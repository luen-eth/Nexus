//! Request scheduler with continuous batching (vLLM-style).
//! Manages concurrent inference requests with efficient KV cache allocation.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

/// Request state
#[derive(Debug, Clone)]
pub enum RequestState {
    /// Waiting to be scheduled
    Pending,
    /// Currently being processed (prefill or decode)
    Running {
        /// Tokens generated so far
        generated: usize,
        /// Max tokens to generate
        max_tokens: usize,
        /// Position in context
        position: usize,
    },
    /// Request completed
    Done,
}

/// Individual inference request
#[derive(Debug, Clone)]
pub struct Request {
    pub id: String,
    pub tokens: Vec<u32>,
    pub state: RequestState,
    pub prompt_len: usize,
    pub max_tokens: usize,
    pub kv_pages: Vec<u64>,
}

impl Request {
    pub fn new(id: String, tokens: Vec<u32>, max_tokens: usize) -> Self {
        let prompt_len = tokens.len();
        Request {
            id,
            tokens,
            state: RequestState::Pending,
            prompt_len,
            max_tokens,
            kv_pages: Vec::new(),
        }
    }

    pub fn advance(&mut self) -> bool {
        if let RequestState::Running {
            generated,
            max_tokens,
            position,
        } = &mut self.state
        {
            if *generated >= *max_tokens {
                self.state = RequestState::Done;
                return false;
            }

            *generated += 1;
            *position += 1;

            if *generated >= *max_tokens {
                self.state = RequestState::Done;
            }
            true
        } else {
            false
        }
    }

    pub fn mark_running(&mut self, kv_pages: Vec<u64>) {
        self.kv_pages = kv_pages;
        self.state = RequestState::Running {
            generated: 0,
            max_tokens: self.max_tokens,
            position: self.prompt_len,
        };
    }

    pub fn is_running(&self) -> bool {
        matches!(self.state, RequestState::Running { .. })
    }

    pub fn is_done(&self) -> bool {
        matches!(self.state, RequestState::Done)
    }

    pub fn generated_tokens(&self) -> usize {
        match self.state {
            RequestState::Running { generated, .. } => generated,
            RequestState::Done => self.tokens.len().saturating_sub(self.prompt_len),
            RequestState::Pending => 0,
        }
    }

    pub fn position(&self) -> usize {
        match self.state {
            RequestState::Running { position, .. } => position,
            _ => self.tokens.len(),
        }
    }
}

/// One active sequence item for token-by-token decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledDecode {
    pub id: String,
    pub token: u32,
    pub position: usize,
    pub generated: usize,
    pub prompt_len: usize,
    pub kv_pages: Vec<u64>,
}

/// KV page allocation assigned to one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPageAllocation {
    pub request_id: String,
    pub page_ids: Vec<u64>,
    pub page_size_tokens: usize,
    pub capacity_tokens: usize,
}

/// Scheduler manages request lifecycle and batching
pub struct Scheduler {
    pending: Mutex<VecDeque<Request>>,
    running: Mutex<Vec<Request>>,
    completed: Mutex<Vec<Request>>,
    max_batch_size: usize,
    page_size_tokens: usize,
    max_kv_pages: usize,
    next_page_id: Mutex<u64>,
    kv_pages: Mutex<HashMap<String, KvPageAllocation>>,
}

impl Scheduler {
    pub fn new(max_batch_size: usize) -> Self {
        Scheduler {
            pending: Mutex::new(VecDeque::new()),
            running: Mutex::new(Vec::new()),
            completed: Mutex::new(Vec::new()),
            max_batch_size,
            page_size_tokens: 16,
            max_kv_pages: usize::MAX / 2,
            next_page_id: Mutex::new(0),
            kv_pages: Mutex::new(HashMap::new()),
        }
    }

    pub fn with_kv_pages(
        max_batch_size: usize,
        page_size_tokens: usize,
        max_kv_pages: usize,
    ) -> Self {
        Scheduler {
            pending: Mutex::new(VecDeque::new()),
            running: Mutex::new(Vec::new()),
            completed: Mutex::new(Vec::new()),
            max_batch_size,
            page_size_tokens: page_size_tokens.max(1),
            max_kv_pages,
            next_page_id: Mutex::new(0),
            kv_pages: Mutex::new(HashMap::new()),
        }
    }

    /// Add a new request to the scheduler
    pub fn add_request(&self, id: String, tokens: Vec<u32>, max_tokens: usize) {
        let mut pending = self.pending.lock().unwrap();
        pending.push_back(Request::new(id, tokens, max_tokens));
    }

    /// Get the next batch of requests to process
    pub fn get_next_batch(&self) -> Vec<Request> {
        let mut pending = self.pending.lock().unwrap();
        let mut running = self.running.lock().unwrap();
        self.fill_running(&mut pending, &mut running)
    }

    fn fill_running(
        &self,
        pending: &mut VecDeque<Request>,
        running: &mut Vec<Request>,
    ) -> Vec<Request> {
        let mut batch = Vec::new();
        let available_slots = self.max_batch_size.saturating_sub(running.len());
        while batch.len() < available_slots {
            if let Some(mut req) = pending.pop_front() {
                let Some(allocation) = self.allocate_kv_pages(&req) else {
                    pending.push_front(req);
                    break;
                };
                req.mark_running(allocation.page_ids.clone());
                batch.push(req);
            } else {
                break;
            }
        }

        running.extend(batch.clone());
        batch
    }

    /// Return one decode item for each running request, admitting pending work first.
    pub fn decode_batch(&self) -> Vec<ScheduledDecode> {
        {
            let mut pending = self.pending.lock().unwrap();
            let mut running = self.running.lock().unwrap();
            self.fill_running(&mut pending, &mut running);
        }

        let running = self.running.lock().unwrap();
        running
            .iter()
            .filter_map(|req| {
                let token = *req.tokens.last()?;
                Some(ScheduledDecode {
                    id: req.id.clone(),
                    token,
                    position: req.position(),
                    generated: req.generated_tokens(),
                    prompt_len: req.prompt_len,
                    kv_pages: req.kv_pages.clone(),
                })
            })
            .collect()
    }

    /// Append one generated token to a running request and advance its state.
    pub fn append_generated(&self, id: &str, token: u32) -> bool {
        let mut running = self.running.lock().unwrap();
        let Some(index) = running.iter().position(|req| req.id == id) else {
            return false;
        };

        let mut req = running.remove(index);
        let advanced = req.advance();
        if advanced {
            req.tokens.push(token);
        }

        if req.is_done() {
            self.release_kv_pages(&req.id);
            self.completed.lock().unwrap().push(req);
        } else {
            running.insert(index, req);
        }
        true
    }

    /// Advance running request positions by one scheduler tick.
    pub fn step(&self) -> Vec<String> {
        let mut running = self.running.lock().unwrap();
        let mut completed = self.completed.lock().unwrap();
        let mut results = Vec::new();

        let mut i = 0;
        while i < running.len() {
            let req = &mut running[i];
            if matches!(req.state, RequestState::Running { .. }) {
                if req.advance() {
                    results.push(req.id.clone());
                }
                if req.is_done() {
                    let req = running.remove(i);
                    self.release_kv_pages(&req.id);
                    completed.push(req);
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }

        results
    }

    /// Get completed requests
    pub fn get_completed(&self) -> Vec<Request> {
        let mut completed = self.completed.lock().unwrap();
        std::mem::take(&mut *completed)
    }

    /// Mark a request as completed, removing it from pending or running queues.
    pub fn complete_request(&self, id: &str) -> bool {
        {
            let mut running = self.running.lock().unwrap();
            if let Some(index) = running.iter().position(|req| req.id == id) {
                let mut req = running.remove(index);
                req.state = RequestState::Done;
                self.release_kv_pages(&req.id);
                self.completed.lock().unwrap().push(req);
                return true;
            }
        }

        let mut pending = self.pending.lock().unwrap();
        if let Some(index) = pending.iter().position(|req| req.id == id) {
            let mut req = pending.remove(index).unwrap();
            req.state = RequestState::Done;
            self.release_kv_pages(&req.id);
            self.completed.lock().unwrap().push(req);
            return true;
        }

        false
    }

    /// Get scheduler statistics
    pub fn stats(&self) -> (usize, usize, usize) {
        let pending = self.pending.lock().unwrap().len();
        let running = self.running.lock().unwrap().len();
        let completed = self.completed.lock().unwrap().len();
        (pending, running, completed)
    }

    pub fn kv_page_stats(&self) -> (usize, usize) {
        let pages = self.kv_pages.lock().unwrap();
        let used = pages
            .values()
            .map(|allocation| allocation.page_ids.len())
            .sum();
        (used, self.max_kv_pages)
    }

    pub fn kv_allocation(&self, id: &str) -> Option<KvPageAllocation> {
        self.kv_pages.lock().unwrap().get(id).cloned()
    }

    fn allocate_kv_pages(&self, req: &Request) -> Option<KvPageAllocation> {
        let capacity_tokens = req.prompt_len + req.max_tokens;
        let needed = capacity_tokens.div_ceil(self.page_size_tokens).max(1);
        let mut pages = self.kv_pages.lock().unwrap();
        let used: usize = pages
            .values()
            .map(|allocation| allocation.page_ids.len())
            .sum();
        if used + needed > self.max_kv_pages {
            return None;
        }

        let mut next = self.next_page_id.lock().unwrap();
        let page_ids: Vec<u64> = (0..needed)
            .map(|_| {
                let id = *next;
                *next += 1;
                id
            })
            .collect();
        let allocation = KvPageAllocation {
            request_id: req.id.clone(),
            page_ids,
            page_size_tokens: self.page_size_tokens,
            capacity_tokens,
        };
        pages.insert(req.id.clone(), allocation.clone());
        Some(allocation)
    }

    fn release_kv_pages(&self, id: &str) {
        self.kv_pages.lock().unwrap().remove(id);
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_lifecycle() {
        let req = Request::new("test".to_string(), vec![1, 2, 3], 5);
        assert!(matches!(req.state, RequestState::Pending));
        assert!(!req.is_running());
        assert!(!req.is_done());

        let mut req = req;
        req.mark_running(Vec::new());
        assert!(req.is_running());

        req.advance();
        if let RequestState::Running { generated, .. } = req.state {
            assert_eq!(generated, 1);
        } else {
            panic!("Expected Running state");
        }
    }

    #[test]
    fn test_scheduler_add_and_batch() {
        let scheduler = Scheduler::new(4);
        scheduler.add_request("r1".to_string(), vec![1, 2], 3);
        scheduler.add_request("r2".to_string(), vec![3, 4], 3);

        let batch = scheduler.get_next_batch();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].id, "r1");
        assert_eq!(batch[1].id, "r2");

        let (pending, running, completed) = scheduler.stats();
        assert_eq!(pending, 0);
        assert_eq!(running, 2);
        assert_eq!(completed, 0);
    }

    #[test]
    fn test_scheduler_step() {
        let scheduler = Scheduler::new(4);
        scheduler.add_request("r1".to_string(), vec![1], 2);

        let _batch = scheduler.get_next_batch();
        let results = scheduler.step();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_scheduler_max_batch() {
        let scheduler = Scheduler::new(2);
        for i in 0..5 {
            scheduler.add_request(format!("r{}", i), vec![i as u32], 3);
        }

        let batch = scheduler.get_next_batch();
        assert!(batch.len() <= 2);
    }

    #[test]
    fn test_scheduler_honors_request_max_tokens() {
        let scheduler = Scheduler::new(4);
        scheduler.add_request("r1".to_string(), vec![1], 1);

        scheduler.get_next_batch();
        let results = scheduler.step();
        assert_eq!(results, vec!["r1".to_string()]);

        let (pending, running, completed) = scheduler.stats();
        assert_eq!((pending, running, completed), (0, 0, 1));
    }

    #[test]
    fn test_scheduler_respects_running_capacity() {
        let scheduler = Scheduler::new(2);
        scheduler.add_request("r1".to_string(), vec![1], 3);
        scheduler.add_request("r2".to_string(), vec![2], 3);
        scheduler.add_request("r3".to_string(), vec![3], 3);

        let first = scheduler.get_next_batch();
        assert_eq!(first.len(), 2);

        let second = scheduler.get_next_batch();
        assert!(second.is_empty());
        let (pending, running, completed) = scheduler.stats();
        assert_eq!((pending, running, completed), (1, 2, 0));
    }

    #[test]
    fn test_scheduler_complete_request() {
        let scheduler = Scheduler::new(2);
        scheduler.add_request("r1".to_string(), vec![1], 10);
        scheduler.get_next_batch();

        assert!(scheduler.complete_request("r1"));
        let (pending, running, completed) = scheduler.stats();
        assert_eq!((pending, running, completed), (0, 0, 1));
    }

    #[test]
    fn test_decode_batch_and_append_generated() {
        let scheduler = Scheduler::new(2);
        scheduler.add_request("r1".to_string(), vec![10, 11], 2);

        let batch = scheduler.decode_batch();
        assert_eq!(
            batch,
            vec![ScheduledDecode {
                id: "r1".to_string(),
                token: 11,
                position: 2,
                generated: 0,
                prompt_len: 2,
                kv_pages: vec![0],
            }]
        );

        assert!(scheduler.append_generated("r1", 12));
        let batch = scheduler.decode_batch();
        assert_eq!(batch[0].token, 12);
        assert_eq!(batch[0].position, 3);
        assert_eq!(batch[0].generated, 1);

        assert!(scheduler.append_generated("r1", 13));
        let (pending, running, completed) = scheduler.stats();
        assert_eq!((pending, running, completed), (0, 0, 1));
    }

    #[test]
    fn test_scheduler_kv_page_capacity_and_release() {
        let scheduler = Scheduler::with_kv_pages(4, 4, 2);
        scheduler.add_request("r1".to_string(), vec![1, 2, 3, 4], 4);
        scheduler.add_request("r2".to_string(), vec![5, 6, 7, 8], 4);

        let first = scheduler.get_next_batch();
        assert_eq!(first.len(), 1);
        let allocation = scheduler.kv_allocation("r1").unwrap();
        assert_eq!(allocation.page_ids, vec![0, 1]);
        assert_eq!(scheduler.kv_page_stats(), (2, 2));

        let second = scheduler.get_next_batch();
        assert!(second.is_empty());

        assert!(scheduler.complete_request("r1"));
        assert_eq!(scheduler.kv_page_stats(), (0, 2));

        let third = scheduler.get_next_batch();
        assert_eq!(third.len(), 1);
        assert_eq!(third[0].id, "r2");
    }
}
