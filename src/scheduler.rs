//! Request scheduler for continuous batching.
//!
//! Tracks pending/running requests and accounts KV pages so that admission is
//! bounded by both a running-slot limit and a KV page budget. The server's
//! generation worker uses it for admission control; actual KV memory is owned
//! by each engine `GenerationSession`.
//!
//! Completed requests are counted, and only a small bounded history is kept
//! (with token buffers released) so long-running servers do not retain prompts.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Default number of completed requests kept for `get_completed`.
pub const DEFAULT_COMPLETED_HISTORY: usize = 64;
/// Upper bound for default accounting metadata (runtime installs a model-specific budget).
pub const DEFAULT_MAX_KV_PAGES: usize = 65_536;
/// After this many resource-blocked admission opportunities, stop bypassing
/// that request and let running work drain until it fits.
pub const MAX_ADMISSION_BYPASSES: usize = 8;

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
    /// Number of generated tokens; preserved after token buffers are released.
    generated_count: usize,
    admission_skips: usize,
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
            generated_count: 0,
            admission_skips: 0,
        }
    }

    /// Drop token and page buffers once a request is finished.
    fn release_buffers(&mut self) {
        self.tokens = Vec::new();
        self.kv_pages = Vec::new();
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
            self.generated_count = *generated;

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
            RequestState::Done => self.generated_count,
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
    /// Bounded history of finished requests (token buffers released).
    completed: Mutex<VecDeque<Request>>,
    completed_history: usize,
    completed_total: AtomicUsize,
    max_batch_size: usize,
    page_size_tokens: usize,
    max_kv_pages: AtomicUsize,
    next_page_id: Mutex<u64>,
    kv_pages: Mutex<HashMap<String, KvPageAllocation>>,
}

impl Scheduler {
    pub fn new(max_batch_size: usize) -> Self {
        Self::with_kv_pages(max_batch_size, 16, DEFAULT_MAX_KV_PAGES)
    }

    pub fn with_kv_pages(
        max_batch_size: usize,
        page_size_tokens: usize,
        max_kv_pages: usize,
    ) -> Self {
        Scheduler {
            pending: Mutex::new(VecDeque::new()),
            running: Mutex::new(Vec::new()),
            completed: Mutex::new(VecDeque::new()),
            completed_history: DEFAULT_COMPLETED_HISTORY,
            completed_total: AtomicUsize::new(0),
            max_batch_size: max_batch_size.max(1),
            page_size_tokens: page_size_tokens.max(1),
            max_kv_pages: AtomicUsize::new(max_kv_pages),
            next_page_id: Mutex::new(0),
            kv_pages: Mutex::new(HashMap::new()),
        }
    }

    /// Change how many finished requests are retained for `get_completed`.
    pub fn with_completed_history(mut self, completed_history: usize) -> Self {
        self.completed_history = completed_history;
        self
    }

    /// Maximum number of concurrently running requests.
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    /// Tokens per KV page used for accounting.
    pub fn page_size_tokens(&self) -> usize {
        self.page_size_tokens
    }

    /// Replace the KV page budget (e.g. after a model with a different context
    /// length is installed). Existing allocations are kept.
    pub fn set_kv_page_budget(&self, max_kv_pages: usize) {
        self.max_kv_pages.store(max_kv_pages, Ordering::SeqCst);
    }

    /// Number of KV pages a request with this footprint needs.
    pub fn pages_needed(&self, prompt_len: usize, max_tokens: usize) -> usize {
        prompt_len
            .checked_add(max_tokens)
            .map(|tokens| tokens.div_ceil(self.page_size_tokens).max(1))
            .unwrap_or(usize::MAX)
    }

    /// Whether a request of this footprint could ever be admitted under the
    /// current KV page budget. Requests that can never fit must not be queued,
    /// because they would block the head of the pending queue.
    pub fn can_ever_fit(&self, prompt_len: usize, max_tokens: usize) -> bool {
        prompt_len.checked_add(max_tokens).is_some()
            && self.pages_needed(prompt_len, max_tokens) <= self.max_kv_pages.load(Ordering::SeqCst)
    }

    fn record_completed(&self, mut req: Request) {
        req.state = RequestState::Done;
        req.release_buffers();
        self.completed_total.fetch_add(1, Ordering::SeqCst);
        if self.completed_history == 0 {
            return;
        }
        let mut completed = self.completed.lock().unwrap();
        while completed.len() >= self.completed_history {
            completed.pop_front();
        }
        completed.push_back(req);
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
        // Inspect each pending request at most once. Impossible requests are
        // finalized; temporarily blocked requests rotate behind work that fits.
        // FIFO order is retained among requests with the same resource needs.
        for _ in 0..pending.len() {
            if batch.len() >= available_slots {
                break;
            }
            let Some(mut req) = pending.pop_front() else {
                break;
            };
            if !self.can_ever_fit(req.prompt_len, req.max_tokens) {
                self.record_completed(req);
                continue;
            }
            let Some(allocation) = self.allocate_kv_pages(&req) else {
                req.admission_skips = req.admission_skips.saturating_add(1);
                if req.admission_skips >= MAX_ADMISSION_BYPASSES {
                    // Reserve the next admission opportunity for this older
                    // request. New small jobs must not continually consume the
                    // pages released by running jobs.
                    pending.push_front(req);
                    break;
                }
                pending.push_back(req);
                continue;
            };
            req.mark_running(allocation.page_ids.clone());
            batch.push(req);
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
            drop(running);
            self.record_completed(req);
        } else {
            running.insert(index, req);
        }
        true
    }

    /// Advance running request positions by one scheduler tick.
    pub fn step(&self) -> Vec<String> {
        let mut running = self.running.lock().unwrap();
        let mut finished = Vec::new();
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
                    finished.push(req);
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }
        drop(running);
        for req in finished {
            self.record_completed(req);
        }

        results
    }

    /// Drain the bounded history of completed requests (token buffers are
    /// released; `generated_tokens()` remains available).
    pub fn get_completed(&self) -> Vec<Request> {
        let mut completed = self.completed.lock().unwrap();
        completed.drain(..).collect()
    }

    /// Mark a request as completed, removing it from pending or running queues.
    pub fn complete_request(&self, id: &str) -> bool {
        {
            let mut running = self.running.lock().unwrap();
            if let Some(index) = running.iter().position(|req| req.id == id) {
                let req = running.remove(index);
                drop(running);
                self.release_kv_pages(&req.id);
                self.record_completed(req);
                return true;
            }
        }

        let removed = {
            let mut pending = self.pending.lock().unwrap();
            pending
                .iter()
                .position(|req| req.id == id)
                .and_then(|index| pending.remove(index))
        };
        if let Some(req) = removed {
            self.release_kv_pages(&req.id);
            self.record_completed(req);
            return true;
        }

        false
    }

    /// Get scheduler statistics: (pending, running, completed_total).
    ///
    /// `completed_total` is a monotonic counter of finished requests; it does
    /// not imply that those requests are retained in memory.
    pub fn stats(&self) -> (usize, usize, usize) {
        let pending = self.pending.lock().unwrap().len();
        let running = self.running.lock().unwrap().len();
        let completed = self.completed_total.load(Ordering::SeqCst);
        (pending, running, completed)
    }

    pub fn kv_page_stats(&self) -> (usize, usize) {
        let pages = self.kv_pages.lock().unwrap();
        let used = pages
            .values()
            .map(|allocation| allocation.page_ids.len())
            .sum();
        (used, self.max_kv_pages.load(Ordering::SeqCst))
    }

    pub fn kv_allocation(&self, id: &str) -> Option<KvPageAllocation> {
        self.kv_pages.lock().unwrap().get(id).cloned()
    }

    fn allocate_kv_pages(&self, req: &Request) -> Option<KvPageAllocation> {
        let capacity_tokens = req.prompt_len.checked_add(req.max_tokens)?;
        let needed = self.pages_needed(req.prompt_len, req.max_tokens);
        let mut pages = self.kv_pages.lock().unwrap();
        let used = pages.values().try_fold(0usize, |used, allocation| {
            used.checked_add(allocation.page_ids.len())
        })?;
        if used.checked_add(needed)? > self.max_kv_pages.load(Ordering::SeqCst) {
            return None;
        }

        let mut next = self.next_page_id.lock().unwrap();
        let end = next.checked_add(u64::try_from(needed).ok()?)?;
        let mut page_ids = Vec::new();
        page_ids.try_reserve_exact(needed).ok()?;
        page_ids.extend(*next..end);
        *next = end;
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

    #[test]
    fn test_completed_history_is_bounded_and_releases_tokens() {
        let scheduler = Scheduler::new(2).with_completed_history(3);
        for i in 0..10 {
            let id = format!("r{i}");
            scheduler.add_request(id.clone(), vec![7; 1000], 4);
            scheduler.get_next_batch();
            assert!(scheduler.append_generated(&id, 1));
            assert!(scheduler.complete_request(&id));
        }

        let (pending, running, completed) = scheduler.stats();
        assert_eq!((pending, running, completed), (0, 0, 10));
        assert_eq!(scheduler.kv_page_stats().0, 0);

        let history = scheduler.get_completed();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].id, "r7");
        for req in &history {
            assert!(req.is_done());
            assert!(req.tokens.is_empty(), "completed prompt must be released");
            assert_eq!(req.generated_tokens(), 1);
        }
        assert!(scheduler.get_completed().is_empty());
        // Counter is monotonic even after draining history.
        assert_eq!(scheduler.stats().2, 10);
    }

    #[test]
    fn test_zero_history_retains_nothing() {
        let scheduler = Scheduler::new(1).with_completed_history(0);
        scheduler.add_request("r".to_string(), vec![1, 2, 3], 1);
        scheduler.get_next_batch();
        assert!(scheduler.complete_request("r"));
        assert!(scheduler.get_completed().is_empty());
        assert_eq!(scheduler.stats(), (0, 0, 1));
    }

    #[test]
    fn test_kv_budget_update_and_fit_check() {
        let scheduler = Scheduler::with_kv_pages(4, 4, 2);
        assert!(scheduler.can_ever_fit(4, 4));
        assert!(!scheduler.can_ever_fit(8, 4));
        scheduler.set_kv_page_budget(3);
        assert!(scheduler.can_ever_fit(8, 4));
        assert_eq!(scheduler.kv_page_stats(), (0, 3));
    }
    #[test]
    fn impossible_or_overflowing_request_does_not_block_a_small_request() {
        let scheduler = Scheduler::with_kv_pages(2, 16, 1);
        scheduler.add_request("oversize".into(), vec![1], 16);
        scheduler.add_request("overflow".into(), vec![1], usize::MAX);
        scheduler.add_request("fits".into(), vec![1], 1);
        let batch = scheduler.get_next_batch();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].id, "fits");
        assert_eq!(scheduler.stats(), (0, 1, 2));
        assert!(!scheduler.can_ever_fit(1, usize::MAX));
    }

    #[test]
    fn temporarily_blocked_request_rotates_and_runs_after_pages_are_released() {
        let scheduler = Scheduler::with_kv_pages(3, 4, 3);
        scheduler.add_request("held".into(), vec![1; 8], 0);
        assert_eq!(scheduler.get_next_batch()[0].id, "held");
        scheduler.add_request("large".into(), vec![1; 8], 0);
        scheduler.add_request("small".into(), vec![1], 0);
        assert_eq!(scheduler.get_next_batch()[0].id, "small");
        assert_eq!(scheduler.stats().0, 1);
        scheduler.complete_request("held");
        assert_eq!(scheduler.get_next_batch()[0].id, "large");
    }
    #[test]
    fn aged_large_request_reserves_admission_before_a_continuous_small_stream() {
        let scheduler = Scheduler::with_kv_pages(2, 4, 3);
        scheduler.add_request("held".into(), vec![1], 0);
        assert_eq!(scheduler.get_next_batch()[0].id, "held");
        scheduler.add_request("large".into(), vec![1; 12], 0);
        let mut held = "held".to_string();
        let mut reserved = false;
        for i in 0..MAX_ADMISSION_BYPASSES + 1 {
            let small = format!("small-{i}");
            scheduler.add_request(small.clone(), vec![1], 0);
            let batch = scheduler.get_next_batch();
            if batch.is_empty() {
                reserved = true;
                scheduler.complete_request(&held);
                break;
            }
            assert_eq!(batch.len(), 1);
            assert_eq!(batch[0].id, small);
            scheduler.complete_request(&held);
            held = small;
        }
        assert!(reserved, "large request was starved by small arrivals");
        let admitted = scheduler.get_next_batch();
        assert_eq!(admitted.len(), 1);
        assert_eq!(admitted[0].id, "large");
        assert_eq!(scheduler.kv_page_stats().0, 3);
        scheduler.complete_request("large");
        assert!(
            !scheduler.get_next_batch().is_empty(),
            "small work resumes after the large request"
        );
    }
}
