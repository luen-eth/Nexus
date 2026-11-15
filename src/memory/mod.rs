//! Unified Memory Manager inspired by MLX's unified memory model.
//! Records paged reservations for weights and KV cache. These are accounting
//! records, not RAM/GPU payload buffers. Region selection is a reservation policy.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

/// Memory region type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemRegion {
    /// Host memory (CPU RAM)
    Host,
    /// Device memory (GPU)
    Device,
    /// Unified memory (Apple Silicon shared memory)
    Unified,
}

impl MemRegion {
    pub fn name(&self) -> &'static str {
        match self {
            MemRegion::Host => "host",
            MemRegion::Device => "device",
            MemRegion::Unified => "unified",
        }
    }
}

/// Memory allocation page
#[derive(Debug, Clone)]
pub struct MemoryPage {
    /// Page ID
    pub id: u64,
    /// Size in bytes
    pub size: usize,
    /// Memory region
    pub region: MemRegion,
    /// Offset within the region
    pub offset: usize,
    /// Explicit compatibility pin count snapshot. Actual shared ownership is
    /// tracked by the enclosing block's lease, not by this public metadata.
    pub ref_count: usize,
}

impl MemoryPage {
    pub fn new(id: u64, size: usize, region: MemRegion) -> Self {
        MemoryPage {
            id,
            size,
            region,
            offset: 0,
            ref_count: 1,
        }
    }
}

/// Memory block - a contiguous allocation
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    pub name: String,
    pub pages: Vec<MemoryPage>,
    pub total_size: usize,
    pub region: MemRegion,
    // Pool records and every returned clone share this allocation identity.
    // External leases pin the reservation against release and eviction.
    lease: Arc<AllocationLease>,
}

#[derive(Debug)]
struct AllocationLease {
    // Compatibility for explicit add_ref/release users; Arc owns actual leases.
    manual_refs: AtomicUsize,
}

impl MemoryBlock {
    pub fn new(name: &str, size: usize, region: MemRegion) -> Self {
        let page_size = Self::default_page_size(region);
        let num_pages = size.div_ceil(page_size);

        let mut pages = Vec::with_capacity(num_pages);
        let mut offset = 0usize;
        for i in 0..num_pages {
            let size = if i == num_pages - 1 && !size.is_multiple_of(page_size) {
                size % page_size
            } else {
                page_size
            };
            let mut page = MemoryPage::new(i as u64, size, region);
            page.offset = offset;
            pages.push(page);
            offset += size;
        }

        MemoryBlock {
            name: name.to_string(),
            pages,
            total_size: size,
            region,
            lease: Arc::new(AllocationLease {
                manual_refs: AtomicUsize::new(1),
            }),
        }
    }

    fn default_page_size(region: MemRegion) -> usize {
        match region {
            MemRegion::Unified => 4096, // 4KB pages for unified memory
            MemRegion::Device => 4096,
            MemRegion::Host => 4096,
        }
    }

    pub fn num_pages(&self) -> usize {
        self.pages.len()
    }

    /// Explicit metadata pin, retained for compatibility. Prefer cloning the
    /// block: its shared lease is released automatically when the clone drops.
    pub fn add_ref(&mut self) {
        self.lease.manual_refs.fetch_add(1, Ordering::Relaxed);
        for page in &mut self.pages {
            page.ref_count = page.ref_count.saturating_add(1);
        }
    }

    /// Release an explicit metadata pin. The pool reservation remains pinned
    /// until every returned block/get handle has also been dropped.
    pub fn release(&mut self) -> bool {
        let refs = &self.lease.manual_refs;
        let mut old = refs.load(Ordering::Relaxed);
        loop {
            match refs.compare_exchange_weak(
                old,
                old.saturating_sub(1),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => old = actual,
            }
        }
        for page in &mut self.pages {
            page.ref_count = old.saturating_sub(1);
        }
        old <= 1
    }

    fn evictable(&self) -> bool {
        Arc::strong_count(&self.lease) == 1 && self.lease.manual_refs.load(Ordering::Relaxed) <= 1
    }
}

/// A reservation registry. One mutex protects capacity checks, insertion,
/// eviction and counters together; no operation takes another pool-state lock.
pub struct MemoryPool {
    state: Mutex<PoolState>,
}

struct PoolEntry {
    block: MemoryBlock,
    last_used: u64,
}

struct PoolState {
    blocks: HashMap<String, PoolEntry>,
    stats: MemoryStats,
    capacity: HashMap<MemRegion, usize>,
    clock: u64,
}

#[derive(Debug, Default, Clone)]
pub struct MemoryStats {
    /// Cumulative reservation bytes, not actual payload allocations.
    pub total_allocated: usize,
    pub total_freed: usize,
    /// Currently registered reservation bytes.
    pub current_bytes: usize,
    pub peak_usage: usize,
    pub block_count: usize,
    pub page_faults: usize,
    pub evictions: usize,
}

impl PoolState {
    fn used_in_region(&self, region: MemRegion) -> usize {
        self.blocks
            .values()
            .filter(|entry| entry.block.region == region)
            .map(|entry| entry.block.total_size)
            .sum()
    }

    fn remove(&mut self, name: &str) -> Option<MemoryBlock> {
        let entry = self.blocks.remove(name)?;
        self.stats.total_freed = self
            .stats
            .total_freed
            .saturating_add(entry.block.total_size);
        self.stats.current_bytes -= entry.block.total_size;
        self.stats.block_count = self.blocks.len();
        Some(entry.block)
    }

    fn evict(&mut self, needed: usize, region: MemRegion) -> usize {
        let mut candidates: Vec<_> = self
            .blocks
            .iter()
            .filter(|(_, e)| {
                e.block.region == region && e.block.total_size > 0 && e.block.evictable()
            })
            .map(|(name, e)| (name.clone(), e.last_used))
            .collect();
        candidates.sort_unstable_by_key(|(_, used)| *used);
        let mut freed = 0usize;
        for (name, _) in candidates {
            if freed >= needed {
                break;
            }
            if let Some(block) = self.remove(&name) {
                freed += block.total_size;
                self.stats.evictions = self.stats.evictions.saturating_add(1);
            }
        }
        freed
    }

    fn select_region(&mut self, size: usize, preferred: MemRegion) -> Option<MemRegion> {
        for region in [preferred, MemRegion::Unified, MemRegion::Host] {
            let cap = self.capacity.get(&region).copied().unwrap_or(0);
            if size > cap {
                continue;
            } // Never evict for an impossible reservation.
            let free = cap.saturating_sub(self.used_in_region(region));
            if free < size {
                let evictable: usize = self
                    .blocks
                    .values()
                    .filter(|e| e.block.region == region && e.block.evictable())
                    .map(|e| e.block.total_size)
                    .sum();
                if evictable < size - free {
                    continue;
                }
                self.evict(size - free, region);
            }
            if cap.saturating_sub(self.used_in_region(region)) >= size {
                return Some(region);
            }
        }
        None
    }
}

impl MemoryPool {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(PoolState {
                blocks: HashMap::new(),
                stats: MemoryStats::default(),
                capacity: HashMap::from([
                    (MemRegion::Host, 8 * 1024 * 1024 * 1024),
                    (MemRegion::Unified, 8 * 1024 * 1024 * 1024),
                    (MemRegion::Device, 4 * 1024 * 1024 * 1024),
                ]),
                clock: 0,
            }),
        }
    }

    fn state(&self) -> MutexGuard<'_, PoolState> {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    pub fn set_capacity(&mut self, region: MemRegion, bytes: usize) {
        self.state().capacity.insert(region, bytes);
    }

    /// Reserve metadata bytes. Duplicate names are rejected without replacing
    /// live reservations or changing counters. Returned handles pin the record.
    pub fn allocate(
        &self,
        name: &str,
        size: usize,
        preferred_region: MemRegion,
    ) -> Option<MemoryBlock> {
        let mut state = self.state();
        if state.blocks.contains_key(name) {
            return None;
        }
        state.stats.current_bytes.checked_add(size)?;
        let region = state.select_region(size, preferred_region)?;
        let block = MemoryBlock::new(name, size, region);
        state.clock = state.clock.saturating_add(1);
        let last_used = state.clock;
        state.stats.current_bytes += size;
        state.stats.total_allocated = state.stats.total_allocated.saturating_add(size);
        state.stats.peak_usage = state.stats.peak_usage.max(state.stats.current_bytes);
        state.blocks.insert(
            name.to_string(),
            PoolEntry {
                block: block.clone(),
                last_used,
            },
        );
        state.stats.block_count = state.blocks.len();
        Some(block)
    }

    /// A clone shares the allocation lease with the pool and other handles.
    pub fn get(&self, name: &str) -> Option<Arc<MemoryBlock>> {
        let mut state = self.state();
        state.clock = state.clock.saturating_add(1);
        let now = state.clock;
        let entry = state.blocks.get_mut(name)?;
        entry.last_used = now;
        Some(Arc::new(entry.block.clone()))
    }

    /// Remove a reservation only when no external lease or explicit pin exists.
    pub fn release(&self, name: &str) -> bool {
        let mut state = self.state();
        if !state.blocks.get(name).is_some_and(|e| e.block.evictable()) {
            return false;
        }
        state.remove(name).is_some()
    }

    /// Evict least-recently-used unpinned reservation records.
    pub fn evict_to_free(&self, needed_bytes: usize, region: MemRegion) -> usize {
        self.state().evict(needed_bytes, region)
    }

    pub fn stats(&self) -> MemoryStats {
        self.state().stats.clone()
    }

    pub fn report(&self) {
        // Copy the report under one lock; slow stdout never holds the state lock.
        let (stats, blocks) = {
            let state = self.state();
            let blocks: Vec<_> = state
                .blocks
                .values()
                .map(|entry| {
                    let b = &entry.block;
                    (b.name.clone(), b.total_size, b.region, b.num_pages())
                })
                .collect();
            (state.stats.clone(), blocks)
        };
        println!("=== Memory Reservation Report (not payload memory) ===");
        println!(
            "Reserved cumulative: {}",
            format_size(stats.total_allocated)
        );
        println!("Released cumulative: {}", format_size(stats.total_freed));
        println!("Live reservations:   {}", format_size(stats.current_bytes));
        println!("Peak reservations:   {}", format_size(stats.peak_usage));
        println!("Active blocks:       {}", stats.block_count);
        println!("Evictions:           {}", stats.evictions);
        for (name, size, region, pages) in blocks {
            println!(
                "  {:<40} {} region={} pages={}",
                name,
                format_size(size),
                region.name(),
                pages
            );
        }
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified memory manager - high-level interface combining weight storage and KV cache
pub struct UnifiedMemory {
    pool: Arc<MemoryPool>,
    kv_cache_pool: Arc<MemoryPool>,
}

impl UnifiedMemory {
    pub fn new() -> Self {
        let mut pool = MemoryPool::new();
        // Default: 50% for weights, 50% for KV cache
        pool.set_capacity(MemRegion::Unified, 4 * 1024 * 1024 * 1024);
        pool.set_capacity(MemRegion::Host, 8 * 1024 * 1024 * 1024);

        UnifiedMemory {
            pool: Arc::new(pool),
            kv_cache_pool: Arc::new(MemoryPool::new()),
        }
    }

    /// Allocate weight memory
    pub fn allocate_weights(&self, name: &str, size: usize) -> Option<MemoryBlock> {
        self.pool.allocate(name, size, MemRegion::Unified)
    }

    /// Allocate KV cache memory
    pub fn allocate_kv_cache(
        &self,
        layer: usize,
        seq_len: usize,
        head_dim: usize,
        bytes_per_elem: usize,
    ) -> Option<MemoryBlock> {
        let name = format!("kv_cache_l{}", layer);
        let size = seq_len.checked_mul(head_dim)?.checked_mul(bytes_per_elem)?;
        self.kv_cache_pool.allocate(&name, size, MemRegion::Unified)
    }

    /// Release weights
    pub fn release_weights(&self, name: &str) {
        self.pool.release(name);
    }

    /// Clear all KV cache
    pub fn clear_kv_cache(&self) {
        let names: Vec<String> = {
            let state = self.kv_cache_pool.state();
            state.blocks.keys().cloned().collect()
        };
        for name in names {
            self.kv_cache_pool.release(&name);
        }
    }

    /// Get memory stats
    pub fn stats(&self) -> (MemoryStats, MemoryStats) {
        (self.pool.stats(), self.kv_cache_pool.stats())
    }

    /// Print report
    pub fn report(&self) {
        println!("\n=== Weight Memory ===");
        self.pool.report();
        println!("\n=== KV Cache Memory ===");
        self.kv_cache_pool.report();
    }
}

impl Default for UnifiedMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to format byte sizes
pub fn format_size(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_page() {
        let page = MemoryPage::new(0, 4096, MemRegion::Host);
        assert_eq!(page.id, 0);
        assert_eq!(page.size, 4096);
        assert_eq!(page.ref_count, 1);
    }

    #[test]
    fn test_memory_block() {
        let mut block = MemoryBlock::new("test", 8192, MemRegion::Host);
        assert_eq!(block.num_pages(), 2);
        assert_eq!(block.total_size, 8192);
        assert_eq!(block.pages[0].offset, 0);
        assert_eq!(block.pages[1].offset, 4096);

        block.add_ref();
        assert_eq!(block.pages[0].ref_count, 2);

        let still_alive = block.release();
        assert!(!still_alive); // ref_count goes from 2 to 1, not zero yet

        let freed = block.release();
        assert!(freed); // ref_count goes from 1 to 0
    }

    #[test]
    fn test_memory_pool_allocate() {
        let pool = MemoryPool::new();
        let block = pool.allocate("test_block", 4096, MemRegion::Host).unwrap();
        assert_eq!(block.name, "test_block");

        // Verify we can retrieve it
        assert!(pool.get("test_block").is_some());

        // Release it
        drop(block);
        pool.release("test_block");
        assert!(pool.get("test_block").is_none());
    }

    #[test]
    fn test_memory_pool_stats() {
        let pool = MemoryPool::new();
        pool.allocate("a", 1024, MemRegion::Host).unwrap();
        pool.allocate("b", 2048, MemRegion::Host).unwrap();

        let stats = pool.stats();
        assert_eq!(stats.block_count, 2);
        assert_eq!(stats.total_allocated, 3072);
    }

    #[test]
    fn test_memory_pool_release_respects_shared_leases() {
        let pool = MemoryPool::new();
        let block = pool.allocate("shared", 1024, MemRegion::Host).unwrap();
        let handle = pool.get("shared").unwrap();
        assert!(Arc::ptr_eq(&block.lease, &handle.lease));
        assert!(!pool.release("shared"));
        drop(block);
        assert!(!pool.release("shared"));
        assert_eq!(pool.evict_to_free(1024, MemRegion::Host), 0);
        drop(handle);
        assert!(pool.release("shared"));
        assert!(pool.get("shared").is_none());
    }

    #[test]
    fn test_unified_memory() {
        let um = UnifiedMemory::new();
        let block = um.allocate_weights("model", 1024 * 1024).unwrap();
        assert_eq!(block.total_size, 1024 * 1024);

        drop(block);
        um.release_weights("model");
        um.allocate_kv_cache(0, 16, 8, 2).unwrap();
        um.clear_kv_cache();

        let (w_stats, kv_stats) = um.stats();
        assert_eq!(w_stats.block_count, 0);
        assert_eq!(kv_stats.block_count, 0);
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1024 * 5), "5.00 KB");
        assert_eq!(format_size(1024 * 1024 * 10), "10.00 MB");
        assert_eq!(format_size(1024 * 1024 * 1024 * 2), "2.00 GB");
    }

    #[test]
    fn test_mem_region_names() {
        assert_eq!(MemRegion::Host.name(), "host");
        assert_eq!(MemRegion::Device.name(), "device");
        assert_eq!(MemRegion::Unified.name(), "unified");
    }
    fn bounded_pool(cap: usize) -> MemoryPool {
        let mut pool = MemoryPool::new();
        pool.set_capacity(MemRegion::Host, cap);
        pool.set_capacity(MemRegion::Unified, 0);
        pool.set_capacity(MemRegion::Device, 0);
        pool
    }

    #[test]
    fn live_and_peak_reservations_do_not_count_freed_history() {
        let pool = bounded_pool(128);
        for _ in 0..4 {
            drop(pool.allocate("a", 100, MemRegion::Host).unwrap());
            assert_eq!(pool.stats().current_bytes, 100);
            assert!(pool.release("a"));
        }
        let stats = pool.stats();
        assert_eq!(stats.total_allocated, 400);
        assert_eq!(stats.total_freed, 400);
        assert_eq!(stats.current_bytes, 0);
        assert_eq!(stats.peak_usage, 100);
        assert_eq!(stats.block_count, 0);
    }

    #[test]
    fn duplicate_names_are_rejected_without_changing_stats() {
        let pool = bounded_pool(128);
        let block = pool.allocate("a", 96, MemRegion::Host).unwrap();
        assert!(pool.allocate("a", 32, MemRegion::Host).is_none());
        assert_eq!(pool.stats().current_bytes, 96);
        assert_eq!(pool.stats().total_allocated, 96);
        assert_eq!(pool.stats().block_count, 1);
        assert_eq!(pool.get("a").unwrap().total_size, block.total_size);
    }

    #[test]
    fn held_allocation_or_get_handle_prevents_replacement() {
        let pool = bounded_pool(128);
        let block = pool.allocate("held", 96, MemRegion::Host).unwrap();
        let handle = pool.get("held").unwrap();
        assert!(pool.allocate("new", 96, MemRegion::Host).is_none());
        drop(block);
        assert!(pool.allocate("new", 96, MemRegion::Host).is_none());
        assert_eq!(pool.evict_to_free(96, MemRegion::Host), 0);
        drop(handle);
        assert!(pool.allocate("new", 96, MemRegion::Host).is_some());
        assert!(pool.get("held").is_none());
        assert_eq!(pool.stats().current_bytes, 96);
        assert_eq!(pool.stats().evictions, 1);
    }

    #[test]
    fn eviction_uses_lru_and_never_evicts_for_impossible_size() {
        let pool = bounded_pool(128);
        drop(pool.allocate("a", 64, MemRegion::Host).unwrap());
        drop(pool.allocate("b", 64, MemRegion::Host).unwrap());
        drop(pool.get("a").unwrap());
        assert!(pool.allocate("impossible", 129, MemRegion::Host).is_none());
        assert_eq!(pool.stats().evictions, 0);
        drop(pool.allocate("c", 64, MemRegion::Host).unwrap());
        assert!(pool.get("b").is_none());
        assert!(pool.get("a").is_some());
        assert_eq!(pool.stats().current_bytes, 128);
    }

    #[test]
    fn concurrent_reservations_cannot_overbook_a_region() {
        use std::sync::Barrier;
        let pool = Arc::new(bounded_pool(128));
        let barrier = Arc::new(Barrier::new(3));
        let threads: Vec<_> = (0..2)
            .map(|id| {
                let pool = pool.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    let reservation = pool.allocate(&format!("r{id}"), 96, MemRegion::Host);
                    barrier.wait();
                    reservation.is_some()
                })
            })
            .collect();
        barrier.wait();
        barrier.wait();
        assert_eq!(
            threads
                .into_iter()
                .map(|t| usize::from(t.join().unwrap()))
                .sum::<usize>(),
            1
        );
        assert_eq!(pool.stats().current_bytes, 96);
    }

    #[test]
    fn concurrent_allocate_release_evict_and_report_make_progress() {
        let pool = Arc::new(bounded_pool(4096));
        let (tx, rx) = std::sync::mpsc::channel();
        let mut threads = Vec::new();
        for worker in 0..4 {
            let pool = pool.clone();
            let tx = tx.clone();
            threads.push(std::thread::spawn(move || {
                for i in 0..200 {
                    let name = format!("{worker}-{i}");
                    if let Some(block) = pool.allocate(&name, 64, MemRegion::Host) {
                        drop(block);
                        pool.release(&name);
                    }
                    pool.evict_to_free(64, MemRegion::Host);
                    if i % 100 == 0 {
                        pool.report();
                    }
                }
                tx.send(()).unwrap();
            }));
        }
        for _ in 0..4 {
            rx.recv_timeout(std::time::Duration::from_secs(5))
                .expect("pool operations deadlocked");
        }
        for thread in threads {
            thread.join().unwrap();
        }
        let state = pool.state();
        let summed: usize = state.blocks.values().map(|e| e.block.total_size).sum();
        assert_eq!(state.stats.current_bytes, summed);
        assert_eq!(
            state.stats.total_allocated - state.stats.total_freed,
            summed
        );
        assert_eq!(state.stats.block_count, state.blocks.len());
        assert!(summed <= 4096);
    }

    #[test]
    fn kv_metadata_footprint_overflow_is_rejected() {
        let memory = UnifiedMemory::new();
        assert!(memory.allocate_kv_cache(0, usize::MAX, 2, 4).is_none());
        assert_eq!(memory.stats().1.total_allocated, 0);
    }
    #[test]
    fn explicit_metadata_pins_share_allocation_state_across_clones() {
        let pool = bounded_pool(128);
        let mut block = pool.allocate("pinned", 96, MemRegion::Host).unwrap();
        block.add_ref();
        let mut cloned = block.clone();
        assert!(!cloned.release()); // explicit pin count 2 -> 1
        drop(block);
        assert_eq!(pool.evict_to_free(96, MemRegion::Host), 0);
        drop(cloned);
        assert_eq!(pool.evict_to_free(96, MemRegion::Host), 96);
    }

    #[test]
    fn failed_reservation_does_not_discard_unrelated_cache_entries() {
        let pool = bounded_pool(128);
        let pinned = pool.allocate("pinned", 96, MemRegion::Host).unwrap();
        drop(pool.allocate("cached", 16, MemRegion::Host).unwrap());
        assert!(pool.allocate("cannot-fit", 64, MemRegion::Host).is_none());
        assert_eq!(pool.stats().current_bytes, 112);
        assert_eq!(pool.stats().evictions, 0);
        assert!(pool.get("cached").is_some());
        drop(pinned);
    }
}
