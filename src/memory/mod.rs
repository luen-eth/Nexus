//! Unified Memory Manager inspired by MLX's unified memory model.
//! Provides paged allocation, zero-copy where possible, and automatic
//! CPU/GPU placement for both weights and KV cache.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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
    /// Reference count
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
}

impl MemoryBlock {
    pub fn new(name: &str, size: usize, region: MemRegion) -> Self {
        let page_size = Self::default_page_size(region);
        let num_pages = size.div_ceil(page_size);

        let mut pages = Vec::with_capacity(num_pages);
        let mut offset = 0usize;
        for i in 0..num_pages {
            let size = if i == num_pages - 1 && size % page_size != 0 {
                size % page_size
            } else {
                page_size
            };
            let mut page = MemoryPage::new(i as u64, size, region);
            page.offset = offset;
            pages.push(MemoryPage::new(page.id, page.size, page.region));
            pages.last_mut().unwrap().offset = page.offset;
            offset += size;
        }

        MemoryBlock {
            name: name.to_string(),
            pages,
            total_size: size,
            region,
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

    pub fn add_ref(&mut self) {
        for page in &mut self.pages {
            page.ref_count += 1;
        }
    }

    pub fn release(&mut self) -> bool {
        let mut all_zero = true;
        for page in &mut self.pages {
            page.ref_count = page.ref_count.saturating_sub(1);
            if page.ref_count > 0 {
                all_zero = false;
            }
        }
        all_zero
    }
}

/// Memory pool manages allocations across regions
pub struct MemoryPool {
    /// Blocks indexed by name
    blocks: Mutex<HashMap<String, MemoryBlock>>,
    /// Statistics
    stats: Mutex<MemoryStats>,
    /// Total capacity per region
    capacity: HashMap<MemRegion, usize>,
}

#[derive(Debug, Default, Clone)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub total_freed: usize,
    pub peak_usage: usize,
    pub block_count: usize,
    pub page_faults: usize,
    pub evictions: usize,
}

impl MemoryPool {
    pub fn new() -> Self {
        MemoryPool {
            blocks: Mutex::new(HashMap::new()),
            stats: Mutex::new(MemoryStats::default()),
            capacity: HashMap::from([
                (MemRegion::Host, 8 * 1024 * 1024 * 1024),    // 8GB
                (MemRegion::Unified, 8 * 1024 * 1024 * 1024), // 8GB
                (MemRegion::Device, 4 * 1024 * 1024 * 1024),  // 4GB
            ]),
        }
    }

    /// Set capacity for a region
    pub fn set_capacity(&mut self, region: MemRegion, bytes: usize) {
        self.capacity.insert(region, bytes);
    }

    /// Allocate a memory block
    pub fn allocate(
        &self,
        name: &str,
        size: usize,
        preferred_region: MemRegion,
    ) -> Option<MemoryBlock> {
        // Determine best region
        let region = self.select_region(size, preferred_region)?;

        let mut stats = self.stats.lock().unwrap();
        stats.total_allocated += size;
        stats.block_count += 1;

        if stats.total_allocated > stats.peak_usage {
            stats.peak_usage = stats.total_allocated;
        }

        drop(stats);

        let block = MemoryBlock::new(name, size, region);
        let mut blocks = self.blocks.lock().unwrap();
        blocks.insert(name.to_string(), block.clone());

        Some(block)
    }

    /// Get a memory block by name (returns Arc for shared ownership)
    pub fn get(&self, name: &str) -> Option<Arc<MemoryBlock>> {
        let blocks = self.blocks.lock().unwrap();
        blocks.get(name).map(|b| Arc::new(b.clone()))
    }

    /// Release a memory block
    pub fn release(&self, name: &str) -> bool {
        let mut blocks = self.blocks.lock().unwrap();
        if let Some(block) = blocks.get_mut(name) {
            let freed = block.total_size;
            if !block.release() {
                return false;
            }
            blocks.remove(name);

            let mut stats = self.stats.lock().unwrap();
            stats.total_freed += freed;
            stats.block_count = blocks.len();
            true
        } else {
            false
        }
    }

    /// Evict least-recently-used blocks to free memory
    pub fn evict_to_free(&self, needed_bytes: usize, region: MemRegion) -> usize {
        let mut freed = 0;
        let mut to_remove = Vec::new();

        {
            let blocks = self.blocks.lock().unwrap();

            // Collect evictable block names sorted by ref_count
            let mut candidates: Vec<_> = blocks
                .iter()
                .filter(|(_, b)| b.region == region && b.total_size > 0)
                .collect();
            candidates.sort_by_key(|(_, b)| b.pages.first().map(|p| p.ref_count).unwrap_or(0));

            for (name, block) in candidates {
                if freed >= needed_bytes {
                    break;
                }
                if block.pages.first().map(|p| p.ref_count).unwrap_or(0) <= 1 {
                    freed += block.total_size;
                    to_remove.push(name.clone());
                }
            }
        }

        // Remove evicted blocks
        {
            let mut blocks = self.blocks.lock().unwrap();
            for name in &to_remove {
                blocks.remove(name);
            }
        }

        {
            let mut stats = self.stats.lock().unwrap();
            stats.evictions += to_remove.len();
            stats.total_freed += freed;
            stats.block_count = self.blocks.lock().unwrap().len();
        }

        freed
    }

    /// Get current usage statistics
    pub fn stats(&self) -> MemoryStats {
        self.stats.lock().unwrap().clone()
    }

    /// Print memory report
    pub fn report(&self) {
        let blocks = self.blocks.lock().unwrap();
        let stats = self.stats.lock().unwrap();

        println!("=== Memory Pool Report ===");
        println!(
            "Total allocated:  {} MB",
            stats.total_allocated / 1024 / 1024
        );
        println!("Total freed:      {} MB", stats.total_freed / 1024 / 1024);
        println!("Peak usage:       {} MB", stats.peak_usage / 1024 / 1024);
        println!("Active blocks:    {}", stats.block_count);
        println!("Page faults:      {}", stats.page_faults);
        println!("Evictions:        {}", stats.evictions);

        for (name, block) in blocks.iter() {
            println!(
                "  {:<40} {} region={} pages={}",
                name,
                format_size(block.total_size),
                block.region.name(),
                block.num_pages()
            );
        }
    }

    fn select_region(&self, size: usize, preferred: MemRegion) -> Option<MemRegion> {
        // Try preferred region first
        if let Some(&cap) = self.capacity.get(&preferred) {
            let used = self.used_in_region(preferred);
            if cap.saturating_sub(used) >= size {
                return Some(preferred);
            }
            if self.evict_to_free(size.saturating_sub(cap.saturating_sub(used)), preferred) > 0
                && cap.saturating_sub(self.used_in_region(preferred)) >= size
            {
                return Some(preferred);
            }
        }

        // Fall back to unified memory
        if preferred != MemRegion::Unified {
            if let Some(&cap) = self.capacity.get(&MemRegion::Unified) {
                let used = self.used_in_region(MemRegion::Unified);
                if cap.saturating_sub(used) >= size {
                    return Some(MemRegion::Unified);
                }
            }
        }

        // Fall back to host
        if preferred != MemRegion::Host {
            if let Some(&cap) = self.capacity.get(&MemRegion::Host) {
                let used = self.used_in_region(MemRegion::Host);
                if cap.saturating_sub(used) >= size {
                    return Some(MemRegion::Host);
                }
            }
        }

        None
    }

    fn used_in_region(&self, region: MemRegion) -> usize {
        let blocks = self.blocks.lock().unwrap();
        blocks
            .values()
            .filter(|block| block.region == region)
            .map(|block| block.total_size)
            .sum()
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
        let size = seq_len * head_dim * bytes_per_elem;
        self.kv_cache_pool.allocate(&name, size, MemRegion::Unified)
    }

    /// Release weights
    pub fn release_weights(&self, name: &str) {
        self.pool.release(name);
    }

    /// Clear all KV cache
    pub fn clear_kv_cache(&self) {
        let names: Vec<String> = {
            let blocks = self.kv_cache_pool.blocks.lock().unwrap();
            blocks.keys().cloned().collect()
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
    fn test_memory_pool_release_respects_ref_count() {
        let pool = MemoryPool::new();
        let mut block = pool.allocate("shared", 1024, MemRegion::Host).unwrap();
        block.add_ref();
        {
            let mut blocks = pool.blocks.lock().unwrap();
            blocks.insert("shared".to_string(), block);
        }

        assert!(!pool.release("shared"));
        assert!(pool.get("shared").is_some());
        assert!(pool.release("shared"));
        assert!(pool.get("shared").is_none());
    }

    #[test]
    fn test_unified_memory() {
        let um = UnifiedMemory::new();
        let block = um.allocate_weights("model", 1024 * 1024).unwrap();
        assert_eq!(block.total_size, 1024 * 1024);

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
}
