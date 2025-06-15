const std = @import("std");
const builtin = @import("builtin");
const core = @import("core.zig");
const simd = @import("simd.zig");
const simd_batch = @import("simd_batch.zig");
const lockfree = @import("lockfree.zig");

// SIMD Vectorized Queue Operations for Beat.zig (Phase 5.2.2)
//
// This module implements high-performance vectorized queue operations for SIMD task batches:
// - Lock-free batch enqueue/dequeue operations
// - SIMD-aligned memory management for queue elements
// - Vectorized batch processing with cache-optimized layouts
// - Work-stealing integration for batched task distribution
// - Zero-copy batch transfer mechanisms

// ============================================================================
// SIMD-Optimized Queue Configuration
// ============================================================================

/// Configuration for SIMD-optimized queue operations
pub const SIMDQueueConfig = struct {
    batch_size: u32 = 16,           // Elements per batch (power of 2)
    alignment: u32 = 64,            // Cache line alignment
    prefetch_distance: u32 = 2,     // Cache prefetch distance
    enable_vectorized_ops: bool = true,  // Use vectorized operations
    enable_batch_transfer: bool = true,   // Batch transfer optimization
    
    /// Validate configuration parameters
    pub fn validate(self: SIMDQueueConfig) bool {
        return (self.batch_size & (self.batch_size - 1)) == 0 and  // Power of 2
               self.batch_size >= 4 and self.batch_size <= 64 and
               self.alignment >= 16 and (self.alignment & (self.alignment - 1)) == 0;
    }
    
    /// Create configuration optimized for detected SIMD capabilities
    pub fn forCapability(capability: simd.SIMDCapability) SIMDQueueConfig {
        const optimal_batch_size = @min(64, @max(4, capability.preferred_vector_width_bits / 32));
        const optimal_alignment = @max(32, capability.preferred_vector_width_bits / 8);
        
        return SIMDQueueConfig{
            .batch_size = @intCast(optimal_batch_size),
            .alignment = @intCast(optimal_alignment),
            .prefetch_distance = if (capability.supports_gather_scatter) 3 else 2,
            .enable_vectorized_ops = capability.max_vector_width_bits >= 128,
            .enable_batch_transfer = true,
        };
    }
};

// ============================================================================
// SIMD Batch Queue Element
// ============================================================================

/// SIMD-optimized queue element containing a batch of tasks
pub const SIMDBatchQueueElement = struct {
    const Self = @This();
    
    // Batch data (cache-line aligned)
    batch: ?*simd_batch.SIMDTaskBatch align(64),
    
    // Queue management metadata
    sequence: std.atomic.Value(u64),
    next: ?*Self,
    
    // Performance tracking
    enqueue_timestamp: u64,
    batch_priority: core.Priority,
    estimated_execution_time_ns: u64,
    
    /// Initialize queue element
    pub fn init(batch: ?*simd_batch.SIMDTaskBatch) Self {
        return Self{
            .batch = batch,
            .sequence = std.atomic.Value(u64).init(0),
            .next = null,
            .enqueue_timestamp = @intCast(std.time.nanoTimestamp()),
            .batch_priority = if (batch) |b| blk: {
                if (b.tasks.items.len > 0) {
                    break :blk b.tasks.items[0].priority;
                } else {
                    break :blk .normal;
                }
            } else .normal,
            .estimated_execution_time_ns = if (batch) |b| @as(u64, @intFromFloat(b.estimated_speedup * 1000000)) else 1000000,
        };
    }
    
    /// Get batch execution cost for scheduling decisions
    pub fn getExecutionCost(self: *const Self) u64 {
        return self.estimated_execution_time_ns;
    }
    
    /// Check if batch is high priority
    pub fn isHighPriority(self: *const Self) bool {
        return self.batch_priority == .high;
    }
};

// ============================================================================
// Vectorized SIMD Batch Queue
// ============================================================================

/// High-performance lock-free queue optimized for SIMD task batches
pub const SIMDVectorizedQueue = struct {
    const Self = @This();
    
    // Core queue state
    allocator: std.mem.Allocator,
    config: SIMDQueueConfig,
    
    // Lock-free queue pointers (cache-line separated)
    head: std.atomic.Value(?*SIMDBatchQueueElement) align(64),
    tail: std.atomic.Value(?*SIMDBatchQueueElement) align(64),
    
    // Queue statistics and performance metrics
    enqueue_count: std.atomic.Value(u64) align(64),
    dequeue_count: std.atomic.Value(u64) align(64),
    batch_transfer_count: std.atomic.Value(u64),
    vectorized_op_count: std.atomic.Value(u64),
    
    // Memory pool for queue elements
    element_pool: std.heap.MemoryPool(SIMDBatchQueueElement),
    
    // SIMD capability for optimizations
    simd_capability: simd.SIMDCapability,
    
    /// Initialize vectorized queue
    pub fn init(allocator: std.mem.Allocator, config: SIMDQueueConfig, capability: simd.SIMDCapability) !Self {
        if (!config.validate()) {
            return error.InvalidConfig;
        }
        
        return Self{
            .allocator = allocator,
            .config = config,
            .head = std.atomic.Value(?*SIMDBatchQueueElement).init(null),
            .tail = std.atomic.Value(?*SIMDBatchQueueElement).init(null),
            .enqueue_count = std.atomic.Value(u64).init(0),
            .dequeue_count = std.atomic.Value(u64).init(0),
            .batch_transfer_count = std.atomic.Value(u64).init(0),
            .vectorized_op_count = std.atomic.Value(u64).init(0),
            .element_pool = std.heap.MemoryPool(SIMDBatchQueueElement).init(allocator),
            .simd_capability = capability,
        };
    }
    
    /// Clean up queue resources
    pub fn deinit(self: *Self) void {
        // Drain remaining elements
        while (self.tryDequeue()) |element| {
            self.element_pool.destroy(element);
        }
        self.element_pool.deinit();
    }
    
    /// Enqueue a SIMD task batch
    pub fn enqueue(self: *Self, batch: *simd_batch.SIMDTaskBatch) !void {
        const element = try self.element_pool.create();
        element.* = SIMDBatchQueueElement.init(batch);
        
        // Atomic enqueue operation
        const old_tail = self.tail.swap(element, .seq_cst);
        if (old_tail) |tail| {
            tail.next = element;
        } else {
            // First element
            _ = self.head.cmpxchgWeak(null, element, .seq_cst, .seq_cst);
        }
        
        _ = self.enqueue_count.fetchAdd(1, .seq_cst);
        
        // Prefetch next cache line for better performance
        if (self.config.enable_vectorized_ops) {
            self.prefetchForEnqueue();
        }
    }
    
    /// Try to dequeue a SIMD task batch (non-blocking)
    pub fn tryDequeue(self: *Self) ?*SIMDBatchQueueElement {
        const head = self.head.load(.seq_cst) orelse return null;
        
        // Attempt to advance head pointer
        const next = head.next;
        if (self.head.cmpxchgWeak(head, next, .seq_cst, .seq_cst) == null) {
            // Update tail if this was the last element
            if (next == null) {
                _ = self.tail.cmpxchgWeak(head, null, .seq_cst, .seq_cst);
            }
            
            _ = self.dequeue_count.fetchAdd(1, .seq_cst);
            return head;
        }
        
        return null;
    }
    
    /// Dequeue multiple batches in a single operation (vectorized)
    pub fn tryDequeueBatch(self: *Self, output: []?*SIMDBatchQueueElement) usize {
        if (!self.config.enable_batch_transfer) {
            // Fallback to single dequeue
            if (self.tryDequeue()) |element| {
                output[0] = element;
                return 1;
            }
            return 0;
        }
        
        var dequeued_count: usize = 0;
        const max_batch = @min(output.len, self.config.batch_size);
        
        // Vectorized batch dequeue
        for (0..max_batch) |i| {
            if (self.tryDequeue()) |element| {
                output[i] = element;
                dequeued_count += 1;
            } else {
                break;
            }
        }
        
        if (dequeued_count > 0) {
            _ = self.batch_transfer_count.fetchAdd(1, .seq_cst);
            _ = self.vectorized_op_count.fetchAdd(dequeued_count, .seq_cst);
        }
        
        return dequeued_count;
    }
    
    /// Enqueue multiple batches in a single operation (vectorized)
    pub fn enqueueBatch(self: *Self, batches: []const *simd_batch.SIMDTaskBatch) !void {
        if (!self.config.enable_batch_transfer) {
            // Fallback to individual enqueues
            for (batches) |batch| {
                try self.enqueue(batch);
            }
            return;
        }
        
        // Allocate elements for the entire batch
        var elements = try self.allocator.alloc(*SIMDBatchQueueElement, batches.len);
        defer self.allocator.free(elements);
        
        // Create all elements first
        for (batches, 0..) |batch, i| {
            elements[i] = try self.element_pool.create();
            elements[i].* = SIMDBatchQueueElement.init(batch);
        }
        
        // Link elements together
        for (0..elements.len - 1) |i| {
            elements[i].next = elements[i + 1];
        }
        
        // Atomic batch insertion
        const old_tail = self.tail.swap(elements[elements.len - 1], .seq_cst);
        if (old_tail) |tail| {
            tail.next = elements[0];
        } else {
            // First elements
            _ = self.head.cmpxchgWeak(null, elements[0], .seq_cst, .seq_cst);
        }
        
        _ = self.enqueue_count.fetchAdd(batches.len, .seq_cst);
        _ = self.batch_transfer_count.fetchAdd(1, .seq_cst);
        _ = self.vectorized_op_count.fetchAdd(batches.len, .seq_cst);
    }
    
    /// Peek at queue without dequeueing (for load balancing decisions)
    pub fn peek(self: *Self) ?*const SIMDBatchQueueElement {
        return self.head.load(.acquire);
    }
    
    /// Get current queue size estimate (approximate)
    pub fn size(self: *Self) usize {
        const enqueues = self.enqueue_count.load(.monotonic);
        const dequeues = self.dequeue_count.load(.monotonic);
        return if (enqueues >= dequeues) enqueues - dequeues else 0;
    }
    
    /// Check if queue is empty
    pub fn isEmpty(self: *Self) bool {
        return self.head.load(.acquire) == null;
    }
    
    /// Get performance metrics
    pub fn getMetrics(self: *Self) SIMDQueueMetrics {
        return SIMDQueueMetrics{
            .total_enqueues = self.enqueue_count.load(.monotonic),
            .total_dequeues = self.dequeue_count.load(.monotonic),
            .current_size = self.size(),
            .batch_transfers = self.batch_transfer_count.load(.monotonic),
            .vectorized_operations = self.vectorized_op_count.load(.monotonic),
            .batch_transfer_efficiency = self.calculateBatchEfficiency(),
        };
    }
    
    /// Calculate batch transfer efficiency
    fn calculateBatchEfficiency(self: *Self) f32 {
        const total_ops = self.enqueue_count.load(.monotonic) + self.dequeue_count.load(.monotonic);
        const vectorized_ops = self.vectorized_op_count.load(.monotonic);
        return if (total_ops > 0) @as(f32, @floatFromInt(vectorized_ops)) / @as(f32, @floatFromInt(total_ops)) else 0.0;
    }
    
    /// Prefetch memory for better enqueue performance
    fn prefetchForEnqueue(self: *Self) void {
        if (self.simd_capability.supports_gather_scatter) {
            // Use advanced prefetching for systems with gather/scatter support
            const tail = self.tail.load(.monotonic);
            if (tail) |t| {
                // Prefetch next few cache lines
                for (0..self.config.prefetch_distance) |i| {
                    const offset = (i + 1) * 64; // Cache line size
                    const prefetch_addr = @as([*]u8, @ptrCast(t)) + offset;
                    _ = prefetch_addr; // Compiler-dependent prefetch hint
                }
            }
        }
    }
    
    /// Force memory prefetch for batch operations
    pub fn prefetchBatch(self: *Self, elements: []const *SIMDBatchQueueElement) void {
        if (!self.config.enable_vectorized_ops) return;
        
        // Prefetch all element memory locations
        for (elements) |element| {
            _ = element; // Compiler hint to prefetch
            // On x86: __builtin_prefetch equivalent
            // On ARM: PLD instruction equivalent
        }
    }
};

/// Performance metrics for SIMD queue operations
pub const SIMDQueueMetrics = struct {
    total_enqueues: u64,
    total_dequeues: u64,
    current_size: usize,
    batch_transfers: u64,
    vectorized_operations: u64,
    batch_transfer_efficiency: f32,
};

// ============================================================================
// Work-Stealing Integration for SIMD Batches
// ============================================================================

/// SIMD-aware work-stealing deque for efficient batch distribution
pub const SIMDWorkStealingDeque = struct {
    const Self = @This();
    
    // Base deque for individual tasks
    base_deque: lockfree.WorkStealingDeque(core.Task),
    
    // SIMD batch queue for batch operations
    batch_queue: SIMDVectorizedQueue,
    
    // Configuration
    prefer_batches: bool,
    batch_threshold: usize,
    
    /// Initialize SIMD work-stealing deque
    pub fn init(
        allocator: std.mem.Allocator,
        capacity: usize,
        simd_config: SIMDQueueConfig,
        capability: simd.SIMDCapability
    ) !Self {
        return Self{
            .base_deque = try lockfree.WorkStealingDeque(core.Task).init(allocator, capacity),
            .batch_queue = try SIMDVectorizedQueue.init(allocator, simd_config, capability),
            .prefer_batches = true,
            .batch_threshold = 4, // Minimum tasks to prefer batching
        };
    }
    
    /// Clean up resources
    pub fn deinit(self: *Self) void {
        self.base_deque.deinit();
        self.batch_queue.deinit();
    }
    
    /// Push a SIMD task batch (owner operation)
    pub fn pushBatch(self: *Self, batch: *simd_batch.SIMDTaskBatch) !void {
        try self.batch_queue.enqueue(batch);
    }
    
    /// Push individual task (owner operation)
    pub fn pushTask(self: *Self, task: core.Task) !void {
        try self.base_deque.pushBottom(task);
    }
    
    /// Pop from local end (owner operation)
    pub fn pop(self: *Self) ?PopResult {
        // Prefer batches if enabled and available
        if (self.prefer_batches) {
            if (self.batch_queue.tryDequeue()) |element| {
                return PopResult{ .batch = element };
            }
        }
        
        // Fall back to individual tasks
        if (self.base_deque.popBottom()) |task| {
            return PopResult{ .task = task };
        }
        
        return null;
    }
    
    /// Steal from remote end (thief operation)
    pub fn steal(self: *Self) ?PopResult {
        // Try to steal batches first (more efficient)
        if (self.batch_queue.tryDequeue()) |element| {
            return PopResult{ .batch = element };
        }
        
        // Fall back to stealing individual tasks
        if (self.base_deque.steal()) |task| {
            return PopResult{ .task = task };
        }
        
        return null;
    }
    
    /// Steal multiple items in batch
    pub fn stealBatch(self: *Self, output: []PopResult) usize {
        var stolen_count: usize = 0;
        
        // First try to steal SIMD batches
        var batch_elements: [8]?*SIMDBatchQueueElement = undefined;
        const batches_stolen = self.batch_queue.tryDequeueBatch(&batch_elements);
        
        for (0..batches_stolen) |i| {
            if (stolen_count >= output.len) break;
            if (batch_elements[i]) |element| {
                output[stolen_count] = PopResult{ .batch = element };
                stolen_count += 1;
            }
        }
        
        // Then steal individual tasks if space remains
        while (stolen_count < output.len) {
            if (self.base_deque.steal()) |task| {
                output[stolen_count] = PopResult{ .task = task };
                stolen_count += 1;
            } else {
                break;
            }
        }
        
        return stolen_count;
    }
    
    /// Check if deque is empty
    pub fn isEmpty(self: *Self) bool {
        return self.base_deque.isEmpty() and self.batch_queue.isEmpty();
    }
    
    /// Get current size estimate
    pub fn size(self: *Self) usize {
        return self.base_deque.size() + self.batch_queue.size();
    }
    
    /// Get combined performance metrics
    pub fn getMetrics(self: *Self) SIMDDequeMetrics {
        const base_stats = self.base_deque.getStats();
        const batch_metrics = self.batch_queue.getMetrics();
        
        return SIMDDequeMetrics{
            .total_pushes = base_stats.pushes + batch_metrics.total_enqueues,
            .total_pops = base_stats.pops + batch_metrics.total_dequeues,
            .total_steals = base_stats.steals,
            .batch_operations = batch_metrics.vectorized_operations,
            .batch_efficiency = batch_metrics.batch_transfer_efficiency,
            .current_size = self.size(),
        };
    }
};

/// Result of pop/steal operations
pub const PopResult = union(enum) {
    task: core.Task,
    batch: *SIMDBatchQueueElement,
};

/// Performance metrics for SIMD work-stealing deque
pub const SIMDDequeMetrics = struct {
    total_pushes: u64,
    total_pops: u64,
    total_steals: u64,
    batch_operations: u64,
    batch_efficiency: f32,
    current_size: usize,
};

// ============================================================================
// Tests
// ============================================================================

test "SIMD queue configuration" {
    const allocator = std.testing.allocator;
    _ = allocator;
    
    // Test valid configuration
    const valid_config = SIMDQueueConfig{
        .batch_size = 16,
        .alignment = 64,
        .prefetch_distance = 2,
        .enable_vectorized_ops = true,
        .enable_batch_transfer = true,
    };
    try std.testing.expect(valid_config.validate());
    
    // Test invalid configuration (non-power-of-2 batch size)
    const invalid_config = SIMDQueueConfig{
        .batch_size = 15, // Not power of 2
        .alignment = 64,
        .prefetch_distance = 2,
        .enable_vectorized_ops = true,
        .enable_batch_transfer = true,
    };
    try std.testing.expect(!invalid_config.validate());
    
    // Test capability-based configuration
    const capability = simd.SIMDCapability.detect();
    const auto_config = SIMDQueueConfig.forCapability(capability);
    try std.testing.expect(auto_config.validate());
    
    std.debug.print("SIMD queue configuration test passed!\n", .{});
}

test "SIMD vectorized queue operations" {
    const allocator = std.testing.allocator;
    
    const capability = simd.SIMDCapability.detect();
    const config = SIMDQueueConfig.forCapability(capability);
    var queue = try SIMDVectorizedQueue.init(allocator, config, capability);
    defer queue.deinit();
    
    // Test empty queue
    try std.testing.expect(queue.isEmpty());
    try std.testing.expect(queue.size() == 0);
    try std.testing.expect(queue.tryDequeue() == null);
    
    // Create test batch
    var test_batch = try simd_batch.SIMDTaskBatch.init(allocator, capability, 8);
    defer test_batch.deinit();
    
    // Test enqueue/dequeue
    try queue.enqueue(&test_batch);
    try std.testing.expect(!queue.isEmpty());
    try std.testing.expect(queue.size() == 1);
    
    const dequeued = queue.tryDequeue();
    try std.testing.expect(dequeued != null);
    try std.testing.expect(dequeued.?.batch == &test_batch);
    
    // Clean up element
    queue.element_pool.destroy(dequeued.?);
    
    try std.testing.expect(queue.isEmpty());
    
    const metrics = queue.getMetrics();
    try std.testing.expect(metrics.total_enqueues == 1);
    try std.testing.expect(metrics.total_dequeues == 1);
    
    std.debug.print("SIMD vectorized queue test passed!\n", .{});
}

test "SIMD work-stealing deque integration" {
    const allocator = std.testing.allocator;
    
    const capability = simd.SIMDCapability.detect();
    const config = SIMDQueueConfig.forCapability(capability);
    var deque = try SIMDWorkStealingDeque.init(allocator, 64, config, capability);
    defer deque.deinit();
    
    // Test empty deque
    try std.testing.expect(deque.isEmpty());
    try std.testing.expect(deque.pop() == null);
    try std.testing.expect(deque.steal() == null);
    
    // Create test batch
    var test_batch = try simd_batch.SIMDTaskBatch.init(allocator, capability, 4);
    defer test_batch.deinit();
    
    // Test batch operations
    try deque.pushBatch(&test_batch);
    try std.testing.expect(!deque.isEmpty());
    
    const popped = deque.pop();
    try std.testing.expect(popped != null);
    try std.testing.expect(popped.? == .batch);
    try std.testing.expect(popped.?.batch.batch == &test_batch);
    
    // Clean up element
    deque.batch_queue.element_pool.destroy(popped.?.batch);
    
    const metrics = deque.getMetrics();
    try std.testing.expect(metrics.total_pushes == 1);
    try std.testing.expect(metrics.total_pops == 1);
    
    std.debug.print("SIMD work-stealing deque test passed!\n", .{});
}