const std = @import("std");

// ============================================================================
// Comprehensive Allocator Error Injection for Beat.zig Fuzz Testing
//
// This module provides sophisticated allocator error injection capabilities
// to test OOM scenarios, allocation failures at critical points, and
// resource exhaustion conditions across the Beat.zig parallelism library.
//
// Key Features:
// - Deterministic failure injection for reproducible testing
// - Probabilistic failure simulation for realistic stress testing
// - Allocation tracking and leak detection
// - Critical path failure targeting
// - Resource exhaustion simulation
// ============================================================================

/// Fuzzing allocator configuration
pub const FuzzingAllocatorConfig = struct {
    // Failure injection modes
    fail_after_count: ?u64 = null,         // Fail after N allocations
    fail_probability: f32 = 0.0,           // Random failure probability (0.0-1.0)
    fail_size_threshold: ?usize = null,    // Fail allocations above size threshold
    fail_pattern: FailurePattern = .none,  // Systematic failure pattern
    
    // Resource limits
    max_total_allocated: ?usize = null,     // Maximum total memory allowed
    max_allocations: ?u64 = null,           // Maximum number of allocations
    
    // Tracking options
    track_allocations: bool = true,         // Track allocation statistics
    detect_leaks: bool = true,              // Enable leak detection
    log_failures: bool = false,             // Log allocation failures
    
    // Critical path targeting (Beat.zig specific)
    target_thread_pool_init: bool = false, // Target ThreadPool initialization
    target_worker_creation: bool = false,  // Target worker thread creation
    target_queue_operations: bool = false, // Target queue allocations
    target_numa_detection: bool = false,   // Target NUMA topology allocations
    target_telemetry: bool = false,         // Target OpenTelemetry allocations
};

/// Systematic failure patterns for comprehensive testing
pub const FailurePattern = enum {
    none,                    // No systematic failures
    every_nth,              // Fail every N-th allocation
    alternating,            // Alternating success/failure
    burst,                  // Bursts of failures followed by success periods
    exponential_backoff,    // Increasingly frequent failures
    random_walk,            // Random walk failure probability
    critical_path_only,     // Only fail on critical path allocations
};

/// Allocation tracking information
const AllocationInfo = struct {
    size: usize,
    timestamp: u64,
    stack_trace: ?[16]usize = null, // Simplified stack trace
    allocation_id: u64,
    is_critical_path: bool = false,
};

/// Fuzzing allocator with comprehensive error injection capabilities
pub const FuzzingAllocator = struct {
    base_allocator: std.mem.Allocator,
    config: FuzzingAllocatorConfig,
    
    // Allocation tracking
    allocations: std.AutoHashMap(usize, AllocationInfo),
    allocation_count: u64 = 0,
    total_allocated: usize = 0,
    total_freed: usize = 0,
    peak_allocated: usize = 0,
    
    // Failure injection state
    failure_count: u64 = 0,
    last_failure_allocation: u64 = 0,
    random: std.Random,
    prng: std.Random.DefaultPrng,
    
    // Critical path detection
    in_critical_path: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    
    const Self = @This();
    
    pub fn init(base_allocator: std.mem.Allocator, config: FuzzingAllocatorConfig) Self {
        var prng = std.Random.DefaultPrng.init(@intCast(std.time.nanoTimestamp()));
        
        return Self{
            .base_allocator = base_allocator,
            .config = config,
            .allocations = std.AutoHashMap(usize, AllocationInfo).init(base_allocator),
            .random = prng.random(),
            .prng = prng,
        };
    }
    
    pub fn deinit(self: *Self) void {
        // Check for memory leaks
        if (self.config.detect_leaks and self.allocations.count() > 0) {
            std.log.warn("FuzzingAllocator: {} allocations not freed (potential leaks)", .{self.allocations.count()});
            
            var iterator = self.allocations.iterator();
            while (iterator.next()) |entry| {
                const addr = entry.key_ptr.*;
                const info = entry.value_ptr.*;
                std.log.warn("  Leak: addr=0x{X}, size={}, id={}, critical={}", 
                    .{addr, info.size, info.allocation_id, info.is_critical_path});
            }
        }
        
        self.allocations.deinit();
        
        // Print final statistics
        if (self.config.track_allocations) {
            self.printStatistics();
        }
    }
    
    pub fn allocator(self: *Self) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .free = free,
            },
        };
    }
    
    /// Check if allocation should be failed based on injection strategy
    fn shouldFailAllocation(self: *Self, size: usize) bool {
        self.allocation_count += 1;
        
        // Check count-based failure
        if (self.config.fail_after_count) |fail_count| {
            if (self.allocation_count >= fail_count) {
                return true;
            }
        }
        
        // Check size threshold
        if (self.config.fail_size_threshold) |threshold| {
            if (size >= threshold) {
                return true;
            }
        }
        
        // Check total allocation limit
        if (self.config.max_total_allocated) |max_total| {
            if (self.total_allocated + size > max_total) {
                return true;
            }
        }
        
        // Check allocation count limit
        if (self.config.max_allocations) |max_allocs| {
            if (self.allocation_count > max_allocs) {
                return true;
            }
        }
        
        // Critical path targeting
        const is_critical = self.in_critical_path.load(.acquire);
        if (is_critical) {
            if (self.config.target_thread_pool_init or 
                self.config.target_worker_creation or
                self.config.target_queue_operations or
                self.config.target_numa_detection or
                self.config.target_telemetry) {
                // Higher failure rate for critical paths
                if (self.random.float(f32) < 0.3) { // 30% failure rate on critical paths
                    return true;
                }
            }
        }
        
        // Pattern-based failure
        switch (self.config.fail_pattern) {
            .none => {},
            .every_nth => {
                if (self.allocation_count % 10 == 0) return true;
            },
            .alternating => {
                if (self.allocation_count % 2 == 0) return true;
            },
            .burst => {
                // Burst of 3 failures every 20 allocations
                const pos = self.allocation_count % 20;
                if (pos >= 10 and pos < 13) return true;
            },
            .exponential_backoff => {
                const gap = @max(1, 50 / @max(1, self.failure_count));
                if (self.allocation_count - self.last_failure_allocation >= gap) {
                    return true;
                }
            },
            .random_walk => {
                // Random walk with configurable probability
                if (self.random.float(f32) < self.config.fail_probability) {
                    return true;
                }
            },
            .critical_path_only => {
                if (is_critical and self.random.float(f32) < 0.5) {
                    return true;
                }
            },
        }
        
        // Probabilistic failure
        if (self.config.fail_probability > 0.0) {
            if (self.random.float(f32) < self.config.fail_probability) {
                return true;
            }
        }
        
        return false;
    }
    
    fn alloc(ctx: *anyopaque, len: usize, log2_ptr_align: std.mem.Alignment, return_address: usize) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        
        // Check if we should fail this allocation
        if (self.shouldFailAllocation(len)) {
            self.failure_count += 1;
            self.last_failure_allocation = self.allocation_count;
            
            if (self.config.log_failures) {
                std.log.debug("FuzzingAllocator: Injected allocation failure (size={}, count={})", 
                    .{len, self.allocation_count});
            }
            
            return null;
        }
        
        // Perform the actual allocation
        const result = self.base_allocator.rawAlloc(len, log2_ptr_align, return_address);
        
        if (result) |ptr| {
            // Track the allocation
            if (self.config.track_allocations) {
                const addr = @intFromPtr(ptr);
                const info = AllocationInfo{
                    .size = len,
                    .timestamp = @intCast(std.time.nanoTimestamp()),
                    .allocation_id = self.allocation_count,
                    .is_critical_path = self.in_critical_path.load(.acquire),
                };
                
                self.allocations.put(addr, info) catch {
                    // If we can't track the allocation, continue anyway
                    std.log.warn("FuzzingAllocator: Failed to track allocation", .{});
                };
                
                self.total_allocated += len;
                self.peak_allocated = @max(self.peak_allocated, self.total_allocated);
            }
        }
        
        return result;
    }
    
    fn resize(ctx: *anyopaque, buf: []u8, log2_buf_align: std.mem.Alignment, new_len: usize, return_address: usize) bool {
        const self: *Self = @ptrCast(@alignCast(ctx));
        
        // For resize, we might want to fail based on the size increase
        if (new_len > buf.len) {
            const size_increase = new_len - buf.len;
            if (self.shouldFailAllocation(size_increase)) {
                self.failure_count += 1;
                
                if (self.config.log_failures) {
                    std.log.debug("FuzzingAllocator: Injected resize failure (old={}, new={})", 
                        .{buf.len, new_len});
                }
                
                return false;
            }
        }
        
        // Update tracking information
        if (self.config.track_allocations) {
            const addr = @intFromPtr(buf.ptr);
            if (self.allocations.getPtr(addr)) |info| {
                if (new_len > buf.len) {
                    self.total_allocated += new_len - buf.len;
                } else {
                    self.total_allocated -= buf.len - new_len;
                }
                info.size = new_len;
                self.peak_allocated = @max(self.peak_allocated, self.total_allocated);
            }
        }
        
        return self.base_allocator.rawResize(buf, log2_buf_align, new_len, return_address);
    }
    
    fn free(ctx: *anyopaque, buf: []u8, log2_buf_align: std.mem.Alignment, return_address: usize) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        
        // Update tracking
        if (self.config.track_allocations) {
            const addr = @intFromPtr(buf.ptr);
            if (self.allocations.fetchRemove(addr)) |kv| {
                self.total_allocated -= kv.value.size;
                self.total_freed += kv.value.size;
            } else {
                std.log.warn("FuzzingAllocator: Attempt to free untracked allocation at 0x{X}", .{addr});
            }
        }
        
        self.base_allocator.rawFree(buf, log2_buf_align, return_address);
    }
    
    /// Mark the beginning of a critical path for targeted failure injection
    pub fn enterCriticalPath(self: *Self) void {
        self.in_critical_path.store(true, .release);
    }
    
    /// Mark the end of a critical path
    pub fn exitCriticalPath(self: *Self) void {
        self.in_critical_path.store(false, .release);
    }
    
    /// RAII wrapper for critical path sections
    pub const CriticalPathGuard = struct {
        allocator: *FuzzingAllocator,
        
        pub fn init(fuzzing_allocator: *FuzzingAllocator) CriticalPathGuard {
            fuzzing_allocator.enterCriticalPath();
            return CriticalPathGuard{ .allocator = fuzzing_allocator };
        }
        
        pub fn deinit(self: CriticalPathGuard) void {
            self.allocator.exitCriticalPath();
        }
    };
    
    /// Get allocation statistics
    pub fn getStatistics(self: *const Self) struct {
        allocations_attempted: u64,
        allocations_succeeded: u64,
        allocation_failures: u64,
        total_allocated: usize,
        total_freed: usize,
        peak_allocated: usize,
        current_allocated: usize,
        active_allocations: u32,
        failure_rate: f32,
    } {
        const succeeded = self.allocation_count - self.failure_count;
        const failure_rate = if (self.allocation_count > 0) 
            @as(f32, @floatFromInt(self.failure_count)) / @as(f32, @floatFromInt(self.allocation_count))
        else 
            0.0;
        
        return .{
            .allocations_attempted = self.allocation_count,
            .allocations_succeeded = succeeded,
            .allocation_failures = self.failure_count,
            .total_allocated = self.total_allocated,
            .total_freed = self.total_freed,
            .peak_allocated = self.peak_allocated,
            .current_allocated = self.total_allocated - self.total_freed,
            .active_allocations = @intCast(self.allocations.count()),
            .failure_rate = failure_rate,
        };
    }
    
    /// Print detailed allocation statistics
    pub fn printStatistics(self: *const Self) void {
        const stats = self.getStatistics();
        
        std.debug.print("\n=== FuzzingAllocator Statistics ===\n", .{});
        std.debug.print("Allocations attempted: {}\n", .{stats.allocations_attempted});
        std.debug.print("Allocations succeeded: {}\n", .{stats.allocations_succeeded});
        std.debug.print("Allocation failures: {} ({d:.1}%)\n", .{stats.allocation_failures, stats.failure_rate * 100.0});
        std.debug.print("Total allocated: {} bytes\n", .{stats.total_allocated});
        std.debug.print("Total freed: {} bytes\n", .{stats.total_freed});
        std.debug.print("Peak allocated: {} bytes\n", .{stats.peak_allocated});
        std.debug.print("Current allocated: {} bytes\n", .{stats.current_allocated});
        std.debug.print("Active allocations: {}\n", .{stats.active_allocations});
        std.debug.print("====================================\n", .{});
    }
    
    /// Reset statistics and failure injection state
    pub fn reset(self: *Self) void {
        self.allocations.clearAndFree();
        self.allocation_count = 0;
        self.total_allocated = 0;
        self.total_freed = 0;
        self.peak_allocated = 0;
        self.failure_count = 0;
        self.last_failure_allocation = 0;
        self.in_critical_path.store(false, .release);
    }
};

// ============================================================================
// Convenience Functions for Common Fuzzing Scenarios
// ============================================================================

/// Create allocator that fails after a specific number of allocations
pub fn createCountBasedFuzzingAllocator(base: std.mem.Allocator, fail_after: u64) FuzzingAllocator {
    return FuzzingAllocator.init(base, .{
        .fail_after_count = fail_after,
        .track_allocations = true,
        .detect_leaks = true,
    });
}

/// Create allocator that fails with a specific probability
pub fn createProbabilisticFuzzingAllocator(base: std.mem.Allocator, probability: f32) FuzzingAllocator {
    return FuzzingAllocator.init(base, .{
        .fail_probability = probability,
        .fail_pattern = .random_walk,
        .track_allocations = true,
        .detect_leaks = true,
    });
}

/// Create allocator that targets critical path allocations
pub fn createCriticalPathFuzzingAllocator(base: std.mem.Allocator) FuzzingAllocator {
    return FuzzingAllocator.init(base, .{
        .fail_pattern = .critical_path_only,
        .target_thread_pool_init = true,
        .target_worker_creation = true,
        .target_queue_operations = true,
        .target_numa_detection = true,
        .target_telemetry = true,
        .track_allocations = true,
        .detect_leaks = true,
        .log_failures = true,
    });
}

/// Create allocator for resource exhaustion testing
pub fn createResourceExhaustionAllocator(base: std.mem.Allocator, max_memory: usize) FuzzingAllocator {
    return FuzzingAllocator.init(base, .{
        .max_total_allocated = max_memory,
        .max_allocations = 1000, // Limit number of allocations too
        .track_allocations = true,
        .detect_leaks = true,
        .log_failures = true,
    });
}

// ============================================================================
// Testing
// ============================================================================

test "FuzzingAllocator basic functionality" {
    var fuzz_alloc = createCountBasedFuzzingAllocator(std.testing.allocator, 5);
    defer fuzz_alloc.deinit();
    
    const allocator = fuzz_alloc.allocator();
    
    // First 4 allocations should succeed
    var ptrs: [4][]u8 = undefined;
    for (&ptrs) |*ptr| {
        ptr.* = try allocator.alloc(u8, 100);
    }
    
    // 5th allocation should fail
    try std.testing.expectError(error.OutOfMemory, allocator.alloc(u8, 100));
    
    // Free the allocations
    for (ptrs) |ptr| {
        allocator.free(ptr);
    }
    
    const stats = fuzz_alloc.getStatistics();
    try std.testing.expect(stats.allocation_failures > 0);
}

test "FuzzingAllocator critical path targeting" {
    var fuzz_alloc = createCriticalPathFuzzingAllocator(std.testing.allocator);
    defer fuzz_alloc.deinit();
    
    const allocator = fuzz_alloc.allocator();
    
    // Normal allocations might succeed
    const ptr1 = allocator.alloc(u8, 100) catch null;
    if (ptr1) |p| allocator.free(p);
    
    // Critical path allocations are more likely to fail
    {
        const guard = FuzzingAllocator.CriticalPathGuard.init(&fuzz_alloc);
        defer guard.deinit();
        
        // Multiple attempts to see failure injection
        var failures: u32 = 0;
        for (0..10) |_| {
            if (allocator.alloc(u8, 100)) |ptr| {
                allocator.free(ptr);
            } else |_| {
                failures += 1;
            }
        }
        
        // Should have some failures in critical path
        try std.testing.expect(failures > 0);
    }
}