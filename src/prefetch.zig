const std = @import("std");
const builtin = @import("builtin");

/// Memory prefetching utilities for Beat.zig
/// Provides cross-platform software prefetch hints to improve memory access patterns

/// Prefetch locality hints
pub const Locality = enum(u2) {
    /// Temporal locality: data will be accessed again soon
    /// Use for: frequently accessed data structures, hot loops
    temporal_high = 3,    // T0 - keep in all cache levels
    
    /// Moderate temporal locality: data might be accessed again
    /// Use for: moderately frequent access patterns
    temporal_medium = 2,  // T1 - keep in L2/L3 cache
    
    /// Low temporal locality: data unlikely to be reused soon
    /// Use for: one-time access patterns, streaming data
    temporal_low = 1,     // T2 - keep in L3 cache only
    
    /// No temporal locality: data will not be reused
    /// Use for: write-only operations, large sequential scans
    non_temporal = 0,     // NTA - don't pollute cache
};

/// Prefetch operation type
pub const Operation = enum {
    /// Prefetch for reading (most common)
    read,
    /// Prefetch for writing (when you know you'll modify the data)
    write,
};

/// Software prefetch hint - brings memory into cache before it's needed
/// 
/// Benefits:
/// - Reduces memory latency by overlapping computation with memory access
/// - Improves cache hit rates for predictable access patterns
/// - Helps with NUMA-aware memory access
/// 
/// Best practices:
/// - Prefetch 1-3 cache lines ahead of actual use
/// - Use temporal hints based on expected reuse patterns
/// - Don't over-prefetch (can evict useful data)
/// - Measure impact - not all workloads benefit equally
pub inline fn prefetch(ptr: anytype, comptime operation: Operation, comptime locality: Locality) void {
    const PtrType = @TypeOf(ptr);
    const ptr_info = @typeInfo(PtrType);
    
    // Ensure we have a pointer type
    if (ptr_info != .pointer) {
        @compileError("prefetch() requires a pointer argument");
    }
    
    // Convert to byte pointer for address calculation
    const addr = @as([*]const u8, @ptrCast(ptr));
    
    // Platform-specific prefetch implementation
    switch (builtin.cpu.arch) {
        .x86_64, .x86 => {
            // x86/x64 prefetch instructions
            switch (operation) {
                .read => {
                    switch (locality) {
                        .temporal_high => asm volatile ("prefetcht0 %[addr]" : : [addr] "m" (addr[0]) : "memory"),
                        .temporal_medium => asm volatile ("prefetcht1 %[addr]" : : [addr] "m" (addr[0]) : "memory"),
                        .temporal_low => asm volatile ("prefetcht2 %[addr]" : : [addr] "m" (addr[0]) : "memory"),
                        .non_temporal => asm volatile ("prefetchnta %[addr]" : : [addr] "m" (addr[0]) : "memory"),
                    }
                },
                .write => {
                    // x86 doesn't distinguish read/write prefetch, use read prefetch
                    switch (locality) {
                        .temporal_high => asm volatile ("prefetcht0 %[addr]" : : [addr] "m" (addr[0]) : "memory"),
                        .temporal_medium => asm volatile ("prefetcht1 %[addr]" : : [addr] "m" (addr[0]) : "memory"),
                        .temporal_low => asm volatile ("prefetcht2 %[addr]" : : [addr] "m" (addr[0]) : "memory"),
                        .non_temporal => asm volatile ("prefetchnta %[addr]" : : [addr] "m" (addr[0]) : "memory"),
                    }
                },
            }
        },
        .aarch64, .arm => {
            // ARM prefetch instructions
            switch (operation) {
                .read => {
                    switch (locality) {
                        .temporal_high => asm volatile ("prfm pldl1keep, %[addr]" : : [addr] "Q" (addr[0]) : "memory"),
                        .temporal_medium => asm volatile ("prfm pldl2keep, %[addr]" : : [addr] "Q" (addr[0]) : "memory"),
                        .temporal_low => asm volatile ("prfm pldl3keep, %[addr]" : : [addr] "Q" (addr[0]) : "memory"),
                        .non_temporal => asm volatile ("prfm pldl1strm, %[addr]" : : [addr] "Q" (addr[0]) : "memory"),
                    }
                },
                .write => {
                    switch (locality) {
                        .temporal_high => asm volatile ("prfm pstl1keep, %[addr]" : : [addr] "Q" (addr[0]) : "memory"),
                        .temporal_medium => asm volatile ("prfm pstl2keep, %[addr]" : : [addr] "Q" (addr[0]) : "memory"),
                        .temporal_low => asm volatile ("prfm pstl3keep, %[addr]" : : [addr] "Q" (addr[0]) : "memory"),
                        .non_temporal => asm volatile ("prfm pstl1strm, %[addr]" : : [addr] "Q" (addr[0]) : "memory"),
                    }
                },
            }
        },
        else => {
            // Fallback: no-op for unsupported architectures
            // The compiler will optimize this away as a no-op
        },
    }
}

/// Prefetch multiple cache lines sequentially
/// Useful for large data structures or arrays
pub inline fn prefetchRange(ptr: anytype, comptime operation: Operation, comptime locality: Locality, len_bytes: usize) void {
    const cache_line_size = 64; // Standard cache line size
    const addr = @as([*]const u8, @ptrCast(ptr));
    
    var offset: usize = 0;
    while (offset < len_bytes) : (offset += cache_line_size) {
        prefetch(addr + offset, operation, locality);
    }
}

/// Smart prefetch for arrays - prefetches based on access pattern
pub inline fn prefetchArray(comptime T: type, arr: []const T, index: usize, comptime ahead: usize, comptime locality: Locality) void {
    const element_size = @sizeOf(T);
    const cache_line_size = 64;
    const elements_per_line = cache_line_size / element_size;
    
    // Calculate prefetch target
    const prefetch_index = index + ahead * elements_per_line;
    
    if (prefetch_index < arr.len) {
        prefetch(&arr[prefetch_index], .read, locality);
    }
}

/// Prefetch for linked list traversal - prefetches the next node
pub inline fn prefetchNext(ptr: anytype, comptime next_field: []const u8, comptime locality: Locality) void {
    // Access the next field using reflection
    const T = @TypeOf(ptr.*);
    if (@hasField(T, next_field)) {
        const next_ptr = @field(ptr.*, next_field);
        if (next_ptr) |next| {
            prefetch(next, .read, locality);
        }
    }
}

/// Adaptive prefetch distance based on memory bandwidth
/// Returns optimal prefetch distance for current system
pub fn getOptimalPrefetchDistance() usize {
    // TODO: Could be made adaptive based on runtime measurement
    // For now, use conservative defaults
    
    switch (builtin.cpu.arch) {
        .x86_64, .x86 => {
            // x86 systems typically benefit from 1-2 cache lines ahead
            return 2;
        },
        .aarch64, .arm => {
            // ARM systems often have different cache hierarchies
            return 3;
        },
        else => {
            return 1; // Conservative default
        }
    }
}

/// Prefetch utilities for specific Beat.zig use cases

/// Prefetch for work-stealing operations
pub inline fn prefetchForWorkStealing(task_ptr: anytype) void {
    // Prefetch the task data for immediate execution
    prefetch(task_ptr, .read, .temporal_high);
    
    // If task has data pointer, prefetch that too
    if (@hasField(@TypeOf(task_ptr.*), "data")) {
        prefetch(task_ptr.*.data, .read, .temporal_medium);
    }
}

/// Prefetch for batch processing
pub inline fn prefetchBatch(tasks: anytype, start_index: usize, batch_size: usize) void {
    const end_index = @min(start_index + batch_size, tasks.len);
    
    for (start_index..end_index) |i| {
        prefetch(&tasks[i], .read, .temporal_medium);
        
        // Prefetch task data if it exists
        if (@hasField(@TypeOf(tasks[i]), "data")) {
            prefetch(tasks[i].data, .read, .temporal_low);
        }
    }
}

/// Prefetch for A3C neural network data
pub inline fn prefetchNeuralNetworkData(weights: anytype, index: usize) void {
    const ahead_distance = getOptimalPrefetchDistance();
    
    // Prefetch upcoming weights for matrix operations
    prefetchArray(@TypeOf(weights[0]), weights, index, ahead_distance, .temporal_high);
}

/// Test and validation utilities

/// Benchmark prefetch effectiveness
pub const PrefetchBenchmark = struct {
    const Self = @This();
    
    iterations: usize,
    data_size: usize,
    
    pub fn init(iterations: usize, data_size: usize) Self {
        return Self{
            .iterations = iterations,
            .data_size = data_size,
        };
    }
    
    /// Test memory access with and without prefetching
    pub fn measurePrefetchImpact(self: Self, allocator: std.mem.Allocator) !struct { without_prefetch: u64, with_prefetch: u64 } {
        const data = try allocator.alloc(u64, self.data_size);
        defer allocator.free(data);
        
        // Initialize data
        for (data, 0..) |*item, i| {
            item.* = i;
        }
        
        // Test without prefetching
        const start_without = std.time.nanoTimestamp();
        for (0..self.iterations) |_| {
            var sum: u64 = 0;
            for (data) |item| {
                sum +%= item;
            }
            std.mem.doNotOptimizeAway(&sum);
        }
        const end_without = std.time.nanoTimestamp();
        
        // Test with prefetching
        const start_with = std.time.nanoTimestamp();
        for (0..self.iterations) |_| {
            var sum: u64 = 0;
            for (data, 0..) |item, i| {
                // Prefetch ahead
                if (i + 8 < data.len) {
                    prefetch(&data[i + 8], .read, .temporal_medium);
                }
                sum +%= item;
            }
            std.mem.doNotOptimizeAway(&sum);
        }
        const end_with = std.time.nanoTimestamp();
        
        return .{
            .without_prefetch = @as(u64, @intCast(end_without - start_without)),
            .with_prefetch = @as(u64, @intCast(end_with - start_with)),
        };
    }
};