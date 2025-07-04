const std = @import("std");
const core = @import("core.zig");
const build_opts = @import("build_config_unified.zig");

// Comptime Work Distribution Patterns for Beat.zig
// 
// This module provides compile-time work distribution strategies that leverage:
// - Zig's comptime metaprogramming capabilities
// - Auto-detected hardware configuration
// - Type-aware optimization decisions
// - SIMD-aware work chunking
// - Zero runtime overhead for distribution logic

// ============================================================================
// Core Work Distribution Framework
// ============================================================================

/// Work distribution strategy determined at compile time
pub const WorkStrategy = enum {
    sequential,        // Single-threaded execution
    parallel_simple,   // Basic parallel distribution
    parallel_chunked,  // Chunked parallel with optimal sizing
    parallel_simd,     // SIMD-aware parallel distribution
    parallel_numa,     // NUMA-aware distribution
    parallel_adaptive, // Dynamic distribution with prediction
};

/// Compile-time work analysis result
pub const WorkAnalysis = struct {
    strategy: WorkStrategy,
    chunk_size: comptime_int,
    worker_count: comptime_int,
    simd_width: comptime_int,
    memory_pattern: MemoryPattern,
    cost_estimate: comptime_int,
    
    pub const MemoryPattern = enum {
        sequential,    // Linear memory access
        random,        // Random memory access
        strided,       // Strided memory access
        hierarchical,  // Tree-like access pattern
    };
};

/// Analyze work characteristics at compile time
pub fn analyzeWork(
    comptime WorkType: type,
    comptime work_size: comptime_int,
    comptime available_workers: comptime_int,
) WorkAnalysis {
    // Type analysis
    const element_size = @sizeOf(WorkType);
    const is_vectorizable = build_opts.shouldVectorize(WorkType);
    const simd_width = if (is_vectorizable) build_opts.cpu_features.simd_width / element_size else 1;
    
    // Cost analysis
    const sequential_cost = work_size * element_size;
    const parallel_overhead = available_workers * 100; // Estimated overhead
    
    // Determine optimal strategy
    const strategy = comptime blk: {
        // Small work: stay sequential
        if (work_size < 1000) break :blk WorkStrategy.sequential;
        
        // Large work with SIMD support: use SIMD-aware
        if (work_size > 10000 and is_vectorizable) break :blk WorkStrategy.parallel_simd;
        
        // Medium work: chunked parallel
        if (work_size > 5000) break :blk WorkStrategy.parallel_chunked;
        
        // Default: simple parallel
        break :blk WorkStrategy.parallel_simple;
    };
    
    // Calculate optimal chunk size
    const chunk_size = comptime blk: {
        switch (strategy) {
            .sequential => break :blk work_size,
            .parallel_simple => break :blk @max(1, work_size / available_workers),
            .parallel_chunked => {
                // Balance between parallelism and overhead
                const ideal_chunks = available_workers * 4; // 4 chunks per worker
                break :blk @max(64, work_size / ideal_chunks);
            },
            .parallel_simd => {
                // Align to SIMD boundaries
                const base_chunk = work_size / (available_workers * 2);
                break :blk ((base_chunk + simd_width - 1) / simd_width) * simd_width;
            },
            else => break :blk @max(1, work_size / available_workers),
        }
    };
    
    // Estimate memory access pattern based on element size
    const memory_pattern = comptime blk: {
        if (element_size <= 8) {
            break :blk WorkAnalysis.MemoryPattern.sequential;
        } else {
            break :blk WorkAnalysis.MemoryPattern.random;
        }
    };
    
    return WorkAnalysis{
        .strategy = strategy,
        .chunk_size = chunk_size,
        .worker_count = @min(available_workers, @max(1, work_size / chunk_size)),
        .simd_width = simd_width,
        .memory_pattern = memory_pattern,
        .cost_estimate = sequential_cost + parallel_overhead,
    };
}

// ============================================================================
// Work Distribution Patterns
// ============================================================================

/// Generate optimal work distribution at compile time
pub fn WorkDistributor(
    comptime WorkType: type, 
    comptime work_size: comptime_int,
    comptime workers: comptime_int,
) type {
    const analysis = analyzeWork(WorkType, work_size, workers);
    
    return struct {
        const Self = @This();
        
        // Compile-time constants
        pub const strategy = analysis.strategy;
        pub const chunk_size = analysis.chunk_size;
        pub const worker_count = analysis.worker_count;
        pub const total_chunks = (work_size + chunk_size - 1) / chunk_size;
        pub const simd_width = analysis.simd_width;
        
        // Work chunk definition
        pub const WorkChunk = struct {
            start: usize,
            end: usize,
            worker_id: u32,
            simd_aligned: bool,
            
            pub fn size(self: WorkChunk) usize {
                return self.end - self.start;
            }
            
            pub fn isEmpty(self: WorkChunk) bool {
                return self.start >= self.end;
            }
        };
        
        /// Generate all work chunks at compile time
        pub const chunks: [total_chunks]WorkChunk = blk: {
            var result: [total_chunks]WorkChunk = undefined;
            var i: usize = 0;
            var pos: usize = 0;
            
            while (i < total_chunks and pos < work_size) : (i += 1) {
                const start = pos;
                const end = @min(work_size, pos + chunk_size);
                const worker_id = @as(u32, @intCast(i % worker_count));
                
                // Check SIMD alignment for vectorizable work
                const simd_aligned = if (strategy == .parallel_simd) 
                    (start % simd_width == 0) and ((end - start) % simd_width == 0)
                else 
                    false;
                
                result[i] = WorkChunk{
                    .start = start,
                    .end = end,
                    .worker_id = worker_id,
                    .simd_aligned = simd_aligned,
                };
                
                pos = end;
            }
            
            break :blk result;
        };
        
        /// Get chunks assigned to a specific worker
        pub fn getWorkerChunks(comptime worker_id: u32) []const WorkChunk {
            comptime {
                var worker_chunks: [total_chunks]WorkChunk = undefined;
                var count: usize = 0;
                
                for (chunks) |chunk| {
                    if (chunk.worker_id == worker_id) {
                        worker_chunks[count] = chunk;
                        count += 1;
                    }
                }
                
                return worker_chunks[0..count];
            }
        }
        
        /// Execute work distribution with a function
        pub fn execute(
            pool: *core.ThreadPool,
            data: []WorkType,
            comptime work_fn: fn([]WorkType, WorkChunk) void,
        ) !void {
            switch (strategy) {
                .sequential => {
                    const chunk = WorkChunk{
                        .start = 0,
                        .end = data.len,
                        .worker_id = 0,
                        .simd_aligned = false,
                    };
                    work_fn(data, chunk);
                },
                .parallel_simple, .parallel_chunked, .parallel_simd => {
                    // Submit chunks to thread pool
                    for (chunks) |chunk| {
                        if (chunk.isEmpty()) continue;
                        
                        const task = core.Task{
                            .func = struct {
                                fn execute_chunk(ctx: *anyopaque) void {
                                    const chunk_data = @as(*const struct {
                                        data: []WorkType,
                                        chunk: WorkChunk,
                                    }, @ptrCast(@alignCast(ctx)));
                                    
                                    work_fn(chunk_data.data, chunk_data.chunk);
                                }
                            }.execute_chunk,
                            .data = @ptrCast(&.{ .data = data, .chunk = chunk }),
                        };
                        
                        try pool.submit(task);
                    }
                    
                    // Wait for completion
                    pool.wait();
                },
                else => @compileError("Strategy not implemented: " ++ @tagName(strategy)),
            }
        }
        
        /// Get performance analysis
        pub fn getAnalysis() WorkAnalysis {
            return analysis;
        }
        
        /// Print distribution summary
        pub fn printSummary() void {
            std.debug.print("Work Distribution Summary:\n", .{});
            std.debug.print("  Strategy: {s}\n", .{@tagName(strategy)});
            std.debug.print("  Work Size: {}\n", .{work_size});
            std.debug.print("  Chunk Size: {}\n", .{chunk_size});
            std.debug.print("  Total Chunks: {}\n", .{total_chunks});
            std.debug.print("  Workers: {}\n", .{worker_count});
            
            if (strategy == .parallel_simd) {
                std.debug.print("  SIMD Width: {}\n", .{simd_width});
                
                var simd_chunks: usize = 0;
                for (chunks) |chunk| {
                    if (chunk.simd_aligned) simd_chunks += 1;
                }
                std.debug.print("  SIMD-Aligned Chunks: {}\n", .{simd_chunks});
            }
        }
    };
}

// ============================================================================
// High-Level Work Distribution API
// ============================================================================

/// Distribute work automatically with optimal configuration (runtime)
pub fn distributeWork(
    comptime WorkType: type,
    data: []WorkType,
    pool: *core.ThreadPool,
    comptime work_fn: fn([]WorkType, usize, usize) void,
) !void {
    // Runtime parallel work distribution
    const num_workers = pool.config.num_workers orelse 1;
    
    if (data.len < 1000 or num_workers == 1) {
        // Small data or single worker: run sequentially
        work_fn(data, 0, data.len);
        return;
    }
    
    // For simplicity, use pool.wait() approach
    // Calculate optimal chunk size based on data size and workers
    const base_chunk_size = @max(1, data.len / num_workers);
    const remainder = data.len % num_workers;
    
    // Simple approach: submit all chunks then wait for completion
    var current_start: usize = 0;
    for (0..num_workers) |worker_id| {
        const chunk_size = base_chunk_size + (if (worker_id < remainder) @as(usize, 1) else 0);
        const chunk_end = @min(current_start + chunk_size, data.len);
        
        if (current_start >= data.len) break;
        
        // Note: Full parallel implementation would submit chunks as tasks
        // For now, we'll use a simpler sequential approach to avoid lifetime complexity
        current_start = chunk_end;
    }
    
    // Fallback to sequential for now to avoid lifetime issues
    work_fn(data, 0, data.len);
}

/// Create a custom work distributor with specific parameters
pub fn createDistributor(
    comptime WorkType: type,
    comptime work_size: comptime_int,
    comptime workers: comptime_int,
) type {
    return WorkDistributor(WorkType, work_size, workers);
}

// ============================================================================
// Specialized Distribution Patterns
// ============================================================================

/// Map operation with automatic parallelization
pub fn parallelMap(
    comptime T: type,
    comptime U: type,
    pool: *core.ThreadPool,
    input: []const T,
    output: []U,
    comptime map_fn: fn(T) U,
) !void {
    if (input.len != output.len) {
        // parallelMap requires input and output arrays to have the same length
        // Input length: {}, Output length: {}
        // Help: Ensure both arrays have identical sizes before calling parallelMap
        // Example: var output: [input.len]OutputType = undefined;
        return error.ParallelMapArraySizeMismatch;
    }
    
    const num_workers = pool.config.num_workers orelse 1;
    
    if (input.len < 1000 or num_workers == 1) {
        // Small data or single worker: run sequentially
        for (input, 0..) |item, i| {
            output[i] = map_fn(item);
        }
        return;
    }
    
    // For now, fall back to sequential execution
    // Full parallel implementation would require careful context management
    for (input, 0..) |item, i| {
        output[i] = map_fn(item);
    }
}

/// Context for parallel reduce operation
fn ReduceContext(comptime T: type, comptime reduce_fn: fn(T, T) T) type {
    return struct {
        const Self = @This();
        
        partial_results: []T,
        initial: T,
        
        pub fn execute(self: *Self, work_data: []const T, chunk: anytype) void {
            var local_result = self.initial;
            for (work_data[chunk.start..chunk.end]) |item| {
                local_result = reduce_fn(local_result, item);
            }
            self.partial_results[chunk.worker_id] = local_result;
        }
    };
}

/// Reduce operation with sequential execution (for now)
pub fn parallelReduce(
    comptime T: type,
    pool: *core.ThreadPool,
    allocator: std.mem.Allocator,
    data: []const T,
    comptime reduce_fn: fn(T, T) T,
    initial: T,
) !T {
    if (data.len == 0) return initial;
    
    const num_workers = pool.config.num_workers orelse 1;
    
    if (data.len < 1000 or num_workers == 1) {
        // Small data or single worker: run sequentially
        var result = initial;
        for (data) |item| {
            result = reduce_fn(result, item);
        }
        return result;
    }
    
    // Parallel reduction approach
    // 1. Divide work among workers to compute partial results
    // 2. Combine partial results sequentially
    
    const effective_workers = @min(num_workers, data.len);
    const partial_results = try allocator.alloc(T, effective_workers);
    defer allocator.free(partial_results);
    
    // Initialize partial results
    for (partial_results) |*result| {
        result.* = initial;
    }
    
    // Calculate chunk sizes
    const base_chunk_size = data.len / effective_workers;
    const remainder = data.len % effective_workers;
    
    // Process chunks sequentially for now (parallel version would need careful synchronization)
    var current_start: usize = 0;
    for (0..effective_workers) |worker_id| {
        const chunk_size = base_chunk_size + (if (worker_id < remainder) @as(usize, 1) else 0);
        const chunk_end = @min(current_start + chunk_size, data.len);
        
        // Process this chunk
        for (data[current_start..chunk_end]) |item| {
            partial_results[worker_id] = reduce_fn(partial_results[worker_id], item);
        }
        
        current_start = chunk_end;
    }
    
    // Combine partial results
    var final_result = initial;
    for (partial_results) |partial| {
        final_result = reduce_fn(final_result, partial);
    }
    
    return final_result;
}

/// Filter operation with sequential execution (for now)
pub fn parallelFilter(
    comptime T: type,
    pool: *core.ThreadPool,
    allocator: std.mem.Allocator,
    input: []const T,
    comptime filter_fn: fn(T) bool,
) ![]T {
    const num_workers = pool.config.num_workers orelse 1;
    
    if (input.len < 1000 or num_workers == 1) {
        // Small data or single worker: run sequentially
        var filtered_items = std.ArrayList(T).init(allocator);
        defer filtered_items.deinit();
        
        for (input) |item| {
            if (filter_fn(item)) {
                try filtered_items.append(item);
            }
        }
        
        return try filtered_items.toOwnedSlice();
    }
    
    // Parallel filter approach
    // 1. Each worker processes a chunk and creates a local result list
    // 2. Combine all local results into final result
    
    const effective_workers = @min(num_workers, input.len);
    const base_chunk_size = input.len / effective_workers;
    const remainder = input.len % effective_workers;
    
    // Create partial result lists for each worker
    var partial_results = try allocator.alloc(std.ArrayList(T), effective_workers);
    defer {
        for (partial_results) |*list| {
            list.deinit();
        }
        allocator.free(partial_results);
    }
    
    // Initialize partial result lists
    for (partial_results) |*list| {
        list.* = std.ArrayList(T).init(allocator);
    }
    
    // Process chunks sequentially (parallel version would need careful coordination)
    var current_start: usize = 0;
    for (0..effective_workers) |worker_id| {
        const chunk_size = base_chunk_size + (if (worker_id < remainder) @as(usize, 1) else 0);
        const chunk_end = @min(current_start + chunk_size, input.len);
        
        // Process this chunk
        for (input[current_start..chunk_end]) |item| {
            if (filter_fn(item)) {
                try partial_results[worker_id].append(item);
            }
        }
        
        current_start = chunk_end;
    }
    
    // Combine partial results
    var total_count: usize = 0;
    for (partial_results) |*list| {
        total_count += list.items.len;
    }
    
    var final_result = try allocator.alloc(T, total_count);
    var write_index: usize = 0;
    
    for (partial_results) |*list| {
        @memcpy(final_result[write_index..write_index + list.items.len], list.items);
        write_index += list.items.len;
    }
    
    return final_result;
}

// ============================================================================
// SIMD-Aware Work Distribution
// ============================================================================

/// SIMD-optimized work distribution for vectorizable types
pub fn SimdDistributor(comptime T: type) type {
    const simd_width = build_opts.cpu_features.simd_width / @sizeOf(T);
    const VectorType = build_opts.OptimalVector(T);
    
    return struct {
        const Self = @This();
        
        pub const vector_width = simd_width;
        pub const Vector = VectorType;
        
        /// Process data with SIMD operations
        pub fn processSimd(
            pool: *core.ThreadPool,
            data: []T,
            comptime simd_fn: fn(Vector) Vector,
        ) !void {
            _ = pool; // For now, sequential implementation
            
            if (!build_opts.cpu_features.has_simd) {
                return error.SimdNotAvailable;
            }
            
            const elements_per_vector = @typeInfo(Vector).vector.len;
            var i: usize = 0;
            
            // Process full vectors
            while (i + elements_per_vector <= data.len) : (i += elements_per_vector) {
                const input_vector: Vector = data[i..i + elements_per_vector][0..elements_per_vector].*;
                const result_vector = simd_fn(input_vector);
                data[i..i + elements_per_vector][0..elements_per_vector].* = result_vector;
            }
            
            // Handle remaining elements scalar
            while (i < data.len) : (i += 1) {
                // Fallback to scalar operation
                const scalar_result = simd_fn(@splat(data[i]))[0];
                data[i] = scalar_result;
            }
        }
    };
}