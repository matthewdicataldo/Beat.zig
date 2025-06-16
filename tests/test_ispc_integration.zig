// ISPC Integration Test for Beat.zig
// Tests the integration of Intel SPMD Program Compiler with Beat.zig's task processing
// Validates performance improvements and correctness of ISPC kernels

const std = @import("std");
const testing = std.testing;
const beat = @import("beat");
// Note: ISPC integration functions are available via C headers

// Import ISPC-generated headers via C interface
const ispc_kernels = @cImport({
    @cInclude("fingerprint_similarity.h");
    @cInclude("batch_optimization.h");
    @cInclude("worker_selection.h");
});

test "ISPC integration: fingerprint similarity computation" {
    const allocator = testing.allocator;
    
    // Test data: 8 pairs of 128-bit fingerprints
    const fingerprint_count = 8;
    var fingerprints_a = try allocator.alloc(u128, fingerprint_count);
    defer allocator.free(fingerprints_a);
    var fingerprints_b = try allocator.alloc(u128, fingerprint_count);
    defer allocator.free(fingerprints_b);
    
    // Initialize test fingerprints
    for (0..fingerprint_count) |i| {
        fingerprints_a[i] = @as(u128, @intCast(i)) << 64 | @as(u128, @intCast(i * 2));
        fingerprints_b[i] = @as(u128, @intCast(i)) << 64 | @as(u128, @intCast(i * 3));
    }
    
    // Native Zig implementation for comparison
    var native_results = try allocator.alloc(f32, fingerprint_count);
    defer allocator.free(native_results);
    
    var timer = try std.time.Timer.start();
    
    // Benchmark native implementation
    timer.reset();
    for (0..fingerprint_count) |i| {
        const diff = fingerprints_a[i] ^ fingerprints_b[i];
        const hamming_distance = @popCount(diff);
        native_results[i] = 1.0 - (@as(f32, @floatFromInt(hamming_distance)) / 128.0);
    }
    const native_time = timer.read();
    
    // ISPC implementation
    const ispc_results = try allocator.alloc(f32, fingerprint_count);
    defer allocator.free(ispc_results);
    
    timer.reset();
    ispc_kernels.ispc_compute_fingerprint_similarity(
        @as([*c]u64, @ptrCast(@constCast(fingerprints_a.ptr))),
        @as([*c]u64, @ptrCast(@constCast(fingerprints_b.ptr))),
        ispc_results.ptr,
        @intCast(fingerprint_count),
    );
    const ispc_time = timer.read();
    
    // Verify correctness: results should be nearly identical
    for (0..fingerprint_count) |i| {
        const diff = @abs(native_results[i] - ispc_results[i]);
        try testing.expect(diff < 0.001); // Allow small floating-point differences
    }
    
    // Performance validation
    const speedup = @as(f32, @floatFromInt(native_time)) / @as(f32, @floatFromInt(ispc_time));
    std.debug.print("Fingerprint similarity ISPC speedup: {d:.2}x\n", .{speedup});
    
    // Expect at least 2x speedup (conservative estimate)
    try testing.expect(speedup >= 2.0);
}

test "ISPC integration: batch formation optimization" {
    const allocator = testing.allocator;
    
    const task_count = 16;
    const max_batch_size = 8;
    
    // Generate test data for batch optimization
    var task_scores = try allocator.alloc(f32, task_count);
    defer allocator.free(task_scores);
    var similarity_matrix = try allocator.alloc(f32, task_count * task_count);
    defer allocator.free(similarity_matrix);
    
    // Initialize with realistic test data
    for (0..task_count) |i| {
        task_scores[i] = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(task_count)) + 0.1;
        
        for (0..task_count) |j| {
            const similarity = if (i == j) 
                1.0 
            else 
                0.8 - @abs(@as(f32, @floatFromInt(i)) - @as(f32, @floatFromInt(j))) / @as(f32, @floatFromInt(task_count));
            similarity_matrix[i * task_count + j] = similarity;
        }
    }
    
    // Test ISPC batch optimization  
    const batch_indices = try allocator.alloc(i32, max_batch_size);
    defer allocator.free(batch_indices);
    
    var timer = try std.time.Timer.start();
    const actual_batch_size = ispc_kernels.ispc_optimize_batch_formation(
        task_scores.ptr,
        similarity_matrix.ptr,
        batch_indices.ptr,
        @intCast(task_count),
        @intCast(max_batch_size),
    );
    const ispc_time = timer.read();
    
    // Validate results
    try testing.expect(actual_batch_size > 0);
    try testing.expect(actual_batch_size <= max_batch_size);
    
    // Verify batch contains valid task indices
    for (0..@intCast(actual_batch_size)) |i| {
        try testing.expect(batch_indices[i] < task_count);
    }
    
    std.debug.print("ISPC batch formation time: {d}ns\n", .{ispc_time});
    std.debug.print("Batch size: {d}/{d} tasks\n", .{ actual_batch_size, max_batch_size });
}

test "ISPC integration: worker selection scoring" {
    const allocator = testing.allocator;
    
    const worker_count = 12;
    
    // Generate test worker data
    var worker_loads = try allocator.alloc(f32, worker_count);
    defer allocator.free(worker_loads);
    var numa_distances = try allocator.alloc(f32, worker_count);
    defer allocator.free(numa_distances);
    var cache_affinities = try allocator.alloc(f32, worker_count);
    defer allocator.free(cache_affinities);
    const worker_scores = try allocator.alloc(f32, worker_count);
    defer allocator.free(worker_scores);
    
    // Initialize realistic worker metrics
    for (0..worker_count) |i| {
        worker_loads[i] = @as(f32, @floatFromInt(i % 4)) / 4.0; // 0-3 load levels
        numa_distances[i] = @as(f32, @floatFromInt(i / 4)) / 3.0; // NUMA distances
        cache_affinities[i] = 0.8 + (@as(f32, @floatFromInt(i % 3)) / 10.0); // Cache affinities
    }
    
    // Benchmark native vs ISPC worker selection
    var native_scores = try allocator.alloc(f32, worker_count);
    defer allocator.free(native_scores);
    
    var timer = try std.time.Timer.start();
    
    // Native implementation
    timer.reset();
    for (0..worker_count) |i| {
        const load_score = 1.0 - std.math.clamp(worker_loads[i], 0.0, 1.0);
        const numa_score = 1.0 - std.math.clamp(numa_distances[i], 0.0, 1.0);
        const cache_score = std.math.clamp(cache_affinities[i], 0.0, 1.0);
        
        native_scores[i] = 0.4 * load_score * load_score + 
                          0.35 * std.math.sqrt(numa_score) + 
                          0.25 * cache_score;
    }
    const native_time = timer.read();
    
    // ISPC implementation
    timer.reset();
    ispc_kernels.ispc_compute_worker_scores(
        worker_loads.ptr,
        numa_distances.ptr,
        cache_affinities.ptr,
        worker_scores.ptr,
        @intCast(worker_count),
    );
    const ispc_time = timer.read();
    
    // Verify score validity
    for (0..worker_count) |i| {
        try testing.expect(worker_scores[i] >= 0.0);
        try testing.expect(worker_scores[i] <= 10.0); // Allow for exponential scaling
        
        // Scores should be reasonably close to native implementation
        const expected_exp = std.math.exp(2.0 * native_scores[i]) - 1.0;
        const diff = @abs(worker_scores[i] - expected_exp);
        try testing.expect(diff < 0.1);
    }
    
    const speedup = @as(f32, @floatFromInt(native_time)) / @as(f32, @floatFromInt(ispc_time));
    std.debug.print("Worker selection ISPC speedup: {d:.2}x\n", .{speedup});
    
    // Expect meaningful speedup for parallel computation
    try testing.expect(speedup >= 1.5);
}

test "ISPC integration: similarity matrix computation" {
    const allocator = testing.allocator;
    
    const fingerprint_count = 8;
    var fingerprints = try allocator.alloc(u128, fingerprint_count);
    defer allocator.free(fingerprints);
    const similarity_matrix = try allocator.alloc(f32, fingerprint_count * fingerprint_count);
    defer allocator.free(similarity_matrix);
    
    // Initialize test fingerprints
    for (0..fingerprint_count) |i| {
        fingerprints[i] = (@as(u128, @intCast(i)) << 32) | @as(u128, @intCast(i * i));
    }
    
    // Compute similarity matrix using ISPC
    var timer = try std.time.Timer.start();
    ispc_kernels.ispc_compute_similarity_matrix(
        @as([*]u64, @ptrCast(fingerprints.ptr)),
        similarity_matrix.ptr,
        @intCast(fingerprint_count),
    );
    const ispc_time = timer.read();
    
    // Verify matrix properties
    for (0..fingerprint_count) |i| {
        for (0..fingerprint_count) |j| {
            const similarity = similarity_matrix[i * fingerprint_count + j];
            
            // Diagonal should be 1.0 (self-similarity)
            if (i == j) {
                try testing.expectApproxEqAbs(@as(f32, 1.0), similarity, 0.001);
            } else {
                // Non-diagonal should be symmetric
                const symmetric = similarity_matrix[j * fingerprint_count + i];
                try testing.expectApproxEqAbs(similarity, symmetric, 0.001);
                
                // Should be between 0 and 1
                try testing.expect(similarity >= 0.0 and similarity <= 1.0);
            }
        }
    }
    
    std.debug.print("ISPC similarity matrix computation: {d}ns\n", .{ispc_time});
}

// Configuration and wrapper tests disabled for now - 
// these require the full ispc integration module which has import path issues in tests
// The core ISPC kernel functionality is validated in the tests above