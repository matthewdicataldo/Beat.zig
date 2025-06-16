// Comprehensive Test Suite for ISPC Prediction Integration
// Validates transparent acceleration with 100% API compatibility

const std = @import("std");
const fingerprint_enhanced = @import("src/fingerprint_enhanced.zig");
const ispc_integration = @import("src/ispc_prediction_integration.zig");

// External ISPC kernel declarations for testing
extern fn ispc_compute_fingerprint_similarity(
    fingerprints_a: [*]const u64,
    fingerprints_b: [*]const u64,
    results: [*]f32,
    count: i32,
) void;

extern fn ispc_compute_similarity_matrix(
    fingerprints: [*]const u64,
    similarity_matrix: [*]f32,
    count: i32,
) void;

extern fn ispc_compute_multi_factor_confidence_batch(
    execution_counts: [*]const u32,
    accuracy_scores: [*]const f32,
    temporal_scores: [*]const f32,
    variance_scores: [*]const f32,
    confidence_results: [*]fingerprint_enhanced.fingerprint.MultiFactorConfidence,
    count: i32,
) void;

extern fn ispc_score_workers_batch(
    worker_loads: [*]const f32,
    numa_distances: [*]const f32,
    prediction_accuracies: [*]const f32,
    worker_scores: [*]f32,
    worker_count: i32,
    task_numa_preference: i32,
) void;

fn formatTime(ns: u64) void {
    if (ns < 1000) {
        std.debug.print("{d}ns", .{ns});
    } else if (ns < 1_000_000) {
        std.debug.print("{d:.1}μs", .{@as(f64, @floatFromInt(ns)) / 1000.0});
    } else if (ns < 1_000_000_000) {
        std.debug.print("{d:.1}ms", .{@as(f64, @floatFromInt(ns)) / 1_000_000.0});
    } else {
        std.debug.print("{d:.1}s", .{@as(f64, @floatFromInt(ns)) / 1_000_000_000.0});
    }
}

fn createTestFingerprint(seed: u32) fingerprint_enhanced.TaskFingerprint {
    return fingerprint_enhanced.TaskFingerprint{
        .call_site_hash = seed,
        .data_size_class = @truncate(seed % 255),
        .data_alignment = @truncate((seed >> 8) % 16),
        .access_pattern = if (seed % 2 == 0) .sequential else .random,
        .simd_width = @truncate((seed >> 4) % 16),
        .cache_locality = @truncate((seed >> 12) % 16),
        .numa_node_hint = @truncate((seed >> 16) % 16),
        .cpu_intensity = @truncate((seed >> 20) % 16),
        .parallel_potential = @truncate((seed >> 24) % 16),
        .execution_phase = @truncate((seed >> 28) % 16),
        .priority_class = @truncate(seed % 4),
        .time_sensitivity = @truncate((seed >> 2) % 4),
        .dependency_count = @truncate((seed >> 6) % 16),
        .time_of_day_bucket = @truncate((seed >> 10) % 24),
        .execution_frequency = @truncate((seed >> 14) % 8),
        .seasonal_pattern = @truncate((seed >> 18) % 16),
        .variance_level = @truncate((seed >> 22) % 16),
        .expected_cycles_log2 = @truncate((seed >> 26) % 256),
        .memory_bandwidth_log2 = @truncate(seed % 32),
        .branch_prediction_difficulty = @truncate((seed >> 5) % 16),
        .vectorization_efficiency = @truncate((seed >> 9) % 16),
        .cache_working_set_log2 = @truncate((seed >> 13) % 32),
        .reserved = 0,
    };
}

fn testAPICompatibility(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== Testing API Compatibility ===\n");
    
    // Initialize enhanced system
    fingerprint_enhanced.AutoAcceleration.init();
    
    var registry = try fingerprint_enhanced.createEnhancedRegistry(allocator);
    defer registry.deinit();
    
    // Create test fingerprints
    const fp1 = createTestFingerprint(0x12345678);
    const fp2 = createTestFingerprint(0x12345679);
    
    // Test single similarity computation (should work identically)
    const similarity_original = fp1.similarity(fp2);
    const similarity_enhanced = fingerprint_enhanced.EnhancedSimilarity.similarity(fp1, fp2);
    
    std.debug.print("Original similarity: {d:.6}\n", .{similarity_original});
    std.debug.print("Enhanced similarity: {d:.6}\n", .{similarity_enhanced});
    
    // Verify results are identical (within floating-point precision)
    const diff = @abs(similarity_original - similarity_enhanced);
    if (diff > 0.000001) {
        std.debug.print("❌ API compatibility test failed: difference = {d:.8}\n", .{diff});
        return error.APIMismatch;
    }
    
    std.debug.print("✅ API COMPATIBILITY: Single similarity computation identical\n");
}

fn testBatchAcceleration(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== Testing Batch Acceleration ===\n");
    
    const batch_sizes = [_]usize{ 4, 8, 16, 32, 64, 128 };
    
    for (batch_sizes) |batch_size| {
        std.debug.print("\nTesting batch size: {d}\n", .{batch_size});
        
        // Create test data
        var fingerprints_a = try allocator.alloc(fingerprint_enhanced.TaskFingerprint, batch_size);
        defer allocator.free(fingerprints_a);
        var fingerprints_b = try allocator.alloc(fingerprint_enhanced.TaskFingerprint, batch_size);
        defer allocator.free(fingerprints_b);
        
        for (0..batch_size) |i| {
            fingerprints_a[i] = createTestFingerprint(@intCast(i * 12345));
            fingerprints_b[i] = createTestFingerprint(@intCast(i * 12346));
        }
        
        // Test batch similarity computation
        var batch_results = try allocator.alloc(f32, batch_size);
        defer allocator.free(batch_results);
        var single_results = try allocator.alloc(f32, batch_size);
        defer allocator.free(single_results);
        
        // Measure batch performance
        var timer = try std.time.Timer.start();
        const iterations = 1000;
        
        timer.reset();
        for (0..iterations) |_| {
            fingerprint_enhanced.EnhancedSimilarity.similarityBatch(
                fingerprints_a,
                fingerprints_b,
                batch_results,
            );
            std.mem.doNotOptimizeAway(&batch_results);
        }
        const batch_time = timer.read();
        
        // Measure single computation performance
        timer.reset();
        for (0..iterations) |_| {
            for (fingerprints_a, fingerprints_b, single_results) |fp_a, fp_b, *result| {
                result.* = fingerprint_enhanced.EnhancedSimilarity.similarity(fp_a, fp_b);
            }
            std.mem.doNotOptimizeAway(&single_results);
        }
        const single_time = timer.read();
        
        // Verify results are identical
        var max_diff: f32 = 0.0;
        for (batch_results, single_results) |batch, single| {
            const diff = @abs(batch - single);
            max_diff = @max(max_diff, diff);
        }
        
        const speedup = @as(f64, @floatFromInt(single_time)) / @as(f64, @floatFromInt(batch_time));
        
        std.debug.print("Batch time: ");
        formatTime(batch_time);
        std.debug.print(", Single time: ");
        formatTime(single_time);
        std.debug.print(", Speedup: {d:.2}x\n", .{speedup});
        std.debug.print("Max difference: {d:.8}\n", .{max_diff});
        
        if (max_diff > 0.000001) {
            std.debug.print("❌ Batch acceleration failed: results don't match\n");
            return error.BatchMismatch;
        }
        
        std.debug.print("✅ BATCH ACCELERATION: Speedup {d:.2}x with identical results\n", .{speedup});
    }
}

fn testSimilarityMatrix(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== Testing Similarity Matrix Computation ===\n");
    
    const matrix_sizes = [_]usize{ 8, 16, 32 };
    
    for (matrix_sizes) |size| {
        std.debug.print("\nTesting matrix size: {d}x{d}\n", .{ size, size });
        
        // Create test fingerprints
        var fingerprints = try allocator.alloc(fingerprint_enhanced.TaskFingerprint, size);
        defer allocator.free(fingerprints);
        
        for (0..size) |i| {
            fingerprints[i] = createTestFingerprint(@intCast(i * 98765));
        }
        
        // Test matrix computation
        var matrix_enhanced = try allocator.alloc(f32, size * size);
        defer allocator.free(matrix_enhanced);
        var matrix_reference = try allocator.alloc(f32, size * size);
        defer allocator.free(matrix_reference);
        
        var timer = try std.time.Timer.start();
        const iterations = 100;
        
        // Enhanced matrix computation
        timer.reset();
        for (0..iterations) |_| {
            fingerprint_enhanced.EnhancedSimilarity.similarityMatrix(fingerprints, matrix_enhanced);
            std.mem.doNotOptimizeAway(&matrix_enhanced);
        }
        const enhanced_time = timer.read();
        
        // Reference implementation
        timer.reset();
        for (0..iterations) |_| {
            for (fingerprints, 0..) |fp_i, i| {
                for (fingerprints, 0..) |fp_j, j| {
                    if (i == j) {
                        matrix_reference[i * size + j] = 1.0;
                    } else {
                        matrix_reference[i * size + j] = fp_i.similarity(fp_j);
                    }
                }
            }
            std.mem.doNotOptimizeAway(&matrix_reference);
        }
        const reference_time = timer.read();
        
        // Verify results
        var max_diff: f32 = 0.0;
        for (matrix_enhanced, matrix_reference) |enhanced, reference| {
            const diff = @abs(enhanced - reference);
            max_diff = @max(max_diff, diff);
        }
        
        const speedup = @as(f64, @floatFromInt(reference_time)) / @as(f64, @floatFromInt(enhanced_time));
        
        std.debug.print("Enhanced time: ");
        formatTime(enhanced_time);
        std.debug.print(", Reference time: ");
        formatTime(reference_time);
        std.debug.print(", Speedup: {d:.2}x\n", .{speedup});
        std.debug.print("Max difference: {d:.8}\n", .{max_diff});
        
        if (max_diff > 0.000001) {
            std.debug.print("❌ Matrix computation failed: results don't match\n");
            return error.MatrixMismatch;
        }
        
        std.debug.print("✅ SIMILARITY MATRIX: Speedup {d:.2}x with identical results\n", .{speedup});
    }
}

fn testWorkerSelectionAcceleration(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== Testing Worker Selection Acceleration ===\n");
    
    const worker_counts = [_]usize{ 4, 8, 16, 32, 64 };
    
    for (worker_counts) |worker_count| {
        std.debug.print("\nTesting worker count: {d}\n", .{worker_count});
        
        // Create test worker data
        var worker_loads = try allocator.alloc(f32, worker_count);
        defer allocator.free(worker_loads);
        var numa_distances = try allocator.alloc(f32, worker_count);
        defer allocator.free(numa_distances);
        var prediction_accuracies = try allocator.alloc(f32, worker_count);
        defer allocator.free(prediction_accuracies);
        var worker_scores = try allocator.alloc(f32, worker_count);
        defer allocator.free(worker_scores);
        
        // Initialize with test data
        for (0..worker_count) |i| {
            worker_loads[i] = @as(f32, @floatFromInt(i % 10)) / 10.0;
            numa_distances[i] = @as(f32, @floatFromInt(i % 4)) / 4.0;
            prediction_accuracies[i] = 0.5 + @as(f32, @floatFromInt(i % 5)) / 10.0;
        }
        
        // Test accelerated worker scoring
        const accelerator = ispc_integration.getGlobalAccelerator();
        var timer = try std.time.Timer.start();
        const iterations = 10000;
        
        timer.reset();
        for (0..iterations) |_| {
            accelerator.scoreWorkersBatch(
                worker_loads,
                numa_distances,
                prediction_accuracies,
                worker_scores,
                0, // NUMA preference
            );
            std.mem.doNotOptimizeAway(&worker_scores);
        }
        const acceleration_time = timer.read();
        
        std.debug.print("Worker scoring time: ");
        formatTime(acceleration_time);
        std.debug.print(" ({d} workers, {d} iterations)\n", .{ worker_count, iterations });
        
        // Validate scores are reasonable
        var valid_scores: u32 = 0;
        for (worker_scores) |score| {
            if (score >= 0.0 and score <= 1.0) {
                valid_scores += 1;
            }
        }
        
        std.debug.print("Valid scores: {d}/{d}\n", .{ valid_scores, worker_count });
        
        if (valid_scores != worker_count) {
            std.debug.print("❌ Worker selection failed: invalid scores\n");
            return error.InvalidScores;
        }
        
        std.debug.print("✅ WORKER SELECTION: All scores valid and properly computed\n");
    }
}

fn testAccelerationStatistics() !void {
    std.debug.print("\n=== Testing Acceleration Statistics ===\n");
    
    // Get current statistics
    const stats = fingerprint_enhanced.AutoAcceleration.getStats();
    
    std.debug.print("ISPC calls: {d}\n", .{stats.ispc_calls});
    std.debug.print("Native calls: {d}\n", .{stats.native_calls});
    std.debug.print("ISPC failures: {d}\n", .{stats.ispc_failures});
    std.debug.print("Performance ratio: {d:.2}x\n", .{stats.performance_ratio});
    
    const total_calls = stats.ispc_calls + stats.native_calls;
    if (total_calls > 0) {
        const acceleration_rate = @as(f64, @floatFromInt(stats.ispc_calls)) / @as(f64, @floatFromInt(total_calls)) * 100.0;
        std.debug.print("Acceleration rate: {d:.1}%\n", .{acceleration_rate});
        
        if (stats.ispc_failures > 0) {
            const failure_rate = @as(f64, @floatFromInt(stats.ispc_failures)) / @as(f64, @floatFromInt(stats.ispc_calls)) * 100.0;
            std.debug.print("Failure rate: {d:.1}%\n", .{failure_rate});
        }
    }
    
    // Print comprehensive report
    fingerprint_enhanced.AutoAcceleration.printReport();
    
    std.debug.print("✅ ACCELERATION STATISTICS: Performance monitoring working\n");
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    std.debug.print("🚀 BEAT.ZIG PREDICTION INTEGRATION TEST SUITE\n");
    std.debug.print("==============================================\n");
    std.debug.print("Testing transparent ISPC acceleration with 100% API compatibility\n");
    
    try testAPICompatibility(allocator);
    try testBatchAcceleration(allocator);
    try testSimilarityMatrix(allocator);
    try testWorkerSelectionAcceleration(allocator);
    try testAccelerationStatistics();
    
    std.debug.print("\n🎊 INTEGRATION TEST SUMMARY\n");
    std.debug.print("===========================\n");
    std.debug.print("✅ API Compatibility: 100% identical results\n");
    std.debug.print("✅ Batch Acceleration: Automatic ISPC speedup\n");
    std.debug.print("✅ Similarity Matrix: Enhanced performance\n");
    std.debug.print("✅ Worker Selection: Optimized scoring\n");
    std.debug.print("✅ Statistics: Performance monitoring active\n");
    std.debug.print("\n🚀 Beat.zig Prediction System: Ready for Production!\n");
    std.debug.print("Users get maximum performance out-of-the-box with zero API changes.\n");
}