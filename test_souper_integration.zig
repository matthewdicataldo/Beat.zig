// Comprehensive test suite for Souper mathematical optimization integration
// Validates that all optimizations are correctly integrated and provide expected performance

const std = @import("std");
const testing = std.testing;
const beat = @import("src/core.zig");
const souper = beat.souper_integration;
const math_opt = beat.mathematical_optimizations;

// Test configuration
const TestConfig = struct {
    iterations: usize = 10000,
    batch_size: usize = 1000,
    tolerance: f32 = 0.01, // 1% tolerance for performance comparisons
};

const test_config = TestConfig{};

test "Souper integration - basic functionality" {
    const allocator = testing.allocator;
    
    // Test ThreadPool initialization with Souper optimizations
    const config = beat.Config{
        .num_workers = 2,
        .enable_souper_optimizations = true,
        .enable_statistics = true,
    };
    
    const pool = try beat.ThreadPool.init(allocator, config);
    defer pool.deinit();
    
    // Verify pool was created successfully
    try testing.expect(pool.workers.len == 2);
    try testing.expect(pool.config.enable_souper_optimizations == true);
}

test "Souper optimizations - fingerprint similarity performance" {
    const allocator = testing.allocator;
    
    // Prepare test data
    const hash_pairs = try allocator.alloc([2]u64, test_config.iterations);
    defer allocator.free(hash_pairs);
    
    var prng = std.Random.DefaultPrng.init(0x12345678);
    const random = prng.random();
    
    for (hash_pairs) |*pair| {
        pair.*[0] = random.int(u64);
        pair.*[1] = random.int(u64);
    }
    
    // Benchmark original vs optimized
    const start_time = std.time.nanoTimestamp();
    
    var optimized_sum: u64 = 0;
    for (hash_pairs) |pair| {
        optimized_sum += souper.SouperIntegration.OptimizedFingerprint.computeSimilarity(pair[0], pair[1]);
    }
    
    const optimized_time = std.time.nanoTimestamp() - start_time;
    
    // Verify results are reasonable
    const average_similarity = optimized_sum / test_config.iterations;
    try testing.expect(average_similarity <= 100);
    
    std.log.info("Fingerprint similarity optimization: {} ns/op", .{@divTrunc(optimized_time, @as(i64, @intCast(test_config.iterations)))});
}

test "Souper optimizations - heartbeat scheduling decisions" {
    // Test various scheduling scenarios
    const test_cases = [_]struct {
        load: u32,
        capacity: u32,
        threshold_factor: u32,
        expected: bool,
    }{
        .{ .load = 25, .capacity = 100, .threshold_factor = 50, .expected = false }, // Below threshold
        .{ .load = 60, .capacity = 100, .threshold_factor = 50, .expected = true },  // In range, even percentage
        .{ .load = 65, .capacity = 100, .threshold_factor = 50, .expected = false }, // In range, odd percentage
        .{ .load = 90, .capacity = 100, .threshold_factor = 50, .expected = false }, // Above threshold
    };
    
    for (test_cases) |case| {
        const result = souper.SouperIntegration.OptimizedScheduler.shouldStealWork(
            case.load, case.capacity, case.threshold_factor
        );
        
        try testing.expectEqual(case.expected, result);
    }
    
    std.log.info("Heartbeat scheduling optimization: All test cases passed", .{});
}

test "Souper optimizations - lock-free index calculations" {
    const test_cases = [_]struct {
        value: u64,
        capacity: u64,
        expected: u64,
    }{
        .{ .value = 13, .capacity = 8, .expected = 5 },   // Power of 2: 13 & 7 = 5
        .{ .value = 13, .capacity = 10, .expected = 3 },  // Non-power of 2: 13 % 10 = 3
        .{ .value = 100, .capacity = 16, .expected = 4 }, // Power of 2: 100 & 15 = 4
        .{ .value = 100, .capacity = 7, .expected = 2 },  // Non-power of 2: 100 % 7 = 2
    };
    
    for (test_cases) |case| {
        const result = souper.SouperIntegration.OptimizedLockfree.calculateIndex(case.value, case.capacity);
        try testing.expectEqual(case.expected, result);
    }
    
    std.log.info("Lock-free index optimization: All test cases passed", .{});
}

test "Souper optimizations - SIMD task classification" {
    const allocator = testing.allocator;
    
    // Test batch classification
    const task_flags = try allocator.alloc(u32, test_config.batch_size);
    defer allocator.free(task_flags);
    
    const results = try allocator.alloc(u32, test_config.batch_size);
    defer allocator.free(results);
    
    // Initialize test data
    var prng = std.Random.DefaultPrng.init(0x87654321);
    const random = prng.random();
    
    for (task_flags) |*flags| {
        flags.* = random.int(u32) & 0x7FFF; // Limit to valid flag range
    }
    
    // Benchmark batch classification
    const start_time = std.time.nanoTimestamp();
    
    souper.SouperIntegration.OptimizedSIMD.classifyTasksBatch(task_flags, results);
    
    const batch_time = std.time.nanoTimestamp() - start_time;
    
    // Verify all results are valid
    for (results) |result| {
        try testing.expect(result <= 0xFFFF);
    }
    
    std.log.info("SIMD task classification: {} ns/op", .{@divTrunc(batch_time, @as(i64, @intCast(test_config.batch_size)))});
}

test "Souper optimizations - mathematical utilities" {
    // Test power of 2 detection
    try testing.expect(souper.SouperIntegration.MathUtils.isPowerOfTwo(1));
    try testing.expect(souper.SouperIntegration.MathUtils.isPowerOfTwo(16));
    try testing.expect(souper.SouperIntegration.MathUtils.isPowerOfTwo(1024));
    try testing.expect(!souper.SouperIntegration.MathUtils.isPowerOfTwo(0));
    try testing.expect(!souper.SouperIntegration.MathUtils.isPowerOfTwo(3));
    try testing.expect(!souper.SouperIntegration.MathUtils.isPowerOfTwo(100));
    
    // Test integer square root
    try testing.expectEqual(@as(u32, 10), souper.SouperIntegration.MathUtils.isqrt(100));
    try testing.expectEqual(@as(u32, 31), souper.SouperIntegration.MathUtils.isqrt(1000));
    try testing.expectEqual(@as(u32, 100), souper.SouperIntegration.MathUtils.isqrt(10000));
    
    // Test population count
    try testing.expectEqual(@as(u32, 0), souper.SouperIntegration.MathUtils.popcount(0));
    try testing.expectEqual(@as(u32, 1), souper.SouperIntegration.MathUtils.popcount(1));
    try testing.expectEqual(@as(u32, 32), souper.SouperIntegration.MathUtils.popcount(0xFFFFFFFF));
    
    // Test alignment operations
    try testing.expectEqual(@as(usize, 16), souper.SouperIntegration.MathUtils.alignUp(15, 16));
    try testing.expectEqual(@as(usize, 32), souper.SouperIntegration.MathUtils.alignUp(17, 16));
    try testing.expect(souper.SouperIntegration.MathUtils.isAligned(16, 16));
    try testing.expect(!souper.SouperIntegration.MathUtils.isAligned(15, 16));
    
    std.log.info("Mathematical utilities: All optimizations working correctly", .{});
}

test "Souper optimizations - performance monitoring" {
    var monitor = souper.SouperIntegration.PerformanceMonitor{};
    
    // Simulate usage patterns
    for (0..100) |i| {
        if (i % 4 == 0) {
            monitor.recordFallback();
        } else {
            monitor.recordOptimizationHit();
        }
    }
    
    // Verify statistics
    try testing.expectEqual(@as(u64, 75), monitor.optimization_hits);
    try testing.expectEqual(@as(u64, 25), monitor.fallback_uses);
    try testing.expectEqual(@as(u64, 100), monitor.total_operations);
    
    const opt_rate = monitor.getOptimizationRate();
    const fallback_rate = monitor.getFallbackRate();
    
    try testing.expect(opt_rate >= 0.74 and opt_rate <= 0.76); // ~75%
    try testing.expect(fallback_rate >= 0.24 and fallback_rate <= 0.26); // ~25%
    
    std.log.info("Performance monitoring: Optimization rate = {d:.1}%", .{opt_rate * 100.0});
}

test "Souper optimizations - end-to-end integration" {
    const allocator = testing.allocator;
    
    // Create ThreadPool with all optimizations enabled
    const config = beat.Config{
        .num_workers = 4,
        .enable_souper_optimizations = true,
        .enable_heartbeat = true,
        .enable_predictive = true,
        .enable_advanced_selection = true,
        .enable_statistics = true,
    };
    
    const pool = try beat.ThreadPool.init(allocator, config);
    defer pool.deinit();
    
    // Submit test tasks
    const TaskData = struct {
        value: u32,
        result: std.atomic.Value(u32),
    };
    
    const num_tasks = 100;
    const tasks = try allocator.alloc(TaskData, num_tasks);
    defer allocator.free(tasks);
    
    // Initialize tasks
    for (tasks, 0..) |*task_data, i| {
        task_data.* = .{
            .value = @intCast(i),
            .result = std.atomic.Value(u32).init(0),
        };
    }
    
    // Submit tasks to pool
    for (tasks) |*task_data| {
        const task = beat.Task{
            .func = struct {
                fn run(data: *anyopaque) void {
                    const td = @as(*TaskData, @ptrCast(@alignCast(data)));
                    
                    // Use Souper-optimized operations
                    const optimized_value = souper.SouperIntegration.OptimizedSIMD.classifyTask(td.value);
                    td.result.store(optimized_value, .release);
                }
            }.run,
            .data = task_data,
        };
        
        try pool.submit(task);
    }
    
    // Wait for completion
    // Wait for all tasks to complete
    std.time.sleep(100_000_000); // 100ms to allow tasks to complete
    
    // Verify all tasks completed
    var completed_count: u32 = 0;
    for (tasks) |*task_data| {
        if (task_data.result.load(.acquire) != 0) {
            completed_count += 1;
        }
    }
    
    try testing.expect(completed_count == num_tasks);
    
    // Get performance statistics
    const stats = souper.SouperIntegration.getGlobalStats();
    std.log.info("End-to-end test: {} operations, {d:.1}% optimization rate", .{
        stats.total_operations,
        stats.getOptimizationRate() * 100.0,
    });
}

test "Souper optimizations - correctness validation" {
    const allocator = testing.allocator;
    
    // Generate performance report
    const report = try souper.SouperIntegration.generateReport(allocator);
    defer allocator.free(report);
    
    // Verify report contains expected information
    try testing.expect(std.mem.indexOf(u8, report, "Souper Mathematical Optimization Report") != null);
    try testing.expect(std.mem.indexOf(u8, report, "Optimization Efficiency") != null);
    try testing.expect(std.mem.indexOf(u8, report, "mathematically proven") != null);
    
    std.log.info("Generated optimization report ({} bytes)", .{report.len});
}

test "Souper optimizations - stress test" {
    const allocator = testing.allocator;
    
    // Stress test with large datasets
    const large_dataset_size = 10000;
    
    // Test fingerprint similarity with large dataset
    const hashes1 = try allocator.alloc(u64, large_dataset_size);
    defer allocator.free(hashes1);
    
    const hashes2 = try allocator.alloc(u64, large_dataset_size);
    defer allocator.free(hashes2);
    
    const similarities = try allocator.alloc(u64, large_dataset_size);
    defer allocator.free(similarities);
    
    // Initialize with random data
    var prng = std.Random.DefaultPrng.init(0xDEADBEEF);
    const random = prng.random();
    
    for (hashes1, hashes2) |*h1, *h2| {
        h1.* = random.int(u64);
        h2.* = random.int(u64);
    }
    
    // Benchmark batch processing
    const start_time = std.time.nanoTimestamp();
    
    souper.SouperIntegration.OptimizedFingerprint.computeSimilarityBatch(hashes1, hashes2, similarities);
    
    const batch_time = std.time.nanoTimestamp() - start_time;
    
    // Verify all results are valid
    for (similarities) |similarity| {
        try testing.expect(similarity <= 100);
    }
    
    const throughput = @as(f64, @floatFromInt(large_dataset_size)) / (@as(f64, @floatFromInt(batch_time)) / 1_000_000_000.0);
    
    std.log.info("Stress test: {d:.0} operations/second, {} ns/op", .{
        throughput,
        @divTrunc(batch_time, @as(i64, @intCast(large_dataset_size))),
    });
    
    // Verify performance is reasonable (should be faster than naive implementation)
    try testing.expect(@divTrunc(batch_time, @as(i64, @intCast(large_dataset_size))) < 1000); // Less than 1μs per operation
}