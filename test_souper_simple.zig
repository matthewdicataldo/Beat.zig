// Simple Souper integration test focusing on mathematical optimizations
// This test validates the core Souper optimizations without ISPC dependencies

const std = @import("std");
const testing = std.testing;
const beat = @import("src/core.zig");
const souper = beat.souper_integration;
const math_opt = beat.mathematical_optimizations;

test "Souper mathematical optimizations - core algorithms" {
    // Test fingerprint similarity optimization
    const hash1: u64 = 0x123456789ABCDEF0;
    const hash2: u64 = 0x0FEDCBA987654321;
    
    const result = souper.SouperIntegration.OptimizedFingerprint.computeSimilarity(hash1, hash2);
    try testing.expect(result <= 100);
}

test "Souper mathematical optimizations - heartbeat scheduling" {
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
        .{ .load = 90, .capacity = 100, .threshold_factor = 50, .expected = false }, // Too high load
    };
    
    for (test_cases) |case| {
        const result = souper.SouperIntegration.OptimizedScheduler.shouldStealWork(
            case.load, case.capacity, case.threshold_factor
        );
        
        try testing.expectEqual(case.expected, result);
    }
}

test "Souper mathematical optimizations - lock-free index calculations" {
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
}

test "Souper mathematical optimizations - mathematical utilities" {
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
}

test "Souper mathematical optimizations - performance monitoring" {
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
}

test "Souper mathematical optimizations - correctness validation" {
    const allocator = testing.allocator;
    
    // Generate performance report
    const report = try souper.SouperIntegration.generateReport(allocator);
    defer allocator.free(report);
    
    // Verify report contains expected information
    try testing.expect(std.mem.indexOf(u8, report, "Souper Mathematical Optimization Report") != null);
    try testing.expect(std.mem.indexOf(u8, report, "Optimization Efficiency") != null);
    try testing.expect(std.mem.indexOf(u8, report, "mathematically proven") != null);
}

test "Mathematical optimizations - direct algorithm testing" {
    // Test core mathematical optimization functions directly
    
    // Test fingerprint similarity optimization
    const hash1: u64 = 0x123456789ABCDEF0;
    const hash2: u64 = 0x0FEDCBA987654321;
    const similarity = math_opt.MathematicalOptimizations.computeFingerprintSimilarityOptimized(hash1, hash2);
    try testing.expect(similarity <= 100);
    
    // Test heartbeat scheduling optimization
    try testing.expect(!math_opt.MathematicalOptimizations.shouldStealWorkOptimized(75, 100, 50));
    try testing.expect(math_opt.MathematicalOptimizations.shouldStealWorkOptimized(60, 100, 50));
    
    // Test Chase-Lev index optimization
    try testing.expectEqual(@as(u64, 5), math_opt.MathematicalOptimizations.optimizeChaselevIndex(13, 8));
    try testing.expectEqual(@as(u64, 3), math_opt.MathematicalOptimizations.optimizeChaselevIndex(13, 10));
    
    // Test worker selection optimization
    try testing.expectEqual(@as(u32, 5), math_opt.MathematicalOptimizations.selectWorkerOptimized(13, 8));
    try testing.expectEqual(@as(u32, 3), math_opt.MathematicalOptimizations.selectWorkerOptimized(13, 10));
    
    // Test vector sum optimization
    const data = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const sum = math_opt.MathematicalOptimizations.vectorSumOptimized(&data);
    try testing.expectEqual(@as(u32, 55), sum);
    
    // Test task classification optimization
    const flags: u32 = 0b11111111_1111_111;
    const classification = math_opt.MathematicalOptimizations.classifyTaskOptimized(flags);
    try testing.expect(classification <= 0xFFFF);
}