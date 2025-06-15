const std = @import("std");
const beat = @import("beat");

// Development Mode Configuration Test
//
// This test demonstrates the development mode configuration features of Beat.zig
// including debugging, validation, and profiling configurations.

test "development configuration creation" {
    
    // Test development configuration
    const dev_config = beat.Config.createDevelopmentConfig();
    try std.testing.expect(dev_config.development_mode);
    try std.testing.expect(dev_config.verbose_logging);
    try std.testing.expect(dev_config.performance_validation);
    try std.testing.expect(dev_config.memory_debugging);
    try std.testing.expect(dev_config.task_tracing);
    try std.testing.expect(dev_config.scheduler_profiling);
    try std.testing.expect(dev_config.deadlock_detection);
    try std.testing.expect(dev_config.resource_leak_detection);
    try std.testing.expect(dev_config.enable_trace);
    try std.testing.expect(dev_config.enable_statistics);
    
    // Check conservative debugging settings
    try std.testing.expectEqual(@as(?usize, 2), dev_config.num_workers);
    try std.testing.expectEqual(@as(u32, 16), dev_config.task_queue_size);
    try std.testing.expectEqual(@as(u32, 50), dev_config.heartbeat_interval_us);
}

test "testing configuration creation" {
    const test_config = beat.Config.createTestingConfig();
    try std.testing.expect(test_config.development_mode);
    try std.testing.expect(!test_config.verbose_logging); // Reduced noise
    try std.testing.expect(test_config.performance_validation);
    try std.testing.expect(test_config.memory_debugging);
    try std.testing.expect(test_config.resource_leak_detection);
    try std.testing.expect(test_config.enable_statistics);
    
    // Check fast test settings
    try std.testing.expectEqual(@as(?usize, 2), test_config.num_workers);
    try std.testing.expectEqual(@as(u32, 8), test_config.task_queue_size);
    try std.testing.expectEqual(@as(u32, 10), test_config.heartbeat_interval_us);
    try std.testing.expectEqual(@as(u64, 5), test_config.promotion_threshold);
}

test "profiling configuration creation" {
    const prof_config = beat.Config.createProfilingConfig();
    try std.testing.expect(prof_config.development_mode);
    try std.testing.expect(!prof_config.verbose_logging);
    try std.testing.expect(!prof_config.performance_validation); // Avoid interference
    try std.testing.expect(prof_config.scheduler_profiling);
    try std.testing.expect(!prof_config.task_tracing); // Reduce overhead
    try std.testing.expect(prof_config.enable_statistics);
}

test "development mode application" {
    var config = beat.Config{};
    config.development_mode = true;
    config.verbose_logging = true;
    config.memory_debugging = true;
    config.deadlock_detection = true;
    config.heartbeat_interval_us = 200; // Will be reduced
    
    config.applyDevelopmentMode();
    
    // Should enable trace due to verbose_logging
    try std.testing.expect(config.enable_trace);
    
    // Should enable statistics due to memory_debugging
    try std.testing.expect(config.enable_statistics);
    
    // Should reduce heartbeat interval due to deadlock_detection
    try std.testing.expectEqual(@as(u32, 100), config.heartbeat_interval_us);
}

test "configuration validation" {
    const allocator = std.testing.allocator;
    
    // Test development configuration validation
    const dev_config = beat.Config.createDevelopmentConfig();
    const analysis = try dev_config.validateDevelopmentConfig(allocator);
    defer allocator.free(analysis);
    
    // Should contain positive confirmations
    try std.testing.expect(std.mem.indexOf(u8, analysis, "✅ Resource leak detection enabled") != null);
    try std.testing.expect(std.mem.indexOf(u8, analysis, "✅ Deadlock detection enabled") != null);
    try std.testing.expect(std.mem.indexOf(u8, analysis, "✅ Performance validation enabled") != null);
    
    // Test production configuration validation
    const prod_config = beat.Config{};
    const prod_analysis = try prod_config.validateDevelopmentConfig(allocator);
    defer allocator.free(prod_analysis);
    
    try std.testing.expect(std.mem.indexOf(u8, prod_analysis, "Production Configuration") != null);
    try std.testing.expect(std.mem.indexOf(u8, prod_analysis, "Development mode disabled") != null);
}

test "configuration with problematic settings" {
    const allocator = std.testing.allocator;
    
    var config = beat.Config{};
    config.development_mode = true;
    config.task_tracing = true;
    config.verbose_logging = false; // This should trigger a recommendation
    config.num_workers = 8; // Too many for development
    config.task_queue_size = 64; // Too large for development
    config.memory_debugging = true;
    config.enable_statistics = false; // This should trigger a recommendation
    
    const analysis = try config.validateDevelopmentConfig(allocator);
    defer allocator.free(analysis);
    
    // Should contain recommendations
    try std.testing.expect(std.mem.indexOf(u8, analysis, "⚠️  Recommendation: Enable verbose_logging") != null);
    try std.testing.expect(std.mem.indexOf(u8, analysis, "⚠️  Recommendation: Enable statistics") != null);
    try std.testing.expect(std.mem.indexOf(u8, analysis, "⚠️  Recommendation: Use 2-4 workers") != null);
    try std.testing.expect(std.mem.indexOf(u8, analysis, "⚠️  Recommendation: Use smaller queue size") != null);
}

// Example usage demonstration
test "development mode thread pool creation" {
    const allocator = std.testing.allocator;
    
    // Create a development configuration
    const config = beat.Config.createDevelopmentConfig();
    
    // Get configuration analysis
    const analysis = try config.validateDevelopmentConfig(allocator);
    defer allocator.free(analysis);
    
    // In development mode, you would see detailed logging and validation
    if (config.verbose_logging) {
        // This would produce detailed logs in a real application
        std.debug.print("Development mode enabled with comprehensive debugging\n", .{});
    }
    
    // Note: We can't easily test ThreadPool creation here due to complex dependencies,
    // but this demonstrates how development mode would be used
    try std.testing.expect(config.development_mode);
}

// Performance comparison test (demonstrative)
test "configuration performance characteristics" {
    // Development config prioritizes debugging over performance
    const dev_config = beat.Config.createDevelopmentConfig();
    try std.testing.expect(dev_config.num_workers.? == 2); // Smaller for debugging
    try std.testing.expect(dev_config.task_queue_size == 16); // Smaller for faster detection
    try std.testing.expect(dev_config.heartbeat_interval_us == 50); // More responsive
    
    // Testing config balances debugging with test speed
    const test_config = beat.Config.createTestingConfig();
    try std.testing.expect(test_config.num_workers.? == 2); // Still small but optimized
    try std.testing.expect(test_config.task_queue_size == 8); // Even smaller for tests
    try std.testing.expect(test_config.heartbeat_interval_us == 10); // Very fast
    try std.testing.expect(test_config.promotion_threshold == 5); // Lower for faster testing
    
    // Profiling config minimizes interference
    const prof_config = beat.Config.createProfilingConfig();
    try std.testing.expect(!prof_config.performance_validation); // No interference
    try std.testing.expect(!prof_config.task_tracing); // Minimal overhead
    try std.testing.expect(prof_config.scheduler_profiling); // Focused profiling
}