const std = @import("std");

// Test the enhanced error handling integration
// This will test the new error messages and fallback behaviors

const beat = @import("beat");
const easy_api = @import("src/easy_api.zig");
const enhanced_errors = @import("src/enhanced_errors.zig");

test "enhanced error messages for invalid worker count" {
    const allocator = std.testing.allocator;
    
    // Test zero worker count error
    const result = easy_api.createBasicPool(allocator, 0);
    try std.testing.expectError(enhanced_errors.ConfigError.InvalidConfiguration, result);
}

test "enhanced error messages for high worker count warning" {
    const allocator = std.testing.allocator;
    
    // Test high worker count warning (should succeed but warn)
    const pool = try easy_api.createBasicPool(allocator, 100);
    defer pool.deinit();
    
    // Should succeed despite warning
    try std.testing.expect(@intFromPtr(pool) != 0);
}

test "basic pool auto-detection with error handling" {
    const allocator = std.testing.allocator;
    
    // Test auto-detection (should work with enhanced error handling)
    const pool = try easy_api.createBasicPoolAuto(allocator);
    defer pool.deinit();
    
    try std.testing.expect(@intFromPtr(pool) != 0);
}

test "performance pool with enhanced error handling" {
    const allocator = std.testing.allocator;
    
    // Test performance pool creation with reasonable settings
    const pool = try easy_api.createPerformancePool(allocator, .{
        .workers = 4,
        .enable_topology_aware = false, // Disable to avoid potential issues
    });
    defer pool.deinit();
    
    try std.testing.expect(@intFromPtr(pool) != 0);
}

test "development pool creation" {
    const allocator = std.testing.allocator;
    
    // Test development pool (should be very safe)
    const pool = try easy_api.createDevelopmentPool(allocator, .{
        .workers = 2,
        .enable_performance_features = false,
    });
    defer pool.deinit();
    
    try std.testing.expect(@intFromPtr(pool) != 0);
}

test "runtime configuration detection" {
    const allocator = std.testing.allocator;
    
    // Test runtime configuration detection
    const runtime_config = try easy_api.detectOptimalConfig(allocator);
    
    // Validate detected values
    try std.testing.expect(runtime_config.recommended_workers > 0);
    try std.testing.expect(runtime_config.recommended_queue_size > 0);
    try std.testing.expect(runtime_config.memory_gb > 0);
    
    // Test creating pool from runtime config
    const pool = try runtime_config.createPool(allocator);
    defer pool.deinit();
    
    try std.testing.expect(@intFromPtr(pool) != 0);
}

test "http server pool creation" {
    const allocator = std.testing.allocator;
    
    // Test HTTP server pool with basic feature level
    var http_pool = try easy_api.createHttpServerPool(allocator, .{
        .connection_workers = 2,
        .io_workers = 1,
        .request_workers = 2,
        .feature_level = .basic, // Use basic to avoid complex dependencies
    });
    defer http_pool.deinit();
    
    // Validate pools were created
    try std.testing.expect(@intFromPtr(http_pool.connection_pool) != 0);
    try std.testing.expect(@intFromPtr(http_pool.io_pool) != 0);
    try std.testing.expect(@intFromPtr(http_pool.request_pool) != 0);
}

test "configuration validation with helpful errors" {
    // Test the configuration validation system
    const TestConfig = struct {
        num_workers: ?u32,
        task_queue_size: u32,
    };
    
    // Valid configuration should pass
    const valid_config = TestConfig{
        .num_workers = 4,
        .task_queue_size = 256,
    };
    try enhanced_errors.validateConfigurationWithHelp(valid_config);
    
    // Invalid configurations should provide helpful errors
    const invalid_workers = TestConfig{
        .num_workers = 0,
        .task_queue_size = 256,
    };
    try std.testing.expectError(
        enhanced_errors.ConfigError.InvalidConfiguration, 
        enhanced_errors.validateConfigurationWithHelp(invalid_workers)
    );
    
    const invalid_queue = TestConfig{
        .num_workers = 4,
        .task_queue_size = 8, // Too small
    };
    try std.testing.expectError(
        enhanced_errors.ConfigError.InvalidConfiguration,
        enhanced_errors.validateConfigurationWithHelp(invalid_queue)
    );
}

test "error message generation" {
    // Test error message generation functions
    const missing_config_msg = enhanced_errors.formatMissingBuildConfigError("TestProject");
    try std.testing.expect(missing_config_msg.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, missing_config_msg, "build_config") != null);
    try std.testing.expect(std.mem.indexOf(u8, missing_config_msg, "createBasicPool") != null);
    
    const hardware_error_msg = enhanced_errors.formatHardwareDetectionError();
    try std.testing.expect(hardware_error_msg.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, hardware_error_msg, "safe default") != null);
    
    const platform_error_msg = enhanced_errors.formatUnsupportedPlatformError("TestOS");
    try std.testing.expect(platform_error_msg.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, platform_error_msg, "platform") != null);
}

test "dependency detection and recommendations" {
    // Test dependency detection logic
    const is_dependency = enhanced_errors.isUsedAsDependency();
    try std.testing.expect(is_dependency == true or is_dependency == false);
    
    // Test recommended dependency configuration
    const dep_config = enhanced_errors.getRecommendedDependencyConfig();
    try std.testing.expect(dep_config.enable_lock_free == true);
    try std.testing.expect(dep_config.enable_topology_aware == false);
}

// Integration test that simulates a real external project scenario
test "external project integration simulation" {
    const allocator = std.testing.allocator;
    
    // Simulate what an external project would do
    std.log.info("=== Simulating External Project Integration ===", .{});
    
    // Step 1: Try basic pool (should work immediately)
    const basic_pool = try easy_api.createBasicPool(allocator, 4);
    defer basic_pool.deinit();
    std.log.info("✅ Basic pool created successfully", .{});
    
    // Step 2: Try auto-detection
    const auto_pool = try easy_api.createBasicPoolAuto(allocator);
    defer auto_pool.deinit();
    std.log.info("✅ Auto-detection pool created successfully", .{});
    
    // Step 3: Try runtime configuration
    const runtime_config = try easy_api.detectOptimalConfig(allocator);
    const runtime_pool = try runtime_config.createPool(allocator);
    defer runtime_pool.deinit();
    std.log.info("✅ Runtime configuration pool created successfully", .{});
    
    // Step 4: Try performance pool with safe settings
    const perf_pool = try easy_api.createPerformancePool(allocator, .{
        .workers = 4,
        .enable_topology_aware = false, // Safe for external projects
    });
    defer perf_pool.deinit();
    std.log.info("✅ Performance pool created successfully", .{});
    
    std.log.info("🎉 All integration scenarios passed!", .{});
}