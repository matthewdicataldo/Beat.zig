const std = @import("std");
const ispc_config = @import("../src/ispc_config.zig");

test "ISPC configuration and fallback" {
    std.debug.print("\n=== ISPC Configuration Test ===\n", .{});
    
    // Test ISPC availability detection
    const ispc_available = ispc_config.ISPCConfig.ISPC_AVAILABLE;
    std.debug.print("ISPC Available: {}\n", .{ispc_available});
    
    // Test runtime initialization
    const init_result = ispc_config.ISPCConfig.initializeISPCRuntime();
    std.debug.print("ISPC Runtime Init: {}\n", .{init_result});
    
    // Test enable/disable acceleration
    const enable_result = ispc_config.ISPCConfig.enableISPCAcceleration();
    std.debug.print("ISPC Acceleration Enabled: {}\n", .{enable_result});
    
    // Print acceleration status
    const should_use = ispc_config.ISPCConfig.shouldUseISPC();
    std.debug.print("Should Use ISPC: {}\n", .{should_use});
    
    // Clean up
    ispc_config.ISPCConfig.disableISPCAcceleration();
    
    std.debug.print("=== ISPC Test Complete ===\n", .{});
}

test "ISPC feature detection" {
    std.debug.print("\n=== ISPC Feature Detection ===\n", .{});
    
    // Test architecture compatibility
    const target = @import("builtin").target;
    std.debug.print("Target Architecture: {}\n", .{target.cpu.arch});
    
    const ispc_compatible = switch (target.cpu.arch) {
        .x86_64, .aarch64 => true,
        else => false,
    };
    std.debug.print("ISPC Compatible Architecture: {}\n", .{ispc_compatible});
    
    // Test acceleration configuration
    const config = ispc_config.ISPCConfig;
    std.debug.print("Enable ISPC Acceleration: {}\n", .{config.enable_ispc_acceleration});
    std.debug.print("ISPC Runtime Initialized: {}\n", .{config.ispc_runtime_initialized});
    
    std.debug.print("=== Feature Detection Complete ===\n", .{});
}

test "ISPC fallback mechanisms" {
    std.debug.print("\n=== ISPC Fallback Test ===\n", .{});
    
    // Test fallback mechanisms without allocator
    
    // Test fallback function wrapper
    const TestFallback = struct {
        fn ispcFunction(x: i32) i32 {
            return x * 2; // ISPC version (faster)
        }
        
        fn fallbackFunction(x: i32) i32 {
            return x + x; // Fallback version (same result)
        }
    };
    
    // Test conditional execution
    const input = 21;
    const expected = 42;
    
    const result = ispc_config.executeWithISPCFallback(
        TestFallback.ispcFunction,
        TestFallback.fallbackFunction,
        .{input}
    );
    
    try std.testing.expect(result == expected);
    std.debug.print("Fallback execution test passed: {} -> {}\n", .{input, result});
    
    // Test that fallback provides correct results
    const fallback_result = TestFallback.fallbackFunction(input);
    try std.testing.expect(fallback_result == expected);
    std.debug.print("Direct fallback test passed: {} -> {}\n", .{input, fallback_result});
    
    std.debug.print("=== Fallback Test Complete ===\n", .{});
}

test "ISPC performance monitoring" {
    std.debug.print("\n=== ISPC Performance Monitoring ===\n", .{});
    
    // Test stats tracking (when ISPC integration is available)
    const prediction_integration = @import("../src/ispc_prediction_integration.zig");
    
    // Get empty stats (ISPC not available)
    const stats = prediction_integration.DiagnosticsAPI.getAccelerationStats();
    std.debug.print("ISPC Calls: {}\n", .{stats.ispc_calls});
    std.debug.print("Native Calls: {}\n", .{stats.native_calls});
    std.debug.print("ISPC Failures: {}\n", .{stats.ispc_failures});
    std.debug.print("Performance Ratio: {d:.2}\n", .{stats.performance_ratio});
    
    // Print performance report
    prediction_integration.DiagnosticsAPI.printPerformanceReport();
    
    std.debug.print("=== Performance Monitoring Complete ===\n", .{});
}