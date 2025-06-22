const std = @import("std");
const testing = std.testing;
const runtime_context = @import("../src/runtime_context.zig");

test "RuntimeContext basic lifecycle" {
    const allocator = testing.allocator;
    
    const config = runtime_context.createTestingConfig();
    var context = try runtime_context.RuntimeContext.init(allocator, config);
    defer context.deinit();
    
    try testing.expect(context.isInitialized());
    try testing.expect(!context.isShuttingDown());
    
    const stats = context.getStatistics();
    try testing.expect(stats.is_initialized);
    try testing.expect(!stats.is_shutting_down);
}

test "RuntimeContext global management" {
    const allocator = testing.allocator;
    
    // Test initialization
    const config = runtime_context.createTestingConfig();
    const context = try runtime_context.initGlobalRuntimeContext(allocator, config);
    defer runtime_context.deinitGlobalRuntimeContext();
    
    try testing.expect(runtime_context.isGlobalRuntimeContextInitialized());
    
    // Test retrieval
    const retrieved_context = try runtime_context.getGlobalRuntimeContext();
    try testing.expect(context == retrieved_context);
    
    // Test statistics
    const stats = context.getStatistics();
    const report = try stats.generateReport(allocator);
    defer allocator.free(report);
    
    try testing.expect(report.len > 0);
}

test "RuntimeContext configuration factories" {
    const dev_config = runtime_context.createDevelopmentConfig();
    try testing.expect(!dev_config.enable_ispc_acceleration);
    try testing.expect(!dev_config.enable_optimization_systems);
    
    const prod_config = runtime_context.createProductionConfig();
    try testing.expect(prod_config.enable_ispc_acceleration);
    try testing.expect(prod_config.enable_optimization_systems);
    
    const test_config = runtime_context.createTestingConfig();
    try testing.expect(!test_config.enable_numa_awareness);
    try testing.expect(!test_config.enable_advanced_worker_selection);
}

test "RuntimeContext subsystem management" {
    const allocator = testing.allocator;
    
    // Test with minimal subsystems enabled
    var config = runtime_context.createTestingConfig();
    config.enable_numa_awareness = true;
    config.enable_task_execution_stats = true;
    
    var context = try runtime_context.RuntimeContext.init(allocator, config);
    defer context.deinit();
    
    try testing.expect(context.isInitialized());
    
    // Test that enabled subsystems are available
    const numa_mapper = context.getNumaMapper();
    try testing.expect(numa_mapper != null);
    
    const task_stats = context.getTaskStatsManager();
    try testing.expect(task_stats != null);
    
    // Test that disabled subsystems are not available
    const memory_monitor = context.getMemoryPressureMonitor();
    try testing.expect(memory_monitor == null);
    
    const optimization_registry = context.getOptimizationRegistry();
    try testing.expect(optimization_registry == null);
}

test "RuntimeContext thread safety" {
    const allocator = testing.allocator;
    
    const config = runtime_context.createTestingConfig();
    const context = try runtime_context.initGlobalRuntimeContext(allocator, config);
    defer runtime_context.deinitGlobalRuntimeContext();
    
    // Test concurrent access to statistics
    const ThreadData = struct {
        context: *runtime_context.RuntimeContext,
        stats_collected: std.atomic.Value(u32),
    };
    
    var thread_data = ThreadData{
        .context = context,
        .stats_collected = std.atomic.Value(u32).init(0),
    };
    
    const StatCollector = struct {
        fn collectStats(data: *ThreadData) void {
            for (0..10) |_| {
                const stats = data.context.getStatistics();
                _ = stats; // Use the stats to prevent optimization
                _ = data.stats_collected.fetchAdd(1, .monotonic);
                std.time.sleep(1_000_000); // 1ms
            }
        }
    };
    
    // Start multiple threads
    var threads: [4]std.Thread = undefined;
    for (&threads) |*thread| {
        thread.* = try std.Thread.spawn(.{}, StatCollector.collectStats, .{&thread_data});
    }
    
    // Wait for all threads
    for (threads) |thread| {
        thread.join();
    }
    
    // Verify all stats were collected
    try testing.expect(thread_data.stats_collected.load(.acquire) == 40);
}

test "RuntimeContext shutdown coordination" {
    const allocator = testing.allocator;
    
    var config = runtime_context.createTestingConfig();
    config.enable_numa_awareness = true;
    config.enable_task_execution_stats = true;
    
    var context = try runtime_context.RuntimeContext.init(allocator, config);
    
    try testing.expect(context.isInitialized());
    try testing.expect(!context.isShuttingDown());
    
    // Verify subsystems are available before shutdown
    try testing.expect(context.getNumaMapper() != null);
    try testing.expect(context.getTaskStatsManager() != null);
    
    // Start shutdown
    context.deinit();
    
    // After deinit, context should not be accessible
    // Note: We can't test this directly since context is freed,
    // but we can test the global context shutdown
}

test "RuntimeContext error handling" {
    const allocator = testing.allocator;
    
    // Test getting global context before initialization
    try testing.expectError(error.RuntimeContextNotInitialized, runtime_context.getGlobalRuntimeContext());
    
    // Test double initialization (should return existing instance)
    const config = runtime_context.createTestingConfig();
    const context1 = try runtime_context.initGlobalRuntimeContext(allocator, config);
    const context2 = try runtime_context.initGlobalRuntimeContext(allocator, config);
    defer runtime_context.deinitGlobalRuntimeContext();
    
    try testing.expect(context1 == context2);
}

test "RuntimeContext statistics report generation" {
    const allocator = testing.allocator;
    
    var config = runtime_context.createTestingConfig();
    config.enable_numa_awareness = true;
    config.enable_task_execution_stats = true;
    
    var context = try runtime_context.RuntimeContext.init(allocator, config);
    defer context.deinit();
    
    const stats = context.getStatistics();
    const report = try stats.generateReport(allocator);
    defer allocator.free(report);
    
    // Verify report contains expected sections
    try testing.expect(std.mem.indexOf(u8, report, "Runtime Context Status") != null);
    try testing.expect(std.mem.indexOf(u8, report, "Initialization Status") != null);
    try testing.expect(std.mem.indexOf(u8, report, "Subsystem Status") != null);
    try testing.expect(std.mem.indexOf(u8, report, "Current State") != null);
    
    // Verify NUMA and task stats are reported as enabled
    try testing.expect(std.mem.indexOf(u8, report, "NUMA mapping: true") != null);
    try testing.expect(std.mem.indexOf(u8, report, "Task execution stats: true") != null);
    
    // Verify disabled subsystems are reported as disabled
    try testing.expect(std.mem.indexOf(u8, report, "ISPC acceleration: false") != null);
    try testing.expect(std.mem.indexOf(u8, report, "Optimization systems: false") != null);
}