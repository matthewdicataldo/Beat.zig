const std = @import("std");
const beat = @import("src/core.zig");

// Demonstration of the Enhanced Parallel Testing Framework
// This shows the completed implementation working correctly

test "Enhanced Parallel Testing Framework - Complete Demo" {
    const testing_framework = beat.testing;
    
    std.debug.print("\n=== Enhanced Parallel Testing Framework Demo ===\n", .{});
    
    // 1. Task Creation Utilities
    std.debug.print("1. Testing task creation utilities...\n", .{});
    const cpu_task = testing_framework.createCpuTask(100);
    const result = cpu_task();
    try std.testing.expect(result > 100);
    std.debug.print("   ✓ CPU task created and executed: {}\n", .{result});
    
    const io_task = testing_framework.createIoTask(42, 1000); // 1ms delay
    const start_time = std.time.nanoTimestamp();
    const io_result = io_task();
    const end_time = std.time.nanoTimestamp();
    const duration_us = @divTrunc((end_time - start_time), 1000);
    try std.testing.expectEqual(@as(i32, 42), io_result);
    std.debug.print("   ✓ IO task with delay: {}μs (result: {})\n", .{ duration_us, io_result });
    
    const memory_task = testing_framework.createMemoryTask(2048);
    memory_task();
    std.debug.print("   ✓ Memory task completed (allocated 2KB)\n", .{});
    
    // 2. Test Pool Creation
    std.debug.print("2. Testing optimized test pool...\n", .{});
    const pool = try testing_framework.createTestPool(std.testing.allocator);
    defer pool.deinit();
    
    try std.testing.expect(pool.workers.len == 4);
    try std.testing.expect(pool.running.load(.acquire) == true);
    try std.testing.expect(pool.config.enable_topology_aware == false);
    try std.testing.expect(pool.config.enable_numa_aware == false);
    try std.testing.expect(pool.config.enable_heartbeat == true);
    std.debug.print("   ✓ Test pool created with 4 workers, simplified config\n", .{});
    
    // 3. Statistics Capture
    std.debug.print("3. Testing statistics capture...\n", .{});
    const initial_stats = testing_framework.TestStats.capture(&pool.stats);
    
    // Submit some tasks
    try testing_framework.submitSimpleTasks(pool, 3);
    std.time.sleep(50_000_000); // 50ms - allow tasks to complete
    
    const final_stats = testing_framework.TestStats.capture(&pool.stats);
    const delta = testing_framework.TestStats.delta(final_stats, initial_stats);
    
    std.debug.print("   ✓ Submitted: {}, Completed: {}, Delta: {}\n", .{
        delta.tasks_submitted, delta.tasks_completed, delta.tasks_submitted - delta.tasks_completed
    });
    try std.testing.expect(delta.tasks_submitted >= 3);
    
    // 4. Resource Cleanup Validation
    std.debug.print("4. Testing resource cleanup validation...\n", .{});
    try testing_framework.validateResourceCleanup(initial_stats, final_stats);
    std.debug.print("   ✓ Resource cleanup validation passed\n", .{});
    
    // 5. Test Helpers
    std.debug.print("5. Testing helper functions...\n", .{});
    
    const TestFunction = struct {
        fn basicTest(test_pool: *beat.ThreadPool) !void {
            try std.testing.expect(test_pool.workers.len > 0);
            try std.testing.expect(test_pool.running.load(.acquire) == true);
        }
        
        fn performanceTest(test_pool: *beat.ThreadPool) !void {
            _ = test_pool;
            std.time.sleep(5_000_000); // 5ms
        }
        
        fn stressTest(test_pool: *beat.ThreadPool, iterations: usize) !void {
            _ = test_pool;
            var sum: u64 = 0;
            for (0..iterations) |i| {
                sum += i;
            }
            try std.testing.expect(sum > 0);
        }
    };
    
    // Test runParallelTest
    try testing_framework.runParallelTest(std.testing.allocator, TestFunction.basicTest);
    std.debug.print("   ✓ runParallelTest completed successfully\n", .{});
    
    // Test runPerformanceTest (should complete within 100ms)
    try testing_framework.runPerformanceTest(std.testing.allocator, TestFunction.performanceTest, 100);
    std.debug.print("   ✓ runPerformanceTest completed within time limit\n", .{});
    
    // Test runStressTest
    try testing_framework.runStressTest(std.testing.allocator, TestFunction.stressTest, 100);
    std.debug.print("   ✓ runStressTest completed 100 iterations\n", .{});
    
    std.debug.print("\n=== All Enhanced Parallel Testing Framework Features Working! ===\n", .{});
}