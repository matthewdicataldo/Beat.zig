const std = @import("std");
const beat = @import("src/core.zig");

// Simple test to verify the testing framework works
test "simple parallel testing framework" {
    // Test the basic utilities without complex parallel operations
    const testing_framework = beat.testing;
    
    // Test task creation
    const cpu_task = testing_framework.createCpuTask(100);
    const result = cpu_task();
    try std.testing.expect(result > 100);
    
    // Test test pool creation
    const pool = try testing_framework.createTestPool(std.testing.allocator);
    defer pool.deinit();
    
    try std.testing.expect(pool.workers.len > 0);
    
    // Test stats capture
    const stats = testing_framework.TestStats.capture(&pool.stats);
    try std.testing.expectEqual(@as(u64, 0), stats.tasks_submitted);
    
    std.debug.print("Parallel testing framework basic functionality verified!\n", .{});
}

test "test framework utilities" {
    const testing_framework = beat.testing;
    
    // Test different task types
    const cpu_task1 = testing_framework.createCpuTask(42);
    const cpu_task2 = testing_framework.createCpuTask(84);
    
    const result1 = cpu_task1();
    const result2 = cpu_task2();
    
    try std.testing.expect(result1 != result2); // Different base values should give different results
    try std.testing.expect(result1 > 42);
    try std.testing.expect(result2 > 84);
    
    // Test IO task
    const io_task = testing_framework.createIoTask(200, 1000); // 1ms delay
    const start_time = std.time.nanoTimestamp();
    const io_result = io_task();
    const end_time = std.time.nanoTimestamp();
    
    try std.testing.expectEqual(@as(i32, 200), io_result);
    try std.testing.expect((end_time - start_time) >= 500_000); // At least 0.5ms (allowing for timing variance)
    
    // Test memory task
    const memory_task = testing_framework.createMemoryTask(1024);
    memory_task(); // Should complete without error
    
    std.debug.print("All task utilities working correctly!\n", .{});
}

test "run parallel test helper" {
    const testing_framework = beat.testing;
    
    const SimpleTest = struct {
        fn testFunction(pool: *beat.ThreadPool) !void {
            // Just verify the pool is working
            try std.testing.expect(pool.workers.len > 0);
            try std.testing.expect(pool.running.load(.acquire) == true);
        }
    };
    
    // Use the runParallelTest helper
    try testing_framework.runParallelTest(std.testing.allocator, SimpleTest.testFunction);
    
    std.debug.print("runParallelTest helper works correctly!\n", .{});
}