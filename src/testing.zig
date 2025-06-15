const std = @import("std");
const core = @import("core.zig");

// Enhanced Parallel Testing Framework for Beat.zig
// Provides specialized testing utilities for parallel code validation

// ============================================================================
// Test Utilities
// ============================================================================

/// Create a thread pool optimized for testing
pub fn createTestPool(allocator: std.mem.Allocator) !*core.ThreadPool {
    const config = core.Config{
        .num_workers = 4,
        .enable_topology_aware = false, // Disable for consistent test behavior
        .enable_numa_aware = false,
        .enable_heartbeat = true,
        .task_queue_size = 256, // Smaller queue for testing
    };
    
    const pool = try core.ThreadPool.init(allocator, config);
    return pool;
}

/// Test statistics capture for validation
pub const TestStats = struct {
    tasks_submitted: u64,
    tasks_completed: u64,
    tasks_stolen: u64,
    tasks_cancelled: u64,
    
    pub fn capture(stats: *const core.ThreadPoolStats) TestStats {
        return TestStats{
            .tasks_submitted = stats.tasks_submitted.load(.acquire),
            .tasks_completed = stats.tasks_completed.load(.acquire),
            .tasks_stolen = stats.tasks_stolen.load(.acquire),
            .tasks_cancelled = stats.tasks_cancelled.load(.acquire),
        };
    }
    
    pub fn delta(final: TestStats, initial: TestStats) TestStats {
        return TestStats{
            .tasks_submitted = final.tasks_submitted - initial.tasks_submitted,
            .tasks_completed = final.tasks_completed - initial.tasks_completed,
            .tasks_stolen = final.tasks_stolen - initial.tasks_stolen,
            .tasks_cancelled = final.tasks_cancelled - initial.tasks_cancelled,
        };
    }
};

/// Validate that all submitted tasks are properly handled
pub fn validateResourceCleanup(initial: TestStats, final: TestStats) !void {
    const delta = TestStats.delta(final, initial);
    
    // All submitted tasks should be either completed or cancelled
    const total_processed = delta.tasks_completed + delta.tasks_cancelled;
    
    if (delta.tasks_submitted != total_processed) {
        std.debug.print("Resource leak detected: {d} tasks submitted, {d} completed, {d} cancelled\n", 
            .{ delta.tasks_submitted, delta.tasks_completed, delta.tasks_cancelled });
        // Thread pool resource leak detected - submitted tasks not properly processed
        // Submitted: {}, Completed: {}, Cancelled: {}, Missing: {}
        // Help: Check for deadlocks, ensure pool.wait() is called, verify task completion
        // Common causes: Tasks stuck in queues, worker threads not running, premature shutdown
        return error.ThreadPoolResourceLeak;
    }
    
    // No tasks should remain in worker queues (this is a basic check)
    if (delta.tasks_submitted > 0 and delta.tasks_completed == 0) {
        std.debug.print("Possible deadlock: tasks submitted but none completed\n", .{});
        // Thread pool deadlock detected - tasks submitted but none completed
        // Submitted: {}, Workers: available but idle
        // Help: Check for circular dependencies, task function panics, or worker thread issues
        // Debug: Use thread pool diagnostics, check task function implementation
        return error.ThreadPoolDeadlockDetected;
    }
}

/// Helper to run a test with automatic resource cleanup validation
pub fn runParallelTest(
    allocator: std.mem.Allocator,
    test_fn: fn(*core.ThreadPool) anyerror!void,
) !void {
    const pool = try createTestPool(allocator);
    defer pool.deinit();
    
    // Record initial statistics
    const initial_stats = TestStats.capture(&pool.stats);
    
    // Run the actual test
    try test_fn(pool);
    
    // Allow some time for cleanup
    std.time.sleep(10_000_000); // 10ms
    
    // Verify resource cleanup and consistency
    const final_stats = TestStats.capture(&pool.stats);
    try validateResourceCleanup(initial_stats, final_stats);
}

/// Helper to run a performance test with timing validation
pub fn runPerformanceTest(
    allocator: std.mem.Allocator,
    test_fn: fn(*core.ThreadPool) anyerror!void,
    expected_max_duration_ms: u64,
) !void {
    const pool = try createTestPool(allocator);
    defer pool.deinit();
    
    const start_time = std.time.nanoTimestamp();
    try test_fn(pool);
    const end_time = std.time.nanoTimestamp();
    
    const duration_ms = @divTrunc((end_time - start_time), 1_000_000);
    if (duration_ms > expected_max_duration_ms) {
        std.debug.print("Performance test took {d}ms, expected <= {d}ms\n", 
            .{ duration_ms, expected_max_duration_ms });
        // Performance regression detected - test execution exceeded expected duration
        // Actual: {}ms, Expected: ≤{}ms, Slowdown: {}%
        // Help: Check for resource contention, optimize task implementation, increase test timeout
        // Consider: CPU load, memory pressure, or algorithmic changes affecting performance
        return error.PerformanceTestRegressionDetected;
    }
}

/// Helper to run a stress test with many iterations
pub fn runStressTest(
    allocator: std.mem.Allocator,
    test_fn: fn(*core.ThreadPool, usize) anyerror!void,
    iterations: usize,
) !void {
    const config = core.Config{
        .num_workers = 8, // More workers for stress testing
        .enable_topology_aware = false,
        .enable_numa_aware = false,
        .enable_heartbeat = true,
        .task_queue_size = 2048, // Larger queue for stress testing
    };
    
    const pool = try core.ThreadPool.init(allocator, config);
    defer pool.deinit();
    
    try test_fn(pool, iterations);
    
    // Stress tests need more time to settle
    std.time.sleep(50_000_000); // 50ms
}

// ============================================================================
// Task Creation Utilities
// ============================================================================

/// Simple CPU-bound task that returns a predictable result
pub fn simpleCpuTask() i32 {
    var sum: i32 = 0;
    for (0..1000) |i| {
        sum += @intCast(i % 100);
    }
    return sum + 42; // Base value 42
}

/// Helper to create a simple CPU-bound task for testing
pub fn createCpuTask(comptime result_value: i32) *const fn() i32 {
    return struct {
        fn compute() i32 {
            var sum: i32 = 0;
            for (0..1000) |i| {
                sum += @intCast(i % 100);
            }
            return sum + result_value;
        }
    }.compute;
}

/// Helper to create a task that simulates I/O wait
pub fn createIoTask(comptime result_value: i32, comptime sleep_us: u64) *const fn() i32 {
    return struct {
        fn ioWait() i32 {
            std.time.sleep(sleep_us * 1000); // Convert to nanoseconds
            return result_value;
        }
    }.ioWait;
}

/// Helper to create a task that allocates memory for testing memory pools
pub fn createMemoryTask(comptime size: usize) *const fn() void {
    return struct {
        fn allocateMemory() void {
            const memory = std.testing.allocator.alloc(u8, size) catch return;
            defer std.testing.allocator.free(memory);
            
            // Touch the memory to prevent optimization
            for (memory) |*byte| {
                byte.* = 42;
            }
        }
    }.allocateMemory;
}

/// Direct task submission helper that works around pcall issues
pub fn submitTaskDirect(pool: *core.ThreadPool, comptime task_fn: fn() void) !void {
    const TaskWrapper = struct {
        var dummy_data: u8 = 0;
        
        fn wrapper(data: *anyopaque) void {
            _ = data;
            task_fn();
        }
    };
    
    const task = core.Task{
        .func = TaskWrapper.wrapper,
        .data = @ptrCast(&TaskWrapper.dummy_data),
    };
    
    try pool.submit(task);
}

/// Helper to submit multiple simple tasks for testing
pub fn submitSimpleTasks(pool: *core.ThreadPool, count: usize) !void {
    const VoidTask = struct {
        fn voidCpuTask() void {
            _ = simpleCpuTask();
        }
    };
    
    for (0..count) |_| {
        try submitTaskDirect(pool, VoidTask.voidCpuTask);
    }
}

// ============================================================================
// Basic Tests
// ============================================================================

test "parallel testing framework - basic functionality" {
    // Test basic task creation utilities
    const cpu_task = createCpuTask(100);
    const result = cpu_task();
    try std.testing.expect(result > 100);
    
    // Test IO task
    const io_task = createIoTask(200, 1000); // 1ms delay
    const start_time = std.time.nanoTimestamp();
    const io_result = io_task();
    const end_time = std.time.nanoTimestamp();
    
    try std.testing.expectEqual(@as(i32, 200), io_result);
    try std.testing.expect((end_time - start_time) >= 500_000); // At least 0.5ms
    
    // Test memory task
    const memory_task = createMemoryTask(1024);
    memory_task();
    
    std.debug.print("Basic task creation utilities working!\n", .{});
}

test "parallel testing framework - pool creation" {
    const pool = try createTestPool(std.testing.allocator);
    defer pool.deinit();
    
    try std.testing.expect(pool.workers.len == 4);
    try std.testing.expect(pool.running.load(.acquire) == true);
    
    // Test stats capture
    const stats = TestStats.capture(&pool.stats);
    try std.testing.expectEqual(@as(u64, 0), stats.tasks_submitted);
    
    std.debug.print("Test pool creation working!\n", .{});
}

test "parallel testing framework - test helpers" {
    const TestFunctions = struct {
        fn basicTest(pool: *core.ThreadPool) !void {
            try std.testing.expect(pool.workers.len > 0);
            try std.testing.expect(pool.running.load(.acquire) == true);
        }
        
        fn performanceTest(_: *core.ThreadPool) !void {
            // Simple delay to test timing
            std.time.sleep(1_000_000); // 1ms
        }
        
        fn stressTest(_: *core.ThreadPool, iterations: usize) !void {
            // Simple computation
            var sum: u64 = 0;
            for (0..iterations) |i| {
                sum += i;
            }
            try std.testing.expect(sum > 0);
        }
    };
    
    // Test runParallelTest
    try runParallelTest(std.testing.allocator, TestFunctions.basicTest);
    
    // Test runPerformanceTest (should complete within 100ms)
    try runPerformanceTest(std.testing.allocator, TestFunctions.performanceTest, 100);
    
    // Test runStressTest
    try runStressTest(std.testing.allocator, TestFunctions.stressTest, 1000);
    
    std.debug.print("All test helpers working correctly!\n", .{});
}

test "parallel testing framework - stats validation" {
    const pool = try createTestPool(std.testing.allocator);
    defer pool.deinit();
    
    const initial_stats = TestStats.capture(&pool.stats);
    
    // Submit some tasks directly to the pool (bypassing pcall)
    try submitSimpleTasks(pool, 5);
    
    // Allow tasks to complete
    std.time.sleep(100_000_000); // 100ms
    
    const final_stats = TestStats.capture(&pool.stats);
    const delta = TestStats.delta(final_stats, initial_stats);
    
    // Should have submitted some tasks
    try std.testing.expect(delta.tasks_submitted >= 5);
    
    std.debug.print("Statistics validation working!\n", .{});
}