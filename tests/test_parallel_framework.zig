const std = @import("std");
const beat = @import("beat");

// Demonstration of the Enhanced Parallel Testing Framework
// This file shows how to use the new testing utilities for Beat.zig

// ============================================================================
// Basic Parallel Tests
// ============================================================================

const BasicParallelTests = struct {
    fn testSimpleTaskExecution(pool: *beat.ThreadPool) !void {
        const task = beat.testing.createCpuTask(42);
        
        var future = beat.pcall.pcall(i32, task);
        const result = try future.get();
        
        // Result should be the base value plus the computed sum
        try std.testing.expect(result > 42);
    }
    
    fn testMultipleTaskExecution(pool: *beat.ThreadPool) !void {
        var tasks: [10]*const fn() i32 = undefined;
        for (tasks, 0..) |*task, i| {
            task.* = beat.testing.createCpuTask(@intCast(i * 10));
        }
        
        const results = try beat.testing.submitAndWaitTasks(pool, i32, &tasks);
        defer std.testing.allocator.free(results);
        
        try std.testing.expectEqual(@as(usize, 10), results.len);
        
        // Each result should be greater than its base value
        for (results, 0..) |result, i| {
            try std.testing.expect(result > @as(i32, @intCast(i * 10)));
        }
    }
    
    fn testWorkStealingDetection(pool: *beat.ThreadPool) !void {
        // Submit enough tasks to trigger work stealing
        try beat.testing.testWorkStealing(pool, 50);
    }
    
    fn testIoSimulation(pool: *beat.ThreadPool) !void {
        const io_task = beat.testing.createIoTask(100, 1000); // 1ms delay
        
        var future = beat.pcall.pcall(i32, io_task);
        const result = try future.get();
        
        try std.testing.expectEqual(@as(i32, 100), result);
    }
};

// ============================================================================
// Performance Tests
// ============================================================================

const PerformanceTests = struct {
    fn testThroughputPerformance(pool: *beat.ThreadPool) !void {
        const task_count = 1000;
        var tasks: [task_count]*const fn() i32 = undefined;
        
        for (tasks, 0..) |*task, i| {
            task.* = beat.testing.createCpuTask(@intCast(i));
        }
        
        const start_time = std.time.nanoTimestamp();
        const results = try beat.testing.submitAndWaitTasks(pool, i32, &tasks);
        const end_time = std.time.nanoTimestamp();
        
        defer std.testing.allocator.free(results);
        
        const duration_ms = (end_time - start_time) / 1_000_000;
        std.debug.print("Processed {d} tasks in {d}ms\n", .{ task_count, duration_ms });
        
        // Should complete within reasonable time (adjust based on system)
        try std.testing.expect(duration_ms < 5000); // 5 seconds max
    }
    
    fn testLatencyPerformance(pool: *beat.ThreadPool) !void {
        const task = beat.testing.createCpuTask(1);
        
        const start_time = std.time.nanoTimestamp();
        var future = beat.pcall.pcall(i32, task);
        const result = try future.get();
        const end_time = std.time.nanoTimestamp();
        
        _ = result;
        
        const latency_us = (end_time - start_time) / 1000;
        std.debug.print("Single task latency: {d}μs\n", .{latency_us});
        
        // Should have low latency (adjust based on system)
        try std.testing.expect(latency_us < 10000); // 10ms max
    }
};

// ============================================================================
// Stress Tests
// ============================================================================

const StressTests = struct {
    fn testHighConcurrency(pool: *beat.ThreadPool, iterations: usize) !void {
        const task_count = iterations;
        var futures = try std.testing.allocator.alloc(beat.pcall.PotentialFuture(i32), task_count);
        defer std.testing.allocator.free(futures);
        
        // Submit all tasks concurrently
        for (futures, 0..) |*future, i| {
            const task = beat.testing.createCpuTask(@intCast(i));
            future.* = beat.pcall.pcall(i32, task);
        }
        
        // Wait for all results
        for (futures, 0..) |*future, i| {
            const result = try future.get();
            try std.testing.expect(result > @as(i32, @intCast(i)));
        }
    }
    
    fn testMemoryPressure(pool: *beat.ThreadPool, iterations: usize) !void {
        var futures = try std.testing.allocator.alloc(beat.pcall.PotentialFuture(void), iterations);
        defer std.testing.allocator.free(futures);
        
        // Submit memory-intensive tasks
        for (futures) |*future| {
            const task = beat.testing.createMemoryTask(std.testing.allocator, 1024);
            future.* = beat.pcall.pcall(void, task);
        }
        
        // Wait for completion
        for (futures) |*future| {
            try future.get();
        }
    }
};

// ============================================================================
// Actual Test Declarations
// ============================================================================

// Use the parallel testing framework
beat.testing.parallelTest("simple task execution", BasicParallelTests.testSimpleTaskExecution);
beat.testing.parallelTest("multiple task execution", BasicParallelTests.testMultipleTaskExecution);
beat.testing.parallelTest("work stealing detection", BasicParallelTests.testWorkStealingDetection);
beat.testing.parallelTest("io simulation", BasicParallelTests.testIoSimulation);

// Performance tests with timing constraints
beat.testing.parallelPerfTest("throughput performance", PerformanceTests.testThroughputPerformance, 5000);
beat.testing.parallelPerfTest("latency performance", PerformanceTests.testLatencyPerformance, 100);

// Stress tests with high iteration counts
beat.testing.parallelStressTest("high concurrency", StressTests.testHighConcurrency, 500);
beat.testing.parallelStressTest("memory pressure", StressTests.testMemoryPressure, 100);

// ============================================================================
// Integration Tests for Existing Modules
// ============================================================================

const IntegrationTests = struct {
    fn testMemoryPoolIntegration(pool: *beat.ThreadPool) !void {
        // Test that memory pools work correctly under parallel load
        const MemoryTask = struct {
            fn allocateAndProcess() i32 {
                var typed_pool = beat.memory.TypedPool(u64).init(std.testing.allocator);
                defer typed_pool.deinit();
                
                const ptr = typed_pool.alloc() catch return -1;
                ptr.* = 12345;
                const value = ptr.*;
                typed_pool.free(ptr);
                
                return @intCast(value);
            }
        };
        
        var futures: [20]beat.pcall.PotentialFuture(i32) = undefined;
        for (&futures) |*future| {
            future.* = beat.pcall.pcall(i32, MemoryTask.allocateAndProcess);
        }
        
        for (futures) |*future| {
            const result = try future.get();
            try std.testing.expectEqual(@as(i32, 12345), result);
        }
    }
    
    fn testTopologyAwareScheduling(pool: *beat.ThreadPool) !void {
        // Test that topology-aware features work correctly
        if (pool.topology == null) {
            return std.testing.skip(); // Skip if topology detection disabled
        }
        
        const tasks = [_]*const fn() i32{
            beat.testing.createCpuTask(1),
            beat.testing.createCpuTask(2),
            beat.testing.createCpuTask(3),
            beat.testing.createCpuTask(4),
        };
        
        const results = try beat.testing.submitAndWaitTasks(pool, i32, &tasks);
        defer std.testing.allocator.free(results);
        
        try std.testing.expectEqual(@as(usize, 4), results.len);
    }
};

beat.testing.parallelTest("memory pool integration", IntegrationTests.testMemoryPoolIntegration);
beat.testing.parallelTest("topology aware scheduling", IntegrationTests.testTopologyAwareScheduling);

// ============================================================================
// Documentation Test
// ============================================================================

test "parallel testing framework documentation" {
    // This test demonstrates the API and serves as documentation
    
    // 1. Basic parallel test - provides ThreadPool and validates cleanup
    const MyTest = struct {
        fn myParallelTest(pool: *beat.ThreadPool) !void {
            // Your test code here - pool is ready to use
            var future = beat.pcall.pcall(i32, beat.testing.createCpuTask(42));
            const result = try future.get();
            try std.testing.expect(result > 42);
        }
    };
    
    // Register the test (this would normally be at module level)
    // beat.testing.parallelTest("my test", MyTest.myParallelTest);
    
    // 2. Performance test - includes timing validation
    const MyPerfTest = struct {
        fn myPerformanceTest(pool: *beat.ThreadPool) !void {
            // Test that should complete within time limit
            var future = beat.pcall.pcall(i32, beat.testing.createCpuTask(1));
            _ = try future.get();
        }
    };
    
    // Register with 100ms time limit
    // beat.testing.parallelPerfTest("my perf test", MyPerfTest.myPerformanceTest, 100);
    
    // 3. Stress test - runs with many iterations
    const MyStressTest = struct {
        fn myStressTest(pool: *beat.ThreadPool, iterations: usize) !void {
            // Test with high load
            for (0..iterations) |_| {
                var future = beat.pcall.pcall(i32, beat.testing.createCpuTask(1));
                _ = try future.get();
            }
        }
    };
    
    // Register with 1000 iterations
    // beat.testing.parallelStressTest("my stress test", MyStressTest.myStressTest, 1000);
    
    std.debug.print("Parallel testing framework ready for use!\n", .{});
}