const std = @import("std");
const beat = @import("beat.zig");

// Comprehensive examples demonstrating Beat.zig features

// =============================================================================
// Helper Functions
// =============================================================================

fn fibonacci(n: u32) u64 {
    if (n <= 1) return n;
    var a: u64 = 0;
    var b: u64 = 1;
    var i: u32 = 2;
    while (i <= n) : (i += 1) {
        const temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

// =============================================================================
// Example 1: Basic Thread Pool Usage (V1 features)
// =============================================================================

fn example_basic_thread_pool() !void {
    std.debug.print("\n=== Example 1: Basic Thread Pool ===\n", .{});
    
    const allocator = std.heap.page_allocator;
    const pool = try beat.createPool(allocator);
    defer pool.deinit();
    
    // Simple task submission
    const print_task = struct {
        fn run(data: *anyopaque) void {
            const msg = @as(*[]const u8, @ptrCast(@alignCast(data))).*;
            std.debug.print("Worker says: {s}\n", .{msg});
        }
    }.run;
    
    var messages = [_][]const u8{
        "Hello from ZigPulse!",
        "Parallel execution rocks!",
        "Tasks are fun!",
    };
    
    for (&messages) |*msg| {
        try pool.submit(print_task, @as(*anyopaque, @ptrCast(msg)));
    }
    
    pool.wait();
}

// =============================================================================
// Example 2: Futures and Results
// =============================================================================

fn example_futures() !void {
    std.debug.print("\n=== Example 2: Futures with Results ===\n", .{});
    
    const allocator = std.heap.page_allocator;
    const pool = try beat.createPoolWithConfig(allocator, .{ .num_workers = 4 });
    defer pool.deinit();
    
    // Compute fibonacci numbers in parallel
    const compute_fib = struct {
        fn run(data: *anyopaque) !u64 {
            const n = @as(*u32, @ptrCast(@alignCast(data))).*;
            return fibonacci(n);
        }
    }.run;
    
    var nums = [_]u32{ 10, 20, 30, 35 };
    var futures: [4]beat.Future(u64) = undefined;
    
    // Spawn tasks
    for (&nums, 0..) |*n, i| {
        futures[i] = try pool.spawn(u64, compute_fib, @as(*anyopaque, @ptrCast(n)));
    }
    
    // Collect results
    for (futures, nums) |future, n| {
        const result = try future.wait();
        std.debug.print("Fibonacci({}) = {}\n", .{ n, result });
    }
}

// =============================================================================
// Example 3: Priority Scheduling
// =============================================================================

fn example_priority_scheduling() !void {
    std.debug.print("\n=== Example 3: Priority Scheduling ===\n", .{});
    
    const allocator = std.heap.page_allocator;
    const pool = try beat.createPool(allocator);
    defer pool.deinit();
    
    const priority_task = struct {
        fn run(data: *anyopaque) void {
            const info = @as(*struct { name: []const u8, delay_ms: u32 }, @ptrCast(@alignCast(data)));
            std.time.sleep(info.delay_ms * std.time.ns_per_ms);
            std.debug.print("{s} task completed\n", .{info.name});
        }
    }.run;
    
    var tasks = [_]struct { name: []const u8, delay_ms: u32 }{
        .{ .name = "Low priority", .delay_ms = 50 },
        .{ .name = "HIGH PRIORITY", .delay_ms = 10 },
        .{ .name = "Normal priority", .delay_ms = 30 },
    };
    
    try pool.submitWithPriority(priority_task, @as(*anyopaque, @ptrCast(&tasks[0])), .low);
    try pool.submitWithPriority(priority_task, @as(*anyopaque, @ptrCast(&tasks[1])), .high);
    try pool.submitWithPriority(priority_task, @as(*anyopaque, @ptrCast(&tasks[2])), .normal);
    
    pool.wait();
}

// =============================================================================
// Example 4: Parallel Patterns
// =============================================================================

fn example_parallel_patterns() !void {
    std.debug.print("\n=== Example 4: Parallel Patterns ===\n", .{});
    
    const allocator = std.heap.page_allocator;
    const pool = try beat.createPool(allocator);
    defer pool.deinit();
    
    // Parallel for-each
    var numbers: [8]i32 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    
    const double_fn = struct {
        fn run(idx: usize, item: *i32) void {
            _ = idx;
            item.* *= 2;
        }
    }.run;
    
    try pool.parallelFor(i32, &numbers, double_fn);
    
    std.debug.print("Numbers after doubling: ", .{});
    for (numbers) |n| std.debug.print("{} ", .{n});
    std.debug.print("\n", .{});
    
    // Fork-join pattern
    const task1 = struct {
        fn run() !i32 {
            std.time.sleep(10 * std.time.ns_per_ms);
            return 100;
        }
    }.run;
    
    const task2 = struct {
        fn run() !i32 {
            std.time.sleep(5 * std.time.ns_per_ms);
            return 200;
        }
    }.run;
    
    const results = try beat.join2(pool, i32, i32, task1, task2);
    std.debug.print("Fork-join results: {} and {}\n", .{ results.left, results.right });
}

// =============================================================================
// Example 5: Heartbeat Scheduling with pcall (V2 features)
// =============================================================================

fn example_heartbeat_scheduling() !void {
    std.debug.print("\n=== Example 5: Heartbeat Scheduling (V2) ===\n", .{});
    
    const allocator = std.heap.page_allocator;
    const config = beat.Config{
        .num_workers = 4,
        .enable_heartbeat = true,
        .heartbeat_interval_us = 100,
        .promotion_threshold = 10,
    };
    
    const pool = try beat.createPoolWithConfig(allocator, config);
    defer pool.deinit();
    
    // Initialize thread-local state
    beat.initThread(pool);
    
    std.debug.print("Heartbeat scheduler enabled with {}μs interval\n", .{config.heartbeat_interval_us});
    
    // Light work - should execute immediately
    const light_work = struct {
        fn compute() i32 {
            var sum: i32 = 0;
            for (0..100) |i| {
                sum += @as(i32, @intCast(i));
            }
            return sum;
        }
    }.compute;
    
    // Heavy work - should be promoted to parallel
    const heavy_work = struct {
        fn compute() u64 {
            return fibonacci(35);
        }
    }.compute;
    
    // Execute multiple pcalls
    for (0..5) |i| {
        var light_future = beat.pcall(i32, light_work);
        const light_result = try light_future.get();
        std.debug.print("Light work {}: result = {}\n", .{ i, light_result });
        
        if (i >= 3) {
            var heavy_future = beat.pcall(u64, heavy_work);
            const heavy_result = try heavy_future.get();
            std.debug.print("Heavy work {}: result = {} (likely promoted)\n", .{ i, heavy_result });
        }
    }
    
    pool.wait();
}

// =============================================================================
// Example 6: Zero-Overhead pcallMinimal
// =============================================================================

fn example_zero_overhead() !void {
    std.debug.print("\n=== Example 6: Zero-Overhead pcallMinimal ===\n", .{});
    
    const allocator = std.heap.page_allocator;
    const pool = try beat.createPool(allocator);
    defer pool.deinit();
    
    beat.initThread(pool);
    
    const compute = struct {
        fn run() i32 {
            var sum: i32 = 0;
            for (0..1000) |i| {
                sum += @as(i32, @intCast(i));
            }
            return sum;
        }
    }.run;
    
    // This has zero overhead in release mode
    const result = beat.pcallMinimal(i32, compute);
    std.debug.print("Result: {} (computed with zero overhead in release)\n", .{result});
}

// =============================================================================
// Example 7: Real-World Use Cases
// =============================================================================

fn example_real_world() !void {
    std.debug.print("\n=== Example 7: Real-World Use Cases ===\n", .{});
    
    const allocator = std.heap.page_allocator;
    const pool = try beat.createPool(allocator);
    defer pool.deinit();
    
    // Parallel HTTP request parsing (simulated)
    std.debug.print("\n--- Parallel HTTP Parsing ---\n", .{});
    const parseRequest = struct {
        fn parse(data: *anyopaque) void {
            const request = @as(*[]const u8, @ptrCast(@alignCast(data))).*;
            var checksum: u32 = 0;
            for (request) |byte| {
                checksum +%= byte;
            }
            std.debug.print("Parsed request with checksum: {}\n", .{checksum});
        }
    }.parse;
    
    var requests = [_][]const u8{
        "GET /api/users HTTP/1.1\r\nHost: example.com\r\n\r\n",
        "POST /api/data HTTP/1.1\r\nHost: example.com\r\n\r\n",
        "GET /health HTTP/1.1\r\nHost: example.com\r\n\r\n",
    };
    
    for (&requests) |*req| {
        try pool.submit(parseRequest, @as(*anyopaque, @ptrCast(req)));
    }
    
    // Parallel compression (simulated)
    std.debug.print("\n--- Parallel Compression ---\n", .{});
    const DataBlock = struct {
        id: u32,
        size: u32,
        compressed_size: u32 = 0,
    };
    
    const compressBlock = struct {
        fn compress(data: *anyopaque) void {
            const block = @as(*DataBlock, @ptrCast(@alignCast(data)));
            // Simulate compression
            block.compressed_size = block.size * 6 / 10;
            std.debug.print("Compressed block {} from {} to {} bytes\n", .{
                block.id, block.size, block.compressed_size
            });
        }
    }.compress;
    
    var blocks = [_]DataBlock{
        .{ .id = 0, .size = 1024 },
        .{ .id = 1, .size = 2048 },
        .{ .id = 2, .size = 4096 },
    };
    
    for (&blocks) |*block| {
        try pool.submit(compressBlock, @as(*anyopaque, @ptrCast(block)));
    }
    
    pool.wait();
}

// =============================================================================
// Example 8: Performance Comparison
// =============================================================================

fn example_performance_comparison() !void {
    std.debug.print("\n=== Example 8: Performance Comparison ===\n", .{});
    
    const allocator = std.heap.page_allocator;
    
    // Test different configurations
    const configs = [_]struct {
        name: []const u8,
        config: beat.Config,
    }{
        .{
            .name = "Basic (V1)",
            .config = .{ .num_workers = 4, .enable_statistics = true },
        },
        .{
            .name = "With Heartbeat (V2)",
            .config = .{ .num_workers = 4, .enable_heartbeat = true, .enable_statistics = true },
        },
        .{
            .name = "Optimized",
            .config = .{ .num_workers = 4, .enable_heartbeat = true, .use_fast_rdtsc = true },
        },
    };
    
    const benchmark_task = struct {
        fn run(data: *anyopaque) void {
            const iterations = @as(*u32, @ptrCast(@alignCast(data))).*;
            var sum: u64 = 0;
            for (0..iterations) |i| {
                sum += i;
            }
            std.mem.doNotOptimizeAway(&sum);
        }
    }.run;
    
    for (configs) |test_config| {
        std.debug.print("\n--- {s} Configuration ---\n", .{test_config.name});
        
        const pool = try beat.createPoolWithConfig(allocator, test_config.config);
        defer pool.deinit();
        
        var iterations: u32 = 10000;
        const start = std.time.milliTimestamp();
        
        for (0..100) |_| {
            try pool.submit(benchmark_task, @as(*anyopaque, @ptrCast(&iterations)));
        }
        
        pool.wait();
        const elapsed = std.time.milliTimestamp() - start;
        
        std.debug.print("Completed 100 tasks in {}ms\n", .{elapsed});
        
        if (test_config.config.enable_statistics) {
            const stats = pool.getStats();
            std.debug.print("  Tasks completed: {}\n", .{stats.tasks_completed.load(.acquire)});
        }
    }
}

// =============================================================================
// Main Function
// =============================================================================

pub fn main() !void {
    std.debug.print("=== ZigPulse Examples ===\n", .{});
    std.debug.print("Ultra-low-overhead parallelism for Zig\n", .{});
    
    try example_basic_thread_pool();
    try example_futures();
    try example_priority_scheduling();
    try example_parallel_patterns();
    try example_heartbeat_scheduling();
    try example_zero_overhead();
    try example_real_world();
    try example_performance_comparison();
    
    std.debug.print("\n=== All examples completed successfully! ===\n", .{});
}