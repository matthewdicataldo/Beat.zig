const std = @import("std");
const beat = @import("src/core.zig");

test "Smart Worker Selection - Task Affinity" {
    const allocator = std.testing.allocator;
    
    // Create pool with explicit NUMA configuration
    const pool = try beat.createPoolWithConfig(allocator, .{
        .num_workers = 4,
        .enable_topology_aware = true,
        .enable_statistics = true,
    });
    defer pool.deinit();
    
    // Test task submission with affinity hint
    var counter = std.atomic.Value(i32).init(0);
    
    const increment_task = struct {
        fn run(data: *anyopaque) void {
            const cnt = @as(*std.atomic.Value(i32), @ptrCast(@alignCast(data)));
            _ = cnt.fetchAdd(1, .monotonic);
        }
    }.run;
    
    // Submit tasks with NUMA affinity hints
    try pool.submit(.{ 
        .func = increment_task, 
        .data = &counter, 
        .affinity_hint = 0 // Prefer NUMA node 0
    });
    
    try pool.submit(.{ 
        .func = increment_task, 
        .data = &counter, 
        .affinity_hint = 1 // Prefer NUMA node 1 (if available)
    });
    
    pool.wait();
    
    // Verify tasks completed
    try std.testing.expect(counter.load(.acquire) == 2);
}

test "Smart Worker Selection - Load Balancing" {
    const allocator = std.testing.allocator;
    
    const pool = try beat.createPoolWithConfig(allocator, .{
        .num_workers = 4,
        .enable_topology_aware = true,
        .enable_statistics = true,
    });
    defer pool.deinit();
    
    var counter = std.atomic.Value(i32).init(0);
    
    const increment_task = struct {
        fn run(data: *anyopaque) void {
            const cnt = @as(*std.atomic.Value(i32), @ptrCast(@alignCast(data)));
            _ = cnt.fetchAdd(1, .monotonic);
            // Add small delay to create queue buildup
            std.time.sleep(1_000_000); // 1ms
        }
    }.run;
    
    // Submit many tasks to test load balancing
    for (0..20) |_| {
        try pool.submit(.{ 
            .func = increment_task, 
            .data = &counter 
        });
    }
    
    pool.wait();
    
    // Verify all tasks completed
    try std.testing.expect(counter.load(.acquire) == 20);
}

test "Smart Worker Selection - Performance" {
    const allocator = std.testing.allocator;
    
    const pool = try beat.createOptimalPool(allocator);
    defer pool.deinit();
    
    var counter = std.atomic.Value(u64).init(0);
    
    const fast_task = struct {
        fn run(data: *anyopaque) void {
            const cnt = @as(*std.atomic.Value(u64), @ptrCast(@alignCast(data)));
            _ = cnt.fetchAdd(1, .monotonic);
        }
    }.run;
    
    const start_time = std.time.nanoTimestamp();
    
    // Submit many small tasks to test selection overhead
    for (0..1000) |_| {
        try pool.submit(.{ 
            .func = fast_task, 
            .data = &counter 
        });
    }
    
    pool.wait();
    const end_time = std.time.nanoTimestamp();
    
    // Verify all tasks completed
    try std.testing.expect(counter.load(.acquire) == 1000);
    
    const total_time = @as(u64, @intCast(end_time - start_time));
    const avg_time_per_task = total_time / 1000;
    
    std.debug.print("Smart worker selection: {} tasks in {}μs ({}ns per task)\n", .{
        1000, total_time / 1000, avg_time_per_task
    });
    
    // Performance should be reasonable (less than 100μs per task)
    try std.testing.expect(avg_time_per_task < 100_000);
}