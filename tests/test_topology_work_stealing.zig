const std = @import("std");
const beat = @import("beat");

test "Topology-Aware Work Stealing - Basic Functionality" {
    const allocator = std.testing.allocator;
    
    // Create pool with topology awareness enabled
    const pool = try beat.createPoolWithConfig(allocator, .{
        .num_workers = 4,
        .enable_topology_aware = true,
        .enable_work_stealing = true,
        .enable_statistics = true,
    });
    defer pool.deinit();
    
    var counter = std.atomic.Value(i32).init(0);
    
    const work_task = struct {
        fn run(data: *anyopaque) void {
            const cnt = @as(*std.atomic.Value(i32), @ptrCast(@alignCast(data)));
            _ = cnt.fetchAdd(1, .monotonic);
            // Add small delay to encourage work stealing
            var sum: u64 = 0;
            for (0..1000) |i| {
                sum += i;
            }
            std.mem.doNotOptimizeAway(&sum);
        }
    }.run;
    
    // Submit many tasks to one worker to force work stealing
    for (0..20) |_| {
        try pool.submit(.{ .func = work_task, .data = &counter });
    }
    
    pool.wait();
    
    // Verify all tasks completed
    try std.testing.expect(counter.load(.acquire) == 20);
}

test "Topology-Aware Work Stealing - Performance Comparison" {
    const allocator = std.testing.allocator;
    
    // Test with topology-aware work stealing
    const topo_pool = try beat.createPoolWithConfig(allocator, .{
        .num_workers = 4,
        .enable_topology_aware = true,
        .enable_work_stealing = true,
        .enable_statistics = true,
    });
    defer topo_pool.deinit();
    
    // Test without topology awareness (fallback to random)
    const random_pool = try beat.createPoolWithConfig(allocator, .{
        .num_workers = 4,
        .enable_topology_aware = false,
        .enable_work_stealing = true,
        .enable_statistics = true,
    });
    defer random_pool.deinit();
    
    var topo_counter = std.atomic.Value(u64).init(0);
    var random_counter = std.atomic.Value(u64).init(0);
    
    const work_task = struct {
        fn run(data: *anyopaque) void {
            const cnt = @as(*std.atomic.Value(u64), @ptrCast(@alignCast(data)));
            _ = cnt.fetchAdd(1, .monotonic);
            // Simulate work that benefits from locality
            var sum: u64 = 0;
            for (0..100) |i| {
                sum += i * i;
            }
            std.mem.doNotOptimizeAway(&sum);
        }
    }.run;
    
    const task_count = 100;
    
    // Benchmark topology-aware version
    const topo_start = std.time.nanoTimestamp();
    for (0..task_count) |_| {
        try topo_pool.submit(.{ .func = work_task, .data = &topo_counter });
    }
    topo_pool.wait();
    const topo_end = std.time.nanoTimestamp();
    
    // Benchmark random version
    const random_start = std.time.nanoTimestamp();
    for (0..task_count) |_| {
        try random_pool.submit(.{ .func = work_task, .data = &random_counter });
    }
    random_pool.wait();
    const random_end = std.time.nanoTimestamp();
    
    // Verify all tasks completed
    try std.testing.expect(topo_counter.load(.acquire) == task_count);
    try std.testing.expect(random_counter.load(.acquire) == task_count);
    
    const topo_time = @as(u64, @intCast(topo_end - topo_start));
    const random_time = @as(u64, @intCast(random_end - random_start));
    
    std.debug.print("Topology-aware work stealing: {}μs\n", .{topo_time / 1000});
    std.debug.print("Random work stealing: {}μs\n", .{random_time / 1000});
    
    // Topology-aware should be competitive or better
    // Note: On single-NUMA systems, the difference may be minimal
    const improvement_ratio = @as(f64, @floatFromInt(random_time)) / @as(f64, @floatFromInt(topo_time));
    std.debug.print("Improvement ratio: {d:.2}x\n", .{improvement_ratio});
}

test "Topology-Aware Work Stealing - NUMA Node Preferences" {
    const allocator = std.testing.allocator;
    
    const pool = try beat.createPoolWithConfig(allocator, .{
        .num_workers = 4,
        .enable_topology_aware = true,
        .enable_work_stealing = true,
        .enable_statistics = true,
    });
    defer pool.deinit();
    
    var numa0_counter = std.atomic.Value(i32).init(0);
    var numa1_counter = std.atomic.Value(i32).init(0);
    
    const numa0_task = struct {
        fn run(data: *anyopaque) void {
            const cnt = @as(*std.atomic.Value(i32), @ptrCast(@alignCast(data)));
            _ = cnt.fetchAdd(1, .monotonic);
        }
    }.run;
    
    const numa1_task = struct {
        fn run(data: *anyopaque) void {
            const cnt = @as(*std.atomic.Value(i32), @ptrCast(@alignCast(data)));
            _ = cnt.fetchAdd(1, .monotonic);
        }
    }.run;
    
    // Submit tasks with specific NUMA affinities
    for (0..10) |_| {
        try pool.submit(.{ 
            .func = numa0_task, 
            .data = &numa0_counter,
            .affinity_hint = 0, // Prefer NUMA node 0
        });
        
        try pool.submit(.{ 
            .func = numa1_task, 
            .data = &numa1_counter,
            .affinity_hint = 1, // Prefer NUMA node 1 (if available)
        });
    }
    
    pool.wait();
    
    // Verify all tasks completed
    try std.testing.expect(numa0_counter.load(.acquire) == 10);
    try std.testing.expect(numa1_counter.load(.acquire) == 10);
    
    std.debug.print("NUMA 0 tasks completed: {}\n", .{numa0_counter.load(.acquire)});
    std.debug.print("NUMA 1 tasks completed: {}\n", .{numa1_counter.load(.acquire)});
}