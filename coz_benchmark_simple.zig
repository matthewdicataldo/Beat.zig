const std = @import("std");
const beat = @import("zigpulse");

// Simplified COZ benchmark with guaranteed progress points

var global_counter: std.atomic.Value(u64) = std.atomic.Value(u64).init(0);

fn hotPathTask(data: *anyopaque) void {
    const iterations = @as(*u32, @ptrCast(@alignCast(data))).*;
    
    // Hot computation path - prime candidate for optimization
    var sum: u64 = 0;
    for (0..iterations) |i| {
        sum += i * i + 42;
        
        // Every 100 iterations, add progress point
        if (i % 100 == 0) {
            beat.coz.throughput(beat.coz.Points.task_execution);
        }
    }
    
    // Increment global counter and add progress point
    _ = global_counter.fetchAdd(1, .monotonic);
    beat.coz.throughput(beat.coz.Points.task_completed);
    
    std.mem.doNotOptimizeAway(&sum);
}

fn memoryTask(data: *anyopaque) void {
    const size = @as(*u32, @ptrCast(@alignCast(data))).*;
    
    // Memory allocation/deallocation - potential bottleneck
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    
    const buffer = allocator.alloc(u64, size) catch return;
    
    // Memory access pattern
    for (buffer, 0..) |*item, i| {
        item.* = i;
        if (i % 50 == 0) {
            beat.coz.throughput(beat.coz.Points.task_stolen);
        }
    }
    
    _ = global_counter.fetchAdd(1, .monotonic);
    beat.coz.throughput(beat.coz.Points.task_completed);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){}; 
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("🔬 COZ Profiling - Simple Benchmark\n", .{});
    std.debug.print("=====================================\n", .{});
    
    // Create pool optimized for profiling
    const config = beat.Config{
        .num_workers = 4,
        .enable_a3c_scheduling = true,
        .a3c_learning_rate = 0.001,
        .a3c_confidence_threshold = 0.5,
        .enable_statistics = true,
    };
    
    const pool = try beat.createPoolWithConfig(allocator, config);
    defer pool.deinit();
    
    std.debug.print("Pool created with {} workers\n", .{config.num_workers.?});
    std.debug.print("Starting workload...\n", .{});
    
    const start_time = std.time.nanoTimestamp();
    
    // Submit 1000 tasks with guaranteed progress points
    for (0..1000) |i| {
        var work_size: u32 = if (i % 3 == 0) 1000 else 500;
        
        const task = if (i % 2 == 0) 
            beat.Task{
                .func = hotPathTask,
                .data = &work_size,
                .priority = .normal,
            }
        else
            beat.Task{
                .func = memoryTask,
                .data = &work_size,
                .priority = .normal,
            };
        
        try pool.submit(task);
        
        // Progress point for submission
        if (i % 100 == 0) {
            beat.coz.throughput(beat.coz.Points.worker_idle);
            std.debug.print("Submitted {} tasks\n", .{i + 1});
        }
    }
    
    std.debug.print("All tasks submitted, waiting for completion...\n", .{});
    pool.wait();
    
    const end_time = std.time.nanoTimestamp();
    const duration_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
    const completed_tasks = global_counter.load(.acquire);
    
    std.debug.print("\n🏁 Results:\n", .{});
    std.debug.print("Duration: {d:.2}ms\n", .{duration_ms});
    std.debug.print("Tasks completed: {}\n", .{completed_tasks});
    std.debug.print("Tasks per second: {d:.0}\n", .{@as(f64, @floatFromInt(completed_tasks)) * 1000.0 / duration_ms});
    
    // Print stats
    std.debug.print("\n📊 Statistics:\n", .{});
    std.debug.print("Tasks submitted: {}\n", .{pool.stats.hot.tasks_submitted.load(.acquire)});
    std.debug.print("Tasks stolen: {}\n", .{pool.stats.cold.tasks_stolen.load(.acquire)});
    
    std.debug.print("\n✅ COZ progress points fired throughout execution\n", .{});
}