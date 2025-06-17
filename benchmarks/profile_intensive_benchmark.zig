const std = @import("std");
const beat = @import("zigpulse");

// Intensive benchmark for profiling hotspots

const ITERATIONS = 50;
const TASKS_PER_ITERATION = 200;
const WORK_SIZES = [_]u32{ 100, 1000, 5000, 10000 };

fn lightComputeTask(data: *anyopaque) void {
    const work_size = @as(*u32, @ptrCast(@alignCast(data))).*;
    
    var sum: u64 = 0;
    for (0..work_size) |i| {
        sum += i * 17; // Prime number for less predictable computation
    }
    
    // Prevent optimization
    std.mem.doNotOptimizeAway(&sum);
    
    // COZ progress point
    beat.coz.throughput(beat.coz.Points.task_completed);
}

fn mediumComputeTask(data: *anyopaque) void {
    const work_size = @as(*u32, @ptrCast(@alignCast(data))).*;
    
    var sum: u64 = 0;
    var temp: u64 = 1;
    
    for (0..work_size) |i| {
        temp = temp * 31 + i; // More complex computation
        sum += temp % 1000;
        
        // Memory access pattern
        if (i % 100 == 0) {
            sum = sum ^ (temp >> 8);
        }
    }
    
    std.mem.doNotOptimizeAway(&sum);
    beat.coz.throughput(beat.coz.Points.task_execution);
}

fn heavyComputeTask(data: *anyopaque) void {
    const work_size = @as(*u32, @ptrCast(@alignCast(data))).*;
    
    // Simulate heavy computation with memory allocation
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    
    const buffer = allocator.alloc(u64, work_size / 10) catch return;
    
    var sum: u64 = 0;
    for (buffer, 0..) |*item, i| {
        item.* = i * i + 42;
        sum += item.*;
        
        // More memory access
        if (i > 0) {
            sum += buffer[i - 1];
        }
    }
    
    std.mem.doNotOptimizeAway(&sum);
    beat.coz.throughput(beat.coz.Points.task_stolen);
}

fn memoryIntensiveTask(data: *anyopaque) void {
    const work_size = @as(*u32, @ptrCast(@alignCast(data))).*;
    
    // Large memory allocation to stress memory subsystem
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    
    const large_buffer = allocator.alloc(u64, work_size) catch return;
    
    // Random memory access pattern
    var rng = std.Random.DefaultPrng.init(@as(u64, @intCast(std.time.nanoTimestamp())));
    
    for (0..work_size / 10) |_| {
        const idx = rng.random().uintLessThan(usize, large_buffer.len);
        large_buffer[idx] = large_buffer[idx] +% 1;
    }
    
    beat.coz.throughput(beat.coz.Points.worker_idle);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){}; 
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.log.info("🔥 Intensive Performance Profiling Benchmark", .{});
    std.log.info("Iterations: {}, Tasks per iteration: {}", .{ ITERATIONS, TASKS_PER_ITERATION });
    
    // Create A3C-enabled pool with statistics
    const config = beat.Config{
        .num_workers = 8, // More workers for better parallelism
        .enable_a3c_scheduling = true,
        .a3c_learning_rate = 0.001,
        .a3c_confidence_threshold = 0.6,
        .a3c_exploration_rate = 0.15,
        .enable_statistics = true,
        .enable_trace = false, // Disable trace for performance
    };
    
    const pool = try beat.createPoolWithConfig(allocator, config);
    defer pool.deinit();
    
    std.log.info("A3C Pool initialized with {} workers", .{config.num_workers.?});
    
    const start_time = std.time.nanoTimestamp();
    
    // Run intensive benchmark
    for (0..ITERATIONS) |iteration| {
        if (iteration % 10 == 0) {
            std.log.info("Progress: {}/{} iterations", .{ iteration, ITERATIONS });
        }
        
        // Submit varied workload
        for (0..TASKS_PER_ITERATION) |i| {
            const work_size_idx = i % WORK_SIZES.len;
            var work_size = WORK_SIZES[work_size_idx];
            
            const task = switch (i % 10) {
                0, 1, 2, 3, 4 => beat.Task{ // 50% light tasks
                    .func = lightComputeTask,
                    .data = &work_size,
                    .priority = .normal,
                    .data_size_hint = work_size,
                },
                5, 6, 7 => beat.Task{ // 30% medium tasks
                    .func = mediumComputeTask,
                    .data = &work_size,
                    .priority = .normal,
                    .data_size_hint = work_size * 2,
                },
                8 => beat.Task{ // 10% heavy tasks
                    .func = heavyComputeTask,
                    .data = &work_size,
                    .priority = .high,
                    .data_size_hint = work_size * 5,
                },
                9 => beat.Task{ // 10% memory intensive
                    .func = memoryIntensiveTask,
                    .data = &work_size,
                    .priority = .high,
                    .data_size_hint = work_size * 10,
                },
                else => unreachable,
            };
            
            try pool.submit(task);
        }
        
        // Wait for completion
        pool.wait();
        
        // Small breathing room
        if (iteration % 5 == 4) {
            std.time.sleep(1_000_000); // 1ms every 5 iterations
        }
    }
    
    const end_time = std.time.nanoTimestamp();
    const total_time_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
    const total_tasks = ITERATIONS * TASKS_PER_ITERATION;
    
    std.log.info("", .{});
    std.log.info("🏁 Benchmark Complete!", .{});
    std.log.info("Total time: {d:.2}ms", .{total_time_ms});
    std.log.info("Total tasks: {}", .{total_tasks});
    std.log.info("Tasks per second: {d:.0}", .{@as(f64, @floatFromInt(total_tasks)) * 1000.0 / total_time_ms});
    std.log.info("Average task time: {d:.3}ms", .{total_time_ms / @as(f64, @floatFromInt(total_tasks))});
    
    // Print A3C performance metrics
    if (pool.a3c_scheduler) |a3c| {
        std.log.info("", .{});
        std.log.info("🧠 A3C Performance Metrics:", .{});
        a3c.logPerformanceMetrics();
    }
    
    // Print work-stealing statistics
    std.log.info("", .{});
    std.log.info("📊 Work-Stealing Statistics:", .{});
    std.log.info("Tasks submitted: {}", .{pool.stats.hot.tasks_submitted.load(.acquire)});
    std.log.info("Tasks completed: {}", .{pool.stats.cold.tasks_completed.load(.acquire)});
    std.log.info("Tasks stolen: {}", .{pool.stats.cold.tasks_stolen.load(.acquire)});
    
    const steal_rate = @as(f64, @floatFromInt(pool.stats.cold.tasks_stolen.load(.acquire))) / 
                      @as(f64, @floatFromInt(pool.stats.cold.tasks_completed.load(.acquire))) * 100.0;
    std.log.info("Work-stealing rate: {d:.1}%", .{steal_rate});
}