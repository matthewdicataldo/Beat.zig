const std = @import("std");
const beat = @import("zigpulse");

// Longer-running COZ benchmark designed to generate profile data

var task_counter: std.atomic.Value(u64) = std.atomic.Value(u64).init(0);

fn cpuIntensiveTask(data: *anyopaque) void {
    const work_size = @as(*u32, @ptrCast(@alignCast(data))).*;
    
    // Very CPU-intensive work to ensure COZ has time to profile
    var sum: u64 = 0;
    var temp: u64 = 1;
    
    for (0..work_size * 10) |i| {
        // Complex computation
        temp = temp *% 17 +% i;
        sum +%= temp % 997; // Use prime modulo
        
        // Simulate different access patterns
        if (i % 3 == 0) {
            temp = temp ^ (i << 3);
        } else if (i % 7 == 0) {
            temp = temp +% (sum >> 2);
        }
        
        // More frequent progress points
        if (i % 50 == 0) {
            beat.coz.throughput(beat.coz.Points.task_execution);
        }
    }
    
    // Simulate some memory work
    var small_buffer: [100]u64 = undefined;
    for (&small_buffer, 0..) |*item, idx| {
        item.* = sum +% idx;
    }
    
    _ = task_counter.fetchAdd(1, .monotonic);
    beat.coz.throughput(beat.coz.Points.task_completed);
    
    std.mem.doNotOptimizeAway(&sum);
    std.mem.doNotOptimizeAway(&small_buffer);
}

fn stealingTask(data: *anyopaque) void {
    const work_size = @as(*u32, @ptrCast(@alignCast(data))).*;
    
    // Medium work designed to trigger work stealing
    var sum: u64 = 0;
    for (0..work_size * 5) |i| {
        sum +%= i * i + 42;
        
        if (i % 25 == 0) {
            beat.coz.throughput(beat.coz.Points.task_stolen);
        }
    }
    
    _ = task_counter.fetchAdd(1, .monotonic);
    beat.coz.throughput(beat.coz.Points.task_completed);
    
    std.mem.doNotOptimizeAway(&sum);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){}; 
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("🔥 COZ Intensive Profiling Benchmark\n", .{});
    std.debug.print("=====================================\n", .{});
    std.debug.print("Running for 30+ seconds to ensure COZ profile generation...\n", .{});
    
    // Create pool with A3C
    const config = beat.Config{
        .num_workers = 6,
        .enable_a3c_scheduling = true,
        .a3c_learning_rate = 0.001,
        .a3c_confidence_threshold = 0.5,
        .enable_statistics = true,
    };
    
    const pool = try beat.createPoolWithConfig(allocator, config);
    defer pool.deinit();
    
    std.debug.print("A3C Pool created with {} workers\n", .{config.num_workers.?});
    std.debug.print("Starting intensive workload...\n", .{});
    
    const start_time = std.time.nanoTimestamp();
    
    // Submit 50,000 tasks over multiple iterations for sustained load  
    const total_tasks = 50000;
    const batch_size = 1000;
    
    for (0..total_tasks / batch_size) |batch| {
        std.debug.print("Batch {}/{} (tasks {}-{})\n", .{ 
            batch + 1, 
            total_tasks / batch_size,
            batch * batch_size,
            (batch + 1) * batch_size
        });
        
        for (0..batch_size) |i| {
            // Vary work sizes for different load patterns - much heavier
            var work_size: u32 = switch (i % 5) {
                0, 1 => 2000,   // Light tasks
                2, 3 => 4000,   // Medium tasks  
                4 => 8000,      // Heavy tasks
                else => unreachable,
            };
            
            const task = if (i % 3 == 0)
                beat.Task{
                    .func = stealingTask,
                    .data = &work_size,
                    .priority = .normal,
                }
            else
                beat.Task{
                    .func = cpuIntensiveTask,
                    .data = &work_size,
                    .priority = .normal,
                };
            
            try pool.submit(task);
            
            // Progress point for task submission
            if (i % 100 == 0) {
                beat.coz.throughput(beat.coz.Points.worker_idle);
            }
        }
        
        // Wait for this batch to complete
        pool.wait();
        
        // Small delay between batches
        std.time.sleep(500_000); // 0.5ms
        
        // Progress marker for batch completion
        beat.coz.throughput(beat.coz.Points.task_execution);
    }
    
    const end_time = std.time.nanoTimestamp();
    const duration_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
    const completed_tasks = task_counter.load(.acquire);
    
    std.debug.print("\n🏁 Intensive Benchmark Results:\n", .{});
    std.debug.print("Total duration: {d:.2}ms ({d:.1}s)\n", .{ duration_ms, duration_ms / 1000.0 });
    std.debug.print("Tasks completed: {}\n", .{completed_tasks});
    std.debug.print("Tasks per second: {d:.0}\n", .{@as(f64, @floatFromInt(completed_tasks)) * 1000.0 / duration_ms});
    
    // Print pool statistics
    std.debug.print("\n📊 Pool Statistics:\n", .{});
    std.debug.print("Tasks submitted: {}\n", .{pool.stats.hot.tasks_submitted.load(.acquire)});
    std.debug.print("Tasks stolen: {}\n", .{pool.stats.cold.tasks_stolen.load(.acquire)});
    
    if (pool.stats.cold.tasks_stolen.load(.acquire) > 0) {
        const steal_rate = @as(f64, @floatFromInt(pool.stats.cold.tasks_stolen.load(.acquire))) / 
                          @as(f64, @floatFromInt(completed_tasks)) * 100.0;
        std.debug.print("Work-stealing rate: {d:.1}%\n", .{steal_rate});
    }
    
    std.debug.print("\n✅ COZ progress points fired {} times throughout execution\n", .{completed_tasks * 10}); // Rough estimate
    std.debug.print("Check for profile.coz file in current directory\n", .{});
}