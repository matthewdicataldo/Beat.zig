const std = @import("std");
const beat = @import("beat");

// Work-stealing efficiency benchmark with fast path optimization

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    std.debug.print("=== Work-Stealing Efficiency Benchmark ===\n", .{});
    
    // Test 1: Small task performance (should trigger fast path)
    {
        std.debug.print("\n-- Small Task Performance Test --\n", .{});
        
        var pool = try beat.ThreadPool.init(allocator, .{
            .num_workers = 4,
            .enable_work_stealing = true,
        });
        defer pool.deinit();
        
        const TaskData = struct { value: u32 };
        var task_data_array: [1000]TaskData = undefined;
        
        // Initialize data
        for (&task_data_array, 0..) |*data, i| {
            data.value = @intCast(i);
        }
        
        const start = std.time.nanoTimestamp();
        
        // Submit 1000 small tasks
        for (&task_data_array) |*data| {
            const task = beat.Task{
                .func = struct {
                    fn simple_increment(ptr: *anyopaque) void {
                        const typed_data = @as(*TaskData, @ptrCast(@alignCast(ptr)));
                        typed_data.value +%= 1; // Small, fast operation
                    }
                }.simple_increment,
                .data = @ptrCast(data),
                .priority = .normal,
                .data_size_hint = @sizeOf(TaskData), // Small size hint
            };
            
            try pool.submit(task);
        }
        
        pool.wait();
        const end = std.time.nanoTimestamp();
        
        const total_time = @as(u64, @intCast(end - start));
        const avg_time_per_task = total_time / 1000;
        
        // Get statistics
        const submitted = pool.stats.tasks_submitted.load(.monotonic);
        const completed = pool.stats.tasks_completed.load(.monotonic);
        const fast_path = pool.stats.fast_path_executions.load(.monotonic);
        const work_stealing_efficiency = pool.stats.getWorkStealingEfficiency();
        
        std.debug.print("  Total time: {:.2} ms\n", .{@as(f64, @floatFromInt(total_time)) / 1_000_000.0});
        std.debug.print("  Avg per task: {} ns\n", .{avg_time_per_task});
        std.debug.print("  Tasks submitted: {}\n", .{submitted});
        std.debug.print("  Tasks completed: {}\n", .{completed});
        std.debug.print("  Fast path executions: {}\n", .{fast_path});
        std.debug.print("  Fast path percentage: {:.1}%\n", .{(@as(f64, @floatFromInt(fast_path)) / @as(f64, @floatFromInt(submitted))) * 100.0});
        std.debug.print("  Work-stealing efficiency: {:.1}%\n", .{work_stealing_efficiency * 100.0});
        
        // Verify correctness
        var all_correct = true;
        for (task_data_array, 0..) |data, i| {
            if (data.value != i + 1) {
                all_correct = false;
                break;
            }
        }
        std.debug.print("  Correctness: {s}\n", .{if (all_correct) "PASS" else "FAIL"});
    }
    
    // Test 2: Medium task performance (should avoid fast path)
    {
        std.debug.print("\n-- Medium Task Performance Test --\n", .{});
        
        var pool = try beat.ThreadPool.init(allocator, .{
            .num_workers = 4,
            .enable_work_stealing = true,
        });
        defer pool.deinit();
        
        const TaskData = struct { values: [1024]f32 }; // Larger data structure
        var task_data_array: [100]TaskData = undefined;
        
        // Initialize data
        for (&task_data_array, 0..) |*data, i| {
            for (&data.values, 0..) |*val, j| {
                val.* = @as(f32, @floatFromInt(i * 1024 + j));
            }
        }
        
        const start = std.time.nanoTimestamp();
        
        // Submit 100 medium tasks
        for (&task_data_array) |*data| {
            const task = beat.Task{
                .func = struct {
                    fn matrix_operation(ptr: *anyopaque) void {
                        const typed_data = @as(*TaskData, @ptrCast(@alignCast(ptr)));
                        // More complex operation
                        for (&typed_data.values) |*val| {
                            val.* = val.* * 1.5 + 0.1;
                        }
                    }
                }.matrix_operation,
                .data = @ptrCast(data),
                .priority = .normal,
                .data_size_hint = @sizeOf(TaskData), // Larger size hint
            };
            
            try pool.submit(task);
        }
        
        pool.wait();
        const end = std.time.nanoTimestamp();
        
        const total_time = @as(u64, @intCast(end - start));
        const avg_time_per_task = total_time / 100;
        
        // Get statistics
        const submitted = pool.stats.tasks_submitted.load(.monotonic);
        const completed = pool.stats.tasks_completed.load(.monotonic);
        const fast_path = pool.stats.fast_path_executions.load(.monotonic);
        const work_stealing_efficiency = pool.stats.getWorkStealingEfficiency();
        
        std.debug.print("  Total time: {:.2} ms\n", .{@as(f64, @floatFromInt(total_time)) / 1_000_000.0});
        std.debug.print("  Avg per task: {} ns\n", .{avg_time_per_task});
        std.debug.print("  Tasks submitted: {}\n", .{submitted});
        std.debug.print("  Tasks completed: {}\n", .{completed});
        std.debug.print("  Fast path executions: {}\n", .{fast_path});
        std.debug.print("  Fast path percentage: {:.1}%\n", .{(@as(f64, @floatFromInt(fast_path)) / @as(f64, @floatFromInt(submitted))) * 100.0});
        std.debug.print("  Work-stealing efficiency: {:.1}%\n", .{work_stealing_efficiency * 100.0});
    }
    
    // Test 3: Mixed workload (realistic scenario)
    {
        std.debug.print("\n-- Mixed Workload Test --\n", .{});
        
        var pool = try beat.ThreadPool.init(allocator, .{
            .num_workers = 4,
            .enable_work_stealing = true,
        });
        defer pool.deinit();
        
        const SmallData = struct { value: u32 };
        const LargeData = struct { values: [512]f32 };
        
        var small_data_array: [500]SmallData = undefined;
        var large_data_array: [50]LargeData = undefined;
        
        // Initialize data
        for (&small_data_array, 0..) |*data, i| {
            data.value = @intCast(i);
        }
        for (&large_data_array, 0..) |*data, i| {
            for (&data.values, 0..) |*val, j| {
                val.* = @as(f32, @floatFromInt(i * 512 + j));
            }
        }
        
        const start = std.time.nanoTimestamp();
        
        // Submit mixed workload
        for (&small_data_array) |*data| {
            const task = beat.Task{
                .func = struct {
                    fn simple_increment(ptr: *anyopaque) void {
                        const typed_data = @as(*SmallData, @ptrCast(@alignCast(ptr)));
                        typed_data.value +%= 1;
                    }
                }.simple_increment,
                .data = @ptrCast(data),
                .priority = .normal,
                .data_size_hint = @sizeOf(SmallData),
            };
            try pool.submit(task);
        }
        
        for (&large_data_array) |*data| {
            const task = beat.Task{
                .func = struct {
                    fn complex_operation(ptr: *anyopaque) void {
                        const typed_data = @as(*LargeData, @ptrCast(@alignCast(ptr)));
                        for (&typed_data.values) |*val| {
                            val.* = val.* * 1.5 + 0.1;
                        }
                    }
                }.complex_operation,
                .data = @ptrCast(data),
                .priority = .normal,
                .data_size_hint = @sizeOf(LargeData),
            };
            try pool.submit(task);
        }
        
        pool.wait();
        const end = std.time.nanoTimestamp();
        
        const total_time = @as(u64, @intCast(end - start));
        const avg_time_per_task = total_time / 550; // 500 + 50 tasks
        
        // Get statistics
        const submitted = pool.stats.tasks_submitted.load(.monotonic);
        const completed = pool.stats.tasks_completed.load(.monotonic);
        const fast_path = pool.stats.fast_path_executions.load(.monotonic);
        const work_stealing_efficiency = pool.stats.getWorkStealingEfficiency();
        
        std.debug.print("  Total time: {:.2} ms\n", .{@as(f64, @floatFromInt(total_time)) / 1_000_000.0});
        std.debug.print("  Avg per task: {} ns\n", .{avg_time_per_task});
        std.debug.print("  Tasks submitted: {}\n", .{submitted});
        std.debug.print("  Tasks completed: {}\n", .{completed});
        std.debug.print("  Fast path executions: {}\n", .{fast_path});
        std.debug.print("  Fast path percentage: {:.1}%\n", .{(@as(f64, @floatFromInt(fast_path)) / @as(f64, @floatFromInt(submitted))) * 100.0});
        std.debug.print("  Work-stealing efficiency: {:.1}%\n", .{work_stealing_efficiency * 100.0});
    }
}