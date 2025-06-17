const std = @import("std");
const zigpulse = @import("zigpulse");

// Simple COZ benchmark for profiling Beat.zig A3C performance

fn simpleTask(data: *anyopaque) void {
    const value = @as(*u32, @ptrCast(@alignCast(data)));
    
    // Simple computation with COZ progress point
    var sum: u64 = 0;
    for (0..1000) |i| {
        sum += i * value.*;
    }
    
    value.* = @intCast(sum % 1000);
    
    // COZ progress point for task completion
    zigpulse.coz.throughput(zigpulse.coz.Points.task_completed);
}

fn heavyTask(data: *anyopaque) void {
    const value = @as(*u32, @ptrCast(@alignCast(data)));
    
    // Heavy computation for work-stealing analysis
    var sum: u64 = 0;
    for (0..10000) |i| {
        sum += i * i * value.*;
    }
    
    value.* = @intCast(sum % 1000);
    
    // COZ progress point for task execution
    zigpulse.coz.throughput(zigpulse.coz.Points.task_execution);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){}; 
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.log.info("COZ Profiling Benchmark - Beat.zig A3C Performance", .{});
    
    // Create A3C-enabled pool for profiling
    const config = zigpulse.Config{
        .num_workers = 4,
        .enable_a3c_scheduling = true,
        .a3c_learning_rate = 0.001,
        .a3c_confidence_threshold = 0.7,
        .enable_statistics = true,
    };
    
    const pool = try zigpulse.createPoolWithConfig(allocator, config);
    defer pool.deinit();
    
    std.log.info("Running workload for COZ profiling...", .{});
    
    // Test data
    var test_values: [100]u32 = undefined;
    for (&test_values, 0..) |*value, i| {
        value.* = @intCast(i + 1);
    }
    
    // Submit mixed workload for 10 iterations
    for (0..10) |iteration| {
        std.log.info("Iteration {}/10", .{iteration + 1});
        
        // Submit mix of light and heavy tasks
        for (&test_values, 0..) |*value, i| {
            const task = if (i % 4 == 0)
                zigpulse.Task{
                    .func = heavyTask,
                    .data = value,
                    .priority = .high,
                    .data_size_hint = 10000,
                }
            else
                zigpulse.Task{
                    .func = simpleTask,
                    .data = value,
                    .priority = .normal,
                    .data_size_hint = 1000,
                };
            
            try pool.submit(task);
        }
        
        // Wait for completion
        pool.wait();
        
        // Small delay between iterations
        std.time.sleep(1_000_000); // 1ms
    }
    
    std.log.info("Benchmark complete! Profile data generated.", .{});
    std.log.info("Total tasks: {}", .{100 * 10});
    std.log.info("Heavy tasks: {}", .{25 * 10});
    std.log.info("Light tasks: {}", .{75 * 10});
}