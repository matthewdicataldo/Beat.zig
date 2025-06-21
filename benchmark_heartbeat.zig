// Heartbeat Performance Benchmark
// Tests fixed vs adaptive heartbeat timing under various workload scenarios

const std = @import("std");
const core = @import("src/core.zig");

const BenchmarkResult = struct {
    scenario: []const u8,
    total_time_ms: u64,
    heartbeat_wake_ups: u64,
    promotions_triggered: u64,
    avg_response_time_us: f64,
    cpu_utilization_pct: f64,
};

const WorkloadScenario = enum {
    idle,           // No tasks - should have minimal wake-ups
    light_steady,   // Few tasks, steady rate
    heavy_steady,   // Many tasks, steady rate  
    burst,          // Sudden spike in tasks
    variable,       // Fluctuating workload
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Heartbeat Performance Benchmark ===\n\n", .{});
    
    const scenarios = [_]WorkloadScenario{ .idle, .light_steady, .heavy_steady, .burst, .variable };
    
    // Test current fixed heartbeat implementation
    std.debug.print("BASELINE: Fixed Heartbeat (100μs interval)\n", .{});
    std.debug.print("{s}\n", .{"-" ** 50});
    
    for (scenarios) |scenario| {
        const result = try benchmarkScenario(allocator, scenario, false);
        printBenchmarkResult(result);
    }
    
    std.debug.print("\nOPTIMIZED: Adaptive Heartbeat\n", .{});
    std.debug.print("{s}\n", .{"-" ** 50});
    
    for (scenarios) |scenario| {
        const result = try benchmarkScenario(allocator, scenario, true);
        printBenchmarkResult(result);
    }
}

fn benchmarkScenario(allocator: std.mem.Allocator, scenario: WorkloadScenario, adaptive: bool) !BenchmarkResult {
    // Configure thread pool based on adaptive flag
    var config = core.Config{
        .num_workers = 4,
        .enable_heartbeat = true,
        .heartbeat_interval_us = if (adaptive) 50 else 100, // Will be overridden by adaptive logic
        .enable_statistics = true,
    };
    
    var pool = try core.ThreadPool.init(allocator, config);
    defer pool.deinit();
    
    const start_time = std.time.milliTimestamp();
    
    // Simulate workload based on scenario
    switch (scenario) {
        .idle => {
            // No tasks for 1 second - measure heartbeat overhead
            std.time.sleep(1_000_000_000); // 1 second
        },
        .light_steady => {
            // Submit 10 tasks per second for 2 seconds
            for (0..20) |i| {
                try pool.submit(allocator, lightTask, @intFromPtr(&i));
                std.time.sleep(100_000_000); // 100ms between tasks
            }
        },
        .heavy_steady => {
            // Submit 100 tasks per second for 1 second
            for (0..100) |i| {
                try pool.submit(allocator, mediumTask, @intFromPtr(&i));
                std.time.sleep(10_000_000); // 10ms between tasks
            }
        },
        .burst => {
            // Idle for 500ms, then 50 tasks rapidly, then idle
            std.time.sleep(500_000_000);
            for (0..50) |i| {
                try pool.submit(allocator, heavyTask, @intFromPtr(&i));
                std.time.sleep(1_000_000); // 1ms between tasks (burst)
            }
            std.time.sleep(500_000_000);
        },
        .variable => {
            // Fluctuating pattern: light -> heavy -> light -> heavy
            for (0..4) |cycle| {
                const is_heavy = (cycle % 2 == 1);
                const task_count: usize = if (is_heavy) 25 else 5;
                const task_interval: u64 = if (is_heavy) 5_000_000 else 50_000_000; // 5ms vs 50ms
                
                for (0..task_count) |i| {
                    try pool.submit(allocator, if (is_heavy) heavyTask else lightTask, @intFromPtr(&i));
                    std.time.sleep(task_interval);
                }
            }
        },
    }
    
    // Wait for all tasks to complete
    pool.wait();
    
    const end_time = std.time.milliTimestamp();
    const total_time = @as(u64, @intCast(end_time - start_time));
    
    // Get scheduler statistics
    const scheduler = pool.scheduler;
    const promotions = scheduler.getPromotionCount();
    
    return BenchmarkResult{
        .scenario = @tagName(scenario),
        .total_time_ms = total_time,
        .heartbeat_wake_ups = estimateWakeUps(total_time, config.heartbeat_interval_us),
        .promotions_triggered = promotions,
        .avg_response_time_us = 0.0, // TODO: Implement response time tracking
        .cpu_utilization_pct = 0.0, // TODO: Implement CPU utilization tracking
    };
}

fn estimateWakeUps(total_time_ms: u64, interval_us: u32) u64 {
    const total_time_us = total_time_ms * 1000;
    return total_time_us / interval_us;
}

fn printBenchmarkResult(result: BenchmarkResult) void {
    std.debug.print("{s:12} | {d:4}ms | {d:4} wake-ups | {d:3} promotions\n", 
        .{ result.scenario, result.total_time_ms, result.heartbeat_wake_ups, result.promotions_triggered });
}

// Sample task functions for benchmarking
fn lightTask(_: *anyopaque) void {
    // Simulate light work (10μs)
    var sum: u64 = 0;
    for (0..100) |i| {
        sum +%= i;
    }
    std.mem.doNotOptimizeAway(&sum);
}

fn mediumTask(_: *anyopaque) void {
    // Simulate medium work (100μs)  
    var sum: u64 = 0;
    for (0..1000) |i| {
        sum +%= i * i;
    }
    std.mem.doNotOptimizeAway(&sum);
}

fn heavyTask(_: *anyopaque) void {
    // Simulate heavy work (1ms)
    var sum: u64 = 0;
    for (0..10000) |i| {
        sum +%= i * i * i;
    }
    std.mem.doNotOptimizeAway(&sum);
}