const std = @import("std");
const zigpulse = @import("zigpulse");

// Benchmark specifically designed for COZ profiling
// Tests both throughput (tasks/second) and latency scenarios

const WorkloadType = enum {
    cpu_bound,
    memory_bound,
    mixed,
};

// Simulated work functions
fn cpuBoundWork(data: *anyopaque) void {
    const iterations = @as(*const u32, @ptrCast(@alignCast(data))).*;
    var sum: u64 = 0;
    for (0..iterations) |i| {
        sum +%= i * i + (i << 3) - (i >> 2);
    }
    // Prevent optimization
    std.mem.doNotOptimizeAway(&sum);
}

fn memoryBoundWork(data: *anyopaque) void {
    const size = @as(*const usize, @ptrCast(@alignCast(data))).*;
    var buffer = std.heap.page_allocator.alloc(u8, size) catch return;
    defer std.heap.page_allocator.free(buffer);
    
    // Random memory access pattern
    var rng = std.Random.DefaultPrng.init(@as(u64, @intCast(std.time.milliTimestamp())));
    const random = rng.random();
    
    for (0..size / 64) |_| {
        const idx = random.uintLessThan(usize, size - 8);
        buffer[idx] = @truncate(idx);
    }
}

fn mixedWork(data: *anyopaque) void {
    _ = data;
    // Some CPU work
    var sum: u64 = 0;
    for (0..1000) |i| {
        sum +%= i * i;
    }
    
    // Some memory work
    var buffer: [1024]u8 = undefined;
    for (&buffer, 0..) |*byte, i| {
        byte.* = @truncate(i + sum);
    }
    
    std.mem.doNotOptimizeAway(&buffer);
}

// Benchmark scenarios
const BenchmarkScenario = struct {
    name: []const u8,
    workload: WorkloadType,
    task_count: usize,
    task_size: usize,
    expected_tasks_per_second: f64,
};

const scenarios = [_]BenchmarkScenario{
    .{
        .name = "high_throughput_small",
        .workload = .cpu_bound,
        .task_count = 10000,
        .task_size = 100,
        .expected_tasks_per_second = 50000,
    },
    .{
        .name = "low_latency_large",
        .workload = .cpu_bound,
        .task_count = 100,
        .task_size = 10000,
        .expected_tasks_per_second = 1000,
    },
    .{
        .name = "memory_intensive",
        .workload = .memory_bound,
        .task_count = 1000,
        .task_size = 65536,
        .expected_tasks_per_second = 500,
    },
    .{
        .name = "mixed_workload",
        .workload = .mixed,
        .task_count = 5000,
        .task_size = 0,
        .expected_tasks_per_second = 10000,
    },
};

pub fn main() !void {
    std.debug.print("=== ZigPulse COZ Profiling Benchmark ===\n", .{});
    std.debug.print("Build with: zig build-exe benchmark_coz.zig -O ReleaseSafe -fno-omit-frame-pointer\n", .{});
    std.debug.print("Run with: coz run --- ./benchmark_coz\n\n", .{});
    
    // Use page allocator to avoid potential GPA issues in ReleaseSafe
    const allocator = std.heap.page_allocator;
    
    // Detect optimal worker count
    const cpu_count = try std.Thread.getCpuCount();
    const optimal_workers = @min(cpu_count / 2, 8); // Use half of CPUs, max 8
    
    std.debug.print("Detected {} CPUs, using {} workers\n", .{cpu_count, optimal_workers});
    
    // Create thread pool with optimized worker count
    const config = zigpulse.Config{
        .num_workers = optimal_workers,
        .task_queue_size = 16384,  // Increase queue size for high-throughput scenarios
        .enable_statistics = true,
        .enable_trace = false,
        .enable_topology_aware = false, // Disable to avoid potential issues
        .enable_lock_free = false,  // Use mutex queue which doesn't have size limits
    };
    
    const pool = try zigpulse.createPoolWithConfig(allocator, config);
    defer pool.deinit();
    
    std.debug.print("Thread pool created with {} workers\n\n", .{pool.workers.len});
    
    // Run each scenario
    for (scenarios) |scenario| {
        std.debug.print("=== Scenario: {s} ===\n", .{scenario.name});
        std.debug.print("  Workload: {}\n", .{scenario.workload});
        std.debug.print("  Tasks: {}\n", .{scenario.task_count});
        std.debug.print("  Task size: {}\n", .{scenario.task_size});
        std.debug.print("  Target: {d:.0} tasks/sec\n", .{scenario.expected_tasks_per_second});
        
        const start_time = std.time.milliTimestamp();
        
        // Allocate task data
        const task_data = try allocator.alloc(usize, scenario.task_count);
        defer allocator.free(task_data);
        
        for (task_data) |*data| {
            data.* = scenario.task_size;
        }
        
        // Submit all tasks
        for (task_data) |*data| {
            const task = zigpulse.Task{
                .func = switch (scenario.workload) {
                    .cpu_bound => cpuBoundWork,
                    .memory_bound => memoryBoundWork,
                    .mixed => mixedWork,
                },
                .data = @ptrCast(data),
                .priority = .normal,
            };
            
            try pool.submit(task);
        }
        
        // Wait for completion
        pool.wait();
        
        const end_time = std.time.milliTimestamp();
        const duration_ms = @as(f64, @floatFromInt(end_time - start_time));
        const tasks_per_second = @as(f64, @floatFromInt(scenario.task_count)) * 1000.0 / duration_ms;
        
        std.debug.print("  Duration: {d:.2}ms\n", .{duration_ms});
        std.debug.print("  Achieved: {d:.0} tasks/sec\n", .{tasks_per_second});
        std.debug.print("  Efficiency: {d:.1}%\n", .{tasks_per_second / scenario.expected_tasks_per_second * 100.0});
        
        // Print statistics
        const stats = &pool.stats;
        std.debug.print("  Stats:\n", .{});
        std.debug.print("    Submitted: {}\n", .{stats.tasks_submitted.load(.acquire)});
        std.debug.print("    Completed: {}\n", .{stats.tasks_completed.load(.acquire)});
        std.debug.print("    Stolen: {}\n", .{stats.tasks_stolen.load(.acquire)});
        
        const steal_rate = if (stats.tasks_completed.load(.acquire) > 0)
            @as(f64, @floatFromInt(stats.tasks_stolen.load(.acquire))) * 100.0 / 
            @as(f64, @floatFromInt(stats.tasks_completed.load(.acquire)))
        else 
            0.0;
        
        std.debug.print("    Steal rate: {d:.1}%\n\n", .{steal_rate});
        
        // Small delay between scenarios
        std.time.sleep(100_000_000); // 100ms
    }
    
    std.debug.print("=== COZ Profiling Notes ===\n", .{});
    std.debug.print("Key progress points to analyze:\n", .{});
    std.debug.print("  - zigpulse_task_completed: Overall throughput\n", .{});
    std.debug.print("  - zigpulse_task_execution: Task latency\n", .{});
    std.debug.print("  - zigpulse_task_stolen: Work-stealing efficiency\n", .{});
    std.debug.print("  - zigpulse_worker_idle: Load balancing\n", .{});
    std.debug.print("\nLook for virtual speedup opportunities in:\n", .{});
    std.debug.print("  1. Task submission path\n", .{});
    std.debug.print("  2. Work-stealing algorithm\n", .{});
    std.debug.print("  3. Memory allocation\n", .{});
    std.debug.print("  4. Queue operations\n", .{});
}

// Test harness for continuous load (useful for COZ)
pub fn continuousLoad() !void {
    // Use page allocator to avoid potential GPA issues in ReleaseSafe
    const allocator = std.heap.page_allocator;
    
    const pool = try zigpulse.createPool(allocator);
    defer pool.deinit();
    
    // Run for 60 seconds with continuous task submission
    const start = std.time.milliTimestamp();
    const duration_ms = 60_000;
    
    var task_count: u64 = 0;
    var task_size: u32 = 1000;
    
    while (std.time.milliTimestamp() - start < duration_ms) {
        const task = zigpulse.Task{
            .func = cpuBoundWork,
            .data = @ptrCast(&task_size),
        };
        
        try pool.submit(task);
        task_count += 1;
        
        // Vary the workload
        if (task_count % 1000 == 0) {
            task_size = @truncate((task_size * 7 + 13) % 10000 + 100);
        }
    }
    
    pool.wait();
    
    const actual_duration = @as(f64, @floatFromInt(std.time.milliTimestamp() - start));
    const tasks_per_second = @as(f64, @floatFromInt(task_count)) * 1000.0 / actual_duration;
    
    std.debug.print("Continuous load test:\n", .{});
    std.debug.print("  Tasks: {}\n", .{task_count});
    std.debug.print("  Rate: {d:.0} tasks/sec\n", .{tasks_per_second});
}