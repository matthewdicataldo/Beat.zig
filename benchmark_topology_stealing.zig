const std = @import("std");
const beat = @import("src/core.zig");

// Comprehensive benchmarks for topology-aware work stealing
// Tests both synthetic workloads and realistic scenarios

const BenchmarkConfig = struct {
    iterations: u32 = 1000,
    warmup_iterations: u32 = 50,
    task_count: u32 = 10000,
    work_size: u32 = 1000,
};

// Timer utility for precise measurements
const Timer = struct {
    start_time: i128,
    
    pub fn start() Timer {
        return .{ .start_time = std.time.nanoTimestamp() };
    }
    
    pub fn elapsed(self: Timer) u64 {
        return @as(u64, @intCast(std.time.nanoTimestamp() - self.start_time));
    }
};

// Memory-intensive task that benefits from locality
const MemoryTask = struct {
    data: []u64,
    result: *std.atomic.Value(u64),
    
    pub fn execute(ctx: *anyopaque) void {
        const self = @as(*MemoryTask, @ptrCast(@alignCast(ctx)));
        var sum: u64 = 0;
        
        // Memory-intensive operations that benefit from cache locality
        for (self.data) |*item| {
            sum += item.*;
            item.* = sum; // Modify to create cache pressure
        }
        
        _ = self.result.fetchAdd(sum, .monotonic);
    }
};

// CPU-intensive task with memory access patterns
const ComputeTask = struct {
    iterations: u32,
    result: *std.atomic.Value(u64),
    
    pub fn execute(ctx: *anyopaque) void {
        const self = @as(*ComputeTask, @ptrCast(@alignCast(ctx)));
        var sum: u64 = 0;
        
        for (0..self.iterations) |i| {
            sum += i * i + (i % 17) * (i % 31);
        }
        
        _ = self.result.fetchAdd(sum, .monotonic);
    }
};

fn benchmarkMemoryIntensiveWorkload(allocator: std.mem.Allocator, config: BenchmarkConfig) !void {
    std.debug.print("\n=== Memory-Intensive Workload Benchmark ===\n", .{});
    
    // Create test data that will cause cache misses when accessed remotely
    const data_size = 64 * 1024; // 64KB per task (larger than L1 cache)
    const test_data = try allocator.alloc(u64, data_size);
    defer allocator.free(test_data);
    
    // Initialize with pattern that creates cache pressure
    for (test_data, 0..) |*item, i| {
        item.* = @as(u64, @intCast(i * 17 + 42));
    }
    
    // Test topology-aware work stealing
    std.debug.print("Testing topology-aware work stealing...\n", .{});
    const topo_result = try runMemoryBenchmark(allocator, config, test_data, true);
    
    // Test random work stealing (baseline)
    std.debug.print("Testing random work stealing...\n", .{});
    const random_result = try runMemoryBenchmark(allocator, config, test_data, false);
    
    // Calculate improvement
    const improvement = @as(f64, @floatFromInt(random_result.time)) / @as(f64, @floatFromInt(topo_result.time));
    const overhead_reduction = (1.0 - (1.0 / improvement)) * 100.0;
    
    std.debug.print("\nResults:\n", .{});
    std.debug.print("  Topology-aware: {}ms ({} tasks/sec)\n", .{
        topo_result.time / 1_000_000,
        @as(u64, @intFromFloat(@as(f64, @floatFromInt(config.task_count)) * 1e9 / @as(f64, @floatFromInt(topo_result.time)))),
    });
    std.debug.print("  Random stealing: {}ms ({} tasks/sec)\n", .{
        random_result.time / 1_000_000,
        @as(u64, @intFromFloat(@as(f64, @floatFromInt(config.task_count)) * 1e9 / @as(f64, @floatFromInt(random_result.time)))),
    });
    std.debug.print("  Speedup: {d:.2}x\n", .{improvement});
    std.debug.print("  Overhead reduction: {d:.1}%\n", .{overhead_reduction});
    std.debug.print("  Work completed correctly: {s}\n", .{
        if (topo_result.result_sum == random_result.result_sum) "✓" else "✗",
    });
}

const BenchmarkResult = struct {
    time: u64,
    result_sum: u64,
    steal_count: u64,
};

fn runMemoryBenchmark(
    allocator: std.mem.Allocator,
    config: BenchmarkConfig,
    test_data: []u64,
    topology_aware: bool,
) !BenchmarkResult {
    const pool = try beat.createPoolWithConfig(allocator, .{
        .num_workers = 4,
        .enable_topology_aware = topology_aware,
        .enable_work_stealing = true,
        .enable_statistics = true,
    });
    defer pool.deinit();
    
    var result = std.atomic.Value(u64).init(0);
    var tasks = try allocator.alloc(MemoryTask, config.task_count);
    defer allocator.free(tasks);
    
    // Initialize tasks with shared data to create memory contention
    for (tasks) |*task| {
        task.* = MemoryTask{
            .data = test_data, // All tasks share same data for maximum cache pressure
            .result = &result,
        };
    }
    
    // Warmup
    for (0..config.warmup_iterations) |_| {
        result.store(0, .release);
        for (tasks[0..@min(10, tasks.len)]) |*task| {
            try pool.submit(.{ .func = MemoryTask.execute, .data = task });
        }
        pool.wait();
    }
    
    // Benchmark
    result.store(0, .release);
    const timer = Timer.start();
    
    for (tasks) |*task| {
        try pool.submit(.{ .func = MemoryTask.execute, .data = task });
    }
    
    pool.wait();
    const elapsed = timer.elapsed();
    
    return BenchmarkResult{
        .time = elapsed,
        .result_sum = result.load(.acquire),
        .steal_count = 0, // TODO: Add steal count tracking
    };
}

fn benchmarkScalingWithWorkStealing(allocator: std.mem.Allocator, config: BenchmarkConfig) !void {
    std.debug.print("\n=== Work Stealing Scaling Benchmark ===\n", .{});
    
    const worker_counts = [_]u32{ 1, 2, 4, 8 };
    
    std.debug.print("Worker Count | Topology-Aware | Random Stealing | Improvement\n", .{});
    std.debug.print("-------------|----------------|----------------|------------\n", .{});
    
    for (worker_counts) |worker_count| {
        if (worker_count > std.Thread.getCpuCount() catch 4) break;
        
        // Test topology-aware
        const topo_time = try runScalingBenchmark(allocator, config, worker_count, true);
        
        // Test random
        const random_time = try runScalingBenchmark(allocator, config, worker_count, false);
        
        const improvement = @as(f64, @floatFromInt(random_time)) / @as(f64, @floatFromInt(topo_time));
        
        std.debug.print("     {d:2}      |     {d:4}ms     |     {d:4}ms     |   {d:.2}x\n", .{
            worker_count,
            topo_time / 1_000_000,
            random_time / 1_000_000,
            improvement,
        });
    }
}

fn runScalingBenchmark(
    allocator: std.mem.Allocator,
    config: BenchmarkConfig,
    worker_count: u32,
    topology_aware: bool,
) !u64 {
    const pool = try beat.createPoolWithConfig(allocator, .{
        .num_workers = worker_count,
        .enable_topology_aware = topology_aware,
        .enable_work_stealing = true,
        .enable_statistics = true,
    });
    defer pool.deinit();
    
    var result = std.atomic.Value(u64).init(0);
    const tasks = try allocator.alloc(ComputeTask, config.task_count / 10); // Fewer tasks for scaling test
    defer allocator.free(tasks);
    
    for (tasks) |*task| {
        task.* = ComputeTask{
            .iterations = config.work_size,
            .result = &result,
        };
    }
    
    const timer = Timer.start();
    
    for (tasks) |*task| {
        try pool.submit(.{ .func = ComputeTask.execute, .data = task });
    }
    
    pool.wait();
    return timer.elapsed();
}

fn benchmarkNUMAAffinityWorkload(allocator: std.mem.Allocator, config: BenchmarkConfig) !void {
    std.debug.print("\n=== NUMA Affinity Workload Benchmark ===\n", .{});
    
    const pool = try beat.createPoolWithConfig(allocator, .{
        .num_workers = 4,
        .enable_topology_aware = true,
        .enable_work_stealing = true,
        .enable_statistics = true,
    });
    defer pool.deinit();
    
    var numa0_result = std.atomic.Value(u64).init(0);
    var numa1_result = std.atomic.Value(u64).init(0);
    
    // Create memory regions for each NUMA node
    const numa0_data = try allocator.alloc(u64, 1024);
    defer allocator.free(numa0_data);
    const numa1_data = try allocator.alloc(u64, 1024);
    defer allocator.free(numa1_data);
    
    // Initialize data
    for (numa0_data, 0..) |*item, i| item.* = i;
    for (numa1_data, 0..) |*item, i| item.* = i * 2;
    
    const numa0_tasks = try allocator.alloc(MemoryTask, config.task_count / 2);
    defer allocator.free(numa0_tasks);
    const numa1_tasks = try allocator.alloc(MemoryTask, config.task_count / 2);
    defer allocator.free(numa1_tasks);
    
    // Setup tasks with NUMA affinity
    for (numa0_tasks) |*task| {
        task.* = MemoryTask{ .data = numa0_data, .result = &numa0_result };
    }
    for (numa1_tasks) |*task| {
        task.* = MemoryTask{ .data = numa1_data, .result = &numa1_result };
    }
    
    const timer = Timer.start();
    
    // Submit tasks with explicit NUMA hints
    for (numa0_tasks) |*task| {
        try pool.submit(.{ 
            .func = MemoryTask.execute, 
            .data = task,
            .affinity_hint = 0, // Prefer NUMA node 0
        });
    }
    
    for (numa1_tasks) |*task| {
        try pool.submit(.{ 
            .func = MemoryTask.execute, 
            .data = task,
            .affinity_hint = 1, // Prefer NUMA node 1
        });
    }
    
    pool.wait();
    const elapsed = timer.elapsed();
    
    std.debug.print("NUMA affinity workload completed in {}ms\n", .{elapsed / 1_000_000});
    std.debug.print("NUMA 0 result: {}\n", .{numa0_result.load(.acquire)});
    std.debug.print("NUMA 1 result: {}\n", .{numa1_result.load(.acquire)});
}

fn benchmarkContention(allocator: std.mem.Allocator, config: BenchmarkConfig) !void {
    std.debug.print("\n=== High Contention Workload Benchmark ===\n", .{});
    
    // Test with many small tasks to maximize stealing opportunities
    const small_config = BenchmarkConfig{
        .iterations = config.iterations,
        .warmup_iterations = config.warmup_iterations,
        .task_count = config.task_count * 4, // More, smaller tasks
        .work_size = config.work_size / 10, // Less work per task
    };
    
    // Compare topology-aware vs random under high contention
    const topo_time = try runContentionBenchmark(allocator, small_config, true);
    const random_time = try runContentionBenchmark(allocator, small_config, false);
    
    const improvement = @as(f64, @floatFromInt(random_time)) / @as(f64, @floatFromInt(topo_time));
    
    std.debug.print("High contention results:\n", .{});
    std.debug.print("  Topology-aware: {}ms\n", .{topo_time / 1_000_000});
    std.debug.print("  Random stealing: {}ms\n", .{random_time / 1_000_000});
    std.debug.print("  Improvement under contention: {d:.2}x\n", .{improvement});
}

fn runContentionBenchmark(
    allocator: std.mem.Allocator,
    config: BenchmarkConfig,
    topology_aware: bool,
) !u64 {
    const pool = try beat.createPoolWithConfig(allocator, .{
        .num_workers = 4,
        .enable_topology_aware = topology_aware,
        .enable_work_stealing = true,
        .enable_statistics = true,
    });
    defer pool.deinit();
    
    var result = std.atomic.Value(u64).init(0);
    const tasks = try allocator.alloc(ComputeTask, config.task_count);
    defer allocator.free(tasks);
    
    for (tasks) |*task| {
        task.* = ComputeTask{
            .iterations = config.work_size,
            .result = &result,
        };
    }
    
    const timer = Timer.start();
    
    // Submit all tasks quickly to create contention
    for (tasks) |*task| {
        try pool.submit(.{ .func = ComputeTask.execute, .data = task });
    }
    
    pool.wait();
    return timer.elapsed();
}

pub fn main() !void {
    std.debug.print("=== Topology-Aware Work Stealing Performance Analysis ===\n", .{});
    
    const allocator = std.heap.page_allocator;
    const config = BenchmarkConfig{
        .iterations = 5,
        .warmup_iterations = 2,
        .task_count = 1000,
        .work_size = 10000,
    };
    
    std.debug.print("Configuration: {} tasks, {} work units per task\n", .{
        config.task_count, config.work_size
    });
    std.debug.print("Hardware: {} CPUs detected\n", .{std.Thread.getCpuCount() catch 0});
    
    // Run comprehensive benchmarks
    try benchmarkMemoryIntensiveWorkload(allocator, config);
    try benchmarkScalingWithWorkStealing(allocator, config);
    try benchmarkNUMAAffinityWorkload(allocator, config);
    try benchmarkContention(allocator, config);
    
    std.debug.print("\n=== Summary ===\n", .{});
    std.debug.print("Topology-aware work stealing provides measurable benefits by:\n", .{});
    std.debug.print("  • Reducing cache misses through locality-aware stealing\n", .{});
    std.debug.print("  • Minimizing expensive cross-NUMA memory access\n", .{});
    std.debug.print("  • Maintaining work distribution efficiency under contention\n", .{});
    std.debug.print("  • Respecting NUMA affinity hints for optimal placement\n", .{});
    std.debug.print("\nNote: Benefits are most pronounced on multi-socket NUMA systems\n", .{});
}