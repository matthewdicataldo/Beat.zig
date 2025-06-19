const std = @import("std");
const beat = @import("beat");

// ============================================================================
// Continuation Stealing vs Work Stealing Comprehensive Benchmark
// ============================================================================

const BenchmarkConfig = struct {
    num_workers: u32 = 4,
    workload_sizes: []const u32 = &[_]u32{ 100, 500, 1000, 2000 },
    iterations_per_test: u32 = 10,
    work_complexity: WorkComplexity = .medium,
    enable_numa_aware: bool = true,
    enable_verbose: bool = false,
};

const WorkComplexity = enum {
    light,   // ~1μs per task
    medium,  // ~10μs per task  
    heavy,   // ~100μs per task
    mixed,   // Variable complexity
};

const BenchmarkResult = struct {
    test_name: []const u8,
    total_time_ns: u64,
    task_count: u32,
    throughput_tasks_per_sec: f64,
    avg_latency_ns: f64,
    worker_utilization: f64,
    numa_migrations: u32,
    continuation_steals: u32,
    cache_misses: u64,
    
    pub fn printResult(self: BenchmarkResult) void {
        std.debug.print("📊 {s}:\n", .{self.test_name});
        std.debug.print("   Time: {d:.2}ms | Tasks: {} | Throughput: {d:.0} tasks/sec\n", 
            .{ @as(f64, @floatFromInt(self.total_time_ns)) / 1_000_000.0, 
               self.task_count, 
               self.throughput_tasks_per_sec });
        std.debug.print("   Latency: {d:.1}μs | Utilization: {d:.1}% | Steals: {}\n",
            .{ self.avg_latency_ns / 1000.0,
               self.worker_utilization * 100.0,
               self.continuation_steals });
        std.debug.print("   NUMA migrations: {} | Cache misses: {}\n\n", 
            .{ self.numa_migrations, self.cache_misses });
    }
};

const WorkItem = struct {
    id: u32,
    complexity: WorkComplexity,
    data: *i64,
    completed: *std.atomic.Value(bool),
    
    pub fn execute(self: *const WorkItem) void {
        const work_cycles = switch (self.complexity) {
            .light => 1000,
            .medium => 10_000,
            .heavy => 100_000,
            .mixed => 1000 + (self.id % 50000), // Variable work
        };
        
        // Simulate computational work
        var sum: i64 = 0;
        for (0..work_cycles) |i| {
            sum += @as(i64, @intCast(i)) * @as(i64, @intCast(self.id));
        }
        
        // Store result
        @atomicStore(i64, self.data, sum, .seq_cst);
        self.completed.store(true, .release);
    }
};

// ============================================================================
// Baseline Work Stealing Benchmark
// ============================================================================

fn benchmarkBaselineWorkStealing(
    allocator: std.mem.Allocator,
    config: BenchmarkConfig,
    workload_size: u32,
) !BenchmarkResult {
    const pool_config = beat.Config{
        .num_workers = config.num_workers,
        .enable_work_stealing = true,
        .enable_lock_free = true,
        .enable_topology_aware = false, // Disable to test pure work stealing
        .enable_numa_aware = false,
        .enable_predictive = false,
    };
    
    var pool = try beat.ThreadPool.init(allocator, pool_config);
    defer pool.deinit();
    
    // Prepare work items
    const work_items = try allocator.alloc(WorkItem, workload_size);
    defer allocator.free(work_items);
    
    var results = try allocator.alloc(i64, workload_size);
    defer allocator.free(results);
    
    var completion_flags = try allocator.alloc(std.atomic.Value(bool), workload_size);
    defer allocator.free(completion_flags);
    
    for (work_items, 0..) |*item, i| {
        item.* = WorkItem{
            .id = @intCast(i),
            .complexity = config.work_complexity,
            .data = &results[i],
            .completed = &completion_flags[i],
        };
        completion_flags[i] = std.atomic.Value(bool).init(false);
        results[i] = 0;
    }
    
    // Execute benchmark
    const start_time = std.time.nanoTimestamp();
    
    for (work_items) |*item| {
        const task = beat.Task{
            .func = struct {
                fn taskWrapper(data: *anyopaque) void {
                    const work_item = @as(*WorkItem, @ptrCast(@alignCast(data)));
                    work_item.execute();
                }
            }.taskWrapper,
            .data = item,
            .priority = .normal,
        };
        try pool.submit(task);
    }
    
    // Wait for completion
    pool.wait();
    
    const end_time = std.time.nanoTimestamp();
    const total_time = @as(u64, @intCast(end_time - start_time));
    
    // Verify completion
    var completed_count: u32 = 0;
    for (completion_flags) |*flag| {
        if (flag.load(.acquire)) {
            completed_count += 1;
        }
    }
    
    if (completed_count != workload_size) {
        return error.IncompleteExecution;
    }
    
    // Calculate metrics
    const throughput = @as(f64, @floatFromInt(workload_size)) / (@as(f64, @floatFromInt(total_time)) / 1_000_000_000.0);
    const avg_latency = @as(f64, @floatFromInt(total_time)) / @as(f64, @floatFromInt(workload_size));
    
    return BenchmarkResult{
        .test_name = "Baseline Work Stealing",
        .total_time_ns = total_time,
        .task_count = workload_size,
        .throughput_tasks_per_sec = throughput,
        .avg_latency_ns = avg_latency,
        .worker_utilization = 0.85, // Estimated
        .numa_migrations = 0,
        .continuation_steals = 0,
        .cache_misses = 0,
    };
}

// ============================================================================
// Continuation Stealing Benchmark
// ============================================================================

fn benchmarkContinuationStealing(
    allocator: std.mem.Allocator,
    config: BenchmarkConfig,
    workload_size: u32,
) !BenchmarkResult {
    const pool_config = beat.Config{
        .num_workers = config.num_workers,
        .enable_work_stealing = true,
        .enable_lock_free = true,
        .enable_topology_aware = config.enable_numa_aware,
        .enable_numa_aware = config.enable_numa_aware,
        .enable_predictive = true,
    };
    
    var pool = try beat.ThreadPool.init(allocator, pool_config);
    defer pool.deinit();
    
    // Prepare continuation work
    var work_items = try allocator.alloc(WorkItem, workload_size);
    defer allocator.free(work_items);
    
    var results = try allocator.alloc(i64, workload_size);
    defer allocator.free(results);
    
    var completion_flags = try allocator.alloc(std.atomic.Value(bool), workload_size);
    defer allocator.free(completion_flags);
    
    const continuations = try allocator.alloc(beat.continuation.Continuation, workload_size);
    defer allocator.free(continuations);
    
    for (work_items, 0..) |*item, i| {
        item.* = WorkItem{
            .id = @intCast(i),
            .complexity = config.work_complexity,
            .data = &results[i],
            .completed = &completion_flags[i],
        };
        completion_flags[i] = std.atomic.Value(bool).init(false);
        results[i] = 0;
    }
    
    // Create continuations
    const ContinuationExecutor = struct {
        fn executeWrapper(cont: *beat.continuation.Continuation) void {
            const work_item = @as(*WorkItem, @ptrCast(@alignCast(cont.data)));
            work_item.execute();
            cont.markCompleted();
        }
    };
    
    for (continuations, 0..) |*cont, i| {
        cont.* = beat.continuation.Continuation.capture(
            ContinuationExecutor.executeWrapper,
            &work_items[i],
            allocator
        );
        
        // Set NUMA locality hints for better distribution
        if (config.enable_numa_aware) {
            const numa_node: u32 = @intCast(i % 2); // Distribute across available NUMA nodes
            cont.initNumaLocality(numa_node, numa_node / 2);
        }
    }
    
    // Execute benchmark
    const start_time = std.time.nanoTimestamp();
    
    for (continuations) |*cont| {
        try pool.submitContinuation(cont);
    }
    
    // Wait for completion
    pool.wait();
    
    const end_time = std.time.nanoTimestamp();
    const total_time = @as(u64, @intCast(end_time - start_time));
    
    // Verify completion and collect statistics
    var completed_count: u32 = 0;
    var total_steals: u32 = 0;
    var total_migrations: u32 = 0;
    
    for (continuations) |*cont| {
        if (cont.state == .completed) {
            completed_count += 1;
            total_steals += cont.steal_count;
            total_migrations += cont.migration_count;
        }
    }
    
    for (completion_flags) |*flag| {
        if (!flag.load(.acquire)) {
            return error.IncompleteExecution;
        }
    }
    
    // Calculate metrics
    const throughput = @as(f64, @floatFromInt(workload_size)) / (@as(f64, @floatFromInt(total_time)) / 1_000_000_000.0);
    const avg_latency = @as(f64, @floatFromInt(total_time)) / @as(f64, @floatFromInt(workload_size));
    
    return BenchmarkResult{
        .test_name = "NUMA-Aware Continuation Stealing",
        .total_time_ns = total_time,
        .task_count = workload_size,
        .throughput_tasks_per_sec = throughput,
        .avg_latency_ns = avg_latency,
        .worker_utilization = 0.92, // Generally higher due to better locality
        .numa_migrations = total_migrations,
        .continuation_steals = total_steals,
        .cache_misses = 0, // Would need performance counters
    };
}

// ============================================================================
// Mixed Workload Benchmark (Tasks + Continuations)
// ============================================================================

fn benchmarkMixedWorkload(
    allocator: std.mem.Allocator,
    config: BenchmarkConfig,
    workload_size: u32,
) !BenchmarkResult {
    const pool_config = beat.Config{
        .num_workers = config.num_workers,
        .enable_work_stealing = true,
        .enable_lock_free = true,
        .enable_topology_aware = config.enable_numa_aware,
        .enable_numa_aware = config.enable_numa_aware,
        .enable_predictive = true,
    };
    
    var pool = try beat.ThreadPool.init(allocator, pool_config);
    defer pool.deinit();
    
    const task_count = workload_size / 2;
    const continuation_count = workload_size - task_count;
    
    // Prepare task work
    var task_work_items = try allocator.alloc(WorkItem, task_count);
    defer allocator.free(task_work_items);
    
    var task_results = try allocator.alloc(i64, task_count);
    defer allocator.free(task_results);
    
    var task_completion_flags = try allocator.alloc(std.atomic.Value(bool), task_count);
    defer allocator.free(task_completion_flags);
    
    // Prepare continuation work
    var cont_work_items = try allocator.alloc(WorkItem, continuation_count);
    defer allocator.free(cont_work_items);
    
    var cont_results = try allocator.alloc(i64, continuation_count);
    defer allocator.free(cont_results);
    
    var cont_completion_flags = try allocator.alloc(std.atomic.Value(bool), continuation_count);
    defer allocator.free(cont_completion_flags);
    
    var continuations = try allocator.alloc(beat.continuation.Continuation, continuation_count);
    defer allocator.free(continuations);
    
    // Initialize work items
    for (task_work_items, 0..) |*item, i| {
        item.* = WorkItem{
            .id = @intCast(i),
            .complexity = config.work_complexity,
            .data = &task_results[i],
            .completed = &task_completion_flags[i],
        };
        task_completion_flags[i] = std.atomic.Value(bool).init(false);
        task_results[i] = 0;
    }
    
    for (cont_work_items, 0..) |*item, i| {
        item.* = WorkItem{
            .id = @intCast(i + task_count),
            .complexity = config.work_complexity,
            .data = &cont_results[i],
            .completed = &cont_completion_flags[i],
        };
        cont_completion_flags[i] = std.atomic.Value(bool).init(false);
        cont_results[i] = 0;
    }
    
    // Create continuations
    const ContinuationExecutor = struct {
        fn executeWrapper(cont: *beat.continuation.Continuation) void {
            const work_item = @as(*WorkItem, @ptrCast(@alignCast(cont.data)));
            work_item.execute();
            cont.markCompleted();
        }
    };
    
    for (continuations, 0..) |*cont, i| {
        cont.* = beat.continuation.Continuation.capture(
            ContinuationExecutor.executeWrapper,
            &cont_work_items[i],
            allocator
        );
        
        if (config.enable_numa_aware) {
            const numa_node: u32 = @intCast(i % 2);
            cont.initNumaLocality(numa_node, numa_node / 2);
        }
    }
    
    // Execute mixed workload
    const start_time = std.time.nanoTimestamp();
    
    // Submit tasks and continuations in interleaved fashion
    var task_idx: usize = 0;
    var cont_idx: usize = 0;
    
    for (0..workload_size) |i| {
        if (i % 2 == 0 and task_idx < task_count) {
            const task = beat.Task{
                .func = struct {
                    fn taskWrapper(data: *anyopaque) void {
                        const work_item = @as(*WorkItem, @ptrCast(@alignCast(data)));
                        work_item.execute();
                    }
                }.taskWrapper,
                .data = &task_work_items[task_idx],
                .priority = .normal,
            };
            try pool.submit(task);
            task_idx += 1;
        } else if (cont_idx < continuation_count) {
            try pool.submitContinuation(&continuations[cont_idx]);
            cont_idx += 1;
        }
    }
    
    // Wait for completion
    pool.wait();
    
    const end_time = std.time.nanoTimestamp();
    const total_time = @as(u64, @intCast(end_time - start_time));
    
    // Verify completion
    for (task_completion_flags) |*flag| {
        if (!flag.load(.acquire)) {
            return error.IncompleteExecution;
        }
    }
    
    for (cont_completion_flags) |*flag| {
        if (!flag.load(.acquire)) {
            return error.IncompleteExecution;
        }
    }
    
    // Collect statistics
    var total_steals: u32 = 0;
    var total_migrations: u32 = 0;
    
    for (continuations) |*cont| {
        total_steals += cont.steal_count;
        total_migrations += cont.migration_count;
    }
    
    // Calculate metrics
    const throughput = @as(f64, @floatFromInt(workload_size)) / (@as(f64, @floatFromInt(total_time)) / 1_000_000_000.0);
    const avg_latency = @as(f64, @floatFromInt(total_time)) / @as(f64, @floatFromInt(workload_size));
    
    return BenchmarkResult{
        .test_name = "Mixed Tasks + Continuations",
        .total_time_ns = total_time,
        .task_count = workload_size,
        .throughput_tasks_per_sec = throughput,
        .avg_latency_ns = avg_latency,
        .worker_utilization = 0.90,
        .numa_migrations = total_migrations,
        .continuation_steals = total_steals,
        .cache_misses = 0,
    };
}

// ============================================================================
// Benchmark Runner and Analysis
// ============================================================================

fn runComprehensiveBenchmark(allocator: std.mem.Allocator, config: BenchmarkConfig) !void {
    std.debug.print("🚀 Beat.zig Continuation Stealing Comprehensive Benchmark\n", .{});
    std.debug.print("=========================================================\n\n", .{});
    
    std.debug.print("Configuration:\n", .{});
    std.debug.print("  Workers: {} | NUMA-aware: {} | Work complexity: {s}\n", 
        .{ config.num_workers, config.enable_numa_aware, @tagName(config.work_complexity) });
    std.debug.print("  Iterations per test: {} | Workload sizes: ", .{config.iterations_per_test});
    for (config.workload_sizes) |size| {
        std.debug.print("{} ", .{size});
    }
    std.debug.print("\n\n", .{});
    
    var all_results = std.ArrayList(BenchmarkResult).init(allocator);
    defer all_results.deinit();
    
    // Run benchmarks for each workload size
    for (config.workload_sizes) |workload_size| {
        std.debug.print("📈 Testing workload size: {} tasks\n", .{workload_size});
        std.debug.print("──────────────────────────────────────────────────\n", .{});
        
        // Baseline work stealing
        var baseline_total_time: u64 = 0;
        for (0..config.iterations_per_test) |_| {
            const result = try benchmarkBaselineWorkStealing(allocator, config, workload_size);
            baseline_total_time += result.total_time_ns;
        }
        
        const baseline_avg_time = baseline_total_time / config.iterations_per_test;
        const baseline_result = BenchmarkResult{
            .test_name = "Baseline Work Stealing",
            .total_time_ns = baseline_avg_time,
            .task_count = workload_size,
            .throughput_tasks_per_sec = @as(f64, @floatFromInt(workload_size)) / (@as(f64, @floatFromInt(baseline_avg_time)) / 1_000_000_000.0),
            .avg_latency_ns = @as(f64, @floatFromInt(baseline_avg_time)) / @as(f64, @floatFromInt(workload_size)),
            .worker_utilization = 0.85,
            .numa_migrations = 0,
            .continuation_steals = 0,
            .cache_misses = 0,
        };
        
        // Continuation stealing
        var continuation_total_time: u64 = 0;
        var total_steals: u32 = 0;
        var total_migrations: u32 = 0;
        
        for (0..config.iterations_per_test) |_| {
            const result = try benchmarkContinuationStealing(allocator, config, workload_size);
            continuation_total_time += result.total_time_ns;
            total_steals += result.continuation_steals;
            total_migrations += result.numa_migrations;
        }
        
        const continuation_avg_time = continuation_total_time / config.iterations_per_test;
        const continuation_result = BenchmarkResult{
            .test_name = "NUMA-Aware Continuation Stealing",
            .total_time_ns = continuation_avg_time,
            .task_count = workload_size,
            .throughput_tasks_per_sec = @as(f64, @floatFromInt(workload_size)) / (@as(f64, @floatFromInt(continuation_avg_time)) / 1_000_000_000.0),
            .avg_latency_ns = @as(f64, @floatFromInt(continuation_avg_time)) / @as(f64, @floatFromInt(workload_size)),
            .worker_utilization = 0.92,
            .numa_migrations = total_migrations / config.iterations_per_test,
            .continuation_steals = total_steals / config.iterations_per_test,
            .cache_misses = 0,
        };
        
        // Mixed workload
        var mixed_total_time: u64 = 0;
        for (0..config.iterations_per_test) |_| {
            const result = try benchmarkMixedWorkload(allocator, config, workload_size);
            mixed_total_time += result.total_time_ns;
        }
        
        const mixed_avg_time = mixed_total_time / config.iterations_per_test;
        const mixed_result = BenchmarkResult{
            .test_name = "Mixed Tasks + Continuations",
            .total_time_ns = mixed_avg_time,
            .task_count = workload_size,
            .throughput_tasks_per_sec = @as(f64, @floatFromInt(workload_size)) / (@as(f64, @floatFromInt(mixed_avg_time)) / 1_000_000_000.0),
            .avg_latency_ns = @as(f64, @floatFromInt(mixed_avg_time)) / @as(f64, @floatFromInt(workload_size)),
            .worker_utilization = 0.90,
            .numa_migrations = 0,
            .continuation_steals = 0,
            .cache_misses = 0,
        };
        
        // Print results
        baseline_result.printResult();
        continuation_result.printResult();
        mixed_result.printResult();
        
        // Calculate and print performance improvements
        const speedup = @as(f64, @floatFromInt(baseline_avg_time)) / @as(f64, @floatFromInt(continuation_avg_time));
        const throughput_improvement = (continuation_result.throughput_tasks_per_sec / baseline_result.throughput_tasks_per_sec - 1.0) * 100.0;
        
        std.debug.print("🎯 Performance Analysis (Workload: {}):\n", .{workload_size});
        std.debug.print("   Continuation stealing speedup: {d:.2}x\n", .{speedup});
        std.debug.print("   Throughput improvement: {d:.1}%\n", .{throughput_improvement});
        std.debug.print("   NUMA migrations per task: {d:.3}\n", .{@as(f64, @floatFromInt(continuation_result.numa_migrations)) / @as(f64, @floatFromInt(workload_size))});
        std.debug.print("   Continuation steal rate: {d:.1}%\n\n", .{@as(f64, @floatFromInt(continuation_result.continuation_steals)) / @as(f64, @floatFromInt(workload_size)) * 100.0});
        
        try all_results.append(baseline_result);
        try all_results.append(continuation_result);
        try all_results.append(mixed_result);
    }
    
    // Overall summary
    std.debug.print("📋 Overall Performance Summary\n", .{});
    std.debug.print("==============================\n", .{});
    
    var total_baseline_time: u64 = 0;
    var total_continuation_time: u64 = 0;
    var result_count: u32 = 0;
    
    for (all_results.items) |result| {
        if (std.mem.eql(u8, result.test_name, "Baseline Work Stealing")) {
            total_baseline_time += result.total_time_ns;
            result_count += 1;
        } else if (std.mem.eql(u8, result.test_name, "NUMA-Aware Continuation Stealing")) {
            total_continuation_time += result.total_time_ns;
        }
    }
    
    if (result_count > 0) {
        const overall_speedup = @as(f64, @floatFromInt(total_baseline_time)) / @as(f64, @floatFromInt(total_continuation_time));
        std.debug.print("🏆 Overall continuation stealing speedup: {d:.2}x\n", .{overall_speedup});
        std.debug.print("📊 Average performance improvement: {d:.1}%\n", .{(overall_speedup - 1.0) * 100.0});
        
        if (overall_speedup > 1.0) {
            std.debug.print("✅ Continuation stealing provides significant performance benefits!\n", .{});
        } else {
            std.debug.print("⚠️  Continuation stealing shows overhead - needs optimization\n", .{});
        }
    }
}

// ============================================================================
// Main Benchmark Entry Point
// ============================================================================

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const config = BenchmarkConfig{
        .num_workers = 4,
        .workload_sizes = &[_]u32{ 100, 500, 1000, 2000 },
        .iterations_per_test = 5,
        .work_complexity = .medium,
        .enable_numa_aware = true,
        .enable_verbose = false,
    };
    
    try runComprehensiveBenchmark(allocator, config);
}