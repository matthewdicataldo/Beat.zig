const std = @import("std");
const beat = @import("src/core.zig");

// Cache Performance Measurement Framework (Task: Optimization 2)
//
// This framework measures the performance impact of prediction lookup optimizations
// identified from COZ profiling analysis.

pub const OptimizationValidationConfig = struct {
    test_duration_seconds: u32 = 10,
    task_submission_rate: u32 = 1000,
    num_workers: usize = 4,
    num_test_iterations: usize = 5,
    enable_detailed_metrics: bool = true,
};

pub const PerformanceMetrics = struct {
    throughput_tasks_per_second: f64,
    average_latency_ns: f64,
    scheduling_overhead_ns: f64,
    prediction_lookup_time_ns: f64,
    worker_selection_time_ns: f64,
    total_execution_time_ns: u64,
    
    // Cache-specific metrics (when available)
    cache_hit_rate: f64 = 0.0,
    cache_lookup_time_ns: f64 = 0.0,
    
    pub fn calculateImprovement(optimized: PerformanceMetrics, baseline: PerformanceMetrics) ImprovementAnalysis {
        return ImprovementAnalysis{
            .throughput_improvement_percent = ((optimized.throughput_tasks_per_second - baseline.throughput_tasks_per_second) / baseline.throughput_tasks_per_second) * 100.0,
            .latency_reduction_percent = ((baseline.average_latency_ns - optimized.average_latency_ns) / baseline.average_latency_ns) * 100.0,
            .overhead_reduction_percent = ((baseline.scheduling_overhead_ns - optimized.scheduling_overhead_ns) / baseline.scheduling_overhead_ns) * 100.0,
            .prediction_speedup_factor = baseline.prediction_lookup_time_ns / optimized.prediction_lookup_time_ns,
        };
    }
};

pub const ImprovementAnalysis = struct {
    throughput_improvement_percent: f64,
    latency_reduction_percent: f64,
    overhead_reduction_percent: f64,
    prediction_speedup_factor: f64,
    
    pub fn isSignificantImprovement(self: ImprovementAnalysis) bool {
        return self.throughput_improvement_percent > 5.0 or
               self.latency_reduction_percent > 10.0 or
               self.overhead_reduction_percent > 15.0;
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){}; 
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const config = OptimizationValidationConfig{};
    
    std.debug.print("=== Optimization Validation Framework ===\n", .{});
    std.debug.print("Configuration:\n", .{});
    std.debug.print("  Test duration: {}s\n", .{config.test_duration_seconds});
    std.debug.print("  Target rate: {} tasks/s\n", .{config.task_submission_rate});
    std.debug.print("  Workers: {}\n", .{config.num_workers});
    std.debug.print("  Iterations: {}\n\n", .{config.num_test_iterations});
    
    // Test baseline (legacy) performance
    std.debug.print("Measuring baseline performance...\n", .{});
    const baseline_metrics = try measureConfiguration(allocator, config, ConfigurationType.baseline);
    
    // Test optimized performance
    std.debug.print("Measuring optimized performance...\n", .{});
    const optimized_metrics = try measureConfiguration(allocator, config, ConfigurationType.optimized);
    
    // Analyze results
    printPerformanceComparison(baseline_metrics, optimized_metrics);
    
    // Validate against COZ findings
    validateAgainstCOZFindings(baseline_metrics, optimized_metrics);
}

const ConfigurationType = enum {
    baseline,
    optimized,
};

fn measureConfiguration(allocator: std.mem.Allocator, config: OptimizationValidationConfig, config_type: ConfigurationType) !PerformanceMetrics {
    var total_metrics = PerformanceMetrics{
        .throughput_tasks_per_second = 0,
        .average_latency_ns = 0,
        .scheduling_overhead_ns = 0,
        .prediction_lookup_time_ns = 0,
        .worker_selection_time_ns = 0,
        .total_execution_time_ns = 0,
    };
    
    // Run multiple iterations for statistical validity
    for (0..config.num_test_iterations) |iteration| {
        if (config.enable_detailed_metrics) {
            std.debug.print("  Iteration {}/{}...\n", .{ iteration + 1, config.num_test_iterations });
        }
        
        const iteration_metrics = try runSinglePerformanceTest(allocator, config, config_type);
        
        // Accumulate metrics
        total_metrics.throughput_tasks_per_second += iteration_metrics.throughput_tasks_per_second;
        total_metrics.average_latency_ns += iteration_metrics.average_latency_ns;
        total_metrics.scheduling_overhead_ns += iteration_metrics.scheduling_overhead_ns;
        total_metrics.prediction_lookup_time_ns += iteration_metrics.prediction_lookup_time_ns;
        total_metrics.worker_selection_time_ns += iteration_metrics.worker_selection_time_ns;
        total_metrics.total_execution_time_ns += iteration_metrics.total_execution_time_ns;
        total_metrics.cache_hit_rate += iteration_metrics.cache_hit_rate;
        total_metrics.cache_lookup_time_ns += iteration_metrics.cache_lookup_time_ns;
    }
    
    // Calculate averages
    const iterations_f = @as(f64, @floatFromInt(config.num_test_iterations));
    total_metrics.throughput_tasks_per_second /= iterations_f;
    total_metrics.average_latency_ns /= iterations_f;
    total_metrics.scheduling_overhead_ns /= iterations_f;
    total_metrics.prediction_lookup_time_ns /= iterations_f;
    total_metrics.worker_selection_time_ns /= iterations_f;
    total_metrics.total_execution_time_ns /= @as(u64, @intFromFloat(iterations_f));
    total_metrics.cache_hit_rate /= iterations_f;
    total_metrics.cache_lookup_time_ns /= iterations_f;
    
    return total_metrics;
}

fn runSinglePerformanceTest(allocator: std.mem.Allocator, config: OptimizationValidationConfig, config_type: ConfigurationType) !PerformanceMetrics {
    // Configure thread pool based on test type
    const pool_config = switch (config_type) {
        .baseline => beat.Config{
            .num_workers = config.num_workers,
            .enable_predictive = false,
            .enable_advanced_selection = false,
            .enable_topology_aware = false,
        },
        .optimized => beat.Config{
            .num_workers = config.num_workers,
            .enable_predictive = true,
            .enable_advanced_selection = true,
            .enable_topology_aware = true,
        },
    };
    
    var pool = try beat.ThreadPool.init(allocator, pool_config);
    defer pool.deinit();
    
    // Performance tracking
    var tasks_completed = std.atomic.Value(u64).init(0);
    var total_scheduling_time = std.atomic.Value(u64).init(0);
    var total_prediction_time = std.atomic.Value(u64).init(0);
    var total_worker_selection_time = std.atomic.Value(u64).init(0);
    
    const test_start = std.time.nanoTimestamp();
    const test_end = test_start + (config.test_duration_seconds * 1_000_000_000);
    
    var task_id: u64 = 0;
    const target_interval_ns = 1_000_000_000 / config.task_submission_rate;
    
    // Task submission loop
    while (std.time.nanoTimestamp() < test_end) {
        const submission_start = std.time.nanoTimestamp();
        
        // Measure prediction lookup time (if optimized)
        var prediction_time: u64 = 0;
        if (config_type == .optimized and pool.fingerprint_registry != null) {
            const pred_start = std.time.nanoTimestamp();
            
            // Generate fingerprint and lookup prediction
            var context = beat.fingerprint.ExecutionContext.init();
            const test_task = createTestTask(task_id, &tasks_completed);
            const fingerprint = beat.fingerprint.generateTaskFingerprint(&test_task, &context);
            _ = pool.fingerprint_registry.?.getPredictionWithConfidence(fingerprint);
            
            prediction_time = @as(u64, @intCast(std.time.nanoTimestamp() - pred_start));
            _ = total_prediction_time.fetchAdd(prediction_time, .monotonic);
        }
        
        // Measure worker selection time
        const selection_start = std.time.nanoTimestamp();
        const test_task = createTestTask(task_id, &tasks_completed);
        _ = pool.selectWorker(test_task);
        const selection_time = @as(u64, @intCast(std.time.nanoTimestamp() - selection_start));
        _ = total_worker_selection_time.fetchAdd(selection_time, .monotonic);
        
        // Submit actual task
        const task = createTestTask(task_id, &tasks_completed);
        try pool.submit(task);
        
        const submission_time = @as(u64, @intCast(std.time.nanoTimestamp() - submission_start));
        _ = total_scheduling_time.fetchAdd(submission_time, .monotonic);
        
        task_id += 1;
        
        // Rate limiting
        const elapsed = @as(u64, @intCast(std.time.nanoTimestamp() - submission_start));
        if (elapsed < target_interval_ns) {
            std.time.sleep(target_interval_ns - elapsed);
        }
    }
    
    // Wait for all tasks to complete
    pool.wait();
    const actual_end = std.time.nanoTimestamp();
    
    // Calculate metrics
    const completed = tasks_completed.load(.acquire);
    const total_time = @as(u64, @intCast(actual_end - test_start));
    const throughput = @as(f64, @floatFromInt(completed)) / (@as(f64, @floatFromInt(total_time)) / 1_000_000_000.0);
    const avg_latency = @as(f64, @floatFromInt(total_time)) / @as(f64, @floatFromInt(completed));
    const avg_scheduling_overhead = @as(f64, @floatFromInt(total_scheduling_time.load(.acquire))) / @as(f64, @floatFromInt(completed));
    const avg_prediction_time = if (completed > 0) 
        @as(f64, @floatFromInt(total_prediction_time.load(.acquire))) / @as(f64, @floatFromInt(completed))
    else 0.0;
    const avg_selection_time = @as(f64, @floatFromInt(total_worker_selection_time.load(.acquire))) / @as(f64, @floatFromInt(completed));
    
    return PerformanceMetrics{
        .throughput_tasks_per_second = throughput,
        .average_latency_ns = avg_latency,
        .scheduling_overhead_ns = avg_scheduling_overhead,
        .prediction_lookup_time_ns = avg_prediction_time,
        .worker_selection_time_ns = avg_selection_time,
        .total_execution_time_ns = total_time,
        .cache_hit_rate = 0.0, // Would be measured by cached registry
        .cache_lookup_time_ns = 0.0, // Simplified for now
    };
}

fn createTestTask(task_id: u64, counter: *std.atomic.Value(u64)) beat.Task {
    return beat.Task{
        .func = benchmarkTask,
        .data = @ptrCast(counter),
        .priority = switch (task_id % 3) {
            0 => .high,
            1 => .normal,
            2 => .low,
            else => .normal,
        },
        .data_size_hint = @as(usize, @intCast(64 + (task_id % 512))), // Variable size hint
        .affinity_hint = @as(u32, @intCast(task_id % 2)), // Alternate NUMA nodes
    };
}

fn benchmarkTask(data: *anyopaque) void {
    const counter = @as(*std.atomic.Value(u64), @ptrCast(@alignCast(data)));
    
    // Simulate work with some variety
    var result: u64 = 42;
    const work_amount = 1000 + (counter.load(.acquire) % 2000);
    
    for (0..work_amount) |i| {
        result = result *% (i + 1) +% 7;
    }
    std.mem.doNotOptimizeAway(result);
    
    _ = counter.fetchAdd(1, .monotonic);
}

fn printPerformanceComparison(baseline: PerformanceMetrics, optimized: PerformanceMetrics) void {
    std.debug.print("\n=== Performance Comparison Results ===\n", .{});
    
    std.debug.print("\nBaseline Performance:\n", .{});
    printMetrics("  ", baseline);
    
    std.debug.print("\nOptimized Performance:\n", .{});
    printMetrics("  ", optimized);
    
    const improvement = PerformanceMetrics.calculateImprovement(optimized, baseline);
    
    std.debug.print("\nImprovement Analysis:\n", .{});
    std.debug.print("  Throughput change: {d:.1}%\n", .{improvement.throughput_improvement_percent});
    std.debug.print("  Latency change: {d:.1}%\n", .{improvement.latency_reduction_percent});
    std.debug.print("  Scheduling overhead change: {d:.1}%\n", .{improvement.overhead_reduction_percent});
    std.debug.print("  Prediction speedup: {d:.1}x\n", .{improvement.prediction_speedup_factor});
    
    if (improvement.isSignificantImprovement()) {
        std.debug.print("  ✅ Significant performance improvement detected!\n", .{});
    } else {
        std.debug.print("  ⚠️  No significant performance improvement detected\n", .{});
    }
}

fn printMetrics(prefix: []const u8, metrics: PerformanceMetrics) void {
    std.debug.print("{s}Throughput: {d:.0} tasks/second\n", .{ prefix, metrics.throughput_tasks_per_second });
    std.debug.print("{s}Average latency: {d:.1} μs\n", .{ prefix, metrics.average_latency_ns / 1000.0 });
    std.debug.print("{s}Scheduling overhead: {d:.1} μs/task\n", .{ prefix, metrics.scheduling_overhead_ns / 1000.0 });
    std.debug.print("{s}Prediction lookup: {d:.1} ns/task\n", .{ prefix, metrics.prediction_lookup_time_ns });
    std.debug.print("{s}Worker selection: {d:.1} ns/task\n", .{ prefix, metrics.worker_selection_time_ns });
    std.debug.print("{s}Total execution: {d:.1} ms\n", .{ prefix, @as(f64, @floatFromInt(metrics.total_execution_time_ns)) / 1_000_000.0 });
}

fn validateAgainstCOZFindings(baseline: PerformanceMetrics, optimized: PerformanceMetrics) void {
    std.debug.print("\n=== COZ Findings Validation ===\n", .{});
    
    // COZ showed 2.7% throughput reduction (452 -> 440 tasks/s)
    const expected_overhead_percent = -2.7;
    const actual_change = ((optimized.throughput_tasks_per_second - baseline.throughput_tasks_per_second) / baseline.throughput_tasks_per_second) * 100.0;
    
    std.debug.print("COZ findings: 2.7% throughput reduction with advanced scheduling\n", .{});
    std.debug.print("Our measurement: {d:.1}% throughput change\n", .{actual_change});
    
    const difference = @abs(actual_change - expected_overhead_percent);
    if (difference < 5.0) {
        std.debug.print("✅ Results consistent with COZ profiling findings\n", .{});
    } else {
        std.debug.print("⚠️  Results differ from COZ findings by {d:.1}%\n", .{difference});
    }
    
    // Analysis of optimization opportunities
    std.debug.print("\nOptimization Analysis:\n", .{});
    if (optimized.prediction_lookup_time_ns > 0) {
        std.debug.print("  Prediction lookup overhead: {d:.1} ns/task\n", .{optimized.prediction_lookup_time_ns});
        if (optimized.prediction_lookup_time_ns > 1000) {
            std.debug.print("  🎯 Caching could reduce this overhead\n", .{});
        }
    }
    
    if (optimized.worker_selection_time_ns > baseline.worker_selection_time_ns * 2) {
        std.debug.print("  🎯 Worker selection shows {d:.1}x overhead\n", .{optimized.worker_selection_time_ns / baseline.worker_selection_time_ns});
        std.debug.print("  💡 Fast path for simple tasks recommended\n", .{});
    }
    
    std.debug.print("\nNext Steps:\n", .{});
    std.debug.print("  1. Implement prediction lookup caching\n", .{});
    std.debug.print("  2. Add worker selection fast path\n", .{});
    std.debug.print("  3. Validate with A/B testing framework\n", .{});
    std.debug.print("  4. Re-run COZ profiling to measure impact\n", .{});
}