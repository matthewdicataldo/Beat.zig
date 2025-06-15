const std = @import("std");
const beat = @import("src/core.zig");

// Enhanced COZ Profiler Integration for Benchmarking (Task 2.5.1.3)
//
// This benchmark provides deep integration with COZ profiler for causal profiling
// of scheduling decisions, latency analysis, and throughput optimization points.

const COZBenchmarkConfig = struct {
    test_duration_seconds: u32 = 5,    // Reduced for testing
    task_submission_rate_hz: u32 = 500, // Reduced rate
    enable_detailed_progress_points: bool = true,
    enable_latency_tracking: bool = true,
    enable_throughput_tracking: bool = true,
    worker_count: usize = 4,
    task_variety_count: usize = 5,
    enable_scheduling_profiling: bool = true,
    enable_numa_profiling: bool = true,
};

// Workaround for accessing private getWorkerQueueSize
fn getWorkerQueueSizeWorkaround(pool: *beat.ThreadPool, worker_id: usize) usize {
    const worker = &pool.workers[worker_id];
    
    return switch (worker.queue) {
        .mutex => |*q| blk: {
            q.mutex.lock();
            defer q.mutex.unlock();
            break :blk q.tasks[0].items.len + q.tasks[1].items.len + q.tasks[2].items.len;
        },
        .lockfree => |*q| q.size(),
    };
}

const COZProgressPoints = struct {
    // Enhanced scheduling decision points
    pub const worker_selection_begin = "beat_worker_selection_begin";
    pub const worker_selection_end = "beat_worker_selection_end";
    pub const advanced_selection_begin = "beat_advanced_selection_begin";
    pub const advanced_selection_end = "beat_advanced_selection_end";
    pub const prediction_lookup_begin = "beat_prediction_lookup_begin";
    pub const prediction_lookup_end = "beat_prediction_lookup_end";
    pub const fingerprint_generation_begin = "beat_fingerprint_generation_begin";
    pub const fingerprint_generation_end = "beat_fingerprint_generation_end";
    
    // NUMA and topology optimization points
    pub const numa_placement_begin = "beat_numa_placement_begin";
    pub const numa_placement_end = "beat_numa_placement_end";
    pub const topology_aware_steal_begin = "beat_topology_aware_steal_begin";
    pub const topology_aware_steal_end = "beat_topology_aware_steal_end";
    
    // Confidence and decision framework points
    pub const confidence_calculation_begin = "beat_confidence_calculation_begin";
    pub const confidence_calculation_end = "beat_confidence_calculation_end";
    pub const decision_framework_begin = "beat_decision_framework_begin";
    pub const decision_framework_end = "beat_decision_framework_end";
    
    // Task execution phases
    pub const task_queue_push = "beat_task_queue_push";
    pub const task_queue_pop = "beat_task_queue_pop";
    pub const task_execution_begin = "beat_task_execution_begin";
    pub const task_execution_end = "beat_task_execution_end";
    pub const task_completion = "beat_task_completion";
    
    // Worker and load balancing
    pub const load_balance_check = "beat_load_balance_check";
    pub const worker_idle = "beat_worker_idle";
    pub const worker_active = "beat_worker_active";
    pub const work_stealing_attempt = "beat_work_stealing_attempt";
    pub const work_stealing_success = "beat_work_stealing_success";
    
    // Predictive scheduling components
    pub const prediction_update = "beat_prediction_update";
    pub const filter_adaptation = "beat_filter_adaptation";
    pub const confidence_update = "beat_confidence_update";
    pub const learning_adaptation = "beat_learning_adaptation";
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){}; 
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const config = COZBenchmarkConfig{};
    
    std.debug.print("=== Beat.zig Enhanced COZ Profiler Benchmark ===\n", .{});
    std.debug.print("Configuration:\n", .{});
    std.debug.print("  Test duration: {}s\n", .{config.test_duration_seconds});
    std.debug.print("  Task submission rate: {} Hz\n", .{config.task_submission_rate_hz});
    std.debug.print("  Workers: {}\n", .{config.worker_count});
    std.debug.print("  COZ integration: {}\n", .{config.enable_detailed_progress_points});
    std.debug.print("\nRunning benchmark with COZ profiling...\n", .{});
    
    // Run different profiling scenarios
    try runLegacyProfilingScenario(allocator, config);
    try runAdvancedProfilingScenario(allocator, config);
    try runDetailedComponentProfiling(allocator, config);
    
    std.debug.print("\n=== COZ Profiling Complete ===\n", .{});
    std.debug.print("COZ Data Collection Points:\n", .{});
    printCOZProgressPoints();
    std.debug.print("\nTo analyze results:\n", .{});
    std.debug.print("  1. Run with: coz run --- ./zig-out/bin/benchmark_coz_enhanced\n", .{});
    std.debug.print("  2. View results: coz plot\n", .{});
    std.debug.print("  3. Focus on scheduling optimization points\n", .{});
}

fn runLegacyProfilingScenario(allocator: std.mem.Allocator, config: COZBenchmarkConfig) !void {
    std.debug.print("\n--- Legacy Scheduling COZ Profile ---\n", .{});
    
    const pool_config = beat.Config{
        .num_workers = config.worker_count,
        .enable_predictive = false,
        .enable_advanced_selection = false,
        .enable_topology_aware = false,
    };
    
    var pool = try beat.ThreadPool.init(allocator, pool_config);
    defer pool.deinit();
    
    try runProfiledWorkload(pool, config, "Legacy");
}

fn runAdvancedProfilingScenario(allocator: std.mem.Allocator, config: COZBenchmarkConfig) !void {
    std.debug.print("\n--- Advanced Scheduling COZ Profile ---\n", .{});
    
    const pool_config = beat.Config{
        .num_workers = config.worker_count,
        .enable_predictive = true,
        .enable_advanced_selection = true,
        .enable_topology_aware = true,
    };
    
    var pool = try beat.ThreadPool.init(allocator, pool_config);
    defer pool.deinit();
    
    try runProfiledWorkload(pool, config, "Advanced");
}

fn runDetailedComponentProfiling(allocator: std.mem.Allocator, config: COZBenchmarkConfig) !void {
    std.debug.print("\n--- Detailed Component COZ Profile ---\n", .{});
    
    const pool_config = beat.Config{
        .num_workers = config.worker_count,
        .enable_predictive = true,
        .enable_advanced_selection = true,
        .enable_topology_aware = true,
        .enable_statistics = true,
    };
    
    var pool = try beat.ThreadPool.init(allocator, pool_config);
    defer pool.deinit();
    
    // Focus on detailed component profiling
    try runComponentProfiledWorkload(pool, config);
}

fn runProfiledWorkload(pool: *beat.ThreadPool, config: COZBenchmarkConfig, scenario_name: []const u8) !void {
    std.debug.print("Running {s} scenario for {}s...\n", .{ scenario_name, config.test_duration_seconds });
    
    var task_counter = std.atomic.Value(u64).init(0);
    const start_time = std.time.milliTimestamp();
    const end_time = start_time + (config.test_duration_seconds * 1000);
    
    var task_id: u64 = 0;
    
    while (std.time.milliTimestamp() < end_time) {
        // COZ throughput tracking
        beat.coz.throughput(beat.coz.Points.task_submitted);
        
        // Task submission with COZ latency tracking
        beat.coz.latencyBegin(COZProgressPoints.task_queue_push);
        
        const task_type = task_id % config.task_variety_count;
        const task = createProfiledTask(task_type, &task_counter, task_id);
        
        // Profile worker selection
        if (config.enable_scheduling_profiling) {
            beat.coz.latencyBegin(COZProgressPoints.worker_selection_begin);
            _ = pool.selectWorker(task);
            beat.coz.latencyEnd(COZProgressPoints.worker_selection_end);
        }
        
        try pool.submit(task);
        beat.coz.latencyEnd(COZProgressPoints.task_queue_push);
        
        task_id += 1;
        
        // Rate limiting
        const delay_us = 1_000_000 / config.task_submission_rate_hz;
        std.time.sleep(delay_us * 1000);
    }
    
    // Wait for completion with progress tracking
    beat.coz.latencyBegin("workload_completion");
    pool.wait();
    beat.coz.latencyEnd("workload_completion");
    
    const completed_tasks = task_counter.load(.acquire);
    const actual_duration = std.time.milliTimestamp() - start_time;
    const throughput = @as(f64, @floatFromInt(completed_tasks)) / (@as(f64, @floatFromInt(actual_duration)) / 1000.0);
    
    std.debug.print("  Completed {} tasks in {}ms ({d:.0} tasks/s)\n", .{ completed_tasks, actual_duration, throughput });
}

fn runComponentProfiledWorkload(pool: *beat.ThreadPool, config: COZBenchmarkConfig) !void {
    std.debug.print("Running detailed component profiling for {}s...\n", .{config.test_duration_seconds});
    
    var task_counter = std.atomic.Value(u64).init(0);
    const start_time = std.time.milliTimestamp();
    const end_time = start_time + (config.test_duration_seconds * 1000);
    
    var task_id: u64 = 0;
    
    while (std.time.milliTimestamp() < end_time) {
        beat.coz.throughput("component_analysis_task");
        
        const task_type = task_id % config.task_variety_count;
        
        // Profile fingerprint generation
        beat.coz.latencyBegin(COZProgressPoints.fingerprint_generation_begin);
        var context = beat.fingerprint.ExecutionContext.init();
        const temp_task = createProfiledTask(task_type, &task_counter, task_id);
        const fingerprint = beat.fingerprint.generateTaskFingerprint(&temp_task, &context);
        beat.coz.latencyEnd(COZProgressPoints.fingerprint_generation_end);
        
        // Profile prediction lookup
        if (pool.fingerprint_registry) |registry| {
            beat.coz.latencyBegin(COZProgressPoints.prediction_lookup_begin);
            const profile = registry.getProfile(fingerprint);
            beat.coz.latencyEnd(COZProgressPoints.prediction_lookup_end);
            
            // Profile confidence calculation
            if (profile) |p| {
                beat.coz.latencyBegin(COZProgressPoints.confidence_calculation_begin);
                _ = p.getMultiFactorConfidence();
                beat.coz.latencyEnd(COZProgressPoints.confidence_calculation_end);
            }
        }
        
        // Profile advanced worker selection
        if (pool.advanced_selector) |_| {
            beat.coz.latencyBegin(COZProgressPoints.advanced_selection_begin);
            _ = pool.selectWorker(temp_task);
            beat.coz.latencyEnd(COZProgressPoints.advanced_selection_end);
        }
        
        // Profile decision framework
        if (pool.decision_framework) |framework| {
            beat.coz.latencyBegin(COZProgressPoints.decision_framework_begin);
            
            // Create worker info for decision making
            var worker_infos = std.ArrayList(beat.intelligent_decision.WorkerInfo).init(std.heap.page_allocator);
            defer worker_infos.deinit();
            
            for (pool.workers, 0..) |*worker, i| {
                const worker_info = beat.intelligent_decision.WorkerInfo{
                    .id = worker.id,
                    .numa_node = worker.numa_node,
                    .queue_size = getWorkerQueueSizeWorkaround(pool, i),
                    .max_queue_size = 1024,
                };
                worker_infos.append(worker_info) catch break;
            }
            
            if (worker_infos.items.len > 0) {
                _ = framework.makeSchedulingDecision(&temp_task, worker_infos.items, pool.topology);
            }
            
            beat.coz.latencyEnd(COZProgressPoints.decision_framework_end);
        }
        
        // Submit actual task
        beat.coz.latencyBegin(COZProgressPoints.task_queue_push);
        const task = createProfiledTask(task_type, &task_counter, task_id);
        try pool.submit(task);
        beat.coz.latencyEnd(COZProgressPoints.task_queue_push);
        
        task_id += 1;
        
        // COZ progress point for adaptive learning
        if (task_id % 100 == 0) {
            beat.coz.throughput(COZProgressPoints.learning_adaptation);
        }
        
        // Rate limiting
        const delay_us = 1_000_000 / config.task_submission_rate_hz;
        std.time.sleep(delay_us * 1000);
    }
    
    beat.coz.latencyBegin("detailed_workload_completion");
    pool.wait();
    beat.coz.latencyEnd("detailed_workload_completion");
    
    const completed_tasks = task_counter.load(.acquire);
    const actual_duration = std.time.milliTimestamp() - start_time;
    const throughput = @as(f64, @floatFromInt(completed_tasks)) / (@as(f64, @floatFromInt(actual_duration)) / 1000.0);
    
    std.debug.print("  Component analysis: {} tasks in {}ms ({d:.0} tasks/s)\n", .{ completed_tasks, actual_duration, throughput });
}

fn createProfiledTask(task_type: u64, counter: *std.atomic.Value(u64), task_id: u64) beat.Task {
    _ = task_id; // Simplified - just use counter
    return beat.Task{
        .func = switch (task_type) {
            0 => profiledCpuIntensiveTask,
            1 => profiledMemoryIntensiveTask,
            2 => profiledMixedWorkloadTask,
            3 => profiledShortBurstTask,
            4 => profiledLongRunningTask,
            else => profiledCpuIntensiveTask,
        },
        .data = @ptrCast(counter), // Simplified - just use counter directly
        .priority = switch (task_type) {
            3 => .high,    // short burst
            0, 1 => .normal, // cpu/memory intensive  
            2, 4 => .low,  // mixed/long running
            else => .normal,
        },
        .data_size_hint = switch (task_type) {
            0 => 64,
            1 => 1024,
            2 => 256,
            3 => 16,
            4 => 512,
            else => 64,
        },
        .affinity_hint = if (task_type % 2 == 0) 0 else 1, // Alternate NUMA nodes
    };
}

const TaskContext = struct {
    counter: *std.atomic.Value(u64),
    task_id: u64,
    task_type: u32,
};

// Task implementations with COZ profiling

fn profiledCpuIntensiveTask(data: *anyopaque) void {
    const counter = @as(*std.atomic.Value(u64), @ptrCast(@alignCast(data)));
    
    beat.coz.latencyBegin(COZProgressPoints.task_execution_begin);
    beat.coz.throughput(COZProgressPoints.worker_active);
    
    var result: u64 = 42;
    for (0..10000) |i| { // Reduced workload
        result = result *% (i + 1) +% 17;
        
        // COZ progress point for computation
        if (i % 2000 == 0) {
            beat.coz.throughput("cpu_computation_progress");
        }
    }
    std.mem.doNotOptimizeAway(result);
    
    _ = counter.fetchAdd(1, .monotonic);
    beat.coz.throughput(COZProgressPoints.task_completion);
    beat.coz.latencyEnd(COZProgressPoints.task_execution_end);
}

fn profiledMemoryIntensiveTask(data: *anyopaque) void {
    const counter = @as(*std.atomic.Value(u64), @ptrCast(@alignCast(data)));
    
    beat.coz.latencyBegin(COZProgressPoints.task_execution_begin);
    beat.coz.throughput(COZProgressPoints.worker_active);
    
    var buffer: [256]u64 = undefined; // Reduced size
    for (0..50) |iteration| { // Reduced iterations
        for (0..256) |i| {
            buffer[i] = buffer[(i + 37) % 256] *% iteration +% i;
        }
        
        // COZ progress point for memory operations
        if (iteration % 10 == 0) {
            beat.coz.throughput("memory_access_progress");
        }
    }
    std.mem.doNotOptimizeAway(buffer);
    
    _ = counter.fetchAdd(1, .monotonic);
    beat.coz.throughput(COZProgressPoints.task_completion);
    beat.coz.latencyEnd(COZProgressPoints.task_execution_end);
}

fn profiledMixedWorkloadTask(data: *anyopaque) void {
    const counter = @as(*std.atomic.Value(u64), @ptrCast(@alignCast(data)));
    
    beat.coz.latencyBegin(COZProgressPoints.task_execution_begin);
    beat.coz.throughput(COZProgressPoints.worker_active);
    
    var result: u64 = 42;
    var buffer: [128]u64 = undefined;
    
    for (0..10) |iteration| {
        // CPU work
        for (0..500) |i| {
            result = result *% (i + 1) +% iteration;
        }
        // Memory work
        for (0..128) |i| {
            buffer[i] = result +% (i * iteration);
        }
        
        // COZ progress point for mixed operations
        if (iteration % 2 == 0) {
            beat.coz.throughput("mixed_workload_progress");
        }
    }
    
    std.mem.doNotOptimizeAway(result);
    std.mem.doNotOptimizeAway(buffer);
    
    _ = counter.fetchAdd(1, .monotonic);
    beat.coz.throughput(COZProgressPoints.task_completion);
    beat.coz.latencyEnd(COZProgressPoints.task_execution_end);
}

fn profiledShortBurstTask(data: *anyopaque) void {
    const counter = @as(*std.atomic.Value(u64), @ptrCast(@alignCast(data)));
    
    beat.coz.latencyBegin(COZProgressPoints.task_execution_begin);
    beat.coz.throughput(COZProgressPoints.worker_active);
    
    var result: u64 = 42;
    for (0..100) |i| {
        result = result *% (i + 1) +% 7;
    }
    std.mem.doNotOptimizeAway(result);
    
    _ = counter.fetchAdd(1, .monotonic);
    beat.coz.throughput(COZProgressPoints.task_completion);
    beat.coz.latencyEnd(COZProgressPoints.task_execution_end);
}

fn profiledLongRunningTask(data: *anyopaque) void {
    const counter = @as(*std.atomic.Value(u64), @ptrCast(@alignCast(data)));
    
    beat.coz.latencyBegin(COZProgressPoints.task_execution_begin);
    beat.coz.throughput(COZProgressPoints.worker_active);
    
    var result: u64 = 42;
    var buffer: [128]u64 = undefined;
    
    for (0..20) |iteration| { // Reduced
        for (0..128) |i| {
            result = result *% (i + iteration + 1) +% 13;
            buffer[i] = result;
        }
        for (0..128) |i| {
            result = result +% buffer[(i + 31) % 128];
        }
        
        // COZ progress point for long-running operations
        if (iteration % 5 == 0) {
            beat.coz.throughput("long_running_progress");
        }
    }
    
    std.mem.doNotOptimizeAway(result);
    std.mem.doNotOptimizeAway(buffer);
    
    _ = counter.fetchAdd(1, .monotonic);
    beat.coz.throughput(COZProgressPoints.task_completion);
    beat.coz.latencyEnd(COZProgressPoints.task_execution_end);
}

fn printCOZProgressPoints() void {
    std.debug.print("  Core Scheduling:\n", .{});
    std.debug.print("    - {s}\n", .{COZProgressPoints.worker_selection_begin});
    std.debug.print("    - {s}\n", .{COZProgressPoints.worker_selection_end});
    std.debug.print("    - {s}\n", .{COZProgressPoints.advanced_selection_begin});
    std.debug.print("    - {s}\n", .{COZProgressPoints.advanced_selection_end});
    
    std.debug.print("  Prediction System:\n", .{});
    std.debug.print("    - {s}\n", .{COZProgressPoints.fingerprint_generation_begin});
    std.debug.print("    - {s}\n", .{COZProgressPoints.fingerprint_generation_end});
    std.debug.print("    - {s}\n", .{COZProgressPoints.prediction_lookup_begin});
    std.debug.print("    - {s}\n", .{COZProgressPoints.prediction_lookup_end});
    std.debug.print("    - {s}\n", .{COZProgressPoints.confidence_calculation_begin});
    std.debug.print("    - {s}\n", .{COZProgressPoints.confidence_calculation_end});
    
    std.debug.print("  NUMA Optimization:\n", .{});
    std.debug.print("    - {s}\n", .{COZProgressPoints.numa_placement_begin});
    std.debug.print("    - {s}\n", .{COZProgressPoints.numa_placement_end});
    std.debug.print("    - {s}\n", .{COZProgressPoints.topology_aware_steal_begin});
    std.debug.print("    - {s}\n", .{COZProgressPoints.topology_aware_steal_end});
    
    std.debug.print("  Task Execution:\n", .{});
    std.debug.print("    - {s}\n", .{COZProgressPoints.task_execution_begin});
    std.debug.print("    - {s}\n", .{COZProgressPoints.task_execution_end});
    std.debug.print("    - {s}\n", .{COZProgressPoints.task_completion});
    
    std.debug.print("  Adaptive Learning:\n", .{});
    std.debug.print("    - {s}\n", .{COZProgressPoints.prediction_update});
    std.debug.print("    - {s}\n", .{COZProgressPoints.filter_adaptation});
    std.debug.print("    - {s}\n", .{COZProgressPoints.learning_adaptation});
}