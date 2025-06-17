const std = @import("std");
const beat = @import("beat");

// ============================================================================
// COZ Profiling Integration for Continuation Stealing Analysis
// ============================================================================

// COZ progress points for bottleneck identification
const coz = beat.coz;

const BenchmarkConfig = struct {
    num_workers: u32 = 4,
    workload_size: u32 = 1000,
    iterations: u32 = 100,
    work_complexity: WorkComplexity = .heavy,
    enable_numa_aware: bool = true,
};

const WorkComplexity = enum {
    light,   // ~1μs per task
    medium,  // ~10μs per task  
    heavy,   // ~100μs per task
};

const WorkItem = struct {
    id: u32,
    complexity: WorkComplexity,
    data: *i64,
    completed: *std.atomic.Value(bool),
    
    pub fn execute(self: *const WorkItem) void {
        // COZ progress point: Work execution start
        coz.throughput("work_execution");
        
        const work_cycles: u32 = switch (self.complexity) {
            .light => 1000,
            .medium => 10_000,
            .heavy => 100_000,
        };
        
        // Simulate computational work with memory access patterns
        var sum: i64 = 0;
        for (0..work_cycles) |i| {
            sum += @as(i64, @intCast(i)) * @as(i64, @intCast(self.id));
            
            // Prevent compiler optimization of the loop
            if (i % 1000 == 0) {
                std.mem.doNotOptimizeAway(&sum);
            }
        }
        
        // Store result with proper memory ordering
        @atomicStore(i64, self.data, sum, .seq_cst);
        self.completed.store(true, .release);
        
        // COZ progress point: Work execution complete
        coz.throughput("work_completion");
    }
};

// ============================================================================
// COZ-Instrumented Continuation Stealing Benchmark
// ============================================================================

fn benchmarkContinuationStealingWithCoz(
    allocator: std.mem.Allocator,
    config: BenchmarkConfig,
) !void {
    std.debug.print("🔍 COZ Profiling: NUMA-Aware Continuation Stealing\n", .{});
    std.debug.print("Configuration: {} workers, {} tasks, {s} complexity\n", 
        .{ config.num_workers, config.workload_size, @tagName(config.work_complexity) });
    
    const pool_config = beat.Config{
        .num_workers = config.num_workers,
        .enable_work_stealing = true,
        .enable_lock_free = true,
        .enable_topology_aware = config.enable_numa_aware,
        .enable_numa_aware = config.enable_numa_aware,
        .enable_predictive = true,
        .enable_statistics = true,
    };
    
    var pool = try beat.ThreadPool.init(allocator, pool_config);
    defer pool.deinit();
    
    // COZ progress point: Pool initialization complete
    coz.throughput("pool_init");
    
    // Prepare continuation work
    const work_items = try allocator.alloc(WorkItem, config.workload_size);
    defer allocator.free(work_items);
    
    var results = try allocator.alloc(i64, config.workload_size);
    defer allocator.free(results);
    
    var completion_flags = try allocator.alloc(std.atomic.Value(bool), config.workload_size);
    defer allocator.free(completion_flags);
    
    const continuations = try allocator.alloc(beat.continuation.Continuation, config.workload_size);
    defer allocator.free(continuations);
    
    // Initialize work items
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
    
    // COZ progress point: Work initialization complete
    coz.throughput("work_init");
    
    // Create continuations with COZ instrumentation
    const ContinuationExecutor = struct {
        fn executeWrapper(cont: *beat.continuation.Continuation) void {
            // COZ progress point: Continuation execution start
            coz.throughput("continuation_start");
            
            const work_item = @as(*WorkItem, @ptrCast(@alignCast(cont.data)));
            work_item.execute();
            cont.markCompleted();
            
            // COZ progress point: Continuation execution complete
            coz.throughput("continuation_complete");
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
            const numa_node: u32 = @intCast(i % 2);
            cont.initNumaLocality(numa_node, numa_node / 2);
        }
    }
    
    // COZ progress point: Continuation setup complete
    coz.throughput("continuation_setup");
    
    std.debug.print("Running {} iterations for statistical significance...\n", .{config.iterations});
    
    var total_time: u64 = 0;
    var total_steals: u32 = 0;
    var total_migrations: u32 = 0;
    
    for (0..config.iterations) |iteration| {
        // Reset completion flags
        for (completion_flags) |*flag| {
            flag.store(false, .release);
        }
        
        for (results) |*result| {
            result.* = 0;
        }
        
        // COZ progress point: Iteration start
        coz.throughput("iteration_start");
        
        const start_time = std.time.nanoTimestamp();
        
        // Submit all continuations
        for (continuations) |*cont| {
            // COZ progress point: Continuation submission
            coz.throughput("continuation_submit");
            
            try pool.submitContinuation(cont);
        }
        
        // COZ progress point: All continuations submitted
        coz.throughput("submission_complete");
        
        // Wait for completion
        pool.wait();
        
        const end_time = std.time.nanoTimestamp();
        const iteration_time = @as(u64, @intCast(end_time - start_time));
        total_time += iteration_time;
        
        // COZ progress point: Iteration complete
        coz.throughput("iteration_complete");
        
        // Collect statistics
        var steals: u32 = 0;
        var migrations: u32 = 0;
        
        for (continuations) |*cont| {
            steals += cont.steal_count;
            migrations += cont.migration_count;
            
            // Reset for next iteration
            cont.steal_count = 0;
            cont.migration_count = 0;
            cont.state = .pending;
        }
        
        total_steals += steals;
        total_migrations += migrations;
        
        // Verify completion
        for (completion_flags) |*flag| {
            if (!flag.load(.acquire)) {
                return error.IncompleteExecution;
            }
        }
        
        if (iteration % 10 == 0) {
            std.debug.print("  Iteration {}: {d:.2}ms, {} steals, {} migrations\n", 
                .{ iteration, @as(f64, @floatFromInt(iteration_time)) / 1_000_000.0, steals, migrations });
        }
    }
    
    // COZ progress point: All iterations complete
    coz.throughput("benchmark_complete");
    
    // Calculate and report final statistics
    const avg_time = total_time / config.iterations;
    const avg_steals = @as(f32, @floatFromInt(total_steals)) / @as(f32, @floatFromInt(config.iterations));
    const avg_migrations = @as(f32, @floatFromInt(total_migrations)) / @as(f32, @floatFromInt(config.iterations));
    
    const throughput = @as(f64, @floatFromInt(config.workload_size)) / (@as(f64, @floatFromInt(avg_time)) / 1_000_000_000.0);
    const avg_latency = @as(f64, @floatFromInt(avg_time)) / @as(f64, @floatFromInt(config.workload_size));
    
    std.debug.print("\n🎯 COZ Profiling Results:\n", .{});
    std.debug.print("=======================\n", .{});
    std.debug.print("Average execution time: {d:.2}ms\n", .{@as(f64, @floatFromInt(avg_time)) / 1_000_000.0});
    std.debug.print("Throughput: {d:.0} tasks/sec\n", .{throughput});
    std.debug.print("Average latency: {d:.1}μs\n", .{avg_latency / 1000.0});
    std.debug.print("Average steals per iteration: {d:.1}\n", .{avg_steals});
    std.debug.print("Average NUMA migrations: {d:.1}\n", .{avg_migrations});
    std.debug.print("Steal rate: {d:.1}%\n", .{avg_steals / @as(f32, @floatFromInt(config.workload_size)) * 100.0});
    
    // Note: ThreadPool statistics would be displayed here if available
    std.debug.print("\n📊 ThreadPool Statistics: (Not implemented in current API)\n", .{});
    
    std.debug.print("\n💡 COZ Usage:\n", .{});
    std.debug.print("To analyze bottlenecks, run with: coz run --- ./benchmark\n", .{});
    std.debug.print("Key progress points to analyze:\n", .{});
    std.debug.print("  - continuation_submit: Submission overhead\n", .{});
    std.debug.print("  - continuation_start: Scheduling latency\n", .{});
    std.debug.print("  - work_execution: Actual work time\n", .{});
    std.debug.print("  - continuation_complete: Completion overhead\n", .{});
}

// ============================================================================
// Comparative COZ Analysis: Baseline vs Continuation Stealing
// ============================================================================

fn comparativeAnalysisWithCoz(
    allocator: std.mem.Allocator,
    config: BenchmarkConfig,
) !void {
    std.debug.print("\n🔍 COZ Comparative Analysis: Baseline vs Continuation Stealing\n", .{});
    std.debug.print("===============================================================\n", .{});
    
    // Baseline work stealing analysis
    std.debug.print("\n📈 Phase 1: Baseline Work Stealing Analysis\n", .{});
    
    const baseline_config = beat.Config{
        .num_workers = config.num_workers,
        .enable_work_stealing = true,
        .enable_lock_free = true,
        .enable_topology_aware = false,
        .enable_numa_aware = false,
        .enable_predictive = false,
        .enable_statistics = true,
    };
    
    var baseline_pool = try beat.ThreadPool.init(allocator, baseline_config);
    defer baseline_pool.deinit();
    
    // Prepare baseline work
    const baseline_work_items = try allocator.alloc(WorkItem, config.workload_size);
    defer allocator.free(baseline_work_items);
    
    var baseline_results = try allocator.alloc(i64, config.workload_size);
    defer allocator.free(baseline_results);
    
    var baseline_completion_flags = try allocator.alloc(std.atomic.Value(bool), config.workload_size);
    defer allocator.free(baseline_completion_flags);
    
    for (baseline_work_items, 0..) |*item, i| {
        item.* = WorkItem{
            .id = @intCast(i),
            .complexity = config.work_complexity,
            .data = &baseline_results[i],
            .completed = &baseline_completion_flags[i],
        };
        baseline_completion_flags[i] = std.atomic.Value(bool).init(false);
        baseline_results[i] = 0;
    }
    
    // COZ progress point: Baseline setup
    coz.throughput("baseline_setup");
    
    const baseline_start = std.time.nanoTimestamp();
    
    for (baseline_work_items) |*item| {
        // COZ progress point: Task submission
        coz.throughput("task_submit");
        
        const task = beat.Task{
            .func = struct {
                fn taskWrapper(data: *anyopaque) void {
                    // COZ progress point: Task execution start
                    coz.throughput("task_start");
                    
                    const work_item = @as(*WorkItem, @ptrCast(@alignCast(data)));
                    work_item.execute();
                    
                    // COZ progress point: Task execution complete
                    coz.throughput("task_complete");
                }
            }.taskWrapper,
            .data = item,
            .priority = .normal,
        };
        try baseline_pool.submit(task);
    }
    
    // COZ progress point: All tasks submitted
    coz.throughput("baseline_submit_complete");
    
    baseline_pool.wait();
    
    const baseline_end = std.time.nanoTimestamp();
    const baseline_time = @as(u64, @intCast(baseline_end - baseline_start));
    
    // COZ progress point: Baseline complete
    coz.throughput("baseline_complete");
    
    // Verify baseline completion
    for (baseline_completion_flags) |*flag| {
        if (!flag.load(.acquire)) {
            return error.BaselineIncompleteExecution;
        }
    }
    
    std.debug.print("Baseline execution time: {d:.2}ms\n", .{@as(f64, @floatFromInt(baseline_time)) / 1_000_000.0});
    
    // Continuation stealing analysis
    std.debug.print("\n📈 Phase 2: Continuation Stealing Analysis\n", .{});
    
    const continuation_config_coz = BenchmarkConfig{
        .num_workers = config.num_workers,
        .workload_size = config.workload_size,
        .iterations = 1, // Single run for COZ analysis
        .work_complexity = config.work_complexity,
        .enable_numa_aware = config.enable_numa_aware,
    };
    
    try benchmarkContinuationStealingWithCoz(allocator, continuation_config_coz);
    
    std.debug.print("\n🎯 COZ Analysis Summary:\n", .{});
    std.debug.print("========================\n", .{});
    std.debug.print("This benchmark includes detailed COZ progress points for:\n", .{});
    std.debug.print("1. Task vs Continuation submission overhead\n", .{});
    std.debug.print("2. Work scheduling and stealing latency\n", .{});
    std.debug.print("3. NUMA-aware placement efficiency\n", .{});
    std.debug.print("4. Memory access patterns and cache behavior\n", .{});
    std.debug.print("\nRun with 'coz run' to identify specific bottlenecks.\n", .{});
}

// ============================================================================
// Main COZ Profiling Entry Point
// ============================================================================

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const config = BenchmarkConfig{
        .num_workers = 4,
        .workload_size = 1000,
        .iterations = 50,
        .work_complexity = .heavy,
        .enable_numa_aware = true,
    };
    
    std.debug.print("🚀 Beat.zig Continuation Stealing COZ Profiling\n", .{});
    std.debug.print("==============================================\n\n", .{});
    
    try comparativeAnalysisWithCoz(allocator, config);
}