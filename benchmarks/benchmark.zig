const std = @import("std");
const builtin = @import("builtin");
const zigpulse = @import("zigpulse.zig");

// Comprehensive benchmarks for ZigPulse

// =============================================================================
// Benchmark Configuration
// =============================================================================

const BenchmarkConfig = struct {
    iterations: u32 = 1000,
    warmup_iterations: u32 = 100,
    num_tasks: u32 = 1000,
    verbose: bool = false,
};

// =============================================================================
// Timing Utilities
// =============================================================================

fn Timer() type {
    return struct {
        start_time: i64,
        
        const Self = @This();
        
        pub fn start() Self {
            return .{ .start_time = std.time.nanoTimestamp() };
        }
        
        pub fn lap(self: *Self) u64 {
            const now = std.time.nanoTimestamp();
            const elapsed = @as(u64, @intCast(now - self.start_time));
            self.start_time = now;
            return elapsed;
        }
        
        pub fn elapsed(self: Self) u64 {
            return @as(u64, @intCast(std.time.nanoTimestamp() - self.start_time));
        }
    };
}

// =============================================================================
// Benchmark 1: Task Submission Overhead
// =============================================================================

fn benchmarkTaskSubmission(allocator: std.mem.Allocator, config: BenchmarkConfig) !void {
    std.debug.print("\n=== Benchmark: Task Submission Overhead ===\n", .{});
    
    const pool = try zigpulse.createPool(allocator);
    defer pool.deinit();
    
    const noop_task = struct {
        fn run(data: *anyopaque) void {
            _ = data;
        }
    }.run;
    
    // Warmup
    for (0..config.warmup_iterations) |_| {
        try pool.submit(noop_task, undefined);
    }
    pool.wait();
    
    // Measure
    var timer = Timer().start();
    for (0..config.num_tasks) |_| {
        try pool.submit(noop_task, undefined);
    }
    const submit_time = timer.elapsed();
    
    pool.wait();
    const total_time = timer.elapsed();
    
    const avg_submit_ns = submit_time / config.num_tasks;
    const avg_total_ns = total_time / config.num_tasks;
    
    std.debug.print("  Tasks submitted: {}\n", .{config.num_tasks});
    std.debug.print("  Average submission time: {}ns\n", .{avg_submit_ns});
    std.debug.print("  Average total time: {}ns\n", .{avg_total_ns});
}

// =============================================================================
// Benchmark 2: ZigPulse vs std.Thread
// =============================================================================

fn benchmarkVsStdThread(allocator: std.mem.Allocator, config: BenchmarkConfig) !void {
    std.debug.print("\n=== Benchmark: ZigPulse vs std.Thread ===\n", .{});
    
    var counter = std.atomic.Value(i64).init(0);
    
    const increment = struct {
        fn run(data: *anyopaque) void {
            const cnt = @as(*std.atomic.Value(i64), @ptrCast(@alignCast(data)));
            _ = cnt.fetchAdd(1, .monotonic);
        }
    }.run;
    
    // Benchmark std.Thread
    counter.store(0, .release);
    var timer = Timer().start();
    
    const threads = try allocator.alloc(std.Thread, config.num_tasks);
    defer allocator.free(threads);
    
    for (threads) |*thread| {
        thread.* = try std.Thread.spawn(.{}, struct {
            fn threadRun(cnt: *std.atomic.Value(i64)) void {
                _ = cnt.fetchAdd(1, .monotonic);
            }
        }.threadRun, .{&counter});
    }
    
    for (threads) |thread| {
        thread.join();
    }
    
    const std_time = timer.lap();
    std.debug.assert(counter.load(.acquire) == @as(i64, @intCast(config.num_tasks)));
    
    // Benchmark ZigPulse
    const pool = try zigpulse.createPool(allocator);
    defer pool.deinit();
    
    counter.store(0, .release);
    
    for (0..config.num_tasks) |_| {
        try pool.submit(increment, &counter);
    }
    
    pool.wait();
    const zp_time = timer.elapsed();
    std.debug.assert(counter.load(.acquire) == @as(i64, @intCast(config.num_tasks)));
    
    const speedup = @as(f64, @floatFromInt(std_time)) / @as(f64, @floatFromInt(zp_time));
    
    std.debug.print("  Tasks: {}\n", .{config.num_tasks});
    std.debug.print("  std.Thread: {}μs ({}ns per task)\n", .{ 
        std_time / 1000, 
        std_time / config.num_tasks 
    });
    std.debug.print("  ZigPulse: {}μs ({}ns per task)\n", .{ 
        zp_time / 1000, 
        zp_time / config.num_tasks 
    });
    std.debug.print("  Speedup: {d:.2}x\n", .{speedup});
}

// =============================================================================
// Benchmark 3: Work Stealing Efficiency
// =============================================================================

fn benchmarkWorkStealing(allocator: std.mem.Allocator, config: BenchmarkConfig) !void {
    std.debug.print("\n=== Benchmark: Work Stealing Efficiency ===\n", .{});
    
    const pool = try zigpulse.createPoolWithConfig(allocator, .{
        .num_workers = 4,
        .enable_statistics = true,
    });
    defer pool.deinit();
    
    const work_task = struct {
        fn run(data: *anyopaque) void {
            const iterations = @as(*u32, @ptrCast(@alignCast(data))).*;
            var sum: u64 = 0;
            for (0..iterations) |i| {
                sum += i;
            }
            std.mem.doNotOptimizeAway(&sum);
        }
    }.run;
    
    var iterations: u32 = 1000;
    var timer = Timer().start();
    
    // Submit all tasks to worker 0 to force stealing
    for (0..config.num_tasks) |_| {
        try pool.submit(work_task, @as(*anyopaque, @ptrCast(&iterations)));
    }
    
    pool.wait();
    const elapsed = timer.elapsed();
    
    const stats = pool.getStats();
    const steals = stats.tasks_stolen.load(.acquire);
    const steal_rate = @as(f64, @floatFromInt(steals)) / @as(f64, @floatFromInt(config.num_tasks)) * 100.0;
    
    std.debug.print("  Tasks: {}, Stolen: {} ({d:.1}%)\n", .{ config.num_tasks, steals, steal_rate });
    std.debug.print("  Total time: {}ms\n", .{elapsed / 1_000_000});
    std.debug.print("  Tasks per second: {d:.0}\n", .{
        @as(f64, @floatFromInt(config.num_tasks)) * 1e9 / @as(f64, @floatFromInt(elapsed))
    });
}

// =============================================================================
// Benchmark 4: pcall Overhead (V2 features)
// =============================================================================

fn benchmarkPcallOverhead(allocator: std.mem.Allocator, config: BenchmarkConfig) !void {
    std.debug.print("\n=== Benchmark: pcall Overhead ===\n", .{});
    
    const pool = try zigpulse.createPoolWithConfig(allocator, .{
        .enable_heartbeat = true,
        .use_fast_rdtsc = true,
    });
    defer pool.deinit();
    
    zigpulse.initThread(pool);
    
    // Baseline: direct function call
    const compute = struct {
        fn run() i32 {
            var sum: i32 = 0;
            for (0..100) |i| {
                sum += @as(i32, @intCast(i));
            }
            return sum;
        }
    }.run;
    
    // Measure baseline
    var timer = Timer().start();
    var baseline_sum: i64 = 0;
    for (0..config.iterations) |_| {
        baseline_sum += compute();
    }
    const baseline_time = timer.lap();
    
    // Measure pcall
    var pcall_sum: i64 = 0;
    for (0..config.iterations) |_| {
        var future = zigpulse.pcall(i32, compute);
        pcall_sum += try future.get();
    }
    const pcall_time = timer.lap();
    
    // Measure pcallMinimal
    var minimal_sum: i64 = 0;
    for (0..config.iterations) |_| {
        minimal_sum += zigpulse.pcallMinimal(i32, compute);
    }
    const minimal_time = timer.elapsed();
    
    std.mem.doNotOptimizeAway(&baseline_sum);
    std.mem.doNotOptimizeAway(&pcall_sum);
    std.mem.doNotOptimizeAway(&minimal_sum);
    
    const baseline_ns = baseline_time / config.iterations;
    const pcall_ns = pcall_time / config.iterations;
    const minimal_ns = minimal_time / config.iterations;
    const pcall_overhead = @as(i64, @intCast(pcall_ns)) - @as(i64, @intCast(baseline_ns));
    const minimal_overhead = @as(i64, @intCast(minimal_ns)) - @as(i64, @intCast(baseline_ns));
    
    std.debug.print("  Iterations: {}\n", .{config.iterations});
    std.debug.print("  Baseline: {}ns per call\n", .{baseline_ns});
    std.debug.print("  pcall: {}ns per call (overhead: {}ns)\n", .{ pcall_ns, pcall_overhead });
    std.debug.print("  pcallMinimal: {}ns per call (overhead: {}ns)\n", .{ minimal_ns, minimal_overhead });
    
    if (builtin.mode == .ReleaseFast) {
        if (minimal_overhead == 0) {
            std.debug.print("  ✅ Zero overhead achieved in release mode!\n", .{});
        }
    }
}

// =============================================================================
// Benchmark 5: Fork-Join Performance
// =============================================================================

fn benchmarkForkJoin(allocator: std.mem.Allocator, config: BenchmarkConfig) !void {
    std.debug.print("\n=== Benchmark: Fork-Join Performance ===\n", .{});
    
    const pool = try zigpulse.createPool(allocator);
    defer pool.deinit();
    
    const computeA = struct {
        fn run() !i32 {
            var sum: i32 = 0;
            for (0..1000) |i| {
                sum +%= @as(i32, @intCast(i));
            }
            return sum;
        }
    }.run;
    
    const computeB = struct {
        fn run() !i32 {
            var sum: i32 = 0;
            for (0..1000) |i| {
                sum +%= @as(i32, @intCast(i * 2));
            }
            return sum;
        }
    }.run;
    
    // Warmup
    for (0..config.warmup_iterations) |_| {
        const result = try zigpulse.join2(pool, i32, i32, computeA, computeB);
        std.mem.doNotOptimizeAway(result);
    }
    
    // Measure
    var timer = Timer().start();
    var total: i64 = 0;
    
    for (0..config.iterations) |_| {
        const result = try zigpulse.join2(pool, i32, i32, computeA, computeB);
        total += result.left + result.right;
    }
    
    const elapsed = timer.elapsed();
    std.mem.doNotOptimizeAway(&total);
    
    const per_join_ns = elapsed / config.iterations;
    
    std.debug.print("  Iterations: {}\n", .{config.iterations});
    std.debug.print("  Total time: {}ms\n", .{elapsed / 1_000_000});
    std.debug.print("  Per join: {}ns\n", .{per_join_ns});
    std.debug.print("  Join operations/sec: {d:.0}\n", .{
        1e9 / @as(f64, @floatFromInt(per_join_ns))
    });
}

// =============================================================================
// Benchmark 6: Parallel Patterns
// =============================================================================

fn benchmarkParallelPatterns(allocator: std.mem.Allocator, config: BenchmarkConfig) !void {
    std.debug.print("\n=== Benchmark: Parallel Patterns ===\n", .{});
    
    const pool = try zigpulse.createPool(allocator);
    defer pool.deinit();
    
    // Benchmark parallelFor
    const array_size = 10000;
    var numbers: [array_size]i32 = undefined;
    for (&numbers, 0..) |*n, i| {
        n.* = @intCast(i);
    }
    
    const double_fn = struct {
        fn run(idx: usize, item: *i32) void {
            _ = idx;
            item.* *= 2;
        }
    }.run;
    
    var timer = Timer().start();
    try pool.parallelFor(i32, &numbers, double_fn);
    const parallel_time = timer.lap();
    
    // Compare with sequential
    for (&numbers) |*n| {
        n.* *= 2;
    }
    const sequential_time = timer.elapsed();
    
    const speedup = @as(f64, @floatFromInt(sequential_time)) / @as(f64, @floatFromInt(parallel_time));
    
    std.debug.print("  Array size: {}\n", .{array_size});
    std.debug.print("  Parallel time: {}μs\n", .{parallel_time / 1000});
    std.debug.print("  Sequential time: {}μs\n", .{sequential_time / 1000});
    std.debug.print("  Speedup: {d:.2}x\n", .{speedup});
}

// =============================================================================
// Benchmark 7: Scaling Test
// =============================================================================

fn benchmarkScaling(allocator: std.mem.Allocator, config: BenchmarkConfig) !void {
    std.debug.print("\n=== Benchmark: Scaling Test ===\n", .{});
    
    const work_task = struct {
        fn run(data: *anyopaque) void {
            const iterations = @as(*u32, @ptrCast(@alignCast(data))).*;
            var sum: u64 = 0;
            for (0..iterations) |i| {
                sum += i * i;
            }
            std.mem.doNotOptimizeAway(&sum);
        }
    }.run;
    
    const cpu_count = std.Thread.getCpuCount() catch 4;
    const max_workers = @min(cpu_count, 8);
    var iterations: u32 = 10000;
    
    std.debug.print("  CPU count: {}\n", .{cpu_count});
    
    var worker_count: usize = 1;
    while (worker_count <= max_workers) : (worker_count *= 2) {
        const pool = try zigpulse.createPoolWithConfig(allocator, .{
            .num_workers = worker_count,
        });
        defer pool.deinit();
        
        var timer = Timer().start();
        
        for (0..config.num_tasks) |_| {
            try pool.submit(work_task, @as(*anyopaque, @ptrCast(&iterations)));
        }
        
        pool.wait();
        const elapsed = timer.elapsed();
        
        const tasks_per_sec = @as(f64, @floatFromInt(config.num_tasks)) * 1e9 / @as(f64, @floatFromInt(elapsed));
        
        std.debug.print("  Workers: {}, Time: {}ms, Tasks/sec: {d:.0}\n", .{
            worker_count,
            elapsed / 1_000_000,
            tasks_per_sec,
        });
    }
}

// =============================================================================
// Main Benchmark Runner
// =============================================================================

pub fn main() !void {
    std.debug.print("=== ZigPulse Benchmarks ===\n", .{});
    std.debug.print("Build mode: {}\n", .{builtin.mode});
    
    const allocator = std.heap.page_allocator;
    const config = BenchmarkConfig{
        .iterations = 10000,
        .warmup_iterations = 100,
        .num_tasks = 1000,
        .verbose = false,
    };
    
    // Parse command line arguments
    var args = std.process.args();
    _ = args.next(); // Skip program name
    
    var run_all = true;
    var selected_benchmark: ?[]const u8 = null;
    
    if (args.next()) |arg| {
        run_all = false;
        selected_benchmark = arg;
    }
    
    // Run benchmarks
    if (run_all or (selected_benchmark != null and std.mem.eql(u8, selected_benchmark.?, "submission"))) {
        try benchmarkTaskSubmission(allocator, config);
    }
    
    if (run_all or (selected_benchmark != null and std.mem.eql(u8, selected_benchmark.?, "vs-std"))) {
        try benchmarkVsStdThread(allocator, config);
    }
    
    if (run_all or (selected_benchmark != null and std.mem.eql(u8, selected_benchmark.?, "stealing"))) {
        try benchmarkWorkStealing(allocator, config);
    }
    
    if (run_all or (selected_benchmark != null and std.mem.eql(u8, selected_benchmark.?, "pcall"))) {
        try benchmarkPcallOverhead(allocator, config);
    }
    
    if (run_all or (selected_benchmark != null and std.mem.eql(u8, selected_benchmark.?, "fork-join"))) {
        try benchmarkForkJoin(allocator, config);
    }
    
    if (run_all or (selected_benchmark != null and std.mem.eql(u8, selected_benchmark.?, "patterns"))) {
        try benchmarkParallelPatterns(allocator, config);
    }
    
    if (run_all or (selected_benchmark != null and std.mem.eql(u8, selected_benchmark.?, "scaling"))) {
        try benchmarkScaling(allocator, config);
    }
    
    std.debug.print("\n=== Benchmarks Complete ===\n", .{});
    
    if (!run_all and selected_benchmark != null) {
        std.debug.print("\nRun with no arguments to execute all benchmarks\n", .{});
    }
}