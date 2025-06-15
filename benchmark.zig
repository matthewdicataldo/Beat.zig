const std = @import("std");
const builtin = @import("builtin");
const beat = @import("src/core.zig");

// Root-level benchmark wrapper that imports the comprehensive benchmarks
// from the benchmarks/ directory and adapts them to use the current API

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
        start_time: i128,
        
        const Self = @This();
        
        pub fn start() Self {
            return .{ .start_time = std.time.nanoTimestamp() };
        }
        
        pub fn lap(self: *Self) u64 {
            const now = std.time.nanoTimestamp();
            const lap_time = @as(u64, @intCast(now - self.start_time));
            self.start_time = now;
            return lap_time;
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
    
    const pool = try beat.createOptimalPool(allocator);
    defer pool.deinit();
    
    const noop_task = struct {
        fn run(data: *anyopaque) void {
            _ = data;
        }
    }.run;
    
    var dummy_data: u8 = 0;
    
    // Warmup
    for (0..config.warmup_iterations) |_| {
        try pool.submit(.{ .func = noop_task, .data = &dummy_data });
    }
    pool.wait();
    
    // Measure
    var timer = Timer().start();
    for (0..config.num_tasks) |_| {
        try pool.submit(.{ .func = noop_task, .data = &dummy_data });
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
// Benchmark 2: Beat vs std.Thread
// =============================================================================

fn benchmarkVsStdThread(allocator: std.mem.Allocator, config: BenchmarkConfig) !void {
    std.debug.print("\n=== Benchmark: Beat vs std.Thread ===\n", .{});
    
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
    
    // Benchmark Beat
    const pool = try beat.createOptimalPool(allocator);
    defer pool.deinit();
    
    counter.store(0, .release);
    
    for (0..config.num_tasks) |_| {
        try pool.submit(.{ .func = increment, .data = &counter });
    }
    
    pool.wait();
    const beat_time = timer.elapsed();
    std.debug.assert(counter.load(.acquire) == @as(i64, @intCast(config.num_tasks)));
    
    const speedup = @as(f64, @floatFromInt(std_time)) / @as(f64, @floatFromInt(beat_time));
    
    std.debug.print("  Tasks: {}\n", .{config.num_tasks});
    std.debug.print("  std.Thread: {}μs ({}ns per task)\n", .{ 
        std_time / 1000, 
        std_time / config.num_tasks 
    });
    std.debug.print("  Beat: {}μs ({}ns per task)\n", .{ 
        beat_time / 1000, 
        beat_time / config.num_tasks 
    });
    std.debug.print("  Speedup: {d:.2}x\n", .{speedup});
}

// =============================================================================
// Benchmark 3: Work Stealing Efficiency
// =============================================================================

fn benchmarkWorkStealing(allocator: std.mem.Allocator, config: BenchmarkConfig) !void {
    std.debug.print("\n=== Benchmark: Work Stealing Efficiency ===\n", .{});
    
    const pool_config = beat.Config{
        .num_workers = 4,
        .enable_statistics = true,
    };
    const pool = try beat.createPoolWithConfig(allocator, pool_config);
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
    
    // Submit all tasks to force stealing
    for (0..config.num_tasks) |_| {
        try pool.submit(.{ .func = work_task, .data = &iterations });
    }
    
    pool.wait();
    const elapsed = timer.elapsed();
    
    std.debug.print("  Tasks: {}\n", .{config.num_tasks});
    std.debug.print("  Total time: {}ms\n", .{elapsed / 1_000_000});
    std.debug.print("  Tasks per second: {d:.0}\n", .{
        @as(f64, @floatFromInt(config.num_tasks)) * 1e9 / @as(f64, @floatFromInt(elapsed))
    });
}

// =============================================================================
// Benchmark 4: Scaling Test
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
        const pool_config = beat.Config{
            .num_workers = worker_count,
        };
        const pool = try beat.createPoolWithConfig(allocator, pool_config);
        defer pool.deinit();
        
        var timer = Timer().start();
        
        for (0..config.num_tasks) |_| {
            try pool.submit(.{ .func = work_task, .data = &iterations });
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
// Benchmark 5: Comptime Work Distribution
// =============================================================================

fn benchmarkComptimeWork(allocator: std.mem.Allocator, config: BenchmarkConfig) !void {
    std.debug.print("\n=== Benchmark: Comptime Work Distribution ===\n", .{});
    
    const pool = try beat.createOptimalPool(allocator);
    defer pool.deinit();
    
    // Test different work sizes to show strategy selection
    const work_sizes = [_]comptime_int{ 100, 1000, 10000, 100000 };
    
    inline for (work_sizes) |size| {
        const analysis = beat.comptime_work.analyzeWork(f32, size, 4);
        std.debug.print("  Work size {}: {s} strategy, chunk size {}\n", .{
            size, @tagName(analysis.strategy), analysis.chunk_size
        });
    }
    
    // Benchmark parallel map performance
    const data_size = config.num_tasks;
    const input = try allocator.alloc(f32, data_size);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, data_size);
    defer allocator.free(output);
    
    // Initialize data
    for (input, 0..) |*item, i| {
        item.* = @as(f32, @floatFromInt(i));
    }
    
    const square = struct {
        fn apply(x: f32) f32 {
            return x * x;
        }
    }.apply;
    
    var timer = Timer().start();
    try beat.comptime_work.parallelMap(f32, f32, pool, input, output, square);
    const parallel_time = timer.elapsed();
    
    std.debug.print("  Parallel map of {} elements: {}μs\n", .{
        data_size, parallel_time / 1000
    });
}

// =============================================================================
// Main Benchmark Runner
// =============================================================================

pub fn main() !void {
    std.debug.print("=== Beat.zig Benchmarks ===\n", .{});
    std.debug.print("Build mode: {}\n", .{builtin.mode});
    std.debug.print("Auto-configuration: {} workers, {} SIMD width\n", .{
        beat.build_opts.hardware.optimal_workers,
        beat.build_opts.cpu_features.simd_width,
    });
    
    const allocator = std.heap.page_allocator;
    const config = BenchmarkConfig{
        .iterations = 10000,
        .warmup_iterations = 100,
        .num_tasks = 1000,
        .verbose = false,
    };
    
    // Parse command line arguments
    const args_list = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args_list);
    
    var run_all = true;
    var selected_benchmark: ?[]const u8 = null;
    
    if (args_list.len > 1) {
        run_all = false;
        selected_benchmark = args_list[1];
    }
    
    // Run selected benchmarks
    if (selected_benchmark) |bench_name| {
        std.debug.print("Running benchmark: {s}\n", .{bench_name});
        
        if (std.mem.eql(u8, bench_name, "submission")) {
            benchmarkTaskSubmission(allocator, config) catch |err| {
                std.debug.print("Error in task submission benchmark: {}\n", .{err});
            };
        } else if (std.mem.eql(u8, bench_name, "vs-std")) {
            benchmarkVsStdThread(allocator, config) catch |err| {
                std.debug.print("Error in vs-std benchmark: {}\n", .{err});
            };
        } else if (std.mem.eql(u8, bench_name, "stealing")) {
            benchmarkWorkStealing(allocator, config) catch |err| {
                std.debug.print("Error in work stealing benchmark: {}\n", .{err});
            };
        } else if (std.mem.eql(u8, bench_name, "scaling")) {
            benchmarkScaling(allocator, config) catch |err| {
                std.debug.print("Error in scaling benchmark: {}\n", .{err});
            };
        } else if (std.mem.eql(u8, bench_name, "comptime")) {
            benchmarkComptimeWork(allocator, config) catch |err| {
                std.debug.print("Error in comptime work benchmark: {}\n", .{err});
            };
        } else {
            std.debug.print("Unknown benchmark: {s}\n", .{bench_name});
        }
    } else {
        // Run all benchmarks
        benchmarkTaskSubmission(allocator, config) catch |err| {
            std.debug.print("Error in task submission benchmark: {}\n", .{err});
        };
        
        benchmarkVsStdThread(allocator, config) catch |err| {
            std.debug.print("Error in vs-std benchmark: {}\n", .{err});
        };
        
        benchmarkWorkStealing(allocator, config) catch |err| {
            std.debug.print("Error in work stealing benchmark: {}\n", .{err});
        };
        
        benchmarkScaling(allocator, config) catch |err| {
            std.debug.print("Error in scaling benchmark: {}\n", .{err});
        };
        
        benchmarkComptimeWork(allocator, config) catch |err| {
            std.debug.print("Error in comptime work benchmark: {}\n", .{err});
        };
    }
    
    std.debug.print("\n=== Benchmarks Complete ===\n", .{});
    
    if (!run_all and selected_benchmark != null) {
        std.debug.print("\nRun with specific benchmark name or no arguments to execute all\n", .{});
        std.debug.print("Available: submission, vs-std, stealing, scaling, comptime\n", .{});
    }
}