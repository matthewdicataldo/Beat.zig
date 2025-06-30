const std = @import("std");
const beat = @import("core.zig");
const easy_api = @import("easy_api.zig");

const BenchmarkConfig = struct {
    benchmark: struct {
        name: []const u8,
        description: []const u8,
        fibonacci_numbers: []const u32,
        sample_count: u32,
        warmup_runs: u32,
        timeout_seconds: u32,
    },
};

// Sequential Fibonacci implementation
fn fibSequential(n: u32) u64 {
    if (n <= 1) return n;
    return fibSequential(n - 1) + fibSequential(n - 2);
}

// Beat.zig parallel Fibonacci using work-stealing
const FibData = struct {
    n: u32,
    result: u64 = 0,
};

fn fibWorker(data: *anyopaque) void {
    const fib_data = @as(*FibData, @ptrCast(@alignCast(data)));
    fib_data.result = fibSequential(fib_data.n);
}

fn fibBeatParallel(pool: *beat.ThreadPool, allocator: std.mem.Allocator, n: u32) !u64 {
    _ = allocator; // Not used in this implementation
    if (n <= 1) return n;
    if (n <= 30) return fibSequential(n); // Avoid task overhead for small values
    
    var left_data = FibData{ .n = n - 1 };
    var right_data = FibData{ .n = n - 2 };
    
    const left_task = beat.Task{
        .func = fibWorker,
        .data = @ptrCast(&left_data),
    };
    
    const right_task = beat.Task{
        .func = fibWorker,
        .data = @ptrCast(&right_data),
    };
    
    try pool.submit(left_task);
    try pool.submit(right_task);
    
    // Wait for completion
    pool.wait();
    
    return left_data.result + right_data.result;
}

fn benchmarkFibonacci(allocator: std.mem.Allocator, pool: *beat.ThreadPool, fib_num: u32, sample_count: u32, warmup_runs: u32) !void {
    // Warmup
    for (0..warmup_runs) |_| {
        _ = fibSequential(fib_num);
        _ = try fibBeatParallel(pool, allocator, fib_num);
    }
    
    // Sequential timing
    const seq_times = try allocator.alloc(u64, sample_count);
    defer allocator.free(seq_times);
    
    for (seq_times, 0..) |*time, i| {
        _ = i;
        const start = std.time.nanoTimestamp();
        const result = fibSequential(fib_num);
        const end = std.time.nanoTimestamp();
        time.* = @intCast(end - start);
        std.mem.doNotOptimizeAway(result);
    }
    
    // Parallel timing
    const par_times = try allocator.alloc(u64, sample_count);
    defer allocator.free(par_times);
    
    for (par_times, 0..) |*time, i| {
        _ = i;
        const start = std.time.nanoTimestamp();
        const result = try fibBeatParallel(pool, allocator, fib_num);
        const end = std.time.nanoTimestamp();
        time.* = @intCast(end - start);
        std.mem.doNotOptimizeAway(result);
    }
    
    // Calculate median times
    std.mem.sort(u64, seq_times, {}, std.sort.asc(u64));
    std.mem.sort(u64, par_times, {}, std.sort.asc(u64));
    
    const seq_median_ns = seq_times[seq_times.len / 2];
    const par_median_ns = par_times[par_times.len / 2];
    
    const seq_median_ms = seq_median_ns / 1_000_000;
    const par_median_ms = par_median_ns / 1_000_000;
    
    const speedup = @as(f64, @floatFromInt(seq_median_ns)) / @as(f64, @floatFromInt(par_median_ns));
    
    std.debug.print("{d:<8} {d:<12} {d:<12} {d:<12.2} {s:<12}\n", .{
        fib_num, seq_median_ms, par_median_ms, speedup, "success"
    });
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Create Beat.zig thread pool with advanced features
    var pool = try easy_api.createAdvancedPool(allocator, .{});
    defer pool.deinit();
    
    // Read config from JSON file
    const config_file = try std.fs.cwd().openFile("benchmark_config.json", .{});
    defer config_file.close();
    
    const config_contents = try config_file.readToEndAlloc(allocator, 1024 * 1024);
    defer allocator.free(config_contents);
    
    const parsed = try std.json.parseFromSlice(BenchmarkConfig, allocator, config_contents, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    
    const config = parsed.value;
    
    std.debug.print("BEAT.ZIG FIBONACCI BENCHMARK RESULTS\n", .{});
    std.debug.print("====================================\n", .{});
    std.debug.print("{s:<8} {s:<12} {s:<12} {s:<12} {s:<12}\n", .{
        "Fib(n)", "Seq (ms)", "Par (ms)", "Speedup", "Status"
    });
    std.debug.print("------------------------------------------------------------\n", .{});
    
    for (config.benchmark.fibonacci_numbers) |fib_num| {
        try benchmarkFibonacci(allocator, pool, fib_num, config.benchmark.sample_count, config.benchmark.warmup_runs);
    }
    
    std.debug.print("\nNOTES:\n", .{});
    std.debug.print("• Beat.zig: Work-stealing with SIMD and NUMA optimizations\n", .{});
    std.debug.print("• Task parallelization with advanced worker selection\n", .{});
    std.debug.print("• Sample count: {}, Warmup runs: {}\n", .{ config.benchmark.sample_count, config.benchmark.warmup_runs });
}