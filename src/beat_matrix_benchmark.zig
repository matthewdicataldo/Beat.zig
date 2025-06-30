const std = @import("std");
const beat = @import("core.zig");
const easy_api = @import("easy_api.zig");

const BenchmarkConfig = struct {
    benchmark: struct {
        name: []const u8,
        description: []const u8,
        matrix_sizes: []const u32,
        sample_count: u32,
        warmup_runs: u32,
        timeout_seconds: u32,
    },
};

// Sequential matrix multiplication
fn multiplyMatricesSequential(allocator: std.mem.Allocator, a: [][]f32, b: [][]f32, result: [][]f32, size: u32) !void {
    for (0..size) |i| {
        for (0..size) |j| {
            result[i][j] = 0.0;
            for (0..size) |k| {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    _ = allocator;
}

// Beat.zig parallel matrix multiplication
const MatrixRowTask = struct {
    a: [][]f32,
    b: [][]f32,
    result: [][]f32,
    row: u32,
    size: u32,
};

fn matrixRowWorker(data: *anyopaque) void {
    const task_data = @as(*MatrixRowTask, @ptrCast(@alignCast(data)));
    const row = task_data.row;
    
    for (0..task_data.size) |j| {
        task_data.result[row][j] = 0.0;
        for (0..task_data.size) |k| {
            task_data.result[row][j] += task_data.a[row][k] * task_data.b[k][j];
        }
    }
}

fn multiplyMatricesParallel(allocator: std.mem.Allocator, pool: *beat.ThreadPool, a: [][]f32, b: [][]f32, result: [][]f32, size: u32) !void {
    var tasks = try allocator.alloc(MatrixRowTask, size);
    defer allocator.free(tasks);
    
    var beat_tasks = try allocator.alloc(beat.Task, size);
    defer allocator.free(beat_tasks);
    
    // Create tasks for each row
    for (0..size) |i| {
        tasks[i] = MatrixRowTask{
            .a = a,
            .b = b,
            .result = result,
            .row = @intCast(i),
            .size = size,
        };
        
        beat_tasks[i] = beat.Task{
            .func = matrixRowWorker,
            .data = @ptrCast(&tasks[i]),
        };
        
        try pool.submit(beat_tasks[i]);
    }
    
    // Wait for all tasks to complete
    pool.wait();
}

fn createMatrix(allocator: std.mem.Allocator, size: u32) ![][]f32 {
    const matrix = try allocator.alloc([]f32, size);
    for (0..size) |i| {
        matrix[i] = try allocator.alloc(f32, size);
        for (0..size) |j| {
            matrix[i][j] = @as(f32, @floatFromInt(i + j)) * 0.5;
        }
    }
    return matrix;
}

fn freeMatrix(allocator: std.mem.Allocator, matrix: [][]f32) void {
    for (matrix) |row| {
        allocator.free(row);
    }
    allocator.free(matrix);
}

fn benchmarkMatrix(allocator: std.mem.Allocator, pool: *beat.ThreadPool, size: u32, sample_count: u32, warmup_runs: u32) !void {
    const a = try createMatrix(allocator, size);
    defer freeMatrix(allocator, a);
    
    const b = try createMatrix(allocator, size);
    defer freeMatrix(allocator, b);
    
    const result_seq = try createMatrix(allocator, size);
    defer freeMatrix(allocator, result_seq);
    
    const result_par = try createMatrix(allocator, size);
    defer freeMatrix(allocator, result_par);
    
    // Warmup
    for (0..warmup_runs) |_| {
        try multiplyMatricesSequential(allocator, a, b, result_seq, size);
        try multiplyMatricesParallel(allocator, pool, a, b, result_par, size);
    }
    
    // Sequential timing
    const seq_times = try allocator.alloc(u64, sample_count);
    defer allocator.free(seq_times);
    
    for (seq_times, 0..) |*time, i| {
        _ = i;
        const start = std.time.nanoTimestamp();
        try multiplyMatricesSequential(allocator, a, b, result_seq, size);
        const end = std.time.nanoTimestamp();
        time.* = @intCast(end - start);
    }
    
    // Parallel timing
    const par_times = try allocator.alloc(u64, sample_count);
    defer allocator.free(par_times);
    
    for (par_times, 0..) |*time, i| {
        _ = i;
        const start = std.time.nanoTimestamp();
        try multiplyMatricesParallel(allocator, pool, a, b, result_par, size);
        const end = std.time.nanoTimestamp();
        time.* = @intCast(end - start);
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
        size, seq_median_ms, par_median_ms, speedup, "success"
    });
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Create Beat.zig thread pool with advanced features (ISPC optimizations)
    var pool = try easy_api.createAdvancedPool(allocator, .{});
    defer pool.deinit();
    
    // Read config from JSON file
    const config_file = try std.fs.cwd().openFile("benchmark_config.json", .{});
    defer config_file.close();
    
    const config_contents = try config_file.readToEndAlloc(allocator, 1024 * 1024);
    defer allocator.free(config_contents);
    
    const parsed = try std.json.parseFromSlice(BenchmarkConfig, allocator, config_contents, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    
    const benchmark_config = parsed.value;
    
    std.debug.print("BEAT.ZIG MATRIX BENCHMARK RESULTS\n", .{});
    std.debug.print("=================================\n", .{});
    std.debug.print("{s:<8} {s:<12} {s:<12} {s:<12} {s:<12}\n", .{
        "Size", "Seq (ms)", "Par (ms)", "Speedup", "Status"
    });
    std.debug.print("------------------------------------------------------------\n", .{});
    
    for (benchmark_config.benchmark.matrix_sizes) |size| {
        try benchmarkMatrix(allocator, pool, size, benchmark_config.benchmark.sample_count, benchmark_config.benchmark.warmup_runs);
    }
    
    std.debug.print("\nNOTES:\n", .{});
    std.debug.print("• Beat.zig: Work-stealing with SIMD and NUMA optimizations\n", .{});
    std.debug.print("• Row-wise task parallelization with advanced worker selection\n", .{});
    std.debug.print("• Sample count: {}, Warmup runs: {}\n", .{ benchmark_config.benchmark.sample_count, benchmark_config.benchmark.warmup_runs });
}