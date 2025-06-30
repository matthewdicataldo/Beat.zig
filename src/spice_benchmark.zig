const std = @import("std");
const spice = @import("root.zig");

const BenchmarkConfig = struct {
    benchmark: struct {
        name: []const u8,
        description: []const u8,
        tree_sizes: []const u32,
        sample_count: u32,
        warmup_runs: u32,
        timeout_seconds: u32,
    },
};

const Node = struct {
    value: i64,
    left: ?*Node = null,
    right: ?*Node = null,
};

fn createTree(allocator: std.mem.Allocator, size: usize) !*Node {
    if (size == 0) {
        return error.InvalidSize;
    }
    
    const node = try allocator.create(Node);
    node.value = @intCast(size);
    
    if (size == 1) {
        node.left = null;
        node.right = null;
        return node;
    }
    
    const left_size = (size - 1) / 2;
    const right_size = size - 1 - left_size;
    
    node.left = if (left_size > 0) try createTree(allocator, left_size) else null;
    node.right = if (right_size > 0) try createTree(allocator, right_size) else null;
    
    return node;
}

fn destroyTree(allocator: std.mem.Allocator, node: ?*Node) void {
    if (node) |n| {
        destroyTree(allocator, n.left);
        destroyTree(allocator, n.right);
        allocator.destroy(n);
    }
}

fn sequentialSum(node: ?*Node) i64 {
    if (node) |n| {
        return n.value + sequentialSum(n.left) + sequentialSum(n.right);
    }
    return 0;
}

fn spiceParallelSum(task: *spice.Task, node: ?*Node) i64 {
    if (node) |n| {
        var result = n.value;
        
        // Use task.call for potentially parallel execution
        if (n.left) |left_child| {
            result += task.call(i64, spiceParallelSum, left_child);
        }
        
        if (n.right) |right_child| {
            result += task.call(i64, spiceParallelSum, right_child);
        }
        
        return result;
    }
    return 0;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Read config from JSON file
    const config_file = try std.fs.cwd().openFile("benchmark_config.json", .{});
    defer config_file.close();
    
    const config_contents = try config_file.readToEndAlloc(allocator, 1024 * 1024);
    defer allocator.free(config_contents);
    
    const parsed = try std.json.parseFromSlice(BenchmarkConfig, allocator, config_contents, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    
    const config = parsed.value;
    
    std.debug.print("SPICE BENCHMARK RESULTS\n", .{});
    std.debug.print("=======================\n", .{});
    std.debug.print("{s:<12} {s:<12} {s:<12} {s:<12} {s:<12}\n", .{
        "Tree Size", "Seq (μs)", "Par (μs)", "Speedup", "Overhead"
    });
    std.debug.print("------------------------------------------------------------\n", .{});
    
    for (config.benchmark.tree_sizes) |size| {
        const tree = try createTree(allocator, size);
        defer destroyTree(allocator, tree);
        
        // Create thread pool
        var thread_pool = spice.ThreadPool.init(allocator);
        defer thread_pool.deinit();
        thread_pool.start(.{});
        
        // Warmup
        for (0..config.benchmark.warmup_runs) |_| {
            _ = sequentialSum(tree);
            _ = thread_pool.call(i64, spiceParallelSum, tree);
        }
        
        // Sequential timing
        const seq_times = try allocator.alloc(u64, config.benchmark.sample_count);
        defer allocator.free(seq_times);
        
        for (seq_times, 0..) |*time, i| {
            _ = i;
            const start = std.time.nanoTimestamp();
            const result = sequentialSum(tree);
            const end = std.time.nanoTimestamp();
            time.* = @intCast(end - start);
            std.mem.doNotOptimizeAway(result);
        }
        
        // Parallel timing
        const par_times = try allocator.alloc(u64, config.benchmark.sample_count);
        defer allocator.free(par_times);
        
        for (par_times, 0..) |*time, i| {
            _ = i;
            const start = std.time.nanoTimestamp();
            const result = thread_pool.call(i64, spiceParallelSum, tree);
            const end = std.time.nanoTimestamp();
            time.* = @intCast(end - start);
            std.mem.doNotOptimizeAway(result);
        }
        
        // Calculate median times
        std.mem.sort(u64, seq_times, {}, std.sort.asc(u64));
        std.mem.sort(u64, par_times, {}, std.sort.asc(u64));
        
        const seq_median_ns = seq_times[seq_times.len / 2];
        const par_median_ns = par_times[par_times.len / 2];
        
        const seq_median_us = seq_median_ns / 1_000;
        const par_median_us = par_median_ns / 1_000;
        
        const speedup = @as(f64, @floatFromInt(seq_median_ns)) / @as(f64, @floatFromInt(par_median_ns));
        const overhead_ns = if (par_median_ns > seq_median_ns) par_median_ns - seq_median_ns else 0;
        
        var overhead_buffer: [32]u8 = undefined;
        const overhead_str = if (overhead_ns < 1000) 
            "sub-ns" 
        else 
            std.fmt.bufPrint(overhead_buffer[0..], "{}ns", .{overhead_ns}) catch "overflow";
        
        std.debug.print("{d:<12} {d:<12} {d:<12} {d:<12.2} {s:<12}\n", .{
            size, seq_median_us, par_median_us, speedup, overhead_str
        });
    }
}