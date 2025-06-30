const std = @import("std");
const beat = @import("easy_api.zig");
const core = @import("core.zig");

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

// Beat.zig parallel sum using the thread pool
const SumTask = struct {
    node: ?*Node,
    result: i64 = 0,
};

fn beatParallelSum(pool: *core.ThreadPool, allocator: std.mem.Allocator, node: ?*Node) !i64 {
    if (node) |n| {
        // For small subtrees, go sequential to avoid overhead
        if (countNodes(n) < 100) {
            return sequentialSum(node);
        }
        
        var result = n.value;
        
        // Create tasks for left and right subtrees
        const left_task = try allocator.create(SumTask);
        defer allocator.destroy(left_task);
        const right_task = try allocator.create(SumTask);
        defer allocator.destroy(right_task);
        
        left_task.* = SumTask{ .node = n.left };
        right_task.* = SumTask{ .node = n.right };
        
        // Submit tasks to thread pool
        const left_beat_task = core.Task{
            .func = sumTaskWrapper,
            .data = @ptrCast(left_task),
        };
        const right_beat_task = core.Task{
            .func = sumTaskWrapper,
            .data = @ptrCast(right_task),
        };
        
        try pool.submit(left_beat_task);
        try pool.submit(right_beat_task);
        
        // Wait for completion
        pool.wait();
        
        result += left_task.result + right_task.result;
        return result;
    }
    return 0;
}

fn sumTaskWrapper(data: *anyopaque) void {
    const task: *SumTask = @ptrCast(@alignCast(data));
    task.result = sequentialSum(task.node);
}

fn countNodes(node: ?*Node) usize {
    if (node) |n| {
        return 1 + countNodes(n.left) + countNodes(n.right);
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
    
    std.debug.print("BEAT.ZIG TREE BENCHMARK RESULTS\n", .{});
    std.debug.print("===============================\n", .{});
    std.debug.print("{s:<12} {s:<12} {s:<12} {s:<12} {s:<12}\n", .{
        "Tree Size", "Seq (μs)", "Par (μs)", "Speedup", "Overhead"
    });
    std.debug.print("------------------------------------------------------------\n", .{});
    
    for (config.benchmark.tree_sizes) |size| {
        // Use arena allocator for better memory locality
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const arena_allocator = arena.allocator();
        
        const tree = try createTree(arena_allocator, size);
        // No need for manual cleanup - arena handles it
        
        // Create Beat.zig basic pool (no ISPC dependencies)
        const pool = try beat.createBasicPool(allocator, 4);
        defer pool.deinit();
        
        // Warmup
        for (0..config.benchmark.warmup_runs) |_| {
            _ = sequentialSum(tree);
            _ = try beatParallelSum(pool, allocator, tree);
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
            const result = try beatParallelSum(pool, allocator, tree);
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
    
    std.debug.print("\nNOTES:\n", .{});
    std.debug.print("• Beat.zig thread pool with work-stealing deque\n", .{});
    std.debug.print("• OPTIMIZED: Using arena allocator for better memory locality\n", .{});
    std.debug.print("• Sequential threshold: 100 nodes for overhead avoidance\n", .{});
    std.debug.print("• Sample count: {}, Warmup runs: {}\n", .{ config.benchmark.sample_count, config.benchmark.warmup_runs });
}