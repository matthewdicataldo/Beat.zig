const std = @import("std");

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

// Thread context for manual parallelization
const ThreadContext = struct {
    node: ?*Node,
    result: i64 = 0,
};

fn threadWorker(context: *ThreadContext) void {
    context.result = sequentialSum(context.node);
}

fn stdThreadParallelSum(_: std.mem.Allocator, node: ?*Node) !i64 {
    if (node) |n| {
        var result = n.value;
        
        // Create contexts for left and right subtrees
        var left_context = ThreadContext{ .node = n.left };
        var right_context = ThreadContext{ .node = n.right };
        
        // Spawn threads for each subtree (if they exist)
        var left_thread: ?std.Thread = null;
        var right_thread: ?std.Thread = null;
        
        if (n.left != null) {
            left_thread = try std.Thread.spawn(.{}, threadWorker, .{&left_context});
        }
        
        if (n.right != null) {
            right_thread = try std.Thread.spawn(.{}, threadWorker, .{&right_context});
        }
        
        // Join threads and collect results
        if (left_thread) |thread| {
            thread.join();
            result += left_context.result;
        }
        
        if (right_thread) |thread| {
            thread.join();
            result += right_context.result;
        }
        
        return result;
    }
    return 0;
}

// Bounded parallel sum with limited thread depth to prevent explosion
fn boundedParallelSum(_: std.mem.Allocator, node: ?*Node, max_depth: u32) !i64 {
    if (max_depth == 0 or node == null) {
        return sequentialSum(node);
    }
    
    if (node) |n| {
        var result = n.value;
        
        var left_context = ThreadContext{ .node = n.left };
        var right_context = ThreadContext{ .node = n.right };
        
        var left_thread: ?std.Thread = null;
        var right_thread: ?std.Thread = null;
        
        // Only spawn threads for significant subtrees
        if (n.left != null and max_depth > 1) {
            left_thread = try std.Thread.spawn(.{}, threadWorker, .{&left_context});
        } else {
            left_context.result = sequentialSum(n.left);
        }
        
        if (n.right != null and max_depth > 1) {
            right_thread = try std.Thread.spawn(.{}, threadWorker, .{&right_context});
        } else {
            right_context.result = sequentialSum(n.right);
        }
        
        // Join threads
        if (left_thread) |thread| {
            thread.join();
        }
        
        if (right_thread) |thread| {
            thread.join();
        }
        
        result += left_context.result + right_context.result;
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
    
    std.debug.print("ZIG STD.THREAD BASELINE RESULTS\n", .{});
    std.debug.print("===============================\n", .{});
    std.debug.print("{s:<12} {s:<12} {s:<12} {s:<12} {s:<12} {s:<12}\n", .{
        "Tree Size", "Seq (μs)", "Raw (μs)", "Bounded (μs)", "Raw Speedup", "Bounded Speedup"
    });
    std.debug.print("--------------------------------------------------------------------------------\n", .{});
    
    for (config.benchmark.tree_sizes) |size| {
        const tree = try createTree(allocator, size);
        defer destroyTree(allocator, tree);
        
        // Warmup
        for (0..config.benchmark.warmup_runs) |_| {
            _ = sequentialSum(tree);
            _ = try stdThreadParallelSum(allocator, tree);
            _ = try boundedParallelSum(allocator, tree, 3); // 3 levels of parallelism
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
        
        // Raw parallel timing (can create many threads)
        const raw_par_times = try allocator.alloc(u64, config.benchmark.sample_count);
        defer allocator.free(raw_par_times);
        
        for (raw_par_times, 0..) |*time, i| {
            _ = i;
            const start = std.time.nanoTimestamp();
            const result = try stdThreadParallelSum(allocator, tree);
            const end = std.time.nanoTimestamp();
            time.* = @intCast(end - start);
            std.mem.doNotOptimizeAway(result);
        }
        
        // Bounded parallel timing (limited thread depth)
        const bounded_par_times = try allocator.alloc(u64, config.benchmark.sample_count);
        defer allocator.free(bounded_par_times);
        
        for (bounded_par_times, 0..) |*time, i| {
            _ = i;
            const start = std.time.nanoTimestamp();
            const result = try boundedParallelSum(allocator, tree, 3);
            const end = std.time.nanoTimestamp();
            time.* = @intCast(end - start);
            std.mem.doNotOptimizeAway(result);
        }
        
        // Calculate median times
        std.mem.sort(u64, seq_times, {}, std.sort.asc(u64));
        std.mem.sort(u64, raw_par_times, {}, std.sort.asc(u64));
        std.mem.sort(u64, bounded_par_times, {}, std.sort.asc(u64));
        
        const seq_median_ns = seq_times[seq_times.len / 2];
        const raw_par_median_ns = raw_par_times[raw_par_times.len / 2];
        const bounded_par_median_ns = bounded_par_times[bounded_par_times.len / 2];
        
        const seq_median_us = seq_median_ns / 1_000;
        const raw_par_median_us = raw_par_median_ns / 1_000;
        const bounded_par_median_us = bounded_par_median_ns / 1_000;
        
        const raw_speedup = @as(f64, @floatFromInt(seq_median_ns)) / @as(f64, @floatFromInt(raw_par_median_ns));
        const bounded_speedup = @as(f64, @floatFromInt(seq_median_ns)) / @as(f64, @floatFromInt(bounded_par_median_ns));
        
        std.debug.print("{d:<12} {d:<12} {d:<12} {d:<12} {d:<12.2} {d:<12.2}\n", .{
            size, seq_median_us, raw_par_median_us, bounded_par_median_us, raw_speedup, bounded_speedup
        });
    }
    
    std.debug.print("\nNOTES:\n", .{});
    std.debug.print("• Raw: std.Thread.spawn() for every subtree (can create many threads)\n", .{});
    std.debug.print("• Bounded: Limited to 3 levels of thread spawning for realistic comparison\n", .{});
    std.debug.print("• This shows the baseline threading overhead vs sophisticated libraries\n", .{});
}