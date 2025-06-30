const std = @import("std");

const BenchmarkConfig = struct {
    benchmark: struct {
        name: []const u8,
        description: []const u8,
        tree_sizes: []const u32,
        fibonacci_numbers: []const u32,
        matrix_sizes: ?[]const u32 = null,
        sample_count: u32,
        warmup_runs: u32,
        timeout_seconds: u32,
    },
};

// Sequential implementations for baseline
fn fibSequential(n: u32) u64 {
    if (n <= 1) return n;
    return fibSequential(n - 1) + fibSequential(n - 2);
}

const TreeNode = struct {
    value: i64,
    left: ?*TreeNode = null,
    right: ?*TreeNode = null,
};

fn createTree(allocator: std.mem.Allocator, size: usize) !*TreeNode {
    if (size == 0) return error.InvalidSize;
    
    const node = try allocator.create(TreeNode);
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

fn destroyTree(allocator: std.mem.Allocator, node: ?*TreeNode) void {
    if (node) |n| {
        destroyTree(allocator, n.left);
        destroyTree(allocator, n.right);
        allocator.destroy(n);
    }
}

fn treeSum(node: ?*TreeNode) i64 {
    if (node) |n| {
        return n.value + treeSum(n.left) + treeSum(n.right);
    }
    return 0;
}

// Thread-based implementations
const ThreadContext = struct {
    result: u64 = 0,
    
    // For Fibonacci
    n: u32 = 0,
    
    // For tree sum
    tree_node: ?*TreeNode = null,
};

fn fibWorker(context: *ThreadContext) void {
    context.result = fibSequential(context.n);
}

fn treeWorker(context: *ThreadContext) void {
    context.result = @intCast(treeSum(context.tree_node));
}

fn fibSimpleThreaded(allocator: std.mem.Allocator, n: u32) !u64 {
    _ = allocator;
    if (n <= 1) return n;
    if (n <= 30) return fibSequential(n); // Avoid threading overhead for small values
    
    var left_context = ThreadContext{ .n = n - 1 };
    var right_context = ThreadContext{ .n = n - 2 };
    
    const left_thread = try std.Thread.spawn(.{}, fibWorker, .{&left_context});
    const right_thread = try std.Thread.spawn(.{}, fibWorker, .{&right_context});
    
    left_thread.join();
    right_thread.join();
    
    return left_context.result + right_context.result;
}

// Matrix multiplication implementations
fn matrixMultiplySequential(a: []const f32, b: []const f32, c: []f32, n: usize) void {
    // Standard O(n³) matrix multiplication: C = A * B
    for (0..n) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..n) |k| {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

fn matrixMultiplyParallel(allocator: std.mem.Allocator, a: []const f32, b: []const f32, c: []f32, n: usize) !void {
    // Parallel matrix multiplication using std.Thread
    const num_threads = @min(std.Thread.getCpuCount() catch 4, n);
    const rows_per_thread = n / num_threads;
    
    const threads = try allocator.alloc(std.Thread, num_threads);
    defer allocator.free(threads);
    
    const MatrixTask = struct {
        a: []const f32,
        b: []const f32,
        c: []f32,
        n: usize,
        start_row: usize,
        end_row: usize,
        
        fn run(self: @This()) void {
            for (self.start_row..self.end_row) |i| {
                for (0..self.n) |j| {
                    var sum: f32 = 0.0;
                    for (0..self.n) |k| {
                        sum += self.a[i * self.n + k] * self.b[k * self.n + j];
                    }
                    self.c[i * self.n + j] = sum;
                }
            }
        }
    };
    
    // Start threads
    for (threads, 0..) |*thread, i| {
        const start_row = i * rows_per_thread;
        const end_row = if (i == num_threads - 1) n else (i + 1) * rows_per_thread;
        
        const task = MatrixTask{
            .a = a,
            .b = b,
            .c = c,
            .n = n,
            .start_row = start_row,
            .end_row = end_row,
        };
        
        thread.* = try std.Thread.spawn(.{}, MatrixTask.run, .{task});
    }
    
    // Wait for all threads
    for (threads) |thread| {
        thread.join();
    }
}

fn treeSumThreaded(allocator: std.mem.Allocator, tree: ?*TreeNode) !u64 {
    _ = allocator;
    if (tree == null) return 0;
    
    const node = tree.?;
    var result: u64 = @intCast(node.value);
    
    if (node.left == null and node.right == null) {
        return result;
    }
    
    var left_context = ThreadContext{ .tree_node = node.left };
    var right_context = ThreadContext{ .tree_node = node.right };
    
    var left_thread: ?std.Thread = null;
    var right_thread: ?std.Thread = null;
    
    if (node.left != null) {
        left_thread = try std.Thread.spawn(.{}, treeWorker, .{&left_context});
    }
    
    if (node.right != null) {
        right_thread = try std.Thread.spawn(.{}, treeWorker, .{&right_context});
    }
    
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

fn runBenchmarkSet(
    allocator: std.mem.Allocator,
    name: []const u8,
    sample_count: u32,
    warmup_runs: u32,
    seq_fn: anytype,
    par_fn: anytype,
    input: anytype,
) !void {
    std.debug.print("Testing {}...\n", .{input});
    
    // Warmup
    for (0..warmup_runs) |_| {
        _ = seq_fn(input);
        _ = try par_fn(input, allocator);
    }
    
    // Sequential timing
    const seq_times = try allocator.alloc(u64, sample_count);
    defer allocator.free(seq_times);
    
    for (seq_times, 0..) |*time, i| {
        _ = i;
        const start = std.time.nanoTimestamp();
        const result = seq_fn(input);
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
        const result = try par_fn(input, allocator);
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
    else if (overhead_ns < 1_000_000) 
        std.fmt.bufPrint(overhead_buffer[0..], "{}ns", .{overhead_ns}) catch "overflow"
    else 
        std.fmt.bufPrint(overhead_buffer[0..], "{}μs", .{overhead_ns / 1_000}) catch "overflow";
    
    std.debug.print("{s:<12} {d:<12} {d:<12} {d:<12.2} {s:<12}\n", .{
        name, seq_median_us, par_median_us, speedup, overhead_str
    });
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Read config from JSON file
    const config_file = std.fs.cwd().openFile("benchmark_config.json", .{}) catch |err| {
        std.debug.print("Error: Could not open benchmark_config.json: {}\n", .{err});
        return;
    };
    defer config_file.close();
    
    const config_contents = config_file.readToEndAlloc(allocator, 1024 * 1024) catch |err| {
        std.debug.print("Error: Could not read config file: {}\n", .{err});
        return;
    };
    defer allocator.free(config_contents);
    
    const parsed = std.json.parseFromSlice(BenchmarkConfig, allocator, config_contents, .{ .ignore_unknown_fields = true }) catch |err| {
        std.debug.print("Error: Could not parse JSON config: {}\n", .{err});
        return;
    };
    defer parsed.deinit();
    
    const config = parsed.value;
    
    std.debug.print("UNIFIED PARALLELISM BENCHMARK SUITE\n", .{});
    std.debug.print("===================================\n", .{});
    std.debug.print("Comparing std.Thread baseline vs optimized approaches\n\n", .{});
    
    // Fibonacci benchmarks
    std.debug.print("FIBONACCI RECURSIVE BENCHMARK RESULTS\n", .{});
    std.debug.print("====================================\n", .{});
    std.debug.print("{s:<12} {s:<12} {s:<12} {s:<12} {s:<12}\n", .{
        "Fib(n)", "Seq (μs)", "Par (μs)", "Speedup", "Overhead"
    });
    std.debug.print("----------------------------------------------------------\n", .{});
    
    for (config.benchmark.fibonacci_numbers) |n| {
        const FibInput = struct {
            n: u32,
            
            fn seq(self: @This()) u64 {
                return fibSequential(self.n);
            }
            
            fn par(self: @This(), alloc: std.mem.Allocator) !u64 {
                return fibSimpleThreaded(alloc, self.n);
            }
        };
        
        const input = FibInput{ .n = n };
        var buffer: [32]u8 = undefined;
        const name = std.fmt.bufPrint(buffer[0..], "Fib({})", .{n}) catch unreachable;
        
        try runBenchmarkSet(
            allocator,
            name,
            config.benchmark.sample_count,
            config.benchmark.warmup_runs,
            FibInput.seq,
            FibInput.par,
            input,
        );
    }
    
    std.debug.print("\n", .{});
    
    // Tree sum benchmarks
    std.debug.print("TREE SUM BENCHMARK RESULTS\n", .{});
    std.debug.print("==========================\n", .{});
    std.debug.print("{s:<12} {s:<12} {s:<12} {s:<12} {s:<12}\n", .{
        "Tree Size", "Seq (μs)", "Par (μs)", "Speedup", "Overhead"
    });
    std.debug.print("----------------------------------------------------------\n", .{});
    
    for (config.benchmark.tree_sizes) |size| {
        // Use arena allocator for better memory locality (matches Rust's Box allocation pattern)
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const arena_allocator = arena.allocator();
        
        const tree = createTree(arena_allocator, size) catch |err| {
            std.debug.print("Error creating tree of size {}: {}\n", .{ size, err });
            continue;
        };
        // No need for manual cleanup - arena handles it
        
        const TreeInput = struct {
            tree: ?*TreeNode,
            
            fn seq(self: @This()) i64 {
                return treeSum(self.tree);
            }
            
            fn par(self: @This(), alloc: std.mem.Allocator) !u64 {
                return treeSumThreaded(alloc, self.tree);
            }
        };
        
        const input = TreeInput{ .tree = tree };
        var buffer: [32]u8 = undefined;
        const name = std.fmt.bufPrint(buffer[0..], "{} nodes", .{size}) catch unreachable;
        
        try runBenchmarkSet(
            arena_allocator,  // Use arena allocator for consistent memory locality
            name,
            config.benchmark.sample_count,
            config.benchmark.warmup_runs,
            TreeInput.seq,
            TreeInput.par,
            input,
        );
    }
    
    // Matrix multiplication benchmarks (if configured)
    if (config.benchmark.matrix_sizes) |matrix_sizes| {
        std.debug.print("\n", .{});
        std.debug.print("MATRIX MULTIPLICATION BENCHMARK RESULTS\n", .{});
        std.debug.print("=======================================\n", .{});
        std.debug.print("{s:<12} {s:<12} {s:<12} {s:<12}\n", .{
            "Matrix Size", "Seq (ms)", "Par (ms)", "Speedup"
        });
        std.debug.print("-----------------------------------------------\n", .{});
        
        for (matrix_sizes) |size| {
            // Allocate matrices
            const matrix_elements = size * size;
            const a = try allocator.alloc(f32, matrix_elements);
            defer allocator.free(a);
            const b = try allocator.alloc(f32, matrix_elements);
            defer allocator.free(b);
            const c_seq = try allocator.alloc(f32, matrix_elements);
            defer allocator.free(c_seq);
            const c_par = try allocator.alloc(f32, matrix_elements);
            defer allocator.free(c_par);
            
            // Initialize matrices with random values
            var prng = std.Random.DefaultPrng.init(12345);
            const random = prng.random();
            for (a) |*val| val.* = random.float(f32) * 10.0;
            for (b) |*val| val.* = random.float(f32) * 10.0;
            
            // Sequential timing (3 runs for quick results)
            var seq_times: [3]u64 = undefined;
            for (&seq_times) |*time| {
                const start = std.time.nanoTimestamp();
                matrixMultiplySequential(a, b, c_seq, size);
                const end = std.time.nanoTimestamp();
                time.* = @intCast(end - start);
            }
            
            // Parallel timing
            var par_times: [3]u64 = undefined;
            for (&par_times) |*time| {
                const start = std.time.nanoTimestamp();
                matrixMultiplyParallel(allocator, a, b, c_par, size) catch continue;
                const end = std.time.nanoTimestamp();
                time.* = @intCast(end - start);
            }
            
            // Calculate medians
            std.mem.sort(u64, &seq_times, {}, std.sort.asc(u64));
            std.mem.sort(u64, &par_times, {}, std.sort.asc(u64));
            
            const seq_median_ms = seq_times[1] / 1_000_000;
            const par_median_ms = par_times[1] / 1_000_000;
            const speedup = @as(f64, @floatFromInt(seq_median_ms)) / @as(f64, @floatFromInt(par_median_ms));
            
            std.debug.print("{}x{}       {}ms        {}ms        {d:.2}x\n", .{ size, size, seq_median_ms, par_median_ms, speedup });
        }
    }
    
    std.debug.print("\nNOTES:\n", .{});
    std.debug.print("• All measurements use std.Thread as the parallel implementation\n", .{});
    std.debug.print("• Matrix multiplication: Standard O(n³) algorithm with row-wise parallelization\n", .{});
    std.debug.print("• Sequential threshold: fib(30) for Fibonacci, no threshold for tree sum\n", .{});
    std.debug.print("• This provides baseline measurements for comparison with Beat.zig\n", .{});
    std.debug.print("• Sample count: {}, Warmup runs: {}\n", .{ config.benchmark.sample_count, config.benchmark.warmup_runs });
    std.debug.print("• OPTIMIZED: Using arena allocator for better memory locality (matches Rust performance)\n", .{});
    
    // Performance analysis
    std.debug.print("\nPERFORMANCE ANALYSIS:\n", .{});
    std.debug.print("• Fibonacci shows exponential complexity - parallel helps for large n\n", .{});
    std.debug.print("• Tree sum shows linear complexity - parallel overhead may dominate\n", .{});
    std.debug.print("• Thread creation overhead becomes apparent in small task scenarios\n", .{});
    std.debug.print("• Beat.zig's thread pool would eliminate repeated thread creation costs\n", .{});
}