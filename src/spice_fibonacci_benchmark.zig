const std = @import("std");
const spice = @import("spice.zig");

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

// Spice parallel Fibonacci using heartbeat scheduling
fn spiceFibonacci(task: *spice.Task, n: u32) u64 {
    if (n <= 1) return n;
    
    // Use sequential for small numbers to avoid excessive task creation
    if (n <= 30) {
        return fibSequential(n);
    }
    
    // Fork two parallel tasks using Spice's heartbeat system
    const result1 = task.call(u64, spiceFibonacci, n - 1);
    const result2 = task.call(u64, spiceFibonacci, n - 2);
    
    return result1 + result2;
}

// Aggressive recursive Spice Fibonacci (for comparison)
fn spiceFibonacciRecursive(task: *spice.Task, n: u32, depth: u32) u64 {
    if (n <= 1) return n;
    
    // Limit recursion depth to prevent task explosion
    if (depth > 6 or n <= 25) {
        return fibSequential(n);
    }
    
    const FibWrapper = struct {
        n: u32,
        depth: u32,
        
        fn compute(self: @This(), inner_task: *spice.Task) u64 {
            return spiceFibonacciRecursive(inner_task, self.n, self.depth);
        }
    };
    
    const wrapper1 = FibWrapper{ .n = n - 1, .depth = depth + 1 };
    const wrapper2 = FibWrapper{ .n = n - 2, .depth = depth + 1 };
    
    const result1 = task.call(u64, FibWrapper.compute, wrapper1);
    const result2 = task.call(u64, FibWrapper.compute, wrapper2);
    
    return result1 + result2;
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
    
    const parsed = try std.json.parseFromSlice(BenchmarkConfig, allocator, config_contents, .{});
    defer parsed.deinit();
    
    const config = parsed.value;
    
    std.debug.print("SPICE FIBONACCI BENCHMARK RESULTS\n", .{});
    std.debug.print("=================================\n", .{});
    std.debug.print("{s:<12} {s:<12} {s:<12} {s:<12} {s:<12} {s:<12}\n", .{
        "Fib(n)", "Seq (ms)", "Simple (ms)", "Recursive (ms)", "Simple Speedup", "Recursive Speedup"
    });
    std.debug.print("------------------------------------------------------------------------------\n", .{});
    
    // Initialize Spice thread pool
    var thread_pool = spice.ThreadPool.init(allocator);
    thread_pool.start(.{});
    defer thread_pool.deinit();
    
    for (config.benchmark.fibonacci_numbers) |n| {
        std.debug.print("Testing Fibonacci({d})...\n", .{n});
        
        // Warmup
        for (0..config.benchmark.warmup_runs) |_| {
            _ = fibSequential(n);
            _ = thread_pool.call(u64, spiceFibonacci, n);
            _ = thread_pool.call(u64, spiceFibonacciRecursive, n, 0);
        }
        
        // Sequential timing
        var seq_times = try allocator.alloc(u64, config.benchmark.sample_count);
        defer allocator.free(seq_times);
        
        for (seq_times, 0..) |*time, i| {
            _ = i;
            const start = std.time.nanoTimestamp();
            const result = fibSequential(n);
            const end = std.time.nanoTimestamp();
            time.* = @intCast(end - start);
            std.mem.doNotOptimizeAway(result);
        }
        
        // Simple Spice timing
        var simple_times = try allocator.alloc(u64, config.benchmark.sample_count);
        defer allocator.free(simple_times);
        
        for (simple_times, 0..) |*time, i| {
            _ = i;
            const start = std.time.nanoTimestamp();
            const result = thread_pool.call(u64, spiceFibonacci, n);
            const end = std.time.nanoTimestamp();
            time.* = @intCast(end - start);
            std.mem.doNotOptimizeAway(result);
        }
        
        // Recursive Spice timing
        var recursive_times = try allocator.alloc(u64, config.benchmark.sample_count);
        defer allocator.free(recursive_times);
        
        for (recursive_times, 0..) |*time, i| {
            _ = i;
            const start = std.time.nanoTimestamp();
            const result = thread_pool.call(u64, spiceFibonacciRecursive, n, 0);
            const end = std.time.nanoTimestamp();
            time.* = @intCast(end - start);
            std.mem.doNotOptimizeAway(result);
        }
        
        // Calculate median times
        std.mem.sort(u64, seq_times, {}, std.sort.asc(u64));
        std.mem.sort(u64, simple_times, {}, std.sort.asc(u64));
        std.mem.sort(u64, recursive_times, {}, std.sort.asc(u64));
        
        const seq_median_ns = seq_times[seq_times.len / 2];
        const simple_median_ns = simple_times[simple_times.len / 2];
        const recursive_median_ns = recursive_times[recursive_times.len / 2];
        
        const seq_median_ms = seq_median_ns / 1_000_000;
        const simple_median_ms = simple_median_ns / 1_000_000;
        const recursive_median_ms = recursive_median_ns / 1_000_000;
        
        const simple_speedup = @as(f64, @floatFromInt(seq_median_ns)) / @as(f64, @floatFromInt(simple_median_ns));
        const recursive_speedup = @as(f64, @floatFromInt(seq_median_ns)) / @as(f64, @floatFromInt(recursive_median_ns));
        
        std.debug.print("{d:<12} {d:<12} {d:<12} {d:<12} {d:<12.2} {d:<12.2}\n", .{
            n, seq_median_ms, simple_median_ms, recursive_median_ms, simple_speedup, recursive_speedup
        });
    }
    
    std.debug.print("\nNOTES:\n", .{});
    std.debug.print("• Simple: Basic fork-join using Spice task.call()\n", .{});
    std.debug.print("• Recursive: Full recursive parallelization with heartbeat scheduling\n", .{});
    std.debug.print("• Sequential threshold varies to prevent task explosion\n", .{});
    std.debug.print("• Demonstrates Spice's sub-nanosecond overhead design\n", .{});
}