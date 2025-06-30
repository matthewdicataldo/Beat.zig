const std = @import("std");

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

// Thread context for manual parallelization
const FibContext = struct {
    n: u32,
    result: u64 = 0,
};

fn fibWorker(context: *FibContext) void {
    context.result = fibSequential(context.n);
}

// Simple two-thread Fibonacci
fn fibSimpleThreaded(allocator: std.mem.Allocator, n: u32) !u64 {
    _ = allocator; // Not used but kept for API consistency
    if (n <= 1) return n;
    if (n <= 30) return fibSequential(n); // Avoid threading overhead for small values
    
    var left_context = FibContext{ .n = n - 1 };
    var right_context = FibContext{ .n = n - 2 };
    
    const left_thread = try std.Thread.spawn(.{}, fibWorker, .{&left_context});
    const right_thread = try std.Thread.spawn(.{}, fibWorker, .{&right_context});
    
    left_thread.join();
    right_thread.join();
    
    return left_context.result + right_context.result;
}

// Recursive threaded Fibonacci with depth limiting
fn fibRecursiveThreaded(allocator: std.mem.Allocator, n: u32, depth: u32) !u64 {
    _ = allocator; // Not used but kept for API consistency
    
    if (n <= 1) return n;
    
    // Limit depth to prevent thread explosion
    if (depth > 4 or n <= 30) {
        return fibSequential(n);
    }
    
    var left_context = FibContext{ .n = n - 1 };
    var right_context = FibContext{ .n = n - 2 };
    
    // For recursive case, we still only spawn two threads per level
    const left_thread = try std.Thread.spawn(.{}, fibWorker, .{&left_context});
    const right_thread = try std.Thread.spawn(.{}, fibWorker, .{&right_context});
    
    left_thread.join();
    right_thread.join();
    
    return left_context.result + right_context.result;
}

// Thread pool simulation with fixed worker count
const ThreadPoolSim = struct {
    allocator: std.mem.Allocator,
    worker_count: u32,
    
    const Self = @This();
    
    fn init(allocator: std.mem.Allocator, worker_count: u32) Self {
        return Self{
            .allocator = allocator,
            .worker_count = worker_count,
        };
    }
    
    fn fibWithWorkers(self: *Self, n: u32) !u64 {
        if (n <= 1) return n;
        if (n <= 30) return fibSequential(n);
        
        // Simulate work distribution among fixed number of workers
        // For simplicity, just use the simple two-thread approach
        return fibSimpleThreaded(self.allocator, n);
    }
};

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
    
    std.debug.print("ZIG STD.THREAD FIBONACCI BENCHMARK RESULTS\n", .{});
    std.debug.print("==========================================\n", .{});
    std.debug.print("{s:<12} {s:<12} {s:<12} {s:<12} {s:<12} {s:<12}\n", .{
        "Fib(n)", "Seq (ms)", "Simple (ms)", "Pool (ms)", "Simple Speedup", "Pool Speedup"
    });
    std.debug.print("------------------------------------------------------------------------------\n", .{});
    
    // Initialize simulated thread pool
    var pool_sim = ThreadPoolSim.init(allocator, 4);
    
    for (config.benchmark.fibonacci_numbers) |n| {
        std.debug.print("Testing Fibonacci({d})...\n", .{n});
        
        // Warmup
        for (0..config.benchmark.warmup_runs) |_| {
            _ = fibSequential(n);
            _ = try fibSimpleThreaded(allocator, n);
            _ = try pool_sim.fibWithWorkers(n);
        }
        
        // Sequential timing
        const seq_times = try allocator.alloc(u64, config.benchmark.sample_count);
        defer allocator.free(seq_times);
        
        for (seq_times, 0..) |*time, i| {
            _ = i;
            const start = std.time.nanoTimestamp();
            const result = fibSequential(n);
            const end = std.time.nanoTimestamp();
            time.* = @intCast(end - start);
            std.mem.doNotOptimizeAway(result);
        }
        
        // Simple threaded timing
        const simple_times = try allocator.alloc(u64, config.benchmark.sample_count);
        defer allocator.free(simple_times);
        
        for (simple_times, 0..) |*time, i| {
            _ = i;
            const start = std.time.nanoTimestamp();
            const result = try fibSimpleThreaded(allocator, n);
            const end = std.time.nanoTimestamp();
            time.* = @intCast(end - start);
            std.mem.doNotOptimizeAway(result);
        }
        
        // Thread pool simulation timing
        const pool_times = try allocator.alloc(u64, config.benchmark.sample_count);
        defer allocator.free(pool_times);
        
        for (pool_times, 0..) |*time, i| {
            _ = i;
            const start = std.time.nanoTimestamp();
            const result = try pool_sim.fibWithWorkers(n);
            const end = std.time.nanoTimestamp();
            time.* = @intCast(end - start);
            std.mem.doNotOptimizeAway(result);
        }
        
        // Calculate median times
        std.mem.sort(u64, seq_times, {}, std.sort.asc(u64));
        std.mem.sort(u64, simple_times, {}, std.sort.asc(u64));
        std.mem.sort(u64, pool_times, {}, std.sort.asc(u64));
        
        const seq_median_ns = seq_times[seq_times.len / 2];
        const simple_median_ns = simple_times[simple_times.len / 2];
        const pool_median_ns = pool_times[pool_times.len / 2];
        
        const seq_median_ms = seq_median_ns / 1_000_000;
        const simple_median_ms = simple_median_ns / 1_000_000;
        const pool_median_ms = pool_median_ns / 1_000_000;
        
        const simple_speedup = @as(f64, @floatFromInt(seq_median_ns)) / @as(f64, @floatFromInt(simple_median_ns));
        const pool_speedup = @as(f64, @floatFromInt(seq_median_ns)) / @as(f64, @floatFromInt(pool_median_ns));
        
        std.debug.print("{d:<12} {d:<12} {d:<12} {d:<12} {d:<12.2} {d:<12.2}\n", .{
            n, seq_median_ms, simple_median_ms, pool_median_ms, simple_speedup, pool_speedup
        });
    }
    
    std.debug.print("\nNOTES:\n", .{});
    std.debug.print("• Simple: Basic std.Thread.spawn() for two tasks\n", .{});
    std.debug.print("• Pool: Simulated thread pool with fixed worker count\n", .{});
    std.debug.print("• Sequential threshold at fib(30) to prevent thread explosion\n", .{});
    std.debug.print("• Demonstrates raw threading overhead vs sophisticated libraries\n", .{});
}