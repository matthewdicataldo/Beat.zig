const std = @import("std");
const builtin = @import("builtin");
const lockfree = @import("zigpulse_v3_lockfree_deque.zig");

// Simple mutex-based queue for comparison
const MutexQueue = struct {
    tasks: std.ArrayList(*Task),
    mutex: std.Thread.Mutex,
    
    const Task = struct {
        data: u64,
    };
    
    pub fn init(allocator: std.mem.Allocator) @This() {
        return .{
            .tasks = std.ArrayList(*Task).init(allocator),
            .mutex = .{},
        };
    }
    
    pub fn deinit(self: *@This()) void {
        self.tasks.deinit();
    }
    
    pub fn push(self: *@This(), task: *Task) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.tasks.append(task);
    }
    
    pub fn pop(self: *@This()) ?*Task {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.tasks.items.len > 0) {
            return self.tasks.pop();
        }
        return null;
    }
    
    pub fn steal(self: *@This()) ?*Task {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.tasks.items.len > 0) {
            return self.tasks.orderedRemove(0);
        }
        return null;
    }
};

fn Timer() type {
    return struct {
        start_time: i64,
        
        const Self = @This();
        
        pub fn start() Self {
            return .{ .start_time = @as(i64, @intCast(std.time.nanoTimestamp())) };
        }
        
        pub fn elapsed(self: Self) u64 {
            const now = @as(i64, @intCast(std.time.nanoTimestamp()));
            return @as(u64, @intCast(now - self.start_time));
        }
    };
}

fn benchmarkSingleThread(allocator: std.mem.Allocator, iterations: usize) !void {
    std.debug.print("\n=== Single Thread Benchmark ({} operations) ===\n", .{iterations});
    
    // Lock-free deque
    {
        var deque = try lockfree.WorkStealingDeque(u64).init(allocator, iterations);
        defer deque.deinit();
        
        const tasks = try allocator.alloc(lockfree.Task(u64), iterations);
        defer allocator.free(tasks);
        
        var timer = Timer().start();
        
        // Push all
        for (tasks, 0..) |*task, i| {
            task.* = .{ .data = i };
            try deque.pushBottom(task);
        }
        
        // Pop all
        for (0..iterations) |_| {
            _ = deque.popBottom();
        }
        
        const elapsed = timer.elapsed();
        const ops_per_sec = @as(f64, @floatFromInt(iterations * 2)) * 1e9 / @as(f64, @floatFromInt(elapsed));
        
        std.debug.print("Lock-free deque: {}ms ({:.0} ops/sec)\n", .{
            elapsed / 1_000_000,
            ops_per_sec,
        });
    }
    
    // Mutex-based queue
    {
        var queue = MutexQueue.init(allocator);
        defer queue.deinit();
        
        const tasks = try allocator.alloc(MutexQueue.Task, iterations);
        defer allocator.free(tasks);
        
        var timer = Timer().start();
        
        // Push all
        for (tasks, 0..) |*task, i| {
            task.* = .{ .data = i };
            try queue.push(task);
        }
        
        // Pop all
        for (0..iterations) |_| {
            _ = queue.pop();
        }
        
        const elapsed = timer.elapsed();
        const ops_per_sec = @as(f64, @floatFromInt(iterations * 2)) * 1e9 / @as(f64, @floatFromInt(elapsed));
        
        std.debug.print("Mutex queue: {}ms ({:.0} ops/sec)\n", .{
            elapsed / 1_000_000,
            ops_per_sec,
        });
    }
}

fn benchmarkContention(allocator: std.mem.Allocator, threads: usize, ops_per_thread: usize) !void {
    std.debug.print("\n=== Contention Benchmark ({} threads, {} ops/thread) ===\n", .{
        threads, ops_per_thread
    });
    
    const total_ops = threads * ops_per_thread;
    
    // Lock-free deque
    {
        var deque = try lockfree.WorkStealingDeque(u64).init(allocator, total_ops);
        defer deque.deinit();
        
        const thread_handles = try allocator.alloc(std.Thread, threads);
        defer allocator.free(thread_handles);
        
        var completed = std.atomic.Value(u64).init(0);
        
        var timer = Timer().start();
        
        // Spawn threads
        for (thread_handles, 0..) |*handle, i| {
            handle.* = try std.Thread.spawn(.{}, struct {
                fn worker(
                    d: *lockfree.WorkStealingDeque(u64), 
                    thread_id: usize,
                    ops: usize,
                    done: *std.atomic.Value(u64),
                    alloc: std.mem.Allocator,
                ) !void {
                    const tasks = try alloc.alloc(lockfree.Task(u64), ops);
                    defer alloc.free(tasks);
                    
                    // Mix of push/pop/steal operations
                    for (tasks, 0..) |*task, j| {
                        task.* = .{ .data = thread_id * 1000 + j };
                        
                        // Push
                        try d.pushBottom(task);
                        
                        // Sometimes pop
                        if (j % 3 == 0) {
                            _ = d.popBottom();
                        }
                        
                        // Sometimes steal
                        if (j % 5 == 0) {
                            _ = d.steal();
                        }
                    }
                    
                    // Drain remaining
                    while (d.popBottom() != null or d.steal() != null) {}
                    
                    _ = done.fetchAdd(1, .release);
                }
            }.worker, .{ &deque, i, ops_per_thread, &completed, allocator });
        }
        
        // Wait for all threads
        for (thread_handles) |handle| {
            handle.join();
        }
        
        const elapsed = timer.elapsed();
        const ops_per_sec = @as(f64, @floatFromInt(total_ops)) * 1e9 / @as(f64, @floatFromInt(elapsed));
        
        std.debug.print("Lock-free deque: {}ms ({:.0} ops/sec)\n", .{
            elapsed / 1_000_000,
            ops_per_sec,
        });
        
        const stats = deque.getStats();
        std.debug.print("  Stats: pushes={}, pops={}, steals={}\n", .{
            stats.pushes, stats.pops, stats.steals
        });
    }
    
    // Mutex-based queue
    {
        var queue = MutexQueue.init(allocator);
        defer queue.deinit();
        
        const thread_handles = try allocator.alloc(std.Thread, threads);
        defer allocator.free(thread_handles);
        
        var completed = std.atomic.Value(u64).init(0);
        
        var timer = Timer().start();
        
        // Spawn threads
        for (thread_handles, 0..) |*handle, i| {
            handle.* = try std.Thread.spawn(.{}, struct {
                fn worker(
                    q: *MutexQueue,
                    thread_id: usize,
                    ops: usize,
                    done: *std.atomic.Value(u64),
                    alloc: std.mem.Allocator,
                ) !void {
                    const tasks = try alloc.alloc(MutexQueue.Task, ops);
                    defer alloc.free(tasks);
                    
                    // Mix of push/pop/steal operations
                    for (tasks, 0..) |*task, j| {
                        task.* = .{ .data = thread_id * 1000 + j };
                        
                        // Push
                        try q.push(task);
                        
                        // Sometimes pop
                        if (j % 3 == 0) {
                            _ = q.pop();
                        }
                        
                        // Sometimes steal
                        if (j % 5 == 0) {
                            _ = q.steal();
                        }
                    }
                    
                    // Drain remaining
                    while (q.pop() != null or q.steal() != null) {}
                    
                    _ = done.fetchAdd(1, .release);
                }
            }.worker, .{ &queue, i, ops_per_thread, &completed, allocator });
        }
        
        // Wait for all threads
        for (thread_handles) |handle| {
            handle.join();
        }
        
        const elapsed = timer.elapsed();
        const ops_per_sec = @as(f64, @floatFromInt(total_ops)) * 1e9 / @as(f64, @floatFromInt(elapsed));
        
        std.debug.print("Mutex queue: {}ms ({:.0} ops/sec)\n", .{
            elapsed / 1_000_000,
            ops_per_sec,
        });
    }
}

pub fn main() !void {
    std.debug.print("=== Lock-Free vs Mutex Queue Benchmark ===\n", .{});
    std.debug.print("Build mode: {}\n", .{builtin.mode});
    
    const allocator = std.heap.page_allocator;
    
    // Single-threaded benchmark
    try benchmarkSingleThread(allocator, 100_000);
    
    // Contention benchmarks
    const cpu_count = try std.Thread.getCpuCount();
    std.debug.print("\nCPU count: {}\n", .{cpu_count});
    
    // Low contention
    try benchmarkContention(allocator, 2, 10_000);
    
    // Medium contention
    try benchmarkContention(allocator, cpu_count / 2, 10_000);
    
    // High contention
    try benchmarkContention(allocator, cpu_count, 10_000);
    
    // Extreme contention
    if (cpu_count > 4) {
        try benchmarkContention(allocator, cpu_count * 2, 5_000);
    }
    
    std.debug.print("\n=== Benchmark Complete ===\n", .{});
}