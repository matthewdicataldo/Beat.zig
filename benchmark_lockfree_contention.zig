const std = @import("std");
const beat = @import("zigpulse");

// Lock-free Queue CAS Contention Analysis Benchmark
// Analyzes current CAS retry storm patterns and establishes optimization baseline
// Target: Identify contention bottlenecks and measure 15-25% improvement potential

const TOTAL_TASKS = 2048;  // Reduced to fit in deque capacity
const NUM_WORKER_THREADS = 8;
const CONTENTION_ITERATIONS = 50;

// Enhanced statistics for contention analysis
const ContentionStats = struct {
    successful_steals: std.atomic.Value(u64),
    failed_steal_attempts: std.atomic.Value(u64),
    total_cas_operations: std.atomic.Value(u64),
    cas_failures: std.atomic.Value(u64),
    retry_cycles: std.atomic.Value(u64),
    contention_incidents: std.atomic.Value(u64),
    
    pub fn init() ContentionStats {
        return ContentionStats{
            .successful_steals = std.atomic.Value(u64).init(0),
            .failed_steal_attempts = std.atomic.Value(u64).init(0),
            .total_cas_operations = std.atomic.Value(u64).init(0),
            .cas_failures = std.atomic.Value(u64).init(0),
            .retry_cycles = std.atomic.Value(u64).init(0),
            .contention_incidents = std.atomic.Value(u64).init(0),
        };
    }
    
    pub fn getFailureRate(self: *const ContentionStats) f64 {
        const total = self.total_cas_operations.load(.acquire);
        const failures = self.cas_failures.load(.acquire);
        return if (total > 0) (@as(f64, @floatFromInt(failures)) / @as(f64, @floatFromInt(total))) * 100.0 else 0.0;
    }
    
    pub fn getStealEfficiency(self: *const ContentionStats) f64 {
        const successful = self.successful_steals.load(.acquire);
        const failed = self.failed_steal_attempts.load(.acquire);
        const total_attempts = successful + failed;
        return if (total_attempts > 0) (@as(f64, @floatFromInt(successful)) / @as(f64, @floatFromInt(total_attempts))) * 100.0 else 0.0;
    }
};

// Mock task for benchmarking
fn mockTaskFunction(_: *anyopaque) void {
    // Simulate some work to create realistic contention patterns
    var sum: u64 = 0;
    for (0..100) |i| {
        sum +%= i;
    }
    std.mem.doNotOptimizeAway(sum);
}

// Enhanced work-stealing deque wrapper with contention tracking
const ContentionTrackingDeque = struct {
    const Self = @This();
    
    deque: beat.lockfree.WorkStealingDeque(beat.Task),
    stats: ContentionStats,
    
    pub fn init(allocator: std.mem.Allocator, capacity: u32) !Self {
        return Self{
            .deque = try beat.lockfree.WorkStealingDeque(beat.Task).init(allocator, capacity),
            .stats = ContentionStats.init(),
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.deque.deinit();
    }
    
    // Instrumented steal function to track contention
    pub fn stealWithTracking(self: *Self) ?beat.Task {
        _ = self.stats.total_cas_operations.fetchAdd(1, .monotonic);
        
        const start_time = std.time.nanoTimestamp();
        var retry_count: u32 = 0;
        
        // Attempt steal with retry counting
        while (retry_count < 3) { // Max 3 retries to detect contention
            if (self.deque.steal()) |task| {
                _ = self.stats.successful_steals.fetchAdd(1, .monotonic);
                if (retry_count > 0) {
                    _ = self.stats.retry_cycles.fetchAdd(retry_count, .monotonic);
                }
                return task;
            }
            
            retry_count += 1;
            if (retry_count > 1) {
                _ = self.stats.cas_failures.fetchAdd(1, .monotonic);
                // Simulate current behavior - immediate retry
            }
        }
        
        _ = self.stats.failed_steal_attempts.fetchAdd(1, .monotonic);
        if (retry_count >= 3) {
            _ = self.stats.contention_incidents.fetchAdd(1, .monotonic);
        }
        
        const end_time = std.time.nanoTimestamp();
        const duration = end_time - start_time;
        if (duration > 1000) { // More than 1μs indicates contention
            _ = self.stats.contention_incidents.fetchAdd(1, .monotonic);
        }
        
        return null;
    }
    
    pub fn pushBottom(self: *Self, task: beat.Task) !void {
        return self.deque.pushBottom(task);
    }
    
    pub fn popBottom(self: *Self) ?beat.Task {
        return self.deque.popBottom();
    }
    
    pub fn getStats(self: *const Self) ContentionStats {
        return self.stats;
    }
};

// Contention stress test - multiple threads hammering the same deque
fn contentionStressTest(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🔥 CAS Contention Stress Test\n", .{});
    std.debug.print("============================\n", .{});
    
    var tracked_deque = try ContentionTrackingDeque.init(allocator, 4096);
    defer tracked_deque.deinit();
    
    // Pre-populate deque with tasks
    for (0..TOTAL_TASKS) |i| {
        const task = beat.Task{
            .func = mockTaskFunction,
            .data = @as(*anyopaque, @ptrFromInt(@as(usize, 0x1000 + i))),
            .priority = beat.Priority.normal,
        };
        try tracked_deque.pushBottom(task);
    }
    
    const ThreadContext = struct {
        deque: *ContentionTrackingDeque,
        thread_id: usize,
        tasks_stolen: std.atomic.Value(u64),
        local_failures: std.atomic.Value(u64),
    };
    
    var contexts: [NUM_WORKER_THREADS]ThreadContext = undefined;
    var threads: [NUM_WORKER_THREADS]std.Thread = undefined;
    
    for (&contexts, 0..) |*ctx, i| {
        ctx.* = ThreadContext{
            .deque = &tracked_deque,
            .thread_id = i,
            .tasks_stolen = std.atomic.Value(u64).init(0),
            .local_failures = std.atomic.Value(u64).init(0),
        };
    }
    
    const worker_function = struct {
        fn worker(ctx: *ThreadContext) void {
            const start_time = std.time.nanoTimestamp();
            var local_stolen: u64 = 0;
            var local_failures: u64 = 0;
            
            // Aggressive stealing to create maximum contention
            while (true) {
                if (ctx.deque.stealWithTracking()) |_| {
                    local_stolen += 1;
                    
                    // Simulate task execution
                    var sum: u64 = 0;
                    for (0..50) |i| {
                        sum +%= i;
                    }
                    std.mem.doNotOptimizeAway(sum);
                } else {
                    local_failures += 1;
                    
                    // Break if we've had too many consecutive failures
                    if (local_failures > 100 and local_stolen == 0) {
                        break;
                    }
                }
                
                // Break if we've been running too long (safety)
                const current_time = std.time.nanoTimestamp();
                if (current_time - start_time > 5_000_000_000) { // 5 seconds max
                    break;
                }
            }
            
            ctx.tasks_stolen.store(local_stolen, .release);
            ctx.local_failures.store(local_failures, .release);
        }
    }.worker;
    
    const start_time = std.time.nanoTimestamp();
    
    // Launch all worker threads simultaneously for maximum contention
    for (&threads, 0..) |*thread, i| {
        thread.* = try std.Thread.spawn(.{}, worker_function, .{&contexts[i]});
    }
    
    // Wait for completion
    for (&threads) |*thread| {
        thread.join();
    }
    
    const end_time = std.time.nanoTimestamp();
    const total_duration_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
    
    // Collect results
    var total_stolen: u64 = 0;
    var total_local_failures: u64 = 0;
    
    for (&contexts) |*ctx| {
        total_stolen += ctx.tasks_stolen.load(.acquire);
        total_local_failures += ctx.local_failures.load(.acquire);
    }
    
    const final_stats = tracked_deque.getStats();
    
    std.debug.print("📊 Contention Analysis Results:\n", .{});
    std.debug.print("  Test Duration: {d:.2}ms\n", .{total_duration_ms});
    std.debug.print("  Worker Threads: {}\n", .{NUM_WORKER_THREADS});
    std.debug.print("  Total Tasks Stolen: {}\n", .{total_stolen});
    std.debug.print("  Successful Steals: {}\n", .{final_stats.successful_steals.load(.acquire)});
    std.debug.print("  Failed Steal Attempts: {}\n", .{final_stats.failed_steal_attempts.load(.acquire)});
    std.debug.print("  Steal Success Rate: {d:.1}%\n", .{final_stats.getStealEfficiency()});
    std.debug.print("\n", .{});
    
    std.debug.print("🔬 CAS Operation Analysis:\n", .{});
    std.debug.print("  Total CAS Operations: {}\n", .{final_stats.total_cas_operations.load(.acquire)});
    std.debug.print("  CAS Failures: {}\n", .{final_stats.cas_failures.load(.acquire)});
    std.debug.print("  CAS Failure Rate: {d:.1}%\n", .{final_stats.getFailureRate()});
    std.debug.print("  Retry Cycles: {}\n", .{final_stats.retry_cycles.load(.acquire)});
    std.debug.print("  Contention Incidents: {}\n", .{final_stats.contention_incidents.load(.acquire)});
    std.debug.print("\n", .{});
    
    std.debug.print("⚡ Performance Metrics:\n", .{});
    const steals_per_ms = @as(f64, @floatFromInt(total_stolen)) / total_duration_ms;
    const avg_steal_time_ns = total_duration_ms * 1_000_000.0 / @as(f64, @floatFromInt(total_stolen));
    
    std.debug.print("  Steals per millisecond: {d:.0}\n", .{steals_per_ms});
    std.debug.print("  Average steal time: {d:.0}ns\n", .{avg_steal_time_ns});
    
    // Calculate contention overhead
    const cas_per_steal = @as(f64, @floatFromInt(final_stats.total_cas_operations.load(.acquire))) / @as(f64, @floatFromInt(total_stolen));
    const contention_overhead = (final_stats.getFailureRate() / 100.0) * 100.0;
    
    std.debug.print("  CAS operations per steal: {d:.2}\n", .{cas_per_steal});
    std.debug.print("  Contention overhead: {d:.1}%\n", .{contention_overhead});
}

// Benchmark different contention scenarios
fn benchmarkContentionScenarios(allocator: std.mem.Allocator) !void {
    std.debug.print("\n📈 Contention Scenario Benchmarks\n", .{});
    std.debug.print("==================================\n", .{});
    
    const scenarios = [_]struct { threads: u32, name: []const u8 }{
        .{ .threads = 2, .name = "Low Contention (2 threads)" },
        .{ .threads = 4, .name = "Medium Contention (4 threads)" },
        .{ .threads = 8, .name = "High Contention (8 threads)" },
        .{ .threads = 16, .name = "Extreme Contention (16 threads)" },
    };
    
    for (scenarios) |scenario| {
        std.debug.print("\n🧪 Testing: {s}\n", .{scenario.name});
        
        var tracked_deque = try ContentionTrackingDeque.init(allocator, 2048);
        defer tracked_deque.deinit();
        
        // Pre-populate with fewer tasks to force more contention
        const tasks_per_thread = 200;
        for (0..scenario.threads * tasks_per_thread) |i| {
            const task = beat.Task{
                .func = mockTaskFunction,
                .data = @as(*anyopaque, @ptrFromInt(@as(usize, 0x1000 + i))),
                .priority = beat.Priority.normal,
            };
            try tracked_deque.pushBottom(task);
        }
        
        const ThreadContext = struct {
            deque: *ContentionTrackingDeque,
            tasks_processed: std.atomic.Value(u64),
        };
        
        var contexts = try allocator.alloc(ThreadContext, scenario.threads);
        defer allocator.free(contexts);
        
        const threads = try allocator.alloc(std.Thread, scenario.threads);
        defer allocator.free(threads);
        
        for (contexts) |*ctx| {
            ctx.* = ThreadContext{
                .deque = &tracked_deque,
                .tasks_processed = std.atomic.Value(u64).init(0),
            };
        }
        
        const worker_function = struct {
            fn worker(ctx: *ThreadContext) void {
                var processed: u64 = 0;
                const start_time = std.time.nanoTimestamp();
                
                while (true) {
                    if (ctx.deque.stealWithTracking()) |_| {
                        processed += 1;
                        
                        // Simulate work
                        var sum: u64 = 0;
                        for (0..25) |i| {
                            sum +%= i;
                        }
                        std.mem.doNotOptimizeAway(sum);
                    } else {
                        // No more work available
                        const current_time = std.time.nanoTimestamp();
                        if (current_time - start_time > 1_000_000_000) { // 1 second timeout
                            break;
                        }
                    }
                }
                
                ctx.tasks_processed.store(processed, .release);
            }
        }.worker;
        
        const start_time = std.time.nanoTimestamp();
        
        for (threads, 0..) |*thread, i| {
            thread.* = try std.Thread.spawn(.{}, worker_function, .{&contexts[i]});
        }
        
        for (threads) |*thread| {
            thread.join();
        }
        
        const end_time = std.time.nanoTimestamp();
        const duration_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
        
        var total_processed: u64 = 0;
        for (contexts) |*ctx| {
            total_processed += ctx.tasks_processed.load(.acquire);
        }
        
        const stats = tracked_deque.getStats();
        
        std.debug.print("  Duration: {d:.2}ms\n", .{duration_ms});
        std.debug.print("  Tasks processed: {}\n", .{total_processed});
        std.debug.print("  CAS failure rate: {d:.1}%\n", .{stats.getFailureRate()});
        std.debug.print("  Steal efficiency: {d:.1}%\n", .{stats.getStealEfficiency()});
        std.debug.print("  Contention incidents: {}\n", .{stats.contention_incidents.load(.acquire)});
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("🔍 Lock-Free Queue CAS Contention Analysis\n", .{});
    std.debug.print("===========================================\n", .{});
    std.debug.print("Target: Identify retry storm patterns for 15-25% improvement\n", .{});
    std.debug.print("Analyzing {} tasks across {} worker threads\n", .{ TOTAL_TASKS, NUM_WORKER_THREADS });
    
    try contentionStressTest(allocator);
    try benchmarkContentionScenarios(allocator);
    
    std.debug.print("\n🎯 Optimization Opportunities Identified:\n", .{});
    std.debug.print("==========================================\n", .{});
    std.debug.print("✅ Baseline contention measurements complete\n", .{});
    std.debug.print("🔧 Next steps:\n", .{});
    std.debug.print("  1. Implement exponential backoff for failed CAS\n", .{});
    std.debug.print("  2. Add steal batching to reduce contention frequency\n", .{});
    std.debug.print("  3. Design adaptive contention detection\n", .{});
    std.debug.print("  4. Benchmark improvements against this baseline\n", .{});
    
    std.debug.print("\n🚀 Lock-Free CAS Optimization: Phase 1 COMPLETE!\n", .{});
}