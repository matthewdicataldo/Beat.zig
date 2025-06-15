const std = @import("std");
const beat = @import("src/core.zig");

// Work-Stealing Performance Analysis
// This benchmark analyzes the current 40% efficiency for small tasks

const TaskType = enum {
    tiny,       // < 100 cycles
    small,      // 100-1000 cycles  
    medium,     // 1000-10000 cycles
    large,      // > 10000 cycles
};

const WorkStealingMetrics = struct {
    tasks_submitted: u64 = 0,
    tasks_completed: u64 = 0,
    tasks_stolen: u64 = 0,
    total_execution_time_ns: u64 = 0,
    total_overhead_ns: u64 = 0,
    stealing_attempts: u64 = 0,
    successful_steals: u64 = 0,
    failed_steals: u64 = 0,
    
    pub fn efficiency(self: WorkStealingMetrics) f64 {
        if (self.tasks_submitted == 0) return 0.0;
        return @as(f64, @floatFromInt(self.tasks_completed)) / @as(f64, @floatFromInt(self.tasks_submitted));
    }
    
    pub fn steal_success_rate(self: WorkStealingMetrics) f64 {
        if (self.stealing_attempts == 0) return 0.0;
        return @as(f64, @floatFromInt(self.successful_steals)) / @as(f64, @floatFromInt(self.stealing_attempts));
    }
    
    pub fn overhead_ratio(self: WorkStealingMetrics) f64 {
        if (self.total_execution_time_ns == 0) return 0.0;
        return @as(f64, @floatFromInt(self.total_overhead_ns)) / @as(f64, @floatFromInt(self.total_execution_time_ns));
    }
};

fn simulateWork(task_type: TaskType) void {
    var sum: u64 = 0;
    const cycles: u32 = switch (task_type) {
        .tiny => 50,
        .small => 500,
        .medium => 5000,
        .large => 50000,
    };
    
    for (0..cycles) |i| {
        sum += i;
    }
    std.mem.doNotOptimizeAway(&sum);
}

fn benchmarkWorkStealingEfficiency(allocator: std.mem.Allocator, task_type: TaskType, num_tasks: u32) !WorkStealingMetrics {
    std.debug.print("\n=== Benchmarking Work-Stealing for {s} tasks ===\n", .{@tagName(task_type)});
    
    // Create pool with statistics enabled
    const pool = try beat.createPoolWithConfig(allocator, .{
        .num_workers = 4,
        .enable_work_stealing = true,
        .enable_statistics = true,
        .enable_lock_free = true,
        .task_queue_size = 32,
    });
    defer pool.deinit();
    
    var metrics = WorkStealingMetrics{};
    var task_counter = std.atomic.Value(u64).init(0);
    
    const work_task = struct {
        fn run(data: *anyopaque) void {
            const args = @as(*struct { task_type: TaskType, counter: *std.atomic.Value(u64) }, @ptrCast(@alignCast(data)));
            simulateWork(args.task_type);
            _ = args.counter.fetchAdd(1, .monotonic);
        }
    }.run;
    
    const start_time = std.time.nanoTimestamp();
    
    // Submit all tasks to create imbalance and force stealing
    var task_args = .{ .task_type = task_type, .counter = &task_counter };
    for (0..num_tasks) |_| {
        try pool.submit(.{ 
            .func = work_task, 
            .data = &task_args,
            .priority = .normal 
        });
        metrics.tasks_submitted += 1;
    }
    
    // Wait for completion
    pool.wait();
    
    const end_time = std.time.nanoTimestamp();
    
    metrics.tasks_completed = task_counter.load(.acquire);
    metrics.total_execution_time_ns = @intCast(end_time - start_time);
    
    // Get statistics from pool
    const pool_stats = &pool.stats;
    metrics.tasks_stolen = pool_stats.tasks_stolen.load(.acquire);
    
    std.debug.print("Tasks submitted: {}\n", .{metrics.tasks_submitted});
    std.debug.print("Tasks completed: {}\n", .{metrics.tasks_completed});
    std.debug.print("Tasks stolen: {}\n", .{metrics.tasks_stolen});
    std.debug.print("Total time: {:.2}ms\n", .{@as(f64, @floatFromInt(metrics.total_execution_time_ns)) / 1_000_000.0});
    std.debug.print("Efficiency: {:.1}%\n", .{metrics.efficiency() * 100.0});
    
    if (metrics.tasks_submitted > 0) {
        const steal_percentage = @as(f64, @floatFromInt(metrics.tasks_stolen)) / @as(f64, @floatFromInt(metrics.tasks_submitted)) * 100.0;
        std.debug.print("Steal rate: {:.1}%\n", .{steal_percentage});
        
        const avg_time_per_task = @as(f64, @floatFromInt(metrics.total_execution_time_ns)) / @as(f64, @floatFromInt(metrics.tasks_completed));
        std.debug.print("Avg time per task: {:.0}ns\n", .{avg_time_per_task});
    }
    
    return metrics;
}

fn analyzeWorkStealingBottlenecks() !void {
    std.debug.print("=== Work-Stealing Performance Bottleneck Analysis ===\n", .{});
    
    const allocator = std.heap.page_allocator;
    
    // Test different task sizes
    const tiny_metrics = try benchmarkWorkStealingEfficiency(allocator, .tiny, 1000);
    const small_metrics = try benchmarkWorkStealingEfficiency(allocator, .small, 1000);
    const medium_metrics = try benchmarkWorkStealingEfficiency(allocator, .medium, 100);
    const large_metrics = try benchmarkWorkStealingEfficiency(allocator, .large, 50);
    
    std.debug.print("\n=== Summary Analysis ===\n", .{});
    std.debug.print("Tiny tasks efficiency: {:.1}%\n", .{tiny_metrics.efficiency() * 100.0});
    std.debug.print("Small tasks efficiency: {:.1}%\n", .{small_metrics.efficiency() * 100.0});
    std.debug.print("Medium tasks efficiency: {:.1}%\n", .{medium_metrics.efficiency() * 100.0});
    std.debug.print("Large tasks efficiency: {:.1}%\n", .{large_metrics.efficiency() * 100.0});
    
    // Identify bottlenecks
    std.debug.print("\n=== Bottleneck Analysis ===\n", .{});
    
    if (tiny_metrics.efficiency() < 0.5) {
        std.debug.print("🔍 ISSUE: Tiny task efficiency is very low ({:.1}%)\n", .{tiny_metrics.efficiency() * 100.0});
        std.debug.print("   Likely causes:\n", .{});
        std.debug.print("   - Task submission overhead dominates execution time\n", .{});
        std.debug.print("   - Work-stealing overhead too high for small tasks\n", .{});
        std.debug.print("   - Queue contention in lock-free operations\n", .{});
    }
    
    if (small_metrics.efficiency() < 0.6) {
        std.debug.print("🔍 ISSUE: Small task efficiency is low ({:.1}%)\n", .{small_metrics.efficiency() * 100.0});
        std.debug.print("   Likely causes:\n", .{});
        std.debug.print("   - Memory allocation overhead for task pointers\n", .{});
        std.debug.print("   - CAS operation overhead in steal attempts\n", .{});
        std.debug.print("   - Worker thread sleep/wake cycles\n", .{});
    }
    
    // Calculate steal effectiveness
    const total_steals = tiny_metrics.tasks_stolen + small_metrics.tasks_stolen + 
                        medium_metrics.tasks_stolen + large_metrics.tasks_stolen;
    const total_tasks = tiny_metrics.tasks_submitted + small_metrics.tasks_submitted + 
                       medium_metrics.tasks_submitted + large_metrics.tasks_submitted;
    
    if (total_tasks > 0) {
        const overall_steal_rate = @as(f64, @floatFromInt(total_steals)) / @as(f64, @floatFromInt(total_tasks)) * 100.0;
        std.debug.print("\nOverall steal rate: {:.1}%\n", .{overall_steal_rate});
        
        if (overall_steal_rate < 10.0) {
            std.debug.print("🔍 ISSUE: Low steal rate suggests:\n", .{});
            std.debug.print("   - Workers not finding enough work to steal\n", .{});
            std.debug.print("   - Task distribution is already balanced\n", .{});
            std.debug.print("   - Stealing mechanism is not aggressive enough\n", .{});
        } else if (overall_steal_rate > 50.0) {
            std.debug.print("🔍 ISSUE: High steal rate suggests:\n", .{});
            std.debug.print("   - Poor initial task distribution\n", .{});
            std.debug.print("   - Workers spending too much time stealing\n", .{});
            std.debug.print("   - Need better load balancing in submit()\n", .{});
        }
    }
}

pub fn main() !void {
    try analyzeWorkStealingBottlenecks();
    
    std.debug.print("\n=== Recommended Optimizations ===\n", .{});
    std.debug.print("1. Implement task batching for small tasks\n", .{});
    std.debug.print("2. Add adaptive stealing based on task size\n", .{});
    std.debug.print("3. Optimize CAS operations in steal() path\n", .{});
    std.debug.print("4. Reduce memory allocations for task pointers\n", .{});
    std.debug.print("5. Implement exponential backoff in worker idle loop\n", .{});
    std.debug.print("6. Add NUMA-aware stealing order\n", .{});
    std.debug.print("7. Optimize queue operations for cache efficiency\n", .{});
}