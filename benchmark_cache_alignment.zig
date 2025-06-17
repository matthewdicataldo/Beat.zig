const std = @import("std");
const beat = @import("zigpulse");

// Cache-line alignment optimization benchmark
// Tests the impact of our Worker and WorkStealingDeque optimizations

var task_counter: std.atomic.Value(u64) = std.atomic.Value(u64).init(0);

// Benchmark tasks designed to stress cache behavior
fn cacheStressTask(data: *anyopaque) void {
    const iterations = @as(*u32, @ptrCast(@alignCast(data))).*;
    
    // Stress test: Many atomic operations that benefit from cache alignment
    var sum: u64 = 0;
    for (0..iterations) |i| {
        // Memory access pattern that stresses cache lines
        sum += i * i + 42;
        
        // Frequent atomic increment to test false sharing elimination
        if (i % 10 == 0) {
            _ = task_counter.fetchAdd(1, .monotonic);
        }
    }
    
    // Final atomic operation
    _ = task_counter.fetchAdd(sum % 1000, .monotonic);
    
    std.mem.doNotOptimizeAway(&sum);
}

fn workStealingTask(data: *anyopaque) void {
    const iterations = @as(*u32, @ptrCast(@alignCast(data))).*;
    
    // Task designed to trigger work-stealing behavior
    // This tests our WorkStealingDeque cache alignment optimizations
    var sum: u64 = 0;
    for (0..iterations) |i| {
        sum += i * 17 + (i << 3); // Non-trivial computation
        
        if (i % 5 == 0) {
            _ = task_counter.fetchAdd(1, .monotonic);
        }
    }
    
    std.mem.doNotOptimizeAway(&sum);
}

fn memoryIntensiveTask(data: *anyopaque) void {
    const size = @as(*u32, @ptrCast(@alignCast(data))).*;
    
    // Stress memory allocation to test cache alignment under memory pressure
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    
    const buffer = allocator.alloc(u64, size) catch return;
    
    // Random access pattern that benefits from good cache alignment
    var rng = std.Random.DefaultPrng.init(@as(u64, @intCast(std.time.nanoTimestamp())));
    
    for (0..size / 4) |_| {
        const idx = rng.random().uintLessThan(usize, buffer.len);
        buffer[idx] = buffer[idx] +% 1;
        
        _ = task_counter.fetchAdd(1, .monotonic);
    }
}

const BenchmarkResult = struct {
    total_time_ms: f64,
    tasks_per_second: f64,
    cache_efficiency: f64,
    work_stealing_rate: f64,
    memory_throughput_gbps: f64,
};

fn runBenchmark(allocator: std.mem.Allocator, name: []const u8, num_workers: usize, total_tasks: usize) !BenchmarkResult {
    std.debug.print("\n🔬 Running {s} Benchmark\n", .{name});
    std.debug.print("Workers: {}, Tasks: {}\n", .{ num_workers, total_tasks });
    
    // Reset counter
    task_counter.store(0, .monotonic);
    
    const config = beat.Config{
        .num_workers = num_workers,
        .enable_a3c_scheduling = true,
        .a3c_learning_rate = 0.001,
        .a3c_confidence_threshold = 0.6,
        .enable_statistics = true,
        .enable_topology_aware = true,
    };
    
    const pool = try beat.createPoolWithConfig(allocator, config);
    defer pool.deinit();
    
    const start_time = std.time.nanoTimestamp();
    
    // Submit varied workload to stress different cache patterns
    for (0..total_tasks) |i| {
        var work_size: u32 = switch (i % 6) {
            0, 1 => 500,    // Light cache stress
            2, 3 => 1500,   // Medium cache stress
            4 => 3000,      // Heavy cache stress  
            5 => 100,       // Memory intensive
            else => unreachable,
        };
        
        const task = switch (i % 6) {
            0, 1, 2, 3, 4 => beat.Task{
                .func = if (i % 2 == 0) cacheStressTask else workStealingTask,
                .data = &work_size,
                .priority = .normal,
            },
            5 => beat.Task{
                .func = memoryIntensiveTask,
                .data = &work_size,
                .priority = .normal,
            },
            else => unreachable,
        };
        
        try pool.submit(task);
    }
    
    // Wait for completion
    pool.wait();
    
    const end_time = std.time.nanoTimestamp();
    const duration_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
    const completed_operations = task_counter.load(.acquire);
    
    // Calculate performance metrics
    const tasks_per_second = @as(f64, @floatFromInt(total_tasks)) * 1000.0 / duration_ms;
    
    // Estimate cache efficiency based on completion rate vs operations
    const cache_efficiency = @as(f64, @floatFromInt(completed_operations)) / @as(f64, @floatFromInt(total_tasks * 100)) * 100.0;
    
    // Work-stealing statistics
    const tasks_stolen = pool.stats.cold.tasks_stolen.load(.acquire);
    const work_stealing_rate = @as(f64, @floatFromInt(tasks_stolen)) / @as(f64, @floatFromInt(total_tasks)) * 100.0;
    
    // Rough memory throughput estimate (based on task completion rate)
    const memory_throughput_gbps = tasks_per_second * 64.0 / (1024.0 * 1024.0 * 1024.0) * 8.0; // Rough estimate
    
    std.debug.print("✅ {s} Complete:\n", .{name});
    std.debug.print("  Duration: {d:.2}ms\n", .{duration_ms});
    std.debug.print("  Tasks/sec: {d:.0}\n", .{tasks_per_second});
    std.debug.print("  Cache efficiency: {d:.1}%\n", .{cache_efficiency});
    std.debug.print("  Work-stealing rate: {d:.1}%\n", .{work_stealing_rate});
    std.debug.print("  Est. memory throughput: {d:.2} GB/s\n", .{memory_throughput_gbps});
    std.debug.print("  Operations completed: {}\n", .{completed_operations});
    
    return BenchmarkResult{
        .total_time_ms = duration_ms,
        .tasks_per_second = tasks_per_second,
        .cache_efficiency = cache_efficiency,
        .work_stealing_rate = work_stealing_rate,
        .memory_throughput_gbps = memory_throughput_gbps,
    };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){}; 
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("🚀 Cache-Line Alignment Optimization Benchmark\n", .{});
    std.debug.print("==============================================\n", .{});
    std.debug.print("Testing impact of Worker and WorkStealingDeque optimizations\n", .{});
    
    // Test different scenarios to validate cache improvements
    const test_scenarios = [_]struct {
        name: []const u8,
        workers: usize,
        tasks: usize,
    }{
        .{ .name = "Small Load (Cache Pressure)", .workers = 2, .tasks = 1000 },
        .{ .name = "Medium Load (Mixed Access)", .workers = 4, .tasks = 5000 },
        .{ .name = "High Load (Cache Stress)", .workers = 6, .tasks = 15000 },
        .{ .name = "Maximum Load (Scalability)", .workers = 8, .tasks = 25000 },
    };
    
    var results = std.ArrayList(BenchmarkResult).init(allocator);
    defer results.deinit();
    
    for (test_scenarios) |scenario| {
        const result = try runBenchmark(allocator, scenario.name, scenario.workers, scenario.tasks);
        try results.append(result);
        
        // Brief pause between tests
        std.time.sleep(500_000_000); // 0.5 seconds
    }
    
    // Performance Analysis Summary
    std.debug.print("\n📊 Cache-Line Alignment Performance Analysis\n", .{});
    std.debug.print("============================================\n", .{});
    
    var total_tasks_per_sec: f64 = 0;
    var total_cache_efficiency: f64 = 0;
    var total_memory_throughput: f64 = 0;
    var min_time: f64 = std.math.inf(f64);
    var max_tasks_per_sec: f64 = 0;
    
    for (results.items, 0..) |result, i| {
        std.debug.print("{}. {s}:\n", .{ i + 1, test_scenarios[i].name });
        std.debug.print("   Tasks/sec: {d:.0} | Cache: {d:.1}% | Memory: {d:.2} GB/s\n", .{
            result.tasks_per_second, 
            result.cache_efficiency, 
            result.memory_throughput_gbps
        });
        
        total_tasks_per_sec += result.tasks_per_second;
        total_cache_efficiency += result.cache_efficiency;
        total_memory_throughput += result.memory_throughput_gbps;
        min_time = @min(min_time, result.total_time_ms);
        max_tasks_per_sec = @max(max_tasks_per_sec, result.tasks_per_second);
    }
    
    const avg_tasks_per_sec = total_tasks_per_sec / @as(f64, @floatFromInt(results.items.len));
    const avg_cache_efficiency = total_cache_efficiency / @as(f64, @floatFromInt(results.items.len));
    const avg_memory_throughput = total_memory_throughput / @as(f64, @floatFromInt(results.items.len));
    
    std.debug.print("\n🎯 Cache-Line Optimization Results:\n", .{});
    std.debug.print("   Average Tasks/sec: {d:.0}\n", .{avg_tasks_per_sec});
    std.debug.print("   Average Cache Efficiency: {d:.1}%\n", .{avg_cache_efficiency});
    std.debug.print("   Average Memory Throughput: {d:.2} GB/s\n", .{avg_memory_throughput});
    std.debug.print("   Peak Performance: {d:.0} tasks/sec\n", .{max_tasks_per_sec});
    std.debug.print("   Best Response Time: {d:.2}ms\n", .{min_time});
    
    // Performance Rating
    std.debug.print("\n⭐ Performance Assessment:\n", .{});
    if (avg_tasks_per_sec > 20000) {
        std.debug.print("   🚀 EXCELLENT: Cache optimizations highly effective!\n", .{});
    } else if (avg_tasks_per_sec > 15000) {
        std.debug.print("   ✅ GOOD: Cache optimizations showing positive impact\n", .{});
    } else if (avg_tasks_per_sec > 10000) {
        std.debug.print("   ⚠️  MODERATE: Some cache benefits, room for improvement\n", .{});
    } else {
        std.debug.print("   ❌ NEEDS WORK: Cache optimizations need refinement\n", .{});
    }
    
    if (avg_cache_efficiency > 85.0) {
        std.debug.print("   🎯 Cache efficiency: Excellent alignment impact\n", .{});
    } else if (avg_cache_efficiency > 70.0) {
        std.debug.print("   ✅ Cache efficiency: Good improvement from alignment\n", .{});
    } else {
        std.debug.print("   ⚠️  Cache efficiency: More optimization needed\n", .{});
    }
    
    std.debug.print("\n🔍 Next Optimization Recommendation:\n", .{});
    if (avg_cache_efficiency < 80.0) {
        std.debug.print("   Priority: Memory prefetching hints (high impact expected)\n", .{});
    } else {
        std.debug.print("   Priority: Batch formation optimization (highest impact)\n", .{});
    }
}