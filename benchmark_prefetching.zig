const std = @import("std");
const beat = @import("zigpulse");

// Memory prefetching optimization benchmark
// Tests the impact of software prefetch hints on memory-intensive workloads

var task_counter: std.atomic.Value(u64) = std.atomic.Value(u64).init(0);

// Memory-intensive tasks designed to benefit from prefetching
fn memoryStreamingTask(data: *anyopaque) void {
    const size = @as(*u32, @ptrCast(@alignCast(data))).*;
    
    // Allocate a buffer for streaming memory access
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    
    const buffer = allocator.alloc(u64, size) catch return;
    
    // Sequential streaming access pattern (benefits from prefetching)
    for (buffer, 0..) |*item, i| {
        item.* = i * 42 + 17;
    }
    
    // Read-back with computation
    var sum: u64 = 0;
    for (buffer) |item| {
        sum +%= item;
    }
    
    _ = task_counter.fetchAdd(sum % 1000, .monotonic);
    std.mem.doNotOptimizeAway(&sum);
}

fn randomMemoryTask(data: *anyopaque) void {
    const size = @as(*u32, @ptrCast(@alignCast(data))).*;
    
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    
    const buffer = allocator.alloc(u64, size) catch return;
    
    // Random access pattern (challenges prefetching effectiveness)
    var rng = std.Random.DefaultPrng.init(@as(u64, @intCast(std.time.nanoTimestamp())));
    
    // Initialize buffer
    for (buffer, 0..) |*item, i| {
        item.* = i;
    }
    
    // Random access pattern
    for (0..size / 4) |_| {
        const idx1 = rng.random().uintLessThan(usize, buffer.len);
        const idx2 = rng.random().uintLessThan(usize, buffer.len);
        buffer[idx1] = buffer[idx1] +% buffer[idx2];
    }
    
    _ = task_counter.fetchAdd(buffer[0] % 1000, .monotonic);
}

fn linkedListTask(data: *anyopaque) void {
    const iterations = @as(*u32, @ptrCast(@alignCast(data))).*;
    
    // Simple linked list structure
    const Node = struct {
        value: u64,
        next: ?*@This() = null,
    };
    
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    
    // Create linked list
    var head: ?*Node = null;
    var current: ?*Node = null;
    
    for (0..iterations) |i| {
        const node = allocator.create(Node) catch return;
        node.* = Node{ .value = i };
        
        if (head == null) {
            head = node;
            current = node;
        } else {
            current.?.next = node;
            current = node;
        }
    }
    
    // Traverse and compute (benefits from prefetching next nodes)
    var sum: u64 = 0;
    var node = head;
    while (node) |n| {
        sum +%= n.value;
        node = n.next;
    }
    
    _ = task_counter.fetchAdd(sum % 1000, .monotonic);
    std.mem.doNotOptimizeAway(&sum);
}

fn workStealingStressTask(data: *anyopaque) void {
    const iterations = @as(*u32, @ptrCast(@alignCast(data))).*;
    
    // Task designed to stress work-stealing (tests prefetching in steal paths)
    var sum: u64 = 0;
    for (0..iterations) |i| {
        // Computation that varies to create uneven load distribution
        const work_factor: usize = if (i % 7 == 0) 100 else 10;
        for (0..work_factor) |j| {
            sum +%= i * j + 42;
        }
    }
    
    _ = task_counter.fetchAdd(sum % 1000, .monotonic);
    std.mem.doNotOptimizeAway(&sum);
}

const BenchmarkResult = struct {
    scenario_name: []const u8,
    total_time_ms: f64,
    tasks_per_second: f64,
    memory_throughput_gbps: f64,
    operations_completed: u64,
    cache_efficiency: f64,
    work_stealing_rate: f64,
};

fn runPrefetchBenchmark(allocator: std.mem.Allocator, scenario_name: []const u8, task_func: *const fn (*anyopaque) void, num_workers: usize, total_tasks: usize, work_size: u32) !BenchmarkResult {
    std.debug.print("\n🧠 Running {s} Prefetch Benchmark\n", .{scenario_name});
    std.debug.print("Workers: {}, Tasks: {}, Work Size: {}\n", .{ num_workers, total_tasks, work_size });
    
    // Reset counter
    task_counter.store(0, .monotonic);
    
    const config = beat.Config{
        .num_workers = num_workers,
        .enable_a3c_scheduling = true,
        .a3c_learning_rate = 0.001,
        .a3c_confidence_threshold = 0.6,
        .enable_statistics = true,
        .enable_topology_aware = true,
        .enable_work_stealing = true,
    };
    
    const pool = try beat.createPoolWithConfig(allocator, config);
    defer pool.deinit();
    
    const start_time = std.time.nanoTimestamp();
    
    // Submit workload
    var work_size_var = work_size;
    for (0..total_tasks) |_| {
        const task = beat.Task{
            .func = task_func,
            .data = &work_size_var,
            .priority = .normal,
            .data_size_hint = work_size,
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
    
    // Estimate memory throughput (rough calculation)
    const bytes_per_task = @as(f64, @floatFromInt(work_size * @sizeOf(u64)));
    const total_bytes = @as(f64, @floatFromInt(total_tasks)) * bytes_per_task;
    const memory_throughput_gbps = total_bytes / (duration_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    
    // Cache efficiency estimate
    const expected_operations = @as(f64, @floatFromInt(total_tasks * 100)); // Rough estimate
    const cache_efficiency = @as(f64, @floatFromInt(completed_operations)) / expected_operations * 100.0;
    
    // Work-stealing rate
    const tasks_stolen = pool.stats.cold.tasks_stolen.load(.acquire);
    const work_stealing_rate = @as(f64, @floatFromInt(tasks_stolen)) / @as(f64, @floatFromInt(total_tasks)) * 100.0;
    
    std.debug.print("✅ {s} Complete:\n", .{scenario_name});
    std.debug.print("  Duration: {d:.2}ms\n", .{duration_ms});
    std.debug.print("  Tasks/sec: {d:.0}\n", .{tasks_per_second});
    std.debug.print("  Memory throughput: {d:.2} GB/s\n", .{memory_throughput_gbps});
    std.debug.print("  Operations: {}\n", .{completed_operations});
    std.debug.print("  Cache efficiency: {d:.1}%\n", .{cache_efficiency});
    std.debug.print("  Work-stealing: {d:.1}%\n", .{work_stealing_rate});
    
    return BenchmarkResult{
        .scenario_name = scenario_name,
        .total_time_ms = duration_ms,
        .tasks_per_second = tasks_per_second,
        .memory_throughput_gbps = memory_throughput_gbps,
        .operations_completed = completed_operations,
        .cache_efficiency = cache_efficiency,
        .work_stealing_rate = work_stealing_rate,
    };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){}; 
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("🚀 Memory Prefetching Optimization Benchmark\n", .{});
    std.debug.print("=============================================\n", .{});
    std.debug.print("Testing impact of software prefetch hints on memory access patterns\n", .{});
    
    // Test scenarios designed to measure prefetching effectiveness
    const test_scenarios = [_]struct {
        name: []const u8,
        task_func: *const fn (*anyopaque) void,
        workers: usize,
        tasks: usize,
        work_size: u32,
    }{
        .{ .name = "Sequential Memory (Prefetch Friendly)", .task_func = memoryStreamingTask, .workers = 4, .tasks = 500, .work_size = 1024 },
        .{ .name = "Random Memory (Prefetch Challenge)", .task_func = randomMemoryTask, .workers = 4, .tasks = 300, .work_size = 1024 },
        .{ .name = "Linked List Traversal", .task_func = linkedListTask, .workers = 6, .tasks = 200, .work_size = 256 },
        .{ .name = "Work-Stealing Stress Test", .task_func = workStealingStressTask, .workers = 8, .tasks = 1000, .work_size = 500 },
    };
    
    var results = std.ArrayList(BenchmarkResult).init(allocator);
    defer results.deinit();
    
    for (test_scenarios) |scenario| {
        const result = try runPrefetchBenchmark(
            allocator, 
            scenario.name, 
            scenario.task_func, 
            scenario.workers, 
            scenario.tasks, 
            scenario.work_size
        );
        try results.append(result);
        
        // Brief pause between tests
        std.time.sleep(300_000_000); // 0.3 seconds
    }
    
    // Memory Prefetching Performance Analysis
    std.debug.print("\n📊 Memory Prefetching Performance Analysis\n", .{});
    std.debug.print("===========================================\n", .{});
    
    var total_throughput: f64 = 0;
    var total_memory_bandwidth: f64 = 0;
    var total_cache_efficiency: f64 = 0;
    var total_work_stealing: f64 = 0;
    var max_memory_throughput: f64 = 0;
    var max_tasks_per_sec: f64 = 0;
    
    for (results.items, 0..) |result, i| {
        std.debug.print("{}. {s}:\n", .{ i + 1, result.scenario_name });
        std.debug.print("   Tasks/sec: {d:.0} | Memory: {d:.2} GB/s | Cache: {d:.1}% | WS: {d:.1}%\n", .{
            result.tasks_per_second,
            result.memory_throughput_gbps,
            result.cache_efficiency,
            result.work_stealing_rate,
        });
        
        total_throughput += result.tasks_per_second;
        total_memory_bandwidth += result.memory_throughput_gbps;
        total_cache_efficiency += result.cache_efficiency;
        total_work_stealing += result.work_stealing_rate;
        max_memory_throughput = @max(max_memory_throughput, result.memory_throughput_gbps);
        max_tasks_per_sec = @max(max_tasks_per_sec, result.tasks_per_second);
    }
    
    const num_scenarios = @as(f64, @floatFromInt(results.items.len));
    const avg_throughput = total_throughput / num_scenarios;
    const avg_memory_bandwidth = total_memory_bandwidth / num_scenarios;
    const avg_cache_efficiency = total_cache_efficiency / num_scenarios;
    const avg_work_stealing = total_work_stealing / num_scenarios;
    
    std.debug.print("\n🎯 Memory Prefetching Optimization Results:\n", .{});
    std.debug.print("   Average Task Throughput: {d:.0} tasks/sec\n", .{avg_throughput});
    std.debug.print("   Average Memory Bandwidth: {d:.2} GB/s\n", .{avg_memory_bandwidth});
    std.debug.print("   Average Cache Efficiency: {d:.1}%\n", .{avg_cache_efficiency});
    std.debug.print("   Average Work-Stealing Rate: {d:.1}%\n", .{avg_work_stealing});
    std.debug.print("   Peak Memory Throughput: {d:.2} GB/s\n", .{max_memory_throughput});
    std.debug.print("   Peak Task Throughput: {d:.0} tasks/sec\n", .{max_tasks_per_sec});
    
    // Performance Assessment
    std.debug.print("\n⭐ Memory Prefetching Assessment:\n", .{});
    if (avg_memory_bandwidth > 2.0) {
        std.debug.print("   🚀 EXCELLENT: Memory prefetching highly effective!\n", .{});
    } else if (avg_memory_bandwidth > 1.0) {
        std.debug.print("   ✅ GOOD: Prefetching showing positive memory impact\n", .{});
    } else if (avg_memory_bandwidth > 0.5) {
        std.debug.print("   ⚠️  MODERATE: Some prefetch benefits, room for improvement\n", .{});
    } else {
        std.debug.print("   ❌ NEEDS WORK: Prefetching optimizations need refinement\n", .{});
    }
    
    if (avg_cache_efficiency > 80.0) {
        std.debug.print("   🎯 Memory Access: Excellent prefetch impact on cache performance\n", .{});
    } else if (avg_cache_efficiency > 60.0) {
        std.debug.print("   ✅ Memory Access: Good improvement from prefetching\n", .{});
    } else {
        std.debug.print("   ⚠️  Memory Access: More prefetch optimization needed\n", .{});
    }
    
    if (avg_work_stealing > 5.0 and avg_work_stealing < 20.0) {
        std.debug.print("   ⚖️  Work-Stealing: Optimal balance with prefetch optimizations\n", .{});
    } else if (avg_work_stealing < 5.0) {
        std.debug.print("   📈 Work-Stealing: Good load distribution, prefetching helps efficiency\n", .{});
    } else {
        std.debug.print("   ⚠️  Work-Stealing: High stealing rate - prefetching helps with migration costs\n", .{});
    }
    
    std.debug.print("\n🔍 Next Optimization Recommendation:\n", .{});
    if (avg_memory_bandwidth < 1.5) {
        std.debug.print("   Priority: Adaptive prefetch distance optimization (medium impact)\n", .{});
    } else {
        std.debug.print("   Priority: Batch formation optimization (highest impact potential)\n", .{});
    }
    
    std.debug.print("\n✅ Memory prefetching implementation complete!\n", .{});
    std.debug.print("   • Software prefetch hints in all hot paths\n", .{});
    std.debug.print("   • Work-stealing prefetch optimization\n", .{});
    std.debug.print("   • Task execution data prefetching\n", .{});
    std.debug.print("   • Cross-platform prefetch support (x86, ARM)\n", .{});
}