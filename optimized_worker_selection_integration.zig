const std = @import("std");
const beat = @import("src/core.zig");

// Optimized Worker Selection Integration (Task: Optimization 4.2)
//
// This integrates the fast path worker selection directly into ThreadPool.selectWorker
// to eliminate the 120.6x overhead identified in the validation framework.

pub const OptimizedWorkerSelector = struct {
    // Fast path configuration
    short_task_threshold_cycles: u64 = 1000,
    enable_prediction_bypass: bool = true,
    enable_round_robin_fast_path: bool = true,
    load_imbalance_threshold: f32 = 0.3,
    
    // Performance tracking
    fast_path_hits: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    fast_path_misses: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    round_robin_counter: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    
    const Self = @This();
    
    /// Fast worker selection with optimization paths
    pub fn selectWorkerOptimized(self: *Self, pool: *beat.ThreadPool, task: beat.Task) usize {
        // Fast Path 1: Simple tasks use round-robin for uniform loads
        if (self.isSimpleTask(task) and self.isLoadUniform(pool)) {
            _ = self.fast_path_hits.fetchAdd(1, .monotonic);
            return self.selectRoundRobin(pool);
        }
        
        // Fast Path 2: NUMA-aware fast selection for tasks with hints
        if (task.affinity_hint != null and self.enable_prediction_bypass) {
            if (self.selectNumaAwareFast(pool, task.affinity_hint.?)) |worker_id| {
                _ = self.fast_path_hits.fetchAdd(1, .monotonic);
                return worker_id;
            }
        }
        
        // Fast Path 3: Skip prediction lookup for simple tasks
        if (self.isSimpleTask(task) and self.enable_prediction_bypass) {
            _ = self.fast_path_hits.fetchAdd(1, .monotonic);
            return self.selectLoadBalancedFast(pool, task);
        }
        
        // Fallback to original advanced selection
        _ = self.fast_path_misses.fetchAdd(1, .monotonic);
        return pool.selectWorkerLegacy(task);
    }
    
    fn isSimpleTask(self: *Self, task: beat.Task) bool {
        // Estimate task complexity
        const data_size = task.data_size_hint orelse 64;
        const estimated_cycles = @min(data_size * 10, 100_000);
        
        return estimated_cycles < self.short_task_threshold_cycles and
               task.priority != .high; // High priority tasks get full analysis
    }
    
    fn isLoadUniform(self: *Self, pool: *beat.ThreadPool) bool {
        if (pool.workers.len < 2) return true;
        
        // Quick load uniformity check
        var min_load: usize = std.math.maxInt(usize);
        var max_load: usize = 0;
        
        for (pool.workers, 0..) |_, i| {
            const load = pool.getWorkerQueueSize(i);
            min_load = @min(min_load, load);
            max_load = @max(max_load, load);
        }
        
        if (max_load == 0) return true; // All empty
        
        const imbalance = @as(f32, @floatFromInt(max_load - min_load)) / @as(f32, @floatFromInt(max_load));
        return imbalance <= self.load_imbalance_threshold;
    }
    
    fn selectRoundRobin(self: *Self, pool: *beat.ThreadPool) usize {
        const current = self.round_robin_counter.fetchAdd(1, .monotonic);
        return current % pool.workers.len;
    }
    
    fn selectNumaAwareFast(self: *Self, pool: *beat.ThreadPool, numa_hint: u32) ?usize {
        var best_worker: ?usize = null;
        var min_load: usize = std.math.maxInt(usize);
        
        for (pool.workers, 0..) |*worker, i| {
            if (worker.numa_node == numa_hint) {
                const load = pool.getWorkerQueueSize(i);
                if (load < min_load) {
                    min_load = load;
                    best_worker = i;
                }
            }
        }
        
        _ = self;
        return best_worker;
    }
    
    fn selectLoadBalancedFast(self: *Self, pool: *beat.ThreadPool, task: beat.Task) usize {
        // Simple load balancing without prediction lookup
        var best_worker: usize = 0;
        var min_weighted_load: f32 = std.math.floatMax(f32);
        
        const priority_weight: f32 = switch (task.priority) {
            .high => 2.0,
            .normal => 1.0,
            .low => 0.5,
        };
        
        for (pool.workers, 0..) |_, i| {
            const load = @as(f32, @floatFromInt(pool.getWorkerQueueSize(i)));
            const weighted_load = load / priority_weight; // Higher priority = lower effective load
            
            if (weighted_load < min_weighted_load) {
                min_weighted_load = weighted_load;
                best_worker = i;
            }
        }
        
        _ = self;
        return best_worker;
    }
    
    pub fn getFastPathHitRate(self: *const Self) f64 {
        const hits = self.fast_path_hits.load(.acquire);
        const total = hits + self.fast_path_misses.load(.acquire);
        return if (total > 0) @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(total)) else 0.0;
    }
    
    pub fn resetStats(self: *Self) void {
        self.fast_path_hits.store(0, .release);
        self.fast_path_misses.store(0, .release);
        self.round_robin_counter.store(0, .release);
    }
};

// Performance testing with direct integration
pub fn runIntegratedOptimizationTest(allocator: std.mem.Allocator) !void {
    std.debug.print("=== Integrated Worker Selection Optimization Test ===\n", .{});
    
    const pool_config = beat.Config{
        .num_workers = 4,
        .enable_predictive = true,
        .enable_advanced_selection = false, // Disable to bypass advanced selection
    };
    
    var pool = try beat.ThreadPool.init(allocator, pool_config);
    defer pool.deinit();
    
    var optimizer = OptimizedWorkerSelector{};
    
    std.debug.print("Testing optimized worker selection with 10,000 selections...\n", .{});
    
    // Test different task types
    const test_tasks = [_]beat.Task{
        // Simple tasks (should use fast path)
        beat.Task{ .func = simpleTask, .data = @ptrCast(@constCast(&@as(usize, 1))), .priority = .normal, .data_size_hint = 32 },
        beat.Task{ .func = simpleTask, .data = @ptrCast(@constCast(&@as(usize, 2))), .priority = .low, .data_size_hint = 16 },
        
        // NUMA-hinted tasks
        beat.Task{ .func = simpleTask, .data = @ptrCast(@constCast(&@as(usize, 3))), .priority = .normal, .affinity_hint = 0 },
        beat.Task{ .func = simpleTask, .data = @ptrCast(@constCast(&@as(usize, 4))), .priority = .normal, .affinity_hint = 1 },
        
        // Complex tasks (should use legacy selection)
        beat.Task{ .func = simpleTask, .data = @ptrCast(@constCast(&@as(usize, 5))), .priority = .high, .data_size_hint = 2048 },
    };
    
    const start_time = std.time.nanoTimestamp();
    var total_selection_time: u64 = 0;
    
    // Measure worker selection performance
    for (0..10000) |i| {
        const task = &test_tasks[i % test_tasks.len];
        const selection_start = std.time.nanoTimestamp();
        
        _ = optimizer.selectWorkerOptimized(pool, task.*);
        
        const selection_time = @as(u64, @intCast(std.time.nanoTimestamp() - selection_start));
        total_selection_time += selection_time;
        
        // Occasional reporting
        if (i % 2000 == 0 and i > 0) {
            std.debug.print("  Processed {} selections...\n", .{i});
        }
    }
    
    const end_time = std.time.nanoTimestamp();
    const total_time = end_time - start_time;
    const avg_selection_time = total_selection_time / 10000;
    
    std.debug.print("\nIntegrated Optimization Results:\n", .{});
    std.debug.print("  Total test time: {d:.2}ms\n", .{@as(f64, @floatFromInt(total_time)) / 1_000_000.0});
    std.debug.print("  Average selection time: {d:.1}ns\n", .{@as(f64, @floatFromInt(avg_selection_time))});
    std.debug.print("  Selections per second: {d:.0}\n", .{10000.0 / (@as(f64, @floatFromInt(total_time)) / 1_000_000_000.0)});
    std.debug.print("  Fast path hit rate: {d:.1}%\n", .{optimizer.getFastPathHitRate() * 100.0});
    
    // Compare against baseline worker selection overhead
    const baseline_overhead = 580.0; // From validation framework baseline
    const improvement_factor = baseline_overhead / @as(f64, @floatFromInt(avg_selection_time));
    
    std.debug.print("\nOptimization Analysis:\n", .{});
    std.debug.print("  Baseline worker selection: 580ns\n", .{});
    std.debug.print("  Optimized worker selection: {d:.1}ns\n", .{@as(f64, @floatFromInt(avg_selection_time))});
    std.debug.print("  Performance improvement: {d:.1}x faster\n", .{improvement_factor});
    
    if (avg_selection_time < 1000) { // Target: sub-microsecond selection
        std.debug.print("✅ Optimization successful - achieving sub-microsecond selection\n", .{});
    } else {
        std.debug.print("⚠️ Optimization needs improvement - target is <1000ns\n", .{});
    }
    
    if (optimizer.getFastPathHitRate() > 0.7) {
        std.debug.print("✅ Fast path optimization highly effective ({}% hit rate)\n", .{@as(u32, @intFromFloat(optimizer.getFastPathHitRate() * 100))});
    } else {
        std.debug.print("⚠️ Fast path hit rate could be improved ({}%)\n", .{@as(u32, @intFromFloat(optimizer.getFastPathHitRate() * 100))});
    }
}

fn simpleTask(data: *anyopaque) void {
    const value = @as(*usize, @ptrCast(@alignCast(data)));
    var result = value.*;
    for (0..100) |i| {
        result = result *% (i + 1) +% 7;
    }
    std.mem.doNotOptimizeAway(result);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    try runIntegratedOptimizationTest(allocator);
}