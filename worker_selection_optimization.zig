const std = @import("std");
const beat = @import("src/core.zig");

// Worker Selection Fast Path Optimization (Task: Optimization 4)
//
// Based on validation framework findings showing 120.6x overhead in worker selection,
// this implements fast paths for different task categories to minimize scheduling overhead.

pub const FastPathWorkerSelector = struct {
    base_pool: *beat.ThreadPool,
    allocator: std.mem.Allocator,
    
    // Fast path configuration
    fast_path_config: FastPathConfig,
    
    // Performance tracking
    fast_path_stats: FastPathStats,
    
    // Worker load tracking for fast decisions
    worker_loads: []std.atomic.Value(u32),
    last_round_robin: std.atomic.Value(usize),
    
    const Self = @This();
    
    pub const FastPathConfig = struct {
        short_task_threshold_ns: u64 = 10_000,     // Tasks < 10μs estimated time
        simple_task_threshold_cycles: u64 = 1000,  // Tasks < 1000 cycles
        load_imbalance_threshold: f32 = 0.3,       // When to skip load balancing (30% imbalance)
        enable_prediction_bypass: bool = true,     // Skip predictions for simple tasks
        enable_round_robin_fast_path: bool = true, // Use round-robin for uniform loads
        numa_affinity_preference: bool = true,     // Prefer local NUMA but don't force
    };
    
    pub const FastPathStats = struct {
        total_selections: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        fast_path_hits: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        round_robin_selections: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        load_balanced_selections: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        prediction_bypassed: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        numa_local_placements: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        
        pub fn getFastPathHitRate(self: *const FastPathStats) f64 {
            const total = self.total_selections.load(.acquire);
            const hits = self.fast_path_hits.load(.acquire);
            return if (total > 0) @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(total)) else 0.0;
        }
    };
    
    pub const SelectionResult = struct {
        worker_id: usize,
        selection_method: SelectionMethod,
        time_taken_ns: u64,
        bypassed_prediction: bool,
        used_numa_hint: bool,
    };
    
    pub const SelectionMethod = enum {
        round_robin_fast,        // Fast round-robin for uniform loads
        numa_aware_fast,         // Fast NUMA-aware selection
        load_balanced_fast,      // Fast load balancing
        advanced_fallback,       // Fallback to advanced selection
        legacy_fallback,         // Fallback to legacy selection
    };
    
    pub fn init(allocator: std.mem.Allocator, base_pool: *beat.ThreadPool, config: FastPathConfig) !Self {
        const worker_count = base_pool.workers.len;
        const worker_loads = try allocator.alloc(std.atomic.Value(u32), worker_count);
        
        // Initialize worker load tracking
        for (worker_loads) |*load| {
            load.* = std.atomic.Value(u32).init(0);
        }
        
        return Self{
            .base_pool = base_pool,
            .allocator = allocator,
            .fast_path_config = config,
            .fast_path_stats = FastPathStats{},
            .worker_loads = worker_loads,
            .last_round_robin = std.atomic.Value(usize).init(0),
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.worker_loads);
    }
    
    /// Fast worker selection with multiple optimization paths
    pub fn selectWorker(self: *Self, task: beat.Task) SelectionResult {
        const selection_start = std.time.nanoTimestamp();
        _ = self.fast_path_stats.total_selections.fetchAdd(1, .monotonic);
        
        // Step 1: Analyze task characteristics for fast path eligibility
        const task_analysis = self.analyzeTask(task);
        
        // Step 2: Check for fast path opportunities
        if (task_analysis.is_simple_task) {
            if (self.tryFastPath(task, task_analysis)) |result| {
                const selection_time = @as(u64, @intCast(std.time.nanoTimestamp() - selection_start));
                _ = self.fast_path_stats.fast_path_hits.fetchAdd(1, .monotonic);
                
                return SelectionResult{
                    .worker_id = result.worker_id,
                    .selection_method = result.method,
                    .time_taken_ns = selection_time,
                    .bypassed_prediction = result.bypassed_prediction,
                    .used_numa_hint = result.used_numa_hint,
                };
            }
        }
        
        // Step 3: Fallback to advanced or legacy selection
        const fallback_result = self.fallbackToAdvancedSelection(task);
        const selection_time = @as(u64, @intCast(std.time.nanoTimestamp() - selection_start));
        
        return SelectionResult{
            .worker_id = fallback_result,
            .selection_method = .advanced_fallback,
            .time_taken_ns = selection_time,
            .bypassed_prediction = false,
            .used_numa_hint = false,
        };
    }
    
    const TaskAnalysis = struct {
        is_simple_task: bool,
        estimated_cycles: u64,
        has_numa_hint: bool,
        numa_hint: u32,
        priority_weight: f32,
        requires_advanced_selection: bool,
    };
    
    fn analyzeTask(self: *Self, task: beat.Task) TaskAnalysis {
        // Quick task classification based on available hints
        const data_size = task.data_size_hint orelse 64;
        
        // Estimate complexity from data size and priority
        const estimated_cycles = @min(data_size * 10, 100_000); // Rough heuristic
        
        // Determine if this is a simple task eligible for fast path
        const is_simple = 
            estimated_cycles < self.fast_path_config.simple_task_threshold_cycles and
            task.priority != .high; // High priority tasks get advanced selection
        
        // Check for NUMA hints
        const has_numa_hint = task.affinity_hint != null;
        const numa_hint = task.affinity_hint orelse 0;
        
        // Priority weighting for load balancing
        const priority_weight: f32 = switch (task.priority) {
            .high => 2.0,
            .normal => 1.0,
            .low => 0.5,
        };
        
        // Determine if advanced selection is required
        const requires_advanced = 
            !is_simple or 
            task.priority == .high;
        
        return TaskAnalysis{
            .is_simple_task = is_simple,
            .estimated_cycles = estimated_cycles,
            .has_numa_hint = has_numa_hint,
            .numa_hint = numa_hint,
            .priority_weight = priority_weight,
            .requires_advanced_selection = requires_advanced,
        };
    }
    
    const FastPathResult = struct {
        worker_id: usize,
        method: SelectionMethod,
        bypassed_prediction: bool,
        used_numa_hint: bool,
    };
    
    fn tryFastPath(self: *Self, task: beat.Task, analysis: TaskAnalysis) ?FastPathResult {
        // Fast Path 1: Round-robin for uniform loads
        if (self.fast_path_config.enable_round_robin_fast_path) {
            if (self.isLoadUniform()) {
                const worker_id = self.selectRoundRobin();
                _ = self.fast_path_stats.round_robin_selections.fetchAdd(1, .monotonic);
                
                return FastPathResult{
                    .worker_id = worker_id,
                    .method = .round_robin_fast,
                    .bypassed_prediction = true,
                    .used_numa_hint = false,
                };
            }
        }
        
        // Fast Path 2: NUMA-aware fast selection
        if (analysis.has_numa_hint and self.fast_path_config.numa_affinity_preference) {
            if (self.selectNumaAwareFast(analysis.numa_hint)) |worker_id| {
                _ = self.fast_path_stats.numa_local_placements.fetchAdd(1, .monotonic);
                
                return FastPathResult{
                    .worker_id = worker_id,
                    .method = .numa_aware_fast,
                    .bypassed_prediction = self.fast_path_config.enable_prediction_bypass,
                    .used_numa_hint = true,
                };
            }
        }
        
        // Fast Path 3: Load-balanced fast selection (no prediction lookup)
        if (self.fast_path_config.enable_prediction_bypass) {
            const worker_id = self.selectLoadBalancedFast(analysis.priority_weight);
            _ = self.fast_path_stats.load_balanced_selections.fetchAdd(1, .monotonic);
            _ = self.fast_path_stats.prediction_bypassed.fetchAdd(1, .monotonic);
            
            return FastPathResult{
                .worker_id = worker_id,
                .method = .load_balanced_fast,
                .bypassed_prediction = true,
                .used_numa_hint = false,
            };
        }
        
        _ = task; // Mark as used
        return null; // No fast path available
    }
    
    fn isLoadUniform(self: *Self) bool {
        if (self.worker_loads.len < 2) return true;
        
        // Quick load uniformity check
        var min_load: u32 = std.math.maxInt(u32);
        var max_load: u32 = 0;
        
        for (self.worker_loads) |*load| {
            const current_load = load.load(.acquire);
            min_load = @min(min_load, current_load);
            max_load = @max(max_load, current_load);
        }
        
        if (max_load == 0) return true; // All empty
        
        const imbalance = @as(f32, @floatFromInt(max_load - min_load)) / @as(f32, @floatFromInt(max_load));
        return imbalance <= self.fast_path_config.load_imbalance_threshold;
    }
    
    fn selectRoundRobin(self: *Self) usize {
        const current = self.last_round_robin.fetchAdd(1, .monotonic);
        return current % self.worker_loads.len;
    }
    
    fn selectNumaAwareFast(self: *Self, numa_hint: u32) ?usize {
        // Find workers on the preferred NUMA node with lowest load
        var best_worker: ?usize = null;
        var min_load: u32 = std.math.maxInt(u32);
        
        for (self.base_pool.workers, 0..) |*worker, i| {
            if (worker.numa_node == numa_hint) {
                const load = self.worker_loads[i].load(.acquire);
                if (load < min_load) {
                    min_load = load;
                    best_worker = i;
                }
            }
        }
        
        // If found a worker on preferred node, use it
        if (best_worker) |worker_id| {
            self.incrementWorkerLoad(worker_id);
            return worker_id;
        }
        
        return null; // No worker on preferred NUMA node
    }
    
    fn selectLoadBalancedFast(self: *Self, priority_weight: f32) usize {
        // Simple load balancing based on tracked loads
        var best_worker: usize = 0;
        var min_weighted_load: f32 = std.math.floatMax(f32);
        
        for (self.worker_loads, 0..) |*load, i| {
            const current_load = @as(f32, @floatFromInt(load.load(.acquire)));
            const weighted_load = current_load / priority_weight; // Higher priority = lower effective load
            
            if (weighted_load < min_weighted_load) {
                min_weighted_load = weighted_load;
                best_worker = i;
            }
        }
        
        self.incrementWorkerLoad(best_worker);
        return best_worker;
    }
    
    fn incrementWorkerLoad(self: *Self, worker_id: usize) void {
        _ = self.worker_loads[worker_id].fetchAdd(1, .monotonic);
    }
    
    fn decrementWorkerLoad(self: *Self, worker_id: usize) void {
        _ = self.worker_loads[worker_id].fetchSub(1, .monotonic);
    }
    
    fn fallbackToAdvancedSelection(self: *Self, task: beat.Task) usize {
        // Use the thread pool's existing worker selection logic
        return self.base_pool.selectWorker(task);
    }
    
    /// Notify when a task completes (for load tracking)
    pub fn notifyTaskComplete(self: *Self, worker_id: usize) void {
        if (worker_id < self.worker_loads.len) {
            self.decrementWorkerLoad(worker_id);
        }
    }
    
    pub fn getStats(self: *const Self) FastPathStats {
        return self.fast_path_stats;
    }
    
    pub fn resetStats(self: *Self) void {
        self.fast_path_stats = FastPathStats{};
    }
};

// Performance testing and validation
pub fn runWorkerSelectionOptimizationTest(allocator: std.mem.Allocator) !void {
    std.debug.print("=== Worker Selection Fast Path Optimization Test ===\n", .{});
    
    const pool_config = beat.Config{
        .num_workers = 4,
        .enable_predictive = true,
        .enable_advanced_selection = true,
    };
    
    var pool = try beat.ThreadPool.init(allocator, pool_config);
    defer pool.deinit();
    
    const fast_path_config = FastPathWorkerSelector.FastPathConfig{
        .short_task_threshold_ns = 10_000,
        .simple_task_threshold_cycles = 2000,
        .enable_prediction_bypass = true,
        .enable_round_robin_fast_path = true,
    };
    
    var fast_selector = try FastPathWorkerSelector.init(allocator, pool, fast_path_config);
    defer fast_selector.deinit();
    
    std.debug.print("Testing worker selection performance with 10,000 selections...\n", .{});
    
    // Test different task types
    const test_tasks = [_]beat.Task{
        // Simple tasks (should use fast path)
        beat.Task{ .func = simpleTask, .data = @ptrCast(@constCast(&@as(usize, 1))), .priority = .normal, .data_size_hint = 32 },
        beat.Task{ .func = simpleTask, .data = @ptrCast(@constCast(&@as(usize, 2))), .priority = .low, .data_size_hint = 16 },
        
        // NUMA-hinted tasks
        beat.Task{ .func = simpleTask, .data = @ptrCast(@constCast(&@as(usize, 3))), .priority = .normal, .affinity_hint = 0 },
        beat.Task{ .func = simpleTask, .data = @ptrCast(@constCast(&@as(usize, 4))), .priority = .normal, .affinity_hint = 1 },
        
        // Complex tasks (should use advanced selection)
        beat.Task{ .func = simpleTask, .data = @ptrCast(@constCast(&@as(usize, 5))), .priority = .high, .data_size_hint = 2048 },
    };
    
    const start_time = std.time.nanoTimestamp();
    var total_fast_time: u64 = 0;
    
    // Test fast path selection performance
    for (0..10000) |i| {
        const task = &test_tasks[i % test_tasks.len];
        const selection_start = std.time.nanoTimestamp();
        
        const result = fast_selector.selectWorker(task.*);
        
        const selection_time = @as(u64, @intCast(std.time.nanoTimestamp() - selection_start));
        total_fast_time += selection_time;
        
        // Simulate task completion for load tracking
        fast_selector.notifyTaskComplete(result.worker_id);
        
        // Occasional reporting
        if (i % 2000 == 0 and i > 0) {
            std.debug.print("  Processed {} selections...\n", .{i});
        }
    }
    
    const end_time = std.time.nanoTimestamp();
    const total_time = end_time - start_time;
    
    // Performance analysis
    const stats = fast_selector.getStats();
    const avg_selection_time = total_fast_time / 10000;
    
    std.debug.print("\nPerformance Results:\n", .{});
    std.debug.print("  Total test time: {d:.2}ms\n", .{@as(f64, @floatFromInt(total_time)) / 1_000_000.0});
    std.debug.print("  Average selection time: {d:.1}ns\n", .{@as(f64, @floatFromInt(avg_selection_time))});
    std.debug.print("  Selections per second: {d:.0}\n", .{10000.0 / (@as(f64, @floatFromInt(total_time)) / 1_000_000_000.0)});
    
    std.debug.print("\nFast Path Statistics:\n", .{});
    std.debug.print("  Total selections: {}\n", .{stats.total_selections.load(.acquire)});
    std.debug.print("  Fast path hit rate: {d:.1}%\n", .{stats.getFastPathHitRate() * 100.0});
    std.debug.print("  Round-robin selections: {}\n", .{stats.round_robin_selections.load(.acquire)});
    std.debug.print("  Load-balanced selections: {}\n", .{stats.load_balanced_selections.load(.acquire)});
    std.debug.print("  NUMA local placements: {}\n", .{stats.numa_local_placements.load(.acquire)});
    std.debug.print("  Predictions bypassed: {}\n", .{stats.prediction_bypassed.load(.acquire)});
    
    // Expected performance analysis
    if (stats.getFastPathHitRate() > 0.6) {
        std.debug.print("✅ Fast path optimization successful ({}% hit rate)\n", .{@as(u32, @intFromFloat(stats.getFastPathHitRate() * 100))});
    } else {
        std.debug.print("⚠️ Fast path hit rate lower than expected ({}%)\n", .{@as(u32, @intFromFloat(stats.getFastPathHitRate() * 100))});
    }
    
    if (avg_selection_time < 1000) { // Target: sub-microsecond selection
        std.debug.print("✅ Selection performance excellent ({d:.1}ns average)\n", .{@as(f64, @floatFromInt(avg_selection_time))});
    } else {
        std.debug.print("⚠️ Selection performance needs improvement ({d:.1}ns average)\n", .{@as(f64, @floatFromInt(avg_selection_time))});
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
    
    try runWorkerSelectionOptimizationTest(allocator);
}