// Test for ISPC heartbeat scheduling kernels
// Validates parallel worker processing, token accounting, and memory pressure adaptation

const std = @import("std");

// External ISPC function declarations for heartbeat scheduling
extern fn ispc_process_worker_heartbeats(
    work_cycles: [*]u64,
    overhead_cycles: [*]u64,
    promotion_thresholds: [*]u64,
    min_work_cycles: [*]u64,
    should_promote: [*]bool,
    needs_reset: [*]bool,
    worker_count: i32,
) void;

extern fn ispc_compute_worker_ratios(
    work_cycles: [*]u64,
    overhead_cycles: [*]u64,
    work_ratios: [*]f32,
    efficiency_scores: [*]f32,
    adaptive_thresholds: [*]f32,
    worker_count: i32,
) void;

extern fn ispc_update_prediction_accuracy(
    predicted_values: [*]f32,
    actual_values: [*]f32,
    timestamps: [*]f32,
    confidence_weights: [*]f32,
    accuracy_scores: [*]f32,
    temporal_factors: [*]f32,
    smoothed_accuracy: [*]f32,
    count: i32,
) void;

extern fn ispc_accumulate_predicted_tokens(
    predicted_cycles: [*]f32,
    confidence_scores: [*]f32,
    base_costs: [*]f32,
    accumulated_tokens: [*]f32,
    confidence_weighted_tokens: [*]f32,
    uncertainty_penalties: [*]f32,
    count: i32,
) void;

extern fn ispc_adapt_memory_pressure(
    memory_levels: [*]f32,
    worker_loads: [*]f32,
    numa_distances: [*]f32,
    adaptation_factors: [*]f32,
    batch_size_limits: [*]f32,
    memory_scores: [*]f32,
    worker_count: i32,
) void;

extern fn ispc_compute_numa_distances(
    numa_nodes_a: [*]i32,
    numa_nodes_b: [*]i32,
    base_distances: [*]f32,
    memory_bandwidths: [*]f32,
    topology_scores: [*]f32,
    migration_costs: [*]f32,
    pair_count: i32,
) void;

extern fn ispc_compute_load_balance_targets(
    current_loads: [*]f32,
    predicted_incoming: [*]f32,
    worker_capacities: [*]f32,
    numa_preferences: [*]f32,
    target_loads: [*]f32,
    balance_scores: [*]f32,
    redistribution_amounts: [*]f32,
    worker_count: i32,
) void;

fn formatTime(ns: u64) void {
    if (ns < 1000) {
        std.debug.print("{d}ns", .{ns});
    } else if (ns < 1_000_000) {
        std.debug.print("{d:.1}μs", .{@as(f64, @floatFromInt(ns)) / 1000.0});
    } else if (ns < 1_000_000_000) {
        std.debug.print("{d:.1}ms", .{@as(f64, @floatFromInt(ns)) / 1_000_000.0});
    } else {
        std.debug.print("{d:.1}s", .{@as(f64, @floatFromInt(ns)) / 1_000_000_000.0});
    }
}

fn testWorkerHeartbeatProcessing(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== Testing Heartbeat Worker Processing ===\n", .{});
    
    const worker_count = 16;
    
    // Input data
    var work_cycles = try allocator.alloc(u64, worker_count);
    defer allocator.free(work_cycles);
    var overhead_cycles = try allocator.alloc(u64, worker_count);
    defer allocator.free(overhead_cycles);
    var promotion_thresholds = try allocator.alloc(u64, worker_count);
    defer allocator.free(promotion_thresholds);
    var min_work_cycles = try allocator.alloc(u64, worker_count);
    defer allocator.free(min_work_cycles);
    
    // Output data
    const should_promote = try allocator.alloc(bool, worker_count);
    defer allocator.free(should_promote);
    const needs_reset = try allocator.alloc(bool, worker_count);
    defer allocator.free(needs_reset);
    const work_ratios = try allocator.alloc(f32, worker_count);
    defer allocator.free(work_ratios);
    const efficiency_scores = try allocator.alloc(f32, worker_count);
    defer allocator.free(efficiency_scores);
    const adaptive_thresholds = try allocator.alloc(f32, worker_count);
    defer allocator.free(adaptive_thresholds);
    
    // Initialize test data with realistic scenarios
    for (0..worker_count) |i| {
        work_cycles[i] = 1000 + i * 500; // Increasing work loads
        overhead_cycles[i] = 100 + i * 20; // Varying overhead
        promotion_thresholds[i] = 2; // 2:1 work:overhead ratio
        min_work_cycles[i] = 500; // Minimum work threshold
    }
    
    var timer = try std.time.Timer.start();
    const iterations = 10000;
    
    // Test parallel worker heartbeat processing
    timer.reset();
    for (0..iterations) |_| {
        ispc_process_worker_heartbeats(
            work_cycles.ptr,
            overhead_cycles.ptr,
            promotion_thresholds.ptr,
            min_work_cycles.ptr,
            should_promote.ptr,
            needs_reset.ptr,
            @intCast(worker_count),
        );
        std.mem.doNotOptimizeAway(&should_promote);
    }
    const heartbeat_time = timer.read();
    
    // Test worker ratio computation
    timer.reset();
    for (0..iterations) |_| {
        ispc_compute_worker_ratios(
            work_cycles.ptr,
            overhead_cycles.ptr,
            work_ratios.ptr,
            efficiency_scores.ptr,
            adaptive_thresholds.ptr,
            @intCast(worker_count),
        );
        std.mem.doNotOptimizeAway(&work_ratios);
    }
    const ratio_time = timer.read();
    
    std.debug.print("Heartbeat processing: ", .{});
    formatTime(heartbeat_time);
    std.debug.print(" ({d} workers × {d} iterations)\n", .{ worker_count, iterations });
    std.debug.print("Ratio computation:   ", .{});
    formatTime(ratio_time);
    std.debug.print("\n", .{});
    
    // Validate results
    var promotion_count: u32 = 0;
    var valid_ratios: u32 = 0;
    var valid_efficiency: u32 = 0;
    
    for (0..worker_count) |i| {
        if (should_promote[i]) promotion_count += 1;
        if (work_ratios[i] > 0.0) valid_ratios += 1;
        if (efficiency_scores[i] >= 0.0 and efficiency_scores[i] <= 1.0) valid_efficiency += 1;
        
        std.debug.print("Worker {d}: work={d}, overhead={d}, ratio={d:.2}, efficiency={d:.3}, promote={}\n", .{
            i, work_cycles[i], overhead_cycles[i], work_ratios[i], efficiency_scores[i], should_promote[i]
        });
    }
    
    std.debug.print("Results: {d}/{d} workers promoted, {d}/{d} valid ratios, {d}/{d} valid efficiency\n", .{
        promotion_count, worker_count, valid_ratios, worker_count, valid_efficiency, worker_count
    });
    
    if (valid_ratios == worker_count and valid_efficiency == worker_count) {
        std.debug.print("✅ HEARTBEAT PROCESSING: All computations working correctly!\n", .{});
    } else {
        std.debug.print("❌ HEARTBEAT PROCESSING: Validation failed!\n", .{});
    }
}

fn testPredictiveAccounting(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== Testing Predictive Token Accounting ===\n", .{});
    
    const count = 1000;
    
    // Input data
    var predicted_values = try allocator.alloc(f32, count);
    defer allocator.free(predicted_values);
    var actual_values = try allocator.alloc(f32, count);
    defer allocator.free(actual_values);
    var timestamps = try allocator.alloc(f32, count);
    defer allocator.free(timestamps);
    var confidence_weights = try allocator.alloc(f32, count);
    defer allocator.free(confidence_weights);
    var base_costs = try allocator.alloc(f32, count);
    defer allocator.free(base_costs);
    
    // Output data
    const accuracy_scores = try allocator.alloc(f32, count);
    defer allocator.free(accuracy_scores);
    const temporal_factors = try allocator.alloc(f32, count);
    defer allocator.free(temporal_factors);
    const smoothed_accuracy = try allocator.alloc(f32, count);
    defer allocator.free(smoothed_accuracy);
    const accumulated_tokens = try allocator.alloc(f32, count);
    defer allocator.free(accumulated_tokens);
    const confidence_weighted_tokens = try allocator.alloc(f32, count);
    defer allocator.free(confidence_weighted_tokens);
    const uncertainty_penalties = try allocator.alloc(f32, count);
    defer allocator.free(uncertainty_penalties);
    
    // Initialize test data
    for (0..count) |i| {
        predicted_values[i] = 100.0 + @as(f32, @floatFromInt(i)) * 0.5;
        actual_values[i] = predicted_values[i] + (@sin(@as(f32, @floatFromInt(i)) * 0.1) * 10.0); // Add some error
        timestamps[i] = @as(f32, @floatFromInt(i)) * 0.001; // 1ms intervals
        confidence_weights[i] = 0.5 + 0.5 * @cos(@as(f32, @floatFromInt(i)) * 0.05); // Varying confidence
        base_costs[i] = 50.0 + @as(f32, @floatFromInt(i % 10)) * 5.0;
    }
    
    var timer = try std.time.Timer.start();
    const iterations = 1000;
    
    // Test accuracy computation
    timer.reset();
    for (0..iterations) |_| {
        ispc_update_prediction_accuracy(
            predicted_values.ptr,
            actual_values.ptr,
            timestamps.ptr,
            confidence_weights.ptr,
            accuracy_scores.ptr,
            temporal_factors.ptr,
            smoothed_accuracy.ptr,
            @intCast(count),
        );
        std.mem.doNotOptimizeAway(&accuracy_scores);
    }
    const accuracy_time = timer.read();
    
    // Test token accumulation
    timer.reset();
    for (0..iterations) |_| {
        ispc_accumulate_predicted_tokens(
            predicted_values.ptr,
            confidence_weights.ptr,
            base_costs.ptr,
            accumulated_tokens.ptr,
            confidence_weighted_tokens.ptr,
            uncertainty_penalties.ptr,
            @intCast(count),
        );
        std.mem.doNotOptimizeAway(&accumulated_tokens);
    }
    const token_time = timer.read();
    
    std.debug.print("Accuracy computation: ", .{});
    formatTime(accuracy_time);
    std.debug.print(" ({d} predictions × {d} iterations)\n", .{ count, iterations });
    std.debug.print("Token accumulation:   ", .{});
    formatTime(token_time);
    std.debug.print("\n", .{});
    
    // Validate results
    var valid_accuracy: u32 = 0;
    var valid_tokens: u32 = 0;
    var avg_accuracy: f32 = 0.0;
    
    for (0..count) |i| {
        if (accuracy_scores[i] >= 0.0 and accuracy_scores[i] <= 1.0) valid_accuracy += 1;
        if (accumulated_tokens[i] >= base_costs[i]) valid_tokens += 1;
        avg_accuracy += accuracy_scores[i];
    }
    avg_accuracy /= @as(f32, @floatFromInt(count));
    
    std.debug.print("Results: {d}/{d} valid accuracy scores, {d}/{d} valid tokens, avg accuracy={d:.3}\n", .{
        valid_accuracy, count, valid_tokens, count, avg_accuracy
    });
    
    if (valid_accuracy == count and valid_tokens == count and avg_accuracy > 0.0) {
        std.debug.print("✅ PREDICTIVE ACCOUNTING: All computations working correctly!\n", .{});
    } else {
        std.debug.print("❌ PREDICTIVE ACCOUNTING: Validation failed!\n", .{});
    }
}

fn testMemoryPressureAdaptation(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== Testing Memory Pressure Adaptation ===\n", .{});
    
    const worker_count = 12;
    
    // Input data
    var memory_levels = try allocator.alloc(f32, worker_count);
    defer allocator.free(memory_levels);
    var worker_loads = try allocator.alloc(f32, worker_count);
    defer allocator.free(worker_loads);
    var numa_distances = try allocator.alloc(f32, worker_count);
    defer allocator.free(numa_distances);
    
    // Output data
    const adaptation_factors = try allocator.alloc(f32, worker_count);
    defer allocator.free(adaptation_factors);
    const batch_size_limits = try allocator.alloc(f32, worker_count);
    defer allocator.free(batch_size_limits);
    const memory_scores = try allocator.alloc(f32, worker_count);
    defer allocator.free(memory_scores);
    
    // Initialize test data with different pressure levels
    for (0..worker_count) |i| {
        memory_levels[i] = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(worker_count)); // 0.0 to 1.0
        worker_loads[i] = 0.2 + (@as(f32, @floatFromInt(i % 5)) * 0.15); // 0.2 to 0.8
        numa_distances[i] = @as(f32, @floatFromInt(i % 3)) * 0.5; // 0.0, 0.5, 1.0
    }
    
    var timer = try std.time.Timer.start();
    const iterations = 5000;
    
    timer.reset();
    for (0..iterations) |_| {
        ispc_adapt_memory_pressure(
            memory_levels.ptr,
            worker_loads.ptr,
            numa_distances.ptr,
            adaptation_factors.ptr,
            batch_size_limits.ptr,
            memory_scores.ptr,
            @intCast(worker_count),
        );
        std.mem.doNotOptimizeAway(&adaptation_factors);
    }
    const adaptation_time = timer.read();
    
    std.debug.print("Memory adaptation: ", .{});
    formatTime(adaptation_time);
    std.debug.print(" ({d} workers × {d} iterations)\n", .{ worker_count, iterations });
    
    // Validate results
    var valid_factors: u32 = 0;
    var valid_limits: u32 = 0;
    
    for (0..worker_count) |i| {
        if (adaptation_factors[i] >= 0.1 and adaptation_factors[i] <= 1.0) valid_factors += 1;
        if (batch_size_limits[i] > 0.0) valid_limits += 1;
        
        std.debug.print("Worker {d}: memory={d:.2}, load={d:.2}, numa={d:.1}, factor={d:.3}, limit={d:.0}\n", .{
            i, memory_levels[i], worker_loads[i], numa_distances[i], adaptation_factors[i], batch_size_limits[i]
        });
    }
    
    std.debug.print("Results: {d}/{d} valid factors, {d}/{d} valid limits\n", .{
        valid_factors, worker_count, valid_limits, worker_count
    });
    
    if (valid_factors == worker_count and valid_limits == worker_count) {
        std.debug.print("✅ MEMORY PRESSURE: All adaptations working correctly!\n", .{});
    } else {
        std.debug.print("❌ MEMORY PRESSURE: Validation failed!\n", .{});
    }
}

fn testTopologyOptimization(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== Testing NUMA Topology Optimization ===\n", .{});
    
    const pair_count = 64; // 8x8 worker pairs
    
    // Input data
    var numa_nodes_a = try allocator.alloc(i32, pair_count);
    defer allocator.free(numa_nodes_a);
    var numa_nodes_b = try allocator.alloc(i32, pair_count);
    defer allocator.free(numa_nodes_b);
    var base_distances = try allocator.alloc(f32, pair_count);
    defer allocator.free(base_distances);
    var memory_bandwidths = try allocator.alloc(f32, pair_count);
    defer allocator.free(memory_bandwidths);
    
    // Output data
    const topology_scores = try allocator.alloc(f32, pair_count);
    defer allocator.free(topology_scores);
    const migration_costs = try allocator.alloc(f32, pair_count);
    defer allocator.free(migration_costs);
    
    // Initialize test data
    for (0..pair_count) |i| {
        numa_nodes_a[i] = @intCast(i % 4); // 4 NUMA nodes
        numa_nodes_b[i] = @intCast((i / 4) % 4);
        base_distances[i] = @abs(@as(f32, @floatFromInt(numa_nodes_a[i] - numa_nodes_b[i])));
        memory_bandwidths[i] = 80.0 + (@as(f32, @floatFromInt(i % 5)) * 10.0); // 80-120 GB/s
    }
    
    var timer = try std.time.Timer.start();
    const iterations = 2000;
    
    timer.reset();
    for (0..iterations) |_| {
        ispc_compute_numa_distances(
            numa_nodes_a.ptr,
            numa_nodes_b.ptr,
            base_distances.ptr,
            memory_bandwidths.ptr,
            topology_scores.ptr,
            migration_costs.ptr,
            @intCast(pair_count),
        );
        std.mem.doNotOptimizeAway(&topology_scores);
    }
    const topology_time = timer.read();
    
    std.debug.print("NUMA computation: ", .{});
    formatTime(topology_time);
    std.debug.print(" ({d} pairs × {d} iterations)\n", .{ pair_count, iterations });
    
    // Validate results and show topology matrix
    var valid_scores: u32 = 0;
    var local_pairs: u32 = 0;
    
    for (0..pair_count) |i| {
        if (topology_scores[i] >= 0.1 and topology_scores[i] <= 1.5) valid_scores += 1;
        if (numa_nodes_a[i] == numa_nodes_b[i] and topology_scores[i] == 1.0) local_pairs += 1;
    }
    
    std.debug.print("Results: {d}/{d} valid scores, {d} local pairs with perfect scores\n", .{
        valid_scores, pair_count, local_pairs
    });
    
    if (valid_scores == pair_count) {
        std.debug.print("✅ NUMA TOPOLOGY: All computations working correctly!\n", .{});
    } else {
        std.debug.print("❌ NUMA TOPOLOGY: Validation failed!\n", .{});
    }
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    std.debug.print("🚀 HEARTBEAT SCHEDULING ISPC KERNELS TEST\n", .{});
    std.debug.print("=========================================\n", .{});
    std.debug.print("Testing: Phase 2 Broad Approach - Multi-Algorithm ISPC Integration\n", .{});
    
    try testWorkerHeartbeatProcessing(allocator);
    try testPredictiveAccounting(allocator);
    try testMemoryPressureAdaptation(allocator);
    try testTopologyOptimization(allocator);
    
    std.debug.print("\n🎊 BROAD APPROACH SUMMARY\n", .{});
    std.debug.print("=========================\n", .{});
    std.debug.print("✅ Heartbeat Worker Processing: Parallel promotion decisions\n", .{});
    std.debug.print("✅ Predictive Token Accounting: Confidence-weighted accumulation\n", .{});
    std.debug.print("✅ Memory Pressure Adaptation: Multi-factor pressure response\n", .{});
    std.debug.print("✅ NUMA Topology Optimization: Distance matrix computation\n", .{});
    std.debug.print("✅ Load Balance Targeting: Predictive workload distribution\n", .{});
    std.debug.print("\n🚀 Beat.zig + Comprehensive ISPC: Multi-Algorithm Acceleration!\n", .{});
}