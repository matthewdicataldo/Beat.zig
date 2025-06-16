// SoA vs AoS Performance Benchmark for ISPC Optimization
// Demonstrates the dramatic performance impact of Structure of Arrays layout
// Target: 4-8x improvement over AoS + 3-6x ISPC improvement = 12-48x total

const std = @import("std");

// External ISPC function declarations
extern fn ispc_compute_fingerprint_similarity(
    fingerprints_a: [*]u64,
    fingerprints_b: [*]u64,
    results: [*]f32,
    count: i32,
) void;

extern fn ispc_compute_fingerprint_similarity_soa(
    fingerprints_a_low: [*]u64,
    fingerprints_a_high: [*]u64,
    fingerprints_b_low: [*]u64,
    fingerprints_b_high: [*]u64,
    results: [*]f32,
    count: i32,
) void;

extern fn ispc_one_euro_filter_batch(
    raw_values: [*]f32,
    timestamps: [*]f32,
    states: [*]OneEuroState,
    filtered_values: [*]f32,
    count: i32,
) void;

const OneEuroState = extern struct {
    x_prev: f32 = 0.0,
    dx_prev: f32 = 0.0,
    initialized: bool = false,
    min_cutoff: f32 = 1.0,
    beta: f32 = 0.1,
    derivate_cutoff: f32 = 1.0,
};

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

fn benchmarkFingerprintSimilarity(allocator: std.mem.Allocator, count: usize, iterations: u32) !void {
    std.debug.print("\n=== Fingerprint Similarity: AoS vs SoA Comparison ===\n", .{});
    std.debug.print("Dataset: {} fingerprint pairs, {} iterations\n", .{ count, iterations });
    
    // Prepare test data
    var fingerprints_a_aos = try allocator.alloc(u64, count * 2);
    defer allocator.free(fingerprints_a_aos);
    var fingerprints_b_aos = try allocator.alloc(u64, count * 2);
    defer allocator.free(fingerprints_b_aos);
    
    // SoA arrays
    var fingerprints_a_low = try allocator.alloc(u64, count);
    defer allocator.free(fingerprints_a_low);
    var fingerprints_a_high = try allocator.alloc(u64, count);
    defer allocator.free(fingerprints_a_high);
    var fingerprints_b_low = try allocator.alloc(u64, count);
    defer allocator.free(fingerprints_b_low);
    var fingerprints_b_high = try allocator.alloc(u64, count);
    defer allocator.free(fingerprints_b_high);
    
    // Initialize test data
    for (0..count) |i| {
        const fp_a = (@as(u128, @intCast(i)) * 0x123456789ABCDEF) ^ 0xFEDCBA9876543210;
        const fp_b = (@as(u128, @intCast(i)) * 0xDEADBEEFCAFEBABE) ^ 0x0123456789ABCDEF;
        
        // AoS layout
        fingerprints_a_aos[i * 2] = @as(u64, @truncate(fp_a));
        fingerprints_a_aos[i * 2 + 1] = @as(u64, @truncate(fp_a >> 64));
        fingerprints_b_aos[i * 2] = @as(u64, @truncate(fp_b));
        fingerprints_b_aos[i * 2 + 1] = @as(u64, @truncate(fp_b >> 64));
        
        // SoA layout
        fingerprints_a_low[i] = @as(u64, @truncate(fp_a));
        fingerprints_a_high[i] = @as(u64, @truncate(fp_a >> 64));
        fingerprints_b_low[i] = @as(u64, @truncate(fp_b));
        fingerprints_b_high[i] = @as(u64, @truncate(fp_b >> 64));
    }
    
    var results_aos = try allocator.alloc(f32, count);
    defer allocator.free(results_aos);
    var results_soa = try allocator.alloc(f32, count);
    defer allocator.free(results_soa);
    
    var timer = try std.time.Timer.start();
    
    // Benchmark ISPC AoS (with gather operations)
    timer.reset();
    for (0..iterations) |_| {
        ispc_compute_fingerprint_similarity(
            fingerprints_a_aos.ptr,
            fingerprints_b_aos.ptr,
            results_aos.ptr,
            @intCast(count),
        );
        std.mem.doNotOptimizeAway(&results_aos);
    }
    const aos_time = timer.read();
    
    // Benchmark ISPC SoA (optimized vectorized access)
    timer.reset();
    for (0..iterations) |_| {
        ispc_compute_fingerprint_similarity_soa(
            fingerprints_a_low.ptr,
            fingerprints_a_high.ptr,
            fingerprints_b_low.ptr,
            fingerprints_b_high.ptr,
            results_soa.ptr,
            @intCast(count),
        );
        std.mem.doNotOptimizeAway(&results_soa);
    }
    const soa_time = timer.read();
    
    // Verify correctness
    var max_diff: f32 = 0.0;
    for (0..count) |i| {
        const diff = @abs(results_aos[i] - results_soa[i]);
        max_diff = @max(max_diff, diff);
    }
    
    const speedup = @as(f64, @floatFromInt(aos_time)) / @as(f64, @floatFromInt(soa_time));
    
    std.debug.print("AoS ISPC time:  ", .{});
    formatTime(aos_time);
    std.debug.print(" (with gather warnings)\n", .{});
    std.debug.print("SoA ISPC time:  ", .{});
    formatTime(soa_time);
    std.debug.print(" (vectorized access)\n", .{});
    std.debug.print("SoA Speedup:    {d:.2}x\n", .{speedup});
    std.debug.print("Max difference: {d:.6}\n", .{max_diff});
    
    if (speedup >= 4.0) {
        std.debug.print("🎯 EXCELLENT: SoA achieves {d:.1}x speedup!\n", .{speedup});
    } else if (speedup >= 2.0) {
        std.debug.print("✅ GOOD: SoA achieves {d:.1}x speedup\n", .{speedup});
    } else {
        std.debug.print("📊 Result: {d:.1}x speedup (may vary by hardware)\n", .{speedup});
    }
}

fn benchmarkOneEuroFilter(allocator: std.mem.Allocator, count: usize, iterations: u32) !void {
    std.debug.print("\n=== One Euro Filter: ISPC vs Native Comparison ===\n", .{});
    std.debug.print("Dataset: {} predictions, {} iterations\n", .{ count, iterations });
    
    // Prepare test data
    var raw_values = try allocator.alloc(f32, count);
    defer allocator.free(raw_values);
    var timestamps = try allocator.alloc(f32, count);
    defer allocator.free(timestamps);
    var states_ispc = try allocator.alloc(OneEuroState, count);
    defer allocator.free(states_ispc);
    var filtered_ispc = try allocator.alloc(f32, count);
    defer allocator.free(filtered_ispc);
    var filtered_native = try allocator.alloc(f32, count);
    defer allocator.free(filtered_native);
    
    // Initialize test data
    for (0..count) |i| {
        raw_values[i] = @sin(@as(f32, @floatFromInt(i)) * 0.1) + @as(f32, @floatFromInt(i)) * 0.01;
        timestamps[i] = @as(f32, @floatFromInt(i)) * 0.016; // 60fps
        states_ispc[i] = OneEuroState{};
    }
    
    var timer = try std.time.Timer.start();
    
    // Native One Euro Filter implementation
    timer.reset();
    for (0..iterations) |_| {
        var prev_value: f32 = 0.0;
        var prev_derivative: f32 = 0.0;
        var initialized = false;
        
        for (0..count) |i| {
            if (!initialized) {
                filtered_native[i] = raw_values[i];
                prev_value = raw_values[i];
                initialized = true;
                continue;
            }
            
            const dt = if (i > 0) timestamps[i] - timestamps[i-1] else 0.016;
            const dx = (raw_values[i] - prev_value) / dt;
            
            // Simple smoothing (simplified One Euro)
            const alpha = 0.1;
            const filtered = alpha * raw_values[i] + (1.0 - alpha) * prev_value;
            
            filtered_native[i] = filtered;
            prev_value = filtered;
            prev_derivative = dx;
        }
        std.mem.doNotOptimizeAway(&filtered_native);
    }
    const native_time = timer.read();
    
    // ISPC One Euro Filter implementation
    timer.reset();
    for (0..iterations) |_| {
        // Reset states
        for (states_ispc) |*state| {
            state.* = OneEuroState{};
        }
        
        ispc_one_euro_filter_batch(
            raw_values.ptr,
            timestamps.ptr,
            states_ispc.ptr,
            filtered_ispc.ptr,
            @intCast(count),
        );
        std.mem.doNotOptimizeAway(&filtered_ispc);
    }
    const ispc_time = timer.read();
    
    const speedup = @as(f64, @floatFromInt(native_time)) / @as(f64, @floatFromInt(ispc_time));
    
    std.debug.print("Native time:   ", .{});
    formatTime(native_time);
    std.debug.print("\n", .{});
    std.debug.print("ISPC time:     ", .{});
    formatTime(ispc_time);
    std.debug.print("\n", .{});
    std.debug.print("ISPC Speedup:  {d:.2}x\n", .{speedup});
    
    if (speedup >= 10.0) {
        std.debug.print("🚀 OUTSTANDING: ISPC achieves {d:.1}x speedup!\n", .{speedup});
    } else if (speedup >= 5.0) {
        std.debug.print("🎯 EXCELLENT: ISPC achieves {d:.1}x speedup!\n", .{speedup});
    } else if (speedup >= 2.0) {
        std.debug.print("✅ GOOD: ISPC achieves {d:.1}x speedup\n", .{speedup});
    } else {
        std.debug.print("📊 Result: {d:.1}x speedup (compute-heavy algorithms show better gains)\n", .{speedup});
    }
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    std.debug.print("🚀 ISPC Performance Optimization Benchmark\n", .{});
    std.debug.print("===========================================\n", .{});
    std.debug.print("Testing: Structure of Arrays + ISPC SPMD Acceleration\n", .{});
    
    // Test different dataset sizes
    const test_sizes = [_]usize{ 1000, 5000, 20000 };
    
    for (test_sizes) |size| {
        std.debug.print("\n" ++ "="**50 ++ "\n", .{});
        std.debug.print("Dataset Size: {} elements\n", .{size});
        std.debug.print("="**50 ++ "\n", .{});
        
        // Benchmark fingerprint similarity (memory-bound → SoA benefits)
        try benchmarkFingerprintSimilarity(allocator, size, 100);
        
        // Benchmark One Euro Filter (compute-bound → ISPC benefits)
        try benchmarkOneEuroFilter(allocator, size, 50);
    }
    
    std.debug.print("\n🎊 OPTIMIZATION SUMMARY\n", .{});
    std.debug.print("=======================\n", .{});
    std.debug.print("✅ Structure of Arrays (SoA): Eliminates gather operations\n", .{});
    std.debug.print("✅ ISPC SPMD Parallelism: Vectorizes math-heavy computations\n", .{});
    std.debug.print("✅ Combined Optimization: Multiple performance multipliers\n", .{});
    std.debug.print("\n🚀 Beat.zig + Optimized ISPC: Maximum Performance!\n", .{});
}