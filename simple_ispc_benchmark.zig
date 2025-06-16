// Simple ISPC Performance Test
// Quick demonstration of ISPC speedup vs native implementation

const std = @import("std");

extern fn ispc_compute_fingerprint_similarity(
    fingerprints_a: [*]u64,
    fingerprints_b: [*]u64,
    results: [*]f32,
    count: i32,
) void;

fn nativeCompute(fingerprints_a: []u64, fingerprints_b: []u64, results: []f32) void {
    for (0..results.len) |i| {
        const fp_a_low = fingerprints_a[i * 2];
        const fp_a_high = fingerprints_a[i * 2 + 1];
        const fp_b_low = fingerprints_b[i * 2];
        const fp_b_high = fingerprints_b[i * 2 + 1];
        
        const diff_low = fp_a_low ^ fp_b_low;
        const diff_high = fp_a_high ^ fp_b_high;
        const hamming_distance = @popCount(diff_low) + @popCount(diff_high);
        
        results[i] = 1.0 - (@as(f32, @floatFromInt(hamming_distance)) / 128.0);
    }
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    std.debug.print("⚡ ISPC vs Native Performance Test\n", .{});
    std.debug.print("==================================\n", .{});
    
    // Test with larger dataset to overcome overhead
    const count: usize = 10000;
    const iterations: u32 = 100;
    
    var fingerprints_a = try allocator.alloc(u64, count * 2);
    defer allocator.free(fingerprints_a);
    var fingerprints_b = try allocator.alloc(u64, count * 2);
    defer allocator.free(fingerprints_b);
    var native_results = try allocator.alloc(f32, count);
    defer allocator.free(native_results);
    var ispc_results = try allocator.alloc(f32, count);
    defer allocator.free(ispc_results);
    
    // Initialize with pseudo-random data
    for (0..count * 2) |i| {
        fingerprints_a[i] = (@as(u64, @intCast(i)) * 0x123456789ABCDEF) ^ 0xFEDCBA9876543210;
        fingerprints_b[i] = (@as(u64, @intCast(i)) * 0xDEADBEEFCAFEBABE) ^ 0x0123456789ABCDEF;
    }
    
    var timer = try std.time.Timer.start();
    
    // Benchmark native implementation
    timer.reset();
    for (0..iterations) |_| {
        nativeCompute(fingerprints_a, fingerprints_b, native_results);
        std.mem.doNotOptimizeAway(&native_results);
    }
    const native_time = timer.read();
    
    // Benchmark ISPC implementation
    timer.reset();
    for (0..iterations) |_| {
        ispc_compute_fingerprint_similarity(
            fingerprints_a.ptr,
            fingerprints_b.ptr,
            ispc_results.ptr,
            @intCast(count),
        );
        std.mem.doNotOptimizeAway(&ispc_results);
    }
    const ispc_time = timer.read();
    
    // Verify correctness
    var max_diff: f32 = 0.0;
    for (0..count) |i| {
        const diff = @abs(native_results[i] - ispc_results[i]);
        max_diff = @max(max_diff, diff);
    }
    
    const speedup = @as(f64, @floatFromInt(native_time)) / @as(f64, @floatFromInt(ispc_time));
    
    std.debug.print("Dataset: {} fingerprint pairs\n", .{count});
    std.debug.print("Iterations: {}\n", .{iterations});
    std.debug.print("\n", .{});
    std.debug.print("Native time:   {d:.2}ms\n", .{@as(f64, @floatFromInt(native_time)) / 1_000_000.0});
    std.debug.print("ISPC time:     {d:.2}ms\n", .{@as(f64, @floatFromInt(ispc_time)) / 1_000_000.0});
    std.debug.print("SPEEDUP:       {d:.2}x\n", .{speedup});
    std.debug.print("Max difference: {d:.6}\n", .{max_diff});
    
    if (speedup >= 2.0) {
        std.debug.print("\n🎯 TARGET ACHIEVED: {d:.1}x speedup with ISPC SPMD!\n", .{speedup});
    } else {
        std.debug.print("\n📊 Speedup: {d:.1}x (varies by workload and hardware)\n", .{speedup});
    }
    
    std.debug.print("\n🚀 Beat.zig + ISPC: Production-ready acceleration!\n", .{});
}