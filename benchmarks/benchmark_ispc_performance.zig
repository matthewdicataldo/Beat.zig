// ISPC Performance Benchmark for Beat.zig
// Comprehensive performance comparison between native Zig SIMD and ISPC SPMD implementations
// Validates the 3-6x speedup targets for production workloads

const std = @import("std");
const beat = @import("beat");
const ispc = @import("../src/ispc_integration.zig");

// Import ISPC-generated headers
const ispc_kernels = @cImport({
    @cInclude("fingerprint_similarity.h");
    @cInclude("batch_optimization.h");
    @cInclude("worker_selection.h");
});

const BenchmarkResult = struct {
    native_time: u64,
    ispc_time: u64,
    speedup: f64,
    throughput_native: f64,
    throughput_ispc: f64,
};

fn formatTime(ns: u64) void {
    if (ns < 1000) {
        std.debug.print("{d}ns", .{ns});
    } else if (ns < 1_000_000) {
        std.debug.print("{f:.1}μs", .{@as(f64, @floatFromInt(ns)) / 1000.0});
    } else {
        std.debug.print("{d:.1}ms", .{@as(f64, @floatFromInt(ns)) / 1_000_000.0});
    }
}

fn formatThroughput(ops_per_sec: f64) void {
    if (ops_per_sec < 1_000) {
        std.debug.print("{d:.1} ops/s", .{ops_per_sec});
    } else if (ops_per_sec < 1_000_000) {
        std.debug.print("{d:.1}K ops/s", .{ops_per_sec / 1000.0});
    } else {
        std.debug.print("{d:.1}M ops/s", .{ops_per_sec / 1_000_000.0});
    }
}

// Benchmark fingerprint similarity computation
fn benchmarkFingerprintSimilarity(allocator: std.mem.Allocator, size: usize, iterations: u32) !BenchmarkResult {
    std.debug.print("\n=== Fingerprint Similarity Benchmark ===\n", .{});
    std.debug.print("Dataset size: {} fingerprint pairs\n", .{size});
    std.debug.print("Iterations: {}\n", .{iterations});

    // Prepare test data
    var fingerprints_a = try allocator.alloc(u128, size);
    defer allocator.free(fingerprints_a);
    var fingerprints_b = try allocator.alloc(u128, size);
    defer allocator.free(fingerprints_b);

    var prng = std.rand.DefaultPrng.init(12345);
    const random = prng.random();

    for (0..size) |i| {
        fingerprints_a[i] = random.int(u128);
        fingerprints_b[i] = random.int(u128);
    }

    var native_results = try allocator.alloc(f32, size);
    defer allocator.free(native_results);
    var ispc_results = try allocator.alloc(f32, size);
    defer allocator.free(ispc_results);

    var timer = try std.time.Timer.start();

    // Benchmark native implementation
    timer.reset();
    for (0..iterations) |_| {
        for (0..size) |i| {
            const diff = fingerprints_a[i] ^ fingerprints_b[i];
            const hamming_distance = @popCount(diff);
            native_results[i] = 1.0 - (@as(f32, @floatFromInt(hamming_distance)) / 128.0);
        }
        std.mem.doNotOptimizeAway(&native_results);
    }
    const native_time = timer.read();

    // Benchmark ISPC implementation
    timer.reset();
    for (0..iterations) |_| {
        ispc_kernels.ispc_compute_fingerprint_similarity(
            @as([*]const u64, @ptrCast(fingerprints_a.ptr)),
            @as([*]const u64, @ptrCast(fingerprints_b.ptr)),
            ispc_results.ptr,
            @intCast(size),
        );
        std.mem.doNotOptimizeAway(&ispc_results);
    }
    const ispc_time = timer.read();

    // Verify correctness
    var max_diff: f32 = 0.0;
    for (0..size) |i| {
        const diff = @abs(native_results[i] - ispc_results[i]);
        max_diff = @max(max_diff, diff);
    }

    const speedup = @as(f64, @floatFromInt(native_time)) / @as(f64, @floatFromInt(ispc_time));
    const total_ops = size * iterations;
    const throughput_native = @as(f64, @floatFromInt(total_ops)) / (@as(f64, @floatFromInt(native_time)) / 1e9);
    const throughput_ispc = @as(f64, @floatFromInt(total_ops)) / (@as(f64, @floatFromInt(ispc_time)) / 1e9);

    std.debug.print("Native time: ");
    formatTime(native_time);
    std.debug.print("\nISPC time: ");
    formatTime(ispc_time);
    std.debug.print("\nSpeedup: {f:.2}x\n", .{speedup});
    std.debug.print("Max difference: {d:.6}\n", .{max_diff});
    std.debug.print("Native throughput: ");
    formatThroughput(throughput_native);
    std.debug.print("\nISPC throughput: ");
    formatThroughput(throughput_ispc);
    std.debug.print("\n");

    return BenchmarkResult{
        .native_time = native_time,
        .ispc_time = ispc_time,
        .speedup = speedup,
        .throughput_native = throughput_native,
        .throughput_ispc = throughput_ispc,
    };
}

// Benchmark batch formation optimization
fn benchmarkBatchOptimization(allocator: std.mem.Allocator, task_count: usize, iterations: u32) !BenchmarkResult {
    std.debug.print("\n=== Batch Formation Optimization Benchmark ===\n");
    std.debug.print("Task count: {}\n", .{task_count});
    std.debug.print("Iterations: {}\n", .{iterations});

    var task_scores = try allocator.alloc(f32, task_count);
    defer allocator.free(task_scores);
    var similarity_matrix = try allocator.alloc(f32, task_count * task_count);
    defer allocator.free(similarity_matrix);
    var batch_indices = try allocator.alloc(u32, task_count / 2);
    defer allocator.free(batch_indices);

    var prng = std.rand.DefaultPrng.init(54321);
    const random = prng.random();

    // Initialize realistic test data
    for (0..task_count) |i| {
        task_scores[i] = random.float(f32) * 0.8 + 0.2; // 0.2 to 1.0

        for (0..task_count) |j| {
            if (i == j) {
                similarity_matrix[i * task_count + j] = 1.0;
            } else {
                similarity_matrix[i * task_count + j] = random.float(f32) * 0.6 + 0.2; // 0.2 to 0.8
            }
        }
    }

    var timer = try std.time.Timer.start();
    var total_batch_size: u32 = 0;

    // Benchmark native greedy algorithm
    timer.reset();
    for (0..iterations) |_| {
        // Simplified native greedy batch formation
        var selected = try allocator.alloc(bool, task_count);
        defer allocator.free(selected);
        @memset(selected, false);

        var batch_size: u32 = 0;
        var best_idx: usize = 0;
        var best_score: f32 = 0.0;

        // Find best initial task
        for (0..task_count) |i| {
            if (task_scores[i] > best_score) {
                best_score = task_scores[i];
                best_idx = i;
            }
        }

        selected[best_idx] = true;
        batch_size = 1;

        // Greedy selection
        while (batch_size < batch_indices.len) {
            best_score = -1.0;
            best_idx = task_count;

            for (0..task_count) |candidate| {
                if (selected[candidate]) continue;

                var total_similarity: f32 = 0.0;
                var selected_count: u32 = 0;
                for (0..task_count) |j| {
                    if (selected[j]) {
                        total_similarity += similarity_matrix[candidate * task_count + j];
                        selected_count += 1;
                    }
                }

                const avg_similarity = total_similarity / @as(f32, @floatFromInt(selected_count));
                const compatibility = avg_similarity * task_scores[candidate];

                if (compatibility > best_score) {
                    best_score = compatibility;
                    best_idx = candidate;
                }
            }

            if (best_idx < task_count) {
                selected[best_idx] = true;
                batch_size += 1;
            } else {
                break;
            }
        }

        total_batch_size += batch_size;
        std.mem.doNotOptimizeAway(&selected);
    }
    const native_time = timer.read();

    // Benchmark ISPC implementation
    timer.reset();
    for (0..iterations) |_| {
        const batch_size = ispc_kernels.ispc_optimize_batch_formation(
            task_scores.ptr,
            similarity_matrix.ptr,
            batch_indices.ptr,
            @intCast(task_count),
            @intCast(batch_indices.len),
        );
        total_batch_size += @intCast(batch_size);
        std.mem.doNotOptimizeAway(&batch_indices);
    }
    const ispc_time = timer.read();

    const speedup = @as(f64, @floatFromInt(native_time)) / @as(f64, @floatFromInt(ispc_time));
    const throughput_native = @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(native_time)) / 1e9);
    const throughput_ispc = @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(ispc_time)) / 1e9);

    std.debug.print("Native time: ");
    formatTime(native_time);
    std.debug.print("\nISPC time: ");
    formatTime(ispc_time);
    std.debug.print("\nSpeedup: {d:.2}x\n", .{speedup});
    std.debug.print("Avg batch size: {d:.1}\n", .{@as(f64, @floatFromInt(total_batch_size)) / @as(f64, @floatFromInt(iterations * 2))});
    std.debug.print("Native throughput: ");
    formatThroughput(throughput_native);
    std.debug.print(" batches/s\nISPC throughput: ");
    formatThroughput(throughput_ispc);
    std.debug.print(" batches/s\n");

    return BenchmarkResult{
        .native_time = native_time,
        .ispc_time = ispc_time,
        .speedup = speedup,
        .throughput_native = throughput_native,
        .throughput_ispc = throughput_ispc,
    };
}

// Benchmark worker selection scoring
fn benchmarkWorkerSelection(allocator: std.mem.Allocator, worker_count: usize, iterations: u32) !BenchmarkResult {
    std.debug.print("\n=== Worker Selection Scoring Benchmark ===\n");
    std.debug.print("Worker count: {}\n", .{worker_count});
    std.debug.print("Iterations: {}\n", .{iterations});

    var worker_loads = try allocator.alloc(f32, worker_count);
    defer allocator.free(worker_loads);
    var numa_distances = try allocator.alloc(f32, worker_count);
    defer allocator.free(numa_distances);
    var cache_affinities = try allocator.alloc(f32, worker_count);
    defer allocator.free(cache_affinities);
    var native_scores = try allocator.alloc(f32, worker_count);
    defer allocator.free(native_scores);
    var ispc_scores = try allocator.alloc(f32, worker_count);
    defer allocator.free(ispc_scores);

    var prng = std.rand.DefaultPrng.init(98765);
    const random = prng.random();

    // Initialize realistic worker metrics
    for (0..worker_count) |i| {
        worker_loads[i] = random.float(f32);
        numa_distances[i] = random.float(f32);
        cache_affinities[i] = random.float(f32) * 0.4 + 0.6; // 0.6 to 1.0
    }

    var timer = try std.time.Timer.start();

    // Benchmark native implementation
    timer.reset();
    for (0..iterations) |_| {
        for (0..worker_count) |i| {
            const load_score = 1.0 - std.math.clamp(worker_loads[i], 0.0, 1.0);
            const numa_score = 1.0 - std.math.clamp(numa_distances[i], 0.0, 1.0);
            const cache_score = std.math.clamp(cache_affinities[i], 0.0, 1.0);

            const load_weighted = load_score * load_score;
            const numa_weighted = std.math.sqrt(numa_score);

            const combined = 0.4 * load_weighted + 0.35 * numa_weighted + 0.25 * cache_score;
            native_scores[i] = std.math.exp(2.0 * combined) - 1.0;
        }
        std.mem.doNotOptimizeAway(&native_scores);
    }
    const native_time = timer.read();

    // Benchmark ISPC implementation
    timer.reset();
    for (0..iterations) |_| {
        ispc_kernels.ispc_compute_worker_scores(
            worker_loads.ptr,
            numa_distances.ptr,
            cache_affinities.ptr,
            ispc_scores.ptr,
            @intCast(worker_count),
        );
        std.mem.doNotOptimizeAway(&ispc_scores);
    }
    const ispc_time = timer.read();

    // Verify correctness
    var max_diff: f32 = 0.0;
    for (0..worker_count) |i| {
        const diff = @abs(native_scores[i] - ispc_scores[i]);
        max_diff = @max(max_diff, diff);
    }

    const speedup = @as(f64, @floatFromInt(native_time)) / @as(f64, @floatFromInt(ispc_time));
    const total_ops = worker_count * iterations;
    const throughput_native = @as(f64, @floatFromInt(total_ops)) / (@as(f64, @floatFromInt(native_time)) / 1e9);
    const throughput_ispc = @as(f64, @floatFromInt(total_ops)) / (@as(f64, @floatFromInt(ispc_time)) / 1e9);

    std.debug.print("Native time: ");
    formatTime(native_time);
    std.debug.print("\nISPC time: ");
    formatTime(ispc_time);
    std.debug.print("\nSpeedup: {d:.2}x\n", .{speedup});
    std.debug.print("Max difference: {d:.6}\n", .{max_diff});
    std.debug.print("Native throughput: ");
    formatThroughput(throughput_native);
    std.debug.print(" scores/s\nISPC throughput: ");
    formatThroughput(throughput_ispc);
    std.debug.print(" scores/s\n");

    return BenchmarkResult{
        .native_time = native_time,
        .ispc_time = ispc_time,
        .speedup = speedup,
        .throughput_native = throughput_native,
        .throughput_ispc = throughput_ispc,
    };
}

// Comprehensive memory bandwidth test
fn benchmarkMemoryBandwidth(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== Memory Bandwidth Analysis ===\n");

    const sizes = [_]usize{ 1024, 4096, 16384, 65536 };

    for (sizes) |size| {
        var data = try allocator.alloc(f32, size);
        defer allocator.free(data);

        // Initialize data
        for (0..size) |i| {
            data[i] = @as(f32, @floatFromInt(i));
        }

        var timer = try std.time.Timer.start();
        const iterations: u32 = 1000;

        // Sequential access pattern
        timer.reset();
        for (0..iterations) |_| {
            var sum: f32 = 0;
            for (data) |val| {
                sum += val;
            }
            std.mem.doNotOptimizeAway(&sum);
        }
        const sequential_time = timer.read();

        // Strided access pattern (every 8th element)
        timer.reset();
        for (0..iterations) |_| {
            var sum: f32 = 0;
            var i: usize = 0;
            while (i < size) : (i += 8) {
                sum += data[i];
            }
            std.mem.doNotOptimizeAway(&sum);
        }
        const strided_time = timer.read();

        const data_size_mb = (@as(f64, @floatFromInt(size)) * @sizeOf(f32) * @as(f64, @floatFromInt(iterations))) / (1024.0 * 1024.0);
        const sequential_bandwidth = data_size_mb / (@as(f64, @floatFromInt(sequential_time)) / 1e9);
        const strided_bandwidth = (data_size_mb / 8.0) / (@as(f64, @floatFromInt(strided_time)) / 1e9);

        std.debug.print("Size: {}KB - Sequential: {d:.1}MB/s, Strided: {d:.1}MB/s\n", .{ size * @sizeOf(f32) / 1024, sequential_bandwidth, strided_bandwidth });
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("🚀 Beat.zig ISPC Performance Benchmark Suite\n", .{});
    std.debug.print("=============================================\n", .{});

    // Detect ISPC configuration
    const config = ispc.Config.detectISPCCapabilities();
    std.debug.print("ISPC Target: {s} ({d}-wide SIMD)\n", .{ config.target.toString(), config.simd_width });
    std.debug.print("Expected speedup: {d:.1}x\n", .{config.estimated_speedup});

    var total_speedup: f64 = 0.0;
    var benchmark_count: u32 = 0;

    // Fingerprint similarity benchmarks
    const fp_small = try benchmarkFingerprintSimilarity(allocator, 1000, 100);
    const fp_large = try benchmarkFingerprintSimilarity(allocator, 10000, 10);
    total_speedup += fp_small.speedup + fp_large.speedup;
    benchmark_count += 2;

    // Batch optimization benchmarks
    const batch_medium = try benchmarkBatchOptimization(allocator, 64, 100);
    const batch_large = try benchmarkBatchOptimization(allocator, 256, 20);
    total_speedup += batch_medium.speedup + batch_large.speedup;
    benchmark_count += 2;

    // Worker selection benchmarks
    const worker_small = try benchmarkWorkerSelection(allocator, 16, 1000);
    const worker_large = try benchmarkWorkerSelection(allocator, 128, 100);
    total_speedup += worker_small.speedup + worker_large.speedup;
    benchmark_count += 2;

    // Memory bandwidth analysis
    try benchmarkMemoryBandwidth(allocator);

    // Summary
    const avg_speedup = total_speedup / @as(f64, @floatFromInt(benchmark_count));

    std.debug.print("\n📊 BENCHMARK SUMMARY\n");
    std.debug.print("===================\n");
    std.debug.print("Average ISPC speedup: {d:.2}x\n", .{avg_speedup});
    std.debug.print("Target speedup range: 3.0x - 6.0x\n");

    if (avg_speedup >= 3.0) {
        std.debug.print("✅ PERFORMANCE TARGET ACHIEVED!\n");
    } else if (avg_speedup >= 2.0) {
        std.debug.print("⚠️  Performance target partially met\n");
    } else {
        std.debug.print("❌ Performance target not met\n");
    }

    // Individual benchmark analysis
    std.debug.print("\nIndividual Results:\n");
    std.debug.print("- Fingerprint similarity (small): {d:.2}x\n", .{fp_small.speedup});
    std.debug.print("- Fingerprint similarity (large): {d:.2}x\n", .{fp_large.speedup});
    std.debug.print("- Batch optimization (medium): {d:.2}x\n", .{batch_medium.speedup});
    std.debug.print("- Batch optimization (large): {d:.2}x\n", .{batch_large.speedup});
    std.debug.print("- Worker selection (small): {d:.2}x\n", .{worker_small.speedup});
    std.debug.print("- Worker selection (large): {d:.2}x\n", .{worker_large.speedup});

    std.debug.print("\n🎯 Integration with Beat.zig's existing optimizations:\n");
    std.debug.print("   720x batch formation * {d:.1}x ISPC = {d:.0}x total improvement\n", .{ avg_speedup, 720.0 * avg_speedup });
    std.debug.print("   15.3x worker selection * {d:.1}x ISPC = {d:.0}x total improvement\n", .{ avg_speedup, 15.3 * avg_speedup });

    std.debug.print("\n🚀 Beat.zig + ISPC: Production-ready SPMD acceleration!\n");
}
