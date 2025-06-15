const std = @import("std");
const beat = @import("src/core.zig");

// Demonstration of Build System Auto-Configuration in Beat.zig

test "Build System Auto-Configuration Demo" {
    std.debug.print("\n============================================================\n", .{});
    std.debug.print("Beat.zig Build System Auto-Configuration Demo\n", .{});
    std.debug.print("============================================================\n", .{});
    
    // 1. Display detected hardware configuration
    std.debug.print("\n1. Hardware Detection:\n", .{});
    beat.build_opts.printSummary();
    
    // 2. Show different pool configurations
    std.debug.print("\n2. Auto-Configured Thread Pools:\n", .{});
    
    const allocator = std.testing.allocator;
    
    // Standard pool
    const standard_pool = try beat.createPool(allocator);
    defer standard_pool.deinit();
    std.debug.print("   Standard Pool: {} workers\n", .{standard_pool.workers.len});
    
    // Optimal pool (auto-configured)
    const optimal_pool = try beat.createOptimalPool(allocator);
    defer optimal_pool.deinit();
    std.debug.print("   Optimal Pool: {} workers (auto-detected)\n", .{optimal_pool.workers.len});
    
    // Test pool
    const test_pool = try beat.createTestPool(allocator);
    defer test_pool.deinit();
    std.debug.print("   Test Pool: {} workers (simplified for testing)\n", .{test_pool.workers.len});
    
    // Benchmark pool
    const benchmark_pool = try beat.createBenchmarkPool(allocator);
    defer benchmark_pool.deinit();
    std.debug.print("   Benchmark Pool: {} workers (optimized for performance)\n", .{benchmark_pool.workers.len});
    
    // 3. Show SIMD optimization capabilities
    std.debug.print("\n3. SIMD Optimization Features:\n", .{});
    std.debug.print("   SIMD Available: {}\n", .{beat.build_opts.cpu_features.has_simd});
    std.debug.print("   SIMD Width: {} bytes\n", .{beat.build_opts.cpu_features.simd_width});
    
    if (beat.build_opts.cpu_features.has_simd) {
        // Demonstrate vector types
        const FloatVec = beat.build_opts.OptimalVector(f32);
        const IntVec = beat.build_opts.OptimalVector(i32);
        
        if (@typeInfo(FloatVec) == .vector) {
            std.debug.print("   Optimal f32 Vector: {} elements\n", .{@typeInfo(FloatVec).vector.len});
        } else {
            std.debug.print("   Optimal f32 Vector: scalar (no vectorization)\n", .{});
        }
        
        if (@typeInfo(IntVec) == .vector) {
            std.debug.print("   Optimal i32 Vector: {} elements\n", .{@typeInfo(IntVec).vector.len});
        } else {
            std.debug.print("   Optimal i32 Vector: scalar (no vectorization)\n", .{});
        }
        
        std.debug.print("   Should vectorize f32: {}\n", .{beat.build_opts.shouldVectorize(f32)});
        std.debug.print("   Should vectorize f64: {}\n", .{beat.build_opts.shouldVectorize(f64)});
        std.debug.print("   SIMD alignment for f32: {} bytes\n", .{beat.build_opts.getSimdAlignment(f32)});
    }
    
    // 4. Configuration comparison
    std.debug.print("\n4. Configuration Comparison:\n", .{});
    
    const default_config = beat.Config{};
    const optimal_config = beat.build_opts.getOptimalConfig();
    const test_config = beat.build_opts.getTestConfig();
    const benchmark_config = beat.build_opts.getBenchmarkConfig();
    
    std.debug.print("   Configuration        Workers  Queue   OneEuro(β)  NUMA  Topo\n", .{});
    std.debug.print("   ------------------- -------- ------- ----------- ----- -----\n", .{});
    std.debug.print("   Default             {}        {}      {d:.3}       {}    {}\n", .{
        default_config.num_workers orelse 0,
        default_config.task_queue_size,
        default_config.prediction_beta,
        default_config.enable_numa_aware,
        default_config.enable_topology_aware,
    });
    std.debug.print("   Optimal (Auto)      {}        {}       {d:.3}       {}     {}\n", .{
        optimal_config.num_workers.?,
        optimal_config.task_queue_size,
        optimal_config.prediction_beta,
        optimal_config.enable_numa_aware,
        optimal_config.enable_topology_aware,
    });
    std.debug.print("   Test                {}        {}       {d:.3}       {}     {}\n", .{
        test_config.num_workers.?,
        test_config.task_queue_size,
        test_config.prediction_beta,
        test_config.enable_numa_aware,
        test_config.enable_topology_aware,
    });
    std.debug.print("   Benchmark           {}        {}      {d:.3}       {}     {}\n", .{
        benchmark_config.num_workers.?,
        benchmark_config.task_queue_size,
        benchmark_config.prediction_beta,
        benchmark_config.enable_numa_aware,
        benchmark_config.enable_topology_aware,
    });
    
    // 5. Build-time vs Runtime detection
    std.debug.print("\n5. Build-Time vs Runtime Detection:\n", .{});
    std.debug.print("   Build-time CPU count: {}\n", .{beat.build_opts.hardware.cpu_count});
    
    const runtime_cpu_count = std.Thread.getCpuCount() catch 0;
    std.debug.print("   Runtime CPU count: {}\n", .{runtime_cpu_count});
    
    if (beat.build_opts.hardware.cpu_count == runtime_cpu_count) {
        std.debug.print("   ✓ Build-time detection matches runtime\n", .{});
    } else {
        std.debug.print("   ⚠ Build-time detection differs from runtime\n", .{});
        std.debug.print("     (This is normal for cross-compilation)\n", .{});
    }
    
    // 6. Performance implications
    std.debug.print("\n6. Performance Optimizations Applied:\n", .{});
    
    if (beat.build_opts.performance.enable_numa_aware) {
        std.debug.print("   ✓ NUMA-aware allocation enabled\n", .{});
    }
    
    if (beat.build_opts.performance.enable_topology_aware) {
        std.debug.print("   ✓ Topology-aware scheduling enabled\n", .{});
    } else {
        std.debug.print("   - Topology-aware scheduling disabled (fewer cores)\n", .{});
    }
    
    if (beat.build_opts.cpu_features.has_avx512) {
        std.debug.print("   ✓ AVX-512 vectorization available\n", .{});
    } else if (beat.build_opts.cpu_features.has_avx2) {
        std.debug.print("   ✓ AVX2 vectorization available\n", .{});
    } else if (beat.build_opts.cpu_features.has_avx) {
        std.debug.print("   ✓ AVX vectorization available\n", .{});
    }
    
    std.debug.print("   ✓ One Euro Filter tuned for detected hardware\n", .{});
    std.debug.print("     min_cutoff: {d:.2} Hz, beta: {d:.3}\n", .{
        beat.build_opts.performance.one_euro_min_cutoff,
        beat.build_opts.performance.one_euro_beta,
    });
    
    std.debug.print("\n============================================================\n", .{});
    std.debug.print("Build System Auto-Configuration Complete!\n", .{});
    std.debug.print("Beat.zig automatically optimized for your system.\n", .{});
    std.debug.print("============================================================\n", .{});
}

test "Zero-Configuration Usage Examples" {
    std.debug.print("\nZero-Configuration Usage Examples:\n", .{});
    
    const allocator = std.testing.allocator;
    
    // Example 1: Just works out of the box
    std.debug.print("\n1. Instant optimal performance:\n", .{});
    std.debug.print("   const pool = try beat.createOptimalPool(allocator);\n", .{});
    std.debug.print("   // Automatically uses {} workers, {} queue size\n", .{
        beat.build_opts.hardware.optimal_workers,
        beat.build_opts.hardware.optimal_queue_size,
    });
    
    const demo_pool = try beat.createOptimalPool(allocator);
    defer demo_pool.deinit();
    
    try std.testing.expect(demo_pool.workers.len == beat.build_opts.hardware.optimal_workers);
    
    // Example 2: Testing made simple
    std.debug.print("\n2. Testing with optimal configuration:\n", .{});
    std.debug.print("   const test_pool = try beat.createTestPool(allocator);\n", .{});
    std.debug.print("   // Automatically uses {} workers for stable testing\n", .{
        beat.build_opts.hardware.optimal_test_threads,
    });
    
    // Example 3: Benchmarking optimized
    std.debug.print("\n3. Benchmarking with maximum performance:\n", .{});
    std.debug.print("   const bench_pool = try beat.createBenchmarkPool(allocator);\n", .{});
    std.debug.print("   // Automatically uses large queues and aggressive tuning\n", .{});
    
    std.debug.print("\nNo configuration needed - Beat.zig adapts to your hardware!\n", .{});
}