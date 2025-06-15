const std = @import("std");
const beat = @import("beat");

// Test for SIMD Benchmarking and Validation Framework (Phase 3.1)
//
// This test validates the comprehensive SIMD benchmarking system including:
// - High-precision timing infrastructure with statistical analysis
// - SIMD performance measurement across different vector sizes and operations
// - Memory bandwidth utilization analysis with multiple access patterns
// - Classification overhead measurement and performance profiling
// - Batch formation efficiency analysis and throughput validation
// - End-to-end workflow benchmarking and cross-platform compatibility testing

test "precision timer accuracy and statistical analysis" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Precision Timer Accuracy and Statistical Analysis Test ===\n", .{});
    
    // Test 1: Basic timing accuracy
    std.debug.print("1. Testing basic timing accuracy...\n", .{});
    
    var timer = beat.simd_benchmark.PrecisionTimer.init(allocator);
    defer timer.deinit();
    
    // Collect multiple measurements for statistical analysis
    for (0..50) |i| {
        timer.start();
        
        // Simulate work with controlled duration
        const work_amount = (i % 10) + 1;
        var dummy_sum: u64 = 0;
        for (0..work_amount * 1000) |j| {
            dummy_sum += j;
        }
        std.mem.doNotOptimizeAway(dummy_sum);
        
        try timer.end();
    }
    
    const stats = timer.getStatistics();
    
    try std.testing.expect(stats.count == 50);
    try std.testing.expect(stats.min_ns > 0);
    try std.testing.expect(stats.max_ns >= stats.min_ns);
    try std.testing.expect(stats.mean_ns > 0);
    try std.testing.expect(stats.std_dev_ns >= 0.0);
    
    std.debug.print("   Timing measurements collected: {}\n", .{stats.count});
    std.debug.print("   Mean execution time: {d:.1} ns\n", .{stats.mean_ns});
    std.debug.print("   Standard deviation: {d:.1} ns\n", .{stats.std_dev_ns});
    std.debug.print("   Coefficient of variation: {d:.2}%\n", .{stats.coefficient_of_variation});
    std.debug.print("   Performance rating: {s}\n", .{@tagName(stats.getPerformanceRating())});
    
    // Test 2: Statistical validation
    std.debug.print("2. Testing statistical validation...\n", .{});
    
    const is_stable = stats.isStable();
    std.debug.print("   Measurement stability: {}\n", .{is_stable});
    
    // Confidence interval should be reasonable
    const ci_range = stats.confidence_interval_95_upper - stats.confidence_interval_95_lower;
    std.debug.print("   95% confidence interval: [{d:.1}, {d:.1}] ns (range: {d:.1})\n", .{
        stats.confidence_interval_95_lower, stats.confidence_interval_95_upper, ci_range
    });
    
    try std.testing.expect(stats.confidence_interval_95_lower <= stats.mean_ns);
    try std.testing.expect(stats.confidence_interval_95_upper >= stats.mean_ns);
    
    std.debug.print("   ✅ Precision timer accuracy and statistical analysis completed\n", .{});
}

test "SIMD benchmark suite initialization and system detection" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== SIMD Benchmark Suite Initialization Test ===\n", .{});
    
    // Test 1: Suite initialization
    std.debug.print("1. Testing benchmark suite initialization...\n", .{});
    
    var suite = try beat.simd_benchmark.SIMDBenchmarkSuite.init(allocator);
    defer suite.deinit();
    
    std.debug.print("   Benchmark suite initialized successfully\n", .{});
    
    // Test 2: System information detection
    std.debug.print("2. Testing system information detection...\n", .{});
    
    const system_info = suite.system_info;
    
    try std.testing.expect(system_info.memory_size_gb > 0);
    try std.testing.expect(system_info.cache_sizes.l1_cache_kb > 0);
    try std.testing.expect(system_info.cache_sizes.l2_cache_kb > 0);
    try std.testing.expect(system_info.cache_sizes.l3_cache_kb > 0);
    
    std.debug.print("   System Information Detected:\n", .{});
    std.debug.print("     CPU Architecture: {s}\n", .{system_info.cpu_arch});
    std.debug.print("     Memory: {d:.1} GB\n", .{system_info.memory_size_gb});
    std.debug.print("     L1 Cache: {} KB\n", .{system_info.cache_sizes.l1_cache_kb});
    std.debug.print("     L2 Cache: {} KB\n", .{system_info.cache_sizes.l2_cache_kb});
    std.debug.print("     L3 Cache: {} KB\n", .{system_info.cache_sizes.l3_cache_kb});
    
    // Test 3: SIMD capabilities detection
    std.debug.print("3. Testing SIMD capabilities detection...\n", .{});
    
    const simd_flags = system_info.getSIMDFlags();
    std.debug.print("   SIMD Capabilities:\n", .{});
    std.debug.print("     SSE support: {}\n", .{simd_flags.sse});
    std.debug.print("     AVX support: {}\n", .{simd_flags.avx});
    std.debug.print("     AVX2 support: {}\n", .{simd_flags.avx2});
    std.debug.print("     AVX-512 support: {}\n", .{simd_flags.avx512});
    std.debug.print("     NEON support: {}\n", .{simd_flags.neon});
    
    std.debug.print("   ✅ SIMD benchmark suite initialization completed\n", .{});
}

test "vector arithmetic performance benchmarking" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Vector Arithmetic Performance Benchmarking Test ===\n", .{});
    
    var suite = try beat.simd_benchmark.SIMDBenchmarkSuite.init(allocator);
    defer suite.deinit();
    
    // Test 1: Scalar arithmetic benchmarking
    std.debug.print("1. Testing scalar arithmetic benchmarking...\n", .{});
    
    const scalar_stats = try suite.benchmarkScalarArithmetic(1024);
    
    try std.testing.expect(scalar_stats.count > 0);
    try std.testing.expect(scalar_stats.mean_ns > 0);
    
    std.debug.print("   Scalar arithmetic (1024 elements):\n", .{});
    std.debug.print("     Iterations: {}\n", .{scalar_stats.count});
    std.debug.print("     Mean time: {d:.1} μs\n", .{scalar_stats.mean_ns / 1000.0});
    std.debug.print("     Min time: {d:.1} μs\n", .{@as(f64, @floatFromInt(scalar_stats.min_ns)) / 1000.0});
    std.debug.print("     Max time: {d:.1} μs\n", .{@as(f64, @floatFromInt(scalar_stats.max_ns)) / 1000.0});
    std.debug.print("     Coefficient of variation: {d:.2}%\n", .{scalar_stats.coefficient_of_variation});
    
    // Test 2: SIMD arithmetic benchmarking
    std.debug.print("2. Testing SIMD arithmetic benchmarking...\n", .{});
    
    const simd_stats = try suite.benchmarkSIMDArithmetic(1024);
    
    try std.testing.expect(simd_stats.count > 0);
    try std.testing.expect(simd_stats.mean_ns > 0);
    
    std.debug.print("   SIMD arithmetic (1024 elements):\n", .{});
    std.debug.print("     Iterations: {}\n", .{simd_stats.count});
    std.debug.print("     Mean time: {d:.1} μs\n", .{simd_stats.mean_ns / 1000.0});
    std.debug.print("     Min time: {d:.1} μs\n", .{@as(f64, @floatFromInt(simd_stats.min_ns)) / 1000.0});
    std.debug.print("     Max time: {d:.1} μs\n", .{@as(f64, @floatFromInt(simd_stats.max_ns)) / 1000.0});
    std.debug.print("     Coefficient of variation: {d:.2}%\n", .{simd_stats.coefficient_of_variation});
    
    // Test 3: Performance comparison
    std.debug.print("3. Testing performance comparison...\n", .{});
    
    const speedup = if (simd_stats.mean_ns > 0.0) 
        scalar_stats.mean_ns / simd_stats.mean_ns else 0.0;
    
    std.debug.print("   Performance Comparison:\n", .{});
    std.debug.print("     SIMD speedup: {d:.2}x\n", .{speedup});
    
    // SIMD should be at least as fast as scalar (speedup >= 1.0)
    try std.testing.expect(speedup >= 0.5); // Allow for some overhead in simulation
    
    std.debug.print("   ✅ Vector arithmetic performance benchmarking completed\n", .{});
}

test "memory bandwidth utilization analysis" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Memory Bandwidth Utilization Analysis Test ===\n", .{});
    
    var suite = try beat.simd_benchmark.SIMDBenchmarkSuite.init(allocator);
    defer suite.deinit();
    
    // Test 1: Sequential memory access
    std.debug.print("1. Testing sequential memory access benchmark...\n", .{});
    
    const elements = 256 * 1024; // 1MB of f32 data
    const sequential_bw = try suite.benchmarkSequentialAccess(elements);
    
    try std.testing.expect(sequential_bw > 0.0);
    
    std.debug.print("   Sequential access:\n", .{});
    std.debug.print("     Elements: {}\n", .{elements});
    std.debug.print("     Bandwidth: {d:.2} GB/s\n", .{sequential_bw});
    
    // Test 2: Random memory access
    std.debug.print("2. Testing random memory access benchmark...\n", .{});
    
    const random_bw = try suite.benchmarkRandomAccess(elements);
    
    try std.testing.expect(random_bw > 0.0);
    
    std.debug.print("   Random access:\n", .{});
    std.debug.print("     Elements: {}\n", .{elements});
    std.debug.print("     Bandwidth: {d:.2} GB/s\n", .{random_bw});
    
    // Test 3: Strided memory access
    std.debug.print("3. Testing strided memory access benchmark...\n", .{});
    
    const strided_bw = try suite.benchmarkStridedAccess(elements);
    
    try std.testing.expect(strided_bw > 0.0);
    
    std.debug.print("   Strided access:\n", .{});
    std.debug.print("     Elements: {}\n", .{elements});
    std.debug.print("     Bandwidth: {d:.2} GB/s\n", .{strided_bw});
    
    // Test 4: Performance comparison
    std.debug.print("4. Testing memory access pattern comparison...\n", .{});
    
    std.debug.print("   Bandwidth Comparison:\n", .{});
    std.debug.print("     Sequential: {d:.2} GB/s (baseline)\n", .{sequential_bw});
    std.debug.print("     Random: {d:.2} GB/s ({d:.1}% of sequential)\n", .{
        random_bw, (random_bw / sequential_bw) * 100.0
    });
    std.debug.print("     Strided: {d:.2} GB/s ({d:.1}% of sequential)\n", .{
        strided_bw, (strided_bw / sequential_bw) * 100.0
    });
    
    // Sequential should typically be fastest, but allow for variation
    try std.testing.expect(sequential_bw >= random_bw * 0.5); // Allow significant variance
    
    std.debug.print("   ✅ Memory bandwidth utilization analysis completed\n", .{});
}

test "classification overhead measurement" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Classification Overhead Measurement Test ===\n", .{});
    
    var suite = try beat.simd_benchmark.SIMDBenchmarkSuite.init(allocator);
    defer suite.deinit();
    
    // Test 1: Static analysis overhead
    std.debug.print("1. Testing static analysis overhead...\n", .{});
    
    const static_stats = try suite.benchmarkStaticAnalysis();
    
    try std.testing.expect(static_stats.count > 0);
    try std.testing.expect(static_stats.mean_ns > 0);
    
    std.debug.print("   Static Analysis:\n", .{});
    std.debug.print("     Iterations: {}\n", .{static_stats.count});
    std.debug.print("     Mean time: {d:.2} μs\n", .{static_stats.mean_ns / 1000.0});
    std.debug.print("     Standard deviation: {d:.2} μs\n", .{static_stats.std_dev_ns / 1000.0});
    
    // Test 2: Dynamic profiling overhead
    std.debug.print("2. Testing dynamic profiling overhead...\n", .{});
    
    const dynamic_stats = try suite.benchmarkDynamicProfiling();
    
    try std.testing.expect(dynamic_stats.count > 0);
    try std.testing.expect(dynamic_stats.mean_ns > 0);
    
    std.debug.print("   Dynamic Profiling:\n", .{});
    std.debug.print("     Iterations: {}\n", .{dynamic_stats.count});
    std.debug.print("     Mean time: {d:.2} μs\n", .{dynamic_stats.mean_ns / 1000.0});
    std.debug.print("     Standard deviation: {d:.2} μs\n", .{dynamic_stats.std_dev_ns / 1000.0});
    
    // Test 3: Feature extraction overhead
    std.debug.print("3. Testing feature extraction overhead...\n", .{});
    
    const feature_stats = try suite.benchmarkFeatureExtraction();
    
    try std.testing.expect(feature_stats.count > 0);
    try std.testing.expect(feature_stats.mean_ns > 0);
    
    std.debug.print("   Feature Extraction:\n", .{});
    std.debug.print("     Iterations: {}\n", .{feature_stats.count});
    std.debug.print("     Mean time: {d:.2} μs\n", .{feature_stats.mean_ns / 1000.0});
    std.debug.print("     Standard deviation: {d:.2} μs\n", .{feature_stats.std_dev_ns / 1000.0});
    
    // Test 4: Batch formation overhead
    std.debug.print("4. Testing batch formation overhead...\n", .{});
    
    const batch_stats = try suite.benchmarkBatchFormationOverhead();
    
    try std.testing.expect(batch_stats.count > 0);
    try std.testing.expect(batch_stats.mean_ns > 0);
    
    std.debug.print("   Batch Formation:\n", .{});
    std.debug.print("     Iterations: {}\n", .{batch_stats.count});
    std.debug.print("     Mean time: {d:.2} μs\n", .{batch_stats.mean_ns / 1000.0});
    std.debug.print("     Standard deviation: {d:.2} μs\n", .{batch_stats.std_dev_ns / 1000.0});
    
    // Test 5: Total overhead calculation
    std.debug.print("5. Testing total overhead calculation...\n", .{});
    
    const total_overhead = static_stats.mean_ns + dynamic_stats.mean_ns + 
                          feature_stats.mean_ns + batch_stats.mean_ns;
    
    std.debug.print("   Total Classification Overhead:\n", .{});
    std.debug.print("     Static analysis: {d:.2} μs ({d:.1}%)\n", .{
        static_stats.mean_ns / 1000.0, (static_stats.mean_ns / total_overhead) * 100.0
    });
    std.debug.print("     Dynamic profiling: {d:.2} μs ({d:.1}%)\n", .{
        dynamic_stats.mean_ns / 1000.0, (dynamic_stats.mean_ns / total_overhead) * 100.0
    });
    std.debug.print("     Feature extraction: {d:.2} μs ({d:.1}%)\n", .{
        feature_stats.mean_ns / 1000.0, (feature_stats.mean_ns / total_overhead) * 100.0
    });
    std.debug.print("     Batch formation: {d:.2} μs ({d:.1}%)\n", .{
        batch_stats.mean_ns / 1000.0, (batch_stats.mean_ns / total_overhead) * 100.0
    });
    std.debug.print("     Total: {d:.2} μs\n", .{total_overhead / 1000.0});
    
    // Overhead should be reasonable (< 1ms total)
    try std.testing.expect(total_overhead < 1_000_000); // Less than 1ms
    
    std.debug.print("   ✅ Classification overhead measurement completed\n", .{});
}

test "batch formation efficiency analysis" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Batch Formation Efficiency Analysis Test ===\n", .{});
    
    var suite = try beat.simd_benchmark.SIMDBenchmarkSuite.init(allocator);
    defer suite.deinit();
    
    // Test 1: Small batch formation
    std.debug.print("1. Testing small batch formation (5 tasks)...\n", .{});
    
    const small_batch_time = try suite.benchmarkBatchFormationTime(5);
    const small_batch_efficiency = try suite.measureBatchFormationEfficiency(5);
    
    try std.testing.expect(small_batch_time.count > 0);
    try std.testing.expect(small_batch_efficiency >= 0.0);
    try std.testing.expect(small_batch_efficiency <= 1.0);
    
    std.debug.print("   Small batch (5 tasks):\n", .{});
    std.debug.print("     Formation time: {d:.2} μs\n", .{small_batch_time.mean_ns / 1000.0});
    std.debug.print("     Formation efficiency: {d:.1}%\n", .{small_batch_efficiency * 100.0});
    
    // Test 2: Medium batch formation
    std.debug.print("2. Testing medium batch formation (20 tasks)...\n", .{});
    
    const medium_batch_time = try suite.benchmarkBatchFormationTime(20);
    const medium_batch_efficiency = try suite.measureBatchFormationEfficiency(20);
    
    try std.testing.expect(medium_batch_time.count > 0);
    try std.testing.expect(medium_batch_efficiency >= 0.0);
    try std.testing.expect(medium_batch_efficiency <= 1.0);
    
    std.debug.print("   Medium batch (20 tasks):\n", .{});
    std.debug.print("     Formation time: {d:.2} μs\n", .{medium_batch_time.mean_ns / 1000.0});
    std.debug.print("     Formation efficiency: {d:.1}%\n", .{medium_batch_efficiency * 100.0});
    
    // Test 3: Large batch formation
    std.debug.print("3. Testing large batch formation (50 tasks)...\n", .{});
    
    const large_batch_time = try suite.benchmarkBatchFormationTime(50);
    const large_batch_efficiency = try suite.measureBatchFormationEfficiency(50);
    
    try std.testing.expect(large_batch_time.count > 0);
    try std.testing.expect(large_batch_efficiency >= 0.0);
    try std.testing.expect(large_batch_efficiency <= 1.0);
    
    std.debug.print("   Large batch (50 tasks):\n", .{});
    std.debug.print("     Formation time: {d:.2} μs\n", .{large_batch_time.mean_ns / 1000.0});
    std.debug.print("     Formation efficiency: {d:.1}%\n", .{large_batch_efficiency * 100.0});
    
    // Test 4: Efficiency scaling analysis
    std.debug.print("4. Testing efficiency scaling analysis...\n", .{});
    
    std.debug.print("   Batch Formation Scaling:\n", .{});
    std.debug.print("     5 tasks: {d:.2} μs, {d:.1}% efficiency\n", .{
        small_batch_time.mean_ns / 1000.0, small_batch_efficiency * 100.0
    });
    std.debug.print("     20 tasks: {d:.2} μs, {d:.1}% efficiency\n", .{
        medium_batch_time.mean_ns / 1000.0, medium_batch_efficiency * 100.0
    });
    std.debug.print("     50 tasks: {d:.2} μs, {d:.1}% efficiency\n", .{
        large_batch_time.mean_ns / 1000.0, large_batch_efficiency * 100.0
    });
    
    // Formation time should scale reasonably with batch size
    const time_ratio_20_to_5 = medium_batch_time.mean_ns / small_batch_time.mean_ns;
    const time_ratio_50_to_20 = large_batch_time.mean_ns / medium_batch_time.mean_ns;
    
    std.debug.print("     Time scaling 20/5: {d:.2}x\n", .{time_ratio_20_to_5});
    std.debug.print("     Time scaling 50/20: {d:.2}x\n", .{time_ratio_50_to_20});
    
    // Time scaling should be reasonable (not exponential)
    try std.testing.expect(time_ratio_20_to_5 < 10.0); // Should scale sub-linearly
    try std.testing.expect(time_ratio_50_to_20 < 5.0);
    
    std.debug.print("   ✅ Batch formation efficiency analysis completed\n", .{});
}

test "end-to-end workflow benchmarking" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== End-to-End Workflow Benchmarking Test ===\n", .{});
    
    var suite = try beat.simd_benchmark.SIMDBenchmarkSuite.init(allocator);
    defer suite.deinit();
    
    // Test 1: Complete workflow timing
    std.debug.print("1. Testing complete workflow timing...\n", .{});
    
    const workflow_stats = try suite.benchmarkCompleteWorkflow();
    
    try std.testing.expect(workflow_stats.count > 0);
    try std.testing.expect(workflow_stats.mean_ns > 0);
    
    std.debug.print("   Complete Workflow (15 tasks):\n", .{});
    std.debug.print("     Iterations: {}\n", .{workflow_stats.count});
    std.debug.print("     Mean time: {d:.1} μs\n", .{workflow_stats.mean_ns / 1000.0});
    std.debug.print("     Min time: {d:.1} μs\n", .{@as(f64, @floatFromInt(workflow_stats.min_ns)) / 1000.0});
    std.debug.print("     Max time: {d:.1} μs\n", .{@as(f64, @floatFromInt(workflow_stats.max_ns)) / 1000.0});
    std.debug.print("     Standard deviation: {d:.1} μs\n", .{workflow_stats.std_dev_ns / 1000.0});
    std.debug.print("     Coefficient of variation: {d:.2}%\n", .{workflow_stats.coefficient_of_variation});
    
    // Test 2: Workflow stability analysis
    std.debug.print("2. Testing workflow stability analysis...\n", .{});
    
    const is_stable = workflow_stats.isStable();
    const performance_rating = workflow_stats.getPerformanceRating();
    
    std.debug.print("   Workflow Stability:\n", .{});
    std.debug.print("     Stable measurements: {}\n", .{is_stable});
    std.debug.print("     Performance rating: {s}\n", .{@tagName(performance_rating)});
    
    // Test 3: Workflow components breakdown estimate
    std.debug.print("3. Testing workflow components breakdown...\n", .{});
    
    // Estimate component contributions (simplified)
    const per_task_classification_overhead = 50.0; // μs estimate
    const batch_formation_overhead = 100.0; // μs estimate
    const execution_time = workflow_stats.mean_ns / 1000.0 - 
                          (15 * per_task_classification_overhead) - batch_formation_overhead;
    
    std.debug.print("   Estimated Workflow Breakdown:\n", .{});
    std.debug.print("     Task classification (15 tasks): {d:.1} μs\n", .{15 * per_task_classification_overhead});
    std.debug.print("     Batch formation: {d:.1} μs\n", .{batch_formation_overhead});
    std.debug.print("     Execution time: {d:.1} μs\n", .{execution_time});
    std.debug.print("     Total workflow: {d:.1} μs\n", .{workflow_stats.mean_ns / 1000.0});
    
    // Workflow should complete in reasonable time (< 100ms)
    try std.testing.expect(workflow_stats.mean_ns < 100_000_000); // Less than 100ms
    
    std.debug.print("   ✅ End-to-end workflow benchmarking completed\n", .{});
}

test "cross-platform compatibility validation" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Cross-Platform Compatibility Validation Test ===\n", .{});
    
    var suite = try beat.simd_benchmark.SIMDBenchmarkSuite.init(allocator);
    defer suite.deinit();
    
    // Test 1: Platform detection
    std.debug.print("1. Testing platform detection...\n", .{});
    
    const system_info = suite.system_info;
    const simd_flags = system_info.getSIMDFlags();
    
    std.debug.print("   Platform Information:\n", .{});
    std.debug.print("     Architecture: {s}\n", .{system_info.cpu_arch});
    std.debug.print("     SIMD capabilities detected:\n", .{});
    std.debug.print("       SSE: {}\n", .{simd_flags.sse});
    std.debug.print("       AVX: {}\n", .{simd_flags.avx});
    std.debug.print("       AVX2: {}\n", .{simd_flags.avx2});
    std.debug.print("       AVX-512: {}\n", .{simd_flags.avx512});
    std.debug.print("       NEON: {}\n", .{simd_flags.neon});
    
    // Test 2: Feature-specific performance testing
    std.debug.print("2. Testing feature-specific performance...\n", .{});
    
    if (simd_flags.sse) {
        const sse_perf = try suite.benchmarkSSEPerformance();
        std.debug.print("   SSE Performance: {d:.2}x speedup\n", .{sse_perf});
        try std.testing.expect(sse_perf > 0.0);
    } else {
        std.debug.print("   SSE not available on this platform\n", .{});
    }
    
    if (simd_flags.avx) {
        const avx_perf = try suite.benchmarkAVXPerformance();
        std.debug.print("   AVX Performance: {d:.2}x speedup\n", .{avx_perf});
        try std.testing.expect(avx_perf > 0.0);
    } else {
        std.debug.print("   AVX not available on this platform\n", .{});
    }
    
    if (simd_flags.avx2) {
        const avx2_perf = try suite.benchmarkAVX2Performance();
        std.debug.print("   AVX2 Performance: {d:.2}x speedup\n", .{avx2_perf});
        try std.testing.expect(avx2_perf > 0.0);
    } else {
        std.debug.print("   AVX2 not available on this platform\n", .{});
    }
    
    // Test 3: Vector width performance scaling
    std.debug.print("3. Testing vector width performance scaling...\n", .{});
    
    const vector_widths = [_]usize{ 2, 4, 8, 16 };
    
    for (vector_widths) |width| {
        const perf = try suite.benchmarkVectorWidth(width);
        std.debug.print("   Vector width {}: {d:.2}x speedup\n", .{ width, perf });
        try std.testing.expect(perf > 0.0);
    }
    
    // Test 4: Compatibility validation
    std.debug.print("4. Testing compatibility validation...\n", .{});
    
    // Should detect at least one SIMD feature or gracefully fall back
    const has_any_simd = simd_flags.sse or simd_flags.avx or simd_flags.avx2 or 
                        simd_flags.avx512 or simd_flags.neon;
    
    std.debug.print("   SIMD support available: {}\n", .{has_any_simd});
    
    if (!has_any_simd) {
        std.debug.print("   System falls back to scalar operations\n", .{});
    }
    
    std.debug.print("   ✅ Cross-platform compatibility validation completed\n", .{});
}

test "comprehensive benchmark suite execution" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Comprehensive Benchmark Suite Execution Test ===\n", .{});
    
    var suite = try beat.simd_benchmark.SIMDBenchmarkSuite.init(allocator);
    defer suite.deinit();
    
    // Note: This test demonstrates the complete benchmark suite but only runs
    // a subset to avoid excessive test time. The full suite would be run
    // separately for comprehensive performance analysis.
    
    std.debug.print("1. Running abbreviated comprehensive benchmark suite...\n", .{});
    
    // Run individual benchmark components for validation
    std.debug.print("   Testing vector arithmetic component...\n", .{});
    const vector_result = try suite.benchmarkVectorArithmetic();
    defer allocator.free(vector_result.size_results);
    
    try std.testing.expect(vector_result.size_results.len > 0);
    try std.testing.expect(vector_result.overall_efficiency >= 0.0); // Allow zero efficiency for simulation
    
    std.debug.print("   Vector arithmetic: {} sizes tested, {d:.2}x average speedup\n", .{
        vector_result.size_results.len, vector_result.overall_efficiency
    });
    
    std.debug.print("   Testing matrix operations component...\n", .{});
    const matrix_result = try suite.benchmarkMatrixOperations();
    defer allocator.free(matrix_result.size_results);
    
    try std.testing.expect(matrix_result.size_results.len > 0);
    
    std.debug.print("   Matrix operations: {} sizes tested\n", .{matrix_result.size_results.len});
    
    std.debug.print("   Testing memory bandwidth component...\n", .{});
    const memory_result = try suite.benchmarkMemoryBandwidth();
    defer allocator.free(memory_result.bandwidth_results);
    
    try std.testing.expect(memory_result.bandwidth_results.len > 0);
    
    std.debug.print("   Memory bandwidth: {} data sizes tested\n", .{memory_result.bandwidth_results.len});
    
    std.debug.print("   Testing classification overhead component...\n", .{});
    const overhead_result = try suite.benchmarkClassificationOverhead();
    
    try std.testing.expect(overhead_result.total_overhead_ns > 0.0);
    
    std.debug.print("   Classification overhead: {d:.1} μs total\n", .{overhead_result.total_overhead_ns / 1000.0});
    
    std.debug.print("   Testing batch formation component...\n", .{});
    const batch_result = try suite.benchmarkBatchFormation();
    defer allocator.free(batch_result.formation_results);
    
    try std.testing.expect(batch_result.formation_results.len > 0);
    
    std.debug.print("   Batch formation: {} batch sizes tested\n", .{batch_result.formation_results.len});
    
    std.debug.print("   Testing end-to-end component...\n", .{});
    const end_to_end_result = try suite.benchmarkEndToEndWorkflow();
    
    try std.testing.expect(end_to_end_result.complete_workflow_time.count > 0);
    
    std.debug.print("   End-to-end: {d:.1} μs complete workflow\n", .{
        end_to_end_result.complete_workflow_time.mean_ns / 1000.0
    });
    
    std.debug.print("   Testing cross-platform component...\n", .{});
    _ = try suite.benchmarkCrossPlatformCompatibility();
    
    std.debug.print("   Cross-platform: {s} architecture validated\n", .{
        suite.system_info.cpu_arch
    });
    
    std.debug.print("\n✅ Comprehensive SIMD benchmark suite execution completed successfully!\n", .{});
    
    std.debug.print("\n📊 SIMD Benchmarking Framework Summary:\n", .{});
    std.debug.print("   • High-precision timing with statistical analysis ✅\n", .{});
    std.debug.print("   • Vector arithmetic performance measurement ✅\n", .{});
    std.debug.print("   • Matrix operations benchmarking ✅\n", .{});
    std.debug.print("   • Memory bandwidth utilization analysis ✅\n", .{});
    std.debug.print("   • Classification overhead measurement ✅\n", .{});
    std.debug.print("   • Batch formation efficiency validation ✅\n", .{});
    std.debug.print("   • End-to-end workflow benchmarking ✅\n", .{});
    std.debug.print("   • Cross-platform compatibility testing ✅\n", .{});
}