const std = @import("std");
const testing = std.testing;
const time = std.time;
const print = std.debug.print;

const ml_classifier = @import("src/ml_classifier.zig");
const performance_profiler = @import("src/performance_profiler.zig");
const bayesian_classifier = @import("src/bayesian_classifier.zig");
const fingerprint = @import("src/fingerprint.zig");

// ML Performance Benchmark and Validation Suite
//
// This test focuses specifically on the ML components without GPU dependencies
// to validate performance, accuracy, and overhead characteristics.

fn createTestFingerprint(data_size_class: u8, parallel_potential: u4, cpu_intensity: u4) fingerprint.TaskFingerprint {
    return fingerprint.TaskFingerprint{
        .call_site_hash = 0x12345678,
        .data_size_class = data_size_class,
        .data_alignment = 8,
        .access_pattern = .sequential,
        .simd_width = 8,
        .cache_locality = 10,
        .numa_node_hint = 0,
        .cpu_intensity = cpu_intensity,
        .parallel_potential = parallel_potential,
        .execution_phase = 1,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 2,
        .time_of_day_bucket = 12,
        .execution_frequency = 3,
        .seasonal_pattern = 1,
        .variance_level = 5,
        .expected_cycles_log2 = 20,
        .memory_footprint_log2 = @intCast(data_size_class),
        .io_intensity = 1,
        .cache_miss_rate = 3,
        .branch_predictability = 12,
        .vectorization_benefit = parallel_potential,
    };
}

fn benchmarkFeatureExtraction(allocator: std.mem.Allocator, iterations: u32) !u64 {
    var extractor = ml_classifier.MLFeatureExtractor.init(allocator);
    defer extractor.deinit();
    
    const test_fingerprint = createTestFingerprint(20, 14, 12);
    const system_state = ml_classifier.SystemState.getCurrentState();
    
    const start_time = time.nanoTimestamp();
    
    var i: u32 = 0;
    while (i < iterations) : (i += 1) {
        _ = try extractor.extractFeatures(test_fingerprint, null, system_state);
    }
    
    const end_time = time.nanoTimestamp();
    return @intCast(end_time - start_time);
}

fn benchmarkMLClassification(allocator: std.mem.Allocator, iterations: u32) !u64 {
    var classifier = ml_classifier.OnlineLinearClassifier.init(allocator, 0.01);
    
    // Create test features favoring GPU
    const gpu_features = ml_classifier.MLTaskFeatures{
        .data_size_log2 = 0.8,
        .computational_intensity = 0.9,
        .memory_footprint_log2 = 0.7,
        .parallelization_potential = 0.95,
        .vectorization_suitability = 0.9,
        .memory_access_locality = 0.6,
        .system_load_cpu = 0.8,
        .system_load_gpu = 0.2,
        .available_cpu_cores = 0.3,
        .available_gpu_memory = 0.9,
        .numa_locality_hint = 0.0,
        .power_budget_constraint = 1.0,
        .latency_sensitivity = 0.0,
        .throughput_priority = 1.0,
        .time_of_day_normalized = 0.5,
        .workload_frequency = 0.5,
        .recent_performance_trend = 0.7,
        .execution_time_variance = 0.2,
        .resource_contention_history = 0.1,
        .thermal_state = 0.9,
        .gpu_compute_capability = 0.8,
        .memory_bandwidth_ratio = 0.9,
        .device_specialization_match = 0.9,
        .energy_efficiency_ratio = 0.8,
    };
    
    const start_time = time.nanoTimestamp();
    
    var i: u32 = 0;
    while (i < iterations) : (i += 1) {
        _ = classifier.predict(gpu_features);
    }
    
    const end_time = time.nanoTimestamp();
    return @intCast(end_time - start_time);
}

fn benchmarkBayesianClassification(allocator: std.mem.Allocator, iterations: u32) !u64 {
    var bayesian_classifier_instance = bayesian_classifier.BayesianGPUClassifier.init(allocator, 0.5);
    
    const test_features = ml_classifier.MLTaskFeatures{
        .data_size_log2 = 0.7,
        .computational_intensity = 0.8,
        .memory_footprint_log2 = 0.6,
        .parallelization_potential = 0.9,
        .vectorization_suitability = 0.8,
        .memory_access_locality = 0.7,
        .system_load_cpu = 0.5,
        .system_load_gpu = 0.3,
        .available_cpu_cores = 0.8,
        .available_gpu_memory = 0.9,
        .numa_locality_hint = 0.5,
        .power_budget_constraint = 1.0,
        .latency_sensitivity = 0.2,
        .throughput_priority = 0.8,
        .time_of_day_normalized = 0.4,
        .workload_frequency = 0.6,
        .recent_performance_trend = 0.3,
        .execution_time_variance = 0.3,
        .resource_contention_history = 0.2,
        .thermal_state = 0.8,
        .gpu_compute_capability = 0.7,
        .memory_bandwidth_ratio = 0.8,
        .device_specialization_match = 0.8,
        .energy_efficiency_ratio = 0.7,
    };
    
    const start_time = time.nanoTimestamp();
    
    var i: u32 = 0;
    while (i < iterations) : (i += 1) {
        _ = bayesian_classifier_instance.classify(test_features);
    }
    
    const end_time = time.nanoTimestamp();
    return @intCast(end_time - start_time);
}

fn benchmarkPerformanceProfiler(allocator: std.mem.Allocator, iterations: u32) !u64 {
    var profiler = performance_profiler.PerformanceProfiler.init(allocator);
    defer profiler.deinit();
    
    const start_time = time.nanoTimestamp();
    
    var i: u32 = 0;
    while (i < iterations) : (i += 1) {
        const context = profiler.startMeasurement();
        // Simulate small work
        var j: u32 = 0;
        var sum: u32 = 0;
        while (j < 100) : (j += 1) {
            sum +%= j;
        }
        std.mem.doNotOptimizeAway(sum);
        _ = try profiler.recordMeasurement(context, .cpu);
    }
    
    const end_time = time.nanoTimestamp();
    return @intCast(end_time - start_time);
}

fn testMLAccuracy(allocator: std.mem.Allocator) !void {
    print("\n=== ML Classification Accuracy Test ===\n", .{});
    
    var classifier = ml_classifier.OnlineLinearClassifier.init(allocator, 0.01);
    
    // Create GPU-favorable and CPU-favorable features
    const gpu_features = ml_classifier.MLTaskFeatures{
        .data_size_log2 = 0.9,           // Large data
        .computational_intensity = 0.95,  // High compute
        .memory_footprint_log2 = 0.8,
        .parallelization_potential = 0.98, // High parallelism
        .vectorization_suitability = 0.95,
        .memory_access_locality = 0.6,   // Lower cache locality
        .system_load_cpu = 0.9,          // High CPU load
        .system_load_gpu = 0.1,          // Low GPU load
        .available_cpu_cores = 0.2,      // Few CPU cores available
        .available_gpu_memory = 0.95,    // Lots of GPU memory
        .numa_locality_hint = 0.0,
        .power_budget_constraint = 1.0,
        .latency_sensitivity = 0.0,      // Not latency sensitive
        .throughput_priority = 1.0,      // Throughput focused
        .time_of_day_normalized = 0.5,
        .workload_frequency = 0.5,
        .recent_performance_trend = 0.8, // GPU trending better
        .execution_time_variance = 0.1,  // Stable
        .resource_contention_history = 0.1,
        .thermal_state = 0.9,            // Cool
        .gpu_compute_capability = 0.9,
        .memory_bandwidth_ratio = 0.9,
        .device_specialization_match = 0.95,
        .energy_efficiency_ratio = 0.8,
    };
    
    const cpu_features = ml_classifier.MLTaskFeatures{
        .data_size_log2 = 0.2,           // Small data
        .computational_intensity = 0.3,  // Low compute
        .memory_footprint_log2 = 0.2,
        .parallelization_potential = 0.1, // Low parallelism
        .vectorization_suitability = 0.2,
        .memory_access_locality = 0.95,  // High cache locality
        .system_load_cpu = 0.2,          // Low CPU load
        .system_load_gpu = 0.8,          // High GPU load
        .available_cpu_cores = 0.9,      // Many CPU cores available
        .available_gpu_memory = 0.1,     // Little GPU memory
        .numa_locality_hint = 0.5,
        .power_budget_constraint = 0.7,
        .latency_sensitivity = 0.9,      // Very latency sensitive
        .throughput_priority = 0.1,      // Latency focused
        .time_of_day_normalized = 0.5,
        .workload_frequency = 0.5,
        .recent_performance_trend = -0.6, // CPU trending better
        .execution_time_variance = 0.8,  // Variable
        .resource_contention_history = 0.7,
        .thermal_state = 0.4,            // Warm
        .gpu_compute_capability = 0.3,
        .memory_bandwidth_ratio = 0.3,
        .device_specialization_match = 0.1,
        .energy_efficiency_ratio = 0.4,
    };
    
    // Train the classifier
    print("1. Training ML classifier with 200 examples...\n", .{});
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        classifier.updateModel(gpu_features, true, 2.0);  // GPU 2x faster
        classifier.updateModel(cpu_features, false, 0.5); // CPU 2x faster
    }
    
    // Test predictions
    const gpu_prediction = classifier.predict(gpu_features);
    const cpu_prediction = classifier.predict(cpu_features);
    
    print("2. Testing classification accuracy:\n", .{});
    print("   GPU-favorable task: use_gpu={}, probability={d:.3}, confidence={d:.3}\n", 
          .{gpu_prediction.use_gpu, gpu_prediction.probability, gpu_prediction.confidence});
    print("   CPU-favorable task: use_gpu={}, probability={d:.3}, confidence={d:.3}\n", 
          .{cpu_prediction.use_gpu, cpu_prediction.probability, cpu_prediction.confidence});
    
    const accuracy = classifier.getAccuracy();
    print("   Overall accuracy: {d:.1}%\n", .{accuracy * 100});
    
    try testing.expect(gpu_prediction.use_gpu);
    try testing.expect(!cpu_prediction.use_gpu);
    try testing.expect(accuracy > 0.8); // Should achieve >80% accuracy
    print("✅ ML classification accuracy test passed\n", .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    print("🚀 ML-Based Classification Performance Benchmark\n", .{});
    print("==================================================\n", .{});
    
    const iterations = 1000;
    
    // Benchmark feature extraction
    print("\n=== Feature Extraction Performance ===\n", .{});
    const feature_time = try benchmarkFeatureExtraction(allocator, iterations);
    const feature_avg = @as(f64, @floatFromInt(feature_time)) / @as(f64, @floatFromInt(iterations));
    print("Feature extraction: {d:.1} ns per operation ({} iterations)\n", .{feature_avg, iterations});
    
    // Benchmark ML classification
    print("\n=== ML Classification Performance ===\n", .{});
    const ml_time = try benchmarkMLClassification(allocator, iterations);
    const ml_avg = @as(f64, @floatFromInt(ml_time)) / @as(f64, @floatFromInt(iterations));
    print("ML classification: {d:.1} ns per operation ({} iterations)\n", .{ml_avg, iterations});
    
    // Benchmark Bayesian classification
    print("\n=== Bayesian Classification Performance ===\n", .{});
    const bayesian_time = try benchmarkBayesianClassification(allocator, iterations);
    const bayesian_avg = @as(f64, @floatFromInt(bayesian_time)) / @as(f64, @floatFromInt(iterations));
    print("Bayesian classification: {d:.1} ns per operation ({} iterations)\n", .{bayesian_avg, iterations});
    
    // Benchmark performance profiler
    print("\n=== Performance Profiler Overhead ===\n", .{});
    const profiler_iterations = 100; // Fewer iterations for profiler
    const profiler_time = try benchmarkPerformanceProfiler(allocator, profiler_iterations);
    const profiler_avg = @as(f64, @floatFromInt(profiler_time)) / @as(f64, @floatFromInt(profiler_iterations));
    print("Performance profiler: {d:.1} ns per operation ({} iterations)\n", .{profiler_avg, profiler_iterations});
    
    // Test ML accuracy
    try testMLAccuracy(allocator);
    
    // Summary
    print("\n=== Performance Summary ===\n", .{});
    const total_ml_overhead = feature_avg + ml_avg + bayesian_avg;
    print("Total ML classification overhead: {d:.1} ns\n", .{total_ml_overhead});
    print("Performance profiler overhead: {d:.1} ns\n", .{profiler_avg});
    
    if (total_ml_overhead < 10000) { // Less than 10 microseconds
        print("✅ ML classification performance: EXCELLENT (< 10μs)\n", .{});
    } else if (total_ml_overhead < 50000) { // Less than 50 microseconds
        print("✅ ML classification performance: GOOD (< 50μs)\n", .{});
    } else {
        print("⚠️  ML classification performance: NEEDS OPTIMIZATION (> 50μs)\n", .{});
    }
    
    print("\n🎯 ML Performance Benchmark Complete!\n", .{});
}