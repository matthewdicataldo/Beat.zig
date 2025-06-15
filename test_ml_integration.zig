const std = @import("std");
const testing = std.testing;
const fingerprint = @import("src/fingerprint.zig");
const ml_classifier = @import("src/ml_classifier.zig");
const performance_profiler = @import("src/performance_profiler.zig");
const bayesian_classifier = @import("src/bayesian_classifier.zig");
const ml_gpu_integration = @import("src/ml_gpu_integration.zig");
const gpu_integration = @import("src/gpu_integration.zig");

// Comprehensive Test Suite for ML-Based GPU Classification (Task 3.2.2)
//
// This test suite validates the entire ML-based classification pipeline:
// - Feature extraction from task fingerprints
// - Performance profiling and statistical analysis
// - Bayesian uncertainty quantification
// - Adaptive learning with feedback loops
// - Integration with existing GPU classifier
// - End-to-end classification scenarios

// ============================================================================
// Test Data Generation
// ============================================================================

/// Create a high-parallelism task fingerprint suitable for GPU execution
fn createGPUFavorableFingerprint() fingerprint.TaskFingerprint {
    return fingerprint.TaskFingerprint{
        .call_site_hash = 0x87654321,
        .data_size_class = 26,        // Large data (64MB)
        .data_alignment = 8,
        .access_pattern = .sequential,
        .simd_width = 16,
        .cache_locality = 8,          // Lower cache locality (favors GPU)
        .numa_node_hint = 0,
        .cpu_intensity = 14,          // High compute intensity
        .parallel_potential = 15,     // Maximum parallelism
        .execution_phase = 1,
        .priority_class = 0,
        .time_sensitivity = 0,        // Not latency sensitive
        .dependency_count = 1,
        .time_of_day_bucket = 10,
        .execution_frequency = 5,
        .seasonal_pattern = 2,
        .variance_level = 3,
        .expected_cycles_log2 = 28,   // Very high compute
        .memory_footprint_log2 = 26,
        .io_intensity = 1,
        .cache_miss_rate = 8,         // Higher cache misses
        .branch_predictability = 14,  // Predictable branches
        .vectorization_benefit = 15,  // Maximum vectorization benefit
    };
}

/// Create a low-parallelism task fingerprint suitable for CPU execution
fn createCPUFavorableFingerprint() fingerprint.TaskFingerprint {
    return fingerprint.TaskFingerprint{
        .call_site_hash = 0x12345678,
        .data_size_class = 12,        // Small data (4KB)
        .data_alignment = 8,
        .access_pattern = .random,
        .simd_width = 4,
        .cache_locality = 14,         // High cache locality
        .numa_node_hint = 0,
        .cpu_intensity = 6,           // Low compute intensity
        .parallel_potential = 3,      // Low parallelism
        .execution_phase = 1,
        .priority_class = 0,
        .time_sensitivity = 3,        // Latency sensitive
        .dependency_count = 5,
        .time_of_day_bucket = 14,
        .execution_frequency = 2,
        .seasonal_pattern = 1,
        .variance_level = 8,
        .expected_cycles_log2 = 16,   // Low compute
        .memory_footprint_log2 = 12,
        .io_intensity = 3,
        .cache_miss_rate = 1,         // Low cache misses
        .branch_predictability = 6,   // Unpredictable branches
        .vectorization_benefit = 2,   // Low vectorization benefit
    };
}

/// Create mixed workload fingerprint
fn createMixedFingerprint() fingerprint.TaskFingerprint {
    return fingerprint.TaskFingerprint{
        .call_site_hash = 0xABCDEF00,
        .data_size_class = 20,        // Medium data (1MB)
        .data_alignment = 8,
        .access_pattern = .strided,
        .simd_width = 8,
        .cache_locality = 8,          // Medium cache locality
        .numa_node_hint = 1,
        .cpu_intensity = 10,          // Medium compute intensity
        .parallel_potential = 9,      // Medium parallelism
        .execution_phase = 1,
        .priority_class = 1,
        .time_sensitivity = 1,        // Some latency sensitivity
        .dependency_count = 3,
        .time_of_day_bucket = 8,
        .execution_frequency = 4,
        .seasonal_pattern = 1,
        .variance_level = 6,
        .expected_cycles_log2 = 22,   // Medium compute
        .memory_footprint_log2 = 20,
        .io_intensity = 2,
        .cache_miss_rate = 4,         // Medium cache misses
        .branch_predictability = 9,   // Medium predictability
        .vectorization_benefit = 8,   // Medium vectorization benefit
    };
}

// ============================================================================
// Feature Extraction Tests
// ============================================================================

test "ML feature extraction comprehensive validation" {
    const allocator = testing.allocator;
    
    var extractor = ml_classifier.MLFeatureExtractor.init(allocator);
    defer extractor.deinit();
    
    const gpu_task = createGPUFavorableFingerprint();
    const cpu_task = createCPUFavorableFingerprint();
    const mixed_task = createMixedFingerprint();
    
    const system_state = ml_classifier.SystemState.getCurrentState();
    
    // Extract features for different task types
    const gpu_features = try extractor.extractFeatures(gpu_task, null, system_state);
    const cpu_features = try extractor.extractFeatures(cpu_task, null, system_state);
    const mixed_features = try extractor.extractFeatures(mixed_task, null, system_state);
    
    // Validate feature ranges
    const gpu_array = gpu_features.toArray();
    const cpu_array = cpu_features.toArray();
    const mixed_array = mixed_features.toArray();
    
    for (gpu_array) |feature| {
        try testing.expect(feature >= -1.1 and feature <= 1.1); // Allow small normalization errors
    }
    
    for (cpu_array) |feature| {
        try testing.expect(feature >= -1.1 and feature <= 1.1);
    }
    
    for (mixed_array) |feature| {
        try testing.expect(feature >= -1.1 and feature <= 1.1);
    }
    
    // GPU-favorable tasks should have higher parallelization potential
    try testing.expect(gpu_features.parallelization_potential > cpu_features.parallelization_potential);
    try testing.expect(gpu_features.vectorization_suitability > cpu_features.vectorization_suitability);
    try testing.expect(gpu_features.computational_intensity > cpu_features.computational_intensity);
    
    // CPU-favorable tasks should have higher cache locality and latency sensitivity
    try testing.expect(cpu_features.memory_access_locality > gpu_features.memory_access_locality);
    try testing.expect(cpu_features.latency_sensitivity > gpu_features.latency_sensitivity);
    
    std.debug.print("✓ Feature extraction validation passed\n", .{});
}

test "performance profiler comprehensive measurement" {
    const allocator = testing.allocator;
    
    var profiler = performance_profiler.PerformanceProfiler.init(allocator);
    defer profiler.deinit();
    
    // Simulate multiple CPU measurements
    var i: u32 = 0;
    while (i < 10) : (i += 1) {
        const context = profiler.startMeasurement();
        std.time.sleep(100); // 100ns delay
        _ = try profiler.recordMeasurement(context, .cpu);
    }
    
    // Simulate multiple GPU measurements
    i = 0;
    while (i < 10) : (i += 1) {
        const context = profiler.startMeasurement();
        std.time.sleep(50); // 50ns delay (simulating GPU being faster)
        _ = try profiler.recordMeasurement(context, .gpu);
    }
    
    // Verify measurements recorded
    try testing.expect(profiler.cpu_measurements.items.len == 10);
    try testing.expect(profiler.gpu_measurements.items.len == 10);
    
    // Verify statistics updated
    try testing.expect(profiler.cpu_stats.sample_count == 10);
    try testing.expect(profiler.gpu_stats.sample_count == 10);
    try testing.expect(profiler.cpu_stats.mean_execution_time > 0);
    try testing.expect(profiler.gpu_stats.mean_execution_time > 0);
    
    // Test performance comparison
    const test_fingerprint = createGPUFavorableFingerprint();
    const comparison = profiler.compareCPUvsGPU(test_fingerprint);
    
    try testing.expect(comparison.confidence_level > 0.0);
    try testing.expect(comparison.performance_ratio > 0.0);
    try testing.expect(comparison.energy_ratio > 0.0);
    
    // Test trend analysis
    const trend = profiler.getPerformanceTrend(5);
    try testing.expect(trend.trend_confidence >= 0.0 and trend.trend_confidence <= 1.0);
    
    std.debug.print("✓ Performance profiler validation passed\n", .{});
}

test "Bayesian classifier uncertainty quantification" {
    const allocator = testing.allocator;
    
    var adaptive_system = bayesian_classifier.AdaptiveLearningSystem.init(allocator, 0.5);
    defer adaptive_system.deinit();
    
    const gpu_task = createGPUFavorableFingerprint();
    const cpu_task = createCPUFavorableFingerprint();
    const system_state = ml_classifier.SystemState.getCurrentState();
    
    // Test initial classification (should be uncertain)
    const initial_gpu_result = try adaptive_system.classifyTask(gpu_task, null, system_state);
    const initial_cpu_result = try adaptive_system.classifyTask(cpu_task, null, system_state);
    
    try testing.expect(initial_gpu_result.confidence >= 0.0 and initial_gpu_result.confidence <= 1.0);
    try testing.expect(initial_cpu_result.confidence >= 0.0 and initial_cpu_result.confidence <= 1.0);
    
    // Provide feedback to improve confidence
    const mock_performance = performance_profiler.PerformanceMeasurement{
        .execution_time_ns = 500_000, // 0.5ms
        .cpu_cycles = 1_500_000,
        .memory_accesses = 2000,
        .cache_misses = 100,
        .instructions_executed = 20_000,
        .power_consumption_mw = 80_000,
        .temperature_celsius = 70,
        .memory_bandwidth_used = 0.8,
        .cpu_utilization = 0.2,
        .gpu_utilization = 0.9,
    };
    
    // Train with GPU-favorable examples
    var i: u32 = 0;
    while (i < 20) : (i += 1) {
        const result = try adaptive_system.classifyTask(gpu_task, null, system_state);
        try adaptive_system.provideFeedback(result, .gpu, mock_performance);
    }
    
    // Train with CPU-favorable examples
    i = 0;
    while (i < 20) : (i += 1) {
        const result = try adaptive_system.classifyTask(cpu_task, null, system_state);
        try adaptive_system.provideFeedback(result, .cpu, mock_performance);
    }
    
    // Test improved confidence after training
    const trained_gpu_result = try adaptive_system.classifyTask(gpu_task, null, system_state);
    const trained_cpu_result = try adaptive_system.classifyTask(cpu_task, null, system_state);
    
    // Confidence should improve with training data
    try testing.expect(trained_gpu_result.confidence >= initial_gpu_result.confidence);
    try testing.expect(trained_cpu_result.confidence >= initial_cpu_result.confidence);
    
    // GPU task should favor GPU, CPU task should favor CPU
    try testing.expect(trained_gpu_result.use_gpu);
    try testing.expect(!trained_cpu_result.use_gpu);
    
    // Test system statistics
    const stats = adaptive_system.getSystemStatistics();
    try testing.expect(stats.total_classifications > 0);
    try testing.expect(stats.feedback_count > 0);
    
    std.debug.print("✓ Bayesian classifier validation passed\n", .{});
}

test "Enhanced GPU classifier integration end-to-end" {
    const allocator = testing.allocator;
    
    var enhanced_classifier = ml_gpu_integration.EnhancedGPUClassifier.init(allocator, 0.5);
    defer enhanced_classifier.deinit();
    
    const gpu_task = createGPUFavorableFingerprint();
    const cpu_task = createCPUFavorableFingerprint();
    const mixed_task = createMixedFingerprint();
    
    // Test initial classifications
    const gpu_result = try enhanced_classifier.classifyTask(gpu_task, null);
    const cpu_result = try enhanced_classifier.classifyTask(cpu_task, null);
    const mixed_result = try enhanced_classifier.classifyTask(mixed_task, null);
    
    // Validate enhanced classification results
    try testing.expect(gpu_result.confidence >= 0.0 and gpu_result.confidence <= 1.0);
    try testing.expect(cpu_result.confidence >= 0.0 and cpu_result.confidence <= 1.0);
    try testing.expect(mixed_result.confidence >= 0.0 and mixed_result.confidence <= 1.0);
    
    try testing.expect(gpu_result.classification_time_ns > 0);
    try testing.expect(cpu_result.classification_time_ns > 0);
    try testing.expect(mixed_result.classification_time_ns > 0);
    
    // Test feedback mechanism with realistic performance data
    const gpu_performance = performance_profiler.PerformanceMeasurement{
        .execution_time_ns = 200_000, // GPU faster for parallel task
        .cpu_cycles = 600_000,
        .memory_accesses = 5000,
        .cache_misses = 500,
        .instructions_executed = 50_000,
        .power_consumption_mw = 120_000,
        .temperature_celsius = 75,
        .memory_bandwidth_used = 0.9,
        .cpu_utilization = 0.1,
        .gpu_utilization = 0.95,
    };
    
    const cpu_performance = performance_profiler.PerformanceMeasurement{
        .execution_time_ns = 100_000, // CPU faster for sequential task
        .cpu_cycles = 300_000,
        .memory_accesses = 1000,
        .cache_misses = 50,
        .instructions_executed = 10_000,
        .power_consumption_mw = 40_000,
        .temperature_celsius = 60,
        .memory_bandwidth_used = 0.3,
        .cpu_utilization = 0.8,
        .gpu_utilization = 0.1,
    };
    
    // Provide feedback for learning
    try enhanced_classifier.provideFeedback(gpu_result, .gpu, gpu_performance);
    try enhanced_classifier.provideFeedback(cpu_result, .cpu, cpu_performance);
    try enhanced_classifier.provideFeedback(mixed_result, .cpu, cpu_performance); // Mixed task performs better on CPU
    
    // Test classification weight adaptation
    _ = enhanced_classifier.ml_weight; // Store initial weights (for potential future use)
    _ = enhanced_classifier.rule_weight;
    
    enhanced_classifier.setClassificationWeights(0.8, 0.2);
    try testing.expect(std.math.fabs(enhanced_classifier.ml_weight - 0.8) < 0.01);
    try testing.expect(std.math.fabs(enhanced_classifier.rule_weight - 0.2) < 0.01);
    
    // Test adaptive mode
    enhanced_classifier.setAdaptiveMode(false);
    try testing.expect(!enhanced_classifier.adaptation_enabled);
    
    enhanced_classifier.setAdaptiveMode(true);
    try testing.expect(enhanced_classifier.adaptation_enabled);
    
    // Test system statistics
    const stats = enhanced_classifier.getSystemStatistics();
    try testing.expect(stats.total_classifications >= 3); // We made 3 classifications
    try testing.expect(stats.ml_weight + stats.rule_weight == 1.0);
    try testing.expect(stats.confidence_threshold > 0.0);
    
    std.debug.print("✓ Enhanced GPU classifier integration passed\n", .{});
}

test "ML classification pipeline stress test" {
    const allocator = testing.allocator;
    
    var enhanced_classifier = ml_gpu_integration.EnhancedGPUClassifier.init(allocator, 0.6);
    defer enhanced_classifier.deinit();
    
    // Create variety of task types
    const task_templates = [_]fingerprint.TaskFingerprint{
        createGPUFavorableFingerprint(),
        createCPUFavorableFingerprint(),
        createMixedFingerprint(),
    };
    
    const performance_templates = [_]performance_profiler.PerformanceMeasurement{
        // GPU-favorable performance
        performance_profiler.PerformanceMeasurement{
            .execution_time_ns = 150_000,
            .cpu_cycles = 450_000,
            .memory_accesses = 3000,
            .cache_misses = 300,
            .instructions_executed = 30_000,
            .power_consumption_mw = 100_000,
            .temperature_celsius = 70,
            .memory_bandwidth_used = 0.85,
            .cpu_utilization = 0.15,
            .gpu_utilization = 0.9,
        },
        // CPU-favorable performance
        performance_profiler.PerformanceMeasurement{
            .execution_time_ns = 80_000,
            .cpu_cycles = 240_000,
            .memory_accesses = 800,
            .cache_misses = 40,
            .instructions_executed = 8_000,
            .power_consumption_mw = 35_000,
            .temperature_celsius = 55,
            .memory_bandwidth_used = 0.25,
            .cpu_utilization = 0.75,
            .gpu_utilization = 0.05,
        },
        // Mixed performance
        performance_profiler.PerformanceMeasurement{
            .execution_time_ns = 120_000,
            .cpu_cycles = 360_000,
            .memory_accesses = 1500,
            .cache_misses = 150,
            .instructions_executed = 15_000,
            .power_consumption_mw = 60_000,
            .temperature_celsius = 65,
            .memory_bandwidth_used = 0.5,
            .cpu_utilization = 0.5,
            .gpu_utilization = 0.4,
        },
    };
    
    // Simulate 100 classification cycles with feedback
    var total_classifications: u32 = 0;
    var gpu_recommendations: u32 = 0;
    var cpu_recommendations: u32 = 0;
    
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        const task_idx = i % task_templates.len;
        const task = task_templates[task_idx];
        const performance = performance_templates[task_idx];
        
        // Classify task
        const result = try enhanced_classifier.classifyTask(task, null);
        total_classifications += 1;
        
        if (result.final_decision) {
            gpu_recommendations += 1;
        } else {
            cpu_recommendations += 1;
        }
        
        // Provide feedback (assume correct device was used based on template)
        const actual_device: ml_classifier.DeviceType = switch (task_idx) {
            0 => .gpu,  // GPU-favorable task
            1 => .cpu,  // CPU-favorable task
            2 => .cpu,  // Mixed task (assume CPU performed better)
            else => .cpu,
        };
        
        try enhanced_classifier.provideFeedback(result, actual_device, performance);
        
        // Validate result structure
        try testing.expect(result.confidence >= 0.0 and result.confidence <= 1.0);
        try testing.expect(result.classification_time_ns > 0);
        try testing.expect(result.timestamp > 0);
        try testing.expect(result.ml_result.confidence >= 0.0);
        try testing.expect(result.rule_result.confidence_score >= 0.0);
    }
    
    // Validate stress test results
    try testing.expect(total_classifications == 100);
    try testing.expect(gpu_recommendations > 0);
    try testing.expect(cpu_recommendations > 0);
    try testing.expect(gpu_recommendations + cpu_recommendations == total_classifications);
    
    // Test final system state
    const final_stats = enhanced_classifier.getSystemStatistics();
    try testing.expect(final_stats.total_classifications == 100);
    try testing.expect(final_stats.adaptive_stats.total_classifications > 0);
    try testing.expect(final_stats.trend_stats.sample_count > 0);
    
    // Ensure weights are still normalized
    try testing.expect(std.math.fabs(final_stats.ml_weight + final_stats.rule_weight - 1.0) < 0.01);
    
    std.debug.print("✓ ML classification pipeline stress test passed\n", .{});
    std.debug.print("  Total classifications: {}\n", .{total_classifications});
    std.debug.print("  GPU recommendations: {}\n", .{gpu_recommendations});
    std.debug.print("  CPU recommendations: {}\n", .{cpu_recommendations});
    std.debug.print("  Final ML weight: {d:.3}\n", .{final_stats.ml_weight});
    std.debug.print("  Final rule weight: {d:.3}\n", .{final_stats.rule_weight});
}

test "ML feature normalization and edge cases" {
    // Test feature normalization edge cases
    var features = ml_classifier.MLTaskFeatures{
        .data_size_log2 = 100.0,        // Extreme value
        .computational_intensity = -5.0, // Negative value
        .memory_footprint_log2 = 50.0,  // Large value
        .parallelization_potential = 1.5, // Above range
        .vectorization_suitability = -0.5, // Below range
        .memory_access_locality = 0.5,
        .system_load_cpu = 2.0,         // Above range
        .system_load_gpu = -1.0,        // Below range
        .available_cpu_cores = 0.8,
        .available_gpu_memory = 0.9,
        .numa_locality_hint = 0.5,
        .power_budget_constraint = 1.2, // Above range
        .latency_sensitivity = 0.3,
        .throughput_priority = 0.7,
        .time_of_day_normalized = 0.6,
        .workload_frequency = 0.4,
        .recent_performance_trend = 2.0, // Above range
        .execution_time_variance = 0.2,
        .resource_contention_history = 0.1,
        .thermal_state = 0.8,
        .gpu_compute_capability = 0.9,
        .memory_bandwidth_ratio = 0.6,
        .device_specialization_match = 0.7,
        .energy_efficiency_ratio = 0.5,
    };
    
    features.normalize();
    
    // Verify all features are in valid ranges after normalization
    const feature_array = features.toArray();
    for (feature_array) |feature| {
        try testing.expect(feature >= -1.0 and feature <= 1.0);
    }
    
    // Test array conversion roundtrip
    const reconstructed = ml_classifier.MLTaskFeatures.fromArray(feature_array);
    const reconstructed_array = reconstructed.toArray();
    
    for (feature_array, reconstructed_array) |original, reconstructed_val| {
        try testing.expect(std.math.fabs(original - reconstructed_val) < 0.001);
    }
    
    std.debug.print("✓ Feature normalization and edge cases passed\n", .{});
}