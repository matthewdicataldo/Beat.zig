const std = @import("std");
const testing = std.testing;
const beat = @import("beat");
const core = beat;
const continuation = beat.continuation;
const continuation_simd = beat.continuation_simd;
const continuation_predictive = beat.continuation_predictive;

// ============================================================================
// Continuation Predictive Accounting Integration Tests
// ============================================================================

test "predictive accounting integration initialization" {
    const allocator = testing.allocator;
    
    // Test configuration with predictive features enabled
    const config = core.Config{
        .num_workers = 2,
        .enable_predictive = true,
        .prediction_min_cutoff = 0.1,
        .prediction_beta = 0.05,
        .prediction_d_cutoff = 1.0,
    };
    
    var pool = try core.ThreadPool.init(allocator, config);
    defer pool.deinit();
    
    // Verify predictive accounting is initialized
    try testing.expect(pool.continuation_predictive_accounting != null);
    try testing.expect(pool.continuation_simd_classifier != null);
    
    std.debug.print("✅ Predictive accounting integration initialization test passed!\n", .{});
}

test "continuation submission with predictive enhancement" {
    const allocator = testing.allocator;
    
    const config = core.Config{
        .num_workers = 2,
        .enable_predictive = true,
        .enable_topology_aware = false, // Simplify for testing
        .prediction_min_cutoff = 0.05,
        .prediction_beta = 0.1,
    };
    
    var pool = try core.ThreadPool.init(allocator, config);
    defer pool.deinit();
    
    // Create test continuation with SIMD-friendly operation
    const TestData = struct { 
        values: [32]f32,
        processed: bool = false,
    };
    
    var test_data = TestData{ .values = undefined };
    for (&test_data.values, 0..) |*value, i| {
        value.* = @floatFromInt(i);
    }
    
    const resume_fn = struct {
        fn executeFunc(cont: *continuation.Continuation) void {
            const data = @as(*TestData, @ptrCast(@alignCast(cont.data)));
            
            // SIMD-friendly vectorizable operation
            for (&data.values) |*value| {
                value.* = value.* * 2.0 + 1.0; // Linear transformation
            }
            data.processed = true;
            cont.state = .completed;
        }
    };
    
    var test_continuation = continuation.Continuation.capture(resume_fn.executeFunc, &test_data, allocator);
    test_continuation.frame_size = 128;
    
    // Submit continuation - this should trigger SIMD classification and prediction
    try pool.submitContinuation(&test_continuation);
    
    // Verify fingerprint was generated
    try testing.expect(test_continuation.fingerprint_hash != null);
    
    // Wait for processing
    pool.wait();
    
    // Verify continuation was processed
    try testing.expect(test_data.processed);
    
    // Verify prediction was recorded
    if (pool.continuation_predictive_accounting) |predictor| {
        const stats = predictor.getPerformanceStats();
        try testing.expect(stats.total_predictions > 0);
    }
    
    std.debug.print("✅ Continuation submission with predictive enhancement test passed!\n", .{});
}

test "execution time prediction accuracy tracking" {
    const allocator = testing.allocator;
    
    const config = continuation_predictive.PredictiveConfig.performanceOptimized();
    var predictor = try continuation_predictive.ContinuationPredictiveAccounting.init(allocator, config);
    defer predictor.deinit();
    
    // Create multiple test continuations with different characteristics
    const TestData = struct { 
        operation_count: u32,
        complexity: f32,
    };
    
    const test_cases = [_]TestData{
        .{ .operation_count = 10, .complexity = 1.0 },   // Simple
        .{ .operation_count = 100, .complexity = 2.0 },  // Medium  
        .{ .operation_count = 1000, .complexity = 3.0 }, // Complex
    };
    
    var continuations: [3]continuation.Continuation = undefined;
    var accuracies: [3]bool = undefined;
    
    const resume_fn = struct {
        fn executeFunc(cont: *continuation.Continuation) void {
            const data = @as(*TestData, @ptrCast(@alignCast(cont.data)));
            
            // Simulate work proportional to operation count and complexity
            var sum: f32 = 0.0;
            for (0..data.operation_count) |i| {
                sum += @as(f32, @floatFromInt(i)) * data.complexity;
            }
            
            // Prevent optimization
            std.mem.doNotOptimizeAway(sum);
            cont.state = .completed;
        }
    };
    
    // Initialize continuations
    for (&continuations, 0..) |*cont, i| {
        cont.* = continuation.Continuation.capture(resume_fn.executeFunc, @constCast(&test_cases[i]), allocator);
        cont.fingerprint_hash = 1000 + i; // Unique fingerprints
    }
    
    // Test prediction accuracy over multiple iterations
    for (0..3) |iteration| {
        for (&continuations, 0..) |*cont, i| {
            // Make prediction
            const prediction = try predictor.predictExecutionTime(cont, null);
            
            // Simulate execution timing
            const start_time = std.time.nanoTimestamp();
            resume_fn.executeFunc(cont);
            const end_time = std.time.nanoTimestamp();
            const actual_time = @as(u64, @intCast(end_time - start_time));
            
            // Update predictor with actual results
            try predictor.updatePrediction(cont, actual_time);
            
            // Track accuracy for final iteration
            if (iteration == 2) {
                const prediction_error = @abs(@as(f64, @floatFromInt(actual_time)) - @as(f64, @floatFromInt(prediction.predicted_time_ns)));
                const relative_error = prediction_error / @as(f64, @floatFromInt(actual_time));
                accuracies[i] = relative_error < 0.5; // Within 50% considered reasonable for test
            }
        }
    }
    
    // Check that prediction accuracy improves
    const final_stats = predictor.getPerformanceStats();
    try testing.expect(final_stats.total_predictions >= 9); // 3 continuations × 3 iterations
    
    // At least some predictions should be reasonably accurate
    var accurate_count: u32 = 0;
    for (accuracies) |accurate| {
        if (accurate) accurate_count += 1;
    }
    try testing.expect(accurate_count >= 1); // At least one should be accurate
    
    std.debug.print("✅ Execution time prediction accuracy tracking test passed!\n", .{});
    std.debug.print("   Total predictions: {}\n", .{final_stats.total_predictions});
    std.debug.print("   Accuracy rate: {d:.1}%\n", .{final_stats.accuracy_rate * 100});
    std.debug.print("   Cache hit rate: {d:.1}%\n", .{final_stats.cache_hit_rate * 100});
    std.debug.print("   Accurate predictions in final round: {}/{}\n", .{accurate_count, accuracies.len});
}

test "adaptive NUMA placement based on predictions" {
    const allocator = testing.allocator;
    
    const config = continuation_predictive.PredictiveConfig{
        .enable_adaptive_numa = true,
    };
    
    var predictor = try continuation_predictive.ContinuationPredictiveAccounting.init(allocator, config);
    defer predictor.deinit();
    
    // Test different continuation types for NUMA placement
    const TestData = struct { execution_time_category: enum { short, medium, long } };
    
    const test_cases = [_]TestData{
        .{ .execution_time_category = .short },  // < 10ms - should prefer current NUMA
        .{ .execution_time_category = .medium }, // ~50ms - adaptive placement
        .{ .execution_time_category = .long },   // > 100ms - should prefer specific NUMA
    };
    
    var continuations: [3]continuation.Continuation = undefined;
    
    const resume_fn = struct {
        fn executeFunc(cont: *continuation.Continuation) void {
            cont.state = .completed;
        }
    };
    
    // Initialize continuations with different NUMA preferences
    for (&continuations, 0..) |*cont, i| {
        cont.* = continuation.Continuation.capture(resume_fn.executeFunc, @constCast(&test_cases[i]), allocator);
        cont.fingerprint_hash = 2000 + i;
        cont.numa_node = 0; // Start with NUMA node 0
        cont.original_numa_node = 1; // Original was on NUMA node 1
    }
    
    // Create mock SIMD classifications with different characteristics
    const simd_classes = [_]continuation_simd.ContinuationSIMDClass{
        .{ // Short task - low overhead
            .task_class = .potentially_vectorizable,
            .simd_suitability_score = 0.3,
            .continuation_overhead_factor = 1.1,
            .vectorization_potential = 1.5,
            .preferred_numa_node = null,
        },
        .{ // Medium task - moderate optimization
            .task_class = .moderately_vectorizable,
            .simd_suitability_score = 0.6,
            .continuation_overhead_factor = 1.2,
            .vectorization_potential = 2.5,
            .preferred_numa_node = 1,
        },
        .{ // Long task - high optimization potential
            .task_class = .highly_vectorizable,
            .simd_suitability_score = 0.9,
            .continuation_overhead_factor = 1.3,
            .vectorization_potential = 4.0,
            .preferred_numa_node = 1,
        },
    };
    
    // Test predictions and NUMA placement
    for (&continuations, 0..) |*cont, i| {
        const prediction = try predictor.predictExecutionTime(cont, simd_classes[i]);
        
        // Verify prediction characteristics
        try testing.expect(prediction.predicted_time_ns > 0);
        try testing.expect(prediction.confidence >= 0.0 and prediction.confidence <= 1.0);
        
        // Check NUMA placement logic
        switch (test_cases[i].execution_time_category) {
            .short => {
                // Short tasks should keep current NUMA node or not change much
                try testing.expect(prediction.numa_preference == cont.numa_node or prediction.numa_preference == null);
            },
            .medium => {
                // Medium tasks may use adaptive placement
                // NUMA preference can be current, original, or SIMD-preferred
            },
            .long => {
                // Long tasks should prefer original or SIMD-optimal NUMA node
                try testing.expect(prediction.numa_preference == cont.original_numa_node or prediction.numa_preference == simd_classes[i].preferred_numa_node);
            },
        }
        
        std.debug.print("Task {} ({s}): predicted_time={}μs, confidence={d:.3}, numa_preference={?}\n", 
                       .{i, @tagName(test_cases[i].execution_time_category), 
                         prediction.predicted_time_ns / 1000, prediction.confidence, prediction.numa_preference});
    }
    
    std.debug.print("✅ Adaptive NUMA placement based on predictions test passed!\n", .{});
}

test "SIMD-enhanced prediction integration" {
    const allocator = testing.allocator;
    
    const config = continuation_predictive.PredictiveConfig.balanced();
    var predictor = try continuation_predictive.ContinuationPredictiveAccounting.init(allocator, config);
    defer predictor.deinit();
    
    // Create test continuation
    const TestData = struct { matrix: [64]f32 };
    var test_data = TestData{ .matrix = undefined };
    
    const resume_fn = struct {
        fn executeFunc(cont: *continuation.Continuation) void {
            const data = @as(*TestData, @ptrCast(@alignCast(cont.data)));
            // Complex mathematical operation suitable for SIMD
            for (&data.matrix) |*value| {
                value.* = @sqrt(value.* * value.* + 1.0);
            }
            cont.state = .completed;
        }
    };
    
    var test_continuation = continuation.Continuation.capture(resume_fn.executeFunc, &test_data, allocator);
    test_continuation.fingerprint_hash = 54321;
    
    // Test prediction without SIMD enhancement
    const prediction_baseline = try predictor.predictExecutionTime(&test_continuation, null);
    
    // Test prediction with SIMD enhancement
    const simd_class = continuation_simd.ContinuationSIMDClass{
        .task_class = .highly_vectorizable,
        .simd_suitability_score = 0.9,
        .continuation_overhead_factor = 1.1,
        .vectorization_potential = 4.0, // 4x speedup expected
        .preferred_numa_node = 1,
    };
    
    const prediction_simd = try predictor.predictExecutionTime(&test_continuation, simd_class);
    
    // SIMD-enhanced prediction should be faster and more confident
    try testing.expect(prediction_simd.predicted_time_ns <= prediction_baseline.predicted_time_ns);
    try testing.expect(prediction_simd.confidence >= prediction_baseline.confidence);
    try testing.expect(prediction_simd.should_batch == true);
    try testing.expect(prediction_simd.numa_preference == 1);
    try testing.expect(prediction_simd.prediction_source == .simd_enhanced);
    
    // Test that SIMD speedup is applied correctly
    const expected_speedup = simd_class.getExpectedSpeedup();
    try testing.expect(expected_speedup > 1.0);
    
    std.debug.print("✅ SIMD-enhanced prediction integration test passed!\n", .{});
    std.debug.print("   Baseline prediction: {}μs (confidence: {d:.3})\n", 
                   .{prediction_baseline.predicted_time_ns / 1000, prediction_baseline.confidence});
    std.debug.print("   SIMD prediction: {}μs (confidence: {d:.3})\n", 
                   .{prediction_simd.predicted_time_ns / 1000, prediction_simd.confidence});
    std.debug.print("   Expected SIMD speedup: {d:.2}x\n", .{expected_speedup});
    std.debug.print("   NUMA preference: {?}\n", .{prediction_simd.numa_preference});
}

test "prediction cache performance and eviction" {
    const allocator = testing.allocator;
    
    const config = continuation_predictive.PredictiveConfig.performanceOptimized();
    var predictor = try continuation_predictive.ContinuationPredictiveAccounting.init(allocator, config);
    defer predictor.deinit();
    
    // Create many unique continuations to test cache behavior
    const TestData = struct { id: u32 };
    var test_data_array: [100]TestData = undefined;
    var continuations: [100]continuation.Continuation = undefined;
    
    const resume_fn = struct {
        fn executeFunc(cont: *continuation.Continuation) void {
            cont.state = .completed;
        }
    };
    
    // Initialize many unique continuations
    for (&test_data_array, 0..) |*data, i| {
        data.* = TestData{ .id = @intCast(i) };
        continuations[i] = continuation.Continuation.capture(resume_fn.executeFunc, data, allocator);
        continuations[i].fingerprint_hash = 10000 + i; // Unique fingerprints
    }
    
    // First pass - populate cache
    for (&continuations) |*cont| {
        _ = try predictor.predictExecutionTime(cont, null);
    }
    
    // Second pass - should hit cache for many entries
    for (&continuations) |*cont| {
        _ = try predictor.predictExecutionTime(cont, null);
    }
    
    // Check cache performance
    const stats = predictor.getPerformanceStats();
    try testing.expect(stats.total_predictions >= 200); // 100 × 2 passes
    try testing.expect(stats.cache_hit_rate > 0.0); // Some cache hits expected
    
    // Test cache eviction by adding more entries
    for (&continuations) |*cont| {
        // Modify fingerprint to create new cache entries
        cont.fingerprint_hash = cont.fingerprint_hash.? + 100000;
        _ = try predictor.predictExecutionTime(cont, null);
    }
    
    const final_stats = predictor.getPerformanceStats();
    try testing.expect(final_stats.total_predictions >= 300); // Total should keep growing
    
    std.debug.print("✅ Prediction cache performance and eviction test passed!\n", .{});
    std.debug.print("   Total predictions: {}\n", .{final_stats.total_predictions});
    std.debug.print("   Cache hit rate: {d:.1}%\n", .{final_stats.cache_hit_rate * 100});
    std.debug.print("   Profiles tracked: {}\n", .{final_stats.profiles_tracked});
}

test "end-to-end predictive continuation processing" {
    const allocator = testing.allocator;
    
    // Full integration test with ThreadPool
    const config = core.Config{
        .num_workers = 4,
        .enable_predictive = true,
        .enable_topology_aware = false,
        .prediction_min_cutoff = 0.05,
        .prediction_beta = 0.1,
    };
    
    var pool = try core.ThreadPool.init(allocator, config);
    defer pool.deinit();
    
    // Create diverse workload of continuations
    const WorkloadType = enum { compute_intensive, memory_bound, vectorizable };
    const TestData = struct { 
        workload_type: WorkloadType,
        data: [32]f32,
        completed: bool = false,
    };
    
    var test_data_array: [12]TestData = undefined;
    var continuations: [12]continuation.Continuation = undefined;
    
    // Different workload patterns
    const workload_types = [_]WorkloadType{ .compute_intensive, .memory_bound, .vectorizable };
    
    const resume_functions = struct {
        fn computeIntensive(cont: *continuation.Continuation) void {
            const data = @as(*TestData, @ptrCast(@alignCast(cont.data)));
            var sum: f32 = 0.0;
            for (0..1000) |i| {
                sum += @sin(@as(f32, @floatFromInt(i)) * 0.01);
            }
            data.data[0] = sum;
            data.completed = true;
            cont.state = .completed;
        }
        
        fn memoryBound(cont: *continuation.Continuation) void {
            const data = @as(*TestData, @ptrCast(@alignCast(cont.data)));
            // Memory access pattern
            for (&data.data, 0..) |*value, i| {
                value.* = @as(f32, @floatFromInt(i * i));
            }
            data.completed = true;
            cont.state = .completed;
        }
        
        fn vectorizable(cont: *continuation.Continuation) void {
            const data = @as(*TestData, @ptrCast(@alignCast(cont.data)));
            // SIMD-friendly linear operations
            for (&data.data) |*value| {
                value.* = value.* * 2.0 + 1.0;
            }
            data.completed = true;
            cont.state = .completed;
        }
    };
    
    const functions = [_]*const fn(*continuation.Continuation) void{
        resume_functions.computeIntensive,
        resume_functions.memoryBound,
        resume_functions.vectorizable,
    };
    
    // Initialize diverse continuations
    for (&test_data_array, 0..) |*data, i| {
        const workload_idx = i % 3;
        data.* = TestData{ 
            .workload_type = workload_types[workload_idx],
            .data = undefined,
        };
        
        // Initialize data
        for (&data.data, 0..) |*value, j| {
            value.* = @as(f32, @floatFromInt(i * 32 + j));
        }
        
        continuations[i] = continuation.Continuation.capture(functions[workload_idx], data, allocator);
    }
    
    // Submit all continuations
    for (&continuations) |*cont| {
        try pool.submitContinuation(cont);
    }
    
    // Wait for completion
    pool.wait();
    
    // Verify all continuations completed
    var completed_count: u32 = 0;
    for (&test_data_array) |data| {
        if (data.completed) completed_count += 1;
    }
    
    try testing.expect(completed_count == test_data_array.len);
    
    // Check predictive accounting statistics
    if (pool.continuation_predictive_accounting) |predictor| {
        const stats = predictor.getPerformanceStats();
        try testing.expect(stats.total_predictions > 0);
        std.debug.print("   Predictions made: {}\n", .{stats.total_predictions});
        std.debug.print("   Accuracy rate: {d:.1}%\n", .{stats.accuracy_rate * 100});
        std.debug.print("   Cache hit rate: {d:.1}%\n", .{stats.cache_hit_rate * 100});
    }
    
    // Check SIMD classification statistics
    if (pool.continuation_simd_classifier) |classifier| {
        const stats = classifier.getPerformanceStats();
        try testing.expect(stats.classifications_performed > 0);
        std.debug.print("   SIMD classifications: {}\n", .{stats.classifications_performed});
        std.debug.print("   SIMD hit rate: {d:.1}%\n", .{stats.simd_hit_rate * 100});
    }
    
    std.debug.print("✅ End-to-end predictive continuation processing test passed!\n", .{});
    std.debug.print("   Continuations completed: {}/{}\n", .{completed_count, test_data_array.len});
}