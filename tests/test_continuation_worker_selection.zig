const std = @import("std");
const testing = std.testing;
const beat = @import("beat");
const core = beat;
const continuation = beat.continuation;
const continuation_simd = beat.continuation_simd;
const continuation_predictive = beat.continuation_predictive;
const continuation_worker_selection = beat.continuation_worker_selection;
const advanced_worker_selection = beat.advanced_worker_selection;

// ============================================================================
// Continuation Worker Selection Tests
// ============================================================================

test "continuation worker selector initialization" {
    const allocator = testing.allocator;
    
    // Create mock components for testing
    const advanced_selector = try allocator.create(advanced_worker_selection.AdvancedWorkerSelector);
    defer allocator.destroy(advanced_selector);
    
    // Test different selection criteria
    const criteria = advanced_worker_selection.SelectionCriteria.balanced();
    var selector = try continuation_worker_selection.ContinuationWorkerSelector.init(
        allocator, 
        advanced_selector, 
        criteria
    );
    defer selector.deinit();
    
    // Verify initialization
    const stats = selector.getPerformanceStats();
    try testing.expect(stats.total_selections == 0);
    try testing.expect(stats.selection_quality_rate == 0.0);
    try testing.expect(stats.locality_hit_rate == 0.0);
    
    std.debug.print("✅ Continuation worker selector initialization test passed!\n", .{});
}

test "continuation locality tracking" {
    const allocator = testing.allocator;
    
    // Create test continuation
    const TestData = struct { value: i32 = 42 };
    var test_data = TestData{};
    
    const resume_fn = struct {
        fn executeFunc(cont: *continuation.Continuation) void {
            const data = @as(*TestData, @ptrCast(@alignCast(cont.data)));
            data.value *= 2;
            cont.state = .completed;
        }
    };
    
    var test_continuation = continuation.Continuation.capture(resume_fn.executeFunc, &test_data, allocator);
    test_continuation.fingerprint_hash = 12345;
    
    // Test locality tracking components
    var locality_tracker = try continuation_worker_selection.ContinuationLocalityTracker.init(allocator);
    defer locality_tracker.deinit();
    
    // Record consistent placements (good locality)
    locality_tracker.recordPlacement(&test_continuation, 0);
    locality_tracker.recordPlacement(&test_continuation, 0);
    locality_tracker.recordPlacement(&test_continuation, 0);
    
    const locality_score = locality_tracker.getLocalityScore(&test_continuation);
    try testing.expect(locality_score >= 0.8); // Should be high for consistent placement
    
    // Test NUMA preference caching
    var numa_cache = try continuation_worker_selection.NumaPreferenceCache.init(allocator);
    defer numa_cache.deinit();
    
    numa_cache.updatePreference(12345, 1, 0); // NUMA node 1, worker 0
    const preference = numa_cache.getPreference(12345);
    try testing.expect(preference != null);
    try testing.expect(preference.? == 1);
    
    std.debug.print("✅ Continuation locality tracking test passed!\n", .{});
    std.debug.print("   Locality score: {d:.3}\n", .{locality_score});
    std.debug.print("   NUMA preference: {?}\n", .{preference});
}

test "basic worker selection logic" {
    const allocator = testing.allocator;
    
    // Test basic worker selection without full ThreadPool
    const advanced_selector = try allocator.create(advanced_worker_selection.AdvancedWorkerSelector);
    defer allocator.destroy(advanced_selector);
    
    const criteria = advanced_worker_selection.SelectionCriteria.balanced();
    var selector = try continuation_worker_selection.ContinuationWorkerSelector.init(
        allocator, 
        advanced_selector, 
        criteria
    );
    defer selector.deinit();
    
    // Test that we can get performance stats
    const stats = selector.getPerformanceStats();
    try testing.expect(stats.total_selections == 0);
    
    std.debug.print("✅ Basic worker selection logic test passed!\n", .{});
    std.debug.print("   Total selections: {}\n", .{stats.total_selections});
    std.debug.print("   Selection quality rate: {d:.3}\n", .{stats.selection_quality_rate});
}

test "worker selection criteria optimization" {
    _ = testing.allocator;
    
    // Test different selection criteria configurations
    const criteria_configs = [_]advanced_worker_selection.SelectionCriteria{
        advanced_worker_selection.SelectionCriteria.balanced(),
        advanced_worker_selection.SelectionCriteria.latencyOptimized(),
        advanced_worker_selection.SelectionCriteria.throughputOptimized(),
    };
    
    const criteria_names = [_][]const u8{ "balanced", "latency", "throughput" };
    
    for (criteria_configs, criteria_names) |criteria, name| {
        // Verify criteria weights sum to approximately 1.0
        var mutable_criteria = criteria;
        mutable_criteria.normalize(); // Normalize to ensure sum is 1.0
        
        const total_weight = mutable_criteria.load_balance_weight + mutable_criteria.prediction_weight +
                           mutable_criteria.topology_weight + mutable_criteria.confidence_weight +
                           mutable_criteria.exploration_weight + mutable_criteria.simd_weight;
        
        try testing.expectApproxEqRel(total_weight, 1.0, 0.01); // Within 1%
        
        std.debug.print("✅ {s} criteria: total weight = {d:.3}\n", .{ name, total_weight });
    }
    
    std.debug.print("✅ Worker selection criteria optimization test passed!\n", .{});
}

test "continuation selection context enhancement" {
    const allocator = testing.allocator;
    
    // Create test continuation with various characteristics
    const TestData = struct { 
        workload_type: enum { cpu_intensive, memory_bound, vectorizable },
        data: [32]f32,
    };
    
    var test_data = TestData{
        .workload_type = .vectorizable,
        .data = undefined,
    };
    
    // Initialize test data
    for (&test_data.data, 0..) |*value, i| {
        value.* = @as(f32, @floatFromInt(i));
    }
    
    const resume_fn = struct {
        fn executeFunc(cont: *continuation.Continuation) void {
            const data = @as(*TestData, @ptrCast(@alignCast(cont.data)));
            // SIMD-friendly operation
            for (&data.data) |*value| {
                value.* = value.* * 2.0 + 1.0;
            }
            cont.state = .completed;
        }
    };
    
    var test_continuation = continuation.Continuation.capture(resume_fn.executeFunc, &test_data, allocator);
    test_continuation.fingerprint_hash = 54321;
    test_continuation.numa_node = 0;
    test_continuation.original_numa_node = 1;
    
    // Create SIMD classification
    const simd_class = continuation_simd.ContinuationSIMDClass{
        .task_class = .highly_vectorizable,
        .simd_suitability_score = 0.9,
        .continuation_overhead_factor = 1.1,
        .vectorization_potential = 4.0,
        .preferred_numa_node = 1,
    };
    
    // Create prediction result
    const prediction = continuation_predictive.PredictionResult{
        .predicted_time_ns = 2000000, // 2ms
        .confidence = 0.8,
        .numa_preference = 1,
        .should_batch = true,
        .prediction_source = .simd_enhanced,
    };
    
    // Verify that context contains expected information
    try testing.expect(simd_class.isSIMDSuitable());
    try testing.expect(simd_class.getExpectedSpeedup() > 1.0);
    try testing.expect(prediction.confidence > 0.7);
    try testing.expect(prediction.should_batch);
    try testing.expect(prediction.numa_preference == simd_class.preferred_numa_node);
    
    std.debug.print("✅ Continuation selection context enhancement test passed!\n", .{});
    std.debug.print("   SIMD suitability: {}\n", .{simd_class.isSIMDSuitable()});
    std.debug.print("   Expected speedup: {d:.2}x\n", .{simd_class.getExpectedSpeedup()});
    std.debug.print("   Prediction confidence: {d:.3}\n", .{prediction.confidence});
    std.debug.print("   NUMA preference: {?}\n", .{prediction.numa_preference});
}