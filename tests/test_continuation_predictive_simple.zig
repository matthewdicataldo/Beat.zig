const std = @import("std");
const testing = std.testing;
const beat = @import("beat");
const core = beat;
const continuation = beat.continuation;
const continuation_simd = beat.continuation_simd;
const continuation_predictive = beat.continuation_predictive;

// ============================================================================
// Simple Continuation Predictive Accounting Tests
// ============================================================================

test "predictive accounting basic functionality" {
    const allocator = testing.allocator;
    
    const config = continuation_predictive.PredictiveConfig.balanced();
    var predictor = try continuation_predictive.ContinuationPredictiveAccounting.init(allocator, config);
    defer predictor.deinit();
    
    // Create simple test continuation
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
    
    // Test prediction
    const prediction = try predictor.predictExecutionTime(&test_continuation, null);
    
    // Verify basic prediction properties
    try testing.expect(prediction.predicted_time_ns > 0);
    try testing.expect(prediction.confidence >= 0.0 and prediction.confidence <= 1.0);
    
    // Test prediction update
    const actual_time: u64 = 1000000; // 1ms
    try predictor.updatePrediction(&test_continuation, actual_time);
    
    // Get statistics
    const stats = predictor.getPerformanceStats();
    try testing.expect(stats.total_predictions >= 1);
    
    std.debug.print("✅ Basic predictive accounting test passed!\n", .{});
    std.debug.print("   Prediction time: {}μs\n", .{prediction.predicted_time_ns / 1000});
    std.debug.print("   Confidence: {d:.3}\n", .{prediction.confidence});
}

test "SIMD enhanced prediction" {
    const allocator = testing.allocator;
    
    const config = continuation_predictive.PredictiveConfig.performanceOptimized();
    var predictor = try continuation_predictive.ContinuationPredictiveAccounting.init(allocator, config);
    defer predictor.deinit();
    
    // Create test continuation
    const TestData = struct { values: [16]f32 };
    var test_data = TestData{ .values = undefined };
    
    const resume_fn = struct {
        fn executeFunc(cont: *continuation.Continuation) void {
            const data = @as(*TestData, @ptrCast(@alignCast(cont.data)));
            for (&data.values) |*value| {
                value.* = value.* * 2.0 + 1.0;
            }
            cont.state = .completed;
        }
    };
    
    var test_continuation = continuation.Continuation.capture(resume_fn.executeFunc, &test_data, allocator);
    test_continuation.fingerprint_hash = 54321;
    
    // Test without SIMD
    const prediction_baseline = try predictor.predictExecutionTime(&test_continuation, null);
    
    // Test with SIMD enhancement
    const simd_class = continuation_simd.ContinuationSIMDClass{
        .task_class = .highly_vectorizable,
        .simd_suitability_score = 0.9,
        .continuation_overhead_factor = 1.1,
        .vectorization_potential = 4.0,
        .preferred_numa_node = 1,
    };
    
    const prediction_simd = try predictor.predictExecutionTime(&test_continuation, simd_class);
    
    // SIMD should improve prediction
    try testing.expect(prediction_simd.predicted_time_ns <= prediction_baseline.predicted_time_ns);
    try testing.expect(prediction_simd.confidence >= prediction_baseline.confidence);
    try testing.expect(prediction_simd.should_batch);
    try testing.expect(prediction_simd.numa_preference == 1);
    
    std.debug.print("✅ SIMD enhanced prediction test passed!\n", .{});
    std.debug.print("   Baseline: {}μs (confidence: {d:.3})\n", 
                   .{prediction_baseline.predicted_time_ns / 1000, prediction_baseline.confidence});
    std.debug.print("   SIMD: {}μs (confidence: {d:.3})\n", 
                   .{prediction_simd.predicted_time_ns / 1000, prediction_simd.confidence});
}

test "ThreadPool integration" {
    const allocator = testing.allocator;
    
    const config = core.Config{
        .num_workers = 2,
        .enable_predictive = true,
        .enable_topology_aware = false,
    };
    
    var pool = try core.ThreadPool.init(allocator, config);
    defer pool.deinit();
    
    // Verify predictive components are initialized
    try testing.expect(pool.continuation_predictive_accounting != null);
    try testing.expect(pool.continuation_simd_classifier != null);
    
    // Create simple continuation
    const TestData = struct { processed: bool = false };
    var test_data = TestData{};
    
    const resume_fn = struct {
        fn executeFunc(cont: *continuation.Continuation) void {
            const data = @as(*TestData, @ptrCast(@alignCast(cont.data)));
            data.processed = true;
            cont.state = .completed;
        }
    };
    
    var test_continuation = continuation.Continuation.capture(resume_fn.executeFunc, &test_data, allocator);
    
    // Submit continuation
    try pool.submitContinuation(&test_continuation);
    
    // Wait briefly and check
    std.time.sleep(100 * std.time.ns_per_ms); // 100ms
    pool.wait();
    
    // Verify processing
    try testing.expect(test_data.processed);
    try testing.expect(test_continuation.fingerprint_hash != null);
    
    std.debug.print("✅ ThreadPool integration test passed!\n", .{});
}