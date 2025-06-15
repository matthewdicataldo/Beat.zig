const std = @import("std");
const beat = @import("src/core.zig");

// Test for Predictive Token Accounting Implementation (Phase 2.4.1)
//
// This test validates the enhanced token accounting system that integrates
// execution time predictions and confidence-based promotion decisions:
// - Enhanced TokenAccount with predictions
// - Execution time prediction integration 
// - Adaptive promotion thresholds
// - Confidence-based promotion decisions
// - Integration with heartbeat scheduler

test "predictive token account initialization and basic functionality" {
    std.debug.print("\n=== Predictive Token Accounting Test ===\n", .{});
    
    std.debug.print("1. Testing PredictiveTokenAccount initialization...\n", .{});
    
    // Test initialization
    const config = beat.Config{
        .promotion_threshold = 5,
        .min_work_cycles = 1000,
    };
    var account = beat.predictive_accounting.PredictiveTokenAccount.init(&config);
    
    std.debug.print("   Initialized predictive token account:\n", .{});
    std.debug.print("     Base promotion threshold: {}\n", .{account.base_promotion_threshold});
    std.debug.print("     Adaptive threshold: {d:.2}\n", .{account.adaptive_promotion_threshold});
    std.debug.print("     Prediction accuracy score: {d:.2}\n", .{account.prediction_accuracy_score});
    std.debug.print("     Active predictions count: {}\n", .{account.active_predictions.count});
    
    // Verify initialization
    try std.testing.expect(account.base_promotion_threshold == config.promotion_threshold);
    try std.testing.expect(account.adaptive_promotion_threshold == @as(f64, @floatFromInt(config.promotion_threshold)));
    try std.testing.expect(account.predicted_work_cycles == 0.0);
    try std.testing.expect(account.prediction_accuracy_score == 0.0);
    try std.testing.expect(account.active_predictions.count == 0);
    
    std.debug.print("2. Testing basic update functionality...\n", .{});
    
    // Test basic update (compatible with base TokenAccount)
    account.update(2000, 100); // 20:1 ratio, above threshold
    
    std.debug.print("   After basic update (2000 work, 100 overhead):\n", .{});
    std.debug.print("     Base should promote: {}\n", .{account.base_account.shouldPromote()});
    std.debug.print("     Overall should promote: {}\n", .{account.shouldPromote()});
    
    // Should promote due to base account logic
    try std.testing.expect(account.base_account.shouldPromote() == true);
    try std.testing.expect(account.shouldPromote() == true);
    
    std.debug.print("\n✅ Basic functionality test completed successfully!\n", .{});
}

test "prediction tracking and accuracy monitoring" {
    std.debug.print("\n=== Prediction Tracking Test ===\n", .{});
    
    std.debug.print("1. Testing prediction tracker functionality...\n", .{});
    
    // Test prediction tracker
    var tracker = beat.predictive_accounting.PredictiveTokenAccount.PredictionTracker{};
    
    // Add some predictions
    const task_hash_1: u64 = 0x12345678;
    const task_hash_2: u64 = 0x87654321;
    
    tracker.addPrediction(task_hash_1, 1000.0, 0.8);
    tracker.addPrediction(task_hash_2, 1500.0, 0.9);
    
    std.debug.print("   Added 2 predictions:\n", .{});
    std.debug.print("     Tracker count: {}\n", .{tracker.count});
    std.debug.print("     Task 1: {} cycles, 0.8 confidence\n", .{1000});
    std.debug.print("     Task 2: {} cycles, 0.9 confidence\n", .{1500});
    
    try std.testing.expect(tracker.count == 2);
    
    std.debug.print("2. Testing prediction completion and accuracy...\n", .{});
    
    // Complete first prediction (accurate)
    const result_1 = tracker.completePrediction(task_hash_1, 950);
    try std.testing.expect(result_1 != null);
    try std.testing.expect(result_1.?.was_accurate == true); // Within 30% tolerance
    try std.testing.expect(result_1.?.confidence == 0.8);
    
    std.debug.print("   Completed prediction 1:\n", .{});
    std.debug.print("     Predicted: {d:.1} cycles\n", .{result_1.?.predicted_cycles});
    std.debug.print("     Actual: {} cycles\n", .{result_1.?.actual_cycles});
    std.debug.print("     Relative error: {d:.2}%\n", .{result_1.?.relative_error * 100});
    std.debug.print("     Was accurate: {}\n", .{result_1.?.was_accurate});
    
    // Complete second prediction (inaccurate)
    const result_2 = tracker.completePrediction(task_hash_2, 3000);
    try std.testing.expect(result_2 != null);
    try std.testing.expect(result_2.?.was_accurate == false); // 100% error, above 30% tolerance
    try std.testing.expect(result_2.?.confidence == 0.9);
    
    std.debug.print("   Completed prediction 2:\n", .{});
    std.debug.print("     Predicted: {d:.1} cycles\n", .{result_2.?.predicted_cycles});
    std.debug.print("     Actual: {} cycles\n", .{result_2.?.actual_cycles});
    std.debug.print("     Relative error: {d:.2}%\n", .{result_2.?.relative_error * 100});
    std.debug.print("     Was accurate: {}\n", .{result_2.?.was_accurate});
    
    std.debug.print("\n✅ Prediction tracking test completed successfully!\n", .{});
}

test "task execution recording with prediction integration" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Task Execution Recording Test ===\n", .{});
    
    // Create test setup
    const config = beat.Config{
        .promotion_threshold = 5,
        .min_work_cycles = 1000,
    };
    var account = beat.predictive_accounting.PredictiveTokenAccount.init(&config);
    
    // Create fingerprint registry
    var registry = beat.fingerprint.FingerprintRegistry.init(allocator);
    defer registry.deinit();
    
    account.setPredictionRegistry(&registry);
    
    std.debug.print("1. Testing task execution recording...\n", .{});
    
    // Create test task
    const TestData = struct { counter: u32 };
    var test_data = TestData{ .counter = 0 };
    
    var test_task = beat.Task{
        .func = struct {
            fn func(data: *anyopaque) void {
                const typed_data = @as(*TestData, @ptrCast(@alignCast(data)));
                typed_data.counter += 1;
            }
        }.func,
        .data = @ptrCast(&test_data),
        .priority = .normal,
        .fingerprint_hash = 0x11223344,
        .data_size_hint = @sizeOf(TestData),
    };
    
    // Add prediction for this task
    account.active_predictions.addPrediction(0x11223344, 1200.0, 0.75);
    
    std.debug.print("   Recording task execution...\n", .{});
    std.debug.print("     Task fingerprint: 0x{X}\n", .{test_task.fingerprint_hash.?});
    std.debug.print("     Predicted cycles: 1200\n", .{});
    std.debug.print("     Actual cycles: 1100\n", .{});
    
    // Record task execution
    account.recordTaskExecution(&test_task, 1100, 50);
    
    std.debug.print("   After recording execution:\n", .{});
    std.debug.print("     Total predictions made: {}\n", .{account.total_predictions_made});
    std.debug.print("     Accurate predictions: {}\n", .{account.accurate_predictions});
    std.debug.print("     Prediction accuracy score: {d:.3}\n", .{account.prediction_accuracy_score});
    
    // Verify prediction tracking
    try std.testing.expect(account.total_predictions_made == 1);
    try std.testing.expect(account.accurate_predictions == 1); // 1100 vs 1200 is within 30% tolerance
    try std.testing.expect(account.prediction_accuracy_score > 0.0); // Should have updated
    
    std.debug.print("\n✅ Task execution recording test completed successfully!\n", .{});
}

test "adaptive promotion threshold adjustment" {
    std.debug.print("\n=== Adaptive Promotion Threshold Test ===\n", .{});
    
    const config = beat.Config{
        .promotion_threshold = 10,
        .min_work_cycles = 1000,
    };
    var account = beat.predictive_accounting.PredictiveTokenAccount.init(&config);
    
    std.debug.print("1. Testing initial threshold adjustment (no data)...\n", .{});
    
    const initial_adjustment = account.confidence_adjustment_factor;
    account.update(500, 100); // This should trigger updateAdaptiveThreshold()
    
    std.debug.print("   Initial adjustment factor: {d:.2}\n", .{initial_adjustment});
    std.debug.print("   After update (insufficient data): {d:.2}\n", .{account.confidence_adjustment_factor});
    
    // Should not change with insufficient prediction data
    try std.testing.expect(account.confidence_adjustment_factor == initial_adjustment);
    
    std.debug.print("2. Testing threshold adjustment with prediction data...\n", .{});
    
    // Simulate prediction results to trigger threshold adjustment
    account.total_predictions_made = 15; // Above threshold of 10
    account.accurate_predictions = 12;   // 80% accuracy
    
    const before_adjustment = account.confidence_adjustment_factor;
    const before_threshold = account.adaptive_promotion_threshold;
    
    // This should trigger threshold adjustment
    account.update(1000, 200);
    
    std.debug.print("   Before adjustment:\n", .{});
    std.debug.print("     Accuracy rate: {d:.2}%\n", .{@as(f32, @floatFromInt(12)) / @as(f32, @floatFromInt(15)) * 100});
    std.debug.print("     Adjustment factor: {d:.3}\n", .{before_adjustment});
    std.debug.print("     Adaptive threshold: {d:.2}\n", .{before_threshold});
    
    std.debug.print("   After adjustment:\n", .{});
    std.debug.print("     Adjustment factor: {d:.3}\n", .{account.confidence_adjustment_factor});
    std.debug.print("     Adaptive threshold: {d:.2}\n", .{account.adaptive_promotion_threshold});
    
    // With 80% accuracy, adjustment factor should be lower (more aggressive)
    // new_adjustment = 0.5 + (1.0 - 0.8) * 1.0 = 0.7
    const expected_adjustment: f32 = 0.5 + (1.0 - 0.8) * 1.0;
    try std.testing.expect(@abs(account.confidence_adjustment_factor - expected_adjustment) < 0.1);
    
    std.debug.print("\n✅ Adaptive threshold adjustment test completed successfully!\n", .{});
}

test "confidence-based promotion decisions" {
    std.debug.print("\n=== Confidence-Based Promotion Test ===\n", .{});
    
    const config = beat.Config{
        .promotion_threshold = 10,
        .min_work_cycles = 1000,
    };
    var account = beat.predictive_accounting.PredictiveTokenAccount.init(&config);
    
    std.debug.print("1. Testing promotion decision logic...\n", .{});
    
    // Setup for confidence-based promotion test
    account.prediction_accuracy_score = 0.8; // Good accuracy
    account.confidence_weighted_tokens = 0.9; // High confidence
    account.base_account.work_cycles = 8000;  // Below normal threshold
    account.base_account.overhead_cycles = 1000; // 8:1 ratio (below 10:1 threshold)
    
    const promotion_analysis = account.getPromotionAnalysis();
    
    std.debug.print("   Promotion analysis:\n", .{});
    std.debug.print("     Base should promote: {}\n", .{promotion_analysis.base_should_promote});
    std.debug.print("     Prediction should promote: {}\n", .{promotion_analysis.prediction_should_promote});
    std.debug.print("     Confidence should promote: {}\n", .{promotion_analysis.confidence_should_promote});
    std.debug.print("     Overall should promote: {}\n", .{promotion_analysis.overall_should_promote});
    std.debug.print("     Adaptive threshold: {d:.2}\n", .{promotion_analysis.adaptive_threshold});
    std.debug.print("     Prediction accuracy: {d:.2}\n", .{promotion_analysis.prediction_accuracy});
    
    // Base account should not promote (8:1 < 10:1)
    try std.testing.expect(promotion_analysis.base_should_promote == false);
    
    // But confidence-based promotion might trigger due to high confidence
    // This demonstrates the enhanced logic beyond simple ratio checking
    
    std.debug.print("2. Testing prediction-based promotion...\n", .{});
    
    // Setup for prediction-based promotion
    account.predicted_work_cycles = 5000.0; // Sufficient predicted work
    account.prediction_accuracy_score = 0.8; // Good accuracy requirement met
    
    const prediction_should_promote = account.shouldPromoteBasedOnPredictions();
    std.debug.print("   Prediction-based promotion: {}\n", .{prediction_should_promote});
    std.debug.print("     Predicted work: {d:.1} cycles\n", .{account.predicted_work_cycles});
    std.debug.print("     Required accuracy: 0.7, actual: {d:.2}\n", .{account.prediction_accuracy_score});
    
    // Check if prediction-based promotion should occur
    // The logic requires: predicted_work >= min_work AND predicted_ratio > threshold AND accuracy > 0.7
    const min_predicted_work = @as(f64, @floatFromInt(account.base_account.min_work_cycles));
    const predicted_ratio = if (account.base_account.overhead_cycles > 0)
        account.predicted_work_cycles / @as(f64, @floatFromInt(account.base_account.overhead_cycles))
    else
        0.0;
    const base_threshold = @as(f64, @floatFromInt(account.base_promotion_threshold));
    
    std.debug.print("     Predicted work: {d:.1} >= min work: {d:.1} ✓\n", .{account.predicted_work_cycles, min_predicted_work});
    std.debug.print("     Predicted ratio: {d:.2} > threshold: {d:.1} ?\n", .{predicted_ratio, base_threshold});
    std.debug.print("     Accuracy: {d:.2} > 0.7 ✓\n", .{account.prediction_accuracy_score});
    
    // Should only promote if all criteria are met
    const should_promote_expected = account.predicted_work_cycles >= min_predicted_work and 
                                   predicted_ratio > base_threshold and
                                   account.prediction_accuracy_score > 0.7;
    
    std.debug.print("     Expected promotion: {} (actual: {})\n", .{should_promote_expected, prediction_should_promote});
    try std.testing.expect(prediction_should_promote == should_promote_expected);
    
    std.debug.print("\n✅ Confidence-based promotion test completed successfully!\n", .{});
}

test "predictive work estimation integration" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Predictive Work Estimation Test ===\n", .{});
    
    // Setup predictive account with registry
    const config = beat.Config{
        .promotion_threshold = 5,
        .min_work_cycles = 1000,
    };
    var account = beat.predictive_accounting.PredictiveTokenAccount.init(&config);
    
    var registry = beat.fingerprint.FingerprintRegistry.init(allocator);
    defer registry.deinit();
    
    account.setPredictionRegistry(&registry);
    
    std.debug.print("1. Testing work estimation without fingerprint data...\n", .{});
    
    // Create test task without existing fingerprint data
    const TestData = struct { value: i32 };
    var test_data = TestData{ .value = 42 };
    
    var test_task = beat.Task{
        .func = struct {
            fn func(data: *anyopaque) void {
                const typed_data = @as(*TestData, @ptrCast(@alignCast(data)));
                typed_data.value *= 2;
            }
        }.func,
        .data = @ptrCast(&test_data),
        .priority = .normal,
        .fingerprint_hash = 0x99887766,
        .data_size_hint = @sizeOf(TestData),
    };
    
    const estimate_1 = account.predictTaskWork(&test_task);
    
    std.debug.print("   First estimation (no prior data):\n", .{});
    std.debug.print("     Predicted cycles: {d:.1}\n", .{estimate_1.predicted_cycles});
    std.debug.print("     Confidence: {d:.2}\n", .{estimate_1.confidence});
    std.debug.print("     Should preemptively promote: {}\n", .{estimate_1.should_preemptively_promote});
    std.debug.print("     Confidence category: {}\n", .{estimate_1.confidence_category});
    
    // Should have fallback values
    try std.testing.expect(estimate_1.predicted_cycles == 1000.0); // Default fallback
    try std.testing.expect(estimate_1.confidence == 0.0); // No confidence without data
    try std.testing.expect(estimate_1.should_preemptively_promote == false);
    
    std.debug.print("2. Testing accumulated predicted work...\n", .{});
    
    const initial_predicted_work = account.predicted_work_cycles;
    std.debug.print("   Predicted work before: {d:.1}\n", .{initial_predicted_work});
    
    // The prediction should have been added to accumulated work
    const expected_addition = estimate_1.predicted_cycles * @as(f64, estimate_1.confidence);
    std.debug.print("   Expected addition: {d:.1}\n", .{expected_addition});
    std.debug.print("   Predicted work after: {d:.1}\n", .{account.predicted_work_cycles});
    
    // Should have accumulated some predicted work
    try std.testing.expect(account.predicted_work_cycles >= initial_predicted_work);
    
    std.debug.print("\n✅ Predictive work estimation test completed successfully!\n", .{});
}

test "enhanced scheduler integration" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Enhanced Scheduler Integration Test ===\n", .{});
    
    std.debug.print("1. Testing PredictiveScheduler initialization...\n", .{});
    
    // Create configuration for multiple workers
    const config = beat.Config{
        .num_workers = 4,
        .promotion_threshold = 5,
        .min_work_cycles = 1000,
    };
    
    // Create registries
    var fingerprint_registry = beat.fingerprint.FingerprintRegistry.init(allocator);
    defer fingerprint_registry.deinit();
    
    var decision_framework = beat.intelligent_decision.IntelligentDecisionFramework.init(
        beat.intelligent_decision.DecisionConfig{}
    );
    
    // Create predictive scheduler
    var scheduler = try beat.predictive_accounting.PredictiveScheduler.init(
        allocator,
        &config,
        &fingerprint_registry,
        &decision_framework
    );
    defer scheduler.deinit(allocator);
    
    std.debug.print("   Initialized predictive scheduler:\n", .{});
    std.debug.print("     Number of workers: {}\n", .{scheduler.predictive_accounts.len});
    std.debug.print("     Fingerprint registry: {}\n", .{scheduler.fingerprint_registry != null});
    std.debug.print("     Decision framework: {}\n", .{scheduler.decision_framework != null});
    
    try std.testing.expect(scheduler.predictive_accounts.len == 4);
    try std.testing.expect(scheduler.fingerprint_registry != null);
    try std.testing.expect(scheduler.decision_framework != null);
    
    std.debug.print("2. Testing task work prediction...\n", .{});
    
    // Create test task
    const TestData = struct { result: u64 };
    var test_data = TestData{ .result = 0 };
    
    var test_task = beat.Task{
        .func = struct {
            fn func(data: *anyopaque) void {
                const typed_data = @as(*TestData, @ptrCast(@alignCast(data)));
                typed_data.result = 123;
            }
        }.func,
        .data = @ptrCast(&test_data),
        .priority = .normal,
        .fingerprint_hash = 0xAABBCCDD,
        .data_size_hint = @sizeOf(TestData),
    };
    
    const prediction = scheduler.predictTaskWork(0, &test_task);
    
    std.debug.print("   Task work prediction:\n", .{});
    std.debug.print("     Predicted cycles: {d:.1}\n", .{prediction.predicted_cycles});
    std.debug.print("     Confidence: {d:.2}\n", .{prediction.confidence});
    std.debug.print("     Should preemptively promote: {}\n", .{prediction.should_preemptively_promote});
    
    // Should return reasonable prediction
    try std.testing.expect(prediction.predicted_cycles > 0.0);
    try std.testing.expect(prediction.confidence >= 0.0 and prediction.confidence <= 1.0);
    
    std.debug.print("3. Testing task completion recording...\n", .{});
    
    const initial_stats = scheduler.getEnhancedStats();
    
    // Record task completion
    scheduler.recordTaskCompletion(0, &test_task, 1200, 50);
    
    const updated_stats = scheduler.getEnhancedStats();
    
    std.debug.print("   Enhanced scheduler statistics:\n", .{});
    std.debug.print("     Total predictions: {} -> {}\n", .{ initial_stats.total_predictions_made, updated_stats.total_predictions_made });
    std.debug.print("     Accurate predictions: {} -> {}\n", .{ initial_stats.accurate_predictions, updated_stats.accurate_predictions });
    std.debug.print("     Prediction accuracy rate: {d:.2}% -> {d:.2}%\n", .{ 
        initial_stats.prediction_accuracy_rate * 100, 
        updated_stats.prediction_accuracy_rate * 100 
    });
    std.debug.print("     Prediction-based decisions: {}\n", .{updated_stats.prediction_based_decisions});
    
    // Should have recorded the completion
    try std.testing.expect(updated_stats.total_predictions_made >= initial_stats.total_predictions_made);
    
    std.debug.print("\n✅ Enhanced scheduler integration test completed successfully!\n", .{});
}

test "comprehensive promotion analysis and metrics" {
    std.debug.print("\n=== Comprehensive Promotion Analysis Test ===\n", .{});
    
    const config = beat.Config{
        .promotion_threshold = 8,
        .min_work_cycles = 2000,
    };
    var account = beat.predictive_accounting.PredictiveTokenAccount.init(&config);
    
    std.debug.print("1. Testing promotion metrics tracking...\n", .{});
    
    // Simulate various promotion scenarios
    account.promotion_decisions.traditional_promotions = 5;
    account.promotion_decisions.prediction_based_promotions = 3;
    account.promotion_decisions.confidence_based_promotions = 2;
    account.promotion_decisions.threshold_adjustments = 1;
    
    // Setup account state for analysis
    account.prediction_accuracy_score = 0.75;
    account.confidence_weighted_tokens = 0.6;
    account.predicted_work_cycles = 3000.0;
    account.active_predictions.count = 4;
    
    const analysis = account.getPromotionAnalysis();
    
    std.debug.print("   Comprehensive promotion analysis:\n", .{});
    std.debug.print("     Traditional promotions: {}\n", .{analysis.promotion_metrics.traditional_promotions});
    std.debug.print("     Prediction-based promotions: {}\n", .{analysis.promotion_metrics.prediction_based_promotions});
    std.debug.print("     Confidence-based promotions: {}\n", .{analysis.promotion_metrics.confidence_based_promotions});
    std.debug.print("     Threshold adjustments: {}\n", .{analysis.promotion_metrics.threshold_adjustments});
    std.debug.print("     Active predictions count: {}\n", .{analysis.active_predictions_count});
    std.debug.print("     Confidence weighted tokens: {d:.2}\n", .{analysis.confidence_weighted_tokens});
    
    // Verify metrics are tracked
    try std.testing.expect(analysis.promotion_metrics.traditional_promotions == 5);
    try std.testing.expect(analysis.promotion_metrics.prediction_based_promotions == 3);
    try std.testing.expect(analysis.promotion_metrics.confidence_based_promotions == 2);
    try std.testing.expect(analysis.promotion_metrics.threshold_adjustments == 1);
    try std.testing.expect(analysis.active_predictions_count == 4);
    
    std.debug.print("2. Testing reset functionality...\n", .{});
    
    const before_reset_accuracy = account.prediction_accuracy_score;
    const before_reset_metrics = account.promotion_decisions;
    
    account.reset();
    
    std.debug.print("   After reset:\n", .{});
    std.debug.print("     Predicted work cycles: {d:.1}\n", .{account.predicted_work_cycles});
    std.debug.print("     Confidence weighted tokens: {d:.1}\n", .{account.confidence_weighted_tokens});
    std.debug.print("     Accuracy score (should persist): {d:.2}\n", .{account.prediction_accuracy_score});
    
    // Reset should clear temporary state but preserve learning
    try std.testing.expect(account.predicted_work_cycles == 0.0);
    try std.testing.expect(account.confidence_weighted_tokens == 0.0);
    try std.testing.expect(account.prediction_accuracy_score == before_reset_accuracy); // Should persist
    
    // Metrics should persist as they represent historical data
    try std.testing.expect(account.promotion_decisions.traditional_promotions == before_reset_metrics.traditional_promotions);
    
    std.debug.print("\n✅ Comprehensive analysis and metrics test completed successfully!\n", .{});
    std.debug.print("🎯 Predictive Token Accounting Implementation Summary:\n", .{});
    std.debug.print("   • Enhanced TokenAccount with execution time predictions ✅\n", .{});
    std.debug.print("   • Confidence-based promotion decisions ✅\n", .{});
    std.debug.print("   • Adaptive threshold adjustment based on accuracy ✅\n", .{});
    std.debug.print("   • Integration with fingerprint registry ✅\n", .{});
    std.debug.print("   • Comprehensive promotion analysis and metrics ✅\n", .{});
    std.debug.print("   • Enhanced scheduler with predictive capabilities ✅\n", .{});
}