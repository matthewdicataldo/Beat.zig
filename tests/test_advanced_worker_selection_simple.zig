const std = @import("std");
const beat = @import("beat");

// Simplified Advanced Worker Selection Test
// This test validates the core functionality without complex formatting

test "advanced worker selection - basic functionality" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Advanced Worker Selection - Basic Test ===\n", .{});
    
    // Test 1: Criteria normalization
    std.debug.print("1. Testing selection criteria...\n", .{});
    
    var criteria = beat.advanced_worker_selection.SelectionCriteria{
        .load_balance_weight = 2.0,
        .prediction_weight = 3.0,
        .topology_weight = 1.0,
        .confidence_weight = 2.0,
        .exploration_weight = 2.0,
    };
    
    criteria.normalize();
    
    const total = criteria.load_balance_weight + criteria.prediction_weight + 
                 criteria.topology_weight + criteria.confidence_weight + criteria.exploration_weight + criteria.simd_weight;
    
    try std.testing.expect(@abs(total - 1.0) < 0.001);
    std.debug.print("   ✅ Criteria normalization works\n", .{});
    
    // Test 2: Worker evaluation
    std.debug.print("2. Testing worker evaluation...\n", .{});
    
    var evaluation = beat.advanced_worker_selection.WorkerEvaluation.init(42);
    
    try std.testing.expect(evaluation.worker_id == 42);
    try std.testing.expect(evaluation.weighted_score == 0.0);
    
    evaluation.load_balance_score = 0.8;
    evaluation.prediction_score = 0.6;
    evaluation.topology_score = 0.9;
    evaluation.confidence_score = 0.7;
    evaluation.exploration_score = 0.5;
    evaluation.simd_score = 0.8;
    
    const balanced_criteria = beat.advanced_worker_selection.SelectionCriteria.balanced();
    evaluation.calculateWeightedScore(balanced_criteria);
    
    try std.testing.expect(evaluation.weighted_score > 0.0);
    std.debug.print("   ✅ Worker evaluation scoring works\n", .{});
    
    // Test 3: Selector initialization
    std.debug.print("3. Testing selector initialization...\n", .{});
    
    var selector = beat.advanced_worker_selection.AdvancedWorkerSelector.init(allocator, balanced_criteria);
    
    try std.testing.expect(selector.enable_prediction == true);
    try std.testing.expect(selector.enable_exploration == true);
    try std.testing.expect(selector.selection_history.total_selections == 0);
    
    // Set up components
    var fingerprint_registry = beat.fingerprint.FingerprintRegistry.init(allocator);
    defer fingerprint_registry.deinit();
    
    var decision_framework = beat.intelligent_decision.IntelligentDecisionFramework.init(
        beat.intelligent_decision.DecisionConfig{}
    );
    
    selector.setComponents(&fingerprint_registry, null, &decision_framework, null);
    
    try std.testing.expect(selector.fingerprint_registry != null);
    try std.testing.expect(selector.decision_framework != null);
    std.debug.print("   ✅ Selector initialization works\n", .{});
    
    std.debug.print("\n✅ Advanced Worker Selection basic test completed successfully!\n", .{});
}

test "advanced worker selection - selection algorithm" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Worker Selection Algorithm Test ===\n", .{});
    
    // Create selector
    const criteria = beat.advanced_worker_selection.SelectionCriteria.balanced();
    var selector = beat.advanced_worker_selection.AdvancedWorkerSelector.init(allocator, criteria);
    
    // Create test task
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
        .data_size_hint = @sizeOf(TestData),
    };
    
    // Create test workers
    const test_workers = [_]beat.intelligent_decision.WorkerInfo{
        .{ .id = 0, .numa_node = 0, .queue_size = 5, .max_queue_size = 100 },
        .{ .id = 1, .numa_node = 0, .queue_size = 50, .max_queue_size = 100 },
        .{ .id = 2, .numa_node = 1, .queue_size = 10, .max_queue_size = 100 },
        .{ .id = 3, .numa_node = 1, .queue_size = 80, .max_queue_size = 100 },
    };
    
    std.debug.print("1. Testing worker selection...\n", .{});
    
    // Make selection decision
    var decision = try selector.selectWorker(&test_task, &test_workers, null);
    defer decision.deinit(allocator);
    
    std.debug.print("   Selected worker: {}\n", .{decision.selected_worker_id});
    
    // Should select a valid worker
    try std.testing.expect(decision.selected_worker_id < test_workers.len);
    
    // Verify evaluations were created for all workers
    try std.testing.expect(decision.evaluations.len == test_workers.len);
    
    // Should typically select worker with better load balance (worker 0 or 2)
    try std.testing.expect(decision.selected_worker_id == 0 or decision.selected_worker_id == 2);
    
    std.debug.print("   ✅ Worker selection completed successfully\n", .{});
    
    // Test selection history
    std.debug.print("2. Testing selection history...\n", .{});
    
    const stats = selector.getSelectionStats();
    try std.testing.expect(stats.total_selections >= 1);
    
    std.debug.print("   Total selections: {}\n", .{stats.total_selections});
    std.debug.print("   ✅ Selection history tracking works\n", .{});
    
    std.debug.print("\n✅ Selection algorithm test completed successfully!\n", .{});
}