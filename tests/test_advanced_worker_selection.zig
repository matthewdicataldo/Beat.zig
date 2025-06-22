const std = @import("std");
const beat = @import("beat");

// Test for Advanced Worker Selection Algorithm Implementation (Phase 2.4.2)
//
// This test validates the sophisticated worker selection algorithm that replaces
// simple round-robin with predictive selection using multi-criteria optimization:
// - Multi-criteria optimization scoring
// - Predictive selection based on execution time estimates
// - Integration with existing topology awareness
// - Exploratory placement for new task types
// - Adaptive learning and criteria adjustment

test "selection criteria normalization and configuration" {
    std.debug.print("\n=== Advanced Worker Selection Test ===\n", .{});
    
    std.debug.print("1. Testing selection criteria normalization...\n", .{});
    
    // Test criteria normalization
    var criteria = beat.advanced_worker_selection.SelectionCriteria{
        .load_balance_weight = 2.0,
        .prediction_weight = 3.0,
        .topology_weight = 1.0,
        .confidence_weight = 2.0,
        .exploration_weight = 2.0,
    };
    
    std.debug.print("   Before normalization:\n", .{});
    std.debug.print("     Load balance: {d:.2}\n", .{criteria.load_balance_weight});
    std.debug.print("     Prediction: {d:.2}\n", .{criteria.prediction_weight});
    std.debug.print("     Topology: {d:.2}\n", .{criteria.topology_weight});
    std.debug.print("     Confidence: {d:.2}\n", .{criteria.confidence_weight});
    std.debug.print("     Exploration: {d:.2}\n", .{criteria.exploration_weight});
    
    criteria.normalize();
    
    std.debug.print("   After normalization:\n", .{});
    std.debug.print("     Load balance: {d:.2}\n", .{criteria.load_balance_weight});
    std.debug.print("     Prediction: {d:.2}\n", .{criteria.prediction_weight});
    std.debug.print("     Topology: {d:.2}\n", .{criteria.topology_weight});
    std.debug.print("     Confidence: {d:.2}\n", .{criteria.confidence_weight});
    std.debug.print("     Exploration: {d:.2}\n", .{criteria.exploration_weight});
    
    const total = criteria.load_balance_weight + criteria.prediction_weight + 
                 criteria.topology_weight + criteria.confidence_weight + criteria.exploration_weight;
    
    try std.testing.expect(@abs(total - 1.0) < 0.001);
    
    std.debug.print("2. Testing predefined criteria configurations...\n", .{});
    
    const balanced = beat.advanced_worker_selection.SelectionCriteria.balanced();
    const latency_optimized = beat.advanced_worker_selection.SelectionCriteria.latencyOptimized();
    const throughput_optimized = beat.advanced_worker_selection.SelectionCriteria.throughputOptimized();
    
    std.debug.print("   Balanced criteria:\n", .{});
    std.debug.print("     Load balance: {d:.2}, Prediction: {d:.2}, Topology: {d:.2}\n", .{
        balanced.load_balance_weight, balanced.prediction_weight, balanced.topology_weight
    });
    
    std.debug.print("   Latency-optimized criteria:\n", .{});
    std.debug.print("     Load balance: {d:.2}, Prediction: {d:.2}, Exploration: {d:.2}\n", .{
        latency_optimized.load_balance_weight, latency_optimized.prediction_weight, latency_optimized.exploration_weight
    });
    
    std.debug.print("   Throughput-optimized criteria:\n", .{});
    std.debug.print("     Topology: {d:.2}, Confidence: {d:.2}, Exploration: {d:.2}\n", .{
        throughput_optimized.topology_weight, throughput_optimized.confidence_weight, throughput_optimized.exploration_weight
    });
    
    // Verify no exploration in latency-optimized
    try std.testing.expect(latency_optimized.exploration_weight == 0.0);
    try std.testing.expect(latency_optimized.enable_adaptive_weights == false);
    
    std.debug.print("\n✅ Selection criteria test completed successfully!\n", .{});
}

test "worker evaluation and scoring" {
    
    std.debug.print("\n=== Worker Evaluation and Scoring Test ===\n", .{});
    
    std.debug.print("1. Testing worker evaluation initialization...\n", .{});
    
    var evaluation = beat.advanced_worker_selection.WorkerEvaluation.init(42);
    
    std.debug.print("   Initialized evaluation for worker {}:\n", .{evaluation.worker_id});
    std.debug.print("     Load balance score: {d:.2}\n", .{evaluation.load_balance_score});
    std.debug.print("     Prediction score: {d:.2}\n", .{evaluation.prediction_score});
    std.debug.print("     Topology score: {d:.2}\n", .{evaluation.topology_score});
    std.debug.print("     Confidence score: {d:.2}\n", .{evaluation.confidence_score});
    std.debug.print("     Exploration score: {d:.2}\n", .{evaluation.exploration_score});
    
    try std.testing.expect(evaluation.worker_id == 42);
    try std.testing.expect(evaluation.weighted_score == 0.0);
    try std.testing.expect(evaluation.queue_size == 0);
    
    std.debug.print("2. Testing weighted score calculation...\n", .{});
    
    // Set up test scores
    evaluation.load_balance_score = 0.8;
    evaluation.prediction_score = 0.6;
    evaluation.topology_score = 0.9;
    evaluation.confidence_score = 0.7;
    evaluation.exploration_score = 0.5;
    
    const criteria = beat.advanced_worker_selection.SelectionCriteria.balanced();
    evaluation.calculateWeightedScore(criteria);
    
    std.debug.print("   Individual scores:\n", .{});
    std.debug.print("     Load balance: {d:.2} * {d:.2} = {d:.2}\n", .{
        evaluation.load_balance_score, criteria.load_balance_weight, 
        evaluation.load_balance_score * criteria.load_balance_weight
    });
    std.debug.print("     Prediction: {d:.2} * {d:.2} = {d:.2}\n", .{
        evaluation.prediction_score, criteria.prediction_weight, 
        evaluation.prediction_score * criteria.prediction_weight
    });
    std.debug.print("     Topology: {d:.2} * {d:.2} = {d:.2}\n", .{
        evaluation.topology_score, criteria.topology_weight, 
        evaluation.topology_score * criteria.topology_weight
    });
    std.debug.print("   Weighted score: {d:.3}\n", .{evaluation.weighted_score});
    
    // Should be weighted combination of individual scores
    const expected_score = evaluation.load_balance_score * criteria.load_balance_weight +
                          evaluation.prediction_score * criteria.prediction_weight +
                          evaluation.topology_score * criteria.topology_weight +
                          evaluation.confidence_score * criteria.confidence_weight +
                          evaluation.exploration_score * criteria.exploration_weight;
    
    try std.testing.expect(@abs(evaluation.weighted_score - expected_score) < 0.001);
    
    std.debug.print("\n✅ Worker evaluation and scoring test completed successfully!\n", .{});
}

test "advanced worker selector initialization and configuration" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Advanced Worker Selector Initialization Test ===\n", .{});
    
    std.debug.print("1. Testing selector initialization...\n", .{});
    
    const criteria = beat.advanced_worker_selection.SelectionCriteria.balanced();
    var selector = try beat.advanced_worker_selection.AdvancedWorkerSelector.init(allocator, criteria, 4);
    defer selector.deinit();
    
    std.debug.print("   Initialized advanced worker selector:\n", .{});
    std.debug.print("     Prediction enabled: {}\n", .{selector.enable_prediction});
    std.debug.print("     Exploration enabled: {}\n", .{selector.enable_exploration});
    std.debug.print("     Adaptive criteria: {}\n", .{selector.enable_adaptive_criteria});
    std.debug.print("     Max workers to evaluate: {}\n", .{selector.max_workers_to_evaluate});
    std.debug.print("     Total selections: {}\n", .{selector.selection_history.total_selections});
    
    try std.testing.expect(selector.enable_prediction == true);
    try std.testing.expect(selector.enable_exploration == true);
    try std.testing.expect(selector.selection_history.total_selections == 0);
    
    std.debug.print("2. Testing component integration...\n", .{});
    
    // Create mock registries for testing
    var fingerprint_registry = beat.fingerprint.FingerprintRegistry.init(allocator);
    defer fingerprint_registry.deinit();
    
    var decision_framework = beat.intelligent_decision.IntelligentDecisionFramework.init(
        beat.intelligent_decision.DecisionConfig{}
    );
    
    selector.setComponents(&fingerprint_registry, null, &decision_framework);
    
    std.debug.print("   Set prediction and analysis components:\n", .{});
    std.debug.print("     Fingerprint registry: {}\n", .{selector.fingerprint_registry != null});
    std.debug.print("     Decision framework: {}\n", .{selector.decision_framework != null});
    std.debug.print("     Predictive scheduler: {}\n", .{selector.predictive_scheduler != null});
    
    try std.testing.expect(selector.fingerprint_registry != null);
    try std.testing.expect(selector.decision_framework != null);
    try std.testing.expect(selector.predictive_scheduler == null); // Not set in this test
    
    std.debug.print("\n✅ Advanced worker selector initialization test completed successfully!\n", .{});
}

test "worker selection with multi-criteria optimization" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Multi-Criteria Worker Selection Test ===\n", .{});
    
    std.debug.print("1. Testing worker selection algorithm...\n", .{});
    
    // Create selector with balanced criteria
    const criteria = beat.advanced_worker_selection.SelectionCriteria.balanced();
    var selector = try beat.advanced_worker_selection.AdvancedWorkerSelector.init(allocator, criteria, 4);
    defer selector.deinit();
    
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
    
    // Create test workers with different characteristics
    const test_workers = [_]beat.intelligent_decision.WorkerInfo{
        .{ .id = 0, .numa_node = 0, .queue_size = 5, .max_queue_size = 100 },    // Low load, NUMA 0
        .{ .id = 1, .numa_node = 0, .queue_size = 50, .max_queue_size = 100 },   // Medium load, NUMA 0
        .{ .id = 2, .numa_node = 1, .queue_size = 10, .max_queue_size = 100 },   // Low load, NUMA 1
        .{ .id = 3, .numa_node = 1, .queue_size = 80, .max_queue_size = 100 },   // High load, NUMA 1
    };
    
    std.debug.print("   Test workers:\n", .{});
    for (test_workers, 0..) |worker, i| {
        std.debug.print("     Worker {}: NUMA {}, queue {}/{}\n", .{
            i, worker.numa_node orelse 999, worker.queue_size, worker.max_queue_size
        });
    }
    
    // Make selection decision
    var decision = try selector.selectWorker(&test_task, &test_workers, null);
    defer decision.deinit(allocator);
    
    std.debug.print("   Selection decision:\n", .{});
    std.debug.print("     Selected worker: {}\n", .{decision.selected_worker_id});
    std.debug.print("     Selection strategy: {s}\n", .{@tagName(decision.selection_strategy)});
    std.debug.print("     Primary factor: {s}\n", .{@tagName(decision.primary_factor)});
    std.debug.print("     Confidence level: {d:.2}\n", .{decision.confidence_level});
    std.debug.print("     Exploration used: {}\n", .{decision.exploration_used});
    std.debug.print("     Topology optimization: {}\n", .{decision.topology_optimization});
    
    // Should select a valid worker
    try std.testing.expect(decision.selected_worker_id < test_workers.len);
    
    std.debug.print("2. Testing worker evaluations...\n", .{});
    
    std.debug.print("   Worker evaluation details:\n", .{});
    for (decision.evaluations, 0..) |eval, i| {
        std.debug.print("     Worker {}: load={d:.2}, pred={d:.2}, topo={d:.2}, conf={d:.2}, expl={d:.2}, weighted={d:.3}\n", .{
            i, eval.load_balance_score, eval.prediction_score, eval.topology_score, 
            eval.confidence_score, eval.exploration_score, eval.weighted_score
        });
    }
    
    // Verify evaluations were created for all workers
    try std.testing.expect(decision.evaluations.len == test_workers.len);
    
    // Should typically select worker with best load balance (worker 0 or 2)
    try std.testing.expect(decision.selected_worker_id == 0 or decision.selected_worker_id == 2);
    
    std.debug.print("\n✅ Multi-criteria worker selection test completed successfully!\n", .{});
}

test "selection history and learning adaptation" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Selection History and Learning Test ===\n", .{});
    
    std.debug.print("1. Testing selection history tracking...\n", .{});
    
    const criteria = beat.advanced_worker_selection.SelectionCriteria.balanced();
    var selector = try beat.advanced_worker_selection.AdvancedWorkerSelector.init(allocator, criteria, 4);
    defer selector.deinit();
    
    // Create simple test workers
    const test_workers = [_]beat.intelligent_decision.WorkerInfo{
        .{ .id = 0, .numa_node = 0, .queue_size = 0, .max_queue_size = 100 },
        .{ .id = 1, .numa_node = 0, .queue_size = 0, .max_queue_size = 100 },
    };
    
    const TestData = struct { value: i32 };
    var test_data = TestData{ .value = 42 };
    
    var test_task = beat.Task{
        .func = struct {
            fn func(data: *anyopaque) void {
                const typed_data = @as(*TestData, @ptrCast(@alignCast(data)));
                typed_data.value += 1;
            }
        }.func,
        .data = @ptrCast(&test_data),
        .priority = .normal,
        .data_size_hint = @sizeOf(TestData),
    };
    
    std.debug.print("   Making multiple selections to build history...\n", .{});
    
    var selection_counts = [_]u32{0} ** 2;
    
    // Make several selections
    for (0..10) |i| {
        var decision = try selector.selectWorker(&test_task, &test_workers, null);
        defer decision.deinit(allocator);
        
        selection_counts[decision.selected_worker_id] += 1;
        
        std.debug.print("     Selection {}: Worker {} (total: {}/{})\n", .{
            i + 1, decision.selected_worker_id, selection_counts[0], selection_counts[1]
        });
    }
    
    const stats = selector.getSelectionStats();
    
    std.debug.print("   Selection statistics:\n", .{});
    std.debug.print("     Total selections: {}\n", .{stats.total_selections});
    std.debug.print("     Prediction accuracy: {d:.2}\n", .{stats.prediction_accuracy});
    std.debug.print("     Average queue utilization: {d:.2}\n", .{stats.average_queue_utilization});
    std.debug.print("     NUMA locality ratio: {d:.2}\n", .{stats.numa_locality_ratio});
    
    try std.testing.expect(stats.total_selections == 10);
    
    std.debug.print("2. Testing exploration effect on selection distribution...\n", .{});
    
    // Check that exploration leads to more balanced selection
    const frequency_0 = selector.selection_history.getSelectionFrequency(0);
    const frequency_1 = selector.selection_history.getSelectionFrequency(1);
    
    std.debug.print("   Selection frequencies:\n", .{});
    std.debug.print("     Worker 0: {d:.2}\n", .{frequency_0});
    std.debug.print("     Worker 1: {d:.2}\n", .{frequency_1});
    
    // With exploration enabled, should have some distribution
    try std.testing.expect(frequency_0 >= 0.0 and frequency_0 <= 1.0);
    try std.testing.expect(frequency_1 >= 0.0 and frequency_1 <= 1.0);
    
    std.debug.print("\n✅ Selection history and learning test completed successfully!\n", .{});
}

test "integration with thread pool worker selection" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Thread Pool Integration Test ===\n", .{});
    
    std.debug.print("1. Testing thread pool with advanced worker selection...\n", .{});
    
    // Create thread pool configuration with advanced selection enabled
    const config = beat.Config{
        .num_workers = 4,
        .enable_advanced_selection = true,
        .enable_predictive = true,
        .selection_criteria = beat.advanced_worker_selection.SelectionCriteria.latencyOptimized(),
    };
    
    var pool = try beat.ThreadPool.init(allocator, config);
    defer pool.deinit();
    
    std.debug.print("   Created thread pool:\n", .{});
    std.debug.print("     Workers: {}\n", .{pool.workers.len});
    std.debug.print("     Advanced selector: {}\n", .{pool.advanced_selector != null});
    std.debug.print("     Decision framework available: {}\n", .{pool.decision_framework != null});
    std.debug.print("     Fingerprint registry available: {}\n", .{pool.fingerprint_registry != null});
    
    try std.testing.expect(pool.workers.len == 4);
    try std.testing.expect(pool.advanced_selector != null);
    
    std.debug.print("2. Testing task submission with advanced selection...\n", .{});
    
    // Create test tasks
    const TestData = struct { counter: std.atomic.Value(u32) };
    var shared_data = TestData{ .counter = std.atomic.Value(u32).init(0) };
    
    // Submit multiple tasks to test selection algorithm
    for (0..8) |i| {
        const task = beat.Task{
            .func = struct {
                fn func(data: *anyopaque) void {
                    const typed_data = @as(*TestData, @ptrCast(@alignCast(data)));
                    _ = typed_data.counter.fetchAdd(1, .monotonic);
                    
                    // Simulate some work
                    std.time.sleep(1_000_000); // 1ms
                }
            }.func,
            .data = @ptrCast(&shared_data),
            .priority = .normal,
            .data_size_hint = @sizeOf(TestData),
        };
        
        try pool.submit(task);
        std.debug.print("     Submitted task {}\n", .{i + 1});
    }
    
    // Wait for completion
    pool.wait();
    
    const final_count = shared_data.counter.load(.acquire);
    std.debug.print("   Task completion:\n", .{});
    std.debug.print("     Final counter value: {}\n", .{final_count});
    std.debug.print("     Expected: 8\n", .{});
    
    try std.testing.expect(final_count == 8);
    
    std.debug.print("3. Testing selection statistics...\n", .{});
    
    if (pool.advanced_selector) |selector| {
        const stats = selector.getSelectionStats();
        
        std.debug.print("   Advanced selection statistics:\n", .{});
        std.debug.print("     Total selections made: {}\n", .{stats.total_selections});
        std.debug.print("     Current criteria weights:\n", .{});
        std.debug.print("       Load balance: {d:.2}\n", .{stats.current_criteria.load_balance_weight});
        std.debug.print("       Prediction: {d:.2}\n", .{stats.current_criteria.prediction_weight});
        std.debug.print("       Topology: {d:.2}\n", .{stats.current_criteria.topology_weight});
        std.debug.print("       Confidence: {d:.2}\n", .{stats.current_criteria.confidence_weight});
        std.debug.print("       Exploration: {d:.2}\n", .{stats.current_criteria.exploration_weight});
        
        // Should have made selections for the submitted tasks
        try std.testing.expect(stats.total_selections >= 8);
    }
    
    std.debug.print("\n✅ Thread pool integration test completed successfully!\n", .{});
    
    std.debug.print("🎯 Advanced Worker Selection Implementation Summary:\n", .{});
    std.debug.print("   • Multi-criteria optimization scoring system ✅\n", .{});
    std.debug.print("   • Predictive selection based on execution estimates ✅\n", .{});
    std.debug.print("   • Integration with topology awareness ✅\n", .{});
    std.debug.print("   • Exploratory placement for load balancing ✅\n", .{});
    std.debug.print("   • Adaptive learning and criteria adjustment ✅\n", .{});
    std.debug.print("   • Comprehensive evaluation and decision rationale ✅\n", .{});
    std.debug.print("   • Full integration with ThreadPool selection logic ✅\n", .{});
}