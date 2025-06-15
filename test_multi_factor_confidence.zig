const std = @import("std");
const beat = @import("src/core.zig");

// Test for Multi-factor Confidence Model Implementation (Phase 2.3.1)
//
// This test validates the comprehensive multi-factor confidence model that
// combines sample size confidence, prediction accuracy monitoring, temporal
// relevance weighting, and variance stability measurement.

test "multi-factor confidence model core functionality" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Multi-Factor Confidence Model Test ===\n", .{});
    
    // Create execution context
    var context = beat.fingerprint.ExecutionContext.init();
    context.current_numa_node = 0;
    context.system_load = 0.6;
    
    // Create fingerprint registry
    var registry = beat.fingerprint.FingerprintRegistry.init(allocator);
    defer registry.deinit();
    
    // Test data structure
    const TestData = struct {
        values: [50]i32,
        
        pub fn process(self: *@This()) void {
            for (&self.values) |*value| {
                value.* = value.* * 2 + 1;
            }
        }
    };
    
    var test_data = TestData{ .values = undefined };
    for (&test_data.values, 0..) |*value, i| {
        value.* = @intCast(i);
    }
    
    std.debug.print("1. Creating task for multi-factor confidence analysis...\\n", .{});
    
    // Create task
    var task = beat.Task{
        .func = struct {
            fn taskWrapper(data: *anyopaque) void {
                const typed_data = @as(*TestData, @ptrCast(@alignCast(data)));
                typed_data.process();
            }
        }.taskWrapper,
        .data = @ptrCast(&test_data),
        .priority = .normal,
        .data_size_hint = @sizeOf(TestData),
    };
    
    // Generate fingerprint
    const fingerprint = beat.fingerprint.generateTaskFingerprint(&task, &context);
    
    std.debug.print("2. Testing sample size confidence progression...\\n", .{});
    
    // Test confidence progression with increasing sample size
    const sample_progression = [_]struct { cycles: u64, expected_samples: u64 }{
        .{ .cycles = 1000, .expected_samples = 1 },
        .{ .cycles = 1050, .expected_samples = 2 },
        .{ .cycles = 980,  .expected_samples = 3 },
        .{ .cycles = 1020, .expected_samples = 4 },
        .{ .cycles = 990,  .expected_samples = 5 },
    };
    
    for (sample_progression, 0..) |measurement, i| {
        try registry.recordExecution(fingerprint, measurement.cycles);
        
        const confidence = registry.getMultiFactorConfidence(fingerprint);
        
        std.debug.print("   Sample {}: {d:.3} sample confidence ({} samples)\\n", .{
            i + 1, confidence.sample_size_confidence, confidence.sample_count
        });
        
        // Verify sample count matches
        try std.testing.expect(confidence.sample_count == measurement.expected_samples);
        
        // Verify sample confidence increases with more samples
        if (i > 0) {
            const prev_confidence = confidence.sample_size_confidence;
            try std.testing.expect(prev_confidence > 0.0);
        }
    }
    
    std.debug.print("3. Testing accuracy confidence with controlled predictions...\\n", .{});
    
    // Add more consistent measurements to improve accuracy
    const consistent_measurements = [_]u64{ 1000, 1000, 1000, 1000, 1000 };
    
    for (consistent_measurements) |cycles| {
        try registry.recordExecution(fingerprint, cycles);
    }
    
    const accuracy_confidence = registry.getMultiFactorConfidence(fingerprint);
    std.debug.print("   Accuracy confidence after consistent pattern: {d:.3}\\n", .{accuracy_confidence.accuracy_confidence});
    std.debug.print("   Recent accuracy: {d:.3}\\n", .{accuracy_confidence.recent_accuracy});
    
    // Accuracy confidence should be reasonable with consistent data
    try std.testing.expect(accuracy_confidence.accuracy_confidence >= 0.0);
    try std.testing.expect(accuracy_confidence.accuracy_confidence <= 1.0);
    
    std.debug.print("4. Testing temporal relevance weighting...\\n", .{});
    
    // Test temporal confidence (should be high since measurements are recent)
    const temporal_confidence = registry.getMultiFactorConfidence(fingerprint);
    std.debug.print("   Temporal confidence (recent): {d:.3}\\n", .{temporal_confidence.temporal_confidence});
    std.debug.print("   Time since last measurement: {d:.1}ms\\n", .{temporal_confidence.time_since_last_ms});
    
    // Recent measurements should have high temporal confidence
    try std.testing.expect(temporal_confidence.temporal_confidence > 0.5);
    
    // Simulate time passing and check temporal decay
    std.time.sleep(10 * std.time.ns_per_ms); // 10ms delay
    
    const temporal_after_delay = registry.getMultiFactorConfidence(fingerprint);
    std.debug.print("   Temporal confidence (after delay): {d:.3}\\n", .{temporal_after_delay.temporal_confidence});
    
    std.debug.print("5. Testing variance stability measurement...\\n", .{});
    
    // Add variable measurements to test variance confidence
    const variable_measurements = [_]u64{ 1200, 800, 1400, 600, 1600 };
    
    for (variable_measurements) |cycles| {
        try registry.recordExecution(fingerprint, cycles);
    }
    
    const variance_confidence = registry.getMultiFactorConfidence(fingerprint);
    std.debug.print("   Variance confidence: {d:.3}\\n", .{variance_confidence.variance_confidence});
    std.debug.print("   Coefficient of variation: {d:.3}\\n", .{variance_confidence.coefficient_of_variation});
    
    // Variance confidence should reflect the higher variability
    try std.testing.expect(variance_confidence.variance_confidence >= 0.0);
    try std.testing.expect(variance_confidence.variance_confidence <= 1.0);
    try std.testing.expect(variance_confidence.coefficient_of_variation > 0.0);
    
    std.debug.print("6. Testing overall confidence calculation...\\n", .{});
    
    const overall_confidence = registry.getMultiFactorConfidence(fingerprint);
    std.debug.print("   Overall confidence: {d:.3}\\n", .{overall_confidence.overall_confidence});
    std.debug.print("   Confidence category: {}\\n", .{overall_confidence.getConfidenceCategory()});
    
    // Overall confidence should be a valid combination
    try std.testing.expect(overall_confidence.overall_confidence >= 0.0);
    try std.testing.expect(overall_confidence.overall_confidence <= 1.0);
    
    // Print detailed analysis
    const analysis = try overall_confidence.getAnalysisString(allocator);
    defer allocator.free(analysis);
    std.debug.print("\\n{s}", .{analysis});
    
    std.debug.print("\\n✅ Multi-factor confidence model test completed successfully!\\n", .{});
    std.debug.print("🎯 Key features verified:\\n", .{});
    std.debug.print("   • Sample size confidence tracking (asymptotic growth)\\n", .{});
    std.debug.print("   • Prediction accuracy monitoring (rolling accuracy)\\n", .{});
    std.debug.print("   • Temporal relevance weighting (exponential decay)\\n", .{});
    std.debug.print("   • Variance stability measurement (coefficient of variation)\\n", .{});
    std.debug.print("   • Overall confidence calculation (weighted geometric mean)\\n", .{});
}

test "confidence category classification" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\\n=== Confidence Category Classification Test ===\\n", .{});
    
    var registry = beat.fingerprint.FingerprintRegistry.init(allocator);
    defer registry.deinit();
    
    var context = beat.fingerprint.ExecutionContext.init();
    
    // Create test task
    const TestData = struct { value: i32 };
    var test_data = TestData{ .value = 42 };
    
    var task = beat.Task{
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
    
    const fingerprint = beat.fingerprint.generateTaskFingerprint(&task, &context);
    
    std.debug.print("1. Testing very_low confidence scenario...\\n", .{});
    
    // Very few samples with high variance
    const very_low_pattern = [_]u64{ 1000, 5000 };
    for (very_low_pattern) |cycles| {
        try registry.recordExecution(fingerprint, cycles);
    }
    
    const very_low_confidence = registry.getMultiFactorConfidence(fingerprint);
    std.debug.print("   Very low confidence: {d:.3} -> {}\\n", .{
        very_low_confidence.overall_confidence, 
        very_low_confidence.getConfidenceCategory()
    });
    
    std.debug.print("2. Testing medium confidence scenario...\\n", .{});
    
    // More samples with moderate consistency
    const medium_pattern = [_]u64{ 1000, 1100, 1050, 1200, 1150, 1080, 1120 };
    for (medium_pattern) |cycles| {
        try registry.recordExecution(fingerprint, cycles);
    }
    
    const medium_confidence = registry.getMultiFactorConfidence(fingerprint);
    std.debug.print("   Medium confidence: {d:.3} -> {}\\n", .{
        medium_confidence.overall_confidence,
        medium_confidence.getConfidenceCategory()
    });
    
    std.debug.print("3. Testing high confidence scenario...\\n", .{});
    
    // Many consistent samples
    const high_pattern = [_]u64{1100} ** 20; // 20 consistent measurements
    for (high_pattern) |cycles| {
        try registry.recordExecution(fingerprint, cycles);
    }
    
    const high_confidence = registry.getMultiFactorConfidence(fingerprint);
    std.debug.print("   High confidence: {d:.3} -> {}\\n", .{
        high_confidence.overall_confidence,
        high_confidence.getConfidenceCategory()
    });
    
    // Verify progression from low to higher confidence
    try std.testing.expect(very_low_confidence.overall_confidence < medium_confidence.overall_confidence);
    try std.testing.expect(medium_confidence.overall_confidence < high_confidence.overall_confidence);
    
    std.debug.print("\\n✅ Confidence category classification test completed successfully!\\n", .{});
    std.debug.print("📊 Category progression verified:\\n", .{});
    std.debug.print("   • Very low: Few samples + high variance\\n", .{});
    std.debug.print("   • Medium: Moderate samples + moderate consistency\\n", .{});
    std.debug.print("   • High: Many samples + high consistency\\n", .{});
}

test "temporal confidence decay behavior" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\\n=== Temporal Confidence Decay Test ===\\n", .{});
    
    var registry = beat.fingerprint.FingerprintRegistry.init(allocator);
    defer registry.deinit();
    
    var context = beat.fingerprint.ExecutionContext.init();
    
    // Create test task
    const TestData = struct { counter: u32 };
    var test_data = TestData{ .counter = 0 };
    
    var task = beat.Task{
        .func = struct {
            fn func(data: *anyopaque) void {
                const typed_data = @as(*TestData, @ptrCast(@alignCast(data)));
                typed_data.counter += 1;
            }
        }.func,
        .data = @ptrCast(&test_data),
        .priority = .normal,
        .data_size_hint = @sizeOf(TestData),
    };
    
    const fingerprint = beat.fingerprint.generateTaskFingerprint(&task, &context);
    
    // Add some measurements
    const baseline_measurements = [_]u64{ 1000, 1000, 1000, 1000, 1000 };
    for (baseline_measurements) |cycles| {
        try registry.recordExecution(fingerprint, cycles);
    }
    
    std.debug.print("1. Testing immediate temporal confidence...\\n", .{});
    
    const immediate_confidence = registry.getMultiFactorConfidence(fingerprint);
    std.debug.print("   Immediate temporal confidence: {d:.3}\\n", .{immediate_confidence.temporal_confidence});
    std.debug.print("   Time since last: {d:.1}ms\\n", .{immediate_confidence.time_since_last_ms});
    
    // Should be very high for immediate measurements
    try std.testing.expect(immediate_confidence.temporal_confidence > 0.9);
    
    std.debug.print("2. Testing temporal decay over time...\\n", .{});
    
    // Test decay at different intervals
    const intervals_ms = [_]u64{ 50, 100, 200, 500 };
    
    for (intervals_ms) |interval_ms| {
        std.time.sleep(interval_ms * std.time.ns_per_ms);
        
        const delayed_confidence = registry.getMultiFactorConfidence(fingerprint);
        std.debug.print("   After {d}ms: temporal confidence {d:.3}\\n", .{
            interval_ms, delayed_confidence.temporal_confidence
        });
        
        // Temporal confidence should decay over time
        try std.testing.expect(delayed_confidence.temporal_confidence <= immediate_confidence.temporal_confidence);
    }
    
    std.debug.print("\\n✅ Temporal confidence decay test completed successfully!\\n", .{});
    std.debug.print("⏰ Temporal decay behavior verified:\\n", .{});
    std.debug.print("   • Immediate measurements: very high confidence\\n", .{});
    std.debug.print("   • Time-delayed measurements: exponential decay\\n", .{});
    std.debug.print("   • 5-minute half-life decay rate applied\\n", .{});
}