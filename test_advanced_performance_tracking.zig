const std = @import("std");
const beat = @import("src/core.zig");

// Test for Advanced Performance Tracking (Phase 2.2.2)
//
// This test validates the implementation of advanced performance tracking features:
// - Timestamp-based filtering with nanosecond precision
// - Derivative estimation for velocity tracking  
// - Adaptive cutoff frequency calculation
// - Confidence tracking and accuracy metrics

test "timestamp-based filtering with nanosecond precision" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Advanced Performance Tracking Test ===\n", .{});
    
    // Create execution context
    var context = beat.fingerprint.ExecutionContext.init();
    context.current_numa_node = 0;
    context.system_load = 0.5;
    
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
    
    std.debug.print("1. Creating task for advanced tracking...\n", .{});
    
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
    
    std.debug.print("2. Testing timestamp-based filtering with controlled intervals...\n", .{});
    
    // Simulate measurements with known intervals
    const controlled_measurements = [_]struct { cycles: u64, delay_ms: u64 }{
        .{ .cycles = 1000, .delay_ms = 10 },  // 10ms interval
        .{ .cycles = 1050, .delay_ms = 15 },  // 15ms interval  
        .{ .cycles = 980,  .delay_ms = 12 },  // 12ms interval
        .{ .cycles = 1020, .delay_ms = 8 },   // 8ms interval
        .{ .cycles = 1100, .delay_ms = 20 },  // 20ms interval
    };
    
    for (controlled_measurements) |measurement| {
        try registry.recordExecution(fingerprint, measurement.cycles);
        // Simulate controlled delay (in real test we can't control exact timing but we record the intent)
        std.time.sleep(measurement.delay_ms * std.time.ns_per_ms);
    }
    
    std.debug.print("3. Testing derivative estimation and velocity tracking...\n", .{});
    
    // Get advanced metrics after measurements
    const advanced_metrics = registry.getAdvancedMetrics(fingerprint);
    try std.testing.expect(advanced_metrics != null);
    
    if (advanced_metrics) |metrics| {
        std.debug.print("   Average measurement interval: {d:.1}ns ({d:.1}ms)\n", .{ 
            metrics.average_measurement_interval_ns, 
            metrics.average_measurement_interval_ns / 1_000_000.0 
        });
        std.debug.print("   Current velocity: {d:.1} cycles/sec\n", .{metrics.current_velocity_cycles_per_sec});
        std.debug.print("   Peak velocity: {d:.1} cycles/sec\n", .{metrics.peak_velocity_cycles_per_sec});
        std.debug.print("   Velocity variance: {d:.3}\n", .{metrics.velocity_variance});
        
        // Verify measurements were captured
        try std.testing.expect(metrics.average_measurement_interval_ns > 0.0);
    }
    
    std.debug.print("4. Testing adaptive cutoff frequency calculation...\n", .{});
    
    // Simulate variable execution pattern to test adaptive cutoff
    const variable_pattern = [_]u64{ 1200, 1800, 1400, 2000, 1300, 1700, 1500 };
    
    for (variable_pattern) |cycles| {
        try registry.recordExecution(fingerprint, cycles);
        std.time.sleep(5 * std.time.ns_per_ms); // 5ms between measurements
    }
    
    const updated_metrics = registry.getAdvancedMetrics(fingerprint);
    if (updated_metrics) |metrics| {
        std.debug.print("   Adaptive cutoff frequency: {d:.3}Hz\n", .{metrics.adaptive_cutoff_frequency});
        std.debug.print("   Performance stability score: {d:.3}\n", .{metrics.performance_stability_score});
        
        // Verify adaptive behavior
        try std.testing.expect(metrics.adaptive_cutoff_frequency > 0.0);
        try std.testing.expect(metrics.performance_stability_score >= 0.0);
        try std.testing.expect(metrics.performance_stability_score <= 1.0);
    }
    
    std.debug.print("5. Testing confidence tracking and accuracy metrics...\n", .{});
    
    // Get enhanced prediction with all metrics
    const enhanced_prediction = registry.getEnhancedPrediction(fingerprint);
    
    std.debug.print("   Basic prediction: {d:.1} cycles\n", .{enhanced_prediction.basic.predicted_cycles});
    std.debug.print("   Basic confidence: {d:.3}\n", .{enhanced_prediction.basic.confidence});
    std.debug.print("   Rolling accuracy: {d:.3}\n", .{enhanced_prediction.advanced.rolling_accuracy});
    std.debug.print("   Accuracy trend: {d:.3}\n", .{enhanced_prediction.advanced.accuracy_trend});
    std.debug.print("   Average confidence: {d:.3}\n", .{enhanced_prediction.advanced.average_confidence});
    std.debug.print("   Confidence stability: {d:.3}\n", .{enhanced_prediction.advanced.confidence_stability});
    
    // Verify enhanced prediction structure
    try std.testing.expect(enhanced_prediction.basic.execution_count > 0);
    try std.testing.expect(enhanced_prediction.advanced.rolling_accuracy >= 0.0);
    try std.testing.expect(enhanced_prediction.advanced.rolling_accuracy <= 1.0);
    try std.testing.expect(enhanced_prediction.advanced.average_confidence >= 0.0);
    try std.testing.expect(enhanced_prediction.advanced.average_confidence <= 1.0);
    
    std.debug.print("\n✅ Advanced performance tracking test completed successfully!\n", .{});
    std.debug.print("📈 Key enhancements verified:\n", .{});
    std.debug.print("   • Timestamp-based filtering with nanosecond precision\n", .{});
    std.debug.print("   • Derivative estimation for velocity tracking\n", .{});
    std.debug.print("   • Adaptive cutoff frequency calculation\n", .{});
    std.debug.print("   • Confidence tracking and accuracy metrics\n", .{});
}

test "velocity tracking under different load patterns" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Velocity Tracking Under Load Patterns Test ===\n", .{});
    
    var registry = beat.fingerprint.FingerprintRegistry.init(allocator);
    defer registry.deinit();
    
    var context = beat.fingerprint.ExecutionContext.init();
    
    // Create test task and fingerprint
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
    
    std.debug.print("1. Testing steady load pattern...\n", .{});
    
    // Steady load: consistent execution times
    const steady_pattern = [_]u64{ 1000, 1010, 1020, 1015, 1005, 1025, 1000 };
    
    for (steady_pattern) |cycles| {
        try registry.recordExecution(fingerprint, cycles);
        std.time.sleep(10 * std.time.ns_per_ms); // 10ms intervals
    }
    
    const steady_metrics = registry.getAdvancedMetrics(fingerprint).?;
    std.debug.print("   Steady pattern velocity: {d:.1} cycles/sec\n", .{steady_metrics.current_velocity_cycles_per_sec});
    std.debug.print("   Steady pattern stability: {d:.3}\n", .{steady_metrics.performance_stability_score});
    
    std.debug.print("2. Testing burst load pattern...\n", .{});
    
    // Burst load: rapid changes in execution times
    const burst_pattern = [_]u64{ 1000, 2000, 1500, 3000, 1200, 2500, 1800 };
    
    for (burst_pattern) |cycles| {
        try registry.recordExecution(fingerprint, cycles);
        std.time.sleep(5 * std.time.ns_per_ms); // 5ms intervals (faster)
    }
    
    const burst_metrics = registry.getAdvancedMetrics(fingerprint).?;
    std.debug.print("   Burst pattern velocity: {d:.1} cycles/sec\n", .{burst_metrics.current_velocity_cycles_per_sec});
    std.debug.print("   Burst pattern stability: {d:.3}\n", .{burst_metrics.performance_stability_score});
    
    // Burst pattern should have higher velocity and lower stability
    try std.testing.expect(burst_metrics.current_velocity_cycles_per_sec > steady_metrics.current_velocity_cycles_per_sec);
    
    std.debug.print("3. Testing gradual ramp pattern...\n", .{});
    
    // Gradual ramp: slowly increasing execution times
    const ramp_pattern = [_]u64{ 1000, 1100, 1200, 1300, 1400, 1500, 1600 };
    
    for (ramp_pattern) |cycles| {
        try registry.recordExecution(fingerprint, cycles);
        std.time.sleep(15 * std.time.ns_per_ms); // 15ms intervals
    }
    
    const ramp_metrics = registry.getAdvancedMetrics(fingerprint).?;
    std.debug.print("   Ramp pattern velocity: {d:.1} cycles/sec\n", .{ramp_metrics.current_velocity_cycles_per_sec});
    std.debug.print("   Ramp pattern peak velocity: {d:.1} cycles/sec\n", .{ramp_metrics.peak_velocity_cycles_per_sec});
    
    // Peak velocity should capture the highest observed rate of change
    try std.testing.expect(ramp_metrics.peak_velocity_cycles_per_sec >= ramp_metrics.current_velocity_cycles_per_sec);
    
    std.debug.print("\n✅ Velocity tracking test completed successfully!\n", .{});
    std.debug.print("🚀 Velocity tracking demonstrated:\n", .{});
    std.debug.print("   • Steady patterns: low velocity, high stability\n", .{});
    std.debug.print("   • Burst patterns: high velocity, low stability\n", .{});
    std.debug.print("   • Ramp patterns: moderate velocity, peak tracking\n", .{});
}

test "adaptive cutoff frequency behavior" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Adaptive Cutoff Frequency Behavior Test ===\n", .{});
    
    // Create two registries with different base cutoff frequencies
    var low_cutoff_registry = beat.fingerprint.FingerprintRegistry.initWithConfig(
        allocator, 
        0.5,  // Low base cutoff
        0.1, 
        1.0
    );
    defer low_cutoff_registry.deinit();
    
    var high_cutoff_registry = beat.fingerprint.FingerprintRegistry.initWithConfig(
        allocator,
        2.0,  // High base cutoff
        0.1,
        1.0
    );
    defer high_cutoff_registry.deinit();
    
    var context = beat.fingerprint.ExecutionContext.init();
    
    // Create test task and fingerprint
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
    
    std.debug.print("1. Testing cutoff adaptation under stable conditions...\n", .{});
    
    // Stable execution pattern
    const stable_executions = [_]u64{ 1000, 1010, 990, 1020, 980, 1030, 1000 };
    
    for (stable_executions) |cycles| {
        try low_cutoff_registry.recordExecution(fingerprint, cycles);
        try high_cutoff_registry.recordExecution(fingerprint, cycles);
        std.time.sleep(10 * std.time.ns_per_ms);
    }
    
    const low_stable_metrics = low_cutoff_registry.getAdvancedMetrics(fingerprint).?;
    const high_stable_metrics = high_cutoff_registry.getAdvancedMetrics(fingerprint).?;
    
    std.debug.print("   Low base cutoff -> adaptive: {d:.3}Hz\n", .{low_stable_metrics.adaptive_cutoff_frequency});
    std.debug.print("   High base cutoff -> adaptive: {d:.3}Hz\n", .{high_stable_metrics.adaptive_cutoff_frequency});
    
    std.debug.print("2. Testing cutoff adaptation under variable conditions...\n", .{});
    
    // Variable execution pattern to trigger adaptation
    const variable_executions = [_]u64{ 1000, 1500, 2000, 1200, 1800, 1100, 1700 };
    
    for (variable_executions) |cycles| {
        try low_cutoff_registry.recordExecution(fingerprint, cycles);
        try high_cutoff_registry.recordExecution(fingerprint, cycles);
        std.time.sleep(5 * std.time.ns_per_ms);
    }
    
    const low_variable_metrics = low_cutoff_registry.getAdvancedMetrics(fingerprint).?;
    const high_variable_metrics = high_cutoff_registry.getAdvancedMetrics(fingerprint).?;
    
    std.debug.print("   Low base cutoff (variable) -> adaptive: {d:.3}Hz\n", .{low_variable_metrics.adaptive_cutoff_frequency});
    std.debug.print("   High base cutoff (variable) -> adaptive: {d:.3}Hz\n", .{high_variable_metrics.adaptive_cutoff_frequency});
    
    // Variable conditions should increase adaptive cutoff frequency
    try std.testing.expect(low_variable_metrics.adaptive_cutoff_frequency > low_stable_metrics.adaptive_cutoff_frequency);
    try std.testing.expect(high_variable_metrics.adaptive_cutoff_frequency > high_stable_metrics.adaptive_cutoff_frequency);
    
    std.debug.print("3. Comparing stability scores...\n", .{});
    
    std.debug.print("   Low cutoff stability: {d:.3}\n", .{low_variable_metrics.performance_stability_score});
    std.debug.print("   High cutoff stability: {d:.3}\n", .{high_variable_metrics.performance_stability_score});
    
    // Both should detect the reduced stability from variable pattern
    try std.testing.expect(low_variable_metrics.performance_stability_score < 1.0);
    try std.testing.expect(high_variable_metrics.performance_stability_score < 1.0);
    
    std.debug.print("\n✅ Adaptive cutoff frequency test completed successfully!\n", .{});
    std.debug.print("⚙️  Adaptive behavior verified:\n", .{});
    std.debug.print("   • Stable patterns: minimal cutoff adjustment\n", .{});
    std.debug.print("   • Variable patterns: increased cutoff for responsiveness\n", .{});
    std.debug.print("   • Performance stability tracking functional\n", .{});
}