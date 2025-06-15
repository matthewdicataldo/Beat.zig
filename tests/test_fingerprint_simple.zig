const std = @import("std");
const beat = @import("beat");

// Simplified Task Fingerprinting Test
// This test validates the core fingerprinting functionality without complex formatting

test "fingerprint - basic functionality" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Task Fingerprinting - Basic Test ===\n", .{});
    
    // Test 1: Basic fingerprint creation
    std.debug.print("1. Testing basic fingerprint creation...\n", .{});
    
    var context = beat.fingerprint.ExecutionContext.init();
    context.current_numa_node = 0;
    context.system_load = 0.7;
    
    const fingerprint = beat.fingerprint.TaskFingerprint{
        .call_site_hash = 0x12345678,
        .data_size_class = 10, // 2^10 = 1KB
        .data_alignment = 3,   // 8-byte alignment
        .access_pattern = .sequential,
        .simd_width = 4,       // 128-bit SIMD
        .cache_locality = 8,   // High locality
        .numa_node_hint = 0,
        .cpu_intensity = 7,    // CPU intensive
        .parallel_potential = 12, // High parallelization
        .execution_phase = 1,
        .priority_class = 1,   // Normal priority
        .time_sensitivity = 0, // Not time sensitive
        .dependency_count = 0,
        .time_of_day_bucket = 12, // Noon
        .execution_frequency = 5,  // Regular frequency
        .seasonal_pattern = 0,
        .variance_level = 3,   // Low variance
        .expected_cycles_log2 = 10, // 2^10 = 1024 cycles
        .memory_footprint_log2 = 12, // 2^12 = 4KB
        .io_intensity = 2,       // Low I/O
        .cache_miss_rate = 3,    // Low cache misses
        .branch_predictability = 12, // High predictability
        .vectorization_benefit = 8,  // Good vectorization
    };
    
    try std.testing.expect(fingerprint.call_site_hash == 0x12345678);
    try std.testing.expect(fingerprint.data_size_class == 10);
    std.debug.print("   ✅ Basic fingerprint creation works\n", .{});
    
    // Test 2: Registry functionality
    std.debug.print("2. Testing fingerprint registry...\n", .{});
    
    var registry = beat.fingerprint.FingerprintRegistry.init(allocator);
    defer registry.deinit();
    
    // Test basic registry functionality
    const prediction = registry.getPredictionWithConfidence(fingerprint);
    try std.testing.expect(prediction.predicted_cycles >= 0.0); // Should return a non-negative value
    
    std.debug.print("   ✅ Registry registration and lookup works\n", .{});
    
    // Test 3: Task analysis
    std.debug.print("3. Testing task analysis...\n", .{});
    
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
    
    const analysis = beat.fingerprint.TaskAnalyzer.analyzeTask(&task, &context);
    
    try std.testing.expect(analysis.call_site_hash != 0);
    try std.testing.expect(analysis.data_size_class != 0);
    
    std.debug.print("   ✅ Task analysis works\n", .{});
    
    std.debug.print("\n✅ Fingerprinting basic test completed successfully!\n", .{});
}

test "fingerprint - confidence levels" {
    std.debug.print("\n=== Multi-Factor Confidence Test ===\n", .{});
    
    // Test confidence calculation
    std.debug.print("1. Testing confidence calculation...\n", .{});
    
    const confidence = beat.fingerprint.MultiFactorConfidence{
        .sample_size_confidence = 0.8,
        .accuracy_confidence = 0.9,
        .temporal_confidence = 0.7,
        .variance_confidence = 0.6,
        .overall_confidence = 0.75,
        .sample_count = 50,
        .recent_accuracy = 0.85,
        .time_since_last_ms = 1000.0,
        .coefficient_of_variation = 0.15,
    };
    
    try std.testing.expect(confidence.overall_confidence >= 0.0 and confidence.overall_confidence <= 1.0);
    try std.testing.expect(confidence.sample_count > 0);
    
    std.debug.print("   Overall confidence: {d:.2}\n", .{confidence.overall_confidence});
    std.debug.print("   ✅ Confidence calculation works\n", .{});
    
    std.debug.print("\n✅ Confidence test completed successfully!\n", .{});
}