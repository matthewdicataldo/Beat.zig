const std = @import("std");
const beat = @import("beat");
const testing = std.testing;

// ============================================================================
// SIMD-Enhanced Continuation Stealing Integration Tests
// ============================================================================

test "SIMD continuation classifier initialization and basic functionality" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Create test configuration with SIMD features enabled
    const config = beat.Config{
        .num_workers = 4,
        .enable_work_stealing = true,
        .enable_lock_free = true,
        .enable_topology_aware = true,
        .enable_numa_aware = true,
        .enable_predictive = true,
        .enable_statistics = true,
    };
    
    // Initialize ThreadPool with SIMD continuation support
    var pool = try beat.ThreadPool.init(allocator, config);
    defer pool.deinit();
    
    // Verify SIMD classifier was initialized
    try testing.expect(pool.continuation_simd_classifier != null);
    
    // Test SIMD classifier functionality
    const classifier = pool.continuation_simd_classifier.?;
    
    // Create test continuation
    const TestData = struct { values: [32]f32 };
    var test_data = TestData{ .values = undefined };
    for (&test_data.values, 0..) |*value, i| {
        value.* = @floatFromInt(i);
    }
    
    const resume_fn = struct {
        fn executeFunc(cont: *beat.continuation.Continuation) void {
            const data = @as(*TestData, @ptrCast(@alignCast(cont.data)));
            for (&data.values) |*value| {
                value.* = value.* * 2.0 + 1.0; // SIMD-friendly operation
            }
            cont.state = .completed;
        }
    };
    
    var test_continuation = beat.continuation.Continuation.capture(resume_fn.executeFunc, &test_data, allocator);
    test_continuation.frame_size = 128; // Reasonable frame size
    
    // Test classification
    const classification = try classifier.classifyContinuation(&test_continuation);
    
    // Verify classification results
    try testing.expect(classification.simd_suitability_score >= 0.0);
    try testing.expect(classification.simd_suitability_score <= 1.0);
    try testing.expect(classification.continuation_overhead_factor >= 1.0);
    try testing.expect(classification.vectorization_potential > 0.0);
    
    std.debug.print("✅ SIMD continuation classifier initialization test passed!\n", .{});
    std.debug.print("   SIMD suitability: {d:.3}\n", .{classification.simd_suitability_score});
    std.debug.print("   Vectorization potential: {d:.2}x\n", .{classification.vectorization_potential});
}

test "continuation submission with SIMD classification" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const config = beat.Config{
        .num_workers = 2,
        .enable_work_stealing = true,
        .enable_lock_free = true,
        .enable_statistics = true,
    };
    
    var pool = try beat.ThreadPool.init(allocator, config);
    defer pool.deinit();
    
    // Test data for SIMD-friendly operations
    const TestData = struct { 
        values: [16]f32,
        completed: std.atomic.Value(bool),
    };
    var test_data = TestData{ 
        .values = undefined,
        .completed = std.atomic.Value(bool).init(false),
    };
    
    for (&test_data.values, 0..) |*value, i| {
        value.* = @floatFromInt(i);
    }
    
    const resume_fn = struct {
        fn executeFunc(cont: *beat.continuation.Continuation) void {
            const data = @as(*TestData, @ptrCast(@alignCast(cont.data)));
            
            // Perform SIMD-friendly operations
            for (&data.values) |*value| {
                value.* = value.* * 1.5 + 0.5;
            }
            
            cont.state = .completed;
            data.completed.store(true, .release);
        }
    };
    
    var test_continuation = beat.continuation.Continuation.capture(resume_fn.executeFunc, &test_data, allocator);
    
    // Submit continuation (should trigger SIMD classification)
    try pool.submitContinuation(&test_continuation);
    
    // Wait for completion
    pool.wait();
    
    // Verify completion
    try testing.expect(test_data.completed.load(.acquire));
    try testing.expect(test_continuation.state == .completed);
    
    // Verify SIMD classification was applied (fingerprint_hash should be set)
    try testing.expect(test_continuation.fingerprint_hash != null);
    
    // Check classifier statistics
    if (pool.continuation_simd_classifier) |classifier| {
        const stats = classifier.getPerformanceStats();
        try testing.expect(stats.classifications_performed >= 1);
        
        std.debug.print("✅ Continuation submission with SIMD classification test passed!\n", .{});
        std.debug.print("   Classifications performed: {}\n", .{stats.classifications_performed});
        std.debug.print("   Cache hit rate: {d:.1}%\n", .{stats.cache_hit_rate * 100});
        std.debug.print("   SIMD hit rate: {d:.1}%\n", .{stats.simd_hit_rate * 100});
    }
}

test "SIMD continuation batch formation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const config = beat.Config{
        .num_workers = 2,
        .enable_work_stealing = true,
        .enable_lock_free = true,
        .enable_statistics = true,
    };
    
    var pool = try beat.ThreadPool.init(allocator, config);
    defer pool.deinit();
    
    const classifier = pool.continuation_simd_classifier.?;
    
    // Create multiple similar continuations for batching
    const TestData = struct { 
        values: [8]f32,
        id: u32,
        completed: std.atomic.Value(bool),
    };
    var test_data_array: [6]TestData = undefined;
    var continuations: [6]beat.continuation.Continuation = undefined;
    
    const resume_fn = struct {
        fn executeFunc(cont: *beat.continuation.Continuation) void {
            const data = @as(*TestData, @ptrCast(@alignCast(cont.data)));
            
            // Similar SIMD-friendly operations that should batch well
            for (&data.values) |*value| {
                value.* = value.* * 1.2 + 0.3;
            }
            
            cont.state = .completed;
            data.completed.store(true, .release);
        }
    };
    
    // Initialize test data and continuations
    for (&test_data_array, 0..) |*data, i| {
        data.* = TestData{
            .values = undefined,
            .id = @intCast(i),
            .completed = std.atomic.Value(bool).init(false),
        };
        
        for (&data.values, 0..) |*value, j| {
            value.* = @as(f32, @floatFromInt(i * 8 + j));
        }
        
        continuations[i] = beat.continuation.Continuation.capture(resume_fn.executeFunc, data, allocator);
        continuations[i].frame_size = 64; // Similar frame sizes for batching
    }
    
    // Submit all continuations
    for (&continuations) |*cont| {
        try pool.submitContinuation(cont);
    }
    
    // Force batch formation
    _ = try classifier.formContinuationBatches();
    
    // Wait for all to complete
    pool.wait();
    
    // Verify all completed
    for (&test_data_array) |*data| {
        try testing.expect(data.completed.load(.acquire));
    }
    
    // Check batch formation statistics
    const stats = classifier.getPerformanceStats();
    try testing.expect(stats.classifications_performed >= 6);
    
    const batch_stats = stats.batch_formation_stats;
    
    std.debug.print("✅ SIMD continuation batch formation test passed!\n", .{});
    std.debug.print("   Total tasks submitted: {}\n", .{batch_stats.total_tasks_submitted});
    std.debug.print("   Tasks in batches: {}\n", .{batch_stats.tasks_in_batches});
    std.debug.print("   Formation efficiency: {d:.1}%\n", .{batch_stats.formation_efficiency * 100});
    std.debug.print("   Average estimated speedup: {d:.2}x\n", .{batch_stats.average_estimated_speedup});
}

test "continuation SIMD performance comparison" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Configuration for performance testing
    const config = beat.Config{
        .num_workers = 4,
        .enable_work_stealing = true,
        .enable_lock_free = true,
        .enable_topology_aware = true,
        .enable_numa_aware = true,
        .enable_statistics = true,
    };
    
    var pool = try beat.ThreadPool.init(allocator, config);
    defer pool.deinit();
    
    const num_continuations = 32;
    
    // Test data for performance comparison
    const TestData = struct { 
        values: [64]f32,
        result: f32,
        completed: std.atomic.Value(bool),
    };
    var test_data_array: [num_continuations]TestData = undefined;
    var continuations: [num_continuations]beat.continuation.Continuation = undefined;
    
    const resume_fn = struct {
        fn executeFunc(cont: *beat.continuation.Continuation) void {
            const data = @as(*TestData, @ptrCast(@alignCast(cont.data)));
            
            // SIMD-optimizable computation
            var sum: f32 = 0.0;
            for (&data.values) |*value| {
                value.* = value.* * 2.0 + 1.0;
                sum += value.*;
            }
            data.result = sum;
            
            cont.state = .completed;
            data.completed.store(true, .release);
        }
    };
    
    // Initialize test data
    for (&test_data_array, 0..) |*data, i| {
        data.* = TestData{
            .values = undefined,
            .result = 0.0,
            .completed = std.atomic.Value(bool).init(false),
        };
        
        for (&data.values, 0..) |*value, j| {
            value.* = @as(f32, @floatFromInt(i * 64 + j));
        }
        
        continuations[i] = beat.continuation.Continuation.capture(resume_fn.executeFunc, data, allocator);
    }
    
    // Measure execution time
    const start_time = std.time.nanoTimestamp();
    
    // Submit all continuations
    for (&continuations) |*cont| {
        try pool.submitContinuation(cont);
    }
    
    // Wait for completion
    pool.wait();
    
    const end_time = std.time.nanoTimestamp();
    const execution_time = @as(u64, @intCast(end_time - start_time));
    
    // Verify all completed
    for (&test_data_array) |*data| {
        try testing.expect(data.completed.load(.acquire));
        try testing.expect(data.result > 0.0); // Should have computed something
    }
    
    // Check performance statistics
    const classifier = pool.continuation_simd_classifier.?;
    const stats = classifier.getPerformanceStats();
    
    const avg_time_per_continuation = @as(f64, @floatFromInt(execution_time)) / num_continuations;
    const throughput = 1_000_000_000.0 / avg_time_per_continuation; // continuations per second
    
    std.debug.print("✅ Continuation SIMD performance test passed!\n", .{});
    std.debug.print("   Total execution time: {d:.2}ms\n", .{@as(f64, @floatFromInt(execution_time)) / 1_000_000});
    std.debug.print("   Average time per continuation: {d:.1}μs\n", .{avg_time_per_continuation / 1000});
    std.debug.print("   Throughput: {d:.0} continuations/sec\n", .{throughput});
    std.debug.print("   Classifications performed: {}\n", .{stats.classifications_performed});
    std.debug.print("   SIMD hit rate: {d:.1}%\n", .{stats.simd_hit_rate * 100});
    
    // Performance should be reasonable
    try testing.expect(avg_time_per_continuation < 100_000); // Less than 100μs per continuation
    try testing.expect(stats.classifications_performed >= num_continuations);
}

test "continuation SIMD integration with NUMA awareness" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const config = beat.Config{
        .num_workers = 4,
        .enable_work_stealing = true,
        .enable_lock_free = true,
        .enable_topology_aware = true,
        .enable_numa_aware = true,
        .enable_statistics = true,
    };
    
    var pool = try beat.ThreadPool.init(allocator, config);
    defer pool.deinit();
    
    // Test NUMA-aware SIMD continuation processing
    const TestData = struct { 
        values: [32]f32,
        numa_node: ?u32,
        completed: std.atomic.Value(bool),
    };
    var test_data = TestData{
        .values = undefined,
        .numa_node = null,
        .completed = std.atomic.Value(bool).init(false),
    };
    
    for (&test_data.values, 0..) |*value, i| {
        value.* = @floatFromInt(i);
    }
    
    const resume_fn = struct {
        fn executeFunc(cont: *beat.continuation.Continuation) void {
            const data = @as(*TestData, @ptrCast(@alignCast(cont.data)));
            
            // Store NUMA node information
            data.numa_node = cont.numa_node;
            
            // SIMD-friendly computation
            for (&data.values) |*value| {
                value.* = @sqrt(value.* * value.* + 1.0);
            }
            
            cont.state = .completed;
            data.completed.store(true, .release);
        }
    };
    
    var test_continuation = beat.continuation.Continuation.capture(resume_fn.executeFunc, &test_data, allocator);
    
    // Submit continuation
    try pool.submitContinuation(&test_continuation);
    
    // Wait for completion
    pool.wait();
    
    // Verify completion and NUMA awareness
    try testing.expect(test_data.completed.load(.acquire));
    try testing.expect(test_continuation.state == .completed);
    
    // Check that NUMA locality was set
    if (pool.topology != null) {
        try testing.expect(test_continuation.numa_node != null);
        try testing.expect(test_continuation.locality_score > 0.0);
        
        std.debug.print("✅ SIMD continuation NUMA integration test passed!\n", .{});
        std.debug.print("   NUMA node: {?}\n", .{test_continuation.numa_node});
        std.debug.print("   Locality score: {d:.3}\n", .{test_continuation.locality_score});
        std.debug.print("   Migration count: {}\n", .{test_continuation.migration_count});
    } else {
        std.debug.print("✅ SIMD continuation integration test passed (no NUMA topology detected)!\n", .{});
    }
}

test "continuation SIMD error handling and edge cases" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const config = beat.Config{
        .num_workers = 2,
        .enable_work_stealing = true,
        .enable_lock_free = true,
        .enable_statistics = true,
    };
    
    var pool = try beat.ThreadPool.init(allocator, config);
    defer pool.deinit();
    
    // Test edge case: very small data (should still work)
    const SmallData = struct { value: i32 };
    var small_data = SmallData{ .value = 42 };
    _ = std.atomic.Value(bool).init(false); // Unused in this test
    
    const small_resume_fn = struct {
        fn executeFunc(cont: *beat.continuation.Continuation) void {
            const data = @as(*SmallData, @ptrCast(@alignCast(cont.data)));
            data.value *= 2;
            cont.state = .completed;
        }
    };
    
    var small_continuation = beat.continuation.Continuation.capture(small_resume_fn.executeFunc, &small_data, allocator);
    
    // Submit small continuation
    try pool.submitContinuation(&small_continuation);
    
    // Test edge case: large data (should handle gracefully)
    const LargeData = struct { values: [1024]f64 };
    var large_data = LargeData{ .values = undefined };
    for (&large_data.values, 0..) |*value, i| {
        value.* = @floatFromInt(i);
    }
    
    const large_resume_fn = struct {
        fn executeFunc(cont: *beat.continuation.Continuation) void {
            const data = @as(*LargeData, @ptrCast(@alignCast(cont.data)));
            for (&data.values) |*value| {
                value.* = value.* * 0.5;
            }
            cont.state = .completed;
        }
    };
    
    var large_continuation = beat.continuation.Continuation.capture(large_resume_fn.executeFunc, &large_data, allocator);
    
    // Submit large continuation
    try pool.submitContinuation(&large_continuation);
    
    // Wait for all to complete
    pool.wait();
    
    // Verify both completed successfully
    try testing.expect(small_continuation.state == .completed);
    try testing.expect(large_continuation.state == .completed);
    try testing.expect(small_data.value == 84); // 42 * 2
    
    std.debug.print("✅ SIMD continuation edge cases test passed!\n", .{});
    std.debug.print("   Small data value: {}\n", .{small_data.value});
    std.debug.print("   Large data first value: {d:.3}\n", .{large_data.values[0]});
}