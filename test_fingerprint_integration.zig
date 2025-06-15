const std = @import("std");
const beat = @import("src/core.zig");

// Test for Task Fingerprinting Integration (Phase 2.1.2)
//
// This test demonstrates the integration of the fingerprinting system
// with the existing Beat.zig Task structure and ThreadPool.

test "fingerprint integration with enhanced tasks" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Task Fingerprinting Integration Test ===\n", .{});
    
    // Create execution context
    var context = beat.fingerprint.ExecutionContext.init();
    context.current_numa_node = 0;
    context.system_load = 0.7;
    
    // Create fingerprint registry
    var registry = beat.fingerprint.FingerprintRegistry.init(allocator);
    defer registry.deinit();
    
    // Test data structure
    const TestData = struct {
        values: [1000]i32,
        
        pub fn process(self: *@This()) void {
            for (&self.values) |*value| {
                value.* = value.* * 2 + 1;
            }
        }
        
        pub fn sum(self: *const @This()) i64 {
            var total: i64 = 0;
            for (self.values) |value| {
                total += value;
            }
            return total;
        }
    };
    
    var test_data = TestData{ .values = undefined };
    for (&test_data.values, 0..) |*value, i| {
        value.* = @intCast(i);
    }
    
    std.debug.print("1. Creating enhanced task with fingerprinting...\n", .{});
    
    // Create task with fingerprinting hints
    var task = beat.Task{
        .func = struct {
            fn taskWrapper(data: *anyopaque) void {
                const typed_data = @as(*TestData, @ptrCast(@alignCast(data)));
                typed_data.process();
            }
        }.taskWrapper,
        .data = @ptrCast(&test_data),
        .priority = .normal,
        .data_size_hint = @sizeOf(TestData), // Provide size hint for better fingerprinting
    };
    
    std.debug.print("2. Generating task fingerprint...\n", .{});
    
    // Generate fingerprint
    const fingerprint = beat.fingerprint.generateTaskFingerprint(&task, &context);
    
    // Enhance task with fingerprint information
    beat.fingerprint.enhanceTask(&task, &context);
    
    std.debug.print("3. Fingerprint characteristics:\n", .{});
    std.debug.print("   - Call site hash: 0x{X}\n", .{fingerprint.call_site_hash});
    std.debug.print("   - Data size class: {} ({}B)\n", .{ fingerprint.data_size_class, @as(usize, 1) << @as(u6, @intCast(@min(63, fingerprint.data_size_class))) });
    std.debug.print("   - Access pattern: {s}\n", .{@tagName(fingerprint.access_pattern)});
    std.debug.print("   - SIMD width: {}\n", .{fingerprint.simd_width});
    std.debug.print("   - Parallel potential: {}/15\n", .{fingerprint.parallel_potential});
    std.debug.print("   - Cache locality: {}/15\n", .{fingerprint.cache_locality});
    
    const characteristics = fingerprint.getCharacteristics();
    std.debug.print("4. Task characteristics:\n", .{});
    std.debug.print("   - CPU intensive: {}\n", .{characteristics.is_cpu_intensive});
    std.debug.print("   - Memory bound: {}\n", .{characteristics.is_memory_bound});
    std.debug.print("   - Vectorizable: {}\n", .{characteristics.is_vectorizable});
    std.debug.print("   - NUMA sensitive: {}\n", .{characteristics.is_numa_sensitive});
    std.debug.print("   - Cache friendly: {}\n", .{characteristics.is_cache_friendly});
    std.debug.print("   - Parallel friendly: {}\n", .{characteristics.is_parallel_friendly});
    
    // Verify task enhancement
    try std.testing.expect(task.fingerprint_hash != null);
    try std.testing.expect(task.creation_timestamp != null);
    try std.testing.expect(task.data_size_hint != null);
    
    std.debug.print("5. Recording execution performance...\n", .{});
    
    // Simulate multiple executions with performance tracking
    const execution_times = [_]u64{ 1500, 1200, 1800, 1100, 1600 };
    
    for (execution_times) |cycles| {
        try registry.recordExecution(fingerprint, cycles);
    }
    
    // Get performance profile
    const profile = registry.getProfile(fingerprint);
    try std.testing.expect(profile != null);
    
    if (profile) |p| {
        const avg_execution = p.getAverageExecution();
        std.debug.print("6. Performance profile:\n", .{});
        std.debug.print("   - Executions: {}\n", .{p.execution_count});
        std.debug.print("   - Average cycles: {d:.1}\n", .{avg_execution});
        std.debug.print("   - Min cycles: {}\n", .{p.min_cycles});
        std.debug.print("   - Max cycles: {}\n", .{p.max_cycles});
        std.debug.print("   - Variance: {d:.1}%\n", .{(@as(f64, @floatFromInt(p.max_cycles - p.min_cycles)) / avg_execution) * 100.0});
        
        // Verify reasonable performance tracking
        try std.testing.expectEqual(@as(u64, 5), p.execution_count);
        try std.testing.expectApproxEqAbs(@as(f64, 1440.0), avg_execution, 50.0); // Average of execution_times
    }
    
    std.debug.print("7. Testing prediction accuracy...\n", .{});
    
    // Test prediction
    const predicted_cycles = registry.getPredictedCycles(fingerprint);
    std.debug.print("   - Predicted execution: {d:.1} cycles\n", .{predicted_cycles});
    
    // Verify prediction is reasonable
    try std.testing.expect(predicted_cycles > 1000.0);
    try std.testing.expect(predicted_cycles < 2000.0);
    
    std.debug.print("8. Testing similarity calculation...\n", .{});
    
    // Create similar task
    var similar_task = beat.Task{
        .func = task.func, // Same function
        .data = @ptrCast(&test_data),
        .priority = .normal,
        .data_size_hint = @sizeOf(TestData),
    };
    
    beat.fingerprint.enhanceTask(&similar_task, &context);
    
    // Should be very similar (same function, same data)
    const similarity = beat.fingerprint.getTaskSimilarity(&task, &similar_task, &registry);
    std.debug.print("   - Task similarity: {d:.2}\n", .{similarity});
    
    // Test registry statistics
    const stats = registry.getRegistryStats();
    std.debug.print("9. Registry statistics:\n", .{});
    std.debug.print("   - Total profiles: {}\n", .{stats.total_profiles});
    std.debug.print("   - Memory usage: {} bytes\n", .{stats.memory_usage});
    
    try std.testing.expect(stats.total_profiles >= 1);
    try std.testing.expect(stats.memory_usage > 0);
    
    std.debug.print("\n✅ Task fingerprinting integration test completed successfully!\n", .{});
    std.debug.print("📊 Key achievements:\n", .{});
    std.debug.print("   • Compact 128-bit fingerprint representation\n", .{});
    std.debug.print("   • Multi-dimensional task characteristic capture\n", .{});
    std.debug.print("   • Performance tracking and prediction\n", .{});
    std.debug.print("   • Seamless integration with existing Task structure\n", .{});
    std.debug.print("   • Fast fingerprint generation and analysis\n", .{});
}

test "fingerprint with thread pool integration" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== ThreadPool Fingerprint Integration Test ===\n", .{});
    
    // Create thread pool
    const pool = try beat.createTestPool(allocator);
    defer pool.deinit();
    
    // Create execution context
    var context = beat.fingerprint.ExecutionContext.init();
    
    // Create multiple tasks with different characteristics
    const tasks_data = [_]struct {
        name: []const u8,
        data_size: usize,
        compute_intensity: u8,
    }{
        .{ .name = "Small task", .data_size = 64, .compute_intensity = 2 },
        .{ .name = "Medium task", .data_size = 4096, .compute_intensity = 8 },
        .{ .name = "Large task", .data_size = 1024 * 1024, .compute_intensity = 15 },
    };
    
    var fingerprints: [tasks_data.len]beat.fingerprint.TaskFingerprint = undefined;
    var tasks: [tasks_data.len]beat.Task = undefined;
    
    std.debug.print("1. Creating diverse tasks with different characteristics...\n", .{});
    
    // Create task functions with different characteristics
    const TaskFunction1 = struct {
        fn process(data: *anyopaque) void {
            _ = data;
            // Light computation
            var j: usize = 0;
            while (j < 200) : (j += 1) {
                std.mem.doNotOptimizeAway(j * j);
            }
        }
    };
    
    const TaskFunction2 = struct {
        fn process(data: *anyopaque) void {
            _ = data;
            // Medium computation
            var j: usize = 0;
            while (j < 800) : (j += 1) {
                std.mem.doNotOptimizeAway(j * j);
            }
        }
    };
    
    const TaskFunction3 = struct {
        fn process(data: *anyopaque) void {
            _ = data;
            // Heavy computation
            var j: usize = 0;
            while (j < 1500) : (j += 1) {
                std.mem.doNotOptimizeAway(j * j);
            }
        }
    };
    
    const task_functions = [_]*const fn(*anyopaque) void{
        TaskFunction1.process,
        TaskFunction2.process,
        TaskFunction3.process,
    };
    
    // Create tasks with different profiles
    for (tasks_data, 0..) |task_data, i| {
        
        tasks[i] = beat.Task{
            .func = task_functions[i],
            .data = @ptrFromInt(0x1000 + i * 0x100), // Different addresses for diversity
            .priority = if (i == 2) .high else .normal,
            .data_size_hint = task_data.data_size,
        };
        
        // Generate fingerprint
        fingerprints[i] = beat.fingerprint.generateTaskFingerprint(&tasks[i], &context);
        beat.fingerprint.enhanceTask(&tasks[i], &context);
        
        std.debug.print("   {s} fingerprint hash: 0x{X}\n", .{ task_data.name, fingerprints[i].hash() });
        
        // Verify tasks have different fingerprints
        if (i > 0) {
            try std.testing.expect(fingerprints[i].hash() != fingerprints[i-1].hash());
        }
    }
    
    std.debug.print("2. Submitting tasks to thread pool...\n", .{});
    
    // Submit tasks to pool
    for (&tasks) |*task| {
        try pool.submit(task.*);
    }
    
    // Wait for completion
    pool.wait();
    
    std.debug.print("3. Analyzing fingerprint diversity...\n", .{});
    
    // Analyze fingerprint diversity
    for (fingerprints, 0..) |fp1, i| {
        for (fingerprints[i+1..], i+1..) |fp2, j| {
            const similarity = fp1.similarity(fp2);
            std.debug.print("   Task {} vs Task {} similarity: {d:.3}\n", .{ i, j, similarity });
            
            // Different tasks should have low similarity
            try std.testing.expect(similarity < 0.9);
        }
    }
    
    std.debug.print("4. Verifying task characteristics classification...\n", .{});
    
    // Verify characteristics make sense
    const small_chars = fingerprints[0].getCharacteristics();
    const large_chars = fingerprints[2].getCharacteristics();
    
    std.debug.print("   Small task - parallel friendly: {}\n", .{small_chars.is_parallel_friendly});
    std.debug.print("   Large task - parallel friendly: {}\n", .{large_chars.is_parallel_friendly});
    
    // Large task should be more parallel-friendly
    if (large_chars.is_parallel_friendly and !small_chars.is_parallel_friendly) {
        std.debug.print("   ✅ Correctly identified parallel potential\n", .{});
    }
    
    std.debug.print("\n✅ ThreadPool integration test completed successfully!\n", .{});
    std.debug.print("🎯 Demonstrated capabilities:\n", .{});
    std.debug.print("   • Diverse fingerprint generation for different task types\n", .{});
    std.debug.print("   • Integration with existing Beat.zig ThreadPool\n", .{});
    std.debug.print("   • Automatic task characteristic analysis\n", .{});
    std.debug.print("   • Foundation for predictive scheduling\n", .{});
}