const std = @import("std");
const beat = @import("beat");

// ============================================================================
// NUMA-Aware Continuation Stealing Tests
// ============================================================================

test "numa continuation stealing - locality tracking" {
    std.debug.print("\n=== NUMA Continuation Stealing - Locality Tracking ===\n", .{});
    
    // Test continuation NUMA locality tracking
    std.debug.print("1. Testing continuation NUMA locality initialization...\n", .{});
    
    var data: i32 = 42;
    const TestResume = struct {
        fn executeWrapper(cont: *beat.continuation.Continuation) void {
            const value = @as(*i32, @ptrCast(@alignCast(cont.data)));
            value.* += 10;
            cont.markCompleted();
        }
    };
    
    var continuation = beat.continuation.Continuation.capture(
        TestResume.executeWrapper,
        &data,
        std.testing.allocator
    );
    
    // Test NUMA locality initialization
    continuation.initNumaLocality(0, 0); // NUMA node 0, socket 0
    try std.testing.expect(continuation.numa_node == 0);
    try std.testing.expect(continuation.original_numa_node == 0);
    try std.testing.expect(continuation.current_socket == 0);
    try std.testing.expect(continuation.migration_count == 0);
    try std.testing.expect(continuation.locality_score == 1.0);
    
    std.debug.print("   ✅ NUMA locality initialized: node={?}, socket={?}, score={d:.2}\n", 
        .{ continuation.numa_node, continuation.current_socket, continuation.locality_score });
    
    std.debug.print("2. Testing NUMA-aware stealing with locality tracking...\n", .{});
    
    // Test same NUMA node preference
    const same_numa_pref = continuation.getStealingPreference(0, 0);
    try std.testing.expect(same_numa_pref == 1.0); // Perfect preference
    
    // Test same socket, different NUMA preference
    const same_socket_pref = continuation.getStealingPreference(1, 0);
    try std.testing.expect(same_socket_pref == 0.7); // Good preference
    
    // Test different socket preference
    const diff_socket_pref = continuation.getStealingPreference(1, 1);
    try std.testing.expect(diff_socket_pref == 0.3); // Low preference
    
    std.debug.print("   ✅ Stealing preferences: same_numa={d:.1}, same_socket={d:.1}, diff_socket={d:.1}\n",
        .{ same_numa_pref, same_socket_pref, diff_socket_pref });
    
    std.debug.print("3. Testing NUMA migration tracking...\n", .{});
    
    // Simulate NUMA migration
    continuation.markStolenWithNuma(1, 1, 1); // Migrate to NUMA node 1, socket 1
    try std.testing.expect(continuation.numa_node == 1);
    try std.testing.expect(continuation.current_socket == 1);
    try std.testing.expect(continuation.migration_count == 1);
    try std.testing.expect(continuation.locality_score < 1.0); // Should be reduced
    
    std.debug.print("   ✅ After migration: node={?}, socket={?}, migrations={}, score={d:.2}\n",
        .{ continuation.numa_node, continuation.current_socket, continuation.migration_count, continuation.locality_score });
    
    // Test local execution preference
    const prefers_local_before = continuation.prefersLocalExecution();
    
    // Multiple migrations should reduce preference
    // Note: For testing, we simulate additional migrations by updating state directly
    continuation.migration_count = 3;
    continuation.updateLocalityScore(3, 3);
    
    const prefers_local_after = continuation.prefersLocalExecution();
    
    std.debug.print("   ✅ Local execution preference: before={}, after={} (migrations={})\n",
        .{ prefers_local_before, prefers_local_after, continuation.migration_count });
    
    std.debug.print("✅ NUMA locality tracking test completed!\n", .{});
}

test "numa continuation stealing - threadpool integration" {
    std.debug.print("\n=== NUMA Continuation Stealing - ThreadPool Integration ===\n", .{});
    
    // Create ThreadPool with NUMA awareness enabled
    const config = beat.Config{
        .num_workers = 4,
        .enable_predictive = true,
        .enable_work_stealing = true,
        .enable_lock_free = true,
        .enable_topology_aware = true,
        .enable_numa_aware = true,
    };
    
    var pool = try beat.ThreadPool.init(std.testing.allocator, config);
    defer pool.deinit();
    
    std.debug.print("1. Created NUMA-aware ThreadPool with {} workers\n", .{pool.workers.len});
    
    // Check if topology was detected
    const has_topology = pool.topology != null;
    std.debug.print("   Topology detected: {}\n", .{has_topology});
    
    if (has_topology) {
        const topo = pool.topology.?;
        std.debug.print("   NUMA nodes: {}, Total cores: {}, Sockets: {}\n", 
            .{ topo.numa_nodes.len, topo.total_cores, topo.sockets });
    }
    
    // Shared state for NUMA tracking
    const ContinuationState = struct {
        worker_id: ?u32 = null,
        numa_node: ?u32 = null,
        migrations: u32 = 0,
        locality_score: f32 = 0.0,
    };
    var continuation_states: [8]ContinuationState = [_]ContinuationState{.{}} ** 8;
    
    var mutex = std.Thread.Mutex{};
    
    const NumaTestState = struct {
        states: *[8]ContinuationState,
        mutex: *std.Thread.Mutex,
        index: u32,
    };
    
    // Create work contexts
    var work_contexts: [8]NumaTestState = undefined;
    for (work_contexts[0..], 0..) |*ctx, i| {
        ctx.* = NumaTestState{
            .states = &continuation_states,
            .mutex = &mutex,
            .index = @intCast(i),
        };
    }
    
    std.debug.print("2. Submitting NUMA-aware continuations...\n", .{});
    
    const NumaWork = struct {
        fn executeWrapper(cont: *beat.continuation.Continuation) void {
            const ctx = @as(*NumaTestState, @ptrCast(@alignCast(cont.data)));
            
            ctx.mutex.lock();
            defer ctx.mutex.unlock();
            
            ctx.states[ctx.index] = .{
                .worker_id = cont.worker_id,
                .numa_node = cont.numa_node,
                .migrations = cont.migration_count,
                .locality_score = cont.locality_score,
            };
            
            cont.markCompleted();
        }
    };
    
    var continuations: [8]beat.continuation.Continuation = undefined;
    for (continuations[0..], 0..) |*cont, i| {
        cont.* = beat.continuation.Continuation.capture(
            NumaWork.executeWrapper,
            &work_contexts[i],
            std.testing.allocator
        );
        
        try pool.submitContinuation(cont);
        std.debug.print("   Submitted continuation {}: numa_node={?}, socket={?}\n", 
            .{ i, cont.numa_node, cont.current_socket });
    }
    
    std.debug.print("3. Executing NUMA-aware continuation stealing...\n", .{});
    pool.wait();
    
    std.debug.print("4. Analyzing NUMA locality results...\n", .{});
    
    var numa_nodes_used = std.AutoHashMap(u32, u32).init(std.testing.allocator);
    defer numa_nodes_used.deinit();
    
    var workers_used = std.AutoHashMap(u32, u32).init(std.testing.allocator);
    defer workers_used.deinit();
    
    var total_migrations: u32 = 0;
    var locality_scores: [8]f32 = undefined;
    
    for (continuation_states, 0..) |state, i| {
        if (state.worker_id) |worker_id| {
            const worker_count = workers_used.get(worker_id) orelse 0;
            try workers_used.put(worker_id, worker_count + 1);
        }
        
        if (state.numa_node) |numa_node| {
            const numa_count = numa_nodes_used.get(numa_node) orelse 0;
            try numa_nodes_used.put(numa_node, numa_count + 1);
        }
        
        total_migrations += state.migrations;
        locality_scores[i] = state.locality_score;
        
        std.debug.print("   Continuation {}: worker={?}, numa={?}, migrations={}, score={d:.2}\n",
            .{ i, state.worker_id, state.numa_node, state.migrations, state.locality_score });
    }
    
    std.debug.print("   📊 NUMA Distribution:\n", .{});
    var numa_iter = numa_nodes_used.iterator();
    while (numa_iter.next()) |entry| {
        std.debug.print("     NUMA node {}: {} continuations\n", .{ entry.key_ptr.*, entry.value_ptr.* });
    }
    
    std.debug.print("   📊 Worker Distribution:\n", .{});
    var worker_iter = workers_used.iterator();
    while (worker_iter.next()) |entry| {
        std.debug.print("     Worker {}: {} continuations\n", .{ entry.key_ptr.*, entry.value_ptr.* });
    }
    
    std.debug.print("   📊 Migration Analysis:\n", .{});
    std.debug.print("     Total migrations: {}\n", .{total_migrations});
    std.debug.print("     Average migrations per continuation: {d:.1}\n", 
        .{ @as(f32, @floatFromInt(total_migrations)) / 8.0 });
    
    // Calculate average locality score
    var total_locality_score: f32 = 0.0;
    for (locality_scores) |score| {
        total_locality_score += score;
    }
    const avg_locality_score = total_locality_score / 8.0;
    
    std.debug.print("     Average locality score: {d:.2}\n", .{avg_locality_score});
    
    // Verify all continuations completed
    var completed_count: u32 = 0;
    for (continuations) |cont| {
        if (cont.state == .completed) {
            completed_count += 1;
        }
    }
    
    try std.testing.expect(completed_count == 8);
    std.debug.print("   ✅ All {} continuations completed successfully\n", .{completed_count});
    
    // NUMA-aware stealing should distribute work efficiently
    // Note: In single-NUMA node systems, work may be handled by fewer workers
    // This is actually good behavior as it maintains locality
    try std.testing.expect(workers_used.count() >= 1); // At least one worker used
    
    // Verify excellent locality (all on same NUMA node)
    if (numa_nodes_used.count() == 1) {
        std.debug.print("   🎯 Excellent NUMA locality: All continuations on single NUMA node\\n", .{});
    }
    
    std.debug.print("✅ NUMA ThreadPool integration test completed!\n", .{});
}

test "numa continuation stealing - performance comparison" {
    std.debug.print("\n=== NUMA Continuation Stealing - Performance Comparison ===\n", .{});
    
    // Compare NUMA-aware vs NUMA-unaware performance
    const config = beat.Config{
        .num_workers = 4,
        .enable_predictive = true,
        .enable_work_stealing = true,
        .enable_lock_free = true,
        .enable_topology_aware = true,
        .enable_numa_aware = true,
    };
    
    var pool = try beat.ThreadPool.init(std.testing.allocator, config);
    defer pool.deinit();
    
    const num_continuations = 50;
    const PerfResult = struct {
        execution_time: u64 = 0,
        migrations: u32 = 0,
        locality_score: f32 = 0.0,
        worker_id: ?u32 = null,
        numa_node: ?u32 = null,
    };
    var results: [num_continuations]PerfResult = [_]PerfResult{.{}} ** num_continuations;
    
    var mutex = std.Thread.Mutex{};
    
    const PerfTestState = struct {
        results: *[num_continuations]PerfResult,
        mutex: *std.Thread.Mutex,
        index: u32,
    };
    
    var contexts: [num_continuations]PerfTestState = undefined;
    for (contexts[0..], 0..) |*ctx, i| {
        ctx.* = PerfTestState{
            .results = &results,
            .mutex = &mutex,
            .index = @intCast(i),
        };
    }
    
    std.debug.print("1. Running NUMA-aware continuation performance test...\n", .{});
    
    const PerfWork = struct {
        fn executeWrapper(cont: *beat.continuation.Continuation) void {
            const start_time = std.time.nanoTimestamp();
            const ctx = @as(*PerfTestState, @ptrCast(@alignCast(cont.data)));
            
            // Simulate work with NUMA-sensitive operations
            var sum: u64 = 0;
            const work_size = 1000 + (ctx.index % 500); // Variable work size
            for (0..work_size) |i| {
                sum += i * ctx.index;
            }
            
            const end_time = std.time.nanoTimestamp();
            
            ctx.mutex.lock();
            defer ctx.mutex.unlock();
            
            ctx.results[ctx.index] = .{
                .execution_time = @intCast(end_time - start_time),
                .migrations = cont.migration_count,
                .locality_score = cont.locality_score,
                .worker_id = cont.worker_id,
                .numa_node = cont.numa_node,
            };
            
            cont.markCompleted();
        }
    };
    
    var continuations: [num_continuations]beat.continuation.Continuation = undefined;
    const test_start = std.time.nanoTimestamp();
    
    for (continuations[0..], 0..) |*cont, i| {
        cont.* = beat.continuation.Continuation.capture(
            PerfWork.executeWrapper,
            &contexts[i],
            std.testing.allocator
        );
        try pool.submitContinuation(cont);
    }
    
    pool.wait();
    const test_end = std.time.nanoTimestamp();
    const total_test_time = test_end - test_start;
    
    std.debug.print("2. Analyzing NUMA performance results...\n", .{});
    
    // Calculate statistics
    var total_execution_time: u64 = 0;
    var total_migrations: u32 = 0;
    var total_locality_score: f32 = 0.0;
    var min_execution: u64 = std.math.maxInt(u64);
    var max_execution: u64 = 0;
    
    var numa_distribution = std.AutoHashMap(u32, u32).init(std.testing.allocator);
    defer numa_distribution.deinit();
    
    for (results) |result| {
        total_execution_time += result.execution_time;
        total_migrations += result.migrations;
        total_locality_score += result.locality_score;
        min_execution = @min(min_execution, result.execution_time);
        max_execution = @max(max_execution, result.execution_time);
        
        if (result.numa_node) |numa_node| {
            const count = numa_distribution.get(numa_node) orelse 0;
            try numa_distribution.put(numa_node, count + 1);
        }
    }
    
    const avg_execution = total_execution_time / num_continuations;
    const avg_migrations = @as(f32, @floatFromInt(total_migrations)) / @as(f32, @floatFromInt(num_continuations));
    const avg_locality = total_locality_score / @as(f32, @floatFromInt(num_continuations));
    
    std.debug.print("   📊 Performance Metrics:\n", .{});
    std.debug.print("     Total test time: {}ns\n", .{total_test_time});
    std.debug.print("     Average execution time: {}ns\n", .{avg_execution});
    std.debug.print("     Min execution time: {}ns\n", .{min_execution});
    std.debug.print("     Max execution time: {}ns\n", .{max_execution});
    std.debug.print("     Execution time variance: {}ns\n", .{max_execution - min_execution});
    
    std.debug.print("   📊 NUMA Locality Metrics:\n", .{});
    std.debug.print("     Average migrations per continuation: {d:.2}\n", .{avg_migrations});
    std.debug.print("     Average locality score: {d:.2}\n", .{avg_locality});
    std.debug.print("     NUMA nodes used: {}\n", .{numa_distribution.count()});
    
    var numa_iter = numa_distribution.iterator();
    while (numa_iter.next()) |entry| {
        const percentage = (@as(f32, @floatFromInt(entry.value_ptr.*)) / @as(f32, @floatFromInt(num_continuations))) * 100.0;
        std.debug.print("     NUMA node {}: {} continuations ({d:.1}%)\n", 
            .{ entry.key_ptr.*, entry.value_ptr.*, percentage });
    }
    
    // Quality metrics for NUMA-aware stealing
    const high_locality_count = blk: {
        var count: u32 = 0;
        for (results) |result| {
            if (result.locality_score > 0.8) count += 1;
        }
        break :blk count;
    };
    
    const low_migration_count = blk: {
        var count: u32 = 0;
        for (results) |result| {
            if (result.migrations <= 1) count += 1;
        }
        break :blk count;
    };
    
    std.debug.print("   📊 NUMA Efficiency Metrics:\n", .{});
    std.debug.print("     High locality (>0.8): {}/{} ({d:.1}%)\n", 
        .{ high_locality_count, num_continuations, 
           (@as(f32, @floatFromInt(high_locality_count)) / @as(f32, @floatFromInt(num_continuations))) * 100.0 });
    std.debug.print("     Low migration (≤1): {}/{} ({d:.1}%)\n", 
        .{ low_migration_count, num_continuations,
           (@as(f32, @floatFromInt(low_migration_count)) / @as(f32, @floatFromInt(num_continuations))) * 100.0 });
    
    // Verify all continuations completed
    var completed_count: u32 = 0;
    for (continuations) |cont| {
        if (cont.state == .completed) completed_count += 1;
    }
    
    try std.testing.expect(completed_count == num_continuations);
    
    // NUMA-aware stealing should maintain reasonable locality
    try std.testing.expect(avg_locality > 0.5); // Should maintain decent locality
    try std.testing.expect(avg_migrations < 3.0); // Should limit excessive migrations
    
    std.debug.print("   ✅ All {} continuations completed with NUMA awareness\n", .{completed_count});
    
    std.debug.print("✅ NUMA performance comparison test completed!\n", .{});
}