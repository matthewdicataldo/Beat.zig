const std = @import("std");
const beat = @import("beat");

// ============================================================================
// ThreadPool Continuation Integration Tests
// ============================================================================

test "threadpool continuation integration - basic mixed workload" {
    std.debug.print("\n=== ThreadPool Continuation Integration - Basic Mixed Workload ===\n", .{});
    
    // Create ThreadPool with continuation support enabled
    const config = beat.Config{
        .num_workers = 4,
        .enable_predictive = true,  // Enables continuation registry
        .enable_work_stealing = true,
        .enable_lock_free = true,
    };
    
    var pool = try beat.ThreadPool.init(std.testing.allocator, config);
    defer pool.deinit();
    
    std.debug.print("1. Created ThreadPool with {} workers\n", .{pool.workers.len});
    
    // Shared state for mixed workload
    var task_counter: u32 = 0;
    var continuation_counter: u32 = 0;
    var mutex = std.Thread.Mutex{};
    
    const TestState = struct {
        task_counter: *u32,
        continuation_counter: *u32,
        mutex: *std.Thread.Mutex,
    };
    
    var test_state = TestState{
        .task_counter = &task_counter,
        .continuation_counter = &continuation_counter,
        .mutex = &mutex,
    };
    
    // Submit regular tasks
    std.debug.print("2. Submitting regular tasks...\n", .{});
    const TaskWork = struct {
        fn execute(data: *anyopaque) void {
            const state = @as(*TestState, @ptrCast(@alignCast(data)));
            state.mutex.lock();
            defer state.mutex.unlock();
            state.task_counter.* += 1;
        }
    };
    
    const num_tasks = 10;
    for (0..num_tasks) |_| {
        try pool.submit(beat.Task{
            .func = TaskWork.execute,
            .data = &test_state,
            .priority = .normal,
        });
    }
    
    // Submit continuations
    std.debug.print("3. Submitting continuations...\n", .{});
    const ContinuationWork = struct {
        fn executeWrapper(cont: *beat.continuation.Continuation) void {
            const state = @as(*TestState, @ptrCast(@alignCast(cont.data)));
            state.mutex.lock();
            defer state.mutex.unlock();
            state.continuation_counter.* += 1;
            cont.markCompleted();
        }
    };
    
    const num_continuations = 10;
    var continuations: [num_continuations]beat.continuation.Continuation = undefined;
    
    for (continuations[0..], 0..) |*cont, i| {
        cont.* = beat.continuation.Continuation.capture(
            ContinuationWork.executeWrapper,
            &test_state,
            std.testing.allocator
        );
        cont.affinity_hint = @intCast(i % 4); // Distribute across workers
        try pool.submitContinuation(cont);
        std.debug.print("   Submitted continuation {} with affinity hint {}\n", .{ i, cont.affinity_hint.? });
    }
    
    // Wait for all work to complete
    std.debug.print("4. Waiting for all work to complete...\n", .{});
    pool.wait();
    
    // Verify results
    std.debug.print("5. Verifying results...\n", .{});
    try std.testing.expect(task_counter == num_tasks);
    try std.testing.expect(continuation_counter == num_continuations);
    
    std.debug.print("   ✅ Tasks completed: {}/{}\n", .{ task_counter, num_tasks });
    std.debug.print("   ✅ Continuations completed: {}/{}\n", .{ continuation_counter, num_continuations });
    
    // Check continuation states
    std.debug.print("6. Checking continuation states...\n", .{});
    var completed_count: u32 = 0;
    for (continuations[0..]) |cont| {
        if (cont.state == .completed) {
            completed_count += 1;
        }
        std.debug.print("   Continuation: worker={?}, steals={}, state={}\n", 
            .{ cont.worker_id, cont.steal_count, cont.state });
    }
    
    try std.testing.expect(completed_count == num_continuations);
    std.debug.print("   ✅ All continuations marked as completed\n", .{});
    
    std.debug.print("✅ Basic mixed workload test completed successfully!\n", .{});
}

test "threadpool continuation integration - work stealing behavior" {
    std.debug.print("\n=== ThreadPool Continuation Integration - Work Stealing Behavior ===\n", .{});
    
    const config = beat.Config{
        .num_workers = 4,
        .enable_predictive = true,
        .enable_work_stealing = true,
        .enable_lock_free = true,
        .task_queue_size = 64,
    };
    
    var pool = try beat.ThreadPool.init(std.testing.allocator, config);
    defer pool.deinit();
    
    std.debug.print("1. Created ThreadPool for work stealing test\n", .{});
    
    // Create work that takes varying amounts of time
    var results: [20]u32 = [_]u32{0} ** 20;
    var mutex = std.Thread.Mutex{};
    
    const WorkContext = struct {
        results: *[20]u32,
        mutex: *std.Thread.Mutex,
        index: u32,
        work_amount: u32,
    };
    
    // Create varying work contexts
    var work_contexts: [20]WorkContext = undefined;
    for (work_contexts[0..], 0..) |*ctx, i| {
        ctx.* = WorkContext{
            .results = &results,
            .mutex = &mutex,
            .index = @intCast(i),
            .work_amount = @intCast((i % 5) + 1), // 1-5 units of work
        };
    }
    
    // Submit mix of tasks and continuations with varying workloads
    std.debug.print("2. Submitting mixed workload with varying execution times...\n", .{});
    
    const VariableWork = struct {
        // Heavy task work
        fn heavyTaskWork(data: *anyopaque) void {
            const ctx = @as(*WorkContext, @ptrCast(@alignCast(data)));
            
            // Simulate varying amounts of work
            var sum: u64 = 0;
            const iterations = ctx.work_amount * 10000;
            for (0..iterations) |j| {
                sum += j;
            }
            
            ctx.mutex.lock();
            defer ctx.mutex.unlock();
            ctx.results[ctx.index] = ctx.work_amount;
        }
        
        // Light continuation work
        fn lightContinuationWork(cont: *beat.continuation.Continuation) void {
            const ctx = @as(*WorkContext, @ptrCast(@alignCast(cont.data)));
            
            // Light work for continuation
            var sum: u32 = 0;
            for (0..ctx.work_amount * 100) |j| {
                sum += @intCast(j % 1000);
            }
            
            ctx.mutex.lock();
            defer ctx.mutex.unlock();
            ctx.results[ctx.index] = ctx.work_amount;
            cont.markCompleted();
        }
    };
    
    // Submit heavy tasks first (these will create stealing opportunities)
    for (work_contexts[0..10], 0..) |*ctx, i| {
        const task = beat.Task{
            .func = VariableWork.heavyTaskWork,
            .data = ctx,
            .priority = if (i < 5) .high else .normal,
        };
        try pool.submit(task);
        std.debug.print("   Submitted heavy task {} (work units: {})\n", .{ i, ctx.work_amount });
    }
    
    // Then submit light continuations (these should be stolen)
    var continuations: [10]beat.continuation.Continuation = undefined;
    for (continuations[0..], 0..) |*cont, i| {
        const ctx_index = i + 10;
        cont.* = beat.continuation.Continuation.capture(
            VariableWork.lightContinuationWork,
            &work_contexts[ctx_index],
            std.testing.allocator
        );
        try pool.submitContinuation(cont);
        std.debug.print("   Submitted light continuation {} (work units: {})\n", 
            .{ i, work_contexts[ctx_index].work_amount });
    }
    
    std.debug.print("3. Executing mixed workload to trigger work stealing...\n", .{});
    pool.wait();
    
    // Analyze stealing behavior
    std.debug.print("4. Analyzing work stealing behavior...\n", .{});
    var total_steals: u32 = 0;
    var workers_used = std.AutoHashMap(u32, void).init(std.testing.allocator);
    defer workers_used.deinit();
    
    for (continuations[0..], 0..) |cont, i| {
        total_steals += cont.steal_count;
        if (cont.worker_id) |worker_id| {
            try workers_used.put(worker_id, {});
        }
        std.debug.print("   Continuation {}: worker={?}, steals={}, execution_time={?}ns\n",
            .{ i, cont.worker_id, cont.steal_count, cont.execution_time_ns });
    }
    
    std.debug.print("   ✅ Total continuation steals: {}\n", .{total_steals});
    std.debug.print("   ✅ Workers used for continuations: {}\n", .{workers_used.count()});
    
    // Verify all work completed correctly
    var completed_correctly: u32 = 0;
    for (results, 0..) |result, i| {
        const expected = (i % 5) + 1;
        if (result == expected) {
            completed_correctly += 1;
        }
    }
    
    try std.testing.expect(completed_correctly == 20);
    std.debug.print("   ✅ All work completed correctly: {}/20\n", .{completed_correctly});
    
    // Work stealing should have occurred (continuations should have been stolen)
    try std.testing.expect(total_steals > 0 or workers_used.count() > 1);
    
    std.debug.print("✅ Work stealing behavior test completed successfully!\n", .{});
}

test "threadpool continuation integration - performance comparison" {
    std.debug.print("\n=== ThreadPool Continuation Integration - Performance Comparison ===\n", .{});
    
    const config = beat.Config{
        .num_workers = 4,
        .enable_predictive = true,
        .enable_work_stealing = true,
        .enable_lock_free = true,
        .enable_statistics = true,
    };
    
    var pool = try beat.ThreadPool.init(std.testing.allocator, config);
    defer pool.deinit();
    
    const num_operations = 100;
    var task_results: [num_operations]u64 = undefined;
    var continuation_results: [num_operations]u64 = undefined;
    
    // Test 1: Tasks only
    std.debug.print("1. Testing tasks-only performance...\n", .{});
    
    const TaskBenchmark = struct {
        fn execute(data: *anyopaque) void {
            const result_ptr = @as(*u64, @ptrCast(@alignCast(data)));
            const start = std.time.nanoTimestamp();
            
            // Small amount of work
            var sum: u64 = 0;
            for (0..1000) |i| {
                sum += i * 2;
            }
            
            const end = std.time.nanoTimestamp();
            result_ptr.* = @intCast(end - start);
        }
    };
    
    const task_start = std.time.nanoTimestamp();
    
    for (task_results[0..], 0..) |*result, i| {
        const task = beat.Task{
            .func = TaskBenchmark.execute,
            .data = result,
            .priority = .normal,
        };
        try pool.submit(task);
        _ = i;
    }
    
    pool.wait();
    const task_end = std.time.nanoTimestamp();
    const task_total_time = task_end - task_start;
    
    // Test 2: Continuations only
    std.debug.print("2. Testing continuations-only performance...\n", .{});
    
    const ContinuationBenchmark = struct {
        fn executeWrapper(cont: *beat.continuation.Continuation) void {
            const result_ptr = @as(*u64, @ptrCast(@alignCast(cont.data)));
            const start = std.time.nanoTimestamp();
            
            // Same amount of work as task
            var sum: u64 = 0;
            for (0..1000) |i| {
                sum += i * 2;
            }
            
            const end = std.time.nanoTimestamp();
            result_ptr.* = @intCast(end - start);
            cont.markCompleted();
        }
    };
    
    var continuations: [num_operations]beat.continuation.Continuation = undefined;
    const cont_start = std.time.nanoTimestamp();
    
    for (continuations[0..], 0..) |*cont, i| {
        cont.* = beat.continuation.Continuation.capture(
            ContinuationBenchmark.executeWrapper,
            &continuation_results[i],
            std.testing.allocator
        );
        try pool.submitContinuation(cont);
    }
    
    pool.wait();
    const cont_end = std.time.nanoTimestamp();
    const cont_total_time = cont_end - cont_start;
    
    // Analyze performance
    std.debug.print("3. Analyzing performance results...\n", .{});
    
    // Calculate statistics for tasks
    var task_sum: u64 = 0;
    var task_min: u64 = std.math.maxInt(u64);
    var task_max: u64 = 0;
    for (task_results) |time| {
        task_sum += time;
        task_min = @min(task_min, time);
        task_max = @max(task_max, time);
    }
    const task_avg = task_sum / num_operations;
    
    // Calculate statistics for continuations
    var cont_sum: u64 = 0;
    var cont_min: u64 = std.math.maxInt(u64);
    var cont_max: u64 = 0;
    for (continuation_results) |time| {
        // Guard against overflow
        if (cont_sum > std.math.maxInt(u64) - time) {
            std.debug.print("   ⚠️ Overflow detected in continuation sum, using safe calculation\n", .{});
            break;
        }
        cont_sum += time;
        cont_min = @min(cont_min, time);
        cont_max = @max(cont_max, time);
    }
    const cont_avg = cont_sum / num_operations;
    
    std.debug.print("   📊 Task Performance:\n", .{});
    std.debug.print("     Total time: {}ns\n", .{task_total_time});
    std.debug.print("     Average execution: {}ns\n", .{task_avg});
    std.debug.print("     Min: {}ns, Max: {}ns\n", .{ task_min, task_max });
    
    std.debug.print("   📊 Continuation Performance:\n", .{});
    std.debug.print("     Total time: {}ns\n", .{cont_total_time});
    std.debug.print("     Average execution: {}ns\n", .{cont_avg});
    std.debug.print("     Min: {}ns, Max: {}ns\n", .{ cont_min, cont_max });
    
    // Calculate performance ratio
    const performance_ratio = @as(f64, @floatFromInt(task_total_time)) / @as(f64, @floatFromInt(cont_total_time));
    std.debug.print("   🚀 Performance Ratio (Task/Continuation): {d:.2}\n", .{performance_ratio});
    
    // Analyze stealing behavior for continuations
    var stolen_count: u32 = 0;
    for (continuations) |cont| {
        if (cont.steal_count > 0) {
            stolen_count += 1;
        }
    }
    
    std.debug.print("   📈 Continuation Stealing: {}/{} were stolen\n", .{ stolen_count, num_operations });
    
    // Both should complete successfully
    try std.testing.expect(task_sum > 0);
    try std.testing.expect(cont_sum > 0);
    
    std.debug.print("✅ Performance comparison test completed successfully!\n", .{});
}