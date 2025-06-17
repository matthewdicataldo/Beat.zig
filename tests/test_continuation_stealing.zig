const std = @import("std");
const beat = @import("beat");

// ============================================================================
// Continuation Stealing Tests
// ============================================================================

test "continuation stealing - basic functionality" {
    std.debug.print("\n=== Continuation Stealing Basic Functionality Test ===\n", .{});
    
    var data: i32 = 42;
    
    const TestResume = struct {
        fn executeWrapper(cont: *beat.continuation.Continuation) void {
            const value = @as(*i32, @ptrCast(@alignCast(cont.data)));
            value.* *= 2;
            cont.markCompleted();
        }
    };
    
    // Test continuation creation and capture
    std.debug.print("1. Testing continuation creation...\n", .{});
    var continuation = beat.continuation.Continuation.capture(
        TestResume.executeWrapper, 
        &data, 
        std.testing.allocator
    );
    
    try std.testing.expect(continuation.state == .pending);
    try std.testing.expect(continuation.canBeStolen());
    try std.testing.expect(continuation.frame_ptr != 0);
    try std.testing.expect(continuation.return_address != 0);
    
    std.debug.print("   ✅ Continuation created successfully\n", .{});
    std.debug.print("     Frame pointer: 0x{x}\n", .{continuation.frame_ptr});
    std.debug.print("     Return address: 0x{x}\n", .{continuation.return_address});
    std.debug.print("     Initial state: {}\n", .{continuation.state});
    
    // Test continuation execution
    std.debug.print("2. Testing continuation execution...\n", .{});
    continuation.markRunning(0);
    try std.testing.expect(continuation.state == .running);
    try std.testing.expect(!continuation.canBeStolen());
    
    continuation.execute();
    try std.testing.expect(continuation.state == .completed);
    try std.testing.expect(data == 84); // 42 * 2
    
    std.debug.print("   ✅ Continuation executed successfully\n", .{});
    std.debug.print("     Final data value: {}\n", .{data});
    std.debug.print("     Final state: {}\n", .{continuation.state});
    
    std.debug.print("✅ Basic continuation functionality test completed!\n", .{});
}

test "continuation stealing - work stealing simulation" {
    std.debug.print("\n=== Continuation Work Stealing Simulation Test ===\n", .{});
    
    var shared_counter: u32 = 0;
    var mutex = std.Thread.Mutex{};
    
    const SharedWork = struct {
        fn executeWrapper(cont: *beat.continuation.Continuation) void {
            const context = @as(*@This(), @ptrCast(@alignCast(cont.data)));
            context.mutex.lock();
            defer context.mutex.unlock();
            context.counter.* += 1;
            cont.markCompleted();
        }
        
        counter: *u32,
        mutex: *std.Thread.Mutex,
    };
    
    var work_context = SharedWork{
        .counter = &shared_counter,
        .mutex = &mutex,
    };
    
    // Create multiple continuations
    std.debug.print("1. Creating multiple continuations for stealing simulation...\n", .{});
    
    var continuations: [4]beat.continuation.Continuation = undefined;
    for (continuations[0..], 0..) |*cont, i| {
        cont.* = beat.continuation.Continuation.capture(
            SharedWork.executeWrapper,
            &work_context,
            std.testing.allocator
        );
        std.debug.print("   Created continuation {}: frame=0x{x}\n", .{ i, cont.frame_ptr });
    }
    
    // Simulate work stealing between workers
    std.debug.print("2. Simulating work stealing between workers...\n", .{});
    
    // Worker 0 steals continuation 1
    continuations[1].markStolen(0);
    try std.testing.expect(continuations[1].state == .stolen);
    try std.testing.expect(continuations[1].worker_id == 0);
    try std.testing.expect(continuations[1].steal_count == 1);
    std.debug.print("   Worker 0 stole continuation 1\n", .{});
    
    // Worker 1 steals continuation 2  
    continuations[2].markStolen(1);
    try std.testing.expect(continuations[2].state == .stolen);
    try std.testing.expect(continuations[2].worker_id == 1);
    std.debug.print("   Worker 1 stole continuation 2\n", .{});
    
    // Execute all continuations on their assigned workers
    std.debug.print("3. Executing continuations on assigned workers...\n", .{});
    
    for (continuations[0..], 0..) |*cont, i| {
        const worker_id: u32 = @intCast(i % 2); // Alternate between workers 0 and 1
        if (cont.state == .stolen) {
            cont.markRunning(cont.worker_id.?);
        } else {
            cont.markRunning(worker_id);
        }
        cont.execute();
        
        std.debug.print("   Executed continuation {} on worker {}\n", .{ i, cont.worker_id.? });
    }
    
    // Verify all work was completed
    try std.testing.expect(shared_counter == 4);
    std.debug.print("   ✅ All {} continuations executed successfully\n", .{shared_counter});
    
    // Verify stealing statistics
    var total_steals: u32 = 0;
    for (continuations[0..]) |cont| {
        total_steals += cont.steal_count;
    }
    
    try std.testing.expect(total_steals == 2); // 2 continuations were stolen
    std.debug.print("   ✅ Stealing statistics: {} steals total\n", .{total_steals});
    
    std.debug.print("✅ Work stealing simulation test completed!\n", .{});
}

test "continuation stealing - WorkItem integration" {
    std.debug.print("\n=== WorkItem Integration Test ===\n", .{});
    
    var data: i32 = 100;
    
    const TestWork = struct {
        fn executeWrapper(cont: *beat.continuation.Continuation) void {
            const value = @as(*i32, @ptrCast(@alignCast(cont.data)));
            value.* += 50;
            cont.markCompleted();
        }
    };
    
    // Test WorkItem creation from continuation
    std.debug.print("1. Testing WorkItem creation from continuation...\n", .{});
    var continuation = beat.continuation.Continuation.capture(
        TestWork.executeWrapper,
        &data,
        std.testing.allocator
    );
    
    var work_item = beat.lockfree.WorkItem.fromContinuation(&continuation);
    try std.testing.expect(work_item.isContinuation());
    try std.testing.expect(!work_item.isTask());
    try std.testing.expect(work_item.canBeStolen());
    
    std.debug.print("   ✅ WorkItem created from continuation\n", .{});
    std.debug.print("     Is continuation: {}\n", .{work_item.isContinuation()});
    std.debug.print("     Can be stolen: {}\n", .{work_item.canBeStolen()});
    std.debug.print("     Priority: {}\n", .{work_item.getPriority()});
    
    // Test WorkItem execution
    std.debug.print("2. Testing WorkItem execution...\n", .{});
    work_item.execute(1);
    
    try std.testing.expect(continuation.state == .completed);
    try std.testing.expect(continuation.worker_id == 1);
    try std.testing.expect(data == 150); // 100 + 50
    
    std.debug.print("   ✅ WorkItem executed successfully\n", .{});
    std.debug.print("     Final data value: {}\n", .{data});
    std.debug.print("     Continuation worker ID: {}\n", .{continuation.worker_id.?});
    
    // Test WorkItem stealing
    std.debug.print("3. Testing WorkItem stealing behavior...\n", .{});
    var data2: i32 = 200;
    var continuation2 = beat.continuation.Continuation.capture(
        TestWork.executeWrapper,
        &data2,
        std.testing.allocator
    );
    
    var work_item2 = beat.lockfree.WorkItem.fromContinuation(&continuation2);
    try std.testing.expect(work_item2.canBeStolen());
    
    work_item2.markStolen(2);
    try std.testing.expect(continuation2.state == .stolen);
    try std.testing.expect(continuation2.worker_id == 2);
    try std.testing.expect(continuation2.steal_count == 1);
    
    std.debug.print("   ✅ WorkItem stealing completed\n", .{});
    std.debug.print("     Stolen by worker: {}\n", .{continuation2.worker_id.?});
    std.debug.print("     Steal count: {}\n", .{continuation2.steal_count});
    
    std.debug.print("✅ WorkItem integration test completed!\n", .{});
}

test "continuation stealing - registry management" {
    std.debug.print("\n=== Continuation Registry Management Test ===\n", .{});
    
    var registry = beat.continuation.ContinuationRegistry.init(std.testing.allocator);
    defer registry.deinit();
    
    // Create test continuations
    std.debug.print("1. Creating and registering continuations...\n", .{});
    
    var data1: i32 = 10;
    var data2: i32 = 20;
    var data3: i32 = 30;
    
    const TestWork = struct {
        fn executeWrapper(cont: *beat.continuation.Continuation) void {
            const value = @as(*i32, @ptrCast(@alignCast(cont.data)));
            value.* *= 3;
            cont.markCompleted();
        }
    };
    
    var cont1 = beat.continuation.Continuation.capture(TestWork.executeWrapper, &data1, std.testing.allocator);
    var cont2 = beat.continuation.Continuation.capture(TestWork.executeWrapper, &data2, std.testing.allocator);
    var cont3 = beat.continuation.Continuation.capture(TestWork.executeWrapper, &data3, std.testing.allocator);
    
    try registry.registerContinuation(&cont1);
    try registry.registerContinuation(&cont2);
    try registry.registerContinuation(&cont3);
    
    var stats = registry.getStatistics();
    try std.testing.expect(stats.active_count == 3);
    try std.testing.expect(stats.completed_count == 0);
    
    std.debug.print("   ✅ Registered {} continuations\n", .{stats.active_count});
    
    // Execute and complete continuations
    std.debug.print("2. Executing continuations with stealing...\n", .{});
    
    // Steal some continuations
    cont2.markStolen(1);
    cont3.markStolen(2);
    
    // Execute all continuations
    cont1.markRunning(0);
    cont1.execute();
    try registry.completeContinuation(&cont1);
    
    cont2.markRunning(1);
    cont2.execute();
    try registry.completeContinuation(&cont2);
    
    cont3.markRunning(2);
    cont3.execute();
    try registry.completeContinuation(&cont3);
    
    // Check final statistics
    stats = registry.getStatistics();
    try std.testing.expect(stats.active_count == 0);
    try std.testing.expect(stats.completed_count == 3);
    try std.testing.expect(stats.total_steals == 2); // cont2 and cont3 were stolen
    
    std.debug.print("   ✅ Execution completed\n", .{});
    std.debug.print("     Active: {}, Completed: {}\n", .{ stats.active_count, stats.completed_count });
    std.debug.print("     Total steals: {}\n", .{stats.total_steals});
    std.debug.print("     Average execution time: {d:.2} ns\n", .{stats.avg_execution_time_ns});
    
    // Verify data was processed correctly
    try std.testing.expect(data1 == 30);  // 10 * 3
    try std.testing.expect(data2 == 60);  // 20 * 3
    try std.testing.expect(data3 == 90);  // 30 * 3
    
    std.debug.print("   ✅ Data processing verified: {}, {}, {}\n", .{ data1, data2, data3 });
    
    std.debug.print("✅ Registry management test completed!\n", .{});
}

test "continuation stealing - performance characteristics" {
    std.debug.print("\n=== Continuation Performance Characteristics Test ===\n", .{});
    
    const num_continuations = 100;
    var continuations: [num_continuations]beat.continuation.Continuation = undefined;
    var data_array: [num_continuations]u64 = undefined;
    
    // Initialize data
    for (data_array[0..], 0..) |*data, i| {
        data.* = i;
    }
    
    const PerfWork = struct {
        fn executeWrapper(cont: *beat.continuation.Continuation) void {
            const value = @as(*u64, @ptrCast(@alignCast(cont.data)));
            // Simulate some work
            value.* = value.* * 2 + 1;
            cont.markCompleted();
        }
    };
    
    std.debug.print("1. Creating {} continuations for performance test...\n", .{num_continuations});
    
    const start_time = std.time.nanoTimestamp();
    
    // Create continuations
    for (continuations[0..], 0..) |*cont, i| {
        cont.* = beat.continuation.Continuation.capture(
            PerfWork.executeWrapper,
            &data_array[i],
            std.testing.allocator
        );
    }
    
    const creation_time = std.time.nanoTimestamp();
    
    // Simulate stealing pattern (every 3rd continuation gets stolen)
    std.debug.print("2. Simulating realistic stealing pattern...\n", .{});
    var steal_count: u32 = 0;
    for (continuations[0..], 0..) |*cont, i| {
        if (i % 3 == 0) {
            cont.markStolen(@intCast((i + 1) % 4)); // Steal to different workers
            steal_count += 1;
        }
    }
    
    // Execute all continuations
    std.debug.print("3. Executing all continuations...\n", .{});
    for (continuations[0..], 0..) |*cont, i| {
        const worker_id: u32 = if (cont.state == .stolen) 
            cont.worker_id.? 
        else 
            @intCast(i % 4);
        
        cont.markRunning(worker_id);
        cont.execute();
    }
    
    const execution_time = std.time.nanoTimestamp();
    
    // Calculate performance metrics
    const creation_duration = creation_time - start_time;
    const execution_duration = execution_time - creation_time;
    const total_duration = execution_time - start_time;
    
    const avg_creation_ns = @divTrunc(creation_duration, num_continuations);
    const avg_execution_ns = @divTrunc(execution_duration, num_continuations);
    
    std.debug.print("   ✅ Performance Results:\n", .{});
    std.debug.print("     Total continuations: {}\n", .{num_continuations});
    std.debug.print("     Continuations stolen: {}\n", .{steal_count});
    std.debug.print("     Total creation time: {} ns\n", .{creation_duration});
    std.debug.print("     Total execution time: {} ns\n", .{execution_duration});
    std.debug.print("     Average creation: {} ns/continuation\n", .{avg_creation_ns});
    std.debug.print("     Average execution: {} ns/continuation\n", .{avg_execution_ns});
    std.debug.print("     Total test time: {} ns\n", .{total_duration});
    
    // Verify all work was done correctly
    var verification_passed = true;
    for (data_array[0..], 0..) |data, i| {
        const expected = i * 2 + 1;
        if (data != expected) {
            verification_passed = false;
            break;
        }
    }
    
    try std.testing.expect(verification_passed);
    std.debug.print("   ✅ Data verification passed\n", .{});
    
    std.debug.print("✅ Performance characteristics test completed!\n", .{});
}