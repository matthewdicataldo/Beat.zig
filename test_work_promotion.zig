const std = @import("std");
const beat = @import("src/core.zig");
const scheduler = @import("src/scheduler.zig");

test "Work Promotion Trigger - Basic Functionality" {
    const allocator = std.testing.allocator;
    
    // Create scheduler with promotion enabled
    const config = beat.Config{
        .enable_heartbeat = true,
        .heartbeat_interval_us = 10, // Fast heartbeat for testing
        .promotion_threshold = 2,    // Low threshold to trigger easily
        .min_work_cycles = 10,       // Low minimum for testing
        .num_workers = 2,
    };
    
    const sched = try scheduler.Scheduler.init(allocator, &config);
    defer sched.deinit();
    
    // Initially no promotions
    try std.testing.expect(sched.getPromotionCount() == 0);
    
    // Simulate work that would trigger promotion
    const tokens = &sched.worker_tokens[0];
    
    // Add enough work and minimal overhead to trigger promotion
    tokens.update(1000, 100); // 10:1 ratio, above threshold of 2:1
    
    // Verify promotion would be triggered
    try std.testing.expect(tokens.shouldPromote());
    
    // Let the heartbeat run briefly to trigger promotion
    std.time.sleep(50_000_000); // 50ms
    
    // Force a promotion cycle if conditions are met
    if (tokens.shouldPromote()) {
        // Manually call promotion (simulating heartbeat behavior)
        const promo_count_before = sched.getPromotionCount();
        sched.triggerPromotion(0);
        const promo_count_after = sched.getPromotionCount();
        
        try std.testing.expect(promo_count_after > promo_count_before);
    } else {
        // Manually trigger to test the mechanism
        const promo_count_before = sched.getPromotionCount();
        sched.triggerPromotion(0);
        const promo_count_after = sched.getPromotionCount();
        
        try std.testing.expect(promo_count_after > promo_count_before);
    }
    
    std.debug.print("Work promotion test completed. Promotions triggered: {}\n", .{sched.getPromotionCount()});
}

test "Work Promotion Trigger - Multiple Workers" {
    const allocator = std.testing.allocator;
    
    const config = beat.Config{
        .enable_heartbeat = true,
        .heartbeat_interval_us = 20,
        .promotion_threshold = 3,
        .min_work_cycles = 50,
        .num_workers = 4,
    };
    
    const sched = try scheduler.Scheduler.init(allocator, &config);
    defer sched.deinit();
    
    // Test multiple workers triggering promotions
    for (sched.worker_tokens, 0..) |*tokens, i| {
        // Give each worker different work characteristics
        const work_cycles = (i + 1) * 500;  // Increasing work
        const overhead_cycles = 100;         // Fixed overhead
        
        tokens.update(work_cycles, overhead_cycles);
        
        if (tokens.shouldPromote()) {
            sched.triggerPromotion(@intCast(i));
        }
    }
    
    // Should have triggered promotions for workers with good ratios
    try std.testing.expect(sched.getPromotionCount() > 0);
    
    std.debug.print("Multi-worker promotion test: {} promotions\n", .{sched.getPromotionCount()});
}

test "Work Promotion Trigger - Integration with ThreadPool" {
    const allocator = std.testing.allocator;
    
    // Create pool with heartbeat scheduling enabled
    const pool = try beat.createPoolWithConfig(allocator, .{
        .num_workers = 2,
        .enable_heartbeat = true,
        .enable_statistics = true,
        .heartbeat_interval_us = 10,
        .promotion_threshold = 5,
        .min_work_cycles = 100,
    });
    defer pool.deinit();
    
    // Submit some work to generate scheduling activity
    var counter = std.atomic.Value(u32).init(0);
    
    const test_task = struct {
        fn execute(data: *anyopaque) void {
            const cnt = @as(*std.atomic.Value(u32), @ptrCast(@alignCast(data)));
            _ = cnt.fetchAdd(1, .monotonic);
            
            // Do some work to generate cycles
            var sum: u64 = 0;
            for (0..1000) |i| {
                sum += i;
            }
            std.mem.doNotOptimizeAway(&sum);
        }
    }.execute;
    
    // Submit tasks
    for (0..10) |_| {
        try pool.submit(.{ .func = test_task, .data = &counter });
    }
    
    pool.wait();
    
    // Verify tasks completed
    try std.testing.expect(counter.load(.acquire) == 10);
    
    // Check if scheduler recorded any promotions
    if (pool.scheduler) |sched| {
        const promotions = sched.getPromotionCount();
        std.debug.print("ThreadPool integration test: {} promotions recorded\n", .{promotions});
        // Promotions may or may not have triggered depending on timing and work characteristics
    }
}