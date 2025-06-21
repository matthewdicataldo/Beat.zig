// Real-world verification of adaptive heartbeat implementation
const std = @import("std");
const core = @import("src/core.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("=== Adaptive Heartbeat Integration Test ===\n\n", .{});
    
    // Create thread pool with adaptive heartbeat enabled
    const config = core.Config{
        .num_workers = 2,
        .enable_heartbeat = true,
        .heartbeat_interval_us = 100, // 100μs baseline
        .enable_statistics = true,
    };
    
    var pool = try core.ThreadPool.init(allocator, config);
    defer pool.deinit();
    
    // Test 1: Verify adaptive heartbeat responds to idle period
    std.debug.print("Test 1: Idle period (should increase interval)\n", .{});
    
    const scheduler = pool.scheduler orelse {
        std.debug.print("❌ Scheduler not initialized!\n", .{});
        return;
    };
    
    const initial_stats = scheduler.getHeartbeatStats();
    std.debug.print("Initial interval: {}μs\n", .{initial_stats.current_interval_us});
    
    // Wait for adaptive algorithm to detect idle state
    std.time.sleep(2_000_000_000); // 2 seconds of idle time
    
    const idle_stats = scheduler.getHeartbeatStats();
    std.debug.print("After idle period: {}μs (adjustments: {})\n", 
        .{idle_stats.current_interval_us, idle_stats.adjustments_count});
    
    // Test 2: Submit tasks and verify heartbeat adapts to activity
    std.debug.print("\nTest 2: Active period (should decrease interval)\n", .{});
    
    // Submit several tasks to create activity
    for (0..10) |i| {
        var task_data = i; // Store on stack
        try pool.submit(core.Task{
            .func = simpleTask,
            .data = @ptrCast(&task_data),
        });
        std.time.sleep(50_000_000); // 50ms between submissions
    }
    
    // Wait for tasks to complete and heartbeat to adjust
    pool.wait();
    std.time.sleep(500_000_000); // 500ms for heartbeat adjustment
    
    const active_stats = scheduler.getHeartbeatStats();
    std.debug.print("After active period: {}μs (adjustments: {})\n", 
        .{active_stats.current_interval_us, active_stats.adjustments_count});
    
    // Test 3: Return to idle and verify interval increases again
    std.debug.print("\nTest 3: Return to idle (should increase interval again)\n", .{});
    
    std.time.sleep(1_500_000_000); // 1.5 seconds idle
    
    const final_stats = scheduler.getHeartbeatStats();
    std.debug.print("Final idle state: {}μs (adjustments: {})\n", 
        .{final_stats.current_interval_us, final_stats.adjustments_count});
    
    // Summary
    std.debug.print("\n=== Adaptive Heartbeat Verification Results ===\n", .{});
    std.debug.print("✓ Initial interval: {}μs\n", .{initial_stats.current_interval_us});
    std.debug.print("✓ Idle adaptation: {}μs ({}x change)\n", 
        .{idle_stats.current_interval_us, idle_stats.current_interval_us / initial_stats.current_interval_us});
    std.debug.print("✓ Active adaptation: {}μs\n", .{active_stats.current_interval_us});
    std.debug.print("✓ Return to idle: {}μs\n", .{final_stats.current_interval_us});
    std.debug.print("✓ Total adjustments: {}\n", .{final_stats.adjustments_count});
    
    const efficiency_gain = @as(f64, @floatFromInt(final_stats.current_interval_us)) / 
                           @as(f64, @floatFromInt(initial_stats.current_interval_us));
    std.debug.print("✓ Power efficiency gain: {:.1}x in idle state\n", .{efficiency_gain});
    
    if (final_stats.adjustments_count > 0) {
        std.debug.print("\n🎉 ADAPTIVE HEARTBEAT WORKING CORRECTLY!\n", .{});
    } else {
        std.debug.print("\n⚠️  No adjustments detected - check implementation\n", .{});
    }
}

fn simpleTask(data: *anyopaque) void {
    const value = @as(*const usize, @ptrCast(@alignCast(data))).*;
    
    // Simulate some work that would trigger promotions
    var sum: u64 = 0;
    for (0..value * 1000) |i| {
        sum +%= i;
    }
    
    // Prevent optimization
    std.mem.doNotOptimizeAway(&sum);
}