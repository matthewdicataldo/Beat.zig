// Simple heartbeat timing test
const std = @import("std");

pub fn main() !void {
    std.debug.print("=== Current Heartbeat Timing Test ===\n", .{});
    
    // Test 1: Measure heartbeat frequency during idle period
    std.debug.print("Testing idle heartbeat overhead...\n", .{});
    
    const start_time = std.time.nanoTimestamp();
    
    // Simulate heartbeat loop with fixed 100μs interval (current default)
    const fixed_interval_ns = 100 * 1000; // 100μs in nanoseconds
    var wake_ups: u32 = 0;
    
    const test_duration_ns = 1_000_000_000; // 1 second
    var elapsed: u64 = 0;
    
    while (elapsed < test_duration_ns) {
        std.time.sleep(fixed_interval_ns);
        wake_ups += 1;
        elapsed = @as(u64, @intCast(std.time.nanoTimestamp() - start_time));
    }
    
    const actual_duration_ms = elapsed / 1_000_000;
    const expected_wake_ups = test_duration_ns / fixed_interval_ns;
    
    std.debug.print("Duration: {}ms\n", .{actual_duration_ms});
    std.debug.print("Actual wake-ups: {}\n", .{wake_ups});
    std.debug.print("Expected wake-ups: {}\n", .{expected_wake_ups});
    std.debug.print("Wake-up frequency: {:.1} Hz\n", .{@as(f64, @floatFromInt(wake_ups)) / (@as(f64, @floatFromInt(actual_duration_ms)) / 1000.0)});
    
    // Show the overhead this creates
    const overhead_pct = (@as(f64, @floatFromInt(wake_ups)) * 0.001) / (@as(f64, @floatFromInt(actual_duration_ms)) / 1000.0) * 100.0;
    std.debug.print("Estimated overhead: {:.2}% of CPU time\n", .{overhead_pct});
    
    std.debug.print("\n=== Adaptive Heartbeat Benefit Analysis ===\n", .{});
    std.debug.print("Current fixed approach wakes up ~10,000 times per second\n", .{});
    std.debug.print("Adaptive approach could reduce this by 50-90% during idle periods\n", .{});
    std.debug.print("Potential power savings: significant on mobile/battery devices\n", .{});
}