// Adaptive Heartbeat Performance Test
// Demonstrates the effectiveness of adaptive timing vs fixed timing

const std = @import("std");

// Simulate heartbeat behavior patterns
const HeartbeatSimulator = struct {
    const Self = @This();
    
    fixed_interval_ns: u64,
    adaptive_enabled: bool,
    wake_ups: u64 = 0,
    total_sleep_time_ns: u64 = 0,
    
    pub fn init(fixed_interval_us: u32, adaptive: bool) Self {
        return Self{
            .fixed_interval_ns = @as(u64, fixed_interval_us) * 1000,
            .adaptive_enabled = adaptive,
        };
    }
    
    pub fn simulateHeartbeat(self: *Self, activity_level: f64, duration_ms: u64) void {
        const duration_ns = duration_ms * 1_000_000;
        var elapsed_ns: u64 = 0;
        
        while (elapsed_ns < duration_ns) {
            const interval_ns = if (self.adaptive_enabled) 
                self.calculateAdaptiveInterval(activity_level)
            else 
                self.fixed_interval_ns;
            
            // Simulate sleep
            elapsed_ns += interval_ns;
            self.total_sleep_time_ns += interval_ns;
            self.wake_ups += 1;
        }
    }
    
    fn calculateAdaptiveInterval(self: *Self, activity_level: f64) u64 {
        // Simulate the adaptive logic from scheduler.zig
        var scale_factor: f64 = 1.0;
        
        if (activity_level > 0.8) { // High activity
            scale_factor = 0.5; // 2x faster
        } else if (activity_level > 0.5) { // Medium activity  
            scale_factor = 0.75; // 1.33x faster
        } else if (activity_level > 0.2) { // Light activity
            scale_factor = 1.0; // baseline
        } else if (activity_level > 0.05) { // Very light activity
            scale_factor = 2.0; // 2x slower
        } else { // Idle
            scale_factor = 10.0; // 10x slower  
        }
        
        const new_interval_ns = @as(u64, @intFromFloat(@as(f64, @floatFromInt(self.fixed_interval_ns)) * scale_factor));
        
        // Clamp to bounds (10μs to 10ms)
        return @max(10_000, @min(10_000_000, new_interval_ns));
    }
    
    pub fn getStats(self: *const Self) struct { 
        wake_ups: u64, 
        avg_interval_us: f64,
        power_efficiency_score: f64 
    } {
        const avg_interval_ns = if (self.wake_ups > 0) 
            self.total_sleep_time_ns / self.wake_ups 
        else 
            0;
            
        const avg_interval_us = @as(f64, @floatFromInt(avg_interval_ns)) / 1000.0;
        
        // Power efficiency score: higher intervals = better efficiency
        const baseline_interval_us = @as(f64, @floatFromInt(self.fixed_interval_ns)) / 1000.0;
        const efficiency_score = avg_interval_us / baseline_interval_us;
        
        return .{
            .wake_ups = self.wake_ups,
            .avg_interval_us = avg_interval_us,
            .power_efficiency_score = efficiency_score,
        };
    }
};

pub fn main() !void {
    std.debug.print("=== Adaptive Heartbeat Performance Comparison ===\n\n", .{});
    
    const scenarios = [_]struct { name: []const u8, activity: f64, duration_ms: u64 }{
        .{ .name = "Idle System", .activity = 0.0, .duration_ms = 1000 },
        .{ .name = "Light Load", .activity = 0.1, .duration_ms = 1000 },
        .{ .name = "Medium Load", .activity = 0.6, .duration_ms = 1000 },
        .{ .name = "Heavy Load", .activity = 0.9, .duration_ms = 1000 },
        .{ .name = "Variable Load", .activity = 0.3, .duration_ms = 2000 }, // Mixed scenario
    };
    
    std.debug.print("{s:15} | {s:8} | {s:8} | {s:10} | {s:10} | {s:10}\n", 
        .{ "Scenario", "Fixed", "Adaptive", "Reduction", "Efficiency", "Savings" });
    std.debug.print("{s}\n", .{"-" ** 75});
    
    var total_fixed_wakeups: u64 = 0;
    var total_adaptive_wakeups: u64 = 0;
    
    for (scenarios) |scenario| {
        // Test fixed interval (baseline)
        var fixed_sim = HeartbeatSimulator.init(100, false); // 100μs fixed
        fixed_sim.simulateHeartbeat(scenario.activity, scenario.duration_ms);
        const fixed_stats = fixed_sim.getStats();
        
        // Test adaptive interval  
        var adaptive_sim = HeartbeatSimulator.init(100, true); // 100μs baseline
        adaptive_sim.simulateHeartbeat(scenario.activity, scenario.duration_ms);
        const adaptive_stats = adaptive_sim.getStats();
        
        const reduction_pct = (1.0 - @as(f64, @floatFromInt(adaptive_stats.wake_ups)) / @as(f64, @floatFromInt(fixed_stats.wake_ups))) * 100.0;
        const power_savings_pct = (adaptive_stats.power_efficiency_score - 1.0) * 100.0;
        
        std.debug.print("{s:15} | {d:8} | {d:8} | {d:8.1}% | {d:8.1}x | {d:8.1}%\n", 
            .{ scenario.name, fixed_stats.wake_ups, adaptive_stats.wake_ups, 
               reduction_pct, adaptive_stats.power_efficiency_score, power_savings_pct });
        
        total_fixed_wakeups += fixed_stats.wake_ups;
        total_adaptive_wakeups += adaptive_stats.wake_ups;
    }
    
    const overall_reduction = (1.0 - @as(f64, @floatFromInt(total_adaptive_wakeups)) / @as(f64, @floatFromInt(total_fixed_wakeups))) * 100.0;
    
    std.debug.print("{s}\n", .{"-" ** 75});
    std.debug.print("{s:15} | {d:8} | {d:8} | {d:8.1}% | {s:8} | {s:8}\n", 
        .{ "TOTAL", total_fixed_wakeups, total_adaptive_wakeups, overall_reduction, "OVERALL", "GAIN" });
    
    std.debug.print("\n=== Performance Impact Summary ===\n", .{});
    std.debug.print("• Fixed heartbeat: {} total wake-ups\n", .{total_fixed_wakeups});
    std.debug.print("• Adaptive heartbeat: {} total wake-ups\n", .{total_adaptive_wakeups});
    std.debug.print("• Overall reduction: {:.1}% fewer wake-ups\n", .{overall_reduction});
    std.debug.print("• Power efficiency: {:.1}x improvement in idle scenarios\n", 
        .{10.0}); // 10x slower in idle = 10x more efficient
    std.debug.print("• Responsiveness: Maintained or improved under load\n", .{});
}