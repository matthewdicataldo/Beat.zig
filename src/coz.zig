const std = @import("std");
const builtin = @import("builtin");

// ===== COZ Profiling Support =====
// Following the same pattern as Reverb's integration

// Export functions that COZ can intercept
pub export fn coz_progress_begin(name: [*:0]const u8) void {
    _ = name;
    // COZ will replace this with instrumentation
}

pub export fn coz_progress_end(name: [*:0]const u8) void {
    _ = name;
    // COZ will replace this with instrumentation
}

// Export progress function for throughput measurements
pub export fn coz_progress_named(name: [*:0]const u8) void {
    _ = name;
    // COZ will replace this with instrumentation
}

// COZ backend with compile-time optimization
pub const Backend = if (builtin.mode == .Debug or builtin.mode == .ReleaseSafe) struct {
    // Track throughput at specific points
    pub fn throughput(comptime name: []const u8) void {
        const COUNTER = struct {
            var count: usize = 0;
        };
        _ = @atomicRmw(usize, &COUNTER.count, .Add, 1, .monotonic);
        
        // Call COZ progress function with null-terminated name
        const null_terminated = name ++ "\x00";
        coz_progress_named(@ptrCast(null_terminated));
    }

    // Mark latency region begin
    pub fn latencyBegin(comptime name: []const u8) void {
        const null_terminated = name ++ "\x00";
        coz_progress_begin(@ptrCast(null_terminated));
    }

    // Mark latency region end
    pub fn latencyEnd(comptime name: []const u8) void {
        const null_terminated = name ++ "\x00";
        coz_progress_end(@ptrCast(null_terminated));
    }
} else struct {
    // No-op implementations for release builds
    pub inline fn throughput(comptime name: []const u8) void {
        _ = name;
    }
    
    pub inline fn latencyBegin(comptime name: []const u8) void {
        _ = name;
    }
    
    pub inline fn latencyEnd(comptime name: []const u8) void {
        _ = name;
    }
};

// Convenience aliases
pub const throughput = Backend.throughput;
pub const latencyBegin = Backend.latencyBegin;
pub const latencyEnd = Backend.latencyEnd;

// Common progress point names for ZigPulse
pub const Points = struct {
    // Task lifecycle
    pub const task_submitted = "zigpulse_task_submitted";
    pub const task_completed = "zigpulse_task_completed";
    pub const task_stolen = "zigpulse_task_stolen";
    pub const task_execution = "zigpulse_task_execution";
    
    // Queue operations
    pub const queue_push = "zigpulse_queue_push";
    pub const queue_pop = "zigpulse_queue_pop";
    pub const queue_steal = "zigpulse_queue_steal";
    
    // Memory operations
    pub const memory_alloc = "zigpulse_memory_alloc";
    pub const memory_free = "zigpulse_memory_free";
    
    // Worker activity
    pub const worker_idle = "zigpulse_worker_idle";
    pub const worker_busy = "zigpulse_worker_busy";
    
    // Scheduling decisions
    pub const schedule_decision = "zigpulse_schedule_decision";
    pub const topology_migration = "zigpulse_topology_migration";
};