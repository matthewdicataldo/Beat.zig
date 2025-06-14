//! Beat Bundle - Single entry point for the Beat library
//! 
//! This file provides a convenient single import for Beat while maintaining
//! the modular structure. For production use, consider importing src/core.zig
//! directly for better compile-time optimization.
//!
//! Example usage:
//!   const beat = @import("beat.zig");
//!   const pool = try beat.createPool(allocator);

// Re-export all modules
pub const lockfree = @import("src/lockfree.zig");
pub const topology = @import("src/topology.zig");
pub const memory = @import("src/memory.zig");
pub const scheduler = @import("src/scheduler.zig");
pub const pcall = @import("src/pcall.zig");
pub const coz = @import("src/coz.zig");

// Import core for main types
const core = @import("src/core.zig");

// Re-export version info
pub const version = core.version;
pub const cache_line_size = core.cache_line_size;

// Re-export configuration
pub const Config = core.Config;

// Re-export core types
pub const Priority = core.Priority;
pub const TaskStatus = core.TaskStatus;
pub const TaskError = core.TaskError;
pub const Task = core.Task;
pub const ThreadPoolStats = core.ThreadPoolStats;
pub const ThreadPool = core.ThreadPool;

// Re-export main API functions
pub const createPool = core.createPool;
pub const createPoolWithConfig = core.createPoolWithConfig;

// Convenience test function
pub fn version_string() []const u8 {
    return std.fmt.comptimePrint("ZigPulse v{d}.{d}.{d}", .{
        version.major,
        version.minor,
        version.patch,
    });
}

const std = @import("std");

// Tests are run from the individual modules
// Use `zig build test` to run all tests