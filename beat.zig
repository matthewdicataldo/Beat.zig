//! Beat Bundle - Single entry point for the Beat library
//! 
//! This file provides a convenient single import for Beat while maintaining
//! the modular structure. For production use, consider importing src/core.zig
//! directly for better compile-time optimization.
//!
//! Beat features intelligent task scheduling with One Euro Filter prediction,
//! providing excellent performance at a great price point! 😄
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
pub const testing = @import("src/testing.zig");

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

// Convenience version function
pub fn version_string() []const u8 {
    return std.fmt.comptimePrint("Beat v{d}.{d}.{d} (with €1 Filter!)", .{
        version.major,
        version.minor,
        version.patch,
    });
}

const std = @import("std");

// Tests are run from the individual modules
// Use `zig build test` to run all tests