// Main examples entry point
// This file provides easy access to all Beat.zig examples

const std = @import("std");

pub fn main() !void {
    std.log.info("Beat.zig Examples", .{});
    std.log.info("==================", .{});
    std.log.info("", .{});
    std.log.info("Available examples:", .{});
    std.log.info("  - Basic Usage: examples/basic_usage.zig", .{});
    std.log.info("  - Modular Usage: examples/modular_usage.zig", .{});
    std.log.info("  - Single File Usage: examples/single_file_usage.zig", .{});
    std.log.info("  - Comprehensive Demo: examples/comprehensive_demo.zig", .{});
    std.log.info("", .{});
    std.log.info("Run individual examples with:", .{});
    std.log.info("  zig build example-modular", .{});
    std.log.info("  zig build example-bundle", .{});
    std.log.info("", .{});
    std.log.info("Check the examples/ directory for more detailed examples!", .{});
}