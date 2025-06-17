const std = @import("std");

pub fn main() !void {
    std.debug.print("Beat.zig Benchmark Suite\n", .{});
    std.debug.print("========================\n\n", .{});
    
    std.debug.print("Available benchmarks:\n", .{});
    std.debug.print("- zig build bench-cache-alignment\n", .{});
    std.debug.print("- zig build bench-prefetching\n", .{});
    std.debug.print("- zig build bench-batch-formation\n", .{});
    std.debug.print("- zig build bench-lockfree-contention\n", .{});
    std.debug.print("- zig build bench-worker-selection\n", .{});
    std.debug.print("- zig build bench-worker-selection-optimized\n", .{});
    std.debug.print("- zig build bench-coz\n", .{});
    
    if (@import("builtin").mode == .Debug) {
        std.debug.print("\nNote: Running in debug mode. Use --release-fast for accurate benchmarks.\n", .{});
    }
    
    std.debug.print("\nMain benchmark functionality moved to individual specialized benchmarks.\n", .{});
    std.debug.print("Use the commands above to run specific performance tests.\n", .{});
}