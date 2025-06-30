const std = @import("std");

pub fn main() !void {
    std.debug.print("Hello from Zig!\n", .{});
    
    // Test basic JSON parsing
    const json_str = "{ \"test\": 42 }";
    var parsed = try std.json.parseFromSlice(std.json.Value, std.heap.page_allocator, json_str, .{});
    defer parsed.deinit();
    
    std.debug.print("JSON test passed!\n", .{});
}