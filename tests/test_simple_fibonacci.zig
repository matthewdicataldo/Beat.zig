const std = @import("std");

fn fib(n: u32) u64 {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}

pub fn main() !void {
    const result = fib(35);
    std.debug.print("Fibonacci(35) = {}\n", .{result});
    
    const start = std.time.nanoTimestamp();
    _ = fib(35);
    const end = std.time.nanoTimestamp();
    
    const duration_ms = (end - start) / 1_000_000;
    std.debug.print("Sequential time: {}ms\n", .{duration_ms});
}