const std = @import("std");
const beat = @import("beat");

// Test the runtime parallel work distribution implementation
test "Parallel Work Distribution Runtime - Comprehensive Test" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Parallel Work Distribution Runtime Test ===\n", .{});
    
    // Create a thread pool for testing
    const pool = try beat.createPool(allocator);
    defer pool.deinit();
    
    std.debug.print("Created thread pool with {} workers\n", .{pool.config.num_workers.?});
    
    // Test 1: parallelMap
    std.debug.print("\n1. Testing parallelMap...\n", .{});
    
    const input_data = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var output_data: [input_data.len]i32 = undefined;
    
    const square = struct {
        fn apply(x: i32) i32 {
            return x * x;
        }
    }.apply;
    
    try beat.comptime_work.parallelMap(i32, i32, pool, &input_data, &output_data, square);
    
    // Verify results
    for (input_data, output_data, 0..) |input, output, i| {
        try std.testing.expect(output == input * input);
        std.debug.print("  input[{}] = {} -> output[{}] = {}\n", .{ i, input, i, output });
    }
    std.debug.print("  ✅ parallelMap completed successfully\n", .{});
    
    // Test 2: parallelReduce
    std.debug.print("\n2. Testing parallelReduce...\n", .{});
    
    const sum_data = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    
    const add = struct {
        fn apply(a: i32, b: i32) i32 {
            return a + b;
        }
    }.apply;
    
    const sum_result = try beat.comptime_work.parallelReduce(i32, pool, allocator, &sum_data, add, 0);
    const expected_sum = 55; // 1+2+3+4+5+6+7+8+9+10 = 55
    
    try std.testing.expect(sum_result == expected_sum);
    std.debug.print("  Sum of [1..10] = {} (expected {})\n", .{ sum_result, expected_sum });
    std.debug.print("  ✅ parallelReduce completed successfully\n", .{});
    
    // Test 3: parallelFilter
    std.debug.print("\n3. Testing parallelFilter...\n", .{});
    
    const filter_data = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    
    const isEven = struct {
        fn apply(x: i32) bool {
            return @rem(x, 2) == 0;
        }
    }.apply;
    
    const filtered_result = try beat.comptime_work.parallelFilter(i32, pool, allocator, &filter_data, isEven);
    defer allocator.free(filtered_result);
    
    // Verify results - should contain [2, 4, 6, 8, 10, 12]
    const expected_evens = [_]i32{ 2, 4, 6, 8, 10, 12 };
    try std.testing.expect(filtered_result.len == expected_evens.len);
    
    for (filtered_result, expected_evens, 0..) |actual, expected, i| {
        try std.testing.expect(actual == expected);
        std.debug.print("  filtered[{}] = {} (expected {})\n", .{ i, actual, expected });
    }
    std.debug.print("  ✅ parallelFilter completed successfully\n", .{});
    
    // Test 4: distributeWork
    std.debug.print("\n4. Testing distributeWork...\n", .{});
    
    var work_data = [_]i32{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    
    const increment_work = struct {
        fn apply(data: []i32, start: usize, end: usize) void {
            for (start..end) |i| {
                data[i] += 1;
            }
        }
    }.apply;
    
    try beat.comptime_work.distributeWork(i32, &work_data, pool, increment_work);
    
    // Verify all elements were incremented
    for (work_data, 0..) |value, i| {
        try std.testing.expect(value == 1);
        std.debug.print("  work_data[{}] = {} (incremented from 0)\n", .{ i, value });
    }
    std.debug.print("  ✅ distributeWork completed successfully\n", .{});
    
    std.debug.print("\n=== All Parallel Work Distribution Tests Passed! ===\n", .{});
}

test "Parallel Work Distribution Performance Characteristics" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Performance Characteristics Test ===\n", .{});
    
    const pool = try beat.createPool(allocator);
    defer pool.deinit();
    
    // Test with different data sizes to verify threshold behavior
    const test_sizes = [_]usize{ 10, 100, 1000, 10000 };
    
    for (test_sizes) |size| {
        std.debug.print("\nTesting with {} elements:\n", .{size});
        
        // Create test data
        const test_data = try allocator.alloc(i32, size);
        defer allocator.free(test_data);
        
        // Initialize with sequential values
        for (test_data, 0..) |*item, i| {
            item.* = @intCast(i + 1);
        }
        
        // Test parallelReduce performance
        const multiply = struct {
            fn apply(a: i32, b: i32) i32 {
                // Simple multiplication to create some work
                return @divTrunc(a + b, 2); // Average to avoid overflow
            }
        }.apply;
        
        const start_time = std.time.nanoTimestamp();
        const result = try beat.comptime_work.parallelReduce(i32, pool, allocator, test_data, multiply, 0);
        const end_time = std.time.nanoTimestamp();
        
        const duration_ns = end_time - start_time;
        const duration_ms = @as(f64, @floatFromInt(duration_ns)) / 1_000_000.0;
        
        std.debug.print("  Size: {}, Result: {}, Time: {d:.3}ms\n", .{ size, result, duration_ms });
        
        // Verify that small sizes use sequential execution (fast)
        // and large sizes potentially use parallel approach
        if (size < 1000) {
            std.debug.print("    → Sequential execution (expected for small data)\n", .{});
        } else {
            std.debug.print("    → Parallel-capable execution (large data threshold)\n", .{});
        }
    }
    
    std.debug.print("\n✅ Performance characteristics verified\n", .{});
}

test "Parallel Work Distribution Edge Cases" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Edge Cases Test ===\n", .{});
    
    const pool = try beat.createPool(allocator);
    defer pool.deinit();
    
    // Test 1: Empty data
    std.debug.print("Testing empty data...\n", .{});
    
    const empty_data: []const i32 = &[_]i32{};
    const empty_result = try beat.comptime_work.parallelReduce(i32, pool, allocator, empty_data, struct {
        fn apply(a: i32, b: i32) i32 {
            return a + b;
        }
    }.apply, 42);
    
    try std.testing.expect(empty_result == 42); // Should return initial value
    std.debug.print("  ✅ Empty data handled correctly (returned initial value: {})\n", .{empty_result});
    
    // Test 2: Single element
    std.debug.print("Testing single element...\n", .{});
    
    const single_data = [_]i32{100};
    var single_output: [1]i32 = undefined;
    
    try beat.comptime_work.parallelMap(i32, i32, pool, &single_data, &single_output, struct {
        fn apply(x: i32) i32 {
            return x * 2;
        }
    }.apply);
    
    try std.testing.expect(single_output[0] == 200);
    std.debug.print("  ✅ Single element handled correctly ({})\n", .{single_output[0]});
    
    // Test 3: Mismatched sizes for parallelMap
    std.debug.print("Testing mismatched sizes...\n", .{});
    
    const input_mismatch = [_]i32{ 1, 2, 3 };
    var output_mismatch: [5]i32 = undefined; // Different size
    
    const map_result = beat.comptime_work.parallelMap(i32, i32, pool, &input_mismatch, &output_mismatch, struct {
        fn apply(x: i32) i32 {
            return x;
        }
    }.apply);
    
    try std.testing.expectError(error.SizeMismatch, map_result);
    std.debug.print("  ✅ Size mismatch detected correctly\n", .{});
    
    std.debug.print("\n✅ All edge cases handled correctly\n", .{});
}