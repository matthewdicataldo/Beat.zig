const std = @import("std");
const beat = @import("src/core.zig");

// Comprehensive demonstration of Comptime Work Distribution Patterns

test "Comptime Work Distribution - Complete Demo" {
    std.debug.print("\n============================================================\n", .{});
    std.debug.print("Beat.zig Comptime Work Distribution Patterns Demo\n", .{});
    std.debug.print("============================================================\n", .{});
    
    const allocator = std.testing.allocator;
    const pool = try beat.createOptimalPool(allocator);
    defer pool.deinit();
    
    // 1. Compile-Time Work Analysis
    std.debug.print("\n1. Compile-Time Work Analysis:\n", .{});
    
    // Analyze different work patterns at compile time
    const small_work = beat.comptime_work.analyzeWork(f32, 100, 4);
    const medium_work = beat.comptime_work.analyzeWork(f32, 10000, 4);
    const large_work = beat.comptime_work.analyzeWork(f32, 1000000, 4);
    
    std.debug.print("   Small Work (100 f32s):\n", .{});
    std.debug.print("     Strategy: {s}\n", .{@tagName(small_work.strategy)});
    std.debug.print("     Chunk Size: {}\n", .{small_work.chunk_size});
    std.debug.print("     Workers: {}\n", .{small_work.worker_count});
    
    std.debug.print("   Medium Work (10K f32s):\n", .{});
    std.debug.print("     Strategy: {s}\n", .{@tagName(medium_work.strategy)});
    std.debug.print("     Chunk Size: {}\n", .{medium_work.chunk_size});
    std.debug.print("     Workers: {}\n", .{medium_work.worker_count});
    
    std.debug.print("   Large Work (1M f32s):\n", .{});
    std.debug.print("     Strategy: {s}\n", .{@tagName(large_work.strategy)});
    std.debug.print("     Chunk Size: {}\n", .{large_work.chunk_size});
    std.debug.print("     Workers: {}\n", .{large_work.worker_count});
    std.debug.print("     SIMD Width: {}\n", .{large_work.simd_width});
    
    // 2. Automatic Work Distribution
    std.debug.print("\n2. Automatic Work Distribution Example:\n", .{});
    
    // Create test data
    const data_size = 1000;
    const test_data = try allocator.alloc(f32, data_size);
    defer allocator.free(test_data);
    
    // Initialize with test values
    for (test_data, 0..) |*item, i| {
        item.* = @as(f32, @floatFromInt(i));
    }
    
    // Demonstrate automatic work distribution
    const Distributor = beat.comptime_work.WorkDistributor(f32, data_size, 4);
    Distributor.printSummary();
    
    std.debug.print("   Generated {} chunks for {} workers\n", .{ Distributor.total_chunks, Distributor.worker_count });
    
    // Show first few chunks
    std.debug.print("   First 3 chunks:\n", .{});
    for (Distributor.chunks[0..@min(3, Distributor.chunks.len)]) |chunk| {
        std.debug.print("     Worker {}: [{}, {}) size={} simd={}\n", .{
            chunk.worker_id, chunk.start, chunk.end, chunk.size(), chunk.simd_aligned
        });
    }
    
    // 3. Parallel Map Operation
    std.debug.print("\n3. Parallel Map Operation:\n", .{});
    
    const map_input = try allocator.alloc(f32, 100);
    defer allocator.free(map_input);
    const map_output = try allocator.alloc(f32, 100);
    defer allocator.free(map_output);
    
    // Initialize input
    for (map_input, 0..) |*item, i| {
        item.* = @as(f32, @floatFromInt(i));
    }
    
    // Define map function: square each element
    const square = struct {
        fn apply(x: f32) f32 {
            return x * x;
        }
    }.apply;
    
    // Execute parallel map
    try beat.comptime_work.parallelMap(f32, f32, pool, map_input, map_output, square);
    
    // Verify results
    var correct: usize = 0;
    for (map_input, map_output) |input, output| {
        const expected = input * input;
        if (output == expected) correct += 1;
    }
    
    std.debug.print("   Mapped {} elements, {} correct\n", .{ map_input.len, correct });
    std.debug.print("   Sample: {d:.1} -> {d:.1}, {d:.1} -> {d:.1}\n", .{
        map_input[0], map_output[0], map_input[10], map_output[10]
    });
    
    // 4. Parallel Reduce Operation
    std.debug.print("\n4. Parallel Reduce Operation:\n", .{});
    
    const reduce_data = try allocator.alloc(i32, 1000);
    defer allocator.free(reduce_data);
    
    // Initialize with numbers 1 to 1000
    for (reduce_data, 0..) |*item, i| {
        item.* = @as(i32, @intCast(i + 1));
    }
    
    // Define reduce function: sum
    const add = struct {
        fn apply(a: i32, b: i32) i32 {
            return a + b;
        }
    }.apply;
    
    // Execute parallel reduce
    const sum = try beat.comptime_work.parallelReduce(i32, pool, allocator, reduce_data, add, 0);
    const expected_sum = (1000 * 1001) / 2; // Sum of 1 to 1000
    
    std.debug.print("   Reduced {} elements to sum: {}\n", .{ reduce_data.len, sum });
    std.debug.print("   Expected sum: {}, Correct: {}\n", .{ expected_sum, sum == expected_sum });
    
    // 5. Parallel Filter Operation  
    std.debug.print("\n5. Parallel Filter Operation:\n", .{});
    
    const filter_data = try allocator.alloc(i32, 100);
    defer allocator.free(filter_data);
    
    // Initialize with numbers 0 to 99
    for (filter_data, 0..) |*item, i| {
        item.* = @as(i32, @intCast(i));
    }
    
    // Define filter function: even numbers only
    const is_even = struct {
        fn apply(x: i32) bool {
            return @rem(x, 2) == 0;
        }
    }.apply;
    
    // Execute parallel filter
    const filtered = try beat.comptime_work.parallelFilter(i32, pool, allocator, filter_data, is_even);
    defer allocator.free(filtered);
    
    std.debug.print("   Filtered {} elements down to {} even numbers\n", .{ filter_data.len, filtered.len });
    std.debug.print("   First 5 results: ", .{});
    for (filtered[0..@min(5, filtered.len)]) |item| {
        std.debug.print("{} ", .{item});
    }
    std.debug.print("\n", .{});
    
    // 6. SIMD-Aware Distribution
    std.debug.print("\n6. SIMD-Aware Work Distribution:\n", .{});
    
    if (beat.build_opts.cpu_features.has_simd) {
        const SimdDist = beat.comptime_work.SimdDistributor(f32);
        std.debug.print("   SIMD Width: {} elements\n", .{SimdDist.vector_width});
        
        const simd_data = try allocator.alloc(f32, 64); // SIMD-friendly size
        defer allocator.free(simd_data);
        
        // Initialize data
        for (simd_data, 0..) |*item, i| {
            item.* = @as(f32, @floatFromInt(i));
        }
        
        // Define SIMD operation: add 1.0 to each element
        const add_one = struct {
            fn apply(vec: SimdDist.Vector) SimdDist.Vector {
                return vec + @as(SimdDist.Vector, @splat(1.0));
            }
        }.apply;
        
        // Execute SIMD operation
        try SimdDist.processSimd(pool, simd_data, add_one);
        
        std.debug.print("   Processed {} elements with SIMD\n", .{simd_data.len});
        std.debug.print("   Sample results: {d:.1} {d:.1} {d:.1}\n", .{
            simd_data[0], simd_data[1], simd_data[2]
        });
        
        // Verify results (should be original value + 1)
        var simd_correct: usize = 0;
        for (simd_data, 0..) |item, i| {
            const expected = @as(f32, @floatFromInt(i)) + 1.0;
            if (item == expected) simd_correct += 1;
        }
        std.debug.print("   SIMD Results correct: {}/{}\n", .{ simd_correct, simd_data.len });
    } else {
        std.debug.print("   SIMD not available on this platform\n", .{});
    }
    
    // 7. Cost Analysis and Strategy Selection
    std.debug.print("\n7. Compile-Time Cost Analysis:\n", .{});
    
    const tiny_analysis = beat.comptime_work.analyzeWork(i32, 10, 4);
    const huge_analysis = beat.comptime_work.analyzeWork(f64, 10_000_000, 8);
    
    std.debug.print("   Tiny Work (10 i32s): {s} strategy\n", .{@tagName(tiny_analysis.strategy)});
    std.debug.print("   Huge Work (10M f64s): {s} strategy\n", .{@tagName(huge_analysis.strategy)});
    std.debug.print("   Cost estimates: {} vs {}\n", .{ tiny_analysis.cost_estimate, huge_analysis.cost_estimate });
    
    // 8. Integration with Auto-Configuration
    std.debug.print("\n8. Integration with Auto-Configuration:\n", .{});
    std.debug.print("   Using {} optimal workers from build-time detection\n", .{beat.build_opts.hardware.optimal_workers});
    std.debug.print("   SIMD width: {} bytes\n", .{beat.build_opts.cpu_features.simd_width});
    std.debug.print("   Auto-configured work distribution adapts to your hardware!\n", .{});
    
    std.debug.print("\n============================================================\n", .{});
    std.debug.print("Comptime Work Distribution Patterns Complete!\n", .{});
    std.debug.print("Zero runtime overhead with compile-time optimization\n", .{});
    std.debug.print("============================================================\n", .{});
}

test "Advanced Work Distribution Patterns" {
    std.debug.print("\nAdvanced Work Distribution Examples:\n", .{});
    
    // 1. Custom Work Distributor
    std.debug.print("\n1. Custom Work Distributor:\n", .{});
    
    const CustomDist = beat.comptime_work.createDistributor(f64, 5000, 6);
    CustomDist.printSummary();
    
    std.debug.print("   Custom distributor created for specific requirements\n", .{});
    
    // 2. Different Data Types and Strategies
    std.debug.print("\n2. Type-Aware Strategy Selection:\n", .{});
    
    const i8_analysis = beat.comptime_work.analyzeWork(i8, 10000, 4);
    const f64_analysis = beat.comptime_work.analyzeWork(f64, 10000, 4);
    const large_struct_analysis = beat.comptime_work.analyzeWork([64]u8, 10000, 4);
    
    std.debug.print("   i8 work: {s} strategy\n", .{@tagName(i8_analysis.strategy)});
    std.debug.print("   f64 work: {s} strategy\n", .{@tagName(f64_analysis.strategy)});
    std.debug.print("   Large struct work: {s} strategy\n", .{@tagName(large_struct_analysis.strategy)});
    
    // 3. Memory Pattern Analysis
    std.debug.print("\n3. Memory Pattern Optimization:\n", .{});
    std.debug.print("   i8 memory pattern: {s}\n", .{@tagName(i8_analysis.memory_pattern)});
    std.debug.print("   f64 memory pattern: {s}\n", .{@tagName(f64_analysis.memory_pattern)});
    std.debug.print("   Large struct memory pattern: {s}\n", .{@tagName(large_struct_analysis.memory_pattern)});
    
    std.debug.print("\nComptime work distribution provides zero-overhead optimization!\n", .{});
}

test "Performance Characteristics" {
    std.debug.print("\nPerformance Characteristics Analysis:\n", .{});
    
    // Compare strategies for different work sizes
    std.debug.print("   Work Size    Strategy          Chunk Size    Workers\n", .{});
    std.debug.print("   ---------    ---------------   ----------    -------\n", .{});
    
    // Use inline for comptime loop
    inline for ([_]comptime_int{ 100, 1_000, 10_000, 100_000, 1_000_000 }) |size| {
        const analysis = beat.comptime_work.analyzeWork(f32, size, 4);
        std.debug.print("   {d:9}    {s:15}   {d:10}    {d:7}\n", .{
            size, @tagName(analysis.strategy), analysis.chunk_size, analysis.worker_count
        });
    }
    
    std.debug.print("\nOptimal strategy selection based on work characteristics!\n", .{});
}