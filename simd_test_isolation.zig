const std = @import("std");

// Isolated SIMD performance test to understand the root issue

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    const size = 1024;
    const data = try allocator.alloc(f32, size);
    defer allocator.free(data);
    
    // Initialize data
    for (data, 0..) |*value, i| {
        value.* = @as(f32, @floatFromInt(i)) * 0.1;
    }
    
    const iterations = 1000;
    
    // Test 1: Pure scalar
    var scalar_times: [10]u64 = undefined;
    for (&scalar_times) |*time| {
        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            for (data) |*value| {
                value.* = value.* * 1.5 + 0.5;
            }
        }
        const end = std.time.nanoTimestamp();
        time.* = @as(u64, @intCast(end - start));
    }
    
    // Reset data
    for (data, 0..) |*value, i| {
        value.* = @as(f32, @floatFromInt(i)) * 0.1;
    }
    
    // Test 2: SIMD with current implementation
    var simd_times: [10]u64 = undefined;
    for (&simd_times) |*time| {
        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            const vector_width = 8;
            const VectorType = @Vector(vector_width, f32);
            const aligned_size = (size / vector_width) * vector_width;
            
            for (0..aligned_size / vector_width) |chunk| {
                const base_idx = chunk * vector_width;
                const vec_data: VectorType = data[base_idx..base_idx + vector_width][0..vector_width].*;
                const vec_result = vec_data * @as(VectorType, @splat(1.5)) + @as(VectorType, @splat(0.5));
                const result_array: [vector_width]f32 = vec_result;
                @memcpy(data[base_idx..base_idx + vector_width], &result_array);
            }
            
            // Handle remaining elements
            for (aligned_size..size) |i| {
                data[i] = data[i] * 1.5 + 0.5;
            }
        }
        const end = std.time.nanoTimestamp();
        time.* = @as(u64, @intCast(end - start));
    }
    
    // Calculate averages
    var scalar_avg: u64 = 0;
    var simd_avg: u64 = 0;
    
    for (scalar_times) |time| scalar_avg += time;
    for (simd_times) |time| simd_avg += time;
    
    scalar_avg /= scalar_times.len;
    simd_avg /= simd_times.len;
    
    const speedup = @as(f64, @floatFromInt(scalar_avg)) / @as(f64, @floatFromInt(simd_avg));
    
    std.debug.print("=== Isolated SIMD Performance Test ===\n", .{});
    std.debug.print("Size: {} elements, {} iterations\n", .{ size, iterations });
    std.debug.print("Scalar average: {} ns\n", .{scalar_avg});
    std.debug.print("SIMD average: {} ns\n", .{simd_avg});
    std.debug.print("Speedup: {d:.2}x\n", .{speedup});
    
    if (speedup < 1.5) {
        std.debug.print("❌ SIMD performance is poor (<1.5x speedup)\n", .{});
        std.debug.print("This confirms our SIMD implementation needs optimization\n", .{});
    } else if (speedup < 3.0) {
        std.debug.print("⚠️  SIMD performance is moderate (1.5-3x speedup)\n", .{});
    } else {
        std.debug.print("✅ SIMD performance is good (>3x speedup)\n", .{});
    }
}