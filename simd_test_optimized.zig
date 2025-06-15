const std = @import("std");

// Test different SIMD approaches to find the most efficient one

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    const size = 1024;
    const iterations = 1000;
    
    // Test 1: Current approach (load vector, store scalar)
    {
        const data = try allocator.alloc(f32, size);
        defer allocator.free(data);
        
        for (data, 0..) |*value, i| {
            value.* = @as(f32, @floatFromInt(i)) * 0.1;
        }
        
        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            const vector_width = 8;
            const VectorType = @Vector(vector_width, f32);
            const aligned_size = (size / vector_width) * vector_width;
            
            for (0..aligned_size / vector_width) |chunk| {
                const base_idx = chunk * vector_width;
                const vec_data: VectorType = data[base_idx..base_idx + vector_width][0..vector_width].*;
                const vec_result = vec_data * @as(VectorType, @splat(1.5)) + @as(VectorType, @splat(0.5));
                // Scalar storage
                for (0..vector_width) |lane| {
                    data[base_idx + lane] = vec_result[lane];
                }
            }
        }
        const end = std.time.nanoTimestamp();
        const time1 = @as(u64, @intCast(end - start));
        std.debug.print("Approach 1 (vector load + scalar store): {} ns\n", .{time1});
    }
    
    // Test 2: Bulk memory copy approach 
    {
        const data = try allocator.alloc(f32, size);
        defer allocator.free(data);
        
        for (data, 0..) |*value, i| {
            value.* = @as(f32, @floatFromInt(i)) * 0.1;
        }
        
        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            const vector_width = 8;
            const VectorType = @Vector(vector_width, f32);
            const aligned_size = (size / vector_width) * vector_width;
            
            for (0..aligned_size / vector_width) |chunk| {
                const base_idx = chunk * vector_width;
                const vec_data: VectorType = data[base_idx..base_idx + vector_width][0..vector_width].*;
                const vec_result = vec_data * @as(VectorType, @splat(1.5)) + @as(VectorType, @splat(0.5));
                // Bulk copy
                const result_array: [vector_width]f32 = vec_result;
                @memcpy(data[base_idx..base_idx + vector_width], &result_array);
            }
        }
        const end = std.time.nanoTimestamp();
        const time2 = @as(u64, @intCast(end - start));
        std.debug.print("Approach 2 (vector load + memcpy store): {} ns\n", .{time2});
    }
    
    // Test 3: Pure scalar baseline
    {
        const data = try allocator.alloc(f32, size);
        defer allocator.free(data);
        
        for (data, 0..) |*value, i| {
            value.* = @as(f32, @floatFromInt(i)) * 0.1;
        }
        
        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            for (data) |*value| {
                value.* = value.* * 1.5 + 0.5;
            }
        }
        const end = std.time.nanoTimestamp();
        const time3 = @as(u64, @intCast(end - start));
        std.debug.print("Approach 3 (pure scalar): {} ns\n", .{time3});
    }
    
    // Test 4: Unrolled scalar (manual vectorization)
    {
        const data = try allocator.alloc(f32, size);
        defer allocator.free(data);
        
        for (data, 0..) |*value, i| {
            value.* = @as(f32, @floatFromInt(i)) * 0.1;
        }
        
        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            const unroll_factor = 8;
            const aligned_size = (size / unroll_factor) * unroll_factor;
            
            var i: usize = 0;
            while (i < aligned_size) {
                data[i] = data[i] * 1.5 + 0.5;
                data[i+1] = data[i+1] * 1.5 + 0.5;
                data[i+2] = data[i+2] * 1.5 + 0.5;
                data[i+3] = data[i+3] * 1.5 + 0.5;
                data[i+4] = data[i+4] * 1.5 + 0.5;
                data[i+5] = data[i+5] * 1.5 + 0.5;
                data[i+6] = data[i+6] * 1.5 + 0.5;
                data[i+7] = data[i+7] * 1.5 + 0.5;
                i += unroll_factor;
            }
        }
        const end = std.time.nanoTimestamp();
        const time4 = @as(u64, @intCast(end - start));
        std.debug.print("Approach 4 (unrolled scalar): {} ns\n", .{time4});
    }
}