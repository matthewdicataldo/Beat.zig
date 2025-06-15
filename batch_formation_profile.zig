const std = @import("std");
const beat = @import("beat");

// Minimal isolated test to profile batch formation bottlenecks

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    std.debug.print("=== Batch Formation Profiling ===\n", .{});
    
    // Test 1: Create simple tasks
    const TestData = struct { values: [64]f32 };
    var test_data_array: [15]TestData = undefined;
    
    var tasks: [15]beat.Task = undefined;
    for (&tasks, 0..) |*task, i| {
        task.* = beat.Task{
            .func = struct {
                fn func(data: *anyopaque) void {
                    const typed_data = @as(*TestData, @ptrCast(@alignCast(data)));
                    for (&typed_data.values) |*value| {
                        value.* = value.* * 1.2 + 0.3;
                    }
                }
            }.func,
            .data = @ptrCast(&test_data_array[i]),
            .priority = .normal,
            .data_size_hint = @sizeOf(TestData),
        };
    }
    
    // Test 2: Time just the batch former creation
    {
        const start = std.time.nanoTimestamp();
        const criteria = beat.simd_classifier.BatchFormationCriteria.performanceOptimized();
        var batch_former = beat.simd_classifier.IntelligentBatchFormer.init(allocator, criteria);
        defer batch_former.deinit();
        const end = std.time.nanoTimestamp();
        const creation_time = @as(u64, @intCast(end - start));
        std.debug.print("Batch former creation: {} ns ({:.2} μs)\n", .{ creation_time, @as(f64, @floatFromInt(creation_time)) / 1000.0 });
    }
    
    // Test 3: Detailed task addition profiling
    {
        const criteria = beat.simd_classifier.BatchFormationCriteria.performanceOptimized();
        var batch_former = beat.simd_classifier.IntelligentBatchFormer.init(allocator, criteria);
        defer batch_former.deinit();
        
        // Profile individual task additions
        var individual_times: [15]u64 = undefined;
        
        for (tasks, 0..) |task, i| {
            const start = std.time.nanoTimestamp();
            batch_former.addTask(task, false) catch unreachable;
            const end = std.time.nanoTimestamp();
            individual_times[i] = @as(u64, @intCast(end - start));
        }
        
        var total_time: u64 = 0;
        for (individual_times, 0..) |time, i| {
            total_time += time;
            std.debug.print("  Task {}: {} ns ({:.2} μs)\n", .{ i + 1, time, @as(f64, @floatFromInt(time)) / 1000.0 });
        }
        
        std.debug.print("Task addition total: {} ns ({:.2} μs)\n", .{ total_time, @as(f64, @floatFromInt(total_time)) / 1000.0 });
        std.debug.print("Average per task: {} ns ({:.2} μs)\n", .{ total_time / 15, @as(f64, @floatFromInt(total_time / 15)) / 1000.0 });
    }
    
    // Test 4: Time batch formation only
    {
        const criteria = beat.simd_classifier.BatchFormationCriteria.performanceOptimized();
        var batch_former = beat.simd_classifier.IntelligentBatchFormer.init(allocator, criteria);
        defer batch_former.deinit();
        
        for (tasks) |task| {
            batch_former.addTask(task, false) catch unreachable;
        }
        
        const start = std.time.nanoTimestamp();
        batch_former.attemptBatchFormation() catch unreachable;
        const end = std.time.nanoTimestamp();
        const formation_time = @as(u64, @intCast(end - start));
        std.debug.print("Batch formation only: {} ns ({:.2} μs)\n", .{ formation_time, @as(f64, @floatFromInt(formation_time)) / 1000.0 });
    }
    
    // Test 5: Time complete workflow
    {
        const start = std.time.nanoTimestamp();
        
        const criteria = beat.simd_classifier.BatchFormationCriteria.performanceOptimized();
        var batch_former = beat.simd_classifier.IntelligentBatchFormer.init(allocator, criteria);
        defer batch_former.deinit();
        
        for (tasks) |task| {
            batch_former.addTask(task, false) catch unreachable;
        }
        
        batch_former.attemptBatchFormation() catch unreachable;
        
        const end = std.time.nanoTimestamp();
        const total_time = @as(u64, @intCast(end - start));
        std.debug.print("Complete workflow: {} ns ({:.2} μs)\n", .{ total_time, @as(f64, @floatFromInt(total_time)) / 1000.0 });
    }
    
    // Test 6: Time repeated formation (separate instances)
    {
        const iterations = 10;
        const start = std.time.nanoTimestamp();
        
        for (0..iterations) |_| {
            const criteria = beat.simd_classifier.BatchFormationCriteria.performanceOptimized();
            var batch_former = beat.simd_classifier.IntelligentBatchFormer.init(allocator, criteria);
            defer batch_former.deinit();
            
            // Add tasks
            for (tasks) |task| {
                batch_former.addTask(task, false) catch unreachable;
            }
            
            // Form batches
            batch_former.attemptBatchFormation() catch unreachable;
        }
        
        const end = std.time.nanoTimestamp();
        const total_time = @as(u64, @intCast(end - start));
        const avg_time = total_time / iterations;
        std.debug.print("Average per iteration (10x): {} ns ({:.2} μs)\n", .{ avg_time, @as(f64, @floatFromInt(avg_time)) / 1000.0 });
    }
}