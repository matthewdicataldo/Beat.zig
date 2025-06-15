const std = @import("std");
const beat = @import("beat");

// Ultra-specific test to isolate the template issue

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    std.debug.print("=== Template Debug ===\n", .{});
    
    // Test 1: Just template creation
    {
        const start = std.time.nanoTimestamp();
        const criteria = beat.simd_classifier.BatchFormationCriteria.performanceOptimized();
        var batch_former = beat.simd_classifier.IntelligentBatchFormer.init(allocator, criteria);
        defer batch_former.deinit();
        const end = std.time.nanoTimestamp();
        const creation_time = @as(u64, @intCast(end - start));
        std.debug.print("Init with template: {} ns ({:.2} μs)\n", .{ creation_time, @as(f64, @floatFromInt(creation_time)) / 1000.0 });
    }
    
    // Test 2: Individual addTask timing
    const TestData = struct { values: [64]f32 };
    var test_data: TestData = undefined;
    
    const task = beat.Task{
        .func = struct {
            fn func(data: *anyopaque) void {
                const typed_data = @as(*TestData, @ptrCast(@alignCast(data)));
                for (&typed_data.values) |*value| {
                    value.* = value.* * 1.2 + 0.3;
                }
            }
        }.func,
        .data = @ptrCast(&test_data),
        .priority = .normal,
        .data_size_hint = @sizeOf(TestData),
    };
    
    {
        const criteria = beat.simd_classifier.BatchFormationCriteria.performanceOptimized();
        var batch_former = beat.simd_classifier.IntelligentBatchFormer.init(allocator, criteria);
        defer batch_former.deinit();
        
        // Test single task addition
        for (0..5) |i| {
            const start = std.time.nanoTimestamp();
            batch_former.addTask(task, false) catch unreachable;
            const end = std.time.nanoTimestamp();
            const add_time = @as(u64, @intCast(end - start));
            std.debug.print("  AddTask {}: {} ns ({:.2} μs)\n", .{ i + 1, add_time, @as(f64, @floatFromInt(add_time)) / 1000.0 });
        }
    }
    
    // Test 3: Check template access time
    {
        const criteria = beat.simd_classifier.BatchFormationCriteria.performanceOptimized();
        var batch_former = beat.simd_classifier.IntelligentBatchFormer.init(allocator, criteria);
        defer batch_former.deinit();
        
        const start = std.time.nanoTimestamp();
        var fast_classified = batch_former.template_classification;
        fast_classified.task = task;
        const end = std.time.nanoTimestamp();
        const template_time = @as(u64, @intCast(end - start));
        std.debug.print("Template copy: {} ns ({:.2} μs)\n", .{ template_time, @as(f64, @floatFromInt(template_time)) / 1000.0 });
    }
}