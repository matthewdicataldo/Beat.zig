const std = @import("std");
const builtin = @import("builtin");
const core = @import("core.zig");
const scheduler = @import("scheduler.zig");

// Zero-overhead potentially parallel call implementations

// ============================================================================
// Thread-Local State
// ============================================================================

threadlocal var tls_pool: ?*core.ThreadPool = null;

pub fn initThread(pool: *core.ThreadPool) void {
    tls_pool = pool;
}

// ============================================================================
// Future Types
// ============================================================================

pub fn Future(comptime T: type) type {
    return struct {
        result: ?T = null,
        error_value: ?anyerror = null,
        completed: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
        
        const Self = @This();
        
        pub fn wait(self: *Self) !T {
            while (!self.completed.load(.acquire)) {
                std.time.sleep(10_000); // 10 microseconds
            }
            
            if (self.error_value) |err| {
                return err;
            }
            
            return self.result.?;
        }
        
        pub fn isReady(self: *const Self) bool {
            return self.completed.load(.acquire);
        }
    };
}

pub fn PotentialFuture(comptime T: type) type {
    return union(enum) {
        immediate: T,
        deferred: *Future(T),
        
        pub fn get(self: *@This()) !T {
            switch (self.*) {
                .immediate => |val| return val,
                .deferred => |future| return try future.wait(),
            }
        }
        
        pub fn isReady(self: *const @This()) bool {
            switch (self.*) {
                .immediate => return true,
                .deferred => |future| return future.isReady(),
            }
        }
    };
}

// ============================================================================
// V1: Basic pcall
// ============================================================================

pub fn pcallBasic(comptime T: type, func: *const fn () T) PotentialFuture(T) {
    const pool = tls_pool orelse return .{ .immediate = func() };
    
    // Create wrapper struct to hold both future and function
    const TaskData = struct {
        future: Future(T),
        func: *const fn () T,
    };
    
    const task_data = pool.allocator.create(TaskData) catch {
        return .{ .immediate = func() };
    };
    
    task_data.* = TaskData{
        .future = Future(T){},
        .func = func,
    };
    
    const task = core.Task{
        .func = struct {
            fn wrapper(data: *anyopaque) void {
                const task_ptr = @as(*TaskData, @ptrCast(@alignCast(data)));
                task_ptr.future.result = task_ptr.func();
                task_ptr.future.completed.store(true, .release);
            }
        }.wrapper,
        .data = task_data,
    };
    
    pool.submit(task) catch {
        pool.allocator.destroy(task_data);
        return .{ .immediate = func() };
    };
    
    return .{ .deferred = &task_data.future };
}

// ============================================================================
// V2: Heartbeat pcall with token accounting
// ============================================================================

pub fn pcall(comptime T: type, func: *const fn () T) PotentialFuture(T) {
    const pool = tls_pool orelse return .{ .immediate = func() };
    
    if (!pool.config.enable_heartbeat) {
        return pcallBasic(T, func);
    }
    
    const tokens = scheduler.getLocalTokens() orelse return .{ .immediate = func() };
    
    const start = scheduler.rdtsc();
    
    if (!tokens.shouldPromote()) {
        // Fast path: execute inline
        const work_start = scheduler.rdtsc();
        const result = func();
        const work_end = scheduler.rdtsc();
        
        if (pool.config.enable_heartbeat) {
            tokens.update(work_end -% work_start, work_start -% start);
        }
        
        return .{ .immediate = result };
    }
    
    // Slow path: defer to thread pool
    return pcallBasic(T, func);
}

// ============================================================================
// V3: Zero-overhead pcall variants
// ============================================================================

pub inline fn pcallMinimal(comptime T: type, comptime func: fn () T) T {
    if (comptime builtin.mode == .ReleaseFast) {
        // Zero overhead in release mode
        return @call(.always_inline, func, .{});
    }
    
    // In debug mode, use standard pcall
    var future = pcall(T, &func);
    return future.get() catch |err| {
        // Zero-overhead parallel call failed unexpectedly
        // Error: {}, Function: pcallMinimal
        // Help: Check task function implementation, thread pool initialization, memory allocation
        // Common causes: Task function panic, thread pool shutdown, out of memory
        std.debug.panic("pcallMinimal execution failed: {}", .{err});
    };
}

pub inline fn pcallAuto(comptime T: type, comptime func: fn () T) T {
    // Compile-time decision based on result size
    const size = @sizeOf(T);
    
    if (comptime size <= 64) {
        // Small results: always inline
        return @call(.always_inline, func, .{});
    } else if (comptime size <= 1024) {
        // Medium results: use pcall
        var future = pcall(T, &func);
        return future.get() catch |err| {
            // Automatic parallel call failed for medium-sized result type
            // Error: {}, Result size: {}B, Strategy: parallel call
            // Help: Check task function implementation, thread pool state, memory availability
            std.debug.panic("pcallAuto (medium) execution failed: {}", .{err});
        };
    } else {
        // Large results: always parallelize
        var future = pcallBasic(T, &func);
        return future.get() catch |err| {
            // Automatic parallel call failed for large result type
            // Error: {}, Result size: {}B, Strategy: force parallel (large data)
            // Help: Check task function implementation, thread pool availability, memory constraints
            std.debug.panic("pcallAuto (large) execution failed: {}", .{err});
        };
    }
}

// ============================================================================
// Fork-Join Helpers
// ============================================================================

pub fn join2(
    pool: *core.ThreadPool,
    comptime T1: type,
    comptime T2: type,
    func1: *const fn () anyerror!T1,
    func2: *const fn () anyerror!T2,
) !struct { left: T1, right: T2 } {
    // Create wrapper struct to hold both future and function
    const TaskData = struct {
        future: Future(T1),
        func: *const fn () anyerror!T1,
    };
    
    const task_data = pool.allocator.create(TaskData) catch {
        // If allocation fails, run both sequentially
        const result1 = try func1();
        const result2 = try func2();
        return .{ .left = result1, .right = result2 };
    };
    
    task_data.* = TaskData{
        .future = Future(T1){},
        .func = func1,
    };
    
    const task = core.Task{
        .func = struct {
            fn wrapper(data: *anyopaque) void {
                const task_ptr = @as(*TaskData, @ptrCast(@alignCast(data)));
                task_ptr.future.result = task_ptr.func() catch |err| {
                    task_ptr.future.error_value = err;
                    task_ptr.future.completed.store(true, .release);
                    return;
                };
                task_ptr.future.completed.store(true, .release);
            }
        }.wrapper,
        .data = task_data,
    };
    
    pool.submit(task) catch {
        // If submit fails, run both sequentially
        pool.allocator.destroy(task_data);
        const result1 = try func1();
        const result2 = try func2();
        return .{ .left = result1, .right = result2 };
    };
    
    // Execute second function on current thread
    const result2 = try func2();
    
    // Wait for first result
    const result1 = try task_data.future.wait();
    pool.allocator.destroy(task_data);
    
    return .{ .left = result1, .right = result2 };
}

pub fn joinN(
    pool: *core.ThreadPool,
    comptime T: type,
    funcs: []const *const fn () anyerror!T,
) ![]T {
    if (funcs.len == 0) return &[0]T{};
    if (funcs.len == 1) return &[1]T{try funcs[0]()};
    
    const futures = try pool.allocator.alloc(*Future(T), funcs.len - 1);
    defer pool.allocator.free(futures);
    
    // Submit all but last function
    for (funcs[0..funcs.len-1], 0..) |func, i| {
        futures[i] = try pool.allocator.create(Future(T));
        futures[i].* = Future(T){};
        
        const task = core.Task{
            .func = struct {
                fn wrapper(data: *anyopaque) void {
                    const context = @as(*const struct { f: *const fn () anyerror!T, fut: *Future(T) }, @ptrCast(@alignCast(data)));
                    context.fut.result = context.f() catch |err| {
                        context.fut.error_value = err;
                        context.fut.completed.store(true, .release);
                        return;
                    };
                    context.fut.completed.store(true, .release);
                }
            }.wrapper,
            .data = &.{ .f = func, .fut = futures[i] },
        };
        
        try pool.submit(task);
    }
    
    // Execute last function on current thread
    const results = try pool.allocator.alloc(T, funcs.len);
    results[funcs.len - 1] = try funcs[funcs.len - 1]();
    
    // Wait for all results
    for (futures, 0..) |future, i| {
        results[i] = try future.wait();
        pool.allocator.destroy(future);
    }
    
    return results;
}

// ============================================================================
// Tests
// ============================================================================

test "basic pcall" {
    const allocator = std.testing.allocator;
    const pool = try core.createPool(allocator);
    defer pool.deinit();
    
    initThread(pool);
    
    const compute = struct {
        fn run() i32 {
            return 42;
        }
    }.run;
    
    var future = pcallBasic(i32, &compute);
    const result = try future.get();
    
    try std.testing.expectEqual(@as(i32, 42), result);
}

test "pcallMinimal" {
    const compute = struct {
        fn run() i32 {
            return 100;
        }
    }.run;
    
    const result = pcallMinimal(i32, compute);
    try std.testing.expectEqual(@as(i32, 100), result);
}

test "join2" {
    const allocator = std.testing.allocator;
    const pool = try core.createPool(allocator);
    defer pool.deinit();
    
    const func1 = struct {
        fn run() !i32 {
            return 10;
        }
    }.run;
    
    const func2 = struct {
        fn run() !i32 {
            return 20;
        }
    }.run;
    
    const results = try join2(pool, i32, i32, &func1, &func2);
    try std.testing.expectEqual(@as(i32, 10), results.left);
    try std.testing.expectEqual(@as(i32, 20), results.right);
}