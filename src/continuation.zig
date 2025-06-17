const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

// ============================================================================
// Continuation Stealing Implementation
// ============================================================================

/// Continuation state enumeration
pub const ContinuationState = enum {
    pending,    // Created but not yet executing
    running,    // Currently executing on a worker
    stolen,     // Stolen by another worker
    completed,  // Execution finished
    failed,     // Execution failed with error
};

/// Continuation structure for stack frame capture and stealing
/// Designed to work without native Zig async/await support
pub const Continuation = struct {
    // HOT PATH: Frequently accessed during execution (first 32 bytes)
    frame_ptr: usize,                           // 8 bytes - stack frame pointer (@frameAddress)
    resume_fn: *const fn(*Continuation) void,   // 8 bytes - resume function pointer  
    data: *anyopaque,                          // 8 bytes - continuation data
    state: ContinuationState,                  // 4 bytes - current state
    worker_id: ?u32,                           // 4 bytes - owning worker ID
    
    // WARM PATH: Moderately accessed during scheduling (next 32 bytes)
    parent: ?*Continuation,                    // 8 bytes - parent continuation chain
    return_address: usize,                     // 8 bytes - return address (@returnAddress)
    frame_size: u32,                          // 4 bytes - estimated frame size
    steal_count: u32,                         // 4 bytes - number of times stolen
    creation_timestamp: u64,                   // 8 bytes - creation time (nanoseconds)
    
    // COLD PATH: Rarely accessed metadata (remaining bytes, total: 112 bytes = 1.75 cache lines)
    stack_base: ?usize,                       // 8 bytes - stack base for bounds checking
    stack_limit: ?usize,                      // 8 bytes - stack limit for overflow detection
    error_info: ?*anyopaque,                  // 8 bytes - error context if failed
    numa_node: ?u32,                          // 4 bytes - preferred NUMA node
    affinity_hint: ?u32,                      // 4 bytes - worker affinity hint
    fingerprint_hash: ?u64,                   // 8 bytes - continuation fingerprint
    execution_time_ns: ?u64,                  // 8 bytes - actual execution time
    
    // NUMA locality tracking (16 bytes)
    original_numa_node: ?u32,                 // 4 bytes - NUMA node where created
    current_socket: ?u32,                     // 4 bytes - current socket ID
    migration_count: u32,                     // 4 bytes - number of NUMA migrations
    locality_score: f32,                      // 4 bytes - locality preference score (0.0-1.0)
    
    // Memory management
    allocator: ?std.mem.Allocator,            // 8 bytes - allocator for cleanup
    
    const Self = @This();
    
    /// Initialize a new continuation
    pub fn init(
        frame_ptr: usize,
        resume_fn: *const fn(*Continuation) void,
        data: *anyopaque,
        allocator: ?std.mem.Allocator,
    ) Self {
        return Self{
            .frame_ptr = frame_ptr,
            .resume_fn = resume_fn,
            .data = data,
            .state = .pending,
            .worker_id = null,
            .parent = null,
            .return_address = @returnAddress(),
            .frame_size = 0, // Will be calculated later
            .steal_count = 0,
            .creation_timestamp = @as(u64, @intCast(std.time.nanoTimestamp())),
            .stack_base = null,
            .stack_limit = null,
            .error_info = null,
            .numa_node = null,
            .affinity_hint = null,
            .fingerprint_hash = null,
            .execution_time_ns = null,
            .original_numa_node = null,
            .current_socket = null,
            .migration_count = 0,
            .locality_score = 1.0, // Start with perfect locality
            .allocator = allocator,
        };
    }
    
    /// Capture current stack frame for continuation
    pub fn capture(
        resume_fn: *const fn(*Continuation) void,
        data: *anyopaque,
        allocator: ?std.mem.Allocator,
    ) Self {
        const frame_ptr = @frameAddress();
        var cont = init(frame_ptr, resume_fn, data, allocator);
        cont.estimateFrameSize();
        return cont;
    }
    
    /// Estimate stack frame size using heuristics
    pub fn estimateFrameSize(self: *Self) void {
        // Use difference between current frame and return address as rough estimate
        const frame_diff = if (self.return_address > self.frame_ptr) 
            self.return_address - self.frame_ptr 
        else 
            self.frame_ptr - self.return_address;
        
        // Clamp to reasonable bounds (64 bytes to 4KB)
        self.frame_size = @intCast(std.math.clamp(frame_diff, 64, 4096));
    }
    
    /// Mark continuation as stolen by another worker
    pub fn markStolen(self: *Self, new_worker_id: u32) void {
        assert(self.state == .pending or self.state == .running);
        self.state = .stolen;
        self.worker_id = new_worker_id;
        self.steal_count += 1;
    }
    
    /// Mark continuation as stolen with NUMA locality tracking
    pub fn markStolenWithNuma(self: *Self, new_worker_id: u32, new_numa_node: ?u32, new_socket: ?u32) void {
        self.markStolen(new_worker_id);
        
        // Track NUMA migration
        if (self.numa_node != null and new_numa_node != null and self.numa_node.? != new_numa_node.?) {
            self.migration_count += 1;
            self.updateLocalityScore(new_numa_node, new_socket);
        }
        
        self.numa_node = new_numa_node;
        self.current_socket = new_socket;
    }
    
    /// Mark continuation as running on a worker
    pub fn markRunning(self: *Self, worker_id: u32) void {
        assert(self.state == .pending or self.state == .stolen);
        self.state = .running;
        self.worker_id = worker_id;
    }
    
    /// Mark continuation as completed
    pub fn markCompleted(self: *Self) void {
        assert(self.state == .running);
        self.state = .completed;
        self.execution_time_ns = @as(u64, @intCast(std.time.nanoTimestamp())) - self.creation_timestamp;
    }
    
    /// Mark continuation as failed with error context
    pub fn markFailed(self: *Self, error_context: ?*anyopaque) void {
        self.state = .failed;
        self.error_info = error_context;
        self.execution_time_ns = @as(u64, @intCast(std.time.nanoTimestamp())) - self.creation_timestamp;
    }
    
    /// Execute continuation
    pub fn execute(self: *Self) void {
        assert(self.state == .running);
        self.resume_fn(self);
    }
    
    /// Check if continuation can be stolen
    pub fn canBeStolen(self: *const Self) bool {
        return self.state == .pending and self.worker_id == null;
    }
    
    /// Get continuation priority (higher steal_count = lower priority)
    pub fn getPriority(self: *const Self) u32 {
        // Lower steal count means higher priority
        return std.math.maxInt(u32) - self.steal_count;
    }
    
    /// Set stack bounds for safety checking
    pub fn setStackBounds(self: *Self, stack_base: usize, stack_limit: usize) void {
        self.stack_base = stack_base;
        self.stack_limit = stack_limit;
    }
    
    /// Validate stack bounds
    pub fn validateStackBounds(self: *const Self) bool {
        if (self.stack_base == null or self.stack_limit == null) return true;
        
        const base = self.stack_base.?;
        const limit = self.stack_limit.?;
        
        return self.frame_ptr >= limit and self.frame_ptr <= base;
    }
    
    /// Initialize NUMA locality tracking
    pub fn initNumaLocality(self: *Self, numa_node: ?u32, socket: ?u32) void {
        self.numa_node = numa_node;
        self.original_numa_node = numa_node;
        self.current_socket = socket;
        self.migration_count = 0;
        self.locality_score = 1.0;
    }
    
    /// Update locality score based on NUMA migration
    pub fn updateLocalityScore(self: *Self, new_numa_node: ?u32, new_socket: ?u32) void {
        // Calculate locality penalty based on migration distance
        var penalty: f32 = 0.0;
        
        if (self.original_numa_node != null and new_numa_node != null) {
            if (self.original_numa_node.? != new_numa_node.?) {
                // NUMA node migration - significant penalty
                penalty += 0.3;
                
                // Additional penalty if crossing socket boundaries
                if (self.current_socket != null and new_socket != null and self.current_socket.? != new_socket.?) {
                    penalty += 0.2;
                }
            }
        }
        
        // Apply penalty with exponential decay
        self.locality_score = @max(0.1, self.locality_score - penalty);
    }
    
    /// Get stealing preference based on NUMA locality
    pub fn getStealingPreference(self: *const Self, target_numa_node: ?u32, target_socket: ?u32) f32 {
        if (self.numa_node == null or target_numa_node == null) return 0.5; // Neutral
        
        var preference: f32 = 1.0;
        
        // Same NUMA node = highest preference
        if (self.numa_node.? == target_numa_node.?) {
            preference = 1.0;
        }
        // Same socket, different NUMA = medium preference
        else if (self.current_socket != null and target_socket != null and self.current_socket.? == target_socket.?) {
            preference = 0.7;
        }
        // Different socket = low preference
        else {
            preference = 0.3;
        }
        
        // Apply locality score modifier
        return preference * self.locality_score;
    }
    
    /// Check if continuation should prefer local execution
    pub fn prefersLocalExecution(self: *const Self) bool {
        return self.locality_score > 0.8 and self.migration_count <= 2;
    }
};

/// Continuation frame for stack allocation
pub const ContinuationFrame = struct {
    continuation: Continuation,
    data: [64]u8, // 64 bytes for continuation data
    
    pub fn init(
        resume_fn: *const fn(*Continuation) void,
        data: *anyopaque,
    ) ContinuationFrame {
        return ContinuationFrame{
            .continuation = Continuation.capture(resume_fn, data, null),
            .data = std.mem.zeroes([64]u8),
        };
    }
};

/// Continuation registry for tracking and management
pub const ContinuationRegistry = struct {
    active_continuations: std.ArrayList(*Continuation),
    completed_continuations: std.ArrayList(*Continuation),
    mutex: std.Thread.Mutex,
    allocator: std.mem.Allocator,
    
    const Self = @This();
    
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .active_continuations = std.ArrayList(*Continuation).init(allocator),
            .completed_continuations = std.ArrayList(*Continuation).init(allocator),
            .mutex = std.Thread.Mutex{},
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        self.active_continuations.deinit();
        self.completed_continuations.deinit();
    }
    
    /// Register a new continuation
    pub fn registerContinuation(self: *Self, continuation: *Continuation) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        try self.active_continuations.append(continuation);
    }
    
    /// Mark continuation as completed and move to completed list
    pub fn completeContinuation(self: *Self, continuation: *Continuation) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        // Find and remove from active list
        for (self.active_continuations.items, 0..) |active, i| {
            if (active == continuation) {
                _ = self.active_continuations.swapRemove(i);
                break;
            }
        }
        
        // Add to completed list
        try self.completed_continuations.append(continuation);
    }
    
    /// Get statistics about continuation usage
    pub fn getStatistics(self: *Self) struct {
        active_count: usize,
        completed_count: usize,
        total_steals: u64,
        avg_execution_time_ns: f64,
    } {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var total_steals: u64 = 0;
        var total_execution_time: u64 = 0;
        var execution_count: usize = 0;
        
        for (self.active_continuations.items) |cont| {
            total_steals += cont.steal_count;
        }
        
        for (self.completed_continuations.items) |cont| {
            total_steals += cont.steal_count;
            if (cont.execution_time_ns) |exec_time| {
                total_execution_time += exec_time;
                execution_count += 1;
            }
        }
        
        const avg_execution_time = if (execution_count > 0) 
            @as(f64, @floatFromInt(total_execution_time)) / @as(f64, @floatFromInt(execution_count))
        else 
            0.0;
        
        return .{
            .active_count = self.active_continuations.items.len,
            .completed_count = self.completed_continuations.items.len,
            .total_steals = total_steals,
            .avg_execution_time_ns = avg_execution_time,
        };
    }
};

/// Helper functions for continuation creation
pub const ContinuationHelper = struct {
    /// Create a continuation from a function and data
    pub fn createContinuation(
        comptime ResumeFn: type,
        resume_fn: ResumeFn,
        data: anytype,
        allocator: std.mem.Allocator,
    ) !*Continuation {
        const continuation = try allocator.create(Continuation);
        
        // Type-erased wrapper function
        const wrapper = struct {
            fn executeWrapper(cont: *Continuation) void {
                const typed_data = @as(*@TypeOf(data), @ptrCast(@alignCast(cont.data)));
                resume_fn(typed_data);
            }
        };
        
        const data_ptr = try allocator.create(@TypeOf(data));
        data_ptr.* = data;
        
        continuation.* = Continuation.capture(wrapper.executeWrapper, data_ptr, allocator);
        return continuation;
    }
    
    /// Free a continuation and its data
    pub fn freeContinuation(continuation: *Continuation) void {
        if (continuation.allocator) |allocator| {
            if (continuation.data != @as(*anyopaque, @ptrFromInt(0))) {
                // Note: We can't safely free the data without knowing its type
                // This would need to be handled by the caller or through a cleanup function
            }
            allocator.destroy(continuation);
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "continuation basic functionality" {
    var data: i32 = 42;
    
    const TestResume = struct {
        fn executeWrapper(cont: *Continuation) void {
            const value = @as(*i32, @ptrCast(@alignCast(cont.data)));
            value.* += 1;
            cont.markCompleted();
        }
    };
    
    var continuation = Continuation.capture(TestResume.executeWrapper, &data, std.testing.allocator);
    
    try std.testing.expect(continuation.state == .pending);
    try std.testing.expect(continuation.canBeStolen());
    
    continuation.markRunning(0);
    try std.testing.expect(continuation.state == .running);
    try std.testing.expect(!continuation.canBeStolen());
    
    continuation.execute();
    try std.testing.expect(continuation.state == .completed);
    try std.testing.expect(data == 43);
}

test "continuation stealing" {
    var data: i32 = 100;
    
    const TestResume = struct {
        fn executeWrapper(cont: *Continuation) void {
            const value = @as(*i32, @ptrCast(@alignCast(cont.data)));
            value.* *= 2;
            cont.markCompleted();
        }
    };
    
    var continuation = Continuation.capture(TestResume.executeWrapper, &data, std.testing.allocator);
    
    // Test stealing
    continuation.markStolen(1);
    try std.testing.expect(continuation.state == .stolen);
    try std.testing.expect(continuation.worker_id == 1);
    try std.testing.expect(continuation.steal_count == 1);
    
    // Stolen continuation can start running
    continuation.markRunning(1);
    continuation.execute();
    try std.testing.expect(data == 200);
}

test "continuation registry" {
    var registry = ContinuationRegistry.init(std.testing.allocator);
    defer registry.deinit();
    
    var data: i32 = 10;
    const TestResume = struct {
        fn executeWrapper(cont: *Continuation) void {
            cont.markCompleted();
        }
    };
    
    var continuation = Continuation.capture(TestResume.executeWrapper, &data, std.testing.allocator);
    
    try registry.registerContinuation(&continuation);
    var stats = registry.getStatistics();
    try std.testing.expect(stats.active_count == 1);
    try std.testing.expect(stats.completed_count == 0);
    
    try registry.completeContinuation(&continuation);
    stats = registry.getStatistics();
    try std.testing.expect(stats.active_count == 0);
    try std.testing.expect(stats.completed_count == 1);
}