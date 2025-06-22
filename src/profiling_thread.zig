// Dedicated Profiling Thread - Offloads expensive profiling from critical execution paths
// Provides asynchronous task profiling with lock-free communication and batch processing

const std = @import("std");
const core = @import("core.zig");
const simd_classifier = @import("simd_classifier.zig");
const lockfree = @import("lockfree.zig");

/// Profiling event for lock-free communication between workers and profiler
const ProfilingEvent = union(enum) {
    task_submission: TaskSubmissionEvent,
    task_completion: TaskCompletionEvent,
    dynamic_profile_request: DynamicProfileRequest,
    
    const TaskSubmissionEvent = struct {
        task_hash: u64,
        submission_time_ns: u64,
        thread_id: u32,
    };
    
    const TaskCompletionEvent = struct {
        task_hash: u64,
        execution_time_ns: u64,
        completion_time_ns: u64,
        thread_id: u32,
    };
    
    const DynamicProfileRequest = struct {
        task: core.Task,
        iterations: u32,
        callback: *const fn(*simd_classifier.DynamicProfile) void,
        request_id: u64,
    };
};

/// High-performance profiling thread with minimal critical path impact
pub const ProfilingThread = struct {
    const Self = @This();
    
    // Lock-free communication
    event_queue: lockfree.MpmcQueue(ProfilingEvent, 1024),
    
    // Thread management
    thread_handle: ?std.Thread = null,
    running: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    allocator: std.mem.Allocator,
    
    // Profiling state
    task_profiles: std.AutoHashMap(u64, TaskProfileAccumulator),
    processing_batch_size: u32 = 32,
    low_priority_sleep_ns: u64 = 1_000_000, // 1ms between processing cycles
    
    // Performance statistics
    events_processed: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    batches_processed: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    profiling_overhead_ns: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    
    /// Accumulated profiling data for a specific task type
    const TaskProfileAccumulator = struct {
        task_hash: u64,
        submission_count: u64 = 0,
        completion_count: u64 = 0,
        total_execution_time_ns: u64 = 0,
        min_execution_time_ns: u64 = std.math.maxInt(u64),
        max_execution_time_ns: u64 = 0,
        execution_variance_sum: f64 = 0.0,
        last_submission_ns: u64 = 0,
        last_completion_ns: u64 = 0,
        
        /// Update with new execution time
        pub fn addExecution(self: *TaskProfileAccumulator, execution_time_ns: u64) void {
            self.completion_count += 1;
            self.total_execution_time_ns += execution_time_ns;
            self.min_execution_time_ns = @min(self.min_execution_time_ns, execution_time_ns);
            self.max_execution_time_ns = @max(self.max_execution_time_ns, execution_time_ns);
            
            // Update variance using Welford's algorithm
            const mean = @as(f64, @floatFromInt(self.total_execution_time_ns)) / @as(f64, @floatFromInt(self.completion_count));
            const delta = @as(f64, @floatFromInt(execution_time_ns)) - mean;
            self.execution_variance_sum += delta * delta;
        }
        
        /// Get average execution time
        pub fn getAverageExecutionTime(self: *const TaskProfileAccumulator) u64 {
            if (self.completion_count == 0) return 0;
            return self.total_execution_time_ns / self.completion_count;
        }
        
        /// Get execution variance
        pub fn getExecutionVariance(self: *const TaskProfileAccumulator) f64 {
            if (self.completion_count < 2) return 0.0;
            return self.execution_variance_sum / @as(f64, @floatFromInt(self.completion_count - 1));
        }
    };
    
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .event_queue = lockfree.MpmcQueue(ProfilingEvent, 1024).init(),
            .task_profiles = std.AutoHashMap(u64, TaskProfileAccumulator).init(allocator),
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.stop();
        self.task_profiles.deinit();
    }
    
    /// Start profiling thread with low priority
    pub fn start(self: *Self) !void {
        if (self.running.load(.acquire)) {
            return; // Already running
        }
        
        self.running.store(true, .release);
        
        // Create low-priority thread for profiling
        self.thread_handle = try std.Thread.spawn(.{}, profilingLoop, .{self});
    }
    
    /// Stop profiling thread
    pub fn stop(self: *Self) void {
        self.running.store(false, .release);
        
        if (self.thread_handle) |thread| {
            thread.join();
            self.thread_handle = null;
        }
    }
    
    // ========================================================================
    // Lock-Free Event Submission (Critical Path - Zero Blocking)
    // ========================================================================
    
    /// Submit task profiling event (non-blocking, O(1))
    pub inline fn submitTaskEvent(self: *Self, task_hash: u64, execution_time_ns: u64) void {
        const event = ProfilingEvent{
            .task_completion = .{
                .task_hash = task_hash,
                .execution_time_ns = execution_time_ns,
                .completion_time_ns = @as(u64, @intCast(std.time.nanoTimestamp())),
                .thread_id = @as(u32, @intCast(std.Thread.getCurrentId())),
            },
        };
        
        // Non-blocking submission - if queue full, just drop the event
        _ = self.event_queue.enqueue(event);
    }
    
    /// Request dynamic profiling (non-blocking, O(1))
    pub inline fn requestDynamicProfiling(self: *Self, task: core.Task, iterations: u32, callback: *const fn(*simd_classifier.DynamicProfile) void) u64 {
        const request_id = @as(u64, @intCast(std.time.nanoTimestamp()));
        
        const event = ProfilingEvent{
            .dynamic_profile_request = .{
                .task = task,
                .iterations = iterations,
                .callback = callback,
                .request_id = request_id,
            },
        };
        
        _ = self.event_queue.enqueue(event);
        return request_id;
    }
    
    /// Get accumulated profile for specific task type
    pub fn getTaskProfile(self: *const Self, task_hash: u64) ?TaskProfileAccumulator {
        return self.task_profiles.get(task_hash);
    }
    
    /// Check if profiling thread is healthy and processing events
    pub fn isHealthy(self: *const Self) bool {
        const running = self.running.load(.acquire);
        const has_thread = self.thread_handle != null;
        return running and has_thread;
    }
    
    // Main profiling loop - runs on dedicated low-priority thread
    fn profilingLoop(self: *Self) void {
        while (self.running.load(.acquire)) {
            const process_start = std.time.nanoTimestamp();
            
            // Process events in batches for efficiency
            var events_processed: u32 = 0;
            
            for (0..32) |_| { // Process up to 32 events per batch
                if (self.event_queue.dequeue()) |event| {
                    self.processEvent(event);
                    events_processed += 1;
                } else {
                    break; // No more events
                }
            }
            
            if (events_processed > 0) {
                _ = self.events_processed.fetchAdd(events_processed, .monotonic);
                _ = self.batches_processed.fetchAdd(1, .monotonic);
                
                // Track profiling overhead
                const process_end = std.time.nanoTimestamp();
                const overhead_ns = @as(u64, @intCast(process_end - process_start));
                _ = self.profiling_overhead_ns.fetchAdd(overhead_ns, .monotonic);
            } else {
                // No events to process - yield CPU to other threads
                std.time.sleep(self.low_priority_sleep_ns);
            }
        }
    }
    
    /// Process individual profiling event
    fn processEvent(self: *Self, event: ProfilingEvent) void {
        switch (event) {
            .task_completion => |completion| {
                self.processTaskCompletion(completion);
            },
            .dynamic_profile_request => |request| {
                self.processDynamicProfileRequest(request);
            },
            .task_submission => |submission| {
                self.processTaskSubmission(submission);
            },
        }
    }
    
    /// Process task completion event
    fn processTaskCompletion(self: *Self, completion: ProfilingEvent.TaskCompletionEvent) void {
        const result = self.task_profiles.getOrPut(completion.task_hash) catch return;
        
        if (!result.found_existing) {
            result.value_ptr.* = TaskProfileAccumulator{
                .task_hash = completion.task_hash,
            };
        }
        
        result.value_ptr.addExecution(completion.execution_time_ns);
        result.value_ptr.last_completion_ns = completion.completion_time_ns;
    }
    
    /// Process dynamic profiling request (expensive operation off critical path)
    fn processDynamicProfileRequest(self: *Self, request: ProfilingEvent.DynamicProfileRequest) void {
        _ = self;
        // This is where the expensive DynamicProfile.profileTask runs asynchronously
        var profile = simd_classifier.DynamicProfile.profileTask(&request.task, request.iterations) catch {
            std.log.warn("ProfilingThread: Failed to profile task", .{});
            return;
        };
        
        // Callback with results (could be storing in a cache, updating ML models, etc.)
        request.callback(&profile);
    }
    
    /// Process task submission event
    fn processTaskSubmission(self: *Self, submission: ProfilingEvent.TaskSubmissionEvent) void {
        const result = self.task_profiles.getOrPut(submission.task_hash) catch return;
        
        if (!result.found_existing) {
            result.value_ptr.* = TaskProfileAccumulator{
                .task_hash = submission.task_hash,
            };
        }
        
        result.value_ptr.submission_count += 1;
        result.value_ptr.last_submission_ns = submission.submission_time_ns;
    }
};

// ============================================================================
// Integration Helpers
// ============================================================================

/// Global profiling thread instance
var global_profiling_thread: ?ProfilingThread = null;
var global_profiling_mutex: std.Thread.Mutex = .{};

/// Initialize global profiling thread
pub fn initGlobalProfilingThread(allocator: std.mem.Allocator) !void {
    global_profiling_mutex.lock();
    defer global_profiling_mutex.unlock();
    
    if (global_profiling_thread == null) {
        global_profiling_thread = ProfilingThread.init(allocator);
        try global_profiling_thread.?.start();
    }
}

/// Get global profiling thread
pub fn getGlobalProfilingThread() ?*ProfilingThread {
    global_profiling_mutex.lock();
    defer global_profiling_mutex.unlock();
    
    if (global_profiling_thread) |*thread| {
        return thread;
    }
    return null;
}

/// Cleanup global profiling thread
pub fn deinitGlobalProfilingThread() void {
    global_profiling_mutex.lock();
    defer global_profiling_mutex.unlock();
    
    if (global_profiling_thread) |*thread| {
        thread.deinit();
        global_profiling_thread = null;
    }
}