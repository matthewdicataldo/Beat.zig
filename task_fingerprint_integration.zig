const std = @import("std");
const core = @import("src/core.zig");
const TaskFingerprint = @import("task_fingerprint_design.zig").TaskFingerprint;
const ExecutionContext = @import("task_fingerprint_design.zig").ExecutionContext;

// Integration of TaskFingerprint with existing Beat.zig architecture
// This file demonstrates how to integrate the fingerprint system with
// the current Task structure and scheduler implementation.

// ============================================================================
// Enhanced Task Structure
// ============================================================================

/// Enhanced Task structure with fingerprinting capability
pub const EnhancedTask = struct {
    // Existing Beat.zig Task fields
    func: *const fn (*anyopaque) void,
    data: *anyopaque,
    priority: core.Priority = .normal,
    affinity_hint: ?u32 = null,
    
    // New fingerprinting fields
    fingerprint: ?TaskFingerprint = null,
    source_location: std.builtin.SourceLocation = @src(),
    creation_timestamp: u64 = 0,
    
    // Performance tracking
    estimated_cycles: u64 = 0,
    data_size_hint: usize = 0,
    access_pattern_hint: TaskFingerprint.AccessPattern = .mixed,
    
    /// Generate fingerprint for this task
    pub fn generateFingerprint(self: *EnhancedTask, context: *const ExecutionContext) void {
        self.fingerprint = TaskFingerprint.generate(self, context, self.source_location);
    }
    
    /// Convert to legacy Task for backward compatibility
    pub fn toLegacyTask(self: *const EnhancedTask) core.Task {
        return core.Task{
            .func = self.func,
            .data = self.data,
            .priority = self.priority,
            .affinity_hint = self.affinity_hint,
        };
    }
    
    /// Create from legacy Task
    pub fn fromLegacyTask(legacy_task: core.Task, source_location: std.builtin.SourceLocation) EnhancedTask {
        return EnhancedTask{
            .func = legacy_task.func,
            .data = legacy_task.data,
            .priority = legacy_task.priority,
            .affinity_hint = legacy_task.affinity_hint,
            .source_location = source_location,
            .creation_timestamp = std.time.nanoTimestamp(),
        };
    }
};

// ============================================================================
// Execution Context Implementation
// ============================================================================

/// Global execution context for fingerprint generation
pub const GlobalExecutionContext = struct {
    context: ExecutionContext,
    mutex: std.Thread.Mutex = .{},
    
    pub fn init() GlobalExecutionContext {
        return GlobalExecutionContext{
            .context = ExecutionContext{
                .current_numa_node = 0,
                .application_phase = 0,
                .current_hour = @intCast(std.time.timestamp() / 3600 % 24),
                .estimated_cycles = 1000,
                .system_load = 0.5,
                .available_cores = std.Thread.getCpuCount() catch 4,
                .recent_task_types = [_]u32{0} ** 8,
                .execution_history_count = 0,
            },
        };
    }
    
    pub fn update(self: *GlobalExecutionContext, numa_node: u32, system_load: f32) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        self.context.current_numa_node = numa_node;
        self.context.system_load = system_load;
        self.context.current_hour = @intCast(std.time.timestamp() / 3600 % 24);
        self.context.execution_history_count += 1;
    }
    
    pub fn recordTaskExecution(self: *GlobalExecutionContext, fingerprint: TaskFingerprint, cycles: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        // Update recent task types ring buffer
        const index = self.context.execution_history_count % 8;
        self.context.recent_task_types[index] = @truncate(fingerprint.hash());
        
        // Update estimated cycles with simple averaging (to be replaced with One Euro Filter)
        self.context.estimated_cycles = (self.context.estimated_cycles + cycles) / 2;
    }
    
    pub fn getContext(self: *GlobalExecutionContext) ExecutionContext {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.context;
    }
};

// Thread-local global context
threadlocal var global_context: ?*GlobalExecutionContext = null;

pub fn initGlobalContext(context: *GlobalExecutionContext) void {
    global_context = context;
}

pub fn getGlobalContext() ?*GlobalExecutionContext {
    return global_context;
}

// ============================================================================
// Fingerprint Registry
// ============================================================================

/// Registry for storing and managing task fingerprints and their performance data
pub const FingerprintRegistry = struct {
    const Self = @This();
    
    const ProfileEntry = struct {
        fingerprint: TaskFingerprint,
        execution_count: u64,
        total_cycles: u64,
        min_cycles: u64,
        max_cycles: u64,
        last_execution: u64,
        
        pub fn getAverageExecution(self: *const ProfileEntry) f64 {
            if (self.execution_count == 0) return 0.0;
            return @as(f64, @floatFromInt(self.total_cycles)) / @as(f64, @floatFromInt(self.execution_count));
        }
        
        pub fn updateExecution(self: *ProfileEntry, cycles: u64) void {
            if (self.execution_count == 0) {
                self.min_cycles = cycles;
                self.max_cycles = cycles;
            } else {
                self.min_cycles = @min(self.min_cycles, cycles);
                self.max_cycles = @max(self.max_cycles, cycles);
            }
            
            self.total_cycles += cycles;
            self.execution_count += 1;
            self.last_execution = std.time.nanoTimestamp();
        }
    };
    
    profiles: std.AutoHashMap(u64, ProfileEntry), // Key: fingerprint hash
    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex = .{},
    
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .profiles = std.AutoHashMap(u64, ProfileEntry).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.profiles.deinit();
    }
    
    pub fn recordExecution(self: *Self, fingerprint: TaskFingerprint, cycles: u64) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const hash = fingerprint.hash();
        
        if (self.profiles.getPtr(hash)) |entry| {
            entry.updateExecution(cycles);
        } else {
            var new_entry = ProfileEntry{
                .fingerprint = fingerprint,
                .execution_count = 0,
                .total_cycles = 0,
                .min_cycles = 0,
                .max_cycles = 0,
                .last_execution = 0,
            };
            new_entry.updateExecution(cycles);
            try self.profiles.put(hash, new_entry);
        }
    }
    
    pub fn getProfile(self: *Self, fingerprint: TaskFingerprint) ?ProfileEntry {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        return self.profiles.get(fingerprint.hash());
    }
    
    pub fn getPredictedCycles(self: *Self, fingerprint: TaskFingerprint) f64 {
        if (self.getProfile(fingerprint)) |profile| {
            return profile.getAverageExecution();
        }
        return 1000.0; // Default estimate
    }
    
    pub fn getStats(self: *Self) RegistryStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        return RegistryStats{
            .total_profiles = self.profiles.count(),
            .memory_usage = self.profiles.count() * @sizeOf(ProfileEntry),
        };
    }
    
    pub const RegistryStats = struct {
        total_profiles: u32,
        memory_usage: usize,
    };
};

// ============================================================================
// Helper Functions Implementation
// ============================================================================

/// Analyze task data to extract characteristics for fingerprinting
pub const TaskAnalyzer = struct {
    pub fn getDataSize(task: anytype) usize {
        // Try to extract data size from common task patterns
        if (@hasField(@TypeOf(task.*), "data_size_hint")) {
            return task.data_size_hint;
        }
        
        // For now, use a heuristic based on data pointer alignment
        const ptr_value = @intFromPtr(task.data);
        if (ptr_value == 0) return 0;
        
        // Simple heuristic: assume aligned pointers indicate larger data
        const alignment = @ctz(ptr_value);
        return @as(usize, 1) << @min(alignment, 20); // Cap at 1MB for sanity
    }
    
    pub fn analyzeAccessPattern(task: anytype) TaskFingerprint.AccessPattern {
        const data_size = getDataSize(task);
        const ptr_value = @intFromPtr(task.data);
        
        // Heuristics for access pattern detection
        if (data_size < 64) return .sequential;
        if (ptr_value % 64 == 0) return .sequential; // Cache-aligned suggests sequential
        if (data_size > 1024 * 1024) return .strided; // Large data often strided
        
        return .mixed;
    }
    
    pub fn estimateIntensity(task: anytype) u8 {
        // Heuristic: function pointer value can hint at function complexity
        const func_ptr = @intFromPtr(task.func);
        
        // Simple heuristic based on function address
        const complexity_hint = @truncate(func_ptr >> 4);
        return @as(u8, @intCast(complexity_hint % 16));
    }
    
    pub fn classifyParallelPotential(task: anytype) u8 {
        const data_size = getDataSize(task);
        
        // Larger data sizes typically have better parallel potential
        if (data_size >= 1024 * 1024) return 15; // Very high potential
        if (data_size >= 64 * 1024) return 12;   // High potential
        if (data_size >= 4 * 1024) return 8;     // Medium potential
        if (data_size >= 256) return 4;          // Low potential
        
        return 1; // Very low potential
    }
};

// ============================================================================
// Integration with Beat.zig Core
// ============================================================================

/// Enhanced ThreadPool with fingerprinting support
pub const FingerprintAwareThreadPool = struct {
    base_pool: *core.ThreadPool,
    fingerprint_registry: FingerprintRegistry,
    global_context: GlobalExecutionContext,
    
    const Self = @This();
    
    pub fn init(allocator: std.mem.Allocator, config: core.Config) !*Self {
        const self = try allocator.create(Self);
        
        self.* = Self{
            .base_pool = try core.ThreadPool.init(allocator, config),
            .fingerprint_registry = FingerprintRegistry.init(allocator),
            .global_context = GlobalExecutionContext.init(),
        };
        
        // Initialize global context for fingerprint generation
        initGlobalContext(&self.global_context);
        
        return self;
    }
    
    pub fn deinit(self: *Self) void {
        self.fingerprint_registry.deinit();
        self.base_pool.deinit();
    }
    
    pub fn submitEnhanced(self: *Self, enhanced_task: *EnhancedTask) !void {
        const start_time = std.time.nanoTimestamp();
        
        // Generate fingerprint if not already present
        if (enhanced_task.fingerprint == null) {
            const context = self.global_context.getContext();
            enhanced_task.generateFingerprint(&context);
        }
        
        // Convert to legacy task for submission
        const legacy_task = enhanced_task.toLegacyTask();
        
        // Record task submission with fingerprint
        if (enhanced_task.fingerprint) |fp| {
            // Predict execution time based on historical data
            const predicted_cycles = self.fingerprint_registry.getPredictedCycles(fp);
            enhanced_task.estimated_cycles = @intFromFloat(predicted_cycles);
        }
        
        // Submit to base thread pool
        try self.base_pool.submit(legacy_task);
        
        // Record execution (simplified - in real implementation, this would be done after task completion)
        if (enhanced_task.fingerprint) |fp| {
            const execution_time = std.time.nanoTimestamp() - start_time;
            self.fingerprint_registry.recordExecution(fp, @intCast(execution_time)) catch {};
            self.global_context.recordTaskExecution(fp, @intCast(execution_time));
        }
    }
    
    pub fn submitLegacy(self: *Self, legacy_task: core.Task) !void {
        var enhanced_task = EnhancedTask.fromLegacyTask(legacy_task, @src());
        try self.submitEnhanced(&enhanced_task);
    }
    
    pub fn wait(self: *Self) void {
        self.base_pool.wait();
    }
    
    pub fn getRegistryStats(self: *Self) FingerprintRegistry.RegistryStats {
        return self.fingerprint_registry.getStats();
    }
};

// ============================================================================
// Convenience Macros and Functions
// ============================================================================

/// Convenience function to submit a task with automatic fingerprinting
pub fn submitTask(
    pool: *FingerprintAwareThreadPool,
    comptime func: anytype,
    data: anytype,
    priority: core.Priority,
) !void {
    const TaskWrapper = struct {
        var task_data: @TypeOf(data) = undefined;
        
        fn taskFunction(ctx: *anyopaque) void {
            _ = ctx;
            func(&task_data);
        }
    };
    
    TaskWrapper.task_data = data;
    
    var enhanced_task = EnhancedTask{
        .func = TaskWrapper.taskFunction,
        .data = @ptrCast(&TaskWrapper.task_data),
        .priority = priority,
        .source_location = @src(),
        .creation_timestamp = std.time.nanoTimestamp(),
        .data_size_hint = @sizeOf(@TypeOf(data)),
    };
    
    try pool.submitEnhanced(&enhanced_task);
}

// ============================================================================
// Testing and Validation
// ============================================================================

const testing = std.testing;

test "Enhanced Task fingerprint generation" {
    var global_ctx = GlobalExecutionContext.init();
    const context = global_ctx.getContext();
    
    const TestData = struct { value: i32 };
    var test_data = TestData{ .value = 42 };
    
    const test_func = struct {
        fn func(data: *anyopaque) void {
            const typed_data = @as(*TestData, @ptrCast(@alignCast(data)));
            typed_data.value *= 2;
        }
    }.func;
    
    var enhanced_task = EnhancedTask{
        .func = test_func,
        .data = @ptrCast(&test_data),
        .priority = .normal,
        .source_location = @src(),
        .creation_timestamp = std.time.nanoTimestamp(),
        .data_size_hint = @sizeOf(TestData),
    };
    
    enhanced_task.generateFingerprint(&context);
    
    try testing.expect(enhanced_task.fingerprint != null);
    
    if (enhanced_task.fingerprint) |fp| {
        // Verify fingerprint has reasonable values
        try testing.expect(fp.call_site_hash != 0);
        try testing.expect(fp.data_size_class > 0);
        
        // Test hash consistency
        const hash1 = fp.hash();
        const hash2 = fp.hash();
        try testing.expectEqual(hash1, hash2);
    }
}

test "Fingerprint registry operations" {
    var registry = FingerprintRegistry.init(testing.allocator);
    defer registry.deinit();
    
    // Create a test fingerprint
    const test_fp = TaskFingerprint{
        .call_site_hash = 0x12345678,
        .data_size_class = 10,
        .data_alignment = 3,
        .access_pattern = .sequential,
        .simd_width = 4,
        .cache_locality = 8,
        .numa_node_hint = 0,
        .cpu_intensity = 8,
        .parallel_potential = 10,
        .execution_phase = 1,
        .priority_class = 1,
        .time_sensitivity = 1,
        .dependency_count = 0,
        .time_of_day_bucket = 14,
        .execution_frequency = 4,
        .seasonal_pattern = 0,
        .variance_level = 6,
        .expected_cycles_log2 = 16,
        .memory_footprint_log2 = 12,
        .io_intensity = 2,
        .cache_miss_rate = 5,
        .branch_predictability = 8,
        .vectorization_benefit = 6,
    };
    
    // Record some executions
    try registry.recordExecution(test_fp, 1000);
    try registry.recordExecution(test_fp, 1200);
    try registry.recordExecution(test_fp, 800);
    
    // Verify profile was created
    const profile = registry.getProfile(test_fp);
    try testing.expect(profile != null);
    
    if (profile) |p| {
        try testing.expectEqual(@as(u64, 3), p.execution_count);
        try testing.expectEqual(@as(u64, 3000), p.total_cycles);
        try testing.expectEqual(@as(u64, 800), p.min_cycles);
        try testing.expectEqual(@as(u64, 1200), p.max_cycles);
        
        const avg = p.getAverageExecution();
        try testing.expectApproxEqAbs(@as(f64, 1000.0), avg, 0.1);
    }
    
    // Test prediction
    const predicted = registry.getPredictedCycles(test_fp);
    try testing.expectApproxEqAbs(@as(f64, 1000.0), predicted, 0.1);
}

test "FingerprintAwareThreadPool basic functionality" {
    var pool = try FingerprintAwareThreadPool.init(testing.allocator, core.Config{});
    defer pool.deinit();
    
    // Test submitting an enhanced task
    const TestData = struct { 
        value: i32,
        
        pub fn process(self: *@This()) void {
            self.value *= 2;
        }
    };
    
    var test_data = TestData{ .value = 21 };
    
    try submitTask(pool, TestData.process, &test_data, .normal);
    
    pool.wait();
    
    // Verify task executed
    try testing.expectEqual(@as(i32, 42), test_data.value);
    
    // Verify registry has recorded the execution
    const stats = pool.getRegistryStats();
    try testing.expect(stats.total_profiles > 0);
}