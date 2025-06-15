const std = @import("std");
const builtin = @import("builtin");

// Compact 128-bit task fingerprint using bitfield optimization
// Total: 128 bits (16 bytes) - cache-line friendly, SIMD-ready
pub const TaskFingerprint = packed struct {
    // Call site identification (32 bits)
    call_site_hash: u32,           // Hash of function pointer + instruction pointer
    
    // Data characteristics (24 bits)
    data_size_class: u8,           // Log2 size class (0-255, covers 1B to 2^255B)
    data_alignment: u4,            // Alignment requirement (0-15, covers 1B to 32KB)
    access_pattern: AccessPattern, // Memory access pattern (4 bits)
    simd_width: u4,               // Optimal SIMD width hint (0-15)
    cache_locality: u4,           // Expected cache behavior (0-15)
    
    // Execution context (24 bits)
    numa_node_hint: u4,           // Preferred NUMA node (0-15)
    cpu_intensity: u4,            // CPU vs memory bound ratio (0-15)
    parallel_potential: u4,       // Parallelization suitability (0-15)
    execution_phase: u4,          // Application lifecycle phase (0-15)
    priority_class: u2,           // Task priority (0-3)
    time_sensitivity: u2,         // Real-time requirements (0-3)
    dependency_count: u4,         // Number of dependencies (0-15)
    
    // Temporal characteristics (16 bits)
    time_of_day_bucket: u5,       // Hour of day (0-23, with padding)
    execution_frequency: u3,      // How often this task type runs (0-7)
    seasonal_pattern: u4,         // Day/week/month pattern (0-15)
    variance_level: u4,           // Execution time variance (0-15)
    
    // Performance hints (32 bits)
    expected_cycles_log2: u8,     // Log2 of expected execution cycles
    memory_footprint_log2: u8,   // Log2 of memory usage
    io_intensity: u4,             // I/O vs compute ratio (0-15)
    cache_miss_rate: u4,          // Expected cache miss characteristics (0-15)
    branch_predictability: u4,   // Branch prediction friendliness (0-15)
    vectorization_benefit: u4,    // SIMD optimization potential (0-15)
    
    comptime {
        // Ensure the struct is exactly 128 bits (16 bytes)
        std.debug.assert(@sizeOf(TaskFingerprint) == 16);
        std.debug.assert(@bitSizeOf(TaskFingerprint) == 128);
    }
    
    // Memory access patterns for scheduling optimization
    pub const AccessPattern = enum(u4) {
        sequential = 0,     // Linear memory access
        random = 1,         // Random memory access  
        strided = 2,        // Regular stride pattern
        hierarchical = 3,   // Tree-like access
        gather_scatter = 4, // SIMD gather/scatter
        read_only = 5,      // No writes, high cache reuse
        write_heavy = 6,    // Many writes, cache invalidation
        mixed = 7,          // Mixed access patterns
        // 8-15 reserved for future patterns
    };
    
    /// Generate fingerprint from task and execution context
    pub fn generate(
        task: anytype, // Generic task type
        context: *const ExecutionContext,
        comptime call_site: std.builtin.SourceLocation
    ) TaskFingerprint {
        return TaskFingerprint{
            // Call site fingerprinting - combines function identity with call location
            .call_site_hash = hashCallSite(task, call_site),
            
            // Data characteristics analysis
            .data_size_class = classifyDataSize(getTaskDataSize(task)),
            .data_alignment = @intCast(@ctz(@intFromPtr(getTaskDataPtr(task)))),
            .access_pattern = analyzeAccessPattern(task),
            .simd_width = detectOptimalSimdWidth(task),
            .cache_locality = estimateCacheLocality(task),
            
            // Execution context
            .numa_node_hint = @intCast(context.current_numa_node % 16),
            .cpu_intensity = estimateCpuIntensity(task),
            .parallel_potential = analyzeParallelPotential(task),
            .execution_phase = @intCast(context.application_phase),
            .priority_class = @intCast(@intFromEnum(getTaskPriority(task))),
            .time_sensitivity = classifyTimeSensitivity(task),
            .dependency_count = @intCast(@min(getTaskDependencies(task), 15)),
            
            // Temporal characteristics
            .time_of_day_bucket = @intCast(context.current_hour % 24),
            .execution_frequency = classifyExecutionFrequency(task, context),
            .seasonal_pattern = detectSeasonalPattern(context),
            .variance_level = estimateVarianceLevel(task),
            
            // Performance hints
            .expected_cycles_log2 = @intCast(@min(31, std.math.log2_int(u64, context.estimated_cycles + 1))),
            .memory_footprint_log2 = @intCast(@min(31, std.math.log2_int(u64, getTaskMemoryFootprint(task) + 1))),
            .io_intensity = estimateIoIntensity(task),
            .cache_miss_rate = predictCacheMissRate(task),
            .branch_predictability = analyzeBranchPredictability(task),
            .vectorization_benefit = assessVectorizationBenefit(task),
        };
    }
    
    /// Fast hash for HashMap usage - optimized for cache performance
    pub fn hash(self: TaskFingerprint) u64 {
        // Use the first 64 bits directly, XOR with second 64 bits for better distribution
        const first_half = @as(u64, @bitCast([@sizeOf(u64)]u8, @as([@sizeOf(TaskFingerprint)]u8, @bitCast(self))[0..@sizeOf(u64)].*));
        const second_half = @as(u64, @bitCast([@sizeOf(u64)]u8, @as([@sizeOf(TaskFingerprint)]u8, @bitCast(self))[@sizeOf(u64)..].*));
        return first_half ^ second_half;
    }
    
    /// Check similarity for clustering similar tasks
    pub fn similarity(self: TaskFingerprint, other: TaskFingerprint) f32 {
        const self_bits = @as(u128, @bitCast(self));
        const other_bits = @as(u128, @bitCast(other));
        const diff_bits = self_bits ^ other_bits;
        const different_count = @popCount(diff_bits);
        return 1.0 - (@as(f32, @floatFromInt(different_count)) / 128.0);
    }
    
    /// Extract major characteristics for quick classification
    pub fn getCharacteristics(self: TaskFingerprint) TaskCharacteristics {
        return TaskCharacteristics{
            .is_cpu_intensive = self.cpu_intensity >= 12,
            .is_memory_bound = self.cpu_intensity <= 4,
            .is_vectorizable = self.vectorization_benefit >= 8 and self.simd_width >= 4,
            .is_numa_sensitive = self.memory_footprint_log2 >= 20, // 1MB+
            .is_cache_friendly = self.cache_locality >= 8,
            .needs_low_latency = self.time_sensitivity >= 2,
            .is_parallel_friendly = self.parallel_potential >= 8,
        };
    }
    
    pub const TaskCharacteristics = struct {
        is_cpu_intensive: bool,
        is_memory_bound: bool,
        is_vectorizable: bool,
        is_numa_sensitive: bool,
        is_cache_friendly: bool,
        needs_low_latency: bool,
        is_parallel_friendly: bool,
    };
};

// Execution context for fingerprint generation
pub const ExecutionContext = struct {
    current_numa_node: u32,
    application_phase: u8,
    current_hour: u8,
    estimated_cycles: u64,
    system_load: f32,
    available_cores: u32,
    
    // Historical execution patterns
    recent_task_types: [8]u32, // Ring buffer of recent task fingerprints
    execution_history_count: u64,
};

// Helper functions for fingerprint generation
fn hashCallSite(task: anytype, call_site: std.builtin.SourceLocation) u32 {
    // Combine function pointer, source file, and line number
    const func_ptr = @intFromPtr(@field(task, "func"));
    const file_hash = std.hash_map.hashString(call_site.file);
    const line_hash = call_site.line;
    
    // Fast 32-bit hash combining all components
    return @as(u32, @truncate(func_ptr ^ file_hash ^ line_hash));
}

fn classifyDataSize(size: usize) u8 {
    if (size == 0) return 0;
    return @intCast(@min(255, std.math.log2_int(usize, size)));
}

fn analyzeAccessPattern(task: anytype) TaskFingerprint.AccessPattern {
    // Analyze task data access patterns through heuristics
    const data_size = getTaskDataSize(task);
    const data_ptr = getTaskDataPtr(task);
    
    // Simple heuristics - can be enhanced with ML later
    if (data_size < 1024) return .sequential;
    if (@intFromPtr(data_ptr) % 64 == 0) return .sequential; // Cache-aligned suggests sequential
    if (data_size > 1024 * 1024) return .strided; // Large data often strided
    return .mixed;
}

fn detectOptimalSimdWidth(task: anytype) u4 {
    // Detect optimal SIMD width based on data type and size
    const data_size = getTaskDataSize(task);
    if (data_size < 16) return 1; // Too small for SIMD
    if (data_size >= 64) return 8; // Large enough for wide SIMD
    return 4; // Default SIMD width
}

fn estimateCacheLocality(task: anytype) u4 {
    // Estimate cache locality based on data access patterns
    const data_size = getTaskDataSize(task);
    if (data_size <= 32 * 1024) return 15; // Fits in L1
    if (data_size <= 256 * 1024) return 12; // Fits in L2
    if (data_size <= 8 * 1024 * 1024) return 8; // Fits in L3
    return 4; // Likely cache misses
}

// Placeholder implementations - these would be implemented based on task analysis
fn getTaskDataSize(task: anytype) usize {
    // Extract data size from task structure
    _ = task;
    return 1024; // Placeholder
}

fn getTaskDataPtr(task: anytype) *const anyopaque {
    // Extract data pointer from task structure
    _ = task;
    return @ptrFromInt(0x1000); // Placeholder
}

fn getTaskPriority(task: anytype) u2 {
    // Extract priority from task
    _ = task;
    return 1; // Normal priority
}

fn getTaskDependencies(task: anytype) usize {
    // Count task dependencies
    _ = task;
    return 0;
}

fn getTaskMemoryFootprint(task: anytype) usize {
    // Estimate memory footprint
    _ = task;
    return 4096;
}

fn estimateCpuIntensity(task: anytype) u4 {
    // Analyze CPU vs memory intensity
    _ = task;
    return 8; // Balanced
}

fn analyzeParallelPotential(task: anytype) u4 {
    // Assess parallelization potential
    _ = task;
    return 10; // Good potential
}

fn classifyTimeSensitivity(task: anytype) u2 {
    // Classify time sensitivity requirements
    _ = task;
    return 1; // Normal sensitivity
}

fn classifyExecutionFrequency(task: anytype, context: *const ExecutionContext) u3 {
    // Analyze how frequently this task type executes
    _ = task;
    _ = context;
    return 4; // Medium frequency
}

fn detectSeasonalPattern(context: *const ExecutionContext) u4 {
    // Detect daily/weekly/monthly patterns
    _ = context;
    return 0; // No pattern detected
}

fn estimateVarianceLevel(task: anytype) u4 {
    // Estimate execution time variance
    _ = task;
    return 6; // Medium variance
}

fn estimateIoIntensity(task: anytype) u4 {
    // Estimate I/O vs compute ratio
    _ = task;
    return 2; // Low I/O
}

fn predictCacheMissRate(task: anytype) u4 {
    // Predict cache miss characteristics
    _ = task;
    return 5; // Medium cache misses
}

fn analyzeBranchPredictability(task: anytype) u4 {
    // Analyze branch prediction friendliness
    _ = task;
    return 8; // Good predictability
}

fn assessVectorizationBenefit(task: anytype) u4 {
    // Assess SIMD optimization potential
    _ = task;
    return 6; // Medium benefit
}

// Example usage and testing
const testing = std.testing;

test "TaskFingerprint size and alignment" {
    try testing.expectEqual(16, @sizeOf(TaskFingerprint));
    try testing.expectEqual(128, @bitSizeOf(TaskFingerprint));
    
    // Ensure it's properly aligned for SIMD operations
    try testing.expect(@alignOf(TaskFingerprint) >= 8);
}

test "TaskFingerprint hash consistency" {
    const fp1 = TaskFingerprint{
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
    
    const fp2 = fp1; // Same fingerprint
    
    try testing.expectEqual(fp1.hash(), fp2.hash());
}

test "TaskFingerprint similarity calculation" {
    const fp1 = TaskFingerprint{
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
    
    // Identical fingerprints
    try testing.expectEqual(@as(f32, 1.0), fp1.similarity(fp1));
    
    // Different call site (32 bits different)
    var fp2 = fp1;
    fp2.call_site_hash = 0x87654321;
    const similarity = fp1.similarity(fp2);
    try testing.expect(similarity < 1.0 and similarity > 0.7); // Should be quite similar
}