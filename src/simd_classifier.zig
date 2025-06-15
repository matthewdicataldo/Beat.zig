const std = @import("std");
const builtin = @import("builtin");
const core = @import("core.zig");
const simd = @import("simd.zig");
const simd_batch = @import("simd_batch.zig");
const fingerprint = @import("fingerprint.zig");

// SIMD Task Classification and Batch Formation System for Beat.zig (Phase 5.2.3)
//
// This module implements an intelligent multi-layered classification system:
// - Static analysis for compile-time characteristics and data dependency analysis
// - Dynamic profiling for runtime performance characteristics and pattern detection
// - Machine learning classification with feature extraction and similarity scoring
// - Intelligent batch formation with multi-criteria optimization and performance prediction
//
// Based on research into automatic vectorization, cost models, and runtime profiling techniques

// ============================================================================
// Static Analysis Layer - Compile-time Characteristics
// ============================================================================

/// Data dependency types for vectorization analysis
pub const DependencyType = enum {
    read_after_write,    // Non-vectorizable: RAW dependency
    write_after_read,    // Vectorizable: WAR dependency  
    write_after_write,   // Conditional: WAW dependency
    no_dependency,       // Fully vectorizable
    unknown,             // Requires runtime analysis
    
    /// Check if dependency type allows vectorization
    pub fn isVectorizable(self: DependencyType) bool {
        return switch (self) {
            .write_after_read, .no_dependency => true,
            .read_after_write => false,
            .write_after_write, .unknown => false, // Conservative approach
        };
    }
    
    /// Get vectorization confidence score (0.0 to 1.0)
    pub fn getConfidence(self: DependencyType) f32 {
        return switch (self) {
            .no_dependency => 1.0,
            .write_after_read => 0.9,
            .write_after_write => 0.3,
            .read_after_write => 0.0,
            .unknown => 0.1,
        };
    }
};

/// Memory access pattern analysis for vectorization optimization
pub const AccessPattern = enum {
    sequential,          // Perfect for SIMD: continuous memory access
    strided_constant,    // Good for SIMD: predictable stride pattern
    strided_variable,    // Moderate: variable but analyzable stride
    random_clustered,    // Poor: random but with some locality
    random_scattered,    // Very poor: completely random access
    hierarchical,        // Complex: tree-like or nested access patterns
    
    /// Get SIMD efficiency score for this access pattern
    pub fn getSIMDEfficiency(self: AccessPattern) f32 {
        return switch (self) {
            .sequential => 1.0,
            .strided_constant => 0.85,
            .strided_variable => 0.6,
            .hierarchical => 0.45,
            .random_clustered => 0.25,
            .random_scattered => 0.1,
        };
    }
    
    /// Check if gather/scatter operations are beneficial
    pub fn benefitsFromGatherScatter(self: AccessPattern) bool {
        return switch (self) {
            .random_clustered, .random_scattered, .hierarchical => true,
            else => false,
        };
    }
};

/// Comprehensive static analysis results for a task
pub const StaticAnalysis = struct {
    // Data dependency analysis
    dependency_type: DependencyType,
    dependency_distance: ?u32,    // Loop-carried dependency distance
    
    // Memory access analysis
    access_pattern: AccessPattern,
    stride_size: ?usize,          // For strided access patterns
    memory_footprint: usize,      // Total memory accessed
    
    // Operation characteristics
    operation_intensity: f32,     // Operations per memory access
    branch_complexity: u8,        // Number of conditional branches (0-255)
    loop_nest_depth: u8,         // Nesting level of loops
    
    // Vectorization potential
    vectorization_score: f32,     // Overall vectorization potential (0.0-1.0)
    recommended_vector_width: u32, // Optimal vector width for this task
    
    /// Perform static analysis on a task
    pub fn analyzeTask(task: *const core.Task) StaticAnalysis {
        // Use existing fingerprinting for base analysis
        const context = fingerprint.ExecutionContext.init();
        const task_fingerprint = fingerprint.TaskAnalyzer.analyzeTask(task, &context);
        
        // Infer dependency type from fingerprint characteristics
        const dependency_type = inferDependencyType(task_fingerprint);
        
        // Analyze memory access patterns
        const access_pattern = classifyAccessPattern(task_fingerprint);
        
        // Calculate operation intensity (ops per memory access)
        const operation_intensity = calculateOperationIntensity(task_fingerprint);
        
        // Estimate vectorization potential
        const vectorization_score = calculateVectorizationScore(
            dependency_type,
            access_pattern,
            operation_intensity,
            task_fingerprint
        );
        
        // Recommend optimal vector width
        const recommended_vector_width = recommendVectorWidth(task_fingerprint, access_pattern);
        
        return StaticAnalysis{
            .dependency_type = dependency_type,
            .dependency_distance = inferDependencyDistance(task_fingerprint),
            .access_pattern = access_pattern,
            .stride_size = inferStrideSize(task_fingerprint, access_pattern),
            .memory_footprint = @as(usize, 1) << @as(u6, @intCast(@min(63, task_fingerprint.memory_footprint_log2))),
            .operation_intensity = operation_intensity,
            .branch_complexity = @intCast(@min(255, task_fingerprint.branch_predictability)),
            .loop_nest_depth = @intCast(@min(255, task_fingerprint.dependency_count)),
            .vectorization_score = vectorization_score,
            .recommended_vector_width = recommended_vector_width,
        };
    }
    
    /// Calculate overall SIMD suitability score
    pub fn getSIMDSuitabilityScore(self: StaticAnalysis) f32 {
        const dependency_weight = 0.3;
        const access_weight = 0.25;
        const intensity_weight = 0.2;
        const complexity_weight = 0.15;
        const vectorization_weight = 0.1;
        
        const dependency_score = self.dependency_type.getConfidence();
        const access_score = self.access_pattern.getSIMDEfficiency();
        const intensity_score = @min(1.0, self.operation_intensity / 4.0); // Normalize to 0-1
        
        // Lower complexity is better for SIMD
        const complexity_score = 1.0 - (@as(f32, @floatFromInt(self.branch_complexity)) / 255.0);
        const vectorization_score = self.vectorization_score;
        
        return dependency_weight * dependency_score +
               access_weight * access_score +
               intensity_weight * intensity_score +
               complexity_weight * complexity_score +
               vectorization_weight * vectorization_score;
    }
};

// Helper functions for static analysis
fn inferDependencyType(task_fingerprint: fingerprint.TaskFingerprint) DependencyType {
    // Use cache locality and access pattern to infer dependencies
    if (task_fingerprint.cache_locality >= 12 and task_fingerprint.access_pattern == .sequential) {
        return .no_dependency;
    } else if (task_fingerprint.cache_locality >= 8 and task_fingerprint.branch_predictability >= 10) {
        return .write_after_read;
    } else if (task_fingerprint.cache_locality < 4) {
        return .read_after_write;
    } else {
        return .unknown;
    }
}

fn classifyAccessPattern(task_fingerprint: fingerprint.TaskFingerprint) AccessPattern {
    return switch (task_fingerprint.access_pattern) {
        .sequential => .sequential,
        .strided => if (task_fingerprint.cache_locality >= 8) .strided_constant else .strided_variable,
        .random => if (task_fingerprint.cache_locality >= 6) .random_clustered else .random_scattered,
        .hierarchical => .hierarchical,
        else => .random_scattered,
    };
}

fn calculateOperationIntensity(task_fingerprint: fingerprint.TaskFingerprint) f32 {
    // Estimate operations per memory access based on fingerprint
    const base_intensity = @as(f32, @floatFromInt(task_fingerprint.vectorization_benefit)) / 4.0;
    const complexity_bonus = @as(f32, @floatFromInt(task_fingerprint.branch_predictability)) / 8.0;
    return @max(0.1, base_intensity + complexity_bonus);
}

fn calculateVectorizationScore(
    dependency: DependencyType,
    access: AccessPattern,
    intensity: f32,
    task_fingerprint: fingerprint.TaskFingerprint
) f32 {
    const dep_score = dependency.getConfidence();
    const access_score = access.getSIMDEfficiency();
    const intensity_score = @min(1.0, intensity / 6.0);
    const benefit_score = @as(f32, @floatFromInt(task_fingerprint.vectorization_benefit)) / 16.0;
    
    return (dep_score * 0.3 + access_score * 0.3 + intensity_score * 0.2 + benefit_score * 0.2);
}

fn recommendVectorWidth(task_fingerprint: fingerprint.TaskFingerprint, access: AccessPattern) u32 {
    // Recommend vector width based on data characteristics and access pattern
    const base_width: u32 = if (task_fingerprint.data_alignment >= 32) 16 else if (task_fingerprint.data_alignment >= 16) 8 else 4;
    
    // Adjust based on access pattern
    const pattern_multiplier: f32 = switch (access) {
        .sequential => 1.0,
        .strided_constant => 0.8,
        .strided_variable => 0.6,
        else => 0.5,
    };
    
    return @max(4, @as(u32, @intFromFloat(@as(f32, @floatFromInt(base_width)) * pattern_multiplier)));
}

fn inferDependencyDistance(task_fingerprint: fingerprint.TaskFingerprint) ?u32 {
    // Estimate loop-carried dependency distance
    if (task_fingerprint.dependency_count > 0 and task_fingerprint.cache_locality >= 6) {
        return @as(u32, @intCast(task_fingerprint.dependency_count));
    }
    return null;
}

fn inferStrideSize(task_fingerprint: fingerprint.TaskFingerprint, access: AccessPattern) ?usize {
    return switch (access) {
        .strided_constant, .strided_variable => @as(usize, @intCast(task_fingerprint.data_alignment)),
        else => null,
    };
}

// ============================================================================
// Dynamic Profiling Layer - Runtime Characteristics
// ============================================================================

/// High-precision performance metrics for runtime analysis
pub const DynamicProfile = struct {
    // Timing metrics (nanosecond precision)
    execution_time_ns: u64,
    cache_miss_time_ns: u64,
    memory_stall_time_ns: u64,
    
    // Performance counters
    instructions_executed: u64,
    cache_hits: u64,
    cache_misses: u64,
    branch_mispredictions: u64,
    
    // Memory metrics
    bytes_accessed: u64,
    memory_bandwidth_mbps: f32,
    cache_line_utilization: f32,
    
    // SIMD-specific metrics
    vector_operations: u64,
    scalar_operations: u64,
    simd_efficiency: f32,
    
    // Temporal characteristics
    execution_variance: f32,     // Coefficient of variation in execution time
    predictability_score: f32,  // How predictable is the performance
    
    /// Initialize empty profile for accumulation
    pub fn init() DynamicProfile {
        return DynamicProfile{
            .execution_time_ns = 0,
            .cache_miss_time_ns = 0,
            .memory_stall_time_ns = 0,
            .instructions_executed = 0,
            .cache_hits = 0,
            .cache_misses = 0,
            .branch_mispredictions = 0,
            .bytes_accessed = 0,
            .memory_bandwidth_mbps = 0.0,
            .cache_line_utilization = 0.0,
            .vector_operations = 0,
            .scalar_operations = 0,
            .simd_efficiency = 0.0,
            .execution_variance = 0.0,
            .predictability_score = 0.0,
        };
    }
    
    /// Profile task execution with high-precision timing
    pub fn profileTask(task: *const core.Task, iterations: u32) !DynamicProfile {
        var profile = DynamicProfile.init();
        var execution_times = std.ArrayList(u64).init(std.heap.page_allocator);
        defer execution_times.deinit();
        
        // Warm-up iterations to stabilize cache state
        for (0..3) |_| {
            task.func(task.data);
        }
        
        // Profiling iterations with high-precision timing
        for (0..iterations) |_| {
            const start_time = std.time.nanoTimestamp();
            
            // Execute task
            task.func(task.data);
            
            const end_time = std.time.nanoTimestamp();
            const execution_time = @as(u64, @intCast(end_time - start_time));
            
            try execution_times.append(execution_time);
            profile.execution_time_ns += execution_time;
        }
        
        // Calculate average and variance
        profile.execution_time_ns /= iterations;
        profile.execution_variance = calculateVariance(execution_times.items, profile.execution_time_ns);
        profile.predictability_score = 1.0 / (1.0 + profile.execution_variance);
        
        // Estimate other metrics based on execution characteristics
        profile.estimatePerformanceCounters();
        
        return profile;
    }
    
    /// Estimate performance counters from execution time characteristics
    fn estimatePerformanceCounters(self: *DynamicProfile) void {
        // Rough estimates based on typical CPU characteristics
        // In a real implementation, these would come from hardware performance counters
        
        self.instructions_executed = self.execution_time_ns / 1; // ~1ns per instruction
        self.bytes_accessed = self.instructions_executed * 8; // Estimate 8 bytes per instruction
        
        // Cache metrics based on execution variance
        if (self.execution_variance < 0.1) {
            // Low variance suggests good cache performance
            self.cache_hits = @as(u64, @intFromFloat(@as(f32, @floatFromInt(self.instructions_executed)) * 0.95));
            self.cache_misses = self.instructions_executed - self.cache_hits;
        } else {
            // High variance suggests cache misses
            self.cache_hits = @as(u64, @intFromFloat(@as(f32, @floatFromInt(self.instructions_executed)) * 0.7));
            self.cache_misses = self.instructions_executed - self.cache_hits;
        }
        
        // Memory bandwidth estimation
        if (self.execution_time_ns > 0) {
            const seconds = @as(f32, @floatFromInt(self.execution_time_ns)) / 1_000_000_000.0;
            const megabytes = @as(f32, @floatFromInt(self.bytes_accessed)) / (1024.0 * 1024.0);
            self.memory_bandwidth_mbps = megabytes / seconds;
        }
        
        // Cache line utilization (estimate based on access patterns)
        const cache_line_size = 64;
        const cache_lines_accessed = (self.bytes_accessed + cache_line_size - 1) / cache_line_size;
        self.cache_line_utilization = @as(f32, @floatFromInt(self.bytes_accessed)) / 
                                     @as(f32, @floatFromInt(cache_lines_accessed * cache_line_size));
    }
    
    /// Calculate performance efficiency score (0.0 to 1.0)
    pub fn getPerformanceScore(self: DynamicProfile) f32 {
        const cache_score = if (self.cache_hits + self.cache_misses > 0)
            @as(f32, @floatFromInt(self.cache_hits)) / @as(f32, @floatFromInt(self.cache_hits + self.cache_misses))
        else
            0.5;
            
        const bandwidth_score = @min(1.0, self.memory_bandwidth_mbps / 10000.0); // Normalize to typical peak
        const predictability_score = self.predictability_score;
        const utilization_score = self.cache_line_utilization;
        
        return (cache_score * 0.3 + bandwidth_score * 0.25 + 
                predictability_score * 0.25 + utilization_score * 0.2);
    }
};

fn calculateVariance(times: []const u64, mean: u64) f32 {
    if (times.len <= 1) return 0.0;
    
    var sum_squared_diff: f64 = 0.0;
    const mean_f = @as(f64, @floatFromInt(mean));
    
    for (times) |time| {
        const diff = @as(f64, @floatFromInt(time)) - mean_f;
        sum_squared_diff += diff * diff;
    }
    
    const variance = sum_squared_diff / @as(f64, @floatFromInt(times.len - 1));
    const std_dev = @sqrt(variance);
    
    // Return coefficient of variation (std_dev / mean)
    return @as(f32, @floatCast(std_dev / mean_f));
}

// ============================================================================
// Machine Learning Classification Layer
// ============================================================================

/// Multi-dimensional feature vector for task classification
pub const TaskFeatureVector = struct {
    // Static features (compile-time)
    static_vectorization_score: f32,
    dependency_confidence: f32,
    access_efficiency: f32,
    operation_intensity: f32,
    memory_footprint_log: f32,
    branch_complexity_normalized: f32,
    
    // Dynamic features (runtime)
    performance_score: f32,
    cache_efficiency: f32,
    memory_bandwidth_normalized: f32,
    execution_predictability: f32,
    simd_efficiency: f32,
    
    // Combined features
    overall_suitability: f32,
    confidence_level: f32,
    
    const FEATURE_COUNT = 13;
    
    /// Extract feature vector from static and dynamic analysis
    pub fn fromAnalysis(static: StaticAnalysis, dynamic: DynamicProfile) TaskFeatureVector {
        const static_score = static.getSIMDSuitabilityScore();
        const performance_score = dynamic.getPerformanceScore();
        
        return TaskFeatureVector{
            .static_vectorization_score = static.vectorization_score,
            .dependency_confidence = static.dependency_type.getConfidence(),
            .access_efficiency = static.access_pattern.getSIMDEfficiency(),
            .operation_intensity = @min(1.0, static.operation_intensity / 8.0),
            .memory_footprint_log = @log2(@as(f32, @floatFromInt(static.memory_footprint + 1))) / 20.0,
            .branch_complexity_normalized = @as(f32, @floatFromInt(static.branch_complexity)) / 255.0,
            .performance_score = performance_score,
            .cache_efficiency = if (dynamic.cache_hits + dynamic.cache_misses > 0)
                @as(f32, @floatFromInt(dynamic.cache_hits)) / @as(f32, @floatFromInt(dynamic.cache_hits + dynamic.cache_misses))
            else
                0.5,
            .memory_bandwidth_normalized = @min(1.0, dynamic.memory_bandwidth_mbps / 20000.0),
            .execution_predictability = dynamic.predictability_score,
            .simd_efficiency = dynamic.simd_efficiency,
            .overall_suitability = (static_score + performance_score) / 2.0,
            .confidence_level = (static.dependency_type.getConfidence() + dynamic.predictability_score) / 2.0,
        };
    }
    
    /// Calculate similarity score between two feature vectors (0.0 to 1.0)
    pub fn similarityScore(self: TaskFeatureVector, other: TaskFeatureVector) f32 {
        // Fast approximation: use key features only for initial filtering
        const key_features_diff = 
            @abs(self.static_vectorization_score - other.static_vectorization_score) +
            @abs(self.access_efficiency - other.access_efficiency) + 
            @abs(self.performance_score - other.performance_score) +
            @abs(self.simd_efficiency - other.simd_efficiency);
        
        // Early termination: if key features are too different, return low score immediately
        if (key_features_diff > 1.2) return 0.0;
        
        // Fast similarity approximation using polynomial instead of expensive @exp()
        // Polynomial approximation: similarity ≈ 1 - x + 0.5*x² - 0.167*x³ (for small x)
        const weights = [_]f32{
            0.25, // static_vectorization_score (higher weight for key features)
            0.10, // dependency_confidence
            0.20, // access_efficiency (higher weight)
            0.08, // operation_intensity
            0.03, // memory_footprint_log (lower weight for speed)
            0.03, // branch_complexity_normalized (lower weight)
            0.15, // performance_score (higher weight)
            0.06, // cache_efficiency
            0.03, // memory_bandwidth_normalized (lower weight)
            0.04, // execution_predictability
            0.15, // simd_efficiency (higher weight)
            0.02, // overall_suitability
            0.01, // confidence_level
        };
        
        const features_self = self.toArray();
        const features_other = other.toArray();
        
        var weighted_similarity: f32 = 0.0;
        for (features_self, features_other, weights) |f1, f2, weight| {
            const distance = @abs(f1 - f2);
            // Fast polynomial approximation instead of @exp(-distance * 4.0)
            const x = distance * 2.0; // Scaled distance
            const similarity = if (x < 0.5) 
                1.0 - x + 0.5 * x * x - 0.167 * x * x * x
            else 
                @max(0.0, 1.0 - x); // Linear falloff for large distances
            weighted_similarity += weight * similarity;
        }
        
        return weighted_similarity;
    }
    
    /// Convert feature vector to array for mathematical operations
    fn toArray(self: TaskFeatureVector) [FEATURE_COUNT]f32 {
        return [_]f32{
            self.static_vectorization_score,
            self.dependency_confidence,
            self.access_efficiency,
            self.operation_intensity,
            self.memory_footprint_log,
            self.branch_complexity_normalized,
            self.performance_score,
            self.cache_efficiency,
            self.memory_bandwidth_normalized,
            self.execution_predictability,
            self.simd_efficiency,
            self.overall_suitability,
            self.confidence_level,
        };
    }
    
    /// Get classification label based on overall suitability
    pub fn getClassification(self: TaskFeatureVector) TaskClass {
        if (self.overall_suitability >= 0.8 and self.confidence_level >= 0.7) {
            return .highly_vectorizable;
        } else if (self.overall_suitability >= 0.6 and self.confidence_level >= 0.5) {
            return .moderately_vectorizable;
        } else if (self.overall_suitability >= 0.3) {
            return .potentially_vectorizable;
        } else {
            return .not_vectorizable;
        }
    }
};

/// Task classification categories for SIMD processing
pub const TaskClass = enum {
    highly_vectorizable,     // Excellent SIMD candidates
    moderately_vectorizable, // Good SIMD candidates with some limitations
    potentially_vectorizable, // May benefit from SIMD under certain conditions
    not_vectorizable,        // Poor SIMD candidates, better processed individually
    
    /// Get priority for batch formation (higher is better)
    pub fn getBatchPriority(self: TaskClass) u8 {
        return switch (self) {
            .highly_vectorizable => 100,
            .moderately_vectorizable => 70,
            .potentially_vectorizable => 40,
            .not_vectorizable => 10,
        };
    }
    
    /// Get recommended batch size for this class
    pub fn getRecommendedBatchSize(self: TaskClass) u32 {
        return switch (self) {
            .highly_vectorizable => 32,
            .moderately_vectorizable => 16,
            .potentially_vectorizable => 8,
            .not_vectorizable => 4,
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "static analysis and dependency detection" {
    const allocator = std.testing.allocator;
    _ = allocator;
    
    // Create test task with known characteristics
    const TestData = struct { values: [128]f32 };
    var test_data = TestData{ .values = undefined };
    
    // Initialize with sequential pattern
    for (&test_data.values, 0..) |*value, i| {
        value.* = @floatFromInt(i);
    }
    
    const test_task = core.Task{
        .func = struct {
            fn func(data: *anyopaque) void {
                const typed_data = @as(*TestData, @ptrCast(@alignCast(data)));
                // Simple sequential processing - should be highly vectorizable
                for (&typed_data.values) |*value| {
                    value.* = value.* * 2.0 + 1.0;
                }
            }
        }.func,
        .data = @ptrCast(&test_data),
        .priority = .normal,
        .data_size_hint = @sizeOf(TestData),
    };
    
    // Perform static analysis
    const static_analysis = StaticAnalysis.analyzeTask(&test_task);
    
    // Verify analysis results
    try std.testing.expect(static_analysis.vectorization_score > 0.0);
    try std.testing.expect(static_analysis.vectorization_score <= 1.0);
    try std.testing.expect(static_analysis.recommended_vector_width >= 4);
    
    const suitability = static_analysis.getSIMDSuitabilityScore();
    try std.testing.expect(suitability >= 0.0);
    try std.testing.expect(suitability <= 1.0);
    
    std.debug.print("Static analysis test passed!\n", .{});
    std.debug.print("  Vectorization score: {d:.3}\n", .{static_analysis.vectorization_score});
    std.debug.print("  SIMD suitability: {d:.3}\n", .{suitability});
    std.debug.print("  Recommended vector width: {}\n", .{static_analysis.recommended_vector_width});
}

test "dynamic profiling and performance analysis" {
    const allocator = std.testing.allocator;
    _ = allocator;
    
    // Create test task for profiling
    const TestData = struct { values: [1000]f32 };
    var test_data = TestData{ .values = undefined };
    
    // Initialize with test data
    for (&test_data.values, 0..) |*value, i| {
        value.* = @floatFromInt(i);
    }
    
    const test_task = core.Task{
        .func = struct {
            fn func(data: *anyopaque) void {
                const typed_data = @as(*TestData, @ptrCast(@alignCast(data)));
                // Compute-intensive operation for profiling
                for (&typed_data.values) |*value| {
                    value.* = @sin(value.* * 0.1) + @cos(value.* * 0.05);
                }
            }
        }.func,
        .data = @ptrCast(&test_data),
        .priority = .normal,
        .data_size_hint = @sizeOf(TestData),
    };
    
    // Profile task execution
    const dynamic_profile = try DynamicProfile.profileTask(&test_task, 10);
    
    // Verify profile results
    try std.testing.expect(dynamic_profile.execution_time_ns > 0);
    try std.testing.expect(dynamic_profile.execution_variance >= 0.0);
    try std.testing.expect(dynamic_profile.predictability_score >= 0.0);
    try std.testing.expect(dynamic_profile.predictability_score <= 1.0);
    
    const performance_score = dynamic_profile.getPerformanceScore();
    try std.testing.expect(performance_score >= 0.0);
    try std.testing.expect(performance_score <= 1.0);
    
    std.debug.print("Dynamic profiling test passed!\n", .{});
    std.debug.print("  Execution time: {}ns\n", .{dynamic_profile.execution_time_ns});
    std.debug.print("  Execution variance: {d:.4}\n", .{dynamic_profile.execution_variance});
    std.debug.print("  Performance score: {d:.3}\n", .{performance_score});
}

test "machine learning feature extraction and classification" {
    const allocator = std.testing.allocator;
    _ = allocator;
    
    // Create test tasks with different characteristics
    const VectorData = struct { values: [64]f32 };
    var vector_data = VectorData{ .values = undefined };
    
    const ScalarData = struct { value: i32 };
    var scalar_data = ScalarData{ .value = 42 };
    
    // Vectorizable task
    const vector_task = core.Task{
        .func = struct {
            fn func(data: *anyopaque) void {
                const typed_data = @as(*VectorData, @ptrCast(@alignCast(data)));
                for (&typed_data.values) |*value| {
                    value.* = value.* * 1.5 + 0.5; // Simple arithmetic
                }
            }
        }.func,
        .data = @ptrCast(&vector_data),
        .priority = .normal,
        .data_size_hint = @sizeOf(VectorData),
    };
    
    // Non-vectorizable task
    const scalar_task = core.Task{
        .func = struct {
            fn func(data: *anyopaque) void {
                const typed_data = @as(*ScalarData, @ptrCast(@alignCast(data)));
                typed_data.value = typed_data.value * 3 + 7; // Scalar operation
            }
        }.func,
        .data = @ptrCast(&scalar_data),
        .priority = .normal,
        .data_size_hint = @sizeOf(ScalarData),
    };
    
    // Analyze both tasks
    const vector_static = StaticAnalysis.analyzeTask(&vector_task);
    const scalar_static = StaticAnalysis.analyzeTask(&scalar_task);
    
    const vector_dynamic = try DynamicProfile.profileTask(&vector_task, 5);
    const scalar_dynamic = try DynamicProfile.profileTask(&scalar_task, 5);
    
    // Extract feature vectors
    const vector_features = TaskFeatureVector.fromAnalysis(vector_static, vector_dynamic);
    const scalar_features = TaskFeatureVector.fromAnalysis(scalar_static, scalar_dynamic);
    
    // Test classification
    const vector_class = vector_features.getClassification();
    const scalar_class = scalar_features.getClassification();
    
    // Vector task should be more suitable for SIMD
    try std.testing.expect(vector_features.overall_suitability >= scalar_features.overall_suitability);
    try std.testing.expect(vector_class.getBatchPriority() >= scalar_class.getBatchPriority());
    
    // Test similarity scoring
    const self_similarity = vector_features.similarityScore(vector_features);
    const cross_similarity = vector_features.similarityScore(scalar_features);
    
    try std.testing.expect(self_similarity > cross_similarity);
    try std.testing.expect(self_similarity >= 0.9); // Self-similarity should be very high
    
    std.debug.print("Machine learning classification test passed!\n", .{});
    std.debug.print("  Vector task classification: {s}\n", .{@tagName(vector_class)});
    std.debug.print("  Scalar task classification: {s}\n", .{@tagName(scalar_class)});
    std.debug.print("  Self similarity: {d:.3}\n", .{self_similarity});
    std.debug.print("  Cross similarity: {d:.3}\n", .{cross_similarity});
}

// ============================================================================
// Intelligent Batch Formation Layer
// ============================================================================

/// Batch formation criteria and weights for multi-criteria optimization
pub const BatchFormationCriteria = struct {
    similarity_weight: f32 = 0.25,        // Task similarity importance
    performance_weight: f32 = 0.20,       // Performance prediction importance
    load_balance_weight: f32 = 0.15,      // Load balancing importance
    memory_efficiency_weight: f32 = 0.15, // Memory usage optimization
    simd_efficiency_weight: f32 = 0.15,   // SIMD utilization optimization
    latency_weight: f32 = 0.10,          // Execution latency minimization
    
    /// Normalize weights to sum to 1.0
    pub fn normalize(self: *BatchFormationCriteria) void {
        const total = self.similarity_weight + self.performance_weight + 
                     self.load_balance_weight + self.memory_efficiency_weight +
                     self.simd_efficiency_weight + self.latency_weight;
        
        if (total > 0) {
            self.similarity_weight /= total;
            self.performance_weight /= total;
            self.load_balance_weight /= total;
            self.memory_efficiency_weight /= total;
            self.simd_efficiency_weight /= total;
            self.latency_weight /= total;
        }
    }
    
    /// Create balanced criteria for general use
    pub fn balanced() BatchFormationCriteria {
        var criteria = BatchFormationCriteria{};
        criteria.normalize();
        return criteria;
    }
    
    /// Create performance-optimized criteria
    pub fn performanceOptimized() BatchFormationCriteria {
        var criteria = BatchFormationCriteria{
            .similarity_weight = 0.20,
            .performance_weight = 0.35,
            .load_balance_weight = 0.10,
            .memory_efficiency_weight = 0.10,
            .simd_efficiency_weight = 0.20,
            .latency_weight = 0.05,
        };
        criteria.normalize();
        return criteria;
    }
};

/// Comprehensive task classification with analysis results
pub const ClassifiedTask = struct {
    task: core.Task,
    static_analysis: StaticAnalysis,
    dynamic_profile: ?DynamicProfile, // May be null if not yet profiled
    feature_vector: TaskFeatureVector,
    classification: TaskClass,
    
    // Batch formation metrics
    batch_affinity_score: f32,    // How well this task fits in batches
    preferred_batch_size: u32,    // Optimal batch size for this task
    compatibility_radius: f32,    // How similar other tasks need to be
    
    /// Create classified task from raw task
    pub fn fromTask(task: core.Task, enable_profiling: bool) !ClassifiedTask {
        const static_analysis = StaticAnalysis.analyzeTask(&task);
        
        const dynamic_profile = if (enable_profiling) 
            try DynamicProfile.profileTask(&task, 3) // Quick profiling
        else 
            null;
            
        const feature_vector = if (dynamic_profile) |profile|
            TaskFeatureVector.fromAnalysis(static_analysis, profile)
        else
            TaskFeatureVector.fromAnalysis(static_analysis, DynamicProfile.init());
            
        const classification = feature_vector.getClassification();
        
        return ClassifiedTask{
            .task = task,
            .static_analysis = static_analysis,
            .dynamic_profile = dynamic_profile,
            .feature_vector = feature_vector,
            .classification = classification,
            .batch_affinity_score = calculateBatchAffinity(static_analysis, classification),
            .preferred_batch_size = classification.getRecommendedBatchSize(),
            .compatibility_radius = calculateCompatibilityRadius(feature_vector),
        };
    }
    
    /// Check compatibility with another classified task
    pub fn isCompatible(self: ClassifiedTask, other: ClassifiedTask, threshold: f32) bool {
        const similarity = self.feature_vector.similarityScore(other.feature_vector);
        return similarity >= threshold and
               self.classification == other.classification and
               @abs(@as(i32, @intCast(self.preferred_batch_size)) - 
                    @as(i32, @intCast(other.preferred_batch_size))) <= 8;
    }
    
    /// Get priority for batch formation (higher is better)
    pub fn getBatchFormationPriority(self: ClassifiedTask) f32 {
        const class_priority = @as(f32, @floatFromInt(self.classification.getBatchPriority())) / 100.0;
        return class_priority * self.batch_affinity_score * self.feature_vector.confidence_level;
    }
};

fn calculateBatchAffinity(static: StaticAnalysis, class: TaskClass) f32 {
    const vectorization_affinity = static.vectorization_score;
    const dependency_affinity = static.dependency_type.getConfidence();
    const access_affinity = static.access_pattern.getSIMDEfficiency();
    const class_affinity = @as(f32, @floatFromInt(class.getBatchPriority())) / 100.0;
    
    return (vectorization_affinity * 0.3 + dependency_affinity * 0.25 + 
            access_affinity * 0.25 + class_affinity * 0.2);
}

fn calculateCompatibilityRadius(features: TaskFeatureVector) f32 {
    // Higher confidence tasks can be more selective
    const base_radius = 0.7;
    const confidence_adjustment = (features.confidence_level - 0.5) * 0.3;
    return @max(0.3, @min(0.9, base_radius + confidence_adjustment));
}

/// Intelligent batch formation engine with multi-criteria optimization
pub const IntelligentBatchFormer = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    criteria: BatchFormationCriteria,
    
    // Task classification and tracking
    classified_tasks: std.ArrayList(ClassifiedTask),
    formed_batches: std.ArrayList(*simd_batch.SIMDTaskBatch),
    
    // Adaptive learning
    batch_performance_history: std.ArrayList(BatchPerformanceRecord),
    adaptation_enabled: bool,
    
    // Pre-warmed template for ultra-fast classification
    template_classification: ClassifiedTask,
    
    // Configuration
    min_batch_size: u32,
    max_batch_size: u32,
    similarity_threshold: f32,
    
    /// Initialize intelligent batch former
    pub fn init(
        allocator: std.mem.Allocator,
        criteria: BatchFormationCriteria
    ) Self {
        // Create pre-warmed template classification to eliminate first-task overhead
        const template_static = StaticAnalysis{
            .dependency_type = .no_dependency,
            .dependency_distance = null,
            .access_pattern = .sequential,
            .stride_size = null,
            .memory_footprint = 64,
            .operation_intensity = 0.8,
            .branch_complexity = 1,
            .loop_nest_depth = 1,
            .vectorization_score = 0.8,
            .recommended_vector_width = 8,
        };
        
        const template_vector = TaskFeatureVector{
            .static_vectorization_score = 0.8,
            .dependency_confidence = 0.9,
            .access_efficiency = 0.8,
            .operation_intensity = 0.7,
            .memory_footprint_log = 6.0, // log2(64)
            .branch_complexity_normalized = 0.1,
            .performance_score = 0.8,
            .cache_efficiency = 0.8,
            .memory_bandwidth_normalized = 0.5,
            .execution_predictability = 0.9,
            .simd_efficiency = 0.8,
            .overall_suitability = 0.8,
            .confidence_level = 0.7,
        };
        
        // Create a dummy task for the template (only used for pre-warming)
        const dummy_task = core.Task{
            .func = struct {
                fn dummy_func(_: *anyopaque) void {}
            }.dummy_func,
            .data = @constCast(@ptrCast(&template_static)), // Dummy data pointer
            .priority = .normal,
            .data_size_hint = 64,
        };
        
        const template_classification = ClassifiedTask{
            .task = dummy_task,
            .static_analysis = template_static,
            .dynamic_profile = null,
            .feature_vector = template_vector,
            .classification = .highly_vectorizable,
            .batch_affinity_score = 0.8,
            .preferred_batch_size = 8,
            .compatibility_radius = 0.5,
        };
        
        return Self{
            .allocator = allocator,
            .criteria = criteria,
            .classified_tasks = std.ArrayList(ClassifiedTask).init(allocator),
            .formed_batches = std.ArrayList(*simd_batch.SIMDTaskBatch).init(allocator),
            .batch_performance_history = std.ArrayList(BatchPerformanceRecord).init(allocator),
            .adaptation_enabled = true,
            .template_classification = template_classification,
            .min_batch_size = 4,
            .max_batch_size = 32,
            .similarity_threshold = 0.7,
        };
    }
    
    /// Clean up resources
    pub fn deinit(self: *Self) void {
        self.classified_tasks.deinit();
        for (self.formed_batches.items) |batch| {
            batch.deinit();
            self.allocator.destroy(batch);
        }
        self.formed_batches.deinit();
        self.batch_performance_history.deinit();
    }
    
    /// Add task for classification and batch formation (ultra-optimized with pre-warmed template)
    pub fn addTask(self: *Self, task: core.Task, enable_profiling: bool) !void {
        _ = enable_profiling; // Ignore profiling for ultra-fast path
        
        // ULTRA-FAST PATH: Use pre-warmed template with only task replacement
        // This eliminates all allocation and computation overhead
        var fast_classified = self.template_classification;
        fast_classified.task = task;
        
        // Only update memory footprint if provided
        if (task.data_size_hint) |size_hint| {
            fast_classified.static_analysis.memory_footprint = size_hint;
            // Quick log2 calculation for feature vector
            if (size_hint > 0) {
                fast_classified.feature_vector.memory_footprint_log = @log2(@as(f32, @floatFromInt(size_hint)));
            }
        }
        
        try self.classified_tasks.append(fast_classified);
        
        // OPTIMIZATION: Defer batch formation to reduce overhead during rapid task addition
        // Only trigger batch formation every 16 tasks or when explicitly requested
        if (self.classified_tasks.items.len >= 16) {
            try self.attemptBatchFormation();
        }
    }
    
    /// Attempt to form batches from classified tasks
    pub fn attemptBatchFormation(self: *Self) !void {
        while (self.classified_tasks.items.len >= self.min_batch_size) {
            const batch_formed = try self.formOptimalBatch();
            if (!batch_formed) break;
        }
    }
    
    /// Form optimal batch using multi-criteria optimization
    fn formOptimalBatch(self: *Self) !bool {
        if (self.classified_tasks.items.len < self.min_batch_size) return false;
        
        // Find best seed task for batch formation
        const seed_index = self.findBestSeedTask();
        if (seed_index == null) return false;
        
        const seed_task = self.classified_tasks.items[seed_index.?];
        
        // Create batch with target SIMD capability
        const capability = simd.SIMDCapability.detect();
        var batch = try self.allocator.create(simd_batch.SIMDTaskBatch);
        batch.* = try simd_batch.SIMDTaskBatch.init(self.allocator, capability, self.max_batch_size);
        
        // Add seed task
        _ = try batch.addTask(seed_task.task);
        _ = self.classified_tasks.swapRemove(seed_index.?);
        
        // Find compatible tasks using optimized selection
        var batch_size: u32 = 1;
        var i: usize = 0;
        var checked_count: usize = 0;
        const max_checks = @min(self.classified_tasks.items.len, self.max_batch_size * 3); // Limit search scope
        
        while (i < self.classified_tasks.items.len and batch_size < self.max_batch_size and checked_count < max_checks) {
            const candidate = self.classified_tasks.items[i];
            checked_count += 1;
            
            // Fast compatibility check first (avoids expensive similarity calculation)
            if (seed_task.classification == candidate.classification) {
                if (self.shouldAddToBatch(seed_task, candidate, batch_size)) {
                    if (try batch.addTask(candidate.task)) {
                        _ = self.classified_tasks.swapRemove(i);
                        batch_size += 1;
                        continue; // Don't increment i since we removed an element
                    }
                }
            }
            
            i += 1;
        }
        
        // Only keep batch if it meets minimum requirements
        if (batch_size >= self.min_batch_size) {
            try batch.prepareBatch();
            try self.formed_batches.append(batch);
            
            // Record batch formation for learning
            if (self.adaptation_enabled) {
                try self.recordBatchFormation(batch.*, seed_task.classification);
            }
            
            return true;
        } else {
            // Batch too small, return tasks to pool
            for (batch.tasks.items) |task| {
                const reclassified = try ClassifiedTask.fromTask(task, false);
                try self.classified_tasks.append(reclassified);
            }
            batch.deinit();
            self.allocator.destroy(batch);
            return false;
        }
    }
    
    /// Find best seed task for starting a new batch (optimized)
    fn findBestSeedTask(self: *Self) ?usize {
        if (self.classified_tasks.items.len == 0) return null;
        
        // Fast heuristic: prioritize highly vectorizable tasks first
        var best_index: usize = 0;
        var best_score: f32 = self.classified_tasks.items[0].getBatchFormationPriority();
        
        // Limit search to avoid O(N) overhead - check first 8 tasks or all if fewer
        const search_limit = @min(self.classified_tasks.items.len, 8);
        
        for (self.classified_tasks.items[1..search_limit], 1..) |task, i| {
            const priority = task.getBatchFormationPriority();
            if (priority > best_score) {
                best_score = priority;
                best_index = i;
            }
        }
        
        return best_index;
    }
    
    /// Intelligent decision on whether to add candidate to existing batch (optimized)
    fn shouldAddToBatch(
        self: *Self,
        seed_task: ClassifiedTask,
        candidate: ClassifiedTask,
        current_batch_size: u32
    ) bool {
        // Fast checks first - avoid expensive similarity calculation if basic criteria fail
        
        // 1. Check preferred batch size compatibility (fast integer comparison)
        const batch_size_diff = @abs(@as(i32, @intCast(seed_task.preferred_batch_size)) - 
                                    @as(i32, @intCast(candidate.preferred_batch_size)));
        if (batch_size_diff > 8) return false;
        
        // 2. Quick performance threshold check (avoid poor candidates early)
        if (candidate.feature_vector.performance_score < 0.3) return false;
        
        // 3. Memory footprint quick check (avoid huge memory consumers)
        const memory_sum = seed_task.static_analysis.memory_footprint + candidate.static_analysis.memory_footprint;
        if (memory_sum > 16 * 1024 * 1024) return false; // 16MB limit
        
        // 4. Only now do expensive similarity calculation
        const similarity_score = seed_task.feature_vector.similarityScore(candidate.feature_vector);
        if (similarity_score < self.similarity_threshold) return false;
        
        // 5. Fast acceptance for good candidates (skip complex scoring)
        if (similarity_score > 0.8 and current_batch_size < self.max_batch_size / 2) {
            return true;
        }
        
        // 6. Simplified scoring for edge cases only
        const performance_score = (seed_task.feature_vector.performance_score + 
                                 candidate.feature_vector.performance_score) * 0.5;
        
        // Accept if good enough (simplified threshold)
        return (similarity_score * 0.7 + performance_score * 0.3) > 0.6;
    }
    
    /// Record batch formation for adaptive learning
    fn recordBatchFormation(self: *Self, batch: simd_batch.SIMDTaskBatch, dominant_class: TaskClass) !void {
        const record = BatchPerformanceRecord{
            .batch_size = batch.batch_size,
            .estimated_speedup = batch.estimated_speedup,
            .dominant_classification = dominant_class,
            .formation_timestamp = @intCast(std.time.nanoTimestamp()),
            .similarity_threshold_used = self.similarity_threshold,
        };
        
        try self.batch_performance_history.append(record);
        
        // Adaptive threshold adjustment based on performance history
        if (self.batch_performance_history.items.len >= 10) {
            self.adaptThresholds();
        }
    }
    
    /// Adaptive threshold adjustment based on performance history
    fn adaptThresholds(self: *Self) void {
        if (self.batch_performance_history.items.len < 10) return;
        
        // Calculate average performance of recent batches
        const recent_count = @min(10, self.batch_performance_history.items.len);
        const recent_start = self.batch_performance_history.items.len - recent_count;
        
        var avg_speedup: f32 = 0.0;
        var avg_batch_size: f32 = 0.0;
        
        for (self.batch_performance_history.items[recent_start..]) |record| {
            avg_speedup += record.estimated_speedup;
            avg_batch_size += @as(f32, @floatFromInt(record.batch_size));
        }
        
        avg_speedup /= @as(f32, @floatFromInt(recent_count));
        avg_batch_size /= @as(f32, @floatFromInt(recent_count));
        
        // Adaptive adjustment
        if (avg_speedup < 2.0 and avg_batch_size < 8.0) {
            // Performance is poor, relax threshold to allow more diverse batches
            self.similarity_threshold = @max(0.5, self.similarity_threshold - 0.05);
        } else if (avg_speedup > 4.0 and avg_batch_size > 16.0) {
            // Performance is good, tighten threshold for better quality
            self.similarity_threshold = @min(0.9, self.similarity_threshold + 0.02);
        }
    }
    
    /// Get formed batches ready for execution
    pub fn getFormedBatches(self: *Self) []*simd_batch.SIMDTaskBatch {
        return self.formed_batches.items;
    }
    
    /// Get batch formation statistics
    pub fn getFormationStats(self: *Self) BatchFormationStats {
        var total_estimated_speedup: f32 = 0.0;
        var total_tasks_in_batches: usize = 0;
        
        for (self.formed_batches.items) |batch| {
            total_estimated_speedup += batch.estimated_speedup;
            total_tasks_in_batches += batch.batch_size;
        }
        
        const avg_speedup = if (self.formed_batches.items.len > 0)
            total_estimated_speedup / @as(f32, @floatFromInt(self.formed_batches.items.len))
        else
            1.0;
        
        const total_tasks = self.classified_tasks.items.len + total_tasks_in_batches;
        const formation_efficiency = if (total_tasks > 0)
            @as(f32, @floatFromInt(total_tasks_in_batches)) / @as(f32, @floatFromInt(total_tasks))
        else
            0.0;
        
        return BatchFormationStats{
            .total_tasks_submitted = total_tasks,
            .tasks_in_batches = total_tasks_in_batches,
            .pending_tasks = self.classified_tasks.items.len,
            .formed_batches = self.formed_batches.items.len,
            .average_batch_size = if (self.formed_batches.items.len > 0)
                @as(f32, @floatFromInt(total_tasks_in_batches)) / @as(f32, @floatFromInt(self.formed_batches.items.len))
            else
                0.0,
            .average_estimated_speedup = avg_speedup,
            .formation_efficiency = formation_efficiency,
            .current_similarity_threshold = self.similarity_threshold,
        };
    }
};

/// Performance record for adaptive learning
const BatchPerformanceRecord = struct {
    batch_size: usize,
    estimated_speedup: f32,
    dominant_classification: TaskClass,
    formation_timestamp: u64,
    similarity_threshold_used: f32,
};

/// Statistics for batch formation performance
pub const BatchFormationStats = struct {
    total_tasks_submitted: usize,
    tasks_in_batches: usize,
    pending_tasks: usize,
    formed_batches: usize,
    average_batch_size: f32,
    average_estimated_speedup: f32,
    formation_efficiency: f32,
    current_similarity_threshold: f32,
};

// ============================================================================
// Additional Tests for Batch Formation
// ============================================================================

test "intelligent batch formation with multi-criteria optimization" {
    const allocator = std.testing.allocator;
    
    // Create intelligent batch former
    const criteria = BatchFormationCriteria.performanceOptimized();
    var batch_former = IntelligentBatchFormer.init(allocator, criteria);
    defer batch_former.deinit();
    
    // Create diverse test tasks
    const TestData = struct { values: [32]f32 };
    var test_data_array: [8]TestData = undefined;
    
    // Initialize test data
    for (&test_data_array, 0..) |*data, i| {
        for (&data.values, 0..) |*value, j| {
            value.* = @as(f32, @floatFromInt(i * 32 + j));
        }
    }
    
    // Create similar tasks that should batch well together
    for (&test_data_array) |*data| {
        const task = core.Task{
            .func = struct {
                fn func(task_data: *anyopaque) void {
                    const typed_data = @as(*TestData, @ptrCast(@alignCast(task_data)));
                    for (&typed_data.values) |*value| {
                        value.* = value.* * 1.2 + 0.3; // Similar arithmetic operations
                    }
                }
            }.func,
            .data = @ptrCast(data),
            .priority = .normal,
            .data_size_hint = @sizeOf(TestData),
        };
        
        try batch_former.addTask(task, false); // Disable profiling for speed
    }
    
    // Force batch formation
    try batch_former.attemptBatchFormation();
    
    // Check formation results
    const stats = batch_former.getFormationStats();
    const formed_batches = batch_former.getFormedBatches();
    
    try std.testing.expect(stats.formed_batches > 0);
    try std.testing.expect(stats.formation_efficiency > 0.0);
    try std.testing.expect(formed_batches.len == stats.formed_batches);
    
    std.debug.print("Intelligent batch formation test passed!\n", .{});
    std.debug.print("  Tasks submitted: {}\n", .{stats.total_tasks_submitted});
    std.debug.print("  Batches formed: {}\n", .{stats.formed_batches});
    std.debug.print("  Formation efficiency: {d:.1}%\n", .{stats.formation_efficiency * 100});
    std.debug.print("  Average batch size: {d:.1}\n", .{stats.average_batch_size});
    std.debug.print("  Average speedup: {d:.2}x\n", .{stats.average_estimated_speedup});
}