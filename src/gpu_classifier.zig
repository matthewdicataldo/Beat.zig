const std = @import("std");
const builtin = @import("builtin");
const core = @import("core.zig");
const fingerprint = @import("fingerprint.zig");
const simd_classifier = @import("simd_classifier.zig");
const gpu_integration = @import("gpu_integration.zig");
const topology = @import("topology.zig");

// Automatic GPU Suitability Detection for Beat.zig (Task 3.2.1)
//
// This module implements intelligent GPU vs CPU task classification using:
// - Data-parallel pattern recognition with semantic analysis
// - Computational intensity analysis using roofline model principles
// - Memory access pattern classification for coalescing optimization
// - Branch divergence detection and performance impact analysis
// - Adaptive learning with real-time performance feedback
//
// The system integrates with existing SIMD classification and extends it
// with GPU-specific analysis to make optimal heterogeneous computing decisions.

// ============================================================================
// Core GPU Classification Types and Constants
// ============================================================================

/// GPU suitability classification levels based on performance analysis
pub const GPUSuitability = enum(u8) {
    highly_unsuitable = 0,  // CPU strongly preferred (< 0.5x GPU performance)
    unsuitable = 1,         // CPU preferred (0.5-0.8x GPU performance)
    neutral = 2,            // Either CPU or GPU acceptable (0.8-1.2x)
    suitable = 3,           // GPU preferred (1.2-2.0x GPU performance)
    highly_suitable = 4,    // GPU strongly preferred (> 2.0x GPU performance)
    
    /// Get numeric suitability score (0.0-1.0)
    pub fn getScore(self: GPUSuitability) f32 {
        return switch (self) {
            .highly_unsuitable => 0.0,
            .unsuitable => 0.25,
            .neutral => 0.5,
            .suitable => 0.75,
            .highly_suitable => 1.0,
        };
    }
    
    /// Get expected performance multiplier vs CPU
    pub fn getPerformanceMultiplier(self: GPUSuitability) f32 {
        return switch (self) {
            .highly_unsuitable => 0.3,
            .unsuitable => 0.65,
            .neutral => 1.0,
            .suitable => 1.6,
            .highly_suitable => 3.0,
        };
    }
};

/// Detailed analysis of GPU performance factors
pub const GPUAnalysis = struct {
    // Core classification metrics
    arithmetic_intensity: f32,          // Operations per memory access
    parallelization_potential: f32,     // 0.0-1.0 parallel fraction
    memory_coalescing_score: f32,       // 0.0-1.0 memory access efficiency
    branch_uniformity: f32,             // 0.0-1.0 execution uniformity
    
    // Workload characteristics
    problem_size_score: f32,            // 0.0-1.0 based on data size
    compute_intensity: f32,             // Compute vs memory ratio
    data_locality_score: f32,           // Memory access locality
    dependency_freedom: f32,            // 0.0-1.0 independence level
    
    // Performance estimates
    gpu_efficiency_estimate: f32,       // Expected GPU utilization
    transfer_overhead_ratio: f32,       // Data transfer cost ratio
    overall_suitability: GPUSuitability,
    confidence_score: f32,              // 0.0-1.0 classification confidence
    
    // Detailed pattern analysis
    detected_patterns: std.EnumSet(DataParallelPattern),
    memory_access_patterns: std.EnumSet(MemoryAccessPattern),
    execution_patterns: std.EnumSet(ExecutionPattern),
    
    pub fn init() GPUAnalysis {
        return GPUAnalysis{
            .arithmetic_intensity = 0.0,
            .parallelization_potential = 0.0,
            .memory_coalescing_score = 0.0,
            .branch_uniformity = 1.0,
            .problem_size_score = 0.0,
            .compute_intensity = 0.0,
            .data_locality_score = 0.0,
            .dependency_freedom = 0.0,
            .gpu_efficiency_estimate = 0.0,
            .transfer_overhead_ratio = 1.0,
            .overall_suitability = .neutral,
            .confidence_score = 0.0,
            .detected_patterns = std.EnumSet(DataParallelPattern).init(.{}),
            .memory_access_patterns = std.EnumSet(MemoryAccessPattern).init(.{}),
            .execution_patterns = std.EnumSet(ExecutionPattern).init(.{}),
        };
    }
    
    /// Calculate overall GPU suitability based on all factors
    pub fn calculateSuitability(self: *GPUAnalysis) void {
        // Multi-factor weighted scoring
        const weights = struct {
            const arithmetic_intensity: f32 = 0.25;
            const parallelization: f32 = 0.20;
            const memory_coalescing: f32 = 0.15;
            const branch_uniformity: f32 = 0.15;
            const problem_size: f32 = 0.10;
            const compute_intensity: f32 = 0.10;
            const dependency_freedom: f32 = 0.05;
        };
        
        // Calculate weighted score
        var score: f32 = 0.0;
        score += self.arithmetic_intensity * weights.arithmetic_intensity;
        score += self.parallelization_potential * weights.parallelization;
        score += self.memory_coalescing_score * weights.memory_coalescing;
        score += self.branch_uniformity * weights.branch_uniformity;
        score += self.problem_size_score * weights.problem_size;
        score += self.compute_intensity * weights.compute_intensity;
        score += self.dependency_freedom * weights.dependency_freedom;
        
        // Apply pattern bonuses/penalties
        score = self.applyPatternAdjustments(score);
        
        // Apply transfer overhead penalty
        score *= (1.0 - self.transfer_overhead_ratio * 0.3);
        
        // Clamp and classify
        score = @max(0.0, @min(1.0, score));
        self.overall_suitability = self.scoreToSuitability(score);
        
        // Calculate confidence based on data quality
        self.confidence_score = self.calculateConfidence();
    }
    
    fn applyPatternAdjustments(self: *const GPUAnalysis, base_score: f32) f32 {
        var adjusted_score = base_score;
        
        // Data parallel pattern bonuses
        if (self.detected_patterns.contains(.embarrassingly_parallel)) {
            adjusted_score += 0.15;
        }
        if (self.detected_patterns.contains(.map_reduce)) {
            adjusted_score += 0.10;
        }
        if (self.detected_patterns.contains(.stencil_computation)) {
            adjusted_score += 0.08;
        }
        
        // Memory pattern adjustments
        if (self.memory_access_patterns.contains(.random_access)) {
            adjusted_score -= 0.12;
        }
        if (self.memory_access_patterns.contains(.pointer_chasing)) {
            adjusted_score -= 0.20;
        }
        if (self.memory_access_patterns.contains(.coalesced_access)) {
            adjusted_score += 0.10;
        }
        
        // Execution pattern adjustments
        if (self.execution_patterns.contains(.divergent_branching)) {
            adjusted_score -= 0.15;
        }
        if (self.execution_patterns.contains(.synchronization_heavy)) {
            adjusted_score -= 0.18;
        }
        if (self.execution_patterns.contains(.uniform_execution)) {
            adjusted_score += 0.08;
        }
        
        return adjusted_score;
    }
    
    fn scoreToSuitability(self: *const GPUAnalysis, score: f32) GPUSuitability {
        _ = self;
        if (score < 0.2) return .highly_unsuitable;
        if (score < 0.4) return .unsuitable;
        if (score < 0.6) return .neutral;
        if (score < 0.8) return .suitable;
        return .highly_suitable;
    }
    
    fn calculateConfidence(self: *const GPUAnalysis) f32 {
        var confidence: f32 = 0.5; // Base confidence
        
        // Increase confidence with clear patterns
        const pattern_count = self.detected_patterns.count() + 
                             self.memory_access_patterns.count() + 
                             self.execution_patterns.count();
        confidence += @as(f32, @floatFromInt(pattern_count)) * 0.05;
        
        // Increase confidence with extreme values
        if (self.arithmetic_intensity > 10.0 or self.arithmetic_intensity < 0.1) {
            confidence += 0.2;
        }
        if (self.parallelization_potential > 0.9 or self.parallelization_potential < 0.1) {
            confidence += 0.15;
        }
        
        return @max(0.0, @min(1.0, confidence));
    }
};

/// Data-parallel computation patterns that indicate GPU suitability
pub const DataParallelPattern = enum {
    embarrassingly_parallel,    // Independent parallel tasks
    map_reduce,                // Map-reduce style computations
    stencil_computation,       // Neighbor-based grid computations
    matrix_operations,         // Linear algebra operations
    image_processing,          // Pixel-wise image operations
    monte_carlo,              // Monte Carlo simulations
    sorting_algorithms,       // Parallel sorting operations
    graph_algorithms,         // Parallel graph processing
    signal_processing,        // DSP and filtering operations
    numerical_simulation,     // Scientific computing simulations
};

/// Memory access patterns that affect GPU performance
pub const MemoryAccessPattern = enum {
    coalesced_access,         // Sequential memory access (optimal for GPU)
    strided_access,           // Regular stride patterns (good for GPU)
    random_access,            // Random memory access (poor for GPU)
    pointer_chasing,          // Linked data structures (very poor for GPU)
    broadcast_access,         // Read-only shared data (good for GPU)
    scatter_gather,           // Indirect addressing (moderate for GPU)
    temporal_locality,        // High temporal reuse (good for caching)
    spatial_locality,         // High spatial reuse (good for coalescing)
};

/// Execution patterns that affect GPU efficiency
pub const ExecutionPattern = enum {
    uniform_execution,        // All threads execute same path (optimal)
    divergent_branching,      // Threads take different branches (poor)
    synchronization_heavy,    // Frequent synchronization (poor)
    compute_intensive,        // High arithmetic intensity (good)
    memory_intensive,         // Memory-bound operations (depends on pattern)
    control_flow_heavy,       // Complex control flow (poor)
    recursive_algorithms,     // Recursive execution (poor)
    streaming_computation,    // Streaming data processing (good)
};

// ============================================================================
// Arithmetic Intensity Analyzer
// ============================================================================

/// Arithmetic intensity analyzer using roofline model principles
pub const ArithmeticIntensityAnalyzer = struct {
    const Self = @This();
    
    // Operation counting
    floating_point_ops: u64,
    integer_ops: u64,
    memory_accesses: u64,
    cache_hits: u64,
    cache_misses: u64,
    
    // Memory characteristics
    memory_footprint: usize,
    working_set_size: usize,
    
    pub fn init() Self {
        return Self{
            .floating_point_ops = 0,
            .integer_ops = 0,
            .memory_accesses = 0,
            .cache_hits = 0,
            .cache_misses = 0,
            .memory_footprint = 0,
            .working_set_size = 0,
        };
    }
    
    /// Analyze task fingerprint for arithmetic intensity
    pub fn analyzeFingerprint(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) void {
        // Extract operation counts from fingerprint
        self.floating_point_ops = self.estimateFloatingPointOps(task_fingerprint);
        self.integer_ops = self.estimateIntegerOps(task_fingerprint);
        self.memory_accesses = self.estimateMemoryAccesses(task_fingerprint);
        
        // Estimate working set and memory footprint
        self.memory_footprint = task_fingerprint.data_size;
        self.working_set_size = self.estimateWorkingSet(task_fingerprint);
    }
    
    /// Calculate arithmetic intensity (operations per byte transferred)
    pub fn calculateArithmeticIntensity(self: *const Self) f32 {
        const total_ops = self.floating_point_ops + self.integer_ops;
        if (total_ops == 0) return 0.0;
        
        // Estimate actual memory traffic (accounting for cache effects)
        const cache_hit_ratio = if (self.memory_accesses > 0) 
            @as(f32, @floatFromInt(self.cache_hits)) / @as(f32, @floatFromInt(self.memory_accesses))
            else 0.0;
        
        const memory_traffic = @as(f32, @floatFromInt(self.memory_accesses)) * (1.0 - cache_hit_ratio);
        const bytes_transferred = memory_traffic * 8.0; // Assume 8 bytes per access average
        
        if (bytes_transferred == 0.0) return 100.0; // Compute-only task
        
        return @as(f32, @floatFromInt(total_ops)) / bytes_transferred;
    }
    
    /// Estimate GPU efficiency based on roofline model
    pub fn estimateGPUEfficiency(self: *const Self, gpu_device: *const gpu_integration.GPUDeviceInfo) f32 {
        const ai = self.calculateArithmeticIntensity();
        
        // Roofline model parameters (simplified)
        const peak_performance = @as(f32, @floatFromInt(gpu_device.compute_units)) * 1000.0; // GFLOPS estimate
        const peak_bandwidth = gpu_device.global_memory_gb * 100.0; // GB/s estimate
        const ridge_point = peak_performance / peak_bandwidth;
        
        if (ai < ridge_point) {
            // Memory-bound region
            return ai / ridge_point;
        } else {
            // Compute-bound region  
            return @min(1.0, ridge_point / ai);
        }
    }
    
    // Private helper methods for operation estimation
    
    fn estimateFloatingPointOps(self: *const Self, task_fingerprint: fingerprint.TaskFingerprint) u64 {
        _ = self;
        // Heuristic based on task characteristics
        var ops: u64 = 0;
        
        // Base estimate from data size and complexity
        const complexity_factor = switch (task_fingerprint.computational_complexity) {
            0...25 => 1,    // Simple operations
            26...50 => 4,   // Moderate complexity
            51...75 => 16,  // High complexity
            else => 64,     // Very high complexity
        };
        
        ops += (task_fingerprint.data_size / 8) * complexity_factor;
        
        // Additional ops based on optimization potential (likely floating point intensive)
        if (task_fingerprint.optimization_potential > 50) {
            ops *= 2;
        }
        
        return ops;
    }
    
    fn estimateIntegerOps(self: *const Self, task_fingerprint: fingerprint.TaskFingerprint) u64 {
        _ = self;
        // Base integer operations (array indexing, comparisons, etc.)
        var ops: u64 = 0;
        
        // Index operations
        ops += task_fingerprint.data_size / 8; // One index per 8 bytes
        
        // Control flow operations
        ops += (task_fingerprint.dependency_count + 1) * 10;
        
        // Branch operations based on predictability
        const branch_ops = (100 - task_fingerprint.branch_predictability) * 2;
        ops += branch_ops;
        
        return ops;
    }
    
    fn estimateMemoryAccesses(self: *const Self, task_fingerprint: fingerprint.TaskFingerprint) u64 {
        _ = self;
        // Estimate memory accesses based on data access patterns
        var accesses: u64 = 0;
        
        // Base accesses (at least one read per data element)
        accesses += task_fingerprint.data_size / 8;
        
        // Additional accesses based on memory intensity
        const memory_factor = switch (task_fingerprint.access_pattern) {
            0...25 => 1,    // Sequential access
            26...50 => 2,   // Some random access
            51...75 => 4,   // Heavy random access
            else => 8,      // Very scattered access
        };
        
        accesses *= memory_factor;
        
        // Write accesses
        accesses += task_fingerprint.data_size / 16; // Assume 50% writes
        
        return accesses;
    }
    
    fn estimateWorkingSet(self: *const Self, task_fingerprint: fingerprint.TaskFingerprint) usize {
        _ = self;
        // Working set is typically smaller than total data size
        const locality_factor: f32 = switch (task_fingerprint.access_pattern) {
            0...25 => 0.1,   // High locality
            26...50 => 0.3,  // Moderate locality
            51...75 => 0.6,  // Low locality
            else => 1.0,     // No locality
        };
        
        return @as(usize, @intFromFloat(@as(f32, @floatFromInt(task_fingerprint.data_size)) * locality_factor));
    }
};

// ============================================================================
// Data-Parallel Pattern Detector
// ============================================================================

/// Detector for data-parallel computation patterns
pub const DataParallelPatternDetector = struct {
    const Self = @This();
    
    detected_patterns: std.EnumSet(DataParallelPattern),
    pattern_confidence: std.EnumMap(DataParallelPattern, f32),
    
    pub fn init() Self {
        return Self{
            .detected_patterns = std.EnumSet(DataParallelPattern).init(.{}),
            .pattern_confidence = std.EnumMap(DataParallelPattern, f32).init(.{}),
        };
    }
    
    /// Analyze task characteristics for data-parallel patterns
    pub fn analyzeTaskPatterns(self: *Self, task_fingerprint: fingerprint.TaskFingerprint, simd_analysis: ?simd_classifier.StaticAnalysis) void {
        self.detectEmbarrassinglyParallel(task_fingerprint);
        self.detectMapReduce(task_fingerprint);
        self.detectMatrixOperations(task_fingerprint, simd_analysis);
        self.detectImageProcessing(task_fingerprint);
        self.detectNumericalSimulation(task_fingerprint);
        self.detectStencilComputation(task_fingerprint);
        self.detectSorting(task_fingerprint);
        self.detectMonteCarloSimulation(task_fingerprint);
    }
    
    /// Get overall parallelization potential (0.0-1.0)
    pub fn getParallelizationPotential(self: *const Self) f32 {
        var total_confidence: f32 = 0.0;
        var pattern_count: u32 = 0;
        
        var iter = self.pattern_confidence.iterator();
        while (iter.next()) |entry| {
            total_confidence += entry.value_ptr.*;
            pattern_count += 1;
        }
        
        if (pattern_count == 0) return 0.0;
        
        const avg_confidence = total_confidence / @as(f32, @floatFromInt(pattern_count));
        
        // Boost score if multiple strong patterns are detected
        const detected_count = self.detected_patterns.count();
        const pattern_bonus = @min(0.3, @as(f32, @floatFromInt(detected_count)) * 0.1);
        
        return @min(1.0, avg_confidence + pattern_bonus);
    }
    
    // Pattern detection methods
    
    fn detectEmbarrassinglyParallel(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) void {
        var confidence: f32 = 0.0;
        
        // Low dependency count indicates independent operations
        if (task_fingerprint.dependency_count < 5) {
            confidence += 0.4;
        }
        
        // High optimization potential suggests parallelizable operations
        if (task_fingerprint.optimization_potential > 70) {
            confidence += 0.3;
        }
        
        // Large data size with low dependencies is embarrassingly parallel
        if (task_fingerprint.data_size > 1024 * 1024 and task_fingerprint.dependency_count < 3) {
            confidence += 0.3;
        }
        
        if (confidence >= 0.5) {
            self.detected_patterns.setPresent(.embarrassingly_parallel, true);
            self.pattern_confidence.put(.embarrassingly_parallel, confidence);
        }
    }
    
    fn detectMapReduce(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) void {
        var confidence: f32 = 0.0;
        
        // Moderate dependencies suggest map-reduce structure
        if (task_fingerprint.dependency_count >= 3 and task_fingerprint.dependency_count <= 10) {
            confidence += 0.3;
        }
        
        // High computational complexity with structured dependencies
        if (task_fingerprint.computational_complexity > 50 and task_fingerprint.dependency_count > 0) {
            confidence += 0.3;
        }
        
        // Large data size suggests aggregation operations
        if (task_fingerprint.data_size > 512 * 1024) {
            confidence += 0.2;
        }
        
        if (confidence >= 0.4) {
            self.detected_patterns.setPresent(.map_reduce, true);
            self.pattern_confidence.put(.map_reduce, confidence);
        }
    }
    
    fn detectMatrixOperations(self: *Self, task_fingerprint: fingerprint.TaskFingerprint, simd_analysis: ?simd_classifier.StaticAnalysis) void {
        var confidence: f32 = 0.0;
        
        // SIMD analysis can indicate matrix-like operations
        if (simd_analysis) |analysis| {
            if (analysis.vectorization_score > 0.7) {
                confidence += 0.4;
            }
            if (analysis.data_dependencies.contains(.none)) {
                confidence += 0.2;
            }
        }
        
        // High optimization potential with medium complexity
        if (task_fingerprint.optimization_potential > 60 and 
           task_fingerprint.computational_complexity > 40) {
            confidence += 0.3;
        }
        
        // Regular access patterns suggest matrix operations
        if (task_fingerprint.access_pattern < 30) { // Regular access
            confidence += 0.2;
        }
        
        if (confidence >= 0.5) {
            self.detected_patterns.setPresent(.matrix_operations, true);
            self.pattern_confidence.put(.matrix_operations, confidence);
        }
    }
    
    fn detectImageProcessing(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) void {
        var confidence: f32 = 0.0;
        
        // Image processing typically has specific data size patterns
        const data_size = task_fingerprint.data_size;
        if (self.isPowerOfTwo(data_size) or self.isImageLikeSize(data_size)) {
            confidence += 0.3;
        }
        
        // Local access patterns typical in image processing
        if (task_fingerprint.access_pattern < 40) {
            confidence += 0.3;
        }
        
        // High optimization potential with moderate complexity
        if (task_fingerprint.optimization_potential > 70 and 
           task_fingerprint.computational_complexity > 30) {
            confidence += 0.3;
        }
        
        if (confidence >= 0.6) {
            self.detected_patterns.setPresent(.image_processing, true);
            self.pattern_confidence.put(.image_processing, confidence);
        }
    }
    
    fn detectNumericalSimulation(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) void {
        var confidence: f32 = 0.0;
        
        // High computational complexity suggests numerical work
        if (task_fingerprint.computational_complexity > 70) {
            confidence += 0.4;
        }
        
        // High optimization potential in numerical algorithms
        if (task_fingerprint.optimization_potential > 80) {
            confidence += 0.3;
        }
        
        // Large data with complex operations
        if (task_fingerprint.data_size > 1024 * 1024 and 
           task_fingerprint.computational_complexity > 60) {
            confidence += 0.3;
        }
        
        if (confidence >= 0.7) {
            self.detected_patterns.setPresent(.numerical_simulation, true);
            self.pattern_confidence.put(.numerical_simulation, confidence);
        }
    }
    
    fn detectStencilComputation(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) void {
        var confidence: f32 = 0.0;
        
        // Stencil computations have specific dependency patterns
        if (task_fingerprint.dependency_count >= 4 and task_fingerprint.dependency_count <= 8) {
            confidence += 0.4;
        }
        
        // Local access patterns in stencil operations
        if (task_fingerprint.access_pattern < 25) {
            confidence += 0.3;
        }
        
        // Moderate to high computational complexity
        if (task_fingerprint.computational_complexity > 40) {
            confidence += 0.2;
        }
        
        if (confidence >= 0.6) {
            self.detected_patterns.setPresent(.stencil_computation, true);
            self.pattern_confidence.put(.stencil_computation, confidence);
        }
    }
    
    fn detectSorting(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) void {
        var confidence: f32 = 0.0;
        
        // Sorting has unpredictable branches
        if (task_fingerprint.branch_predictability < 60) {
            confidence += 0.3;
        }
        
        // High access pattern variance in sorting
        if (task_fingerprint.access_pattern > 60) {
            confidence += 0.3;
        }
        
        // Moderate complexity with optimization potential
        if (task_fingerprint.computational_complexity > 30 and 
           task_fingerprint.optimization_potential > 50) {
            confidence += 0.2;
        }
        
        if (confidence >= 0.5) {
            self.detected_patterns.setPresent(.sorting_algorithms, true);
            self.pattern_confidence.put(.sorting_algorithms, confidence);
        }
    }
    
    fn detectMonteCarloSimulation(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) void {
        var confidence: f32 = 0.0;
        
        // Monte Carlo has unpredictable patterns
        if (task_fingerprint.branch_predictability < 50) {
            confidence += 0.2;
        }
        
        // Random access patterns
        if (task_fingerprint.access_pattern > 70) {
            confidence += 0.3;
        }
        
        // High optimization potential despite complexity
        if (task_fingerprint.optimization_potential > 70) {
            confidence += 0.4;
        }
        
        if (confidence >= 0.6) {
            self.detected_patterns.setPresent(.monte_carlo, true);
            self.pattern_confidence.put(.monte_carlo, confidence);
        }
    }
    
    // Helper methods
    
    fn isPowerOfTwo(self: *const Self, n: usize) bool {
        _ = self;
        return n > 0 and (n & (n - 1)) == 0;
    }
    
    fn isImageLikeSize(self: *const Self, size: usize) bool {
        _ = self;
        // Check for common image dimensions
        const common_sizes = [_]usize{
            640 * 480,    // VGA
            1024 * 768,   // XGA
            1920 * 1080,  // Full HD
            3840 * 2160,  // 4K
        };
        
        for (common_sizes) |img_size| {
            if (size >= img_size - 1024 and size <= img_size + 1024) {
                return true;
            }
        }
        
        return false;
    }
};

// ============================================================================
// Memory Access Pattern Analyzer
// ============================================================================

/// Analyzer for memory access patterns and GPU coalescing potential
pub const MemoryAccessAnalyzer = struct {
    const Self = @This();
    
    detected_patterns: std.EnumSet(MemoryAccessPattern),
    coalescing_score: f32,
    cache_efficiency_score: f32,
    
    pub fn init() Self {
        return Self{
            .detected_patterns = std.EnumSet(MemoryAccessPattern).init(.{}),
            .coalescing_score = 0.0,
            .cache_efficiency_score = 0.0,
        };
    }
    
    /// Analyze memory access patterns from task characteristics
    pub fn analyzeMemoryPatterns(self: *Self, task_fingerprint: fingerprint.TaskFingerprint, simd_analysis: ?simd_classifier.StaticAnalysis) void {
        self.detectCoalescedAccess(task_fingerprint, simd_analysis);
        self.detectStridedAccess(task_fingerprint);
        self.detectRandomAccess(task_fingerprint);
        self.detectPointerChasing(task_fingerprint);
        self.detectBroadcastAccess(task_fingerprint);
        self.detectScatterGather(task_fingerprint);
        self.calculateCoalescingScore();
        self.calculateCacheEfficiency(task_fingerprint);
    }
    
    /// Get overall memory coalescing score for GPU efficiency
    pub fn getCoalescingScore(self: *const Self) f32 {
        return self.coalescing_score;
    }
    
    /// Get cache efficiency score
    pub fn getCacheEfficiencyScore(self: *const Self) f32 {
        return self.cache_efficiency_score;
    }
    
    // Pattern detection methods
    
    fn detectCoalescedAccess(self: *Self, task_fingerprint: fingerprint.TaskFingerprint, simd_analysis: ?simd_classifier.StaticAnalysis) void {
        var is_coalesced = false;
        
        // Sequential access pattern indicates coalescing potential
        if (task_fingerprint.access_pattern < 30) {
            is_coalesced = true;
        }
        
        // SIMD analysis can confirm sequential patterns
        if (simd_analysis) |analysis| {
            if (analysis.access_patterns.contains(.sequential)) {
                is_coalesced = true;
            }
        }
        
        if (is_coalesced) {
            self.detected_patterns.setPresent(.coalesced_access, true);
        }
    }
    
    fn detectStridedAccess(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) void {
        // Moderate access pattern suggests strided access
        if (task_fingerprint.access_pattern >= 30 and task_fingerprint.access_pattern <= 60) {
            self.detected_patterns.setPresent(.strided_access, true);
        }
    }
    
    fn detectRandomAccess(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) void {
        // High access pattern variance indicates random access
        if (task_fingerprint.access_pattern > 70) {
            self.detected_patterns.setPresent(.random_access, true);
        }
    }
    
    fn detectPointerChasing(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) void {
        // High dependencies with random access suggests pointer chasing
        if (task_fingerprint.dependency_count > 8 and task_fingerprint.access_pattern > 80) {
            self.detected_patterns.setPresent(.pointer_chasing, true);
        }
    }
    
    fn detectBroadcastAccess(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) void {
        // Low dependencies with low access pattern suggests broadcast
        if (task_fingerprint.dependency_count < 3 and task_fingerprint.access_pattern < 20) {
            self.detected_patterns.setPresent(.broadcast_access, true);
        }
    }
    
    fn detectScatterGather(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) void {
        // Moderate dependencies with irregular access
        if (task_fingerprint.dependency_count >= 3 and 
           task_fingerprint.dependency_count <= 8 and 
           task_fingerprint.access_pattern > 50) {
            self.detected_patterns.setPresent(.scatter_gather, true);
        }
    }
    
    fn calculateCoalescingScore(self: *Self) void {
        var score: f32 = 0.5; // Base score
        
        // Positive patterns
        if (self.detected_patterns.contains(.coalesced_access)) {
            score += 0.4;
        }
        if (self.detected_patterns.contains(.strided_access)) {
            score += 0.2;
        }
        if (self.detected_patterns.contains(.broadcast_access)) {
            score += 0.3;
        }
        
        // Negative patterns
        if (self.detected_patterns.contains(.random_access)) {
            score -= 0.3;
        }
        if (self.detected_patterns.contains(.pointer_chasing)) {
            score -= 0.5;
        }
        if (self.detected_patterns.contains(.scatter_gather)) {
            score -= 0.2;
        }
        
        self.coalescing_score = @max(0.0, @min(1.0, score));
    }
    
    fn calculateCacheEfficiency(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) void {
        var efficiency: f32 = 0.5; // Base efficiency
        
        // Sequential access improves cache efficiency
        if (task_fingerprint.access_pattern < 30) {
            efficiency += 0.3;
        }
        
        // Low access pattern variance helps caching
        if (task_fingerprint.access_pattern < 50) {
            efficiency += 0.2;
        }
        
        // Random access hurts cache efficiency
        if (task_fingerprint.access_pattern > 70) {
            efficiency -= 0.4;
        }
        
        self.cache_efficiency_score = @max(0.0, @min(1.0, efficiency));
    }
};

// ============================================================================
// Branch Divergence Analyzer
// ============================================================================

/// Analyzer for branch divergence patterns that affect GPU performance
pub const BranchDivergenceAnalyzer = struct {
    const Self = @This();
    
    detected_patterns: std.EnumSet(ExecutionPattern),
    uniformity_score: f32,
    divergence_penalty: f32,
    
    pub fn init() Self {
        return Self{
            .detected_patterns = std.EnumSet(ExecutionPattern).init(.{}),
            .uniformity_score = 1.0,
            .divergence_penalty = 0.0,
        };
    }
    
    /// Analyze execution patterns for branch divergence
    pub fn analyzeExecutionPatterns(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) void {
        self.detectUniformExecution(task_fingerprint);
        self.detectDivergentBranching(task_fingerprint);
        self.detectSynchronizationHeavy(task_fingerprint);
        self.detectComputeIntensive(task_fingerprint);
        self.detectControlFlowHeavy(task_fingerprint);
        self.calculateUniformityScore(task_fingerprint);
    }
    
    /// Get branch uniformity score (higher is better for GPU)
    pub fn getUniformityScore(self: *const Self) f32 {
        return self.uniformity_score;
    }
    
    /// Get divergence penalty factor
    pub fn getDivergencePenalty(self: *const Self) f32 {
        return self.divergence_penalty;
    }
    
    // Pattern detection methods
    
    fn detectUniformExecution(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) void {
        // High branch predictability suggests uniform execution
        if (task_fingerprint.branch_predictability > 80 and task_fingerprint.dependency_count < 5) {
            self.detected_patterns.setPresent(.uniform_execution, true);
        }
    }
    
    fn detectDivergentBranching(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) void {
        // Low branch predictability indicates divergent branches
        if (task_fingerprint.branch_predictability < 60) {
            self.detected_patterns.setPresent(.divergent_branching, true);
        }
    }
    
    fn detectSynchronizationHeavy(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) void {
        // High dependency count suggests frequent synchronization
        if (task_fingerprint.dependency_count > 10) {
            self.detected_patterns.setPresent(.synchronization_heavy, true);
        }
    }
    
    fn detectComputeIntensive(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) void {
        // High computational complexity with high optimization potential
        if (task_fingerprint.computational_complexity > 70 and 
           task_fingerprint.optimization_potential > 70) {
            self.detected_patterns.setPresent(.compute_intensive, true);
        }
    }
    
    fn detectControlFlowHeavy(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) void {
        // High dependency count with low branch predictability
        if (task_fingerprint.dependency_count > 5 and 
           task_fingerprint.branch_predictability < 50) {
            self.detected_patterns.setPresent(.control_flow_heavy, true);
        }
    }
    
    fn calculateUniformityScore(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) void {
        var score: f32 = @as(f32, @floatFromInt(task_fingerprint.branch_predictability)) / 100.0;
        
        // Pattern adjustments
        if (self.detected_patterns.contains(.uniform_execution)) {
            score += 0.2;
        }
        if (self.detected_patterns.contains(.divergent_branching)) {
            score -= 0.3;
        }
        if (self.detected_patterns.contains(.synchronization_heavy)) {
            score -= 0.2;
        }
        if (self.detected_patterns.contains(.control_flow_heavy)) {
            score -= 0.25;
        }
        if (self.detected_patterns.contains(.compute_intensive)) {
            score += 0.1;
        }
        
        self.uniformity_score = @max(0.0, @min(1.0, score));
        self.divergence_penalty = 1.0 - self.uniformity_score;
    }
};

// ============================================================================
// Main GPU Classifier
// ============================================================================

/// Main GPU task classifier that orchestrates all analysis components
pub const GPUTaskClassifier = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    
    // Analysis components
    arithmetic_analyzer: ArithmeticIntensityAnalyzer,
    pattern_detector: DataParallelPatternDetector,
    memory_analyzer: MemoryAccessAnalyzer,
    branch_analyzer: BranchDivergenceAnalyzer,
    
    // Performance tracking
    classification_count: u64,
    accuracy_tracker: std.ArrayList(f32),
    
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .arithmetic_analyzer = ArithmeticIntensityAnalyzer.init(),
            .pattern_detector = DataParallelPatternDetector.init(),
            .memory_analyzer = MemoryAccessAnalyzer.init(),
            .branch_analyzer = BranchDivergenceAnalyzer.init(),
            .classification_count = 0,
            .accuracy_tracker = std.ArrayList(f32).init(allocator),
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.accuracy_tracker.deinit();
    }
    
    /// Perform comprehensive GPU suitability analysis
    pub fn classifyTask(self: *Self, task_fingerprint: fingerprint.TaskFingerprint, gpu_device: ?*const gpu_integration.GPUDeviceInfo) !GPUAnalysis {
        var analysis = GPUAnalysis.init();
        
        // Get SIMD analysis if available (reuse existing classifier)
        const simd_analysis = try self.getSIMDAnalysis(task_fingerprint);
        
        // Run all analysis components
        self.arithmetic_analyzer.analyzeFingerprint(task_fingerprint);
        self.pattern_detector.analyzeTaskPatterns(task_fingerprint, simd_analysis);
        self.memory_analyzer.analyzeMemoryPatterns(task_fingerprint, simd_analysis);
        self.branch_analyzer.analyzeExecutionPatterns(task_fingerprint);
        
        // Populate analysis results
        analysis.arithmetic_intensity = self.arithmetic_analyzer.calculateArithmeticIntensity();
        analysis.parallelization_potential = self.pattern_detector.getParallelizationPotential();
        analysis.memory_coalescing_score = self.memory_analyzer.getCoalescingScore();
        analysis.branch_uniformity = self.branch_analyzer.getUniformityScore();
        
        // Calculate derived metrics
        analysis.problem_size_score = self.calculateProblemSizeScore(task_fingerprint);
        analysis.compute_intensity = self.calculateComputeIntensity(task_fingerprint);
        analysis.data_locality_score = self.memory_analyzer.getCacheEfficiencyScore();
        analysis.dependency_freedom = self.calculateDependencyFreedom(task_fingerprint);
        
        // GPU-specific estimates
        if (gpu_device) |device| {
            analysis.gpu_efficiency_estimate = self.arithmetic_analyzer.estimateGPUEfficiency(device);
            analysis.transfer_overhead_ratio = self.calculateTransferOverhead(task_fingerprint, device);
        } else {
            analysis.gpu_efficiency_estimate = 0.5; // Default estimate
            analysis.transfer_overhead_ratio = 0.3; // Conservative estimate
        }
        
        // Copy detected patterns
        analysis.detected_patterns = self.pattern_detector.detected_patterns;
        analysis.memory_access_patterns = self.memory_analyzer.detected_patterns;
        analysis.execution_patterns = self.branch_analyzer.detected_patterns;
        
        // Calculate final suitability
        analysis.calculateSuitability();
        
        self.classification_count += 1;
        
        return analysis;
    }
    
    /// Record actual performance for adaptive learning
    pub fn recordPerformance(self: *Self, predicted_suitability: GPUSuitability, actual_gpu_speedup: f32) !void {
        const predicted_speedup = predicted_suitability.getPerformanceMultiplier();
        const accuracy = 1.0 - @abs(predicted_speedup - actual_gpu_speedup) / @max(predicted_speedup, actual_gpu_speedup);
        
        try self.accuracy_tracker.append(accuracy);
        
        // Keep only recent accuracy measurements
        if (self.accuracy_tracker.items.len > 1000) {
            _ = self.accuracy_tracker.orderedRemove(0);
        }
    }
    
    /// Get current classification accuracy
    pub fn getClassificationAccuracy(self: *const Self) f32 {
        if (self.accuracy_tracker.items.len == 0) return 0.0;
        
        var sum: f32 = 0.0;
        for (self.accuracy_tracker.items) |accuracy| {
            sum += accuracy;
        }
        
        return sum / @as(f32, @floatFromInt(self.accuracy_tracker.items.len));
    }
    
    // Private helper methods
    
    fn getSIMDAnalysis(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) !?simd_classifier.StaticAnalysis {
        _ = self;
        // Create a basic SIMD analysis for memory pattern detection
        // This reuses our existing SIMD infrastructure
        var static_analysis = simd_classifier.StaticAnalysis.init();
        
        // Convert task fingerprint to SIMD analysis
        if (task_fingerprint.access_pattern < 30) {
            static_analysis.access_patterns.setPresent(.sequential, true);
        }
        if (task_fingerprint.dependency_count < 3) {
            static_analysis.data_dependencies.setPresent(.none, true);
        }
        
        static_analysis.vectorization_score = @as(f32, @floatFromInt(task_fingerprint.optimization_potential)) / 100.0;
        
        return static_analysis;
    }
    
    fn calculateProblemSizeScore(self: *const Self, task_fingerprint: fingerprint.TaskFingerprint) f32 {
        _ = self;
        // Larger problems are better for GPU
        const size_mb = @as(f32, @floatFromInt(task_fingerprint.data_size)) / (1024.0 * 1024.0);
        
        if (size_mb < 1.0) return 0.1;
        if (size_mb < 10.0) return 0.3;
        if (size_mb < 100.0) return 0.6;
        if (size_mb < 1000.0) return 0.8;
        return 1.0;
    }
    
    fn calculateComputeIntensity(self: *const Self, task_fingerprint: fingerprint.TaskFingerprint) f32 {
        _ = self;
        // Ratio of computation to data movement
        const complexity_ratio = @as(f32, @floatFromInt(task_fingerprint.computational_complexity)) / 100.0;
        const access_efficiency = 1.0 - (@as(f32, @floatFromInt(task_fingerprint.access_pattern)) / 100.0);
        
        return (complexity_ratio + access_efficiency) / 2.0;
    }
    
    fn calculateDependencyFreedom(self: *const Self, task_fingerprint: fingerprint.TaskFingerprint) f32 {
        _ = self;
        // Lower dependency count means higher freedom
        const max_deps: f32 = 20.0; // Assumed maximum meaningful dependencies
        const freedom = 1.0 - (@as(f32, @floatFromInt(task_fingerprint.dependency_count)) / max_deps);
        return @max(0.0, freedom);
    }
    
    fn calculateTransferOverhead(self: *const Self, task_fingerprint: fingerprint.TaskFingerprint, gpu_device: *const gpu_integration.GPUDeviceInfo) f32 {
        _ = self;
        // Estimate data transfer overhead
        const data_size_gb = @as(f32, @floatFromInt(task_fingerprint.data_size)) / (1024.0 * 1024.0 * 1024.0);
        const transfer_bandwidth_gb_s: f32 = 10.0; // PCIe bandwidth estimate
        const compute_time_estimate = data_size_gb / (@as(f32, @floatFromInt(gpu_device.compute_units)) * 0.1);
        const transfer_time = data_size_gb / transfer_bandwidth_gb_s;
        
        return transfer_time / @max(compute_time_estimate, transfer_time);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "GPU arithmetic intensity calculation" {
    _ = std.testing.allocator;
    
    var analyzer = ArithmeticIntensityAnalyzer.init();
    
    // Create test fingerprint
    var test_fingerprint = fingerprint.TaskFingerprint.init();
    test_fingerprint.data_size = 1024 * 1024; // 1MB
    test_fingerprint.computational_complexity = 80;
    test_fingerprint.optimization_potential = 90;
    test_fingerprint.access_pattern = 20; // Sequential
    
    analyzer.analyzeFingerprint(test_fingerprint);
    const ai = analyzer.calculateArithmeticIntensity();
    
    try std.testing.expect(ai > 0.0);
    try std.testing.expect(ai < 1000.0); // Reasonable range
    
    std.debug.print("Arithmetic intensity: {d:.2}\n", .{ai});
}

test "GPU pattern detection" {
    _ = std.testing.allocator;
    
    var detector = DataParallelPatternDetector.init();
    
    // Test embarrassingly parallel pattern
    var parallel_fingerprint = fingerprint.TaskFingerprint.init();
    parallel_fingerprint.dependency_count = 1;
    parallel_fingerprint.optimization_potential = 90;
    parallel_fingerprint.data_size = 10 * 1024 * 1024;
    
    detector.analyzeTaskPatterns(parallel_fingerprint, null);
    
    try std.testing.expect(detector.detected_patterns.contains(.embarrassingly_parallel));
    
    const potential = detector.getParallelizationPotential();
    try std.testing.expect(potential > 0.5);
    
    std.debug.print("Parallelization potential: {d:.2}\n", .{potential});
}

test "GPU memory access analysis" {
    _ = std.testing.allocator;
    
    var analyzer = MemoryAccessAnalyzer.init();
    
    // Test coalesced access pattern
    var coalesced_fingerprint = fingerprint.TaskFingerprint.init();
    coalesced_fingerprint.access_pattern = 15; // Very sequential
    coalesced_fingerprint.dependency_count = 2;
    
    analyzer.analyzeMemoryPatterns(coalesced_fingerprint, null);
    
    try std.testing.expect(analyzer.detected_patterns.contains(.coalesced_access));
    
    const coalescing_score = analyzer.getCoalescingScore();
    try std.testing.expect(coalescing_score > 0.7);
    
    std.debug.print("Coalescing score: {d:.2}\n", .{coalescing_score});
}

test "GPU branch divergence analysis" {
    _ = std.testing.allocator;
    
    var analyzer = BranchDivergenceAnalyzer.init();
    
    // Test uniform execution pattern
    var uniform_fingerprint = fingerprint.TaskFingerprint.init();
    uniform_fingerprint.branch_predictability = 95;
    uniform_fingerprint.dependency_count = 2;
    
    analyzer.analyzeExecutionPatterns(uniform_fingerprint);
    
    try std.testing.expect(analyzer.detected_patterns.contains(.uniform_execution));
    
    const uniformity = analyzer.getUniformityScore();
    try std.testing.expect(uniformity > 0.8);
    
    std.debug.print("Branch uniformity: {d:.2}\n", .{uniformity});
}

test "GPU task classification integration" {
    const allocator = std.testing.allocator;
    
    var classifier = GPUTaskClassifier.init(allocator);
    defer classifier.deinit();
    
    // Test high-suitability task
    var gpu_friendly_fingerprint = fingerprint.TaskFingerprint.init();
    gpu_friendly_fingerprint.data_size = 50 * 1024 * 1024; // 50MB
    gpu_friendly_fingerprint.computational_complexity = 85;
    gpu_friendly_fingerprint.optimization_potential = 90;
    gpu_friendly_fingerprint.access_pattern = 10; // Very sequential
    gpu_friendly_fingerprint.branch_predictability = 90;
    gpu_friendly_fingerprint.dependency_count = 1;
    
    const analysis = try classifier.classifyTask(gpu_friendly_fingerprint, null);
    
    try std.testing.expect(analysis.overall_suitability == .suitable or analysis.overall_suitability == .highly_suitable);
    try std.testing.expect(analysis.confidence_score > 0.5);
    try std.testing.expect(analysis.arithmetic_intensity > 0.0);
    
    std.debug.print("GPU suitability: {s} (confidence: {d:.2})\n", .{
        @tagName(analysis.overall_suitability), analysis.confidence_score
    });
    std.debug.print("Arithmetic intensity: {d:.2}\n", .{analysis.arithmetic_intensity});
    std.debug.print("Parallelization potential: {d:.2}\n", .{analysis.parallelization_potential});
}