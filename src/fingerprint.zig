const std = @import("std");
const builtin = @import("builtin");
const core = @import("core.zig");
const scheduler = @import("scheduler.zig");

// Task Fingerprinting System for Beat.zig
//
// This module implements the multi-dimensional task fingerprinting system
// for advanced predictive scheduling (Phase 2.1.1 and 2.1.2).
//
// Key Features:
// - Compact 128-bit fingerprint representation
// - Multi-dimensional task characteristic capture
// - Fast hashing and similarity calculation
// - Integration with existing Beat.zig architecture

// ============================================================================
// Core Fingerprint Structure
// ============================================================================

/// Compact 128-bit task fingerprint using bitfield optimization
/// Total: 128 bits (16 bytes) - cache-line friendly, SIMD-ready
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
    branch_predictability: u4,    // Branch prediction friendliness (0-15)
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
    
    /// Fast hash for HashMap usage - optimized for cache performance
    pub fn hash(self: TaskFingerprint) u64 {
        // Convert to array of bytes for manipulation
        const bytes = @as([@sizeOf(TaskFingerprint)]u8, @bitCast(self));
        
        // Use the first 64 bits directly, XOR with second 64 bits for better distribution
        const first_half = std.mem.readInt(u64, bytes[0..8], .little);
        const second_half = std.mem.readInt(u64, bytes[8..16], .little);
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

// ============================================================================
// Execution Context
// ============================================================================

/// Execution context for fingerprint generation
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
    
    pub fn init() ExecutionContext {
        return ExecutionContext{
            .current_numa_node = 0,
            .application_phase = 0,
            .current_hour = @intCast(@rem(@divTrunc(std.time.timestamp(), 3600), 24)),
            .estimated_cycles = 1000,
            .system_load = 0.5,
            .available_cores = @intCast(std.Thread.getCpuCount() catch 4),
            .recent_task_types = [_]u32{0} ** 8,
            .execution_history_count = 0,
        };
    }
    
    pub fn update(self: *ExecutionContext, numa_node: u32, system_load: f32) void {
        self.current_numa_node = numa_node;
        self.system_load = system_load;
        self.current_hour = @intCast(@rem(@divTrunc(std.time.timestamp(), 3600), 24));
        self.execution_history_count += 1;
    }
};

// ============================================================================
// Task Analysis Functions
// ============================================================================

/// Analyze task to extract characteristics for fingerprinting
pub const TaskAnalyzer = struct {
    pub fn analyzeTask(task: *const core.Task, context: *const ExecutionContext) TaskFingerprint {
        return TaskFingerprint{
            // Call site fingerprinting - combines function identity
            .call_site_hash = hashCallSite(task),
            
            // Data characteristics analysis
            .data_size_class = classifyDataSize(getDataSize(task)),
            .data_alignment = @intCast(@min(15, @ctz(@intFromPtr(task.data)))),
            .access_pattern = analyzeAccessPattern(task),
            .simd_width = detectOptimalSimdWidth(task),
            .cache_locality = estimateCacheLocality(task),
            
            // Execution context
            .numa_node_hint = @intCast(context.current_numa_node % 16),
            .cpu_intensity = estimateCpuIntensity(task),
            .parallel_potential = analyzeParallelPotential(task),
            .execution_phase = @intCast(context.application_phase % 16),
            .priority_class = @intCast(@intFromEnum(task.priority)),
            .time_sensitivity = classifyTimeSensitivity(task),
            .dependency_count = 0, // TODO: Add dependency tracking
            
            // Temporal characteristics
            .time_of_day_bucket = @intCast(context.current_hour % 24),
            .execution_frequency = classifyExecutionFrequency(context),
            .seasonal_pattern = detectSeasonalPattern(context),
            .variance_level = estimateVarianceLevel(task),
            
            // Performance hints
            .expected_cycles_log2 = @intCast(@min(255, std.math.log2_int_ceil(u64, context.estimated_cycles))),
            .memory_footprint_log2 = @intCast(@min(255, std.math.log2_int_ceil(usize, getMemoryFootprint(task)))),
            .io_intensity = estimateIoIntensity(task),
            .cache_miss_rate = predictCacheMissRate(task),
            .branch_predictability = analyzeBranchPredictability(task),
            .vectorization_benefit = assessVectorizationBenefit(task),
        };
    }
    
    fn hashCallSite(task: *const core.Task) u32 {
        // Use function pointer as primary identifier
        const func_ptr = @intFromPtr(task.func);
        
        // Simple hash - combine function pointer with some entropy
        return @truncate(func_ptr ^ (func_ptr >> 32));
    }
    
    fn getDataSize(task: *const core.Task) usize {
        // Use hint if available
        if (task.data_size_hint) |size| {
            return size;
        }
        
        // Heuristic: use pointer alignment to estimate size
        const ptr_value = @intFromPtr(task.data);
        if (ptr_value == 0) return 0;
        
        const alignment = @ctz(ptr_value);
        return @as(usize, 1) << @min(alignment, 16); // Cap at 64KB
    }
    
    fn classifyDataSize(size: usize) u8 {
        if (size == 0) return 0;
        return @intCast(@min(255, std.math.log2_int_ceil(usize, size)));
    }
    
    fn analyzeAccessPattern(task: *const core.Task) TaskFingerprint.AccessPattern {
        const data_size = getDataSize(task);
        const ptr_value = @intFromPtr(task.data);
        
        // Simple heuristics
        if (data_size < 64) return .sequential;
        if (ptr_value % 64 == 0) return .sequential; // Cache-aligned
        if (data_size > 1024 * 1024) return .strided; // Large data
        
        return .mixed;
    }
    
    fn detectOptimalSimdWidth(task: *const core.Task) u4 {
        const data_size = getDataSize(task);
        if (data_size < 16) return 1;
        if (data_size >= 64) return 8;
        return 4;
    }
    
    fn estimateCacheLocality(task: *const core.Task) u4 {
        const data_size = getDataSize(task);
        if (data_size <= 32 * 1024) return 15; // L1 cache
        if (data_size <= 256 * 1024) return 12; // L2 cache
        if (data_size <= 8 * 1024 * 1024) return 8; // L3 cache
        return 4; // Main memory
    }
    
    fn estimateCpuIntensity(task: *const core.Task) u4 {
        // Heuristic based on function pointer
        const func_ptr = @intFromPtr(task.func);
        return @intCast((func_ptr >> 8) % 16);
    }
    
    fn analyzeParallelPotential(task: *const core.Task) u4 {
        const data_size = getDataSize(task);
        
        // Larger data typically has better parallel potential
        if (data_size >= 1024 * 1024) return 15;
        if (data_size >= 64 * 1024) return 12;
        if (data_size >= 4 * 1024) return 8;
        if (data_size >= 256) return 4;
        
        return 1;
    }
    
    fn classifyTimeSensitivity(task: *const core.Task) u2 {
        // Use priority as proxy for time sensitivity
        return switch (task.priority) {
            .low => 0,
            .normal => 1,
            .high => 2,
        };
    }
    
    fn classifyExecutionFrequency(context: *const ExecutionContext) u3 {
        // Based on historical execution count
        const freq = context.execution_history_count % 1000;
        return @intCast(@min(7, freq / 125));
    }
    
    fn detectSeasonalPattern(context: *const ExecutionContext) u4 {
        // Simple time-based pattern
        return @intCast(context.current_hour % 8);
    }
    
    fn estimateVarianceLevel(task: *const core.Task) u4 {
        // Heuristic based on data characteristics
        const data_size = getDataSize(task);
        if (data_size > 1024 * 1024) return 12; // Large data = high variance
        return 6; // Default medium variance
    }
    
    fn getMemoryFootprint(task: *const core.Task) usize {
        return getDataSize(task) * 2; // Estimate including working memory
    }
    
    fn estimateIoIntensity(task: *const core.Task) u4 {
        // Default to low I/O for compute tasks
        _ = task;
        return 2;
    }
    
    fn predictCacheMissRate(task: *const core.Task) u4 {
        const data_size = getDataSize(task);
        if (data_size <= 32 * 1024) return 2; // Low misses for L1
        if (data_size <= 8 * 1024 * 1024) return 6; // Medium for L3
        return 12; // High misses for large data
    }
    
    fn analyzeBranchPredictability(task: *const core.Task) u4 {
        // Default to reasonable predictability
        _ = task;
        return 8;
    }
    
    fn assessVectorizationBenefit(task: *const core.Task) u4 {
        const data_size = getDataSize(task);
        if (data_size >= 256) return 8; // Good benefit for larger data
        return 4; // Medium benefit
    }
};

// ============================================================================
// Fingerprint Registry for Performance Tracking
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
        
        // Enhanced One Euro Filter for adaptive prediction
        execution_filter: scheduler.OneEuroFilter,
        
        // Advanced performance tracking
        variance_sum: f64 = 0.0,            // Welford's variance calculation
        prediction_error_sum: f64 = 0.0,    // Prediction accuracy tracking
        accuracy_count: u64 = 0,            // Number of predictions made
        last_prediction: ?f32 = null,       // Last prediction for accuracy calculation
        
        // Phase change detection
        phase_change_detector: PhaseChangeDetector,
        
        // Outlier detection and resilience
        outlier_detector: OutlierDetector,
        
        // Phase change adaptation state
        suppress_outlier_detection_until: u64 = 0, // Timestamp until which outlier detection is suppressed
        
        /// Phase change detection for adaptive response
        const PhaseChangeDetector = struct {
            window_size: u32 = 10,
            measurements: [10]f32 = [_]f32{0.0} ** 10,
            write_index: u32 = 0,
            filled: bool = false,
            last_mean: f32 = 0.0,
            
            pub fn addMeasurement(self: *PhaseChangeDetector, value: f32) bool {
                // Add to circular buffer
                self.measurements[self.write_index] = value;
                self.write_index = (self.write_index + 1) % self.window_size;
                if (self.write_index == 0) self.filled = true;
                
                // Calculate current window mean
                const count = if (self.filled) self.window_size else self.write_index;
                if (count < 3) return false; // Need at least 3 samples
                
                var sum: f32 = 0.0;
                for (self.measurements[0..count]) |measurement| {
                    sum += measurement;
                }
                const current_mean = sum / @as(f32, @floatFromInt(count));
                
                // Detect significant phase change (>15% change in mean)
                const phase_change = if (self.last_mean > 0.0) 
                    @abs(current_mean - self.last_mean) / self.last_mean > 0.15
                else 
                    false;
                
                self.last_mean = current_mean;
                return phase_change;
            }
            
            pub fn reset(self: *PhaseChangeDetector) void {
                self.write_index = 0;
                self.filled = false;
                self.last_mean = 0.0;
                self.measurements = [_]f32{0.0} ** 10;
            }
        };
        
        /// Outlier detection using modified Z-score
        const OutlierDetector = struct {
            median_buffer: [21]f32 = [_]f32{0.0} ** 21,
            buffer_index: u32 = 0,
            filled: bool = false,
            
            pub fn isOutlier(self: *OutlierDetector, value: f32) bool {
                // Add to buffer
                self.median_buffer[self.buffer_index] = value;
                self.buffer_index = (self.buffer_index + 1) % 21;
                if (self.buffer_index == 0) self.filled = true;
                
                const count = if (self.filled) 21 else self.buffer_index;
                if (count < 7) return false; // Need sufficient samples
                
                // Calculate median absolute deviation (MAD)
                var sorted_buffer: [21]f32 = undefined;
                @memcpy(sorted_buffer[0..count], self.median_buffer[0..count]);
                std.mem.sort(f32, sorted_buffer[0..count], {}, std.sort.asc(f32));
                
                const median = sorted_buffer[count / 2];
                
                // Calculate MAD
                var deviations: [21]f32 = undefined;
                for (sorted_buffer[0..count], 0..) |sorted_val, i| {
                    deviations[i] = @abs(sorted_val - median);
                }
                std.mem.sort(f32, deviations[0..count], {}, std.sort.asc(f32));
                const mad = deviations[count / 2];
                
                // Modified Z-score threshold of 3.5 (commonly used)
                if (mad > 0.0) {
                    const modified_z_score = 0.6745 * @abs(value - median) / mad;
                    return modified_z_score > 3.5;
                }
                
                return false;
            }
            
            pub fn reset(self: *OutlierDetector) void {
                self.buffer_index = 0;
                self.filled = false;
                self.median_buffer = [_]f32{0.0} ** 21;
            }
        };
        
        pub fn init(fingerprint: TaskFingerprint) ProfileEntry {
            return ProfileEntry{
                .fingerprint = fingerprint,
                .execution_count = 0,
                .total_cycles = 0,
                .min_cycles = 0,
                .max_cycles = 0,
                .last_execution = 0,
                .execution_filter = scheduler.OneEuroFilter.initDefault(),
                .phase_change_detector = PhaseChangeDetector{},
                .outlier_detector = OutlierDetector{},
            };
        }
        
        pub fn initWithConfig(fingerprint: TaskFingerprint, min_cutoff: f32, beta: f32, d_cutoff: f32) ProfileEntry {
            return ProfileEntry{
                .fingerprint = fingerprint,
                .execution_count = 0,
                .total_cycles = 0,
                .min_cycles = 0,
                .max_cycles = 0,
                .last_execution = 0,
                .execution_filter = scheduler.OneEuroFilter.init(min_cutoff, beta, d_cutoff),
                .phase_change_detector = PhaseChangeDetector{},
                .outlier_detector = OutlierDetector{},
            };
        }
        
        /// Get adaptive prediction using One Euro Filter (replaces simple averaging)
        pub fn getAdaptivePrediction(self: *const ProfileEntry) f64 {
            if (self.execution_count == 0) return 1000.0;
            
            // Use One Euro Filter estimate if available
            if (self.execution_filter.getCurrentEstimate()) |estimate| {
                return estimate;
            }
            
            // Fallback to simple average for first measurement
            return @as(f64, @floatFromInt(self.total_cycles)) / @as(f64, @floatFromInt(self.execution_count));
        }
        
        /// Legacy method name for backward compatibility
        pub fn getAverageExecution(self: *const ProfileEntry) f64 {
            return self.getAdaptivePrediction();
        }
        
        /// Enhanced execution update with adaptive filtering and outlier detection
        pub fn updateExecution(self: *ProfileEntry, cycles: u64) void {
            const cycles_f32 = @as(f32, @floatFromInt(cycles));
            const timestamp_ns = @as(u64, @intCast(std.time.nanoTimestamp()));
            
            // Phase change detection (need mutable access)
            var phase_detector_mut = &@constCast(self).phase_change_detector;
            const phase_change = phase_detector_mut.addMeasurement(cycles_f32);
            
            // Adapt filter parameters based on phase change
            var filter_mut = &@constCast(self).execution_filter;
            var self_mut = @constCast(self);
            if (phase_change) {
                // Increase responsiveness during phase changes
                filter_mut.reset();
                // Suppress outlier detection for 5 seconds to allow adaptation
                self_mut.suppress_outlier_detection_until = timestamp_ns + 5_000_000_000;
                // Debug: Phase change detected (can be enabled for debugging)
            }
            
            // Outlier detection and resilience (with phase change suppression)
            var outlier_detector_mut = &self_mut.outlier_detector;
            const outlier_suppressed = timestamp_ns < self.suppress_outlier_detection_until;
            const is_outlier = if (outlier_suppressed) false else outlier_detector_mut.isOutlier(cycles_f32);
            
            // Apply One Euro Filter with adaptive outlier handling
            var filtered_value: f32 = undefined;
            if (is_outlier and self.execution_count > 5) {
                // For outliers, use current filter estimate instead of raw measurement
                // This provides outlier resilience
                filtered_value = filter_mut.getCurrentEstimate() orelse cycles_f32;
                // Debug: Outlier rejected (can be enabled for debugging)
            } else {
                // Normal case: apply filter to measurement
                filtered_value = filter_mut.filter(cycles_f32, timestamp_ns);
                // Debug: Normal filtering (can be enabled for debugging)
            }
            
            // Track prediction accuracy
            if (self.last_prediction) |prediction| {
                const prediction_error = @abs(prediction - cycles_f32);
                self.prediction_error_sum += prediction_error;
                self.accuracy_count += 1;
            }
            self.last_prediction = filtered_value;
            
            // Update basic statistics
            if (self.execution_count == 0) {
                self.min_cycles = cycles;
                self.max_cycles = cycles;
            } else {
                self.min_cycles = @min(self.min_cycles, cycles);
                self.max_cycles = @max(self.max_cycles, cycles);
            }
            
            self.total_cycles += cycles;
            self.execution_count += 1;
            self.last_execution = timestamp_ns;
            
            // Update variance using Welford's algorithm
            if (self.execution_count > 1) {
                const mean = @as(f64, @floatFromInt(self.total_cycles)) / @as(f64, @floatFromInt(self.execution_count));
                const delta = @as(f64, @floatFromInt(cycles)) - mean;
                self.variance_sum += delta * delta;
            }
        }
        
        /// Get prediction confidence based on multiple factors
        pub fn getConfidence(self: *const ProfileEntry) f32 {
            if (self.execution_count == 0) return 0.0;
            
            // Sample size confidence (asymptotic to 1.0)
            const sample_confidence = @min(1.0, @as(f32, @floatFromInt(self.execution_count)) / 50.0);
            
            // Accuracy confidence (lower error = higher confidence)
            var accuracy_confidence: f32 = 1.0;
            if (self.accuracy_count > 0) {
                const avg_error = self.prediction_error_sum / @as(f64, @floatFromInt(self.accuracy_count));
                const avg_cycles = @as(f64, @floatFromInt(self.total_cycles)) / @as(f64, @floatFromInt(self.execution_count));
                if (avg_cycles > 0.0) {
                    const relative_error = avg_error / avg_cycles;
                    accuracy_confidence = @max(0.0, 1.0 - @as(f32, @floatCast(relative_error)));
                }
            }
            
            // Temporal relevance (recent samples weighted more heavily)
            const current_time = @as(u64, @intCast(std.time.nanoTimestamp()));
            const time_diff_ns = current_time - self.last_execution;
            const time_diff_s = @as(f64, @floatFromInt(time_diff_ns)) / 1_000_000_000.0;
            const temporal_confidence = @exp(-time_diff_s / 300.0); // 5-minute decay
            
            // Variance stability (lower variance = higher confidence)
            var variance_confidence: f32 = 1.0;
            if (self.execution_count > 2) {
                const variance = self.variance_sum / @as(f64, @floatFromInt(self.execution_count - 1));
                const avg_cycles = @as(f64, @floatFromInt(self.total_cycles)) / @as(f64, @floatFromInt(self.execution_count));
                if (avg_cycles > 0.0) {
                    const coefficient_of_variation = @sqrt(variance) / avg_cycles;
                    variance_confidence = @max(0.0, 1.0 - @as(f32, @floatCast(coefficient_of_variation)));
                }
            }
            
            // Combined confidence
            return sample_confidence * accuracy_confidence * @as(f32, @floatCast(temporal_confidence)) * variance_confidence;
        }
        
        /// Get execution time variance
        pub fn getVariance(self: *const ProfileEntry) f64 {
            if (self.execution_count < 2) return 0.0;
            return self.variance_sum / @as(f64, @floatFromInt(self.execution_count - 1));
        }
    };
    
    profiles: std.AutoHashMap(u64, ProfileEntry),
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
            var new_entry = ProfileEntry.init(fingerprint);
            new_entry.updateExecution(cycles);
            try self.profiles.put(hash, new_entry);
        }
    }
    
    pub fn getProfile(self: *Self, fingerprint: TaskFingerprint) ?ProfileEntry {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        return self.profiles.get(fingerprint.hash());
    }
    
    /// Get adaptive prediction using One Euro Filter (enhanced replacement for simple averaging)
    pub fn getPredictedCycles(self: *Self, fingerprint: TaskFingerprint) f64 {
        if (self.getProfile(fingerprint)) |profile| {
            return profile.getAdaptivePrediction();
        }
        return 1000.0; // Default estimate
    }
    
    /// Get prediction with confidence score
    pub fn getPredictionWithConfidence(self: *Self, fingerprint: TaskFingerprint) PredictionResult {
        if (self.getProfile(fingerprint)) |profile| {
            return PredictionResult{
                .predicted_cycles = profile.getAdaptivePrediction(),
                .confidence = profile.getConfidence(),
                .variance = profile.getVariance(),
                .execution_count = profile.execution_count,
            };
        }
        return PredictionResult{
            .predicted_cycles = 1000.0,
            .confidence = 0.0,
            .variance = 0.0,
            .execution_count = 0,
        };
    }
    
    /// Add customized registry with configurable One Euro Filter parameters
    pub fn initWithConfig(allocator: std.mem.Allocator, min_cutoff: f32, beta: f32, d_cutoff: f32) EnhancedFingerprintRegistry {
        return EnhancedFingerprintRegistry{
            .base_registry = Self{
                .profiles = std.AutoHashMap(u64, ProfileEntry).init(allocator),
                .allocator = allocator,
            },
            .min_cutoff = min_cutoff,
            .beta = beta,
            .d_cutoff = d_cutoff,
        };
    }
    
    /// Enhanced prediction result
    pub const PredictionResult = struct {
        predicted_cycles: f64,
        confidence: f32,
        variance: f64,
        execution_count: u64,
    };
    
    pub fn getRegistryStats(self: *Self) RegistryStats {
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

/// Enhanced FingerprintRegistry with configurable One Euro Filter parameters
pub const EnhancedFingerprintRegistry = struct {
    base_registry: FingerprintRegistry,
    min_cutoff: f32,
    beta: f32,
    d_cutoff: f32,
    
    const Self = @This();
    
    pub fn deinit(self: *Self) void {
        self.base_registry.deinit();
    }
    
    /// Record execution with custom One Euro Filter parameters
    pub fn recordExecution(self: *Self, fingerprint: TaskFingerprint, cycles: u64) !void {
        self.base_registry.mutex.lock();
        defer self.base_registry.mutex.unlock();
        
        const hash = fingerprint.hash();
        
        if (self.base_registry.profiles.getPtr(hash)) |entry| {
            entry.updateExecution(cycles);
        } else {
            var new_entry = FingerprintRegistry.ProfileEntry.initWithConfig(fingerprint, self.min_cutoff, self.beta, self.d_cutoff);
            new_entry.updateExecution(cycles);
            try self.base_registry.profiles.put(hash, new_entry);
        }
    }
    
    /// Get prediction with enhanced One Euro Filter
    pub fn getPredictionWithConfidence(self: *Self, fingerprint: TaskFingerprint) FingerprintRegistry.PredictionResult {
        return self.base_registry.getPredictionWithConfidence(fingerprint);
    }
    
    /// Get simple prediction cycles
    pub fn getPredictedCycles(self: *Self, fingerprint: TaskFingerprint) f64 {
        return self.base_registry.getPredictedCycles(fingerprint);
    }
    
    /// Get registry statistics
    pub fn getRegistryStats(self: *Self) FingerprintRegistry.RegistryStats {
        return self.base_registry.getRegistryStats();
    }
};

// ============================================================================
// Public API Functions
// ============================================================================

/// Generate fingerprint for a task with current execution context
pub fn generateTaskFingerprint(task: *const core.Task, context: *const ExecutionContext) TaskFingerprint {
    return TaskAnalyzer.analyzeTask(task, context);
}

/// Enhance a task with fingerprinting information
pub fn enhanceTask(task: *core.Task, context: *const ExecutionContext) void {
    const fingerprint = generateTaskFingerprint(task, context);
    task.fingerprint_hash = fingerprint.hash();
    task.creation_timestamp = @intCast(std.time.nanoTimestamp());
}

/// Get similarity between two tasks based on their fingerprint hashes
pub fn getTaskSimilarity(task1: *const core.Task, task2: *const core.Task, registry: *FingerprintRegistry) f32 {
    _ = registry; // TODO: Use registry for more sophisticated similarity
    
    if (task1.fingerprint_hash == null or task2.fingerprint_hash == null) {
        return 0.0;
    }
    
    // Simple similarity based on hash equality (can be enhanced)
    if (task1.fingerprint_hash.? == task2.fingerprint_hash.?) {
        return 1.0;
    }
    
    return 0.0; // TODO: Implement fuzzy similarity
}

// ============================================================================
// Testing
// ============================================================================

const testing = std.testing;

test "TaskFingerprint structure validation" {
    // Verify size and alignment
    try testing.expectEqual(16, @sizeOf(TaskFingerprint));
    try testing.expectEqual(128, @bitSizeOf(TaskFingerprint));
    
    // Test fingerprint creation
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
    
    // Test hash consistency
    const hash1 = test_fp.hash();
    const hash2 = test_fp.hash();
    try testing.expectEqual(hash1, hash2);
    
    // Test similarity
    try testing.expectEqual(@as(f32, 1.0), test_fp.similarity(test_fp));
    
    // Test characteristics
    const chars = test_fp.getCharacteristics();
    try testing.expect(!chars.is_cpu_intensive); // cpu_intensity = 8 < 12
    try testing.expect(chars.is_parallel_friendly); // parallel_potential = 10 >= 8
}

test "Task analysis integration" {
    const TestData = struct { value: i32 };
    var test_data = TestData{ .value = 42 };
    
    const test_func = struct {
        fn func(data: *anyopaque) void {
            const typed_data = @as(*TestData, @ptrCast(@alignCast(data)));
            typed_data.value *= 2;
        }
    }.func;
    
    var task = core.Task{
        .func = test_func,
        .data = @ptrCast(&test_data),
        .priority = .normal,
        .data_size_hint = @sizeOf(TestData),
    };
    
    var context = ExecutionContext.init();
    
    const fingerprint = generateTaskFingerprint(&task, &context);
    
    // Verify fingerprint generation
    try testing.expect(fingerprint.call_site_hash != 0);
    try testing.expect(fingerprint.data_size_class > 0);
    try testing.expectEqual(core.Priority.normal, @as(core.Priority, @enumFromInt(fingerprint.priority_class)));
}

test "Fingerprint registry functionality" {
    var registry = FingerprintRegistry.init(testing.allocator);
    defer registry.deinit();
    
    // Create test fingerprint
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
    
    // Record executions
    try registry.recordExecution(test_fp, 1000);
    try registry.recordExecution(test_fp, 1200);
    try registry.recordExecution(test_fp, 800);
    
    // Verify tracking
    const profile = registry.getProfile(test_fp);
    try testing.expect(profile != null);
    
    if (profile) |p| {
        try testing.expectEqual(@as(u64, 3), p.execution_count);
        // One Euro Filter may give slightly different results than simple averaging
        try testing.expectApproxEqAbs(@as(f64, 1000.0), p.getAverageExecution(), 30.0);
    }
    
    // Test prediction
    const predicted = registry.getPredictedCycles(test_fp);
    try testing.expectApproxEqAbs(@as(f64, 1000.0), predicted, 30.0);
    
    // Test stats
    const stats = registry.getRegistryStats();
    try testing.expect(stats.total_profiles > 0);
}