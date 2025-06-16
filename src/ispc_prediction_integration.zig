// ISPC Prediction System Integration
// Transparent acceleration layer that maintains full API compatibility
// while providing maximum out-of-the-box performance through ISPC kernels

const std = @import("std");
const builtin = @import("builtin");
const core = @import("core.zig");
const fingerprint = @import("fingerprint.zig");
const predictive_accounting = @import("predictive_accounting.zig");
const intelligent_decision = @import("intelligent_decision.zig");
const simd_classifier = @import("simd_classifier.zig");

/// Compile-time ISPC availability detection
const ISPC_AVAILABLE = @hasDecl(@This(), "ispc_compute_fingerprint_similarity") and 
                       @hasDecl(@This(), "ispc_process_one_euro_filter_batch");

/// Runtime thresholds for ISPC acceleration
const ISPC_BATCH_THRESHOLD = 4; // Minimum batch size for ISPC acceleration
const ISPC_SIMILARITY_THRESHOLD = 8; // Minimum fingerprint count for similarity matrix
const ISPC_WORKER_THRESHOLD = 4; // Minimum worker count for vectorized selection

/// External ISPC kernel declarations (conditionally compiled)
extern fn ispc_compute_fingerprint_similarity(
    fingerprints_a: [*]const u64,
    fingerprints_b: [*]const u64,
    results: [*]f32,
    count: i32,
) void;

extern fn ispc_compute_similarity_matrix(
    fingerprints: [*]const u64,
    similarity_matrix: [*]f32,
    count: i32,
) void;

extern fn ispc_process_one_euro_filter_batch(
    measurements: [*]const f32,
    timestamps: [*]const u64,
    states: [*]fingerprint.OneEuroState,
    results: [*]f32,
    count: i32,
    dt_scale: f32,
    beta: f32,
    fc_min: f32,
) void;

extern fn ispc_compute_multi_factor_confidence_batch(
    execution_counts: [*]const u32,
    accuracy_scores: [*]const f32,
    temporal_scores: [*]const f32,
    variance_scores: [*]const f32,
    confidence_results: [*]intelligent_decision.MultiFactorConfidence,
    count: i32,
) void;

extern fn ispc_score_workers_batch(
    worker_loads: [*]const f32,
    numa_distances: [*]const f32,
    prediction_accuracies: [*]const f32,
    worker_scores: [*]f32,
    worker_count: i32,
    task_numa_preference: i32,
) void;

/// Intelligent ISPC Integration Layer
pub const PredictionAccelerator = struct {
    config: AcceleratorConfig,
    fallback_stats: FallbackStats,
    
    const AcceleratorConfig = struct {
        enable_ispc: bool = true,
        auto_detection: bool = true,
        prefer_accuracy: bool = false, // If true, always use verified native implementations
        batch_threshold: u32 = ISPC_BATCH_THRESHOLD,
        performance_tracking: bool = false,
    };
    
    const FallbackStats = struct {
        ispc_calls: u64 = 0,
        native_calls: u64 = 0,
        ispc_failures: u64 = 0,
        performance_ratio: f64 = 1.0, // ISPC speedup vs native
    };
    
    pub fn init(config: AcceleratorConfig) PredictionAccelerator {
        return PredictionAccelerator{
            .config = config,
            .fallback_stats = FallbackStats{},
        };
    }
    
    /// Transparent fingerprint similarity computation with automatic ISPC acceleration
    pub fn computeFingerprintSimilarity(
        self: *PredictionAccelerator,
        fingerprint_a: fingerprint.TaskFingerprint,
        fingerprint_b: fingerprint.TaskFingerprint,
    ) f32 {
        // For single-pair similarity, always use native (ISPC overhead not worth it)
        self.fallback_stats.native_calls += 1;
        return fingerprint_a.similarity(fingerprint_b);
    }
    
    /// Batch fingerprint similarity with intelligent ISPC acceleration
    pub fn computeFingerprintSimilarityBatch(
        self: *PredictionAccelerator,
        fingerprints_a: []const fingerprint.TaskFingerprint,
        fingerprints_b: []const fingerprint.TaskFingerprint,
        results: []f32,
    ) void {
        std.debug.assert(fingerprints_a.len == fingerprints_b.len);
        std.debug.assert(fingerprints_a.len == results.len);
        
        const count = fingerprints_a.len;
        
        // Intelligent acceleration decision
        if (shouldUseISPC(count, ISPC_BATCH_THRESHOLD)) {
            self.computeFingerprintSimilarityISPC(fingerprints_a, fingerprints_b, results) catch {
                // Graceful fallback on ISPC failure
                self.fallback_stats.ispc_failures += 1;
                self.computeFingerprintSimilarityNative(fingerprints_a, fingerprints_b, results);
                return;
            };
            self.fallback_stats.ispc_calls += 1;
        } else {
            self.computeFingerprintSimilarityNative(fingerprints_a, fingerprints_b, results);
            self.fallback_stats.native_calls += 1;
        }
    }
    
    /// Similarity matrix computation with automatic ISPC acceleration
    pub fn computeSimilarityMatrix(
        self: *PredictionAccelerator,
        fingerprints: []const fingerprint.TaskFingerprint,
        similarity_matrix: []f32,
    ) void {
        const count = fingerprints.len;
        std.debug.assert(similarity_matrix.len == count * count);
        
        if (shouldUseISPC(count, ISPC_SIMILARITY_THRESHOLD)) {
            self.computeSimilarityMatrixISPC(fingerprints, similarity_matrix) catch {
                self.fallback_stats.ispc_failures += 1;
                self.computeSimilarityMatrixNative(fingerprints, similarity_matrix);
                return;
            };
            self.fallback_stats.ispc_calls += 1;
        } else {
            self.computeSimilarityMatrixNative(fingerprints, similarity_matrix);
            self.fallback_stats.native_calls += 1;
        }
    }
    
    /// Batch One Euro Filter processing with ISPC acceleration
    pub fn processOneEuroFilterBatch(
        self: *PredictionAccelerator,
        measurements: []const f32,
        timestamps: []const u64,
        states: []fingerprint.OneEuroState,
        results: []f32,
        filter_config: fingerprint.OneEuroConfig,
    ) void {
        std.debug.assert(measurements.len == timestamps.len);
        std.debug.assert(measurements.len == states.len);
        std.debug.assert(measurements.len == results.len);
        
        const count = measurements.len;
        
        if (shouldUseISPC(count, ISPC_BATCH_THRESHOLD)) {
            self.processOneEuroFilterISPC(measurements, timestamps, states, results, filter_config) catch {
                self.fallback_stats.ispc_failures += 1;
                self.processOneEuroFilterNative(measurements, timestamps, states, results, filter_config);
                return;
            };
            self.fallback_stats.ispc_calls += 1;
        } else {
            self.processOneEuroFilterNative(measurements, timestamps, states, results, filter_config);
            self.fallback_stats.native_calls += 1;
        }
    }
    
    /// Multi-factor confidence batch computation
    pub fn computeMultiFactorConfidenceBatch(
        self: *PredictionAccelerator,
        profiles: []const fingerprint.FingerprintRegistry.ProfileEntry,
        confidence_results: []intelligent_decision.MultiFactorConfidence,
    ) void {
        std.debug.assert(profiles.len == confidence_results.len);
        
        const count = profiles.len;
        
        if (shouldUseISPC(count, ISPC_BATCH_THRESHOLD)) {
            self.computeMultiFactorConfidenceISPC(profiles, confidence_results) catch {
                self.fallback_stats.ispc_failures += 1;
                self.computeMultiFactorConfidenceNative(profiles, confidence_results);
                return;
            };
            self.fallback_stats.ispc_calls += 1;
        } else {
            self.computeMultiFactorConfidenceNative(profiles, confidence_results);
            self.fallback_stats.native_calls += 1;
        }
    }
    
    /// Worker selection scoring with ISPC acceleration
    pub fn scoreWorkersBatch(
        self: *PredictionAccelerator,
        worker_loads: []const f32,
        numa_distances: []const f32,
        prediction_accuracies: []const f32,
        worker_scores: []f32,
        task_numa_preference: ?u32,
    ) void {
        std.debug.assert(worker_loads.len == numa_distances.len);
        std.debug.assert(worker_loads.len == prediction_accuracies.len);
        std.debug.assert(worker_loads.len == worker_scores.len);
        
        const count = worker_loads.len;
        
        if (shouldUseISPC(count, ISPC_WORKER_THRESHOLD)) {
            const numa_pref = if (task_numa_preference) |pref| @intCast(pref) else -1;
            self.scoreWorkersISPC(worker_loads, numa_distances, prediction_accuracies, worker_scores, numa_pref) catch {
                self.fallback_stats.ispc_failures += 1;
                self.scoreWorkersNative(worker_loads, numa_distances, prediction_accuracies, worker_scores, task_numa_preference);
                return;
            };
            self.fallback_stats.ispc_calls += 1;
        } else {
            self.scoreWorkersNative(worker_loads, numa_distances, prediction_accuracies, worker_scores, task_numa_preference);
            self.fallback_stats.native_calls += 1;
        }
    }
    
    /// Get acceleration statistics for performance monitoring
    pub fn getStats(self: *const PredictionAccelerator) FallbackStats {
        return self.fallback_stats;
    }
    
    /// Reset performance statistics
    pub fn resetStats(self: *PredictionAccelerator) void {
        self.fallback_stats = FallbackStats{};
    }
    
    // Private ISPC implementation functions
    fn computeFingerprintSimilarityISPC(
        self: *PredictionAccelerator,
        fingerprints_a: []const fingerprint.TaskFingerprint,
        fingerprints_b: []const fingerprint.TaskFingerprint,
        results: []f32,
    ) !void {
        _ = self;
        if (!ISPC_AVAILABLE) return error.ISPCNotAvailable;
        
        // Convert fingerprints to u64 arrays for ISPC
        const count = @intCast(i32, fingerprints_a.len);
        
        // TODO: Consider memory pooling for frequent allocations
        var a_buffer = try std.heap.page_allocator.alloc(u64, fingerprints_a.len * 2);
        defer std.heap.page_allocator.free(a_buffer);
        var b_buffer = try std.heap.page_allocator.alloc(u64, fingerprints_b.len * 2);
        defer std.heap.page_allocator.free(b_buffer);
        
        // Pack fingerprints into u64 arrays
        for (fingerprints_a, 0..) |fp, i| {
            const bits = @as(u128, @bitCast(fp));
            a_buffer[i * 2] = @truncate(u64, bits);
            a_buffer[i * 2 + 1] = @truncate(u64, bits >> 64);
        }
        
        for (fingerprints_b, 0..) |fp, i| {
            const bits = @as(u128, @bitCast(fp));
            b_buffer[i * 2] = @truncate(u64, bits);
            b_buffer[i * 2 + 1] = @truncate(u64, bits >> 64);
        }
        
        // Call ISPC kernel
        ispc_compute_fingerprint_similarity(
            a_buffer.ptr,
            b_buffer.ptr,
            results.ptr,
            count,
        );
    }
    
    fn computeSimilarityMatrixISPC(
        self: *PredictionAccelerator,
        fingerprints: []const fingerprint.TaskFingerprint,
        similarity_matrix: []f32,
    ) !void {
        _ = self;
        if (!ISPC_AVAILABLE) return error.ISPCNotAvailable;
        
        const count = @intCast(i32, fingerprints.len);
        
        var fingerprint_buffer = try std.heap.page_allocator.alloc(u64, fingerprints.len * 2);
        defer std.heap.page_allocator.free(fingerprint_buffer);
        
        // Pack fingerprints
        for (fingerprints, 0..) |fp, i| {
            const bits = @as(u128, @bitCast(fp));
            fingerprint_buffer[i * 2] = @truncate(u64, bits);
            fingerprint_buffer[i * 2 + 1] = @truncate(u64, bits >> 64);
        }
        
        ispc_compute_similarity_matrix(
            fingerprint_buffer.ptr,
            similarity_matrix.ptr,
            count,
        );
    }
    
    fn processOneEuroFilterISPC(
        self: *PredictionAccelerator,
        measurements: []const f32,
        timestamps: []const u64,
        states: []fingerprint.OneEuroState,
        results: []f32,
        filter_config: fingerprint.OneEuroConfig,
    ) !void {
        _ = self;
        if (!ISPC_AVAILABLE) return error.ISPCNotAvailable;
        
        const count = @intCast(i32, measurements.len);
        const dt_scale = 1.0 / 1_000_000_000.0; // ns to seconds
        
        ispc_process_one_euro_filter_batch(
            measurements.ptr,
            timestamps.ptr,
            @ptrCast([*]fingerprint.OneEuroState, states.ptr),
            results.ptr,
            count,
            dt_scale,
            filter_config.beta,
            filter_config.fc_min,
        );
    }
    
    fn computeMultiFactorConfidenceISPC(
        self: *PredictionAccelerator,
        profiles: []const fingerprint.FingerprintRegistry.ProfileEntry,
        confidence_results: []intelligent_decision.MultiFactorConfidence,
    ) !void {
        _ = self;
        if (!ISPC_AVAILABLE) return error.ISPCNotAvailable;
        
        const count = @intCast(i32, profiles.len);
        
        // Extract data for ISPC processing
        var execution_counts = try std.heap.page_allocator.alloc(u32, profiles.len);
        defer std.heap.page_allocator.free(execution_counts);
        var accuracy_scores = try std.heap.page_allocator.alloc(f32, profiles.len);
        defer std.heap.page_allocator.free(accuracy_scores);
        var temporal_scores = try std.heap.page_allocator.alloc(f32, profiles.len);
        defer std.heap.page_allocator.free(temporal_scores);
        var variance_scores = try std.heap.page_allocator.alloc(f32, profiles.len);
        defer std.heap.page_allocator.free(variance_scores);
        
        for (profiles, 0..) |profile, i| {
            execution_counts[i] = profile.execution_count;
            accuracy_scores[i] = profile.prediction_accuracy;
            temporal_scores[i] = profile.temporal_consistency;
            variance_scores[i] = profile.execution_variance;
        }
        
        ispc_compute_multi_factor_confidence_batch(
            execution_counts.ptr,
            accuracy_scores.ptr,
            temporal_scores.ptr,
            variance_scores.ptr,
            @ptrCast([*]intelligent_decision.MultiFactorConfidence, confidence_results.ptr),
            count,
        );
    }
    
    fn scoreWorkersISPC(
        self: *PredictionAccelerator,
        worker_loads: []const f32,
        numa_distances: []const f32,
        prediction_accuracies: []const f32,
        worker_scores: []f32,
        task_numa_preference: i32,
    ) !void {
        _ = self;
        if (!ISPC_AVAILABLE) return error.ISPCNotAvailable;
        
        const count = @intCast(i32, worker_loads.len);
        
        ispc_score_workers_batch(
            worker_loads.ptr,
            numa_distances.ptr,
            prediction_accuracies.ptr,
            worker_scores.ptr,
            count,
            task_numa_preference,
        );
    }
    
    // Native fallback implementations
    fn computeFingerprintSimilarityNative(
        self: *PredictionAccelerator,
        fingerprints_a: []const fingerprint.TaskFingerprint,
        fingerprints_b: []const fingerprint.TaskFingerprint,
        results: []f32,
    ) void {
        _ = self;
        for (fingerprints_a, fingerprints_b, results) |fp_a, fp_b, *result| {
            result.* = fp_a.similarity(fp_b);
        }
    }
    
    fn computeSimilarityMatrixNative(
        self: *PredictionAccelerator,
        fingerprints: []const fingerprint.TaskFingerprint,
        similarity_matrix: []f32,
    ) void {
        _ = self;
        const count = fingerprints.len;
        
        for (fingerprints, 0..) |fp_i, i| {
            for (fingerprints, 0..) |fp_j, j| {
                if (i == j) {
                    similarity_matrix[i * count + j] = 1.0;
                } else {
                    similarity_matrix[i * count + j] = fp_i.similarity(fp_j);
                }
            }
        }
    }
    
    fn processOneEuroFilterNative(
        self: *PredictionAccelerator,
        measurements: []const f32,
        timestamps: []const u64,
        states: []fingerprint.OneEuroState,
        results: []f32,
        filter_config: fingerprint.OneEuroConfig,
    ) void {
        _ = self;
        _ = filter_config;
        
        for (measurements, timestamps, states, results) |measurement, timestamp, *state, *result| {
            // Native One Euro Filter implementation
            result.* = fingerprint.OneEuroFilter.processValue(state, measurement, timestamp);
        }
    }
    
    fn computeMultiFactorConfidenceNative(
        self: *PredictionAccelerator,
        profiles: []const fingerprint.FingerprintRegistry.ProfileEntry,
        confidence_results: []intelligent_decision.MultiFactorConfidence,
    ) void {
        _ = self;
        
        for (profiles, confidence_results) |profile, *result| {
            result.* = intelligent_decision.MultiFactorConfidence.calculate(&profile);
        }
    }
    
    fn scoreWorkersNative(
        self: *PredictionAccelerator,
        worker_loads: []const f32,
        numa_distances: []const f32,
        prediction_accuracies: []const f32,
        worker_scores: []f32,
        task_numa_preference: ?u32,
    ) void {
        _ = self;
        
        for (worker_loads, numa_distances, prediction_accuracies, worker_scores) |load, numa_dist, accuracy, *score| {
            // Native worker scoring algorithm
            const load_score = 1.0 - load; // Lower load = higher score
            const numa_score = if (task_numa_preference != null) 1.0 - numa_dist else 1.0;
            const accuracy_score = accuracy;
            
            // Weighted combination
            score.* = (load_score * 0.4) + (numa_score * 0.3) + (accuracy_score * 0.3);
        }
    }
    
    /// Intelligent decision function for ISPC usage
    fn shouldUseISPC(count: usize, threshold: u32) bool {
        if (!ISPC_AVAILABLE) return false;
        
        // Runtime ISPC availability check
        if (builtin.mode == .Debug) {
            // In debug mode, prefer native for better debugging
            return count >= threshold * 2;
        }
        
        return count >= threshold;
    }
};

/// Global accelerator instance for transparent API integration
var global_accelerator: ?PredictionAccelerator = null;

/// Initialize global prediction accelerator (called automatically by Beat.zig)
pub fn initGlobalAccelerator(config: PredictionAccelerator.AcceleratorConfig) void {
    global_accelerator = PredictionAccelerator.init(config);
}

/// Get global accelerator instance
pub fn getGlobalAccelerator() *PredictionAccelerator {
    if (global_accelerator == null) {
        // Auto-initialize with default config
        initGlobalAccelerator(PredictionAccelerator.AcceleratorConfig{});
    }
    return &global_accelerator.?;
}

/// Transparent API extensions for existing modules
pub const TransparentAPI = struct {
    /// Enhanced fingerprint similarity with automatic ISPC acceleration
    pub fn enhancedSimilarity(fp_a: fingerprint.TaskFingerprint, fp_b: fingerprint.TaskFingerprint) f32 {
        return getGlobalAccelerator().computeFingerprintSimilarity(fp_a, fp_b);
    }
    
    /// Enhanced batch processing for simd_classifier
    pub fn enhancedSimilarityMatrix(
        fingerprints: []const fingerprint.TaskFingerprint,
        similarity_matrix: []f32,
    ) void {
        getGlobalAccelerator().computeSimilarityMatrix(fingerprints, similarity_matrix);
    }
    
    /// Enhanced One Euro Filter processing
    pub fn enhancedOneEuroFilterBatch(
        measurements: []const f32,
        timestamps: []const u64,
        states: []fingerprint.OneEuroState,
        results: []f32,
        config: fingerprint.OneEuroConfig,
    ) void {
        getGlobalAccelerator().processOneEuroFilterBatch(measurements, timestamps, states, results, config);
    }
};

// Performance monitoring and diagnostics
pub const DiagnosticsAPI = struct {
    pub fn getAccelerationStats() PredictionAccelerator.FallbackStats {
        return getGlobalAccelerator().getStats();
    }
    
    pub fn printPerformanceReport() void {
        const stats = getAccelerationStats();
        const total_calls = stats.ispc_calls + stats.native_calls;
        
        if (total_calls == 0) {
            std.debug.print("No prediction acceleration calls made yet.\n");
            return;
        }
        
        const ispc_percentage = @as(f64, @floatFromInt(stats.ispc_calls)) / @as(f64, @floatFromInt(total_calls)) * 100.0;
        
        std.debug.print("\n=== Beat.zig Prediction Acceleration Report ===\n");
        std.debug.print("Total prediction calls: {d}\n", .{total_calls});
        std.debug.print("ISPC accelerated: {d} ({d:.1}%)\n", .{ stats.ispc_calls, ispc_percentage });
        std.debug.print("Native fallback: {d} ({d:.1}%)\n", .{ stats.native_calls, 100.0 - ispc_percentage });
        std.debug.print("ISPC failures: {d}\n", .{stats.ispc_failures});
        
        if (stats.performance_ratio > 1.0) {
            std.debug.print("Average ISPC speedup: {d:.2}x\n", .{stats.performance_ratio});
        }
        
        std.debug.print("=== End Report ===\n\n");
    }
};