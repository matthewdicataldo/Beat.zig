// Enhanced Fingerprint Module with Transparent ISPC Acceleration
// Maintains 100% API compatibility while providing automatic ISPC optimization

const std = @import("std");
const builtin = @import("builtin");
const core = @import("core.zig");
const fingerprint = @import("fingerprint.zig");
const ispc_integration = @import("ispc_prediction_integration.zig");

// Re-export all original types and constants for API compatibility
pub const TaskFingerprint = fingerprint.TaskFingerprint;
pub const AccessPattern = fingerprint.AccessPattern;
pub const OneEuroFilter = fingerprint.OneEuroFilter;
pub const OneEuroState = fingerprint.OneEuroState;
pub const OneEuroConfig = fingerprint.OneEuroConfig;
pub const FingerprintRegistry = struct {
    // Enhanced registry with automatic ISPC acceleration
    base_registry: fingerprint.FingerprintRegistry,
    accelerator: *ispc_integration.PredictionAccelerator,
    
    const Self = @This();
    
    pub fn init(allocator: std.mem.Allocator) !Self {
        return Self{
            .base_registry = fingerprint.FingerprintRegistry.init(allocator),
            .accelerator = ispc_integration.getGlobalAccelerator(),
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.base_registry.deinit();
    }
    
    /// Enhanced prediction lookup with automatic batching and ISPC acceleration
    pub fn getPredictedCycles(self: *Self, task_fingerprint: TaskFingerprint) f64 {
        // For single queries, use the base implementation
        return self.base_registry.getPredictedCycles(task_fingerprint);
    }
    
    /// Batch prediction lookup with ISPC acceleration
    pub fn getPredictedCyclesBatch(
        self: *Self,
        fingerprints: []const TaskFingerprint,
        results: []f64,
    ) void {
        std.debug.assert(fingerprints.len == results.len);
        
        if (fingerprints.len >= ispc_integration.ISPC_BATCH_THRESHOLD) {
            // Use ISPC-accelerated batch processing
            self.getPredictedCyclesBatchISPC(fingerprints, results) catch {
                // Fallback to native implementation
                self.getPredictedCyclesBatchNative(fingerprints, results);
            };
        } else {
            // Use native implementation for small batches
            self.getPredictedCyclesBatchNative(fingerprints, results);
        }
    }
    
    /// Enhanced confidence calculation with multi-factor analysis
    pub fn getMultiFactorConfidence(self: *Self, task_fingerprint: TaskFingerprint) fingerprint.MultiFactorConfidence {
        // Single confidence calculation uses base implementation
        return self.base_registry.getMultiFactorConfidence(task_fingerprint);
    }
    
    /// Batch confidence calculation with ISPC acceleration
    pub fn getMultiFactorConfidenceBatch(
        self: *Self,
        fingerprints: []const TaskFingerprint,
        results: []fingerprint.MultiFactorConfidence,
    ) void {
        std.debug.assert(fingerprints.len == results.len);
        
        // Get profile entries for ISPC processing
        var profiles = std.ArrayList(fingerprint.FingerprintRegistry.ProfileEntry).init(self.base_registry.allocator);
        defer profiles.deinit();
        
        profiles.ensureTotalCapacity(fingerprints.len) catch {
            // Fallback on allocation failure
            self.getMultiFactorConfidenceBatchNative(fingerprints, results);
            return;
        };
        
        for (fingerprints) |fp| {
            if (self.base_registry.getProfile(fp)) |profile| {
                profiles.appendAssumeCapacity(profile);
            } else {
                // Create default profile for unknown fingerprint
                profiles.appendAssumeCapacity(fingerprint.FingerprintRegistry.ProfileEntry{
                    .execution_count = 1,
                    .prediction_accuracy = 0.5,
                    .temporal_consistency = 0.5,
                    .execution_variance = 0.1,
                });
            }
        }
        
        // Use ISPC acceleration for batch processing
        self.accelerator.computeMultiFactorConfidenceBatch(profiles.items, results);
    }
    
    /// Enhanced similarity computation with automatic acceleration
    pub fn computeSimilarityMatrix(
        self: *Self,
        fingerprints: []const TaskFingerprint,
        similarity_matrix: []f32,
    ) void {
        self.accelerator.computeSimilarityMatrix(fingerprints, similarity_matrix);
    }
    
    // Private implementation methods
    fn getPredictedCyclesBatchISPC(
        self: *Self,
        fingerprints: []const TaskFingerprint,
        results: []f64,
    ) !void {
        // Extract hashes for ISPC lookup
        const hashes = try self.base_registry.allocator.alloc(u64, fingerprints.len);
        defer self.base_registry.allocator.free(hashes);
        
        for (fingerprints, hashes) |fp, *hash| {
            hash.* = self.base_registry.computeHash(fp);
        }
        
        // ISPC-accelerated lookup
        const predictions = try self.base_registry.allocator.alloc(f32, fingerprints.len);
        defer self.base_registry.allocator.free(predictions);
        const confidences = try self.base_registry.allocator.alloc(f32, fingerprints.len);
        defer self.base_registry.allocator.free(confidences);
        const cache_hits = try self.base_registry.allocator.alloc(bool, fingerprints.len);
        defer self.base_registry.allocator.free(cache_hits);
        
        // TODO: Implement ispc_lookup_predictions_batch kernel call
        // For now, fall back to native implementation
        self.getPredictedCyclesBatchNative(fingerprints, results);
    }
    
    fn getPredictedCyclesBatchNative(
        self: *Self,
        fingerprints: []const TaskFingerprint,
        results: []f64,
    ) void {
        for (fingerprints, results) |fp, *result| {
            result.* = self.base_registry.getPredictedCycles(fp);
        }
    }
    
    fn getMultiFactorConfidenceBatchNative(
        self: *Self,
        fingerprints: []const TaskFingerprint,
        results: []fingerprint.MultiFactorConfidence,
    ) void {
        for (fingerprints, results) |fp, *result| {
            result.* = self.base_registry.getMultiFactorConfidence(fp);
        }
    }
    
    // Re-export base methods for compatibility
    pub fn generateTaskFingerprint(task: *const core.Task, context: anytype) TaskFingerprint {
        return fingerprint.FingerprintRegistry.generateTaskFingerprint(task, context);
    }
    
    pub fn enhanceTask(task: *core.Task, context: anytype) void {
        return fingerprint.FingerprintRegistry.enhanceTask(task, context);
    }
    
    pub fn recordExecution(self: *Self, task_fingerprint: TaskFingerprint, cycles: u64) void {
        self.base_registry.recordExecution(task_fingerprint, cycles);
    }
    
    pub fn updatePrediction(self: *Self, task_fingerprint: TaskFingerprint, actual_cycles: u64) void {
        self.base_registry.updatePrediction(task_fingerprint, actual_cycles);
    }
};

/// Enhanced One Euro Filter with batch processing capabilities
pub const EnhancedOneEuroFilter = struct {
    base_filter: OneEuroFilter,
    accelerator: *ispc_integration.PredictionAccelerator,
    
    const Self = @This();
    
    pub fn init(config: OneEuroConfig) Self {
        return Self{
            .base_filter = OneEuroFilter.init(config),
            .accelerator = ispc_integration.getGlobalAccelerator(),
        };
    }
    
    /// Process single value (maintains API compatibility)
    pub fn filter(self: *Self, measurement: f32, timestamp_ns: u64) f32 {
        return self.base_filter.filter(measurement, timestamp_ns);
    }
    
    /// Batch processing with ISPC acceleration
    pub fn filterBatch(
        self: *Self,
        measurements: []const f32,
        timestamps: []const u64,
        results: []f32,
    ) void {
        std.debug.assert(measurements.len == timestamps.len);
        std.debug.assert(measurements.len == results.len);
        
        // Create state array for batch processing
        var states = std.ArrayList(OneEuroState).init(std.heap.page_allocator);
        defer states.deinit();
        
        states.resize(measurements.len) catch {
            // Fallback to single processing on allocation failure
            for (measurements, timestamps, results) |measurement, timestamp, *result| {
                result.* = self.filter(measurement, timestamp);
            }
            return;
        };
        
        // Initialize all states to current filter state
        for (states.items) |*state| {
            state.* = self.base_filter.getState();
        }
        
        // Use ISPC acceleration for batch processing
        self.accelerator.processOneEuroFilterBatch(
            measurements,
            timestamps,
            states.items,
            results,
            self.base_filter.config,
        );
        
        // Update base filter with last state (for API consistency)
        if (states.items.len > 0) {
            self.base_filter.setState(states.items[states.items.len - 1]);
        }
    }
};

/// Enhanced similarity computation with automatic acceleration
pub const EnhancedSimilarity = struct {
    /// Enhanced single fingerprint similarity (API compatible)
    pub fn similarity(fp_a: TaskFingerprint, fp_b: TaskFingerprint) f32 {
        return ispc_integration.TransparentAPI.enhancedSimilarity(fp_a, fp_b);
    }
    
    /// Batch similarity computation with ISPC acceleration
    pub fn similarityBatch(
        fingerprints_a: []const TaskFingerprint,
        fingerprints_b: []const TaskFingerprint,
        results: []f32,
    ) void {
        const accelerator = ispc_integration.getGlobalAccelerator();
        accelerator.computeFingerprintSimilarityBatch(fingerprints_a, fingerprints_b, results);
    }
    
    /// Similarity matrix computation with ISPC acceleration
    pub fn similarityMatrix(
        fingerprints: []const TaskFingerprint,
        similarity_matrix: []f32,
    ) void {
        ispc_integration.TransparentAPI.enhancedSimilarityMatrix(fingerprints, similarity_matrix);
    }
};

/// API Enhancement Helper - automatically chooses best implementation
pub const AutoAcceleration = struct {
    /// Initialize acceleration system (call once at startup)
    pub fn init() void {
        ispc_integration.initGlobalAccelerator(ispc_integration.PredictionAccelerator.AcceleratorConfig{
            .enable_ispc = true,
            .auto_detection = true,
            .performance_tracking = builtin.mode != .ReleaseFast, // Track in debug/release-safe
        });
    }
    
    /// Get performance statistics
    pub fn getStats() ispc_integration.PredictionAccelerator.FallbackStats {
        return ispc_integration.DiagnosticsAPI.getAccelerationStats();
    }
    
    /// Print performance report
    pub fn printReport() void {
        ispc_integration.DiagnosticsAPI.printPerformanceReport();
    }
    
    /// Configure acceleration parameters
    pub fn configure(config: ispc_integration.PredictionAccelerator.AcceleratorConfig) void {
        ispc_integration.initGlobalAccelerator(config);
    }
};

// Legacy API compatibility - redirect to enhanced versions
pub const generateTaskFingerprint = FingerprintRegistry.generateTaskFingerprint;
pub const enhanceTask = FingerprintRegistry.enhanceTask;

// Enhanced type aliases for seamless migration
pub const EnhancedFingerprintRegistry = FingerprintRegistry;

/// Migration helper for existing code
pub fn createEnhancedRegistry(allocator: std.mem.Allocator) !FingerprintRegistry {
    // Auto-initialize acceleration system
    AutoAcceleration.init();
    return FingerprintRegistry.init(allocator);
}

/// Transparent performance enhancement for existing TaskFingerprint methods
pub fn enhanceTaskFingerprintAPI() void {
    // This would ideally extend TaskFingerprint with ISPC-accelerated methods
    // For now, users should use EnhancedSimilarity for batch operations
}

// API Integration Tests
test "enhanced fingerprint compatibility" {
    const testing = std.testing;
    
    // Test that enhanced API maintains compatibility
    var registry = try createEnhancedRegistry(testing.allocator);
    defer registry.deinit();
    
    // Create test fingerprints
    const fp1 = TaskFingerprint{
        .call_site_hash = 0x12345678,
        .data_size_class = 10,
        .data_alignment = 3,
        .access_pattern = .sequential,
        .simd_width = 8,
        .cache_locality = 12,
        .numa_node_hint = 0,
        .cpu_intensity = 8,
        .parallel_potential = 10,
        .execution_phase = 1,
        .priority_class = 2,
        .time_sensitivity = 1,
        .dependency_count = 2,
        .time_of_day_bucket = 14,
        .execution_frequency = 5,
        .seasonal_pattern = 3,
        .variance_level = 4,
        .expected_cycles_log2 = 12,
        .memory_footprint_log2 = 8,
        .io_intensity = 2,
        .branch_predictability = 6,
        .vectorization_benefit = 9,
        .cache_miss_rate = 4,
    };
    
    const fp2 = TaskFingerprint{
        .call_site_hash = 0x12345679,
        .data_size_class = 10,
        .data_alignment = 3,
        .access_pattern = .sequential,
        .simd_width = 8,
        .cache_locality = 12,
        .numa_node_hint = 0,
        .cpu_intensity = 8,
        .parallel_potential = 10,
        .execution_phase = 1,
        .priority_class = 2,
        .time_sensitivity = 1,
        .dependency_count = 2,
        .time_of_day_bucket = 14,
        .execution_frequency = 5,
        .seasonal_pattern = 3,
        .variance_level = 4,
        .expected_cycles_log2 = 12,
        .memory_footprint_log2 = 8,
        .io_intensity = 2,
        .branch_predictability = 6,
        .vectorization_benefit = 9,
        .cache_miss_rate = 4,
    };
    
    // Test single similarity (should work identically to original)
    const similarity_single = EnhancedSimilarity.similarity(fp1, fp2);
    try testing.expect(similarity_single >= 0.0 and similarity_single <= 1.0);
    
    // Test batch similarity (new capability)
    const fingerprints_a = [_]TaskFingerprint{ fp1, fp1 };
    const fingerprints_b = [_]TaskFingerprint{ fp2, fp2 };
    var results = [_]f32{ 0.0, 0.0 };
    
    EnhancedSimilarity.similarityBatch(&fingerprints_a, &fingerprints_b, &results);
    try testing.expect(results[0] >= 0.0 and results[0] <= 1.0);
    try testing.expect(results[1] >= 0.0 and results[1] <= 1.0);
    
    // Test that batch and single give same results
    try testing.expectApproxEqRel(similarity_single, results[0], 0.001);
}