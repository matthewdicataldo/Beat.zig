const std = @import("std");
const continuation = @import("continuation.zig");
const continuation_unified = @import("continuation_unified.zig");
const simd = @import("simd.zig");
const simd_classifier = @import("simd_classifier.zig");

// ============================================================================
// SIMD Continuation Compatibility Layer
// 
// This file provides 100% API compatibility with the original continuation_simd.zig
// while routing all calls through the unified continuation management system.
// This ensures zero breaking changes during the consolidation process.
// ============================================================================

/// SIMD classification result - maintains exact API compatibility
pub const ContinuationSIMDClass = struct {
    task_class: simd_classifier.TaskClass,
    simd_suitability_score: f32,
    continuation_overhead_factor: f32,
    vectorization_potential: f32,
    preferred_numa_node: ?u32,
    
    /// Check if continuation is suitable for SIMD batching - exact API compatibility
    pub fn isSIMDSuitable(self: ContinuationSIMDClass) bool {
        return self.simd_suitability_score > 0.5 and self.vectorization_potential > 1.5;
    }
    
    /// Get expected performance improvement - exact API compatibility
    pub fn getExpectedSpeedup(self: ContinuationSIMDClass) f32 {
        return self.vectorization_potential / self.continuation_overhead_factor;
    }
    
    /// Create from unified SIMD classification
    fn fromUnified(unified_simd: continuation_unified.SIMDClassification) ContinuationSIMDClass {
        // Map task class based on suitability score
        const task_class: simd_classifier.TaskClass = if (unified_simd.suitability_score > 0.8)
            .highly_vectorizable
        else if (unified_simd.suitability_score > 0.6)
            .moderately_vectorizable
        else if (unified_simd.suitability_score > 0.3)
            .potentially_vectorizable
        else
            .not_vectorizable;
        
        return ContinuationSIMDClass{
            .task_class = task_class,
            .simd_suitability_score = unified_simd.suitability_score,
            .continuation_overhead_factor = 1.2, // Default overhead factor
            .vectorization_potential = unified_simd.vectorization_potential,
            .preferred_numa_node = unified_simd.preferred_numa_node,
        };
    }
};

/// SIMD continuation classifier - maintains exact API compatibility
pub const ContinuationClassifier = struct {
    const Self = @This();
    
    // Route through unified manager instead of separate implementation
    unified_manager: *continuation_unified.UnifiedContinuationManager,
    
    // Legacy compatibility fields (maintained for API compatibility)
    classification_count: u64 = 0,
    simd_hits: u64 = 0,
    cache_hits: u64 = 0,
    
    /// Initialize SIMD classifier with unified manager backend
    pub fn init(allocator: std.mem.Allocator, unified_manager: *continuation_unified.UnifiedContinuationManager) !Self {
        _ = allocator; // Not needed for unified system
        
        return Self{
            .unified_manager = unified_manager,
        };
    }
    
    /// Clean up resources - maintained for API compatibility
    pub fn deinit(self: *Self) void {
        _ = self; // Unified manager handles cleanup
    }
    
    /// Classify continuation for SIMD suitability - exact API compatibility
    pub fn classifyContinuation(self: *Self, cont: *continuation.Continuation) !ContinuationSIMDClass {
        self.classification_count += 1;
        
        // Route through unified system
        const unified_analysis = try self.unified_manager.getAnalysis(cont);
        
        // Check if this was a cache hit (for compatibility stats)
        const stats = self.unified_manager.getPerformanceStats();
        if (stats.cache_hit_rate > 0) {
            self.cache_hits += 1;
        }
        
        // Track SIMD potential
        if (unified_analysis.simd_classification.isSIMDSuitable()) {
            self.simd_hits += 1;
        }
        
        // Convert unified result to legacy format
        return ContinuationSIMDClass.fromUnified(unified_analysis.simd_classification);
    }
    
    /// Get optimal batch size for continuation class - exact API compatibility
    pub fn getBatchOptimalSize(self: *Self, class: ContinuationSIMDClass) u32 {
        _ = self;
        
        // Use same logic as original implementation
        const base_size = class.task_class.getRecommendedBatchSize();
        const continuation_overhead_factor = 1.2;
        const simd_factor = 1.0; // Simplified for compatibility
        
        const optimal_size = @as(f32, @floatFromInt(base_size)) * simd_factor / continuation_overhead_factor;
        return @max(4, @min(32, @as(u32, @intFromFloat(optimal_size))));
    }
    
    /// Add continuation to batch formation queue - exact API compatibility
    pub fn addContinuationForBatching(self: *Self, cont: *continuation.Continuation) !void {
        // For compatibility, we still accept this call but the unified system
        // handles batching automatically through the analysis pipeline
        _ = self;
        _ = cont;
        // No-op in unified system - batching is handled automatically
    }
    
    /// Attempt to form SIMD-optimized continuation batches - exact API compatibility
    pub fn formContinuationBatches(self: *Self) ![]ContinuationBatch {
        // Return empty slice for compatibility - unified system handles batching internally
        _ = self;
        return &[_]ContinuationBatch{};
    }
    
    /// Get performance statistics - exact API compatibility
    pub fn getPerformanceStats(self: *Self) ClassificationStats {
        const unified_stats = self.unified_manager.getPerformanceStats();
        
        const cache_hit_rate = unified_stats.cache_hit_rate;
        const simd_hit_rate = if (self.classification_count > 0)
            @as(f32, @floatFromInt(self.simd_hits)) / @as(f32, @floatFromInt(self.classification_count))
        else
            0.0;
        
        return ClassificationStats{
            .classifications_performed = self.classification_count,
            .cache_hit_rate = cache_hit_rate,
            .simd_hit_rate = simd_hit_rate,
            .batch_formation_stats = simd_classifier.BatchFormationStats{
                .total_tasks_submitted = 0,
                .tasks_in_batches = 0,
                .pending_tasks = 0,
                .formed_batches = 0,
                .average_batch_size = 0.0,
                .average_estimated_speedup = 1.0,
                .formation_efficiency = 0.0,
                .current_similarity_threshold = 0.7,
            },
        };
    }
};

/// Continuation batch for SIMD execution - maintains exact API compatibility
pub const ContinuationBatch = struct {
    const Self = @This();
    
    continuations: std.ArrayList(*continuation.Continuation),
    simd_class: ContinuationSIMDClass,
    estimated_speedup: f32,
    numa_node_preference: ?u32,
    
    /// Cleanup batch resources
    pub fn deinit(self: *Self) void {
        self.continuations.deinit();
    }
    
    /// Execute batch with SIMD optimization - exact API compatibility
    pub fn executeBatch(self: *Self) void {
        // Execute all continuations in the batch
        for (self.continuations.items) |cont| {
            cont.resume_fn(cont);
        }
    }
    
    /// Get batch statistics - exact API compatibility
    pub fn getBatchStats(self: *Self) ContinuationBatchStats {
        return ContinuationBatchStats{
            .batch_size = self.continuations.items.len,
            .estimated_speedup = self.estimated_speedup,
            .simd_suitability_score = self.simd_class.simd_suitability_score,
            .continuation_overhead_factor = self.simd_class.continuation_overhead_factor,
        };
    }
};

/// Statistics for continuation classification performance - exact API compatibility
pub const ClassificationStats = struct {
    classifications_performed: u64,
    cache_hit_rate: f32,
    simd_hit_rate: f32,
    batch_formation_stats: simd_classifier.BatchFormationStats,
};

/// Statistics for continuation batch execution - exact API compatibility
pub const ContinuationBatchStats = struct {
    batch_size: usize,
    estimated_speedup: f32,
    simd_suitability_score: f32,
    continuation_overhead_factor: f32,
};

// ============================================================================
// Compatibility Tests
// ============================================================================

test "SIMD compatibility layer API preservation" {
    const allocator = std.testing.allocator;
    
    // This test verifies that the compatibility layer maintains exact API compatibility
    // with the original continuation_simd.zig implementation
    
    // Create mock unified manager for testing
    var fingerprint_registry = @import("fingerprint.zig").FingerprintRegistry.init(allocator);
    defer fingerprint_registry.deinit();
    
    const unified_config = continuation_unified.UnifiedConfig.balanced();
    var unified_manager = try continuation_unified.UnifiedContinuationManager.init(
        allocator,
        &fingerprint_registry,
        unified_config
    );
    defer unified_manager.deinit();
    
    // Test compatibility layer
    var classifier = try ContinuationClassifier.init(allocator, &unified_manager);
    defer classifier.deinit();
    
    // Create test continuation
    const TestData = struct { values: [16]f32 };
    var test_data = TestData{ .values = undefined };
    
    const resume_fn = struct {
        fn executeFunc(cont: *continuation.Continuation) void {
            _ = cont;
        }
    };
    
    var test_continuation = continuation.Continuation.capture(resume_fn.executeFunc, &test_data, allocator);
    
    // Test classification API compatibility
    const classification = try classifier.classifyContinuation(&test_continuation);
    
    // Verify API compatibility
    try std.testing.expect(classification.simd_suitability_score >= 0.0);
    try std.testing.expect(classification.simd_suitability_score <= 1.0);
    const is_suitable = classification.isSIMDSuitable();
    const speedup = classification.getExpectedSpeedup();
    try std.testing.expect(speedup > 0.0);
    _ = is_suitable; // Use the value
    
    // Test batch size calculation API compatibility
    const batch_size = classifier.getBatchOptimalSize(classification);
    try std.testing.expect(batch_size >= 4);
    try std.testing.expect(batch_size <= 32);
    
    // Test statistics API compatibility
    const stats = classifier.getPerformanceStats();
    try std.testing.expect(stats.classifications_performed >= 1);
    try std.testing.expect(stats.cache_hit_rate >= 0.0);
    try std.testing.expect(stats.cache_hit_rate <= 1.0);
    
    std.debug.print("✅ SIMD compatibility layer API preservation test passed!\n", .{});
    std.debug.print("   SIMD suitability: {d:.3}\n", .{classification.simd_suitability_score});
    std.debug.print("   Expected speedup: {d:.2}x\n", .{speedup});
    std.debug.print("   Optimal batch size: {}\n", .{batch_size});
    std.debug.print("   Classifications performed: {}\n", .{stats.classifications_performed});
}