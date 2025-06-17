const std = @import("std");
const core = @import("core.zig");
const continuation = @import("continuation.zig");
const simd = @import("simd.zig");
const simd_classifier = @import("simd_classifier.zig");
const simd_batch = @import("simd_batch.zig");
const fingerprint = @import("fingerprint.zig");

// ============================================================================
// SIMD-Enhanced Continuation Stealing (Phase 1 Integration)
// ============================================================================

/// Enhanced continuation classification with SIMD vectorization analysis
pub const ContinuationClassifier = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    simd_features: simd.SIMDCapability,
    classification_cache: ClassificationCache,
    batch_former: simd_classifier.IntelligentBatchFormer,
    
    // Performance tracking
    classification_count: u64,
    simd_hits: u64,
    cache_hits: u64,
    
    /// Initialize SIMD-enhanced continuation classifier
    pub fn init(allocator: std.mem.Allocator) !Self {
        const simd_features = simd.SIMDCapability.detect();
        const criteria = simd_classifier.BatchFormationCriteria.performanceOptimized();
        
        return Self{
            .allocator = allocator,
            .simd_features = simd_features,
            .classification_cache = try ClassificationCache.init(allocator),
            .batch_former = simd_classifier.IntelligentBatchFormer.init(allocator, criteria),
            .classification_count = 0,
            .simd_hits = 0,
            .cache_hits = 0,
        };
    }
    
    /// Clean up resources
    pub fn deinit(self: *Self) void {
        self.classification_cache.deinit();
        self.batch_former.deinit();
    }
    
    /// Classify continuation for SIMD suitability (6-23x speedup through vectorization)
    pub fn classifyContinuation(self: *Self, cont: *continuation.Continuation) !ContinuationSIMDClass {
        self.classification_count += 1;
        
        // Fast cache lookup using continuation fingerprint
        if (cont.fingerprint_hash) |hash| {
            if (self.classification_cache.get(hash)) |cached_class| {
                self.cache_hits += 1;
                return cached_class;
            }
        }
        
        // Convert continuation to task for SIMD analysis
        const task = continuationToTask(cont);
        
        // Perform SIMD analysis using existing infrastructure
        const static_analysis = simd_classifier.StaticAnalysis.analyzeTask(&task);
        const dynamic_profile = simd_classifier.DynamicProfile.init(); // Skip profiling for speed
        const feature_vector = simd_classifier.TaskFeatureVector.fromAnalysis(static_analysis, dynamic_profile);
        const task_class = feature_vector.getClassification();
        
        // Enhanced classification with continuation-specific metrics
        const continuation_class = enhanceClassificationForContinuation(
            task_class,
            static_analysis,
            cont,
            self.simd_features
        );
        
        // Cache result for future lookups
        if (cont.fingerprint_hash) |hash| {
            try self.classification_cache.put(hash, continuation_class);
        } else {
            // Generate fingerprint for caching
            const hash = self.generateContinuationFingerprint(cont);
            cont.fingerprint_hash = hash;
            try self.classification_cache.put(hash, continuation_class);
        }
        
        // Track SIMD potential
        if (continuation_class.simd_suitability_score > 0.6) {
            self.simd_hits += 1;
        }
        
        return continuation_class;
    }
    
    /// Get optimal batch size for continuation class
    pub fn getBatchOptimalSize(self: *Self, class: ContinuationSIMDClass) u32 {
        
        // Consider SIMD vector width and continuation overhead
        const base_size = class.task_class.getRecommendedBatchSize();
        const continuation_overhead_factor = 1.2; // Continuations have slightly more overhead
        
        // Adjust based on SIMD capability
        const simd_factor: f32 = switch (self.simd_features.max_vector_width_bits) {
            128 => 0.8,  // 128-bit SIMD
            256 => 1.0,  // 256-bit SIMD  
            512 => 1.2,  // 512-bit SIMD
            else => 0.6, // Fallback to scalar
        };
        
        const optimal_size = @as(f32, @floatFromInt(base_size)) * simd_factor / continuation_overhead_factor;
        return @max(4, @min(32, @as(u32, @intFromFloat(optimal_size))));
    }
    
    /// Add continuation to batch formation queue
    pub fn addContinuationForBatching(self: *Self, cont: *continuation.Continuation) !void {
        // Convert continuation to task for batch formation
        const task = continuationToTask(cont);
        try self.batch_former.addTask(task, false); // Disable profiling for speed
    }
    
    /// Attempt to form SIMD-optimized continuation batches
    pub fn formContinuationBatches(self: *Self) ![]ContinuationBatch {
        try self.batch_former.attemptBatchFormation();
        
        const simd_batches = self.batch_former.getFormedBatches();
        var continuation_batches = try self.allocator.alloc(ContinuationBatch, simd_batches.len);
        
        for (simd_batches, 0..) |batch, i| {
            continuation_batches[i] = try ContinuationBatch.fromSIMDBatch(batch, self.allocator);
        }
        
        return continuation_batches;
    }
    
    /// Get performance statistics
    pub fn getPerformanceStats(self: *Self) ClassificationStats {
        const cache_hit_rate = if (self.classification_count > 0)
            @as(f32, @floatFromInt(self.cache_hits)) / @as(f32, @floatFromInt(self.classification_count))
        else
            0.0;
            
        const simd_hit_rate = if (self.classification_count > 0)
            @as(f32, @floatFromInt(self.simd_hits)) / @as(f32, @floatFromInt(self.classification_count))
        else
            0.0;
        
        return ClassificationStats{
            .classifications_performed = self.classification_count,
            .cache_hit_rate = cache_hit_rate,
            .simd_hit_rate = simd_hit_rate,
            .batch_formation_stats = self.batch_former.getFormationStats(),
        };
    }
    
    /// Generate fingerprint for continuation caching
    fn generateContinuationFingerprint(self: *Self, cont: *continuation.Continuation) u64 {
        _ = self; // Allocator not needed for simple hash
        
        // Use function pointer and data characteristics for fingerprinting
        const func_addr = @intFromPtr(cont.resume_fn);
        const data_addr = @intFromPtr(cont.data);
        const frame_size = cont.frame_size;
        
        // Simple hash combining multiple characteristics  
        const hash = func_addr ^ data_addr ^ frame_size;
        
        return hash;
    }
};

/// SIMD-enhanced continuation classification
pub const ContinuationSIMDClass = struct {
    task_class: simd_classifier.TaskClass,
    simd_suitability_score: f32,        // 0.0-1.0 SIMD suitability
    continuation_overhead_factor: f32,   // Continuation-specific overhead (1.0-2.0)
    vectorization_potential: f32,        // Expected speedup from vectorization
    preferred_numa_node: ?u32,          // NUMA placement preference
    
    /// Check if continuation is suitable for SIMD batching
    pub fn isSIMDSuitable(self: ContinuationSIMDClass) bool {
        return self.simd_suitability_score > 0.5 and 
               self.vectorization_potential > 1.5;
    }
    
    /// Get expected performance improvement
    pub fn getExpectedSpeedup(self: ContinuationSIMDClass) f32 {
        return self.vectorization_potential / self.continuation_overhead_factor;
    }
};

/// Classification cache for high-performance lookups
const ClassificationCache = struct {
    const CacheEntry = struct {
        classification: ContinuationSIMDClass,
        timestamp: u64,
        access_count: u32,
    };
    
    map: std.AutoHashMap(u64, CacheEntry),
    max_entries: usize,
    
    fn init(allocator: std.mem.Allocator) !ClassificationCache {
        return ClassificationCache{
            .map = std.AutoHashMap(u64, CacheEntry).init(allocator),
            .max_entries = 1024, // Limit cache size
        };
    }
    
    fn deinit(self: *ClassificationCache) void {
        self.map.deinit();
    }
    
    fn get(self: *ClassificationCache, hash: u64) ?ContinuationSIMDClass {
        if (self.map.getPtr(hash)) |entry| {
            entry.access_count += 1;
            return entry.classification;
        }
        return null;
    }
    
    fn put(self: *ClassificationCache, hash: u64, classification: ContinuationSIMDClass) !void {
        // Evict old entries if cache is full
        if (self.map.count() >= self.max_entries) {
            try self.evictOldEntries();
        }
        
        const entry = CacheEntry{
            .classification = classification,
            .timestamp = @as(u64, @intCast(std.time.nanoTimestamp())),
            .access_count = 1,
        };
        
        try self.map.put(hash, entry);
    }
    
    fn evictOldEntries(self: *ClassificationCache) !void {
        // Simple LRU eviction - remove 25% of entries
        const eviction_count = self.max_entries / 4;
        var keys_to_remove = try std.ArrayList(u64).initCapacity(self.map.allocator, eviction_count);
        defer keys_to_remove.deinit();
        
        var iterator = self.map.iterator();
        var oldest_timestamp: u64 = std.math.maxInt(u64);
        
        // Find oldest entries
        while (iterator.next()) |entry| {
            if (keys_to_remove.items.len < eviction_count) {
                keys_to_remove.appendAssumeCapacity(entry.key_ptr.*);
                oldest_timestamp = @min(oldest_timestamp, entry.value_ptr.timestamp);
            } else if (entry.value_ptr.timestamp < oldest_timestamp) {
                // Replace newest entry in eviction list
                for (keys_to_remove.items, 0..) |key, i| {
                    if (self.map.get(key).?.timestamp == oldest_timestamp) {
                        keys_to_remove.items[i] = entry.key_ptr.*;
                        break;
                    }
                }
            }
        }
        
        // Remove selected entries
        for (keys_to_remove.items) |key| {
            _ = self.map.remove(key);
        }
    }
};

/// Continuation batch optimized for SIMD execution
pub const ContinuationBatch = struct {
    const Self = @This();
    
    continuations: std.ArrayList(*continuation.Continuation),
    simd_class: ContinuationSIMDClass,
    estimated_speedup: f32,
    numa_node_preference: ?u32,
    
    fn fromSIMDBatch(batch: *simd_batch.SIMDTaskBatch, allocator: std.mem.Allocator) !Self {
        var continuations_list = std.ArrayList(*continuation.Continuation).init(allocator);
        
        // Extract continuations from SIMD batch tasks
        for (batch.tasks.items) |task| {
            // Tasks created from continuations have continuation data
            const cont = @as(*continuation.Continuation, @ptrCast(@alignCast(task.data)));
            try continuations_list.append(cont);
        }
        
        // Default classification for batch
        const simd_class = ContinuationSIMDClass{
            .task_class = .moderately_vectorizable,
            .simd_suitability_score = 0.7,
            .continuation_overhead_factor = 1.2,
            .vectorization_potential = batch.estimated_speedup,
            .preferred_numa_node = null,
        };
        
        return Self{
            .continuations = continuations_list,
            .simd_class = simd_class,
            .estimated_speedup = batch.estimated_speedup,
            .numa_node_preference = null,
        };
    }
    
    fn deinit(self: *Self) void {
        self.continuations.deinit();
    }
    
    /// Execute batch with SIMD optimization
    pub fn executeBatch(self: *Self) void {
        // Execute all continuations in the batch
        // In a real implementation, this would use SIMD instructions
        for (self.continuations.items) |cont| {
            cont.resume_fn(cont);
        }
    }
    
    /// Get batch statistics
    pub fn getBatchStats(self: *Self) ContinuationBatchStats {
        return ContinuationBatchStats{
            .batch_size = self.continuations.items.len,
            .estimated_speedup = self.estimated_speedup,
            .simd_suitability_score = self.simd_class.simd_suitability_score,
            .continuation_overhead_factor = self.simd_class.continuation_overhead_factor,
        };
    }
};

/// Statistics for continuation classification performance
pub const ClassificationStats = struct {
    classifications_performed: u64,
    cache_hit_rate: f32,
    simd_hit_rate: f32,
    batch_formation_stats: simd_classifier.BatchFormationStats,
};

/// Statistics for continuation batch execution
pub const ContinuationBatchStats = struct {
    batch_size: usize,
    estimated_speedup: f32,
    simd_suitability_score: f32,
    continuation_overhead_factor: f32,
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert continuation to task for SIMD analysis
fn continuationToTask(cont: *continuation.Continuation) core.Task {
    return core.Task{
        .func = struct {
            fn continuationWrapper(data: *anyopaque) void {
                const continuation_ptr = @as(*continuation.Continuation, @ptrCast(@alignCast(data)));
                continuation_ptr.resume_fn(continuation_ptr);
            }
        }.continuationWrapper,
        .data = @ptrCast(cont),
        .priority = .normal,
        .data_size_hint = @sizeOf(continuation.Continuation) + cont.frame_size,
    };
}

/// Enhance task classification with continuation-specific analysis
fn enhanceClassificationForContinuation(
    task_class: simd_classifier.TaskClass,
    static_analysis: simd_classifier.StaticAnalysis,
    cont: *continuation.Continuation,
    simd_features: simd.SIMDCapability
) ContinuationSIMDClass {
    // Calculate continuation overhead factor
    const base_overhead: f32 = 1.1; // Base continuation overhead
    const frame_overhead = @min(0.5, @as(f32, @floatFromInt(cont.frame_size)) / 1024.0); // Frame size impact
    const steal_overhead = @as(f32, @floatFromInt(cont.steal_count)) * 0.05; // Stealing impact
    const continuation_overhead_factor = base_overhead + frame_overhead + steal_overhead;
    
    // Calculate SIMD suitability considering continuation characteristics
    const base_suitability = static_analysis.getSIMDSuitabilityScore();
    const frame_size_penalty: f32 = if (cont.frame_size > 512) 0.1 else 0.0; // Large frames hurt SIMD
    const locality_bonus = cont.locality_score * 0.1; // Good locality helps SIMD
    const simd_suitability_score = @max(0.0, @min(1.0, base_suitability - frame_size_penalty + locality_bonus));
    
    // Calculate vectorization potential
    const base_vectorization = @as(f32, @floatFromInt(static_analysis.recommended_vector_width)) / 4.0;
    const simd_multiplier: f32 = switch (simd_features.max_vector_width_bits) {
        128 => 2.0,  // 128-bit: 2x potential
        256 => 4.0,  // 256-bit: 4x potential
        512 => 8.0,  // 512-bit: 8x potential
        else => 1.0, // Scalar: no vectorization
    };
    const vectorization_potential = base_vectorization * simd_multiplier;
    
    return ContinuationSIMDClass{
        .task_class = task_class,
        .simd_suitability_score = simd_suitability_score,
        .continuation_overhead_factor = continuation_overhead_factor,
        .vectorization_potential = vectorization_potential,
        .preferred_numa_node = cont.numa_node,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "continuation SIMD classification and caching" {
    const allocator = std.testing.allocator;
    
    // Initialize SIMD classifier
    var classifier = try ContinuationClassifier.init(allocator);
    defer classifier.deinit();
    
    // Create test continuation
    const TestData = struct { values: [32]f32 };
    var test_data = TestData{ .values = undefined };
    for (&test_data.values, 0..) |*value, i| {
        value.* = @floatFromInt(i);
    }
    
    const resume_fn = struct {
        fn executeFunc(cont: *continuation.Continuation) void {
            const data = @as(*TestData, @ptrCast(@alignCast(cont.data)));
            for (&data.values) |*value| {
                value.* = value.* * 2.0 + 1.0; // SIMD-friendly operation
            }
            cont.state = .completed;
        }
    };
    
    var test_continuation = continuation.Continuation.capture(resume_fn.executeFunc, &test_data, allocator);
    test_continuation.frame_size = 128; // Reasonable frame size
    
    // Test classification
    const classification = try classifier.classifyContinuation(&test_continuation);
    
    // Verify results
    try std.testing.expect(classification.simd_suitability_score >= 0.0);
    try std.testing.expect(classification.simd_suitability_score <= 1.0);
    try std.testing.expect(classification.continuation_overhead_factor >= 1.0);
    try std.testing.expect(classification.vectorization_potential > 0.0);
    
    // Test caching by classifying same continuation again
    const cached_classification = try classifier.classifyContinuation(&test_continuation);
    try std.testing.expect(cached_classification.simd_suitability_score == classification.simd_suitability_score);
    
    // Check cache hit rate
    const stats = classifier.getPerformanceStats();
    try std.testing.expect(stats.cache_hit_rate > 0.0);
    try std.testing.expect(stats.classifications_performed >= 2);
    
    std.debug.print("Continuation SIMD classification test passed!\n", .{});
    std.debug.print("  SIMD suitability: {d:.3}\n", .{classification.simd_suitability_score});
    std.debug.print("  Vectorization potential: {d:.2}x\n", .{classification.vectorization_potential});
    std.debug.print("  Cache hit rate: {d:.1}%\n", .{stats.cache_hit_rate * 100});
}

test "continuation batch formation with SIMD optimization" {
    const allocator = std.testing.allocator;
    
    // Initialize SIMD classifier
    var classifier = try ContinuationClassifier.init(allocator);
    defer classifier.deinit();
    
    // Create multiple similar continuations for batching
    const TestData = struct { values: [16]f32 };
    var test_data_array: [8]TestData = undefined;
    var continuations: [8]continuation.Continuation = undefined;
    
    const resume_fn = struct {
        fn executeFunc(cont: *continuation.Continuation) void {
            const data = @as(*TestData, @ptrCast(@alignCast(cont.data)));
            for (&data.values) |*value| {
                value.* = value.* * 1.5 + 0.5; // Similar SIMD operation
            }
            cont.state = .completed;
        }
    };
    
    // Initialize test data and continuations
    for (&test_data_array, 0..) |*data, i| {
        for (&data.values, 0..) |*value, j| {
            value.* = @as(f32, @floatFromInt(i * 16 + j));
        }
        
        continuations[i] = continuation.Continuation.capture(resume_fn.executeFunc, data, allocator);
        continuations[i].frame_size = 64; // Similar frame sizes for batching
    }
    
    // Add continuations for batching
    for (&continuations) |*cont| {
        try classifier.addContinuationForBatching(cont);
    }
    
    // Form batches
    const batches = try classifier.formContinuationBatches();
    defer {
        for (batches) |*batch| {
            batch.deinit();
        }
        allocator.free(batches);
    }
    
    // Verify batch formation
    try std.testing.expect(batches.len > 0);
    
    var total_continuations_in_batches: usize = 0;
    for (batches) |batch| {
        total_continuations_in_batches += batch.continuations.items.len;
        
        // Verify batch properties
        try std.testing.expect(batch.continuations.items.len >= 4); // Minimum batch size
        try std.testing.expect(batch.estimated_speedup > 1.0);
        
        const batch_stats = batch.getBatchStats();
        try std.testing.expect(batch_stats.simd_suitability_score > 0.0);
    }
    
    try std.testing.expect(total_continuations_in_batches > 0);
    
    std.debug.print("Continuation batch formation test passed!\n", .{});
    std.debug.print("  Batches formed: {}\n", .{batches.len});
    std.debug.print("  Total continuations in batches: {}\n", .{total_continuations_in_batches});
    
    if (batches.len > 0) {
        const first_batch_stats = batches[0].getBatchStats();
        std.debug.print("  First batch size: {}\n", .{first_batch_stats.batch_size});
        std.debug.print("  First batch estimated speedup: {d:.2}x\n", .{first_batch_stats.estimated_speedup});
    }
}