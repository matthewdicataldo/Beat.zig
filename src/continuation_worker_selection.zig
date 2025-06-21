const std = @import("std");
const core = @import("core.zig");
const continuation = @import("continuation.zig");
const continuation_simd = @import("continuation_simd.zig");
const continuation_predictive = @import("continuation_predictive.zig");
const advanced_worker_selection = @import("advanced_worker_selection.zig");
const topology = @import("topology.zig");
const fingerprint = @import("fingerprint.zig");
const enhanced_errors = @import("enhanced_errors.zig");

// ============================================================================
// Advanced Worker Selection for Continuation Stealing (Phase 1 Integration)
// ============================================================================

/// Enhanced worker selection for continuation submission with multi-criteria optimization
pub const ContinuationWorkerSelector = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    
    // Advanced worker selection components
    advanced_selector: *advanced_worker_selection.AdvancedWorkerSelector,
    selection_criteria: advanced_worker_selection.SelectionCriteria,
    
    // Continuation-specific enhancements
    continuation_locality_tracker: ContinuationLocalityTracker,
    numa_preference_cache: NumaPreferenceCache,
    
    // Performance tracking
    selection_count: u64,
    optimal_selections: u64,
    locality_hits: u64,
    
    /// Initialize continuation worker selector
    pub fn init(
        allocator: std.mem.Allocator,
        advanced_selector: *advanced_worker_selection.AdvancedWorkerSelector,
        criteria: ?advanced_worker_selection.SelectionCriteria
    ) !Self {
        return Self{
            .allocator = allocator,
            .advanced_selector = advanced_selector,
            .selection_criteria = criteria orelse advanced_worker_selection.SelectionCriteria.balanced(),
            .continuation_locality_tracker = try ContinuationLocalityTracker.init(allocator),
            .numa_preference_cache = try NumaPreferenceCache.init(allocator),
            .selection_count = 0,
            .optimal_selections = 0,
            .locality_hits = 0,
        };
    }
    
    /// Clean up resources
    pub fn deinit(self: *Self) void {
        self.continuation_locality_tracker.deinit();
        self.numa_preference_cache.deinit();
    }
    
    /// Select optimal worker for continuation with enhanced criteria
    pub fn selectWorkerForContinuation(
        self: *Self,
        cont: *continuation.Continuation,
        pool: *core.ThreadPool,
        simd_class: ?continuation_simd.ContinuationSIMDClass,
        prediction: ?continuation_predictive.PredictionResult
    ) !u32 {
        self.selection_count += 1;
        
        // Enhance base task with continuation-specific information
        const task = self.createTaskFromContinuation(cont);
        
        // Create continuation-specific selection context
        const context = ContinuationSelectionContext{
            .continuation = cont,
            .simd_class = simd_class,
            .prediction = prediction,
            .numa_preference = self.getNUMAPreference(cont),
            .locality_history = self.continuation_locality_tracker.getLocalityScore(cont),
        };
        
        // Use advanced worker selection with continuation enhancements
        const worker_evaluations = try self.evaluateWorkersForContinuation(pool, &task, &context);
        defer self.allocator.free(worker_evaluations);
        
        // Select best worker based on weighted criteria
        const selected_worker_id = self.selectBestWorker(worker_evaluations);
        
        // Update tracking and caches
        self.updateTrackingInfo(cont, selected_worker_id, &context);
        
        return selected_worker_id;
    }
    
    /// Create continuation-specific worker evaluations
    fn evaluateWorkersForContinuation(
        self: *Self,
        pool: *core.ThreadPool,
        task: *const core.Task,
        context: *const ContinuationSelectionContext
    ) ![]advanced_worker_selection.WorkerEvaluation {
        _ = task; // For now, task-specific scoring not implemented
        const evaluations = try self.allocator.alloc(advanced_worker_selection.WorkerEvaluation, pool.workers.len);
        
        // Create base evaluations for each worker
        for (evaluations, 0..) |*eval, i| {
            eval.* = advanced_worker_selection.WorkerEvaluation.init(i);
            
            // Calculate base scores
            eval.load_balance_score = self.calculateLoadBalanceScore(&pool.workers[i]);
            eval.topology_score = self.calculateTopologyScore(&pool.workers[i], context);
            eval.simd_score = self.calculateSIMDScore(&pool.workers[i], context);
            eval.confidence_score = self.calculateConfidenceScore(&pool.workers[i], context);
            eval.prediction_score = self.calculatePredictionScore(&pool.workers[i], context);
            eval.exploration_score = 0.5; // Default exploration score
            
            // Calculate weighted score with continuation-specific criteria
            eval.weighted_score = self.calculateContinuationWeightedScore(eval);
        }
        
        // Normalize scores relative to all workers
        self.normalizeEvaluationScores(evaluations);
        
        return evaluations;
    }
    
    /// Calculate load balance score for worker
    fn calculateLoadBalanceScore(self: *Self, worker: *const core.ThreadPool.Worker) f32 {
        _ = self;
        // Simple heuristic - for now assume all workers equally loaded
        // In real implementation, this would check queue lengths
        _ = worker;
        return 0.5;
    }
    
    /// Calculate topology score for worker
    fn calculateTopologyScore(self: *Self, worker: *const core.ThreadPool.Worker, context: *const ContinuationSelectionContext) f32 {
        return self.enhanceTopologyScore(0.5, worker, context);
    }
    
    /// Calculate SIMD score for worker  
    fn calculateSIMDScore(self: *Self, worker: *const core.ThreadPool.Worker, context: *const ContinuationSelectionContext) f32 {
        if (context.simd_class) |simd| {
            return self.enhanceSIMDScore(0.5, worker, simd);
        }
        return 0.5;
    }
    
    /// Calculate confidence score for worker
    fn calculateConfidenceScore(self: *Self, worker: *const core.ThreadPool.Worker, context: *const ContinuationSelectionContext) f32 {
        if (context.prediction) |pred| {
            return self.enhanceConfidenceScore(0.5, worker, pred);
        }
        return 0.5;
    }
    
    /// Calculate prediction score for worker
    fn calculatePredictionScore(self: *Self, worker: *const core.ThreadPool.Worker, context: *const ContinuationSelectionContext) f32 {
        if (context.prediction) |pred| {
            return self.enhancePredictionScore(0.5, worker, pred);
        }
        return 0.5;
    }
    
    /// Enhance topology score with continuation locality information
    fn enhanceTopologyScore(
        self: *Self,
        base_score: f32,
        worker: *const core.ThreadPool.Worker,
        context: *const ContinuationSelectionContext
    ) f32 {
        _ = self; // Self not needed for this scoring
        var enhanced_score = base_score;
        
        // Prefer workers on the same NUMA node as continuation
        if (context.continuation.numa_node) |cont_numa| {
            if (worker.numa_node == cont_numa) {
                enhanced_score += 0.3; // Strong NUMA locality bonus
                
                // Additional bonus for original NUMA node
                if (context.continuation.original_numa_node == cont_numa) {
                    enhanced_score += 0.1;
                }
            }
        }
        
        // Prefer workers with good historical locality
        enhanced_score += context.locality_history * 0.2;
        
        // NUMA preference from cache
        if (context.numa_preference) |pref_numa| {
            if (worker.numa_node == pref_numa) {
                enhanced_score += 0.2;
            }
        }
        
        return @min(1.0, enhanced_score);
    }
    
    /// Enhance SIMD score based on continuation vectorization potential
    fn enhanceSIMDScore(
        self: *Self,
        base_score: f32,
        worker: *const core.ThreadPool.Worker,
        simd_class: continuation_simd.ContinuationSIMDClass
    ) f32 {
        _ = self;
        var enhanced_score = base_score;
        
        // High SIMD suitability strongly favors capable workers
        if (simd_class.isSIMDSuitable()) {
            enhanced_score += simd_class.simd_suitability_score * 0.4;
            
            // Additional bonus for high vectorization potential
            if (simd_class.vectorization_potential > 3.0) {
                enhanced_score += 0.2;
            }
        }
        
        // Prefer workers with compatible NUMA node for SIMD work
        if (simd_class.preferred_numa_node) |simd_numa| {
            if (worker.numa_node == simd_numa) {
                enhanced_score += 0.2;
            }
        }
        
        return @min(1.0, enhanced_score);
    }
    
    /// Enhance confidence score based on prediction reliability
    fn enhanceConfidenceScore(
        self: *Self,
        base_score: f32,
        worker: *const core.ThreadPool.Worker,
        prediction: continuation_predictive.PredictionResult
    ) f32 {
        _ = self;
        _ = worker;
        
        // High confidence predictions boost confidence score
        var enhanced_score = base_score + prediction.confidence * 0.5;
        
        // Bonus for high-quality prediction sources
        switch (prediction.prediction_source) {
            .simd_enhanced => enhanced_score += 0.2,
            .historical_filtered => enhanced_score += 0.1,
            .numa_optimized => enhanced_score += 0.1,
            .default => {},
        }
        
        return @min(1.0, enhanced_score);
    }
    
    /// Enhance prediction score based on execution time estimates
    fn enhancePredictionScore(
        self: *Self,
        base_score: f32,
        worker: *const core.ThreadPool.Worker,
        prediction: continuation_predictive.PredictionResult
    ) f32 {
        _ = self;
        _ = worker;
        
        var enhanced_score = base_score;
        
        // Short predicted times are preferred (lower latency)
        if (prediction.predicted_time_ns < 1_000_000) { // < 1ms
            enhanced_score += 0.3;
        } else if (prediction.predicted_time_ns < 10_000_000) { // < 10ms
            enhanced_score += 0.1;
        }
        
        // High confidence predictions get bonus
        enhanced_score += prediction.confidence * 0.2;
        
        return @min(1.0, enhanced_score);
    }
    
    /// Calculate weighted score with continuation-specific criteria
    fn calculateContinuationWeightedScore(self: *Self, eval: *const advanced_worker_selection.WorkerEvaluation) f32 {
        return eval.load_balance_score * self.selection_criteria.load_balance_weight +
               eval.prediction_score * self.selection_criteria.prediction_weight +
               eval.topology_score * self.selection_criteria.topology_weight +
               eval.confidence_score * self.selection_criteria.confidence_weight +
               eval.exploration_score * self.selection_criteria.exploration_weight +
               eval.simd_score * self.selection_criteria.simd_weight;
    }
    
    /// Normalize evaluation scores across all workers
    fn normalizeEvaluationScores(self: *Self, evaluations: []advanced_worker_selection.WorkerEvaluation) void {
        _ = self;
        
        // Find min and max weighted scores
        var min_score: f32 = std.math.floatMax(f32);
        var max_score: f32 = std.math.floatMin(f32);
        
        for (evaluations) |eval| {
            min_score = @min(min_score, eval.weighted_score);
            max_score = @max(max_score, eval.weighted_score);
        }
        
        // Normalize to 0.0-1.0 range
        const score_range = max_score - min_score;
        for (evaluations) |*eval| {
            eval.normalized_score = if (score_range > 0.0)
                (eval.weighted_score - min_score) / score_range
            else
                0.5; // Equal scores
        }
    }
    
    /// Select best worker from evaluations
    fn selectBestWorker(self: *Self, evaluations: []const advanced_worker_selection.WorkerEvaluation) u32 {
        var best_worker_id: u32 = 0;
        var best_score: f32 = -1.0;
        
        for (evaluations) |eval| {
            if (eval.normalized_score > best_score) {
                best_score = eval.normalized_score;
                best_worker_id = @intCast(eval.worker_id);
            }
        }
        
        // Track if this was an optimal selection
        if (best_score > 0.8) { // High-quality selection
            self.optimal_selections += 1;
        }
        
        return best_worker_id;
    }
    
    /// Update tracking information after worker selection
    fn updateTrackingInfo(
        self: *Self,
        cont: *continuation.Continuation,
        worker_id: u32,
        context: *const ContinuationSelectionContext
    ) void {
        // Update locality tracker
        self.continuation_locality_tracker.recordPlacement(cont, worker_id);
        
        // Update NUMA preference cache
        if (context.numa_preference) |numa| {
            self.numa_preference_cache.updatePreference(cont.fingerprint_hash orelse 0, numa, worker_id);
        }
        
        // Track locality hits
        if (cont.numa_node) |cont_numa| {
            // Check if selected worker is on preferred NUMA node
            // This would require access to pool to check worker.numa_node
            // For now, we'll track based on context
            if (context.numa_preference == cont_numa) {
                self.locality_hits += 1;
            }
        }
    }
    
    /// Get NUMA preference for continuation
    fn getNUMAPreference(self: *Self, cont: *continuation.Continuation) ?u32 {
        if (cont.fingerprint_hash) |hash| {
            return self.numa_preference_cache.getPreference(hash);
        }
        return cont.numa_node;
    }
    
    /// Create task representation from continuation
    fn createTaskFromContinuation(self: *Self, cont: *continuation.Continuation) core.Task {
        _ = self;
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
            .fingerprint_hash = cont.fingerprint_hash,
        };
    }
    
    /// Get performance statistics
    pub fn getPerformanceStats(self: *Self) ContinuationSelectionStats {
        const selection_quality = if (self.selection_count > 0)
            @as(f32, @floatFromInt(self.optimal_selections)) / @as(f32, @floatFromInt(self.selection_count))
        else
            0.0;
            
        const locality_hit_rate = if (self.selection_count > 0)
            @as(f32, @floatFromInt(self.locality_hits)) / @as(f32, @floatFromInt(self.selection_count))
        else
            0.0;
        
        return ContinuationSelectionStats{
            .total_selections = self.selection_count,
            .optimal_selections = self.optimal_selections,
            .selection_quality_rate = selection_quality,
            .locality_hit_rate = locality_hit_rate,
            .numa_cache_entries = self.numa_preference_cache.getEntryCount(),
            .locality_tracker_entries = self.continuation_locality_tracker.getEntryCount(),
        };
    }
};

/// Context for continuation-specific worker selection
const ContinuationSelectionContext = struct {
    continuation: *continuation.Continuation,
    simd_class: ?continuation_simd.ContinuationSIMDClass,
    prediction: ?continuation_predictive.PredictionResult,
    numa_preference: ?u32,
    locality_history: f32,
};

/// Track continuation locality patterns for worker selection optimization
pub const ContinuationLocalityTracker = struct {
    const LocalityEntry = struct {
        worker_placements: [8]u32 = [_]u32{0} ** 8, // Track last 8 placements
        placement_count: usize = 0,
        locality_score: f32 = 0.5, // Start neutral
    };
    
    locality_map: std.AutoHashMap(u64, LocalityEntry),
    
    pub fn init(allocator: std.mem.Allocator) !ContinuationLocalityTracker {
        return ContinuationLocalityTracker{
            .locality_map = std.AutoHashMap(u64, LocalityEntry).init(allocator),
        };
    }
    
    pub fn deinit(self: *ContinuationLocalityTracker) void {
        self.locality_map.deinit();
    }
    
    pub fn recordPlacement(self: *ContinuationLocalityTracker, cont: *continuation.Continuation, worker_id: u32) void {
        const hash = cont.fingerprint_hash orelse return;
        
        var entry = self.locality_map.get(hash) orelse LocalityEntry{};
        
        // Add to placement history
        const index = entry.placement_count % 8;
        entry.worker_placements[index] = worker_id;
        entry.placement_count += 1;
        
        // Update locality score based on placement diversity
        self.updateLocalityScore(&entry);
        
        self.locality_map.put(hash, entry) catch |err| {
            enhanced_errors.logEnhancedError(@TypeOf(err), err, "Failed to record continuation placement in locality tracker");
            std.log.debug("Failed to update locality map for continuation hash {}: {}", .{hash, err});
        };
    }
    
    pub fn getLocalityScore(self: *ContinuationLocalityTracker, cont: *continuation.Continuation) f32 {
        const hash = cont.fingerprint_hash orelse return 0.5;
        return if (self.locality_map.get(hash)) |entry| entry.locality_score else 0.5;
    }
    
    fn updateLocalityScore(self: *ContinuationLocalityTracker, entry: *LocalityEntry) void {
        _ = self;
        
        // Calculate locality score based on worker placement patterns
        // Higher score for consistent worker usage (better locality)
        const placements_to_check = @min(entry.placement_count, 8);
        if (placements_to_check < 2) {
            entry.locality_score = 0.5;
            return;
        }
        
        var same_worker_count: u32 = 0;
        const first_worker = entry.worker_placements[0];
        
        for (entry.worker_placements[0..placements_to_check]) |worker_id| {
            if (worker_id == first_worker) {
                same_worker_count += 1;
            }
        }
        
        // Score based on worker consistency
        entry.locality_score = @as(f32, @floatFromInt(same_worker_count)) / @as(f32, @floatFromInt(placements_to_check));
    }
    
    fn getEntryCount(self: *ContinuationLocalityTracker) u32 {
        return @intCast(self.locality_map.count());
    }
};

/// Cache NUMA preferences for continuations
pub const NumaPreferenceCache = struct {
    const PreferenceEntry = struct {
        numa_node: u32,
        worker_id: u32,
        access_count: u32 = 1,
        timestamp: u64,
    };
    
    preference_map: std.AutoHashMap(u64, PreferenceEntry),
    max_entries: usize = 256,
    
    pub fn init(allocator: std.mem.Allocator) !NumaPreferenceCache {
        return NumaPreferenceCache{
            .preference_map = std.AutoHashMap(u64, PreferenceEntry).init(allocator),
        };
    }
    
    pub fn deinit(self: *NumaPreferenceCache) void {
        self.preference_map.deinit();
    }
    
    pub fn updatePreference(self: *NumaPreferenceCache, hash: u64, numa_node: u32, worker_id: u32) void {
        if (self.preference_map.count() >= self.max_entries) {
            self.evictOldEntries() catch |err| {
                enhanced_errors.logEnhancedError(@TypeOf(err), err, "Failed to evict old entries from NUMA preference cache");
                std.log.warn("NUMA preference cache eviction failed, continuing with full cache: {}", .{err});
            };
        }
        
        const entry = PreferenceEntry{
            .numa_node = numa_node,
            .worker_id = worker_id,
            .timestamp = @as(u64, @intCast(std.time.nanoTimestamp())),
        };
        
        self.preference_map.put(hash, entry) catch |err| {
            enhanced_errors.logEnhancedError(@TypeOf(err), err, "Failed to update NUMA preference cache");
            std.log.debug("Failed to update NUMA preference for hash {}: {}", .{hash, err});
        };
    }
    
    pub fn getPreference(self: *NumaPreferenceCache, hash: u64) ?u32 {
        if (self.preference_map.getPtr(hash)) |entry| {
            entry.access_count += 1;
            return entry.numa_node;
        }
        return null;
    }
    
    fn evictOldEntries(self: *NumaPreferenceCache) !void {
        // Simple LRU eviction - remove 25% of entries
        const eviction_count = self.max_entries / 4;
        var keys_to_remove = try std.ArrayList(u64).initCapacity(self.preference_map.allocator, eviction_count);
        defer keys_to_remove.deinit();
        
        var iterator = self.preference_map.iterator();
        var oldest_timestamp: u64 = std.math.maxInt(u64);
        
        // Find oldest entries
        while (iterator.next()) |entry| {
            if (keys_to_remove.items.len < eviction_count) {
                keys_to_remove.appendAssumeCapacity(entry.key_ptr.*);
                oldest_timestamp = @min(oldest_timestamp, entry.value_ptr.timestamp);
            } else if (entry.value_ptr.timestamp < oldest_timestamp) {
                // Replace newest entry in eviction list
                for (keys_to_remove.items, 0..) |key, i| {
                    if (self.preference_map.get(key).?.timestamp == oldest_timestamp) {
                        keys_to_remove.items[i] = entry.key_ptr.*;
                        break;
                    }
                }
            }
        }
        
        // Remove selected entries
        for (keys_to_remove.items) |key| {
            _ = self.preference_map.remove(key);
        }
    }
    
    fn getEntryCount(self: *NumaPreferenceCache) u32 {
        return @intCast(self.preference_map.count());
    }
};

/// Performance statistics for continuation worker selection
pub const ContinuationSelectionStats = struct {
    total_selections: u64,
    optimal_selections: u64,
    selection_quality_rate: f32,
    locality_hit_rate: f32,
    numa_cache_entries: u32,
    locality_tracker_entries: u32,
};

// ============================================================================
// Tests
// ============================================================================

test "continuation worker selector initialization" {
    const allocator = std.testing.allocator;
    
    // Initialize with mock advanced selector
    const advanced_selector = try allocator.create(advanced_worker_selection.AdvancedWorkerSelector);
    defer allocator.destroy(advanced_selector);
    
    // Note: This would require proper initialization of AdvancedWorkerSelector
    // For now, just test the continuation selector structure
    
    const criteria = advanced_worker_selection.SelectionCriteria.balanced();
    var selector = try ContinuationWorkerSelector.init(allocator, advanced_selector, criteria);
    defer selector.deinit();
    
    // Verify initialization
    const stats = selector.getPerformanceStats();
    try std.testing.expect(stats.total_selections == 0);
    try std.testing.expect(stats.selection_quality_rate == 0.0);
    
    std.debug.print("✅ Continuation worker selector initialization test passed!\n", .{});
}

test "continuation locality tracking" {
    const allocator = std.testing.allocator;
    
    var tracker = try ContinuationLocalityTracker.init(allocator);
    defer tracker.deinit();
    
    // Create test continuation
    const TestData = struct { value: i32 = 42 };
    var test_data = TestData{};
    
    const resume_fn = struct {
        fn executeFunc(cont: *continuation.Continuation) void {
            _ = cont;
        }
    };
    
    var test_continuation = continuation.Continuation.capture(resume_fn.executeFunc, &test_data, allocator);
    test_continuation.fingerprint_hash = 12345;
    
    // Record placements on same worker (good locality)
    tracker.recordPlacement(&test_continuation, 0);
    tracker.recordPlacement(&test_continuation, 0);
    tracker.recordPlacement(&test_continuation, 0);
    
    // Check locality score (should be high for consistent placement)
    const locality_score = tracker.getLocalityScore(&test_continuation);
    try std.testing.expect(locality_score > 0.8); // High locality
    
    // Record placement on different worker (reduces locality)
    tracker.recordPlacement(&test_continuation, 1);
    
    const updated_score = tracker.getLocalityScore(&test_continuation);
    try std.testing.expect(updated_score < locality_score); // Locality decreased
    
    std.debug.print("✅ Continuation locality tracking test passed!\n", .{});
    std.debug.print("   Initial locality score: {d:.3}\n", .{locality_score});
    std.debug.print("   Updated locality score: {d:.3}\n", .{updated_score});
}

test "numa preference caching" {
    const allocator = std.testing.allocator;
    
    var cache = try NumaPreferenceCache.init(allocator);
    defer cache.deinit();
    
    // Test preference storage and retrieval
    const hash: u64 = 54321;
    cache.updatePreference(hash, 1, 5); // NUMA node 1, worker 5
    
    const preference = cache.getPreference(hash);
    try std.testing.expect(preference != null);
    try std.testing.expect(preference.? == 1);
    
    // Test cache miss
    const missing_preference = cache.getPreference(99999);
    try std.testing.expect(missing_preference == null);
    
    std.debug.print("✅ NUMA preference caching test passed!\n", .{});
    std.debug.print("   Cached NUMA preference: {?}\n", .{preference});
}