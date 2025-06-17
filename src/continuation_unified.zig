const std = @import("std");
const core = @import("core.zig");
const continuation = @import("continuation.zig");
const fingerprint = @import("fingerprint.zig");
const topology = @import("topology.zig");
const simd = @import("simd.zig");
const scheduler = @import("scheduler.zig");
const advanced_worker_selection = @import("advanced_worker_selection.zig");

// Legacy imports for compatibility during migration
const continuation_simd = @import("continuation_simd.zig");
const continuation_predictive = @import("continuation_predictive.zig");

// ============================================================================
// Unified Continuation Management System
// ============================================================================

/// Unified continuation manager that consolidates SIMD classification,
/// predictive accounting, and worker selection into a single optimized system
pub const UnifiedContinuationManager = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    
    // Core components
    fingerprint_generator: *fingerprint.FingerprintRegistry,
    unified_cache: UnifiedContinuationCache,
    analysis_pipeline: ContinuationAnalysisPipeline,
    decision_coordinator: ContinuationDecisionCoordinator,
    
    // Performance tracking
    performance_tracker: UnifiedPerformanceTracker,
    
    // Configuration
    config: UnifiedConfig,
    
    /// Initialize unified continuation manager
    pub fn init(
        allocator: std.mem.Allocator,
        fingerprint_registry: *fingerprint.FingerprintRegistry,
        config: UnifiedConfig
    ) !Self {
        return Self{
            .allocator = allocator,
            .fingerprint_generator = fingerprint_registry,
            .unified_cache = try UnifiedContinuationCache.init(allocator, config.cache_config),
            .analysis_pipeline = try ContinuationAnalysisPipeline.init(allocator, config.analysis_config),
            .decision_coordinator = try ContinuationDecisionCoordinator.init(allocator, config.decision_config),
            .performance_tracker = UnifiedPerformanceTracker.init(),
            .config = config,
        };
    }
    
    /// Clean up resources
    pub fn deinit(self: *Self) void {
        self.unified_cache.deinit();
        self.analysis_pipeline.deinit();
        self.decision_coordinator.deinit();
    }
    
    /// Get comprehensive analysis for continuation with unified caching
    pub fn getAnalysis(self: *Self, cont: *continuation.Continuation) !ContinuationAnalysis {
        const start_time = std.time.nanoTimestamp();
        
        // Generate advanced fingerprint using existing fingerprint system
        const advanced_fingerprint = self.generateAdvancedFingerprint(cont);
        
        // Check unified cache first
        if (self.unified_cache.get(advanced_fingerprint)) |cached_analysis| {
            self.performance_tracker.recordCacheHit();
            return cached_analysis;
        }
        
        // Perform comprehensive analysis through pipeline
        const analysis = try self.analysis_pipeline.analyzeComprehensive(cont, advanced_fingerprint);
        
        // Coordinate decisions to eliminate conflicts
        const coordinated_analysis = try self.decision_coordinator.coordinate(analysis, cont);
        
        // Cache results for future use
        try self.unified_cache.put(advanced_fingerprint, coordinated_analysis);
        
        // Track performance
        const end_time = std.time.nanoTimestamp();
        self.performance_tracker.recordAnalysis(@as(u64, @intCast(end_time - start_time)));
        
        return coordinated_analysis;
    }
    
    /// Update analysis with actual execution results
    pub fn updateWithResults(
        self: *Self, 
        cont: *continuation.Continuation, 
        actual_execution_time_ns: u64,
        selected_worker_id: u32
    ) !void {
        const advanced_fingerprint = self.generateAdvancedFingerprint(cont);
        
        // Update prediction accuracy
        try self.analysis_pipeline.updatePredictionAccuracy(advanced_fingerprint.hash(), actual_execution_time_ns);
        
        // Update worker selection feedback
        try self.decision_coordinator.updateWorkerFeedback(advanced_fingerprint.hash(), selected_worker_id, actual_execution_time_ns);
        
        // Update cached analysis if present
        if (self.unified_cache.getPtr(advanced_fingerprint)) |cached_analysis| {
            cached_analysis.execution_prediction.updateAccuracy(actual_execution_time_ns);
            cached_analysis.worker_preferences.updateFeedback(selected_worker_id, actual_execution_time_ns);
        }
        
        self.performance_tracker.recordUpdate();
    }
    
    /// Get performance statistics
    pub fn getPerformanceStats(self: *Self) UnifiedPerformanceStats {
        return self.performance_tracker.getStats();
    }
    
    /// Generate advanced fingerprint using existing fingerprint system
    fn generateAdvancedFingerprint(self: *Self, cont: *continuation.Continuation) fingerprint.TaskFingerprint {
        _ = self; // Unused but kept for future registry integration
        // Convert continuation to task for fingerprinting
        const task = core.Task{
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
        
        // Use advanced fingerprinting system
        var context = fingerprint.ExecutionContext.init();
        if (cont.numa_node) |numa_node| {
            context.current_numa_node = numa_node;
        }
        context.application_phase = 0; // Default phase for continuations
        context.estimated_cycles = cont.frame_size;
        
        return fingerprint.generateTaskFingerprint(&task, &context);
    }
};

/// Unified cache entry containing all continuation metadata
const UnifiedCacheEntry = struct {
    // Advanced fingerprinting
    fingerprint: fingerprint.TaskFingerprint,
    
    // Consolidated analysis results
    analysis: ContinuationAnalysis,
    
    // Cache metadata
    timestamp: u64,
    access_count: u32,
    
    fn init(fingerprint_val: fingerprint.TaskFingerprint, analysis_val: ContinuationAnalysis) UnifiedCacheEntry {
        return UnifiedCacheEntry{
            .fingerprint = fingerprint_val,
            .analysis = analysis_val,
            .timestamp = @as(u64, @intCast(std.time.nanoTimestamp())),
            .access_count = 1,
        };
    }
};

/// High-performance unified cache replacing 3 separate cache implementations
const UnifiedContinuationCache = struct {
    map: std.AutoHashMap(u64, UnifiedCacheEntry),
    max_entries: usize,
    
    fn init(allocator: std.mem.Allocator, config: CacheConfig) !UnifiedContinuationCache {
        return UnifiedContinuationCache{
            .map = std.AutoHashMap(u64, UnifiedCacheEntry).init(allocator),
            .max_entries = config.max_entries,
        };
    }
    
    fn deinit(self: *UnifiedContinuationCache) void {
        self.map.deinit();
    }
    
    fn get(self: *UnifiedContinuationCache, fingerprint_val: fingerprint.TaskFingerprint) ?ContinuationAnalysis {
        const hash = self.fingerprintToHash(fingerprint_val);
        if (self.map.getPtr(hash)) |entry| {
            entry.access_count += 1;
            
            // Check if entry is still fresh (within 30 seconds)
            const current_time = @as(u64, @intCast(std.time.nanoTimestamp()));
            if (current_time - entry.timestamp < 30_000_000_000) {
                return entry.analysis;
            } else {
                // Remove stale entry
                _ = self.map.remove(hash);
            }
        }
        return null;
    }
    
    fn getPtr(self: *UnifiedContinuationCache, fingerprint_val: fingerprint.TaskFingerprint) ?*ContinuationAnalysis {
        const hash = self.fingerprintToHash(fingerprint_val);
        if (self.map.getPtr(hash)) |entry| {
            entry.access_count += 1;
            return &entry.analysis;
        }
        return null;
    }
    
    fn put(self: *UnifiedContinuationCache, fingerprint_val: fingerprint.TaskFingerprint, analysis: ContinuationAnalysis) !void {
        // Evict old entries if cache is full
        if (self.map.count() >= self.max_entries) {
            try self.evictOldEntries();
        }
        
        const hash = self.fingerprintToHash(fingerprint_val);
        const entry = UnifiedCacheEntry.init(fingerprint_val, analysis);
        try self.map.put(hash, entry);
    }
    
    fn evictOldEntries(self: *UnifiedContinuationCache) !void {
        // Efficient LRU eviction - remove 25% of entries
        const eviction_count = self.max_entries / 4;
        var keys_to_remove = try std.ArrayList(u64).initCapacity(self.map.allocator, eviction_count);
        defer keys_to_remove.deinit();
        
        var iterator = self.map.iterator();
        var oldest_timestamp: u64 = std.math.maxInt(u64);
        
        // Collect oldest entries for eviction
        while (iterator.next()) |entry| {
            if (keys_to_remove.items.len < eviction_count) {
                keys_to_remove.appendAssumeCapacity(entry.key_ptr.*);
                oldest_timestamp = @min(oldest_timestamp, entry.value_ptr.timestamp);
            } else if (entry.value_ptr.timestamp < oldest_timestamp) {
                // Replace newest entry in eviction list with older one
                for (keys_to_remove.items, 0..) |key, i| {
                    if (self.map.get(key).?.timestamp == oldest_timestamp) {
                        keys_to_remove.items[i] = entry.key_ptr.*;
                        oldest_timestamp = entry.value_ptr.timestamp;
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
    
    fn fingerprintToHash(self: *UnifiedContinuationCache, fingerprint_val: fingerprint.TaskFingerprint) u64 {
        _ = self;
        // Use call site hash as primary key, combine with other fields for uniqueness
        return fingerprint_val.call_site_hash ^ 
               (@as(u64, fingerprint_val.data_size_class) << 32) ^
               (@as(u64, fingerprint_val.numa_node_hint) << 16);
    }
};

/// Integrated analysis pipeline that coordinates SIMD, prediction, and worker selection
const ContinuationAnalysisPipeline = struct {
    allocator: std.mem.Allocator,
    
    // SIMD analysis components
    simd_capability: simd.SIMDCapability,
    
    // Prediction components
    one_euro_filter: scheduler.OneEuroFilter,
    velocity_filter: scheduler.OneEuroFilter,
    execution_history: std.AutoHashMap(u64, ExecutionProfile),
    
    // Performance tracking
    analysis_count: u64,
    
    fn init(allocator: std.mem.Allocator, config: AnalysisConfig) !ContinuationAnalysisPipeline {
        return ContinuationAnalysisPipeline{
            .allocator = allocator,
            .simd_capability = simd.SIMDCapability.detect(),
            .one_euro_filter = scheduler.OneEuroFilter.init(config.min_cutoff, config.beta, config.d_cutoff),
            .velocity_filter = scheduler.OneEuroFilter.init(config.velocity_min_cutoff, config.velocity_beta, config.velocity_d_cutoff),
            .execution_history = std.AutoHashMap(u64, ExecutionProfile).init(allocator),
            .analysis_count = 0,
        };
    }
    
    fn deinit(self: *ContinuationAnalysisPipeline) void {
        self.execution_history.deinit();
    }
    
    /// Perform comprehensive analysis in single coordinated pass
    fn analyzeComprehensive(
        self: *ContinuationAnalysisPipeline, 
        cont: *continuation.Continuation, 
        advanced_fingerprint: fingerprint.TaskFingerprint
    ) !ContinuationAnalysis {
        self.analysis_count += 1;
        
        // Phase 1: SIMD Analysis
        const simd_classification = self.analyzeSIMD(cont, advanced_fingerprint);
        
        // Phase 2: Execution Prediction (using SIMD results)
        const execution_prediction = try self.predictExecution(cont, advanced_fingerprint, simd_classification);
        
        // Phase 3: Worker Preferences (using both SIMD and prediction results)
        const worker_preferences = self.analyzeWorkerPreferences(cont, simd_classification, execution_prediction);
        
        // Phase 4: NUMA Coordination
        const numa_coordination = self.coordinateNUMA(cont, simd_classification, execution_prediction, worker_preferences);
        
        return ContinuationAnalysis{
            .simd_classification = simd_classification,
            .execution_prediction = execution_prediction,
            .worker_preferences = worker_preferences,
            .numa_coordination = numa_coordination,
            .analysis_timestamp = @as(u64, @intCast(std.time.nanoTimestamp())),
            .confidence_score = self.calculateOverallConfidence(simd_classification, execution_prediction, worker_preferences),
        };
    }
    
    /// SIMD analysis using advanced fingerprinting
    fn analyzeSIMD(
        self: *ContinuationAnalysisPipeline, 
        cont: *continuation.Continuation, 
        advanced_fingerprint: fingerprint.TaskFingerprint
    ) SIMDClassification {
        _ = self;
        
        // Use advanced fingerprint characteristics for SIMD analysis
        const vectorization_potential = @as(f32, @floatFromInt(advanced_fingerprint.vectorization_benefit)) / 15.0; // Normalize to 0-1
        const simd_suitability = @as(f32, @floatFromInt(advanced_fingerprint.simd_width)) / 64.0; // Normalize to 0-1
        
        // Factor in continuation-specific characteristics
        const frame_size_factor = @min(1.0, 512.0 / @as(f32, @floatFromInt(cont.frame_size))); // Smaller frames better for SIMD
        const locality_factor = cont.locality_score;
        
        const overall_suitability = (vectorization_potential * 0.4 + simd_suitability * 0.4 + frame_size_factor * 0.1 + locality_factor * 0.1);
        
        return SIMDClassification{
            .suitability_score = overall_suitability,
            .vectorization_potential = vectorization_potential * 8.0, // Scale to expected speedup
            .preferred_numa_node = cont.numa_node,
            .should_batch = overall_suitability > 0.6,
            .analysis_source = .unified_fingerprint,
        };
    }
    
    /// Execution time prediction using One Euro Filter
    fn predictExecution(
        self: *ContinuationAnalysisPipeline,
        cont: *continuation.Continuation,
        advanced_fingerprint: fingerprint.TaskFingerprint,
        simd_classification: SIMDClassification
    ) !ExecutionPrediction {
        const fingerprint_hash = advanced_fingerprint.call_site_hash;
        
        // Get or create execution profile
        const profile = self.execution_history.get(fingerprint_hash) orelse ExecutionProfile.init();
        
        var predicted_time_ns: u64 = 1000000; // 1ms default
        var confidence: f32 = 0.0;
        var prediction_source: ExecutionPrediction.PredictionSource = .default;
        
        // Use historical data with One Euro Filter if available
        if (profile.sample_count > 0) {
            const current_time = @as(u64, @intCast(std.time.nanoTimestamp()));
            
            // Apply One Euro Filter
            const filtered_time = self.one_euro_filter.filter(
                @as(f32, @floatFromInt(profile.average_execution_time_ns)),
                current_time
            );
            
            // Apply SIMD speedup factor
            const simd_factor = if (simd_classification.should_batch) 
                1.0 / @max(1.0, simd_classification.vectorization_potential)
            else 
                1.0;
            
            predicted_time_ns = @as(u64, @intFromFloat(@max(1000, filtered_time * simd_factor)));
            confidence = self.calculatePredictionConfidence(profile, simd_classification);
            prediction_source = .historical_filtered;
        }
        
        // Determine NUMA preference based on prediction and continuation characteristics
        const numa_preference = if (predicted_time_ns > 10_000_000) // > 10ms
            cont.original_numa_node orelse cont.numa_node
        else
            cont.numa_node;
        
        return ExecutionPrediction{
            .predicted_time_ns = predicted_time_ns,
            .confidence = confidence,
            .numa_preference = numa_preference,
            .should_batch = simd_classification.should_batch,
            .prediction_source = prediction_source,
        };
    }
    
    /// Analyze worker preferences using coordinated information
    fn analyzeWorkerPreferences(
        self: *ContinuationAnalysisPipeline,
        cont: *continuation.Continuation,
        simd_classification: SIMDClassification,
        execution_prediction: ExecutionPrediction
    ) WorkerPreferences {
        _ = self;
        
        // Multi-criteria worker scoring
        const load_balance_weight: f32 = 0.25;
        const prediction_weight: f32 = 0.25;
        const topology_weight: f32 = 0.25;
        const simd_weight: f32 = 0.25;
        
        return WorkerPreferences{
            .load_balance_weight = load_balance_weight,
            .prediction_weight = prediction_weight,
            .topology_weight = topology_weight,
            .simd_weight = simd_weight,
            .preferred_numa_node = execution_prediction.numa_preference,
            .requires_simd_capability = simd_classification.should_batch,
            .locality_bonus_factor = cont.locality_score * 0.2,
        };
    }
    
    /// Coordinate NUMA decisions to eliminate conflicts
    fn coordinateNUMA(
        self: *ContinuationAnalysisPipeline,
        cont: *continuation.Continuation,
        simd_classification: SIMDClassification,
        execution_prediction: ExecutionPrediction,
        worker_preferences: WorkerPreferences
    ) NumaCoordination {
        _ = self;
        
        // Priority order: execution prediction > SIMD preference > worker preference > current
        const final_numa_node = execution_prediction.numa_preference orelse
                                simd_classification.preferred_numa_node orelse
                                worker_preferences.preferred_numa_node orelse
                                cont.numa_node;
        
        return NumaCoordination{
            .final_numa_node = final_numa_node,
            .decision_source = if (execution_prediction.numa_preference != null) .execution_prediction
                              else if (simd_classification.preferred_numa_node != null) .simd_analysis
                              else if (worker_preferences.preferred_numa_node != null) .worker_preference
                              else .current_assignment,
            .confidence = @max(execution_prediction.confidence, simd_classification.suitability_score * 0.8),
        };
    }
    
    /// Calculate prediction confidence
    fn calculatePredictionConfidence(
        self: *ContinuationAnalysisPipeline,
        profile: ExecutionProfile,
        simd_classification: SIMDClassification
    ) f32 {
        _ = self;
        
        // Base confidence from execution history
        const sample_confidence = 1.0 - @exp(-@as(f32, @floatFromInt(profile.sample_count)) / 10.0);
        const accuracy_confidence = profile.accuracy_rate;
        
        // SIMD confidence bonus
        const simd_confidence_bonus = if (simd_classification.should_batch) 
            simd_classification.suitability_score * 0.2 
        else 
            0.0;
        
        return @min(1.0, (sample_confidence * 0.5 + accuracy_confidence * 0.5) + simd_confidence_bonus);
    }
    
    /// Calculate overall analysis confidence
    fn calculateOverallConfidence(
        self: *ContinuationAnalysisPipeline,
        simd_classification: SIMDClassification,
        execution_prediction: ExecutionPrediction,
        worker_preferences: WorkerPreferences
    ) f32 {
        _ = self;
        _ = worker_preferences;
        
        // Weighted combination of individual confidences
        return (simd_classification.suitability_score * 0.4 + execution_prediction.confidence * 0.6);
    }
    
    /// Update prediction accuracy with actual results
    fn updatePredictionAccuracy(
        self: *ContinuationAnalysisPipeline,
        fingerprint_hash: u64,
        actual_execution_time_ns: u64
    ) !void {
        var profile = self.execution_history.get(fingerprint_hash) orelse ExecutionProfile.init();
        profile.updateWithExecution(actual_execution_time_ns);
        
        // Update One Euro Filter with actual execution time
        const current_time = @as(u64, @intCast(std.time.nanoTimestamp()));
        _ = self.one_euro_filter.filter(@as(f32, @floatFromInt(actual_execution_time_ns)), current_time);
        
        try self.execution_history.put(fingerprint_hash, profile);
    }
};

/// Decision coordinator that eliminates conflicts between different systems
const ContinuationDecisionCoordinator = struct {
    allocator: std.mem.Allocator,
    
    // Worker feedback tracking
    worker_feedback: std.AutoHashMap(u64, WorkerFeedback),
    
    fn init(allocator: std.mem.Allocator, config: DecisionConfig) !ContinuationDecisionCoordinator {
        _ = config;
        return ContinuationDecisionCoordinator{
            .allocator = allocator,
            .worker_feedback = std.AutoHashMap(u64, WorkerFeedback).init(allocator),
        };
    }
    
    fn deinit(self: *ContinuationDecisionCoordinator) void {
        self.worker_feedback.deinit();
    }
    
    /// Coordinate all decisions to eliminate conflicts
    fn coordinate(
        self: *ContinuationDecisionCoordinator,
        analysis: ContinuationAnalysis,
        cont: *continuation.Continuation
    ) !ContinuationAnalysis {
        _ = self;
        _ = cont;
        
        // For now, return analysis as-is
        // Future: Add sophisticated conflict resolution
        return analysis;
    }
    
    /// Update worker selection feedback
    fn updateWorkerFeedback(
        self: *ContinuationDecisionCoordinator,
        fingerprint_hash: u64,
        worker_id: u32,
        actual_execution_time_ns: u64
    ) !void {
        var feedback = self.worker_feedback.get(fingerprint_hash) orelse WorkerFeedback.init();
        feedback.updateFeedback(worker_id, actual_execution_time_ns);
        try self.worker_feedback.put(fingerprint_hash, feedback);
    }
};

/// Unified performance tracking system
const UnifiedPerformanceTracker = struct {
    cache_hits: u64 = 0,
    cache_misses: u64 = 0,
    total_analyses: u64 = 0,
    total_updates: u64 = 0,
    total_analysis_time_ns: u64 = 0,
    
    fn init() UnifiedPerformanceTracker {
        return UnifiedPerformanceTracker{};
    }
    
    fn recordCacheHit(self: *UnifiedPerformanceTracker) void {
        self.cache_hits += 1;
    }
    
    fn recordAnalysis(self: *UnifiedPerformanceTracker, analysis_time_ns: u64) void {
        self.cache_misses += 1;
        self.total_analyses += 1;
        self.total_analysis_time_ns += analysis_time_ns;
    }
    
    fn recordUpdate(self: *UnifiedPerformanceTracker) void {
        self.total_updates += 1;
    }
    
    fn getStats(self: *UnifiedPerformanceTracker) UnifiedPerformanceStats {
        const total_requests = self.cache_hits + self.cache_misses;
        const cache_hit_rate = if (total_requests > 0) 
            @as(f32, @floatFromInt(self.cache_hits)) / @as(f32, @floatFromInt(total_requests))
        else 
            0.0;
        
        const avg_analysis_time_ns = if (self.total_analyses > 0)
            self.total_analysis_time_ns / self.total_analyses
        else
            0;
        
        return UnifiedPerformanceStats{
            .cache_hit_rate = cache_hit_rate,
            .total_analyses = self.total_analyses,
            .total_updates = self.total_updates,
            .average_analysis_time_ns = avg_analysis_time_ns,
        };
    }
};

// ============================================================================
// Data Structures
// ============================================================================

/// Comprehensive continuation analysis containing all metadata
pub const ContinuationAnalysis = struct {
    simd_classification: SIMDClassification,
    execution_prediction: ExecutionPrediction,
    worker_preferences: WorkerPreferences,
    numa_coordination: NumaCoordination,
    analysis_timestamp: u64,
    confidence_score: f32,
};

/// SIMD classification results
pub const SIMDClassification = struct {
    suitability_score: f32,                // 0.0-1.0 SIMD suitability
    vectorization_potential: f32,          // Expected speedup factor
    preferred_numa_node: ?u32,             // NUMA preference for SIMD
    should_batch: bool,                    // Whether to batch with other tasks
    analysis_source: AnalysisSource,       // Source of analysis
    
    pub const AnalysisSource = enum {
        unified_fingerprint,
        legacy_compatibility,
        fallback_heuristic,
    };
    
    /// Check if continuation is suitable for SIMD processing
    pub fn isSIMDSuitable(self: SIMDClassification) bool {
        return self.suitability_score > 0.5 and self.vectorization_potential > 1.5;
    }
    
    /// Get expected speedup from vectorization
    pub fn getExpectedSpeedup(self: SIMDClassification) f32 {
        return if (self.should_batch) self.vectorization_potential else 1.0;
    }
};

/// Execution time prediction results
pub const ExecutionPrediction = struct {
    predicted_time_ns: u64,
    confidence: f32,
    numa_preference: ?u32,
    should_batch: bool,
    prediction_source: PredictionSource,
    
    pub const PredictionSource = enum {
        default,
        historical_filtered,
        simd_enhanced,
        numa_optimized,
    };
    
    /// Update prediction accuracy with actual results
    pub fn updateAccuracy(self: *ExecutionPrediction, actual_time_ns: u64) void {
        _ = self;
        _ = actual_time_ns;
        // Future: Update internal accuracy tracking
    }
};

/// Worker selection preferences
pub const WorkerPreferences = struct {
    load_balance_weight: f32,
    prediction_weight: f32,
    topology_weight: f32,
    simd_weight: f32,
    preferred_numa_node: ?u32,
    requires_simd_capability: bool,
    locality_bonus_factor: f32,
    
    /// Update preferences with feedback
    pub fn updateFeedback(self: *WorkerPreferences, worker_id: u32, execution_time_ns: u64) void {
        _ = self;
        _ = worker_id;
        _ = execution_time_ns;
        // Future: Update preferences based on feedback
    }
};

/// Coordinated NUMA placement decisions
pub const NumaCoordination = struct {
    final_numa_node: ?u32,
    decision_source: DecisionSource,
    confidence: f32,
    
    pub const DecisionSource = enum {
        execution_prediction,
        simd_analysis,
        worker_preference,
        current_assignment,
    };
};

/// Execution profile for prediction tracking
const ExecutionProfile = struct {
    sample_count: u64 = 0,
    average_execution_time_ns: u64 = 0,
    accuracy_rate: f32 = 0.0,
    
    fn init() ExecutionProfile {
        return ExecutionProfile{};
    }
    
    fn updateWithExecution(self: *ExecutionProfile, execution_time_ns: u64) void {
        if (self.sample_count == 0) {
            self.average_execution_time_ns = execution_time_ns;
        } else {
            // Exponential moving average
            const alpha = 0.1;
            self.average_execution_time_ns = @as(u64, @intFromFloat(
                alpha * @as(f64, @floatFromInt(execution_time_ns)) + 
                (1.0 - alpha) * @as(f64, @floatFromInt(self.average_execution_time_ns))
            ));
        }
        self.sample_count += 1;
    }
};

/// Worker feedback tracking
const WorkerFeedback = struct {
    worker_id: u32 = 0,
    execution_count: u32 = 0,
    average_execution_time_ns: u64 = 0,
    
    fn init() WorkerFeedback {
        return WorkerFeedback{};
    }
    
    fn updateFeedback(self: *WorkerFeedback, worker_id: u32, execution_time_ns: u64) void {
        self.worker_id = worker_id;
        self.execution_count += 1;
        
        if (self.execution_count == 1) {
            self.average_execution_time_ns = execution_time_ns;
        } else {
            // Exponential moving average
            const alpha = 0.2;
            self.average_execution_time_ns = @as(u64, @intFromFloat(
                alpha * @as(f64, @floatFromInt(execution_time_ns)) + 
                (1.0 - alpha) * @as(f64, @floatFromInt(self.average_execution_time_ns))
            ));
        }
    }
};

// ============================================================================
// Configuration
// ============================================================================

/// Unified configuration for all continuation management
pub const UnifiedConfig = struct {
    cache_config: CacheConfig = CacheConfig{},
    analysis_config: AnalysisConfig = AnalysisConfig{},
    decision_config: DecisionConfig = DecisionConfig{},
    
    /// Create performance-optimized configuration
    pub fn performanceOptimized() UnifiedConfig {
        return UnifiedConfig{
            .cache_config = CacheConfig{ .max_entries = 2048 },
            .analysis_config = AnalysisConfig{
                .min_cutoff = 0.05,
                .beta = 0.1,
            },
        };
    }
    
    /// Create balanced configuration
    pub fn balanced() UnifiedConfig {
        return UnifiedConfig{};
    }
};

/// Cache configuration
pub const CacheConfig = struct {
    max_entries: usize = 1024,
    eviction_threshold: f32 = 0.8,
};

/// Analysis pipeline configuration
pub const AnalysisConfig = struct {
    min_cutoff: f32 = 0.1,
    beta: f32 = 0.05,
    d_cutoff: f32 = 1.0,
    velocity_min_cutoff: f32 = 0.05,
    velocity_beta: f32 = 0.01,
    velocity_d_cutoff: f32 = 0.5,
};

/// Decision coordination configuration
pub const DecisionConfig = struct {
    enable_conflict_resolution: bool = true,
    numa_priority_weight: f32 = 0.6,
};

/// Unified performance statistics
pub const UnifiedPerformanceStats = struct {
    cache_hit_rate: f32,
    total_analyses: u64,
    total_updates: u64,
    average_analysis_time_ns: u64,
};

// ============================================================================
// Tests
// ============================================================================

test "unified continuation manager initialization" {
    const allocator = std.testing.allocator;
    
    // Create mock fingerprint registry
    var fingerprint_registry = fingerprint.FingerprintRegistry.init(allocator);
    defer fingerprint_registry.deinit();
    
    const config = UnifiedConfig.balanced();
    var manager = try UnifiedContinuationManager.init(allocator, &fingerprint_registry, config);
    defer manager.deinit();
    
    // Verify initialization
    const stats = manager.getPerformanceStats();
    try std.testing.expect(stats.total_analyses == 0);
    try std.testing.expect(stats.cache_hit_rate == 0.0);
    
    std.debug.print("✅ Unified continuation manager initialization test passed!\n", .{});
}

test "unified cache performance" {
    const allocator = std.testing.allocator;
    
    const cache_config = CacheConfig{ .max_entries = 64 };
    var cache = try UnifiedContinuationCache.init(allocator, cache_config);
    defer cache.deinit();
    
    // Create test fingerprint
    const test_fingerprint = fingerprint.TaskFingerprint{
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
    
    // Create test analysis
    const test_analysis = ContinuationAnalysis{
        .simd_classification = SIMDClassification{
            .suitability_score = 0.8,
            .vectorization_potential = 4.0,
            .preferred_numa_node = 0,
            .should_batch = true,
            .analysis_source = .unified_fingerprint,
        },
        .execution_prediction = ExecutionPrediction{
            .predicted_time_ns = 2000000,
            .confidence = 0.7,
            .numa_preference = 0,
            .should_batch = true,
            .prediction_source = .historical_filtered,
        },
        .worker_preferences = WorkerPreferences{
            .load_balance_weight = 0.25,
            .prediction_weight = 0.25,
            .topology_weight = 0.25,
            .simd_weight = 0.25,
            .preferred_numa_node = 0,
            .requires_simd_capability = true,
            .locality_bonus_factor = 0.2,
        },
        .numa_coordination = NumaCoordination{
            .final_numa_node = 0,
            .decision_source = .execution_prediction,
            .confidence = 0.8,
        },
        .analysis_timestamp = @as(u64, @intCast(std.time.nanoTimestamp())),
        .confidence_score = 0.75,
    };
    
    // Test cache miss
    const missed_result = cache.get(test_fingerprint);
    try std.testing.expect(missed_result == null);
    
    // Test cache put and hit
    try cache.put(test_fingerprint, test_analysis);
    const hit_result = cache.get(test_fingerprint);
    try std.testing.expect(hit_result != null);
    try std.testing.expect(hit_result.?.confidence_score == 0.75);
    
    std.debug.print("✅ Unified cache performance test passed!\n", .{});
}

test "comprehensive analysis pipeline" {
    const allocator = std.testing.allocator;
    
    const analysis_config = AnalysisConfig{};
    var pipeline = try ContinuationAnalysisPipeline.init(allocator, analysis_config);
    defer pipeline.deinit();
    
    // Create test continuation
    const TestData = struct { values: [32]f32 };
    var test_data = TestData{ .values = undefined };
    
    const resume_fn = struct {
        fn executeFunc(cont: *continuation.Continuation) void {
            _ = cont;
        }
    };
    
    var test_continuation = continuation.Continuation.capture(resume_fn.executeFunc, &test_data, allocator);
    test_continuation.numa_node = 1;
    test_continuation.locality_score = 0.8;
    
    // Create test fingerprint
    const test_fingerprint = fingerprint.TaskFingerprint{
        .call_site_hash = 0x87654321,
        .data_size_class = 15,
        .data_alignment = 3,
        .access_pattern = .sequential,
        .simd_width = 15,
        .cache_locality = 14,
        .numa_node_hint = 1,
        .cpu_intensity = 12,
        .parallel_potential = 14,
        .execution_phase = 2,
        .priority_class = 1,
        .time_sensitivity = 2,
        .dependency_count = 1,
        .time_of_day_bucket = 10,
        .execution_frequency = 7,
        .seasonal_pattern = 2,
        .variance_level = 3,
        .expected_cycles_log2 = 14,
        .memory_footprint_log2 = 10,
        .io_intensity = 1,
        .branch_predictability = 8,
        .vectorization_benefit = 12,
        .cache_miss_rate = 2,
    };
    
    // Test comprehensive analysis
    const analysis = try pipeline.analyzeComprehensive(&test_continuation, test_fingerprint);
    
    // Verify results
    try std.testing.expect(analysis.simd_classification.suitability_score >= 0.0);
    try std.testing.expect(analysis.simd_classification.suitability_score <= 1.0);
    try std.testing.expect(analysis.execution_prediction.predicted_time_ns > 0);
    try std.testing.expect(analysis.confidence_score >= 0.0);
    try std.testing.expect(analysis.confidence_score <= 1.0);
    
    std.debug.print("✅ Comprehensive analysis pipeline test passed!\n", .{});
    std.debug.print("   SIMD suitability: {d:.3}\n", .{analysis.simd_classification.suitability_score});
    std.debug.print("   Prediction confidence: {d:.3}\n", .{analysis.execution_prediction.confidence});
    std.debug.print("   Overall confidence: {d:.3}\n", .{analysis.confidence_score});
}