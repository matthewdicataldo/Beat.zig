const std = @import("std");
const core = @import("core.zig");
const continuation = @import("continuation.zig");
const continuation_simd = @import("continuation_simd.zig");
const scheduler = @import("scheduler.zig");
const fingerprint = @import("fingerprint.zig");
const predictive_accounting = @import("predictive_accounting.zig");
const intelligent_decision = @import("intelligent_decision.zig");

// ============================================================================
// Predictive Accounting Integration for Continuation Stealing
// ============================================================================

/// Enhanced continuation processing with intelligent execution time prediction
/// Integrates One Euro Filter for adaptive prediction with SIMD-enhanced classification
pub const ContinuationPredictiveAccounting = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    
    // Core prediction infrastructure
    one_euro_filter: scheduler.OneEuroFilter,
    velocity_filter: scheduler.OneEuroFilter,
    
    // Continuation-specific execution tracking
    execution_history: std.AutoHashMap(u64, ExecutionProfile),
    prediction_cache: PredictionCache,
    
    // Performance metrics
    total_predictions: u64,
    accurate_predictions: u64,
    cache_hits: u64,
    
    // Configuration
    prediction_confidence_threshold: f32,
    adaptive_numa_enabled: bool,
    
    /// Initialize predictive accounting for continuations
    pub fn init(allocator: std.mem.Allocator, config: PredictiveConfig) !Self {
        return Self{
            .allocator = allocator,
            .one_euro_filter = scheduler.OneEuroFilter.init(
                config.min_cutoff,
                config.beta,
                config.d_cutoff
            ),
            .velocity_filter = scheduler.OneEuroFilter.init(
                config.velocity_min_cutoff,
                config.velocity_beta,
                config.velocity_d_cutoff
            ),
            .execution_history = std.AutoHashMap(u64, ExecutionProfile).init(allocator),
            .prediction_cache = try PredictionCache.init(allocator),
            .total_predictions = 0,
            .accurate_predictions = 0,
            .cache_hits = 0,
            .prediction_confidence_threshold = config.confidence_threshold,
            .adaptive_numa_enabled = config.enable_adaptive_numa,
        };
    }
    
    /// Clean up resources
    pub fn deinit(self: *Self) void {
        self.execution_history.deinit();
        self.prediction_cache.deinit();
    }
    
    /// Predict execution time for continuation with SIMD analysis integration
    pub fn predictExecutionTime(self: *Self, cont: *continuation.Continuation, simd_class: ?continuation_simd.ContinuationSIMDClass) !PredictionResult {
        self.total_predictions += 1;
        
        // Generate fingerprint for continuation
        const fingerprint_hash = cont.fingerprint_hash orelse 
            self.generateContinuationFingerprint(cont);
        
        // Check prediction cache first
        if (self.prediction_cache.get(fingerprint_hash)) |cached_prediction| {
            self.cache_hits += 1;
            return self.enhancePredictionWithSIMD(cached_prediction, simd_class);
        }
        
        // Get or create execution profile
        var profile = self.execution_history.get(fingerprint_hash) orelse ExecutionProfile.init();
        
        var prediction = PredictionResult{
            .predicted_time_ns = 1000000, // 1ms default
            .confidence = 0.0,
            .numa_preference = cont.numa_node,
            .should_batch = false,
            .prediction_source = .default,
        };
        
        // If we have historical data, use One Euro Filter
        if (profile.sample_count > 0) {
            const current_time = @as(u64, @intCast(std.time.nanoTimestamp()));
            
            // Apply One Euro Filter to execution time
            const filtered_time = self.one_euro_filter.filter(
                @as(f32, @floatFromInt(profile.average_execution_time_ns)),
                current_time
            );
            
            // Apply velocity estimation
            const velocity = self.calculateVelocity(&profile, current_time);
            const velocity_filtered = self.velocity_filter.filter(velocity, current_time);
            
            // Combine filtered time with velocity for prediction
            prediction.predicted_time_ns = @as(u64, @intFromFloat(@max(1000, filtered_time + velocity_filtered * 1000000))); // Add 1ms for velocity
            prediction.confidence = self.calculateConfidence(&profile);
            prediction.prediction_source = .historical_filtered;
        }
        
        // Enhance prediction with SIMD analysis if available
        if (simd_class) |simd| {
            prediction = self.enhancePredictionWithSIMD(prediction, simd);
        }
        
        // Determine NUMA placement strategy based on prediction
        if (self.adaptive_numa_enabled) {
            prediction.numa_preference = self.determineOptimalNumaNode(cont, &prediction);
        }
        
        // Cache the prediction for future use
        try self.prediction_cache.put(fingerprint_hash, prediction);
        
        return prediction;
    }
    
    /// Update prediction accuracy when continuation completes
    pub fn updatePrediction(self: *Self, cont: *continuation.Continuation, actual_time_ns: u64) !void {
        const fingerprint_hash = cont.fingerprint_hash orelse return;
        
        // Get or create execution profile
        var profile = self.execution_history.get(fingerprint_hash) orelse ExecutionProfile.init();
        
        // Update execution profile with actual time
        profile.updateWithExecution(actual_time_ns);
        
        // Check prediction accuracy if we had a cached prediction
        if (self.prediction_cache.get(fingerprint_hash)) |cached_prediction| {
            const prediction_error = if (cached_prediction.predicted_time_ns > actual_time_ns)
                cached_prediction.predicted_time_ns - actual_time_ns
            else
                actual_time_ns - cached_prediction.predicted_time_ns;
                
            const relative_error = @as(f32, @floatFromInt(prediction_error)) / @as(f32, @floatFromInt(actual_time_ns));
            const was_accurate = relative_error < 0.3; // Within 30% considered accurate
            
            // Update accuracy tracking
            profile.updateAccuracy(was_accurate);
            
            // Update cache quality score for hybrid LFU+age algorithm
            self.prediction_cache.updateQualityScore(fingerprint_hash, was_accurate);
            
            if (was_accurate) {
                self.accurate_predictions += 1;
            }
        }
        
        // Store updated profile
        try self.execution_history.put(fingerprint_hash, profile);
        
        // Update One Euro Filter with actual execution time
        const current_time = @as(u64, @intCast(std.time.nanoTimestamp()));
        _ = self.one_euro_filter.filter(@as(f32, @floatFromInt(actual_time_ns)), current_time);
        
        // Update velocity filter
        const velocity = self.calculateVelocityFromActual(&profile, actual_time_ns, current_time);
        _ = self.velocity_filter.filter(velocity, current_time);
    }
    
    /// Get performance statistics with enhanced cache metrics
    pub fn getPerformanceStats(self: *Self) PredictiveAccountingStats {
        const accuracy_rate = if (self.total_predictions > 0)
            @as(f32, @floatFromInt(self.accurate_predictions)) / @as(f32, @floatFromInt(self.total_predictions))
        else
            0.0;
            
        const cache_hit_rate = if (self.total_predictions > 0)
            @as(f32, @floatFromInt(self.cache_hits)) / @as(f32, @floatFromInt(self.total_predictions))
        else
            0.0;
        
        return PredictiveAccountingStats{
            .total_predictions = self.total_predictions,
            .accurate_predictions = self.accurate_predictions,
            .accuracy_rate = accuracy_rate,
            .cache_hit_rate = cache_hit_rate,
            .profiles_tracked = self.execution_history.count(),
            .current_confidence = self.one_euro_filter.getCurrentEstimate() orelse 0.0,
        };
    }
    
    /// Get detailed cache performance statistics for hybrid LFU+age algorithm
    pub fn getCacheStats(self: *const Self) PredictionCache.CacheStats {
        return self.prediction_cache.getStats();
    }
    
    /// Generate fingerprint for continuation caching
    fn generateContinuationFingerprint(self: *Self, cont: *continuation.Continuation) u64 {
        _ = self; // Allocator not needed for simple hash
        
        // Use function pointer, data characteristics, and frame size for fingerprinting
        const func_addr = @intFromPtr(cont.resume_fn);
        const data_addr = @intFromPtr(cont.data);
        const frame_size = cont.frame_size;
        const numa_node = cont.numa_node orelse 0;
        
        // Combine characteristics into hash
        const hash = func_addr ^ data_addr ^ frame_size ^ numa_node;
        return hash;
    }
    
    /// Enhance prediction using SIMD classification results
    fn enhancePredictionWithSIMD(self: *Self, base_prediction: PredictionResult, simd_class: ?continuation_simd.ContinuationSIMDClass) PredictionResult {
        _ = self;
        var enhanced = base_prediction;
        
        if (simd_class) |simd| {
            // Adjust prediction based on SIMD characteristics
            const simd_speedup_factor = simd.getExpectedSpeedup();
            enhanced.predicted_time_ns = @as(u64, @intFromFloat(@as(f32, @floatFromInt(enhanced.predicted_time_ns)) / simd_speedup_factor));
            
            // Increase confidence if SIMD is highly suitable
            if (simd.simd_suitability_score > 0.7) {
                enhanced.confidence = @min(1.0, enhanced.confidence + 0.2);
                enhanced.should_batch = simd.isSIMDSuitable();
            }
            
            // Prefer NUMA node for SIMD-optimized continuations
            if (simd.preferred_numa_node) |numa| {
                enhanced.numa_preference = numa;
            }
            
            enhanced.prediction_source = .simd_enhanced;
        }
        
        return enhanced;
    }
    
    /// Calculate velocity of execution time changes
    fn calculateVelocity(self: *Self, profile: *const ExecutionProfile, current_time_ns: u64) f32 {
        _ = self;
        if (profile.sample_count < 2) return 0.0;
        
        // Simple velocity calculation based on recent trend
        const time_delta = @as(f32, @floatFromInt(current_time_ns - profile.last_execution_time_ns));
        if (time_delta <= 0) return 0.0;
        
        const execution_delta = @as(f32, @floatFromInt(profile.average_execution_time_ns)) - @as(f32, @floatFromInt(profile.previous_execution_time_ns));
        return execution_delta / (time_delta / 1_000_000_000.0); // ns per second
    }
    
    /// Calculate velocity from actual execution
    fn calculateVelocityFromActual(self: *Self, profile: *const ExecutionProfile, actual_time_ns: u64, current_time_ns: u64) f32 {
        _ = self;
        if (profile.sample_count == 0) return 0.0;
        
        const time_delta = @as(f32, @floatFromInt(current_time_ns - profile.last_execution_time_ns));
        if (time_delta <= 0) return 0.0;
        
        const execution_delta = @as(f32, @floatFromInt(actual_time_ns)) - @as(f32, @floatFromInt(profile.average_execution_time_ns));
        return execution_delta / (time_delta / 1_000_000_000.0);
    }
    
    /// Calculate prediction confidence based on execution profile
    fn calculateConfidence(self: *Self, profile: *const ExecutionProfile) f32 {
        _ = self;
        
        // Multi-factor confidence calculation
        const sample_confidence = 1.0 - @exp(-@as(f32, @floatFromInt(profile.sample_count)) / 10.0);
        const accuracy_confidence = profile.accuracy_rate;
        const stability_confidence = 1.0 / (1.0 + profile.variance_coefficient);
        
        // Weighted geometric mean for overall confidence
        const weights = [3]f32{ 0.4, 0.4, 0.2 }; // Sample, accuracy, stability
        const values = [3]f32{ sample_confidence, accuracy_confidence, stability_confidence };
        
        var weighted_product: f32 = 1.0;
        var weight_sum: f32 = 0.0;
        
        for (values, weights) |value, weight| {
            if (value > 0) {
                weighted_product *= std.math.pow(f32, value, weight);
                weight_sum += weight;
            }
        }
        
        return if (weight_sum > 0) weighted_product else 0.0;
    }
    
    /// Determine optimal NUMA node based on prediction and continuation characteristics
    fn determineOptimalNumaNode(self: *Self, cont: *continuation.Continuation, prediction: *const PredictionResult) ?u32 {
        _ = self;
        
        // For long-running continuations (>10ms), prefer specific NUMA node
        if (prediction.predicted_time_ns > 10_000_000) {
            return cont.original_numa_node orelse cont.numa_node;
        }
        
        // For short continuations, current NUMA node is fine
        return cont.numa_node;
    }
};

/// Configuration for continuation predictive accounting
pub const PredictiveConfig = struct {
    // One Euro Filter parameters
    min_cutoff: f32 = 0.1,
    beta: f32 = 0.05,
    d_cutoff: f32 = 1.0,
    
    // Velocity filter parameters (more stable)
    velocity_min_cutoff: f32 = 0.05,
    velocity_beta: f32 = 0.01,
    velocity_d_cutoff: f32 = 0.5,
    
    // Prediction parameters
    confidence_threshold: f32 = 0.5,
    enable_adaptive_numa: bool = true,
    
    /// Create balanced configuration for general use
    pub fn balanced() PredictiveConfig {
        return PredictiveConfig{};
    }
    
    /// Create performance-optimized configuration
    pub fn performanceOptimized() PredictiveConfig {
        return PredictiveConfig{
            .min_cutoff = 0.05,
            .beta = 0.1,
            .confidence_threshold = 0.3,
            .enable_adaptive_numa = true,
        };
    }
};

/// Execution profile for continuation prediction
const ExecutionProfile = struct {
    sample_count: u64,
    average_execution_time_ns: u64,
    previous_execution_time_ns: u64,
    last_execution_time_ns: u64,
    variance_coefficient: f32,
    accuracy_rate: f32,
    accurate_count: u64,
    total_count: u64,
    
    fn init() ExecutionProfile {
        return ExecutionProfile{
            .sample_count = 0,
            .average_execution_time_ns = 0,
            .previous_execution_time_ns = 0,
            .last_execution_time_ns = 0,
            .variance_coefficient = 0.0,
            .accuracy_rate = 0.0,
            .accurate_count = 0,
            .total_count = 0,
        };
    }
    
    fn updateWithExecution(self: *ExecutionProfile, execution_time_ns: u64) void {
        self.previous_execution_time_ns = self.average_execution_time_ns;
        
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
        self.last_execution_time_ns = @as(u64, @intCast(std.time.nanoTimestamp()));
        
        // Update variance coefficient
        if (self.sample_count > 1) {
            const diff = @as(f32, @floatFromInt(execution_time_ns)) - @as(f32, @floatFromInt(self.average_execution_time_ns));
            const variance_update = diff * diff;
            self.variance_coefficient = self.variance_coefficient * 0.9 + variance_update * 0.1 / @as(f32, @floatFromInt(self.average_execution_time_ns));
        }
    }
    
    fn updateAccuracy(self: *ExecutionProfile, was_accurate: bool) void {
        self.total_count += 1;
        if (was_accurate) {
            self.accurate_count += 1;
        }
        self.accuracy_rate = @as(f32, @floatFromInt(self.accurate_count)) / @as(f32, @floatFromInt(self.total_count));
    }
};

/// Result of execution time prediction
pub const PredictionResult = struct {
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
};

/// High-performance prediction cache with hybrid LFU + age eviction policy
const PredictionCache = struct {
    const CacheEntry = struct {
        prediction: PredictionResult,
        timestamp: u64,         // Entry creation time
        last_access: u64,       // Last access time
        access_count: u32,      // Total access count (LFU component)
        access_frequency: f32,  // Weighted access frequency per time unit
        quality_score: f32,     // Prediction accuracy score for this entry
        
        /// Calculate hybrid eviction score (lower = more likely to evict)
        /// Combines LFU (frequency), recency (age), and prediction quality
        fn calculateEvictionScore(self: *const CacheEntry, current_time: u64) f32 {
            const age_seconds = @as(f32, @floatFromInt(current_time - self.timestamp)) / 1_000_000_000.0;
            const recency_seconds = @as(f32, @floatFromInt(current_time - self.last_access)) / 1_000_000_000.0;
            
            // Prevent division by zero for very new entries
            const safe_age = @max(0.001, age_seconds);
            const safe_recency = @max(0.001, recency_seconds);
            
            // Frequency score: higher access frequency = higher score
            const frequency_score = self.access_frequency / safe_age;
            
            // Recency score: more recent access = higher score
            const recency_score = 1.0 / safe_recency;
            
            // Quality score: better predictions = higher score
            const quality_score = self.quality_score;
            
            // Age penalty: very old entries get penalized exponentially
            const age_penalty = if (age_seconds > 30.0) std.math.exp(-(age_seconds - 30.0) / 10.0) else 1.0;
            
            // Weighted combination with tuned coefficients for prediction workloads
            const weights = struct {
                const frequency: f32 = 0.35;  // LFU component
                const recency: f32 = 0.25;    // LRU component  
                const quality: f32 = 0.30;    // Accuracy component
                const age_factor: f32 = 0.10;  // Age penalty factor
            };
            
            return (frequency_score * weights.frequency + 
                   recency_score * weights.recency + 
                   quality_score * weights.quality) * 
                   (age_penalty * weights.age_factor + (1.0 - weights.age_factor));
        }
        
        /// Update access statistics for this entry
        fn updateAccess(self: *CacheEntry, current_time: u64) void {
            const time_since_last = @as(f32, @floatFromInt(current_time - self.last_access)) / 1_000_000_000.0;
            
            // Update access tracking
            self.access_count += 1;
            self.last_access = current_time;
            
            // Update weighted frequency using exponential smoothing
            const alpha: f32 = 0.3; // Smoothing factor
            const instant_frequency = if (time_since_last > 0) 1.0 / time_since_last else 1.0;
            self.access_frequency = alpha * instant_frequency + (1.0 - alpha) * self.access_frequency;
        }
    };
    
    map: std.AutoHashMap(u64, CacheEntry),
    max_entries: usize,
    total_accesses: u64,      // Statistics for adaptive tuning
    total_evictions: u64,
    
    fn init(allocator: std.mem.Allocator) !PredictionCache {
        return PredictionCache{
            .map = std.AutoHashMap(u64, CacheEntry).init(allocator),
            .max_entries = 512, // Optimal size for predictions
            .total_accesses = 0,
            .total_evictions = 0,
        };
    }
    
    fn deinit(self: *PredictionCache) void {
        self.map.deinit();
    }
    
    pub fn get(self: *PredictionCache, hash: u64) ?PredictionResult {
        const current_time = @as(u64, @intCast(std.time.nanoTimestamp()));
        
        if (self.map.getPtr(hash)) |entry| {
            // Check if prediction is still fresh (adaptive freshness based on confidence)
            const max_age_ns: u64 = if (entry.prediction.confidence > 0.8)
                20_000_000_000  // 20 seconds for high-confidence predictions
            else
                5_000_000_000;  // 5 seconds for low-confidence predictions
                
            if (current_time - entry.timestamp < max_age_ns) {
                // Update access statistics for hybrid algorithm
                entry.updateAccess(current_time);
                self.total_accesses += 1;
                return entry.prediction;
            } else {
                // Remove stale entry immediately
                _ = self.map.remove(hash);
            }
        }
        return null;
    }
    
    fn put(self: *PredictionCache, hash: u64, prediction: PredictionResult) !void {
        const current_time = @as(u64, @intCast(std.time.nanoTimestamp()));
        
        // Evict entries if cache is full using hybrid LFU+age policy
        if (self.map.count() >= self.max_entries) {
            try self.evictHybridLFUAge(current_time);
        }
        
        // Initialize quality score based on prediction confidence and source
        const initial_quality = switch (prediction.prediction_source) {
            .historical_filtered => prediction.confidence * 0.9,
            .simd_enhanced => prediction.confidence * 1.1,
            .numa_optimized => prediction.confidence * 0.95,
            .default => prediction.confidence * 0.7,
        };
        
        const entry = CacheEntry{
            .prediction = prediction,
            .timestamp = current_time,
            .last_access = current_time,
            .access_count = 1,
            .access_frequency = 1.0,  // Initial frequency
            .quality_score = @min(1.0, @max(0.0, initial_quality)),
        };
        
        try self.map.put(hash, entry);
    }
    
    /// Hybrid LFU + Age eviction algorithm optimized for prediction workloads
    fn evictHybridLFUAge(self: *PredictionCache, current_time: u64) !void {
        // Adaptive eviction: remove 20-40% based on cache pressure
        const cache_pressure = @as(f32, @floatFromInt(self.map.count())) / @as(f32, @floatFromInt(self.max_entries));
        const eviction_rate: f32 = if (cache_pressure > 0.95) 0.4 else if (cache_pressure > 0.90) 0.3 else 0.2;
        const eviction_count = @max(1, @as(usize, @intFromFloat(@as(f32, @floatFromInt(self.max_entries)) * eviction_rate)));
        
        // Collect entries with their eviction scores
        const EntryScore = struct {
            hash: u64,
            score: f32,
        };
        
        var entries_scores = try std.ArrayList(EntryScore).initCapacity(
            self.map.allocator, self.map.count()
        );
        defer entries_scores.deinit();
        
        var iterator = self.map.iterator();
        while (iterator.next()) |entry| {
            const score = entry.value_ptr.calculateEvictionScore(current_time);
            entries_scores.appendAssumeCapacity(EntryScore{
                .hash = entry.key_ptr.*,
                .score = score,
            });
        }
        
        // Sort by eviction score (ascending - lowest scores evicted first)
        std.sort.pdq(EntryScore, entries_scores.items, {}, struct {
            fn lessThan(_: void, a: EntryScore, b: EntryScore) bool {
                return a.score < b.score;
            }
        }.lessThan);
        
        // Evict lowest-scoring entries
        const actual_eviction_count = @min(eviction_count, entries_scores.items.len);
        for (entries_scores.items[0..actual_eviction_count]) |entry_score| {
            _ = self.map.remove(entry_score.hash);
            self.total_evictions += 1;
        }
        
        // Optional: Emergency cleanup for very stale entries regardless of score
        if (cache_pressure > 0.98) {
            try self.emergencyStaleCleanup(current_time);
        }
    }
    
    /// Emergency cleanup to remove very stale entries when cache is critically full
    fn emergencyStaleCleanup(self: *PredictionCache, current_time: u64) !void {
        const emergency_age_threshold = 60_000_000_000; // 60 seconds
        
        var keys_to_remove = std.ArrayList(u64).init(self.map.allocator);
        defer keys_to_remove.deinit();
        
        var iterator = self.map.iterator();
        while (iterator.next()) |entry| {
            if (current_time - entry.value_ptr.timestamp > emergency_age_threshold) {
                try keys_to_remove.append(entry.key_ptr.*);
            }
        }
        
        for (keys_to_remove.items) |key| {
            _ = self.map.remove(key);
            self.total_evictions += 1;
        }
    }
    
    /// Update quality score for an entry based on prediction accuracy
    fn updateQualityScore(self: *PredictionCache, hash: u64, was_accurate: bool) void {
        if (self.map.getPtr(hash)) |entry| {
            const accuracy_delta: f32 = if (was_accurate) 0.1 else -0.1;
            const smoothing: f32 = 0.2;
            
            entry.quality_score = @min(1.0, @max(0.0, 
                entry.quality_score * (1.0 - smoothing) + accuracy_delta * smoothing
            ));
        }
    }
    
    /// Get cache performance statistics
    fn getStats(self: *const PredictionCache) CacheStats {
        const hit_rate = if (self.total_accesses > 0)
            1.0 - (@as(f32, @floatFromInt(self.total_evictions)) / @as(f32, @floatFromInt(self.total_accesses)))
        else
            0.0;
            
        var avg_quality: f32 = 0.0;
        var total_frequency: f32 = 0.0;
        
        var iterator = self.map.iterator();
        while (iterator.next()) |entry| {
            avg_quality += entry.value_ptr.quality_score;
            total_frequency += entry.value_ptr.access_frequency;
        }
        
        const entry_count = @as(f32, @floatFromInt(self.map.count()));
        if (entry_count > 0) {
            avg_quality /= entry_count;
            total_frequency /= entry_count;
        }
        
        return CacheStats{
            .entries = self.map.count(),
            .hit_rate = hit_rate,
            .avg_quality_score = avg_quality,
            .avg_access_frequency = total_frequency,
            .total_evictions = self.total_evictions,
        };
    }
    
    const CacheStats = struct {
        entries: usize,
        hit_rate: f32,
        avg_quality_score: f32,
        avg_access_frequency: f32,
        total_evictions: u64,
    };
};

/// Performance statistics for predictive accounting
pub const PredictiveAccountingStats = struct {
    total_predictions: u64,
    accurate_predictions: u64,
    accuracy_rate: f32,
    cache_hit_rate: f32,
    profiles_tracked: u32,
    current_confidence: f32,
};

// ============================================================================
// Tests
// ============================================================================

test "continuation predictive accounting initialization and basic functionality" {
    const allocator = std.testing.allocator;
    
    // Initialize predictive accounting
    const config = PredictiveConfig.performanceOptimized();
    var predictor = try ContinuationPredictiveAccounting.init(allocator, config);
    defer predictor.deinit();
    
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
    test_continuation.frame_size = 128;
    test_continuation.fingerprint_hash = 12345; // Mock fingerprint
    
    // Test initial prediction
    const prediction = try predictor.predictExecutionTime(&test_continuation, null);
    
    // Verify prediction structure
    try std.testing.expect(prediction.predicted_time_ns > 0);
    try std.testing.expect(prediction.confidence >= 0.0);
    try std.testing.expect(prediction.confidence <= 1.0);
    
    // Test update after execution
    const actual_time: u64 = 1500000; // 1.5ms
    try predictor.updatePrediction(&test_continuation, actual_time);
    
    // Test second prediction (should use historical data)
    const second_prediction = try predictor.predictExecutionTime(&test_continuation, null);
    try std.testing.expect(second_prediction.confidence > prediction.confidence);
    
    // Check statistics
    const stats = predictor.getPerformanceStats();
    try std.testing.expect(stats.total_predictions >= 2);
    try std.testing.expect(stats.profiles_tracked >= 1);
    
    std.debug.print("✅ Continuation predictive accounting basic test passed!\n", .{});
    std.debug.print("   Total predictions: {}\n", .{stats.total_predictions});
    std.debug.print("   Accuracy rate: {d:.3}\n", .{stats.accuracy_rate});
    std.debug.print("   Cache hit rate: {d:.3}\n", .{stats.cache_hit_rate});
}

test "predictive accounting with SIMD integration" {
    const allocator = std.testing.allocator;
    
    const config = PredictiveConfig.balanced();
    var predictor = try ContinuationPredictiveAccounting.init(allocator, config);
    defer predictor.deinit();
    
    // Create test continuation
    const TestData = struct { values: [64]f32 };
    var test_data = TestData{ .values = undefined };
    
    const resume_fn = struct {
        fn executeFunc(cont: *continuation.Continuation) void {
            const data = @as(*TestData, @ptrCast(@alignCast(cont.data)));
            for (&data.values) |*value| {
                value.* = @sqrt(value.* * value.* + 1.0); // Complex SIMD operation
            }
            cont.state = .completed;
        }
    };
    
    var test_continuation = continuation.Continuation.capture(resume_fn.executeFunc, &test_data, allocator);
    test_continuation.fingerprint_hash = 54321;
    
    // Create mock SIMD classification
    const simd_class = continuation_simd.ContinuationSIMDClass{
        .task_class = .highly_vectorizable,
        .simd_suitability_score = 0.9,
        .continuation_overhead_factor = 1.1,
        .vectorization_potential = 4.0, // 4x speedup
        .preferred_numa_node = 1,
    };
    
    // Test prediction with SIMD enhancement
    const prediction_with_simd = try predictor.predictExecutionTime(&test_continuation, simd_class);
    const prediction_without_simd = try predictor.predictExecutionTime(&test_continuation, null);
    
    // SIMD-enhanced prediction should be faster and more confident
    try std.testing.expect(prediction_with_simd.predicted_time_ns <= prediction_without_simd.predicted_time_ns);
    try std.testing.expect(prediction_with_simd.confidence >= prediction_without_simd.confidence);
    try std.testing.expect(prediction_with_simd.should_batch);
    try std.testing.expect(prediction_with_simd.numa_preference == 1);
    
    std.debug.print("✅ Predictive accounting with SIMD integration test passed!\n", .{});
    std.debug.print("   SIMD prediction time: {}μs\n", .{prediction_with_simd.predicted_time_ns / 1000});
    std.debug.print("   Base prediction time: {}μs\n", .{prediction_without_simd.predicted_time_ns / 1000});
    std.debug.print("   SIMD confidence: {d:.3}\n", .{prediction_with_simd.confidence});
}

test "prediction cache performance and accuracy tracking" {
    const allocator = std.testing.allocator;
    
    const config = PredictiveConfig.performanceOptimized();
    var predictor = try ContinuationPredictiveAccounting.init(allocator, config);
    defer predictor.deinit();
    
    // Create multiple similar continuations
    const TestData = struct { value: i32 };
    var test_data_array: [10]TestData = undefined;
    var continuations: [10]continuation.Continuation = undefined;
    
    const resume_fn = struct {
        fn executeFunc(cont: *continuation.Continuation) void {
            const data = @as(*TestData, @ptrCast(@alignCast(cont.data)));
            data.value *= 2; // Simple operation
            cont.state = .completed;
        }
    };
    
    // Initialize similar continuations
    for (&test_data_array, 0..) |*data, i| {
        data.* = TestData{ .value = @intCast(i) };
        continuations[i] = continuation.Continuation.capture(resume_fn.executeFunc, data, allocator);
        continuations[i].fingerprint_hash = 1000 + i; // Different fingerprints
    }
    
    // Test cache behavior
    for (&continuations) |*cont| {
        _ = try predictor.predictExecutionTime(cont, null);
    }
    
    // Second round should hit cache
    for (&continuations) |*cont| {
        _ = try predictor.predictExecutionTime(cont, null);
    }
    
    // Update with actual execution times
    for (&continuations, 0..) |*cont, i| {
        const actual_time: u64 = 500000 + i * 100000; // Variable execution times
        try predictor.updatePrediction(cont, actual_time);
    }
    
    // Check statistics
    const stats = predictor.getPerformanceStats();
    try std.testing.expect(stats.total_predictions >= 20);
    try std.testing.expect(stats.cache_hit_rate > 0.0);
    try std.testing.expect(stats.profiles_tracked >= 10);
    
    std.debug.print("✅ Prediction cache and accuracy tracking test passed!\n", .{});
    std.debug.print("   Cache hit rate: {d:.1}%\n", .{stats.cache_hit_rate * 100});
    std.debug.print("   Accuracy rate: {d:.1}%\n", .{stats.accuracy_rate * 100});
    std.debug.print("   Profiles tracked: {}\n", .{stats.profiles_tracked});
}

test "hybrid LFU + age eviction policy performance" {
    const allocator = std.testing.allocator;
    
    const config = PredictiveConfig.performanceOptimized();
    var predictor = try ContinuationPredictiveAccounting.init(allocator, config);
    defer predictor.deinit();
    
    // Override cache size for testing
    predictor.prediction_cache.max_entries = 10; // Small cache to force evictions
    
    const TestData = struct { value: u32 };
    
    const resume_fn = struct {
        fn executeFunc(cont: *continuation.Continuation) void {
            const data = @as(*TestData, @ptrCast(@alignCast(cont.data)));
            data.value += 1;
            cont.state = .completed;
        }
    };
    
    // Phase 1: Fill cache with initial entries
    var test_data_array: [15]TestData = undefined;
    var continuations: [15]continuation.Continuation = undefined;
    
    for (&test_data_array, 0..) |*data, i| {
        data.* = TestData{ .value = @intCast(i) };
        continuations[i] = continuation.Continuation.capture(resume_fn.executeFunc, data, allocator);
        continuations[i].fingerprint_hash = 2000 + i;
    }
    
    // Submit predictions to fill cache beyond capacity
    for (&continuations) |*cont| {
        _ = try predictor.predictExecutionTime(cont, null);
    }
    
    const initial_cache_stats = predictor.getCacheStats();
    try std.testing.expect(initial_cache_stats.entries <= 10); // Cache should be capped
    
    // Phase 2: Create access patterns to test hybrid algorithm
    // Access some entries frequently (high LFU score)
    for (0..5) |_| {
        for (0..3) |i| { // Access first 3 entries multiple times
            _ = try predictor.predictExecutionTime(&continuations[i], null);
        }
        std.time.sleep(100_000_000); // 100ms between access rounds
    }
    
    // Phase 3: Update some predictions with high accuracy (high quality score)
    for (0..3) |i| {
        // Simulate accurate predictions
        const predicted = try predictor.predictExecutionTime(&continuations[i], null);
        const accurate_time = predicted.predicted_time_ns + 100_000; // Within 30% accuracy
        try predictor.updatePrediction(&continuations[i], accurate_time);
    }
    
    // Update some with poor accuracy (low quality score)
    for (3..6) |i| {
        const predicted = try predictor.predictExecutionTime(&continuations[i], null);
        const inaccurate_time = predicted.predicted_time_ns * 3; // Poor accuracy
        try predictor.updatePrediction(&continuations[i], inaccurate_time);
    }
    
    // Phase 4: Force more evictions by adding new entries
    var new_test_data: [10]TestData = undefined;
    var new_continuations: [10]continuation.Continuation = undefined;
    
    for (&new_test_data, 0..) |*data, i| {
        data.* = TestData{ .value = @intCast(i + 100) };
        new_continuations[i] = continuation.Continuation.capture(resume_fn.executeFunc, data, allocator);
        new_continuations[i].fingerprint_hash = 3000 + i;
    }
    
    // Add new entries to force evictions
    for (&new_continuations) |*cont| {
        _ = try predictor.predictExecutionTime(cont, null);
    }
    
    const final_cache_stats = predictor.getCacheStats();
    
    // Verify hybrid algorithm behavior
    try std.testing.expect(final_cache_stats.entries <= 10); // Cache size limit respected
    try std.testing.expect(final_cache_stats.total_evictions > 0); // Evictions occurred
    try std.testing.expect(final_cache_stats.hit_rate > 0.0); // Some cache hits occurred
    
    // Verify frequently accessed and high-quality entries are more likely to be retained
    var high_value_entries_retained: u32 = 0;
    for (0..3) |i| { // Check first 3 entries (frequent + high quality)
        if (predictor.prediction_cache.get(continuations[i].fingerprint_hash.?)) |_| {
            high_value_entries_retained += 1;
        }
    }
    
    var low_value_entries_retained: u32 = 0;
    for (10..13) |i| { // Check less accessed entries
        if (predictor.prediction_cache.get(continuations[i].fingerprint_hash.?)) |_| {
            low_value_entries_retained += 1;
        }
    }
    
    // High-value entries should be more likely to be retained
    try std.testing.expect(high_value_entries_retained >= low_value_entries_retained);
    
    // Performance verification
    const overall_stats = predictor.getPerformanceStats();
    
    std.debug.print("✅ Hybrid LFU + age eviction policy test passed!\n", .{});
    std.debug.print("   Cache entries: {}/{}\n", .{ final_cache_stats.entries, 10 });
    std.debug.print("   Total evictions: {}\n", .{final_cache_stats.total_evictions});
    std.debug.print("   Hit rate: {d:.1}%\n", .{final_cache_stats.hit_rate * 100});
    std.debug.print("   Avg quality score: {d:.3}\n", .{final_cache_stats.avg_quality_score});
    std.debug.print("   Avg access frequency: {d:.3}\n", .{final_cache_stats.avg_access_frequency});
    std.debug.print("   High-value entries retained: {}/3\n", .{high_value_entries_retained});
    std.debug.print("   Low-value entries retained: {}/3\n", .{low_value_entries_retained});
    std.debug.print("   Overall accuracy: {d:.1}%\n", .{overall_stats.accuracy_rate * 100});
}