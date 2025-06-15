const std = @import("std");
const builtin = @import("builtin");
const core = @import("core.zig");
const scheduler = @import("scheduler.zig");
const fingerprint = @import("fingerprint.zig");
const intelligent_decision = @import("intelligent_decision.zig");

// Predictive Token Accounting for Beat.zig (Task 2.4.1)
//
// This module enhances the existing token accounting system with forward-looking
// intelligence by integrating execution time predictions and confidence-based
// promotion decisions. Instead of being purely reactive to completed work,
// the system becomes proactive by considering predicted future work.

// ============================================================================
// Enhanced Predictive Token Account
// ============================================================================

/// Enhanced token account that integrates execution time predictions
/// and confidence-based decision making for smarter promotion logic
pub const PredictiveTokenAccount = struct {
    // Base token accounting (maintains compatibility)
    base_account: scheduler.TokenAccount,
    
    // Predictive enhancements
    predicted_work_cycles: f64 = 0.0,           // Accumulated predicted work
    prediction_accuracy_score: f32 = 0.0,       // How accurate our predictions have been
    confidence_weighted_tokens: f64 = 0.0,      // Tokens weighted by prediction confidence
    
    // Adaptive promotion thresholds
    base_promotion_threshold: u64,               // Original static threshold
    adaptive_promotion_threshold: f64,           // Dynamic threshold based on confidence
    confidence_adjustment_factor: f32 = 1.0,    // Multiplier for threshold adjustment
    
    // Prediction tracking
    active_predictions: PredictionTracker,      // Track pending task predictions
    prediction_registry: ?*fingerprint.FingerprintRegistry = null,
    
    // Performance metrics
    total_predictions_made: u64 = 0,
    accurate_predictions: u64 = 0,
    promotion_decisions: PromotionMetrics,
    
    const Self = @This();
    
    /// Prediction tracking for active tasks
    pub const PredictionTracker = struct {
        predictions: [32]PendingPrediction = [_]PendingPrediction{.{}} ** 32,
        count: usize = 0,
        next_index: usize = 0,
        
        const PendingPrediction = struct {
            task_fingerprint_hash: u64 = 0,
            predicted_cycles: f64 = 0.0,
            confidence: f32 = 0.0,
            start_timestamp: u64 = 0,
            is_active: bool = false,
        };
        
        /// Add a new prediction to track
        pub fn addPrediction(self: *PredictionTracker, fingerprint_hash: u64, predicted_cycles: f64, confidence: f32) void {
            const index = self.next_index;
            self.predictions[index] = .{
                .task_fingerprint_hash = fingerprint_hash,
                .predicted_cycles = predicted_cycles,
                .confidence = confidence,
                .start_timestamp = @as(u64, @intCast(std.time.nanoTimestamp())),
                .is_active = true,
            };
            
            self.next_index = (self.next_index + 1) % 32;
            if (self.count < 32) self.count += 1;
        }
        
        /// Complete a prediction and return accuracy information
        pub fn completePrediction(self: *PredictionTracker, fingerprint_hash: u64, actual_cycles: u64) ?PredictionResult {
            for (&self.predictions) |*pred| {
                if (pred.is_active and pred.task_fingerprint_hash == fingerprint_hash) {
                    pred.is_active = false;
                    
                    const prediction_error = @abs(pred.predicted_cycles - @as(f64, @floatFromInt(actual_cycles)));
                    const relative_error = if (pred.predicted_cycles > 0.0) 
                        prediction_error / pred.predicted_cycles 
                    else 
                        1.0;
                        
                    return PredictionResult{
                        .predicted_cycles = pred.predicted_cycles,
                        .actual_cycles = actual_cycles,
                        .confidence = pred.confidence,
                        .relative_error = @as(f32, @floatCast(relative_error)),
                        .was_accurate = relative_error < 0.3, // Within 30% considered accurate
                    };
                }
            }
            return null;
        }
    };
    
    /// Result of comparing prediction to actual execution
    const PredictionResult = struct {
        predicted_cycles: f64,
        actual_cycles: u64,
        confidence: f32,
        relative_error: f32,
        was_accurate: bool,
    };
    
    /// Metrics for promotion decision analysis
    const PromotionMetrics = struct {
        confidence_based_promotions: u64 = 0,
        prediction_based_promotions: u64 = 0,
        traditional_promotions: u64 = 0,
        threshold_adjustments: u64 = 0,
    };
    
    /// Initialize predictive token account
    pub fn init(config: *const core.Config) Self {
        return Self{
            .base_account = scheduler.TokenAccount.init(config),
            .base_promotion_threshold = config.promotion_threshold,
            .adaptive_promotion_threshold = @as(f64, @floatFromInt(config.promotion_threshold)),
            .active_predictions = PredictionTracker{},
            .promotion_decisions = PromotionMetrics{},
        };
    }
    
    /// Set the fingerprint registry for prediction integration
    pub fn setPredictionRegistry(self: *Self, registry: *fingerprint.FingerprintRegistry) void {
        self.prediction_registry = registry;
    }
    
    /// Enhanced update with predictive capabilities
    pub fn update(self: *Self, work: u64, overhead: u64) void {
        // Always update base account for compatibility
        self.base_account.update(work, overhead);
        
        // Update adaptive promotion threshold based on current performance
        self.updateAdaptiveThreshold();
    }
    
    /// Record a task execution with prediction integration
    pub fn recordTaskExecution(
        self: *Self, 
        task: *const core.Task, 
        actual_cycles: u64, 
        overhead: u64
    ) void {
        // Standard token accounting update
        self.update(actual_cycles, overhead);
        
        // Enhanced prediction tracking
        if (task.fingerprint_hash) |fingerprint_hash| {
            if (self.active_predictions.completePrediction(fingerprint_hash, actual_cycles)) |result| {
                self.updatePredictionAccuracy(result);
                self.total_predictions_made += 1;
                if (result.was_accurate) {
                    self.accurate_predictions += 1;
                }
            }
        }
        
        // Update confidence-weighted tokens
        self.updateConfidenceWeightedTokens(actual_cycles);
    }
    
    /// Predict and pre-allocate tokens for upcoming task
    pub fn predictTaskWork(self: *Self, task: *const core.Task) PredictiveWorkEstimate {
        var estimate = PredictiveWorkEstimate{
            .predicted_cycles = 1000.0, // Default fallback
            .confidence = 0.0,
            .should_preemptively_promote = false,
            .confidence_category = .very_low,
        };
        
        // Try to get prediction from fingerprint registry
        if (self.prediction_registry) |registry| {
            if (task.fingerprint_hash) |fingerprint_hash| {
                // Generate or retrieve fingerprint
                var context = fingerprint.ExecutionContext.init();
                const task_fingerprint = fingerprint.generateTaskFingerprint(task, &context);
                
                // Get multi-factor confidence
                const multi_confidence = registry.getMultiFactorConfidence(task_fingerprint);
                estimate.confidence = multi_confidence.overall_confidence;
                estimate.confidence_category = multi_confidence.getConfidenceCategory();
                
                // Get execution time prediction
                estimate.predicted_cycles = registry.getPredictedCycles(task_fingerprint);
                
                // Track this prediction
                self.active_predictions.addPrediction(
                    fingerprint_hash, 
                    estimate.predicted_cycles, 
                    estimate.confidence
                );
                
                // Determine if preemptive promotion is warranted
                estimate.should_preemptively_promote = self.shouldPreemptivelyPromote(estimate);
                
                // Add to predicted work accumulator
                var self_mut = @constCast(self);
                self_mut.predicted_work_cycles += estimate.predicted_cycles * @as(f64, estimate.confidence);
            }
        }
        
        return estimate;
    }
    
    /// Enhanced promotion decision with confidence and prediction factors
    pub fn shouldPromote(self: *const Self) bool {
        // Always check base account first for backward compatibility
        const base_should_promote = self.base_account.shouldPromote();
        
        // If base account says promote, consider additional factors
        if (base_should_promote) {
            var self_mut = @constCast(self);
            self_mut.promotion_decisions.traditional_promotions += 1;
            return true;
        }
        
        // Check predictive promotion criteria
        if (self.shouldPromoteBasedOnPredictions()) {
            var self_mut = @constCast(self);
            self_mut.promotion_decisions.prediction_based_promotions += 1;
            return true;
        }
        
        // Check confidence-based promotion
        if (self.shouldPromoteBasedOnConfidence()) {
            var self_mut = @constCast(self);
            self_mut.promotion_decisions.confidence_based_promotions += 1;
            return true;
        }
        
        return false;
    }
    
    /// Check if promotion should occur based on predictions
    pub fn shouldPromoteBasedOnPredictions(self: *const Self) bool {
        // If we have sufficient predicted work and good accuracy, promote preemptively
        const min_predicted_work = @as(f64, @floatFromInt(self.base_account.min_work_cycles));
        const predicted_ratio = if (self.base_account.overhead_cycles > 0)
            self.predicted_work_cycles / @as(f64, @floatFromInt(self.base_account.overhead_cycles))
        else
            0.0;
            
        return self.predicted_work_cycles >= min_predicted_work and 
               predicted_ratio > @as(f64, @floatFromInt(self.base_promotion_threshold)) and
               self.prediction_accuracy_score > 0.7; // Require good prediction accuracy
    }
    
    /// Check if promotion should occur based on confidence levels
    fn shouldPromoteBasedOnConfidence(self: *const Self) bool {
        // High confidence in work quality can trigger promotion with lower thresholds
        const adjusted_threshold = self.adaptive_promotion_threshold * @as(f64, self.confidence_adjustment_factor);
        const current_ratio = if (self.base_account.overhead_cycles > 0)
            @as(f64, @floatFromInt(self.base_account.work_cycles)) / @as(f64, @floatFromInt(self.base_account.overhead_cycles))
        else
            0.0;
            
        return current_ratio > adjusted_threshold and self.confidence_weighted_tokens > 0.8;
    }
    
    /// Determine if a task should trigger preemptive promotion
    fn shouldPreemptivelyPromote(self: *const Self, estimate: PredictiveWorkEstimate) bool {
        // High confidence, high value tasks can trigger immediate promotion
        return estimate.confidence > 0.8 and 
               estimate.predicted_cycles > @as(f64, @floatFromInt(self.base_account.min_work_cycles)) * 2.0 and
               estimate.confidence_category == .high;
    }
    
    /// Update adaptive promotion threshold based on prediction accuracy
    fn updateAdaptiveThreshold(self: *Self) void {
        if (self.total_predictions_made > 10) {
            const accuracy_rate = @as(f32, @floatFromInt(self.accurate_predictions)) / @as(f32, @floatFromInt(self.total_predictions_made));
            
            // Higher accuracy allows more aggressive promotion (lower threshold)
            // Lower accuracy requires more conservative promotion (higher threshold)
            const new_adjustment = 0.5 + (1.0 - accuracy_rate) * 1.0; // Range: 0.5 to 1.5
            
            if (@abs(new_adjustment - self.confidence_adjustment_factor) > 0.1) {
                self.confidence_adjustment_factor = new_adjustment;
                self.adaptive_promotion_threshold = @as(f64, @floatFromInt(self.base_promotion_threshold)) * new_adjustment;
                self.promotion_decisions.threshold_adjustments += 1;
            }
        }
    }
    
    /// Update prediction accuracy tracking
    fn updatePredictionAccuracy(self: *Self, result: PredictionResult) void {
        // Use exponential smoothing to update accuracy score
        const new_accuracy: f32 = if (result.was_accurate) 1.0 else 0.0;
        const alpha: f32 = 0.1; // Smoothing factor
        self.prediction_accuracy_score = alpha * new_accuracy + (1.0 - alpha) * self.prediction_accuracy_score;
    }
    
    /// Update confidence-weighted token accumulation
    fn updateConfidenceWeightedTokens(self: *Self, actual_cycles: u64) void {
        // Simple confidence weighting based on recent prediction accuracy
        const confidence_weight = @max(0.1, self.prediction_accuracy_score);
        self.confidence_weighted_tokens += @as(f64, @floatFromInt(actual_cycles)) * confidence_weight;
        
        // Decay over time to prevent unbounded accumulation
        self.confidence_weighted_tokens *= 0.99;
    }
    
    /// Get comprehensive promotion analysis
    pub fn getPromotionAnalysis(self: *const Self) PromotionAnalysis {
        return PromotionAnalysis{
            .base_should_promote = self.base_account.shouldPromote(),
            .prediction_should_promote = self.shouldPromoteBasedOnPredictions(),
            .confidence_should_promote = self.shouldPromoteBasedOnConfidence(),
            .overall_should_promote = self.shouldPromote(),
            .adaptive_threshold = self.adaptive_promotion_threshold,
            .prediction_accuracy = self.prediction_accuracy_score,
            .confidence_weighted_tokens = self.confidence_weighted_tokens,
            .active_predictions_count = self.active_predictions.count,
            .promotion_metrics = self.promotion_decisions,
        };
    }
    
    /// Reset account state (enhanced version)
    pub fn reset(self: *Self) void {
        self.base_account.reset();
        self.predicted_work_cycles = 0.0;
        self.confidence_weighted_tokens = 0.0;
        // Note: Don't reset accuracy scores and metrics - they should persist
    }
};

// ============================================================================
// Supporting Structures
// ============================================================================

/// Predictive work estimate for upcoming tasks
pub const PredictiveWorkEstimate = struct {
    predicted_cycles: f64,
    confidence: f32,
    should_preemptively_promote: bool,
    confidence_category: fingerprint.MultiFactorConfidence.ConfidenceCategory,
};

/// Comprehensive analysis of promotion decision factors
pub const PromotionAnalysis = struct {
    base_should_promote: bool,
    prediction_should_promote: bool,
    confidence_should_promote: bool,
    overall_should_promote: bool,
    adaptive_threshold: f64,
    prediction_accuracy: f32,
    confidence_weighted_tokens: f64,
    active_predictions_count: usize,
    promotion_metrics: PredictiveTokenAccount.PromotionMetrics,
};

// ============================================================================
// Enhanced Predictive Scheduler
// ============================================================================

/// Enhanced scheduler that integrates predictive token accounting
pub const PredictiveScheduler = struct {
    base_scheduler: *scheduler.Scheduler,
    predictive_accounts: []PredictiveTokenAccount,
    fingerprint_registry: ?*fingerprint.FingerprintRegistry = null,
    decision_framework: ?*intelligent_decision.IntelligentDecisionFramework = null,
    
    // Enhanced metrics
    total_preemptive_promotions: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    prediction_based_decisions: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    
    const Self = @This();
    
    /// Initialize predictive scheduler wrapper
    pub fn init(
        allocator: std.mem.Allocator, 
        config: *const core.Config,
        fingerprint_registry: ?*fingerprint.FingerprintRegistry,
        decision_framework: ?*intelligent_decision.IntelligentDecisionFramework
    ) !*Self {
        const self = try allocator.create(Self);
        
        // Create base scheduler
        const base = try scheduler.Scheduler.init(allocator, config);
        
        // Create predictive accounts
        const accounts = try allocator.alloc(PredictiveTokenAccount, config.num_workers orelse 1);
        for (accounts) |*account| {
            account.* = PredictiveTokenAccount.init(config);
            if (fingerprint_registry) |registry| {
                account.setPredictionRegistry(registry);
            }
        }
        
        self.* = Self{
            .base_scheduler = base,
            .predictive_accounts = accounts,
            .fingerprint_registry = fingerprint_registry,
            .decision_framework = decision_framework,
        };
        
        return self;
    }
    
    /// Cleanup predictive scheduler
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.base_scheduler.deinit();
        allocator.free(self.predictive_accounts);
        allocator.destroy(self);
    }
    
    /// Enhanced task submission with predictive work estimation
    pub fn predictTaskWork(self: *Self, worker_id: u32, task: *const core.Task) PredictiveWorkEstimate {
        if (worker_id >= self.predictive_accounts.len) {
            return PredictiveWorkEstimate{
                .predicted_cycles = 1000.0,
                .confidence = 0.0,
                .should_preemptively_promote = false,
                .confidence_category = .very_low,
            };
        }
        
        const estimate = self.predictive_accounts[worker_id].predictTaskWork(task);
        
        if (estimate.should_preemptively_promote) {
            _ = self.total_preemptive_promotions.fetchAdd(1, .monotonic);
            self.base_scheduler.triggerPromotion(worker_id);
        }
        
        return estimate;
    }
    
    /// Record task completion with enhanced tracking
    pub fn recordTaskCompletion(
        self: *Self, 
        worker_id: u32, 
        task: *const core.Task, 
        actual_cycles: u64, 
        overhead_cycles: u64
    ) void {
        if (worker_id >= self.predictive_accounts.len) return;
        
        self.predictive_accounts[worker_id].recordTaskExecution(task, actual_cycles, overhead_cycles);
        
        // Check for promotion with enhanced logic
        if (self.predictive_accounts[worker_id].shouldPromote()) {
            self.base_scheduler.triggerPromotion(worker_id);
            self.predictive_accounts[worker_id].reset();
            _ = self.prediction_based_decisions.fetchAdd(1, .monotonic);
        }
    }
    
    /// Get predictive promotion analysis for a worker
    pub fn getWorkerAnalysis(self: *const Self, worker_id: u32) ?PromotionAnalysis {
        if (worker_id >= self.predictive_accounts.len) return null;
        return self.predictive_accounts[worker_id].getPromotionAnalysis();
    }
    
    /// Get comprehensive scheduler statistics
    pub fn getEnhancedStats(self: *const Self) EnhancedSchedulerStats {
        var total_predictions: u64 = 0;
        var total_accurate: u64 = 0;
        var avg_accuracy: f32 = 0.0;
        
        for (self.predictive_accounts) |account| {
            total_predictions += account.total_predictions_made;
            total_accurate += account.accurate_predictions;
            avg_accuracy += account.prediction_accuracy_score;
        }
        
        avg_accuracy /= @as(f32, @floatFromInt(self.predictive_accounts.len));
        
        return EnhancedSchedulerStats{
            .base_promotions = self.base_scheduler.getPromotionCount(),
            .preemptive_promotions = self.total_preemptive_promotions.load(.acquire),
            .prediction_based_decisions = self.prediction_based_decisions.load(.acquire),
            .total_predictions_made = total_predictions,
            .accurate_predictions = total_accurate,
            .overall_prediction_accuracy = avg_accuracy,
            .prediction_accuracy_rate = if (total_predictions > 0) 
                @as(f32, @floatFromInt(total_accurate)) / @as(f32, @floatFromInt(total_predictions))
            else 
                0.0,
        };
    }
};

/// Enhanced scheduler statistics
pub const EnhancedSchedulerStats = struct {
    base_promotions: u64,
    preemptive_promotions: u64,
    prediction_based_decisions: u64,
    total_predictions_made: u64,
    accurate_predictions: u64,
    overall_prediction_accuracy: f32,
    prediction_accuracy_rate: f32,
};

// ============================================================================
// Testing and Validation
// ============================================================================

const testing = std.testing;

test "predictive token account initialization" {
    const config = core.Config{};
    const account = PredictiveTokenAccount.init(&config);
    
    // Test initialization
    try testing.expect(account.base_account.promotion_threshold == config.promotion_threshold);
    try testing.expect(account.predicted_work_cycles == 0.0);
    try testing.expect(account.prediction_accuracy_score == 0.0);
    try testing.expect(account.active_predictions.count == 0);
}

test "prediction tracking functionality" {
    var tracker = PredictiveTokenAccount.PredictionTracker{};
    
    // Add a prediction
    tracker.addPrediction(12345, 1000.0, 0.8);
    try testing.expect(tracker.count == 1);
    
    // Complete the prediction
    const result = tracker.completePrediction(12345, 950);
    try testing.expect(result != null);
    try testing.expect(result.?.was_accurate == true); // Within 30% tolerance
    try testing.expect(result.?.confidence == 0.8);
}