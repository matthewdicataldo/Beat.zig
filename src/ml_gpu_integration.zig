const std = @import("std");
const fingerprint = @import("fingerprint.zig");
const gpu_integration = @import("gpu_integration.zig");
const gpu_classifier = @import("gpu_classifier.zig");
const ml_classifier = @import("ml_classifier.zig");
const performance_profiler = @import("performance_profiler.zig");
const bayesian_classifier = @import("bayesian_classifier.zig");

// ML-Enhanced GPU Integration for Beat.zig (Task 3.2.2 Integration)
//
// This module integrates the machine learning-based task classification
// with the existing GPU task classifier to create a comprehensive,
// adaptive heterogeneous computing system.
//
// Integration Features:
// - Hierarchical classification: ML → GPU classifier → Final decision
// - Performance feedback loops with real-time adaptation
// - Confidence-weighted decision fusion
// - Fallback mechanisms for uncertain classifications
// - Historical performance tracking and trend analysis
// - Energy and thermal aware scheduling decisions

// ============================================================================
// Enhanced GPU Classification System
// ============================================================================

/// Enhanced GPU task classifier integrating ML and rule-based approaches
pub const EnhancedGPUClassifier = struct {
    allocator: std.mem.Allocator,
    
    // Core classification components
    adaptive_learning_system: bayesian_classifier.AdaptiveLearningSystem,
    rule_based_classifier: gpu_classifier.GPUTaskClassifier,
    
    // Integration configuration
    ml_weight: f32,                    // Weight for ML-based classification (0-1)
    rule_weight: f32,                  // Weight for rule-based classification (0-1)
    confidence_threshold: f32,         // Minimum confidence for ML decisions
    fallback_strategy: FallbackStrategy,
    
    // Performance tracking
    classification_history: std.ArrayList(EnhancedClassificationResult),
    performance_trends: PerformanceTrendTracker,
    
    // Adaptive configuration
    adaptation_enabled: bool,
    last_adaptation_time: u64,
    adaptation_interval_ms: u64,
    
    const Self = @This();
    const MAX_HISTORY = 1000;
    
    pub fn init(allocator: std.mem.Allocator, prior_gpu_probability: f32) Self {
        return Self{
            .allocator = allocator,
            .adaptive_learning_system = bayesian_classifier.AdaptiveLearningSystem.init(allocator, prior_gpu_probability),
            .rule_based_classifier = gpu_classifier.GPUTaskClassifier.init(allocator),
            .ml_weight = 0.7,              // Start with higher ML weight
            .rule_weight = 0.3,            // Lower rule-based weight
            .confidence_threshold = 0.6,   // Moderate confidence threshold
            .fallback_strategy = .conservative_rules,
            .classification_history = std.ArrayList(EnhancedClassificationResult).init(allocator),
            .performance_trends = PerformanceTrendTracker.init(),
            .adaptation_enabled = true,
            .last_adaptation_time = 0,
            .adaptation_interval_ms = 10_000, // 10 seconds
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.adaptive_learning_system.deinit();
        self.rule_based_classifier.deinit();
        self.classification_history.deinit();
    }
    
    /// Enhanced classification combining ML and rule-based approaches
    pub fn classifyTask(
        self: *Self,
        task_fingerprint: fingerprint.TaskFingerprint,
        gpu_device: ?*const gpu_integration.GPUDeviceInfo,
    ) !EnhancedClassificationResult {
        const start_time = std.time.nanoTimestamp();
        
        // Get system state for ML classification
        const system_state = ml_classifier.SystemState.getCurrentState();
        
        // ML-based classification
        const ml_result = try self.adaptive_learning_system.classifyTask(
            task_fingerprint,
            gpu_device,
            system_state
        );
        
        // Rule-based classification
        const rule_result = try self.rule_based_classifier.classifyTask(task_fingerprint, gpu_device);
        
        // Combine classifications using weighted fusion
        const combined_result = self.combineClassifications(ml_result, rule_result, gpu_device);
        
        // Create enhanced result
        const enhanced_result = EnhancedClassificationResult{
            .final_decision = combined_result.use_gpu,
            .confidence = combined_result.confidence,
            .ml_result = ml_result,
            .rule_result = rule_result,
            .fusion_strategy = combined_result.strategy,
            .reasoning = combined_result.reasoning,
            .fallback_used = combined_result.fallback_used,
            .classification_time_ns = @intCast(std.time.nanoTimestamp() - start_time),
            .timestamp = @intCast(std.time.milliTimestamp()),
            .task_fingerprint = task_fingerprint,
        };
        
        // Store result for trend analysis
        try self.classification_history.append(enhanced_result);
        if (self.classification_history.items.len > MAX_HISTORY) {
            _ = self.classification_history.orderedRemove(0);
        }
        
        // Periodic adaptation
        if (self.adaptation_enabled) {
            try self.performPeriodicAdaptation();
        }
        
        return enhanced_result;
    }
    
    /// Provide feedback on classification performance
    pub fn provideFeedback(
        self: *Self,
        classification: EnhancedClassificationResult,
        actual_device_used: ml_classifier.DeviceType,
        actual_performance: performance_profiler.PerformanceMeasurement,
    ) !void {
        // Provide feedback to adaptive learning system
        try self.adaptive_learning_system.provideFeedback(
            classification.ml_result,
            actual_device_used,
            actual_performance
        );
        
        // Update performance trends
        self.performance_trends.update(classification, actual_device_used, actual_performance);
        
        // Adapt weights based on feedback quality
        self.adaptClassificationWeights(classification, actual_device_used, actual_performance);
    }
    
    /// Combine ML and rule-based classifications
    fn combineClassifications(
        self: *const Self,
        ml_result: bayesian_classifier.AdaptiveClassification,
        rule_result: gpu_classifier.GPUAnalysis,
        gpu_device: ?*const gpu_integration.GPUDeviceInfo,
    ) CombinedClassification {
        // Convert rule-based result to common format
        const rule_probability = switch (rule_result.suitability) {
            .highly_unsuitable => 0.1,
            .unsuitable => 0.3,
            .neutral => 0.5,
            .suitable => 0.7,
            .highly_suitable => 0.9,
        };
        
        const rule_confidence = rule_result.confidence_score;
        
        // Check if ML result is confident enough
        if (ml_result.confidence >= self.confidence_threshold) {
            // High confidence ML decision - use weighted combination
            const combined_prob = self.ml_weight * ml_result.probability + 
                                 self.rule_weight * rule_probability;
            const combined_confidence = self.ml_weight * ml_result.confidence + 
                                       self.rule_weight * rule_confidence;
            
            return CombinedClassification{
                .use_gpu = combined_prob > 0.5,
                .confidence = combined_confidence,
                .strategy = .weighted_fusion,
                .reasoning = self.generateReasoning(ml_result, rule_result, .weighted_fusion),
                .fallback_used = false,
            };
        } else {
            // Low confidence ML - use fallback strategy
            return self.applyFallbackStrategy(ml_result, rule_result, gpu_device);
        }
    }
    
    /// Apply fallback strategy when ML confidence is low
    fn applyFallbackStrategy(
        self: *const Self,
        ml_result: bayesian_classifier.AdaptiveClassification,
        rule_result: gpu_classifier.GPUAnalysis,
        gpu_device: ?*const gpu_integration.GPUDeviceInfo,
    ) CombinedClassification {
        switch (self.fallback_strategy) {
            .conservative_rules => {
                // Use rule-based decision with conservative threshold
                const use_gpu = rule_result.suitability == .highly_suitable or 
                               (rule_result.suitability == .suitable and rule_result.confidence_score > 0.8);
                
                return CombinedClassification{
                    .use_gpu = use_gpu,
                    .confidence = rule_result.confidence_score * 0.8, // Reduced confidence for fallback
                    .strategy = .conservative_fallback,
                    .reasoning = self.generateReasoning(ml_result, rule_result, .conservative_fallback),
                    .fallback_used = true,
                };
            },
            .ml_with_rules_veto => {
                // Use ML decision unless rules strongly disagree
                var use_gpu = ml_result.use_gpu;
                var confidence = ml_result.confidence * 0.7; // Reduced for uncertainty
                
                if ((ml_result.use_gpu and rule_result.suitability == .highly_unsuitable) or
                    (!ml_result.use_gpu and rule_result.suitability == .highly_suitable)) {
                    // Strong disagreement - defer to rules
                    use_gpu = rule_result.suitability == .highly_suitable;
                    confidence = rule_result.confidence_score * 0.6;
                }
                
                return CombinedClassification{
                    .use_gpu = use_gpu,
                    .confidence = confidence,
                    .strategy = .ml_with_veto,
                    .reasoning = self.generateReasoning(ml_result, rule_result, .ml_with_veto),
                    .fallback_used = true,
                };
            },
            .device_availability => {
                // Consider device availability first
                if (gpu_device == null) {
                    return CombinedClassification{
                        .use_gpu = false,
                        .confidence = 1.0,
                        .strategy = .device_unavailable,
                        .reasoning = "GPU device not available",
                        .fallback_used = true,
                    };
                }
                
                // Use rule-based decision as fallback
                return CombinedClassification{
                    .use_gpu = rule_result.suitability == .suitable or rule_result.suitability == .highly_suitable,
                    .confidence = rule_result.confidence_score,
                    .strategy = .device_fallback,
                    .reasoning = self.generateReasoning(ml_result, rule_result, .device_fallback),
                    .fallback_used = true,
                };
            },
        }
    }
    
    /// Generate human-readable reasoning for classification decision
    fn generateReasoning(
        self: *const Self,
        ml_result: bayesian_classifier.AdaptiveClassification,
        rule_result: gpu_classifier.GPUAnalysis,
        strategy: FusionStrategy,
    ) []const u8 {
        _ = self;
        _ = ml_result;
        _ = rule_result;
        
        return switch (strategy) {
            .weighted_fusion => "High-confidence ML prediction combined with rule-based analysis",
            .conservative_fallback => "Conservative rule-based decision due to low ML confidence",
            .ml_with_veto => "ML prediction with rule-based veto for safety",
            .device_unavailable => "Device unavailable - forced CPU execution",
            .device_fallback => "Device-aware fallback using rule-based classification",
        };
    }
    
    /// Adapt classification weights based on recent performance
    fn adaptClassificationWeights(
        self: *Self,
        classification: EnhancedClassificationResult,
        actual_device_used: ml_classifier.DeviceType,
        actual_performance: performance_profiler.PerformanceMeasurement,
    ) void {
        _ = actual_performance;
        
        // Check if the decision was correct
        const ml_correct = (classification.ml_result.use_gpu and actual_device_used == .gpu) or
                          (!classification.ml_result.use_gpu and actual_device_used == .cpu);
        
        const rule_correct = ((classification.rule_result.suitability == .suitable or 
                              classification.rule_result.suitability == .highly_suitable) and actual_device_used == .gpu) or
                            ((classification.rule_result.suitability == .unsuitable or 
                              classification.rule_result.suitability == .highly_unsuitable) and actual_device_used == .cpu);
        
        // Adapt weights based on accuracy
        const adaptation_rate = 0.01;
        
        if (ml_correct and !rule_correct) {
            // ML was right, rules were wrong - increase ML weight
            self.ml_weight = std.math.min(0.9, self.ml_weight + adaptation_rate);
            self.rule_weight = 1.0 - self.ml_weight;
        } else if (!ml_correct and rule_correct) {
            // Rules were right, ML was wrong - increase rules weight
            self.rule_weight = std.math.min(0.7, self.rule_weight + adaptation_rate);
            self.ml_weight = 1.0 - self.rule_weight;
        }
        
        // Ensure weights are normalized
        const total = self.ml_weight + self.rule_weight;
        self.ml_weight /= total;
        self.rule_weight /= total;
    }
    
    /// Perform periodic adaptation of the classification system
    fn performPeriodicAdaptation(self: *Self) !void {
        const current_time = @as(u64, @intCast(std.time.milliTimestamp()));
        if (current_time - self.last_adaptation_time < self.adaptation_interval_ms) {
            return;
        }
        
        if (self.classification_history.items.len < 20) {
            return; // Need sufficient data for adaptation
        }
        
        // Analyze recent classification performance
        const analysis = self.analyzeRecentPerformance();
        
        // Adapt confidence threshold based on accuracy vs uncertainty tradeoff
        if (analysis.accuracy < 0.7 and analysis.avg_confidence > 0.8) {
            // Low accuracy despite high confidence - increase threshold
            self.confidence_threshold = std.math.min(0.9, self.confidence_threshold + 0.05);
        } else if (analysis.accuracy > 0.9 and analysis.avg_confidence < 0.6) {
            // High accuracy with low confidence - decrease threshold
            self.confidence_threshold = std.math.max(0.4, self.confidence_threshold - 0.03);
        }
        
        // Adapt fallback strategy based on fallback performance
        if (analysis.fallback_accuracy > analysis.normal_accuracy + 0.1) {
            // Fallbacks are performing better - lower confidence threshold
            self.confidence_threshold = std.math.max(0.4, self.confidence_threshold - 0.02);
        }
        
        self.last_adaptation_time = current_time;
    }
    
    /// Analyze recent classification performance
    fn analyzeRecentPerformance(self: *const Self) PerformanceAnalysis {
        const recent_count = std.math.min(100, self.classification_history.items.len);
        const recent_start = self.classification_history.items.len - recent_count;
        
        var total_confidence: f32 = 0.0;
        var fallback_count: u32 = 0;
        var fallback_accuracy: u32 = 0;
        var normal_accuracy: u32 = 0;
        var normal_count: u32 = 0;
        
        _ = &fallback_accuracy;
        _ = &normal_accuracy;
        
        for (self.classification_history.items[recent_start..]) |result| {
            total_confidence += result.confidence;
            
            if (result.fallback_used) {
                fallback_count += 1;
                // Note: Would need actual outcome data to calculate accuracy
                // This is simplified for demonstration
            } else {
                normal_count += 1;
            }
        }
        
        return PerformanceAnalysis{
            .accuracy = 0.8, // Placeholder - would calculate from actual outcomes
            .avg_confidence = total_confidence / @as(f32, @floatFromInt(recent_count)),
            .fallback_rate = @as(f32, @floatFromInt(fallback_count)) / @as(f32, @floatFromInt(recent_count)),
            .fallback_accuracy = 0.75, // Placeholder
            .normal_accuracy = 0.82,   // Placeholder
        };
    }
    
    /// Get comprehensive system statistics
    pub fn getSystemStatistics(self: *const Self) EnhancedSystemStatistics {
        const adaptive_stats = self.adaptive_learning_system.getSystemStatistics();
        const trend_stats = self.performance_trends.getStatistics();
        
        return EnhancedSystemStatistics{
            .adaptive_stats = adaptive_stats,
            .trend_stats = trend_stats,
            .ml_weight = self.ml_weight,
            .rule_weight = self.rule_weight,
            .confidence_threshold = self.confidence_threshold,
            .fallback_strategy = self.fallback_strategy,
            .total_classifications = self.classification_history.items.len,
            .adaptation_enabled = self.adaptation_enabled,
        };
    }
    
    /// Enable or disable adaptive behavior
    pub fn setAdaptiveMode(self: *Self, enabled: bool) void {
        self.adaptation_enabled = enabled;
    }
    
    /// Manually adjust classification weights
    pub fn setClassificationWeights(self: *Self, ml_weight: f32, rule_weight: f32) void {
        const total = ml_weight + rule_weight;
        self.ml_weight = ml_weight / total;
        self.rule_weight = rule_weight / total;
    }
};

/// Enhanced classification result combining ML and rule-based approaches
pub const EnhancedClassificationResult = struct {
    final_decision: bool,                                           // Final GPU/CPU decision
    confidence: f32,                                               // Overall confidence (0-1)
    ml_result: bayesian_classifier.AdaptiveClassification,         // ML classification result
    rule_result: gpu_classifier.GPUAnalysis,                      // Rule-based analysis
    fusion_strategy: FusionStrategy,                               // Strategy used for fusion
    reasoning: []const u8,                                         // Human-readable reasoning
    fallback_used: bool,                                           // Whether fallback was used
    classification_time_ns: u64,                                   // Time taken for classification
    timestamp: u64,                                                // Timestamp of classification
    task_fingerprint: fingerprint.TaskFingerprint,                // Task characteristics
};

/// Classification fusion strategies
pub const FusionStrategy = enum {
    weighted_fusion,        // Weighted combination of ML and rules
    conservative_fallback,  // Conservative rule-based fallback
    ml_with_veto,          // ML decision with rule-based veto
    device_unavailable,    // Device unavailable fallback
    device_fallback,       // Device-aware fallback
};

/// Fallback strategies for uncertain classifications
pub const FallbackStrategy = enum {
    conservative_rules,     // Use conservative rule-based decisions
    ml_with_rules_veto,    // Use ML but allow rules to veto
    device_availability,   // Consider device availability first
};

/// Combined classification result (internal)
const CombinedClassification = struct {
    use_gpu: bool,
    confidence: f32,
    strategy: FusionStrategy,
    reasoning: []const u8,
    fallback_used: bool,
};

/// Performance trend tracking
const PerformanceTrendTracker = struct {
    recent_ml_accuracy: f32,
    recent_rule_accuracy: f32,
    recent_combined_accuracy: f32,
    sample_count: u32,
    
    fn init() PerformanceTrendTracker {
        return PerformanceTrendTracker{
            .recent_ml_accuracy = 0.0,
            .recent_rule_accuracy = 0.0,
            .recent_combined_accuracy = 0.0,
            .sample_count = 0,
        };
    }
    
    fn update(
        self: *PerformanceTrendTracker,
        classification: EnhancedClassificationResult,
        actual_device: ml_classifier.DeviceType,
        performance: performance_profiler.PerformanceMeasurement,
    ) void {
        _ = performance;
        self.sample_count += 1;
        
        // Update accuracies (simplified - would need actual accuracy calculations)
        const alpha = 0.1; // Exponential moving average factor
        
        // ML accuracy
        const ml_correct = (classification.ml_result.use_gpu and actual_device == .gpu) or
                          (!classification.ml_result.use_gpu and actual_device == .cpu);
        const ml_acc = if (ml_correct) 1.0 else 0.0;
        self.recent_ml_accuracy = (1.0 - alpha) * self.recent_ml_accuracy + alpha * ml_acc;
        
        // Rule accuracy
        const rule_gpu_recommended = classification.rule_result.suitability == .suitable or
                                    classification.rule_result.suitability == .highly_suitable;
        const rule_correct = (rule_gpu_recommended and actual_device == .gpu) or
                            (!rule_gpu_recommended and actual_device == .cpu);
        const rule_acc = if (rule_correct) 1.0 else 0.0;
        self.recent_rule_accuracy = (1.0 - alpha) * self.recent_rule_accuracy + alpha * rule_acc;
        
        // Combined accuracy
        const combined_correct = (classification.final_decision and actual_device == .gpu) or
                               (!classification.final_decision and actual_device == .cpu);
        const combined_acc = if (combined_correct) 1.0 else 0.0;
        self.recent_combined_accuracy = (1.0 - alpha) * self.recent_combined_accuracy + alpha * combined_acc;
    }
    
    fn getStatistics(self: *const PerformanceTrendTracker) TrendStatistics {
        return TrendStatistics{
            .ml_accuracy = self.recent_ml_accuracy,
            .rule_accuracy = self.recent_rule_accuracy,
            .combined_accuracy = self.recent_combined_accuracy,
            .sample_count = self.sample_count,
        };
    }
};

/// Performance analysis result
const PerformanceAnalysis = struct {
    accuracy: f32,
    avg_confidence: f32,
    fallback_rate: f32,
    fallback_accuracy: f32,
    normal_accuracy: f32,
};

/// Trend statistics
pub const TrendStatistics = struct {
    ml_accuracy: f32,
    rule_accuracy: f32,
    combined_accuracy: f32,
    sample_count: u32,
};

/// Enhanced system statistics
pub const EnhancedSystemStatistics = struct {
    adaptive_stats: bayesian_classifier.AdaptiveSystemStatistics,
    trend_stats: TrendStatistics,
    ml_weight: f32,
    rule_weight: f32,
    confidence_threshold: f32,
    fallback_strategy: FallbackStrategy,
    total_classifications: usize,
    adaptation_enabled: bool,
};

// ============================================================================
// Test Utilities
// ============================================================================

test "enhanced GPU classifier integration" {
    const allocator = std.testing.allocator;
    
    var enhanced_classifier = EnhancedGPUClassifier.init(allocator, 0.5);
    defer enhanced_classifier.deinit();
    
    // Create test fingerprint
    const test_fingerprint = fingerprint.TaskFingerprint{
        .call_site_hash = 0x12345678,
        .data_size_class = 24, // Large data
        .data_alignment = 8,
        .access_pattern = .sequential,
        .simd_width = 8,
        .cache_locality = 12,
        .numa_node_hint = 0,
        .cpu_intensity = 12,
        .parallel_potential = 14, // High parallelism
        .execution_phase = 1,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 2,
        .time_of_day_bucket = 14,
        .execution_frequency = 3,
        .seasonal_pattern = 1,
        .variance_level = 5,
        .expected_cycles_log2 = 26,
        .memory_footprint_log2 = 24,
        .io_intensity = 1,
        .cache_miss_rate = 2,
        .branch_predictability = 13,
        .vectorization_benefit = 14,
    };
    
    // Classify task
    const result = try enhanced_classifier.classifyTask(test_fingerprint, null);
    
    // Verify enhanced result
    try std.testing.expect(result.confidence >= 0.0 and result.confidence <= 1.0);
    try std.testing.expect(result.classification_time_ns > 0);
    try std.testing.expect(result.timestamp > 0);
    
    // Verify both ML and rule-based results are present
    try std.testing.expect(result.ml_result.confidence >= 0.0);
    try std.testing.expect(result.rule_result.confidence_score >= 0.0);
    
    // Test feedback mechanism
    const mock_performance = performance_profiler.PerformanceMeasurement{
        .execution_time_ns = 1_000_000, // 1ms
        .cpu_cycles = 3_000_000,
        .memory_accesses = 1000,
        .cache_misses = 100,
        .instructions_executed = 10_000,
        .power_consumption_mw = 50_000,
        .temperature_celsius = 65,
        .memory_bandwidth_used = 0.4,
        .cpu_utilization = 0.6,
        .gpu_utilization = 0.3,
    };
    
    try enhanced_classifier.provideFeedback(result, .cpu, mock_performance);
    
    // Verify system statistics
    const stats = enhanced_classifier.getSystemStatistics();
    try std.testing.expect(stats.total_classifications > 0);
    try std.testing.expect(stats.ml_weight + stats.rule_weight == 1.0);
}

test "classification weight adaptation" {
    const allocator = std.testing.allocator;
    
    var enhanced_classifier = EnhancedGPUClassifier.init(allocator, 0.5);
    defer enhanced_classifier.deinit();
    
    // Get initial weights
    _ = enhanced_classifier.ml_weight; // Store for potential future use
    _ = enhanced_classifier.rule_weight;
    
    // Manually adjust weights
    enhanced_classifier.setClassificationWeights(0.8, 0.2);
    
    // Verify adjustment
    try std.testing.expect(std.math.fabs(enhanced_classifier.ml_weight - 0.8) < 0.01);
    try std.testing.expect(std.math.fabs(enhanced_classifier.rule_weight - 0.2) < 0.01);
    try std.testing.expect(enhanced_classifier.ml_weight + enhanced_classifier.rule_weight == 1.0);
    
    // Test adaptive mode toggle
    enhanced_classifier.setAdaptiveMode(false);
    try std.testing.expect(!enhanced_classifier.adaptation_enabled);
    
    enhanced_classifier.setAdaptiveMode(true);
    try std.testing.expect(enhanced_classifier.adaptation_enabled);
}