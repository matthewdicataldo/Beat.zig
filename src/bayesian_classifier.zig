const std = @import("std");
const ml_classifier = @import("ml_classifier.zig");
const performance_profiler = @import("performance_profiler.zig");
const fingerprint = @import("fingerprint.zig");

// Bayesian Uncertainty Quantification for Beat.zig ML Classification (Task 3.2.2)
//
// This module implements Bayesian confidence scoring and uncertainty quantification
// for robust GPU task classification in heterogeneous computing environments.
//
// Based on research in uncertainty-aware predictions and Bayesian deep learning,
// this system provides:
// - Epistemic and aleatory uncertainty quantification
// - Bayesian confidence intervals for classification decisions
// - Evidence-based decision making with uncertainty propagation
// - Adaptive confidence thresholds based on system performance
// - Risk-aware task routing with fallback mechanisms
//
// Key Features:
// - Beta-Bernoulli conjugate prior for binary classification
// - Thompson sampling for exploration vs exploitation
// - Confidence-calibrated probability estimates
// - Multi-armed bandit approach for device selection
// - Uncertainty-aware adaptive learning rates

// ============================================================================
// Bayesian Classification Framework
// ============================================================================

/// Bayesian binary classifier for GPU vs CPU task assignment
/// Uses Beta-Bernoulli conjugate prior for analytical uncertainty quantification
pub const BayesianGPUClassifier = struct {
    allocator: std.mem.Allocator,
    
    // Beta distribution parameters for GPU success probability
    // Beta(alpha, beta) represents our belief about P(GPU better | features)
    alpha: f64,                    // Success count + prior
    beta: f64,                     // Failure count + prior
    
    // Prior knowledge configuration
    prior_alpha: f64,              // Prior belief in GPU success
    prior_beta: f64,               // Prior belief in GPU failure
    
    // Uncertainty quantification
    epistemic_uncertainty: f64,    // Model uncertainty (reducible with data)
    aleatory_uncertainty: f64,     // Inherent noise (irreducible)
    
    // Adaptive confidence parameters
    confidence_threshold: f64,     // Current confidence threshold
    min_confidence: f64,           // Minimum confidence threshold
    max_confidence: f64,           // Maximum confidence threshold
    adaptation_rate: f64,          // Rate of confidence adaptation
    
    // Performance tracking
    total_predictions: u64,
    correct_predictions: u64,
    uncertain_predictions: u64,    // Predictions below confidence threshold
    
    // Bandit algorithm state for exploration
    exploration_rate: f64,         // ε in ε-greedy exploration
    temperature: f64,              // Temperature for softmax exploration
    
    const Self = @This();
    
    pub fn init(allocator: std.mem.Allocator, prior_gpu_probability: f64) Self {
        // Convert probability to Beta parameters
        // For weak prior: α = β = 1 (uniform)
        // For informed prior: use method of moments
        const prior_strength = 2.0; // Equivalent sample size for prior
        const alpha_prior = prior_gpu_probability * prior_strength;
        const beta_prior = (1.0 - prior_gpu_probability) * prior_strength;
        
        return Self{
            .allocator = allocator,
            .alpha = alpha_prior,
            .beta = beta_prior,
            .prior_alpha = alpha_prior,
            .prior_beta = beta_prior,
            .epistemic_uncertainty = 1.0, // High initial uncertainty
            .aleatory_uncertainty = 0.1,  // Assume low inherent noise
            .confidence_threshold = 0.7,
            .min_confidence = 0.5,
            .max_confidence = 0.95,
            .adaptation_rate = 0.01,
            .total_predictions = 0,
            .correct_predictions = 0,
            .uncertain_predictions = 0,
            .exploration_rate = 0.1,
            .temperature = 1.0,
        };
    }
    
    /// Make classification decision with Bayesian uncertainty quantification
    pub fn classify(self: *Self, features: ml_classifier.MLTaskFeatures) BayesianClassification {
        self.total_predictions += 1;
        
        // Calculate posterior mean and variance
        const total_trials = self.alpha + self.beta;
        const posterior_mean = self.alpha / total_trials;
        const posterior_variance = (self.alpha * self.beta) / (total_trials * total_trials * (total_trials + 1.0));
        
        // Epistemic uncertainty (model uncertainty - decreases with data)
        self.epistemic_uncertainty = std.math.sqrt(posterior_variance);
        
        // Calculate confidence intervals
        const confidence_interval = self.calculateCredibleInterval(0.95);
        
        // Confidence score based on interval width
        const interval_width = confidence_interval.upper - confidence_interval.lower;
        const confidence = std.math.max(0.0, 1.0 - interval_width * 2.0); // Normalize to [0,1]
        
        // Decision making strategy
        const decision = self.makeDecision(posterior_mean, confidence, features);
        
        // Track uncertain predictions
        if (confidence < self.confidence_threshold) {
            self.uncertain_predictions += 1;
        }
        
        return BayesianClassification{
            .use_gpu = decision.use_gpu,
            .probability = posterior_mean,
            .confidence = confidence,
            .epistemic_uncertainty = self.epistemic_uncertainty,
            .aleatory_uncertainty = self.aleatory_uncertainty,
            .confidence_interval = confidence_interval,
            .decision_strategy = decision.strategy,
            .exploration_bonus = decision.exploration_bonus,
        };
    }
    
    /// Update classifier with outcome (Bayesian learning)
    pub fn updateWithOutcome(self: *Self, gpu_was_better: bool, performance_ratio: f64) void {
        if (gpu_was_better) {
            self.alpha += 1.0;
            self.correct_predictions += 1;
        } else {
            self.beta += 1.0;
        }
        
        // Update aleatory uncertainty based on performance variance
        const outcome_surprise = if (gpu_was_better) 
            1.0 - (self.alpha / (self.alpha + self.beta))
        else 
            self.alpha / (self.alpha + self.beta);
        
        // Exponential moving average for aleatory uncertainty
        const alpha_ema = 0.1;
        self.aleatory_uncertainty = (1.0 - alpha_ema) * self.aleatory_uncertainty + 
                                   alpha_ema * std.math.fabs(outcome_surprise);
        
        // Adapt confidence threshold based on recent performance
        self.adaptConfidenceThreshold(performance_ratio);
        
        // Adapt exploration rate (decay over time)
        self.exploration_rate = std.math.max(0.01, self.exploration_rate * 0.995);
    }
    
    /// Calculate Bayesian credible interval
    fn calculateCredibleInterval(self: *const Self, credibility: f64) CredibleInterval {
        // For Beta distribution, use quantile function
        const alpha_level = (1.0 - credibility) / 2.0;
        
        // Approximate using normal approximation for computational efficiency
        // For large α, β: Beta(α,β) ≈ N(μ, σ²)
        const total = self.alpha + self.beta;
        const mean = self.alpha / total;
        const variance = (self.alpha * self.beta) / (total * total * (total + 1.0));
        const std_dev = std.math.sqrt(variance);
        
        // Normal quantiles (approximation)
        const z_score = self.normalQuantile(1.0 - alpha_level);
        
        const lower = std.math.max(0.0, mean - z_score * std_dev);
        const upper = std.math.min(1.0, mean + z_score * std_dev);
        
        return CredibleInterval{
            .lower = lower,
            .upper = upper,
            .credibility = credibility,
        };
    }
    
    /// Decision making with exploration and uncertainty
    fn makeDecision(
        self: *const Self, 
        probability: f64, 
        confidence: f64, 
        features: ml_classifier.MLTaskFeatures
    ) Decision {
        // Thompson sampling for exploration
        const should_explore = self.shouldExplore(confidence);
        
        if (should_explore) {
            // Sample from posterior Beta distribution
            const sampled_prob = self.sampleFromPosterior();
            return Decision{
                .use_gpu = sampled_prob > 0.5,
                .strategy = .thompson_sampling,
                .exploration_bonus = std.math.fabs(sampled_prob - probability),
            };
        }
        
        // Confident exploitation
        if (confidence >= self.confidence_threshold) {
            return Decision{
                .use_gpu = probability > 0.5,
                .strategy = .confident_exploitation,
                .exploration_bonus = 0.0,
            };
        }
        
        // Uncertain - use conservative fallback
        return self.conservativeFallback(features);
    }
    
    /// Conservative decision when uncertain
    fn conservativeFallback(self: *const Self, features: ml_classifier.MLTaskFeatures) Decision {
        _ = self;
        
        // Use heuristics for conservative decision
        const high_parallelism = features.parallelization_potential > 0.8;
        const large_data = features.data_size_log2 > 0.7;
        const gpu_available = features.available_gpu_memory > 0.5;
        
        // Conservative: only use GPU if clearly beneficial
        const use_gpu = high_parallelism and large_data and gpu_available;
        
        return Decision{
            .use_gpu = use_gpu,
            .strategy = .conservative_fallback,
            .exploration_bonus = 0.0,
        };
    }
    
    /// Determine if we should explore (Thompson sampling vs ε-greedy)
    fn shouldExplore(self: *const Self, confidence: f64) bool {
        // Higher exploration when confidence is low
        const uncertainty_bonus = 1.0 - confidence;
        const effective_exploration_rate = self.exploration_rate + uncertainty_bonus * 0.1;
        
        // Random exploration decision
        var prng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
        return prng.random().float(f64) < effective_exploration_rate;
    }
    
    /// Sample from posterior Beta distribution (Box-Muller for normal approximation)
    fn sampleFromPosterior(self: *const Self) f64 {
        const total = self.alpha + self.beta;
        const mean = self.alpha / total;
        const variance = (self.alpha * self.beta) / (total * total * (total + 1.0));
        
        // Use normal approximation for efficiency
        var prng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
        const random = prng.random();
        
        const uniform_sample = random.float(f64);
        const u2 = random.float(f64);
        
        // Box-Muller transform
        const z0 = std.math.sqrt(-2.0 * std.math.log(uniform_sample)) * std.math.cos(2.0 * std.math.pi * u2);
        const sample = mean + std.math.sqrt(variance) * z0;
        
        return std.math.clamp(sample, 0.0, 1.0);
    }
    
    /// Adapt confidence threshold based on performance
    fn adaptConfidenceThreshold(self: *Self, performance_ratio: f64) void {
        // If GPU was much better/worse than expected, adjust confidence requirements
        const surprise = std.math.fabs(std.math.log(performance_ratio));
        
        if (surprise > 0.5) { // High surprise - increase confidence requirements
            self.confidence_threshold = std.math.min(
                self.max_confidence,
                self.confidence_threshold + self.adaptation_rate
            );
        } else if (surprise < 0.1) { // Low surprise - can be less conservative
            self.confidence_threshold = std.math.max(
                self.min_confidence,
                self.confidence_threshold - self.adaptation_rate * 0.5
            );
        }
    }
    
    /// Approximate normal quantile function
    fn normalQuantile(self: *const Self, p: f64) f64 {
        _ = self;
        // Beasley-Springer-Moro approximation for standard normal quantile
        const a0 = 2.50662823884;
        const a1 = -18.61500062529;
        const a2 = 41.39119773534;
        const a3 = -25.44106049637;
        
        const b1 = -8.47351093090;
        const b2 = 23.08336743743;
        const b3 = -21.06224101826;
        const b4 = 3.13082909833;
        
        if (p <= 0.5) {
            const t = std.math.sqrt(-2.0 * std.math.log(p));
            return -(((a3 * t + a2) * t + a1) * t + a0) / ((((b4 * t + b3) * t + b2) * t + b1) * t + 1.0);
        } else {
            const t = std.math.sqrt(-2.0 * std.math.log(1.0 - p));
            return (((a3 * t + a2) * t + a1) * t + a0) / ((((b4 * t + b3) * t + b2) * t + b1) * t + 1.0);
        }
    }
    
    /// Get classifier statistics
    pub fn getStatistics(self: *const Self) BayesianStatistics {
        const accuracy = if (self.total_predictions > 0)
            @as(f64, @floatFromInt(self.correct_predictions)) / @as(f64, @floatFromInt(self.total_predictions))
        else 0.0;
        
        const uncertainty_rate = if (self.total_predictions > 0)
            @as(f64, @floatFromInt(self.uncertain_predictions)) / @as(f64, @floatFromInt(self.total_predictions))
        else 0.0;
        
        return BayesianStatistics{
            .total_predictions = self.total_predictions,
            .accuracy = accuracy,
            .epistemic_uncertainty = self.epistemic_uncertainty,
            .aleatory_uncertainty = self.aleatory_uncertainty,
            .confidence_threshold = self.confidence_threshold,
            .uncertainty_rate = uncertainty_rate,
            .exploration_rate = self.exploration_rate,
            .posterior_mean = self.alpha / (self.alpha + self.beta),
            .posterior_variance = (self.alpha * self.beta) / 
                ((self.alpha + self.beta) * (self.alpha + self.beta) * (self.alpha + self.beta + 1.0)),
        };
    }
};

/// Bayesian classification result with uncertainty quantification
pub const BayesianClassification = struct {
    use_gpu: bool,                          // Classification decision
    probability: f64,                       // Posterior probability of GPU being better
    confidence: f64,                        // Confidence in decision (0-1)
    epistemic_uncertainty: f64,             // Model uncertainty (reducible)
    aleatory_uncertainty: f64,              // Inherent uncertainty (irreducible)
    confidence_interval: CredibleInterval,  // Bayesian credible interval
    decision_strategy: DecisionStrategy,    // Strategy used for this decision
    exploration_bonus: f64,                 // Exploration bonus applied
};

/// Bayesian credible interval
pub const CredibleInterval = struct {
    lower: f64,         // Lower bound of interval
    upper: f64,         // Upper bound of interval
    credibility: f64,   // Credibility level (e.g., 0.95 for 95%)
};

/// Decision making strategy
pub const DecisionStrategy = enum {
    confident_exploitation,     // High confidence, exploit best option
    thompson_sampling,          // Exploration via Thompson sampling
    conservative_fallback,      // Low confidence, use heuristics
    epsilon_greedy,            // ε-greedy exploration
};

/// Internal decision structure
const Decision = struct {
    use_gpu: bool,
    strategy: DecisionStrategy,
    exploration_bonus: f64,
};

/// Bayesian classifier statistics
pub const BayesianStatistics = struct {
    total_predictions: u64,
    accuracy: f64,
    epistemic_uncertainty: f64,
    aleatory_uncertainty: f64,
    confidence_threshold: f64,
    uncertainty_rate: f64,
    exploration_rate: f64,
    posterior_mean: f64,
    posterior_variance: f64,
};

// ============================================================================
// Real-Time Feedback System
// ============================================================================

/// Real-time adaptive learning system that integrates ML classification with performance feedback
pub const AdaptiveLearningSystem = struct {
    allocator: std.mem.Allocator,
    
    // Core components
    bayesian_classifier: BayesianGPUClassifier,
    feature_extractor: ml_classifier.MLFeatureExtractor,
    performance_profiler: performance_profiler.PerformanceProfiler,
    
    // Feedback loop configuration
    feedback_window_size: u32,          // Size of feedback window
    min_feedback_samples: u32,          // Minimum samples before adaptation
    adaptation_interval_ms: u64,        // How often to adapt (milliseconds)
    last_adaptation_time: u64,          // Last adaptation timestamp
    
    // Performance tracking
    recent_decisions: std.ArrayList(ClassificationOutcome),
    feedback_metrics: FeedbackMetrics,
    
    const Self = @This();
    
    pub fn init(allocator: std.mem.Allocator, prior_gpu_probability: f64) Self {
        return Self{
            .allocator = allocator,
            .bayesian_classifier = BayesianGPUClassifier.init(allocator, prior_gpu_probability),
            .feature_extractor = ml_classifier.MLFeatureExtractor.init(allocator),
            .performance_profiler = performance_profiler.PerformanceProfiler.init(allocator),
            .feedback_window_size = 100,
            .min_feedback_samples = 10,
            .adaptation_interval_ms = 5000, // 5 seconds
            .last_adaptation_time = 0,
            .recent_decisions = std.ArrayList(ClassificationOutcome).init(allocator),
            .feedback_metrics = FeedbackMetrics.init(),
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.feature_extractor.deinit();
        self.performance_profiler.deinit();
        self.recent_decisions.deinit();
    }
    
    /// Classify task with adaptive learning and feedback
    pub fn classifyTask(
        self: *Self,
        task_fingerprint: fingerprint.TaskFingerprint,
        gpu_device: ?*const @import("gpu_integration.zig").GPUDeviceInfo,
        system_state: ml_classifier.SystemState,
    ) !AdaptiveClassification {
        // Extract ML features
        const features = try self.feature_extractor.extractFeatures(
            task_fingerprint,
            gpu_device,
            system_state
        );
        
        // Bayesian classification with uncertainty
        const bayesian_result = self.bayesian_classifier.classify(features);
        
        // Start performance measurement
        const measurement_context = self.performance_profiler.startMeasurement();
        
        // Create adaptive classification result
        const classification = AdaptiveClassification{
            .use_gpu = bayesian_result.use_gpu,
            .probability = bayesian_result.probability,
            .confidence = bayesian_result.confidence,
            .epistemic_uncertainty = bayesian_result.epistemic_uncertainty,
            .aleatory_uncertainty = bayesian_result.aleatory_uncertainty,
            .confidence_interval = bayesian_result.confidence_interval,
            .decision_strategy = bayesian_result.decision_strategy,
            .features = features,
            .measurement_context = measurement_context,
            .timestamp = @intCast(std.time.milliTimestamp()),
        };
        
        return classification;
    }
    
    /// Provide feedback on classification outcome
    pub fn provideFeedback(
        self: *Self,
        classification: AdaptiveClassification,
        actual_device_used: ml_classifier.DeviceType,
        actual_performance: performance_profiler.PerformanceMeasurement,
    ) !void {
        // Record performance measurement
        const measurement = try self.performance_profiler.recordMeasurement(
            classification.measurement_context,
            actual_device_used
        );
        
        // Calculate performance comparison
        const comparison = self.performance_profiler.compareCPUvsGPU(
            // Create fingerprint from features (simplified)
            self.createFingerprintFromFeatures(classification.features)
        );
        
        // Determine if GPU was actually better
        const gpu_was_better = comparison.performance_ratio < 1.0 and comparison.energy_ratio < 1.5;
        
        // Update Bayesian classifier
        self.bayesian_classifier.updateWithOutcome(gpu_was_better, comparison.performance_ratio);
        
        // Update feature extractor with performance data
        const cpu_time = if (actual_device_used == .cpu) @as(f32, @floatFromInt(measurement.execution_time_ns)) / 1_000_000.0 else 0.0;
        const gpu_time = if (actual_device_used == .gpu) @as(f32, @floatFromInt(measurement.execution_time_ns)) / 1_000_000.0 else 0.0;
        try self.feature_extractor.updatePerformanceHistory(cpu_time, gpu_time, actual_device_used);
        
        // Store outcome for trend analysis
        const outcome = ClassificationOutcome{
            .predicted_gpu = classification.use_gpu,
            .actual_gpu_better = gpu_was_better,
            .confidence = classification.confidence,
            .performance_ratio = comparison.performance_ratio,
            .timestamp = classification.timestamp,
        };
        
        try self.recent_decisions.append(outcome);
        
        // Maintain window size
        if (self.recent_decisions.items.len > self.feedback_window_size) {
            _ = self.recent_decisions.orderedRemove(0);
        }
        
        // Update feedback metrics
        self.feedback_metrics.update(outcome);
        
        // Periodic adaptation
        const current_time = @as(u64, @intCast(std.time.milliTimestamp()));
        if (current_time - self.last_adaptation_time > self.adaptation_interval_ms) {
            self.performAdaptation();
            self.last_adaptation_time = current_time;
        }
    }
    
    /// Perform periodic model adaptation based on recent feedback
    fn performAdaptation(self: *Self) void {
        if (self.recent_decisions.items.len < self.min_feedback_samples) {
            return;
        }
        
        // Analyze recent performance trends
        _ = self.analyzePerformanceTrend(); // For future use
        
        // Adapt confidence threshold based on accuracy
        if (self.feedback_metrics.recent_accuracy < 0.7) {
            // Low accuracy - increase exploration
            self.bayesian_classifier.exploration_rate = std.math.min(0.3, self.bayesian_classifier.exploration_rate * 1.1);
        } else if (self.feedback_metrics.recent_accuracy > 0.9) {
            // High accuracy - reduce exploration
            self.bayesian_classifier.exploration_rate = std.math.max(0.01, self.bayesian_classifier.exploration_rate * 0.9);
        }
        
        // Adapt based on confidence calibration
        const confidence_error = self.calculateConfidenceCalibrationError();
        if (confidence_error > 0.1) {
            // Poor calibration - adjust confidence threshold
            self.bayesian_classifier.adaptConfidenceThreshold(1.0 + confidence_error);
        }
        
        // Update feedback metrics
        self.feedback_metrics.last_adaptation_time = @intCast(std.time.milliTimestamp());
    }
    
    /// Analyze recent performance trends
    fn analyzePerformanceTrend(self: *const Self) PerformanceTrendAnalysis {
        if (self.recent_decisions.items.len < 10) {
            return PerformanceTrendAnalysis{
                .trend_direction = 0.0,
                .trend_strength = 0.0,
                .confidence = 0.0,
            };
        }
        
        // Simple linear regression on performance ratios
        var sum_x: f64 = 0.0;
        var sum_y: f64 = 0.0;
        var sum_xy: f64 = 0.0;
        var sum_x2: f64 = 0.0;
        const n = @as(f64, @floatFromInt(self.recent_decisions.items.len));
        
        for (self.recent_decisions.items, 0..) |outcome, i| {
            const x = @as(f64, @floatFromInt(i));
            const y = outcome.performance_ratio;
            
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
        
        // Calculate slope (trend direction)
        const slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        
        // Calculate correlation coefficient (trend strength)
        const mean_x = sum_x / n;
        const mean_y = sum_y / n;
        
        var ss_xy: f64 = 0.0;
        var ss_xx: f64 = 0.0;
        var ss_yy: f64 = 0.0;
        
        for (self.recent_decisions.items, 0..) |outcome, i| {
            const x = @as(f64, @floatFromInt(i));
            const y = outcome.performance_ratio;
            
            ss_xy += (x - mean_x) * (y - mean_y);
            ss_xx += (x - mean_x) * (x - mean_x);
            ss_yy += (y - mean_y) * (y - mean_y);
        }
        
        const correlation = ss_xy / std.math.sqrt(ss_xx * ss_yy);
        
        return PerformanceTrendAnalysis{
            .trend_direction = slope,
            .trend_strength = std.math.fabs(correlation),
            .confidence = std.math.min(1.0, n / 50.0), // Confidence increases with sample size
        };
    }
    
    /// Calculate confidence calibration error
    fn calculateConfidenceCalibrationError(self: *const Self) f64 {
        if (self.recent_decisions.items.len < 20) {
            return 0.0;
        }
        
        // Bin predictions by confidence and check calibration
        const bin_count = 5;
        var bins: [bin_count]CalibrationBin = [_]CalibrationBin{CalibrationBin.init()} ** bin_count;
        
        for (self.recent_decisions.items) |outcome| {
            const bin_index = @as(usize, @intFromFloat(outcome.confidence * @as(f64, @floatFromInt(bin_count - 1))));
            bins[bin_index].add(outcome);
        }
        
        // Calculate expected calibration error
        var total_error: f64 = 0.0;
        var total_samples: u32 = 0;
        
        for (bins) |bin| {
            if (bin.count > 0) {
                const accuracy = @as(f64, @floatFromInt(bin.correct)) / @as(f64, @floatFromInt(bin.count));
                const avg_confidence = bin.confidence_sum / @as(f64, @floatFromInt(bin.count));
                const calibration_error = std.math.fabs(accuracy - avg_confidence);
                
                total_error += calibration_error * @as(f64, @floatFromInt(bin.count));
                total_samples += bin.count;
            }
        }
        
        return if (total_samples > 0) total_error / @as(f64, @floatFromInt(total_samples)) else 0.0;
    }
    
    /// Create simplified fingerprint from ML features (for compatibility)
    fn createFingerprintFromFeatures(self: *const Self, features: ml_classifier.MLTaskFeatures) fingerprint.TaskFingerprint {
        _ = self;
        return fingerprint.TaskFingerprint{
            .call_site_hash = 0,
            .data_size_class = @intFromFloat(features.data_size_log2 * 32.0),
            .data_alignment = 0,
            .access_pattern = .sequential,
            .simd_width = 0,
            .cache_locality = @intFromFloat(features.memory_access_locality * 15.0),
            .numa_node_hint = 0,
            .cpu_intensity = @intFromFloat(features.computational_intensity * 15.0),
            .parallel_potential = @intFromFloat(features.parallelization_potential * 15.0),
            .execution_phase = 0,
            .priority_class = 0,
            .time_sensitivity = @intFromFloat(features.latency_sensitivity * 3.0),
            .dependency_count = 0,
            .time_of_day_bucket = @intFromFloat(features.time_of_day_normalized * 23.0),
            .execution_frequency = @intFromFloat(features.workload_frequency * 7.0),
            .seasonal_pattern = 0,
            .variance_level = @intFromFloat(features.execution_time_variance * 15.0),
            .expected_cycles_log2 = @intFromFloat(features.computational_intensity * 32.0),
            .memory_footprint_log2 = @intFromFloat(features.memory_footprint_log2 * 32.0),
            .io_intensity = 0,
            .cache_miss_rate = 0,
            .branch_predictability = 0,
            .vectorization_benefit = @intFromFloat(features.vectorization_suitability * 15.0),
        };
    }
    
    /// Get system statistics
    pub fn getSystemStatistics(self: *const Self) AdaptiveSystemStatistics {
        const bayesian_stats = self.bayesian_classifier.getStatistics();
        
        return AdaptiveSystemStatistics{
            .bayesian_stats = bayesian_stats,
            .feedback_metrics = self.feedback_metrics,
            .total_classifications = self.recent_decisions.items.len,
            .adaptation_frequency = if (self.feedback_metrics.last_adaptation_time > 0)
                @as(f64, @floatFromInt(@as(u64, @intCast(std.time.milliTimestamp())) - self.feedback_metrics.last_adaptation_time)) / 1000.0
            else 0.0,
        };
    }
};

/// Adaptive classification result
pub const AdaptiveClassification = struct {
    use_gpu: bool,
    probability: f64,
    confidence: f64,
    epistemic_uncertainty: f64,
    aleatory_uncertainty: f64,
    confidence_interval: CredibleInterval,
    decision_strategy: DecisionStrategy,
    features: ml_classifier.MLTaskFeatures,
    measurement_context: performance_profiler.MeasurementContext,
    timestamp: u64,
};

/// Classification outcome for feedback analysis
const ClassificationOutcome = struct {
    predicted_gpu: bool,
    actual_gpu_better: bool,
    confidence: f64,
    performance_ratio: f64,
    timestamp: u64,
};

/// Feedback metrics tracking
const FeedbackMetrics = struct {
    total_feedback: u64,
    correct_predictions: u64,
    recent_accuracy: f64,
    confidence_calibration_error: f64,
    last_adaptation_time: u64,
    
    fn init() FeedbackMetrics {
        return FeedbackMetrics{
            .total_feedback = 0,
            .correct_predictions = 0,
            .recent_accuracy = 0.0,
            .confidence_calibration_error = 0.0,
            .last_adaptation_time = 0,
        };
    }
    
    fn update(self: *FeedbackMetrics, outcome: ClassificationOutcome) void {
        self.total_feedback += 1;
        
        if ((outcome.predicted_gpu and outcome.actual_gpu_better) or 
            (!outcome.predicted_gpu and !outcome.actual_gpu_better)) {
            self.correct_predictions += 1;
        }
        
        self.recent_accuracy = @as(f64, @floatFromInt(self.correct_predictions)) / @as(f64, @floatFromInt(self.total_feedback));
    }
};

/// Performance trend analysis result
const PerformanceTrendAnalysis = struct {
    trend_direction: f64,    // Positive = improving, negative = degrading
    trend_strength: f64,     // 0-1, how strong the trend is
    confidence: f64,         // 0-1, confidence in trend analysis
};

/// Calibration bin for confidence calibration analysis
const CalibrationBin = struct {
    count: u32,
    correct: u32,
    confidence_sum: f64,
    
    fn init() CalibrationBin {
        return CalibrationBin{
            .count = 0,
            .correct = 0,
            .confidence_sum = 0.0,
        };
    }
    
    fn add(self: *CalibrationBin, outcome: ClassificationOutcome) void {
        self.count += 1;
        self.confidence_sum += outcome.confidence;
        
        if ((outcome.predicted_gpu and outcome.actual_gpu_better) or 
            (!outcome.predicted_gpu and !outcome.actual_gpu_better)) {
            self.correct += 1;
        }
    }
};

/// Combined system statistics
pub const AdaptiveSystemStatistics = struct {
    bayesian_stats: BayesianStatistics,
    feedback_metrics: FeedbackMetrics,
    total_classifications: usize,
    adaptation_frequency: f64,
};

// ============================================================================
// Test Utilities
// ============================================================================

test "Bayesian GPU classifier uncertainty quantification" {
    const allocator = std.testing.allocator;
    
    var classifier = BayesianGPUClassifier.init(allocator, 0.5); // Neutral prior
    
    // Create test features
    const features = ml_classifier.MLTaskFeatures{
        .data_size_log2 = 0.8,
        .computational_intensity = 0.9,
        .memory_footprint_log2 = 0.7,
        .parallelization_potential = 0.9,
        .vectorization_suitability = 0.8,
        .memory_access_locality = 0.7,
        .system_load_cpu = 0.6,
        .system_load_gpu = 0.3,
        .available_cpu_cores = 0.8,
        .available_gpu_memory = 0.9,
        .numa_locality_hint = 0.0,
        .power_budget_constraint = 1.0,
        .latency_sensitivity = 0.2,
        .throughput_priority = 0.8,
        .time_of_day_normalized = 0.5,
        .workload_frequency = 0.3,
        .recent_performance_trend = 0.2,
        .execution_time_variance = 0.1,
        .resource_contention_history = 0.1,
        .thermal_state = 0.9,
        .gpu_compute_capability = 0.8,
        .memory_bandwidth_ratio = 0.9,
        .device_specialization_match = 0.9,
        .energy_efficiency_ratio = 0.8,
    };
    
    // Initial classification should have high uncertainty
    const initial_result = classifier.classify(features);
    try std.testing.expect(initial_result.epistemic_uncertainty > 0.1);
    try std.testing.expect(initial_result.confidence_interval.upper > initial_result.confidence_interval.lower);
    
    // Update with positive outcome
    classifier.updateWithOutcome(true, 0.7); // GPU was better (ratio < 1.0)
    
    // After update, should have slightly less uncertainty
    const updated_result = classifier.classify(features);
    try std.testing.expect(updated_result.epistemic_uncertainty <= initial_result.epistemic_uncertainty);
    
    // Verify statistics
    const stats = classifier.getStatistics();
    try std.testing.expect(stats.total_predictions == 2);
    try std.testing.expect(stats.posterior_mean > 0.5); // Should favor GPU after positive feedback
}

test "adaptive learning system integration" {
    const allocator = std.testing.allocator;
    
    var learning_system = AdaptiveLearningSystem.init(allocator, 0.4); // Slight CPU bias initially
    defer learning_system.deinit();
    
    // Create test fingerprint
    const test_fingerprint = fingerprint.TaskFingerprint{
        .call_site_hash = 0x12345678,
        .data_size_class = 24, // Large data
        .data_alignment = 8,
        .access_pattern = .sequential,
        .simd_width = 8,
        .cache_locality = 10,
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
    
    const system_state = ml_classifier.SystemState.getCurrentState();
    
    // Make classification
    const classification = try learning_system.classifyTask(test_fingerprint, null, system_state);
    
    // Verify classification structure
    try std.testing.expect(classification.confidence >= 0.0 and classification.confidence <= 1.0);
    try std.testing.expect(classification.epistemic_uncertainty >= 0.0);
    try std.testing.expect(classification.aleatory_uncertainty >= 0.0);
    
    // Simulate feedback with GPU being better
    const mock_measurement = performance_profiler.PerformanceMeasurement{
        .execution_time_ns = 500_000, // 0.5ms
        .cpu_cycles = 1_500_000,
        .memory_accesses = 2000,
        .cache_misses = 50,
        .instructions_executed = 20_000,
        .power_consumption_mw = 80_000,
        .temperature_celsius = 70,
        .memory_bandwidth_used = 0.8,
        .cpu_utilization = 0.2,
        .gpu_utilization = 0.9,
    };
    
    try learning_system.provideFeedback(classification, .gpu, mock_measurement);
    
    // Get system statistics
    const stats = learning_system.getSystemStatistics();
    try std.testing.expect(stats.total_classifications >= 0);
    try std.testing.expect(stats.bayesian_stats.total_predictions > 0);
}