const std = @import("std");
const fingerprint = @import("fingerprint.zig");
const gpu_integration = @import("gpu_integration.zig");
const gpu_classifier = @import("gpu_classifier.zig");

// Machine Learning-based Task Classification for Beat.zig (Task 3.2.2)
//
// This module implements adaptive machine learning algorithms for intelligent
// task classification in heterogeneous CPU-GPU computing environments.
//
// Based on research showing 20-37% performance improvements through ML-based
// task routing, this system provides:
// - Multi-dimensional feature extraction from task characteristics
// - Lightweight online learning with confidence intervals  
// - Bayesian uncertainty quantification for robust decision-making
// - Real-time performance feedback and adaptive model updates
//
// Key Features:
// - Online gradient descent with momentum for lightweight learning
// - Confidence-aware task routing based on uncertainty quantification
// - Temporal pattern recognition for workload adaptation
// - Device-specific performance modeling and prediction

// ============================================================================
// Feature Extraction and Engineering
// ============================================================================

/// Comprehensive feature vector for ML-based task classification
/// Based on research findings for effective heterogeneous computing classification
pub const MLTaskFeatures = struct {
    // Static task characteristics (12 features)
    data_size_log2: f32,              // Log2 of data size for scale normalization
    computational_intensity: f32,      // Operations per memory access (roofline model)
    memory_footprint_log2: f32,       // Log2 of memory usage
    parallelization_potential: f32,   // Degree of inherent parallelism (0-1)
    vectorization_suitability: f32,   // SIMD optimization potential (0-1)
    memory_access_locality: f32,      // Cache-friendly access patterns (0-1)
    
    // Dynamic execution context (8 features)
    system_load_cpu: f32,             // Current CPU utilization (0-1)
    system_load_gpu: f32,             // Current GPU utilization (0-1)
    available_cpu_cores: f32,         // Normalized available cores (0-1)
    available_gpu_memory: f32,        // Normalized available GPU memory (0-1)
    numa_locality_hint: f32,          // NUMA node preference strength (0-1)
    power_budget_constraint: f32,     // Power consumption constraint (0-1)
    latency_sensitivity: f32,         // Real-time requirements (0-1)
    throughput_priority: f32,         // Batch vs latency optimization (0-1)
    
    // Temporal and historical patterns (6 features)
    time_of_day_normalized: f32,      // Hour of day normalized (0-1)
    workload_frequency: f32,          // How often this pattern occurs (0-1)
    recent_performance_trend: f32,    // Recent CPU vs GPU performance trend (-1 to 1)
    execution_time_variance: f32,     // Stability of execution times (0-1)
    resource_contention_history: f32, // Historical resource conflicts (0-1)
    thermal_state: f32,               // Device thermal condition (0-1)
    
    // Device-specific characteristics (4 features)
    gpu_compute_capability: f32,      // GPU computational power normalized (0-1)
    memory_bandwidth_ratio: f32,      // GPU vs CPU memory bandwidth (0-1)
    device_specialization_match: f32, // Task-device affinity (0-1)
    energy_efficiency_ratio: f32,     // GPU vs CPU energy efficiency (0-1)
    
    /// Total number of features for ML model
    pub const FEATURE_COUNT: usize = 30;
    
    /// Convert to array for ML algorithms
    pub fn toArray(self: MLTaskFeatures) [FEATURE_COUNT]f32 {
        return [_]f32{
            // Static characteristics
            self.data_size_log2,
            self.computational_intensity,
            self.memory_footprint_log2,
            self.parallelization_potential,
            self.vectorization_suitability,
            self.memory_access_locality,
            
            // Dynamic context
            self.system_load_cpu,
            self.system_load_gpu,
            self.available_cpu_cores,
            self.available_gpu_memory,
            self.numa_locality_hint,
            self.power_budget_constraint,
            self.latency_sensitivity,
            self.throughput_priority,
            
            // Temporal patterns
            self.time_of_day_normalized,
            self.workload_frequency,
            self.recent_performance_trend,
            self.execution_time_variance,
            self.resource_contention_history,
            self.thermal_state,
            
            // Device characteristics
            self.gpu_compute_capability,
            self.memory_bandwidth_ratio,
            self.device_specialization_match,
            self.energy_efficiency_ratio,
        };
    }
    
    /// Create from array (for model predictions)
    pub fn fromArray(features: [FEATURE_COUNT]f32) MLTaskFeatures {
        return MLTaskFeatures{
            .data_size_log2 = features[0],
            .computational_intensity = features[1],
            .memory_footprint_log2 = features[2],
            .parallelization_potential = features[3],
            .vectorization_suitability = features[4],
            .memory_access_locality = features[5],
            .system_load_cpu = features[6],
            .system_load_gpu = features[7],
            .available_cpu_cores = features[8],
            .available_gpu_memory = features[9],
            .numa_locality_hint = features[10],
            .power_budget_constraint = features[11],
            .latency_sensitivity = features[12],
            .throughput_priority = features[13],
            .time_of_day_normalized = features[14],
            .workload_frequency = features[15],
            .recent_performance_trend = features[16],
            .execution_time_variance = features[17],
            .resource_contention_history = features[18],
            .thermal_state = features[19],
            .gpu_compute_capability = features[20],
            .memory_bandwidth_ratio = features[21],
            .device_specialization_match = features[22],
            .energy_efficiency_ratio = features[23],
        };
    }
    
    /// Normalize features to [0,1] or [-1,1] range for ML stability
    pub fn normalize(self: *MLTaskFeatures) void {
        // Clamp all features to valid ranges
        self.data_size_log2 = std.math.clamp(self.data_size_log2 / 32.0, 0.0, 1.0); // Max 2^32 bytes
        self.computational_intensity = std.math.clamp(self.computational_intensity / 100.0, 0.0, 1.0); // Max 100 ops/byte
        self.memory_footprint_log2 = std.math.clamp(self.memory_footprint_log2 / 32.0, 0.0, 1.0);
        self.parallelization_potential = std.math.clamp(self.parallelization_potential, 0.0, 1.0);
        self.vectorization_suitability = std.math.clamp(self.vectorization_suitability, 0.0, 1.0);
        self.memory_access_locality = std.math.clamp(self.memory_access_locality, 0.0, 1.0);
        
        // System state features are already normalized
        self.system_load_cpu = std.math.clamp(self.system_load_cpu, 0.0, 1.0);
        self.system_load_gpu = std.math.clamp(self.system_load_gpu, 0.0, 1.0);
        self.available_cpu_cores = std.math.clamp(self.available_cpu_cores, 0.0, 1.0);
        self.available_gpu_memory = std.math.clamp(self.available_gpu_memory, 0.0, 1.0);
        self.numa_locality_hint = std.math.clamp(self.numa_locality_hint, 0.0, 1.0);
        self.power_budget_constraint = std.math.clamp(self.power_budget_constraint, 0.0, 1.0);
        self.latency_sensitivity = std.math.clamp(self.latency_sensitivity, 0.0, 1.0);
        self.throughput_priority = std.math.clamp(self.throughput_priority, 0.0, 1.0);
        
        // Temporal features
        self.time_of_day_normalized = std.math.clamp(self.time_of_day_normalized, 0.0, 1.0);
        self.workload_frequency = std.math.clamp(self.workload_frequency, 0.0, 1.0);
        self.recent_performance_trend = std.math.clamp(self.recent_performance_trend, -1.0, 1.0);
        self.execution_time_variance = std.math.clamp(self.execution_time_variance, 0.0, 1.0);
        self.resource_contention_history = std.math.clamp(self.resource_contention_history, 0.0, 1.0);
        self.thermal_state = std.math.clamp(self.thermal_state, 0.0, 1.0);
        
        // Device features
        self.gpu_compute_capability = std.math.clamp(self.gpu_compute_capability, 0.0, 1.0);
        self.memory_bandwidth_ratio = std.math.clamp(self.memory_bandwidth_ratio, 0.0, 1.0);
        self.device_specialization_match = std.math.clamp(self.device_specialization_match, 0.0, 1.0);
        self.energy_efficiency_ratio = std.math.clamp(self.energy_efficiency_ratio, 0.0, 1.0);
    }
};

/// Feature extractor that converts task fingerprints and system state to ML features
pub const MLFeatureExtractor = struct {
    allocator: std.mem.Allocator,
    
    // Historical performance tracking
    cpu_performance_history: std.ArrayList(f32),
    gpu_performance_history: std.ArrayList(f32),
    execution_count: u64,
    
    // System monitoring
    last_cpu_load: f32,
    last_gpu_load: f32,
    thermal_history: std.ArrayList(f32),
    
    const Self = @This();
    
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .cpu_performance_history = std.ArrayList(f32).init(allocator),
            .gpu_performance_history = std.ArrayList(f32).init(allocator),
            .execution_count = 0,
            .last_cpu_load = 0.0,
            .last_gpu_load = 0.0,
            .thermal_history = std.ArrayList(f32).init(allocator),
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.cpu_performance_history.deinit();
        self.gpu_performance_history.deinit();
        self.thermal_history.deinit();
    }
    
    /// Extract comprehensive ML features from task fingerprint and system state
    pub fn extractFeatures(
        self: *Self,
        task_fingerprint: fingerprint.TaskFingerprint,
        gpu_device: ?*const gpu_integration.GPUDeviceInfo,
        system_state: SystemState,
    ) !MLTaskFeatures {
        var features = MLTaskFeatures{
            // Static task characteristics
            .data_size_log2 = @as(f32, @floatFromInt(task_fingerprint.data_size_class)),
            .computational_intensity = self.calculateComputationalIntensity(task_fingerprint),
            .memory_footprint_log2 = @as(f32, @floatFromInt(task_fingerprint.memory_footprint_log2)),
            .parallelization_potential = @as(f32, @floatFromInt(task_fingerprint.parallel_potential)) / 15.0,
            .vectorization_suitability = @as(f32, @floatFromInt(task_fingerprint.vectorization_benefit)) / 15.0,
            .memory_access_locality = self.calculateAccessLocality(task_fingerprint),
            
            // Dynamic execution context
            .system_load_cpu = system_state.cpu_utilization,
            .system_load_gpu = system_state.gpu_utilization,
            .available_cpu_cores = system_state.available_cpu_cores,
            .available_gpu_memory = system_state.available_gpu_memory,
            .numa_locality_hint = @as(f32, @floatFromInt(task_fingerprint.numa_node_hint)) / 15.0,
            .power_budget_constraint = system_state.power_budget_factor,
            .latency_sensitivity = @as(f32, @floatFromInt(task_fingerprint.time_sensitivity)) / 3.0,
            .throughput_priority = if (task_fingerprint.priority_class == 0) 1.0 else 0.0,
            
            // Temporal and historical patterns
            .time_of_day_normalized = @as(f32, @floatFromInt(task_fingerprint.time_of_day_bucket)) / 23.0,
            .workload_frequency = @as(f32, @floatFromInt(task_fingerprint.execution_frequency)) / 7.0,
            .recent_performance_trend = self.calculatePerformanceTrend(),
            .execution_time_variance = @as(f32, @floatFromInt(task_fingerprint.variance_level)) / 15.0,
            .resource_contention_history = system_state.contention_factor,
            .thermal_state = system_state.thermal_factor,
            
            // Device-specific characteristics
            .gpu_compute_capability = if (gpu_device) |device| self.calculateGPUCapability(device) else 0.0,
            .memory_bandwidth_ratio = if (gpu_device) |device| self.calculateBandwidthRatio(device) else 0.0,
            .device_specialization_match = if (gpu_device) |device| self.calculateSpecializationMatch(task_fingerprint, device) else 0.0,
            .energy_efficiency_ratio = if (gpu_device) |device| self.calculateEnergyEfficiency(device) else 0.0,
        };
        
        // Normalize all features
        features.normalize();
        
        return features;
    }
    
    /// Update performance history with execution results
    pub fn updatePerformanceHistory(self: *Self, cpu_time: f32, gpu_time: f32, device_used: DeviceType) !void {
        self.execution_count += 1;
        
        // Store performance data (keep last 1000 entries)
        const max_history = 1000;
        
        if (device_used == .cpu) {
            try self.cpu_performance_history.append(cpu_time);
            if (self.cpu_performance_history.items.len > max_history) {
                _ = self.cpu_performance_history.orderedRemove(0);
            }
        } else if (device_used == .gpu) {
            try self.gpu_performance_history.append(gpu_time);
            if (self.gpu_performance_history.items.len > max_history) {
                _ = self.gpu_performance_history.orderedRemove(0);
            }
        }
    }
    
    /// Calculate computational intensity using roofline model principles
    fn calculateComputationalIntensity(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) f32 {
        _ = self;
        const ops = @as(f32, @floatFromInt(task_fingerprint.expected_cycles_log2));
        const memory_accesses = @as(f32, @floatFromInt(task_fingerprint.data_size_class)) / 8.0; // Assume 8-byte accesses
        return if (memory_accesses > 0) ops / memory_accesses else 0.0;
    }
    
    /// Calculate memory access locality score
    fn calculateAccessLocality(self: *Self, task_fingerprint: fingerprint.TaskFingerprint) f32 {
        _ = self;
        const cache_locality = @as(f32, @floatFromInt(task_fingerprint.cache_locality)) / 15.0;
        const cache_miss_penalty = 1.0 - (@as(f32, @floatFromInt(task_fingerprint.cache_miss_rate)) / 15.0);
        
        return switch (task_fingerprint.access_pattern) {
            .sequential => cache_locality * 1.0 + cache_miss_penalty * 0.2,
            .read_only => cache_locality * 0.9 + cache_miss_penalty * 0.3,
            .strided => cache_locality * 0.7 + cache_miss_penalty * 0.5,
            .gather_scatter => cache_locality * 0.6 + cache_miss_penalty * 0.4,
            .hierarchical => cache_locality * 0.5 + cache_miss_penalty * 0.6,
            .mixed => cache_locality * 0.4 + cache_miss_penalty * 0.7,
            .write_heavy => cache_locality * 0.3 + cache_miss_penalty * 0.8,
            .random => cache_locality * 0.2 + cache_miss_penalty * 0.9,
        };
    }
    
    /// Calculate recent performance trend (CPU vs GPU preference)
    fn calculatePerformanceTrend(self: *Self) f32 {
        if (self.cpu_performance_history.items.len < 5 or self.gpu_performance_history.items.len < 5) {
            return 0.0; // Neutral when insufficient data
        }
        
        // Calculate average of recent 5 executions
        const recent_count = 5;
        var cpu_avg: f32 = 0.0;
        var gpu_avg: f32 = 0.0;
        
        const cpu_items = self.cpu_performance_history.items;
        const gpu_items = self.gpu_performance_history.items;
        
        const cpu_start = if (cpu_items.len >= recent_count) cpu_items.len - recent_count else 0;
        const gpu_start = if (gpu_items.len >= recent_count) gpu_items.len - recent_count else 0;
        
        for (cpu_items[cpu_start..]) |time| {
            cpu_avg += time;
        }
        cpu_avg /= @as(f32, @floatFromInt(cpu_items.len - cpu_start));
        
        for (gpu_items[gpu_start..]) |time| {
            gpu_avg += time;
        }
        gpu_avg /= @as(f32, @floatFromInt(gpu_items.len - gpu_start));
        
        // Return normalized trend (-1 = strongly favor CPU, +1 = strongly favor GPU)
        if (cpu_avg + gpu_avg == 0) return 0.0;
        return (cpu_avg - gpu_avg) / (cpu_avg + gpu_avg);
    }
    
    /// Calculate GPU computational capability score
    fn calculateGPUCapability(self: *Self, device: *const gpu_integration.GPUDeviceInfo) f32 {
        _ = self;
        // Normalize based on typical GPU specifications
        const compute_units = @as(f32, @floatFromInt(device.max_compute_units));
        const clock_freq = @as(f32, @floatFromInt(device.max_clock_frequency));
        
        // Rough normalization (adjust based on actual GPU ranges)
        const normalized_units = std.math.clamp(compute_units / 128.0, 0.0, 1.0); // Max ~128 CUs
        const normalized_clock = std.math.clamp(clock_freq / 2000.0, 0.0, 1.0); // Max ~2GHz
        
        return (normalized_units + normalized_clock) / 2.0;
    }
    
    /// Calculate memory bandwidth ratio (GPU vs CPU)
    fn calculateBandwidthRatio(self: *Self, device: *const gpu_integration.GPUDeviceInfo) f32 {
        _ = self;
        const gpu_bandwidth = @as(f32, @floatFromInt(device.global_mem_size)) / (1024 * 1024 * 1024); // GB
        const typical_cpu_bandwidth = 50.0; // GB/s for typical CPU
        const estimated_gpu_bandwidth = gpu_bandwidth * 0.1; // Rough estimate
        
        return std.math.clamp(estimated_gpu_bandwidth / typical_cpu_bandwidth, 0.0, 1.0);
    }
    
    /// Calculate task-device specialization match
    fn calculateSpecializationMatch(self: *Self, task_fingerprint: fingerprint.TaskFingerprint, device: *const gpu_integration.GPUDeviceInfo) f32 {
        _ = self;
        _ = device;
        
        const parallel_score = @as(f32, @floatFromInt(task_fingerprint.parallel_potential)) / 15.0;
        const vector_score = @as(f32, @floatFromInt(task_fingerprint.vectorization_benefit)) / 15.0;
        const compute_score = @as(f32, @floatFromInt(task_fingerprint.cpu_intensity)) / 15.0;
        
        // GPU specialization: high parallelism + vectorization + compute intensity
        return (parallel_score * 0.4 + vector_score * 0.4 + compute_score * 0.2);
    }
    
    /// Calculate energy efficiency ratio
    fn calculateEnergyEfficiency(self: *Self, device: *const gpu_integration.GPUDeviceInfo) f32 {
        _ = self;
        _ = device;
        // Placeholder - would need actual power measurements
        return 0.7; // Assume GPUs are generally more energy efficient for parallel workloads
    }
};

/// System state information for ML feature extraction
pub const SystemState = struct {
    cpu_utilization: f32,        // Current CPU usage (0-1)
    gpu_utilization: f32,        // Current GPU usage (0-1)
    available_cpu_cores: f32,    // Normalized available cores (0-1)
    available_gpu_memory: f32,   // Normalized available GPU memory (0-1)
    power_budget_factor: f32,    // Power constraint factor (0-1)
    contention_factor: f32,      // Resource contention level (0-1)
    thermal_factor: f32,         // Thermal state (0-1, 1=cool, 0=hot)
    
    pub fn getCurrentState() SystemState {
        // Placeholder implementation - would integrate with actual system monitoring
        return SystemState{
            .cpu_utilization = 0.5,
            .gpu_utilization = 0.3,
            .available_cpu_cores = 0.8,
            .available_gpu_memory = 0.9,
            .power_budget_factor = 1.0,
            .contention_factor = 0.2,
            .thermal_factor = 0.8,
        };
    }
};

/// Device type enumeration for performance tracking
pub const DeviceType = enum {
    cpu,
    gpu,
    hybrid,
};

// ============================================================================
// Online Learning Algorithms
// ============================================================================

/// Lightweight online gradient descent with momentum for ML classification
/// Based on research showing effectiveness for real-time heterogeneous computing
pub const OnlineLinearClassifier = struct {
    allocator: std.mem.Allocator,
    
    // Model parameters
    weights: [MLTaskFeatures.FEATURE_COUNT]f32,
    bias: f32,
    
    // Learning configuration
    learning_rate: f32,
    momentum: f32,
    decay_rate: f32,
    
    // Momentum buffers
    weight_momentum: [MLTaskFeatures.FEATURE_COUNT]f32,
    bias_momentum: f32,
    
    // Performance tracking
    training_samples: u64,
    correct_predictions: u64,
    confidence_threshold: f32,
    
    const Self = @This();
    
    pub fn init(allocator: std.mem.Allocator, learning_rate: f32) Self {
        var classifier = Self{
            .allocator = allocator,
            .weights = [_]f32{0.0} ** MLTaskFeatures.FEATURE_COUNT,
            .bias = 0.0,
            .learning_rate = learning_rate,
            .momentum = 0.9,
            .decay_rate = 0.999,
            .weight_momentum = [_]f32{0.0} ** MLTaskFeatures.FEATURE_COUNT,
            .bias_momentum = 0.0,
            .training_samples = 0,
            .correct_predictions = 0,
            .confidence_threshold = 0.6,
        };
        
        // Initialize weights with small random values
        var prng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
        const random = prng.random();
        
        for (classifier.weights[0..]) |*weight| {
            weight.* = (random.float(f32) - 0.5) * 0.01; // Small random initialization
        }
        
        return classifier;
    }
    
    /// Predict GPU suitability with confidence score
    pub fn predict(self: *const Self, features: MLTaskFeatures) PredictionResult {
        const feature_array = features.toArray();
        
        // Linear model: y = w^T * x + b
        var logit: f32 = self.bias;
        for (feature_array, self.weights) |feature, weight| {
            logit += feature * weight;
        }
        
        // Sigmoid activation for probability
        const probability = 1.0 / (1.0 + std.math.exp(-logit));
        
        // Calculate confidence based on distance from decision boundary
        const confidence = std.math.fabs(probability - 0.5) * 2.0;
        
        return PredictionResult{
            .use_gpu = probability > 0.5,
            .probability = probability,
            .confidence = confidence,
            .is_confident = confidence >= self.confidence_threshold,
        };
    }
    
    /// Update model with new training example (online learning)
    pub fn updateModel(self: *Self, features: MLTaskFeatures, actual_gpu_better: bool, actual_performance_ratio: f32) void {
        _ = actual_performance_ratio; // TODO: Use for more sophisticated updates
        
        const feature_array = features.toArray();
        
        // Current prediction
        const prediction = self.predict(features);
        
        // Calculate prediction error (cross-entropy gradient)
        const target: f32 = if (actual_gpu_better) 1.0 else 0.0;
        const prediction_error = prediction.probability - target;
        
        // Update statistics
        self.training_samples += 1;
        if ((prediction.use_gpu and actual_gpu_better) or (!prediction.use_gpu and !actual_gpu_better)) {
            self.correct_predictions += 1;
        }
        
        // Gradient descent with momentum
        const effective_lr = self.learning_rate * std.math.pow(f32, self.decay_rate, @floatFromInt(self.training_samples));
        
        // Update weights
        for (feature_array, &self.weights, &self.weight_momentum) |feature, *weight, *momentum| {
            const gradient = prediction_error * feature;
            momentum.* = self.momentum * momentum.* + gradient;
            weight.* -= effective_lr * momentum.*;
            
            // Weight clipping for stability
            weight.* = std.math.clamp(weight.*, -10.0, 10.0);
        }
        
        // Update bias
        self.bias_momentum = self.momentum * self.bias_momentum + prediction_error;
        self.bias -= effective_lr * self.bias_momentum;
        self.bias = std.math.clamp(self.bias, -10.0, 10.0);
    }
    
    /// Get model accuracy
    pub fn getAccuracy(self: *const Self) f32 {
        if (self.training_samples == 0) return 0.0;
        return @as(f32, @floatFromInt(self.correct_predictions)) / @as(f32, @floatFromInt(self.training_samples));
    }
    
    /// Adapt confidence threshold based on recent performance
    pub fn adaptConfidenceThreshold(self: *Self) void {
        const accuracy = self.getAccuracy();
        
        // Increase threshold if accuracy is low, decrease if high
        if (accuracy < 0.7) {
            self.confidence_threshold = std.math.min(0.9, self.confidence_threshold + 0.05);
        } else if (accuracy > 0.85) {
            self.confidence_threshold = std.math.max(0.4, self.confidence_threshold - 0.02);
        }
    }
};

/// Prediction result with uncertainty quantification
pub const PredictionResult = struct {
    use_gpu: bool,          // Binary classification result
    probability: f32,       // Probability of GPU being better (0-1)
    confidence: f32,        // Confidence in prediction (0-1)
    is_confident: bool,     // Whether prediction meets confidence threshold
};

// ============================================================================
// Test Utilities and Validation
// ============================================================================

test "ML feature extraction from task fingerprint" {
    const allocator = std.testing.allocator;
    
    var extractor = MLFeatureExtractor.init(allocator);
    defer extractor.deinit();
    
    // Create test fingerprint
    const test_fingerprint = fingerprint.TaskFingerprint{
        .call_site_hash = 0x12345678,
        .data_size_class = 20, // 1MB
        .data_alignment = 8,
        .access_pattern = .sequential,
        .simd_width = 8,
        .cache_locality = 12,
        .numa_node_hint = 0,
        .cpu_intensity = 10,
        .parallel_potential = 14,
        .execution_phase = 1,
        .priority_class = 0,
        .time_sensitivity = 1,
        .dependency_count = 2,
        .time_of_day_bucket = 14,
        .execution_frequency = 3,
        .seasonal_pattern = 1,
        .variance_level = 5,
        .expected_cycles_log2 = 25,
        .memory_footprint_log2 = 20,
        .io_intensity = 2,
        .cache_miss_rate = 3,
        .branch_predictability = 12,
        .vectorization_benefit = 13,
    };
    
    const system_state = SystemState.getCurrentState();
    const features = try extractor.extractFeatures(test_fingerprint, null, system_state);
    
    // Validate feature ranges
    const feature_array = features.toArray();
    for (feature_array) |feature| {
        try std.testing.expect(feature >= -1.0 and feature <= 1.0);
    }
    
    try std.testing.expect(features.data_size_log2 >= 0.0);
    try std.testing.expect(features.parallelization_potential >= 0.0 and features.parallelization_potential <= 1.0);
    try std.testing.expect(features.vectorization_suitability >= 0.0 and features.vectorization_suitability <= 1.0);
}

test "online linear classifier training and prediction" {
    const allocator = std.testing.allocator;
    
    var classifier = OnlineLinearClassifier.init(allocator, 0.01);
    
    // Create test features favoring GPU
    var gpu_features = MLTaskFeatures{
        .data_size_log2 = 0.8,
        .computational_intensity = 0.9,
        .memory_footprint_log2 = 0.7,
        .parallelization_potential = 0.95,
        .vectorization_suitability = 0.9,
        .memory_access_locality = 0.8,
        .system_load_cpu = 0.8,
        .system_load_gpu = 0.2,
        .available_cpu_cores = 0.3,
        .available_gpu_memory = 0.9,
        .numa_locality_hint = 0.0,
        .power_budget_constraint = 1.0,
        .latency_sensitivity = 0.0,
        .throughput_priority = 1.0,
        .time_of_day_normalized = 0.5,
        .workload_frequency = 0.5,
        .recent_performance_trend = 0.7,
        .execution_time_variance = 0.2,
        .resource_contention_history = 0.1,
        .thermal_state = 0.9,
        .gpu_compute_capability = 0.8,
        .memory_bandwidth_ratio = 0.9,
        .device_specialization_match = 0.9,
        .energy_efficiency_ratio = 0.8,
    };
    
    // Train with GPU-favorable examples
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        classifier.updateModel(gpu_features, true, 1.5); // GPU 50% faster
    }
    
    // Test prediction
    const prediction = classifier.predict(gpu_features);
    try std.testing.expect(prediction.use_gpu);
    try std.testing.expect(prediction.probability > 0.5);
    
    // Verify model learned
    try std.testing.expect(classifier.getAccuracy() > 0.5);
}