const std = @import("std");

// ============================================================================
// Shared Predictive Configuration Module
// Centralizes predictive accounting configuration to prevent duplication and drift
// ============================================================================

/// Configuration for predictive accounting systems
/// Used by both continuation_predictive.zig and continuation_predictive_compat.zig
/// to ensure consistent behavior and prevent configuration drift
pub const PredictiveConfig = struct {
    // ========================================================================
    // One Euro Filter Parameters
    // ========================================================================
    
    /// Minimum cutoff frequency for the One Euro Filter
    /// Lower values = more smoothing, higher values = more responsiveness
    min_cutoff: f32 = 0.1,
    
    /// Beta parameter for the One Euro Filter  
    /// Controls the cutoff slope - higher values react faster to velocity changes
    beta: f32 = 0.05,
    
    /// Derivative cutoff frequency for the One Euro Filter
    /// Controls the smoothing of the velocity signal
    d_cutoff: f32 = 1.0,
    
    // ========================================================================
    // Velocity Filter Parameters (More Stable Secondary Filter)
    // ========================================================================
    
    /// Minimum cutoff frequency for velocity filtering
    /// More conservative than main filter for stability
    velocity_min_cutoff: f32 = 0.05,
    
    /// Beta parameter for velocity filtering
    /// Lower than main filter for smoother velocity estimates
    velocity_beta: f32 = 0.01,
    
    /// Derivative cutoff frequency for velocity filtering
    velocity_d_cutoff: f32 = 0.5,
    
    // ========================================================================
    // Prediction Parameters
    // ========================================================================
    
    /// Confidence threshold for predictions (0.0-1.0)
    /// Predictions below this threshold are considered unreliable
    confidence_threshold: f32 = 0.5,
    
    /// Enable adaptive NUMA optimization based on predictions
    enable_adaptive_numa: bool = true,
    
    /// Enable SIMD-enhanced prediction analysis
    enable_simd_enhancement: bool = true,
    
    /// Maximum number of historical execution profiles to track
    max_execution_profiles: u32 = 1024,
    
    /// Maximum cache size for prediction results
    max_cache_entries: u32 = 512,
    
    /// Cache entry time-to-live in nanoseconds
    cache_ttl_ns: u64 = 5_000_000_000, // 5 seconds
    
    // ========================================================================
    // Performance Tuning Parameters
    // ========================================================================
    
    /// Prediction accuracy threshold for promoting to high confidence
    accuracy_promotion_threshold: f32 = 0.8,
    
    /// Minimum number of samples before enabling adaptive behavior
    minimum_samples_for_adaptation: u32 = 10,
    
    /// Weight decay factor for old predictions (0.0-1.0)
    historical_weight_decay: f32 = 0.95,
    
    /// Enable prediction result caching to reduce computation overhead
    enable_prediction_caching: bool = true,
    
    /// Enable detailed prediction statistics tracking
    enable_statistics_tracking: bool = true,
    
    // ========================================================================
    // Factory Methods
    // ========================================================================
    
    /// Create balanced configuration for general use
    /// Provides good balance between responsiveness and stability
    pub fn balanced() PredictiveConfig {
        return PredictiveConfig{};
    }
    
    /// Create performance-optimized configuration
    /// More aggressive prediction parameters for high-performance scenarios
    pub fn performanceOptimized() PredictiveConfig {
        return PredictiveConfig{
            .min_cutoff = 0.05,                    // More responsive
            .beta = 0.1,                           // React faster to changes
            .confidence_threshold = 0.3,           // Accept lower confidence predictions
            .enable_adaptive_numa = true,
            .enable_simd_enhancement = true,
            .max_execution_profiles = 2048,        // Track more history
            .max_cache_entries = 1024,             // Larger cache
            .accuracy_promotion_threshold = 0.7,    // Lower promotion threshold
            .minimum_samples_for_adaptation = 5,    // Adapt faster
        };
    }
    
    /// Create conservative configuration for stability-critical scenarios
    /// Prioritizes prediction stability over responsiveness
    pub fn conservative() PredictiveConfig {
        return PredictiveConfig{
            .min_cutoff = 0.2,                    // Less responsive, more stable
            .beta = 0.01,                         // React slowly to changes
            .d_cutoff = 2.0,                      // Heavy derivative smoothing
            .velocity_min_cutoff = 0.01,          // Very smooth velocity
            .velocity_beta = 0.005,               // Very conservative velocity
            .confidence_threshold = 0.7,          // Require high confidence
            .enable_adaptive_numa = false,        // Disable adaptive features
            .enable_simd_enhancement = false,     // Keep it simple
            .max_execution_profiles = 256,        // Smaller memory footprint
            .max_cache_entries = 128,
            .accuracy_promotion_threshold = 0.9,  // High promotion threshold
            .minimum_samples_for_adaptation = 50, // Require many samples
            .historical_weight_decay = 0.99,      // Retain history longer
        };
    }
    
    /// Create testing configuration optimized for unit tests
    /// Fast adaptation with smaller memory usage
    pub fn testing() PredictiveConfig {
        return PredictiveConfig{
            .min_cutoff = 0.01,                   // Very responsive for fast tests
            .beta = 0.2,                          // React immediately
            .confidence_threshold = 0.1,          // Accept low confidence for testing
            .enable_adaptive_numa = false,        // Simpler for testing
            .enable_simd_enhancement = false,     // Avoid complexity in tests
            .max_execution_profiles = 32,         // Small memory footprint
            .max_cache_entries = 16,
            .cache_ttl_ns = 100_000_000,          // 100ms - short for tests
            .minimum_samples_for_adaptation = 1,  // Adapt immediately
            .enable_prediction_caching = false,   // Disable caching for determinism
            .enable_statistics_tracking = true,   // Keep stats for verification
        };
    }
    
    /// Create configuration optimized for benchmarking
    /// Focuses on prediction accuracy and performance measurement
    pub fn benchmarking() PredictiveConfig {
        return PredictiveConfig{
            .min_cutoff = 0.08,                   // Balanced responsiveness
            .beta = 0.08,                         // Balanced adaptation
            .confidence_threshold = 0.4,          // Moderate confidence requirement
            .enable_adaptive_numa = true,
            .enable_simd_enhancement = true,
            .max_execution_profiles = 4096,       // Large profile tracking
            .max_cache_entries = 2048,            // Large cache for benchmarks
            .accuracy_promotion_threshold = 0.75,
            .minimum_samples_for_adaptation = 8,
            .enable_prediction_caching = true,    // Cache for performance
            .enable_statistics_tracking = true,   // Detailed stats for analysis
        };
    }
    
    // ========================================================================
    // Validation and Utility Methods
    // ========================================================================
    
    /// Validate configuration parameters for correctness
    /// Returns error if any parameters are out of valid ranges
    pub fn validate(self: PredictiveConfig) !void {
        // One Euro Filter parameter validation
        if (self.min_cutoff <= 0.0 or self.min_cutoff > 10.0) {
            return error.InvalidMinCutoff;
        }
        
        if (self.beta <= 0.0 or self.beta > 1.0) {
            return error.InvalidBeta;
        }
        
        if (self.d_cutoff <= 0.0 or self.d_cutoff > 10.0) {
            return error.InvalidDCutoff;
        }
        
        // Velocity filter parameter validation
        if (self.velocity_min_cutoff <= 0.0 or self.velocity_min_cutoff > 10.0) {
            return error.InvalidVelocityMinCutoff;
        }
        
        if (self.velocity_beta <= 0.0 or self.velocity_beta > 1.0) {
            return error.InvalidVelocityBeta;
        }
        
        if (self.velocity_d_cutoff <= 0.0 or self.velocity_d_cutoff > 10.0) {
            return error.InvalidVelocityDCutoff;
        }
        
        // Prediction parameter validation
        if (self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0) {
            return error.InvalidConfidenceThreshold;
        }
        
        if (self.accuracy_promotion_threshold < 0.0 or self.accuracy_promotion_threshold > 1.0) {
            return error.InvalidAccuracyPromotionThreshold;
        }
        
        if (self.historical_weight_decay < 0.0 or self.historical_weight_decay > 1.0) {
            return error.InvalidHistoricalWeightDecay;
        }
        
        // Resource limit validation
        if (self.max_execution_profiles == 0 or self.max_execution_profiles > 100_000) {
            return error.InvalidMaxExecutionProfiles;
        }
        
        if (self.max_cache_entries == 0 or self.max_cache_entries > 100_000) {
            return error.InvalidMaxCacheEntries;
        }
        
        if (self.cache_ttl_ns == 0) {
            return error.InvalidCacheTTL;
        }
        
        if (self.minimum_samples_for_adaptation == 0 or self.minimum_samples_for_adaptation > 10_000) {
            return error.InvalidMinimumSamplesForAdaptation;
        }
    }
    
    /// Get a human-readable description of the configuration
    pub fn getDescription(self: PredictiveConfig, allocator: std.mem.Allocator) ![]u8 {
        return std.fmt.allocPrint(allocator,
            "PredictiveConfig(min_cutoff={d:.3}, beta={d:.3}, confidence_threshold={d:.2}, " ++
            "adaptive_numa={}, simd_enhancement={}, max_profiles={}, max_cache={})",
            .{
                self.min_cutoff,
                self.beta,
                self.confidence_threshold,
                self.enable_adaptive_numa,
                self.enable_simd_enhancement,
                self.max_execution_profiles,
                self.max_cache_entries,
            }
        );
    }
    
    /// Clone configuration with modifications
    pub fn withModifications(self: PredictiveConfig, modifications: anytype) PredictiveConfig {
        var new_config = self;
        
        // Use reflection to apply modifications
        const type_info = @typeInfo(@TypeOf(modifications));
        switch (type_info) {
            .@"struct" => |struct_info| {
                inline for (struct_info.fields) |field| {
                    if (@hasField(PredictiveConfig, field.name)) {
                        @field(new_config, field.name) = @field(modifications, field.name);
                    }
                }
            },
            else => {}, // Ignore non-struct types
        }
        
        return new_config;
    }
    
    /// Check if this configuration is suitable for high-performance scenarios
    pub fn isHighPerformance(self: PredictiveConfig) bool {
        return self.min_cutoff <= 0.1 and 
               self.beta >= 0.05 and 
               self.confidence_threshold <= 0.5 and
               self.enable_adaptive_numa and
               self.enable_simd_enhancement and
               self.max_execution_profiles >= 1024;
    }
    
    /// Check if this configuration prioritizes stability
    pub fn isConservative(self: PredictiveConfig) bool {
        return self.min_cutoff >= 0.15 and
               self.beta <= 0.02 and
               self.confidence_threshold >= 0.6 and
               self.minimum_samples_for_adaptation >= 20;
    }
    
    /// Estimate memory usage for this configuration
    pub fn estimateMemoryUsage(self: PredictiveConfig) usize {
        const profile_size = 64; // Approximate size of ExecutionProfile
        const cache_entry_size = 96; // Approximate size of cached prediction
        
        return (self.max_execution_profiles * profile_size) + 
               (self.max_cache_entries * cache_entry_size);
    }
};

// ============================================================================
// Configuration Error Types
// ============================================================================

pub const ConfigValidationError = error{
    InvalidMinCutoff,
    InvalidBeta,
    InvalidDCutoff,
    InvalidVelocityMinCutoff,
    InvalidVelocityBeta,
    InvalidVelocityDCutoff,
    InvalidConfidenceThreshold,
    InvalidAccuracyPromotionThreshold,
    InvalidHistoricalWeightDecay,
    InvalidMaxExecutionProfiles,
    InvalidMaxCacheEntries,
    InvalidCacheTTL,
    InvalidMinimumSamplesForAdaptation,
};

// ============================================================================
// Testing
// ============================================================================

test "PredictiveConfig factory methods" {
    const balanced = PredictiveConfig.balanced();
    const performance = PredictiveConfig.performanceOptimized();
    const conservative = PredictiveConfig.conservative();
    const testing_config = PredictiveConfig.testing();
    const bench_config = PredictiveConfig.benchmarking();
    
    // Validate all configurations
    try balanced.validate();
    try performance.validate();
    try conservative.validate();
    try testing_config.validate();
    try bench_config.validate();
    
    // Check configuration characteristics
    try std.testing.expect(performance.isHighPerformance());
    try std.testing.expect(!conservative.isHighPerformance());
    try std.testing.expect(conservative.isConservative());
    try std.testing.expect(!performance.isConservative());
}

test "PredictiveConfig validation" {
    var invalid_config = PredictiveConfig.balanced();
    
    // Test invalid min_cutoff
    invalid_config.min_cutoff = -1.0;
    try std.testing.expectError(ConfigValidationError.InvalidMinCutoff, invalid_config.validate());
    
    // Test invalid beta
    invalid_config = PredictiveConfig.balanced();
    invalid_config.beta = 2.0;
    try std.testing.expectError(ConfigValidationError.InvalidBeta, invalid_config.validate());
    
    // Test invalid confidence threshold
    invalid_config = PredictiveConfig.balanced();
    invalid_config.confidence_threshold = 1.5;
    try std.testing.expectError(ConfigValidationError.InvalidConfidenceThreshold, invalid_config.validate());
}

test "PredictiveConfig memory estimation" {
    const config = PredictiveConfig.performanceOptimized();
    const memory_usage = config.estimateMemoryUsage();
    
    try std.testing.expect(memory_usage > 0);
    try std.testing.expect(memory_usage < 1_000_000); // Reasonable upper bound
}

test "PredictiveConfig modifications" {
    const base_config = PredictiveConfig.balanced();
    const modified_config = base_config.withModifications(.{
        .min_cutoff = 0.2,
        .enable_adaptive_numa = false,
    });
    
    try std.testing.expect(modified_config.min_cutoff == 0.2);
    try std.testing.expect(modified_config.enable_adaptive_numa == false);
    // Other fields should remain unchanged
    try std.testing.expect(modified_config.beta == base_config.beta);
}