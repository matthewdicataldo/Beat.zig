const std = @import("std");
const builtin = @import("builtin");
const build_opts = @import("build_config_unified.zig");

// ============================================================================
// Advanced Features Configuration for Beat.zig (Production-Ready Defaults)
//
// This module provides intelligent defaults for advanced features while 
// maintaining robust fallback mechanisms and cross-platform compatibility.
// All advanced features are enabled by default with automatic detection
// and graceful degradation when hardware/software support is unavailable.
// ============================================================================

/// Advanced Features Configuration with Production-Ready Defaults
pub const AdvancedFeaturesConfig = struct {
    // ========================================================================
    // Performance Optimization Features (Default: ENABLED)
    // ========================================================================
    
    /// Enable Souper mathematical optimizations with formal verification
    /// Fallback: Disables gracefully if Souper toolchain unavailable
    enable_souper_optimizations: bool = true,
    
    /// Enable ISPC acceleration for SIMD operations (6-23x speedup)  
    /// Fallback: Falls back to Zig SIMD when ISPC compiler unavailable
    enable_ispc_acceleration: bool = true,
    
    /// Enable Minotaur SIMD superoptimization
    /// Fallback: Falls back to standard SIMD when Minotaur unavailable
    enable_minotaur_optimizations: bool = true,
    
    /// Enable unified triple optimization pipeline (Souper + ISPC + Minotaur)
    /// Fallback: Individual optimizations fall back independently
    enable_triple_optimization: bool = true,
    
    /// Enable advanced SIMD task classification and batch formation
    /// Fallback: Uses simple task dispatching without vectorization
    enable_simd_classification: bool = true,
    
    /// Enable topology-aware work stealing and thread affinity
    /// Fallback: Uses round-robin scheduling when topology unavailable
    enable_topology_awareness: bool = true,
    
    /// Enable NUMA-aware memory allocation and worker placement
    /// Fallback: Uses system default allocation on single-node systems
    enable_numa_awareness: bool = true,
    
    /// Enable One Euro Filter predictive scheduling
    /// Fallback: Uses basic heartbeat scheduling without prediction
    enable_predictive_scheduling: bool = true,
    
    /// Enable advanced worker selection with multi-criteria optimization
    /// Fallback: Uses simple load balancing when advanced selection fails
    enable_advanced_worker_selection: bool = true,
    
    // ========================================================================
    // Monitoring and Observability Features (Default: ENABLED)
    // ========================================================================
    
    /// Enable memory pressure monitoring with adaptive scheduling
    /// Fallback: Uses basic memory utilization checks on unsupported platforms
    enable_memory_pressure_monitoring: bool = true,
    
    /// Enable OpenTelemetry distributed tracing and metrics
    /// Fallback: Uses local logging when OpenTelemetry unavailable
    enable_opentelemetry: bool = true,
    
    /// Enable comprehensive error reporting with analytics
    /// Fallback: Uses standard error handling without analytics
    enable_advanced_error_reporting: bool = true,
    
    /// Enable background performance profiling
    /// Fallback: Disables gracefully with minimal overhead
    enable_background_profiling: bool = true,
    
    /// Enable runtime statistics collection and analysis
    /// Fallback: Basic statistics only when advanced collection fails
    enable_advanced_statistics: bool = true,
    
    // ========================================================================
    // Development and Testing Features (Default: OFF in production)
    // ========================================================================
    
    /// Enable development mode with comprehensive debugging
    /// Note: Automatically enabled in debug builds, disabled in release
    enable_development_mode: bool = (builtin.mode == .Debug),
    
    /// Enable verbose logging for detailed operation visibility
    /// Note: Automatically enabled in debug builds, configurable in release
    enable_verbose_logging: bool = (builtin.mode == .Debug),
    
    /// Enable task tracing for individual task execution analysis
    /// Note: Low overhead, safe to enable in production monitoring
    enable_task_tracing: bool = true,
    
    /// Enable scheduler profiling for performance optimization
    /// Note: Minimal overhead, useful for production optimization
    enable_scheduler_profiling: bool = true,
    
    /// Enable deadlock detection with timeout-based analysis
    /// Note: Disabled in release builds for performance
    enable_deadlock_detection: bool = (builtin.mode == .Debug),
    
    /// Enable resource leak detection and validation
    /// Note: Enabled by default for production robustness
    enable_resource_leak_detection: bool = true,
    
    /// Enable memory debugging with detailed allocation tracking
    /// Note: Disabled in release builds for performance
    enable_memory_debugging: bool = (builtin.mode == .Debug),
    
    // ========================================================================
    // Platform-Specific Features (Default: AUTO-DETECT)
    // ========================================================================
    
    /// Enable Linux PSI (Pressure Stall Information) monitoring
    /// Fallback: Uses alternative memory monitoring on non-Linux platforms
    enable_psi_monitoring: bool = (builtin.os.tag == .linux),
    
    /// Enable Windows-specific optimizations
    /// Fallback: Uses cross-platform implementations on other platforms
    enable_windows_optimizations: bool = (builtin.os.tag == .windows),
    
    /// Enable macOS-specific optimizations
    /// Fallback: Uses cross-platform implementations on other platforms  
    enable_macos_optimizations: bool = (builtin.os.tag == .macos),
    
    /// Enable container environment detection and adaptation
    /// Fallback: Uses host environment assumptions when detection fails
    enable_container_awareness: bool = true,
    
    // ========================================================================
    // Robustness and Testing Features (Default: ENABLED for reliability)
    // ========================================================================
    
    /// Enable comprehensive fuzz testing in development builds
    /// Note: Automatically disabled in release builds
    enable_fuzzing_framework: bool = (builtin.mode == .Debug),
    
    /// Enable allocator error injection for robustness testing
    /// Note: Disabled in release builds for performance
    enable_allocator_fuzzing: bool = (builtin.mode == .Debug),
    
    /// Enable hardware absence simulation for degraded environment testing
    /// Note: Disabled in release builds, useful for CI/CD validation
    enable_hardware_simulation: bool = (builtin.mode == .Debug),
    
    /// Enable state corruption testing for race condition detection
    /// Note: Disabled in release builds for stability
    enable_state_fuzzing: bool = (builtin.mode == .Debug),
    
    /// Enable deterministic execution mode for reproducible testing
    /// Note: Disabled by default due to performance impact
    enable_deterministic_mode: bool = false,
    
    // ========================================================================
    // Auto-Configuration Methods
    // ========================================================================
    
    /// Create production-optimized configuration with all advanced features enabled
    pub fn createProductionConfig() AdvancedFeaturesConfig {
        var config = AdvancedFeaturesConfig{};
        
        // Ensure all performance features are enabled
        config.enable_souper_optimizations = true;
        config.enable_ispc_acceleration = true;
        config.enable_minotaur_optimizations = true;
        config.enable_triple_optimization = true;
        config.enable_simd_classification = true;
        config.enable_topology_awareness = true;
        config.enable_numa_awareness = true;
        config.enable_predictive_scheduling = true;
        config.enable_advanced_worker_selection = true;
        
        // Enable monitoring for production insights
        config.enable_memory_pressure_monitoring = true;
        config.enable_opentelemetry = true;
        config.enable_advanced_error_reporting = true;
        config.enable_background_profiling = true;
        config.enable_advanced_statistics = true;
        config.enable_task_tracing = true;
        config.enable_scheduler_profiling = true;
        config.enable_resource_leak_detection = true;
        
        // Disable heavy debugging features for performance
        config.enable_development_mode = false;
        config.enable_verbose_logging = false;
        config.enable_deadlock_detection = false;
        config.enable_memory_debugging = false;
        config.enable_fuzzing_framework = false;
        config.enable_allocator_fuzzing = false;
        config.enable_hardware_simulation = false;
        config.enable_state_fuzzing = false;
        
        // Enable platform detection
        config.enable_container_awareness = true;
        
        return config;
    }
    
    /// Create development configuration with comprehensive debugging enabled
    pub fn createDevelopmentConfig() AdvancedFeaturesConfig {
        var config = AdvancedFeaturesConfig{};
        
        // Enable all performance features for testing
        config.enable_souper_optimizations = true;
        config.enable_ispc_acceleration = true;
        config.enable_minotaur_optimizations = true;
        config.enable_triple_optimization = true;
        config.enable_simd_classification = true;
        config.enable_topology_awareness = true;
        config.enable_numa_awareness = true;
        config.enable_predictive_scheduling = true;
        config.enable_advanced_worker_selection = true;
        
        // Enable comprehensive monitoring and debugging
        config.enable_memory_pressure_monitoring = true;
        config.enable_opentelemetry = true;
        config.enable_advanced_error_reporting = true;
        config.enable_background_profiling = true;
        config.enable_advanced_statistics = true;
        config.enable_development_mode = true;
        config.enable_verbose_logging = true;
        config.enable_task_tracing = true;
        config.enable_scheduler_profiling = true;
        config.enable_deadlock_detection = true;
        config.enable_resource_leak_detection = true;
        config.enable_memory_debugging = true;
        
        // Enable comprehensive testing features
        config.enable_fuzzing_framework = true;
        config.enable_allocator_fuzzing = true;
        config.enable_hardware_simulation = true;
        config.enable_state_fuzzing = true;
        
        // Enable platform detection
        config.enable_container_awareness = true;
        
        return config;
    }
    
    /// Create CI/CD configuration optimized for automated testing
    pub fn createCIConfig() AdvancedFeaturesConfig {
        var config = AdvancedFeaturesConfig{};
        
        // Enable performance features with testing focus
        config.enable_souper_optimizations = true;
        config.enable_ispc_acceleration = true;
        config.enable_minotaur_optimizations = true;
        config.enable_triple_optimization = true;
        config.enable_simd_classification = true;
        config.enable_topology_awareness = true;
        config.enable_numa_awareness = true;
        config.enable_predictive_scheduling = true;
        config.enable_advanced_worker_selection = true;
        
        // Enable monitoring for CI insights
        config.enable_memory_pressure_monitoring = true;
        config.enable_opentelemetry = false; // Reduce external dependencies in CI
        config.enable_advanced_error_reporting = true;
        config.enable_background_profiling = false; // Reduce CI overhead
        config.enable_advanced_statistics = true;
        config.enable_task_tracing = false; // Reduce CI noise
        config.enable_scheduler_profiling = false; // Reduce CI overhead
        config.enable_resource_leak_detection = true;
        
        // Enable targeted debugging for CI reliability
        config.enable_development_mode = false;
        config.enable_verbose_logging = false;
        config.enable_deadlock_detection = true; // Critical for CI stability
        config.enable_memory_debugging = true; // Critical for CI validation
        
        // Enable comprehensive testing for CI validation
        config.enable_fuzzing_framework = true;
        config.enable_allocator_fuzzing = true;
        config.enable_hardware_simulation = true;
        config.enable_state_fuzzing = true;
        
        // Enable container awareness for CI environments
        config.enable_container_awareness = true;
        
        return config;
    }
    
    /// Create embedded/resource-constrained configuration
    pub fn createEmbeddedConfig() AdvancedFeaturesConfig {
        var config = AdvancedFeaturesConfig{};
        
        // Enable lightweight performance features only
        config.enable_souper_optimizations = false; // Too heavy for embedded
        config.enable_ispc_acceleration = false; // May not be available
        config.enable_minotaur_optimizations = false; // Too heavy for embedded
        config.enable_triple_optimization = false; // Too heavy for embedded
        config.enable_simd_classification = true; // Lightweight, good ROI
        config.enable_topology_awareness = true; // Lightweight, good ROI
        config.enable_numa_awareness = false; // Usually single-node in embedded
        config.enable_predictive_scheduling = true; // Lightweight, good ROI
        config.enable_advanced_worker_selection = false; // Reduce complexity
        
        // Minimal monitoring for embedded constraints
        config.enable_memory_pressure_monitoring = true; // Critical for embedded
        config.enable_opentelemetry = false; // Too heavy for embedded
        config.enable_advanced_error_reporting = false; // Reduce overhead
        config.enable_background_profiling = false; // Reduce overhead
        config.enable_advanced_statistics = false; // Reduce overhead
        config.enable_task_tracing = false; // Reduce overhead
        config.enable_scheduler_profiling = false; // Reduce overhead
        config.enable_resource_leak_detection = true; // Critical for embedded
        
        // Minimal debugging for embedded
        config.enable_development_mode = false;
        config.enable_verbose_logging = false;
        config.enable_deadlock_detection = false; // Reduce overhead
        config.enable_memory_debugging = false; // Reduce overhead
        
        // No testing framework in embedded
        config.enable_fuzzing_framework = false;
        config.enable_allocator_fuzzing = false;
        config.enable_hardware_simulation = false;
        config.enable_state_fuzzing = false;
        
        // Basic platform detection
        config.enable_container_awareness = false; // Usually not relevant
        
        return config;
    }
    
    /// Auto-detect optimal configuration based on build mode and platform
    pub fn createAutoConfig() AdvancedFeaturesConfig {
        return switch (builtin.mode) {
            .Debug => createDevelopmentConfig(),
            .ReleaseSafe => createProductionConfig(),
            .ReleaseFast => createProductionConfig(),
            .ReleaseSmall => createEmbeddedConfig(),
        };
    }
    
    /// Validate configuration and provide optimization recommendations
    pub fn validateAndOptimize(self: *AdvancedFeaturesConfig, allocator: std.mem.Allocator) ![]const u8 {
        var recommendations = std.ArrayList(u8).init(allocator);
        var writer = recommendations.writer();
        
        try writer.writeAll("Beat.zig Advanced Features Configuration Analysis:\n\n");
        
        // Analyze performance features
        var performance_score: u32 = 0;
        if (self.enable_souper_optimizations) performance_score += 10;
        if (self.enable_ispc_acceleration) performance_score += 15;
        if (self.enable_minotaur_optimizations) performance_score += 8;
        if (self.enable_triple_optimization) performance_score += 5;
        if (self.enable_simd_classification) performance_score += 12;
        if (self.enable_topology_awareness) performance_score += 8;
        if (self.enable_numa_awareness) performance_score += 6;
        if (self.enable_predictive_scheduling) performance_score += 10;
        if (self.enable_advanced_worker_selection) performance_score += 7;
        
        try writer.print("🚀 Performance Optimization Score: {}/81\n", .{performance_score});
        
        if (performance_score < 50) {
            try writer.writeAll("⚠️  Recommendation: Enable more performance features for better throughput\n");
        } else if (performance_score > 70) {
            try writer.writeAll("✅ Excellent: High-performance configuration enabled\n");
        }
        
        // Analyze monitoring features
        var monitoring_score: u32 = 0;
        if (self.enable_memory_pressure_monitoring) monitoring_score += 10;
        if (self.enable_opentelemetry) monitoring_score += 15;
        if (self.enable_advanced_error_reporting) monitoring_score += 12;
        if (self.enable_background_profiling) monitoring_score += 8;
        if (self.enable_advanced_statistics) monitoring_score += 10;
        if (self.enable_task_tracing) monitoring_score += 6;
        if (self.enable_scheduler_profiling) monitoring_score += 7;
        
        try writer.print("📊 Monitoring and Observability Score: {}/68\n", .{monitoring_score});
        
        // Platform-specific recommendations
        try writer.writeAll("\n🔧 Platform-Specific Recommendations:\n");
        switch (builtin.os.tag) {
            .linux => {
                if (!self.enable_psi_monitoring) {
                    try writer.writeAll("💡 Linux: Enable PSI monitoring for better memory pressure detection\n");
                }
            },
            .windows => {
                if (!self.enable_windows_optimizations) {
                    try writer.writeAll("💡 Windows: Enable Windows-specific optimizations for better performance\n");
                }
            },
            .macos => {
                if (!self.enable_macos_optimizations) {
                    try writer.writeAll("💡 macOS: Enable macOS-specific optimizations for better performance\n");
                }
            },
            else => {
                try writer.writeAll("💡 Cross-platform: Configuration optimized for portability\n");
            },
        }
        
        // Build mode recommendations
        try writer.writeAll("\n🎯 Build Mode Analysis:\n");
        switch (builtin.mode) {
            .Debug => {
                try writer.writeAll("🐛 Debug Mode: All debugging features available\n");
                if (!self.enable_development_mode) {
                    try writer.writeAll("💡 Recommendation: Enable development_mode for better debugging experience\n");
                }
            },
            .ReleaseSafe, .ReleaseFast => {
                try writer.writeAll("⚡ Release Mode: Optimized for production performance\n");
                if (self.enable_memory_debugging) {
                    try writer.writeAll("⚠️  Note: Memory debugging enabled in release mode (performance impact)\n");
                }
            },
            .ReleaseSmall => {
                try writer.writeAll("📦 Size-Optimized: Configured for minimal resource usage\n");
                if (performance_score > 40) {
                    try writer.writeAll("💡 Consider disabling heavy features for smaller binary size\n");
                }
            },
        }
        
        try writer.writeAll("\n✅ Configuration validation complete\n");
        
        return recommendations.toOwnedSlice();
    }
    
    /// Apply auto-detected hardware capabilities to optimize configuration
    pub fn applyHardwareOptimizations(self: *AdvancedFeaturesConfig) void {
        // Auto-detect and optimize based on available hardware
        
        // NUMA awareness optimization
        const numa_nodes = build_opts.hardware.numa_nodes;
        if (numa_nodes <= 1) {
            self.enable_numa_awareness = false; // No benefit on single-node systems
        }
        
        // SIMD feature optimization
        if (!build_opts.hardware.has_avx and !build_opts.hardware.has_sse and !build_opts.hardware.has_neon) {
            self.enable_simd_classification = false; // No SIMD support available
        }
        
        // Worker count optimization
        const cpu_count = build_opts.hardware.cpu_count;
        if (cpu_count <= 2) {
            self.enable_advanced_worker_selection = false; // Not beneficial for small core counts
            self.enable_topology_awareness = false; // Less critical for small systems
        }
        
        // Memory optimization for constrained systems
        if (cpu_count <= 4) {
            self.enable_background_profiling = false; // Reduce overhead on smaller systems
        }
    }
    
    /// Create configuration string for logging and debugging
    pub fn createConfigSummary(self: *const AdvancedFeaturesConfig, allocator: std.mem.Allocator) ![]const u8 {
        var summary = std.ArrayList(u8).init(allocator);
        var writer = summary.writer();
        
        try writer.writeAll("Beat.zig Advanced Features Configuration:\n");
        try writer.print("  Performance: ISPC={}, Souper={}, SIMD={}, Topology={}, NUMA={}\n", .{
            self.enable_ispc_acceleration,
            self.enable_souper_optimizations,
            self.enable_simd_classification,
            self.enable_topology_awareness,
            self.enable_numa_awareness,
        });
        try writer.print("  Monitoring: OpenTelemetry={}, MemoryPressure={}, ErrorReporting={}\n", .{
            self.enable_opentelemetry,
            self.enable_memory_pressure_monitoring,
            self.enable_advanced_error_reporting,
        });
        try writer.print("  Development: DevMode={}, Tracing={}, Profiling={}, Debugging={}\n", .{
            self.enable_development_mode,
            self.enable_task_tracing,
            self.enable_scheduler_profiling,
            self.enable_memory_debugging,
        });
        
        return summary.toOwnedSlice();
    }
};

/// Global advanced features configuration - defaults to auto-detected optimal settings
pub var global_config: AdvancedFeaturesConfig = AdvancedFeaturesConfig.createAutoConfig();

/// Initialize advanced features with specified configuration
pub fn initializeAdvancedFeatures(config: AdvancedFeaturesConfig) void {
    global_config = config;
    global_config.applyHardwareOptimizations();
}

/// Get current advanced features configuration
pub fn getCurrentConfig() AdvancedFeaturesConfig {
    return global_config;
}

/// Check if a specific advanced feature is enabled
pub fn isFeatureEnabled(comptime feature_name: []const u8) bool {
    return @field(global_config, feature_name);
}

// ============================================================================
// Compile-time Feature Detection for Zero-Overhead Conditionals
// ============================================================================

/// Compile-time check for ISPC acceleration availability
pub const has_ispc_support = global_config.enable_ispc_acceleration;

/// Compile-time check for Souper optimization availability  
pub const has_souper_support = global_config.enable_souper_optimizations;

/// Compile-time check for development features
pub const has_development_features = global_config.enable_development_mode;

/// Compile-time check for advanced monitoring
pub const has_advanced_monitoring = global_config.enable_opentelemetry and global_config.enable_advanced_statistics;

// ============================================================================
// Testing Support
// ============================================================================

test "advanced features configuration creation" {
    const testing = std.testing;
    
    // Test production configuration
    const prod_config = AdvancedFeaturesConfig.createProductionConfig();
    try testing.expect(prod_config.enable_ispc_acceleration == true);
    try testing.expect(prod_config.enable_souper_optimizations == true);
    try testing.expect(prod_config.enable_memory_debugging == false); // Should be disabled in production
    
    // Test development configuration
    const dev_config = AdvancedFeaturesConfig.createDevelopmentConfig();
    try testing.expect(dev_config.enable_development_mode == true);
    try testing.expect(dev_config.enable_verbose_logging == true);
    try testing.expect(dev_config.enable_memory_debugging == true);
    
    // Test embedded configuration
    const embedded_config = AdvancedFeaturesConfig.createEmbeddedConfig();
    try testing.expect(embedded_config.enable_souper_optimizations == false); // Should be disabled for embedded
    try testing.expect(embedded_config.enable_memory_pressure_monitoring == true); // Should be enabled for embedded
}

test "hardware optimization application" {
    var config = AdvancedFeaturesConfig.createProductionConfig();
    config.applyHardwareOptimizations();
    
    // Configuration should be automatically optimized based on detected hardware
    // Specific assertions depend on actual hardware, so we just verify it doesn't crash
}

test "configuration validation" {
    const testing = std.testing;
    
    var config = AdvancedFeaturesConfig.createAutoConfig();
    const validation_result = try config.validateAndOptimize(testing.allocator);
    defer testing.allocator.free(validation_result);
    
    // Should contain analysis text
    try testing.expect(validation_result.len > 0);
    try testing.expect(std.mem.indexOf(u8, validation_result, "Performance Optimization Score") != null);
}