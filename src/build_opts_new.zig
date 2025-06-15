const std = @import("std");

// FIXED: Smart build configuration resolver
// Solves dependency chain issues identified by Reverb Code team
const config_resolver = @import("config_resolver.zig");

// Build-time configuration access for Beat.zig
// Now intelligently handles dependency scenarios with graceful fallbacks

// ============================================================================
// Smart Configuration Access (Replaces direct build_config import)
// ============================================================================

/// Auto-detected hardware configuration - now dependency-safe
pub const hardware = struct {
    pub const cpu_count: u32 = config_resolver.hardware.cpu_count;
    pub const optimal_workers: u32 = config_resolver.hardware.optimal_workers;
    pub const numa_nodes: u32 = config_resolver.hardware.numa_nodes;
    pub const cache_line_size: u32 = config_resolver.hardware.cache_line_size;
    pub const memory_gb: u32 = config_resolver.hardware.memory_gb;
    
    // Computed values
    pub const physical_cores: u32 = cpu_count; // Approximation
    pub const optimal_test_threads: u32 = @max(optimal_workers / 2, 2);
    pub const optimal_queue_size: u32 = config_resolver.perf.optimal_queue_size;
};

/// CPU features - now with intelligent detection
pub const cpu_features = struct {
    pub const has_avx2: bool = config_resolver.hardware.has_avx2;
    pub const has_avx512: bool = config_resolver.hardware.has_avx512;
    pub const has_neon: bool = config_resolver.hardware.has_neon;
    
    // Derived features
    pub const has_avx: bool = has_avx2; // AVX2 implies AVX
    pub const has_sse: bool = has_avx2; // AVX2 implies SSE4.2
    
    /// Check if any SIMD features are available
    pub const has_simd: bool = has_avx or has_avx2 or has_avx512 or has_sse or has_neon;
    
    /// Get the widest available SIMD width in bytes
    pub const simd_width: u32 = blk: {
        if (has_avx512) break :blk 64;
        if (has_avx2 or has_avx) break :blk 32;
        if (has_sse) break :blk 16;
        if (has_neon) break :blk 16;
        break :blk 8; // Fallback
    };
};

/// Performance configuration - now with smart defaults
pub const performance = struct {
    pub const one_euro_min_cutoff: f32 = @floatCast(config_resolver.one_euro.min_cutoff);
    pub const one_euro_beta: f32 = @floatCast(config_resolver.one_euro.beta);
    pub const enable_topology_aware: bool = hardware.numa_nodes > 1;
    pub const enable_numa_aware: bool = config_resolver.perf.enable_numa_optimization;
};

/// Build configuration information
pub const build_info = struct {
    pub const is_debug: bool = std.debug.runtime_safety;
    pub const is_release_fast: bool = std.builtin.mode == .ReleaseFast;
    pub const is_optimized: bool = !is_debug;
};

// ============================================================================
// Configuration Mode Diagnostics
// ============================================================================

/// Get information about what configuration mode is being used
pub fn getConfigurationInfo() ConfigurationInfo {
    return ConfigurationInfo{
        .mode = config_resolver.getConfigMode(),
        .has_external_config = @hasDecl(@This(), "build_config"),
        .worker_count = hardware.optimal_workers,
        .queue_size = hardware.optimal_queue_size,
        .simd_available = cpu_features.has_simd,
    };
}

pub const ConfigurationInfo = struct {
    mode: config_resolver.ConfigResolver.ConfigMode,
    has_external_config: bool,
    worker_count: u32,
    queue_size: u32,
    simd_available: bool,
    
    pub fn print(self: ConfigurationInfo) void {
        std.log.info("Beat.zig Configuration:", .{});
        std.log.info("  Mode: {}", .{self.mode});
        std.log.info("  External Config: {}", .{self.has_external_config});
        std.log.info("  Workers: {}", .{self.worker_count});
        std.log.info("  Queue Size: {}", .{self.queue_size});
        std.log.info("  SIMD: {}", .{self.simd_available});
    }
};

// ============================================================================
// Utility Functions (Enhanced with fallback handling)
// ============================================================================

/// Get optimal Beat.zig Config based on detected configuration
pub fn getOptimalConfig() @import("core.zig").Config {
    return @import("core.zig").Config{
        .num_workers = hardware.optimal_workers,
        .task_queue_size = hardware.optimal_queue_size,
        .enable_topology_aware = performance.enable_topology_aware,
        .enable_numa_aware = performance.enable_numa_aware,
        .enable_predictive = true,
        .prediction_min_cutoff = performance.one_euro_min_cutoff,
        .prediction_beta = performance.one_euro_beta,
        .prediction_d_cutoff = 1.0,
        
        // Adjust based on build type
        .enable_statistics = !build_info.is_release_fast,
        .enable_trace = build_info.is_debug,
    };
}

/// Get basic configuration safe for any environment (requested by Reverb team)
pub fn getBasicConfig() @import("core.zig").Config {
    return @import("core.zig").Config{
        .num_workers = @min(hardware.optimal_workers, 8), // Conservative limit
        .task_queue_size = 128, // Small, safe queue size
        .enable_topology_aware = false, // Disable complex features
        .enable_numa_aware = false,
        .enable_predictive = false,
        .enable_advanced_selection = false,
        .enable_lock_free = true, // Keep performance feature
        .enable_statistics = false,
        .enable_trace = false,
    };
}

/// Get performance configuration with minimal dependencies
pub fn getPerformanceConfig() @import("core.zig").Config {
    return @import("core.zig").Config{
        .num_workers = hardware.optimal_workers,
        .task_queue_size = hardware.optimal_queue_size,
        .enable_topology_aware = hardware.numa_nodes > 1, // Only if beneficial
        .enable_numa_aware = false, // Disable complex NUMA logic
        .enable_predictive = false, // Disable prediction complexity
        .enable_advanced_selection = false,
        .enable_lock_free = true,
        .enable_statistics = build_info.is_debug,
        .enable_trace = false,
    };
}

/// Get configuration optimized for testing
pub fn getTestConfig() @import("core.zig").Config {
    var config = getBasicConfig(); // Start with safe basic config
    
    // Test-specific adjustments
    config.num_workers = hardware.optimal_test_threads;
    config.task_queue_size = 64; // Small for faster test execution
    config.enable_statistics = true; // Always enable for testing
    config.enable_trace = build_info.is_debug;
    
    return config;
}

/// Get configuration optimized for benchmarking
pub fn getBenchmarkConfig() @import("core.zig").Config {
    const config = getOptimalConfig(); // Use full optimization for benchmarks
    var bench_config = config;
    
    // Benchmark-specific optimizations
    bench_config.task_queue_size = hardware.optimal_workers * 256; // Large queues
    bench_config.enable_statistics = true; // Need stats for benchmarks
    bench_config.enable_trace = false; // No tracing overhead
    
    return bench_config;
}

/// Print configuration summary with dependency mode information
pub fn printSummary() void {
    const info = getConfigurationInfo();
    
    std.debug.print("=== Beat.zig Smart Configuration Summary ===\n", .{});
    std.debug.print("Configuration Mode: {}\n", .{info.mode});
    std.debug.print("External build_config: {}\n", .{info.has_external_config});
    
    std.debug.print("Hardware:\n", .{});
    std.debug.print("  CPU Count: {}\n", .{hardware.cpu_count});
    std.debug.print("  Optimal Workers: {}\n", .{hardware.optimal_workers});
    std.debug.print("  NUMA Nodes: {}\n", .{hardware.numa_nodes});
    std.debug.print("  Queue Size: {}\n", .{hardware.optimal_queue_size});
    
    std.debug.print("CPU Features:\n", .{});
    std.debug.print("  SIMD Available: {}\n", .{cpu_features.has_simd});
    std.debug.print("  SIMD Width: {} bytes\n", .{cpu_features.simd_width});
    if (cpu_features.has_avx512) std.debug.print("  AVX-512: Yes\n", .{});
    if (cpu_features.has_avx2) std.debug.print("  AVX2: Yes\n", .{});
    if (cpu_features.has_neon) std.debug.print("  NEON: Yes\n", .{});
    
    std.debug.print("Performance:\n", .{});
    std.debug.print("  Topology Aware: {}\n", .{performance.enable_topology_aware});
    std.debug.print("  NUMA Aware: {}\n", .{performance.enable_numa_aware});
    std.debug.print("  One Euro Filter: min_cutoff={d:.2}, beta={d:.3}\n", .{
        performance.one_euro_min_cutoff, performance.one_euro_beta
    });
    
    std.debug.print("Build Info:\n", .{});
    std.debug.print("  Debug: {}\n", .{build_info.is_debug});
    std.debug.print("  Release Fast: {}\n", .{build_info.is_release_fast});
    std.debug.print("============================================\n", .{});
}

/// Enhanced validation with dependency mode awareness
pub fn validateConfiguration() void {
    // Compile-time checks
    if (hardware.optimal_workers == 0) {
        @compileError("Invalid worker count detected in configuration");
    }
    
    if (hardware.optimal_queue_size < 16) {
        @compileError("Queue size too small - minimum 16 required");
    }
    
    if (performance.one_euro_beta <= 0.0 or performance.one_euro_beta > 1.0) {
        @compileError("Invalid One Euro Filter beta parameter - must be (0,1]");
    }
    
    // Runtime validation and warnings
    const info = getConfigurationInfo();
    if (info.mode == .dependency_safe) {
        std.log.warn("Beat.zig running in dependency safe mode - some features disabled", .{});
        std.log.warn("For full features, provide build_config module or use advanced config", .{});
    }
}

// ============================================================================
// SIMD Optimization Helpers (Enhanced)
// ============================================================================

/// Get optimal vector type for the current platform
pub fn OptimalVector(comptime T: type) type {
    const element_size = @sizeOf(T);
    const vector_size = cpu_features.simd_width;
    const elements = vector_size / element_size;
    
    if (elements > 1) {
        return @Vector(elements, T);
    } else {
        return T; // Fallback to scalar
    }
}

/// Check if vectorization is beneficial for the given type
pub fn shouldVectorize(comptime T: type) bool {
    return cpu_features.has_simd and @sizeOf(T) <= cpu_features.simd_width / 2;
}

/// Get alignment for optimal SIMD access
pub fn getSimdAlignment(comptime T: type) u32 {
    return if (shouldVectorize(T)) cpu_features.simd_width else @alignOf(T);
}

// ============================================================================
// Testing Integration
// ============================================================================

test "smart configuration validation" {
    validateConfiguration();
    
    const info = getConfigurationInfo();
    try std.testing.expect(info.worker_count > 0);
    try std.testing.expect(info.queue_size > 0);
    
    // Test all configuration levels
    const basic_config = getBasicConfig();
    try std.testing.expect(basic_config.num_workers.? > 0);
    try std.testing.expect(basic_config.enable_predictive == false); // Should be disabled
    
    const perf_config = getPerformanceConfig();
    try std.testing.expect(perf_config.enable_lock_free == true); // Should be enabled
    
    const test_config = getTestConfig();
    try std.testing.expect(test_config.num_workers.? <= hardware.cpu_count);
}

test "dependency mode handling" {
    const info = getConfigurationInfo();
    
    // Should not crash regardless of configuration mode
    switch (info.mode) {
        .dependency_safe => {
            // Should have safe defaults
            try std.testing.expect(info.worker_count <= 8);
        },
        .runtime_detected => {
            // Should have detected values
            try std.testing.expect(info.worker_count > 0);
        },
        .build_time_detected, .external_provided => {
            // Should have optimal values
            try std.testing.expect(info.worker_count > 0);
        },
    }
}

test "SIMD configuration with fallbacks" {
    if (cpu_features.has_simd) {
        try std.testing.expect(cpu_features.simd_width >= 8);
        
        const FloatVec = OptimalVector(f32);
        try std.testing.expect(@sizeOf(FloatVec) >= @sizeOf(f32));
        
        try std.testing.expect(shouldVectorize(f32));
        try std.testing.expect(getSimdAlignment(f32) >= 4);
    } else {
        // Should gracefully handle no SIMD
        const FloatVec = OptimalVector(f32);
        try std.testing.expect(FloatVec == f32);
    }
}