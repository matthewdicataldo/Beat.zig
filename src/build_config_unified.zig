const std = @import("std");
const builtin = @import("builtin");

// ============================================================================
// Unified Build Configuration for Beat.zig
// Consolidates build_opts.zig and build_opts_new.zig into single source
// Resolves configuration drift and import inconsistencies
// ============================================================================

/// Configuration resolution strategy
pub const ConfigStrategy = enum {
    /// Use externally provided build_config module (highest priority)
    external_build_config,
    /// Use compile-time hardware detection from build system
    build_time_detection,
    /// Use runtime hardware detection via Zig stdlib
    runtime_detection,
    /// Use safe fallback values (lowest priority)
    safe_fallback,
};

/// Detect the best available configuration strategy
pub fn detectConfigStrategy() ConfigStrategy {
    // Try external build_config first
    if (hasExternalBuildConfig()) {
        return .external_build_config;
    }
    
    // Check if we have build-time detection available
    if (hasBuildTimeDetection()) {
        return .build_time_detection;
    }
    
    // Runtime detection is available if we have a specific target architecture
    if (builtin.cpu.arch == .x86_64 or builtin.cpu.arch == .aarch64) {
        return .runtime_detection;
    }
    
    // Fallback to safe defaults
    return .safe_fallback;
}

/// Check if external build_config module is available
fn hasExternalBuildConfig() bool {
    return @hasDecl(@This(), "build_config");
}

/// Check if build-time detection is available
fn hasBuildTimeDetection() bool {
    // This would be set by the build system when available
    return false; // For now, assume not available
}

/// Hardware configuration based on detection strategy
pub const hardware = struct {
    const strategy = detectConfigStrategy();
    
    pub const cpu_count: u32 = switch (strategy) {
        .external_build_config => getExternalBuildConfig().detected_cpu_count,
        .build_time_detection => getBuildTimeConfig().cpu_count,
        .runtime_detection => getRuntimeConfig().cpu_count,
        .safe_fallback => 8,
    };
    
    pub const physical_cores: u32 = switch (strategy) {
        .external_build_config => getExternalBuildConfig().detected_physical_cores,
        .build_time_detection => getBuildTimeConfig().physical_cores,
        .runtime_detection => getRuntimeConfig().physical_cores,
        .safe_fallback => cpu_count,
    };
    
    pub const optimal_workers: u32 = switch (strategy) {
        .external_build_config => getExternalBuildConfig().optimal_workers,
        .build_time_detection => getBuildTimeConfig().optimal_workers,
        .runtime_detection => getRuntimeConfig().optimal_workers,
        .safe_fallback => 6,
    };
    
    pub const optimal_test_threads: u32 = @max(optimal_workers / 2, 2);
    
    pub const optimal_queue_size: u32 = switch (strategy) {
        .external_build_config => getExternalBuildConfig().optimal_queue_size,
        .build_time_detection => getBuildTimeConfig().optimal_queue_size,
        .runtime_detection => getRuntimeConfig().optimal_queue_size,
        .safe_fallback => optimal_workers * 32,
    };
    
    pub const numa_nodes: u32 = switch (strategy) {
        .external_build_config => 1, // External config doesn't provide this yet
        .build_time_detection => getBuildTimeConfig().numa_nodes,
        .runtime_detection => getRuntimeConfig().numa_nodes,
        .safe_fallback => 1,
    };
    
    pub const cache_line_size: u32 = switch (strategy) {
        .external_build_config => 64, // Standard assumption
        .build_time_detection => getBuildTimeConfig().cache_line_size,
        .runtime_detection => std.atomic.cache_line,
        .safe_fallback => 64,
    };
    
    pub const memory_gb: u32 = switch (strategy) {
        .external_build_config => 16, // Reasonable assumption
        .build_time_detection => getBuildTimeConfig().memory_gb,
        .runtime_detection => getRuntimeConfig().memory_gb,
        .safe_fallback => 8,
    };
};

/// CPU features configuration
pub const cpu_features = struct {
    const strategy = detectConfigStrategy();
    
    pub const has_avx: bool = switch (strategy) {
        .external_build_config => getExternalBuildConfig().has_avx,
        .build_time_detection => getBuildTimeConfig().has_avx,
        .runtime_detection => getRuntimeConfig().has_avx,
        .safe_fallback => false,
    };
    
    pub const has_avx2: bool = switch (strategy) {
        .external_build_config => getExternalBuildConfig().has_avx2,
        .build_time_detection => getBuildTimeConfig().has_avx2,
        .runtime_detection => getRuntimeConfig().has_avx2,
        .safe_fallback => false,
    };
    
    pub const has_avx512: bool = switch (strategy) {
        .external_build_config => getExternalBuildConfig().has_avx512,
        .build_time_detection => getBuildTimeConfig().has_avx512,
        .runtime_detection => getRuntimeConfig().has_avx512,
        .safe_fallback => false,
    };
    
    pub const has_sse: bool = switch (strategy) {
        .external_build_config => true, // Assume SSE on x86_64
        .build_time_detection => getBuildTimeConfig().has_sse,
        .runtime_detection => getRuntimeConfig().has_sse,
        .safe_fallback => false,
    };
    
    pub const has_neon: bool = switch (strategy) {
        .external_build_config => false, // Not provided by external config
        .build_time_detection => getBuildTimeConfig().has_neon,
        .runtime_detection => getRuntimeConfig().has_neon,
        .safe_fallback => false,
    };
    
    /// Check if any SIMD features are available
    pub const has_simd: bool = has_avx or has_avx2 or has_avx512 or has_sse or has_neon;
    
    /// Get the widest available SIMD width in bytes
    pub const simd_width: u32 = blk: {
        if (has_avx512) break :blk 64;
        if (has_avx2 or has_avx) break :blk 32;
        if (has_sse or has_neon) break :blk 16;
        break :blk 8; // Fallback
    };
};

/// Performance configuration with auto-tuning
pub const performance = struct {
    const strategy = detectConfigStrategy();
    
    pub const one_euro_min_cutoff: f32 = switch (strategy) {
        .external_build_config => getExternalBuildConfig().optimal_one_euro_min_cutoff,
        .build_time_detection => getBuildTimeConfig().one_euro_min_cutoff,
        .runtime_detection => getRuntimeConfig().one_euro_min_cutoff,
        .safe_fallback => 1.0,
    };
    
    pub const one_euro_beta: f32 = switch (strategy) {
        .external_build_config => getExternalBuildConfig().optimal_one_euro_beta,
        .build_time_detection => getBuildTimeConfig().one_euro_beta,
        .runtime_detection => getRuntimeConfig().one_euro_beta,
        .safe_fallback => 0.007,
    };
    
    pub const enable_topology_aware: bool = switch (strategy) {
        .external_build_config => getExternalBuildConfig().enable_topology_aware,
        .build_time_detection => getBuildTimeConfig().enable_topology_aware,
        .runtime_detection => getRuntimeConfig().enable_topology_aware,
        .safe_fallback => false,
    };
    
    pub const enable_numa_aware: bool = switch (strategy) {
        .external_build_config => getExternalBuildConfig().enable_numa_aware,
        .build_time_detection => getBuildTimeConfig().enable_numa_aware,
        .runtime_detection => hardware.numa_nodes > 1,
        .safe_fallback => false,
    };
};

/// Build information
pub const build_info = struct {
    pub const is_debug: bool = switch (detectConfigStrategy()) {
        .external_build_config => getExternalBuildConfig().is_debug_build,
        .build_time_detection => getBuildTimeConfig().is_debug,
        .runtime_detection => std.debug.runtime_safety,
        .safe_fallback => std.debug.runtime_safety,
    };
    
    pub const is_release_fast: bool = switch (detectConfigStrategy()) {
        .external_build_config => getExternalBuildConfig().is_release_fast_build,
        .build_time_detection => getBuildTimeConfig().is_release_fast,
        .runtime_detection => false, // Cannot detect at runtime
        .safe_fallback => false,
    };
    
    pub const is_optimized: bool = !is_debug;
};

// ============================================================================
// Configuration Sources
// ============================================================================

/// External build config (when available)
fn getExternalBuildConfig() type {
    return @import("build_config");
}

/// Build-time detected configuration
const BuildTimeConfig = struct {
    // This would be populated by the build system
    // For now, falls back to runtime detection
    pub usingnamespace RuntimeConfig;
};

fn getBuildTimeConfig() type {
    return BuildTimeConfig;
}

/// Runtime detected configuration
const RuntimeConfig = struct {
    pub const cpu_count: u32 = 8; // Conservative fallback for comptime
    
    pub const physical_cores: u32 = cpu_count; // Approximation
    pub const optimal_workers: u32 = @max(cpu_count -| 2, 2);
    pub const optimal_queue_size: u32 = optimal_workers * 64;
    
    /// Get actual CPU count at runtime (for runtime usage)
    pub fn getActualCpuCount() u32 {
        return @min(std.Thread.getCpuCount() catch 8, 64);
    }
    
    pub const numa_nodes: u32 = blk: {
        // Estimate NUMA nodes based on CPU count
        if (cpu_count <= 4) break :blk 1;
        if (cpu_count <= 16) break :blk 2;
        if (cpu_count <= 32) break :blk 4;
        break :blk 8;
    };
    
    pub const cache_line_size: u32 = std.atomic.cache_line;
    pub const memory_gb: u32 = @min(cpu_count * 2, 128); // Rough estimate
    
    pub const has_avx: bool = blk: {
        if (builtin.cpu.arch != .x86_64) break :blk false;
        break :blk std.Target.x86.featureSetHas(builtin.cpu.features, .avx);
    };
    
    pub const has_avx2: bool = blk: {
        if (builtin.cpu.arch != .x86_64) break :blk false;
        break :blk std.Target.x86.featureSetHas(builtin.cpu.features, .avx2);
    };
    
    pub const has_avx512: bool = blk: {
        if (builtin.cpu.arch != .x86_64) break :blk false;
        break :blk std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f);
    };
    
    pub const has_sse: bool = blk: {
        if (builtin.cpu.arch != .x86_64) break :blk false;
        break :blk std.Target.x86.featureSetHas(builtin.cpu.features, .sse2);
    };
    
    pub const has_neon: bool = blk: {
        if (builtin.cpu.arch != .aarch64) break :blk false;
        break :blk std.Target.aarch64.featureSetHas(builtin.cpu.features, .neon);
    };
    
    pub const one_euro_min_cutoff: f32 = blk: {
        // Adjust based on system capability
        if (cpu_count >= 16) break :blk 0.8; // High-end systems can handle more prediction
        if (cpu_count >= 8) break :blk 1.0;  // Mid-range systems
        break :blk 1.2; // Conservative for smaller systems
    };
    
    pub const one_euro_beta: f32 = blk: {
        // Adjust based on system capability
        if (cpu_count >= 16) break :blk 0.005; // More aggressive filtering
        if (cpu_count >= 8) break :blk 0.007;  // Balanced
        break :blk 0.01; // Conservative
    };
    
    pub const enable_topology_aware: bool = numa_nodes > 1;
    pub const enable_numa_aware: bool = numa_nodes > 1;
    pub const is_debug: bool = std.debug.runtime_safety;
    pub const is_release_fast: bool = false; // Cannot detect at runtime
};

fn getRuntimeConfig() type {
    return RuntimeConfig;
}

// ============================================================================
// Configuration Factory Functions
// ============================================================================

/// Get optimal Beat.zig Config based on unified configuration
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

/// Get configuration optimized for testing
pub fn getTestConfig() @import("core.zig").Config {
    var config = getOptimalConfig();
    
    // Test-specific optimizations
    config.num_workers = hardware.optimal_test_threads;
    config.task_queue_size = 128; // Smaller queues for tests
    config.enable_topology_aware = false; // Simpler for testing
    config.enable_numa_aware = false;     // Simpler for testing
    config.enable_statistics = true;      // Always enable for testing
    config.enable_trace = build_info.is_debug;
    
    return config;
}

/// Get configuration optimized for benchmarking
pub fn getBenchmarkConfig() @import("core.zig").Config {
    var config = getOptimalConfig();
    
    // Benchmark-specific optimizations
    config.task_queue_size = hardware.optimal_workers * 256; // Large queues
    config.enable_statistics = true;  // Need stats for benchmarks
    config.enable_trace = false;      // No tracing overhead
    config.enable_predictive = true;  // Enable prediction for benchmarks (fixes inconsistency)
    
    // More aggressive One Euro Filter for benchmarks
    config.prediction_beta = config.prediction_beta * 1.2;
    
    return config;
}

/// Get basic configuration safe for any environment
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

// ============================================================================
// SIMD Optimization Helpers
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
// Utility Functions
// ============================================================================

/// Print unified configuration summary
pub fn printSummary() void {
    const strategy = detectConfigStrategy();
    
    std.debug.print("=== Beat.zig Unified Configuration Summary ===\n", .{});
    std.debug.print("Configuration Strategy: {}\n", .{strategy});
    
    std.debug.print("Hardware:\n", .{});
    std.debug.print("  CPU Count: {}\n", .{hardware.cpu_count});
    std.debug.print("  Physical Cores: {}\n", .{hardware.physical_cores});
    std.debug.print("  Optimal Workers: {}\n", .{hardware.optimal_workers});
    std.debug.print("  Queue Size: {}\n", .{hardware.optimal_queue_size});
    std.debug.print("  NUMA Nodes: {}\n", .{hardware.numa_nodes});
    std.debug.print("  Cache Line: {} bytes\n", .{hardware.cache_line_size});
    
    std.debug.print("CPU Features:\n", .{});
    std.debug.print("  SIMD Available: {}\n", .{cpu_features.has_simd});
    std.debug.print("  SIMD Width: {} bytes\n", .{cpu_features.simd_width});
    if (cpu_features.has_avx512) std.debug.print("  AVX-512: Yes\n", .{});
    if (cpu_features.has_avx2) std.debug.print("  AVX2: Yes\n", .{});
    if (cpu_features.has_avx) std.debug.print("  AVX: Yes\n", .{});
    if (cpu_features.has_sse) std.debug.print("  SSE: Yes\n", .{});
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
    std.debug.print("=============================================\n", .{});
}

/// Compile-time validation of unified configuration
pub fn validateConfiguration() void {
    // Compile-time checks
    if (hardware.optimal_workers == 0) {
        @compileError("Invalid worker count detected in unified configuration");
    }
    
    if (hardware.optimal_queue_size < 16) {
        @compileError("Queue size too small in unified configuration");
    }
    
    if (performance.one_euro_beta <= 0.0 or performance.one_euro_beta > 1.0) {
        @compileError("Invalid One Euro Filter beta parameter in unified configuration");
    }
    
    // Strategy-specific validation
    const strategy = detectConfigStrategy();
    if (strategy == .safe_fallback) {
        std.log.warn("Beat.zig using safe fallback configuration - some features may be disabled", .{});
    }
}

/// Get diagnostic information about current configuration
pub fn getDiagnosticInfo() DiagnosticInfo {
    return DiagnosticInfo{
        .strategy = detectConfigStrategy(),
        .has_external_config = hasExternalBuildConfig(),
        .has_build_time_detection = hasBuildTimeDetection(),
        .worker_count = hardware.optimal_workers,
        .queue_size = hardware.optimal_queue_size,
        .simd_available = cpu_features.has_simd,
        .numa_nodes = hardware.numa_nodes,
    };
}

pub const DiagnosticInfo = struct {
    strategy: ConfigStrategy,
    has_external_config: bool,
    has_build_time_detection: bool,
    worker_count: u32,
    queue_size: u32,
    simd_available: bool,
    numa_nodes: u32,
    
    pub fn print(self: DiagnosticInfo) void {
        std.log.info("Beat.zig Unified Configuration Diagnostics:", .{});
        std.log.info("  Strategy: {}", .{self.strategy});
        std.log.info("  External Config: {}", .{self.has_external_config});
        std.log.info("  Build-time Detection: {}", .{self.has_build_time_detection});
        std.log.info("  Workers: {}", .{self.worker_count});
        std.log.info("  Queue Size: {}", .{self.queue_size});
        std.log.info("  SIMD: {}", .{self.simd_available});
        std.log.info("  NUMA Nodes: {}", .{self.numa_nodes});
    }
};

// ============================================================================
// Testing Integration
// ============================================================================

test "unified configuration validation" {
    validateConfiguration();
    
    const config = getOptimalConfig();
    try std.testing.expect(config.num_workers.? > 0);
    try std.testing.expect(config.task_queue_size > 0);
    
    const test_config = getTestConfig();
    try std.testing.expect(test_config.num_workers.? <= hardware.cpu_count);
    
    const bench_config = getBenchmarkConfig();
    try std.testing.expect(bench_config.task_queue_size >= config.task_queue_size);
    try std.testing.expect(bench_config.enable_predictive == true); // Fixed inconsistency
}

test "configuration strategy detection" {
    const strategy = detectConfigStrategy();
    
    // Should not crash regardless of strategy
    switch (strategy) {
        .external_build_config => {
            try std.testing.expect(hasExternalBuildConfig());
        },
        .build_time_detection => {
            try std.testing.expect(hasBuildTimeDetection());
        },
        .runtime_detection => {
            try std.testing.expect(builtin.cpu.arch == .x86_64 or builtin.cpu.arch == .aarch64);
        },
        .safe_fallback => {
            // Safe fallback should always work
        },
    }
}

test "SIMD configuration unified" {
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

test "diagnostic information" {
    const info = getDiagnosticInfo();
    
    try std.testing.expect(info.worker_count > 0);
    try std.testing.expect(info.queue_size > 0);
    try std.testing.expect(info.numa_nodes >= 1);
    
    // Should not crash when printing
    info.print();
}