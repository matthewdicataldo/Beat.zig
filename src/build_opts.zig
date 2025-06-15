const std = @import("std");
const build_config = @import("build_config");

// Build-time configuration access for Beat.zig
// Provides compile-time access to auto-detected system configuration

// ============================================================================
// Build Configuration Access
// ============================================================================

/// Auto-detected hardware configuration available at compile time
pub const hardware = struct {
    pub const cpu_count: u32 = build_config.detected_cpu_count;
    pub const physical_cores: u32 = build_config.detected_physical_cores;
    pub const optimal_workers: u32 = build_config.optimal_workers;
    pub const optimal_test_threads: u32 = build_config.optimal_test_threads;
    pub const optimal_queue_size: u32 = build_config.optimal_queue_size;
};

/// Auto-detected CPU features available at compile time
pub const cpu_features = struct {
    pub const has_avx: bool = build_config.has_avx;
    pub const has_avx2: bool = build_config.has_avx2;
    pub const has_avx512: bool = build_config.has_avx512;
    pub const has_sse: bool = build_config.has_sse;
    pub const has_neon: bool = build_config.has_neon;
    
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

/// Auto-tuned performance configuration
pub const performance = struct {
    pub const one_euro_min_cutoff: f32 = build_config.optimal_one_euro_min_cutoff;
    pub const one_euro_beta: f32 = build_config.optimal_one_euro_beta;
    pub const enable_topology_aware: bool = build_config.enable_topology_aware;
    pub const enable_numa_aware: bool = build_config.enable_numa_aware;
};

/// Build configuration information
pub const build_info = struct {
    pub const is_debug: bool = build_config.is_debug_build;
    pub const is_release_fast: bool = build_config.is_release_fast_build;
    pub const is_optimized: bool = !is_debug;
};

// ============================================================================
// Utility Functions
// ============================================================================

/// Get optimal Beat.zig Config based on build-time detection
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
        .enable_statistics = !build_info.is_release_fast, // Disable stats in release-fast
        .enable_trace = build_info.is_debug,              // Only trace in debug
    };
}

/// Get configuration optimized for testing
pub fn getTestConfig() @import("core.zig").Config {
    var config = getOptimalConfig();
    
    // Test-specific optimizations
    config.num_workers = hardware.optimal_test_threads;
    config.task_queue_size = 256; // Smaller queues for tests
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
    
    // More aggressive One Euro Filter for benchmarks
    config.prediction_beta = config.prediction_beta * 1.2;
    
    return config;
}

/// Print auto-configuration summary
pub fn printSummary() void {
    std.debug.print("=== Beat.zig Auto-Configuration Summary ===\n", .{});
    std.debug.print("Hardware:\n", .{});
    std.debug.print("  CPU Count: {}\n", .{hardware.cpu_count});
    std.debug.print("  Physical Cores: {}\n", .{hardware.physical_cores});
    std.debug.print("  Optimal Workers: {}\n", .{hardware.optimal_workers});
    std.debug.print("  Queue Size: {}\n", .{hardware.optimal_queue_size});
    
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
    std.debug.print("==========================================\n", .{});
}

/// Compile-time validation of configuration
pub fn validateConfiguration() void {
    // Compile-time checks
    if (hardware.optimal_workers == 0) {
        @compileError("Invalid worker count detected");
    }
    
    if (hardware.optimal_queue_size < 16) {
        @compileError("Queue size too small");
    }
    
    if (performance.one_euro_beta <= 0.0 or performance.one_euro_beta > 1.0) {
        @compileError("Invalid One Euro Filter beta parameter");
    }
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
// Testing Integration
// ============================================================================

test "build configuration validation" {
    validateConfiguration();
    
    const config = getOptimalConfig();
    try std.testing.expect(config.num_workers.? > 0);
    try std.testing.expect(config.task_queue_size > 0);
    
    const test_config = getTestConfig();
    try std.testing.expect(test_config.num_workers.? <= hardware.cpu_count);
    
    const bench_config = getBenchmarkConfig();
    try std.testing.expect(bench_config.task_queue_size >= config.task_queue_size);
}

test "SIMD configuration" {
    if (cpu_features.has_simd) {
        try std.testing.expect(cpu_features.simd_width >= 8);
        
        const FloatVec = OptimalVector(f32);
        try std.testing.expect(@sizeOf(FloatVec) >= @sizeOf(f32));
        
        try std.testing.expect(shouldVectorize(f32));
        try std.testing.expect(getSimdAlignment(f32) >= 4);
    }
}