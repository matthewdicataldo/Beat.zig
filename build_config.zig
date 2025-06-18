const std = @import("std");
const builtin = @import("builtin");

// Build System Auto-Configuration for Beat.zig
// Provides intelligent build-time system detection and optimization

// ============================================================================
// System Detection and Auto-Configuration
// ============================================================================

/// Build-time system configuration detected automatically
pub const BuildConfig = struct {
    // Hardware detection
    cpu_count: u32,
    physical_cores: u32,
    cache_line_size: u32,
    numa_nodes: u32,
    
    // CPU features
    has_avx: bool,
    has_avx2: bool,
    has_avx512: bool,
    has_sse: bool,
    has_neon: bool,
    
    // Optimal configurations
    optimal_workers: u32,
    optimal_test_threads: u32,
    optimal_queue_size: u32,
    
    // Build target specific
    is_debug: bool,
    is_release_fast: bool,
    target_arch: std.Target.Cpu.Arch,
    target_os: std.Target.Os.Tag,
    
    // Performance presets
    one_euro_min_cutoff: f32,
    one_euro_beta: f32,
    enable_topology_aware: bool,
    enable_numa_aware: bool,
    
};

/// Detect system configuration at build time
pub fn detectBuildConfig(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) BuildConfig {
    var config = BuildConfig{
        .cpu_count = 1,
        .physical_cores = 1,
        .cache_line_size = 64,
        .numa_nodes = 1,
        .has_avx = false,
        .has_avx2 = false,
        .has_avx512 = false,
        .has_sse = false,
        .has_neon = false,
        .optimal_workers = 1,
        .optimal_test_threads = 1,
        .optimal_queue_size = 256,
        .is_debug = optimize == .Debug,
        .is_release_fast = optimize == .ReleaseFast,
        .target_arch = target.result.cpu.arch,
        .target_os = target.result.os.tag,
        .one_euro_min_cutoff = 1.0,
        .one_euro_beta = 0.1,
        .enable_topology_aware = true,
        .enable_numa_aware = true,
    };
    
    // Detect CPU count (cross-platform)
    config.cpu_count = detectCpuCount(b, target);
    config.physical_cores = estimatePhysicalCores(config.cpu_count, target.result.cpu.arch);
    
    // Detect CPU features based on target
    detectCpuFeatures(&config, target);
    
    // Calculate optimal configurations
    calculateOptimalSettings(&config);
    
    // Apply target-specific optimizations
    applyTargetOptimizations(&config, target, optimize);
    
    // Detect SYCL availability
    
    return config;
}

/// Cross-platform CPU count detection
/// 
/// This function safely detects CPU count for both native compilation and cross-compilation.
/// For native builds, it probes the actual hardware. For cross-compilation, it uses
/// reasonable defaults based on the target architecture to avoid build-time runtime dependencies.
fn detectCpuCount(b: *std.Build, target: std.Build.ResolvedTarget) u32 {
    _ = b; // Build context not needed for CPU detection
    
    // For cross-compilation, avoid runtime hardware probing
    // Only probe hardware when building for the same platform
    const target_info = target.result;
    const is_native_build = builtin.os.tag == target_info.os.tag and 
                           builtin.cpu.arch == target_info.cpu.arch;
    
    if (is_native_build) {
        // Safe to probe hardware on native builds
        const detected = std.Thread.getCpuCount() catch 4;
        const cpu_count = @as(u32, @intCast(detected));
        return std.math.clamp(cpu_count, 1, 128);
    } else {
        // Cross-compilation: use reasonable defaults based on target architecture
        return getDefaultCpuCountForArch(target_info.cpu.arch);
    }
}

/// Get default CPU count for target architecture during cross-compilation
pub fn getDefaultCpuCountForArch(arch: std.Target.Cpu.Arch) u32 {
    // Return reasonable defaults based on common configurations for each architecture
    return switch (arch) {
        .x86_64 => 8,   // Desktop/server x86_64 commonly has 4-16 cores
        .aarch64 => 4,  // ARM64 devices commonly have 4-8 cores
        .arm => 4,      // ARM32 devices commonly have 2-4 cores
        .riscv64 => 4,  // RISC-V typically 4-8 cores
        .wasm32, .wasm64 => 1, // WebAssembly is single-threaded by default
        else => 4,      // Conservative default for unknown architectures
    };
}

/// Estimate physical cores from logical cores
fn estimatePhysicalCores(logical_cores: u32, arch: std.Target.Cpu.Arch) u32 {
    // Heuristic: Most modern CPUs have 2-way SMT
    return switch (arch) {
        .x86_64 => @max(1, logical_cores / 2), // Intel/AMD typically 2-way SMT
        .aarch64 => logical_cores, // ARM typically no SMT
        .riscv64 => logical_cores, // RISC-V typically no SMT
        else => @max(1, logical_cores / 2), // Conservative default
    };
}

/// Detect CPU features based on target
fn detectCpuFeatures(config: *BuildConfig, target: std.Build.ResolvedTarget) void {
    const cpu_features = target.result.cpu.features;
    
    switch (config.target_arch) {
        .x86_64 => {
            // Check x86_64 SIMD features
            config.has_sse = cpu_features.isEnabled(@intFromEnum(std.Target.x86.Feature.sse));
            config.has_avx = cpu_features.isEnabled(@intFromEnum(std.Target.x86.Feature.avx));
            config.has_avx2 = cpu_features.isEnabled(@intFromEnum(std.Target.x86.Feature.avx2));
            config.has_avx512 = cpu_features.isEnabled(@intFromEnum(std.Target.x86.Feature.avx512f));
        },
        .aarch64 => {
            // Check ARM NEON features
            config.has_neon = cpu_features.isEnabled(@intFromEnum(std.Target.aarch64.Feature.neon));
        },
        else => {
            // Other architectures - conservative defaults
        },
    }
}

/// Calculate optimal settings based on detected hardware
fn calculateOptimalSettings(config: *BuildConfig) void {
    // Optimal worker count: Usually physical cores for CPU-bound work
    config.optimal_workers = config.physical_cores;
    
    // For systems with many cores, cap workers to prevent resource contention
    if (config.optimal_workers > 16) {
        config.optimal_workers = 16;
    }
    
    // Test parallelization: Use fewer threads to avoid overwhelming the system
    config.optimal_test_threads = @max(1, config.optimal_workers / 2);
    
    // Queue size based on worker count and expected load
    config.optimal_queue_size = config.optimal_workers * 64;
    
    // NUMA awareness: Enable for systems with multiple NUMA nodes
    // Conservative estimate: more than 8 cores likely has NUMA
    if (config.cpu_count > 8) {
        config.numa_nodes = @max(1, config.cpu_count / 8);
        config.enable_numa_aware = true;
    } else {
        config.enable_numa_aware = false;
    }
    
    // One Euro Filter tuning based on system characteristics
    tuneOneEuroFilter(config);
}

/// Tune One Euro Filter parameters based on system characteristics
fn tuneOneEuroFilter(config: *BuildConfig) void {
    // Base tuning on CPU count and architecture characteristics
    if (config.cpu_count >= 16) {
        // High-performance systems: More aggressive adaptation
        config.one_euro_min_cutoff = 1.5; // More responsive
        config.one_euro_beta = 0.15;      // Faster adaptation
        
        // SIMD-capable systems can handle more aggressive filtering
        if (config.has_avx2 or config.has_avx512) {
            config.one_euro_beta = 0.18; // Even faster for vector workloads
        }
    } else if (config.cpu_count >= 8) {
        // Mid-range systems: Balanced settings
        config.one_euro_min_cutoff = 1.0; // Default
        config.one_euro_beta = 0.1;       // Default
        
        // Adjust for SIMD availability
        if (config.has_avx or config.has_neon) {
            config.one_euro_min_cutoff = 1.2; // Slightly more responsive
        }
    } else {
        // Low-core systems: More conservative
        config.one_euro_min_cutoff = 0.7; // More stable
        config.one_euro_beta = 0.08;      // Slower adaptation
        
        // Even more conservative for embedded/mobile
        if (config.target_arch == .aarch64 and config.cpu_count <= 4) {
            config.one_euro_min_cutoff = 0.5;
            config.one_euro_beta = 0.05;
        }
    }
    
    // Architecture-specific adjustments
    switch (config.target_arch) {
        .x86_64 => {
            // x86_64 typically has good branch prediction and larger caches
            // Can handle slightly more aggressive adaptation
            config.one_euro_beta *= 1.1;
        },
        .aarch64 => {
            // ARM often has simpler cores, be more conservative
            config.one_euro_beta *= 0.9;
        },
        else => {
            // Conservative for unknown architectures
            config.one_euro_beta *= 0.8;
        },
    }
    
    // NUMA-aware adjustment
    if (config.numa_nodes > 1) {
        // NUMA systems have variable memory latencies
        // Use more conservative filtering to handle outliers
        config.one_euro_min_cutoff *= 0.9; // Slightly more stable
        config.one_euro_beta *= 0.95;      // Slightly slower adaptation
    }
}

/// Apply target-specific optimizations
fn applyTargetOptimizations(config: *BuildConfig, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    _ = target;
    
    // Debug builds: Conservative settings for debugging
    if (config.is_debug) {
        config.optimal_workers = @min(config.optimal_workers, 4);
        config.optimal_queue_size = 128;
        config.enable_topology_aware = false; // Simpler debugging
    }
    
    // Release builds: Aggressive optimizations
    if (config.is_release_fast) {
        // Enable all optimizations
        config.enable_topology_aware = true;
        config.enable_numa_aware = config.numa_nodes > 1;
        
        // Larger queues for throughput
        config.optimal_queue_size = config.optimal_workers * 128;
    }
    
    // Test builds: Optimized for test execution
    if (optimize == .ReleaseSafe) {
        // Balance between performance and safety
        config.optimal_test_threads = @min(config.optimal_test_threads, 6);
    }
}

/// Generate build options for the Zig build system
pub fn addBuildOptions(b: *std.Build, exe: *std.Build.Step.Compile, config: BuildConfig) void {
    // Add build-time constants
    const options = b.addOptions();
    
    // Hardware configuration
    options.addOption(u32, "detected_cpu_count", config.cpu_count);
    options.addOption(u32, "detected_physical_cores", config.physical_cores);
    options.addOption(u32, "optimal_workers", config.optimal_workers);
    options.addOption(u32, "optimal_test_threads", config.optimal_test_threads);
    options.addOption(u32, "optimal_queue_size", config.optimal_queue_size);
    
    // CPU features
    options.addOption(bool, "has_avx", config.has_avx);
    options.addOption(bool, "has_avx2", config.has_avx2);
    options.addOption(bool, "has_avx512", config.has_avx512);
    options.addOption(bool, "has_sse", config.has_sse);
    options.addOption(bool, "has_neon", config.has_neon);
    
    // Performance configuration
    options.addOption(f32, "optimal_one_euro_min_cutoff", config.one_euro_min_cutoff);
    options.addOption(f32, "optimal_one_euro_beta", config.one_euro_beta);
    options.addOption(bool, "enable_topology_aware", config.enable_topology_aware);
    options.addOption(bool, "enable_numa_aware", config.enable_numa_aware);
    
    // Build information
    options.addOption(bool, "is_debug_build", config.is_debug);
    options.addOption(bool, "is_release_fast_build", config.is_release_fast);
    
    // GPU/SYCL configuration
    
    exe.root_module.addOptions("build_config", options);
}

/// Print configuration summary for build output
pub fn printConfigSummary(config: BuildConfig) void {
    std.debug.print("\n=== Beat.zig Auto-Configuration ===\n", .{});
    std.debug.print("CPU Count: {} (Physical: {})\n", .{ config.cpu_count, config.physical_cores });
    std.debug.print("Target: {s}-{s}\n", .{ @tagName(config.target_arch), @tagName(config.target_os) });
    
    // CPU Features
    var features = std.ArrayList([]const u8).init(std.heap.page_allocator);
    defer features.deinit();
    
    if (config.has_sse) features.append("SSE") catch {};
    if (config.has_avx) features.append("AVX") catch {};
    if (config.has_avx2) features.append("AVX2") catch {};
    if (config.has_avx512) features.append("AVX512") catch {};
    if (config.has_neon) features.append("NEON") catch {};
    
    std.debug.print("SIMD Features: ", .{});
    for (features.items, 0..) |feature, i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{s}", .{feature});
    }
    std.debug.print("\n", .{});
    
    // Optimal settings
    std.debug.print("Optimal Workers: {}\n", .{config.optimal_workers});
    std.debug.print("Test Threads: {}\n", .{config.optimal_test_threads});
    std.debug.print("Queue Size: {}\n", .{config.optimal_queue_size});
    std.debug.print("NUMA Aware: {}\n", .{config.enable_numa_aware});
    std.debug.print("Topology Aware: {}\n", .{config.enable_topology_aware});
    std.debug.print("One Euro Filter: min_cutoff={d:.2}, beta={d:.3}\n", .{ config.one_euro_min_cutoff, config.one_euro_beta });
    
    // GPU/SYCL configuration
    std.debug.print("\nGPU Integration:\n", .{});
    std.debug.print("=====================================\n\n", .{});
}


// ============================================================================
// Target-Specific Optimizations
// ============================================================================

/// Get recommended compiler flags for the target
pub fn getRecommendedCompilerFlags(config: BuildConfig) []const []const u8 {
    var flags = std.ArrayList([]const u8).init(std.heap.page_allocator);
    
    if (config.target_arch == .x86_64) {
        if (config.has_avx2) {
            flags.append("-mavx2") catch {};
        } else if (config.has_avx) {
            flags.append("-mavx") catch {};
        }
        
        if (config.has_sse) {
            flags.append("-msse4.2") catch {};
        }
    }
    
    if (config.target_arch == .aarch64 and config.has_neon) {
        flags.append("-mfpu=neon") catch {};
    }
    
    return flags.toOwnedSlice() catch &[_][]const u8{};
}

/// Get optimal memory allocator configuration
pub fn getOptimalAllocatorConfig(config: BuildConfig) struct {
    page_size: usize,
    alignment: usize,
    enable_thread_local: bool,
} {
    return .{
        .page_size = if (config.numa_nodes > 1) 2 * 1024 * 1024 else 4096, // 2MB for NUMA
        .alignment = config.cache_line_size,
        .enable_thread_local = config.cpu_count > 4,
    };
}