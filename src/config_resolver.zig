const std = @import("std");
const builtin = @import("builtin");

// Config Resolver: Intelligent build_config detection and fallback system
// Solves the critical dependency chain issue identified by Reverb Code team

pub const ConfigResolver = struct {
    pub fn getBuildConfig(comptime mode: ConfigMode) type {
        return switch (mode) {
            .dependency_safe => FallbackConfig,
            .runtime_detected => RuntimeDetectedConfig,
            .build_time_detected => BuildTimeConfig,
            .external_provided => ExternalConfig,
        };
    }
    
    pub const ConfigMode = enum {
        dependency_safe,     // Safe defaults, no external dependencies
        runtime_detected,    // Runtime hardware detection
        build_time_detected, // Full build-time auto-detection
        external_provided,   // Externally provided build_config
    };
    
    pub fn detectOptimalMode() ConfigMode {
        // Try to determine the best config mode based on available information
        if (@hasDecl(@This(), "build_config")) {
            return .external_provided;
        } else if (comptime isMainProject()) {
            return .build_time_detected;
        } else {
            return .runtime_detected;
        }
    }
    
    fn isMainProject() bool {
        // Heuristic: if we can access build system, we're probably the main project
        return @hasDecl(@This(), "std") and @hasDecl(std, "Build");
    }
};

/// Safe fallback configuration for dependency mode
const FallbackConfig = struct {
    pub const hardware = struct {
        pub const cpu_count: u32 = 8;
        pub const optimal_workers: u32 = 6;
        pub const numa_nodes: u32 = 1;
        pub const cache_line_size: u32 = 64;
        pub const has_avx2: bool = false;
        pub const has_avx512: bool = false;
        pub const has_neon: bool = false;
        pub const memory_gb: u32 = 8;
    };
    
    pub const one_euro = struct {
        pub const frequency: f64 = 60.0;
        pub const min_cutoff: f64 = 1.0;
        pub const beta: f64 = 0.007;
        pub const d_cutoff: f64 = 1.0;
    };
    
    pub const perf = struct {
        pub const optimal_queue_size: u32 = BuildConfig.hardware.optimal_workers * 32;
        pub const enable_numa_optimization: bool = false;
        pub const enable_simd: bool = false;
    };
    
    pub const gpu = struct {
        pub const sycl_available: bool = false;
        pub const enable_integration: bool = false;
        pub const implementation: ?[]const u8 = null;
    };
};

/// Runtime-detected configuration using available Zig standard library functions
const RuntimeDetectedConfig = struct {
    pub const hardware = struct {
        pub const cpu_count: u32 = 8; // Conservative fallback for comptime
        
        pub const optimal_workers: u32 = @max(cpu_count -| 2, 2);
        
        pub const numa_nodes: u32 = blk: {
            // Estimate NUMA nodes based on CPU count
            if (cpu_count <= 4) break :blk 1;
            if (cpu_count <= 16) break :blk 2;
            break :blk 4;
        };
        
        pub const cache_line_size: u32 = std.atomic.cache_line;
        
        pub const has_avx2: bool = blk: {
            if (builtin.cpu.arch != .x86_64) break :blk false;
            break :blk std.Target.x86.featureSetHas(builtin.cpu.features, .avx2);
        };
        
        pub const has_avx512: bool = blk: {
            if (builtin.cpu.arch != .x86_64) break :blk false;
            break :blk std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f);
        };
        
        pub const has_neon: bool = blk: {
            if (builtin.cpu.arch != .aarch64) break :blk false;
            break :blk std.Target.aarch64.featureSetHas(builtin.cpu.features, .neon);
        };
        
        pub const memory_gb: u32 = blk: {
            // Estimate based on CPU count (rough heuristic)
            break :blk @min(cpu_count * 2, 64);
        };
    };
    
    pub const one_euro = struct {
        pub const frequency: f64 = blk: {
            // Adjust frequency based on system capability
            if (RuntimeDetectedConfig.hardware.cpu_count >= 16) break :blk 120.0; // High-end systems
            if (RuntimeDetectedConfig.hardware.cpu_count >= 8) break :blk 90.0;   // Mid-range systems
            break :blk 60.0; // Conservative for smaller systems
        };
        
        pub const min_cutoff: f64 = 1.0;
        pub const beta: f64 = 0.007;
        pub const d_cutoff: f64 = 1.0;
    };
    
    pub const perf = struct {
        pub const optimal_queue_size: u32 = RuntimeDetectedConfig.hardware.optimal_workers * 64;
        pub const enable_numa_optimization: bool = RuntimeDetectedConfig.hardware.numa_nodes > 1;
        pub const enable_simd: bool = RuntimeDetectedConfig.hardware.has_avx2 or RuntimeDetectedConfig.hardware.has_avx512 or RuntimeDetectedConfig.hardware.has_neon;
    };
    
    pub const gpu = struct {
        // Runtime GPU detection is limited, conservative defaults
        pub const sycl_available: bool = false;
        pub const enable_integration: bool = false;
        pub const implementation: ?[]const u8 = null;
    };
};

/// Build-time detected configuration (when available)
const BuildTimeConfig = struct {
    // This would be populated by the build system when available
    // For now, falls back to runtime detection
    pub usingnamespace RuntimeDetectedConfig;
};

/// External configuration (provided by parent project)
const ExternalConfig = struct {
    // This attempts to import external build_config
    pub usingnamespace if (@hasDecl(@This(), "build_config")) 
        @import("build_config").BuildConfig 
    else 
        RuntimeDetectedConfig;
};

/// Smart configuration selector that chooses the best available option
pub fn getSmartConfig() type {
    const mode = ConfigResolver.detectOptimalMode();
    return ConfigResolver.getBuildConfig(mode);
}

/// Export the resolved configuration for use by other modules
pub const BuildConfig = getSmartConfig();

// Re-export for compatibility with existing code
pub const hardware = BuildConfig.hardware;
pub const one_euro = BuildConfig.one_euro;
pub const perf = BuildConfig.perf;

/// Utility function to check what configuration mode is being used
pub fn getConfigMode() ConfigResolver.ConfigMode {
    return ConfigResolver.detectOptimalMode();
}

/// Diagnostic function to help with debugging configuration issues
pub fn printConfigInfo() void {
    const mode = getConfigMode();
    std.log.info("Beat.zig Configuration Mode: {}", .{mode});
    std.log.info("Hardware Config: {} workers, {} NUMA nodes, {} MB cache line", .{
        hardware.optimal_workers,
        hardware.numa_nodes, 
        hardware.cache_line_size
    });
    std.log.info("Features: AVX2={}, AVX512={}, NEON={}", .{
        hardware.has_avx2,
        hardware.has_avx512,
        hardware.has_neon
    });
}