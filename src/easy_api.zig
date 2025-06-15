const std = @import("std");
const core = @import("core.zig");
const build_opts = @import("build_opts_new.zig");
const enhanced_errors = @import("enhanced_errors.zig");

// Easy API for Beat.zig - Progressive Feature Adoption
// Addresses Reverb Code team feedback for simplified integration

/// Progressive feature levels for gradual adoption
pub const FeatureLevel = enum {
    basic,      // No dependencies, just works - for external projects
    performance, // Lock-free + topology, minimal dependencies
    advanced,   // Full predictive scheduling with all features
};

// ============================================================================
// Level 1: Basic Pool - Zero Dependencies, Maximum Compatibility
// ============================================================================

/// Create a basic thread pool with no external dependencies
/// Perfect for external projects integrating Beat as a dependency
pub fn createBasicPool(allocator: std.mem.Allocator, worker_count: u32) !*core.ThreadPool {
    // Enhanced input validation with helpful error messages
    if (worker_count == 0) {
        std.log.err("{s}", .{enhanced_errors.formatMissingBuildConfigError("your project")[0..500]});
        std.log.err(
            \\
            \\❌ Invalid worker count: 0
            \\💡 SOLUTION: Use a positive number:
            \\   const pool = try beat.createBasicPool(allocator, 4);
            \\   OR auto-detect:
            \\   const pool = try beat.createBasicPoolAuto(allocator);
            \\
        , .{});
        return enhanced_errors.ConfigError.InvalidConfiguration;
    }
    
    if (worker_count > 64) {
        std.log.warn(
            \\
            \\⚠️  High worker count: {} workers requested
            \\💡 RECOMMENDATION: Most systems perform best with:
            \\   • Desktop: 4-8 workers
            \\   • Server: CPU count - 2
            \\   • Use auto-detection: beat.createBasicPoolAuto(allocator)
            \\
        , .{worker_count});
    }
    
    const config = core.Config{
        .num_workers = worker_count,
        .task_queue_size = worker_count * 32, // Simple calculation
        .enable_lock_free = true,            // Keep core performance
        .enable_work_stealing = true,        // Keep work stealing
        .enable_topology_aware = false,      // No complex dependencies
        .enable_numa_aware = false,          // No NUMA complexity  
        .enable_predictive = false,          // No prediction complexity
        .enable_advanced_selection = false,  // Simple round-robin
        .enable_heartbeat = false,           // No heartbeat overhead
        .enable_statistics = false,          // No stats overhead
        .enable_trace = false,              // No tracing
    };
    
    return core.ThreadPool.init(allocator, config) catch |err| {
        // Provide enhanced error context for common issues
        switch (err) {
            error.OutOfMemory => {
                std.log.err(
                    \\
                    \\❌ Out of Memory creating thread pool
                    \\💡 SOLUTIONS:
                    \\   • Try fewer workers: beat.createBasicPool(allocator, 2)
                    \\   • Check available memory with: ps aux | grep your-app
                    \\   • Consider using arena allocator for temporary pools
                    \\
                , .{});
                return err;
            },
            else => {
                std.log.err(
                    \\
                    \\❌ Failed to create basic thread pool: {}
                    \\💡 TROUBLESHOOTING:
                    \\   • Try development mode: beat.createDevelopmentPool(allocator, .{{}})
                    \\   • Check system resources: ulimit -a
                    \\   • See integration guide: https://github.com/Beat-zig/Beat.zig/blob/main/INTEGRATION_GUIDE.md
                    \\
                , .{err});
                return err;
            },
        }
    };
}

/// Create basic pool with automatic worker count detection
pub fn createBasicPoolAuto(allocator: std.mem.Allocator) !*core.ThreadPool {
    const worker_count = @max(@as(u32, @intCast(std.Thread.getCpuCount() catch blk: {
        enhanced_errors.logEnhancedError(
            enhanced_errors.ConfigError, 
            enhanced_errors.ConfigError.HardwareDetectionFailed, 
            "CPU count auto-detection"
        );
        std.log.info("🔧 Using conservative fallback: 4 workers", .{});
        break :blk 4;
    })) -| 2, 2);
    
    std.log.info(
        \\
        \\✅ Beat.zig auto-detected configuration:
        \\   Workers: {} (CPU cores minus 2, minimum 2)
        \\   Features: Basic (work-stealing, lock-free queues)
        \\   Compatibility: Maximum (no complex dependencies)
        \\
        \\💡 To customize: beat.createBasicPool(allocator, {})
        \\
    , .{ worker_count, worker_count });
    
    return try createBasicPool(allocator, worker_count);
}

// ============================================================================
// Level 2: Performance Pool - Optimized with Minimal Dependencies  
// ============================================================================

/// Options for performance pool creation
pub const PerformanceOptions = struct {
    workers: ?u32 = null,                   // Auto-detect if null
    enable_work_stealing: bool = true,      // Work stealing for performance
    enable_lock_free: bool = true,          // Lock-free data structures
    enable_topology_aware: bool = true,     // Basic topology awareness
    queue_size_multiplier: u32 = 64,        // Queue size = workers * multiplier
};

/// Create performance-optimized pool with minimal dependencies
pub fn createPerformancePool(allocator: std.mem.Allocator, options: PerformanceOptions) !*core.ThreadPool {
    const worker_count = options.workers orelse build_opts.hardware.optimal_workers;
    
    // Enhanced validation for performance pool
    if (worker_count > 32) {
        std.log.warn(
            \\
            \\⚠️  Very high worker count: {} workers for performance pool
            \\💡 PERFORMANCE NOTES:
            \\   • High worker counts can increase contention
            \\   • Consider using advanced pool for complex workloads
            \\   • Monitor with: pool.getStats() 
            \\
        , .{worker_count});
    }
    
    const config = core.Config{
        .num_workers = worker_count,
        .task_queue_size = worker_count * options.queue_size_multiplier,
        .enable_lock_free = options.enable_lock_free,
        .enable_work_stealing = options.enable_work_stealing,
        .enable_topology_aware = options.enable_topology_aware,
        .enable_numa_aware = false,          // Keep simple for now
        .enable_predictive = false,          // No prediction overhead
        .enable_advanced_selection = false,  // Keep selection simple
        .enable_heartbeat = true,            // Enable heartbeat for performance
        .enable_statistics = false,          // No stats overhead
        .enable_trace = false,              // No tracing overhead
    };
    
    return core.ThreadPool.init(allocator, config) catch |err| {
        // Provide context-specific error handling for performance pool
        switch (err) {
            error.OutOfMemory => {
                std.log.err(
                    \\
                    \\❌ Out of Memory creating performance pool
                    \\💡 SOLUTIONS:
                    \\   • Reduce workers: .workers = {}
                    \\   • Reduce queue size: .queue_size_multiplier = 32
                    \\   • Use basic pool: beat.createBasicPool(allocator, {})
                    \\
                , .{ @max(worker_count / 2, 2), @max(worker_count / 2, 2) });
                return err;
            },
            enhanced_errors.ConfigError.HardwareDetectionFailed => {
                std.log.warn(
                    \\
                    \\⚠️  Hardware detection failed for performance pool
                    \\💡 FALLBACK OPTIONS:
                    \\   • Specify workers manually: .workers = 4
                    \\   • Use basic pool: beat.createBasicPool(allocator, 4)
                    \\   • Try runtime detection: beat.detectOptimalConfig(allocator)
                    \\
                , .{});
                return err;
            },
            else => {
                std.log.err(
                    \\
                    \\❌ Failed to create performance pool: {}
                    \\💡 TROUBLESHOOTING:
                    \\   • Check if topology detection is supported on your platform
                    \\   • Try disabling topology: .enable_topology_aware = false
                    \\   • Fall back to basic pool for guaranteed compatibility
                    \\
                , .{err});
                return err;
            },
        }
    };
}

// ============================================================================
// Level 3: Advanced Pool - Full Features for Power Users
// ============================================================================

/// Options for advanced pool creation
pub const AdvancedOptions = struct {
    workers: ?u32 = null,                   // Auto-detect if null
    enable_predictive: bool = true,         // Predictive scheduling
    enable_advanced_selection: bool = true, // Advanced worker selection  
    enable_numa_aware: bool = true,         // NUMA optimizations
    enable_statistics: bool = true,         // Performance statistics
    enable_trace: bool = false,             // Debug tracing
    
    // Advanced tuning parameters
    prediction_min_cutoff: ?f32 = null,     // One Euro Filter tuning
    prediction_beta: ?f32 = null,           // One Euro Filter tuning
    queue_size_multiplier: u32 = 128,       // Larger queues for advanced features
};

/// Create advanced pool with all features enabled
pub fn createAdvancedPool(allocator: std.mem.Allocator, options: AdvancedOptions) !*core.ThreadPool {
    const worker_count = options.workers orelse build_opts.hardware.optimal_workers;
    
    const config = core.Config{
        .num_workers = worker_count,
        .task_queue_size = worker_count * options.queue_size_multiplier,
        .enable_lock_free = true,
        .enable_work_stealing = true,
        .enable_topology_aware = true,
        .enable_numa_aware = options.enable_numa_aware,
        .enable_predictive = options.enable_predictive,
        .enable_advanced_selection = options.enable_advanced_selection,
        .enable_heartbeat = true,
        .enable_statistics = options.enable_statistics,
        .enable_trace = options.enable_trace,
        
        // Use provided tuning or auto-detected values
        .prediction_min_cutoff = options.prediction_min_cutoff orelse build_opts.performance.one_euro_min_cutoff,
        .prediction_beta = options.prediction_beta orelse build_opts.performance.one_euro_beta,
        .prediction_d_cutoff = 1.0,
    };
    
    return try core.ThreadPool.init(allocator, config);
}

// ============================================================================
// Development and Testing Pools - Enhanced Debugging and Validation
// ============================================================================

/// Create a development pool with comprehensive debugging enabled
/// Perfect for development, debugging, and testing applications
pub fn createDevelopmentPool(allocator: std.mem.Allocator) !*core.ThreadPool {
    const config = core.Config.createDevelopmentConfig();
    
    std.log.info(
        \\
        \\🛠️  Beat.zig Development Mode Activated
        \\   Features: Comprehensive debugging and validation
        \\   Workers: {} (optimized for debugging)
        \\   Queue Size: {} (fast issue detection)
        \\   Heartbeat: {}μs (responsive monitoring)
        \\
        \\✅ Enabled Features:
        \\   • Verbose logging and task tracing
        \\   • Memory debugging and leak detection
        \\   • Performance validation
        \\   • Deadlock detection
        \\   • Resource cleanup validation
        \\
        \\💡 Use in production with: beat.createBasicPool() or beat.createPerformancePool()
        \\
    , .{ config.num_workers.?, config.task_queue_size, config.heartbeat_interval_us });
    
    return core.ThreadPool.init(allocator, config) catch |err| {
        std.log.err(
            \\
            \\❌ Failed to create development pool: {}
            \\💡 DEBUGGING SUGGESTIONS:
            \\   • Check available memory (development mode uses extra resources)
            \\   • Try basic pool for minimal resource usage
            \\   • Verify platform support for all debugging features
            \\
        , .{err});
        return err;
    };
}

/// Create a testing pool optimized for unit tests
/// Balances debugging features with test performance
pub fn createTestingPool(allocator: std.mem.Allocator) !*core.ThreadPool {
    const config = core.Config.createTestingConfig();
    
    return core.ThreadPool.init(allocator, config) catch |err| {
        std.log.err(
            \\
            \\❌ Failed to create testing pool: {}
            \\💡 TEST TROUBLESHOOTING:
            \\   • Testing pool requires minimal resources
            \\   • Check if running in memory-constrained environment
            \\   • Fall back to basic configuration for CI/CD systems
            \\
        , .{err});
        return err;
    };
}

/// Create a profiling pool optimized for performance analysis
/// Minimizes interference while enabling necessary profiling features
pub fn createProfilingPool(allocator: std.mem.Allocator) !*core.ThreadPool {
    const config = core.Config.createProfilingConfig();
    
    std.log.info(
        \\
        \\📊 Beat.zig Profiling Mode Activated
        \\   Features: Minimal interference profiling
        \\   Scheduler Profiling: Enabled
        \\   Performance Validation: Disabled (no interference)
        \\   Task Tracing: Disabled (minimal overhead)
        \\
        \\💡 Use with COZ profiler: zig build bench-coz -Dcoz=true
        \\
    , .{});
    
    return core.ThreadPool.init(allocator, config) catch |err| {
        std.log.err(
            \\
            \\❌ Failed to create profiling pool: {}
            \\💡 PROFILING SETUP:
            \\   • Ensure sufficient system resources
            \\   • Check if hardware performance counters are available
            \\   • Try performance pool for basic profiling
            \\
        , .{err});
        return err;
    };
}

/// Analyze and validate a configuration for development purposes
pub fn analyzeConfiguration(allocator: std.mem.Allocator, config: *const core.Config) ![]const u8 {
    return config.validateDevelopmentConfig(allocator);
}

/// Create a custom pool with development mode enhancements applied
pub fn createCustomDevelopmentPool(allocator: std.mem.Allocator, base_config: core.Config) !*core.ThreadPool {
    var config = base_config;
    config.development_mode = true;
    config.applyDevelopmentMode();
    
    const analysis = try config.validateDevelopmentConfig(allocator);
    defer allocator.free(analysis);
    
    if (config.verbose_logging) {
        std.log.info("{s}", .{analysis});
    }
    
    return core.ThreadPool.init(allocator, config) catch |err| {
        std.log.err(
            \\
            \\❌ Failed to create custom development pool: {}
            \\💡 CONFIGURATION ANALYSIS:
            \\{s}
            \\
        , .{ err, analysis });
        return err;
    };
}

// ============================================================================
// Specialized Pool Creators for Common Use Cases
// ============================================================================

/// Create pool optimized for HTTP server workloads (requested by Reverb team)
pub fn createHttpServerPool(allocator: std.mem.Allocator, options: HttpServerOptions) !HttpServerPool {
    const connection_workers = options.connection_workers orelse @max(build_opts.hardware.optimal_workers / 2, 2);
    const io_workers = options.io_workers orelse @max(build_opts.hardware.optimal_workers / 4, 1);
    const request_workers = options.request_workers orelse build_opts.hardware.optimal_workers;
    
    return HttpServerPool{
        .connection_pool = try createConnectionPool(allocator, connection_workers, options),
        .io_pool = try createIOPool(allocator, io_workers, options),
        .request_pool = try createRequestPool(allocator, request_workers, options),
        .allocator = allocator,
    };
}

pub const HttpServerOptions = struct {
    connection_workers: ?u32 = null,
    io_workers: ?u32 = null, 
    request_workers: ?u32 = null,
    enable_keep_alive_optimization: bool = true,
    enable_request_batching: bool = true,
    memory_pressure_callback: ?*const fn() void = null,
    feature_level: FeatureLevel = .performance, // Default to performance for HTTP
};

pub const HttpServerPool = struct {
    connection_pool: *core.ThreadPool,
    io_pool: *core.ThreadPool,
    request_pool: *core.ThreadPool,
    allocator: std.mem.Allocator,
    
    pub fn deinit(self: *HttpServerPool) void {
        self.connection_pool.deinit();
        self.io_pool.deinit();
        self.request_pool.deinit();
    }
    
    pub fn handleConnection(self: *HttpServerPool, connection_data: anytype) !void {
        const task = core.Task{
            .func = processConnection,
            .data = @ptrCast(connection_data),
            .priority = .high, // Connections are high priority
            .affinity_hint = 0, // Prefer NUMA node 0 for network I/O
        };
        try self.connection_pool.submit(task);
    }
    
    pub fn handleRequest(self: *HttpServerPool, request_data: anytype) !void {
        const task = core.Task{
            .func = processRequest,
            .data = @ptrCast(request_data),
            .priority = .normal,
            // Let advanced worker selection determine optimal placement
        };
        try self.request_pool.submit(task);
    }
    
    pub fn handleIO(self: *HttpServerPool, io_data: anytype) !void {
        const task = core.Task{
            .func = processIO,
            .data = @ptrCast(io_data),
            .priority = .normal,
            .affinity_hint = null, // Let system decide
        };
        try self.io_pool.submit(task);
    }
};

/// Create pool optimized for development and testing
pub fn createDevelopmentPool(allocator: std.mem.Allocator, options: DevelopmentOptions) !*core.ThreadPool {
    const config = core.Config{
        .num_workers = options.workers orelse 4, // Conservative for development
        .task_queue_size = 64, // Small queue for faster feedback
        .enable_lock_free = options.enable_performance_features,
        .enable_work_stealing = options.enable_performance_features,
        .enable_topology_aware = false, // Keep simple for development
        .enable_numa_aware = false,
        .enable_predictive = false,
        .enable_advanced_selection = false,
        .enable_heartbeat = options.enable_performance_features,
        .enable_statistics = true, // Always enable for development
        .enable_trace = options.enable_debug_logging,
    };
    
    return try core.ThreadPool.init(allocator, config);
}

pub const DevelopmentOptions = struct {
    workers: ?u32 = null,
    enable_debug_logging: bool = true,
    enable_performance_features: bool = false, // Conservative for development
    enable_task_tracing: bool = true,
    fallback_to_safe_defaults: bool = true,
};

// ============================================================================
// Runtime Configuration Discovery (Requested Feature)
// ============================================================================

/// Detect optimal configuration at runtime
pub fn detectOptimalConfig(allocator: std.mem.Allocator) !RuntimeConfig {
    _ = allocator; // May be used for future dynamic detection
    
    const cpu_count = std.Thread.getCpuCount() catch 8;
    const memory_estimate = estimateSystemMemory();
    const has_simd = build_opts.cpu_features.has_simd;
    
    // Determine optimal feature level based on system capabilities
    const optimal_level: FeatureLevel = blk: {
        if (cpu_count >= 16 and memory_estimate >= 16) {
            break :blk .advanced; // High-end system
        } else if (cpu_count >= 8 and memory_estimate >= 8) {
            break :blk .performance; // Mid-range system
        } else {
            break :blk .basic; // Conservative for smaller systems
        }
    };
    
    return RuntimeConfig{
        .recommended_workers = @max(@as(u32, @intCast(cpu_count)) -| 2, 2),
        .recommended_queue_size = @as(u32, @intCast(cpu_count)) * 64,
        .optimal_level = optimal_level,
        .has_simd = has_simd,
        .numa_nodes = estimateNumaNodes(@intCast(cpu_count)),
        .memory_gb = memory_estimate,
    };
}

pub const RuntimeConfig = struct {
    recommended_workers: u32,
    recommended_queue_size: u32,
    optimal_level: FeatureLevel,
    has_simd: bool,
    numa_nodes: u32,
    memory_gb: u32,
    
    pub fn createPool(self: RuntimeConfig, allocator: std.mem.Allocator) !*core.ThreadPool {
        return switch (self.optimal_level) {
            .basic => try createBasicPool(allocator, self.recommended_workers),
            .performance => try createPerformancePool(allocator, .{
                .workers = self.recommended_workers,
                .queue_size_multiplier = self.recommended_queue_size / self.recommended_workers,
            }),
            .advanced => try createAdvancedPool(allocator, .{
                .workers = self.recommended_workers,
                .queue_size_multiplier = self.recommended_queue_size / self.recommended_workers,
            }),
        };
    }
};

// ============================================================================
// Helper Functions for Specialized Pools
// ============================================================================

fn createConnectionPool(allocator: std.mem.Allocator, workers: u32, options: HttpServerOptions) !*core.ThreadPool {
    return switch (options.feature_level) {
        .basic => try createBasicPool(allocator, workers),
        .performance => try createPerformancePool(allocator, .{
            .workers = workers,
            .enable_topology_aware = true, // Good for I/O
            .queue_size_multiplier = 32, // Smaller queues for connection handling
        }),
        .advanced => try createAdvancedPool(allocator, .{
            .workers = workers,
            .enable_predictive = false, // I/O patterns are hard to predict
            .queue_size_multiplier = 32,
        }),
    };
}

fn createIOPool(allocator: std.mem.Allocator, workers: u32, options: HttpServerOptions) !*core.ThreadPool {
    return switch (options.feature_level) {
        .basic => try createBasicPool(allocator, workers),
        .performance, .advanced => try createPerformancePool(allocator, .{
            .workers = workers,
            .enable_topology_aware = false, // I/O doesn't benefit much from topology
            .queue_size_multiplier = 16, // Small queues for I/O
        }),
    };
}

fn createRequestPool(allocator: std.mem.Allocator, workers: u32, options: HttpServerOptions) !*core.ThreadPool {
    return switch (options.feature_level) {
        .basic => try createBasicPool(allocator, workers),
        .performance => try createPerformancePool(allocator, .{
            .workers = workers,
            .queue_size_multiplier = 64,
        }),
        .advanced => try createAdvancedPool(allocator, .{
            .workers = workers,
            .enable_predictive = true, // Request processing can benefit from prediction
            .queue_size_multiplier = 128,
        }),
    };
}

fn estimateSystemMemory() u32 {
    // Platform-specific memory estimation
    return switch (@import("builtin").os.tag) {
        .linux => estimateLinuxMemory(),
        .windows => estimateWindowsMemory(),
        .macos => estimateMacOSMemory(),
        else => 8, // Conservative fallback
    };
}

fn estimateNumaNodes(cpu_count: u32) u32 {
    // Rough heuristic for NUMA node estimation
    if (cpu_count <= 4) return 1;
    if (cpu_count <= 16) return 2;
    return 4;
}

fn estimateLinuxMemory() u32 {
    // Could read /proc/meminfo in the future
    return 8; // Conservative fallback
}

fn estimateWindowsMemory() u32 {
    // Could use Windows API in the future
    return 8; // Conservative fallback
}

fn estimateMacOSMemory() u32 {
    // Could use sysctl in the future  
    return 8; // Conservative fallback
}

// Placeholder functions for HTTP server processing
fn processConnection(data: *anyopaque) void {
    _ = data;
    // Connection processing logic would go here
}

fn processRequest(data: *anyopaque) void {
    _ = data;
    // Request processing logic would go here
}

fn processIO(data: *anyopaque) void {
    _ = data;
    // I/O processing logic would go here
}

// ============================================================================
// Testing
// ============================================================================

test "progressive API basic level" {
    const allocator = std.testing.allocator;
    
    // Basic pool should work without any dependencies
    const basic_pool = try createBasicPool(allocator, 2);
    defer basic_pool.deinit();
    
    // Should have disabled advanced features
    // (Would need to expose config in ThreadPool to test this properly)
}

test "runtime configuration detection" {
    const allocator = std.testing.allocator;
    
    const runtime_config = try detectOptimalConfig(allocator);
    try std.testing.expect(runtime_config.recommended_workers > 0);
    try std.testing.expect(runtime_config.recommended_queue_size > 0);
    
    // Should be able to create a pool from runtime config
    const pool = try runtime_config.createPool(allocator);
    defer pool.deinit();
}

test "HTTP server pool creation" {
    const allocator = std.testing.allocator;
    
    var http_pool = try createHttpServerPool(allocator, .{
        .feature_level = .basic, // Use basic level for testing
    });
    defer http_pool.deinit();
    
    // Pools should be created successfully
    try std.testing.expect(@intFromPtr(http_pool.connection_pool) != 0);
    try std.testing.expect(@intFromPtr(http_pool.io_pool) != 0);
    try std.testing.expect(@intFromPtr(http_pool.request_pool) != 0);
}