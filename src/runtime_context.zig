const std = @import("std");
const builtin = @import("builtin");

// Import all global resource managers
const optimization_registry = @import("optimization_registry.zig");
const optimization_orchestrator = @import("optimization_orchestrator.zig");
const unified_triple_optimization = @import("unified_triple_optimization.zig");
const numa_mapping = @import("numa_mapping.zig");
const task_execution_stats = @import("task_execution_stats.zig");
const memory_pressure = @import("memory_pressure.zig");
const ispc_prediction_integration = @import("ispc_prediction_integration.zig");
const advanced_worker_selection = @import("advanced_worker_selection.zig");
const simd_classifier = @import("simd_classifier.zig");
const opentelemetry = @import("opentelemetry.zig");
const error_reporter = @import("error_reporter.zig");

// ============================================================================
// Runtime Context - Deterministic Resource Lifecycle Management
// 
// This module provides the top-level RuntimeContext that owns and manages
// all global resources, background threads, and OS handles in Beat.zig.
//
// ISSUE ADDRESSED:
// - Background threads and OS handles not tied to allocator lifetime
// - Resource cleanup scattered across multiple modules with no coordination
// - Memory leaks when some cleanup functions are missed during shutdown
// - Race conditions during shutdown when resources depend on each other
//
// SOLUTION:
// - Single RuntimeContext owns all global resources
// - Deterministic shutdown order respecting dependencies
// - Automatic resource cleanup with allocator lifetime tracking
// - Thread-safe initialization and cleanup coordination
// ============================================================================

/// Configuration for the runtime context
pub const RuntimeContextConfig = struct {
    /// Enable memory pressure monitoring
    enable_memory_pressure_monitoring: bool = true,
    
    /// Enable ISPC acceleration
    enable_ispc_acceleration: bool = true,
    
    /// Enable optimization systems (Souper/ISPC/Minotaur)
    enable_optimization_systems: bool = true,
    
    /// Enable NUMA topology awareness
    enable_numa_awareness: bool = true,
    
    /// Enable advanced worker selection
    enable_advanced_worker_selection: bool = true,
    
    /// Enable predictive task execution stats
    enable_task_execution_stats: bool = true,
    
    /// Enable SIMD classification system
    enable_simd_classification: bool = true,
    
    /// Enable OpenTelemetry observability
    enable_opentelemetry: bool = true,
    
    /// Enable background profiling threads
    enable_background_profiling: bool = false, // Disabled by default for stability
    
    /// Memory pressure monitoring configuration
    memory_pressure_config: memory_pressure.MemoryPressureConfig = .{},
    
    /// ISPC acceleration configuration
    ispc_config: ispc_prediction_integration.PredictionAccelerator.AcceleratorConfig = .{},
    
    /// Optimization system configuration
    optimization_config: unified_triple_optimization.UnifiedTripleOptimizationConfig = .{},
    
    /// OpenTelemetry configuration
    opentelemetry_config: opentelemetry.OpenTelemetryConfig = .{},
};

/// Centralized runtime context managing all global resources
pub const RuntimeContext = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    config: RuntimeContextConfig,
    
    // State tracking
    is_initialized: std.atomic.Value(bool),
    is_shutting_down: std.atomic.Value(bool),
    init_mutex: std.Thread.Mutex = .{},
    
    // Global resource managers (owned by this context)
    numa_mapper: ?*numa_mapping.NumaMapper = null,
    task_stats_manager: ?*task_execution_stats.TaskExecutionStatsManager = null,
    memory_pressure_monitor: ?*memory_pressure.MemoryPressureMonitor = null,
    
    // Optimization system resources
    optimization_registry: ?*optimization_registry.OptimizationRegistry = null,
    optimization_orchestrator: ?*optimization_orchestrator.OptimizationOrchestrator = null,
    optimization_engine: ?*unified_triple_optimization.UnifiedTripleOptimizationEngine = null,
    
    // Observability system resources
    opentelemetry_tracer: ?*opentelemetry.Tracer = null,
    
    // Background threads (managed explicitly)
    background_threads: std.ArrayList(std.Thread),
    
    // Initialization order tracking for dependencies
    numa_initialized: bool = false,
    task_stats_initialized: bool = false,
    memory_pressure_initialized: bool = false,
    optimization_initialized: bool = false,
    ispc_initialized: bool = false,
    opentelemetry_initialized: bool = false,
    
    /// Initialize the runtime context with all global resources
    pub fn init(allocator: std.mem.Allocator, config: RuntimeContextConfig) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);
        
        self.* = Self{
            .allocator = allocator,
            .config = config,
            .is_initialized = std.atomic.Value(bool).init(false),
            .is_shutting_down = std.atomic.Value(bool).init(false),
            .background_threads = std.ArrayList(std.Thread).init(allocator),
        };
        
        // Initialize all subsystems in dependency order
        try self.initializeSubsystems();
        
        self.is_initialized.store(true, .release);
        std.log.info("RuntimeContext: All global resources initialized successfully", .{});
        
        return self;
    }
    
    /// Clean up all resources in reverse dependency order
    pub fn deinit(self: *Self) void {
        self.is_shutting_down.store(true, .release);
        
        self.init_mutex.lock();
        defer self.init_mutex.unlock();
        
        std.log.info("RuntimeContext: Beginning coordinated shutdown of all global resources", .{});
        
        // Shutdown in reverse dependency order to avoid use-after-free
        self.shutdownOpenTelemetry();
        self.shutdownSIMDClassification();
        self.shutdownAdvancedWorkerSelection();
        self.shutdownISPCResources();
        self.shutdownOptimizationSystems();
        self.shutdownMemoryPressureMonitoring();
        self.shutdownTaskExecutionStats();
        self.shutdownNUMAMapping();
        self.shutdownBackgroundThreads();
        
        // Clean up our own resources
        self.background_threads.deinit();
        
        std.log.info("RuntimeContext: All global resources shut down successfully", .{});
        
        // Finally destroy the context itself
        const allocator = self.allocator;
        allocator.destroy(self);
    }
    
    /// Check if the runtime context is fully initialized
    pub fn isInitialized(self: *const Self) bool {
        return self.is_initialized.load(.acquire);
    }
    
    /// Check if shutdown is in progress
    pub fn isShuttingDown(self: *const Self) bool {
        return self.is_shutting_down.load(.acquire);
    }
    
    /// Get the NUMA mapper (thread-safe)
    pub fn getNumaMapper(self: *Self) ?*numa_mapping.NumaMapper {
        if (!self.isInitialized() or self.isShuttingDown()) return null;
        return self.numa_mapper;
    }
    
    /// Get the task execution stats manager (thread-safe)
    pub fn getTaskStatsManager(self: *Self) ?*task_execution_stats.TaskExecutionStatsManager {
        if (!self.isInitialized() or self.isShuttingDown()) return null;
        return self.task_stats_manager;
    }
    
    /// Get the memory pressure monitor (thread-safe)
    pub fn getMemoryPressureMonitor(self: *Self) ?*memory_pressure.MemoryPressureMonitor {
        if (!self.isInitialized() or self.isShuttingDown()) return null;
        return self.memory_pressure_monitor;
    }
    
    /// Get the optimization registry (thread-safe)
    pub fn getOptimizationRegistry(self: *Self) ?*optimization_registry.OptimizationRegistry {
        if (!self.isInitialized() or self.isShuttingDown()) return null;
        return self.optimization_registry;
    }
    
    /// Get the optimization orchestrator (thread-safe)
    pub fn getOptimizationOrchestrator(self: *Self) ?*optimization_orchestrator.OptimizationOrchestrator {
        if (!self.isInitialized() or self.isShuttingDown()) return null;
        return self.optimization_orchestrator;
    }
    
    /// Get the OpenTelemetry tracer (thread-safe)
    pub fn getOpenTelemetryTracer(self: *Self) ?*opentelemetry.Tracer {
        if (!self.isInitialized() or self.isShuttingDown()) return null;
        return self.opentelemetry_tracer;
    }
    
    /// Get comprehensive runtime statistics
    pub fn getStatistics(self: *Self) RuntimeStatistics {
        var stats = RuntimeStatistics{
            .is_initialized = self.isInitialized(),
            .is_shutting_down = self.isShuttingDown(),
            .numa_initialized = self.numa_initialized,
            .task_stats_initialized = self.task_stats_initialized,
            .memory_pressure_initialized = self.memory_pressure_initialized,
            .optimization_initialized = self.optimization_initialized,
            .ispc_initialized = self.ispc_initialized,
            .opentelemetry_initialized = self.opentelemetry_initialized,
            .background_threads_count = @intCast(self.background_threads.items.len),
        };
        
        // Gather statistics from subsystems
        if (self.task_stats_manager) |manager| {
            stats.task_stats = manager.getManagerStatistics();
        }
        
        if (self.memory_pressure_monitor) |monitor| {
            stats.memory_pressure_level = monitor.getCurrentLevel();
        }
        
        if (self.optimization_registry) |registry| {
            stats.optimization_stats = registry.getStatistics();
        }
        
        return stats;
    }
    
    // Private initialization methods
    
    /// Initialize all subsystems in dependency order
    fn initializeSubsystems(self: *Self) !void {
        // 1. NUMA mapping (foundational - no dependencies)
        if (self.config.enable_numa_awareness) {
            try self.initializeNUMAMapping();
        }
        
        // 2. Task execution stats (depends on NUMA for node-aware stats)
        if (self.config.enable_task_execution_stats) {
            try self.initializeTaskExecutionStats();
        }
        
        // 3. Memory pressure monitoring (depends on NUMA for per-node metrics)
        if (self.config.enable_memory_pressure_monitoring) {
            try self.initializeMemoryPressureMonitoring();
        }
        
        // 4. Optimization systems (depends on task stats for performance tracking)
        if (self.config.enable_optimization_systems) {
            try self.initializeOptimizationSystems();
        }
        
        // 5. ISPC acceleration (can use optimization systems but fallback available)
        if (self.config.enable_ispc_acceleration) {
            try self.initializeISPCAcceleration();
        }
        
        // 6. Advanced worker selection (depends on task stats and memory monitoring)
        if (self.config.enable_advanced_worker_selection) {
            try self.initializeAdvancedWorkerSelection();
        }
        
        // 7. SIMD classification (depends on task stats and ISPC)
        if (self.config.enable_simd_classification) {
            try self.initializeSIMDClassification();
        }
        
        // 8. OpenTelemetry observability (can be initialized independently)
        if (self.config.enable_opentelemetry) {
            try self.initializeOpenTelemetry();
        }
    }
    
    fn initializeNUMAMapping(self: *Self) !void {
        std.log.debug("RuntimeContext: Initializing NUMA mapping", .{});
        
        // Initialize the global NUMA mapper
        self.numa_mapper = try numa_mapping.getGlobalNumaMapper(self.allocator);
        self.numa_initialized = true;
        
        std.log.info("RuntimeContext: NUMA mapping initialized with {} logical nodes", 
            .{self.numa_mapper.?.getNumaNodeCount()});
    }
    
    fn initializeTaskExecutionStats(self: *Self) !void {
        std.log.debug("RuntimeContext: Initializing task execution stats", .{});
        
        // Initialize the global task stats manager
        self.task_stats_manager = try task_execution_stats.getGlobalStatsManager(self.allocator);
        self.task_stats_initialized = true;
        
        std.log.info("RuntimeContext: Task execution statistics manager initialized", .{});
    }
    
    fn initializeMemoryPressureMonitoring(self: *Self) !void {
        std.log.debug("RuntimeContext: Initializing memory pressure monitoring", .{});
        
        // Create memory pressure monitor with NUMA awareness
        self.memory_pressure_monitor = try memory_pressure.MemoryPressureMonitor.init(
            self.allocator, 
            self.config.memory_pressure_config
        );
        
        // Integrate with NUMA mapper if available
        if (self.numa_mapper) |mapper| {
            self.memory_pressure_monitor.?.numa_mapper = mapper;
        }
        
        // Start monitoring (MemoryPressureMonitor handles monitoring internally)
        try self.memory_pressure_monitor.?.start();
        
        self.memory_pressure_initialized = true;
        std.log.info("RuntimeContext: Memory pressure monitoring initialized", .{});
    }
    
    fn initializeOptimizationSystems(self: *Self) !void {
        std.log.debug("RuntimeContext: Initializing optimization systems", .{});
        
        // Initialize optimization registry
        self.optimization_registry = try optimization_registry.getGlobalRegistry(self.allocator);
        
        // Initialize optimization orchestrator
        self.optimization_orchestrator = try optimization_orchestrator.getGlobalOrchestrator(
            self.allocator, 
            self.optimization_registry.?
        );
        
        // Initialize unified triple optimization engine
        self.optimization_engine = try unified_triple_optimization.getGlobalEngine(self.allocator);
        
        self.optimization_initialized = true;
        std.log.info("RuntimeContext: Optimization systems initialized", .{});
    }
    
    fn initializeISPCAcceleration(self: *Self) !void {
        std.log.debug("RuntimeContext: Initializing ISPC acceleration", .{});
        
        // Initialize global ISPC accelerator
        ispc_prediction_integration.initGlobalAccelerator(self.allocator, self.config.ispc_config);
        self.ispc_initialized = true;
        
        std.log.info("RuntimeContext: ISPC acceleration initialized", .{});
    }
    
    fn initializeAdvancedWorkerSelection(self: *Self) !void {
        std.log.debug("RuntimeContext: Initializing advanced worker selection", .{});
        
        // Initialize global worker selection systems (these manage their own globals)
        advanced_worker_selection.initializeGlobalSelection(self.allocator);
        
        std.log.info("RuntimeContext: Advanced worker selection initialized", .{});
    }
    
    fn initializeSIMDClassification(self: *Self) !void {
        std.log.debug("RuntimeContext: Initializing SIMD classification", .{});
        
        // Initialize global SIMD classification systems
        simd_classifier.initializeGlobalClassifier(self.allocator);
        
        std.log.info("RuntimeContext: SIMD classification initialized", .{});
    }
    
    fn initializeOpenTelemetry(self: *Self) !void {
        std.log.debug("RuntimeContext: Initializing OpenTelemetry observability", .{});
        
        // Initialize resource attributes for the config
        self.config.opentelemetry_config.initResourceAttributes(self.allocator);
        
        // Create OpenTelemetry tracer with configuration
        self.opentelemetry_tracer = try self.allocator.create(opentelemetry.Tracer);
        self.opentelemetry_tracer.?.* = try opentelemetry.Tracer.init(self.allocator, self.config.opentelemetry_config);
        
        // Initialize global tracer for system-wide access
        _ = try opentelemetry.initGlobalTracer(self.allocator, self.config.opentelemetry_config);
        
        // Integrate with ErrorReporter if available
        if (error_reporter.getGlobalErrorReporter()) |global_error_reporter| {
            global_error_reporter.setTelemetryHook(opentelemetry.errorReporterTelemetryHook);
            std.log.debug("RuntimeContext: OpenTelemetry integrated with ErrorReporter", .{});
        } else |_| {
            std.log.debug("RuntimeContext: ErrorReporter not available for OpenTelemetry integration", .{});
        }
        
        self.opentelemetry_initialized = true;
        std.log.info("RuntimeContext: OpenTelemetry observability initialized", .{});
    }
    
    // Private shutdown methods (in reverse dependency order)
    
    fn shutdownOpenTelemetry(self: *Self) void {
        if (!self.opentelemetry_initialized) return;
        
        std.log.debug("RuntimeContext: Shutting down OpenTelemetry observability", .{});
        
        if (self.opentelemetry_tracer) |tracer| {
            // Force export any pending telemetry data
            tracer.forceExport() catch |err| {
                std.log.warn("RuntimeContext: Failed to export pending telemetry data: {}", .{err});
            };
            
            // Clean up the tracer
            tracer.deinit();
            self.allocator.destroy(tracer);
            self.opentelemetry_tracer = null;
        }
        
        // Shutdown global tracer
        opentelemetry.deinitGlobalTracer(self.allocator);
        
        self.opentelemetry_initialized = false;
        std.log.info("RuntimeContext: OpenTelemetry observability shut down", .{});
    }
    
    fn shutdownSIMDClassification(self: *Self) void {
        if (!self.config.enable_simd_classification) return;
        
        std.log.debug("RuntimeContext: Shutting down SIMD classification", .{});
        simd_classifier.deinitializeGlobalClassifier();
    }
    
    fn shutdownAdvancedWorkerSelection(self: *Self) void {
        if (!self.config.enable_advanced_worker_selection) return;
        
        std.log.debug("RuntimeContext: Shutting down advanced worker selection", .{});
        advanced_worker_selection.deinitializeGlobalSelection();
    }
    
    fn shutdownISPCResources(self: *Self) void {
        if (!self.ispc_initialized) return;
        
        std.log.debug("RuntimeContext: Shutting down ISPC acceleration", .{});
        
        // Clean up all ISPC resources
        ispc_prediction_integration.deinitGlobalAccelerator();
        self.ispc_initialized = false;
    }
    
    fn shutdownOptimizationSystems(self: *Self) void {
        if (!self.optimization_initialized) return;
        
        std.log.debug("RuntimeContext: Shutting down optimization systems", .{});
        
        // Clean up optimization engine
        if (self.optimization_engine != null) {
            unified_triple_optimization.deinitGlobalEngine();
            self.optimization_engine = null;
        }
        
        // Clean up orchestrator
        if (self.optimization_orchestrator != null) {
            optimization_orchestrator.deinitGlobalOrchestrator();
            self.optimization_orchestrator = null;
        }
        
        // Clean up registry
        if (self.optimization_registry != null) {
            optimization_registry.deinitGlobalRegistry();
            self.optimization_registry = null;
        }
        
        self.optimization_initialized = false;
    }
    
    fn shutdownMemoryPressureMonitoring(self: *Self) void {
        if (!self.memory_pressure_initialized) return;
        
        std.log.debug("RuntimeContext: Shutting down memory pressure monitoring", .{});
        
        if (self.memory_pressure_monitor) |monitor| {
            monitor.deinit();
            self.memory_pressure_monitor = null;
        }
        
        self.memory_pressure_initialized = false;
    }
    
    fn shutdownTaskExecutionStats(self: *Self) void {
        if (!self.task_stats_initialized) return;
        
        std.log.debug("RuntimeContext: Shutting down task execution stats", .{});
        
        if (self.task_stats_manager != null) {
            task_execution_stats.deinitGlobalStatsManager();
            self.task_stats_manager = null;
        }
        
        self.task_stats_initialized = false;
    }
    
    fn shutdownNUMAMapping(self: *Self) void {
        if (!self.numa_initialized) return;
        
        std.log.debug("RuntimeContext: Shutting down NUMA mapping", .{});
        
        if (self.numa_mapper != null) {
            numa_mapping.deinitGlobalNumaMapper();
            self.numa_mapper = null;
        }
        
        self.numa_initialized = false;
    }
    
    fn shutdownBackgroundThreads(self: *Self) void {
        std.log.debug("RuntimeContext: Shutting down {} background threads", .{self.background_threads.items.len});
        
        // Join all background threads
        for (self.background_threads.items) |thread| {
            thread.join();
        }
        
        self.background_threads.clearAndFree();
    }
};

/// Statistics about the runtime context state
pub const RuntimeStatistics = struct {
    // Initialization state
    is_initialized: bool,
    is_shutting_down: bool,
    numa_initialized: bool,
    task_stats_initialized: bool,
    memory_pressure_initialized: bool,
    optimization_initialized: bool,
    ispc_initialized: bool,
    opentelemetry_initialized: bool,
    
    // Resource counts
    background_threads_count: u32,
    
    // Subsystem statistics
    task_stats: ?task_execution_stats.ManagerStatistics = null,
    memory_pressure_level: memory_pressure.MemoryPressureLevel = .none,
    optimization_stats: ?optimization_registry.RegistryStatistics = null,
    
    /// Generate a comprehensive status report
    pub fn generateReport(self: RuntimeStatistics, allocator: std.mem.Allocator) ![]u8 {
        return std.fmt.allocPrint(allocator,
            \\=== Beat.zig Runtime Context Status ===
            \\
            \\Initialization Status:
            \\  • Runtime initialized: {}
            \\  • Shutting down: {}
            \\  • Background threads: {}
            \\
            \\Subsystem Status:
            \\  • NUMA mapping: {}
            \\  • Task execution stats: {}
            \\  • Memory pressure monitoring: {}
            \\  • Optimization systems: {}
            \\  • ISPC acceleration: {}
            \\  • OpenTelemetry observability: {}
            \\
            \\Current State:
            \\  • Memory pressure level: {s}
            \\  • Task statistics: {}
            \\  • Optimization statistics: {}
            \\
        , .{
            self.is_initialized,
            self.is_shutting_down,
            self.background_threads_count,
            self.numa_initialized,
            self.task_stats_initialized,
            self.memory_pressure_initialized,
            self.optimization_initialized,
            self.ispc_initialized,
            self.opentelemetry_initialized,
            @tagName(self.memory_pressure_level),
            self.task_stats != null,
            self.optimization_stats != null,
        });
    }
};

// ============================================================================
// Global Runtime Context Management
// ============================================================================

/// Global runtime context instance
var global_runtime_context: ?*RuntimeContext = null;
var global_runtime_mutex: std.Thread.Mutex = .{};

/// Initialize the global runtime context
pub fn initGlobalRuntimeContext(allocator: std.mem.Allocator, config: RuntimeContextConfig) !*RuntimeContext {
    global_runtime_mutex.lock();
    defer global_runtime_mutex.unlock();
    
    if (global_runtime_context) |context| {
        std.log.warn("RuntimeContext: Global context already initialized, returning existing instance", .{});
        return context;
    }
    
    std.log.info("RuntimeContext: Initializing global runtime context", .{});
    global_runtime_context = try RuntimeContext.init(allocator, config);
    return global_runtime_context.?;
}

/// Get the global runtime context (must be initialized first)
pub fn getGlobalRuntimeContext() !*RuntimeContext {
    global_runtime_mutex.lock();
    defer global_runtime_mutex.unlock();
    
    if (global_runtime_context) |context| {
        return context;
    }
    
    return error.RuntimeContextNotInitialized;
}

/// Shutdown and cleanup the global runtime context
pub fn deinitGlobalRuntimeContext() void {
    global_runtime_mutex.lock();
    defer global_runtime_mutex.unlock();
    
    if (global_runtime_context) |context| {
        std.log.info("RuntimeContext: Shutting down global runtime context", .{});
        context.deinit();
        global_runtime_context = null;
    }
}

/// Check if global runtime context is initialized
pub fn isGlobalRuntimeContextInitialized() bool {
    global_runtime_mutex.lock();
    defer global_runtime_mutex.unlock();
    
    return global_runtime_context != null and global_runtime_context.?.isInitialized();
}

// ============================================================================
// Convenience Functions for Common Configurations
// ============================================================================

/// Create a development-friendly runtime configuration
pub fn createDevelopmentConfig() RuntimeContextConfig {
    return RuntimeContextConfig{
        .enable_memory_pressure_monitoring = false, // Disable for simpler debugging
        .enable_ispc_acceleration = false,           // Disable for better debugging
        .enable_optimization_systems = false,       // Disable for faster builds
        .enable_numa_awareness = true,               // Keep for realistic testing
        .enable_advanced_worker_selection = true,   // Keep for performance testing
        .enable_task_execution_stats = true,        // Keep for performance analysis
        .enable_simd_classification = false,        // Disable for simpler debugging
        .enable_opentelemetry = true,               // Enable for debugging insights
        .enable_background_profiling = false,       // Always disabled in development
    };
}

/// Create a production-optimized runtime configuration
pub fn createProductionConfig() RuntimeContextConfig {
    return RuntimeContextConfig{
        .enable_memory_pressure_monitoring = true,
        .enable_ispc_acceleration = true,
        .enable_optimization_systems = true,
        .enable_numa_awareness = true,
        .enable_advanced_worker_selection = true,
        .enable_task_execution_stats = true,
        .enable_simd_classification = true,
        .enable_opentelemetry = true,                // Enable full observability
        .enable_background_profiling = false,       // Disabled for stability
        .memory_pressure_config = .{},
        .ispc_config = .{
            .enable_ispc = true,
            .auto_detection = true,
            .performance_tracking = true,
        },
        .optimization_config = .{
            .optimization_strategy = .adaptive_performance,
            .enable_performance_feedback = true,
        },
    };
}

/// Create a testing-focused runtime configuration
pub fn createTestingConfig() RuntimeContextConfig {
    return RuntimeContextConfig{
        .enable_memory_pressure_monitoring = false,
        .enable_ispc_acceleration = false,
        .enable_optimization_systems = false,
        .enable_numa_awareness = false,
        .enable_advanced_worker_selection = false,
        .enable_task_execution_stats = false,
        .enable_simd_classification = false,
        .enable_opentelemetry = false,              // Disable for cleaner testing
        .enable_background_profiling = false,
    };
}

// ============================================================================
// Testing
// ============================================================================

test "RuntimeContext basic lifecycle" {
    const allocator = std.testing.allocator;
    
    const config = createTestingConfig();
    var context = try RuntimeContext.init(allocator, config);
    defer context.deinit();
    
    try std.testing.expect(context.isInitialized());
    try std.testing.expect(!context.isShuttingDown());
    
    const stats = context.getStatistics();
    try std.testing.expect(stats.is_initialized);
    try std.testing.expect(!stats.is_shutting_down);
}

test "RuntimeContext global management" {
    const allocator = std.testing.allocator;
    
    // Test initialization
    const config = createTestingConfig();
    const context = try initGlobalRuntimeContext(allocator, config);
    defer deinitGlobalRuntimeContext();
    
    try std.testing.expect(isGlobalRuntimeContextInitialized());
    
    // Test retrieval
    const retrieved_context = try getGlobalRuntimeContext();
    try std.testing.expect(context == retrieved_context);
    
    // Test statistics
    const stats = context.getStatistics();
    const report = try stats.generateReport(allocator);
    defer allocator.free(report);
    
    try std.testing.expect(report.len > 0);
}

test "RuntimeContext configuration factories" {
    const dev_config = createDevelopmentConfig();
    try std.testing.expect(!dev_config.enable_ispc_acceleration);
    try std.testing.expect(!dev_config.enable_optimization_systems);
    try std.testing.expect(dev_config.enable_opentelemetry);
    
    const prod_config = createProductionConfig();
    try std.testing.expect(prod_config.enable_ispc_acceleration);
    try std.testing.expect(prod_config.enable_optimization_systems);
    try std.testing.expect(prod_config.enable_opentelemetry);
    
    const test_config = createTestingConfig();
    try std.testing.expect(!test_config.enable_numa_awareness);
    try std.testing.expect(!test_config.enable_advanced_worker_selection);
    try std.testing.expect(!test_config.enable_opentelemetry);
}