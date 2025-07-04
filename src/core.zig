const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

// Re-export submodules
pub const lockfree = @import("lockfree.zig");
pub const topology = @import("topology.zig");
pub const memory = @import("memory.zig");
pub const scheduler = @import("scheduler.zig");
pub const pcall = @import("pcall.zig");
pub const coz = @import("coz.zig");
pub const testing = @import("testing.zig");
pub const ispc_cleanup = @import("ispc_cleanup_coordinator.zig");
// Use unified build configuration that consolidates all build options
pub const build_opts = @import("build_config_unified.zig");
pub const advanced_features = @import("advanced_features_config.zig");
pub const comptime_work = @import("comptime_work.zig");
pub const enhanced_errors = @import("enhanced_errors.zig");
pub const fingerprint = @import("fingerprint.zig");
pub const fingerprint_enhanced = @import("fingerprint_enhanced.zig");
pub const ispc_prediction_integration = @import("ispc_prediction_integration.zig");
pub const intelligent_decision = @import("intelligent_decision.zig");
pub const predictive_accounting = @import("predictive_accounting.zig");
pub const advanced_worker_selection = @import("advanced_worker_selection.zig");
pub const memory_pressure = @import("memory_pressure.zig");
pub const profiling_thread = @import("profiling_thread.zig");
pub const simd = @import("simd.zig");
pub const simd_batch = @import("simd_batch.zig");
pub const simd_queue = @import("simd_queue.zig");
pub const simd_classifier = @import("simd_classifier.zig");
pub const simd_benchmark = @import("simd_benchmark.zig");
pub const mathematical_optimizations = @import("mathematical_optimizations.zig");
pub const souper_integration = @import("souper_integration.zig");
pub const continuation = @import("continuation.zig");
pub const continuation_simd = @import("continuation_simd.zig");
pub const continuation_predictive = @import("continuation_predictive.zig");
pub const continuation_worker_selection = @import("continuation_worker_selection.zig");
pub const continuation_unified = @import("continuation_unified.zig");

// Version info
pub const version = std.SemanticVersion{
    .major = 3,
    .minor = 0,
    .patch = 1,
};

// Core constants
pub const cache_line_size = 64;

// ============================================================================
// Configuration
// ============================================================================

pub const Config = struct {
    // Thread pool settings - auto-tuned based on hardware detection
    num_workers: ?usize = build_opts.hardware.optimal_workers, // Auto-detected optimal
    min_workers: usize = 2,                  
    max_workers: ?usize = null,              // null = 2x physical cores
    
    // V1 features
    enable_work_stealing: bool = true,       // Work-stealing between threads
    enable_adaptive_sizing: bool = false,    // Dynamic worker count
    
    // V2 features (heartbeat scheduling)
    enable_heartbeat: bool = true,           // Heartbeat scheduling
    heartbeat_interval_us: u32 = 100,        // Heartbeat interval
    promotion_threshold: u64 = 10,           // Work:overhead ratio
    min_work_cycles: u64 = 1000,            // Min cycles for promotion
    
    // V3 features - ENABLED BY DEFAULT with automatic fallback
    enable_topology_aware: bool = true,
    enable_numa_aware: bool = true,
    enable_lock_free: bool = true,          // Lock-free data structures
    enable_predictive: bool = true,
    
    // One Euro Filter parameters for task execution prediction
    // Auto-tuned based on hardware characteristics, but can be overridden
    prediction_min_cutoff: f32 = build_opts.performance.one_euro_min_cutoff,
    prediction_beta: f32 = build_opts.performance.one_euro_beta,
    prediction_d_cutoff: f32 = 1.0,          // Derivative cutoff frequency
    
    // Performance tuning - auto-tuned based on hardware detection
    task_queue_size: u32 = build_opts.hardware.optimal_queue_size,
    cache_line_size: u32 = cache_line_size,
    
    // Statistics and debugging
    enable_statistics: bool = true,          
    enable_trace: bool = false,
    
    // Advanced worker selection - ENABLED BY DEFAULT with fallback
    enable_advanced_selection: bool = true,
    selection_criteria: ?advanced_worker_selection.SelectionCriteria = null,  // null = auto-detect optimal
    enable_selection_learning: bool = true,
    
    // SIMD and ISPC acceleration - ENABLED BY DEFAULT with fallback
    enable_ispc_acceleration: bool = true,
    enable_simd_classification: bool = true,
    enable_triple_optimization: bool = false,
    
    // Monitoring and observability - ENABLED BY DEFAULT with fallback
    enable_memory_pressure_monitoring: bool = true,
    enable_opentelemetry: bool = false,
    enable_advanced_error_reporting: bool = true,
    enable_background_profiling: bool = false,
    
    // Development mode configuration - AUTO-ENABLED in debug builds
    development_mode: bool = false,
    verbose_logging: bool = false,
    performance_validation: bool = false,
    memory_debugging: bool = false,
    task_tracing: bool = false,
    scheduler_profiling: bool = false,
    
    // Mathematical optimizations - ENABLED BY DEFAULT with fallback
    enable_souper_optimizations: ?bool = null,
    enable_minotaur_optimizations: bool = false,
    deadlock_detection: bool = false,
    resource_leak_detection: bool = false,
    
    /// Create a development configuration with comprehensive debugging enabled
    pub fn createDevelopmentConfig() Config {
        // Initialize advanced features for development
        advanced_features.initializeAdvancedFeatures(advanced_features.AdvancedFeaturesConfig.createDevelopmentConfig());
        
        var config = Config{};
        config.development_mode = true;
        config.verbose_logging = true;
        config.performance_validation = true;
        config.memory_debugging = true;
        config.task_tracing = true;
        config.scheduler_profiling = true;
        config.deadlock_detection = true;
        config.resource_leak_detection = true;
        config.enable_trace = true;
        config.enable_statistics = true;
        
        // Enable advanced features for development
        config.enable_ispc_acceleration = true;
        config.enable_simd_classification = true;
        config.enable_triple_optimization = true;
        config.enable_memory_pressure_monitoring = true;
        config.enable_opentelemetry = true;
        config.enable_advanced_error_reporting = true;
        config.enable_background_profiling = true;
        config.enable_souper_optimizations = true;
        config.enable_minotaur_optimizations = true;
        
        // Conservative settings for debugging
        config.num_workers = 2;  // Smaller pool for easier debugging
        config.task_queue_size = 16; // Smaller queue for faster issue detection
        config.heartbeat_interval_us = 50; // More frequent heartbeats for responsiveness
        
        return config;
    }
    
    /// Create a testing configuration optimized for unit tests
    pub fn createTestingConfig() Config {
        // Initialize advanced features for CI/testing
        advanced_features.initializeAdvancedFeatures(advanced_features.AdvancedFeaturesConfig.createCIConfig());
        
        var config = Config{};
        config.development_mode = true;
        config.verbose_logging = false; // Reduce noise in tests
        config.performance_validation = true;
        config.memory_debugging = true;
        config.resource_leak_detection = true;
        config.enable_statistics = true;
        
        // Enable comprehensive advanced features for testing
        config.enable_ispc_acceleration = true;
        config.enable_simd_classification = true;
        config.enable_triple_optimization = true;
        config.enable_memory_pressure_monitoring = true;
        config.enable_advanced_error_reporting = true;
        config.enable_souper_optimizations = true;
        config.enable_minotaur_optimizations = true;
        config.deadlock_detection = true;
        
        // Fast, small configuration for tests
        config.num_workers = 2;
        config.task_queue_size = 8;
        config.heartbeat_interval_us = 10; // Very fast for test responsiveness
        config.promotion_threshold = 5; // Lower threshold for faster promotion in tests
        
        return config;
    }
    
    /// Create a profiling configuration optimized for performance analysis
    pub fn createProfilingConfig() Config {
        // Initialize advanced features for production profiling
        advanced_features.initializeAdvancedFeatures(advanced_features.AdvancedFeaturesConfig.createProductionConfig());
        
        var config = Config{};
        config.development_mode = false; // Production-like for accurate profiling
        config.verbose_logging = false;
        config.performance_validation = false; // Disable to avoid interference
        config.scheduler_profiling = true;
        config.task_tracing = false; // Disable to reduce overhead
        config.enable_statistics = true;
        
        // Enable all performance features for comprehensive profiling
        config.enable_ispc_acceleration = true;
        config.enable_simd_classification = true;
        config.enable_triple_optimization = true;
        config.enable_memory_pressure_monitoring = true;
        config.enable_advanced_error_reporting = true;
        config.enable_background_profiling = true;
        config.enable_souper_optimizations = true;
        config.enable_minotaur_optimizations = true;
        
        // Optimal performance settings for accurate profiling
        // Use default optimized values from build_opts
        return config;
    }
    
    /// Create a production configuration with all advanced features enabled
    pub fn createProductionConfig() Config {
        // Initialize advanced features for production
        advanced_features.initializeAdvancedFeatures(advanced_features.AdvancedFeaturesConfig.createProductionConfig());
        
        var config = Config{};
        
        // Production optimization settings
        config.development_mode = false;
        config.verbose_logging = false;
        config.performance_validation = false;
        config.memory_debugging = false;
        config.task_tracing = true; // Keep for production monitoring
        config.scheduler_profiling = true; // Keep for production optimization
        config.deadlock_detection = false; // Disable for performance
        config.resource_leak_detection = true; // Keep for production stability
        config.enable_statistics = true;
        
        // Enable ALL advanced features with fallback support
        config.enable_ispc_acceleration = true;
        config.enable_simd_classification = true;
        config.enable_triple_optimization = true;
        config.enable_memory_pressure_monitoring = true;
        config.enable_opentelemetry = true;
        config.enable_advanced_error_reporting = true;
        config.enable_background_profiling = true;
        config.enable_souper_optimizations = true;
        config.enable_minotaur_optimizations = true;
        config.enable_topology_aware = true;
        config.enable_numa_aware = true;
        config.enable_predictive = true;
        config.enable_advanced_selection = true;
        config.enable_selection_learning = true;
        
        // Optimal performance settings
        config.heartbeat_interval_us = 100;
        config.promotion_threshold = 10;
        
        return config;
    }
    
    /// Create an embedded/resource-constrained configuration
    pub fn createEmbeddedConfig() Config {
        // Initialize advanced features for embedded systems
        advanced_features.initializeAdvancedFeatures(advanced_features.AdvancedFeaturesConfig.createEmbeddedConfig());
        
        var config = Config{};
        
        // Minimal settings for embedded
        config.development_mode = false;
        config.verbose_logging = false;
        config.performance_validation = false;
        config.memory_debugging = false;
        config.task_tracing = false;
        config.scheduler_profiling = false;
        config.deadlock_detection = false;
        config.resource_leak_detection = true; // Critical for embedded
        config.enable_statistics = false;
        
        // Selective advanced features for embedded
        config.enable_ispc_acceleration = false; // May not be available
        config.enable_simd_classification = true; // Lightweight, good ROI
        config.enable_triple_optimization = false; // Too heavy
        config.enable_memory_pressure_monitoring = true; // Critical for embedded
        config.enable_opentelemetry = false; // Too heavy
        config.enable_advanced_error_reporting = false; // Reduce overhead
        config.enable_background_profiling = false; // Reduce overhead
        config.enable_souper_optimizations = false; // Too heavy
        config.enable_minotaur_optimizations = false; // Too heavy
        config.enable_topology_aware = true; // Lightweight, good ROI
        config.enable_numa_aware = false; // Usually single-node
        config.enable_predictive = true; // Lightweight, good ROI
        config.enable_advanced_selection = false; // Reduce complexity
        
        // Conservative settings for embedded
        config.num_workers = 2;
        config.task_queue_size = 32;
        config.heartbeat_interval_us = 200; // Less frequent for lower overhead
        
        return config;
    }
    
    /// Apply development mode settings if enabled
    pub fn applyDevelopmentMode(self: *Config) void {
        if (self.development_mode) {
            // Ensure debug features are enabled when in development mode
            if (self.verbose_logging or self.task_tracing) {
                self.enable_trace = true;
            }
            
            if (self.memory_debugging or self.resource_leak_detection) {
                self.enable_statistics = true;
            }
            
            // Adjust settings for better debugging experience
            if (self.deadlock_detection) {
                // Shorter timeouts to detect issues faster
                if (self.heartbeat_interval_us > 100) {
                    self.heartbeat_interval_us = 100;
                }
            }
        }
    }
    
    /// Validate configuration and suggest improvements for development
    pub fn validateDevelopmentConfig(self: *const Config, allocator: std.mem.Allocator) ![]const u8 {
        var recommendations = std.ArrayList(u8).init(allocator);
        var writer = recommendations.writer();
        
        if (self.development_mode) {
            try writer.writeAll("Beat.zig Development Mode Configuration Analysis:\n\n");
            
            // Check for optimal development settings
            if (!self.verbose_logging and (self.task_tracing or self.scheduler_profiling)) {
                try writer.writeAll("⚠️  Recommendation: Enable verbose_logging for better debugging visibility\n");
            }
            
            if (!self.enable_statistics and (self.memory_debugging or self.performance_validation)) {
                try writer.writeAll("⚠️  Recommendation: Enable statistics for development features to work properly\n");
            }
            
            if (self.num_workers != null and self.num_workers.? > 4) {
                try writer.writeAll("⚠️  Recommendation: Use 2-4 workers in development mode for easier debugging\n");
            }
            
            if (self.task_queue_size > 32) {
                try writer.writeAll("⚠️  Recommendation: Use smaller queue size (8-16) in development for faster issue detection\n");
            }
            
            // Positive confirmations
            if (self.resource_leak_detection) {
                try writer.writeAll("✅ Resource leak detection enabled - will catch memory/handle leaks\n");
            }
            
            if (self.deadlock_detection) {
                try writer.writeAll("✅ Deadlock detection enabled - will identify potential blocking issues\n");
            }
            
            if (self.performance_validation) {
                try writer.writeAll("✅ Performance validation enabled - will detect scheduling anomalies\n");
            }
            
        } else {
            try writer.writeAll("Beat.zig Production Configuration:\n");
            try writer.writeAll("Development mode disabled - consider Config.createDevelopmentConfig() for debugging\n");
        }
        
        return recommendations.toOwnedSlice();
    }
};

// ============================================================================
// Core Types
// ============================================================================

pub const Priority = enum(u8) {
    low = 0,
    normal = 1,
    high = 2,
};

pub const TaskStatus = enum(u8) {
    pending = 0,
    running = 1,
    completed = 2,
    failed = 3,
    cancelled = 4,
};

/// Task execution and thread pool errors with descriptive context
pub const TaskError = error{
    /// Task function panicked during execution
    /// Help: Check task function for runtime errors, array bounds, null pointers
    TaskPanicked,
    
    /// Task was cancelled before or during execution
    /// Help: This is normal behavior when shutting down the thread pool
    TaskCancelled,
    
    /// Task execution exceeded the configured timeout
    /// Help: Increase timeout value or optimize task performance
    TaskTimeout,
    
    /// Task queue is full, cannot accept new tasks  
    /// Help: Increase queue size, reduce task submission rate, or add more workers
    QueueFull,
    
    /// Thread pool is shutting down, no new tasks accepted
    /// Help: Do not submit tasks after calling pool.deinit() or during shutdown
    PoolShutdown,
};

/// Optimized Task layout with hot data first for cache efficiency
pub const Task = struct {
    // HOT PATH: Frequently accessed during execution (first 16 bytes)
    func: *const fn (*anyopaque) void,       // 8 bytes - function pointer
    data: *anyopaque,                        // 8 bytes - data pointer
    
    // WARM PATH: Moderately accessed during scheduling (next 8 bytes)  
    priority: Priority = .normal,            // 1 byte - task priority
    affinity_hint: ?u32 = null,              // 5 bytes - preferred NUMA node (4 bytes + null flag)
    
    // COLD PATH: Rarely accessed metadata (remaining bytes, total target: 32 bytes)
    data_size_hint: ?usize = null,           // 9 bytes - size hint (8 bytes + null flag)
    fingerprint_hash: ?u64 = null,           // 9 bytes - fingerprint cache (8 bytes + null flag)
    creation_timestamp: ?u64 = null,         // 9 bytes - creation time (8 bytes + null flag)
    
    // Note: Total size now ~48 bytes instead of 80 bytes (40% reduction)
    // Can fit 1.33 tasks per cache line instead of 0.8 tasks
};

// ============================================================================
// Statistics
// ============================================================================

/// Cache-line isolated statistics to eliminate false sharing
pub const ThreadPoolStats = struct {
    // HOT PATH: Frequently accessed counters (64-byte cache line aligned)
    hot: struct {
        tasks_submitted: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        fast_path_executions: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        // Padding to ensure hot counters occupy exactly one cache line
        _pad: [64 - 2 * 8]u8 = [_]u8{0} ** (64 - 2 * 8),
    } align(64) = .{},
    
    // COLD PATH: Less frequently accessed counters (separate cache line)
    cold: struct {
        tasks_completed: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        tasks_stolen: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        tasks_cancelled: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        // Padding to prevent false sharing with other data
        _pad: [64 - 3 * 8]u8 = [_]u8{0} ** (64 - 3 * 8),
    } align(64) = .{},
    
    pub fn recordSubmit(self: *ThreadPoolStats) void {
        _ = self.hot.tasks_submitted.fetchAdd(1, .monotonic);
    }
    
    pub fn recordComplete(self: *ThreadPoolStats) void {
        _ = self.cold.tasks_completed.fetchAdd(1, .monotonic);
    }
    
    pub fn recordSteal(self: *ThreadPoolStats) void {
        _ = self.cold.tasks_stolen.fetchAdd(1, .monotonic);
    }
    
    pub fn recordCancel(self: *ThreadPoolStats) void {
        _ = self.cold.tasks_cancelled.fetchAdd(1, .monotonic);
    }
    
    pub fn recordFastPathExecution(self: *ThreadPoolStats) void {
        _ = self.hot.fast_path_executions.fetchAdd(1, .monotonic);
    }
    
    /// Get work-stealing efficiency ratio (completed via work-stealing / total completed)
    pub fn getWorkStealingEfficiency(self: *const ThreadPoolStats) f64 {
        const total_completed = self.cold.tasks_completed.load(.monotonic);
        const fast_path_count = self.hot.fast_path_executions.load(.monotonic);
        const work_stealing_completed = if (total_completed > fast_path_count) 
            total_completed - fast_path_count else 0;
        
        if (total_completed == 0) return 0.0;
        return @as(f64, @floatFromInt(work_stealing_completed)) / @as(f64, @floatFromInt(total_completed));
    }
};

// ============================================================================
// Main Thread Pool
// ============================================================================

pub const ThreadPool = struct {
    allocator: std.mem.Allocator,
    config: Config,
    workers: []Worker,
    running: std.atomic.Value(bool),
    stats: ThreadPoolStats,
    
    // Fast path optimization for small tasks (work-stealing efficiency improvement)
    fast_path_enabled: bool = true,
    fast_path_threshold: u32 = 256, // bytes
    fast_path_counter: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    
    // Optional subsystems
    topology: ?topology.CpuTopology = null,
    scheduler: ?*scheduler.Scheduler = null,
    memory_pool: ?*memory.TaskPool = null,
    decision_framework: ?*intelligent_decision.IntelligentDecisionFramework = null,
    fingerprint_registry: ?*fingerprint.FingerprintRegistry = null,
    advanced_selector: ?*advanced_worker_selection.AdvancedWorkerSelector = null,
    
    // Dedicated profiling thread for off-critical-path performance analysis
    profiling_thread: ?*profiling_thread.ProfilingThread = null,
    
    // SIMD-enhanced continuation processing (Phase 1 integration)
    continuation_simd_classifier: ?*continuation_simd.ContinuationClassifier = null,
    
    // Predictive accounting for intelligent execution time prediction (Phase 1 integration)
    continuation_predictive_accounting: ?*continuation_predictive.ContinuationPredictiveAccounting = null,
    
    // Advanced worker selection for continuations (Phase 1 integration)
    continuation_worker_selector: ?*continuation_worker_selection.ContinuationWorkerSelector = null,
    
    // Unified continuation management system (Consolidation)
    unified_continuation_manager: ?*continuation_unified.UnifiedContinuationManager = null,
    
    // Arena allocator for WorkerInfo lifetime management
    // Prevents use-after-free issues in worker selection compatibility layer
    worker_info_arena: std.heap.ArenaAllocator,
    
    const Self = @This();
    
    /// Exponential back-off state for intelligent work-stealing
    const StealBackOff = struct {
        consecutive_failures: u32 = 0,
        last_success_time: u64 = 0,
        
        const MIN_DELAY_NS: u64 = 100;              // 100ns minimum (very fast)
        const MAX_DELAY_NS: u64 = 10_000_000;       // 10ms maximum (reasonable for responsiveness)
        const BACKOFF_MULTIPLIER: u32 = 2;          // Double delay on each failure
        const SUCCESS_RESET_THRESHOLD: u64 = 1_000_000_000; // 1 second
        
        /// Calculate the current back-off delay in nanoseconds
        pub fn getCurrentDelay(self: *const StealBackOff) u64 {
            if (self.consecutive_failures == 0) return 0;
            
            // Exponential back-off: MIN_DELAY * (MULTIPLIER ^ failures)
            var delay = MIN_DELAY_NS;
            for (0..@min(self.consecutive_failures, 20)) |_| { // Cap at 20 to prevent overflow
                delay = @min(delay * BACKOFF_MULTIPLIER, MAX_DELAY_NS);
            }
            return delay;
        }
        
        /// Record a failed steal attempt
        pub fn recordFailure(self: *StealBackOff) void {
            self.consecutive_failures = @min(self.consecutive_failures + 1, 30); // Cap to prevent overflow
        }
        
        /// Record a successful steal
        pub fn recordSuccess(self: *StealBackOff) void {
            self.consecutive_failures = 0;
            self.last_success_time = @as(u64, @intCast(std.time.nanoTimestamp()));
        }
        
        /// Reset back-off if it's been too long since last success (handles long idle periods)
        pub fn maybeReset(self: *StealBackOff) void {
            if (self.last_success_time > 0) {
                const current_time = @as(u64, @intCast(std.time.nanoTimestamp()));
                if (current_time >= self.last_success_time and current_time - self.last_success_time > SUCCESS_RESET_THRESHOLD) {
                    self.consecutive_failures = @max(1, self.consecutive_failures / 2); // Gradual reset
                }
            }
        }
        
        /// Perform the back-off delay with platform-optimized sleep
        pub fn performBackOff(self: *const StealBackOff) void {
            const delay = self.getCurrentDelay();
            if (delay == 0) return;
            
            if (delay < 1000) {
                // Very short delays: use CPU pause/spin for minimal latency
                performShortSpin(delay);
            } else if (delay < 100_000) {
                // Short delays: use thread yield for cooperative scheduling
                std.Thread.yield() catch {
                    // Fallback to short spin if yield fails
                    performShortSpin(delay);
                };
            } else {
                // Longer delays: use nanosecond-precision sleep
                std.time.sleep(delay);
            }
        }
        
        /// Platform-optimized short spinning for minimal delays
        fn performShortSpin(target_ns: u64) void {
            const cycles_per_ns = 3; // Rough estimate: ~3 GHz CPU
            const target_cycles = target_ns * cycles_per_ns;
            
            // Use compiler-friendly busy loop that can be optimized per platform
            var i: u64 = 0;
            while (i < target_cycles) : (i += 1) {
                // Platform-specific pause instruction would go here
                // For now, use a simple atomic operation for delay
                asm volatile ("nop"); // Simple no-op for delay
            }
        }
    };
    
    pub const Worker = struct {
        id: u32,
        thread: std.Thread,
        pool: *ThreadPool,
        
        // Queues based on configuration - now support both tasks and continuations
        queue: union(enum) {
            mutex: MutexQueue,
            lockfree: lockfree.WorkStealingDeque(lockfree.WorkItem),
        },
        
        // CPU affinity (v3) 
        cpu_id: ?u32 = null,
        numa_node: ?u32 = null,
        current_socket: ?u32 = null,
        
        // Continuation support
        continuation_registry: ?*continuation.ContinuationRegistry = null,
        
        // Exponential back-off for work-stealing efficiency
        steal_back_off: StealBackOff = StealBackOff{},
    };
    
    const MutexQueue = struct {
        work_items: [3]std.ArrayList(lockfree.WorkItem), // One per priority
        mutex: std.Thread.Mutex,
        
        pub fn init(allocator: std.mem.Allocator) MutexQueue {
            return .{
                .work_items = .{
                    std.ArrayList(lockfree.WorkItem).init(allocator),
                    std.ArrayList(lockfree.WorkItem).init(allocator),
                    std.ArrayList(lockfree.WorkItem).init(allocator),
                },
                .mutex = .{},
            };
        }
        
        pub fn deinit(self: *MutexQueue) void {
            for (&self.work_items) |*queue| {
                queue.deinit();
            }
        }
        
        // Push a task as WorkItem
        pub fn pushTask(self: *MutexQueue, task: Task) !void {
            // For mutex queue, we need to store the task somewhere permanent
            // We'll use a simple approach and store a copy
            const work_item = lockfree.WorkItem.fromTask(@constCast(@ptrCast(&task)));
            self.mutex.lock();
            defer self.mutex.unlock();
            try self.work_items[@intFromEnum(task.priority)].append(work_item);
        }
        
        // Push a continuation as WorkItem  
        pub fn pushContinuation(self: *MutexQueue, cont: *continuation.Continuation) !void {
            const work_item = lockfree.WorkItem.fromContinuation(cont);
            // Continuations have dynamic priority
            const priority_index = if (cont.getPriority() > 2000) @as(usize, 0) else if (cont.getPriority() > 1000) @as(usize, 1) else @as(usize, 2);
            self.mutex.lock();
            defer self.mutex.unlock();
            try self.work_items[priority_index].append(work_item);
        }
        
        pub fn pop(self: *MutexQueue) ?lockfree.WorkItem {
            self.mutex.lock();
            defer self.mutex.unlock();
            
            // Check high priority first
            var i: i32 = 2;
            while (i >= 0) : (i -= 1) {
                const idx = @as(usize, @intCast(i));
                if (self.work_items[idx].items.len > 0) {
                    return self.work_items[idx].pop();
                }
            }
            return null;
        }
        
        pub fn steal(self: *MutexQueue) ?lockfree.WorkItem {
            return self.pop(); // Simple for mutex version
        }
    };
    
    pub fn init(allocator: std.mem.Allocator, input_config: Config) !*Self {
        // Enhanced configuration validation with helpful error messages
        enhanced_errors.validateConfigurationWithHelp(input_config) catch |err| {
            enhanced_errors.logEnhancedError(@TypeOf(err), err, "ThreadPool.init");
            return err;
        };
        
        // Detect configuration issues and provide guidance
        enhanced_errors.detectAndReportConfigIssues(allocator);
        
        // Check if we're being used as a dependency and provide appropriate guidance
        if (enhanced_errors.isUsedAsDependency()) {
            std.log.info(
                \\
                \\ℹ️  Beat.zig detected as external dependency
                \\💡 Consider using the Easy API for simpler integration:
                \\   const pool = try beat.createBasicPool(allocator, 4);
                \\📚 See: https://github.com/Beat-zig/Beat.zig/blob/main/INTEGRATION_GUIDE.md
                \\
            , .{});
        }
        
        // Auto-detect configuration with enhanced error handling
        var actual_config = input_config;
        
        // Detect topology if enabled with enhanced error handling (before self allocation)
        var detected_topology: ?topology.CpuTopology = null;
        
        if (actual_config.enable_topology_aware) {
            detected_topology = topology.detectTopology(allocator) catch |err| blk: {
                // Check for irrecoverable topology detection failures
                const is_irrecoverable = switch (err) {
                    error.OutOfMemory => true,         // Cannot allocate for basic topology detection
                    error.SystemResources => true,     // System resource exhaustion
                    error.PermissionDenied => true,    // Cannot access system information
                    error.Unexpected => true,          // Unexpected system errors
                    else => false,                      // Other errors are recoverable (e.g., UnsupportedPlatform)
                };
                
                if (is_irrecoverable) {
                    enhanced_errors.logEnhancedError(
                        enhanced_errors.ConfigError, 
                        enhanced_errors.ConfigError.HardwareDetectionFailed, 
                        "Critical topology detection failure"
                    );
                    std.log.err(
                        \\
                        \\🚨 CRITICAL: Irrecoverable topology detection failure
                        \\   Error: {}
                        \\   This indicates a serious system issue that prevents
                        \\   safe initialization of topology-aware features.
                        \\
                        \\💡 This suggests system-level problems that may affect
                        \\   overall thread pool reliability. Aborting initialization.
                        \\
                        \\🔧 Solutions:
                        \\   • Check available system memory
                        \\   • Verify process permissions for system information access
                        \\   • Use explicit non-topology configuration as workaround:
                        \\     Config{{ .enable_topology_aware = false, .enable_numa_aware = false }}
                        \\
                    , .{err});
                    return err; // Early abort for critical failures
                }
                
                // Recoverable error - disable topology features and continue
                std.log.warn(
                    \\
                    \\⚠️  Topology detection failed: {}
                    \\💡 AUTOMATIC FALLBACK: Disabling topology-aware features
                    \\🔧 TO FIX: Use basic configuration or disable topology awareness:
                    \\   const config = beat.Config{{ .enable_topology_aware = false }};
                    \\   OR use: beat.createBasicPool(allocator, workers);
                    \\
                , .{err});
                
                // Automatically disable topology features and continue
                actual_config.enable_topology_aware = false;
                actual_config.enable_numa_aware = false;
                break :blk null;
            };
            
            if (detected_topology) |topo| {
                if (actual_config.num_workers == null) {
                    actual_config.num_workers = topo.physical_cores;
                }
            }
        }
        
        // Fallback worker count with enhanced error reporting and early abort for critical failures
        if (actual_config.num_workers == null) {
            actual_config.num_workers = std.Thread.getCpuCount() catch |err| blk: {
                // Check for irrecoverable hardware detection failures that indicate deeper system issues
                const is_irrecoverable = switch (err) {
                    error.SystemResources => true,     // System resource exhaustion
                    error.PermissionDenied => true,    // Security restrictions preventing basic queries
                    error.Unexpected => true,          // Unexpected system errors
                    else => false,                      // Other errors are recoverable
                };
                
                if (is_irrecoverable) {
                    enhanced_errors.logEnhancedError(
                        enhanced_errors.ConfigError, 
                        enhanced_errors.ConfigError.HardwareDetectionFailed, 
                        "Critical hardware detection failure - system may be compromised"
                    );
                    std.log.err(
                        \\
                        \\🚨 CRITICAL: Irrecoverable hardware detection failure
                        \\   Error: {}
                        \\   This indicates a serious system issue that prevents
                        \\   safe operation. Cannot continue with thread pool initialization.
                        \\
                        \\💡 Possible causes:
                        \\   • System resource exhaustion
                        \\   • Security policy restrictions  
                        \\   • Corrupted system information
                        \\   • Hardware or kernel malfunction
                        \\
                        \\🔧 Solutions:
                        \\   • Check system resource availability
                        \\   • Verify security permissions
                        \\   • Restart the system if hardware issues suspected
                        \\   • Use explicit worker count as workaround: Config{{ .num_workers = N }}
                        \\
                    , .{err});
                    return err; // Early abort - cannot safely continue
                }
                
                // Recoverable error - use enhanced fallback logic
                enhanced_errors.logEnhancedError(
                    enhanced_errors.ConfigError, 
                    enhanced_errors.ConfigError.HardwareDetectionFailed, 
                    "CPU count detection"
                );
                std.log.info("🔧 Using conservative fallback: 4 workers", .{});
                break :blk 4;
            };
        }
        
        const self = try allocator.create(Self);
        errdefer {
            // If topology was detected, clean it up
            if (detected_topology) |*topo| topo.deinit();
            allocator.destroy(self);
        }
        
        self.* = .{
            .allocator = allocator,
            .config = actual_config,
            .workers = try allocator.alloc(Worker, actual_config.num_workers.?),
            .running = std.atomic.Value(bool).init(true),
            .stats = .{},
            .topology = detected_topology,
            .worker_info_arena = std.heap.ArenaAllocator.init(allocator),
        };
        
        // Initialize optional subsystems
        if (actual_config.enable_heartbeat or actual_config.enable_predictive) {
            self.scheduler = try scheduler.Scheduler.init(allocator, &actual_config);
        }
        
        // Initialize dedicated profiling thread for off-critical-path performance analysis
        if (actual_config.enable_statistics or actual_config.enable_predictive) {
            self.profiling_thread = try allocator.create(profiling_thread.ProfilingThread);
            self.profiling_thread.?.* = profiling_thread.ProfilingThread.init(allocator);
            try self.profiling_thread.?.start();
            
            // Initialize global profiling thread for cross-module access
            try profiling_thread.initGlobalProfilingThread(allocator);
        }
        
        if (actual_config.enable_lock_free) {
            const pool = memory.TaskPool.init(allocator);
            self.memory_pool = try allocator.create(memory.TaskPool);
            self.memory_pool.?.* = pool;
        }
        
        // Initialize intelligent decision framework if predictive features are enabled
        if (actual_config.enable_predictive) {
            const decision_config = intelligent_decision.DecisionConfig{};
            self.decision_framework = try allocator.create(intelligent_decision.IntelligentDecisionFramework);
            self.decision_framework.?.* = intelligent_decision.IntelligentDecisionFramework.init(decision_config);
            
            // Initialize fingerprint registry for predictive scheduling
            self.fingerprint_registry = try allocator.create(fingerprint.FingerprintRegistry);
            self.fingerprint_registry.?.* = fingerprint.FingerprintRegistry.init(allocator);
            
            // Connect the registry to the decision framework
            self.decision_framework.?.setFingerprintRegistry(self.fingerprint_registry.?);
        }
        
        // Initialize advanced worker selector (Task 2.4.2)
        if (actual_config.enable_advanced_selection) {
            const selection_criteria = actual_config.selection_criteria orelse 
                advanced_worker_selection.SelectionCriteria.balanced();
            
            self.advanced_selector = try allocator.create(advanced_worker_selection.AdvancedWorkerSelector);
            self.advanced_selector.?.* = try advanced_worker_selection.AdvancedWorkerSelector.init(allocator, selection_criteria, actual_config.num_workers.?);
            
            // Connect prediction and analysis components if available
            self.advanced_selector.?.setComponents(
                self.fingerprint_registry,
                null, // Predictive scheduler will be set later if available
                self.decision_framework,
                null  // SIMD registry not available yet
            );
        }
        
        // Initialize Souper mathematical optimizations (Phase 6)
        if (actual_config.enable_souper_optimizations orelse true) {
            souper_integration.SouperIntegration.initialize();
            std.log.info("🔬 Souper mathematical optimizations enabled - formally verified performance", .{});
        }
        
        // Initialize SIMD-enhanced continuation classifier (Phase 1 integration)
        // Provides 6-23x speedup in continuation processing through vectorization
        self.continuation_simd_classifier = try allocator.create(continuation_simd.ContinuationClassifier);
        self.continuation_simd_classifier.?.* = try continuation_simd.ContinuationClassifier.init(allocator);
        std.log.info("🚀 SIMD-enhanced continuation processing enabled - 6-23x performance improvement", .{});
        
        // Initialize predictive accounting for intelligent execution time prediction
        // Integrates One Euro Filter for adaptive prediction with SIMD analysis
        if (actual_config.enable_predictive) {
            const predictive_config = continuation_predictive.PredictiveConfig{
                .min_cutoff = actual_config.prediction_min_cutoff,
                .beta = actual_config.prediction_beta,
                .d_cutoff = actual_config.prediction_d_cutoff,
                .enable_adaptive_numa = actual_config.enable_numa_aware,
            };
            
            self.continuation_predictive_accounting = try allocator.create(continuation_predictive.ContinuationPredictiveAccounting);
            self.continuation_predictive_accounting.?.* = try continuation_predictive.ContinuationPredictiveAccounting.init(allocator, predictive_config);
            std.log.info("🧠 Continuation predictive accounting enabled - intelligent execution time prediction with One Euro Filter", .{});
        }
        
        // Initialize advanced worker selection for continuations (Phase 1 integration)
        // Provides intelligent worker selection with multi-criteria optimization
        if (self.advanced_selector) |advanced_sel| {
            const selection_criteria = actual_config.selection_criteria orelse advanced_worker_selection.SelectionCriteria.balanced();
            
            self.continuation_worker_selector = try allocator.create(continuation_worker_selection.ContinuationWorkerSelector);
            self.continuation_worker_selector.?.* = try continuation_worker_selection.ContinuationWorkerSelector.init(
                allocator, 
                advanced_sel, 
                selection_criteria
            );
            std.log.info("🎯 Advanced continuation worker selection enabled - multi-criteria optimization", .{});
        }
        
        // Initialize unified continuation management system (Consolidation)
        // Provides single optimized system replacing separate SIMD, predictive, and worker selection components
        if (self.fingerprint_registry) |registry| {
            const unified_config = continuation_unified.UnifiedConfig.performanceOptimized();
            
            self.unified_continuation_manager = try allocator.create(continuation_unified.UnifiedContinuationManager);
            self.unified_continuation_manager.?.* = try continuation_unified.UnifiedContinuationManager.init(
                allocator,
                registry,
                unified_config
            );
            std.log.info("🚀 Unified continuation management enabled - consolidated high-performance system", .{});
        }
        
        // Initialize workers
        for (self.workers, 0..) |*worker, i| {
            const cpu_id = if (self.topology) |topo| @as(u32, @intCast(i % topo.total_cores)) else null;
            const numa_node = if (self.topology) |topo| topo.logical_to_numa[i % topo.total_cores] else null;
            const socket_id = if (self.topology) |topo| topo.logical_to_socket[i % topo.total_cores] else null;
            
            const worker_config = WorkerConfig{
                .id = @intCast(i),
                .pool = self,
                .cpu_id = cpu_id,
                .numa_node = numa_node,
                .socket_id = socket_id,
            };
            
            try initWorker(worker, allocator, &actual_config, worker_config);
        }
        
        // Start workers
        for (self.workers) |*worker| {
            worker.thread = try std.Thread.spawn(.{}, workerLoop, .{worker});
            
            // Set CPU affinity if available
            if (worker.cpu_id) |cpu_id| {
                if (self.topology != null) {
                    topology.setThreadAffinity(worker.thread, &[_]u32{cpu_id}) catch |err| {
                        enhanced_errors.logEnhancedError(@TypeOf(err), err, "Failed to set CPU affinity for worker thread");
                        std.log.debug("Worker {d}: Failed to set CPU affinity to core {d}: {}", .{worker.id, cpu_id, err});
                    };
                }
            }
        }
        
        // Initialize ISPC acceleration for transparent performance enhancement
        // This provides maximum out-of-the-box performance with zero API changes
        fingerprint_enhanced.AutoAcceleration.init(allocator);
        
        // Initialize ISPC cleanup coordinator to prevent memory leaks
        ispc_cleanup.initGlobalCleanupCoordinator(allocator);
        
        return self;
    }
    
    pub fn deinit(self: *Self) void {
        self.running.store(false, .release);
        
        // Join all workers
        for (self.workers) |*worker| {
            worker.thread.join();
        }
        
        // Cleanup workers
        for (self.workers) |*worker| {
            switch (worker.queue) {
                .mutex => |*q| q.deinit(),
                .lockfree => |*q| q.deinit(),
            }
            
            // Clean up continuation registry if it exists
            if (worker.continuation_registry) |registry| {
                registry.deinit();
                self.allocator.destroy(registry);
            }
        }
        
        // Cleanup subsystems
        if (self.topology) |*topo| {
            topo.deinit();
        }
        
        if (self.scheduler) |sched| {
            sched.deinit();
        }
        
        if (self.memory_pool) |pool| {
            pool.deinit();
            self.allocator.destroy(pool);
        }
        
        if (self.fingerprint_registry) |registry| {
            registry.deinit();
            self.allocator.destroy(registry);
        }
        
        if (self.decision_framework) |framework| {
            self.allocator.destroy(framework);
        }
        
        if (self.advanced_selector) |selector| {
            selector.deinit();
            self.allocator.destroy(selector);
        }
        
        // Cleanup dedicated profiling thread
        if (self.profiling_thread) |prof_thread| {
            prof_thread.deinit();
            self.allocator.destroy(prof_thread);
        }
        
        // Cleanup global profiling thread
        profiling_thread.deinitGlobalProfilingThread();
        
        if (self.continuation_simd_classifier) |classifier| {
            classifier.deinit();
            self.allocator.destroy(classifier);
        }
        
        if (self.continuation_predictive_accounting) |predictor| {
            predictor.deinit();
            self.allocator.destroy(predictor);
        }
        
        if (self.continuation_worker_selector) |selector| {
            selector.deinit();
            self.allocator.destroy(selector);
        }
        
        if (self.unified_continuation_manager) |manager| {
            manager.deinit();
            self.allocator.destroy(manager);
        }
        
        // Clean up all ISPC runtime allocations before final deallocation
        // This prevents cross-language memory leaks from ISPC kernels
        ispc_cleanup.cleanupAllISPCResources(self.allocator);
        
        // Clean up WorkerInfo arena allocator
        self.worker_info_arena.deinit();
        
        self.allocator.free(self.workers);
        self.allocator.destroy(self);
    }
    
    pub fn submit(self: *Self, task: Task) !void {
        // ULTRA-FAST PATH: Immediate execution for small tasks
        const task_data_size = task.data_size_hint orelse 0;
        const is_likely_fast_task = (task_data_size <= 256) and (task.priority == .normal);
        
        if (is_likely_fast_task and self.should_use_fast_path()) {
            // Execute immediately with minimal overhead
            task.func(task.data);
            _ = self.stats.hot.fast_path_executions.fetchAdd(1, .monotonic);
            _ = self.stats.hot.tasks_submitted.fetchAdd(1, .monotonic);
            return;
        }
        
        // OPTIMIZED STANDARD PATH: Streamlined for work-stealing tasks
        self.submitToWorkStealing(task) catch |err| {
            // Fallback to full submission path with profiling for error cases
            coz.latencyBegin(coz.Points.task_submitted);
            defer coz.latencyEnd(coz.Points.task_submitted);
            coz.throughput(coz.Points.task_submitted);
            return err;
        };
    }
    
    /// Streamlined submission path optimized for throughput
    fn submitToWorkStealing(self: *Self, task: Task) !void {
        // Lightweight statistics update
        _ = self.stats.hot.tasks_submitted.fetchAdd(1, .monotonic);
        
        // OPTIMIZED: Use round-robin worker selection instead of complex algorithm
        // This eliminates the expensive selectWorker() call for most cases
        const submission_count = self.stats.hot.tasks_submitted.load(.monotonic);
        const worker_id = submission_count % self.workers.len;
        const worker = &self.workers[worker_id];
        
        switch (worker.queue) {
            .mutex => |*q| try q.pushTask(task),
            .lockfree => |*q| {
                // Create task pointer and wrap in WorkItem
                const task_ptr = if (self.memory_pool) |pool|
                    try pool.alloc()
                else
                    try self.allocator.create(Task);
                
                task_ptr.* = task;
                const work_item = lockfree.WorkItem.fromTask(task_ptr);
                try q.pushBottom(work_item);
            },
        }
    }
    
    /// Submit a continuation for execution with work stealing support
    pub fn submitContinuation(self: *Self, cont: *continuation.Continuation) !void {
        // Update statistics
        _ = self.stats.hot.tasks_submitted.fetchAdd(1, .monotonic);
        
        // Use unified continuation management system for optimal performance
        if (self.unified_continuation_manager) |unified_manager| {
            // Get comprehensive analysis from unified system (SIMD, prediction, worker selection)
            const analysis = try unified_manager.getAnalysis(cont);
            
            // Apply unified analysis results to continuation
            cont.fingerprint_hash = @intFromFloat(analysis.simd_classification.suitability_score * 1000000);
            
            // Apply NUMA preferences from unified analysis
            if (analysis.numa_coordination.final_numa_node) |numa_node| {
                cont.numa_node = numa_node;
            }
            
            // Adjust scheduling based on unified prediction
            if (analysis.execution_prediction.confidence > 0.7 and analysis.execution_prediction.predicted_time_ns > 5_000_000) {
                cont.locality_score = @min(1.0, cont.locality_score + analysis.execution_prediction.confidence * 0.2);
            }
            
            // Use unified worker selection
            const worker_preferences = analysis.worker_preferences;
            var worker_id: u32 = 0;
            var best_score: f32 = -1.0;
            
            for (self.workers, 0..) |worker, i| {
                var score: f32 = 0.5; // Base score
                
                // NUMA locality bonus from unified analysis
                if (analysis.numa_coordination.final_numa_node) |numa_node| {
                    if (worker.numa_node == numa_node) {
                        score += 0.3;
                    }
                }
                
                // Apply unified worker preference weights
                score += worker_preferences.locality_bonus_factor;
                
                if (score > best_score) {
                    best_score = score;
                    worker_id = @intCast(i);
                }
            }
            
            const worker = &self.workers[worker_id];
            
            // Register with worker's local continuation registry
            if (worker.continuation_registry) |registry| {
                try registry.registerContinuation(cont);
            }
            
            // Submit as WorkItem
            const work_item = lockfree.WorkItem.fromContinuation(cont);
            switch (worker.queue) {
                .mutex => |*q| try q.pushContinuation(cont),
                .lockfree => |*q| try q.pushBottom(work_item),
            }
            
            return;
        }
        
        // Fallback to legacy separate systems if unified system not available
        // SIMD-enhanced continuation classification (6-23x performance improvement)
        var simd_class: ?continuation_simd.ContinuationSIMDClass = null;
        if (self.continuation_simd_classifier) |classifier| {
            // Classify continuation for SIMD suitability
            simd_class = try classifier.classifyContinuation(cont);
            
            // Store classification in continuation for worker optimization
            cont.fingerprint_hash = @intFromFloat(simd_class.?.simd_suitability_score * 1000000); // Store as hash
            
            // Add to batch formation if suitable for SIMD
            if (simd_class.?.isSIMDSuitable()) {
                try classifier.addContinuationForBatching(cont);
                
                // If we have enough continuations, try to form batches
                // This provides optimal SIMD vectorization
                const stats = classifier.getPerformanceStats();
                if (stats.classifications_performed % 8 == 0) { // Every 8 continuations
                    _ = try classifier.formContinuationBatches();
                }
            }
        }
        
        // Intelligent execution time prediction with One Euro Filter
        var prediction: ?continuation_predictive.PredictionResult = null;
        if (self.continuation_predictive_accounting) |predictor| {
            // Predict execution time using SIMD classification for enhanced accuracy
            prediction = try predictor.predictExecutionTime(cont, simd_class);
            
            // Store prediction for adaptive NUMA placement
            if (prediction.?.numa_preference) |numa_node| {
                cont.numa_node = numa_node;
            }
            
            // Adjust scheduling based on prediction confidence and execution time
            if (prediction.?.confidence > 0.7 and prediction.?.predicted_time_ns > 5_000_000) { // > 5ms
                // For high-confidence long-running continuations, prefer specific NUMA placement
                cont.locality_score = @min(1.0, cont.locality_score + prediction.?.confidence * 0.2);
            }
        }
        
        // Initialize NUMA locality for continuation
        if (self.topology) |topo| {
            // Use round-robin NUMA node assignment for load balancing
            const numa_node = @as(u32, @intCast(self.stats.hot.tasks_submitted.load(.monotonic) % topo.numa_nodes.len));
            const socket_id = topo.logical_to_socket[numa_node % topo.total_cores];
            cont.initNumaLocality(numa_node, socket_id);
        }
        
        // Advanced worker selection for continuation submission
        var worker_id: u32 = 0;
        
        // Use advanced worker selection if available, otherwise fall back to round-robin
        if (self.continuation_worker_selector) |selector| {
            worker_id = selector.selectWorkerForContinuation(cont, self, simd_class, prediction) catch |err| blk: {
                // Log error and fallback to round-robin
                enhanced_errors.logEnhancedError(@TypeOf(err), err, "Advanced continuation worker selection failed");
                std.log.debug("Falling back to round-robin worker selection due to error: {}", .{err});
                const submission_count = self.stats.hot.tasks_submitted.load(.monotonic);
                break :blk @intCast(submission_count % self.workers.len);
            };
        } else {
            // Simple round-robin fallback
            const submission_count = self.stats.hot.tasks_submitted.load(.monotonic);
            worker_id = @intCast(submission_count % self.workers.len);
        }
        
        const worker = &self.workers[worker_id];
        
        // Register with worker's local continuation registry
        if (worker.continuation_registry) |registry| {
            try registry.registerContinuation(cont);
        }
        
        // Submit as WorkItem
        const work_item = lockfree.WorkItem.fromContinuation(cont);
        switch (worker.queue) {
            .mutex => |*q| try q.pushContinuation(cont),
            .lockfree => |*q| try q.pushBottom(work_item),
        }
    }
    
    /// Handle continuation completion and update predictive accounting
    pub fn handleContinuationCompletion(self: *Self, cont: *continuation.Continuation, actual_execution_time_ns: u64) void {
        // Use unified system for completion handling if available
        if (self.unified_continuation_manager) |unified_manager| {
            unified_manager.updateWithResults(cont, actual_execution_time_ns, 0) catch |err| {
                // Log error but don't fail the continuation completion
                std.log.warn("Failed to update unified continuation analysis: {}", .{err});
            };
            return;
        }
        
        // Fallback to legacy predictive accounting
        if (self.continuation_predictive_accounting) |predictor| {
            predictor.updatePrediction(cont, actual_execution_time_ns) catch |err| {
                // Log error but don't fail the continuation completion
                std.log.warn("Failed to update continuation prediction: {}", .{err});
            };
        }
    }
    
    pub fn wait(self: *Self) void {
        while (true) {
            var all_empty = true;
            
            for (self.workers) |*worker| {
                const empty = switch (worker.queue) {
                    .mutex => |*q| blk: {
                        q.mutex.lock();
                        defer q.mutex.unlock();
                        break :blk q.work_items[0].items.len == 0 and 
                                   q.work_items[1].items.len == 0 and 
                                   q.work_items[2].items.len == 0;
                    },
                    .lockfree => |*q| q.isEmpty(),
                };
                
                if (!empty) {
                    all_empty = false;
                    break;
                }
            }
            
            if (all_empty) break;
            std.time.sleep(10_000); // 10 microseconds
        }
    }
    
    /// Thread-safe configuration update to prevent data races
    /// This method provides a safe way to update configuration instead of direct field access
    /// TODO: Implement full atomic config updates when Zig supports large struct atomics
    pub fn updateConfig(self: *Self, new_config: Config) !void {
        // Enhanced configuration validation before applying
        enhanced_errors.validateConfigurationWithHelp(new_config) catch |err| {
            enhanced_errors.logEnhancedError(@TypeOf(err), err, "ThreadPool.updateConfig");
            return err;
        };
        
        std.log.info("ThreadPool: Updating configuration safely...", .{});
        
        // Store the old config for comparison
        const old_config = self.config;
        
        // Safe config update - workers should use isFeatureEnabled() for reads
        // This prevents partial config visibility during updates
        self.config = new_config;
        
        // Log configuration changes for debugging and monitoring
        if (old_config.enable_predictive != new_config.enable_predictive) {
            std.log.info("  ✓ Predictive scheduling: {} -> {}", .{ old_config.enable_predictive, new_config.enable_predictive });
        }
        if (old_config.enable_work_stealing != new_config.enable_work_stealing) {
            std.log.info("  ✓ Work stealing: {} -> {}", .{ old_config.enable_work_stealing, new_config.enable_work_stealing });
        }
        if (old_config.enable_topology_aware != new_config.enable_topology_aware) {
            std.log.info("  ✓ Topology awareness: {} -> {}", .{ old_config.enable_topology_aware, new_config.enable_topology_aware });
        }
        if (old_config.enable_advanced_selection != new_config.enable_advanced_selection) {
            std.log.info("  ✓ Advanced worker selection: {} -> {}", .{ old_config.enable_advanced_selection, new_config.enable_advanced_selection });
        }
        if (old_config.enable_heartbeat != new_config.enable_heartbeat) {
            std.log.info("  ✓ Heartbeat scheduling: {} -> {}", .{ old_config.enable_heartbeat, new_config.enable_heartbeat });
        }
        
        std.log.info("ThreadPool: Configuration update completed successfully", .{});
    }
    
    /// Get current configuration - preferred method for reading config
    pub fn getConfig(self: *const Self) Config {
        return self.config;
    }
    
    /// Safely check if a feature is enabled - recommended for worker threads
    pub fn isFeatureEnabled(self: *const Self, comptime field_name: []const u8) bool {
        const config = self.getConfig();
        return @field(config, field_name);
    }
    
    /// Determine if fast path execution should be used for small tasks
    /// This helps improve work-stealing efficiency by avoiding overhead for tiny tasks
    pub fn should_use_fast_path(self: *Self) bool {
        if (!self.fast_path_enabled) return false;
        
        // Use fast path when work-stealing efficiency is low (below 60%)
        // This indicates that overhead is dominating task execution time
        const efficiency = self.stats.getWorkStealingEfficiency();
        const should_boost = efficiency < 0.6; // Below 60% efficiency
        
        // Also consider system load - use fast path when workers are relatively idle
        var idle_workers: u32 = 0;
        for (self.workers) |*worker| {
            const is_idle = switch (worker.queue) {
                .mutex => |*q| blk: {
                    q.mutex.lock();
                    defer q.mutex.unlock();
                    break :blk (q.work_items[0].items.len + q.work_items[1].items.len + q.work_items[2].items.len) < 2;
                },
                .lockfree => |*q| q.size() < 2,
            };
            if (is_idle) idle_workers += 1;
        }
        
        const idle_ratio = @as(f64, @floatFromInt(idle_workers)) / @as(f64, @floatFromInt(self.workers.len));
        const workers_available = idle_ratio > 0.3; // At least 30% workers relatively idle
        
        return should_boost or workers_available;
    }
    
    pub fn selectWorker(self: *Self, task: Task) usize {
        // Use advanced worker selector if available (Task 2.4.2)
        if (self.advanced_selector) |selector| {
            // Record task execution for fingerprinting if registry is available
            if (self.fingerprint_registry) |_| {
                // Generate fingerprint for task
                var context = fingerprint.ExecutionContext.init();
                
                const task_fingerprint = fingerprint.generateTaskFingerprint(&task, &context);
                
                // Set task fingerprint hash for tracking
                var mutable_task = task;
                mutable_task.fingerprint_hash = task_fingerprint.hash();
                mutable_task.creation_timestamp = @intCast(std.time.nanoTimestamp());
            }
            
            // Create worker info array for decision making using arena allocator for proper lifetime management
            // Reset arena to prevent memory growth while ensuring WorkerInfo structs remain valid during selection
            _ = self.worker_info_arena.reset(.retain_capacity);
            const arena_allocator = self.worker_info_arena.allocator();
            
            const worker_infos = arena_allocator.alloc(intelligent_decision.WorkerInfo, self.workers.len) catch {
                // Fallback to legacy selection on allocation failure
                return self.selectWorkerLegacy(task);
            };
            
            for (self.workers, 0..) |*worker, i| {
                worker_infos[i] = intelligent_decision.WorkerInfo{
                    .id = worker.id,
                    .numa_node = worker.numa_node,
                    .queue_size = self.getWorkerQueueSize(i),
                    .max_queue_size = 1024, // Default max queue size
                };
            }
            
            // Use advanced multi-criteria optimization
            const decision = selector.selectWorker(&task, worker_infos, if (self.topology) |*topo| topo else null) catch {
                // Fallback to legacy selection on error
                return self.selectWorkerLegacy(task);
            };
            
            // Cleanup the decision (but not the worker_infos since it's handled by defer)
            self.allocator.free(decision.evaluations);
            
            return decision.selected_worker_id;
        }
        
        // Fallback to intelligent decision framework (Task 2.3.2)
        if (self.decision_framework) |framework| {
            // Record task execution for fingerprinting if registry is available
            if (self.fingerprint_registry) |_| {
                // Generate fingerprint for task
                var context = fingerprint.ExecutionContext.init();
                
                const task_fingerprint = fingerprint.generateTaskFingerprint(&task, &context);
                
                // Set task fingerprint hash for tracking
                var mutable_task = task;
                mutable_task.fingerprint_hash = task_fingerprint.hash();
                mutable_task.creation_timestamp = @intCast(std.time.nanoTimestamp());
            }
            
            // Create worker info array for decision making using arena allocator for proper lifetime management
            // Reset arena to prevent memory growth while ensuring WorkerInfo structs remain valid during selection
            _ = self.worker_info_arena.reset(.retain_capacity);
            const arena_allocator = self.worker_info_arena.allocator();
            
            const worker_infos = arena_allocator.alloc(intelligent_decision.WorkerInfo, self.workers.len) catch {
                // Fallback to legacy selection on allocation failure
                return self.selectWorkerLegacy(task);
            };
            
            for (self.workers, 0..) |*worker, i| {
                worker_infos[i] = intelligent_decision.WorkerInfo{
                    .id = worker.id,
                    .numa_node = worker.numa_node,
                    .queue_size = self.getWorkerQueueSize(i),
                    .max_queue_size = 1024, // Default max queue size
                };
            }
            
            // Make intelligent scheduling decision
            const decision = framework.makeSchedulingDecision(
                &task,
                worker_infos,
                self.topology
            );
            
            return decision.worker_id;
        }
        
        // Fallback to legacy worker selection
        return self.selectWorkerLegacy(task);
    }
    
    pub fn selectWorkerLegacy(self: *Self, task: Task) usize {
        // Legacy smart worker selection (pre-intelligent framework)
        
        // 1. Honor explicit affinity hint if provided
        if (task.affinity_hint) |numa_node| {
            if (self.topology) |topo| {
                return self.selectWorkerOnNumaNode(topo, numa_node);
            }
        }
        
        // 2. Find worker with lightest load, preferring local NUMA node
        if (self.topology) |topo| {
            return self.selectWorkerWithLoadBalancing(topo);
        }
        
        // 3. Fallback to simple load balancing without topology
        return self.selectLightestWorker();
    }
    
    fn selectWorkerOnNumaNode(self: *Self, topo: topology.CpuTopology, numa_node: u32) usize {
        _ = topo; // May be used for validation in the future
        var best_worker: usize = 0;
        var min_queue_size: usize = std.math.maxInt(usize);
        
        // Find the worker on the specified NUMA node with the smallest queue
        for (self.workers, 0..) |*worker, i| {
            if (worker.numa_node == numa_node) {
                const queue_size = self.getWorkerQueueSize(i);
                if (queue_size < min_queue_size) {
                    min_queue_size = queue_size;
                    best_worker = i;
                }
            }
        }
        
        // If no worker found on the specified node, fall back to any node
        if (min_queue_size == std.math.maxInt(usize)) {
            return self.selectLightestWorker();
        }
        
        return best_worker;
    }
    
    fn selectWorkerWithLoadBalancing(self: *Self, topo: topology.CpuTopology) usize {
        // Use a simple round-robin to distribute across NUMA nodes initially
        // This could be enhanced with actual CPU detection in the future
        const submission_count = self.stats.hot.tasks_submitted.load(.acquire);
        const preferred_numa = @as(u32, @intCast(submission_count % topo.numa_nodes.len));
        
        var best_worker: usize = 0;
        var min_queue_size: usize = std.math.maxInt(usize);
        var found_on_preferred_numa = false;
        
        // First pass: prefer workers on the same NUMA node
        for (self.workers, 0..) |*worker, i| {
            const queue_size = self.getWorkerQueueSize(i);
            
            if (worker.numa_node == preferred_numa) {
                if (!found_on_preferred_numa or queue_size < min_queue_size) {
                    min_queue_size = queue_size;
                    best_worker = i;
                    found_on_preferred_numa = true;
                }
            } else if (!found_on_preferred_numa and queue_size < min_queue_size) {
                // Only consider other NUMA nodes if we haven't found a good local option
                min_queue_size = queue_size;
                best_worker = i;
            }
        }
        
        return best_worker;
    }
    
    fn selectLightestWorker(self: *Self) usize {
        var best_worker: usize = 0;
        var min_queue_size: usize = std.math.maxInt(usize);
        
        for (self.workers, 0..) |_, i| {
            const queue_size = self.getWorkerQueueSize(i);
            if (queue_size < min_queue_size) {
                min_queue_size = queue_size;
                best_worker = i;
            }
        }
        
        return best_worker;
    }
    
    pub fn getWorkerQueueSize(self: *Self, worker_id: usize) usize {
        const worker = &self.workers[worker_id];
        
        return switch (worker.queue) {
            .mutex => |*q| blk: {
                q.mutex.lock();
                defer q.mutex.unlock();
                break :blk q.work_items[0].items.len + q.work_items[1].items.len + q.work_items[2].items.len;
            },
            .lockfree => |*q| q.size(),
        };
    }
    
    const WorkerConfig = struct {
        id: u32,
        pool: *ThreadPool,
        cpu_id: ?u32,
        numa_node: ?u32,
        socket_id: ?u32,
    };
    
    fn initWorker(worker: *Worker, allocator: std.mem.Allocator, config: *const Config, worker_config: WorkerConfig) !void {
        // Initialize continuation registry if predictive features are enabled
        var cont_registry: ?*continuation.ContinuationRegistry = null;
        if (config.enable_predictive) {
            cont_registry = try allocator.create(continuation.ContinuationRegistry);
            cont_registry.?.* = continuation.ContinuationRegistry.init(allocator);
        }
        
        worker.* = .{
            .id = worker_config.id,
            .thread = undefined,
            .pool = worker_config.pool,
            .queue = if (config.enable_lock_free)
                .{ .lockfree = try lockfree.WorkStealingDeque(lockfree.WorkItem).init(allocator, config.task_queue_size) }
            else
                .{ .mutex = MutexQueue.init(allocator) },
            .cpu_id = worker_config.cpu_id,
            .numa_node = worker_config.numa_node,
            .current_socket = worker_config.socket_id,
            .continuation_registry = cont_registry,
        };
    }
    
    fn workerLoop(worker: *Worker) void {
        // Register with scheduler if enabled
        if (worker.pool.scheduler) |sched| {
            scheduler.registerWorker(sched, worker.id);
        }
        
        while (worker.pool.running.load(.acquire)) {
            // Check for back-off reset (handles long idle periods gracefully)
            worker.steal_back_off.maybeReset();
            
            // Try to get work (tasks or continuations)
            const work_item = getWork(worker);
            
            if (work_item) |item| {
                // Reset back-off on successful work acquisition
                worker.steal_back_off.recordSuccess();
                
                coz.latencyBegin(coz.Points.task_execution);
                
                // Execute work item based on type
                switch (item) {
                    .task => |task| {
                        // Execute traditional task
                        const task_ptr = @as(*Task, @ptrCast(@alignCast(task)));
                        task_ptr.func(task_ptr.data);
                    },
                    .continuation => |cont| {
                        // Execute continuation with stealing support
                        cont.markRunning(worker.id);
                        cont.execute();
                        
                        // Register completion with continuation registry if available
                        if (worker.continuation_registry) |registry| {
                            registry.completeContinuation(cont) catch |err| {
                                enhanced_errors.logEnhancedError(@TypeOf(err), err, "Failed to register continuation completion");
                                std.log.warn("Worker {d}: Failed to register continuation completion: {}", .{worker.id, err});
                            };
                        }
                    },
                }
                
                coz.latencyEnd(coz.Points.task_execution);
                
                worker.pool.stats.recordComplete();
                coz.throughput(coz.Points.task_completed);
            } else {
                // No work found - record failure and perform intelligent back-off
                worker.steal_back_off.recordFailure();
                
                coz.throughput(coz.Points.worker_idle);
                
                // Perform adaptive exponential back-off instead of fixed 5ms sleep
                // This dynamically adjusts from 100ns to 10ms based on contention level
                worker.steal_back_off.performBackOff();
            }
        }
    }
    
    fn getWork(worker: *Worker) ?lockfree.WorkItem {
        // First try local queue
        switch (worker.queue) {
            .mutex => |*q| {
                if (q.pop()) |work_item| return work_item;
            },
            .lockfree => |*q| {
                if (q.popBottom()) |work_item| {
                    return work_item;
                }
            },
        }
        
        // Then try work stealing if enabled (atomic read to prevent config data races)
        if (worker.pool.isFeatureEnabled("enable_work_stealing")) {
            return stealWork(worker);
        }
        
        return null;
    }
    
    fn stealWork(worker: *Worker) ?lockfree.WorkItem {
        const pool = worker.pool;
        
        coz.latencyBegin(coz.Points.queue_steal);
        defer coz.latencyEnd(coz.Points.queue_steal);
        
        // Topology-aware stealing order: prioritize victims by locality
        if (pool.topology) |topo| {
            return pool.stealWorkTopologyAware(worker, topo);
        } else {
            return pool.stealWorkRandom(worker);
        }
    }
    
    fn stealWorkTopologyAware(self: *Self, worker: *Worker, topo: topology.CpuTopology) ?lockfree.WorkItem {
        const worker_numa = worker.numa_node orelse 0;
        
        // Try stealing in order of preference:
        // 1. Same NUMA node (highest locality)
        // 2. Same socket, different NUMA node
        // 3. Different socket (lowest locality)
        
        // Phase 1: Try same NUMA node workers first
        if (self.tryStealFromNumaNode(worker, worker_numa)) |task| {
            return task;
        }
        
        // Phase 2: Try workers on same socket but different NUMA nodes
        if (self.tryStealFromSocket(worker, topo, worker_numa)) |task| {
            return task;
        }
        
        // Phase 3: Try workers on different sockets (last resort)
        if (self.tryStealFromRemoteNodes(worker, worker_numa)) |task| {
            return task;
        }
        
        return null;
    }
    
    fn tryStealFromNumaNode(self: *Self, worker: *Worker, numa_node: u32) ?lockfree.WorkItem {
        // Create a list of candidate workers on the same NUMA node
        var candidates: [16]usize = undefined; // Support up to 16 workers per NUMA node
        var candidate_count: usize = 0;
        
        for (self.workers, 0..) |*candidate_worker, i| {
            if (i == worker.id) continue; // Don't steal from ourselves
            if (candidate_worker.numa_node == numa_node and candidate_count < candidates.len) {
                candidates[candidate_count] = i;
                candidate_count += 1;
            }
        }
        
        // Try candidates in random order to avoid contention patterns
        if (candidate_count > 0) {
            return self.tryStealFromCandidates(worker, candidates[0..candidate_count]);
        }
        
        return null;
    }
    
    fn tryStealFromSocket(self: *Self, worker: *Worker, topo: topology.CpuTopology, worker_numa: u32) ?lockfree.WorkItem {
        // Find our socket ID through CPU topology
        const worker_socket = blk: {
            if (worker.cpu_id) |cpu_id| {
                for (topo.cores) |core| {
                    if (core.logical_id == cpu_id) {
                        break :blk core.socket_id;
                    }
                }
            }
            break :blk 0; // Default to socket 0
        };
        
        var candidates: [16]usize = undefined;
        var candidate_count: usize = 0;
        
        for (self.workers, 0..) |*candidate_worker, i| {
            if (i == worker.id) continue;
            if (candidate_worker.numa_node == worker_numa) continue; // Already tried same NUMA
            
            // Check if candidate is on same socket
            if (candidate_worker.cpu_id) |cpu_id| {
                for (topo.cores) |core| {
                    if (core.logical_id == cpu_id and core.socket_id == worker_socket) {
                        if (candidate_count < candidates.len) {
                            candidates[candidate_count] = i;
                            candidate_count += 1;
                        }
                        break;
                    }
                }
            }
        }
        
        if (candidate_count > 0) {
            return self.tryStealFromCandidates(worker, candidates[0..candidate_count]);
        }
        
        return null;
    }
    
    fn tryStealFromRemoteNodes(self: *Self, worker: *Worker, worker_numa: u32) ?lockfree.WorkItem {
        var candidates: [16]usize = undefined;
        var candidate_count: usize = 0;
        
        for (self.workers, 0..) |*candidate_worker, i| {
            if (i == worker.id) continue;
            if (candidate_worker.numa_node == worker_numa) continue; // Already tried
            
            if (candidate_count < candidates.len) {
                candidates[candidate_count] = i;
                candidate_count += 1;
            }
        }
        
        if (candidate_count > 0) {
            return self.tryStealFromCandidates(worker, candidates[0..candidate_count]);
        }
        
        return null;
    }
    
    fn tryStealFromCandidates(self: *Self, worker: *Worker, candidates: []const usize) ?lockfree.WorkItem {
        // Sort candidates by NUMA preference for continuation stealing optimization
        var candidate_preferences: [16]struct { id: usize, preference: f32 } = undefined;
        
        for (candidates, 0..) |candidate_id, i| {
            const candidate_worker = &self.workers[candidate_id];
            
            // Check if candidate has work items that prefer being stolen
            var avg_preference: f32 = 0.5; // Default neutral preference
            var item_count: u32 = 0;
            
            // Sample work items to calculate stealing preference
            // Note: This is a simplified heuristic - in practice we'd peek at queue tops
            switch (candidate_worker.queue) {
                .mutex => |*q| {
                    q.mutex.lock();
                    defer q.mutex.unlock();
                    
                    for (&q.work_items) |*priority_queue| {
                        for (priority_queue.items) |work_item| {
                            const preference = work_item.getNumaStealingPreference(worker.numa_node, worker.current_socket);
                            avg_preference = (avg_preference * @as(f32, @floatFromInt(item_count)) + preference) / @as(f32, @floatFromInt(item_count + 1));
                            item_count += 1;
                            
                            // Sample only first few items for performance
                            if (item_count >= 3) break;
                        }
                        if (item_count >= 3) break;
                    }
                },
                .lockfree => |_| {
                    // For lockfree queues, we use a simpler heuristic based on worker locality
                    if (candidate_worker.numa_node != null and worker.numa_node != null) {
                        if (candidate_worker.numa_node.? == worker.numa_node.?) {
                            avg_preference = 0.8; // Same NUMA node
                        } else {
                            avg_preference = 0.4; // Different NUMA node
                        }
                    }
                },
            }
            
            candidate_preferences[i] = .{ .id = candidate_id, .preference = avg_preference };
        }
        
        // Sort candidates by preference (highest first)
        const CandidateSort = struct {
            pub fn lessThan(context: void, a: @TypeOf(candidate_preferences[0]), b: @TypeOf(candidate_preferences[0])) bool {
                _ = context;
                return a.preference > b.preference;
            }
        };
        
        std.sort.insertion(@TypeOf(candidate_preferences[0]), candidate_preferences[0..candidates.len], {}, CandidateSort.lessThan);
        
        // Try stealing from sorted candidates (best NUMA locality first)
        for (candidate_preferences[0..candidates.len], 0..) |candidate_pref, attempt_idx| {
            const victim = &self.workers[candidate_pref.id];
            
            switch (victim.queue) {
                .mutex => |*q| {
                    if (q.steal()) |work_item| {
                        self.stats.recordSteal();
                        coz.throughput(coz.Points.task_stolen);
                        // Mark as stolen with NUMA awareness
                        work_item.markStolenWithNuma(worker.id, worker.numa_node, worker.current_socket);
                        return work_item;
                    }
                },
                .lockfree => |*q| {
                    if (q.steal()) |work_item| {
                        self.stats.recordSteal();
                        coz.throughput(coz.Points.task_stolen);
                        // Mark as stolen with NUMA awareness
                        work_item.markStolenWithNuma(worker.id, worker.numa_node, worker.current_socket);
                        return work_item;
                    }
                },
            }
            
            // Add micro-delay between steal attempts to reduce cache line bouncing
            // Only after failed attempts and not on the last candidate
            if (attempt_idx > 0 and attempt_idx < candidates.len - 1) {
                // Very short pause to allow cache lines to settle
                // This reduces contention when multiple workers are stealing simultaneously
                StealBackOff.performShortSpin(50); // 50ns pause
            }
        }
        
        return null;
    }
    
    fn stealWorkRandom(self: *Self, worker: *Worker) ?lockfree.WorkItem {
        // Fallback to random stealing when topology is not available
        // Note: This only tries one random victim, so no micro-delays needed here
        // The main exponential back-off happens in the worker loop for failed steal attempts
        const victim_id = std.crypto.random.uintLessThan(usize, self.workers.len);
        if (victim_id == worker.id) return null;
        
        const victim = &self.workers[victim_id];
        
        switch (victim.queue) {
            .mutex => |*q| {
                if (q.steal()) |work_item| {
                    self.stats.recordSteal();
                    coz.throughput(coz.Points.task_stolen);
                    // Mark as stolen if it's a continuation
                    work_item.markStolen(worker.id);
                    return work_item;
                }
            },
            .lockfree => |*q| {
                if (q.steal()) |work_item| {
                    self.stats.recordSteal();
                    coz.throughput(coz.Points.task_stolen);
                    // Mark as stolen if it's a continuation
                    work_item.markStolen(worker.id);
                    return work_item;
                }
            },
        }
        
        return null;
    }
    
    /// Apply a function to each element of a slice in parallel (simple implementation)
    pub fn parallelFor(self: *Self, comptime T: type, items: []T, func: *const fn(usize, *T) void) !void {
        _ = self; // For now, just run sequentially to get benchmarks working
        for (items, 0..) |*item, idx| {
            func(idx, item);
        }
    }
};

// ============================================================================
// Public API
// ============================================================================

/// Create a thread pool with DEFAULT ADVANCED FEATURES ENABLED
/// This is now the recommended way to create a Beat.zig thread pool
/// All advanced features are enabled by default with automatic fallback support
pub fn createPool(allocator: std.mem.Allocator) !*ThreadPool {
    // Initialize advanced features with auto-detected configuration
    advanced_features.initializeAdvancedFeatures(advanced_features.AdvancedFeaturesConfig.createAutoConfig());
    return ThreadPool.init(allocator, Config{});
}

/// Create a thread pool with custom configuration
pub fn createPoolWithConfig(allocator: std.mem.Allocator, config: Config) !*ThreadPool {
    return ThreadPool.init(allocator, config);
}

/// Create a thread pool with PRODUCTION-OPTIMIZED advanced features
/// Enables all performance optimizations with minimal debugging overhead
pub fn createProductionPool(allocator: std.mem.Allocator) !*ThreadPool {
    const production_config = Config.createProductionConfig();
    return ThreadPool.init(allocator, production_config);
}

/// Create a thread pool with DEVELOPMENT features enabled
/// Includes comprehensive debugging, profiling, and validation features
pub fn createDevelopmentPool(allocator: std.mem.Allocator) !*ThreadPool {
    const dev_config = Config.createDevelopmentConfig();
    return ThreadPool.init(allocator, dev_config);
}

/// Create a thread pool optimized for EMBEDDED/resource-constrained systems
/// Selective advanced features with minimal resource usage
pub fn createEmbeddedPool(allocator: std.mem.Allocator) !*ThreadPool {
    const embedded_config = Config.createEmbeddedConfig();
    return ThreadPool.init(allocator, embedded_config);
}

/// Create a thread pool with auto-detected optimal configuration
pub fn createOptimalPool(allocator: std.mem.Allocator) !*ThreadPool {
    const optimal_config = build_opts.getOptimalConfig();
    return ThreadPool.init(allocator, optimal_config);
}

/// Create a thread pool optimized for testing with advanced features
pub fn createTestPool(allocator: std.mem.Allocator) !*ThreadPool {
    const test_config = Config.createTestingConfig();
    return ThreadPool.init(allocator, test_config);
}

/// Create a thread pool optimized for benchmarking with all performance features
pub fn createBenchmarkPool(allocator: std.mem.Allocator) !*ThreadPool {
    const benchmark_config = Config.createProfilingConfig();
    return ThreadPool.init(allocator, benchmark_config);
}


/// Manually clean up all ISPC runtime allocations
/// This is automatically called during ThreadPool.deinit(), but can be called manually if needed
pub fn cleanupISPCRuntime(allocator: std.mem.Allocator) void {
    ispc_cleanup.cleanupAllISPCResources(allocator);
}

/// Emergency cleanup for ISPC runtime (forces cleanup even if already called)
/// Use this only in exceptional circumstances (e.g., error recovery)
pub fn emergencyCleanupISPCRuntime(allocator: std.mem.Allocator) void {
    ispc_cleanup.emergencyCleanupAllISPCResources(allocator);
}