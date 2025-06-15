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
// Use smart build configuration that handles dependency scenarios
pub const build_opts = @import("build_opts_new.zig");
pub const comptime_work = @import("comptime_work.zig");
pub const enhanced_errors = @import("enhanced_errors.zig");
pub const fingerprint = @import("fingerprint.zig");
pub const intelligent_decision = @import("intelligent_decision.zig");
pub const predictive_accounting = @import("predictive_accounting.zig");
pub const advanced_worker_selection = @import("advanced_worker_selection.zig");
pub const memory_pressure = @import("memory_pressure.zig");
pub const simd = @import("simd.zig");
pub const simd_batch = @import("simd_batch.zig");
pub const simd_queue = @import("simd_queue.zig");
pub const simd_classifier = @import("simd_classifier.zig");
pub const simd_benchmark = @import("simd_benchmark.zig");
pub const gpu_integration = @import("gpu_integration.zig");
pub const mock_sycl = @import("mock_sycl.zig");
pub const gpu_classifier = @import("gpu_classifier.zig");
pub const sycl_detection = @import("sycl_detection.zig");

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
    
    // V3 features - auto-enabled based on hardware detection
    enable_topology_aware: bool = build_opts.performance.enable_topology_aware,
    enable_numa_aware: bool = build_opts.performance.enable_numa_aware,
    enable_lock_free: bool = true,          // Lock-free data structures
    enable_predictive: bool = true,          // Predictive scheduling with One Euro Filter
    
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
    
    // Advanced worker selection (Task 2.4.2)
    enable_advanced_selection: bool = true,        // Use multi-criteria optimization
    selection_criteria: ?advanced_worker_selection.SelectionCriteria = null,  // null = auto-detect optimal
    enable_selection_learning: bool = true,       // Adaptive criteria adjustment
    
    // Development mode configuration
    development_mode: bool = false,               // Enable comprehensive development features
    verbose_logging: bool = false,                // Detailed operation logging
    performance_validation: bool = false,        // Runtime performance checks
    memory_debugging: bool = false,               // Enhanced memory tracking
    task_tracing: bool = false,                   // Individual task execution tracing
    scheduler_profiling: bool = false,            // Detailed scheduler performance profiling
    deadlock_detection: bool = false,             // Runtime deadlock detection
    resource_leak_detection: bool = false,        // Resource cleanup validation
    
    /// Create a development configuration with comprehensive debugging enabled
    pub fn createDevelopmentConfig() Config {
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
        
        // Conservative settings for debugging
        config.num_workers = 2;  // Smaller pool for easier debugging
        config.task_queue_size = 16; // Smaller queue for faster issue detection
        config.heartbeat_interval_us = 50; // More frequent heartbeats for responsiveness
        
        return config;
    }
    
    /// Create a testing configuration optimized for unit tests
    pub fn createTestingConfig() Config {
        var config = Config{};
        config.development_mode = true;
        config.verbose_logging = false; // Reduce noise in tests
        config.performance_validation = true;
        config.memory_debugging = true;
        config.resource_leak_detection = true;
        config.enable_statistics = true;
        
        // Fast, small configuration for tests
        config.num_workers = 2;
        config.task_queue_size = 8;
        config.heartbeat_interval_us = 10; // Very fast for test responsiveness
        config.promotion_threshold = 5; // Lower threshold for faster promotion in tests
        
        return config;
    }
    
    /// Create a profiling configuration optimized for performance analysis
    pub fn createProfilingConfig() Config {
        var config = Config{};
        config.development_mode = true;
        config.verbose_logging = false;
        config.performance_validation = false; // Disable to avoid interference
        config.scheduler_profiling = true;
        config.task_tracing = false; // Disable to reduce overhead
        config.enable_statistics = true;
        
        // Optimal performance settings for accurate profiling
        // Use default optimized values from build_opts
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

pub const Task = struct {
    func: *const fn (*anyopaque) void,
    data: *anyopaque,
    priority: Priority = .normal,
    affinity_hint: ?u32 = null,              // Preferred NUMA node (v3)
    
    // Optional fingerprinting support (Phase 2 enhancement)
    fingerprint_hash: ?u64 = null,           // Cache fingerprint hash for performance
    creation_timestamp: ?u64 = null,         // Task creation time for temporal analysis
    data_size_hint: ?usize = null,           // Hint about data size for optimization
};

// ============================================================================
// Statistics
// ============================================================================

pub const ThreadPoolStats = struct {
    tasks_submitted: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    tasks_completed: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    tasks_stolen: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    tasks_cancelled: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    
    pub fn recordSubmit(self: *ThreadPoolStats) void {
        _ = self.tasks_submitted.fetchAdd(1, .monotonic);
    }
    
    pub fn recordComplete(self: *ThreadPoolStats) void {
        _ = self.tasks_completed.fetchAdd(1, .monotonic);
    }
    
    pub fn recordSteal(self: *ThreadPoolStats) void {
        _ = self.tasks_stolen.fetchAdd(1, .monotonic);
    }
    
    pub fn recordCancel(self: *ThreadPoolStats) void {
        _ = self.tasks_cancelled.fetchAdd(1, .monotonic);
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
    
    // Optional subsystems
    topology: ?topology.CpuTopology = null,
    scheduler: ?*scheduler.Scheduler = null,
    memory_pool: ?*memory.TaskPool = null,
    decision_framework: ?*intelligent_decision.IntelligentDecisionFramework = null,
    fingerprint_registry: ?*fingerprint.FingerprintRegistry = null,
    advanced_selector: ?*advanced_worker_selection.AdvancedWorkerSelector = null,
    
    const Self = @This();
    
    const Worker = struct {
        id: u32,
        thread: std.Thread,
        pool: *ThreadPool,
        
        // Queues based on configuration
        queue: union(enum) {
            mutex: MutexQueue,
            lockfree: lockfree.WorkStealingDeque(*Task),
        },
        
        // CPU affinity (v3)
        cpu_id: ?u32 = null,
        numa_node: ?u32 = null,
    };
    
    const MutexQueue = struct {
        tasks: [3]std.ArrayList(Task), // One per priority
        mutex: std.Thread.Mutex,
        
        pub fn init(allocator: std.mem.Allocator) MutexQueue {
            return .{
                .tasks = .{
                    std.ArrayList(Task).init(allocator),
                    std.ArrayList(Task).init(allocator),
                    std.ArrayList(Task).init(allocator),
                },
                .mutex = .{},
            };
        }
        
        pub fn deinit(self: *MutexQueue) void {
            for (&self.tasks) |*queue| {
                queue.deinit();
            }
        }
        
        pub fn push(self: *MutexQueue, task: Task) !void {
            self.mutex.lock();
            defer self.mutex.unlock();
            try self.tasks[@intFromEnum(task.priority)].append(task);
        }
        
        pub fn pop(self: *MutexQueue) ?Task {
            self.mutex.lock();
            defer self.mutex.unlock();
            
            // Check high priority first
            var i: i32 = 2;
            while (i >= 0) : (i -= 1) {
                const idx = @as(usize, @intCast(i));
                if (self.tasks[idx].items.len > 0) {
                    return self.tasks[idx].pop();
                }
            }
            return null;
        }
        
        pub fn steal(self: *MutexQueue) ?Task {
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
        
        // Fallback worker count with enhanced error reporting
        if (actual_config.num_workers == null) {
            actual_config.num_workers = std.Thread.getCpuCount() catch blk: {
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
        };
        
        // Initialize optional subsystems
        if (actual_config.enable_heartbeat or actual_config.enable_predictive) {
            self.scheduler = try scheduler.Scheduler.init(allocator, &actual_config);
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
            self.advanced_selector.?.* = advanced_worker_selection.AdvancedWorkerSelector.init(allocator, selection_criteria);
            
            // Connect prediction and analysis components if available
            self.advanced_selector.?.setComponents(
                self.fingerprint_registry,
                null, // Predictive scheduler will be set later if available
                self.decision_framework
            );
        }
        
        // Initialize workers
        for (self.workers, 0..) |*worker, i| {
            const worker_config = WorkerConfig{
                .id = @intCast(i),
                .pool = self,
                .cpu_id = if (self.topology) |topo| @as(u32, @intCast(i % topo.total_cores)) else null,
                .numa_node = if (self.topology) |topo| topo.logical_to_numa[i % topo.total_cores] else null,
            };
            
            try initWorker(worker, allocator, &actual_config, worker_config);
        }
        
        // Start workers
        for (self.workers) |*worker| {
            worker.thread = try std.Thread.spawn(.{}, workerLoop, .{worker});
            
            // Set CPU affinity if available
            if (worker.cpu_id) |cpu_id| {
                if (self.topology != null) {
                    topology.setThreadAffinity(worker.thread, &[_]u32{cpu_id}) catch {};
                }
            }
        }
        
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
            self.allocator.destroy(selector);
        }
        
        self.allocator.free(self.workers);
        self.allocator.destroy(self);
    }
    
    pub fn submit(self: *Self, task: Task) !void {
        coz.latencyBegin(coz.Points.task_submitted);
        defer coz.latencyEnd(coz.Points.task_submitted);
        
        self.stats.recordSubmit();
        coz.throughput(coz.Points.task_submitted);
        
        // Choose worker based on affinity hint and current load
        const worker_id = self.selectWorker(task);
        const worker = &self.workers[worker_id];
        
        switch (worker.queue) {
            .mutex => |*q| try q.push(task),
            .lockfree => |*q| {
                // Allocate task from memory pool if available
                const task_ptr = if (self.memory_pool) |pool|
                    try pool.alloc()
                else
                    try self.allocator.create(Task);
                
                task_ptr.* = task;
                try q.pushBottom(task_ptr);
            },
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
                        break :blk q.tasks[0].items.len == 0 and 
                                   q.tasks[1].items.len == 0 and 
                                   q.tasks[2].items.len == 0;
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
            
            // Create worker info array for decision making
            var worker_infos = std.ArrayList(intelligent_decision.WorkerInfo).init(self.allocator);
            defer worker_infos.deinit();
            
            for (self.workers, 0..) |*worker, i| {
                const worker_info = intelligent_decision.WorkerInfo{
                    .id = worker.id,
                    .numa_node = worker.numa_node,
                    .queue_size = self.getWorkerQueueSize(i),
                    .max_queue_size = 1024, // Default max queue size
                };
                
                worker_infos.append(worker_info) catch {
                    // Fallback to legacy selection on allocation failure
                    return self.selectWorkerLegacy(task);
                };
            }
            
            // Use advanced multi-criteria optimization
            const decision = selector.selectWorker(&task, worker_infos.items, if (self.topology) |*topo| topo else null) catch {
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
            
            // Create worker info array for decision making
            var worker_infos = std.ArrayList(intelligent_decision.WorkerInfo).init(self.allocator);
            defer worker_infos.deinit();
            
            for (self.workers, 0..) |*worker, i| {
                const worker_info = intelligent_decision.WorkerInfo{
                    .id = worker.id,
                    .numa_node = worker.numa_node,
                    .queue_size = self.getWorkerQueueSize(i),
                    .max_queue_size = 1024, // Default max queue size
                };
                
                worker_infos.append(worker_info) catch {
                    // Fallback to legacy selection on allocation failure
                    return self.selectWorkerLegacy(task);
                };
            }
            
            // Make intelligent scheduling decision
            const decision = framework.makeSchedulingDecision(
                &task,
                worker_infos.items,
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
        const submission_count = self.stats.tasks_submitted.load(.acquire);
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
                break :blk q.tasks[0].items.len + q.tasks[1].items.len + q.tasks[2].items.len;
            },
            .lockfree => |*q| q.size(),
        };
    }
    
    const WorkerConfig = struct {
        id: u32,
        pool: *ThreadPool,
        cpu_id: ?u32,
        numa_node: ?u32,
    };
    
    fn initWorker(worker: *Worker, allocator: std.mem.Allocator, config: *const Config, worker_config: WorkerConfig) !void {
        worker.* = .{
            .id = worker_config.id,
            .thread = undefined,
            .pool = worker_config.pool,
            .queue = if (config.enable_lock_free)
                .{ .lockfree = try lockfree.WorkStealingDeque(*Task).init(allocator, config.task_queue_size) }
            else
                .{ .mutex = MutexQueue.init(allocator) },
            .cpu_id = worker_config.cpu_id,
            .numa_node = worker_config.numa_node,
        };
    }
    
    fn workerLoop(worker: *Worker) void {
        // Register with scheduler if enabled
        if (worker.pool.scheduler) |sched| {
            scheduler.registerWorker(sched, worker.id);
        }
        
        while (worker.pool.running.load(.acquire)) {
            // Try to get work
            const task = getWork(worker);
            
            if (task) |t| {
                coz.latencyBegin(coz.Points.task_execution);
                t.func(t.data);
                coz.latencyEnd(coz.Points.task_execution);
                
                worker.pool.stats.recordComplete();
                coz.throughput(coz.Points.task_completed);
            } else {
                coz.throughput(coz.Points.worker_idle);
                // Sleep briefly to avoid busy-waiting
                // Using 5ms sleep to reduce CPU usage while maintaining reasonable responsiveness
                // This reduces idle CPU from ~13% to ~3% with minimal impact on latency
                std.time.sleep(5 * std.time.ns_per_ms);
            }
        }
    }
    
    fn getWork(worker: *Worker) ?Task {
        // First try local queue
        switch (worker.queue) {
            .mutex => |*q| {
                if (q.pop()) |task| return task;
            },
            .lockfree => |*q| {
                if (q.popBottom()) |task_ptr| {
                    // Get task value and potentially free the allocation
                    const task = task_ptr.*;
                    if (worker.pool.memory_pool) |pool| {
                        pool.free(task_ptr);
                    }
                    return task;
                }
            },
        }
        
        // Then try work stealing if enabled
        if (worker.pool.config.enable_work_stealing) {
            return stealWork(worker);
        }
        
        return null;
    }
    
    fn stealWork(worker: *Worker) ?Task {
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
    
    fn stealWorkTopologyAware(self: *Self, worker: *Worker, topo: topology.CpuTopology) ?Task {
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
    
    fn tryStealFromNumaNode(self: *Self, worker: *Worker, numa_node: u32) ?Task {
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
            return self.tryStealFromCandidates(candidates[0..candidate_count]);
        }
        
        return null;
    }
    
    fn tryStealFromSocket(self: *Self, worker: *Worker, topo: topology.CpuTopology, worker_numa: u32) ?Task {
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
            return self.tryStealFromCandidates(candidates[0..candidate_count]);
        }
        
        return null;
    }
    
    fn tryStealFromRemoteNodes(self: *Self, worker: *Worker, worker_numa: u32) ?Task {
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
            return self.tryStealFromCandidates(candidates[0..candidate_count]);
        }
        
        return null;
    }
    
    fn tryStealFromCandidates(self: *Self, candidates: []const usize) ?Task {
        // Shuffle candidates to avoid contention patterns
        var shuffled_candidates: [16]usize = undefined;
        @memcpy(shuffled_candidates[0..candidates.len], candidates);
        
        // Simple Fisher-Yates shuffle for the candidates
        var i = candidates.len;
        while (i > 1) {
            i -= 1;
            const j = std.crypto.random.uintLessThan(usize, i + 1);
            const temp = shuffled_candidates[i];
            shuffled_candidates[i] = shuffled_candidates[j];
            shuffled_candidates[j] = temp;
        }
        
        // Try stealing from shuffled candidates
        for (shuffled_candidates[0..candidates.len]) |victim_id| {
            const victim = &self.workers[victim_id];
            
            switch (victim.queue) {
                .mutex => |*q| {
                    if (q.steal()) |task| {
                        self.stats.recordSteal();
                        coz.throughput(coz.Points.task_stolen);
                        return task;
                    }
                },
                .lockfree => |*q| {
                    if (q.steal()) |task_ptr| {
                        self.stats.recordSteal();
                        coz.throughput(coz.Points.task_stolen);
                        // Get task value and potentially free the allocation
                        const task = task_ptr.*;
                        if (self.memory_pool) |mem_pool| {
                            mem_pool.free(task_ptr);
                        }
                        return task;
                    }
                },
            }
        }
        
        return null;
    }
    
    fn stealWorkRandom(self: *Self, worker: *Worker) ?Task {
        // Fallback to random stealing when topology is not available
        const victim_id = std.crypto.random.uintLessThan(usize, self.workers.len);
        if (victim_id == worker.id) return null;
        
        const victim = &self.workers[victim_id];
        
        switch (victim.queue) {
            .mutex => |*q| {
                if (q.steal()) |task| {
                    self.stats.recordSteal();
                    coz.throughput(coz.Points.task_stolen);
                    return task;
                }
            },
            .lockfree => |*q| {
                if (q.steal()) |task_ptr| {
                    self.stats.recordSteal();
                    coz.throughput(coz.Points.task_stolen);
                    // Get task value and potentially free the allocation
                    const task = task_ptr.*;
                    if (self.memory_pool) |mem_pool| {
                        mem_pool.free(task_ptr);
                    }
                    return task;
                }
            },
        }
        
        return null;
    }
};

// ============================================================================
// Public API
// ============================================================================

/// Create a thread pool with default configuration
pub fn createPool(allocator: std.mem.Allocator) !*ThreadPool {
    return ThreadPool.init(allocator, Config{});
}

/// Create a thread pool with custom configuration
pub fn createPoolWithConfig(allocator: std.mem.Allocator, config: Config) !*ThreadPool {
    return ThreadPool.init(allocator, config);
}

/// Create a thread pool with auto-detected optimal configuration
pub fn createOptimalPool(allocator: std.mem.Allocator) !*ThreadPool {
    const optimal_config = build_opts.getOptimalConfig();
    return ThreadPool.init(allocator, optimal_config);
}

/// Create a thread pool optimized for testing
pub fn createTestPool(allocator: std.mem.Allocator) !*ThreadPool {
    const test_config = build_opts.getTestConfig();
    return ThreadPool.init(allocator, test_config);
}

/// Create a thread pool optimized for benchmarking
pub fn createBenchmarkPool(allocator: std.mem.Allocator) !*ThreadPool {
    const benchmark_config = build_opts.getBenchmarkConfig();
    return ThreadPool.init(allocator, benchmark_config);
}