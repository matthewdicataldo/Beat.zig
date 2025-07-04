const std = @import("std");
const builtin = @import("builtin");
const core = @import("core.zig");
const memory_pressure = @import("memory_pressure.zig");

// Scheduling algorithms for Beat.zig
//
// This module provides intelligent scheduling algorithms including:
// - Token accounting system with work:overhead ratio tracking
// - Heartbeat scheduling for promotion decisions
// - One Euro Filter for adaptive task execution time prediction
// - Performance measurement utilities
//
// The One Euro Filter (€1 Filter) provides superior task execution time prediction
// compared to simple averaging, with excellent price-to-performance ratio! 😄
// It adapts to workload changes, handles outliers gracefully, and costs only
// ~4x computational overhead for significantly better scheduling intelligence.

// ============================================================================
// Token Accounting (v2 Heartbeat)
// ============================================================================
//
// The token accounting system tracks work vs overhead cycles to make intelligent
// promotion decisions. It uses a hybrid caching approach that ensures correctness
// while maintaining performance:
//
// - Always evaluates on first meaningful update
// - Uses interval-based caching for subsequent updates (CHECK_INTERVAL = 1000)
// - Respects both promotion threshold and minimum work cycles
// - Handles edge cases (zero overhead, zero work) safely

pub const TokenAccount = struct {
    work_cycles: u64 = 0,
    overhead_cycles: u64 = 0,
    promotion_threshold: u64,
    min_work_cycles: u64,
    should_promote_cached: bool = false,
    last_check_cycles: u64 = 0,
    first_update_done: bool = false,
    
    const CHECK_INTERVAL = 1000;
    
    pub fn init(config: *const core.Config) TokenAccount {
        return .{
            .promotion_threshold = config.promotion_threshold,
            .min_work_cycles = config.min_work_cycles,
        };
    }
    
    pub inline fn update(self: *TokenAccount, work: u64, overhead: u64) void {
        self.work_cycles +%= work;
        self.overhead_cycles +%= overhead;
        
        // Always check on first meaningful update, then use interval-based caching
        const should_check = !self.first_update_done or 
            (self.overhead_cycles -% self.last_check_cycles > CHECK_INTERVAL);
            
        if (should_check and self.overhead_cycles > 0) {
            self.first_update_done = true;
            self.last_check_cycles = self.overhead_cycles;
            self.should_promote_cached = self.work_cycles >= self.min_work_cycles and 
                self.work_cycles > (self.overhead_cycles * self.promotion_threshold);
        }
    }
    
    pub inline fn shouldPromote(self: *const TokenAccount) bool {
        return self.should_promote_cached;
    }
    
    pub fn reset(self: *TokenAccount) void {
        self.work_cycles = 0;
        self.overhead_cycles = 0;
        self.should_promote_cached = false;
        self.last_check_cycles = 0;
        self.first_update_done = false;
    }
};

// ============================================================================
// Heartbeat Scheduler
// ============================================================================

pub const Scheduler = struct {
    allocator: std.mem.Allocator,
    config: *const core.Config,
    heartbeat_thread: ?std.Thread = null,
    running: std.atomic.Value(bool) = std.atomic.Value(bool).init(true),
    
    // Per-worker token accounts
    worker_tokens: []TokenAccount,
    
    // Promotion tracking
    promotions_triggered: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    
    // Memory pressure monitoring (V3 - Memory-Aware Scheduling)
    memory_monitor: ?*memory_pressure.MemoryPressureMonitor = null,
    memory_pressure_callbacks: memory_pressure.PressureCallbackRegistry,
    
    // Memory-aware scheduling state
    pressure_adaptations: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    last_pressure_check: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    
    // Adaptive heartbeat timing (Performance Optimization)
    adaptive_interval_ns: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    recent_promotions: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    last_activity_check: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    heartbeat_adjustments: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    
    pub fn init(allocator: std.mem.Allocator, config: *const core.Config) !*Scheduler {
        const self = try allocator.create(Scheduler);
        self.* = .{
            .allocator = allocator,
            .config = config,
            .worker_tokens = try allocator.alloc(TokenAccount, config.num_workers orelse 1),
            .memory_pressure_callbacks = memory_pressure.PressureCallbackRegistry.init(allocator),
        };
        
        // Initialize token accounts
        for (self.worker_tokens) |*tokens| {
            tokens.* = TokenAccount.init(config);
        }
        
        // Initialize adaptive heartbeat with baseline interval
        const baseline_interval_ns = @as(u64, config.heartbeat_interval_us) * 1000;
        self.adaptive_interval_ns.store(baseline_interval_ns, .monotonic);
        self.last_activity_check.store(@as(u64, @intCast(std.time.nanoTimestamp())), .monotonic);
        
        // Initialize memory pressure monitoring if enabled
        if (config.enable_numa_aware) { // Reuse NUMA flag for memory awareness
            const pressure_config = memory_pressure.MemoryPressureConfig{
                .update_interval_ms = config.heartbeat_interval_us / 1000, // Sync with heartbeat
            };
            
            self.memory_monitor = memory_pressure.MemoryPressureMonitor.init(allocator, pressure_config) catch |err| blk: {
                // Memory monitoring is optional - log error but continue
                if (builtin.mode == .Debug) {
                    std.debug.print("Scheduler: Failed to initialize memory monitor: {}\n", .{err});
                }
                break :blk null;
            };
            
            // Start memory monitoring if successfully initialized
            if (self.memory_monitor) |monitor| {
                monitor.start() catch |err| {
                    if (builtin.mode == .Debug) {
                        std.debug.print("Scheduler: Failed to start memory monitor: {}\n", .{err});
                    }
                    monitor.deinit();
                    self.memory_monitor = null;
                };
            }
        }
        
        // Start heartbeat thread if enabled
        if (config.enable_heartbeat) {
            self.heartbeat_thread = try std.Thread.spawn(.{}, heartbeatLoop, .{self});
        }
        
        return self;
    }
    
    pub fn deinit(self: *Scheduler) void {
        self.running.store(false, .release);
        
        if (self.heartbeat_thread) |thread| {
            thread.join();
        }
        
        // Cleanup memory pressure monitoring
        if (self.memory_monitor) |monitor| {
            monitor.deinit();
        }
        
        self.memory_pressure_callbacks.deinit();
        self.allocator.free(self.worker_tokens);
        self.allocator.destroy(self);
    }
    
    /// Trigger work promotion for a specific worker
    /// This signals that the worker has favorable work:overhead ratio
    pub fn triggerPromotion(self: *Scheduler, worker_id: u32) void {
        // Record the promotion event
        _ = self.promotions_triggered.fetchAdd(1, .monotonic);
        
        // Log promotion for debugging (in debug builds)
        if (builtin.mode == .Debug) {
            std.debug.print("Scheduler: Work promotion triggered for worker {} (total: {})\n", .{
                worker_id, self.promotions_triggered.load(.monotonic)
            });
        }
        
        // Future enhancement: Could signal thread pool for:
        // - Prioritizing this worker's queue
        // - Adjusting worker thread priorities  
        // - Triggering load balancing
        // - Influencing task placement decisions
    }
    
    /// Get promotion statistics
    pub fn getPromotionCount(self: *const Scheduler) u64 {
        return self.promotions_triggered.load(.acquire);
    }
    
    // ========================================================================
    // Memory-Aware Scheduling (V3)
    // ========================================================================
    
    /// Get current memory pressure level
    pub fn getMemoryPressureLevel(self: *const Scheduler) memory_pressure.MemoryPressureLevel {
        if (self.memory_monitor) |monitor| {
            return monitor.getCurrentLevel();
        }
        return .none;
    }
    
    /// Get current memory pressure metrics
    pub fn getMemoryPressureMetrics(self: *const Scheduler) ?memory_pressure.MemoryPressureMetrics {
        if (self.memory_monitor) |monitor| {
            return monitor.getCurrentMetrics();
        }
        return null;
    }
    
    /// Check if memory pressure suggests deferring new tasks
    pub fn shouldDeferTasksForMemory(self: *const Scheduler) bool {
        if (self.memory_monitor) |monitor| {
            return monitor.shouldDeferTasks();
        }
        return false;
    }
    
    /// Check if memory pressure suggests preferring local NUMA placement
    pub fn shouldPreferLocalNUMAForMemory(self: *const Scheduler) bool {
        if (self.memory_monitor) |monitor| {
            return monitor.shouldPreferLocalNUMA();
        }
        return false;
    }
    
    /// Get recommended task batch limit based on memory pressure
    pub fn getMemoryAwareBatchLimit(self: *const Scheduler, default_limit: u32) u32 {
        if (self.memory_monitor) |monitor| {
            return monitor.getTaskBatchLimit(default_limit);
        }
        return default_limit;
    }
    
    /// Register a callback for memory pressure events
    pub fn registerMemoryPressureCallback(self: *Scheduler, callback: memory_pressure.PressureCallback) !void {
        try self.memory_pressure_callbacks.register(callback);
    }
    
    /// Handle memory pressure change (called by heartbeat loop)
    fn handleMemoryPressureChange(self: *Scheduler, level: memory_pressure.MemoryPressureLevel, metrics: *const memory_pressure.MemoryPressureMetrics) void {
        // Additional defensive check - ensure metrics pointer is valid and metrics are current
        if (metrics.last_update_ns == 0) {
            std.log.warn("Scheduler: handleMemoryPressureChange called with invalid metrics, ignoring", .{});
            return;
        }
        
        // Record adaptation event
        _ = self.pressure_adaptations.fetchAdd(1, .monotonic);
        
        // Log significant pressure changes with defensive value checks
        if (builtin.mode == .Debug) {
            // Guard against invalid metric values that could cause format issues
            const safe_some_avg10 = if (std.math.isNan(metrics.some_avg10) or std.math.isInf(metrics.some_avg10)) 0.0 else metrics.some_avg10;
            const safe_mem_used_pct = if (std.math.isNan(metrics.memory_used_pct) or std.math.isInf(metrics.memory_used_pct)) 0.0 else metrics.memory_used_pct;
            
            std.debug.print("Scheduler: Memory pressure changed to {s} (some_avg10: {:.1}%, mem_used: {:.1}%)\n", 
                .{ @tagName(level), safe_some_avg10, safe_mem_used_pct });
        }
        
        // Trigger registered callbacks with validated metrics
        self.memory_pressure_callbacks.triggerCallbacks(level, metrics);
        
        // Built-in adaptive behaviors based on pressure level
        switch (level) {
            .none, .low => {
                // Normal operation or slight preference for local NUMA
                // No specific action needed - handled by scheduling decisions
            },
            .medium => {
                // Start deferring non-critical tasks and reducing batch sizes
                // This is handled by the scheduling decision functions above
                if (builtin.mode == .Debug) {
                    std.debug.print("Scheduler: Enabling medium memory pressure adaptations\n", .{});
                }
            },
            .high => {
                // Aggressive memory management - significant task deferral
                if (builtin.mode == .Debug) {
                    std.debug.print("Scheduler: Enabling high memory pressure adaptations\n", .{});
                }
            },
            .critical => {
                // Emergency mode - minimal task acceptance
                if (builtin.mode == .Debug) {
                    std.debug.print("Scheduler: CRITICAL memory pressure - emergency adaptations active\n", .{});
                }
            },
        }
    }
    
    /// Get memory pressure adaptation statistics
    pub fn getMemoryAdaptationCount(self: *const Scheduler) u64 {
        return self.pressure_adaptations.load(.acquire);
    }
    
    // ========================================================================
    // Adaptive Heartbeat Timing (Performance Optimization)
    // ========================================================================
    
    /// Calculate adaptive heartbeat interval based on recent activity
    fn calculateAdaptiveInterval(self: *Scheduler, baseline_ns: u64, previous_promotions: u64) u64 {
        const current_time = @as(u64, @intCast(std.time.nanoTimestamp()));
        const current_promotions = self.promotions_triggered.load(.monotonic);
        const last_check = self.last_activity_check.load(.monotonic);
        
        // Calculate promotion rate (promotions per second)
        const time_delta_ns = if (current_time >= last_check) current_time - last_check else 0;
        const promotion_delta = current_promotions - previous_promotions;
        
        if (time_delta_ns < 100_000_000) { // Less than 100ms since last check
            return self.adaptive_interval_ns.load(.monotonic); // Use cached interval
        }
        
        // Update activity tracking
        self.last_activity_check.store(current_time, .monotonic);
        
        // Calculate promotion rate (promotions per second)
        const time_delta_s = @as(f64, @floatFromInt(time_delta_ns)) / 1_000_000_000.0;
        const promotion_rate = if (time_delta_s > 0.001) 
            @as(f64, @floatFromInt(promotion_delta)) / time_delta_s 
        else 
            0.0;
        
        // Adaptive scaling based on activity level
        var scale_factor: f64 = 1.0;
        
        if (promotion_rate > 50.0) {
            // High activity: increase frequency (reduce interval) by up to 50%
            scale_factor = 0.5; // 2x faster heartbeat
        } else if (promotion_rate > 20.0) {
            // Medium activity: increase frequency by 25%
            scale_factor = 0.75; // 1.33x faster heartbeat
        } else if (promotion_rate > 5.0) {
            // Light activity: normal frequency
            scale_factor = 1.0; // baseline frequency
        } else if (promotion_rate > 1.0) {
            // Very light activity: reduce frequency by 50%
            scale_factor = 2.0; // 2x slower heartbeat
        } else {
            // Idle: reduce frequency by up to 90%
            scale_factor = 10.0; // 10x slower heartbeat
        }
        
        // Apply scaling with bounds checking
        const new_interval_ns = @as(u64, @intFromFloat(@as(f64, @floatFromInt(baseline_ns)) * scale_factor));
        
        // Clamp to reasonable bounds (10μs to 10ms)
        const min_interval_ns = 10_000; // 10μs minimum for responsiveness
        const max_interval_ns = 10_000_000; // 10ms maximum for power savings
        const clamped_interval_ns = @max(min_interval_ns, @min(max_interval_ns, new_interval_ns));
        
        // Store new interval and track adjustments
        self.adaptive_interval_ns.store(clamped_interval_ns, .monotonic);
        _ = self.heartbeat_adjustments.fetchAdd(1, .monotonic);
        
        // Debug logging in development builds
        if (builtin.mode == .Debug and scale_factor != 1.0) {
            std.debug.print("Adaptive heartbeat: rate={:.1}/s scale={:.2} interval={}μs\n", 
                .{ promotion_rate, scale_factor, clamped_interval_ns / 1000 });
        }
        
        return clamped_interval_ns;
    }
    
    /// Get current adaptive heartbeat statistics
    pub fn getHeartbeatStats(self: *const Scheduler) struct { 
        current_interval_us: u64, 
        adjustments_count: u64 
    } {
        const current_interval_ns = self.adaptive_interval_ns.load(.acquire);
        const adjustments = self.heartbeat_adjustments.load(.acquire);
        
        return .{
            .current_interval_us = current_interval_ns / 1000,
            .adjustments_count = adjustments,
        };
    }
    
    fn heartbeatLoop(self: *Scheduler) void {
        const baseline_interval_ns = @as(u64, self.config.heartbeat_interval_us) * 1000;
        var previous_pressure_level: memory_pressure.MemoryPressureLevel = .none;
        var previous_promotions: u64 = 0;
        
        while (self.running.load(.acquire)) {
            // Adaptive timing: adjust interval based on recent activity
            const current_interval_ns = self.calculateAdaptiveInterval(baseline_interval_ns, previous_promotions);
            std.time.sleep(current_interval_ns);
            
            // Periodic promotion check with activity tracking
            var promotions_this_cycle: u64 = 0;
            for (self.worker_tokens, 0..) |*tokens, worker_id| {
                if (tokens.shouldPromote()) {
                    // Trigger work promotion: signal that work execution is efficient
                    self.triggerPromotion(@intCast(worker_id));
                    tokens.reset();
                    promotions_this_cycle += 1;
                }
            }
            
            // Update activity tracking for adaptive timing
            previous_promotions = self.promotions_triggered.load(.monotonic);
            
            // Memory pressure monitoring (V3 - Memory-Aware Scheduling)
            if (self.memory_monitor) |monitor| {
                const current_level = monitor.getCurrentLevel();
                
                // Only trigger handler if pressure level changed or on significant updates
                if (current_level != previous_pressure_level) {
                    const metrics = monitor.getCurrentMetrics();
                    
                    // Defensive null guards for memory pressure metrics
                    if (metrics.last_update_ns > 0) {
                        // Metrics are valid - safe to proceed with callback
                        self.handleMemoryPressureChange(current_level, &metrics);
                        previous_pressure_level = current_level;
                    } else {
                        // Metrics are stale or invalid - skip callback but update level to prevent spam
                        std.log.debug("Scheduler: Skipping memory pressure callback due to invalid metrics", .{});
                        previous_pressure_level = current_level;
                    }
                    
                    // Update timestamp of last pressure check
                    const current_time = @as(u64, @intCast(std.time.nanoTimestamp()));
                    self.last_pressure_check.store(current_time, .release);
                }
            }
        }
    }
};

// ============================================================================
// Thread-Local Scheduler State
// ============================================================================

threadlocal var tls_tokens: TokenAccount = undefined;
threadlocal var tls_tokens_init: bool = false;
threadlocal var tls_worker_id: u32 = 0;

pub fn registerWorker(scheduler: *Scheduler, worker_id: u32) void {
    tls_worker_id = worker_id;
    tls_tokens = scheduler.worker_tokens[worker_id];
    tls_tokens_init = true;
}

pub fn getLocalTokens() ?*TokenAccount {
    if (tls_tokens_init) {
        return &tls_tokens;
    }
    return null;
}

// ============================================================================
// Work Prediction (v3) - One Euro Filter
// ============================================================================
//
// Implements adaptive task execution time prediction using the One Euro Filter
// algorithm. This provides superior performance over simple averaging for:
// - Variable workloads with data-dependent execution times
// - Phase changes in application execution
// - Outlier resilience (cache misses, thermal throttling)
// - Microarchitectural adaptation (CPU scaling, NUMA effects)

/// One Euro Filter for adaptive signal smoothing
/// Balances jitter reduction vs lag using adaptive cutoff frequency
pub const OneEuroFilter = struct {
    // Configuration parameters
    min_cutoff: f32,        // Minimum cutoff frequency (Hz) - controls jitter at low speeds
    beta: f32,              // Speed coefficient - controls lag at high speeds  
    d_cutoff: f32,          // Derivative cutoff frequency - smooths velocity estimation
    
    // Internal state
    x_prev: ?f32 = null,    // Previous filtered value
    dx_prev: f32 = 0.0,     // Previous filtered derivative
    t_prev: ?u64 = null,    // Previous timestamp (nanoseconds)
    
    /// Initialize One Euro Filter with tuned parameters for task execution time prediction
    pub fn init(min_cutoff: f32, beta: f32, d_cutoff: f32) OneEuroFilter {
        return .{
            .min_cutoff = min_cutoff,
            .beta = beta,
            .d_cutoff = d_cutoff,
        };
    }
    
    /// Create filter with default parameters optimized for task execution times
    pub fn initDefault() OneEuroFilter {
        return init(
            1.0,    // min_cutoff: 1Hz - good balance for task execution times
            0.1,    // beta: moderate adaptation to workload changes
            1.0     // d_cutoff: standard derivative smoothing
        );
    }
    
    /// Apply One Euro Filter to new measurement
    pub fn filter(self: *OneEuroFilter, measurement: f32, timestamp_ns: u64) f32 {
        // Handle first measurement
        if (self.t_prev == null) {
            self.x_prev = measurement;
            self.t_prev = timestamp_ns;
            return measurement;
        }
        
        // Calculate time delta in seconds
        const dt = @as(f32, @floatFromInt(timestamp_ns - self.t_prev.?)) / 1_000_000_000.0;
        self.t_prev = timestamp_ns;
        
        // Avoid division by zero for very small time deltas
        if (dt <= 0.0) return self.x_prev.?;
        
        // Estimate velocity (rate of change)
        const dx = (measurement - self.x_prev.?) / dt;
        
        // Filter the derivative to smooth velocity estimation
        const dx_alpha = smoothingFactor(self.d_cutoff, dt);
        const dx_filtered = exponentialSmoothing(dx_alpha, dx, self.dx_prev);
        
        // Compute adaptive cutoff frequency based on velocity
        const cutoff = self.min_cutoff + self.beta * @abs(dx_filtered);
        
        // Apply adaptive smoothing to main signal
        const alpha = smoothingFactor(cutoff, dt);
        const filtered = exponentialSmoothing(alpha, measurement, self.x_prev.?);
        
        // Update state
        self.x_prev = filtered;
        self.dx_prev = dx_filtered;
        
        return filtered;
    }
    
    /// Get current filtered estimate without updating
    pub fn getCurrentEstimate(self: *const OneEuroFilter) ?f32 {
        return self.x_prev;
    }
    
    /// Reset filter state
    pub fn reset(self: *OneEuroFilter) void {
        self.x_prev = null;
        self.dx_prev = 0.0;
        self.t_prev = null;
    }
    
    /// Calculate smoothing factor for low-pass filter
    inline fn smoothingFactor(cutoff: f32, dt: f32) f32 {
        const r = 2.0 * std.math.pi * cutoff * dt;
        return r / (r + 1.0);
    }
    
    /// Apply exponential smoothing
    inline fn exponentialSmoothing(alpha: f32, current: f32, previous: f32) f32 {
        return alpha * current + (1.0 - alpha) * previous;
    }
};

/// Enhanced Task Predictor using One Euro Filter for adaptive execution time prediction
pub const TaskPredictor = struct {
    history: std.AutoHashMap(u64, TaskProfile),
    allocator: std.mem.Allocator,
    config: *const core.Config,
    
    /// Per-task execution profile with adaptive filtering
    const TaskProfile = struct {
        // One Euro Filter for execution time prediction
        execution_filter: OneEuroFilter,
        
        // Statistical tracking
        execution_count: u64 = 0,
        total_cycles: u64 = 0,
        variance_sum: f64 = 0.0,       // For Welford's variance calculation
        last_seen: u64 = 0,            // Timestamp of last execution
        
        // Prediction accuracy tracking
        last_prediction: ?f32 = null,
        prediction_error_sum: f64 = 0.0,
        accuracy_count: u64 = 0,
        
        pub fn init() TaskProfile {
            return .{
                .execution_filter = OneEuroFilter.initDefault(),
            };
        }
        
        pub fn initWithConfig(config: *const core.Config) TaskProfile {
            return .{
                .execution_filter = OneEuroFilter.init(
                    config.prediction_min_cutoff,
                    config.prediction_beta,
                    config.prediction_d_cutoff
                ),
            };
        }
        
        /// Record a new execution and update the filter
        pub fn recordExecution(self: *TaskProfile, cycles: u64, timestamp_ns: u64) f32 {
            const cycles_f32 = @as(f32, @floatFromInt(cycles));
            
            // Update One Euro Filter with new measurement
            const filtered_estimate = self.execution_filter.filter(cycles_f32, timestamp_ns);
            
            // Track prediction accuracy if we had a previous prediction
            if (self.last_prediction) |prediction| {
                const prediction_error = @abs(prediction - cycles_f32);
                self.prediction_error_sum += prediction_error;
                self.accuracy_count += 1;
            }
            
            // Update statistics
            self.execution_count += 1;
            self.total_cycles += cycles;
            self.last_seen = timestamp_ns;
            
            // Update variance using Welford's algorithm
            if (self.execution_count > 1) {
                const mean = @as(f64, @floatFromInt(self.total_cycles)) / @as(f64, @floatFromInt(self.execution_count));
                const delta = @as(f64, @floatFromInt(cycles)) - mean;
                self.variance_sum += delta * delta;
            }
            
            self.last_prediction = filtered_estimate;
            return filtered_estimate;
        }
        
        /// Get current execution time prediction
        pub fn getPrediction(self: *const TaskProfile) ?f32 {
            return self.execution_filter.getCurrentEstimate();
        }
        
        /// Get prediction confidence based on sample count and accuracy
        pub fn getConfidence(self: *const TaskProfile) f32 {
            if (self.execution_count == 0) return 0.0;
            
            // Sample size confidence (asymptotic to 1.0)
            const sample_confidence = @min(1.0, @as(f32, @floatFromInt(self.execution_count)) / 100.0);
            
            // Accuracy confidence (lower error = higher confidence)
            var accuracy_confidence: f32 = 1.0;
            if (self.accuracy_count > 0) {
                const avg_error = self.prediction_error_sum / @as(f64, @floatFromInt(self.accuracy_count));
                const avg_cycles = @as(f64, @floatFromInt(self.total_cycles)) / @as(f64, @floatFromInt(self.execution_count));
                const relative_error = avg_error / avg_cycles;
                accuracy_confidence = @max(0.0, 1.0 - @as(f32, @floatCast(relative_error)));
            }
            
            // Temporal relevance (recent samples weighted more heavily)
            const current_time = @as(u64, @intCast(std.time.nanoTimestamp()));
            const time_diff_ns = current_time - self.last_seen;
            const time_diff_s = @as(f64, @floatFromInt(time_diff_ns)) / 1_000_000_000.0;
            const temporal_confidence = @exp(-time_diff_s / 300.0); // 5-minute decay
            
            // Combined confidence
            return sample_confidence * accuracy_confidence * @as(f32, @floatCast(temporal_confidence));
        }
        
        /// Get variance of execution times
        pub fn getVariance(self: *const TaskProfile) f64 {
            if (self.execution_count < 2) return 0.0;
            return self.variance_sum / @as(f64, @floatFromInt(self.execution_count - 1));
        }
    };
    
    pub fn init(allocator: std.mem.Allocator, config: *const core.Config) TaskPredictor {
        return .{
            .history = std.AutoHashMap(u64, TaskProfile).init(allocator),
            .allocator = allocator,
            .config = config,
        };
    }
    
    pub fn deinit(self: *TaskPredictor) void {
        self.history.deinit();
    }
    
    /// Record task execution with adaptive filtering
    pub fn recordExecution(self: *TaskPredictor, task_hash: u64, cycles: u64) !f32 {
        const timestamp_ns = @as(u64, @intCast(std.time.nanoTimestamp()));
        
        const result = try self.history.getOrPut(task_hash);
        if (!result.found_existing) {
            result.value_ptr.* = TaskProfile.initWithConfig(self.config);
        }
        
        return result.value_ptr.recordExecution(cycles, timestamp_ns);
    }
    
    /// Get prediction for a task type
    pub fn predict(self: *TaskPredictor, task_hash: u64) ?PredictedStats {
        if (self.history.get(task_hash)) |profile| {
            if (profile.execution_count > 0) {
                const prediction = profile.getPrediction() orelse return null;
                
                return PredictedStats{
                    .expected_cycles = @as(u64, @intFromFloat(prediction)),
                    .confidence = profile.getConfidence(),
                    .variance = profile.getVariance(),
                    .filtered_estimate = prediction,
                };
            }
        }
        return null;
    }
    
    /// Get detailed profile for a task type (for debugging/monitoring)
    pub fn getProfile(self: *TaskPredictor, task_hash: u64) ?*const TaskProfile {
        return self.history.getPtr(task_hash);
    }
    
    /// Clean up old/unused task profiles to prevent memory growth
    pub fn cleanup(self: *TaskPredictor, max_age_seconds: u64) void {
        const current_time = @as(u64, @intCast(std.time.nanoTimestamp()));
        const max_age_ns = max_age_seconds * 1_000_000_000;
        
        var iterator = self.history.iterator();
        while (iterator.next()) |entry| {
            const profile = entry.value_ptr;
            if (current_time - profile.last_seen > max_age_ns) {
                _ = self.history.remove(entry.key_ptr.*);
            }
        }
    }
    
    /// Enhanced prediction results with One Euro Filter data
    pub const PredictedStats = struct {
        expected_cycles: u64,           // Rounded prediction for scheduling decisions
        confidence: f32,                // Multi-factor confidence score (0.0-1.0)
        variance: f64,                  // Statistical variance of execution times
        filtered_estimate: f32,         // Raw One Euro Filter estimate
    };
};

// ============================================================================
// Performance Measurement
// ============================================================================

pub inline fn rdtsc() u64 {
    if (comptime builtin.cpu.arch == .x86_64) {
        var low: u32 = undefined;
        var high: u32 = undefined;
        asm volatile ("rdtsc" : [low] "={eax}" (low), [high] "={edx}" (high));
        return (@as(u64, high) << 32) | low;
    } else {
        return @as(u64, @intCast(std.time.nanoTimestamp()));
    }
}

// ============================================================================
// Tests
// ============================================================================

test "token accounting" {
    const config = core.Config{
        .promotion_threshold = 10,
        .min_work_cycles = 1000,
    };
    
    var tokens = TokenAccount.init(&config);
    
    // Simulate work
    tokens.update(5000, 100); // 50:1 ratio
    try std.testing.expect(tokens.shouldPromote());
    
    tokens.reset();
    tokens.update(500, 100); // 5:1 ratio
    try std.testing.expect(!tokens.shouldPromote());
}

test "one euro filter" {
    var filter = OneEuroFilter.initDefault();
    
    const base_time: u64 = 1000000000; // 1 second in nanoseconds
    
    // Test initial measurement
    const first = filter.filter(100.0, base_time);
    try std.testing.expectEqual(@as(f32, 100.0), first);
    
    // Test stable measurements (should smooth towards average)
    const second = filter.filter(110.0, base_time + 100_000_000); // +100ms
    try std.testing.expect(second > 100.0 and second < 110.0);
    
    // Test rapid change adaptation
    const third = filter.filter(200.0, base_time + 200_000_000); // +200ms, big jump
    try std.testing.expect(third > second); // Should adapt towards new value
    
    // Test current estimate
    const estimate = filter.getCurrentEstimate().?;
    try std.testing.expect(estimate > 100.0);
    
    // Test reset
    filter.reset();
    try std.testing.expect(filter.getCurrentEstimate() == null);
}

test "task predictor with one euro filter" {
    const allocator = std.testing.allocator;
    
    const config = core.Config{};
    var predictor = TaskPredictor.init(allocator, &config);
    defer predictor.deinit();
    
    const task_hash: u64 = 0x12345678;
    
    // Record some executions with realistic timing
    _ = try predictor.recordExecution(task_hash, 1000);
    std.time.sleep(1_000_000); // 1ms delay
    _ = try predictor.recordExecution(task_hash, 1100);
    std.time.sleep(1_000_000); // 1ms delay
    const filtered = try predictor.recordExecution(task_hash, 900);
    
    // Check that filter is working (should be smoothed)
    try std.testing.expect(filtered != 900.0); // Should be filtered, not raw value
    
    // Check prediction
    const prediction = predictor.predict(task_hash).?;
    try std.testing.expect(prediction.expected_cycles > 0);
    try std.testing.expect(prediction.confidence > 0.0);
    try std.testing.expect(prediction.confidence <= 1.0);
    try std.testing.expect(prediction.variance >= 0.0);
    
    // Test that confidence increases with more samples
    const initial_confidence = prediction.confidence;
    
    // Add more samples
    for (0..10) |_| {
        _ = try predictor.recordExecution(task_hash, 1000);
        std.time.sleep(100_000); // 0.1ms delay
    }
    
    const updated_prediction = predictor.predict(task_hash).?;
    try std.testing.expect(updated_prediction.confidence >= initial_confidence);
}