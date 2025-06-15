const std = @import("std");
const builtin = @import("builtin");
const core = @import("core.zig");

// Scheduling algorithms for Beat.zig
//
// This module provides intelligent scheduling algorithms including:
// - Token accounting system with work:overhead ratio tracking
// - Heartbeat scheduling for promotion decisions
// - Task prediction for performance optimization
// - Performance measurement utilities

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
    
    pub fn init(allocator: std.mem.Allocator, config: *const core.Config) !*Scheduler {
        const self = try allocator.create(Scheduler);
        self.* = .{
            .allocator = allocator,
            .config = config,
            .worker_tokens = try allocator.alloc(TokenAccount, config.num_workers orelse 1),
        };
        
        // Initialize token accounts
        for (self.worker_tokens) |*tokens| {
            tokens.* = TokenAccount.init(config);
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
        
        self.allocator.free(self.worker_tokens);
        self.allocator.destroy(self);
    }
    
    fn heartbeatLoop(self: *Scheduler) void {
        const interval_ns = @as(u64, self.config.heartbeat_interval_us) * 1000;
        
        while (self.running.load(.acquire)) {
            std.time.sleep(interval_ns);
            
            // Periodic promotion check
            for (self.worker_tokens) |*tokens| {
                if (tokens.shouldPromote()) {
                    // TODO: Trigger work promotion
                    tokens.reset();
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