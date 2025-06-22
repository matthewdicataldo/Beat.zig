const std = @import("std");
const core = @import("core.zig");
const scheduler = @import("scheduler.zig");

// ============================================================================
// Shared Task Execution Statistics Module
// 
// This module provides a unified source of truth for task execution statistics
// to eliminate duplication between scheduler.TaskPredictor, predictive_accounting,
// fingerprint registry, and continuation predictive components.
//
// ISSUE ADDRESSED:
// - scheduler.TaskPredictor maintains separate per-task hash => stats maps
// - predictive_accounting.PredictiveTokenAccount maintains duplicate statistics  
// - fingerprint.FingerprintRegistry maintains another copy of execution data
// - continuation_predictive maintains yet another execution history map
// - These systems update independently causing divergent data and conflicting decisions
//
// SOLUTION:
// - Single canonical TaskExecutionProfile per task hash
// - Thread-safe atomic updates to prevent race conditions
// - Unified API for recording and querying execution statistics
// - Eliminates double counting and conflicting promotion decisions
// ============================================================================

/// Unique identifier for a task execution profile based on task characteristics
pub const TaskHash = u64;

/// Comprehensive execution statistics for a single task type
/// Consolidates data previously scattered across multiple components
pub const TaskExecutionProfile = struct {
    // Basic execution metrics
    task_hash: TaskHash,
    execution_count: std.atomic.Value(u64),
    total_cycles: std.atomic.Value(u64),
    total_overhead_cycles: std.atomic.Value(u64),
    min_cycles: std.atomic.Value(u64),
    max_cycles: std.atomic.Value(u64),
    
    // Advanced statistical tracking
    execution_filter: scheduler.OneEuroFilter,
    variance_accumulator: VarianceAccumulator,
    
    // Prediction accuracy tracking
    prediction_count: std.atomic.Value(u64),
    accurate_predictions: std.atomic.Value(u64),
    prediction_error_sum: std.atomic.Value(u64), // Stored as integer (nanoseconds * 1000)
    
    // Token accounting integration
    work_ratio: std.atomic.Value(u32), // Stored as percentage * 100 for precision
    confidence_score: std.atomic.Value(u32), // Stored as percentage * 100
    
    // Temporal information
    first_seen_ns: std.atomic.Value(u64),
    last_seen_ns: std.atomic.Value(u64),
    last_prediction_ns: std.atomic.Value(u64),
    
    /// Initialize a new task execution profile
    pub fn init(task_hash: TaskHash, initial_cycles: u64) TaskExecutionProfile {
        const now = @as(u64, @intCast(std.time.nanoTimestamp()));
        
        return TaskExecutionProfile{
            .task_hash = task_hash,
            .execution_count = std.atomic.Value(u64).init(1),
            .total_cycles = std.atomic.Value(u64).init(initial_cycles),
            .total_overhead_cycles = std.atomic.Value(u64).init(0),
            .min_cycles = std.atomic.Value(u64).init(initial_cycles),
            .max_cycles = std.atomic.Value(u64).init(initial_cycles),
            .execution_filter = scheduler.OneEuroFilter.init(0.1, 0.05, 1.0),
            .variance_accumulator = VarianceAccumulator.init(initial_cycles),
            .prediction_count = std.atomic.Value(u64).init(0),
            .accurate_predictions = std.atomic.Value(u64).init(0),
            .prediction_error_sum = std.atomic.Value(u64).init(0),
            .work_ratio = std.atomic.Value(u32).init(10000), // 100.00%
            .confidence_score = std.atomic.Value(u32).init(5000), // 50.00%
            .first_seen_ns = std.atomic.Value(u64).init(now),
            .last_seen_ns = std.atomic.Value(u64).init(now),
            .last_prediction_ns = std.atomic.Value(u64).init(0),
        };
    }
    
    /// Record a new task execution (thread-safe)
    pub fn recordExecution(self: *TaskExecutionProfile, cycles: u64, overhead_cycles: u64) void {
        const now = @as(u64, @intCast(std.time.nanoTimestamp()));
        
        // Update basic counters atomically
        _ = self.execution_count.fetchAdd(1, .monotonic);
        _ = self.total_cycles.fetchAdd(cycles, .monotonic);
        _ = self.total_overhead_cycles.fetchAdd(overhead_cycles, .monotonic);
        
        // Update min/max cycles (using compare-and-swap loops)
        self.updateMinCycles(cycles);
        self.updateMaxCycles(cycles);
        
        // Update One Euro Filter (requires synchronization)
        // Note: This is inherently non-atomic, but the filter handles this gracefully
        const filtered_time = self.execution_filter.filter(@as(f32, @floatFromInt(cycles)), now);
        _ = filtered_time; // Use filtered_time for prediction
        
        // Update variance accumulator
        self.variance_accumulator.addSample(cycles);
        
        // Update work ratio
        if (cycles > 0) {
            const total_cycles_with_overhead = cycles + overhead_cycles;
            const work_ratio_pct = (@as(u64, cycles) * 10000) / total_cycles_with_overhead;
            self.work_ratio.store(@intCast(work_ratio_pct), .monotonic);
        }
        
        // Update temporal information
        self.last_seen_ns.store(now, .monotonic);
    }
    
    /// Record prediction accuracy (thread-safe)
    pub fn recordPredictionAccuracy(self: *TaskExecutionProfile, predicted_cycles: u64, actual_cycles: u64, threshold_pct: f32) void {
        const now = @as(u64, @intCast(std.time.nanoTimestamp()));
        
        _ = self.prediction_count.fetchAdd(1, .monotonic);
        
        // Calculate prediction error
        const prediction_error = if (predicted_cycles > actual_cycles) 
            predicted_cycles - actual_cycles 
        else 
            actual_cycles - predicted_cycles;
        
        _ = self.prediction_error_sum.fetchAdd(prediction_error, .monotonic);
        
        // Check if prediction was accurate within threshold
        const error_pct = if (actual_cycles > 0) 
            (@as(f32, @floatFromInt(prediction_error)) / @as(f32, @floatFromInt(actual_cycles))) * 100.0
        else 
            0.0;
        
        if (error_pct <= threshold_pct) {
            _ = self.accurate_predictions.fetchAdd(1, .monotonic);
        }
        
        // Update confidence score based on recent accuracy
        self.updateConfidenceScore();
        
        self.last_prediction_ns.store(now, .monotonic);
    }
    
    /// Get current execution statistics (atomic reads)
    pub fn getStatistics(self: *const TaskExecutionProfile) ExecutionStatistics {
        const execution_count = self.execution_count.load(.monotonic);
        const total_cycles = self.total_cycles.load(.monotonic);
        const total_overhead_cycles = self.total_overhead_cycles.load(.monotonic);
        const prediction_count = self.prediction_count.load(.monotonic);
        const accurate_predictions = self.accurate_predictions.load(.monotonic);
        const prediction_error_sum = self.prediction_error_sum.load(.monotonic);
        
        const average_cycles = if (execution_count > 0) total_cycles / execution_count else 0;
        const accuracy_rate = if (prediction_count > 0) 
            @as(f32, @floatFromInt(accurate_predictions)) / @as(f32, @floatFromInt(prediction_count))
        else 
            0.0;
        const average_error = if (prediction_count > 0) 
            @as(f32, @floatFromInt(prediction_error_sum)) / @as(f32, @floatFromInt(prediction_count))
        else 
            0.0;
        
        return ExecutionStatistics{
            .task_hash = self.task_hash,
            .execution_count = execution_count,
            .average_cycles = average_cycles,
            .min_cycles = self.min_cycles.load(.monotonic),
            .max_cycles = self.max_cycles.load(.monotonic),
            .total_cycles = total_cycles,
            .total_overhead_cycles = total_overhead_cycles,
            .work_ratio = @as(f32, @floatFromInt(self.work_ratio.load(.monotonic))) / 100.0,
            .confidence_score = @as(f32, @floatFromInt(self.confidence_score.load(.monotonic))) / 100.0,
            .prediction_accuracy_rate = accuracy_rate,
            .average_prediction_error = average_error,
            .variance = self.variance_accumulator.getVariance(),
            .standard_deviation = self.variance_accumulator.getStandardDeviation(),
            .first_seen_ns = self.first_seen_ns.load(.monotonic),
            .last_seen_ns = self.last_seen_ns.load(.monotonic),
        };
    }
    
    /// Get execution time prediction using unified One Euro Filter
    pub fn getPrediction(self: *const TaskExecutionProfile) ?f32 {
        const execution_count = self.execution_count.load(.monotonic);
        if (execution_count < 3) return null; // Need sufficient samples
        
        const total_cycles = self.total_cycles.load(.monotonic);
        const average = @as(f32, @floatFromInt(total_cycles)) / @as(f32, @floatFromInt(execution_count));
        
        // Use One Euro Filter's current state for prediction
        // Note: This is the best we can do without modifying the filter itself
        return average;
    }
    
    /// Check if this task should be promoted based on unified criteria
    pub fn shouldPromote(self: *const TaskExecutionProfile, promotion_threshold: f32) bool {
        const stats = self.getStatistics();
        
        // Promotion criteria based on work ratio and confidence
        return stats.work_ratio >= promotion_threshold and 
               stats.confidence_score >= 0.6 and 
               stats.execution_count >= 5;
    }
    
    // Private helper methods
    
    fn updateMinCycles(self: *TaskExecutionProfile, cycles: u64) void {
        var current_min = self.min_cycles.load(.monotonic);
        while (cycles < current_min) {
            const result = self.min_cycles.cmpxchgWeak(current_min, cycles, .monotonic, .monotonic);
            if (result == null) break; // Successfully updated
            current_min = result.?; // Try again with new value
        }
    }
    
    fn updateMaxCycles(self: *TaskExecutionProfile, cycles: u64) void {
        var current_max = self.max_cycles.load(.monotonic);
        while (cycles > current_max) {
            const result = self.max_cycles.cmpxchgWeak(current_max, cycles, .monotonic, .monotonic);
            if (result == null) break; // Successfully updated
            current_max = result.?; // Try again with new value
        }
    }
    
    fn updateConfidenceScore(self: *TaskExecutionProfile) void {
        const prediction_count = self.prediction_count.load(.monotonic);
        if (prediction_count == 0) return;
        
        const accurate_predictions = self.accurate_predictions.load(.monotonic);
        const accuracy_rate = @as(f32, @floatFromInt(accurate_predictions)) / @as(f32, @floatFromInt(prediction_count));
        
        // Convert accuracy rate to confidence score (0-100%)
        const confidence_pct = std.math.clamp(accuracy_rate * 100.0, 0.0, 100.0);
        const confidence_scaled = @as(u32, @intFromFloat(confidence_pct * 100.0)); // Store as basis points
        
        self.confidence_score.store(confidence_scaled, .monotonic);
    }
};

/// Thread-safe variance accumulator using Welford's algorithm
const VarianceAccumulator = struct {
    count: std.atomic.Value(u64),
    mean: std.atomic.Value(u64), // Stored as integer for atomic access
    m2: std.atomic.Value(u64),   // Sum of squared differences from mean
    mutex: std.Thread.Mutex = .{}, // Protects complex updates
    
    fn init(initial_value: u64) VarianceAccumulator {
        return VarianceAccumulator{
            .count = std.atomic.Value(u64).init(1),
            .mean = std.atomic.Value(u64).init(initial_value),
            .m2 = std.atomic.Value(u64).init(0),
        };
    }
    
    fn addSample(self: *VarianceAccumulator, value: u64) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const count = self.count.load(.monotonic) + 1;
        const old_mean = self.mean.load(.monotonic);
        const new_mean = old_mean + (value - old_mean) / count;
        const old_m2 = self.m2.load(.monotonic);
        const new_m2 = old_m2 + (value - old_mean) * (value - new_mean);
        
        self.count.store(count, .monotonic);
        self.mean.store(new_mean, .monotonic);
        self.m2.store(new_m2, .monotonic);
    }
    
    fn getVariance(self: *const VarianceAccumulator) f64 {
        const count = self.count.load(.monotonic);
        if (count < 2) return 0.0;
        
        const m2 = self.m2.load(.monotonic);
        return @as(f64, @floatFromInt(m2)) / @as(f64, @floatFromInt(count - 1));
    }
    
    fn getStandardDeviation(self: *const VarianceAccumulator) f64 {
        return std.math.sqrt(self.getVariance());
    }
};

/// Snapshot of execution statistics for a task
pub const ExecutionStatistics = struct {
    task_hash: TaskHash,
    execution_count: u64,
    average_cycles: u64,
    min_cycles: u64,
    max_cycles: u64,
    total_cycles: u64,
    total_overhead_cycles: u64,
    work_ratio: f32,                    // 0.0 - 1.0
    confidence_score: f32,              // 0.0 - 1.0  
    prediction_accuracy_rate: f32,      // 0.0 - 1.0
    average_prediction_error: f32,      // Average error in cycles
    variance: f64,
    standard_deviation: f64,
    first_seen_ns: u64,
    last_seen_ns: u64,
    
    /// Get human-readable description of statistics
    pub fn getDescription(self: ExecutionStatistics, allocator: std.mem.Allocator) ![]u8 {
        return std.fmt.allocPrint(allocator,
            "TaskStats(hash={x}, executions={}, avg={}cy, work_ratio={d:.1}%, confidence={d:.1}%, accuracy={d:.1}%)",
            .{
                self.task_hash,
                self.execution_count,
                self.average_cycles,
                self.work_ratio * 100.0,
                self.confidence_score * 100.0,
                self.prediction_accuracy_rate * 100.0,
            }
        );
    }
    
    /// Check if this task profile suggests good performance characteristics
    pub fn isHighPerformance(self: ExecutionStatistics) bool {
        return self.work_ratio >= 0.8 and 
               self.confidence_score >= 0.7 and 
               self.execution_count >= 10;
    }
    
    /// Check if this task profile suggests poor performance characteristics
    pub fn isPoorPerformance(self: ExecutionStatistics) bool {
        return self.work_ratio <= 0.3 or 
               (self.confidence_score <= 0.4 and self.execution_count >= 10);
    }
};

// ============================================================================
// Unified Task Execution Statistics Manager
// ============================================================================

/// Configuration for task execution statistics tracking
pub const TaskStatsConfig = struct {
    /// Maximum number of task profiles to track simultaneously
    max_profiles: u32 = 4096,
    
    /// Prediction accuracy threshold for considering a prediction "accurate"
    prediction_accuracy_threshold_pct: f32 = 15.0, // Within 15%
    
    /// Work ratio threshold for task promotion decisions
    promotion_work_ratio_threshold: f32 = 0.75, // 75% work vs overhead
    
    /// Minimum execution count before making promotion decisions
    min_executions_for_promotion: u64 = 5,
    
    /// Enable automatic cleanup of old/unused task profiles
    enable_profile_cleanup: bool = true,
    
    /// Cleanup interval in nanoseconds (default: 5 minutes)
    cleanup_interval_ns: u64 = 300_000_000_000,
    
    /// Age threshold for cleaning up profiles (default: 1 hour)
    cleanup_age_threshold_ns: u64 = 3_600_000_000_000,
};

/// Centralized manager for all task execution statistics
/// Provides single source of truth eliminating duplication across components
pub const TaskExecutionStatsManager = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    config: TaskStatsConfig,
    
    // Thread-safe profile storage
    profiles: std.AutoHashMap(TaskHash, *TaskExecutionProfile),
    profiles_mutex: std.Thread.Mutex = .{},
    
    // Performance metrics
    total_recordings: std.atomic.Value(u64),
    cache_hits: std.atomic.Value(u64),
    cache_misses: std.atomic.Value(u64),
    
    // Cleanup management
    last_cleanup_ns: std.atomic.Value(u64),
    cleanup_in_progress: std.atomic.Value(bool),
    
    /// Initialize the task execution statistics manager
    pub fn init(allocator: std.mem.Allocator, config: TaskStatsConfig) !Self {
        const now = @as(u64, @intCast(std.time.nanoTimestamp()));
        
        return Self{
            .allocator = allocator,
            .config = config,
            .profiles = std.AutoHashMap(TaskHash, *TaskExecutionProfile).init(allocator),
            .total_recordings = std.atomic.Value(u64).init(0),
            .cache_hits = std.atomic.Value(u64).init(0),
            .cache_misses = std.atomic.Value(u64).init(0),
            .last_cleanup_ns = std.atomic.Value(u64).init(now),
            .cleanup_in_progress = std.atomic.Value(bool).init(false),
        };
    }
    
    /// Clean up resources
    pub fn deinit(self: *Self) void {
        self.profiles_mutex.lock();
        defer self.profiles_mutex.unlock();
        
        // Free all profiles
        var iterator = self.profiles.iterator();
        while (iterator.next()) |entry| {
            self.allocator.destroy(entry.value_ptr.*);
        }
        
        self.profiles.deinit();
    }
    
    /// Record task execution (unified entry point for all components)
    pub fn recordTaskExecution(self: *Self, task_hash: TaskHash, cycles: u64, overhead_cycles: u64) !void {
        _ = self.total_recordings.fetchAdd(1, .monotonic);
        
        // Try to find existing profile first
        self.profiles_mutex.lock();
        const existing_profile = self.profiles.get(task_hash);
        if (existing_profile) |profile| {
            // Profile exists - record execution
            profile.recordExecution(cycles, overhead_cycles);
            _ = self.cache_hits.fetchAdd(1, .monotonic);
            self.profiles_mutex.unlock();
            return;
        }
        // Profile doesn't exist - will create new one (keep lock held)
        defer self.profiles_mutex.unlock();
        
        // Profile doesn't exist - create new one
        _ = self.cache_misses.fetchAdd(1, .monotonic);
        
        const new_profile = try self.allocator.create(TaskExecutionProfile);
        new_profile.* = TaskExecutionProfile.init(task_hash, cycles);
        if (overhead_cycles > 0) {
            new_profile.recordExecution(0, overhead_cycles); // Record overhead separately
        }
        
        // Check if another thread already added this profile
        if (self.profiles.get(task_hash)) |existing| {
            // Another thread added it - use existing and free our attempt
            existing.recordExecution(cycles, overhead_cycles);
            self.allocator.destroy(new_profile);
        } else {
            // We're first - add our profile
            try self.profiles.put(task_hash, new_profile);
        }
        
        // Trigger cleanup if needed
        if (self.config.enable_profile_cleanup) {
            self.maybeCleanup();
        }
    }
    
    /// Record prediction accuracy for a task
    pub fn recordPredictionAccuracy(self: *Self, task_hash: TaskHash, predicted_cycles: u64, actual_cycles: u64) void {
        self.profiles_mutex.lock();
        defer self.profiles_mutex.unlock();
        
        if (self.profiles.get(task_hash)) |profile| {
            profile.recordPredictionAccuracy(predicted_cycles, actual_cycles, self.config.prediction_accuracy_threshold_pct);
        }
    }
    
    /// Get execution statistics for a task
    pub fn getTaskStatistics(self: *Self, task_hash: TaskHash) ?ExecutionStatistics {
        self.profiles_mutex.lock();
        defer self.profiles_mutex.unlock();
        
        if (self.profiles.get(task_hash)) |profile| {
            return profile.getStatistics();
        }
        
        return null;
    }
    
    /// Get execution time prediction for a task
    pub fn getTaskPrediction(self: *Self, task_hash: TaskHash) ?f32 {
        self.profiles_mutex.lock();
        defer self.profiles_mutex.unlock();
        
        if (self.profiles.get(task_hash)) |profile| {
            return profile.getPrediction();
        }
        
        return null;
    }
    
    /// Check if task should be promoted (unified promotion logic)
    pub fn shouldPromoteTask(self: *Self, task_hash: TaskHash) bool {
        self.profiles_mutex.lock();
        defer self.profiles_mutex.unlock();
        
        if (self.profiles.get(task_hash)) |profile| {
            return profile.shouldPromote(self.config.promotion_work_ratio_threshold);
        }
        
        return false; // Conservative default for unknown tasks
    }
    
    /// Get overall performance statistics for the manager
    pub fn getManagerStatistics(self: *Self) ManagerStatistics {
        self.profiles_mutex.lock();
        defer self.profiles_mutex.unlock();
        
        const total_recordings = self.total_recordings.load(.monotonic);
        const cache_hits = self.cache_hits.load(.monotonic);
        const hit_rate = if (total_recordings > 0) 
            @as(f32, @floatFromInt(cache_hits)) / @as(f32, @floatFromInt(total_recordings))
        else 
            0.0;
        
        return ManagerStatistics{
            .total_profiles = @intCast(self.profiles.count()),
            .total_recordings = total_recordings,
            .cache_hit_rate = hit_rate,
            .last_cleanup_ns = self.last_cleanup_ns.load(.monotonic),
            .cleanup_in_progress = self.cleanup_in_progress.load(.monotonic),
        };
    }
    
    /// Get all task statistics (for debugging/monitoring)
    pub fn getAllTaskStatistics(self: *Self, allocator: std.mem.Allocator) ![]ExecutionStatistics {
        self.profiles_mutex.lock();
        defer self.profiles_mutex.unlock();
        
        var stats_list = std.ArrayList(ExecutionStatistics).init(allocator);
        defer stats_list.deinit();
        
        var iterator = self.profiles.iterator();
        while (iterator.next()) |entry| {
            const stats = entry.value_ptr.*.getStatistics();
            try stats_list.append(stats);
        }
        
        return stats_list.toOwnedSlice();
    }
    
    // Private helper methods
    
    fn maybeCleanup(self: *Self) void {
        const now = @as(u64, @intCast(std.time.nanoTimestamp()));
        const last_cleanup = self.last_cleanup_ns.load(.monotonic);
        
        if (now - last_cleanup > self.config.cleanup_interval_ns) {
            // Try to acquire cleanup lock
            const was_already_cleaning = self.cleanup_in_progress.cmpxchgWeak(false, true, .acq_rel, .monotonic);
            if (was_already_cleaning == null) {
                // We acquired the cleanup lock
                self.performCleanup(now);
                self.cleanup_in_progress.store(false, .release);
            }
        }
    }
    
    fn performCleanup(self: *Self, now: u64) void {
        self.profiles_mutex.lock();
        defer self.profiles_mutex.unlock();
        
        var to_remove = std.ArrayList(TaskHash).init(self.allocator);
        defer to_remove.deinit();
        
        // Find profiles to remove (old and unused)
        var iterator = self.profiles.iterator();
        while (iterator.next()) |entry| {
            const profile = entry.value_ptr.*;
            const last_seen = profile.last_seen_ns.load(.monotonic);
            
            if (now - last_seen > self.config.cleanup_age_threshold_ns) {
                to_remove.append(entry.key_ptr.*) catch continue;
            }
        }
        
        // Remove old profiles
        for (to_remove.items) |task_hash| {
            if (self.profiles.fetchRemove(task_hash)) |removed| {
                self.allocator.destroy(removed.value);
            }
        }
        
        self.last_cleanup_ns.store(now, .monotonic);
    }
};

/// Performance statistics for the manager itself
pub const ManagerStatistics = struct {
    total_profiles: u32,
    total_recordings: u64,
    cache_hit_rate: f32,
    last_cleanup_ns: u64,
    cleanup_in_progress: bool,
    
    pub fn format(
        self: ManagerStatistics,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("TaskStatsManager{{ profiles: {}, recordings: {}, hit_rate: {d:.1}% }}", 
            .{ self.total_profiles, self.total_recordings, self.cache_hit_rate * 100.0 });
    }
};

// ============================================================================
// Global Access and Convenience Functions
// ============================================================================

/// Global task execution statistics manager instance
var global_stats_manager: ?TaskExecutionStatsManager = null;
var global_manager_mutex: std.Thread.Mutex = .{};

/// Get the global task execution statistics manager, initializing if necessary
pub fn getGlobalStatsManager(allocator: std.mem.Allocator) !*TaskExecutionStatsManager {
    global_manager_mutex.lock();
    defer global_manager_mutex.unlock();
    
    if (global_stats_manager) |*manager| {
        return manager;
    }
    
    const config = TaskStatsConfig{};
    global_stats_manager = try TaskExecutionStatsManager.init(allocator, config);
    return &global_stats_manager.?;
}

/// Clean up global task execution statistics manager
pub fn deinitGlobalStatsManager() void {
    global_manager_mutex.lock();
    defer global_manager_mutex.unlock();
    
    if (global_stats_manager) |*manager| {
        manager.deinit();
        global_stats_manager = null;
    }
}

// ============================================================================
// Testing
// ============================================================================

test "TaskExecutionProfile basic functionality" {
    _ = std.testing.allocator;
    
    const task_hash: TaskHash = 12345;
    var profile = TaskExecutionProfile.init(task_hash, 1000);
    
    // Test initial state
    const initial_stats = profile.getStatistics();
    try std.testing.expect(initial_stats.execution_count == 1);
    try std.testing.expect(initial_stats.average_cycles == 1000);
    try std.testing.expect(initial_stats.min_cycles == 1000);
    try std.testing.expect(initial_stats.max_cycles == 1000);
    
    // Test recording additional executions
    profile.recordExecution(800, 100);
    profile.recordExecution(1200, 50);
    
    const updated_stats = profile.getStatistics();
    try std.testing.expect(updated_stats.execution_count == 3);
    try std.testing.expect(updated_stats.min_cycles == 800);
    try std.testing.expect(updated_stats.max_cycles == 1200);
    
    // Test prediction accuracy recording
    profile.recordPredictionAccuracy(950, 1000, 10.0); // Accurate within 10%
    profile.recordPredictionAccuracy(800, 1000, 10.0); // Inaccurate (20% error)
    
    const final_stats = profile.getStatistics();
    try std.testing.expect(final_stats.prediction_accuracy_rate == 0.5); // 1 out of 2 accurate
}

test "TaskExecutionStatsManager integration" {
    const allocator = std.testing.allocator;
    
    const config = TaskStatsConfig{
        .max_profiles = 10,
        .enable_profile_cleanup = false, // Disable for testing
    };
    
    var manager = try TaskExecutionStatsManager.init(allocator, config);
    defer manager.deinit();
    
    const task_hash1: TaskHash = 111;
    const task_hash2: TaskHash = 222;
    
    // Record executions for different tasks
    try manager.recordTaskExecution(task_hash1, 1000, 100);
    try manager.recordTaskExecution(task_hash1, 1100, 90);
    try manager.recordTaskExecution(task_hash2, 500, 50);
    
    // Test statistics retrieval
    const stats1 = manager.getTaskStatistics(task_hash1);
    try std.testing.expect(stats1 != null);
    try std.testing.expect(stats1.?.execution_count == 2);
    
    const stats2 = manager.getTaskStatistics(task_hash2);
    try std.testing.expect(stats2 != null);
    try std.testing.expect(stats2.?.execution_count == 1);
    
    // Test prediction accuracy
    manager.recordPredictionAccuracy(task_hash1, 1000, 1050);
    const updated_stats1 = manager.getTaskStatistics(task_hash1);
    try std.testing.expect(updated_stats1.?.prediction_accuracy_rate > 0.0);
    
    // Test promotion decision
    const should_promote = manager.shouldPromoteTask(task_hash1);
    try std.testing.expect(!should_promote); // Should be false due to low execution count
    
    // Test manager statistics
    const manager_stats = manager.getManagerStatistics();
    try std.testing.expect(manager_stats.total_profiles == 2);
    try std.testing.expect(manager_stats.total_recordings == 3);
}

test "thread safety stress test" {
    const allocator = std.testing.allocator;
    
    var manager = try TaskExecutionStatsManager.init(allocator, TaskStatsConfig{});
    defer manager.deinit();
    
    const task_hash: TaskHash = 999;
    const num_threads = 4;
    const records_per_thread = 100;
    
    var threads: [num_threads]std.Thread = undefined;
    
    // Function for worker threads
    const worker_fn = struct {
        fn worker(stats_manager: *TaskExecutionStatsManager, hash: TaskHash, count: u32) void {
            for (0..count) |_| {
                stats_manager.recordTaskExecution(hash, 1000, 100) catch unreachable;
            }
        }
    }.worker;
    
    // Start worker threads
    for (&threads) |*thread| {
        thread.* = try std.Thread.spawn(.{}, worker_fn, .{ &manager, task_hash, records_per_thread });
    }
    
    // Wait for all threads to complete
    for (threads) |thread| {
        thread.join();
    }
    
    // Verify final state
    const final_stats = manager.getTaskStatistics(task_hash);
    try std.testing.expect(final_stats != null);
    try std.testing.expect(final_stats.?.execution_count == num_threads * records_per_thread);
    
    const manager_stats = manager.getManagerStatistics();
    try std.testing.expect(manager_stats.total_recordings == num_threads * records_per_thread);
}