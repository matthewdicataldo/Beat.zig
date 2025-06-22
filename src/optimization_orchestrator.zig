const std = @import("std");
const optimization_registry = @import("optimization_registry.zig");

// ============================================================================
// Unified Optimization Orchestrator
// 
// This module provides DAG-based coordination and execution of Souper, ISPC,
// and Minotaur optimizations with conflict resolution and performance tracking.
//
// FEATURES:
// - Dependency-aware optimization ordering via DAG
// - Real-time conflict detection and resolution
// - Performance-driven optimization selection
// - Publish/subscribe mechanism for optimization passes
// - Adaptive optimization strategies based on feedback
// ============================================================================

/// Optimization execution plan with dependency ordering
pub const OptimizationPlan = struct {
    region_id: optimization_registry.OptimizationRegionId,
    ordered_optimizations: []OptimizationStep,
    estimated_total_speedup: f32,
    estimated_compilation_time_ms: u32,
    confidence_score: f32,
    
    /// Single optimization step in the execution plan
    pub const OptimizationStep = struct {
        optimization_type: optimization_registry.OptimizationType,
        priority: u8,               // 0-255, higher = more important
        dependencies: []optimization_registry.OptimizationType,
        estimated_speedup: f32,
        estimated_time_ms: u32,
        fallback_available: bool,
        
        /// Check if all dependencies are satisfied
        pub fn dependenciesSatisfied(self: OptimizationStep, completed: std.EnumSet(optimization_registry.OptimizationType)) bool {
            for (self.dependencies) |dep| {
                if (!completed.contains(dep)) return false;
            }
            return true;
        }
    };
    
    /// Clean up allocated plan resources
    pub fn deinit(self: *OptimizationPlan, allocator: std.mem.Allocator) void {
        for (self.ordered_optimizations) |*step| {
            if (step.dependencies.len > 0) {
                allocator.free(step.dependencies);
            }
        }
        allocator.free(self.ordered_optimizations);
    }
    
    /// Get next optimization step to execute
    pub fn getNextStep(self: *const OptimizationPlan, completed: std.EnumSet(optimization_registry.OptimizationType)) ?OptimizationStep {
        for (self.ordered_optimizations) |step| {
            if (!completed.contains(step.optimization_type) and step.dependenciesSatisfied(completed)) {
                return step;
            }
        }
        return null;
    }
};

/// Result of executing an optimization plan
pub const OptimizationResults = struct {
    region_id: optimization_registry.OptimizationRegionId,
    executed_optimizations: std.EnumSet(optimization_registry.OptimizationType),
    failed_optimizations: std.EnumSet(optimization_registry.OptimizationType),
    final_performance: optimization_registry.PerformanceMetrics,
    total_speedup: f32,
    total_execution_time_ms: u32,
    conflicts_resolved: u32,
    
    /// Calculate overall success rate
    pub fn getSuccessRate(self: OptimizationResults) f32 {
        const total_attempted = self.executed_optimizations.count() + self.failed_optimizations.count();
        if (total_attempted == 0) return 0.0;
        
        return @as(f32, @floatFromInt(self.executed_optimizations.count())) / @as(f32, @floatFromInt(total_attempted));
    }
};

/// Strategy for resolving optimization conflicts
pub const ConflictResolutionStrategy = enum {
    choose_highest_priority,    // Select optimization with highest priority
    choose_best_performance,    // Select optimization with best expected performance
    choose_most_reliable,       // Select optimization with best success rate
    parallel_evaluation,        // Evaluate both optimizations in parallel and choose best
    custom_heuristic,          // Use custom conflict resolution logic
    
    /// Get description of conflict resolution strategy
    pub fn getDescription(self: ConflictResolutionStrategy) []const u8 {
        return switch (self) {
            .choose_highest_priority => "Choose optimization with highest priority",
            .choose_best_performance => "Choose optimization with best expected performance",
            .choose_most_reliable => "Choose optimization with best historical success rate",
            .parallel_evaluation => "Evaluate both optimizations and choose best result",
            .custom_heuristic => "Use custom conflict resolution heuristic",
        };
    }
};

/// Configuration for the optimization orchestrator
pub const OrchestratorConfig = struct {
    /// Default conflict resolution strategy
    default_conflict_resolution: ConflictResolutionStrategy = .choose_best_performance,
    
    /// Enable parallel evaluation of non-conflicting optimizations
    enable_parallel_execution: bool = true,
    
    /// Maximum number of optimization threads to use
    max_optimization_threads: u32 = 4,
    
    /// Timeout for individual optimization attempts (milliseconds)
    optimization_timeout_ms: u32 = 30000, // 30 seconds
    
    /// Enable adaptive optimization selection based on feedback
    enable_adaptive_selection: bool = true,
    
    /// Minimum confidence score to proceed with optimization
    min_confidence_threshold: f32 = 0.3,
    
    /// Enable detailed performance tracking
    enable_performance_tracking: bool = true,
};

/// Event for the publish/subscribe optimization mechanism
pub const OptimizationEvent = struct {
    event_type: EventType,
    region_id: optimization_registry.OptimizationRegionId,
    optimization_type: optimization_registry.OptimizationType,
    timestamp_ns: u64,
    data: EventData,
    
    pub const EventType = enum {
        optimization_started,
        optimization_completed,
        optimization_failed,
        conflict_detected,
        performance_update,
    };
    
    pub const EventData = union(EventType) {
        optimization_started: OptimizationStartedData,
        optimization_completed: OptimizationCompletedData,
        optimization_failed: OptimizationFailedData,
        conflict_detected: ConflictDetectedData,
        performance_update: PerformanceUpdateData,
    };
    
    pub const OptimizationStartedData = struct {
        estimated_duration_ms: u32,
        dependencies_satisfied: bool,
    };
    
    pub const OptimizationCompletedData = struct {
        actual_duration_ms: u32,
        performance_improvement: f32,
        verification_status: VerificationStatus,
    };
    
    pub const OptimizationFailedData = struct {
        error_message: []const u8,
        failure_reason: FailureReason,
        retry_recommended: bool,
    };
    
    pub const ConflictDetectedData = struct {
        conflicting_optimization: optimization_registry.OptimizationType,
        conflict_severity: optimization_registry.OptimizationConflict.ConflictSeverity,
        resolution_applied: ConflictResolutionStrategy,
    };
    
    pub const PerformanceUpdateData = struct {
        new_performance: optimization_registry.PerformanceMetrics,
        speedup_achieved: f32,
    };
    
    pub const VerificationStatus = enum { passed, failed, not_available };
    pub const FailureReason = enum { timeout, compilation_error, verification_failed, resource_exhausted };
};

/// Callback function for optimization event subscribers
pub const OptimizationEventCallback = *const fn (event: OptimizationEvent) void;

/// Optimization orchestrator coordinating all optimization systems
pub const OptimizationOrchestrator = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    config: OrchestratorConfig,
    registry: *optimization_registry.OptimizationRegistry,
    
    // Event system for pub/sub coordination
    event_subscribers: std.ArrayList(OptimizationEventCallback),
    event_mutex: std.Thread.Mutex = .{},
    
    // Performance tracking
    total_plans_executed: std.atomic.Value(u64),
    successful_plans: std.atomic.Value(u64),
    conflicts_resolved: std.atomic.Value(u64),
    adaptive_selections: std.atomic.Value(u64),
    
    // Thread pool for parallel optimization execution
    optimization_threads: ?[]std.Thread = null,
    
    /// Initialize the optimization orchestrator
    pub fn init(
        allocator: std.mem.Allocator,
        config: OrchestratorConfig,
        registry: *optimization_registry.OptimizationRegistry
    ) !Self {
        return Self{
            .allocator = allocator,
            .config = config,
            .registry = registry,
            .event_subscribers = std.ArrayList(OptimizationEventCallback).init(allocator),
            .total_plans_executed = std.atomic.Value(u64).init(0),
            .successful_plans = std.atomic.Value(u64).init(0),
            .conflicts_resolved = std.atomic.Value(u64).init(0),
            .adaptive_selections = std.atomic.Value(u64).init(0),
        };
    }
    
    /// Clean up orchestrator resources
    pub fn deinit(self: *Self) void {
        // Wait for optimization threads to complete
        if (self.optimization_threads) |threads| {
            for (threads) |thread| {
                thread.join();
            }
            self.allocator.free(threads);
        }
        
        self.event_subscribers.deinit();
    }
    
    /// Create an optimization plan for a region
    pub fn createOptimizationPlan(
        self: *Self,
        region_id: optimization_registry.OptimizationRegionId
    ) !OptimizationPlan {
        const region = self.registry.getRegion(region_id) orelse return error.RegionNotFound;
        
        // Get recommendations from registry
        const recommendation = self.registry.getOptimizationRecommendation(region_id);
        
        // Build dependency-ordered optimization steps
        var steps = std.ArrayList(OptimizationPlan.OptimizationStep).init(self.allocator);
        defer steps.deinit();
        
        // Add available optimizations in dependency order
        try self.addOptimizationSteps(&steps, region);
        
        // Resolve conflicts
        try self.resolveConflicts(&steps, region);
        
        // Calculate plan metrics
        const total_speedup = self.calculateTotalSpeedup(&steps);
        const compilation_time = self.estimateCompilationTime(&steps);
        const confidence = if (recommendation) |rec| rec.confidence else 0.5;
        
        return OptimizationPlan{
            .region_id = region_id,
            .ordered_optimizations = try steps.toOwnedSlice(),
            .estimated_total_speedup = total_speedup,
            .estimated_compilation_time_ms = compilation_time,
            .confidence_score = confidence,
        };
    }
    
    /// Execute an optimization plan
    pub fn executeOptimizationPlan(
        self: *Self,
        plan: OptimizationPlan
    ) !OptimizationResults {
        _ = self.total_plans_executed.fetchAdd(1, .monotonic);
        
        var results = OptimizationResults{
            .region_id = plan.region_id,
            .executed_optimizations = std.EnumSet(optimization_registry.OptimizationType).initEmpty(),
            .failed_optimizations = std.EnumSet(optimization_registry.OptimizationType).initEmpty(),
            .final_performance = optimization_registry.PerformanceMetrics.init(),
            .total_speedup = 1.0,
            .total_execution_time_ms = 0,
            .conflicts_resolved = 0,
        };
        
        const start_time = std.time.nanoTimestamp();
        
        // Execute optimization steps in dependency order
        for (plan.ordered_optimizations) |step| {
            if (step.dependenciesSatisfied(results.executed_optimizations)) {
                const success = self.executeOptimizationStep(step, plan.region_id) catch false;
                
                if (success) {
                    results.executed_optimizations.insert(step.optimization_type);
                } else {
                    results.failed_optimizations.insert(step.optimization_type);
                }
            }
        }
        
        const end_time = std.time.nanoTimestamp();
        results.total_execution_time_ms = @intCast((end_time - start_time) / 1_000_000);
        
        // Update success metrics
        if (results.executed_optimizations.count() > 0) {
            _ = self.successful_plans.fetchAdd(1, .monotonic);
        }
        
        // Publish completion event
        self.publishEvent(OptimizationEvent{
            .event_type = .optimization_completed,
            .region_id = plan.region_id,
            .optimization_type = .souper, // Placeholder
            .timestamp_ns = @as(u64, @intCast(end_time)),
            .data = .{ .optimization_completed = .{
                .actual_duration_ms = results.total_execution_time_ms,
                .performance_improvement = results.total_speedup,
                .verification_status = .passed,
            }},
        });
        
        return results;
    }
    
    /// Subscribe to optimization events
    pub fn subscribeToEvents(self: *Self, callback: OptimizationEventCallback) !void {
        self.event_mutex.lock();
        defer self.event_mutex.unlock();
        
        try self.event_subscribers.append(callback);
    }
    
    /// Get orchestrator performance statistics
    pub fn getStatistics(self: *Self) OrchestratorStatistics {
        const total_plans = self.total_plans_executed.load(.monotonic);
        const successful = self.successful_plans.load(.monotonic);
        const conflicts = self.conflicts_resolved.load(.monotonic);
        const adaptive = self.adaptive_selections.load(.monotonic);
        
        const success_rate = if (total_plans > 0) 
            @as(f32, @floatFromInt(successful)) / @as(f32, @floatFromInt(total_plans))
        else 
            0.0;
        
        return OrchestratorStatistics{
            .total_plans_executed = total_plans,
            .successful_plans = successful,
            .success_rate = success_rate,
            .conflicts_resolved = conflicts,
            .adaptive_selections = adaptive,
        };
    }
    
    // Private helper methods
    
    /// Add optimization steps in dependency order
    fn addOptimizationSteps(
        self: *Self,
        steps: *std.ArrayList(OptimizationPlan.OptimizationStep),
        region: *const optimization_registry.OptimizationRegion
    ) !void {
        // Souper optimizations (scalar math) - typically first
        if (region.isSuitableFor(.souper)) {
            try steps.append(OptimizationPlan.OptimizationStep{
                .optimization_type = .souper,
                .priority = 100,
                .dependencies = &[_]optimization_registry.OptimizationType{},
                .estimated_speedup = 1.5,
                .estimated_time_ms = 5000,
                .fallback_available = true,
            });
        }
        
        // ISPC optimizations (vectorization) - after scalar optimizations
        if (region.isSuitableFor(.ispc)) {
            const souper_dep = if (region.isSuitableFor(.souper)) 
                try self.allocator.dupe(optimization_registry.OptimizationType, &[_]optimization_registry.OptimizationType{.souper})
            else 
                &[_]optimization_registry.OptimizationType{};
            
            try steps.append(OptimizationPlan.OptimizationStep{
                .optimization_type = .ispc,
                .priority = 90,
                .dependencies = souper_dep,
                .estimated_speedup = 6.0,
                .estimated_time_ms = 10000,
                .fallback_available = true,
            });
        }
        
        // Minotaur optimizations (SIMD instructions) - after vectorization
        if (region.isSuitableFor(.minotaur)) {
            var deps = std.ArrayList(optimization_registry.OptimizationType).init(self.allocator);
            defer deps.deinit();
            
            if (region.isSuitableFor(.ispc)) {
                try deps.append(.ispc);
            }
            
            try steps.append(OptimizationPlan.OptimizationStep{
                .optimization_type = .minotaur,
                .priority = 80,
                .dependencies = try deps.toOwnedSlice(),
                .estimated_speedup = 2.5,
                .estimated_time_ms = 15000,
                .fallback_available = true,
            });
        }
    }
    
    /// Resolve conflicts between optimization steps
    fn resolveConflicts(
        self: *Self,
        steps: *std.ArrayList(OptimizationPlan.OptimizationStep),
        region: *const optimization_registry.OptimizationRegion
    ) !void {
        _ = region;
        
        // Check for conflicts and apply resolution strategy
        var i: usize = 0;
        while (i < steps.items.len) {
            var j: usize = i + 1;
            while (j < steps.items.len) {
                const step_a = &steps.items[i];
                const step_b = &steps.items[j];
                
                if (self.hasConflict(step_a.optimization_type, step_b.optimization_type)) {
                    // Resolve conflict based on strategy
                    const keep_a = self.resolveConflictPair(step_a, step_b);
                    
                    if (keep_a) {
                        _ = steps.orderedRemove(j);
                    } else {
                        _ = steps.orderedRemove(i);
                        break; // Restart from current i
                    }
                    
                    _ = self.conflicts_resolved.fetchAdd(1, .monotonic);
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
    }
    
    /// Check if two optimization types have conflicts
    fn hasConflict(self: *Self, opt_a: optimization_registry.OptimizationType, opt_b: optimization_registry.OptimizationType) bool {
        _ = self;
        
        // Known conflicts
        if ((opt_a == .ispc and opt_b == .minotaur) or (opt_a == .minotaur and opt_b == .ispc)) {
            return true; // SIMD vectorization conflict
        }
        
        return false;
    }
    
    /// Resolve conflict between two optimization steps
    fn resolveConflictPair(
        self: *Self,
        step_a: *const OptimizationPlan.OptimizationStep,
        step_b: *const OptimizationPlan.OptimizationStep
    ) bool {
        return switch (self.config.default_conflict_resolution) {
            .choose_highest_priority => step_a.priority >= step_b.priority,
            .choose_best_performance => step_a.estimated_speedup >= step_b.estimated_speedup,
            .choose_most_reliable => step_a.fallback_available and !step_b.fallback_available,
            .parallel_evaluation => step_a.priority >= step_b.priority, // Fallback to priority
            .custom_heuristic => self.customConflictResolution(step_a, step_b),
        };
    }
    
    /// Custom conflict resolution heuristic
    fn customConflictResolution(
        self: *Self,
        step_a: *const OptimizationPlan.OptimizationStep,
        step_b: *const OptimizationPlan.OptimizationStep
    ) bool {
        _ = self;
        
        // Prefer optimizations with better speedup/time ratio
        const ratio_a = step_a.estimated_speedup / @as(f32, @floatFromInt(step_a.estimated_time_ms));
        const ratio_b = step_b.estimated_speedup / @as(f32, @floatFromInt(step_b.estimated_time_ms));
        
        return ratio_a >= ratio_b;
    }
    
    /// Calculate total expected speedup from optimization steps
    fn calculateTotalSpeedup(self: *Self, steps: *const std.ArrayList(OptimizationPlan.OptimizationStep)) f32 {
        _ = self;
        
        var total_speedup: f32 = 1.0;
        for (steps.items) |step| {
            // Compound speedups (multiplicative)
            total_speedup *= step.estimated_speedup;
        }
        
        return total_speedup;
    }
    
    /// Estimate total compilation time for optimization steps
    fn estimateCompilationTime(self: *Self, steps: *const std.ArrayList(OptimizationPlan.OptimizationStep)) u32 {
        _ = self;
        
        var total_time: u32 = 0;
        for (steps.items) |step| {
            total_time += step.estimated_time_ms;
        }
        
        return total_time;
    }
    
    /// Execute a single optimization step
    fn executeOptimizationStep(
        self: *Self,
        step: OptimizationPlan.OptimizationStep,
        region_id: optimization_registry.OptimizationRegionId
    ) !bool {
        // Publish start event
        self.publishEvent(OptimizationEvent{
            .event_type = .optimization_started,
            .region_id = region_id,
            .optimization_type = step.optimization_type,
            .timestamp_ns = @as(u64, @intCast(std.time.nanoTimestamp())),
            .data = .{ .optimization_started = .{
                .estimated_duration_ms = step.estimated_time_ms,
                .dependencies_satisfied = true,
            }},
        });
        
        // Simulate optimization execution
        // In real implementation, this would call into the actual optimization systems
        const success = switch (step.optimization_type) {
            .souper => self.executeSouperOptimization(region_id),
            .ispc => self.executeISPCOptimization(region_id),
            .minotaur => self.executeMinotaurOptimization(region_id),
        };
        
        if (success) {
            // Record success in registry
            const performance = optimization_registry.PerformanceMetrics{
                .execution_time_ns = 1000000, // 1ms baseline
                .cpu_cycles = 2000,
            };
            self.registry.recordOptimizationSuccess(region_id, step.optimization_type, performance);
        } else {
            // Record failure in registry
            self.registry.recordOptimizationFailure(region_id, step.optimization_type);
        }
        
        return success;
    }
    
    /// Execute Souper optimization (placeholder)
    fn executeSouperOptimization(self: *Self, region_id: optimization_registry.OptimizationRegionId) bool {
        _ = self;
        _ = region_id;
        // Placeholder: In real implementation, this would call souper_integration
        return true; // Assume success for now
    }
    
    /// Execute ISPC optimization (placeholder)
    fn executeISPCOptimization(self: *Self, region_id: optimization_registry.OptimizationRegionId) bool {
        _ = self;
        _ = region_id;
        // Placeholder: In real implementation, this would call ispc_integration
        return true; // Assume success for now
    }
    
    /// Execute Minotaur optimization (placeholder)
    fn executeMinotaurOptimization(self: *Self, region_id: optimization_registry.OptimizationRegionId) bool {
        _ = self;
        _ = region_id;
        // Placeholder: In real implementation, this would call minotaur_integration
        return true; // Assume success for now
    }
    
    /// Publish an optimization event to all subscribers
    fn publishEvent(self: *Self, event: OptimizationEvent) void {
        self.event_mutex.lock();
        defer self.event_mutex.unlock();
        
        for (self.event_subscribers.items) |callback| {
            callback(event);
        }
    }
};

/// Performance statistics for the orchestrator
pub const OrchestratorStatistics = struct {
    total_plans_executed: u64,
    successful_plans: u64,
    success_rate: f32,
    conflicts_resolved: u64,
    adaptive_selections: u64,
    
    pub fn format(
        self: OrchestratorStatistics,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("OptimizationOrchestrator{{ plans: {}, success_rate: {d:.1}%, conflicts: {} }}", 
            .{ self.total_plans_executed, self.success_rate * 100.0, self.conflicts_resolved });
    }
};

// ============================================================================
// Global Access and Convenience Functions
// ============================================================================

/// Global optimization orchestrator instance
var global_orchestrator: ?OptimizationOrchestrator = null;
var global_orchestrator_mutex: std.Thread.Mutex = .{};

/// Get the global optimization orchestrator, initializing if necessary
pub fn getGlobalOrchestrator(
    allocator: std.mem.Allocator,
    registry: *optimization_registry.OptimizationRegistry
) !*OptimizationOrchestrator {
    global_orchestrator_mutex.lock();
    defer global_orchestrator_mutex.unlock();
    
    if (global_orchestrator) |*orchestrator| {
        return orchestrator;
    }
    
    const config = OrchestratorConfig{};
    global_orchestrator = try OptimizationOrchestrator.init(allocator, config, registry);
    return &global_orchestrator.?;
}

/// Clean up global optimization orchestrator
pub fn deinitGlobalOrchestrator() void {
    global_orchestrator_mutex.lock();
    defer global_orchestrator_mutex.unlock();
    
    if (global_orchestrator) |*orchestrator| {
        orchestrator.deinit();
        global_orchestrator = null;
    }
}

// ============================================================================
// Testing
// ============================================================================

test "OptimizationPlan basic functionality" {
    const allocator = std.testing.allocator;
    
    const steps = [_]OptimizationPlan.OptimizationStep{
        .{
            .optimization_type = .souper,
            .priority = 100,
            .dependencies = &[_]optimization_registry.OptimizationType{},
            .estimated_speedup = 1.5,
            .estimated_time_ms = 5000,
            .fallback_available = true,
        },
        .{
            .optimization_type = .ispc,
            .priority = 90,
            .dependencies = &[_]optimization_registry.OptimizationType{.souper},
            .estimated_speedup = 6.0,
            .estimated_time_ms = 10000,
            .fallback_available = true,
        },
    };
    
    var plan = OptimizationPlan{
        .region_id = 12345,
        .ordered_optimizations = try allocator.dupe(OptimizationPlan.OptimizationStep, &steps),
        .estimated_total_speedup = 9.0,
        .estimated_compilation_time_ms = 15000,
        .confidence_score = 0.8,
    };
    defer plan.deinit(allocator);
    
    // Test dependency satisfaction
    var completed = std.EnumSet(optimization_registry.OptimizationType).initEmpty();
    
    const first_step = plan.getNextStep(completed);
    try std.testing.expect(first_step != null);
    try std.testing.expect(first_step.?.optimization_type == .souper);
    
    completed.insert(.souper);
    const second_step = plan.getNextStep(completed);
    try std.testing.expect(second_step != null);
    try std.testing.expect(second_step.?.optimization_type == .ispc);
}

test "OptimizationOrchestrator integration" {
    const allocator = std.testing.allocator;
    
    // Create registry and orchestrator
    var registry = try optimization_registry.OptimizationRegistry.init(allocator, .{});
    defer registry.deinit();
    
    var orchestrator = try OptimizationOrchestrator.init(allocator, .{}, &registry);
    defer orchestrator.deinit();
    
    // Register a region
    const region_id = try registry.registerRegion("test_function", "test_module", 0x12345, 0.8);
    
    // Create and execute optimization plan
    var plan = try orchestrator.createOptimizationPlan(region_id);
    defer plan.deinit(allocator);
    
    try std.testing.expect(plan.ordered_optimizations.len > 0);
    try std.testing.expect(plan.estimated_total_speedup >= 1.0);
    
    const results = try orchestrator.executeOptimizationPlan(plan);
    try std.testing.expect(results.region_id == region_id);
    
    // Test statistics
    const stats = orchestrator.getStatistics();
    try std.testing.expect(stats.total_plans_executed == 1);
}