const std = @import("std");

// ============================================================================
// Unified Optimization Registry
// 
// This module provides centralized coordination for Souper, ISPC, and Minotaur
// optimization systems to eliminate conflicts and duplicated transformations.
//
// ISSUE ADDRESSED:
// - Souper, ISPC, and Minotaur run independently causing potential conflicts
// - Duplicated transformations on same code regions (fingerprint similarity, worker selection)
// - No coordination mechanism for optimization selection
// - Performance impact correlation missing between systems
//
// SOLUTION:
// - Central registry of optimization regions and active transformations
// - Conflict detection and resolution using capability matrices
// - Dependency-aware optimization ordering via DAG
// - Performance-driven optimization selection and adaptation
// ============================================================================

/// Types of optimization systems available in Beat.zig
pub const OptimizationType = enum {
    souper,      // Scalar mathematical optimizations via SMT solving
    ispc,        // SPMD vectorization and parallel execution
    minotaur,    // SIMD instruction synthesis and optimization
    
    /// Get human-readable name for optimization type
    pub fn getName(self: OptimizationType) []const u8 {
        return switch (self) {
            .souper => "Souper",
            .ispc => "ISPC", 
            .minotaur => "Minotaur",
        };
    }
    
    /// Get typical performance characteristics for this optimization type
    pub fn getCharacteristics(self: OptimizationType) OptimizationCharacteristics {
        return switch (self) {
            .souper => OptimizationCharacteristics{
                .primary_target = .scalar_math,
                .typical_speedup = 1.5,
                .compilation_overhead = .low,
                .memory_impact = .minimal,
                .verification_available = true,
            },
            .ispc => OptimizationCharacteristics{
                .primary_target = .parallel_loops,
                .typical_speedup = 6.0,
                .compilation_overhead = .medium,
                .memory_impact = .moderate,
                .verification_available = false,
            },
            .minotaur => OptimizationCharacteristics{
                .primary_target = .simd_instructions,
                .typical_speedup = 2.5,
                .compilation_overhead = .high,
                .memory_impact = .minimal,
                .verification_available = true,
            },
        };
    }
};

/// Performance characteristics of an optimization type
pub const OptimizationCharacteristics = struct {
    primary_target: OptimizationTarget,
    typical_speedup: f32,
    compilation_overhead: CompilationOverhead,
    memory_impact: MemoryImpact,
    verification_available: bool,
    
    pub const OptimizationTarget = enum {
        scalar_math,
        parallel_loops,
        simd_instructions,
        memory_access,
        control_flow,
    };
    
    pub const CompilationOverhead = enum { low, medium, high };
    pub const MemoryImpact = enum { minimal, moderate, significant };
};

/// Unique identifier for a code region that can be optimized
pub const OptimizationRegionId = u64;

/// Represents a code region that can be optimized by one or more systems
pub const OptimizationRegion = struct {
    id: OptimizationRegionId,
    function_name: []const u8,
    module_name: []const u8,
    code_hash: u64,                           // Hash of the code to optimize
    estimated_execution_frequency: f32,       // How often this code is executed (0.0-1.0)
    
    // Current optimization state
    active_optimizations: std.EnumSet(OptimizationType),
    attempted_optimizations: std.EnumSet(OptimizationType),
    failed_optimizations: std.EnumSet(OptimizationType),
    
    // Performance tracking
    baseline_performance: PerformanceMetrics,
    optimized_performance: std.EnumMap(OptimizationType, PerformanceMetrics),
    
    // Dependencies and conflicts
    dependencies: []OptimizationDependency,
    conflicts: []OptimizationConflict,
    
    // Metadata
    creation_time_ns: u64,
    last_update_ns: u64,
    optimization_count: u32,
    
    /// Initialize a new optimization region
    pub fn init(
        allocator: std.mem.Allocator,
        function_name: []const u8,
        module_name: []const u8,
        code_hash: u64,
        estimated_frequency: f32
    ) !OptimizationRegion {
        const now = @as(u64, @intCast(std.time.nanoTimestamp()));
        
        return OptimizationRegion{
            .id = generateRegionId(function_name, module_name, code_hash),
            .function_name = try allocator.dupe(u8, function_name),
            .module_name = try allocator.dupe(u8, module_name),
            .code_hash = code_hash,
            .estimated_execution_frequency = estimated_frequency,
            .active_optimizations = std.EnumSet(OptimizationType).initEmpty(),
            .attempted_optimizations = std.EnumSet(OptimizationType).initEmpty(),
            .failed_optimizations = std.EnumSet(OptimizationType).initEmpty(),
            .baseline_performance = PerformanceMetrics.init(),
            .optimized_performance = std.EnumMap(OptimizationType, PerformanceMetrics).init(.{}),
            .dependencies = &[_]OptimizationDependency{},
            .conflicts = &[_]OptimizationConflict{},
            .creation_time_ns = now,
            .last_update_ns = now,
            .optimization_count = 0,
        };
    }
    
    /// Clean up allocated resources
    pub fn deinit(self: *OptimizationRegion, allocator: std.mem.Allocator) void {
        allocator.free(self.function_name);
        allocator.free(self.module_name);
        if (self.dependencies.len > 0) {
            allocator.free(self.dependencies);
        }
        if (self.conflicts.len > 0) {
            allocator.free(self.conflicts);
        }
    }
    
    /// Check if this region is suitable for a specific optimization type
    pub fn isSuitableFor(self: *const OptimizationRegion, opt_type: OptimizationType) bool {
        // Already failed this optimization
        if (self.failed_optimizations.contains(opt_type)) return false;
        
        // Already optimized with this system
        if (self.active_optimizations.contains(opt_type)) return false;
        
        // Check for conflicts with active optimizations
        for (self.conflicts) |conflict| {
            if (conflict.optimization_a == opt_type or conflict.optimization_b == opt_type) {
                const conflicting_type = if (conflict.optimization_a == opt_type) 
                    conflict.optimization_b 
                else 
                    conflict.optimization_a;
                
                if (self.active_optimizations.contains(conflicting_type)) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    /// Record successful optimization application
    pub fn recordOptimizationSuccess(
        self: *OptimizationRegion, 
        opt_type: OptimizationType, 
        performance: PerformanceMetrics
    ) void {
        self.active_optimizations.insert(opt_type);
        self.attempted_optimizations.insert(opt_type);
        self.optimized_performance.put(opt_type, performance);
        self.optimization_count += 1;
        self.last_update_ns = @as(u64, @intCast(std.time.nanoTimestamp()));
    }
    
    /// Record failed optimization attempt
    pub fn recordOptimizationFailure(self: *OptimizationRegion, opt_type: OptimizationType) void {
        self.failed_optimizations.insert(opt_type);
        self.attempted_optimizations.insert(opt_type);
        self.last_update_ns = @as(u64, @intCast(std.time.nanoTimestamp()));
    }
    
    /// Get the best performing optimization for this region
    pub fn getBestOptimization(self: *const OptimizationRegion) ?OptimizationType {
        var best_type: ?OptimizationType = null;
        var best_speedup: f32 = 1.0;
        
        var iter = self.optimized_performance.iterator();
        while (iter.next()) |entry| {
            const opt_type = entry.key;
            const performance = entry.value;
            
            if (self.active_optimizations.contains(opt_type)) {
                const speedup = self.baseline_performance.calculateSpeedup(performance);
                if (speedup > best_speedup) {
                    best_speedup = speedup;
                    best_type = opt_type;
                }
            }
        }
        
        return best_type;
    }
    
    /// Generate a unique region ID from function and module information
    fn generateRegionId(function_name: []const u8, module_name: []const u8, code_hash: u64) OptimizationRegionId {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(function_name);
        hasher.update(module_name);
        hasher.update(std.mem.asBytes(&code_hash));
        return hasher.final();
    }
};

/// Performance metrics for optimization evaluation
pub const PerformanceMetrics = struct {
    execution_time_ns: u64 = 0,
    cpu_cycles: u64 = 0,
    memory_usage_bytes: u64 = 0,
    cache_misses: u64 = 0,
    instructions_executed: u64 = 0,
    
    /// Initialize default performance metrics
    pub fn init() PerformanceMetrics {
        return PerformanceMetrics{};
    }
    
    /// Calculate speedup compared to another set of metrics
    pub fn calculateSpeedup(self: PerformanceMetrics, optimized: PerformanceMetrics) f32 {
        if (optimized.execution_time_ns == 0) return 1.0;
        return @as(f32, @floatFromInt(self.execution_time_ns)) / @as(f32, @floatFromInt(optimized.execution_time_ns));
    }
    
    /// Calculate efficiency score (speedup per resource unit)
    pub fn calculateEfficiency(self: PerformanceMetrics, optimized: PerformanceMetrics) f32 {
        const speedup = self.calculateSpeedup(optimized);
        const memory_ratio = if (self.memory_usage_bytes > 0) 
            @as(f32, @floatFromInt(optimized.memory_usage_bytes)) / @as(f32, @floatFromInt(self.memory_usage_bytes))
        else 
            1.0;
        
        return speedup / memory_ratio; // Higher is better
    }
};

/// Represents a dependency between optimizations
pub const OptimizationDependency = struct {
    dependent: OptimizationType,
    dependency: OptimizationType,
    dependency_type: DependencyType,
    
    pub const DependencyType = enum {
        requires,           // Dependent optimization requires dependency to be applied first
        benefits_from,      // Dependent optimization works better with dependency
        must_follow,        // Dependent optimization must be applied after dependency
    };
};

/// Represents a conflict between optimizations
pub const OptimizationConflict = struct {
    optimization_a: OptimizationType,
    optimization_b: OptimizationType,
    conflict_type: ConflictType,
    severity: ConflictSeverity,
    resolution_strategy: ResolutionStrategy,
    
    pub const ConflictType = enum {
        incompatible_transformations,  // Optimizations modify code in incompatible ways
        performance_degradation,       // Combined application performs worse than individual
        resource_contention,           // Optimizations compete for same resources
        verification_conflicts,        // Formal verification cannot handle both optimizations
    };
    
    pub const ConflictSeverity = enum {
        low,       // Minor performance impact, warnings only
        medium,    // Moderate impact, avoid if possible
        high,      // Significant impact, must choose one
        critical,  // Correctness issues, never combine
    };
    
    pub const ResolutionStrategy = enum {
        choose_best_performing,   // Select optimization with best performance
        choose_most_reliable,     // Select optimization with best reliability
        choose_by_priority,       // Use predefined optimization priorities
        disable_both,             // Disable both optimizations if conflict too severe
    };
};

// ============================================================================
// Optimization Registry Implementation
// ============================================================================

/// Configuration for the optimization registry
pub const RegistryConfig = struct {
    /// Maximum number of regions to track simultaneously
    max_regions: u32 = 4096,
    
    /// Enable automatic conflict detection
    enable_conflict_detection: bool = true,
    
    /// Enable performance tracking and adaptation
    enable_performance_tracking: bool = true,
    
    /// Minimum execution frequency to consider for optimization
    min_execution_frequency: f32 = 0.01, // 1%
    
    /// Cleanup interval for old regions (nanoseconds)
    cleanup_interval_ns: u64 = 300_000_000_000, // 5 minutes
    
    /// Age threshold for cleaning up unused regions (nanoseconds)
    cleanup_age_threshold_ns: u64 = 3_600_000_000_000, // 1 hour
};

/// Central registry coordinating all optimization systems
pub const OptimizationRegistry = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    config: RegistryConfig,
    
    // Core registry data
    regions: std.AutoHashMap(OptimizationRegionId, OptimizationRegion),
    regions_mutex: std.Thread.Mutex = .{},
    
    // Conflict and dependency management
    global_conflicts: std.ArrayList(OptimizationConflict),
    global_dependencies: std.ArrayList(OptimizationDependency),
    
    // Performance tracking
    total_optimizations: std.atomic.Value(u64),
    successful_optimizations: std.atomic.Value(u64),
    failed_optimizations: std.atomic.Value(u64),
    conflicts_detected: std.atomic.Value(u64),
    
    // Cleanup management
    last_cleanup_ns: std.atomic.Value(u64),
    
    /// Initialize the optimization registry
    pub fn init(allocator: std.mem.Allocator, config: RegistryConfig) !Self {
        const now = @as(u64, @intCast(std.time.nanoTimestamp()));
        
        var registry = Self{
            .allocator = allocator,
            .config = config,
            .regions = std.AutoHashMap(OptimizationRegionId, OptimizationRegion).init(allocator),
            .global_conflicts = std.ArrayList(OptimizationConflict).init(allocator),
            .global_dependencies = std.ArrayList(OptimizationDependency).init(allocator),
            .total_optimizations = std.atomic.Value(u64).init(0),
            .successful_optimizations = std.atomic.Value(u64).init(0),
            .failed_optimizations = std.atomic.Value(u64).init(0),
            .conflicts_detected = std.atomic.Value(u64).init(0),
            .last_cleanup_ns = std.atomic.Value(u64).init(now),
        };
        
        // Initialize default conflicts and dependencies
        try registry.initializeDefaultConflictsAndDependencies();
        
        return registry;
    }
    
    /// Clean up resources
    pub fn deinit(self: *Self) void {
        self.regions_mutex.lock();
        defer self.regions_mutex.unlock();
        
        // Clean up all regions
        var iter = self.regions.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        
        self.regions.deinit();
        self.global_conflicts.deinit();
        self.global_dependencies.deinit();
    }
    
    /// Register a new optimization region
    pub fn registerRegion(
        self: *Self,
        function_name: []const u8,
        module_name: []const u8,
        code_hash: u64,
        estimated_frequency: f32
    ) !OptimizationRegionId {
        if (estimated_frequency < self.config.min_execution_frequency) {
            return error.ExecutionFrequencyTooLow;
        }
        
        var region = try OptimizationRegion.init(
            self.allocator,
            function_name,
            module_name,
            code_hash,
            estimated_frequency
        );
        
        // Add detected conflicts and dependencies
        if (self.config.enable_conflict_detection) {
            try self.detectAndAddConflicts(&region);
        }
        
        self.regions_mutex.lock();
        defer self.regions_mutex.unlock();
        
        const region_id = region.id;
        try self.regions.put(region_id, region);
        
        return region_id;
    }
    
    /// Get optimization region by ID
    pub fn getRegion(self: *Self, region_id: OptimizationRegionId) ?*OptimizationRegion {
        self.regions_mutex.lock();
        defer self.regions_mutex.unlock();
        
        return self.regions.getPtr(region_id);
    }
    
    /// Check if an optimization is suitable for a region
    pub fn isOptimizationSuitable(
        self: *Self,
        region_id: OptimizationRegionId,
        opt_type: OptimizationType
    ) bool {
        self.regions_mutex.lock();
        defer self.regions_mutex.unlock();
        
        if (self.regions.getPtr(region_id)) |region| {
            return region.isSuitableFor(opt_type);
        }
        
        return false;
    }
    
    /// Record successful optimization application
    pub fn recordOptimizationSuccess(
        self: *Self,
        region_id: OptimizationRegionId,
        opt_type: OptimizationType,
        performance: PerformanceMetrics
    ) void {
        _ = self.total_optimizations.fetchAdd(1, .monotonic);
        _ = self.successful_optimizations.fetchAdd(1, .monotonic);
        
        self.regions_mutex.lock();
        defer self.regions_mutex.unlock();
        
        if (self.regions.getPtr(region_id)) |region| {
            region.recordOptimizationSuccess(opt_type, performance);
        }
    }
    
    /// Record failed optimization attempt
    pub fn recordOptimizationFailure(
        self: *Self,
        region_id: OptimizationRegionId,
        opt_type: OptimizationType
    ) void {
        _ = self.total_optimizations.fetchAdd(1, .monotonic);
        _ = self.failed_optimizations.fetchAdd(1, .monotonic);
        
        self.regions_mutex.lock();
        defer self.regions_mutex.unlock();
        
        if (self.regions.getPtr(region_id)) |region| {
            region.recordOptimizationFailure(opt_type);
        }
    }
    
    /// Get optimization recommendation for a region
    pub fn getOptimizationRecommendation(
        self: *Self,
        region_id: OptimizationRegionId
    ) ?OptimizationRecommendation {
        self.regions_mutex.lock();
        defer self.regions_mutex.unlock();
        
        const region = self.regions.getPtr(region_id) orelse return null;
        
        // Find best available optimization
        const available_optimizations = [_]OptimizationType{ .souper, .ispc, .minotaur };
        var best_recommendation: ?OptimizationRecommendation = null;
        var best_score: f32 = 0.0;
        
        for (available_optimizations) |opt_type| {
            if (region.isSuitableFor(opt_type)) {
                const characteristics = opt_type.getCharacteristics();
                const score = self.calculateOptimizationScore(region, opt_type, characteristics);
                
                if (score > best_score) {
                    best_score = score;
                    best_recommendation = OptimizationRecommendation{
                        .optimization_type = opt_type,
                        .confidence = score,
                        .expected_speedup = characteristics.typical_speedup,
                        .rationale = self.getOptimizationRationale(region, opt_type),
                    };
                }
            }
        }
        
        return best_recommendation;
    }
    
    /// Get registry performance statistics
    pub fn getStatistics(self: *Self) RegistryStatistics {
        self.regions_mutex.lock();
        defer self.regions_mutex.unlock();
        
        const total_opts = self.total_optimizations.load(.monotonic);
        const successful_opts = self.successful_optimizations.load(.monotonic);
        const failed_opts = self.failed_optimizations.load(.monotonic);
        const conflicts = self.conflicts_detected.load(.monotonic);
        
        const success_rate = if (total_opts > 0) 
            @as(f32, @floatFromInt(successful_opts)) / @as(f32, @floatFromInt(total_opts))
        else 
            0.0;
        
        return RegistryStatistics{
            .total_regions = @intCast(self.regions.count()),
            .total_optimizations = total_opts,
            .successful_optimizations = successful_opts,
            .failed_optimizations = failed_opts,
            .success_rate = success_rate,
            .conflicts_detected = conflicts,
            .last_cleanup_ns = self.last_cleanup_ns.load(.monotonic),
        };
    }
    
    // Private helper methods
    
    /// Initialize default conflicts and dependencies based on known optimization characteristics
    fn initializeDefaultConflictsAndDependencies(self: *Self) !void {
        // Known conflicts between optimization systems
        
        // ISPC and Minotaur conflict on SIMD instruction optimization
        try self.global_conflicts.append(OptimizationConflict{
            .optimization_a = .ispc,
            .optimization_b = .minotaur,
            .conflict_type = .incompatible_transformations,
            .severity = .medium,
            .resolution_strategy = .choose_best_performing,
        });
        
        // Souper benefits from being applied before ISPC (scalar optimizations first)
        try self.global_dependencies.append(OptimizationDependency{
            .dependent = .ispc,
            .dependency = .souper,
            .dependency_type = .benefits_from,
        });
        
        // Minotaur should be applied after ISPC (vector instruction optimization after vectorization)
        try self.global_dependencies.append(OptimizationDependency{
            .dependent = .minotaur,
            .dependency = .ispc,
            .dependency_type = .must_follow,
        });
    }
    
    /// Detect and add conflicts for a specific region
    fn detectAndAddConflicts(self: *Self, region: *OptimizationRegion) !void {
        // Copy global conflicts to region-specific conflicts
        var conflicts = std.ArrayList(OptimizationConflict).init(self.allocator);
        defer conflicts.deinit();
        
        for (self.global_conflicts.items) |conflict| {
            try conflicts.append(conflict);
        }
        
        // Add region-specific conflict detection logic here
        // (e.g., based on function signature, module, etc.)
        
        region.conflicts = try conflicts.toOwnedSlice();
    }
    
    /// Calculate optimization score for recommendation
    fn calculateOptimizationScore(
        self: *Self,
        region: *const OptimizationRegion,
        opt_type: OptimizationType,
        characteristics: OptimizationCharacteristics
    ) f32 {
        _ = self;
        
        var score: f32 = 0.0;
        
        // Base score from expected speedup
        score += characteristics.typical_speedup * 10.0;
        
        // Weight by execution frequency
        score *= region.estimated_execution_frequency;
        
        // Penalty for compilation overhead
        score *= switch (characteristics.compilation_overhead) {
            .low => 1.0,
            .medium => 0.8,
            .high => 0.6,
        };
        
        // Bonus for verification availability
        if (characteristics.verification_available) {
            score *= 1.1;
        }
        
        // Penalty for previous failures
        if (region.failed_optimizations.contains(opt_type)) {
            score *= 0.1;
        }
        
        return std.math.clamp(score, 0.0, 100.0);
    }
    
    /// Get rationale for optimization recommendation
    fn getOptimizationRationale(
        self: *Self,
        region: *const OptimizationRegion,
        opt_type: OptimizationType
    ) OptimizationRationale {
        _ = self;
        
        return switch (opt_type) {
            .souper => if (region.estimated_execution_frequency > 0.5)
                .high_frequency_scalar_math
            else
                .mathematical_optimization_opportunity,
            .ispc => if (region.function_name.len > 10) // Heuristic for complex functions
                .parallel_vectorization_candidate
            else
                .spmd_optimization_opportunity,
            .minotaur => .simd_instruction_optimization,
        };
    }
};

/// Recommendation for applying an optimization to a region
pub const OptimizationRecommendation = struct {
    optimization_type: OptimizationType,
    confidence: f32,           // 0.0-100.0 confidence score
    expected_speedup: f32,     // Expected performance improvement multiplier
    rationale: OptimizationRationale,
};

/// Rationale for optimization recommendation
pub const OptimizationRationale = enum {
    high_frequency_scalar_math,
    mathematical_optimization_opportunity,
    parallel_vectorization_candidate,
    spmd_optimization_opportunity,
    simd_instruction_optimization,
    performance_critical_path,
    
    pub fn getDescription(self: OptimizationRationale) []const u8 {
        return switch (self) {
            .high_frequency_scalar_math => "High-frequency scalar mathematical operations",
            .mathematical_optimization_opportunity => "Mathematical optimization opportunity detected",
            .parallel_vectorization_candidate => "Function suitable for parallel vectorization",
            .spmd_optimization_opportunity => "SPMD optimization opportunity identified",
            .simd_instruction_optimization => "SIMD instruction optimization potential",
            .performance_critical_path => "Function on performance-critical execution path",
        };
    }
};

/// Performance statistics for the registry
pub const RegistryStatistics = struct {
    total_regions: u32,
    total_optimizations: u64,
    successful_optimizations: u64,
    failed_optimizations: u64,
    success_rate: f32,
    conflicts_detected: u64,
    last_cleanup_ns: u64,
    
    pub fn format(
        self: RegistryStatistics,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("OptimizationRegistry{{ regions: {}, optimizations: {}, success_rate: {d:.1}% }}", 
            .{ self.total_regions, self.total_optimizations, self.success_rate * 100.0 });
    }
};

// ============================================================================
// Global Access and Convenience Functions
// ============================================================================

/// Global optimization registry instance
var global_registry: ?OptimizationRegistry = null;
var global_registry_mutex: std.Thread.Mutex = .{};

/// Get the global optimization registry, initializing if necessary
pub fn getGlobalRegistry(allocator: std.mem.Allocator) !*OptimizationRegistry {
    global_registry_mutex.lock();
    defer global_registry_mutex.unlock();
    
    if (global_registry) |*registry| {
        return registry;
    }
    
    const config = RegistryConfig{};
    global_registry = try OptimizationRegistry.init(allocator, config);
    return &global_registry.?;
}

/// Clean up global optimization registry
pub fn deinitGlobalRegistry() void {
    global_registry_mutex.lock();
    defer global_registry_mutex.unlock();
    
    if (global_registry) |*registry| {
        registry.deinit();
        global_registry = null;
    }
}

// ============================================================================
// Testing
// ============================================================================

test "OptimizationRegion basic functionality" {
    const allocator = std.testing.allocator;
    
    var region = try OptimizationRegion.init(
        allocator,
        "computeSimilarity",
        "fingerprint",
        0x12345678,
        0.8
    );
    defer region.deinit(allocator);
    
    // Test initial state
    try std.testing.expect(!region.active_optimizations.contains(.souper));
    try std.testing.expect(region.isSuitableFor(.souper));
    try std.testing.expect(region.isSuitableFor(.ispc));
    
    // Test optimization recording
    const performance = PerformanceMetrics{
        .execution_time_ns = 1000000,
        .cpu_cycles = 2000,
    };
    
    region.recordOptimizationSuccess(.souper, performance);
    try std.testing.expect(region.active_optimizations.contains(.souper));
    try std.testing.expect(region.optimization_count == 1);
}

test "OptimizationRegistry integration" {
    const allocator = std.testing.allocator;
    
    var registry = try OptimizationRegistry.init(allocator, RegistryConfig{});
    defer registry.deinit();
    
    // Register optimization regions
    const region1 = try registry.registerRegion("fingerprint_similarity", "fingerprint", 0x111, 0.9);
    const region2 = try registry.registerRegion("worker_selection", "scheduler", 0x222, 0.7);
    
    // Test optimization suitability
    try std.testing.expect(registry.isOptimizationSuitable(region1, .souper));
    try std.testing.expect(registry.isOptimizationSuitable(region2, .ispc));
    
    // Test optimization recording
    const performance = PerformanceMetrics{
        .execution_time_ns = 500000,
        .cpu_cycles = 1000,
    };
    
    registry.recordOptimizationSuccess(region1, .souper, performance);
    
    // Test recommendations
    const recommendation = registry.getOptimizationRecommendation(region2);
    try std.testing.expect(recommendation != null);
    try std.testing.expect(recommendation.?.confidence > 0.0);
    
    // Test statistics
    const stats = registry.getStatistics();
    try std.testing.expect(stats.total_regions == 2);
    try std.testing.expect(stats.successful_optimizations == 1);
}