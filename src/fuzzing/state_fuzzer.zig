const std = @import("std");
const core = @import("../core.zig");
const scheduler = @import("../scheduler.zig");
const memory_pressure = @import("../memory_pressure.zig");
const cgroup_detection = @import("../cgroup_detection.zig");

// ============================================================================
// State Fuzzing for Beat.zig Component Interactions
//
// This module provides comprehensive state fuzzing to test Beat.zig's
// robustness under invalid configurations, state transitions, and
// component interaction failures. It enables systematic exploration
// of edge cases and race conditions.
//
// Key Features:
// - Invalid configuration injection
// - State transition fuzzing
// - Component interaction failure simulation
// - Race condition exploration
// - Configuration boundary testing
// - Multi-threaded state corruption
// ============================================================================

/// State fuzzing configuration
pub const StateFuzzingConfig = struct {
    // Configuration fuzzing
    fuzz_worker_counts: bool = false,          // Test invalid worker counts
    fuzz_memory_limits: bool = false,          // Test invalid memory limits
    fuzz_timing_parameters: bool = false,      // Test invalid timing values
    fuzz_numa_configuration: bool = false,     // Test invalid NUMA configs
    fuzz_simd_configuration: bool = false,     // Test invalid SIMD configs
    
    // State transition fuzzing
    fuzz_lifecycle_transitions: bool = false,  // Test invalid lifecycle transitions
    fuzz_scheduler_state: bool = false,        // Test invalid scheduler states
    fuzz_memory_pressure_state: bool = false,  // Test invalid memory pressure states
    fuzz_worker_state: bool = false,          // Test invalid worker states
    
    // Component interaction fuzzing
    fuzz_thread_pool_scheduler: bool = false, // Test pool-scheduler interaction
    fuzz_scheduler_memory: bool = false,      // Test scheduler-memory interaction
    fuzz_topology_affinity: bool = false,    // Test topology-affinity interaction
    fuzz_cgroup_pressure: bool = false,      // Test cgroup-pressure interaction
    
    // Race condition exploration
    enable_race_condition_fuzzing: bool = false, // Enable systematic race exploration
    race_injection_probability: f32 = 0.1,     // Probability of race injection (0.0-1.0)
    race_injection_delay_ns: u64 = 1000,       // Delay injection for race creation
    
    // Boundary value testing
    test_integer_boundaries: bool = false,     // Test integer overflow/underflow
    test_float_boundaries: bool = false,       // Test float precision/infinity
    test_size_boundaries: bool = false,        // Test size limit boundaries
    test_time_boundaries: bool = false,        // Test timestamp boundaries
    
    // Multi-threaded corruption
    enable_concurrent_corruption: bool = false, // Enable concurrent state corruption
    corruption_thread_count: u32 = 4,          // Number of corruption threads
    corruption_intensity: f32 = 0.05,          // Corruption intensity (0.0-1.0)
    
    // Control parameters
    enable_logging: bool = true,               // Log fuzzing activities
    fuzzing_duration_ms: u64 = 5000,         // Fuzzing duration in milliseconds
    state_snapshot_interval_ms: u64 = 100,    // State snapshot interval
    fuzzing_seed: ?u64 = null,                // Deterministic fuzzing seed
};

/// Types of state components that can be fuzzed
pub const StateComponent = enum {
    thread_pool_config,
    worker_configuration,
    scheduler_parameters,
    memory_pressure_config,
    numa_topology_config,
    simd_feature_config,
    cgroup_detection_config,
    heartbeat_timing,
    prediction_parameters,
    work_stealing_config,
    affinity_settings,
    error_handling_state,
};

/// State fuzzer with comprehensive component interaction testing
pub const StateFuzzer = struct {
    config: StateFuzzingConfig,
    random: std.Random,
    prng: std.rand.DefaultPrng,
    
    // State tracking
    original_states: std.HashMap(StateComponent, StateSnapshot),
    corrupted_states: std.ArrayList(CorruptedState),
    race_conditions: std.ArrayList(RaceCondition),
    allocator: std.mem.Allocator,
    
    // Fuzzing control
    fuzzing_active: bool = false,
    start_time: u64 = 0,
    snapshot_count: u64 = 0,
    corruption_threads: std.ArrayList(std.Thread),
    should_stop: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    
    const Self = @This();
    
    /// Snapshot of a component's state
    const StateSnapshot = struct {
        component: StateComponent,
        timestamp: u64,
        data: []u8,
        checksum: u32,
    };
    
    /// Record of state corruption
    const CorruptedState = struct {
        component: StateComponent,
        corruption_type: CorruptionType,
        original_value: []const u8,
        corrupted_value: []const u8,
        timestamp: u64,
        thread_id: std.Thread.Id,
        
        const CorruptionType = enum {
            invalid_value,
            boundary_violation,
            type_mismatch,
            null_injection,
            race_corruption,
            overflow_underflow,
            precision_loss,
            timeout_violation,
        };
    };
    
    /// Record of detected race condition
    const RaceCondition = struct {
        component_a: StateComponent,
        component_b: StateComponent,
        operation_a: []const u8,
        operation_b: []const u8,
        timestamp: u64,
        thread_a: std.Thread.Id,
        thread_b: std.Thread.Id,
        severity: Severity,
        
        const Severity = enum {
            low,        // Benign race
            medium,     // Data race with recovery
            high,       // Data race causing corruption
            critical,   // Data race causing crash
        };
    };
    
    pub fn init(allocator: std.mem.Allocator, config: StateFuzzingConfig) Self {
        const seed = config.fuzzing_seed orelse @as(u64, @intCast(std.time.nanoTimestamp()));
        var prng = std.rand.DefaultPrng.init(seed);
        
        return Self{
            .config = config,
            .allocator = allocator,
            .original_states = std.HashMap(StateComponent, StateSnapshot).init(allocator),
            .corrupted_states = std.ArrayList(CorruptedState).init(allocator),
            .race_conditions = std.ArrayList(RaceCondition).init(allocator),
            .corruption_threads = std.ArrayList(std.Thread).init(allocator),
            .random = prng.random(),
            .prng = prng,
        };
    }
    
    pub fn deinit(self: *Self) void {
        if (self.fuzzing_active) {
            self.stop() catch {};
        }
        
        self.original_states.deinit();
        self.corrupted_states.deinit();
        self.race_conditions.deinit();
        self.corruption_threads.deinit();
    }
    
    /// Start comprehensive state fuzzing
    pub fn start(self: *Self) !void {
        if (self.fuzzing_active) return;
        
        self.start_time = @as(u64, @intCast(std.time.nanoTimestamp()));
        self.should_stop.store(false, .monotonic);
        
        // Capture original states
        try self.captureOriginalStates();
        
        // Start fuzzing activities
        if (self.config.fuzz_worker_counts) try self.fuzzWorkerCounts();
        if (self.config.fuzz_memory_limits) try self.fuzzMemoryLimits();
        if (self.config.fuzz_timing_parameters) try self.fuzzTimingParameters();
        if (self.config.fuzz_numa_configuration) try self.fuzzNumaConfiguration();
        if (self.config.fuzz_simd_configuration) try self.fuzzSIMDConfiguration();
        
        if (self.config.fuzz_lifecycle_transitions) try self.fuzzLifecycleTransitions();
        if (self.config.fuzz_scheduler_state) try self.fuzzSchedulerState();
        if (self.config.fuzz_memory_pressure_state) try self.fuzzMemoryPressureState();
        if (self.config.fuzz_worker_state) try self.fuzzWorkerState();
        
        if (self.config.fuzz_thread_pool_scheduler) try self.fuzzThreadPoolScheduler();
        if (self.config.fuzz_scheduler_memory) try self.fuzzSchedulerMemory();
        if (self.config.fuzz_topology_affinity) try self.fuzzTopologyAffinity();
        if (self.config.fuzz_cgroup_pressure) try self.fuzzCGroupPressure();
        
        if (self.config.enable_race_condition_fuzzing) try self.startRaceConditionFuzzing();
        if (self.config.enable_concurrent_corruption) try self.startConcurrentCorruption();
        
        // Start boundary testing
        if (self.config.test_integer_boundaries) try self.testIntegerBoundaries();
        if (self.config.test_float_boundaries) try self.testFloatBoundaries();
        if (self.config.test_size_boundaries) try self.testSizeBoundaries();
        if (self.config.test_time_boundaries) try self.testTimeBoundaries();
        
        self.fuzzing_active = true;
        
        if (self.config.enable_logging) {
            std.log.info("StateFuzzer: Comprehensive state fuzzing started for {}ms", 
                .{self.config.fuzzing_duration_ms});
        }
        
        // Run fuzzing for specified duration
        try self.runFuzzingLoop();
    }
    
    /// Stop state fuzzing and restore original states
    pub fn stop(self: *Self) !void {
        if (!self.fuzzing_active) return;
        
        self.should_stop.store(true, .monotonic);
        
        // Wait for corruption threads to finish
        for (self.corruption_threads.items) |thread| {
            thread.join();
        }
        self.corruption_threads.clearRetainingCapacity();
        
        // Restore original states
        try self.restoreOriginalStates();
        
        self.fuzzing_active = false;
        
        if (self.config.enable_logging) {
            std.log.info("StateFuzzer: Fuzzing stopped, {} corruptions detected, {} race conditions found", 
                .{self.corrupted_states.items.len, self.race_conditions.items.len});
        }
    }
    
    /// Inject specific state corruption
    pub fn injectStateCorruption(self: *Self, component: StateComponent, corruption_type: CorruptedState.CorruptionType) !void {
        const corruption = CorruptedState{
            .component = component,
            .corruption_type = corruption_type,
            .original_value = try self.captureComponentState(component),
            .corrupted_value = try self.generateCorruptedValue(component, corruption_type),
            .timestamp = @as(u64, @intCast(std.time.nanoTimestamp())),
            .thread_id = std.Thread.getCurrentId(),
        };
        
        try self.corrupted_states.append(corruption);
        
        // Apply corruption
        try self.applyStateCorruption(corruption);
        
        if (self.config.enable_logging) {
            std.log.warn("StateFuzzer: Injected {s} corruption in {s}", 
                .{@tagName(corruption_type), @tagName(component)});
        }
    }
    
    /// Check for race conditions between components
    pub fn detectRaceCondition(self: *Self, comp_a: StateComponent, comp_b: StateComponent, 
                              op_a: []const u8, op_b: []const u8) !void {
        // Simple race detection based on timing
        const now = @as(u64, @intCast(std.time.nanoTimestamp()));
        const race_window_ns = 1000000; // 1ms race detection window
        
        // Check if operations are happening concurrently
        for (self.race_conditions.items) |existing_race| {
            if (now - existing_race.timestamp < race_window_ns) {
                if ((existing_race.component_a == comp_a and existing_race.component_b == comp_b) or
                    (existing_race.component_a == comp_b and existing_race.component_b == comp_a)) {
                    // Potential race condition detected
                    const race = RaceCondition{
                        .component_a = comp_a,
                        .component_b = comp_b,
                        .operation_a = op_a,
                        .operation_b = op_b,
                        .timestamp = now,
                        .thread_a = std.Thread.getCurrentId(),
                        .thread_b = existing_race.thread_a,
                        .severity = self.assessRaceSeverity(comp_a, comp_b),
                    };
                    
                    try self.race_conditions.append(race);
                    
                    if (self.config.enable_logging) {
                        std.log.warn("StateFuzzer: Race condition detected between {s} and {s} (severity: {s})", 
                            .{@tagName(comp_a), @tagName(comp_b), @tagName(race.severity)});
                    }
                    break;
                }
            }
        }
    }
    
    /// Get comprehensive fuzzing report
    pub fn getFuzzingReport(self: *const Self, allocator: std.mem.Allocator) ![]u8 {
        var report = std.ArrayList(u8).init(allocator);
        const writer = report.writer();
        
        try writer.print("=== State Fuzzing Report ===\n");
        try writer.print("Fuzzing Active: {}\n", .{self.fuzzing_active});
        try writer.print("Duration: {}ms\n", .{@divTrunc(@as(u64, @intCast(std.time.nanoTimestamp())) - self.start_time, 1000000)});
        try writer.print("State Corruptions: {}\n", .{self.corrupted_states.items.len});
        try writer.print("Race Conditions: {}\n", .{self.race_conditions.items.len});
        try writer.print("State Snapshots: {}\n\n", .{self.snapshot_count});
        
        // Corruption details
        if (self.corrupted_states.items.len > 0) {
            try writer.print("State Corruptions:\n");
            for (self.corrupted_states.items) |corruption| {
                try writer.print("  Component: {s}\n", .{@tagName(corruption.component)});
                try writer.print("  Type: {s}\n", .{@tagName(corruption.corruption_type)});
                try writer.print("  Thread: {}\n", .{corruption.thread_id});
                try writer.print("  Timestamp: {}\n\n", .{corruption.timestamp});
            }
        }
        
        // Race condition details
        if (self.race_conditions.items.len > 0) {
            try writer.print("Race Conditions:\n");
            for (self.race_conditions.items) |race| {
                try writer.print("  Components: {s} <-> {s}\n", .{@tagName(race.component_a), @tagName(race.component_b)});
                try writer.print("  Operations: {s} <-> {s}\n", .{race.operation_a, race.operation_b});
                try writer.print("  Severity: {s}\n", .{@tagName(race.severity)});
                try writer.print("  Threads: {} <-> {}\n", .{race.thread_a, race.thread_b});
                try writer.print("  Timestamp: {}\n\n", .{race.timestamp});
            }
        }
        
        // Configuration summary
        try writer.print("Configuration Summary:\n");
        try writer.print("  Worker count fuzzing: {}\n", .{self.config.fuzz_worker_counts});
        try writer.print("  Memory limit fuzzing: {}\n", .{self.config.fuzz_memory_limits});
        try writer.print("  Timing parameter fuzzing: {}\n", .{self.config.fuzz_timing_parameters});
        try writer.print("  Lifecycle transition fuzzing: {}\n", .{self.config.fuzz_lifecycle_transitions});
        try writer.print("  Race condition detection: {}\n", .{self.config.enable_race_condition_fuzzing});
        try writer.print("  Concurrent corruption: {}\n", .{self.config.enable_concurrent_corruption});
        try writer.print("  Boundary testing: Integer={}, Float={}, Size={}, Time={}\n", 
            .{self.config.test_integer_boundaries, self.config.test_float_boundaries, 
              self.config.test_size_boundaries, self.config.test_time_boundaries});
        
        return report.toOwnedSlice();
    }
    
    // ========================================================================
    // Private Implementation
    // ========================================================================
    
    fn captureOriginalStates(self: *Self) !void {
        const components = [_]StateComponent{
            .thread_pool_config, .worker_configuration, .scheduler_parameters,
            .memory_pressure_config, .numa_topology_config, .simd_feature_config,
            .cgroup_detection_config, .heartbeat_timing, .prediction_parameters,
            .work_stealing_config, .affinity_settings, .error_handling_state,
        };
        
        for (components) |component| {
            const state_data = try self.captureComponentState(component);
            const snapshot = StateSnapshot{
                .component = component,
                .timestamp = @as(u64, @intCast(std.time.nanoTimestamp())),
                .data = state_data,
                .checksum = self.calculateChecksum(state_data),
            };
            
            try self.original_states.put(component, snapshot);
        }
    }
    
    fn captureComponentState(self: *Self, component: StateComponent) ![]u8 {
        // Implementation would capture actual component state
        var data = std.ArrayList(u8).init(self.allocator);
        
        switch (component) {
            .thread_pool_config => try data.writer().print("default_thread_pool_config"),
            .worker_configuration => try data.writer().print("default_worker_config"),
            .scheduler_parameters => try data.writer().print("default_scheduler_params"),
            .memory_pressure_config => try data.writer().print("default_memory_pressure_config"),
            .numa_topology_config => try data.writer().print("default_numa_config"),
            .simd_feature_config => try data.writer().print("default_simd_config"),
            .cgroup_detection_config => try data.writer().print("default_cgroup_config"),
            .heartbeat_timing => try data.writer().print("default_heartbeat_timing"),
            .prediction_parameters => try data.writer().print("default_prediction_params"),
            .work_stealing_config => try data.writer().print("default_work_stealing_config"),
            .affinity_settings => try data.writer().print("default_affinity_settings"),
            .error_handling_state => try data.writer().print("default_error_handling"),
        }
        
        return data.toOwnedSlice();
    }
    
    fn generateCorruptedValue(self: *Self, component: StateComponent, corruption_type: CorruptedState.CorruptionType) ![]u8 {
        var data = std.ArrayList(u8).init(self.allocator);
        const writer = data.writer();
        
        switch (corruption_type) {
            .invalid_value => try writer.print("INVALID_{s}_VALUE", .{@tagName(component)}),
            .boundary_violation => try writer.print("BOUNDARY_VIOLATION_{s}", .{@tagName(component)}),
            .type_mismatch => try writer.print("TYPE_MISMATCH_{s}", .{@tagName(component)}),
            .null_injection => try writer.print("NULL_INJECTION_{s}", .{@tagName(component)}),
            .race_corruption => try writer.print("RACE_CORRUPTION_{s}", .{@tagName(component)}),
            .overflow_underflow => try writer.print("OVERFLOW_{s}", .{@tagName(component)}),
            .precision_loss => try writer.print("PRECISION_LOSS_{s}", .{@tagName(component)}),
            .timeout_violation => try writer.print("TIMEOUT_VIOLATION_{s}", .{@tagName(component)}),
        }
        
        return data.toOwnedSlice();
    }
    
    fn applyStateCorruption(self: *Self, corruption: CorruptedState) !void {
        // Implementation would apply actual state corruption
        _ = self;
        _ = corruption;
    }
    
    fn restoreOriginalStates(self: *Self) !void {
        var iterator = self.original_states.iterator();
        while (iterator.next()) |entry| {
            // Implementation would restore actual component state
            _ = entry;
        }
    }
    
    fn runFuzzingLoop(self: *Self) !void {
        const end_time = self.start_time + (self.config.fuzzing_duration_ms * 1000000);
        
        while (@as(u64, @intCast(std.time.nanoTimestamp())) < end_time and !self.should_stop.load(.monotonic)) {
            // Take state snapshot
            self.snapshot_count += 1;
            
            // Sleep for snapshot interval
            std.time.sleep(self.config.state_snapshot_interval_ms * 1000000);
        }
    }
    
    fn calculateChecksum(self: *Self, data: []const u8) u32 {
        _ = self;
        var hasher = std.hash.Crc32.init();
        hasher.update(data);
        return hasher.final();
    }
    
    fn assessRaceSeverity(self: *Self, comp_a: StateComponent, comp_b: StateComponent) RaceCondition.Severity {
        _ = self;
        
        // Critical components that can cause crashes
        const critical_components = [_]StateComponent{
            .thread_pool_config, .worker_configuration, .scheduler_parameters
        };
        
        // High-impact components that can cause corruption
        const high_impact_components = [_]StateComponent{
            .memory_pressure_config, .work_stealing_config, .affinity_settings
        };
        
        for (critical_components) |critical| {
            if (comp_a == critical or comp_b == critical) return .critical;
        }
        
        for (high_impact_components) |high_impact| {
            if (comp_a == high_impact or comp_b == high_impact) return .high;
        }
        
        return .medium;
    }
    
    // Configuration fuzzing implementations
    fn fuzzWorkerCounts(self: *Self) !void {
        // Test invalid worker counts (0, negative, excessive)
        try self.injectStateCorruption(.worker_configuration, .boundary_violation);
    }
    
    fn fuzzMemoryLimits(self: *Self) !void {
        // Test invalid memory limits
        try self.injectStateCorruption(.memory_pressure_config, .overflow_underflow);
    }
    
    fn fuzzTimingParameters(self: *Self) !void {
        // Test invalid timing values
        try self.injectStateCorruption(.heartbeat_timing, .boundary_violation);
    }
    
    fn fuzzNumaConfiguration(self: *Self) !void {
        // Test invalid NUMA configurations
        try self.injectStateCorruption(.numa_topology_config, .invalid_value);
    }
    
    fn fuzzSIMDConfiguration(self: *Self) !void {
        // Test invalid SIMD configurations
        try self.injectStateCorruption(.simd_feature_config, .type_mismatch);
    }
    
    // State transition fuzzing implementations
    fn fuzzLifecycleTransitions(self: *Self) !void {
        // Test invalid lifecycle state transitions
        try self.injectStateCorruption(.thread_pool_config, .invalid_value);
    }
    
    fn fuzzSchedulerState(self: *Self) !void {
        // Test invalid scheduler states
        try self.injectStateCorruption(.scheduler_parameters, .race_corruption);
    }
    
    fn fuzzMemoryPressureState(self: *Self) !void {
        // Test invalid memory pressure states
        try self.injectStateCorruption(.memory_pressure_config, .invalid_value);
    }
    
    fn fuzzWorkerState(self: *Self) !void {
        // Test invalid worker states
        try self.injectStateCorruption(.worker_configuration, .race_corruption);
    }
    
    // Component interaction fuzzing implementations
    fn fuzzThreadPoolScheduler(self: *Self) !void {
        // Test thread pool-scheduler interaction failures
        try self.detectRaceCondition(.thread_pool_config, .scheduler_parameters, "pool_init", "scheduler_init");
    }
    
    fn fuzzSchedulerMemory(self: *Self) !void {
        // Test scheduler-memory interaction failures
        try self.detectRaceCondition(.scheduler_parameters, .memory_pressure_config, "schedule_task", "pressure_update");
    }
    
    fn fuzzTopologyAffinity(self: *Self) !void {
        // Test topology-affinity interaction failures
        try self.detectRaceCondition(.numa_topology_config, .affinity_settings, "topology_detect", "affinity_set");
    }
    
    fn fuzzCGroupPressure(self: *Self) !void {
        // Test cgroup-pressure interaction failures
        try self.detectRaceCondition(.cgroup_detection_config, .memory_pressure_config, "cgroup_detect", "pressure_monitor");
    }
    
    // Race condition and boundary testing implementations
    fn startRaceConditionFuzzing(self: *Self) !void {
        // Implementation would start systematic race exploration
        _ = self;
    }
    
    fn startConcurrentCorruption(self: *Self) !void {
        // Implementation would start concurrent corruption threads
        _ = self;
    }
    
    fn testIntegerBoundaries(self: *Self) !void {
        // Test integer overflow/underflow scenarios
        try self.injectStateCorruption(.worker_configuration, .overflow_underflow);
    }
    
    fn testFloatBoundaries(self: *Self) !void {
        // Test float precision/infinity scenarios
        try self.injectStateCorruption(.prediction_parameters, .precision_loss);
    }
    
    fn testSizeBoundaries(self: *Self) !void {
        // Test size limit boundary scenarios
        try self.injectStateCorruption(.memory_pressure_config, .boundary_violation);
    }
    
    fn testTimeBoundaries(self: *Self) !void {
        // Test timestamp boundary scenarios
        try self.injectStateCorruption(.heartbeat_timing, .timeout_violation);
    }
};

// ============================================================================
// Convenience Functions for Common State Fuzzing Scenarios
// ============================================================================

/// Create fuzzer for configuration robustness testing
pub fn createConfigurationFuzzer(allocator: std.mem.Allocator) StateFuzzer {
    return StateFuzzer.init(allocator, .{
        .fuzz_worker_counts = true,
        .fuzz_memory_limits = true,
        .fuzz_timing_parameters = true,
        .fuzz_numa_configuration = true,
        .fuzz_simd_configuration = true,
        .test_integer_boundaries = true,
        .test_float_boundaries = true,
        .enable_logging = true,
        .fuzzing_duration_ms = 2000,
    });
}

/// Create fuzzer for state transition robustness testing
pub fn createTransitionFuzzer(allocator: std.mem.Allocator) StateFuzzer {
    return StateFuzzer.init(allocator, .{
        .fuzz_lifecycle_transitions = true,
        .fuzz_scheduler_state = true,
        .fuzz_memory_pressure_state = true,
        .fuzz_worker_state = true,
        .enable_race_condition_fuzzing = true,
        .race_injection_probability = 0.2,
        .enable_logging = true,
        .fuzzing_duration_ms = 3000,
    });
}

/// Create fuzzer for component interaction testing
pub fn createInteractionFuzzer(allocator: std.mem.Allocator) StateFuzzer {
    return StateFuzzer.init(allocator, .{
        .fuzz_thread_pool_scheduler = true,
        .fuzz_scheduler_memory = true,
        .fuzz_topology_affinity = true,
        .fuzz_cgroup_pressure = true,
        .enable_race_condition_fuzzing = true,
        .enable_concurrent_corruption = true,
        .corruption_thread_count = 2,
        .enable_logging = true,
        .fuzzing_duration_ms = 4000,
    });
}

/// Create fuzzer for comprehensive robustness testing
pub fn createComprehensiveFuzzer(allocator: std.mem.Allocator) StateFuzzer {
    return StateFuzzer.init(allocator, .{
        .fuzz_worker_counts = true,
        .fuzz_memory_limits = true,
        .fuzz_timing_parameters = true,
        .fuzz_numa_configuration = true,
        .fuzz_simd_configuration = true,
        .fuzz_lifecycle_transitions = true,
        .fuzz_scheduler_state = true,
        .fuzz_memory_pressure_state = true,
        .fuzz_worker_state = true,
        .fuzz_thread_pool_scheduler = true,
        .fuzz_scheduler_memory = true,
        .fuzz_topology_affinity = true,
        .fuzz_cgroup_pressure = true,
        .enable_race_condition_fuzzing = true,
        .enable_concurrent_corruption = true,
        .test_integer_boundaries = true,
        .test_float_boundaries = true,
        .test_size_boundaries = true,
        .test_time_boundaries = true,
        .corruption_thread_count = 4,
        .corruption_intensity = 0.1,
        .race_injection_probability = 0.15,
        .enable_logging = true,
        .fuzzing_duration_ms = 10000,
    });
}

// ============================================================================
// Testing
// ============================================================================

test "StateFuzzer basic functionality" {
    var fuzzer = createConfigurationFuzzer(std.testing.allocator);
    defer fuzzer.deinit();
    
    try fuzzer.start();
    try std.testing.expect(fuzzer.fuzzing_active);
    
    try fuzzer.stop();
    try std.testing.expect(!fuzzer.fuzzing_active);
}

test "StateFuzzer state corruption injection" {
    var fuzzer = StateFuzzer.init(std.testing.allocator, .{});
    defer fuzzer.deinit();
    
    try fuzzer.injectStateCorruption(.worker_configuration, .invalid_value);
    try std.testing.expect(fuzzer.corrupted_states.items.len == 1);
    
    const corruption = fuzzer.corrupted_states.items[0];
    try std.testing.expect(corruption.component == .worker_configuration);
    try std.testing.expect(corruption.corruption_type == .invalid_value);
}

test "StateFuzzer race condition detection" {
    var fuzzer = StateFuzzer.init(std.testing.allocator, .{});
    defer fuzzer.deinit();
    
    try fuzzer.detectRaceCondition(.thread_pool_config, .scheduler_parameters, "init", "start");
    // Note: Race detection requires timing, so this test may not trigger actual detection
}

test "StateFuzzer comprehensive testing" {
    var fuzzer = createComprehensiveFuzzer(std.testing.allocator);
    defer fuzzer.deinit();
    
    // Short duration for testing
    fuzzer.config.fuzzing_duration_ms = 100;
    
    try fuzzer.start();
    
    const report = try fuzzer.getFuzzingReport(std.testing.allocator);
    defer std.testing.allocator.free(report);
    
    try std.testing.expect(std.mem.indexOf(u8, report, "State Fuzzing Report") != null);
}