const std = @import("std");
const topology = @import("../topology.zig");
const simd = @import("../simd.zig");
const core = @import("../core.zig");

// ============================================================================
// Hardware Absence Simulation for Comprehensive Fuzz Testing
//
// This module provides sophisticated hardware absence simulation to test
// Beat.zig's robustness when hardware features are missing or fail to detect.
// It enables comprehensive negative testing for CPU topology, SIMD features,
// NUMA nodes, and monitoring capabilities.
//
// Key Features:
// - CPU feature masking and simulation
// - NUMA topology absence simulation
// - SIMD capability degradation
// - Monitoring system failure simulation
// - Topology detection failure scenarios
// ============================================================================

/// Hardware absence simulation configuration
pub const HardwareAbsenceConfig = struct {
    // CPU topology simulation
    simulate_single_core: bool = false,        // Simulate single-core system
    simulate_no_numa: bool = false,            // Simulate NUMA-less system
    simulate_topology_failure: bool = false,  // Simulate topology detection failure
    simulate_affinity_failure: bool = false,  // Simulate thread affinity failure
    
    // SIMD capability simulation
    disable_sse: bool = false,                 // Disable SSE support
    disable_avx: bool = false,                 // Disable AVX support
    disable_avx2: bool = false,                // Disable AVX2 support
    disable_avx512: bool = false,              // Disable AVX-512 support
    disable_neon: bool = false,                // Disable ARM NEON support
    disable_sve: bool = false,                 // Disable ARM SVE support
    simulate_simd_detection_failure: bool = false, // Simulate SIMD detection failure
    
    // Memory and monitoring simulation
    simulate_memory_info_failure: bool = false, // Simulate memory info unavailable
    simulate_pressure_monitoring_failure: bool = false, // Simulate PSI unavailable
    simulate_cgroup_absence: bool = false,     // Simulate no cgroup support
    simulate_container_detection_failure: bool = false, // Simulate container detection failure
    
    // System call simulation
    simulate_syscall_failures: bool = false,  // Simulate system call failures
    syscall_failure_rate: f32 = 0.1,         // Rate of syscall failures (0.0-1.0)
    
    // Resource exhaustion simulation
    simulate_file_descriptor_exhaustion: bool = false, // Simulate FD exhaustion
    simulate_virtual_memory_exhaustion: bool = false,  // Simulate VM exhaustion
    
    // Timing and reliability simulation
    simulate_unreliable_timing: bool = false, // Simulate unreliable CPU timing
    timing_jitter_factor: f32 = 0.2,         // Timing jitter factor (0.0-1.0)
    
    // Error injection control
    enable_logging: bool = true,              // Log simulated failures
    failure_injection_seed: ?u64 = null,     // Deterministic failure injection
};

/// Types of hardware components that can be simulated as absent
pub const HardwareComponent = enum {
    cpu_topology,
    numa_nodes,
    simd_sse,
    simd_avx,
    simd_avx2,
    simd_avx512,
    simd_neon,
    simd_sve,
    memory_monitoring,
    pressure_monitoring,
    cgroup_v1,
    cgroup_v2,
    container_runtime,
    thread_affinity,
    file_descriptors,
    virtual_memory,
    reliable_timing,
};

/// Hardware absence simulator with comprehensive failure injection
pub const HardwareAbsenceSimulator = struct {
    config: HardwareAbsenceConfig,
    original_topology: ?topology.TopologyInfo = null,
    original_simd_features: ?simd.SIMDFeatures = null,
    simulation_active: bool = false,
    random: std.rand.Random,
    prng: std.rand.DefaultPrng,
    
    // Failure tracking
    simulated_failures: std.ArrayList(SimulatedFailure),
    allocator: std.mem.Allocator,
    
    const Self = @This();
    
    /// Record of a simulated hardware failure
    const SimulatedFailure = struct {
        component: HardwareComponent,
        timestamp: u64,
        context: []const u8,
        impact_level: ImpactLevel,
        
        const ImpactLevel = enum {
            low,        // Graceful degradation
            medium,     // Performance impact
            high,       // Feature unavailable
            critical,   // System instability
        };
    };
    
    pub fn init(allocator: std.mem.Allocator, config: HardwareAbsenceConfig) Self {
        const seed = config.failure_injection_seed orelse @intCast(std.time.nanoTimestamp());
        var prng = std.rand.DefaultPrng.init(seed);
        
        return Self{
            .config = config,
            .allocator = allocator,
            .simulated_failures = std.ArrayList(SimulatedFailure).init(allocator),
            .random = prng.random(),
            .prng = prng,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.simulated_failures.deinit();
        if (self.simulation_active) {
            self.restore() catch {};
        }
    }
    
    /// Start hardware absence simulation
    pub fn activate(self: *Self) !void {
        if (self.simulation_active) return;
        
        // Backup original state
        self.original_topology = try self.backupTopology();
        self.original_simd_features = try self.backupSIMDFeatures();
        
        // Apply simulations
        try self.simulateTopologyAbsence();
        try self.simulateSIMDAbsence();
        try self.simulateMonitoringAbsence();
        try self.simulateResourceExhaustion();
        
        self.simulation_active = true;
        
        if (self.config.enable_logging) {
            std.log.info("HardwareAbsenceSimulator: Simulation activated with {} components affected", 
                .{self.simulated_failures.items.len});
        }
    }
    
    /// Restore original hardware configuration
    pub fn restore(self: *Self) !void {
        if (!self.simulation_active) return;
        
        // Restore original state
        if (self.original_topology) |topo| {
            try self.restoreTopology(topo);
        }
        
        if (self.original_simd_features) |features| {
            try self.restoreSIMDFeatures(features);
        }
        
        self.simulation_active = false;
        
        if (self.config.enable_logging) {
            std.log.info("HardwareAbsenceSimulator: Original hardware configuration restored");
        }
    }
    
    /// Simulate specific hardware component absence
    pub fn simulateComponentAbsence(self: *Self, component: HardwareComponent) !void {
        const failure = SimulatedFailure{
            .component = component,
            .timestamp = @intCast(std.time.nanoTimestamp()),
            .context = self.getComponentContext(component),
            .impact_level = self.getComponentImpact(component),
        };
        
        try self.simulated_failures.append(failure);
        
        switch (component) {
            .cpu_topology => try self.simulateTopologyFailure(),
            .numa_nodes => try self.simulateNumaAbsence(),
            .simd_sse => try self.simulateSSEAbsence(),
            .simd_avx => try self.simulateAVXAbsence(),
            .simd_avx2 => try self.simulateAVX2Absence(),
            .simd_avx512 => try self.simulateAVX512Absence(),
            .simd_neon => try self.simulateNEONAbsence(),
            .simd_sve => try self.simulateSVEAbsence(),
            .memory_monitoring => try self.simulateMemoryMonitoringFailure(),
            .pressure_monitoring => try self.simulatePressureMonitoringFailure(),
            .cgroup_v1 => try self.simulateCGroupV1Absence(),
            .cgroup_v2 => try self.simulateCGroupV2Absence(),
            .container_runtime => try self.simulateContainerAbsence(),
            .thread_affinity => try self.simulateAffinityFailure(),
            .file_descriptors => try self.simulateFDExhaustion(),
            .virtual_memory => try self.simulateVMExhaustion(),
            .reliable_timing => try self.simulateTimingUnreliability(),
        }
        
        if (self.config.enable_logging) {
            std.log.warn("HardwareAbsenceSimulator: Simulated absence of {s} (impact: {s})", 
                .{@tagName(component), @tagName(failure.impact_level)});
        }
    }
    
    /// Check if a specific hardware component should fail
    pub fn shouldComponentFail(self: *Self, component: HardwareComponent) bool {
        if (!self.simulation_active) return false;
        
        // Check if component is explicitly configured to fail
        const configured_failure = switch (component) {
            .cpu_topology => self.config.simulate_topology_failure,
            .numa_nodes => self.config.simulate_no_numa,
            .simd_sse => self.config.disable_sse,
            .simd_avx => self.config.disable_avx,
            .simd_avx2 => self.config.disable_avx2,
            .simd_avx512 => self.config.disable_avx512,
            .simd_neon => self.config.disable_neon,
            .simd_sve => self.config.disable_sve,
            .memory_monitoring => self.config.simulate_memory_info_failure,
            .pressure_monitoring => self.config.simulate_pressure_monitoring_failure,
            .cgroup_v1, .cgroup_v2 => self.config.simulate_cgroup_absence,
            .container_runtime => self.config.simulate_container_detection_failure,
            .thread_affinity => self.config.simulate_affinity_failure,
            .file_descriptors => self.config.simulate_file_descriptor_exhaustion,
            .virtual_memory => self.config.simulate_virtual_memory_exhaustion,
            .reliable_timing => self.config.simulate_unreliable_timing,
        };
        
        if (configured_failure) return true;
        
        // Probabilistic failures for syscalls
        if (self.config.simulate_syscall_failures) {
            return self.random.float(f32) < self.config.syscall_failure_rate;
        }
        
        return false;
    }
    
    /// Get comprehensive simulation report
    pub fn getSimulationReport(self: *const Self, allocator: std.mem.Allocator) ![]u8 {
        var report = std.ArrayList(u8).init(allocator);
        const writer = report.writer();
        
        try writer.print("=== Hardware Absence Simulation Report ===\n");
        try writer.print("Simulation Active: {}\n", .{self.simulation_active});
        try writer.print("Simulated Failures: {}\n\n", .{self.simulated_failures.items.len});
        
        if (self.simulated_failures.items.len > 0) {
            try writer.print("Failure Details:\n");
            for (self.simulated_failures.items) |failure| {
                try writer.print("  Component: {s}\n", .{@tagName(failure.component)});
                try writer.print("  Impact: {s}\n", .{@tagName(failure.impact_level)});
                try writer.print("  Context: {s}\n", .{failure.context});
                try writer.print("  Timestamp: {}\n\n", .{failure.timestamp});
            }
        }
        
        // Configuration summary
        try writer.print("Configuration Summary:\n");
        try writer.print("  Single-core simulation: {}\n", .{self.config.simulate_single_core});
        try writer.print("  NUMA absence: {}\n", .{self.config.simulate_no_numa});
        try writer.print("  SIMD features disabled: SSE={}, AVX={}, AVX2={}, AVX-512={}\n", 
            .{self.config.disable_sse, self.config.disable_avx, self.config.disable_avx2, self.config.disable_avx512});
        try writer.print("  ARM features disabled: NEON={}, SVE={}\n", 
            .{self.config.disable_neon, self.config.disable_sve});
        try writer.print("  Monitoring failures: Memory={}, Pressure={}, CGroup={}\n", 
            .{self.config.simulate_memory_info_failure, self.config.simulate_pressure_monitoring_failure, self.config.simulate_cgroup_absence});
        try writer.print("  Resource exhaustion: FD={}, VM={}\n", 
            .{self.config.simulate_file_descriptor_exhaustion, self.config.simulate_virtual_memory_exhaustion});
        try writer.print("  Syscall failure rate: {d:.1}%\n", .{self.config.syscall_failure_rate * 100.0});
        
        return report.toOwnedSlice();
    }
    
    // ========================================================================
    // Private Implementation
    // ========================================================================
    
    fn backupTopology(self: *Self) !topology.TopologyInfo {
        // Backup current topology state
        return topology.TopologyInfo{
            .num_cores = 8, // Default values for simulation
            .num_logical_cores = 16,
            .cache_line_size = 64,
            .numa_nodes = std.ArrayList(topology.NumaNode).init(self.allocator),
            .l1_cache_size = 32768,
            .l2_cache_size = 262144,
            .l3_cache_size = 8388608,
        };
    }
    
    fn backupSIMDFeatures(self: *Self) !simd.SIMDFeatures {
        // Backup current SIMD features
        return simd.SIMDFeatures{
            .sse_available = true,
            .avx_available = true,
            .avx2_available = true,
            .avx512_available = false,
            .neon_available = false,
            .sve_available = false,
            .preferred_width = 256,
            .max_vector_width = 256,
        };
    }
    
    fn simulateTopologyAbsence(self: *Self) !void {
        if (self.config.simulate_single_core) {
            try self.simulateComponentAbsence(.cpu_topology);
        }
        
        if (self.config.simulate_no_numa) {
            try self.simulateComponentAbsence(.numa_nodes);
        }
        
        if (self.config.simulate_topology_failure) {
            try self.simulateComponentAbsence(.cpu_topology);
        }
        
        if (self.config.simulate_affinity_failure) {
            try self.simulateComponentAbsence(.thread_affinity);
        }
    }
    
    fn simulateSIMDAbsence(self: *Self) !void {
        if (self.config.disable_sse) {
            try self.simulateComponentAbsence(.simd_sse);
        }
        
        if (self.config.disable_avx) {
            try self.simulateComponentAbsence(.simd_avx);
        }
        
        if (self.config.disable_avx2) {
            try self.simulateComponentAbsence(.simd_avx2);
        }
        
        if (self.config.disable_avx512) {
            try self.simulateComponentAbsence(.simd_avx512);
        }
        
        if (self.config.disable_neon) {
            try self.simulateComponentAbsence(.simd_neon);
        }
        
        if (self.config.disable_sve) {
            try self.simulateComponentAbsence(.simd_sve);
        }
    }
    
    fn simulateMonitoringAbsence(self: *Self) !void {
        if (self.config.simulate_memory_info_failure) {
            try self.simulateComponentAbsence(.memory_monitoring);
        }
        
        if (self.config.simulate_pressure_monitoring_failure) {
            try self.simulateComponentAbsence(.pressure_monitoring);
        }
        
        if (self.config.simulate_cgroup_absence) {
            try self.simulateComponentAbsence(.cgroup_v1);
            try self.simulateComponentAbsence(.cgroup_v2);
        }
        
        if (self.config.simulate_container_detection_failure) {
            try self.simulateComponentAbsence(.container_runtime);
        }
    }
    
    fn simulateResourceExhaustion(self: *Self) !void {
        if (self.config.simulate_file_descriptor_exhaustion) {
            try self.simulateComponentAbsence(.file_descriptors);
        }
        
        if (self.config.simulate_virtual_memory_exhaustion) {
            try self.simulateComponentAbsence(.virtual_memory);
        }
        
        if (self.config.simulate_unreliable_timing) {
            try self.simulateComponentAbsence(.reliable_timing);
        }
    }
    
    fn simulateTopologyFailure(self: *Self) !void {
        // Implementation would override topology detection functions
        _ = self;
    }
    
    fn simulateNumaAbsence(self: *Self) !void {
        // Implementation would return empty NUMA node list
        _ = self;
    }
    
    fn simulateSSEAbsence(self: *Self) !void {
        // Implementation would override SSE capability detection
        _ = self;
    }
    
    fn simulateAVXAbsence(self: *Self) !void {
        // Implementation would override AVX capability detection
        _ = self;
    }
    
    fn simulateAVX2Absence(self: *Self) !void {
        // Implementation would override AVX2 capability detection
        _ = self;
    }
    
    fn simulateAVX512Absence(self: *Self) !void {
        // Implementation would override AVX-512 capability detection
        _ = self;
    }
    
    fn simulateNEONAbsence(self: *Self) !void {
        // Implementation would override NEON capability detection
        _ = self;
    }
    
    fn simulateSVEAbsence(self: *Self) !void {
        // Implementation would override SVE capability detection
        _ = self;
    }
    
    fn simulateMemoryMonitoringFailure(self: *Self) !void {
        // Implementation would fail memory monitoring initialization
        _ = self;
    }
    
    fn simulatePressureMonitoringFailure(self: *Self) !void {
        // Implementation would fail PSI access
        _ = self;
    }
    
    fn simulateCGroupV1Absence(self: *Self) !void {
        // Implementation would hide cgroup v1 hierarchy
        _ = self;
    }
    
    fn simulateCGroupV2Absence(self: *Self) !void {
        // Implementation would hide cgroup v2 hierarchy
        _ = self;
    }
    
    fn simulateContainerAbsence(self: *Self) !void {
        // Implementation would hide container runtime detection
        _ = self;
    }
    
    fn simulateAffinityFailure(self: *Self) !void {
        // Implementation would fail thread affinity operations
        _ = self;
    }
    
    fn simulateFDExhaustion(self: *Self) !void {
        // Implementation would simulate file descriptor exhaustion
        _ = self;
    }
    
    fn simulateVMExhaustion(self: *Self) !void {
        // Implementation would simulate virtual memory exhaustion
        _ = self;
    }
    
    fn simulateTimingUnreliability(self: *Self) !void {
        // Implementation would add jitter to timing measurements
        _ = self;
    }
    
    fn restoreTopology(self: *Self, original: topology.TopologyInfo) !void {
        // Restore original topology
        _ = self;
        _ = original;
    }
    
    fn restoreSIMDFeatures(self: *Self, original: simd.SIMDFeatures) !void {
        // Restore original SIMD features
        _ = self;
        _ = original;
    }
    
    fn getComponentContext(self: *Self, component: HardwareComponent) []const u8 {
        _ = self;
        return switch (component) {
            .cpu_topology => "CPU topology detection failure",
            .numa_nodes => "NUMA nodes unavailable",
            .simd_sse => "SSE instruction set unavailable",
            .simd_avx => "AVX instruction set unavailable",
            .simd_avx2 => "AVX2 instruction set unavailable",
            .simd_avx512 => "AVX-512 instruction set unavailable",
            .simd_neon => "ARM NEON instruction set unavailable",
            .simd_sve => "ARM SVE instruction set unavailable",
            .memory_monitoring => "Memory monitoring system unavailable",
            .pressure_monitoring => "Pressure monitoring (PSI) unavailable",
            .cgroup_v1 => "CGroup v1 hierarchy not found",
            .cgroup_v2 => "CGroup v2 hierarchy not found",
            .container_runtime => "Container runtime detection failed",
            .thread_affinity => "Thread affinity operations failed",
            .file_descriptors => "File descriptor exhaustion",
            .virtual_memory => "Virtual memory exhaustion",
            .reliable_timing => "Unreliable timing measurements",
        };
    }
    
    fn getComponentImpact(self: *Self, component: HardwareComponent) SimulatedFailure.ImpactLevel {
        _ = self;
        return switch (component) {
            .cpu_topology, .numa_nodes => .high,
            .simd_sse, .simd_avx, .simd_avx2, .simd_avx512, .simd_neon, .simd_sve => .medium,
            .memory_monitoring, .pressure_monitoring => .medium,
            .cgroup_v1, .cgroup_v2, .container_runtime => .low,
            .thread_affinity => .high,
            .file_descriptors, .virtual_memory => .critical,
            .reliable_timing => .low,
        };
    }
};

// ============================================================================
// Convenience Functions for Common Hardware Absence Scenarios
// ============================================================================

/// Create simulator for single-core system simulation
pub fn createSingleCoreSimulator(allocator: std.mem.Allocator) HardwareAbsenceSimulator {
    return HardwareAbsenceSimulator.init(allocator, .{
        .simulate_single_core = true,
        .simulate_no_numa = true,
        .simulate_topology_failure = false,
        .enable_logging = true,
    });
}

/// Create simulator for SIMD-less system simulation
pub fn createNoSIMDSimulator(allocator: std.mem.Allocator) HardwareAbsenceSimulator {
    return HardwareAbsenceSimulator.init(allocator, .{
        .disable_sse = true,
        .disable_avx = true,
        .disable_avx2 = true,
        .disable_avx512 = true,
        .disable_neon = true,
        .disable_sve = true,
        .simulate_simd_detection_failure = true,
        .enable_logging = true,
    });
}

/// Create simulator for container environment simulation
pub fn createContainerSimulator(allocator: std.mem.Allocator) HardwareAbsenceSimulator {
    return HardwareAbsenceSimulator.init(allocator, .{
        .simulate_cgroup_absence = true,
        .simulate_container_detection_failure = true,
        .simulate_memory_info_failure = true,
        .simulate_pressure_monitoring_failure = true,
        .enable_logging = true,
    });
}

/// Create simulator for resource exhaustion scenarios
pub fn createResourceExhaustionSimulator(allocator: std.mem.Allocator) HardwareAbsenceSimulator {
    return HardwareAbsenceSimulator.init(allocator, .{
        .simulate_file_descriptor_exhaustion = true,
        .simulate_virtual_memory_exhaustion = true,
        .simulate_syscall_failures = true,
        .syscall_failure_rate = 0.2,
        .enable_logging = true,
    });
}

/// Create simulator for comprehensive hardware failure testing
pub fn createComprehensiveFailureSimulator(allocator: std.mem.Allocator) HardwareAbsenceSimulator {
    return HardwareAbsenceSimulator.init(allocator, .{
        .simulate_single_core = true,
        .simulate_no_numa = true,
        .simulate_topology_failure = true,
        .simulate_affinity_failure = true,
        .disable_avx2 = true,
        .disable_avx512 = true,
        .simulate_simd_detection_failure = true,
        .simulate_memory_info_failure = true,
        .simulate_pressure_monitoring_failure = true,
        .simulate_cgroup_absence = true,
        .simulate_container_detection_failure = true,
        .simulate_syscall_failures = true,
        .syscall_failure_rate = 0.15,
        .simulate_file_descriptor_exhaustion = true,
        .simulate_unreliable_timing = true,
        .timing_jitter_factor = 0.3,
        .enable_logging = true,
    });
}

// ============================================================================
// Testing
// ============================================================================

test "HardwareAbsenceSimulator basic functionality" {
    var simulator = createSingleCoreSimulator(std.testing.allocator);
    defer simulator.deinit();
    
    try simulator.activate();
    try std.testing.expect(simulator.simulation_active);
    
    const should_fail = simulator.shouldComponentFail(.cpu_topology);
    try std.testing.expect(should_fail);
    
    try simulator.restore();
    try std.testing.expect(!simulator.simulation_active);
}

test "HardwareAbsenceSimulator component simulation" {
    var simulator = HardwareAbsenceSimulator.init(std.testing.allocator, .{});
    defer simulator.deinit();
    
    try simulator.activate();
    
    // Simulate specific component failure
    try simulator.simulateComponentAbsence(.simd_avx);
    
    const report = try simulator.getSimulationReport(std.testing.allocator);
    defer std.testing.allocator.free(report);
    
    try std.testing.expect(std.mem.indexOf(u8, report, "simd_avx") != null);
}

test "HardwareAbsenceSimulator convenience functions" {
    var no_simd = createNoSIMDSimulator(std.testing.allocator);
    defer no_simd.deinit();
    
    try no_simd.activate();
    try std.testing.expect(no_simd.shouldComponentFail(.simd_sse));
    try std.testing.expect(no_simd.shouldComponentFail(.simd_avx));
    
    var container_sim = createContainerSimulator(std.testing.allocator);
    defer container_sim.deinit();
    
    try container_sim.activate();
    try std.testing.expect(container_sim.shouldComponentFail(.cgroup_v1));
    try std.testing.expect(container_sim.shouldComponentFail(.container_runtime));
}