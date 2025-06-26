# Fuzzing Framework API Reference

## Quick Reference

### Build Commands
```bash
zig build test-fuzzing-allocator      # Allocator error injection
zig build test-hardware-absence       # Hardware absence simulation  
zig build test-state-fuzzing          # State corruption testing
zig build test-comprehensive-fuzzing  # Unified framework
zig build test-all-fuzzing           # Complete test suite
```

### Core Modules

- `src/fuzzing.zig` - Unified framework and orchestration
- `src/fuzzing/fuzzing_allocator.zig` - Memory allocation error injection
- `src/fuzzing/hardware_absence_simulator.zig` - Hardware feature simulation
- `src/fuzzing/state_fuzzer.zig` - Component interaction testing

## API Documentation

### ComprehensiveFuzzer

Main orchestration class for coordinated fuzzing campaigns.

```zig
pub const ComprehensiveFuzzer = struct {
    pub fn init(allocator: std.mem.Allocator, config: ComprehensiveFuzzingConfig) Self
    pub fn deinit(self: *Self) void
    pub fn start(self: *Self) !void
    pub fn stop(self: *Self) !void
    pub fn executePhase(self: *Self, phase: FuzzingPhase) !void
    pub fn getComprehensiveReport(self: *const Self, allocator: std.mem.Allocator) ![]u8
}
```

#### Configuration

```zig
pub const ComprehensiveFuzzingConfig = struct {
    allocator_config: FuzzingAllocatorConfig = .{},
    hardware_config: HardwareAbsenceConfig = .{},
    state_config: StateFuzzingConfig = .{},
    fuzzing_duration_ms: u64 = 30000,
    target_branch_coverage: f32 = 0.65,
    enable_coordinated_phases: bool = true,
    enable_coverage_tracking: bool = true,
    enable_detailed_reporting: bool = true,
    max_memory_usage_mb: u64 = 512,
    emergency_stop_on_crash: bool = true,
}
```

#### Convenience Functions

```zig
// Pre-configured fuzzing scenarios
pub fn createComprehensiveRobustnessFuzzer(allocator: std.mem.Allocator) ComprehensiveFuzzer
pub fn createDevelopmentFuzzer(allocator: std.mem.Allocator) ComprehensiveFuzzer
```

### FuzzingAllocator

Memory allocation error injection with realistic failure patterns.

```zig
pub const FuzzingAllocator = struct {
    pub fn init(base_allocator: std.mem.Allocator, config: FuzzingAllocatorConfig) Self
    pub fn deinit(self: *Self) void
    pub fn allocator(self: *Self) std.mem.Allocator
    pub fn activate(self: *Self) !void
    pub fn deactivate(self: *Self) !void
    pub fn shouldFailAllocation(self: *Self, size: usize) bool
    pub fn enterCriticalPath(self: *Self) void
    pub fn exitCriticalPath(self: *Self) void
    pub fn getStatistics(self: *const Self) AllocationStatistics
    pub fn getAllocatorReport(self: *const Self, allocator: std.mem.Allocator) ![]u8
}
```

#### Configuration

```zig
pub const FuzzingAllocatorConfig = struct {
    failure_rate: f32 = 0.1,                           // 10% failure rate
    failure_pattern: FailurePattern = .probabilistic,   // Failure pattern type
    enable_critical_path_targeting: bool = false,       // Target critical operations
    critical_path_failure_multiplier: f32 = 3.0,       // 3x higher failure rate
    enable_allocation_tracking: bool = false,           // Track allocations
    max_allocated_bytes: usize = std.math.maxInt(usize), // Memory limit
    enable_detailed_logging: bool = false,              // Verbose logging
    track_allocations: bool = false,                    // Leak detection
}
```

#### Critical Path Management

```zig
pub const CriticalPathGuard = struct {
    pub fn init(fuzzing_allocator: *FuzzingAllocator) CriticalPathGuard
    pub fn deinit(self: CriticalPathGuard) void
}

// Usage
{
    const guard = FuzzingAllocator.CriticalPathGuard.init(&fuzzing_allocator);
    defer guard.deinit();
    // Allocations in this scope have higher failure probability
}
```

#### Convenience Functions

```zig
pub fn createHighFailureRateFuzzingAllocator(base_allocator: std.mem.Allocator) FuzzingAllocator
pub fn createCriticalPathFuzzingAllocator(base_allocator: std.mem.Allocator) FuzzingAllocator
pub fn createResourceExhaustionFuzzingAllocator(base_allocator: std.mem.Allocator) FuzzingAllocator
```

### HardwareAbsenceSimulator

Hardware feature absence simulation for testing degraded environments.

```zig
pub const HardwareAbsenceSimulator = struct {
    pub fn init(allocator: std.mem.Allocator, config: HardwareAbsenceConfig) Self
    pub fn deinit(self: *Self) void
    pub fn activate(self: *Self) !void
    pub fn restore(self: *Self) !void
    pub fn simulateComponentAbsence(self: *Self, component: HardwareComponent) !void
    pub fn shouldComponentFail(self: *Self, component: HardwareComponent) bool
    pub fn getSimulationReport(self: *const Self, allocator: std.mem.Allocator) ![]u8
}
```

#### Configuration

```zig
pub const HardwareAbsenceConfig = struct {
    // CPU topology simulation
    simulate_single_core: bool = false,
    simulate_no_numa: bool = false,
    simulate_topology_failure: bool = false,
    simulate_affinity_failure: bool = false,
    
    // SIMD capability simulation
    disable_sse: bool = false,
    disable_avx: bool = false,
    disable_avx2: bool = false,
    disable_avx512: bool = false,
    disable_neon: bool = false,
    disable_sve: bool = false,
    
    // Memory and monitoring simulation
    simulate_memory_info_failure: bool = false,
    simulate_pressure_monitoring_failure: bool = false,
    simulate_cgroup_absence: bool = false,
    simulate_container_detection_failure: bool = false,
    
    // Resource exhaustion simulation
    simulate_file_descriptor_exhaustion: bool = false,
    simulate_virtual_memory_exhaustion: bool = false,
    simulate_unreliable_timing: bool = false,
    
    enable_logging: bool = true,
}
```

#### Hardware Components

```zig
pub const HardwareComponent = enum {
    cpu_topology,           // CPU topology detection
    numa_nodes,            // NUMA node availability
    simd_sse,              // SSE instruction set
    simd_avx,              // AVX instruction set
    simd_avx2,             // AVX2 instruction set
    simd_avx512,           // AVX-512 instruction set
    simd_neon,             // ARM NEON instruction set
    simd_sve,              // ARM SVE instruction set
    memory_monitoring,      // Memory usage monitoring
    pressure_monitoring,    // Memory pressure monitoring
    cgroup_v1,             // CGroup v1 support
    cgroup_v2,             // CGroup v2 support
    container_runtime,      // Container runtime detection
    thread_affinity,       // Thread affinity support
    file_descriptors,      // File descriptor availability
    virtual_memory,        // Virtual memory availability
    reliable_timing,       // Reliable timing measurements
}
```

#### Convenience Functions

```zig
pub fn createSingleCoreSimulator(allocator: std.mem.Allocator) HardwareAbsenceSimulator
pub fn createNoSIMDSimulator(allocator: std.mem.Allocator) HardwareAbsenceSimulator
pub fn createContainerSimulator(allocator: std.mem.Allocator) HardwareAbsenceSimulator
pub fn createResourceExhaustionSimulator(allocator: std.mem.Allocator) HardwareAbsenceSimulator
pub fn createComprehensiveFailureSimulator(allocator: std.mem.Allocator) HardwareAbsenceSimulator
```

### StateFuzzer

Component interaction and race condition testing.

```zig
pub const StateFuzzer = struct {
    pub fn init(allocator: std.mem.Allocator, config: StateFuzzingConfig) Self
    pub fn deinit(self: *Self) void
    pub fn start(self: *Self) !void
    pub fn stop(self: *Self) !void
    pub fn injectStateCorruption(self: *Self, component: StateComponent, corruption_type: CorruptionType) !void
    pub fn detectRaceCondition(self: *Self, comp_a: StateComponent, comp_b: StateComponent, op_a: []const u8, op_b: []const u8) !void
    pub fn getFuzzingReport(self: *const Self, allocator: std.mem.Allocator) ![]u8
}
```

#### Configuration

```zig
pub const StateFuzzingConfig = struct {
    // Configuration fuzzing
    fuzz_worker_counts: bool = false,
    fuzz_memory_limits: bool = false,
    fuzz_timing_parameters: bool = false,
    fuzz_numa_configuration: bool = false,
    fuzz_simd_configuration: bool = false,
    
    // State transition fuzzing
    fuzz_lifecycle_transitions: bool = false,
    fuzz_scheduler_state: bool = false,
    fuzz_memory_pressure_state: bool = false,
    fuzz_worker_state: bool = false,
    
    // Component interaction fuzzing
    fuzz_thread_pool_scheduler: bool = false,
    fuzz_scheduler_memory: bool = false,
    fuzz_topology_affinity: bool = false,
    fuzz_cgroup_pressure: bool = false,
    
    // Race condition exploration
    enable_race_condition_fuzzing: bool = false,
    race_injection_probability: f32 = 0.1,
    
    // Boundary value testing
    test_integer_boundaries: bool = false,
    test_float_boundaries: bool = false,
    test_size_boundaries: bool = false,
    test_time_boundaries: bool = false,
    
    // Multi-threaded corruption
    enable_concurrent_corruption: bool = false,
    corruption_thread_count: u32 = 4,
    corruption_intensity: f32 = 0.05,
    
    enable_logging: bool = true,
    fuzzing_duration_ms: u64 = 5000,
}
```

#### State Components

```zig
pub const StateComponent = enum {
    thread_pool_config,      // Thread pool configuration
    worker_configuration,    // Worker thread configuration
    scheduler_parameters,    // Scheduler parameters
    memory_pressure_config,  // Memory pressure configuration
    numa_topology_config,    // NUMA topology configuration
    simd_feature_config,     // SIMD feature configuration
    cgroup_detection_config, // CGroup detection configuration
    heartbeat_timing,        // Heartbeat timing parameters
    prediction_parameters,   // Prediction algorithm parameters
    work_stealing_config,    // Work-stealing configuration
    affinity_settings,       // Thread affinity settings
    error_handling_state,    // Error handling state
}
```

#### Corruption Types

```zig
pub const CorruptionType = enum {
    invalid_value,       // Invalid configuration value
    boundary_violation,  // Boundary condition violation
    type_mismatch,      // Type system violation
    null_injection,     // Null pointer injection
    race_corruption,    // Race condition corruption
    overflow_underflow, // Integer overflow/underflow
    precision_loss,     // Floating point precision loss
    timeout_violation,  // Timeout constraint violation
}
```

#### Convenience Functions

```zig
pub fn createConfigurationFuzzer(allocator: std.mem.Allocator) StateFuzzer
pub fn createTransitionFuzzer(allocator: std.mem.Allocator) StateFuzzer
pub fn createInteractionFuzzer(allocator: std.mem.Allocator) StateFuzzer
pub fn createComprehensiveFuzzer(allocator: std.mem.Allocator) StateFuzzer
```

## Usage Patterns

### Basic Fuzzing Campaign

```zig
const std = @import("std");
const fuzzing = @import("src/fuzzing.zig");

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
defer _ = gpa.deinit();
const allocator = gpa.allocator();

// Create and run comprehensive fuzzer
var fuzzer = fuzzing.createComprehensiveRobustnessFuzzer(allocator);
defer fuzzer.deinit();

try fuzzer.start();
defer fuzzer.stop() catch {};

// Get results
const report = try fuzzer.getComprehensiveReport(allocator);
defer allocator.free(report);
std.log.info("Fuzzing Results:\n{s}", .{report});
```

### Custom Configuration

```zig
const config = fuzzing.ComprehensiveFuzzingConfig{
    .allocator_config = .{
        .failure_rate = 0.20,
        .enable_critical_path_targeting = true,
        .enable_detailed_logging = true,
    },
    .hardware_config = .{
        .simulate_single_core = true,
        .disable_avx2 = true,
        .simulate_cgroup_absence = true,
    },
    .state_config = .{
        .fuzz_worker_counts = true,
        .enable_race_condition_fuzzing = true,
        .corruption_thread_count = 8,
        .fuzzing_duration_ms = 10000,
    },
    .target_branch_coverage = 0.70,  // 70% coverage target
    .fuzzing_duration_ms = 45000,    // 45-second campaign
};

var fuzzer = fuzzing.ComprehensiveFuzzer.init(allocator, config);
defer fuzzer.deinit();
```

### Individual Component Testing

```zig
// Test allocator robustness
var fuzz_alloc = fuzzing.createCriticalPathFuzzingAllocator(std.testing.allocator);
defer fuzz_alloc.deinit();

try fuzz_alloc.activate();
defer fuzz_alloc.deactivate() catch {};

const allocator = fuzz_alloc.allocator();
// Use allocator normally - failures will be injected

// Test hardware absence
var simulator = fuzzing.createNoSIMDSimulator(std.testing.allocator);
defer simulator.deinit();

try simulator.activate();
defer simulator.restore() catch {};

// Hardware features will appear absent

// Test state corruption
var state_fuzzer = fuzzing.createInteractionFuzzer(std.testing.allocator);
defer state_fuzzer.deinit();

try state_fuzzer.start();
defer state_fuzzer.stop() catch {};
```

### Integration with Tests

```zig
test "beat thread pool robustness" {
    var fuzzer = fuzzing.createDevelopmentFuzzer(std.testing.allocator);
    defer fuzzer.deinit();
    
    // Quick 2-second fuzzing for CI
    fuzzer.config.fuzzing_duration_ms = 2000;
    
    try fuzzer.start();
    defer fuzzer.stop() catch {};
    
    // Verify coverage target was met
    try std.testing.expect(fuzzer.branch_coverage >= 0.50);
}
```

## Error Handling

All fuzzing operations return errors for proper handling:

```zig
// Handle fuzzing errors
fuzzer.start() catch |err| switch (err) {
    error.OutOfMemory => std.log.err("Insufficient memory for fuzzing"),
    error.InvalidConfiguration => std.log.err("Invalid fuzzing configuration"),
    error.SystemResourceExhaustion => std.log.err("System resources exhausted"),
    else => return err,
};
```

## Performance Guidelines

### Memory Usage

- Set `max_memory_usage_mb` to prevent excessive memory consumption
- Use `createDevelopmentFuzzer()` for CI environments with limited resources
- Enable allocation tracking only when needed (performance overhead)

### Duration

- **Development**: 2-5 seconds for quick validation
- **CI/CD**: 10-15 seconds for automated testing  
- **Release**: 30-60 seconds for comprehensive validation
- **Research**: 5+ minutes for exhaustive coverage analysis

### Parallel Execution

- Increase `corruption_thread_count` for multi-core systems
- Use coordinated phases for systematic testing
- Enable concurrent corruption for race condition detection

This API reference provides the essential information needed to effectively use Beat.zig's fuzzing framework for robustness testing and validation.