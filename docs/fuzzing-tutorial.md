# Fuzzing Framework Tutorial

This tutorial walks you through using Beat.zig's comprehensive fuzzing framework to validate your application's robustness under extreme conditions.

## Getting Started

### Prerequisites

- Zig compiler installed
- Beat.zig project cloned and building successfully
- Basic understanding of memory allocation and threading concepts

### Quick Validation

First, let's verify the fuzzing framework is working:

```bash
# Test individual components
zig build test-fuzzing-allocator
zig build test-hardware-absence  
zig build test-state-fuzzing

# Run comprehensive suite
zig build test-all-fuzzing
```

If these commands succeed, you're ready to use the fuzzing framework!

## Tutorial 1: Basic Memory Allocation Testing

Let's start with simple allocation failure testing to understand how the framework works.

### Step 1: Create a Simple Test

```zig
const std = @import("std");
const fuzzing = @import("src/fuzzing.zig");

test "basic allocation failure handling" {
    // Create a fuzzing allocator with 20% failure rate
    var fuzz_alloc = fuzzing.FuzzingAllocator.init(std.testing.allocator, .{
        .failure_rate = 0.2,
        .enable_detailed_logging = true,
    });
    defer fuzz_alloc.deinit();
    
    // Activate failure injection
    try fuzz_alloc.activate();
    defer fuzz_alloc.deactivate() catch {};
    
    const allocator = fuzz_alloc.allocator();
    
    // Test your allocation-dependent code
    var successful_allocations: u32 = 0;
    var failed_allocations: u32 = 0;
    
    for (0..100) |_| {
        const memory = allocator.alloc(u8, 1024) catch {
            failed_allocations += 1;
            continue;
        };
        defer allocator.free(memory);
        successful_allocations += 1;
    }
    
    std.log.info("Successful: {}, Failed: {}", .{successful_allocations, failed_allocations});
    
    // Print statistics
    fuzz_alloc.printStatistics();
}
```

### Step 2: Run and Analyze

```bash
zig test your_test.zig
```

You should see output like:
```
Successful: 78, Failed: 22

=== FuzzingAllocator Statistics ===
Allocations attempted: 100
Allocations succeeded: 78
Allocation failures: 22 (22.0%)
Total allocated: 79872 bytes
...
```

This confirms your code gracefully handles allocation failures!

## Tutorial 2: Testing Critical Code Paths

Now let's test critical operations that are essential for your application.

### Step 1: Identify Critical Paths

In Beat.zig, critical paths include:
- Thread pool initialization
- Worker thread creation
- Memory pressure monitor setup

### Step 2: Use Critical Path Targeting

```zig
test "critical path allocation testing" {
    var fuzz_alloc = fuzzing.createCriticalPathFuzzingAllocator(std.testing.allocator);
    defer fuzz_alloc.deinit();
    
    try fuzz_alloc.activate();
    defer fuzz_alloc.deactivate() catch {};
    
    const allocator = fuzz_alloc.allocator();
    
    // Test critical initialization
    {
        // Enter critical path - failures are 3x more likely here
        const guard = fuzzing.FuzzingAllocator.CriticalPathGuard.init(&fuzz_alloc);
        defer guard.deinit();
        
        // Your critical initialization code here
        const important_data = allocator.alloc(u8, 4096) catch |err| {
            std.log.info("Critical allocation failed: {}", .{err});
            // Implement fallback or graceful degradation
            return;
        };
        defer allocator.free(important_data);
        
        std.log.info("Critical allocation succeeded");
    }
    
    // Normal operations have standard failure rate
    const normal_data = allocator.alloc(u8, 1024) catch null;
    if (normal_data) |data| {
        defer allocator.free(data);
        std.log.info("Normal allocation succeeded");
    }
}
```

### Key Insight

Critical path targeting helps you validate that essential operations can gracefully handle failures, while non-critical operations may fail without breaking your application.

## Tutorial 3: Hardware Absence Simulation

Test how your application behaves when hardware features are missing.

### Step 1: Test SIMD Absence

```zig
test "SIMD feature absence" {
    var simulator = fuzzing.createNoSIMDSimulator(std.testing.allocator);
    defer simulator.deinit();
    
    try simulator.activate();
    defer simulator.restore() catch {};
    
    // Your SIMD-dependent code will now fallback to scalar implementations
    
    // Check if AVX2 should be unavailable
    if (simulator.shouldComponentFail(.simd_avx2)) {
        std.log.info("AVX2 simulated as unavailable - testing fallback");
        // Test your scalar fallback code
    }
    
    // Generate report
    const report = try simulator.getSimulationReport(std.testing.allocator);
    defer std.testing.allocator.free(report);
    std.log.info("Hardware Simulation:\n{s}", .{report});
}
```

### Step 2: Test Container Environment

```zig
test "container deployment simulation" {
    var simulator = fuzzing.createContainerSimulator(std.testing.allocator);
    defer simulator.deinit();
    
    try simulator.activate();
    defer simulator.restore() catch {};
    
    // Simulates Docker/Kubernetes environment where:
    // - CGroup information may be unavailable
    // - Memory monitoring might fail
    // - Container detection could fail
    
    // Your container-aware code should gracefully degrade
}
```

## Tutorial 4: Component Interaction Testing

Test how different components interact under stress.

### Step 1: Race Condition Detection

```zig
test "component interaction fuzzing" {
    var state_fuzzer = fuzzing.createInteractionFuzzer(std.testing.allocator);
    defer state_fuzzer.deinit();
    
    // Configure for 3-second fuzzing session
    state_fuzzer.config.fuzzing_duration_ms = 3000;
    
    try state_fuzzer.start();
    defer state_fuzzer.stop() catch {};
    
    // Simulate concurrent operations
    const thread1 = try std.Thread.spawn(.{}, simulateWorkerConfiguration, .{});
    const thread2 = try std.Thread.spawn(.{}, simulateSchedulerUpdate, .{});
    
    thread1.join();
    thread2.join();
    
    // Check for detected race conditions
    const report = try state_fuzzer.getFuzzingReport(std.testing.allocator);
    defer std.testing.allocator.free(report);
    std.log.info("State Fuzzing Results:\n{s}", .{report});
}

fn simulateWorkerConfiguration() void {
    // Simulate worker configuration changes
    std.time.sleep(1000000); // 1ms
}

fn simulateSchedulerUpdate() void {
    // Simulate scheduler state updates
    std.time.sleep(1000000); // 1ms
}
```

### Step 2: Configuration Boundary Testing

```zig
test "configuration boundary testing" {
    var state_fuzzer = fuzzing.createConfigurationFuzzer(std.testing.allocator);
    defer state_fuzzer.deinit();
    
    try state_fuzzer.start();
    defer state_fuzzer.stop() catch {};
    
    // Test with boundary values
    const test_configs = [_]struct{
        worker_count: u32,
        valid: bool,
    }{
        .{ .worker_count = 0, .valid = false },      // Invalid: zero workers
        .{ .worker_count = 1, .valid = true },       // Valid: single worker
        .{ .worker_count = 1000000, .valid = false }, // Invalid: excessive workers
    };
    
    for (test_configs) |config| {
        // Test configuration validation
        const result = validateWorkerCount(config.worker_count);
        if (result != config.valid) {
            std.log.err("Configuration validation failed for worker_count={}", .{config.worker_count});
        }
    }
}

fn validateWorkerCount(count: u32) bool {
    return count > 0 and count <= 1024; // Reasonable bounds
}
```

## Tutorial 5: Comprehensive Testing Campaign

Put it all together for thorough validation.

### Step 1: Development Testing

For quick CI validation:

```zig
test "development fuzzing campaign" {
    var fuzzer = fuzzing.createDevelopmentFuzzer(std.testing.allocator);
    defer fuzzer.deinit();
    
    // Quick 5-second campaign for CI
    try fuzzer.start();
    defer fuzzer.stop() catch {};
    
    // Verify minimum coverage achieved
    try std.testing.expect(fuzzer.branch_coverage >= 0.40);
    
    const report = try fuzzer.getComprehensiveReport(std.testing.allocator);
    defer std.testing.allocator.free(report);
    std.log.info("Development Fuzzing:\n{s}", .{report});
}
```

### Step 2: Production Validation

For release testing:

```zig
test "production fuzzing validation" {
    var fuzzer = fuzzing.createComprehensiveRobustnessFuzzer(std.testing.allocator);
    defer fuzzer.deinit();
    
    // Comprehensive 30-second campaign
    try fuzzer.start();
    defer fuzzer.stop() catch {};
    
    // Verify target coverage achieved
    try std.testing.expect(fuzzer.branch_coverage >= 0.65);
    
    // Ensure no critical integration failures
    try std.testing.expect(fuzzer.integration_failures.items.len == 0);
    
    const report = try fuzzer.getComprehensiveReport(std.testing.allocator);
    defer std.testing.allocator.free(report);
    std.log.info("Production Validation:\n{s}", .{report});
}
```

### Step 3: Custom Campaign

For specific requirements:

```zig
test "custom fuzzing campaign" {
    const config = fuzzing.ComprehensiveFuzzingConfig{
        .allocator_config = .{
            .failure_rate = 0.25,                    // 25% allocation failures
            .enable_critical_path_targeting = true,  // Focus on critical paths
        },
        .hardware_config = .{
            .simulate_single_core = true,            // Test single-core deployment
            .disable_avx2 = true,                    // Test without AVX2
            .simulate_cgroup_absence = true,         // Test non-container deployment
        },
        .state_config = .{
            .fuzz_worker_counts = true,              // Test worker configuration
            .enable_race_condition_fuzzing = true,   // Detect race conditions
            .test_integer_boundaries = true,         // Test overflow conditions
            .fuzzing_duration_ms = 8000,            // 8-second state fuzzing
        },
        .target_branch_coverage = 0.70,             // 70% coverage target
        .fuzzing_duration_ms = 25000,               // 25-second total campaign
    };
    
    var fuzzer = fuzzing.ComprehensiveFuzzer.init(std.testing.allocator, config);
    defer fuzzer.deinit();
    
    try fuzzer.start();
    defer fuzzer.stop() catch {};
    
    const report = try fuzzer.getComprehensiveReport(std.testing.allocator);
    defer std.testing.allocator.free(report);
    std.log.info("Custom Campaign Results:\n{s}", .{report});
}
```

## Best Practices

### 1. Start Simple

Begin with basic allocation testing before moving to complex scenarios:

```zig
// Start here
var fuzz_alloc = fuzzing.FuzzingAllocator.init(allocator, .{ .failure_rate = 0.1 });

// Progress to this
var fuzz_alloc = fuzzing.createCriticalPathFuzzingAllocator(allocator);

// Eventually use this
var fuzzer = fuzzing.createComprehensiveRobustnessFuzzer(allocator);
```

### 2. Gradual Intensity Increase

Start with low failure rates and increase gradually:

```zig
const test_intensities = [_]f32{ 0.05, 0.10, 0.15, 0.25 };

for (test_intensities) |intensity| {
    var fuzz_alloc = fuzzing.FuzzingAllocator.init(allocator, .{
        .failure_rate = intensity,
    });
    defer fuzz_alloc.deinit();
    
    // Test with current intensity
    // If tests pass, continue to higher intensity
}
```

### 3. Targeted Testing

Focus on specific areas of concern:

```zig
// Focus on memory management
test "memory management robustness" {
    var fuzzer = fuzzing.ComprehensiveFuzzer.init(allocator, .{
        .allocator_config = .{ .failure_rate = 0.30 },
        .hardware_config = .{ .simulate_memory_info_failure = true },
        .state_config = .{ .fuzz_memory_limits = true },
    });
    // ...
}

// Focus on SIMD fallbacks
test "SIMD fallback robustness" {
    var fuzzer = fuzzing.ComprehensiveFuzzer.init(allocator, .{
        .hardware_config = .{
            .disable_sse = true,
            .disable_avx = true,
            .disable_avx2 = true,
        },
    });
    // ...
}
```

### 4. CI Integration

Create different test profiles for different environments:

```zig
// For fast CI (commit hooks)
const ci_fast_config = fuzzing.ComprehensiveFuzzingConfig{
    .fuzzing_duration_ms = 3000,        // 3 seconds
    .target_branch_coverage = 0.40,     // 40% coverage
    .enable_detailed_reporting = false, // Reduce output
};

// For thorough CI (pre-merge)
const ci_thorough_config = fuzzing.ComprehensiveFuzzingConfig{
    .fuzzing_duration_ms = 15000,       // 15 seconds
    .target_branch_coverage = 0.60,     // 60% coverage
    .enable_detailed_reporting = true,  // Full reporting
};

// For release validation
const release_config = fuzzing.ComprehensiveFuzzingConfig{
    .fuzzing_duration_ms = 60000,       // 60 seconds
    .target_branch_coverage = 0.70,     // 70% coverage
    .enable_detailed_reporting = true,  // Full reporting
};
```

## Troubleshooting

### Common Issues

**"Tests take too long"**
```zig
// Reduce duration for faster testing
fuzzer.config.fuzzing_duration_ms = 5000;  // 5 seconds instead of 30
```

**"Low coverage achieved"**
```zig
// Increase failure intensity
fuzzer.config.allocator_config.failure_rate = 0.30;  // 30% instead of 10%
fuzzer.config.fuzzing_duration_ms = 45000;  // 45 seconds instead of 30
```

**"Too many integration failures"**
```zig
// Enable detailed logging to understand failures
fuzzer.config.enable_detailed_reporting = true;
fuzzer.config.allocator_config.enable_detailed_logging = true;
```

**"Memory usage too high"**
```zig
// Reduce memory limits
fuzzer.config.max_memory_usage_mb = 256;  // 256MB instead of 512MB
fuzzer.config.allocator_config.max_allocated_bytes = 128 * 1024 * 1024;  // 128MB
```

### Debugging Tips

1. **Enable Verbose Logging**: Always start with detailed logging when debugging
2. **Incremental Testing**: Test components individually before combining
3. **Coverage Analysis**: Use coverage reports to identify untested code paths
4. **Reproduce Issues**: Use deterministic seeds to reproduce specific failures

## Next Steps

After completing this tutorial:

1. **Integrate with Your Tests**: Add fuzzing to your existing test suite
2. **Customize for Your Needs**: Create application-specific fuzzing scenarios
3. **Monitor Over Time**: Track coverage trends as your code evolves
4. **Contribute**: Help improve the fuzzing framework with new capabilities

The fuzzing framework is designed to grow with your needs. Start simple, learn from the results, and gradually increase the testing intensity as your confidence in the system grows.

Happy fuzzing! 🐛🔍