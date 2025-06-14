const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    
    // COZ profiling support
    const enable_coz = b.option(bool, "coz", "Enable COZ profiler support") orelse false;
    
    // Main library (modular)
    const lib = b.addStaticLibrary(.{
        .name = "zigpulse",
        .root_source_file = b.path("src/core.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(lib);
    
    // Tests
    const tests = b.addTest(.{
        .root_source_file = b.path("src/core.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
    
    // Benchmarks
    const benchmark_exe = b.addExecutable(.{
        .name = "benchmark",
        .root_source_file = b.path("benchmark.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    b.installArtifact(benchmark_exe);
    
    const run_benchmark = b.addRunArtifact(benchmark_exe);
    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&run_benchmark.step);
    
    // Examples
    const example_exe = b.addExecutable(.{
        .name = "examples",
        .root_source_file = b.path("examples.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(example_exe);
    
    const run_example = b.addRunArtifact(example_exe);
    const example_step = b.step("examples", "Run examples");
    example_step.dependOn(&run_example.step);
    
    // Modular usage example
    const modular_example = b.addExecutable(.{
        .name = "modular_example",
        .root_source_file = b.path("examples/modular_usage.zig"),
        .target = target,
        .optimize = optimize,
    });
    const zigpulse_module = b.addModule("zigpulse", .{
        .root_source_file = b.path("src/core.zig"),
    });
    modular_example.root_module.addImport("zigpulse", zigpulse_module);
    
    const run_modular = b.addRunArtifact(modular_example);
    const modular_step = b.step("example-modular", "Run modular usage example");
    modular_step.dependOn(&run_modular.step);
    
    // Bundle usage example
    const bundle_example = b.addExecutable(.{
        .name = "bundle_example", 
        .root_source_file = b.path("examples/single_file_usage.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    const run_bundle = b.addRunArtifact(bundle_example);
    const bundle_step = b.step("example-bundle", "Run bundle usage example");
    bundle_step.dependOn(&run_bundle.step);
    
    // Bundle file tests
    const bundle_test = b.addTest(.{
        .root_source_file = b.path("zigpulse.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    const run_bundle_test = b.addRunArtifact(bundle_test);
    const bundle_test_step = b.step("test-bundle", "Test the bundle file");
    bundle_test_step.dependOn(&run_bundle_test.step);
    
    // COZ profiling benchmark
    const coz_benchmark = b.addExecutable(.{
        .name = "benchmark_coz",
        .root_source_file = b.path("benchmark_coz.zig"),
        .target = target,
        .optimize = if (enable_coz) .ReleaseSafe else .ReleaseFast,
    });
    
    if (enable_coz) {
        coz_benchmark.root_module.omit_frame_pointer = false;
        // COZ support is compile-time based on build mode
    }
    
    coz_benchmark.root_module.addImport("zigpulse", zigpulse_module);
    
    const run_coz_benchmark = b.addRunArtifact(coz_benchmark);
    const coz_step = b.step("bench-coz", "Run COZ profiling benchmark");
    coz_step.dependOn(&run_coz_benchmark.step);
    
    // Simple test executable
    const test_simple = b.addExecutable(.{
        .name = "test_simple",
        .root_source_file = b.path("test_simple.zig"),
        .target = target,
        .optimize = .Debug,
    });
    test_simple.root_module.addImport("zigpulse", zigpulse_module);
    
    const run_test_simple = b.addRunArtifact(test_simple);
    const test_simple_step = b.step("test-simple", "Run simple test");
    test_simple_step.dependOn(&run_test_simple.step);
    
    // Minimal benchmark test
    const test_minimal = b.addExecutable(.{
        .name = "test_minimal_benchmark",
        .root_source_file = b.path("test_minimal_benchmark.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
    });
    test_minimal.root_module.addImport("zigpulse", zigpulse_module);
    
    const run_test_minimal = b.addRunArtifact(test_minimal);
    const test_minimal_step = b.step("test-minimal", "Run minimal benchmark test");
    test_minimal_step.dependOn(&run_test_minimal.step);
    
    // Stress test
    const test_stress = b.addExecutable(.{
        .name = "test_stress",
        .root_source_file = b.path("test_stress.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
    });
    test_stress.root_module.addImport("zigpulse", zigpulse_module);
    
    const run_test_stress = b.addRunArtifact(test_stress);
    const test_stress_step = b.step("test-stress", "Run stress test");
    test_stress_step.dependOn(&run_test_stress.step);
    
    // TLS overflow test
    const test_tls = b.addExecutable(.{
        .name = "test_tls_overflow",
        .root_source_file = b.path("test_tls_overflow.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
    });
    test_tls.root_module.addImport("zigpulse", zigpulse_module);
    
    const run_test_tls = b.addRunArtifact(test_tls);
    const test_tls_step = b.step("test-tls", "Test TLS overflow issue");
    test_tls_step.dependOn(&run_test_tls.step);
    
    // Intensive TLS test
    const test_tls_intensive = b.addExecutable(.{
        .name = "test_tls_intensive",
        .root_source_file = b.path("test_tls_intensive.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
    });
    test_tls_intensive.root_module.addImport("zigpulse", zigpulse_module);
    
    const run_test_tls_intensive = b.addRunArtifact(test_tls_intensive);
    const test_tls_intensive_step = b.step("test-tls-intensive", "Test TLS overflow with intensive workload");
    test_tls_intensive_step.dependOn(&run_test_tls_intensive.step);
    
    // COZ conditions test
    const test_coz_conditions = b.addExecutable(.{
        .name = "test_coz_conditions",
        .root_source_file = b.path("test_coz_conditions.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
    });
    test_coz_conditions.root_module.addImport("zigpulse", zigpulse_module);
    test_coz_conditions.root_module.omit_frame_pointer = false;
    
    const run_test_coz_conditions = b.addRunArtifact(test_coz_conditions);
    const test_coz_conditions_step = b.step("test-coz-conditions", "Test under COZ benchmark conditions");
    test_coz_conditions_step.dependOn(&run_test_coz_conditions.step);
    
    // CPU detection test
    const test_cpu_detection = b.addExecutable(.{
        .name = "test_cpu_detection",
        .root_source_file = b.path("test_cpu_detection.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
    });
    test_cpu_detection.root_module.addImport("zigpulse", zigpulse_module);
    
    const run_test_cpu_detection = b.addRunArtifact(test_cpu_detection);
    const test_cpu_detection_step = b.step("test-cpu", "Test CPU detection");
    test_cpu_detection_step.dependOn(&run_test_cpu_detection.step);
    
    // Documentation step
    _ = b.step("docs", "Generate documentation");
    // TODO: Add proper documentation generation when available
}