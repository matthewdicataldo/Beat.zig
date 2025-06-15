const std = @import("std");
const build_config = @import("build_config.zig");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    
    // Auto-detect system configuration
    const auto_config = build_config.detectBuildConfig(b, target, optimize);
    
    // Print configuration summary
    if (b.verbose) {
        build_config.printConfigSummary(auto_config);
    }
    
    // COZ profiling support
    const enable_coz = b.option(bool, "coz", "Enable COZ profiler support") orelse false;
    
    // Main library (modular)
    const lib = b.addStaticLibrary(.{
        .name = "zigpulse",
        .root_source_file = b.path("src/core.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Add auto-configuration to library
    build_config.addBuildOptions(b, lib, auto_config);
    
    b.installArtifact(lib);
    
    // Tests with auto-configuration
    const tests = b.addTest(.{
        .root_source_file = b.path("src/core.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Add auto-configuration to tests
    build_config.addBuildOptions(b, tests, auto_config);
    
    // Set optimal test parallelization
    // Note: Test parallelization will be handled by our enhanced testing framework
    
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
    
    // Benchmarks with auto-configuration
    const benchmark_exe = b.addExecutable(.{
        .name = "benchmark",
        .root_source_file = b.path("benchmark.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    
    // Add auto-configuration optimized for benchmarking
    var bench_config = auto_config;
    bench_config.is_release_fast = true;
    bench_config.optimal_queue_size = auto_config.optimal_workers * 256; // Larger queues for benchmarks
    build_config.addBuildOptions(b, benchmark_exe, bench_config);
    
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
    // Export modules for external use
    _ = b.addModule("beat", .{
        .root_source_file = b.path("beat.zig"),  // Use the bundle file
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
    
    // Build configuration demo (as test)
    const build_config_demo = b.addTest(.{
        .root_source_file = b.path("build_config_demo.zig"),
        .target = target,
        .optimize = .Debug,
    });
    build_config.addBuildOptions(b, build_config_demo, auto_config);
    
    const run_build_config_demo = b.addRunArtifact(build_config_demo);
    const build_config_demo_step = b.step("demo-config", "Run build configuration demo");
    build_config_demo_step.dependOn(&run_build_config_demo.step);
    
    // Comptime work distribution demo
    const comptime_work_demo = b.addTest(.{
        .root_source_file = b.path("comptime_work_demo.zig"),
        .target = target,
        .optimize = .Debug,
    });
    build_config.addBuildOptions(b, comptime_work_demo, auto_config);
    
    const run_comptime_work_demo = b.addRunArtifact(comptime_work_demo);
    const comptime_work_demo_step = b.step("demo-comptime", "Run comptime work distribution demo");
    comptime_work_demo_step.dependOn(&run_comptime_work_demo.step);
    
    // Smart worker selection test
    const smart_worker_test = b.addTest(.{
        .root_source_file = b.path("test_smart_worker_selection.zig"),
        .target = target,
        .optimize = .Debug,
    });
    build_config.addBuildOptions(b, smart_worker_test, auto_config);
    
    const run_smart_worker_test = b.addRunArtifact(smart_worker_test);
    const smart_worker_test_step = b.step("test-smart-worker", "Test smart worker selection");
    smart_worker_test_step.dependOn(&run_smart_worker_test.step);
    
    // Topology-aware work stealing test
    const topology_stealing_test = b.addTest(.{
        .root_source_file = b.path("test_topology_work_stealing.zig"),
        .target = target,
        .optimize = .Debug,
    });
    build_config.addBuildOptions(b, topology_stealing_test, auto_config);
    
    const run_topology_stealing_test = b.addRunArtifact(topology_stealing_test);
    const topology_stealing_test_step = b.step("test-topology-stealing", "Test topology-aware work stealing");
    topology_stealing_test_step.dependOn(&run_topology_stealing_test.step);
    
    // Topology-aware work stealing benchmark
    const topology_stealing_bench = b.addExecutable(.{
        .name = "benchmark_topology_stealing",
        .root_source_file = b.path("benchmark_topology_stealing.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    build_config.addBuildOptions(b, topology_stealing_bench, auto_config);
    
    const run_topology_stealing_bench = b.addRunArtifact(topology_stealing_bench);
    const topology_stealing_bench_step = b.step("bench-topology", "Benchmark topology-aware work stealing performance");
    topology_stealing_bench_step.dependOn(&run_topology_stealing_bench.step);
    
    // Simple topology benchmark
    const simple_topology_bench = b.addExecutable(.{
        .name = "simple_topology_bench",
        .root_source_file = b.path("simple_topology_bench.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    build_config.addBuildOptions(b, simple_topology_bench, auto_config);
    
    const run_simple_topology_bench = b.addRunArtifact(simple_topology_bench);
    const simple_topology_bench_step = b.step("bench-simple", "Simple topology-aware work stealing benchmark");
    simple_topology_bench_step.dependOn(&run_simple_topology_bench.step);
    
    // Topology performance verification
    const verify_topology_perf = b.addExecutable(.{
        .name = "verify_topology_performance",
        .root_source_file = b.path("verify_topology_performance.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    build_config.addBuildOptions(b, verify_topology_perf, auto_config);
    
    const run_verify_topology_perf = b.addRunArtifact(verify_topology_perf);
    const verify_topology_perf_step = b.step("verify-topology", "Verify topology-aware work stealing performance");
    verify_topology_perf_step.dependOn(&run_verify_topology_perf.step);
    
    // Work promotion test
    const work_promotion_test = b.addTest(.{
        .root_source_file = b.path("test_work_promotion.zig"),
        .target = target,
        .optimize = .Debug,
    });
    build_config.addBuildOptions(b, work_promotion_test, auto_config);
    
    const run_work_promotion_test = b.addRunArtifact(work_promotion_test);
    const work_promotion_test_step = b.step("test-promotion", "Test work promotion trigger");
    work_promotion_test_step.dependOn(&run_work_promotion_test.step);
    
    // Auto-configuration integration demo
    const auto_config_integration_demo = b.addTest(.{
        .root_source_file = b.path("auto_config_integration_demo.zig"),
        .target = target,
        .optimize = .Debug,
    });
    build_config.addBuildOptions(b, auto_config_integration_demo, auto_config);
    
    const run_auto_config_integration_demo = b.addRunArtifact(auto_config_integration_demo);
    const auto_config_integration_demo_step = b.step("demo-integration", "Demonstrate auto-configuration integration with One Euro Filter");
    auto_config_integration_demo_step.dependOn(&run_auto_config_integration_demo.step);
    
    // Documentation generation
    const docs = b.addStaticLibrary(.{
        .name = "beat-docs",
        .root_source_file = b.path("src/core.zig"),
        .target = target,
        .optimize = .Debug,
    });
    
    // Add auto-configuration to docs for accurate information
    build_config.addBuildOptions(b, docs, auto_config);
    
    const docs_step = b.step("docs", "Generate documentation");
    const install_docs = b.addInstallDirectory(.{
        .source_dir = docs.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    docs_step.dependOn(&install_docs.step);
    
    // Alternative: Generate docs for the bundle file
    const bundle_docs = b.addStaticLibrary(.{
        .name = "beat-bundle-docs",
        .root_source_file = b.path("beat.zig"),
        .target = target,
        .optimize = .Debug,
    });
    
    const bundle_docs_step = b.step("docs-bundle", "Generate documentation for bundle usage");
    const install_bundle_docs = b.addInstallDirectory(.{
        .source_dir = bundle_docs.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs-bundle",
    });
    bundle_docs_step.dependOn(&install_bundle_docs.step);
    
    // Comprehensive docs generation (includes all modules)
    const comprehensive_docs_step = b.step("docs-all", "Generate comprehensive documentation for all modules");
    comprehensive_docs_step.dependOn(&install_docs.step);
    comprehensive_docs_step.dependOn(&install_bundle_docs.step);
    
    // Open documentation in browser (convenience command)
    const open_docs_step = b.step("docs-open", "Generate and open documentation in browser");
    open_docs_step.dependOn(&install_docs.step);
    
    // Add a step to print documentation locations
    const print_docs_locations = b.addSystemCommand(&[_][]const u8{
        "echo", 
        "📚 Documentation generated at:\n" ++
        "   - Modular API: zig-out/docs/index.html\n" ++
        "   - Bundle API: zig-out/docs-bundle/index.html\n" ++
        "   - Open with: firefox zig-out/docs/index.html"
    });
    docs_step.dependOn(&print_docs_locations.step);
    bundle_docs_step.dependOn(&print_docs_locations.step);
    
    // Thread affinity improvement test
    const thread_affinity_test = b.addTest(.{
        .root_source_file = b.path("test_thread_affinity_improved.zig"),
        .target = target,
        .optimize = .Debug,
    });
    build_config.addBuildOptions(b, thread_affinity_test, auto_config);
    
    const run_thread_affinity_test = b.addRunArtifact(thread_affinity_test);
    const thread_affinity_test_step = b.step("test-affinity", "Test improved thread affinity handling");
    thread_affinity_test_step.dependOn(&run_thread_affinity_test.step);
    
    // Parallel work distribution runtime test
    const parallel_work_test = b.addTest(.{
        .root_source_file = b.path("test_parallel_work_runtime.zig"),
        .target = target,
        .optimize = .Debug,
    });
    build_config.addBuildOptions(b, parallel_work_test, auto_config);
    
    const run_parallel_work_test = b.addRunArtifact(parallel_work_test);
    const parallel_work_test_step = b.step("test-parallel-work", "Test parallel work distribution runtime implementation");
    parallel_work_test_step.dependOn(&run_parallel_work_test.step);
    
    // Enhanced error messages test
    const enhanced_errors_test = b.addTest(.{
        .root_source_file = b.path("test_enhanced_errors.zig"),
        .target = target,
        .optimize = .Debug,
    });
    build_config.addBuildOptions(b, enhanced_errors_test, auto_config);
    
    const run_enhanced_errors_test = b.addRunArtifact(enhanced_errors_test);
    const enhanced_errors_test_step = b.step("test-errors", "Test enhanced error messages with descriptive context");
    enhanced_errors_test_step.dependOn(&run_enhanced_errors_test.step);
    
    // Task fingerprinting integration test
    const fingerprint_test = b.addTest(.{
        .root_source_file = b.path("test_fingerprint_integration.zig"),
        .target = target,
        .optimize = .Debug,
    });
    build_config.addBuildOptions(b, fingerprint_test, auto_config);
    
    const run_fingerprint_test = b.addRunArtifact(fingerprint_test);
    const fingerprint_test_step = b.step("test-fingerprint", "Test task fingerprinting integration with predictive scheduling");
    fingerprint_test_step.dependOn(&run_fingerprint_test.step);
    
    // Enhanced One Euro Filter implementation test
    const enhanced_one_euro_test = b.addTest(.{
        .root_source_file = b.path("test_enhanced_one_euro_filter.zig"),
        .target = target,
        .optimize = .Debug,
    });
    build_config.addBuildOptions(b, enhanced_one_euro_test, auto_config);
    
    const run_enhanced_one_euro_test = b.addRunArtifact(enhanced_one_euro_test);
    const enhanced_one_euro_test_step = b.step("test-enhanced-filter", "Test enhanced One Euro Filter implementation replacing simple averaging");
    enhanced_one_euro_test_step.dependOn(&run_enhanced_one_euro_test.step);
    
    // Advanced performance tracking test
    const advanced_tracking_test = b.addTest(.{
        .root_source_file = b.path("test_advanced_performance_tracking.zig"),
        .target = target,
        .optimize = .Debug,
    });
    build_config.addBuildOptions(b, advanced_tracking_test, auto_config);
    
    const run_advanced_tracking_test = b.addRunArtifact(advanced_tracking_test);
    const advanced_tracking_test_step = b.step("test-advanced-tracking", "Test advanced performance tracking with nanosecond precision and velocity tracking");
    advanced_tracking_test_step.dependOn(&run_advanced_tracking_test.step);
}