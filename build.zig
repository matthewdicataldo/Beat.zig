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
        .root_source_file = b.path("benchmarks/benchmark.zig"),
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
    
    // Note: Legacy test executables removed - all tests now run via main test suite
    
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
        .root_source_file = b.path("tests/test_smart_worker_selection.zig"),
        .target = target,
        .optimize = .Debug,
    });
    smart_worker_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, smart_worker_test, auto_config);
    
    const run_smart_worker_test = b.addRunArtifact(smart_worker_test);
    const smart_worker_test_step = b.step("test-smart-worker", "Test smart worker selection");
    smart_worker_test_step.dependOn(&run_smart_worker_test.step);
    
    // Topology-aware work stealing test
    const topology_stealing_test = b.addTest(.{
        .root_source_file = b.path("tests/test_topology_work_stealing.zig"),
        .target = target,
        .optimize = .Debug,
    });
    topology_stealing_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, topology_stealing_test, auto_config);
    
    const run_topology_stealing_test = b.addRunArtifact(topology_stealing_test);
    const topology_stealing_test_step = b.step("test-topology-stealing", "Test topology-aware work stealing");
    topology_stealing_test_step.dependOn(&run_topology_stealing_test.step);
    
    // Note: Legacy benchmark removed - use benchmarks/benchmark_topology_aware.zig via bench command
    
    // Note: Legacy simple topology benchmark removed - use benchmarks/ directory
    
    // Note: Legacy verification removed - performance is verified through test suite
    
    // Work promotion test
    const work_promotion_test = b.addTest(.{
        .root_source_file = b.path("tests/test_work_promotion.zig"),
        .target = target,
        .optimize = .Debug,
    });
    work_promotion_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, work_promotion_test, auto_config);
    
    const run_work_promotion_test = b.addRunArtifact(work_promotion_test);
    const work_promotion_test_step = b.step("test-promotion", "Test work promotion trigger");
    work_promotion_test_step.dependOn(&run_work_promotion_test.step);
    
    // Memory pressure monitoring test
    const memory_pressure_test = b.addTest(.{
        .root_source_file = b.path("src/memory_pressure.zig"),
        .target = target,
        .optimize = .Debug,
    });
    build_config.addBuildOptions(b, memory_pressure_test, auto_config);
    
    const run_memory_pressure_test = b.addRunArtifact(memory_pressure_test);
    const memory_pressure_test_step = b.step("test-memory-pressure", "Test memory pressure monitoring and scheduling adaptation");
    memory_pressure_test_step.dependOn(&run_memory_pressure_test.step);
    
    // Development mode configuration test
    const development_mode_test = b.addTest(.{
        .root_source_file = b.path("tests/test_development_mode.zig"),
        .target = target,
        .optimize = .Debug,
    });
    development_mode_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, development_mode_test, auto_config);
    
    const run_development_mode_test = b.addRunArtifact(development_mode_test);
    const development_mode_test_step = b.step("test-development-mode", "Test development mode configuration features");
    development_mode_test_step.dependOn(&run_development_mode_test.step);
    
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
        .root_source_file = b.path("tests/test_thread_affinity_improved.zig"),
        .target = target,
        .optimize = .Debug,
    });
    thread_affinity_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, thread_affinity_test, auto_config);
    
    const run_thread_affinity_test = b.addRunArtifact(thread_affinity_test);
    const thread_affinity_test_step = b.step("test-affinity", "Test improved thread affinity handling");
    thread_affinity_test_step.dependOn(&run_thread_affinity_test.step);
    
    // Parallel work distribution runtime test
    const parallel_work_test = b.addTest(.{
        .root_source_file = b.path("tests/test_parallel_work_runtime.zig"),
        .target = target,
        .optimize = .Debug,
    });
    parallel_work_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, parallel_work_test, auto_config);
    
    const run_parallel_work_test = b.addRunArtifact(parallel_work_test);
    const parallel_work_test_step = b.step("test-parallel-work", "Test parallel work distribution runtime implementation");
    parallel_work_test_step.dependOn(&run_parallel_work_test.step);
    
    // Enhanced error messages test
    const enhanced_errors_test = b.addTest(.{
        .root_source_file = b.path("tests/test_enhanced_errors.zig"),
        .target = target,
        .optimize = .Debug,
    });
    enhanced_errors_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, enhanced_errors_test, auto_config);
    
    const run_enhanced_errors_test = b.addRunArtifact(enhanced_errors_test);
    const enhanced_errors_test_step = b.step("test-errors", "Test enhanced error messages with descriptive context");
    enhanced_errors_test_step.dependOn(&run_enhanced_errors_test.step);
    
    // Task fingerprinting integration test
    const fingerprint_test = b.addTest(.{
        .root_source_file = b.path("tests/test_fingerprint_simple.zig"),
        .target = target,
        .optimize = .Debug,
    });
    fingerprint_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, fingerprint_test, auto_config);
    
    const run_fingerprint_test = b.addRunArtifact(fingerprint_test);
    const fingerprint_test_step = b.step("test-fingerprint", "Test task fingerprinting integration with predictive scheduling");
    fingerprint_test_step.dependOn(&run_fingerprint_test.step);
    
    // Enhanced One Euro Filter implementation test
    const enhanced_one_euro_test = b.addTest(.{
        .root_source_file = b.path("tests/test_enhanced_one_euro_filter.zig"),
        .target = target,
        .optimize = .Debug,
    });
    enhanced_one_euro_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, enhanced_one_euro_test, auto_config);
    
    const run_enhanced_one_euro_test = b.addRunArtifact(enhanced_one_euro_test);
    const enhanced_one_euro_test_step = b.step("test-enhanced-filter", "Test enhanced One Euro Filter implementation replacing simple averaging");
    enhanced_one_euro_test_step.dependOn(&run_enhanced_one_euro_test.step);
    
    // Advanced performance tracking test
    const advanced_tracking_test = b.addTest(.{
        .root_source_file = b.path("tests/test_advanced_performance_tracking.zig"),
        .target = target,
        .optimize = .Debug,
    });
    advanced_tracking_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, advanced_tracking_test, auto_config);
    
    const run_advanced_tracking_test = b.addRunArtifact(advanced_tracking_test);
    const advanced_tracking_test_step = b.step("test-advanced-tracking", "Test advanced performance tracking with nanosecond precision and velocity tracking");
    advanced_tracking_test_step.dependOn(&run_advanced_tracking_test.step);
    
    // Multi-factor confidence model test
    const multi_factor_confidence_test = b.addTest(.{
        .root_source_file = b.path("tests/test_multi_factor_confidence.zig"),
        .target = target,
        .optimize = .Debug,
    });
    multi_factor_confidence_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, multi_factor_confidence_test, auto_config);
    
    const run_multi_factor_confidence_test = b.addRunArtifact(multi_factor_confidence_test);
    const multi_factor_confidence_test_step = b.step("test-multi-factor-confidence", "Test multi-factor confidence model for enhanced scheduling decisions");
    multi_factor_confidence_test_step.dependOn(&run_multi_factor_confidence_test.step);
    
    // Intelligent decision framework test
    const intelligent_decision_test = b.addTest(.{
        .root_source_file = b.path("tests/test_intelligent_decision_framework.zig"),
        .target = target,
        .optimize = .Debug,
    });
    intelligent_decision_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, intelligent_decision_test, auto_config);
    
    const run_intelligent_decision_test = b.addRunArtifact(intelligent_decision_test);
    const intelligent_decision_test_step = b.step("test-intelligent-decision", "Test intelligent decision framework for confidence-based scheduling");
    intelligent_decision_test_step.dependOn(&run_intelligent_decision_test.step);
    
    // Predictive token accounting test
    const predictive_accounting_test = b.addTest(.{
        .root_source_file = b.path("tests/test_predictive_accounting.zig"),
        .target = target,
        .optimize = .Debug,
    });
    predictive_accounting_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, predictive_accounting_test, auto_config);
    
    const run_predictive_accounting_test = b.addRunArtifact(predictive_accounting_test);
    const predictive_accounting_test_step = b.step("test-predictive-accounting", "Test predictive token accounting with confidence-based promotion decisions");
    predictive_accounting_test_step.dependOn(&run_predictive_accounting_test.step);
    
    // Advanced worker selection test (simplified version)
    const advanced_worker_selection_test = b.addTest(.{
        .root_source_file = b.path("tests/test_advanced_worker_selection_simple.zig"),
        .target = target,
        .optimize = .Debug,
    });
    advanced_worker_selection_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, advanced_worker_selection_test, auto_config);
    
    const run_advanced_worker_selection_test = b.addRunArtifact(advanced_worker_selection_test);
    const advanced_worker_selection_test_step = b.step("test-advanced-worker-selection", "Test advanced worker selection algorithm with multi-criteria optimization");
    advanced_worker_selection_test_step.dependOn(&run_advanced_worker_selection_test.step);
    
    // SIMD foundation test
    const simd_foundation_test = b.addTest(.{
        .root_source_file = b.path("tests/test_simd_foundation.zig"),
        .target = target,
        .optimize = .Debug,
    });
    simd_foundation_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, simd_foundation_test, auto_config);
    
    const run_simd_foundation_test = b.addRunArtifact(simd_foundation_test);
    const simd_foundation_test_step = b.step("test-simd", "Test SIMD capability detection, registry, and enhanced fingerprinting");
    simd_foundation_test_step.dependOn(&run_simd_foundation_test.step);
    
    // SIMD worker integration test
    const simd_worker_integration_test = b.addTest(.{
        .root_source_file = b.path("tests/test_simd_worker_integration.zig"),
        .target = target,
        .optimize = .Debug,
    });
    simd_worker_integration_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, simd_worker_integration_test, auto_config);
    
    const run_simd_worker_integration_test = b.addRunArtifact(simd_worker_integration_test);
    const simd_worker_integration_test_step = b.step("test-simd-integration", "Test SIMD-aware worker selection integration");
    simd_worker_integration_test_step.dependOn(&run_simd_worker_integration_test.step);
    
    // SIMD batch architecture test
    const simd_batch_architecture_test = b.addTest(.{
        .root_source_file = b.path("tests/test_simd_batch_architecture.zig"),
        .target = target,
        .optimize = .Debug,
    });
    simd_batch_architecture_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, simd_batch_architecture_test, auto_config);
    
    const run_simd_batch_architecture_test = b.addRunArtifact(simd_batch_architecture_test);
    const simd_batch_architecture_test_step = b.step("test-simd-batch", "Test SIMD task batch architecture with type-safe vectorization");
    simd_batch_architecture_test_step.dependOn(&run_simd_batch_architecture_test.step);
    
    // SIMD queue operations test
    const simd_queue_operations_test = b.addTest(.{
        .root_source_file = b.path("tests/test_simd_queue_operations.zig"),
        .target = target,
        .optimize = .Debug,
    });
    simd_queue_operations_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, simd_queue_operations_test, auto_config);
    
    const run_simd_queue_operations_test = b.addRunArtifact(simd_queue_operations_test);
    const simd_queue_operations_test_step = b.step("test-simd-queue", "Test SIMD vectorized queue operations and work-stealing integration");
    simd_queue_operations_test_step.dependOn(&run_simd_queue_operations_test.step);
    
    // SIMD classification and batch formation test
    const simd_classification_test = b.addTest(.{
        .root_source_file = b.path("tests/test_simd_classification.zig"),
        .target = target,
        .optimize = .Debug,
    });
    simd_classification_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, simd_classification_test, auto_config);
    
    const run_simd_classification_test = b.addRunArtifact(simd_classification_test);
    const simd_classification_test_step = b.step("test-simd-classification", "Test SIMD task classification and intelligent batch formation system");
    simd_classification_test_step.dependOn(&run_simd_classification_test.step);
    
    // Advanced scheduling benchmark
    const advanced_scheduling_benchmark = b.addExecutable(.{
        .name = "benchmark_advanced_scheduling",
        .root_source_file = b.path("benchmark_advanced_scheduling.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    build_config.addBuildOptions(b, advanced_scheduling_benchmark, auto_config);
    
    const run_advanced_scheduling_benchmark = b.addRunArtifact(advanced_scheduling_benchmark);
    const advanced_scheduling_benchmark_step = b.step("bench-advanced-scheduling", "Benchmark advanced predictive scheduling performance improvements");
    advanced_scheduling_benchmark_step.dependOn(&run_advanced_scheduling_benchmark.step);
    
    // Simple scheduling benchmark
    const simple_scheduling_benchmark = b.addExecutable(.{
        .name = "benchmark_simple_scheduling",
        .root_source_file = b.path("benchmark_simple_scheduling.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    build_config.addBuildOptions(b, simple_scheduling_benchmark, auto_config);
    
    const run_simple_scheduling_benchmark = b.addRunArtifact(simple_scheduling_benchmark);
    const simple_scheduling_benchmark_step = b.step("bench-simple-scheduling", "Simple focused benchmark for advanced scheduling features");
    simple_scheduling_benchmark_step.dependOn(&run_simple_scheduling_benchmark.step);
    
    // Prediction accuracy micro-benchmark
    const prediction_accuracy_benchmark = b.addExecutable(.{
        .name = "benchmark_prediction_accuracy",
        .root_source_file = b.path("benchmark_prediction_accuracy.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    build_config.addBuildOptions(b, prediction_accuracy_benchmark, auto_config);
    
    const run_prediction_accuracy_benchmark = b.addRunArtifact(prediction_accuracy_benchmark);
    const prediction_accuracy_benchmark_step = b.step("bench-prediction-accuracy", "Micro-benchmarks for prediction accuracy measurement (Task 2.5.1)");
    prediction_accuracy_benchmark_step.dependOn(&run_prediction_accuracy_benchmark.step);
    
    // A/B testing framework
    const ab_testing_framework = b.addExecutable(.{
        .name = "ab_testing_framework",
        .root_source_file = b.path("ab_testing_framework.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    build_config.addBuildOptions(b, ab_testing_framework, auto_config);
    
    const run_ab_testing_framework = b.addRunArtifact(ab_testing_framework);
    const ab_testing_framework_step = b.step("ab-test", "A/B testing infrastructure for scheduling comparison (Task 2.5.1.2)");
    ab_testing_framework_step.dependOn(&run_ab_testing_framework.step);
    
    // Enhanced COZ profiler benchmark
    const coz_enhanced_benchmark = b.addExecutable(.{
        .name = "benchmark_coz_enhanced",
        .root_source_file = b.path("benchmark_coz_enhanced.zig"),
        .target = target,
        .optimize = if (enable_coz) .ReleaseSafe else .ReleaseFast,
    });
    build_config.addBuildOptions(b, coz_enhanced_benchmark, auto_config);
    
    if (enable_coz) {
        coz_enhanced_benchmark.root_module.omit_frame_pointer = false;
    }
    
    const run_coz_enhanced_benchmark = b.addRunArtifact(coz_enhanced_benchmark);
    const coz_enhanced_benchmark_step = b.step("bench-coz-enhanced", "Enhanced COZ profiler integration benchmark (Task 2.5.1.3)");
    coz_enhanced_benchmark_step.dependOn(&run_coz_enhanced_benchmark.step);
    
    // Fingerprint cache optimization
    const cache_optimization = b.addExecutable(.{
        .name = "fingerprint_cache_optimization",
        .root_source_file = b.path("fingerprint_cache_optimization.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    build_config.addBuildOptions(b, cache_optimization, auto_config);
    
    const run_cache_optimization = b.addRunArtifact(cache_optimization);
    const cache_optimization_step = b.step("test-cache-optimization", "Test prediction lookup caching optimization");
    cache_optimization_step.dependOn(&run_cache_optimization.step);
    
    // Optimization validation framework
    const optimization_validation = b.addExecutable(.{
        .name = "optimization_validation",
        .root_source_file = b.path("optimization_validation.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    build_config.addBuildOptions(b, optimization_validation, auto_config);
    
    const run_optimization_validation = b.addRunArtifact(optimization_validation);
    const optimization_validation_step = b.step("validate-optimizations", "Validate optimization performance improvements");
    optimization_validation_step.dependOn(&run_optimization_validation.step);
    
    // Worker selection fast path optimization
    const worker_selection_optimization = b.addExecutable(.{
        .name = "worker_selection_optimization",
        .root_source_file = b.path("worker_selection_optimization.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    build_config.addBuildOptions(b, worker_selection_optimization, auto_config);
    
    const run_worker_selection_optimization = b.addRunArtifact(worker_selection_optimization);
    const worker_selection_optimization_step = b.step("test-worker-selection-optimization", "Test worker selection fast path optimization to reduce 120.6x overhead");
    worker_selection_optimization_step.dependOn(&run_worker_selection_optimization.step);
    
    // Integrated worker selection optimization test
    const integrated_worker_optimization = b.addExecutable(.{
        .name = "optimized_worker_selection_integration",
        .root_source_file = b.path("optimized_worker_selection_integration.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    build_config.addBuildOptions(b, integrated_worker_optimization, auto_config);
    
    const run_integrated_worker_optimization = b.addRunArtifact(integrated_worker_optimization);
    const integrated_worker_optimization_step = b.step("test-integrated-worker-optimization", "Test integrated worker selection optimization for direct ThreadPool integration");
    integrated_worker_optimization_step.dependOn(&run_integrated_worker_optimization.step);
    
    // Fixed cache optimization test
    const fixed_cache_optimization = b.addExecutable(.{
        .name = "fixed_cache_optimization",
        .root_source_file = b.path("fixed_cache_optimization.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    build_config.addBuildOptions(b, fixed_cache_optimization, auto_config);
    
    const run_fixed_cache_optimization = b.addRunArtifact(fixed_cache_optimization);
    const fixed_cache_optimization_step = b.step("test-fixed-cache-optimization", "Test fixed prediction lookup caching with proper memory management");
    fixed_cache_optimization_step.dependOn(&run_fixed_cache_optimization.step);
}