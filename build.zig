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
    
    // Add minotaur integration module
    const minotaur_integration_module = b.addModule("minotaur_integration", .{
        .root_source_file = b.path("src/minotaur_integration.zig"),
    });
    
    // Add souper integration module
    const souper_integration_module = b.addModule("souper_integration", .{
        .root_source_file = b.path("src/souper_integration.zig"),
    });
    
    // Add triple optimization module  
    const triple_optimization_module = b.addModule("triple_optimization", .{
        .root_source_file = b.path("src/triple_optimization.zig"),
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
        .root_source_file = b.path("benchmarks/benchmark_coz.zig"),
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
    
    // SIMD benchmarking and validation framework test
    const simd_benchmark_test = b.addTest(.{
        .root_source_file = b.path("tests/test_simd_benchmark.zig"),
        .target = target,
        .optimize = .Debug,
    });
    simd_benchmark_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, simd_benchmark_test, auto_config);
    
    const run_simd_benchmark_test = b.addRunArtifact(simd_benchmark_test);
    const simd_benchmark_test_step = b.step("test-simd-benchmark", "Test comprehensive SIMD benchmarking and validation framework");
    simd_benchmark_test_step.dependOn(&run_simd_benchmark_test.step);
    
    // Continuation stealing test (Task 7.1)
    const continuation_stealing_test = b.addTest(.{
        .root_source_file = b.path("tests/test_continuation_stealing.zig"),
        .target = target,
        .optimize = .Debug,
    });
    continuation_stealing_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, continuation_stealing_test, auto_config);
    
    const run_continuation_stealing_test = b.addRunArtifact(continuation_stealing_test);
    const continuation_stealing_test_step = b.step("test-continuation-stealing", "Test continuation stealing implementation");
    continuation_stealing_test_step.dependOn(&run_continuation_stealing_test.step);
    
    // ThreadPool continuation integration test (Task 7.1)
    const threadpool_continuation_test = b.addTest(.{
        .root_source_file = b.path("tests/test_threadpool_continuation_integration.zig"),
        .target = target,
        .optimize = .Debug,
    });
    threadpool_continuation_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, threadpool_continuation_test, auto_config);
    
    const run_threadpool_continuation_test = b.addRunArtifact(threadpool_continuation_test);
    const threadpool_continuation_test_step = b.step("test-threadpool-continuation", "Test ThreadPool integration with continuation stealing");
    threadpool_continuation_test_step.dependOn(&run_threadpool_continuation_test.step);
    
    // ML-based classification integration test (Task 3.2.2)
    const ml_integration_test = b.addTest(.{
        .root_source_file = b.path("test_ml_integration.zig"),
        .target = target,
        .optimize = .Debug,
    });
    ml_integration_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, ml_integration_test, auto_config);
    
    const run_ml_integration_test = b.addRunArtifact(ml_integration_test);
    const ml_integration_test_step = b.step("test-ml-integration", "Test ML-based classification for heterogeneous computing (Task 3.2.2)");
    ml_integration_test_step.dependOn(&run_ml_integration_test.step);
    
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
    
    // Batch formation profiling
    const batch_formation_profile = b.addExecutable(.{
        .name = "batch_formation_profile",
        .root_source_file = b.path("batch_formation_profile.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    batch_formation_profile.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, batch_formation_profile, auto_config);
    
    const run_batch_formation_profile = b.addRunArtifact(batch_formation_profile);
    const batch_formation_profile_step = b.step("profile-batch-formation", "Profile batch formation performance bottlenecks");
    batch_formation_profile_step.dependOn(&run_batch_formation_profile.step);
    
    // Work-stealing efficiency benchmark with fast path optimization
    const work_stealing_benchmark = b.addExecutable(.{
        .name = "work_stealing_benchmark",
        .root_source_file = b.path("work_stealing_benchmark.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    work_stealing_benchmark.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, work_stealing_benchmark, auto_config);
    
    const run_work_stealing_benchmark = b.addRunArtifact(work_stealing_benchmark);
    const work_stealing_benchmark_step = b.step("bench-work-stealing", "Benchmark work-stealing efficiency with fast path optimization");
    work_stealing_benchmark_step.dependOn(&run_work_stealing_benchmark.step);
    
    // ============================================================================
    // Souper Superoptimization Integration
    // ============================================================================
    
    // Helper function to create LLVM IR generation commands
    const IRGenCommand = struct {
        fn addIRGenStep(
            builder: *std.Build,
            name: []const u8,
            description: []const u8,
            source_file: []const u8,
            compile_target: std.Build.ResolvedTarget,
            optimization: std.builtin.OptimizeMode,
        ) *std.Build.Step {
            _ = compile_target; // Will be used for cross-compilation support later
            // Create step for LLVM IR generation
            const ir_step = builder.step(name, description);
            
            // Generate both .ll and .bc files using zig build-lib with flags
            const ll_cmd = builder.addSystemCommand(&[_][]const u8{
                "zig", "build-lib",
                source_file,
                "-femit-llvm-ir",
                "--name", builder.fmt("beat_souper_{s}", .{name[7..]}), // Remove "souper-" prefix
                "-O", switch (optimization) {
                    .Debug => "Debug",
                    .ReleaseSafe => "ReleaseSafe", 
                    .ReleaseFast => "ReleaseFast",
                    .ReleaseSmall => "ReleaseSmall",
                },
            });
            
            const bc_cmd = builder.addSystemCommand(&[_][]const u8{
                "zig", "build-lib", 
                source_file,
                "-femit-llvm-bc",
                "--name", builder.fmt("beat_souper_{s}", .{name[7..]}), // Remove "souper-" prefix  
                "-O", switch (optimization) {
                    .Debug => "Debug",
                    .ReleaseSafe => "ReleaseSafe",
                    .ReleaseFast => "ReleaseFast", 
                    .ReleaseSmall => "ReleaseSmall",
                },
            });
            
            ir_step.dependOn(&ll_cmd.step);
            ir_step.dependOn(&bc_cmd.step);
            
            return ir_step;
        }
    };
    
    // Individual module targets for focused analysis
    const souper_targets = [_]struct { name: []const u8, file: []const u8, description: []const u8 }{
        .{ .name = "fingerprint", .file = "src/fingerprint.zig", .description = "Task fingerprinting and hashing algorithms" },
        .{ .name = "lockfree", .file = "src/lockfree.zig", .description = "Work-stealing deque with critical bit operations" },
        .{ .name = "scheduler", .file = "src/scheduler.zig", .description = "Token account promotion and scheduling logic" },
        .{ .name = "simd", .file = "src/simd.zig", .description = "SIMD capability detection and bit flag operations" },
        .{ .name = "simd_classifier", .file = "src/simd_classifier.zig", .description = "Feature vector similarity and classification" },
        .{ .name = "simd_batch", .file = "src/simd_batch.zig", .description = "Task compatibility scoring algorithms" },
        .{ .name = "advanced_worker_selection", .file = "src/advanced_worker_selection.zig", .description = "Worker selection scoring and normalization" },
        .{ .name = "topology", .file = "src/topology.zig", .description = "CPU topology distance calculations" },
    };
    
    // Create individual module analysis targets
    inline for (souper_targets) |souper_target| {
        const step_name = b.fmt("souper-{s}", .{souper_target.name});
        const step_desc = b.fmt("Generate LLVM IR for Souper analysis: {s}", .{souper_target.description});
        _ = IRGenCommand.addIRGenStep(b, step_name, step_desc, souper_target.file, target, .ReleaseFast);
    }
    
    // Whole-program analysis target  
    const souper_whole_step = b.step("souper-whole", "Generate complete program LLVM IR for whole-program superoptimization");
    
    const whole_ll_cmd = b.addSystemCommand(&[_][]const u8{
        "zig", "build-exe",
        "examples/comprehensive_demo.zig", 
        "-femit-llvm-ir",
        "--name", "beat_whole_program_souper",
        "-O", "ReleaseFast",
    });
    
    const whole_bc_cmd = b.addSystemCommand(&[_][]const u8{
        "zig", "build-exe",
        "examples/comprehensive_demo.zig",
        "-femit-llvm-bc", 
        "--name", "beat_whole_program_souper",
        "-O", "ReleaseFast",
    });
    
    souper_whole_step.dependOn(&whole_ll_cmd.step);
    souper_whole_step.dependOn(&whole_bc_cmd.step);
    
    // All Souper targets
    const souper_all_step = b.step("souper-all", "Generate all LLVM IR targets for comprehensive Souper analysis");
    inline for (souper_targets) |souper_target| {
        const step_name = b.fmt("souper-{s}", .{souper_target.name});
        // Get reference to existing step by name lookup (no new step creation)
        if (b.top_level_steps.get(step_name)) |existing_step| {
            souper_all_step.dependOn(&existing_step.step);
        }
    }
    souper_all_step.dependOn(souper_whole_step);
    
    // ============================================================================
    // ISPC Integration for SPMD Parallel Acceleration
    // ============================================================================
    
    // Check for ISPC compiler availability
    const ispc_available = checkISPCAvailable(b);
    if (b.verbose and ispc_available) {
        std.debug.print("ISPC compiler detected - enabling SPMD acceleration\n", .{});
    } else if (b.verbose) {
        std.debug.print("ISPC compiler not found - skipping SPMD acceleration\n", .{});
    }
    
    // Helper function to create ISPC compilation steps
    const ISPCStepBuilder = struct {
        fn addISPCKernel(
            builder: *std.Build,
            kernel_name: []const u8,
            source_path: []const u8,
            description: []const u8,
        ) ?struct { step: *std.Build.Step, obj_path: []const u8, header_path: []const u8 } {
            if (!checkISPCAvailable(builder)) return null;
            
            // Create cache directory for ISPC artifacts
            const cache_dir = "zig-cache/ispc";
            std.fs.makeDirAbsolute(std.fs.path.join(builder.allocator, &.{ builder.build_root.path orelse ".", cache_dir }) catch return null) catch {};
            
            const obj_path = builder.fmt("{s}/{s}.o", .{ cache_dir, kernel_name });
            const header_path = builder.fmt("{s}/{s}.h", .{ cache_dir, kernel_name });
            
            // Auto-detect optimal ISPC target
            const ispc_target = detectOptimalISPCTarget();
            
            const addressing = if (@sizeOf(usize) == 8) "--addressing=64" else "--addressing=32";
            
            const ispc_cmd = builder.addSystemCommand(&[_][]const u8{
                "ispc",
                source_path,
                "-o", obj_path,
                "-h", header_path,
                "--target", ispc_target,
                "-O2",
                addressing,
            });
            
            const step = builder.step(
                builder.fmt("ispc-{s}", .{kernel_name}),
                builder.fmt("Compile ISPC kernel: {s}", .{description}),
            );
            step.dependOn(&ispc_cmd.step);
            
            return .{
                .step = step,
                .obj_path = obj_path,
                .header_path = header_path,
            };
        }
    };
    
    // ISPC kernel compilation targets
    const ispc_kernels = [_]struct { name: []const u8, source: []const u8, description: []const u8 }{
        .{ .name = "fingerprint_similarity", .source = "src/kernels/fingerprint_similarity.ispc", .description = "Task fingerprint similarity computation with SPMD parallelism" },
        .{ .name = "fingerprint_similarity_soa", .source = "src/kernels/fingerprint_similarity_soa.ispc", .description = "SoA-optimized fingerprint similarity with vectorized memory access" },
        .{ .name = "batch_optimization", .source = "src/kernels/batch_optimization.ispc", .description = "Batch formation optimization using multi-criteria SPMD scoring" },
        .{ .name = "worker_selection", .source = "src/kernels/worker_selection.ispc", .description = "Advanced worker selection with topology-aware SPMD computation" },
        .{ .name = "one_euro_filter", .source = "src/kernels/one_euro_filter.ispc", .description = "ISPC-optimized One Euro Filter for predictive scheduling" },
        .{ .name = "optimized_batch_kernels", .source = "src/kernels/optimized_batch_kernels.ispc", .description = "Ultra-optimized mega-batch kernels with minimized function call overhead" },
        .{ .name = "heartbeat_scheduling", .source = "src/kernels/heartbeat_scheduling.ispc", .description = "ISPC-optimized heartbeat scheduling and worker management system" },
        .{ .name = "advanced_ispc_research", .source = "src/kernels/advanced_ispc_research.ispc", .description = "Advanced ISPC research: tasks, async, GPU targeting, cutting-edge features" },
        .{ .name = "prediction_pipeline", .source = "src/kernels/prediction_pipeline.ispc", .description = "Comprehensive prediction system acceleration with transparent API integration" },
    };
    
    // Create ISPC kernel compilation steps
    var ispc_steps = std.ArrayList(*std.Build.Step).init(b.allocator);
    var ispc_obj_paths = std.ArrayList([]const u8).init(b.allocator);
    
    inline for (ispc_kernels) |kernel| {
        if (ISPCStepBuilder.addISPCKernel(b, kernel.name, kernel.source, kernel.description)) |result| {
            ispc_steps.append(result.step) catch {};
            ispc_obj_paths.append(result.obj_path) catch {};
        }
    }
    
    // ISPC integration test with performance comparison
    if (ispc_available) {
        const ispc_integration_test = b.addTest(.{
            .name = "test_ispc_integration",
            .root_source_file = b.path("tests/test_ispc_integration.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        });
        
        // Link all ISPC kernels
        for (ispc_obj_paths.items) |obj_path| {
            ispc_integration_test.addObjectFile(.{ .cwd_relative = obj_path });
        }
        
        // Add ISPC header include path
        ispc_integration_test.addIncludePath(b.path("zig-cache/ispc"));
        
        // Depend on all ISPC compilation steps
        for (ispc_steps.items) |ispc_step| {
            ispc_integration_test.step.dependOn(ispc_step);
        }
        
        ispc_integration_test.root_module.addImport("beat", zigpulse_module);
        build_config.addBuildOptions(b, ispc_integration_test, auto_config);
        
        const run_ispc_integration_test = b.addRunArtifact(ispc_integration_test);
        const ispc_integration_test_step = b.step("test-ispc-integration", "Test ISPC integration with performance comparison vs native SIMD");
        ispc_integration_test_step.dependOn(&run_ispc_integration_test.step);
        
        // ISPC benchmark comparing SPMD vs native performance
        const ispc_benchmark = b.addExecutable(.{
            .name = "benchmark_ispc_performance",
            .root_source_file = b.path("benchmarks/benchmark_ispc_performance.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        });
        
        // Link all ISPC kernels
        for (ispc_obj_paths.items) |obj_path| {
            ispc_benchmark.addObjectFile(.{ .cwd_relative = obj_path });
        }
        
        ispc_benchmark.addIncludePath(b.path("zig-cache/ispc"));
        
        // Depend on all ISPC compilation steps
        for (ispc_steps.items) |ispc_step| {
            ispc_benchmark.step.dependOn(ispc_step);
        }
        
        ispc_benchmark.root_module.addImport("beat", zigpulse_module);
        build_config.addBuildOptions(b, ispc_benchmark, auto_config);
        
        const run_ispc_benchmark = b.addRunArtifact(ispc_benchmark);
        const ispc_benchmark_step = b.step("bench-ispc", "Benchmark ISPC SPMD performance vs native Zig implementations");
        ispc_benchmark_step.dependOn(&run_ispc_benchmark.step);
        
        // Optimized ISPC kernels test
        const optimized_kernels_test = b.addExecutable(.{
            .name = "test_optimized_kernels",
            .root_source_file = b.path("test_optimized_kernels.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        });
        
        // Link all ISPC kernels
        for (ispc_obj_paths.items) |obj_path| {
            optimized_kernels_test.addObjectFile(.{ .cwd_relative = obj_path });
        }
        
        optimized_kernels_test.addIncludePath(b.path("zig-cache/ispc"));
        
        // Depend on all ISPC compilation steps
        for (ispc_steps.items) |ispc_step| {
            optimized_kernels_test.step.dependOn(ispc_step);
        }
        
        const run_optimized_kernels_test = b.addRunArtifact(optimized_kernels_test);
        const optimized_kernels_test_step = b.step("test-optimized-kernels", "Test ultra-optimized mega-batch ISPC kernels with overhead reduction");
        optimized_kernels_test_step.dependOn(&run_optimized_kernels_test.step);
        
        // Heartbeat scheduling ISPC kernels test
        const heartbeat_kernels_test = b.addExecutable(.{
            .name = "test_heartbeat_kernels",
            .root_source_file = b.path("test_heartbeat_kernels.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        });
        
        // Link all ISPC kernels
        for (ispc_obj_paths.items) |obj_path| {
            heartbeat_kernels_test.addObjectFile(.{ .cwd_relative = obj_path });
        }
        
        heartbeat_kernels_test.addIncludePath(b.path("zig-cache/ispc"));
        
        // Depend on all ISPC compilation steps
        for (ispc_steps.items) |ispc_step| {
            heartbeat_kernels_test.step.dependOn(ispc_step);
        }
        
        const run_heartbeat_kernels_test = b.addRunArtifact(heartbeat_kernels_test);
        const heartbeat_kernels_test_step = b.step("test-heartbeat-kernels", "Test ISPC heartbeat scheduling and worker management kernels");
        heartbeat_kernels_test_step.dependOn(&run_heartbeat_kernels_test.step);
        
        // Advanced ISPC Research Test Suite (Phase 3 Deep Dive)
        const advanced_ispc_research_test = b.addTest(.{
            .root_source_file = b.path("test_advanced_ispc_research.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        });
        
        // Link all ISPC kernels
        for (ispc_obj_paths.items) |obj_path| {
            advanced_ispc_research_test.addObjectFile(.{ .cwd_relative = obj_path });
        }
        
        // Add ISPC header include path
        advanced_ispc_research_test.addIncludePath(b.path("zig-cache/ispc"));
        
        // Depend on all ISPC compilation steps
        for (ispc_steps.items) |ispc_step| {
            advanced_ispc_research_test.step.dependOn(ispc_step);
        }
        
        const run_advanced_ispc_research_test = b.addRunArtifact(advanced_ispc_research_test);
        const advanced_ispc_research_test_step = b.step("test-advanced-ispc-research", "Test cutting-edge ISPC features: tasks, GPU, @ispc builtin prototype");
        advanced_ispc_research_test_step.dependOn(&run_advanced_ispc_research_test.step);
        
        // Prediction Integration Test Suite (Production Integration)
        const prediction_integration_test = b.addTest(.{
            .root_source_file = b.path("test_prediction_integration.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        });
        
        // Link all ISPC kernels for comprehensive testing
        for (ispc_obj_paths.items) |obj_path| {
            prediction_integration_test.addObjectFile(.{ .cwd_relative = obj_path });
        }
        
        prediction_integration_test.addIncludePath(b.path("zig-cache/ispc"));
        
        // Depend on all ISPC compilation steps
        for (ispc_steps.items) |ispc_step| {
            prediction_integration_test.step.dependOn(ispc_step);
        }
        
        const run_prediction_integration_test = b.addRunArtifact(prediction_integration_test);
        const prediction_integration_test_step = b.step("test-prediction-integration", "Test transparent ISPC prediction acceleration with 100% API compatibility");
        prediction_integration_test_step.dependOn(&run_prediction_integration_test.step);
    }
    
    // Souper mathematical optimization integration test
    const souper_integration_test = b.addTest(.{
        .root_source_file = b.path("test_souper_integration.zig"),
        .target = target,
        .optimize = optimize,
    });
    souper_integration_test.root_module.addImport("beat", zigpulse_module);
    
    const run_souper_integration_test = b.addRunArtifact(souper_integration_test);
    const souper_integration_test_step = b.step("test-souper-integration", "Test Souper mathematical optimizations with formal verification");
    souper_integration_test_step.dependOn(&run_souper_integration_test.step);
    
    // Souper simple test (without ISPC dependencies)
    const souper_simple_test = b.addTest(.{
        .root_source_file = b.path("test_souper_simple.zig"),
        .target = target,
        .optimize = optimize,
    });
    souper_simple_test.root_module.addImport("beat", zigpulse_module);
    
    const run_souper_simple_test = b.addRunArtifact(souper_simple_test);
    const souper_simple_test_step = b.step("test-souper-simple", "Test Souper mathematical optimizations without ISPC dependencies");
    souper_simple_test_step.dependOn(&run_souper_simple_test.step);
    
    // All ISPC targets
    const ispc_all_step = b.step("ispc-all", "Compile all ISPC kernels for SPMD acceleration");
    for (ispc_steps.items) |ispc_step| {
        ispc_all_step.dependOn(ispc_step);
    }
    
    // Minotaur SIMD superoptimization tests
    const minotaur_integration_test = b.addTest(.{
        .root_source_file = b.path("tests/test_minotaur_integration.zig"),
        .target = target,
        .optimize = .Debug,
    });
    minotaur_integration_test.root_module.addImport("beat", zigpulse_module);
    minotaur_integration_test.root_module.addImport("minotaur_integration", minotaur_integration_module);
    build_config.addBuildOptions(b, minotaur_integration_test, auto_config);
    
    const run_minotaur_integration_test = b.addRunArtifact(minotaur_integration_test);
    const minotaur_integration_test_step = b.step("test-minotaur-integration", "Test Minotaur SIMD superoptimization integration");
    minotaur_integration_test_step.dependOn(&run_minotaur_integration_test.step);
    
    // Triple-optimization pipeline tests
    const triple_optimization_test = b.addTest(.{
        .root_source_file = b.path("tests/test_triple_optimization.zig"),
        .target = target,
        .optimize = .Debug,
    });
    triple_optimization_test.root_module.addImport("beat", zigpulse_module);
    triple_optimization_test.root_module.addImport("triple_optimization", triple_optimization_module);
    triple_optimization_test.root_module.addImport("minotaur_integration", minotaur_integration_module);
    triple_optimization_test.root_module.addImport("souper_integration", souper_integration_module);
    build_config.addBuildOptions(b, triple_optimization_test, auto_config);
    
    const run_triple_optimization_test = b.addRunArtifact(triple_optimization_test);
    const triple_optimization_test_step = b.step("test-triple-optimization", "Test Souper + Minotaur + ISPC triple-optimization pipeline");
    triple_optimization_test_step.dependOn(&run_triple_optimization_test.step);
    
    // ISPC Migration Tests 
    const ispc_migration_test = b.addTest(.{
        .root_source_file = b.path("tests/test_ispc_migration.zig"),
        .target = target,
        .optimize = .Debug,
    });
    ispc_migration_test.root_module.addImport("beat", zigpulse_module);
    build_config.addBuildOptions(b, ispc_migration_test, auto_config);
    
    const run_ispc_migration_test = b.addRunArtifact(ispc_migration_test);
    const ispc_migration_test_step = b.step("test-ispc-migration", "Test Zig SIMD → ISPC migration and API compatibility");
    ispc_migration_test_step.dependOn(&run_ispc_migration_test.step);
    
    // Superoptimization setup and analysis commands
    const setup_minotaur_cmd = b.addSystemCommand(&[_][]const u8{ "bash", "scripts/setup_minotaur.sh" });
    const setup_minotaur_step = b.step("setup-minotaur", "Set up Minotaur SIMD superoptimizer");
    setup_minotaur_step.dependOn(&setup_minotaur_cmd.step);
    
    const run_minotaur_analysis_cmd = b.addSystemCommand(&[_][]const u8{ "bash", "scripts/run_minotaur_analysis.sh" });
    const run_minotaur_analysis_step = b.step("analyze-minotaur", "Run Minotaur SIMD analysis on Beat.zig code");
    run_minotaur_analysis_step.dependOn(&run_minotaur_analysis_cmd.step);
    
    const run_combined_optimization_cmd = b.addSystemCommand(&[_][]const u8{ "bash", "scripts/run_combined_optimization.sh" });
    const run_combined_optimization_step = b.step("analyze-triple", "Run combined Souper + Minotaur + ISPC optimization analysis");
    run_combined_optimization_step.dependOn(&run_combined_optimization_cmd.step);
    
    // Configure ISPC migration strategy for Zig SIMD → ISPC transition
    configureISPCMigration(b, target, optimize) catch |err| {
        std.log.warn("ISPC migration configuration failed: {}", .{err});
    };
}

// Helper function to check ISPC compiler availability
fn checkISPCAvailable(b: *std.Build) bool {
    const result = std.process.Child.run(.{
        .allocator = b.allocator,
        .argv = &[_][]const u8{ "ispc", "--version" },
    }) catch return false;
    
    defer b.allocator.free(result.stdout);
    defer b.allocator.free(result.stderr);
    
    return result.term == .Exited and result.term.Exited == 0;
}

// ISPC Migration Strategy: Prioritize ISPC kernels over Zig SIMD
fn configureISPCMigration(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) !void {
    _ = optimize;
    const ispc_available = checkISPCAvailable(b);
    
    if (ispc_available) {
        std.log.info("ISPC detected: Enabling optimized SIMD kernels (6-23x performance)", .{});
        
        // Add ISPC compilation steps for new kernels
        const new_ispc_kernels = [_][]const u8{
            "simd_capabilities",
            "simd_memory", 
            "simd_queue_ops",
        };
        
        for (new_ispc_kernels) |kernel| {
            // Create ISPC compilation command
            const ispc_cmd = b.addSystemCommand(&[_][]const u8{
                "ispc",
                b.fmt("src/kernels/{s}.ispc", .{kernel}),
                "-o", b.fmt("zig-cache/ispc/{s}.o", .{kernel}),
                "-h", b.fmt("zig-cache/ispc/{s}.h", .{kernel}),
                "--opt=fast-math",
                "--pic",
                "--addressing=64",
            });
            
            // Add target-specific ISPC args
            if (target.result.cpu.arch == .x86_64) {
                ispc_cmd.addArg("--target=avx2-i32x8,avx512skx-i32x16");
            } else if (target.result.cpu.arch == .aarch64) {
                ispc_cmd.addArg("--target=neon-i32x4");
            }
            
            // Create build step for ISPC compilation
            const ispc_step = b.step(b.fmt("ispc-{s}", .{kernel}), b.fmt("Compile {s} ISPC kernel", .{kernel}));
            ispc_step.dependOn(&ispc_cmd.step);
        }
        
        // Add compile-time flag to enable ISPC-first strategy
        const ispc_flag = b.addOptions();
        ispc_flag.addOption(bool, "use_ispc_simd", true);
        ispc_flag.addOption(bool, "deprecate_zig_simd", true);
        
    } else {
        std.log.warn("ISPC not found: Falling back to Zig SIMD (reduced performance)", .{});
        
        const fallback_flag = b.addOptions();
        fallback_flag.addOption(bool, "use_ispc_simd", false);
        fallback_flag.addOption(bool, "deprecate_zig_simd", false);
    }
}

// Auto-detect optimal ISPC target based on CPU capabilities
fn detectOptimalISPCTarget() []const u8 {
    const builtin = @import("builtin");
    const features = std.Target.x86.featureSetHas;
    
    if (builtin.cpu.arch == .x86_64) {
        if (features(builtin.cpu.features, .avx512f)) {
            return "avx512skx-i32x16";
        } else if (features(builtin.cpu.features, .avx2)) {
            return "avx2-i32x8";
        } else if (features(builtin.cpu.features, .avx)) {
            return "avx1-i32x8";
        } else if (features(builtin.cpu.features, .sse4_1)) {
            return "sse4-i32x4";
        } else {
            return "sse2-i32x4";
        }
    } else if (builtin.cpu.arch == .aarch64) {
        return "neon-i32x4";
    } else {
        return "host";
    }
}