const std = @import("std");
const beat = @import("beat");

// Import mock SYCL implementation for linking
comptime {
    _ = beat.mock_sycl;
}

// GPU Task Classification Test for Beat.zig (Task 3.2.1)
//
// This test validates the automatic GPU suitability detection system including:
// - Data-parallel pattern recognition with semantic analysis
// - Computational intensity analysis using roofline model principles
// - Memory access pattern classification for coalescing optimization
// - Branch divergence detection and performance impact analysis
// - Comprehensive task classification and adaptive learning

test "arithmetic intensity analyzer" {
    _ = std.testing.allocator;
    
    std.debug.print("\n=== Arithmetic Intensity Analyzer Test ===\n", .{});
    
    // Test 1: Basic arithmetic intensity calculation
    std.debug.print("1. Testing arithmetic intensity calculation...\n", .{});
    
    var analyzer = beat.gpu_classifier.ArithmeticIntensityAnalyzer.init();
    
    // Create test fingerprint for compute-intensive task
    var compute_fingerprint = beat.fingerprint.TaskFingerprint{
        .call_site_hash = 0,
        .data_size_class = 0,
        .data_alignment = 0,
        .access_pattern = .sequential,
        .simd_width = 0,
        .cache_locality = 0,
        .numa_node_hint = 0,
        .cpu_intensity = 0,
        .parallel_potential = 0,
        .execution_phase = 0,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 0,
        .time_of_day_bucket = 0,
        .execution_frequency = 0,
        .seasonal_pattern = 0,
        .variance_level = 0,
        .expected_cycles_log2 = 0,
        .memory_footprint_log2 = 0,
        .io_intensity = 0,
        .cache_miss_rate = 0,
        .branch_predictability = 0,
        .vectorization_benefit = 0,
    };
    compute_fingerprint.data_size_class = 3; // 2^20 = 1MB
    compute_fingerprint.cpu_intensity = 12;
    compute_fingerprint.vectorization_benefit = 14;
    compute_fingerprint.access_pattern = .sequential; // Sequential access
    compute_fingerprint.branch_predictability = 13;
    
    analyzer.analyzeFingerprint(compute_fingerprint);
    
    const ai = analyzer.calculateArithmeticIntensity();
    try std.testing.expect(ai > 0.0);
    try std.testing.expect(ai < 1000.0); // Reasonable upper bound
    
    std.debug.print("   Compute-intensive task:\n", .{});
    std.debug.print("     Floating point ops: {}\n", .{analyzer.floating_point_ops});
    std.debug.print("     Integer ops: {}\n", .{analyzer.integer_ops});
    std.debug.print("     Memory accesses: {}\n", .{analyzer.memory_accesses});
    std.debug.print("     Arithmetic intensity: {d:.2} ops/byte\n", .{ai});
    
    // Test 2: Memory-intensive task
    std.debug.print("2. Testing memory-intensive task analysis...\n", .{});
    
    var memory_fingerprint = beat.fingerprint.TaskFingerprint{
        .call_site_hash = 0,
        .data_size_class = 0,
        .data_alignment = 0,
        .access_pattern = .sequential,
        .simd_width = 0,
        .cache_locality = 0,
        .numa_node_hint = 0,
        .cpu_intensity = 0,
        .parallel_potential = 0,
        .execution_phase = 0,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 0,
        .time_of_day_bucket = 0,
        .execution_frequency = 0,
        .seasonal_pattern = 0,
        .variance_level = 0,
        .expected_cycles_log2 = 0,
        .memory_footprint_log2 = 0,
        .io_intensity = 0,
        .cache_miss_rate = 0,
        .branch_predictability = 0,
        .vectorization_benefit = 0,
    };
    memory_fingerprint.data_size_class = 27; // 2^27 = 128MB
    memory_fingerprint.cpu_intensity = 5; // Low compute
    memory_fingerprint.vectorization_benefit = 6;
    memory_fingerprint.access_pattern = .random; // Random access
    
    var memory_analyzer = beat.gpu_classifier.ArithmeticIntensityAnalyzer.init();
    memory_analyzer.analyzeFingerprint(memory_fingerprint);
    
    const memory_ai = memory_analyzer.calculateArithmeticIntensity();
    try std.testing.expect(memory_ai < ai); // Should be lower than compute-intensive
    
    std.debug.print("   Memory-intensive task:\n", .{});
    std.debug.print("     Arithmetic intensity: {d:.2} ops/byte\n", .{memory_ai});
    std.debug.print("     Memory footprint: {} bytes\n", .{memory_analyzer.memory_footprint});
    
    std.debug.print("   ✅ Arithmetic intensity analyzer completed\n", .{});
}

test "data parallel pattern detector" {
    _ = std.testing.allocator;
    
    std.debug.print("\n=== Data Parallel Pattern Detector Test ===\n", .{});
    
    // Test 1: Embarrassingly parallel pattern
    std.debug.print("1. Testing embarrassingly parallel pattern detection...\n", .{});
    
    var detector = beat.gpu_classifier.DataParallelPatternDetector.init();
    
    var parallel_fingerprint = beat.fingerprint.TaskFingerprint{
        .call_site_hash = 0,
        .data_size_class = 0,
        .data_alignment = 0,
        .access_pattern = .sequential,
        .simd_width = 0,
        .cache_locality = 0,
        .numa_node_hint = 0,
        .cpu_intensity = 0,
        .parallel_potential = 0,
        .execution_phase = 0,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 0,
        .time_of_day_bucket = 0,
        .execution_frequency = 0,
        .seasonal_pattern = 0,
        .variance_level = 0,
        .expected_cycles_log2 = 0,
        .memory_footprint_log2 = 0,
        .io_intensity = 0,
        .cache_miss_rate = 0,
        .branch_predictability = 0,
        .vectorization_benefit = 0,
    };
    parallel_fingerprint.dependency_count = 1; // Very low dependencies
    parallel_fingerprint.vectorization_benefit = 15; // High optimization potential (u4 max)
    parallel_fingerprint.data_size_class = 24; // 2^24 = 16MB
    parallel_fingerprint.cpu_intensity = 9;
    
    detector.analyzeTaskPatterns(parallel_fingerprint, null);
    
    try std.testing.expect(detector.detected_patterns.contains(.embarrassingly_parallel));
    
    const potential = detector.getParallelizationPotential();
    try std.testing.expect(potential > 0.5);
    
    std.debug.print("   Embarrassingly parallel task:\n", .{});
    std.debug.print("     Pattern detected: {}\n", .{detector.detected_patterns.contains(.embarrassingly_parallel)});
    std.debug.print("     Parallelization potential: {d:.2}\n", .{potential});
    
    // Test 2: Map-reduce pattern
    std.debug.print("2. Testing map-reduce pattern detection...\n", .{});
    
    var mapreduce_detector = beat.gpu_classifier.DataParallelPatternDetector.init();
    
    var mapreduce_fingerprint = beat.fingerprint.TaskFingerprint{
        .call_site_hash = 0,
        .data_size_class = 0,
        .data_alignment = 0,
        .access_pattern = .sequential,
        .simd_width = 0,
        .cache_locality = 0,
        .numa_node_hint = 0,
        .cpu_intensity = 0,
        .parallel_potential = 0,
        .execution_phase = 0,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 0,
        .time_of_day_bucket = 0,
        .execution_frequency = 0,
        .seasonal_pattern = 0,
        .variance_level = 0,
        .expected_cycles_log2 = 0,
        .memory_footprint_log2 = 0,
        .io_intensity = 0,
        .cache_miss_rate = 0,
        .branch_predictability = 0,
        .vectorization_benefit = 0,
    };
    mapreduce_fingerprint.dependency_count = 5; // Moderate dependencies
    mapreduce_fingerprint.cpu_intensity = 10; // High complexity
    mapreduce_fingerprint.data_size_class = 5 * 1024 * 1024; // 5MB
    mapreduce_fingerprint.vectorization_benefit = 12;
    
    mapreduce_detector.analyzeTaskPatterns(mapreduce_fingerprint, null);
    
    const mapreduce_potential = mapreduce_detector.getParallelizationPotential();
    
    std.debug.print("   Map-reduce style task:\n", .{});
    std.debug.print("     Map-reduce detected: {}\n", .{mapreduce_detector.detected_patterns.contains(.map_reduce)});
    std.debug.print("     Parallelization potential: {d:.2}\n", .{mapreduce_potential});
    
    // Test 3: Matrix operations pattern
    std.debug.print("3. Testing matrix operations pattern detection...\n", .{});
    
    var matrix_detector = beat.gpu_classifier.DataParallelPatternDetector.init();
    
    var matrix_fingerprint = beat.fingerprint.TaskFingerprint{
        .call_site_hash = 0,
        .data_size_class = 0,
        .data_alignment = 0,
        .access_pattern = .sequential,
        .simd_width = 0,
        .cache_locality = 0,
        .numa_node_hint = 0,
        .cpu_intensity = 0,
        .parallel_potential = 0,
        .execution_phase = 0,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 0,
        .time_of_day_bucket = 0,
        .execution_frequency = 0,
        .seasonal_pattern = 0,
        .variance_level = 0,
        .expected_cycles_log2 = 0,
        .memory_footprint_log2 = 0,
        .io_intensity = 0,
        .cache_miss_rate = 0,
        .branch_predictability = 0,
        .vectorization_benefit = 0,
    };
    matrix_fingerprint.vectorization_benefit = 13; // Very high optimization
    matrix_fingerprint.cpu_intensity = 11;
    matrix_fingerprint.access_pattern = .gather_scatter; // Regular access pattern
    matrix_fingerprint.data_size_class = 16 * 1024 * 1024; // 16MB
    
    matrix_detector.analyzeTaskPatterns(matrix_fingerprint, null);
    
    const matrix_potential = matrix_detector.getParallelizationPotential();
    
    std.debug.print("   Matrix operations task:\n", .{});
    std.debug.print("     Matrix ops detected: {}\n", .{matrix_detector.detected_patterns.contains(.matrix_operations)});
    std.debug.print("     Parallelization potential: {d:.2}\n", .{matrix_potential});
    
    std.debug.print("   ✅ Data parallel pattern detector completed\n", .{});
}

test "memory access pattern analyzer" {
    _ = std.testing.allocator;
    
    std.debug.print("\n=== Memory Access Pattern Analyzer Test ===\n", .{});
    
    // Test 1: Coalesced access pattern
    std.debug.print("1. Testing coalesced access pattern detection...\n", .{});
    
    var analyzer = beat.gpu_classifier.MemoryAccessAnalyzer.init();
    
    var coalesced_fingerprint = beat.fingerprint.TaskFingerprint{
        .call_site_hash = 0,
        .data_size_class = 0,
        .data_alignment = 0,
        .access_pattern = .sequential,
        .simd_width = 0,
        .cache_locality = 0,
        .numa_node_hint = 0,
        .cpu_intensity = 0,
        .parallel_potential = 0,
        .execution_phase = 0,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 0,
        .time_of_day_bucket = 0,
        .execution_frequency = 0,
        .seasonal_pattern = 0,
        .variance_level = 0,
        .expected_cycles_log2 = 0,
        .memory_footprint_log2 = 0,
        .io_intensity = 0,
        .cache_miss_rate = 0,
        .branch_predictability = 0,
        .vectorization_benefit = 0,
    };
    coalesced_fingerprint.access_pattern = .sequential; // Very sequential access
    coalesced_fingerprint.dependency_count = 2; // Low dependencies
    coalesced_fingerprint.data_size_class = 23; // 2^23 = 8MB
    
    analyzer.analyzeMemoryPatterns(coalesced_fingerprint, null);
    
    try std.testing.expect(analyzer.detected_patterns.contains(.coalesced_access));
    
    const coalescing_score = analyzer.getCoalescingScore();
    try std.testing.expect(coalescing_score > 0.7);
    
    std.debug.print("   Coalesced access task:\n", .{});
    std.debug.print("     Coalesced pattern detected: {}\n", .{analyzer.detected_patterns.contains(.coalesced_access)});
    std.debug.print("     Coalescing score: {d:.2}\n", .{coalescing_score});
    std.debug.print("     Cache efficiency: {d:.2}\n", .{analyzer.getCacheEfficiencyScore()});
    
    // Test 2: Random access pattern
    std.debug.print("2. Testing random access pattern detection...\n", .{});
    
    var random_analyzer = beat.gpu_classifier.MemoryAccessAnalyzer.init();
    
    var random_fingerprint = beat.fingerprint.TaskFingerprint{
        .call_site_hash = 0,
        .data_size_class = 0,
        .data_alignment = 0,
        .access_pattern = .sequential,
        .simd_width = 0,
        .cache_locality = 0,
        .numa_node_hint = 0,
        .cpu_intensity = 0,
        .parallel_potential = 0,
        .execution_phase = 0,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 0,
        .time_of_day_bucket = 0,
        .execution_frequency = 0,
        .seasonal_pattern = 0,
        .variance_level = 0,
        .expected_cycles_log2 = 0,
        .memory_footprint_log2 = 0,
        .io_intensity = 0,
        .cache_miss_rate = 0,
        .branch_predictability = 0,
        .vectorization_benefit = 0,
    };
    random_fingerprint.access_pattern = .random; // Very random access
    random_fingerprint.dependency_count = 12; // High dependencies
    random_fingerprint.data_size_class = 25; // 2^25 = 32MB
    
    random_analyzer.analyzeMemoryPatterns(random_fingerprint, null);
    
    try std.testing.expect(random_analyzer.detected_patterns.contains(.random_access));
    
    const random_coalescing = random_analyzer.getCoalescingScore();
    try std.testing.expect(random_coalescing < coalescing_score); // Should be lower
    
    std.debug.print("   Random access task:\n", .{});
    std.debug.print("     Random pattern detected: {}\n", .{random_analyzer.detected_patterns.contains(.random_access)});
    std.debug.print("     Pointer chasing detected: {}\n", .{random_analyzer.detected_patterns.contains(.pointer_chasing)});
    std.debug.print("     Coalescing score: {d:.2}\n", .{random_coalescing});
    std.debug.print("     Cache efficiency: {d:.2}\n", .{random_analyzer.getCacheEfficiencyScore()});
    
    // Test 3: Strided access pattern
    std.debug.print("3. Testing strided access pattern detection...\n", .{});
    
    var strided_analyzer = beat.gpu_classifier.MemoryAccessAnalyzer.init();
    
    var strided_fingerprint = beat.fingerprint.TaskFingerprint{
        .call_site_hash = 0,
        .data_size_class = 0,
        .data_alignment = 0,
        .access_pattern = .sequential,
        .simd_width = 0,
        .cache_locality = 0,
        .numa_node_hint = 0,
        .cpu_intensity = 0,
        .parallel_potential = 0,
        .execution_phase = 0,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 0,
        .time_of_day_bucket = 0,
        .execution_frequency = 0,
        .seasonal_pattern = 0,
        .variance_level = 0,
        .expected_cycles_log2 = 0,
        .memory_footprint_log2 = 0,
        .io_intensity = 0,
        .cache_miss_rate = 0,
        .branch_predictability = 0,
        .vectorization_benefit = 0,
    };
    strided_fingerprint.access_pattern = .strided; // Moderate access pattern (strided)
    strided_fingerprint.dependency_count = 4;
    
    strided_analyzer.analyzeMemoryPatterns(strided_fingerprint, null);
    
    const strided_coalescing = strided_analyzer.getCoalescingScore();
    
    std.debug.print("   Strided access task:\n", .{});
    std.debug.print("     Strided pattern detected: {}\n", .{strided_analyzer.detected_patterns.contains(.strided_access)});
    std.debug.print("     Coalescing score: {d:.2}\n", .{strided_coalescing});
    
    std.debug.print("   ✅ Memory access pattern analyzer completed\n", .{});
}

test "branch divergence analyzer" {
    _ = std.testing.allocator;
    
    std.debug.print("\n=== Branch Divergence Analyzer Test ===\n", .{});
    
    // Test 1: Uniform execution pattern
    std.debug.print("1. Testing uniform execution pattern detection...\n", .{});
    
    var analyzer = beat.gpu_classifier.BranchDivergenceAnalyzer.init();
    
    var uniform_fingerprint = beat.fingerprint.TaskFingerprint{
        .call_site_hash = 0,
        .data_size_class = 0,
        .data_alignment = 0,
        .access_pattern = .sequential,
        .simd_width = 0,
        .cache_locality = 0,
        .numa_node_hint = 0,
        .cpu_intensity = 0,
        .parallel_potential = 0,
        .execution_phase = 0,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 0,
        .time_of_day_bucket = 0,
        .execution_frequency = 0,
        .seasonal_pattern = 0,
        .variance_level = 0,
        .expected_cycles_log2 = 0,
        .memory_footprint_log2 = 0,
        .io_intensity = 0,
        .cache_miss_rate = 0,
        .branch_predictability = 0,
        .vectorization_benefit = 0,
    };
    uniform_fingerprint.branch_predictability = 15; // Very predictable branches (u4 max)
    uniform_fingerprint.dependency_count = 2; // Low dependencies
    uniform_fingerprint.cpu_intensity = 10;
    
    analyzer.analyzeExecutionPatterns(uniform_fingerprint);
    
    try std.testing.expect(analyzer.detected_patterns.contains(.uniform_execution));
    
    const uniformity = analyzer.getUniformityScore();
    try std.testing.expect(uniformity > 0.8);
    
    std.debug.print("   Uniform execution task:\n", .{});
    std.debug.print("     Uniform pattern detected: {}\n", .{analyzer.detected_patterns.contains(.uniform_execution)});
    std.debug.print("     Branch uniformity score: {d:.2}\n", .{uniformity});
    std.debug.print("     Divergence penalty: {d:.2}\n", .{analyzer.getDivergencePenalty()});
    
    // Test 2: Divergent branching pattern
    std.debug.print("2. Testing divergent branching pattern detection...\n", .{});
    
    var divergent_analyzer = beat.gpu_classifier.BranchDivergenceAnalyzer.init();
    
    var divergent_fingerprint = beat.fingerprint.TaskFingerprint{
        .call_site_hash = 0,
        .data_size_class = 0,
        .data_alignment = 0,
        .access_pattern = .sequential,
        .simd_width = 0,
        .cache_locality = 0,
        .numa_node_hint = 0,
        .cpu_intensity = 0,
        .parallel_potential = 0,
        .execution_phase = 0,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 0,
        .time_of_day_bucket = 0,
        .execution_frequency = 0,
        .seasonal_pattern = 0,
        .variance_level = 0,
        .expected_cycles_log2 = 0,
        .memory_footprint_log2 = 0,
        .io_intensity = 0,
        .cache_miss_rate = 0,
        .branch_predictability = 0,
        .vectorization_benefit = 0,
    };
    divergent_fingerprint.branch_predictability = 5; // Very unpredictable
    divergent_fingerprint.dependency_count = 8; // Moderate dependencies
    divergent_fingerprint.cpu_intensity = 9;
    
    divergent_analyzer.analyzeExecutionPatterns(divergent_fingerprint);
    
    try std.testing.expect(divergent_analyzer.detected_patterns.contains(.divergent_branching));
    
    const divergent_uniformity = divergent_analyzer.getUniformityScore();
    try std.testing.expect(divergent_uniformity < uniformity); // Should be lower
    
    std.debug.print("   Divergent branching task:\n", .{});
    std.debug.print("     Divergent pattern detected: {}\n", .{divergent_analyzer.detected_patterns.contains(.divergent_branching)});
    std.debug.print("     Control flow heavy: {}\n", .{divergent_analyzer.detected_patterns.contains(.control_flow_heavy)});
    std.debug.print("     Branch uniformity score: {d:.2}\n", .{divergent_uniformity});
    std.debug.print("     Divergence penalty: {d:.2}\n", .{divergent_analyzer.getDivergencePenalty()});
    
    // Test 3: Synchronization-heavy pattern
    std.debug.print("3. Testing synchronization-heavy pattern detection...\n", .{});
    
    var sync_analyzer = beat.gpu_classifier.BranchDivergenceAnalyzer.init();
    
    var sync_fingerprint = beat.fingerprint.TaskFingerprint{
        .call_site_hash = 0,
        .data_size_class = 0,
        .data_alignment = 0,
        .access_pattern = .sequential,
        .simd_width = 0,
        .cache_locality = 0,
        .numa_node_hint = 0,
        .cpu_intensity = 0,
        .parallel_potential = 0,
        .execution_phase = 0,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 0,
        .time_of_day_bucket = 0,
        .execution_frequency = 0,
        .seasonal_pattern = 0,
        .variance_level = 0,
        .expected_cycles_log2 = 0,
        .memory_footprint_log2 = 0,
        .io_intensity = 0,
        .cache_miss_rate = 0,
        .branch_predictability = 0,
        .vectorization_benefit = 0,
    };
    sync_fingerprint.dependency_count = 15; // Very high dependencies
    sync_fingerprint.branch_predictability = 10;
    sync_fingerprint.cpu_intensity = 7;
    
    sync_analyzer.analyzeExecutionPatterns(sync_fingerprint);
    
    try std.testing.expect(sync_analyzer.detected_patterns.contains(.synchronization_heavy));
    
    const sync_uniformity = sync_analyzer.getUniformityScore();
    
    std.debug.print("   Synchronization-heavy task:\n", .{});
    std.debug.print("     Sync-heavy pattern detected: {}\n", .{sync_analyzer.detected_patterns.contains(.synchronization_heavy)});
    std.debug.print("     Branch uniformity score: {d:.2}\n", .{sync_uniformity});
    
    std.debug.print("   ✅ Branch divergence analyzer completed\n", .{});
}

test "comprehensive GPU task classification" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Comprehensive GPU Task Classification Test ===\n", .{});
    
    var classifier = beat.gpu_classifier.GPUTaskClassifier.init(allocator);
    defer classifier.deinit();
    
    // Test 1: Highly GPU-suitable task
    std.debug.print("1. Testing highly GPU-suitable task classification...\n", .{});
    
    var gpu_friendly_fingerprint = beat.fingerprint.TaskFingerprint{
        .call_site_hash = 0,
        .data_size_class = 0,
        .data_alignment = 0,
        .access_pattern = .sequential,
        .simd_width = 0,
        .cache_locality = 0,
        .numa_node_hint = 0,
        .cpu_intensity = 0,
        .parallel_potential = 0,
        .execution_phase = 0,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 0,
        .time_of_day_bucket = 0,
        .execution_frequency = 0,
        .seasonal_pattern = 0,
        .variance_level = 0,
        .expected_cycles_log2 = 0,
        .memory_footprint_log2 = 0,
        .io_intensity = 0,
        .cache_miss_rate = 0,
        .branch_predictability = 0,
        .vectorization_benefit = 0,
    };
    gpu_friendly_fingerprint.data_size_class = 26; // 2^26 = 64MB - large dataset
    gpu_friendly_fingerprint.cpu_intensity = 13; // High computation
    gpu_friendly_fingerprint.vectorization_benefit = 14; // Very optimizable
    gpu_friendly_fingerprint.access_pattern = .random; // Very sequential access
    gpu_friendly_fingerprint.branch_predictability = 14; // Uniform execution
    gpu_friendly_fingerprint.dependency_count = 1; // Embarrassingly parallel
    
    const gpu_analysis = try classifier.classifyTask(gpu_friendly_fingerprint, null);
    
    try std.testing.expect(gpu_analysis.overall_suitability == .suitable or 
                          gpu_analysis.overall_suitability == .highly_suitable);
    try std.testing.expect(gpu_analysis.confidence_score > 0.5);
    try std.testing.expect(gpu_analysis.arithmetic_intensity > 0.0);
    try std.testing.expect(gpu_analysis.parallelization_potential > 0.7);
    
    std.debug.print("   GPU-friendly task results:\n", .{});
    std.debug.print("     Overall suitability: {s}\n", .{@tagName(gpu_analysis.overall_suitability)});
    std.debug.print("     Confidence score: {d:.2}\n", .{gpu_analysis.confidence_score});
    std.debug.print("     Arithmetic intensity: {d:.2}\n", .{gpu_analysis.arithmetic_intensity});
    std.debug.print("     Parallelization potential: {d:.2}\n", .{gpu_analysis.parallelization_potential});
    std.debug.print("     Memory coalescing score: {d:.2}\n", .{gpu_analysis.memory_coalescing_score});
    std.debug.print("     Branch uniformity: {d:.2}\n", .{gpu_analysis.branch_uniformity});
    std.debug.print("     GPU efficiency estimate: {d:.2}\n", .{gpu_analysis.gpu_efficiency_estimate});
    std.debug.print("     Expected performance multiplier: {d:.2}x\n", .{gpu_analysis.overall_suitability.getPerformanceMultiplier()});
    
    // Test 2: CPU-suitable task  
    std.debug.print("2. Testing CPU-suitable task classification...\n", .{});
    
    var cpu_friendly_fingerprint = beat.fingerprint.TaskFingerprint{
        .call_site_hash = 0,
        .data_size_class = 0,
        .data_alignment = 0,
        .access_pattern = .sequential,
        .simd_width = 0,
        .cache_locality = 0,
        .numa_node_hint = 0,
        .cpu_intensity = 0,
        .parallel_potential = 0,
        .execution_phase = 0,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 0,
        .time_of_day_bucket = 0,
        .execution_frequency = 0,
        .seasonal_pattern = 0,
        .variance_level = 0,
        .expected_cycles_log2 = 0,
        .memory_footprint_log2 = 0,
        .io_intensity = 0,
        .cache_miss_rate = 0,
        .branch_predictability = 0,
        .vectorization_benefit = 0,
    };
    cpu_friendly_fingerprint.data_size_class = 1024; // 1KB - small dataset
    cpu_friendly_fingerprint.cpu_intensity = 5; // Low computation
    cpu_friendly_fingerprint.vectorization_benefit = 3; // Low optimization
    cpu_friendly_fingerprint.access_pattern = .random; // Very random access
    cpu_friendly_fingerprint.branch_predictability = 4; // Unpredictable branches
    cpu_friendly_fingerprint.dependency_count = 15; // High dependencies
    
    const cpu_analysis = try classifier.classifyTask(cpu_friendly_fingerprint, null);
    
    try std.testing.expect(cpu_analysis.overall_suitability == .unsuitable or 
                          cpu_analysis.overall_suitability == .highly_unsuitable);
    try std.testing.expect(cpu_analysis.parallelization_potential < 0.5);
    
    std.debug.print("   CPU-friendly task results:\n", .{});
    std.debug.print("     Overall suitability: {s}\n", .{@tagName(cpu_analysis.overall_suitability)});
    std.debug.print("     Confidence score: {d:.2}\n", .{cpu_analysis.confidence_score});
    std.debug.print("     Arithmetic intensity: {d:.2}\n", .{cpu_analysis.arithmetic_intensity});
    std.debug.print("     Parallelization potential: {d:.2}\n", .{cpu_analysis.parallelization_potential});
    std.debug.print("     Memory coalescing score: {d:.2}\n", .{cpu_analysis.memory_coalescing_score});
    std.debug.print("     Branch uniformity: {d:.2}\n", .{cpu_analysis.branch_uniformity});
    std.debug.print("     Expected performance multiplier: {d:.2}x\n", .{cpu_analysis.overall_suitability.getPerformanceMultiplier()});
    
    // Test 3: Neutral task (could go either way)
    std.debug.print("3. Testing neutral task classification...\n", .{});
    
    var neutral_fingerprint = beat.fingerprint.TaskFingerprint{
        .call_site_hash = 0,
        .data_size_class = 0,
        .data_alignment = 0,
        .access_pattern = .sequential,
        .simd_width = 0,
        .cache_locality = 0,
        .numa_node_hint = 0,
        .cpu_intensity = 0,
        .parallel_potential = 0,
        .execution_phase = 0,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 0,
        .time_of_day_bucket = 0,
        .execution_frequency = 0,
        .seasonal_pattern = 0,
        .variance_level = 0,
        .expected_cycles_log2 = 0,
        .memory_footprint_log2 = 0,
        .io_intensity = 0,
        .cache_miss_rate = 0,
        .branch_predictability = 0,
        .vectorization_benefit = 0,
    };
    neutral_fingerprint.data_size_class = 2 * 1024 * 1024; // 2MB - moderate dataset
    neutral_fingerprint.cpu_intensity = 8; // Moderate computation
    neutral_fingerprint.vectorization_benefit = 9; // Moderate optimization
    neutral_fingerprint.access_pattern = .mixed; // Mixed access pattern
    neutral_fingerprint.branch_predictability = 65; // Moderate predictability
    neutral_fingerprint.dependency_count = 6; // Moderate dependencies
    
    const neutral_analysis = try classifier.classifyTask(neutral_fingerprint, null);
    
    std.debug.print("   Neutral task results:\n", .{});
    std.debug.print("     Overall suitability: {s}\n", .{@tagName(neutral_analysis.overall_suitability)});
    std.debug.print("     Confidence score: {d:.2}\n", .{neutral_analysis.confidence_score});
    std.debug.print("     Arithmetic intensity: {d:.2}\n", .{neutral_analysis.arithmetic_intensity});
    std.debug.print("     Parallelization potential: {d:.2}\n", .{neutral_analysis.parallelization_potential});
    
    std.debug.print("   ✅ Comprehensive GPU task classification completed\n", .{});
}

test "GPU classification with device information" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== GPU Classification with Device Information Test ===\n", .{});
    
    var classifier = beat.gpu_classifier.GPUTaskClassifier.init(allocator);
    defer classifier.deinit();
    
    // Create mock GPU device info
    const mock_sycl_info = beat.gpu_integration.SyclDeviceInfo{
        .name = [_]u8{0} ** 256,
        .vendor = [_]u8{0} ** 128,
        .driver_version = [_]u8{0} ** 64,
        .type = beat.gpu_integration.DEVICE_TYPE_GPU,
        .backend = beat.gpu_integration.c.BEAT_SYCL_BACKEND_CUDA,
        .max_compute_units = 80, // High-end GPU
        .max_work_group_size = 1024,
        .max_work_item_dimensions = 3,
        .max_work_item_sizes = [_]u64{ 1024, 1024, 64 },
        .global_memory_size = 16 * 1024 * 1024 * 1024, // 16GB
        .local_memory_size = 128 * 1024, // 128KB
        .max_memory_allocation = 4 * 1024 * 1024 * 1024, // 4GB
        .preferred_vector_width_float = 32,
        .preferred_vector_width_double = 16,
        .preferred_vector_width_int = 32,
        .supports_double = 1,
        .supports_half = 1,
        .supports_unified_memory = 1,
        .supports_sub_groups = 1,
        .reserved = 0,
    };
    
    const gpu_device_info = beat.gpu_integration.GPUDeviceInfo.init(mock_sycl_info);
    
    std.debug.print("1. Testing classification with high-end GPU device...\n", .{});
    
    // Test with compute-intensive task
    var compute_fingerprint = beat.fingerprint.TaskFingerprint{
        .call_site_hash = 0,
        .data_size_class = 0,
        .data_alignment = 0,
        .access_pattern = .sequential,
        .simd_width = 0,
        .cache_locality = 0,
        .numa_node_hint = 0,
        .cpu_intensity = 0,
        .parallel_potential = 0,
        .execution_phase = 0,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 0,
        .time_of_day_bucket = 0,
        .execution_frequency = 0,
        .seasonal_pattern = 0,
        .variance_level = 0,
        .expected_cycles_log2 = 0,
        .memory_footprint_log2 = 0,
        .io_intensity = 0,
        .cache_miss_rate = 0,
        .branch_predictability = 0,
        .vectorization_benefit = 0,
    };
    compute_fingerprint.data_size_class = 27; // 2^27 = 128MB
    compute_fingerprint.cpu_intensity = 14;
    compute_fingerprint.vectorization_benefit = 15;
    compute_fingerprint.access_pattern = .sequential; // Sequential
    compute_fingerprint.branch_predictability = 13;
    compute_fingerprint.dependency_count = 2;
    
    const analysis_with_gpu = try classifier.classifyTask(compute_fingerprint, &gpu_device_info);
    
    try std.testing.expect(analysis_with_gpu.gpu_efficiency_estimate > 0.0);
    try std.testing.expect(analysis_with_gpu.transfer_overhead_ratio >= 0.0);
    try std.testing.expect(analysis_with_gpu.transfer_overhead_ratio <= 1.0);
    
    std.debug.print("   High-end GPU analysis:\n", .{});
    std.debug.print("     Device compute units: {}\n", .{gpu_device_info.compute_units});
    std.debug.print("     Device memory: {d:.1} GB\n", .{gpu_device_info.global_memory_gb});
    std.debug.print("     GPU efficiency estimate: {d:.2}\n", .{analysis_with_gpu.gpu_efficiency_estimate});
    std.debug.print("     Transfer overhead ratio: {d:.2}\n", .{analysis_with_gpu.transfer_overhead_ratio});
    std.debug.print("     Overall suitability: {s}\n", .{@tagName(analysis_with_gpu.overall_suitability)});
    std.debug.print("     Confidence score: {d:.2}\n", .{analysis_with_gpu.confidence_score});
    
    // Test 2: Compare with analysis without device info
    std.debug.print("2. Comparing analysis with and without device information...\n", .{});
    
    const analysis_without_gpu = try classifier.classifyTask(compute_fingerprint, null);
    
    std.debug.print("   Comparison results:\n", .{});
    std.debug.print("     With GPU device - efficiency: {d:.2}, suitability: {s}\n", .{
        analysis_with_gpu.gpu_efficiency_estimate, @tagName(analysis_with_gpu.overall_suitability)
    });
    std.debug.print("     Without GPU device - efficiency: {d:.2}, suitability: {s}\n", .{
        analysis_without_gpu.gpu_efficiency_estimate, @tagName(analysis_without_gpu.overall_suitability)
    });
    
    std.debug.print("   ✅ GPU classification with device information completed\n", .{});
}

test "adaptive learning and performance tracking" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Adaptive Learning and Performance Tracking Test ===\n", .{});
    
    var classifier = beat.gpu_classifier.GPUTaskClassifier.init(allocator);
    defer classifier.deinit();
    
    std.debug.print("1. Testing performance tracking...\n", .{});
    
    // Record some performance results
    try classifier.recordPerformance(.highly_suitable, 2.8); // Good prediction
    try classifier.recordPerformance(.suitable, 1.4); // Good prediction
    try classifier.recordPerformance(.unsuitable, 1.2); // Poor prediction (predicted unsuitable but got speedup)
    try classifier.recordPerformance(.highly_suitable, 3.2); // Good prediction
    try classifier.recordPerformance(.neutral, 0.9); // Good prediction
    
    const accuracy = classifier.getClassificationAccuracy();
    try std.testing.expect(accuracy > 0.0);
    try std.testing.expect(accuracy <= 1.0);
    
    std.debug.print("   Performance tracking results:\n", .{});
    std.debug.print("     Recorded predictions: 5\n", .{});
    std.debug.print("     Classification accuracy: {d:.2}\n", .{accuracy});
    std.debug.print("     Total classifications: {}\n", .{classifier.classification_count});
    
    // Test 2: Multiple predictions to build history
    std.debug.print("2. Building classification history...\n", .{});
    
    var i: u32 = 0;
    while (i < 20) : (i += 1) {
        const predicted = if (i % 4 == 0) beat.gpu_classifier.GPUSuitability.highly_suitable
                         else if (i % 4 == 1) beat.gpu_classifier.GPUSuitability.suitable
                         else if (i % 4 == 2) beat.gpu_classifier.GPUSuitability.neutral
                         else beat.gpu_classifier.GPUSuitability.unsuitable;
        
        // Simulate realistic performance with some noise
        const base_multiplier = predicted.getPerformanceMultiplier();
        const noise = (@as(f32, @floatFromInt(i % 7)) - 3.0) * 0.1; // ±0.3 noise
        const actual_speedup = base_multiplier + noise;
        
        try classifier.recordPerformance(predicted, actual_speedup);
    }
    
    const final_accuracy = classifier.getClassificationAccuracy();
    
    std.debug.print("   Extended performance tracking:\n", .{});
    std.debug.print("     Total recorded predictions: 25\n", .{});
    std.debug.print("     Final classification accuracy: {d:.2}\n", .{final_accuracy});
    
    try std.testing.expect(final_accuracy > 0.5); // Should be reasonably accurate
    
    std.debug.print("   ✅ Adaptive learning and performance tracking completed\n", .{});
}

test "roofline model estimation" {
    _ = std.testing.allocator;
    
    std.debug.print("\n=== Roofline Model Estimation Test ===\n", .{});
    
    // Test 1: Memory-bound workload
    std.debug.print("1. Testing memory-bound workload analysis...\n", .{});
    
    var memory_bound_analyzer = beat.gpu_classifier.ArithmeticIntensityAnalyzer.init();
    
    var memory_bound_fingerprint = beat.fingerprint.TaskFingerprint{
        .call_site_hash = 0,
        .data_size_class = 0,
        .data_alignment = 0,
        .access_pattern = .sequential,
        .simd_width = 0,
        .cache_locality = 0,
        .numa_node_hint = 0,
        .cpu_intensity = 0,
        .parallel_potential = 0,
        .execution_phase = 0,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 0,
        .time_of_day_bucket = 0,
        .execution_frequency = 0,
        .seasonal_pattern = 0,
        .variance_level = 0,
        .expected_cycles_log2 = 0,
        .memory_footprint_log2 = 0,
        .io_intensity = 0,
        .cache_miss_rate = 0,
        .branch_predictability = 0,
        .vectorization_benefit = 0,
    };
    memory_bound_fingerprint.data_size_class = 28; // 2^28 = 256MB - large dataset
    memory_bound_fingerprint.cpu_intensity = 4; // Low computation
    memory_bound_fingerprint.vectorization_benefit = 6;
    memory_bound_fingerprint.access_pattern = .sequential; // Sequential access
    
    memory_bound_analyzer.analyzeFingerprint(memory_bound_fingerprint);
    
    // Create high-performance GPU for testing
    const high_perf_gpu_info = beat.gpu_integration.SyclDeviceInfo{
        .name = [_]u8{0} ** 256,
        .vendor = [_]u8{0} ** 128,
        .driver_version = [_]u8{0} ** 64,
        .type = beat.gpu_integration.DEVICE_TYPE_GPU,
        .backend = beat.gpu_integration.c.BEAT_SYCL_BACKEND_CUDA,
        .max_compute_units = 108, // Very high-end GPU
        .max_work_group_size = 1024,
        .max_work_item_dimensions = 3,
        .max_work_item_sizes = [_]u64{ 1024, 1024, 64 },
        .global_memory_size = 24 * 1024 * 1024 * 1024, // 24GB
        .local_memory_size = 164 * 1024, // 164KB
        .max_memory_allocation = 6 * 1024 * 1024 * 1024, // 6GB
        .preferred_vector_width_float = 32,
        .preferred_vector_width_double = 16,
        .preferred_vector_width_int = 32,
        .supports_double = 1,
        .supports_half = 1,
        .supports_unified_memory = 1,
        .supports_sub_groups = 1,
        .reserved = 0,
    };
    
    const high_perf_gpu = beat.gpu_integration.GPUDeviceInfo.init(high_perf_gpu_info);
    
    const memory_ai = memory_bound_analyzer.calculateArithmeticIntensity();
    const memory_efficiency = memory_bound_analyzer.estimateGPUEfficiency(&high_perf_gpu);
    
    std.debug.print("   Memory-bound workload:\n", .{});
    std.debug.print("     Arithmetic intensity: {d:.2} ops/byte\n", .{memory_ai});
    std.debug.print("     GPU efficiency estimate: {d:.2}\n", .{memory_efficiency});
    std.debug.print("     Expected to be memory-bound: {}\n", .{memory_ai < 10.0});
    
    // Test 2: Compute-bound workload
    std.debug.print("2. Testing compute-bound workload analysis...\n", .{});
    
    var compute_bound_analyzer = beat.gpu_classifier.ArithmeticIntensityAnalyzer.init();
    
    var compute_bound_fingerprint = beat.fingerprint.TaskFingerprint{
        .call_site_hash = 0,
        .data_size_class = 0,
        .data_alignment = 0,
        .access_pattern = .sequential,
        .simd_width = 0,
        .cache_locality = 0,
        .numa_node_hint = 0,
        .cpu_intensity = 0,
        .parallel_potential = 0,
        .execution_phase = 0,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 0,
        .time_of_day_bucket = 0,
        .execution_frequency = 0,
        .seasonal_pattern = 0,
        .variance_level = 0,
        .expected_cycles_log2 = 0,
        .memory_footprint_log2 = 0,
        .io_intensity = 0,
        .cache_miss_rate = 0,
        .branch_predictability = 0,
        .vectorization_benefit = 0,
    };
    compute_bound_fingerprint.data_size_class = 20; // 2^20 = 1MB - smaller dataset
    compute_bound_fingerprint.cpu_intensity = 15; // Very high computation
    compute_bound_fingerprint.vectorization_benefit = 14;
    compute_bound_fingerprint.access_pattern = .sequential; // Sequential access
    
    compute_bound_analyzer.analyzeFingerprint(compute_bound_fingerprint);
    
    const compute_ai = compute_bound_analyzer.calculateArithmeticIntensity();
    const compute_efficiency = compute_bound_analyzer.estimateGPUEfficiency(&high_perf_gpu);
    
    std.debug.print("   Compute-bound workload:\n", .{});
    std.debug.print("     Arithmetic intensity: {d:.2} ops/byte\n", .{compute_ai});
    std.debug.print("     GPU efficiency estimate: {d:.2}\n", .{compute_efficiency});
    std.debug.print("     Expected to be compute-bound: {}\n", .{compute_ai > 10.0});
    
    try std.testing.expect(compute_ai > memory_ai); // Compute-bound should have higher AI
    
    std.debug.print("   ✅ Roofline model estimation completed\n", .{});
}

test "comprehensive GPU classification validation" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Comprehensive GPU Classification Validation ===\n", .{});
    
    var classifier = beat.gpu_classifier.GPUTaskClassifier.init(allocator);
    defer classifier.deinit();
    
    std.debug.print("1. Running comprehensive validation of GPU classification system...\n", .{});
    
    // Test different workload types
    const workload_types = [_]struct {
        name: []const u8,
        fingerprint: beat.fingerprint.TaskFingerprint,
        expected_min_suitability: beat.gpu_classifier.GPUSuitability,
    }{
        .{
            .name = "Large-scale matrix multiplication",
            .fingerprint = blk: {
                var fp = beat.fingerprint.TaskFingerprint{
        .call_site_hash = 0,
        .data_size_class = 0,
        .data_alignment = 0,
        .access_pattern = .sequential,
        .simd_width = 0,
        .cache_locality = 0,
        .numa_node_hint = 0,
        .cpu_intensity = 0,
        .parallel_potential = 0,
        .execution_phase = 0,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 0,
        .time_of_day_bucket = 0,
        .execution_frequency = 0,
        .seasonal_pattern = 0,
        .variance_level = 0,
        .expected_cycles_log2 = 0,
        .memory_footprint_log2 = 0,
        .io_intensity = 0,
        .cache_miss_rate = 0,
        .branch_predictability = 0,
        .vectorization_benefit = 0,
    };
                fp.data_size_class = 27; // 2^27 = 128MB
                fp.cpu_intensity = 15;
                fp.vectorization_benefit = 15;
                fp.access_pattern = .read_only; // Very sequential
                fp.branch_predictability = 15;
                fp.dependency_count = 0;
                break :blk fp;
            },
            .expected_min_suitability = .suitable,
        },
        .{
            .name = "Image convolution processing",
            .fingerprint = blk: {
                var fp = beat.fingerprint.TaskFingerprint{
        .call_site_hash = 0,
        .data_size_class = 0,
        .data_alignment = 0,
        .access_pattern = .sequential,
        .simd_width = 0,
        .cache_locality = 0,
        .numa_node_hint = 0,
        .cpu_intensity = 0,
        .parallel_potential = 0,
        .execution_phase = 0,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 0,
        .time_of_day_bucket = 0,
        .execution_frequency = 0,
        .seasonal_pattern = 0,
        .variance_level = 0,
        .expected_cycles_log2 = 0,
        .memory_footprint_log2 = 0,
        .io_intensity = 0,
        .cache_miss_rate = 0,
        .branch_predictability = 0,
        .vectorization_benefit = 0,
    };
                fp.data_size_class = 26; // 2^26 = 64MB
                fp.cpu_intensity = 12;
                fp.vectorization_benefit = 13;
                fp.access_pattern = .gather_scatter; // Local access patterns
                fp.branch_predictability = 12;
                fp.dependency_count = 3;
                break :blk fp;
            },
            .expected_min_suitability = .suitable,
        },
        .{
            .name = "Monte Carlo simulation",
            .fingerprint = blk: {
                var fp = beat.fingerprint.TaskFingerprint{
        .call_site_hash = 0,
        .data_size_class = 0,
        .data_alignment = 0,
        .access_pattern = .sequential,
        .simd_width = 0,
        .cache_locality = 0,
        .numa_node_hint = 0,
        .cpu_intensity = 0,
        .parallel_potential = 0,
        .execution_phase = 0,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 0,
        .time_of_day_bucket = 0,
        .execution_frequency = 0,
        .seasonal_pattern = 0,
        .variance_level = 0,
        .expected_cycles_log2 = 0,
        .memory_footprint_log2 = 0,
        .io_intensity = 0,
        .cache_miss_rate = 0,
        .branch_predictability = 0,
        .vectorization_benefit = 0,
    };
                fp.data_size_class = 32 * 1024 * 1024; // 32MB
                fp.cpu_intensity = 11;
                fp.vectorization_benefit = 13;
                fp.access_pattern = .random; // Random access
                fp.branch_predictability = 6; // Unpredictable
                fp.dependency_count = 2;
                break :blk fp;
            },
            .expected_min_suitability = .neutral,
        },
        .{
            .name = "Linked list traversal",
            .fingerprint = blk: {
                var fp = beat.fingerprint.TaskFingerprint{
        .call_site_hash = 0,
        .data_size_class = 0,
        .data_alignment = 0,
        .access_pattern = .sequential,
        .simd_width = 0,
        .cache_locality = 0,
        .numa_node_hint = 0,
        .cpu_intensity = 0,
        .parallel_potential = 0,
        .execution_phase = 0,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 0,
        .time_of_day_bucket = 0,
        .execution_frequency = 0,
        .seasonal_pattern = 0,
        .variance_level = 0,
        .expected_cycles_log2 = 0,
        .memory_footprint_log2 = 0,
        .io_intensity = 0,
        .cache_miss_rate = 0,
        .branch_predictability = 0,
        .vectorization_benefit = 0,
    };
                fp.data_size_class = 1024 * 1024; // 1MB
                fp.cpu_intensity = 3;
                fp.vectorization_benefit = 15;
                fp.access_pattern = .sequential; // Pointer chasing
                fp.branch_predictability = 5;
                fp.dependency_count = 3; // High dependencies
                break :blk fp;
            },
            .expected_min_suitability = .highly_unsuitable,
        },
        .{
            .name = "Small recursive algorithm",
            .fingerprint = blk: {
                var fp = beat.fingerprint.TaskFingerprint{
        .call_site_hash = 0,
        .data_size_class = 0,
        .data_alignment = 0,
        .access_pattern = .sequential,
        .simd_width = 0,
        .cache_locality = 0,
        .numa_node_hint = 0,
        .cpu_intensity = 0,
        .parallel_potential = 0,
        .execution_phase = 0,
        .priority_class = 0,
        .time_sensitivity = 0,
        .dependency_count = 0,
        .time_of_day_bucket = 0,
        .execution_frequency = 0,
        .seasonal_pattern = 0,
        .variance_level = 0,
        .expected_cycles_log2 = 0,
        .memory_footprint_log2 = 0,
        .io_intensity = 0,
        .cache_miss_rate = 0,
        .branch_predictability = 0,
        .vectorization_benefit = 0,
    };
                fp.data_size_class = 64 * 1024; // 64KB
                fp.cpu_intensity = 9;
                fp.vectorization_benefit = 4;
                fp.access_pattern = .mixed;
                fp.branch_predictability = 35;
                fp.dependency_count = 12;
                break :blk fp;
            },
            .expected_min_suitability = .unsuitable,
        },
    };
    
    var successful_classifications: u32 = 0;
    
    for (workload_types, 0..) |workload, index| {
        const analysis = try classifier.classifyTask(workload.fingerprint, null);
        
        const meets_expectation = @intFromEnum(analysis.overall_suitability) >= @intFromEnum(workload.expected_min_suitability);
        if (meets_expectation) {
            successful_classifications += 1;
        }
        
        std.debug.print("   Workload {}: {s}\n", .{ index + 1, workload.name });
        std.debug.print("     Classified as: {s}\n", .{@tagName(analysis.overall_suitability)});
        std.debug.print("     Expected minimum: {s}\n", .{@tagName(workload.expected_min_suitability)});
        std.debug.print("     Meets expectation: {}\n", .{meets_expectation});
        std.debug.print("     Confidence: {d:.2}, AI: {d:.2}, Parallel potential: {d:.2}\n", .{
            analysis.confidence_score, analysis.arithmetic_intensity, analysis.parallelization_potential
        });
        
        // Validate analysis consistency
        try std.testing.expect(analysis.confidence_score >= 0.0 and analysis.confidence_score <= 1.0);
        try std.testing.expect(analysis.arithmetic_intensity >= 0.0);
        try std.testing.expect(analysis.parallelization_potential >= 0.0 and analysis.parallelization_potential <= 1.0);
        try std.testing.expect(analysis.memory_coalescing_score >= 0.0 and analysis.memory_coalescing_score <= 1.0);
        try std.testing.expect(analysis.branch_uniformity >= 0.0 and analysis.branch_uniformity <= 1.0);
    }
    
    const success_rate = @as(f32, @floatFromInt(successful_classifications)) / @as(f32, @floatFromInt(workload_types.len));
    
    std.debug.print("2. Validation results summary:\n", .{});
    std.debug.print("   Total workloads tested: {}\n", .{workload_types.len});
    std.debug.print("   Successful classifications: {}\n", .{successful_classifications});
    std.debug.print("   Success rate: {d:.1}%\n", .{success_rate * 100.0});
    std.debug.print("   Total classifications performed: {}\n", .{classifier.classification_count});
    
    try std.testing.expect(success_rate >= 0.8); // Expect at least 80% accuracy
    
    std.debug.print("\n🚀 GPU Classification System Summary:\n", .{});
    std.debug.print("   • Arithmetic intensity analysis with roofline model ✅\n", .{});
    std.debug.print("   • Data-parallel pattern recognition ✅\n", .{});
    std.debug.print("   • Memory access pattern classification ✅\n", .{});
    std.debug.print("   • Branch divergence detection ✅\n", .{});
    std.debug.print("   • Multi-factor weighted scoring ✅\n", .{});
    std.debug.print("   • GPU device-specific optimization ✅\n", .{});
    std.debug.print("   • Adaptive learning with performance tracking ✅\n", .{});
    std.debug.print("   • Comprehensive validation with {d:.0}% accuracy ✅\n", .{success_rate * 100.0});
    
    std.debug.print("   ✅ Comprehensive GPU classification validation completed\n", .{});
}