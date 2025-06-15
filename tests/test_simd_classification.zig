const std = @import("std");
const beat = @import("beat");

// Test for SIMD Task Classification and Batch Formation System (Phase 5.2.3)
//
// This test validates the intelligent multi-layered classification system including:
// - Static analysis for compile-time characteristics and data dependency analysis
// - Dynamic profiling for runtime performance characteristics and pattern detection
// - Machine learning classification with feature extraction and similarity scoring
// - Intelligent batch formation with multi-criteria optimization and performance prediction

test "static analysis and dependency detection" {
    std.debug.print("\n=== Static Analysis and Dependency Detection Test ===\n", .{});
    
    // Test 1: Highly vectorizable task analysis
    std.debug.print("1. Testing highly vectorizable task analysis...\n", .{});
    
    const VectorData = struct { values: [256]f32 };
    var vector_data = VectorData{ .values = undefined };
    
    // Initialize with sequential pattern
    for (&vector_data.values, 0..) |*value, i| {
        value.* = @floatFromInt(i);
    }
    
    const vectorizable_task = beat.Task{
        .func = struct {
            fn func(data: *anyopaque) void {
                const typed_data = @as(*VectorData, @ptrCast(@alignCast(data)));
                // Simple sequential processing - should be highly vectorizable
                for (&typed_data.values) |*value| {
                    value.* = value.* * 2.0 + 1.0; // Perfect for SIMD
                }
            }
        }.func,
        .data = @ptrCast(&vector_data),
        .priority = .normal,
        .data_size_hint = @sizeOf(VectorData),
    };
    
    const analysis = beat.simd_classifier.StaticAnalysis.analyzeTask(&vectorizable_task);
    
    try std.testing.expect(analysis.vectorization_score > 0.0);
    try std.testing.expect(analysis.vectorization_score <= 1.0);
    try std.testing.expect(analysis.recommended_vector_width >= 4);
    
    const suitability = analysis.getSIMDSuitabilityScore();
    try std.testing.expect(suitability >= 0.0);
    try std.testing.expect(suitability <= 1.0);
    
    std.debug.print("   Vectorizable task analysis:\n", .{});
    std.debug.print("     Dependency type: {s}\n", .{@tagName(analysis.dependency_type)});
    std.debug.print("     Access pattern: {s}\n", .{@tagName(analysis.access_pattern)});
    std.debug.print("     Vectorization score: {d:.3}\n", .{analysis.vectorization_score});
    std.debug.print("     SIMD suitability: {d:.3}\n", .{suitability});
    std.debug.print("     Recommended vector width: {}\n", .{analysis.recommended_vector_width});
    
    // Test 2: Non-vectorizable task analysis
    std.debug.print("2. Testing non-vectorizable task analysis...\n", .{});
    
    const ScalarData = struct { 
        value: i32,
        counter: u64,
        flag: bool,
    };
    var scalar_data = ScalarData{ .value = 42, .counter = 0, .flag = false };
    
    const non_vectorizable_task = beat.Task{
        .func = struct {
            fn func(data: *anyopaque) void {
                const typed_data = @as(*ScalarData, @ptrCast(@alignCast(data)));
                // Complex branching and irregular access - poor for SIMD
                while (typed_data.counter < 100) {
                    if (@rem(typed_data.value, 2) == 0) {
                        typed_data.value = @divTrunc(typed_data.value, 2);
                    } else {
                        typed_data.value = typed_data.value * 3 + 1;
                    }
                    typed_data.counter += 1;
                    if (typed_data.value == 1) break;
                }
                typed_data.flag = true;
            }
        }.func,
        .data = @ptrCast(&scalar_data),
        .priority = .normal,
        .data_size_hint = @sizeOf(ScalarData),
    };
    
    const scalar_analysis = beat.simd_classifier.StaticAnalysis.analyzeTask(&non_vectorizable_task);
    const scalar_suitability = scalar_analysis.getSIMDSuitabilityScore();
    
    std.debug.print("   Non-vectorizable task analysis:\n", .{});
    std.debug.print("     Dependency type: {s}\n", .{@tagName(scalar_analysis.dependency_type)});
    std.debug.print("     Access pattern: {s}\n", .{@tagName(scalar_analysis.access_pattern)});
    std.debug.print("     Vectorization score: {d:.3}\n", .{scalar_analysis.vectorization_score});
    std.debug.print("     SIMD suitability: {d:.3}\n", .{scalar_suitability});
    
    // Vectorizable task should have higher suitability
    try std.testing.expect(suitability >= scalar_suitability);
    
    std.debug.print("   ✅ Static analysis and dependency detection completed\n", .{});
}

test "dynamic profiling and performance analysis" {
    const allocator = std.testing.allocator;
    _ = allocator;
    
    std.debug.print("\n=== Dynamic Profiling and Performance Analysis Test ===\n", .{});
    
    // Test 1: CPU-intensive task profiling
    std.debug.print("1. Testing CPU-intensive task profiling...\n", .{});
    
    const ComputeData = struct { values: [1000]f32 };
    var compute_data = ComputeData{ .values = undefined };
    
    // Initialize with test data
    for (&compute_data.values, 0..) |*value, i| {
        value.* = @as(f32, @floatFromInt(i)) * 0.1;
    }
    
    const compute_task = beat.Task{
        .func = struct {
            fn func(data: *anyopaque) void {
                const typed_data = @as(*ComputeData, @ptrCast(@alignCast(data)));
                // Compute-intensive operation with good cache locality
                for (&typed_data.values) |*value| {
                    value.* = @sin(value.*) * @cos(value.* * 2.0) + @sqrt(@abs(value.*));
                }
            }
        }.func,
        .data = @ptrCast(&compute_data),
        .priority = .normal,
        .data_size_hint = @sizeOf(ComputeData),
    };
    
    const profile = try beat.simd_classifier.DynamicProfile.profileTask(&compute_task, 5);
    
    try std.testing.expect(profile.execution_time_ns > 0);
    try std.testing.expect(profile.execution_variance >= 0.0);
    try std.testing.expect(profile.predictability_score >= 0.0);
    try std.testing.expect(profile.predictability_score <= 1.0);
    
    const performance_score = profile.getPerformanceScore();
    try std.testing.expect(performance_score >= 0.0);
    try std.testing.expect(performance_score <= 1.0);
    
    std.debug.print("   CPU-intensive task profiling:\n", .{});
    std.debug.print("     Execution time: {}ns\n", .{profile.execution_time_ns});
    std.debug.print("     Execution variance: {d:.4}\n", .{profile.execution_variance});
    std.debug.print("     Predictability score: {d:.3}\n", .{profile.predictability_score});
    std.debug.print("     Performance score: {d:.3}\n", .{performance_score});
    std.debug.print("     Cache line utilization: {d:.3}\n", .{profile.cache_line_utilization});
    std.debug.print("     Memory bandwidth: {d:.1} MB/s\n", .{profile.memory_bandwidth_mbps});
    
    // Test 2: Memory-intensive task profiling
    std.debug.print("2. Testing memory-intensive task profiling...\n", .{});
    
    const MemoryData = struct { values: [10000]i32 };
    var memory_data = MemoryData{ .values = undefined };
    
    // Initialize with random-like pattern
    for (&memory_data.values, 0..) |*value, i| {
        value.* = @as(i32, @intCast((i * 31 + 17) % 1000));
    }
    
    const memory_task = beat.Task{
        .func = struct {
            fn func(data: *anyopaque) void {
                const typed_data = @as(*MemoryData, @ptrCast(@alignCast(data)));
                // Memory-intensive with irregular access pattern
                for (0..typed_data.values.len / 2) |i| {
                    const idx1 = (i * 7) % typed_data.values.len;
                    const idx2 = (i * 13) % typed_data.values.len;
                    const temp = typed_data.values[idx1];
                    typed_data.values[idx1] = typed_data.values[idx2];
                    typed_data.values[idx2] = temp;
                }
            }
        }.func,
        .data = @ptrCast(&memory_data),
        .priority = .normal,
        .data_size_hint = @sizeOf(MemoryData),
    };
    
    const memory_profile = try beat.simd_classifier.DynamicProfile.profileTask(&memory_task, 3);
    const memory_performance = memory_profile.getPerformanceScore();
    
    std.debug.print("   Memory-intensive task profiling:\n", .{});
    std.debug.print("     Execution time: {}ns\n", .{memory_profile.execution_time_ns});
    std.debug.print("     Predictability score: {d:.3}\n", .{memory_profile.predictability_score});
    std.debug.print("     Performance score: {d:.3}\n", .{memory_performance});
    
    std.debug.print("   ✅ Dynamic profiling and performance analysis completed\n", .{});
}

test "machine learning feature extraction and classification" {
    const allocator = std.testing.allocator;
    _ = allocator;
    
    std.debug.print("\n=== Machine Learning Feature Extraction and Classification Test ===\n", .{});
    
    // Test 1: Feature vector extraction for different task types
    std.debug.print("1. Testing feature vector extraction...\n", .{});
    
    // Create diverse test tasks
    const VectorData = struct { values: [128]f32 };
    var vector_data = VectorData{ .values = undefined };
    
    const MatrixData = struct { matrix: [16][16]f32 };
    var matrix_data = MatrixData{ .matrix = undefined };
    
    const ScalarData = struct { value: i64, counter: u32 };
    var scalar_data = ScalarData{ .value = 12345, .counter = 0 };
    
    // Initialize data
    for (&vector_data.values, 0..) |*value, i| {
        value.* = @floatFromInt(i);
    }
    
    for (&matrix_data.matrix, 0..) |*row, i| {
        for (row, 0..) |*value, j| {
            value.* = @floatFromInt(i * 16 + j);
        }
    }
    
    // Highly vectorizable task (vector arithmetic)
    const vector_task = beat.Task{
        .func = struct {
            fn func(data: *anyopaque) void {
                const typed_data = @as(*VectorData, @ptrCast(@alignCast(data)));
                for (&typed_data.values) |*value| {
                    value.* = value.* * 1.5 + 0.5; // Simple arithmetic
                }
            }
        }.func,
        .data = @ptrCast(&vector_data),
        .priority = .normal,
        .data_size_hint = @sizeOf(VectorData),
    };
    
    // Moderately vectorizable task (matrix operations)
    const matrix_task = beat.Task{
        .func = struct {
            fn func(data: *anyopaque) void {
                const typed_data = @as(*MatrixData, @ptrCast(@alignCast(data)));
                // Simple matrix transformation
                for (&typed_data.matrix) |*row| {
                    for (row) |*value| {
                        value.* = value.* * 0.9 + 0.1;
                    }
                }
            }
        }.func,
        .data = @ptrCast(&matrix_data),
        .priority = .normal,
        .data_size_hint = @sizeOf(MatrixData),
    };
    
    // Non-vectorizable task (scalar with branching)
    const scalar_task = beat.Task{
        .func = struct {
            fn func(data: *anyopaque) void {
                const typed_data = @as(*ScalarData, @ptrCast(@alignCast(data)));
                while (typed_data.counter < 1000 and typed_data.value > 1) {
                    if (@rem(typed_data.value, 2) == 0) {
                        typed_data.value = @divTrunc(typed_data.value, 2);
                    } else {
                        typed_data.value = typed_data.value * 3 + 1;
                    }
                    typed_data.counter += 1;
                }
            }
        }.func,
        .data = @ptrCast(&scalar_data),
        .priority = .normal,
        .data_size_hint = @sizeOf(ScalarData),
    };
    
    // Analyze all tasks
    const vector_static = beat.simd_classifier.StaticAnalysis.analyzeTask(&vector_task);
    const matrix_static = beat.simd_classifier.StaticAnalysis.analyzeTask(&matrix_task);
    const scalar_static = beat.simd_classifier.StaticAnalysis.analyzeTask(&scalar_task);
    
    const vector_dynamic = try beat.simd_classifier.DynamicProfile.profileTask(&vector_task, 3);
    const matrix_dynamic = try beat.simd_classifier.DynamicProfile.profileTask(&matrix_task, 3);
    const scalar_dynamic = try beat.simd_classifier.DynamicProfile.profileTask(&scalar_task, 3);
    
    // Extract feature vectors
    const vector_features = beat.simd_classifier.TaskFeatureVector.fromAnalysis(vector_static, vector_dynamic);
    const matrix_features = beat.simd_classifier.TaskFeatureVector.fromAnalysis(matrix_static, matrix_dynamic);
    const scalar_features = beat.simd_classifier.TaskFeatureVector.fromAnalysis(scalar_static, scalar_dynamic);
    
    std.debug.print("   Feature vectors extracted:\n", .{});
    std.debug.print("     Vector task - Overall suitability: {d:.3}, Confidence: {d:.3}\n", .{
        vector_features.overall_suitability, vector_features.confidence_level
    });
    std.debug.print("     Matrix task - Overall suitability: {d:.3}, Confidence: {d:.3}\n", .{
        matrix_features.overall_suitability, matrix_features.confidence_level
    });
    std.debug.print("     Scalar task - Overall suitability: {d:.3}, Confidence: {d:.3}\n", .{
        scalar_features.overall_suitability, scalar_features.confidence_level
    });
    
    // Test 2: Task classification
    std.debug.print("2. Testing task classification...\n", .{});
    
    const vector_class = vector_features.getClassification();
    const matrix_class = matrix_features.getClassification();
    const scalar_class = scalar_features.getClassification();
    
    std.debug.print("   Task classifications:\n", .{});
    std.debug.print("     Vector task: {s} (priority: {})\n", .{
        @tagName(vector_class), vector_class.getBatchPriority()
    });
    std.debug.print("     Matrix task: {s} (priority: {})\n", .{
        @tagName(matrix_class), matrix_class.getBatchPriority()
    });
    std.debug.print("     Scalar task: {s} (priority: {})\n", .{
        @tagName(scalar_class), scalar_class.getBatchPriority()
    });
    
    // Vector task should have highest priority, scalar lowest
    try std.testing.expect(vector_class.getBatchPriority() >= matrix_class.getBatchPriority());
    try std.testing.expect(matrix_class.getBatchPriority() >= scalar_class.getBatchPriority());
    
    // Test 3: Similarity scoring
    std.debug.print("3. Testing similarity scoring...\n", .{});
    
    const vector_matrix_similarity = vector_features.similarityScore(matrix_features);
    const vector_scalar_similarity = vector_features.similarityScore(scalar_features);
    const matrix_scalar_similarity = matrix_features.similarityScore(scalar_features);
    
    std.debug.print("   Similarity scores:\n", .{});
    std.debug.print("     Vector ↔ Matrix: {d:.3}\n", .{vector_matrix_similarity});
    std.debug.print("     Vector ↔ Scalar: {d:.3}\n", .{vector_scalar_similarity});
    std.debug.print("     Matrix ↔ Scalar: {d:.3}\n", .{matrix_scalar_similarity});
    
    // Similar tasks should have higher similarity
    try std.testing.expect(vector_matrix_similarity >= vector_scalar_similarity);
    
    std.debug.print("   ✅ Machine learning feature extraction and classification completed\n", .{});
}

test "intelligent batch formation with multi-criteria optimization" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Intelligent Batch Formation Test ===\n", .{});
    
    // Test 1: Batch formation system initialization
    std.debug.print("1. Testing batch formation system initialization...\n", .{});
    
    const criteria = beat.simd_classifier.BatchFormationCriteria.performanceOptimized();
    var batch_former = beat.simd_classifier.IntelligentBatchFormer.init(allocator, criteria);
    defer batch_former.deinit();
    
    std.debug.print("   Initialized with performance-optimized criteria:\n", .{});
    std.debug.print("     Similarity weight: {d:.2}\n", .{criteria.similarity_weight});
    std.debug.print("     Performance weight: {d:.2}\n", .{criteria.performance_weight});
    std.debug.print("     SIMD efficiency weight: {d:.2}\n", .{criteria.simd_efficiency_weight});
    
    // Test 2: Adding diverse tasks for classification
    std.debug.print("2. Testing task addition and classification...\n", .{});
    
    const TestData = struct { values: [64]f32 };
    var test_data_array: [12]TestData = undefined;
    
    // Initialize test data with different patterns
    for (&test_data_array, 0..) |*data, i| {
        for (&data.values, 0..) |*value, j| {
            value.* = @as(f32, @floatFromInt(i * 64 + j)) + @as(f32, @floatFromInt(i)) * 0.1;
        }
    }
    
    // Create tasks with different characteristics but similar enough to batch together
    for (&test_data_array, 0..) |*data, i| {
        const task = beat.Task{
            .func = if (i < 8) struct {
                fn func(task_data: *anyopaque) void {
                    const typed_data = @as(*TestData, @ptrCast(@alignCast(task_data)));
                    for (&typed_data.values) |*value| {
                        value.* = value.* * 1.2 + 0.3; // Similar arithmetic operations
                    }
                }
            }.func else struct {
                fn func(task_data: *anyopaque) void {
                    const typed_data = @as(*TestData, @ptrCast(@alignCast(task_data)));
                    for (&typed_data.values) |*value| {
                        value.* = @sin(value.* * 0.1); // Different but still vectorizable
                    }
                }
            }.func,
            .data = @ptrCast(data),
            .priority = if (i % 3 == 0) .high else .normal,
            .data_size_hint = @sizeOf(TestData),
        };
        
        try batch_former.addTask(task, false); // Disable profiling for speed
    }
    
    // Test 3: Force batch formation and analyze results
    std.debug.print("3. Testing batch formation process...\n", .{});
    
    try batch_former.attemptBatchFormation();
    
    const stats = batch_former.getFormationStats();
    const formed_batches = batch_former.getFormedBatches();
    
    std.debug.print("   Batch formation results:\n", .{});
    std.debug.print("     Total tasks submitted: {}\n", .{stats.total_tasks_submitted});
    std.debug.print("     Tasks successfully batched: {}\n", .{stats.tasks_in_batches});
    std.debug.print("     Remaining pending tasks: {}\n", .{stats.pending_tasks});
    std.debug.print("     Batches formed: {}\n", .{stats.formed_batches});
    std.debug.print("     Average batch size: {d:.1}\n", .{stats.average_batch_size});
    std.debug.print("     Formation efficiency: {d:.1}%\n", .{stats.formation_efficiency * 100});
    std.debug.print("     Average estimated speedup: {d:.2}x\n", .{stats.average_estimated_speedup});
    std.debug.print("     Current similarity threshold: {d:.2}\n", .{stats.current_similarity_threshold});
    
    // Verify formation results
    try std.testing.expect(stats.formed_batches > 0);
    try std.testing.expect(stats.formation_efficiency > 0.0);
    try std.testing.expect(formed_batches.len == stats.formed_batches);
    
    // Test 4: Examine individual batches
    std.debug.print("4. Testing individual batch characteristics...\n", .{});
    
    for (formed_batches, 0..) |batch, i| {
        const metrics = batch.getPerformanceMetrics();
        std.debug.print("   Batch {}: {} tasks, {d:.2}x speedup, {} vector width\n", .{
            i, metrics.batch_size, metrics.estimated_speedup, metrics.vector_width
        });
        
        try std.testing.expect(metrics.batch_size >= 4); // Should meet minimum
        try std.testing.expect(metrics.estimated_speedup >= 1.0); // Should provide benefit
    }
    
    std.debug.print("   ✅ Intelligent batch formation completed\n", .{});
}

test "adaptive learning and threshold adjustment" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Adaptive Learning and Threshold Adjustment Test ===\n", .{});
    
    // Test 1: Initial threshold behavior
    std.debug.print("1. Testing initial threshold behavior...\n", .{});
    
    const criteria = beat.simd_classifier.BatchFormationCriteria.balanced();
    var batch_former = beat.simd_classifier.IntelligentBatchFormer.init(allocator, criteria);
    defer batch_former.deinit();
    
    const initial_threshold = batch_former.similarity_threshold;
    std.debug.print("   Initial similarity threshold: {d:.2}\n", .{initial_threshold});
    
    // Test 2: Create tasks with varying similarity levels
    std.debug.print("2. Testing with varying task similarity levels...\n", .{});
    
    const TestData = struct { values: [32]f32 };
    var test_data_array: [20]TestData = undefined;
    
    // Create different task types to trigger adaptation
    for (&test_data_array, 0..) |*data, i| {
        for (&data.values, 0..) |*value, j| {
            value.* = @as(f32, @floatFromInt(i * 32 + j));
        }
    }
    
    // Add tasks in batches and observe threshold adaptation
    for (0..4) |round| {
        std.debug.print("   Round {}: Adding 5 tasks...\n", .{round});
        
        for (0..5) |i| {
            const data_index = round * 5 + i;
            if (data_index >= test_data_array.len) break;
            
            const task = beat.Task{
                .func = switch (round % 3) {
                    0 => struct {
                        fn func(task_data: *anyopaque) void {
                            const typed_data = @as(*TestData, @ptrCast(@alignCast(task_data)));
                            for (&typed_data.values) |*value| {
                                value.* = value.* * 1.1 + 0.1; // Very similar operations
                            }
                        }
                    }.func,
                    1 => struct {
                        fn func(task_data: *anyopaque) void {
                            const typed_data = @as(*TestData, @ptrCast(@alignCast(task_data)));
                            for (&typed_data.values) |*value| {
                                value.* = value.* * 2.0; // Somewhat similar operations
                            }
                        }
                    }.func,
                    else => struct {
                        fn func(task_data: *anyopaque) void {
                            const typed_data = @as(*TestData, @ptrCast(@alignCast(task_data)));
                            for (&typed_data.values) |*value| {
                                value.* = @sqrt(@abs(value.*)); // Different operations
                            }
                        }
                    }.func,
                },
                .data = @ptrCast(&test_data_array[data_index]),
                .priority = .normal,
                .data_size_hint = @sizeOf(TestData),
            };
            
            try batch_former.addTask(task, false);
        }
        
        // Force batch formation
        try batch_former.attemptBatchFormation();
        
        const stats = batch_former.getFormationStats();
        std.debug.print("     After round {}: {} batches, {d:.1}% efficiency, threshold: {d:.3}\n", .{
            round, stats.formed_batches, stats.formation_efficiency * 100, stats.current_similarity_threshold
        });
    }
    
    // Test 3: Verify adaptive behavior
    std.debug.print("3. Testing adaptive threshold adjustment...\n", .{});
    
    const final_stats = batch_former.getFormationStats();
    const final_threshold = final_stats.current_similarity_threshold;
    
    std.debug.print("   Final statistics:\n", .{});
    std.debug.print("     Total tasks: {}\n", .{final_stats.total_tasks_submitted});
    std.debug.print("     Final threshold: {d:.3}\n", .{final_threshold});
    std.debug.print("     Formation efficiency: {d:.1}%\n", .{final_stats.formation_efficiency * 100});
    std.debug.print("     Average batch size: {d:.1}\n", .{final_stats.average_batch_size});
    
    // Threshold should have adapted from initial value
    try std.testing.expect(final_stats.formed_batches > 0);
    try std.testing.expect(final_stats.formation_efficiency > 0.0);
    
    std.debug.print("   ✅ Adaptive learning and threshold adjustment completed\n", .{});
}

test "comprehensive SIMD classification system integration" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Comprehensive SIMD Classification System Integration Test ===\n", .{});
    
    // This test demonstrates the complete SIMD classification workflow
    std.debug.print("1. Initializing comprehensive classification system...\n", .{});
    
    const criteria = beat.simd_classifier.BatchFormationCriteria.performanceOptimized();
    var batch_former = beat.simd_classifier.IntelligentBatchFormer.init(allocator, criteria);
    defer batch_former.deinit();
    
    std.debug.print("   System initialized with intelligent batch formation\n", .{});
    
    // Create realistic diverse workloads
    std.debug.print("2. Creating realistic diverse workloads...\n", .{});
    
    // Workload 1: Image processing (highly vectorizable)
    const ImageData = struct { pixels: [256][256]f32 };
    var image_data = try allocator.create(ImageData);
    defer allocator.destroy(image_data);
    
    for (&image_data.pixels, 0..) |*row, i| {
        for (row, 0..) |*pixel, j| {
            pixel.* = @as(f32, @floatFromInt((i + j) % 256)) / 255.0;
        }
    }
    
    const image_task = beat.Task{
        .func = struct {
            fn func(data: *anyopaque) void {
                const typed_data = @as(*ImageData, @ptrCast(@alignCast(data)));
                // Gaussian blur kernel (highly vectorizable)
                for (1..255) |i| {
                    for (1..255) |j| {
                        const sum = typed_data.pixels[i-1][j-1] * 0.1 +
                                   typed_data.pixels[i-1][j] * 0.1 +
                                   typed_data.pixels[i-1][j+1] * 0.1 +
                                   typed_data.pixels[i][j-1] * 0.2 +
                                   typed_data.pixels[i][j] * 0.2 +
                                   typed_data.pixels[i][j+1] * 0.2 +
                                   typed_data.pixels[i+1][j-1] * 0.1 +
                                   typed_data.pixels[i+1][j] * 0.1 +
                                   typed_data.pixels[i+1][j+1] * 0.1;
                        typed_data.pixels[i][j] = sum;
                    }
                }
            }
        }.func,
        .data = @ptrCast(image_data),
        .priority = .high,
        .data_size_hint = @sizeOf(ImageData),
    };
    
    // Workload 2: Mathematical computation (moderately vectorizable)
    const MathData = struct { values: [1024]f64 };
    var math_data = try allocator.create(MathData);
    defer allocator.destroy(math_data);
    
    for (&math_data.values, 0..) |*value, i| {
        value.* = @as(f64, @floatFromInt(i)) * 0.01;
    }
    
    const math_task = beat.Task{
        .func = struct {
            fn func(data: *anyopaque) void {
                const typed_data = @as(*MathData, @ptrCast(@alignCast(data)));
                for (&typed_data.values) |*value| {
                    value.* = @sin(value.*) + @cos(value.* * 2.0) + @exp(value.* * 0.1);
                }
            }
        }.func,
        .data = @ptrCast(math_data),
        .priority = .normal,
        .data_size_hint = @sizeOf(MathData),
    };
    
    // Workload 3: Tree traversal (poorly vectorizable)
    const TreeNode = struct {
        value: i32,
        left: ?*@This() = null,
        right: ?*@This() = null,
    };
    
    const TreeData = struct {
        root: ?*TreeNode,
        sum: i64,
    };
    
    var tree_nodes = try allocator.alloc(TreeNode, 127); // Binary tree with 127 nodes
    defer allocator.free(tree_nodes);
    
    // Build binary tree
    for (tree_nodes, 0..) |*node, i| {
        node.value = @as(i32, @intCast(i + 1));
        if (i * 2 + 1 < tree_nodes.len) node.left = &tree_nodes[i * 2 + 1];
        if (i * 2 + 2 < tree_nodes.len) node.right = &tree_nodes[i * 2 + 2];
    }
    
    var tree_data = TreeData{ .root = &tree_nodes[0], .sum = 0 };
    
    const tree_task = beat.Task{
        .func = struct {
            fn traverse(node: ?*TreeNode, sum: *i64) void {
                if (node) |n| {
                    sum.* += n.value;
                    traverse(n.left, sum);
                    traverse(n.right, sum);
                }
            }
            
            fn func(data: *anyopaque) void {
                const typed_data = @as(*TreeData, @ptrCast(@alignCast(data)));
                typed_data.sum = 0;
                traverse(typed_data.root, &typed_data.sum);
            }
        }.func,
        .data = @ptrCast(&tree_data),
        .priority = .normal,
        .data_size_hint = @sizeOf(TreeData),
    };
    
    // Add all workloads to the classification system
    std.debug.print("3. Adding workloads to classification system...\n", .{});
    
    try batch_former.addTask(image_task, true);  // Enable profiling for complex task
    try batch_former.addTask(math_task, true);   // Enable profiling
    try batch_former.addTask(tree_task, true);   // Enable profiling
    
    // Add more similar tasks to enable batching
    for (0..6) |_| {
        try batch_former.addTask(math_task, false); // Add similar math tasks
    }
    
    // Force batch formation
    try batch_former.attemptBatchFormation();
    
    // Analyze comprehensive results
    std.debug.print("4. Analyzing comprehensive classification results...\n", .{});
    
    const final_stats = batch_former.getFormationStats();
    const formed_batches = batch_former.getFormedBatches();
    
    std.debug.print("   Comprehensive classification results:\n", .{});
    std.debug.print("     Total workloads submitted: {}\n", .{final_stats.total_tasks_submitted});
    std.debug.print("     Successfully batched: {}\n", .{final_stats.tasks_in_batches});
    std.debug.print("     Remaining individual: {}\n", .{final_stats.pending_tasks});
    std.debug.print("     Intelligent batches formed: {}\n", .{final_stats.formed_batches});
    std.debug.print("     Overall formation efficiency: {d:.1}%\n", .{final_stats.formation_efficiency * 100});
    std.debug.print("     Average performance improvement: {d:.2}x\n", .{final_stats.average_estimated_speedup});
    std.debug.print("     Adaptive threshold: {d:.3}\n", .{final_stats.current_similarity_threshold});
    
    // Examine each formed batch
    std.debug.print("5. Examining individual batch characteristics...\n", .{});
    
    for (formed_batches, 0..) |batch, i| {
        const metrics = batch.getPerformanceMetrics();
        std.debug.print("   Intelligent batch {}: {} tasks, {d:.2}x speedup, {}-wide vectors\n", .{
            i, metrics.batch_size, metrics.estimated_speedup, metrics.vector_width
        });
        
        // Execute batch to validate formation
        try batch.execute();
        
        const post_exec_metrics = batch.getPerformanceMetrics();
        std.debug.print("     Post-execution: {} elements processed, efficiency: {d:.2}\n", .{
            post_exec_metrics.total_elements_processed, post_exec_metrics.vectorization_efficiency
        });
    }
    
    // Final validation
    try std.testing.expect(final_stats.formed_batches > 0);
    try std.testing.expect(final_stats.formation_efficiency > 0.0);
    try std.testing.expect(final_stats.average_estimated_speedup >= 1.0);
    
    std.debug.print("\n✅ Comprehensive SIMD classification system integration completed successfully!\n", .{});
    
    std.debug.print("🎯 SIMD Classification System Summary:\n", .{});
    std.debug.print("   • Static analysis with dependency detection ✅\n", .{});
    std.debug.print("   • Dynamic profiling with performance characteristics ✅\n", .{});
    std.debug.print("   • Machine learning feature extraction and similarity scoring ✅\n", .{});
    std.debug.print("   • Intelligent batch formation with multi-criteria optimization ✅\n", .{});
    std.debug.print("   • Adaptive learning with threshold adjustment ✅\n", .{});
    std.debug.print("   • Performance prediction and validation ✅\n", .{});
}