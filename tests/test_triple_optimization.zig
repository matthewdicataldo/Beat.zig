// Triple-Optimization Pipeline Tests
// Validates Souper + Minotaur + ISPC integration for comprehensive optimization

const std = @import("std");
const testing = std.testing;
const beat = @import("beat");

const triple_optimization = @import("triple_optimization");

test "triple optimization config initialization" {
    const config = triple_optimization.TripleOptimizationConfig{
        .enabled = true,
        .optimization_level = .balanced,
        .souper = .{ .enabled = true },
        .minotaur = .{ .enabled = true },
        .ispc = .{ .enabled = true },
        .optimization_order = .sequential,
        .enable_cross_optimization = true,
    };
    
    try testing.expectEqual(true, config.enabled);
    try testing.expectEqual(triple_optimization.OptimizationLevel.balanced, config.optimization_level);
    try testing.expectEqual(true, config.souper.enabled);
    try testing.expectEqual(true, config.minotaur.enabled);
    try testing.expectEqual(true, config.ispc.enabled);
    try testing.expectEqual(triple_optimization.OptimizationOrder.sequential, config.optimization_order);
    try testing.expectEqual(true, config.enable_cross_optimization);
}

test "optimization level enumeration" {
    const levels = [_]triple_optimization.OptimizationLevel{
        .conservative,
        .balanced,
        .aggressive,
        .experimental,
    };
    
    for (levels) |level| {
        try testing.expect(@intFromEnum(level) >= 0);
    }
    
    // Test that balanced is the expected default
    const default_config = triple_optimization.TripleOptimizationConfig{};
    try testing.expectEqual(triple_optimization.OptimizationLevel.balanced, default_config.optimization_level);
}

test "optimization order strategies" {
    const orders = [_]triple_optimization.OptimizationOrder{
        .sequential,
        .parallel,
        .adaptive,
        .iterative,
    };
    
    for (orders) |order| {
        try testing.expect(@intFromEnum(order) >= 0);
    }
}

test "verification level configuration" {
    const levels = [_]triple_optimization.VerificationLevel{
        .none,
        .basic,
        .formal,
        .exhaustive,
    };
    
    for (levels) |level| {
        try testing.expect(@intFromEnum(level) >= 0);
    }
    
    // Test default verification level
    const default_config = triple_optimization.TripleOptimizationConfig{};
    try testing.expectEqual(triple_optimization.VerificationLevel.formal, default_config.verification_level);
}

test "ispc configuration" {
    const ispc_config = triple_optimization.ISPCConfig{
        .enabled = true,
        .target_width = 8,
        .instruction_sets = &[_]triple_optimization.ISPCTarget{.avx2_i32x8},
        .optimization_level = 2,
        .enable_fast_math = true,
        .enable_fma = true,
    };
    
    try testing.expectEqual(true, ispc_config.enabled);
    try testing.expectEqual(@as(u32, 8), ispc_config.target_width);
    try testing.expectEqual(@as(i32, 2), ispc_config.optimization_level);
    try testing.expectEqual(true, ispc_config.enable_fast_math);
    try testing.expectEqual(true, ispc_config.enable_fma);
}

test "ispc target architectures" {
    const targets = [_]triple_optimization.ISPCTarget{
        .sse2_i32x4,
        .sse4_i32x4,
        .avx1_i32x8,
        .avx2_i32x8,
        .avx512knl_i32x16,
        .avx512skx_i32x16,
        .neon_i32x4,
        .generic,
    };
    
    for (targets) |target| {
        try testing.expect(@intFromEnum(target) >= 0);
    }
}

test "triple optimization engine initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const config = triple_optimization.TripleOptimizationConfig{
        .enabled = true,
        .souper = .{ .enabled = false }, // Disabled for testing
        .minotaur = .{ .enabled = false }, // Disabled for testing  
        .ispc = .{ .enabled = false }, // Disabled for testing
    };
    
    var engine = try triple_optimization.TripleOptimizationEngine.init(allocator, config);
    
    try testing.expectEqual(true, engine.config.enabled);
    try testing.expectEqual(@as(u64, 0), engine.total_optimizations_found.load(.acquire));
    try testing.expectEqual(@as(u64, 0), engine.total_optimizations_applied.load(.acquire));
    try testing.expectEqual(@as(u64, 0), engine.total_cycles_saved.load(.acquire));
    
    // Individual engines should be null when disabled
    try testing.expectEqual(@as(?triple_optimization.SouperIntegration, null), engine.souper);
    try testing.expectEqual(@as(?triple_optimization.MinotaurIntegration, null), engine.minotaur);
    try testing.expectEqual(@as(?triple_optimization.ISPCIntegration, null), engine.ispc);
}

test "statistics tracking" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const config = triple_optimization.TripleOptimizationConfig{
        .enabled = true,
        .souper = .{ .enabled = false },
        .minotaur = .{ .enabled = false },
        .ispc = .{ .enabled = false },
    };
    
    var engine = try triple_optimization.TripleOptimizationEngine.init(allocator, config);
    
    // Simulate some optimization activity
    _ = engine.total_optimizations_found.fetchAdd(15, .monotonic);
    _ = engine.total_optimizations_applied.fetchAdd(12, .monotonic);
    _ = engine.total_cycles_saved.fetchAdd(1500, .monotonic);
    _ = engine.scalar_optimizations.fetchAdd(5, .monotonic);
    _ = engine.simd_optimizations.fetchAdd(7, .monotonic);
    _ = engine.spmd_optimizations.fetchAdd(3, .monotonic);
    _ = engine.cross_optimizations.fetchAdd(2, .monotonic);
    
    const stats = engine.getStatistics();
    try testing.expectEqual(@as(u64, 15), stats.total_optimizations_found);
    try testing.expectEqual(@as(u64, 12), stats.total_optimizations_applied);
    try testing.expectEqual(@as(u64, 1500), stats.total_cycles_saved);
    try testing.expectEqual(@as(u32, 5), stats.scalar_optimizations);
    try testing.expectEqual(@as(u32, 7), stats.simd_optimizations);
    try testing.expectEqual(@as(u32, 3), stats.spmd_optimizations);
    try testing.expectEqual(@as(u32, 2), stats.cross_optimizations);
}

test "disabled triple optimization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const config = triple_optimization.TripleOptimizationConfig{
        .enabled = false, // Disabled
    };
    
    var engine = try triple_optimization.TripleOptimizationEngine.init(allocator, config);
    
    // Should return empty report when disabled
    const report = try engine.optimizeCodebase();
    try testing.expectEqual(@as(u32, 0), report.total_optimizations);
}

test "individual configuration components" {
    // Test that individual configs can be customized
    const config = triple_optimization.TripleOptimizationConfig{
        .enabled = true,
        .souper = .{
            .enabled = true,
            .verification_timeout_ms = 10000,
        },
        .minotaur = .{
            .enabled = true,
            .target_vector_width = 512,
            .synthesis_timeout_ms = 15000,
        },
        .ispc = .{
            .enabled = true,
            .target_width = 16,
            .optimization_level = 3,
        },
    };
    
    try testing.expectEqual(true, config.souper.enabled);
    try testing.expectEqual(@as(u32, 512), config.minotaur.target_vector_width);
    try testing.expectEqual(@as(u32, 16), config.ispc.target_width);
    try testing.expectEqual(@as(i32, 3), config.ispc.optimization_level);
}

test "output configuration" {
    const config = triple_optimization.TripleOptimizationConfig{
        .generate_reports = true,
        .output_directory = "custom_output",
        .benchmark_optimizations = true,
    };
    
    try testing.expectEqual(true, config.generate_reports);
    try testing.expectEqualStrings("custom_output", config.output_directory);
    try testing.expectEqual(true, config.benchmark_optimizations);
}

test "cross optimization settings" {
    const config = triple_optimization.TripleOptimizationConfig{
        .enable_cross_optimization = true,
        .optimization_order = .iterative,
    };
    
    try testing.expectEqual(true, config.enable_cross_optimization);
    try testing.expectEqual(triple_optimization.OptimizationOrder.iterative, config.optimization_order);
}

test "threadpool integration" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    var pool = try beat.createTestPool(allocator);
    defer pool.deinit();
    
    const config = triple_optimization.TripleOptimizationConfig{
        .enabled = false, // Disabled for testing to avoid external dependencies
    };
    
    // Should complete without error when disabled
    try triple_optimization.enableTripleOptimization(&pool, config);
}

test "triple optimization test function" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Test the comprehensive test function
    try triple_optimization.testTripleOptimization(allocator);
}

test "default ispc configuration" {
    const default_ispc = triple_optimization.ISPCConfig{};
    
    try testing.expectEqual(true, default_ispc.enabled);
    try testing.expectEqual(@as(u32, 8), default_ispc.target_width);
    try testing.expectEqual(@as(i32, 2), default_ispc.optimization_level);
    try testing.expectEqual(true, default_ispc.enable_fast_math);
    try testing.expectEqual(true, default_ispc.enable_fma);
    try testing.expectEqual(true, default_ispc.auto_vectorization);
    try testing.expectEqual(@as(u32, 10), default_ispc.inline_threshold);
    try testing.expectEqual(true, default_ispc.generate_headers);
    try testing.expectEqual(false, default_ispc.output_assembly);
}

test "comprehensive triple optimization config" {
    const comprehensive_config = triple_optimization.TripleOptimizationConfig{
        .enabled = true,
        .optimization_level = .aggressive,
        .souper = .{
            .enabled = true,
            .redis_cache = true,
            .verification_timeout_ms = 30000,
        },
        .minotaur = .{
            .enabled = true,
            .target_vector_width = 512,
            .verify_optimizations = true,
            .combine_with_souper = true,
            .combine_with_ispc = true,
        },
        .ispc = .{
            .enabled = true,
            .target_width = 16,
            .instruction_sets = &[_]triple_optimization.ISPCTarget{
                .avx512skx_i32x16,
                .avx2_i32x8,
            },
            .optimization_level = 3,
            .enable_fast_math = true,
            .enable_fma = true,
        },
        .optimization_order = .adaptive,
        .enable_cross_optimization = true,
        .benchmark_optimizations = true,
        .verification_level = .formal,
        .generate_reports = true,
        .output_directory = "triple_opt_results",
    };
    
    // Verify all the comprehensive settings
    try testing.expectEqual(true, comprehensive_config.enabled);
    try testing.expectEqual(triple_optimization.OptimizationLevel.aggressive, comprehensive_config.optimization_level);
    try testing.expectEqual(true, comprehensive_config.souper.enabled);
    try testing.expectEqual(true, comprehensive_config.minotaur.enabled);
    try testing.expectEqual(true, comprehensive_config.ispc.enabled);
    try testing.expectEqual(@as(u32, 512), comprehensive_config.minotaur.target_vector_width);
    try testing.expectEqual(@as(u32, 16), comprehensive_config.ispc.target_width);
    try testing.expectEqual(triple_optimization.OptimizationOrder.adaptive, comprehensive_config.optimization_order);
    try testing.expectEqual(true, comprehensive_config.enable_cross_optimization);
    try testing.expectEqual(triple_optimization.VerificationLevel.formal, comprehensive_config.verification_level);
    try testing.expectEqualStrings("triple_opt_results", comprehensive_config.output_directory);
}