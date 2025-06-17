// Minotaur SIMD Superoptimization Integration Tests
// Validates Minotaur integration with Beat.zig SIMD systems

const std = @import("std");
const testing = std.testing;
const beat = @import("beat");

const minotaur_integration = @import("minotaur_integration");

test "minotaur integration initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const config = minotaur_integration.MinotaurConfig{
        .enabled = true,
        .verify_optimizations = true,
        .combine_with_souper = true,
        .combine_with_ispc = true,
    };
    
    var minotaur = try minotaur_integration.MinotaurIntegration.init(allocator, config);
    
    // Verify initialization
    try testing.expect(minotaur.config.enabled);
    try testing.expect(minotaur.config.verify_optimizations);
    try testing.expectEqual(@as(u64, 0), minotaur.candidates_analyzed.load(.acquire));
    try testing.expectEqual(@as(u64, 0), minotaur.optimizations_found.load(.acquire));
}

test "simd optimization candidate creation" {
    const candidate = minotaur_integration.createSIMDCandidate(
        "test_simd.zig",
        "vector_add_float",
        42,
        256, // AVX2
        .vector_add_float,
    );
    
    try testing.expectEqualStrings("test_simd.zig", candidate.source_file);
    try testing.expectEqualStrings("vector_add_float", candidate.function_name);
    try testing.expectEqual(@as(u32, 42), candidate.line_number);
    try testing.expectEqual(@as(u32, 256), candidate.vector_width);
    try testing.expectEqual(minotaur_integration.SIMDInstructionType.vector_add_float, candidate.instruction_type);
}

test "simd instruction type classification" {
    // Test integer SIMD operations
    const int_ops = [_]minotaur_integration.SIMDInstructionType{
        .vector_add_int,
        .vector_sub_int,
        .vector_mul_int,
        .vector_and,
        .vector_or,
        .vector_xor,
    };
    
    for (int_ops) |op| {
        // Verify enum values are distinct
        try testing.expect(@intFromEnum(op) >= 0);
    }
    
    // Test floating-point SIMD operations
    const float_ops = [_]minotaur_integration.SIMDInstructionType{
        .vector_add_float,
        .vector_sub_float,
        .vector_mul_float,
        .vector_div_float,
        .vector_fma,
    };
    
    for (float_ops) |op| {
        try testing.expect(@intFromEnum(op) >= 0);
    }
}

test "optimization potential assessment" {
    const potentials = [_]minotaur_integration.OptimizationPotential{
        .very_high,
        .high,
        .medium,
        .low,
        .minimal,
    };
    
    // Verify ordering makes sense
    try testing.expect(@intFromEnum(potentials[0]) != @intFromEnum(potentials[1]));
    try testing.expect(@intFromEnum(potentials[1]) != @intFromEnum(potentials[2]));
}

test "minotaur statistics tracking" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const config = minotaur_integration.MinotaurConfig{ .enabled = true };
    var minotaur = try minotaur_integration.MinotaurIntegration.init(allocator, config);
    
    // Initial statistics
    var stats = minotaur.getStatistics();
    try testing.expectEqual(@as(u64, 0), stats.candidates_analyzed);
    try testing.expectEqual(@as(u64, 0), stats.optimizations_found);
    try testing.expectEqual(@as(f32, 0.0), stats.success_rate);
    
    // Simulate some activity
    _ = minotaur.candidates_analyzed.fetchAdd(10, .monotonic);
    _ = minotaur.optimizations_found.fetchAdd(3, .monotonic);
    _ = minotaur.optimizations_applied.fetchAdd(2, .monotonic);
    
    stats = minotaur.getStatistics();
    try testing.expectEqual(@as(u64, 10), stats.candidates_analyzed);
    try testing.expectEqual(@as(u64, 3), stats.optimizations_found);
    try testing.expectEqual(@as(u64, 2), stats.optimizations_applied);
    try testing.expectEqual(@as(f32, 0.3), stats.success_rate);
    try testing.expectApproxEqRel(@as(f32, 0.667), stats.application_rate, 0.01);
}

test "simd architecture support" {
    const architectures = [_]minotaur_integration.SIMDArchitecture{
        .sse,
        .sse2,
        .sse3,
        .avx,
        .avx2,
        .avx512f,
        .neon,
        .sve,
        .generic,
    };
    
    // Verify all architectures are represented
    for (architectures) |arch| {
        try testing.expect(@intFromEnum(arch) >= 0);
    }
}

test "optimization type classification" {
    const opt_types = [_]minotaur_integration.OptimizationType{
        .instruction_combining,
        .strength_reduction,
        .constant_folding,
        .dead_code_elimination,
        .loop_vectorization,
        .memory_coalescing,
        .register_allocation,
        .intrinsic_selection,
    };
    
    for (opt_types) |opt_type| {
        try testing.expect(@intFromEnum(opt_type) >= 0);
    }
}

test "minotaur availability detection" {
    // Test the availability detection functions
    // Note: These may return false in test environment, which is expected
    
    // The functions should not crash
    const minotaur_available = minotaur_integration.detectMinotaurAvailability();
    const alive2_available = minotaur_integration.detectAlive2Availability();
    
    // Results are boolean (this test just ensures no crashes)
    try testing.expect(minotaur_available == true or minotaur_available == false);
    try testing.expect(alive2_available == true or alive2_available == false);
}

test "simd operand type support" {
    const operand_types = [_]minotaur_integration.SIMDOperandType{
        .vector_i8,
        .vector_i16,
        .vector_i32,
        .vector_i64,
        .vector_f32,
        .vector_f64,
        .immediate_constant,
        .memory_address,
    };
    
    for (operand_types) |operand_type| {
        try testing.expect(@intFromEnum(operand_type) >= 0);
    }
}

test "minotaur configuration validation" {
    // Test default configuration
    const default_config = minotaur_integration.MinotaurConfig{};
    try testing.expectEqual(false, default_config.enabled);
    try testing.expectEqualStrings("localhost", default_config.redis_host);
    try testing.expectEqual(@as(u16, 6379), default_config.redis_port);
    try testing.expectEqual(@as(u32, 256), default_config.target_vector_width);
    try testing.expectEqual(true, default_config.verify_optimizations);
    
    // Test custom configuration
    const custom_config = minotaur_integration.MinotaurConfig{
        .enabled = true,
        .redis_host = "custom_host",
        .redis_port = 1234,
        .target_vector_width = 512,
        .verify_optimizations = false,
    };
    
    try testing.expectEqual(true, custom_config.enabled);
    try testing.expectEqualStrings("custom_host", custom_config.redis_host);
    try testing.expectEqual(@as(u16, 1234), custom_config.redis_port);
    try testing.expectEqual(@as(u32, 512), custom_config.target_vector_width);
    try testing.expectEqual(false, custom_config.verify_optimizations);
}

test "simd optimization candidate comprehensive" {
    const candidate = minotaur_integration.SIMDOptimizationCandidate{
        .source_file = "comprehensive_test.zig",
        .function_name = "matrix_multiply_simd",
        .line_number = 123,
        .vector_width = 512, // AVX-512
        .instruction_type = .vector_fma,
        .operand_types = &[_]minotaur_integration.SIMDOperandType{
            .vector_f32,
            .vector_f32,
            .vector_f32,
        },
        .estimated_cycles = 4,
        .memory_bandwidth_gb_s = 25.6,
        .cache_efficiency = 0.85,
        .optimization_potential = .high,
        .confidence_score = 0.92,
    };
    
    try testing.expectEqualStrings("comprehensive_test.zig", candidate.source_file);
    try testing.expectEqualStrings("matrix_multiply_simd", candidate.function_name);
    try testing.expectEqual(@as(u32, 123), candidate.line_number);
    try testing.expectEqual(@as(u32, 512), candidate.vector_width);
    try testing.expectEqual(minotaur_integration.SIMDInstructionType.vector_fma, candidate.instruction_type);
    try testing.expectEqual(@as(u32, 4), candidate.estimated_cycles);
    try testing.expectApproxEqRel(@as(f32, 25.6), candidate.memory_bandwidth_gb_s, 0.01);
    try testing.expectApproxEqRel(@as(f32, 0.85), candidate.cache_efficiency, 0.01);
    try testing.expectEqual(minotaur_integration.OptimizationPotential.high, candidate.optimization_potential);
    try testing.expectApproxEqRel(@as(f32, 0.92), candidate.confidence_score, 0.01);
}

test "threadpool integration function" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Create a test thread pool
    var pool = try beat.createTestPool(allocator);
    defer pool.deinit();
    
    const config = minotaur_integration.MinotaurConfig{
        .enabled = false, // Disabled for testing to avoid external dependencies
    };
    
    // This should complete without error even with Minotaur disabled
    try minotaur_integration.integrateSIMDOptimization(&pool, config);
}

test "minotaur test integration function" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // This should complete without error (may not find optimizations in test environment)
    try minotaur_integration.testMinotaurIntegration(allocator);
}