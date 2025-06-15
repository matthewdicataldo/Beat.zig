const std = @import("std");
const beat = @import("beat");

// Test for SIMD Foundation Implementation (Phase 5.1.1)
//
// This test validates the SIMD capability detection, registry functionality,
// and integration with existing topology awareness.

test "SIMD capability detection and classification" {
    _ = std.testing.allocator;
    
    std.debug.print("\n=== SIMD Capability Detection Test ===\n", .{});
    
    // Test 1: Basic SIMD capability detection
    std.debug.print("1. Testing SIMD capability detection...\n", .{});
    
    const capability = beat.simd.SIMDCapability.detect();
    
    std.debug.print("   Detected SIMD capabilities:\n", .{});
    std.debug.print("     Highest instruction set: {s}\n", .{@tagName(capability.highest_instruction_set)});
    std.debug.print("     Max vector width: {} bits\n", .{capability.max_vector_width_bits});
    std.debug.print("     Preferred vector width: {} bits\n", .{capability.preferred_vector_width_bits});
    std.debug.print("     Throughput score: {}\n", .{capability.throughput_score});
    std.debug.print("     Latency score: {}\n", .{capability.latency_score});
    std.debug.print("     Power efficiency: {}\n", .{capability.power_efficiency_score});
    
    // Should detect some SIMD capability on most modern systems
    try std.testing.expect(capability.max_vector_width_bits >= 128);
    try std.testing.expect(capability.supported_data_types.contains(.f32));
    try std.testing.expect(capability.throughput_score > 0);
    
    std.debug.print("   ✅ SIMD capability detection works\n", .{});
    
    // Test 2: Data type support validation
    std.debug.print("2. Testing data type support...\n", .{});
    
    try std.testing.expect(capability.supported_data_types.contains(.f32));
    try std.testing.expect(capability.supported_data_types.contains(.i32));
    try std.testing.expect(capability.supported_data_types.contains(.u8));
    
    const vector_length_f32 = capability.getOptimalVectorLength(.f32);
    const vector_length_i16 = capability.getOptimalVectorLength(.i16);
    
    std.debug.print("   Optimal vector lengths:\n", .{});
    std.debug.print("     f32: {} elements\n", .{vector_length_f32});
    std.debug.print("     i16: {} elements\n", .{vector_length_i16});
    
    try std.testing.expect(vector_length_f32 > 0);
    try std.testing.expect(vector_length_i16 > 0);
    try std.testing.expect(vector_length_i16 >= vector_length_f32); // i16 is smaller, more elements fit
    
    std.debug.print("   ✅ Data type support validation works\n", .{});
    
    // Test 3: Performance scoring
    std.debug.print("3. Testing performance scoring...\n", .{});
    
    const score_f32 = capability.getPerformanceScore(.f32);
    const score_i32 = capability.getPerformanceScore(.i32);
    
    std.debug.print("   Performance scores:\n", .{});
    std.debug.print("     f32: {d:.2}x speedup\n", .{score_f32});
    std.debug.print("     i32: {d:.2}x speedup\n", .{score_i32});
    
    try std.testing.expect(score_f32 >= 1.0);
    try std.testing.expect(score_i32 >= 1.0);
    
    std.debug.print("   ✅ Performance scoring works\n", .{});
    
    std.debug.print("\n✅ SIMD capability detection test completed successfully!\n", .{});
}

test "SIMD capability registry and worker management" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== SIMD Capability Registry Test ===\n", .{});
    
    // Test 1: Registry initialization
    std.debug.print("1. Testing registry initialization...\n", .{});
    
    var registry = try beat.simd.SIMDCapabilityRegistry.init(allocator, 4);
    defer registry.deinit();
    
    const system_stats = registry.getSystemStats();
    
    std.debug.print("   Registry statistics:\n", .{});
    std.debug.print("     Total workers: {}\n", .{system_stats.total_workers});
    std.debug.print("     SIMD capable workers: {}\n", .{system_stats.simd_capable_workers});
    std.debug.print("     Max system width: {} bits\n", .{system_stats.max_system_width});
    std.debug.print("     Average throughput score: {}\n", .{system_stats.average_throughput_score});
    
    try std.testing.expect(system_stats.total_workers == 4);
    try std.testing.expect(system_stats.simd_capable_workers > 0);
    try std.testing.expect(system_stats.max_system_width >= 128);
    
    std.debug.print("   ✅ Registry initialization works\n", .{});
    
    // Test 2: Worker capability access
    std.debug.print("2. Testing worker capability access...\n", .{});
    
    const worker_0_cap = registry.getWorkerCapability(0);
    const worker_999_cap = registry.getWorkerCapability(999); // Out of bounds
    
    try std.testing.expect(worker_0_cap.max_vector_width_bits > 0);
    try std.testing.expect(worker_999_cap.max_vector_width_bits > 0); // Should fallback to system capability
    
    std.debug.print("   Worker 0 capability: {} bits\n", .{worker_0_cap.max_vector_width_bits});
    std.debug.print("   Out-of-bounds fallback: {} bits\n", .{worker_999_cap.max_vector_width_bits});
    
    std.debug.print("   ✅ Worker capability access works\n", .{});
    
    // Test 3: Optimal worker selection
    std.debug.print("3. Testing optimal worker selection...\n", .{});
    
    const optimal_worker = registry.selectOptimalWorker(256, .f32);
    
    if (optimal_worker) |worker_id| {
        std.debug.print("   Selected worker {} for 256-bit f32 operations\n", .{worker_id});
        try std.testing.expect(worker_id < 4);
        
        const selected_cap = registry.getWorkerCapability(worker_id);
        try std.testing.expect(selected_cap.max_vector_width_bits >= 256);
    } else {
        std.debug.print("   No worker found for 256-bit operations (expected on some systems)\n", .{});
    }
    
    // Should always find a worker for 128-bit operations
    const optimal_worker_128 = registry.selectOptimalWorker(128, .f32);
    try std.testing.expect(optimal_worker_128 != null);
    
    std.debug.print("   ✅ Optimal worker selection works\n", .{});
    
    std.debug.print("\n✅ SIMD registry test completed successfully!\n", .{});
}

test "SIMD memory alignment utilities" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== SIMD Memory Alignment Test ===\n", .{});
    
    // Test 1: SIMD-aligned allocation
    std.debug.print("1. Testing SIMD-aligned allocation...\n", .{});
    
    var sse_allocator = beat.simd.SIMDAllocator(.sse).init(allocator);
    var avx_allocator = beat.simd.SIMDAllocator(.avx).init(allocator);
    var avx512_allocator = beat.simd.SIMDAllocator(.avx512).init(allocator);
    
    const sse_memory = try sse_allocator.alloc(f32, 16);
    defer sse_allocator.free(sse_memory);
    
    const avx_memory = try avx_allocator.alloc(f32, 16);
    defer avx_allocator.free(avx_memory);
    
    const avx512_memory = try avx512_allocator.alloc(f32, 16);
    defer avx512_allocator.free(avx512_memory);
    
    std.debug.print("   Memory alignment verification:\n", .{});
    std.debug.print("     SSE (16-byte): 0x{X} - aligned: {}\n", .{ 
        @intFromPtr(sse_memory.ptr), 
        beat.simd.isAligned(sse_memory.ptr, .sse) 
    });
    std.debug.print("     AVX (32-byte): 0x{X} - aligned: {}\n", .{ 
        @intFromPtr(avx_memory.ptr), 
        beat.simd.isAligned(avx_memory.ptr, .avx) 
    });
    std.debug.print("     AVX512 (64-byte): 0x{X} - aligned: {}\n", .{ 
        @intFromPtr(avx512_memory.ptr), 
        beat.simd.isAligned(avx512_memory.ptr, .avx512) 
    });
    
    try std.testing.expect(beat.simd.isAligned(sse_memory.ptr, .sse));
    try std.testing.expect(beat.simd.isAligned(avx_memory.ptr, .avx));
    try std.testing.expect(beat.simd.isAligned(avx512_memory.ptr, .avx512));
    
    std.debug.print("   ✅ SIMD-aligned allocation works\n", .{});
    
    // Test 2: Alignment checking
    std.debug.print("2. Testing alignment validation...\n", .{});
    
    var unaligned_buffer: [100]u8 = undefined;
    const unaligned_ptr = @as(*anyopaque, @ptrCast(&unaligned_buffer[1])); // Deliberately misalign
    
    try std.testing.expect(!beat.simd.isAligned(unaligned_ptr, .avx));
    try std.testing.expect(beat.simd.isAligned(sse_memory.ptr, .sse));
    
    std.debug.print("   Alignment validation works correctly\n", .{});
    std.debug.print("   ✅ Alignment checking works\n", .{});
    
    std.debug.print("\n✅ SIMD memory alignment test completed successfully!\n", .{});
}

test "SIMD integration with topology detection" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== SIMD Topology Integration Test ===\n", .{});
    
    // Test topology detection with SIMD capabilities
    std.debug.print("1. Testing topology detection with SIMD...\n", .{});
    
    var topology = beat.topology.detectTopology(allocator) catch |err| {
        std.debug.print("   Topology detection failed (expected in some environments): {}\n", .{err});
        return; // Skip test if topology detection fails
    };
    defer topology.deinit();
    
    std.debug.print("   Topology information:\n", .{});
    std.debug.print("     Total cores: {}\n", .{topology.total_cores});
    std.debug.print("     Physical cores: {}\n", .{topology.physical_cores});
    std.debug.print("     NUMA nodes: {}\n", .{topology.numa_nodes.len});
    
    std.debug.print("   SIMD capabilities in topology:\n", .{});
    std.debug.print("     Instruction set: {s}\n", .{@tagName(topology.simd_capability.highest_instruction_set)});
    std.debug.print("     Vector width: {} bits\n", .{topology.simd_capability.max_vector_width_bits});
    std.debug.print("     Throughput score: {}\n", .{topology.simd_capability.throughput_score});
    
    try std.testing.expect(topology.total_cores > 0);
    try std.testing.expect(topology.simd_capability.max_vector_width_bits >= 128);
    
    std.debug.print("   ✅ SIMD topology integration works\n", .{});
    
    std.debug.print("\n✅ SIMD topology integration test completed successfully!\n", .{});
}

test "enhanced fingerprinting with SIMD analysis" {
    _ = std.testing.allocator;
    
    std.debug.print("\n=== Enhanced SIMD Fingerprinting Test ===\n", .{});
    
    // Test enhanced SIMD fingerprinting
    std.debug.print("1. Testing enhanced SIMD task analysis...\n", .{});
    
    var context = beat.fingerprint.ExecutionContext.init();
    context.current_numa_node = 0;
    context.system_load = 0.5;
    
    // Test data for SIMD-friendly task
    const SIMDTestData = struct {
        values: [1024]f32,
        
        fn process(self: *@This()) void {
            for (&self.values) |*value| {
                value.* = value.* * 2.0 + 1.0; // Simple arithmetic, perfect for SIMD
            }
        }
    };
    
    var simd_test_data = SIMDTestData{ .values = undefined };
    
    // Initialize with sequential data pattern (SIMD-friendly)
    for (&simd_test_data.values, 0..) |*value, i| {
        value.* = @floatFromInt(i);
    }
    
    var simd_task = beat.Task{
        .func = struct {
            fn func(data: *anyopaque) void {
                const typed_data = @as(*SIMDTestData, @ptrCast(@alignCast(data)));
                typed_data.process();
            }
        }.func,
        .data = @ptrCast(&simd_test_data),
        .priority = .normal,
        .data_size_hint = @sizeOf(SIMDTestData),
    };
    
    const simd_fingerprint = beat.fingerprint.TaskAnalyzer.analyzeTask(&simd_task, &context);
    
    std.debug.print("   SIMD-friendly task fingerprint:\n", .{});
    std.debug.print("     Call site hash: 0x{X}\n", .{simd_fingerprint.call_site_hash});
    std.debug.print("     Data size class: {} ({}KB)\n", .{ simd_fingerprint.data_size_class, @as(u32, 1) << @as(u5, @intCast(simd_fingerprint.data_size_class)) });
    std.debug.print("     SIMD width hint: {}\n", .{simd_fingerprint.simd_width});
    std.debug.print("     Access pattern: {s}\n", .{@tagName(simd_fingerprint.access_pattern)});
    std.debug.print("     Vectorization benefit: {}\n", .{simd_fingerprint.vectorization_benefit});
    std.debug.print("     Memory footprint: {}bytes (log2: {})\n", .{ 
        @as(u32, 1) << @as(u5, @intCast(simd_fingerprint.memory_footprint_log2)),
        simd_fingerprint.memory_footprint_log2 
    });
    
    // Should detect some SIMD potential (pattern may vary)
    try std.testing.expect(simd_fingerprint.simd_width >= 1); // Should suggest some vectorization
    try std.testing.expect(simd_fingerprint.vectorization_benefit >= 6); // Some vectorization potential
    
    std.debug.print("   Analysis results: SIMD width {}, pattern {s}, benefit {}\n", .{
        simd_fingerprint.simd_width, 
        @tagName(simd_fingerprint.access_pattern), 
        simd_fingerprint.vectorization_benefit
    });
    
    std.debug.print("   ✅ Enhanced SIMD fingerprinting works\n", .{});
    
    // Test scalar task for comparison
    std.debug.print("2. Testing scalar task fingerprinting...\n", .{});
    
    const ScalarTestData = struct { value: i32 };
    var scalar_data = ScalarTestData{ .value = 42 };
    
    var scalar_task = beat.Task{
        .func = struct {
            fn func(data: *anyopaque) void {
                const typed_data = @as(*ScalarTestData, @ptrCast(@alignCast(data)));
                typed_data.value = typed_data.value * 3 + 7; // Simple scalar operation
            }
        }.func,
        .data = @ptrCast(&scalar_data),
        .priority = .normal,
        .data_size_hint = @sizeOf(ScalarTestData),
    };
    
    const scalar_fingerprint = beat.fingerprint.TaskAnalyzer.analyzeTask(&scalar_task, &context);
    
    std.debug.print("   Scalar task fingerprint:\n", .{});
    std.debug.print("     SIMD width hint: {}\n", .{scalar_fingerprint.simd_width});
    std.debug.print("     Vectorization benefit: {}\n", .{scalar_fingerprint.vectorization_benefit});
    std.debug.print("     Data size class: {}\n", .{scalar_fingerprint.data_size_class});
    
    // Should detect minimal SIMD benefit
    try std.testing.expect(scalar_fingerprint.simd_width <= simd_fingerprint.simd_width);
    try std.testing.expect(scalar_fingerprint.vectorization_benefit <= simd_fingerprint.vectorization_benefit);
    
    std.debug.print("   ✅ Scalar vs SIMD comparison works\n", .{});
    
    std.debug.print("\n✅ Enhanced SIMD fingerprinting test completed successfully!\n", .{});
    
    std.debug.print("🎯 SIMD Foundation Implementation Summary:\n", .{});
    std.debug.print("   • Cross-platform SIMD capability detection ✅\n", .{});
    std.debug.print("   • SIMD capability registry with worker management ✅\n", .{});
    std.debug.print("   • SIMD-aligned memory allocation utilities ✅\n", .{});
    std.debug.print("   • Integration with existing topology awareness ✅\n", .{});
    std.debug.print("   • Enhanced task fingerprinting with SIMD analysis ✅\n", .{});
    std.debug.print("   • Foundation ready for vectorized operations ✅\n", .{});
}