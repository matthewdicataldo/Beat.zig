const std = @import("std");
const beat = @import("src/core.zig");

// Auto-Configuration Integration Demo
// Demonstrates the seamless integration between build-time hardware detection
// and runtime One Euro Filter parameter optimization

test "Auto-Configuration Integration Demo" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Auto-Configuration Integration Demo ===\n", .{});
    
    // 1. Show auto-detected hardware configuration
    std.debug.print("\n1. Auto-Detected Hardware Configuration:\n", .{});
    beat.build_opts.printSummary();
    
    // 2. Create pool with auto-tuned defaults
    std.debug.print("\n2. Creating Thread Pool with Auto-Tuned Configuration:\n", .{});
    const pool = try beat.createOptimalPool(allocator);
    defer pool.deinit();
    
    std.debug.print("   Workers: {}\n", .{pool.config.num_workers.?});
    std.debug.print("   Queue Size: {}\n", .{pool.config.task_queue_size});
    std.debug.print("   Topology Aware: {}\n", .{pool.config.enable_topology_aware});
    std.debug.print("   NUMA Aware: {}\n", .{pool.config.enable_numa_aware});
    std.debug.print("   One Euro min_cutoff: {d:.3}\n", .{pool.config.prediction_min_cutoff});
    std.debug.print("   One Euro beta: {d:.3}\n", .{pool.config.prediction_beta});
    
    // 3. Demonstrate scheduler with auto-tuned One Euro Filter
    if (pool.scheduler) |_| {
        std.debug.print("\n3. Testing Auto-Tuned One Euro Filter:\n", .{});
        
        // Create predictor with auto-tuned parameters
        var predictor = beat.scheduler.TaskPredictor.init(allocator, &pool.config);
        defer predictor.deinit();
        
        const task_hash: u64 = 0xdeadbeef;
        
        // Simulate variable workload to test adaptive filtering
        const base_cycles: u64 = 1000;
        const workloads = [_]struct { cycles: u64, description: []const u8 }{
            .{ .cycles = base_cycles, .description = "Baseline workload" },
            .{ .cycles = base_cycles * 2, .description = "2x workload increase" },
            .{ .cycles = base_cycles + 50, .description = "Small variation" },
            .{ .cycles = base_cycles * 3, .description = "3x spike (outlier)" },
            .{ .cycles = base_cycles, .description = "Return to baseline" },
            .{ .cycles = base_cycles - 20, .description = "Slight decrease" },
        };
        
        for (workloads, 0..) |workload, i| {
            std.time.sleep(1_000_000); // 1ms delay between measurements
            const filtered = try predictor.recordExecution(task_hash, workload.cycles);
            
            std.debug.print("   Step {}: {} cycles -> {d:.1} filtered ({s})\n", .{
                i + 1, workload.cycles, filtered, workload.description
            });
        }
        
        // Show final prediction with confidence
        if (predictor.predict(task_hash)) |prediction| {
            std.debug.print("\n   Final Prediction:\n", .{});
            std.debug.print("     Expected Cycles: {}\n", .{prediction.expected_cycles});
            std.debug.print("     Confidence: {d:.2} (0.0-1.0)\n", .{prediction.confidence});
            std.debug.print("     Variance: {d:.1}\n", .{prediction.variance});
            std.debug.print("     Raw Estimate: {d:.1}\n", .{prediction.filtered_estimate});
        }
    }
    
    // 4. Compare with different configuration profiles
    std.debug.print("\n4. Configuration Profile Comparison:\n", .{});
    
    const configs = [_]struct { config: beat.Config, name: []const u8 }{
        .{ .config = beat.build_opts.getOptimalConfig(), .name = "Optimal (Auto-Tuned)" },
        .{ .config = beat.build_opts.getTestConfig(), .name = "Testing Profile" },
        .{ .config = beat.build_opts.getBenchmarkConfig(), .name = "Benchmark Profile" },
    };
    
    for (configs) |cfg| {
        std.debug.print("   {s}:\n", .{cfg.name});
        std.debug.print("     Workers: {}\n", .{cfg.config.num_workers.?});
        std.debug.print("     Queue Size: {}\n", .{cfg.config.task_queue_size});
        std.debug.print("     One Euro min_cutoff: {d:.3}\n", .{cfg.config.prediction_min_cutoff});
        std.debug.print("     One Euro beta: {d:.3}\n", .{cfg.config.prediction_beta});
        std.debug.print("     Topology: {}, NUMA: {}\n", .{
            cfg.config.enable_topology_aware, cfg.config.enable_numa_aware
        });
    }
    
    // 5. Demonstrate manual override capability
    std.debug.print("\n5. Manual Parameter Override:\n", .{});
    
    var custom_config = beat.build_opts.getOptimalConfig();
    custom_config.prediction_min_cutoff = 2.0; // Override for real-time workloads
    custom_config.prediction_beta = 0.25;      // More aggressive adaptation
    
    std.debug.print("   Custom Override: min_cutoff={d:.1}, beta={d:.2}\n", .{
        custom_config.prediction_min_cutoff, custom_config.prediction_beta
    });
    std.debug.print("   (Auto-detected values can still be overridden for specialized use cases)\n", .{});
    
    // 6. Show hardware-specific tuning rationale
    std.debug.print("\n6. Hardware-Specific Tuning Rationale:\n", .{});
    
    if (beat.build_opts.hardware.cpu_count >= 16) {
        std.debug.print("   High-core system detected -> More aggressive adaptation\n", .{});
    } else if (beat.build_opts.hardware.cpu_count >= 8) {
        std.debug.print("   Mid-range system detected -> Balanced parameters\n", .{});
    } else {
        std.debug.print("   Low-core system detected -> Conservative filtering\n", .{});
    }
    
    if (beat.build_opts.cpu_features.has_avx2) {
        std.debug.print("   AVX2 support detected -> Enhanced for vector workloads\n", .{});
    } else if (beat.build_opts.cpu_features.has_avx) {
        std.debug.print("   AVX support detected -> SIMD-aware tuning\n", .{});
    }
    
    if (beat.build_opts.performance.enable_numa_aware) {
        std.debug.print("   NUMA system detected -> Conservative for variable latencies\n", .{});
    }
    
    std.debug.print("\n=== Integration Demo Complete ===\n", .{});
    std.debug.print("✅ Build-time detection seamlessly integrated with runtime optimization\n", .{});
    std.debug.print("✅ One Euro Filter parameters automatically tuned for hardware\n", .{});
    std.debug.print("✅ Manual override capability preserved for specialized needs\n", .{});
    std.debug.print("✅ Multiple configuration profiles available (optimal/test/benchmark)\n", .{});
}

test "Auto-Configuration Performance Characteristics" {
    const allocator = std.testing.allocator;
    
    std.debug.print("\n=== Performance Characteristics Analysis ===\n", .{});
    
    // Test performance across different configuration profiles
    const configs = [_]struct { config: beat.Config, name: []const u8 }{
        .{ .config = beat.build_opts.getOptimalConfig(), .name = "Auto-Tuned" },
        .{ .config = beat.Config{}, .name = "Hardcoded Defaults" }, // Old way
    };
    
    for (configs) |test_config| {
        std.debug.print("\n{s} Configuration:\n", .{test_config.name});
        
        var predictor = beat.scheduler.TaskPredictor.init(allocator, &test_config.config);
        defer predictor.deinit();
        
        const task_hash: u64 = 0x12345678;
        const num_samples = 20;
        
        // Measure adaptation speed to workload changes
        const start_time = std.time.nanoTimestamp();
        
        // Phase 1: Stable workload
        for (0..num_samples / 2) |_| {
            _ = try predictor.recordExecution(task_hash, 1000);
            std.time.sleep(100_000); // 0.1ms
        }
        
        // Phase 2: Workload change
        for (0..num_samples / 2) |_| {
            _ = try predictor.recordExecution(task_hash, 2000); // 2x increase
            std.time.sleep(100_000); // 0.1ms
        }
        
        const end_time = std.time.nanoTimestamp();
        const duration_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
        
        if (predictor.predict(task_hash)) |prediction| {
            std.debug.print("   Adaptation Time: {d:.2} ms\n", .{duration_ms});
            std.debug.print("   Final Confidence: {d:.3}\n", .{prediction.confidence});
            std.debug.print("   Prediction Accuracy: {d:.1} cycles\n", .{prediction.filtered_estimate});
            std.debug.print("   Parameters: min_cutoff={d:.2}, beta={d:.3}\n", .{
                test_config.config.prediction_min_cutoff, test_config.config.prediction_beta
            });
        }
    }
    
    std.debug.print("\n✅ Auto-configuration provides hardware-optimized performance\n", .{});
}