const std = @import("std");
const builtin = @import("builtin");
const topology = @import("zigpulse_v3_topology_aware.zig");

// Simple benchmark to show CPU topology impact

fn memoryIntensiveWork(data: []u8, iterations: usize) u64 {
    var sum: u64 = 0;
    
    // Touch memory in patterns that stress cache
    for (0..iterations) |i| {
        // Stride through memory to cause cache misses
        const stride = 64; // Cache line size
        const idx = (i * stride) % data.len;
        sum +%= data[idx];
        data[idx] = @truncate(sum);
    }
    
    return sum;
}

fn benchmarkWithAffinity(allocator: std.mem.Allocator, cpu_list: []const u32, name: []const u8) !u64 {
    const data_size = 8 * 1024 * 1024; // 8MB - larger than L3 cache
    const iterations = 10_000_000;
    
    // Set thread affinity
    if (builtin.os.tag == .linux) {
        topology.setCurrentThreadAffinity(cpu_list) catch {
            std.debug.print("Warning: Could not set thread affinity\n", .{});
        };
    }
    
    // Allocate memory
    const data = try allocator.alloc(u8, data_size);
    defer allocator.free(data);
    
    // Initialize data
    for (data) |*b| {
        b.* = 42;
    }
    
    var timer = std.time.Timer.start() catch unreachable;
    
    const result = memoryIntensiveWork(data, iterations);
    
    const elapsed = timer.read();
    
    std.debug.print("{s}: {}ms (result: {})\n", .{ name, elapsed / 1_000_000, result });
    
    return elapsed;
}

fn benchmarkThreadMigration(allocator: std.mem.Allocator, topo: *topology.CpuTopology) !void {
    std.debug.print("\n=== Thread Migration Cost ===\n", .{});
    
    if (topo.total_cores < 2) {
        std.debug.print("Need at least 2 cores for this test\n", .{});
        return;
    }
    
    const data_size = 1024 * 1024; // 1MB
    const iterations = 1_000_000;
    
    const data = try allocator.alloc(u8, data_size);
    defer allocator.free(data);
    for (data) |*b| b.* = 42;
    
    // Warm up on CPU 0
    if (builtin.os.tag == .linux) {
        topology.setCurrentThreadAffinity(&[_]u32{0}) catch {};
    }
    _ = memoryIntensiveWork(data, iterations / 10);
    
    var timer = std.time.Timer.start() catch unreachable;
    
    // Alternate between CPUs
    const switches = 100;
    for (0..switches) |i| {
        const cpu = if (i % 2 == 0) @as(u32, 0) else @as(u32, topo.total_cores - 1);
        
        if (builtin.os.tag == .linux) {
            topology.setCurrentThreadAffinity(&[_]u32{cpu}) catch {};
        }
        
        _ = memoryIntensiveWork(data, iterations / switches);
    }
    
    const elapsed = timer.read();
    std.debug.print("With migration (100 switches): {}ms\n", .{elapsed / 1_000_000});
    
    // Baseline: no migration
    if (builtin.os.tag == .linux) {
        topology.setCurrentThreadAffinity(&[_]u32{0}) catch {};
    }
    
    timer.reset();
    _ = memoryIntensiveWork(data, iterations);
    const baseline = timer.read();
    
    std.debug.print("Without migration: {}ms\n", .{baseline / 1_000_000});
    std.debug.print("Migration overhead: {}%\n", .{
        ((elapsed - baseline) * 100) / baseline
    });
}

fn benchmarkCacheSharing(allocator: std.mem.Allocator, topo: *topology.CpuTopology) !void {
    std.debug.print("\n=== Cache Sharing Benchmark ===\n", .{});
    
    // Find SMT siblings
    var smt_pair: ?[2]u32 = null;
    var different_cores: ?[2]u32 = null;
    
    for (topo.cores) |core| {
        if (core.smt_siblings.len >= 2) {
            smt_pair = [2]u32{ core.smt_siblings[0], core.smt_siblings[1] };
            break;
        }
    }
    
    // Find cores that don't share cache
    if (topo.cores.len >= 2) {
        const core0 = &topo.cores[0];
        for (topo.cores[1..]) |*core| {
            var shares_cache = false;
            for (core0.l3_sharing) |shared| {
                if (shared == core.logical_id) {
                    shares_cache = true;
                    break;
                }
            }
            if (!shares_cache) {
                different_cores = [2]u32{ core0.logical_id, core.logical_id };
                break;
            }
        }
        
        // Fallback if all cores share L3
        if (different_cores == null and topo.cores.len >= 2) {
            different_cores = [2]u32{ 0, @intCast(topo.cores.len - 1) };
        }
    }
    
    // Run benchmarks
    if (smt_pair) |cpus| {
        std.debug.print("\nSMT siblings (shared L1/L2):\n", .{});
        _ = try benchmarkWithAffinity(allocator, &cpus, "Both threads");
        _ = try benchmarkWithAffinity(allocator, &[_]u32{cpus[0]}, "Single thread");
    } else {
        std.debug.print("No SMT siblings found\n", .{});
    }
    
    if (different_cores) |cpus| {
        std.debug.print("\nDifferent cores:\n", .{});
        _ = try benchmarkWithAffinity(allocator, &cpus, "Both threads");
        _ = try benchmarkWithAffinity(allocator, &[_]u32{cpus[0]}, "Single thread");
    }
}

pub fn main() !void {
    std.debug.print("=== CPU Topology Performance Impact ===\n", .{});
    std.debug.print("Build mode: {}\n", .{builtin.mode});
    
    const allocator = std.heap.page_allocator;
    
    // Detect topology
    var topo = try topology.detectTopology(allocator);
    defer topo.deinit();
    
    std.debug.print("\nSystem topology:\n", .{});
    std.debug.print("  Total cores: {}\n", .{topo.total_cores});
    std.debug.print("  Physical cores: {}\n", .{topo.physical_cores});
    std.debug.print("  NUMA nodes: {}\n", .{topo.numa_nodes.len});
    std.debug.print("  Sockets: {}\n", .{topo.sockets});
    
    // Benchmark 1: Single-threaded on different CPUs
    std.debug.print("\n=== Single Thread Performance ===\n", .{});
    
    const cpu0_time = try benchmarkWithAffinity(allocator, &[_]u32{0}, "CPU 0");
    
    if (topo.total_cores > 1) {
        const last_cpu = topo.total_cores - 1;
        const cpu_last_time = try benchmarkWithAffinity(allocator, &[_]u32{last_cpu}, "CPU last");
        
        const diff = if (cpu_last_time > cpu0_time) 
            ((cpu_last_time - cpu0_time) * 100) / cpu0_time
        else 
            ((cpu0_time - cpu_last_time) * 100) / cpu0_time;
        
        std.debug.print("Performance difference: {}%\n", .{diff});
    }
    
    // Benchmark 2: Thread migration cost
    try benchmarkThreadMigration(allocator, &topo);
    
    // Benchmark 3: Cache sharing effects
    try benchmarkCacheSharing(allocator, &topo);
    
    std.debug.print("\n=== Benchmark Complete ===\n", .{});
}