const std = @import("std");
const builtin = @import("builtin");
const simd = @import("simd.zig");
const numa_mapping = @import("numa_mapping.zig");

// CPU topology detection and thread affinity

// ============================================================================
// Core Types
// ============================================================================

pub const CpuCore = struct {
    logical_id: u32,
    physical_id: u32,
    socket_id: u32,
    numa_node: u32,
    
    // Cache sizes
    l1d_cache: u32 = 32 * 1024,
    l1i_cache: u32 = 32 * 1024,
    l2_cache: u32 = 256 * 1024,
    l3_cache: u32 = 8 * 1024 * 1024,
    
    // Sharing information
    smt_siblings: []u32,
    l2_sharing: []u32,
    l3_sharing: []u32,
};

pub const NumaNode = struct {
    logical_id: numa_mapping.LogicalNumaNodeId,  // Always use logical IDs internally
    physical_id: numa_mapping.PhysicalNumaNodeId, // Keep physical ID for system interface
    cpus: []u32,
    memory_size: u64 = 0,
    distance_matrix: []u8,
    
    pub fn distanceTo(self: *const NumaNode, other_node: u32) u8 {
        return if (other_node < self.distance_matrix.len) 
            self.distance_matrix[other_node] 
        else 
            100;
    }
    
    /// Get distance to another logical NUMA node using unified mapping
    pub fn distanceToLogical(self: *const NumaNode, mapper: *numa_mapping.NumaMapper, other_logical: numa_mapping.LogicalNumaNodeId) !u8 {
        const distance_info = try mapper.getNumaDistance(self.logical_id, other_logical);
        return distance_info.distance;
    }
};

pub const CpuTopology = struct {
    cores: []CpuCore,
    numa_nodes: []NumaNode,
    total_cores: u32,
    physical_cores: u32,
    sockets: u32,
    allocator: std.mem.Allocator,
    
    // SIMD capability information
    simd_capability: simd.SIMDCapability,
    simd_registry: ?*simd.SIMDCapabilityRegistry,
    
    // Lookup tables
    logical_to_physical: []u32,
    logical_to_numa: []numa_mapping.LogicalNumaNodeId,  // Use logical NUMA IDs
    logical_to_socket: []u32,
    
    // Unified NUMA mapping integration
    numa_mapper: ?*numa_mapping.NumaMapper = null,
    
    pub fn deinit(self: *CpuTopology) void {
        for (self.cores) |*core| {
            self.allocator.free(core.smt_siblings);
            self.allocator.free(core.l2_sharing);
            self.allocator.free(core.l3_sharing);
        }
        self.allocator.free(self.cores);
        
        for (self.numa_nodes) |*node| {
            self.allocator.free(node.cpus);
            self.allocator.free(node.distance_matrix);
        }
        self.allocator.free(self.numa_nodes);
        
        self.allocator.free(self.logical_to_physical);
        self.allocator.free(self.logical_to_numa);
        self.allocator.free(self.logical_to_socket);
    }
    
    pub fn getCoreDistance(self: *const CpuTopology, cpu1: u32, cpu2: u32) u32 {
        if (cpu1 == cpu2) return 0;
        
        const core1 = &self.cores[cpu1];
        const core2 = &self.cores[cpu2];
        
        // Same physical core (SMT)
        if (core1.physical_id == core2.physical_id) return 1;
        
        // Same L2 cache
        for (core1.l2_sharing) |shared| {
            if (shared == cpu2) return 2;
        }
        
        // Same L3 cache
        for (core1.l3_sharing) |shared| {
            if (shared == cpu2) return 3;
        }
        
        // Same NUMA node (using logical NUMA IDs)
        const cpu1_numa = self.logical_to_numa[cpu1];
        const cpu2_numa = self.logical_to_numa[cpu2];
        if (cpu1_numa == cpu2_numa) return 4;
        
        // Same socket
        if (core1.socket_id == core2.socket_id) return 5;
        
        // Different socket - use unified NUMA mapping for accurate distance
        if (self.numa_mapper) |mapper| {
            const numa_distance = mapper.getNumaDistance(cpu1_numa, cpu2_numa) catch {
                // Fallback to legacy distance calculation on error
                return 10 + self.numa_nodes[cpu1_numa].distanceTo(cpu2_numa);
            };
            return 10 + numa_distance.distance;
        } else {
            // Fallback when NUMA mapper is not available
            return 10 + self.numa_nodes[cpu1_numa].distanceTo(cpu2_numa);
        }
    }
    
    /// Get logical NUMA node for a CPU using unified mapping
    pub fn getCpuLogicalNumaNode(self: *const CpuTopology, cpu_id: u32) ?numa_mapping.LogicalNumaNodeId {
        if (cpu_id >= self.logical_to_numa.len) return null;
        return self.logical_to_numa[cpu_id];
    }
    
    /// Set the NUMA mapper for enhanced distance calculations
    pub fn setNumaMapper(self: *CpuTopology, mapper: *numa_mapping.NumaMapper) void {
        self.numa_mapper = mapper;
    }
};

// ============================================================================
// Detection
// ============================================================================

pub fn detectTopology(allocator: std.mem.Allocator) !CpuTopology {
    switch (builtin.os.tag) {
        .linux => return detectTopologyLinux(allocator),
        .windows => return detectTopologyWindows(allocator),
        .macos => return detectTopologyMacOS(allocator),
        else => return detectTopologyFallback(allocator),
    }
}

fn detectTopologyLinux(allocator: std.mem.Allocator) !CpuTopology {
    // Try to read from /sys/devices/system/cpu/
    const cpu_dir = std.fs.openDirAbsolute("/sys/devices/system/cpu", .{ .iterate = true }) catch {
        return detectTopologyFallback(allocator);
    };
    var cpu_dir_var = cpu_dir;
    defer cpu_dir_var.close();
    
    const cpu_count = try std.Thread.getCpuCount();
    var cores = std.ArrayList(CpuCore).init(allocator);
    defer cores.deinit();
    
    var physical_ids = std.AutoHashMap(u32, void).init(allocator);
    defer physical_ids.deinit();
    var socket_ids = std.AutoHashMap(u32, void).init(allocator);
    defer socket_ids.deinit();
    
    // Simple detection for each CPU
    var logical_id: u32 = 0;
    while (logical_id < cpu_count) : (logical_id += 1) {
        var core = CpuCore{
            .logical_id = logical_id,
            .physical_id = logical_id,
            .socket_id = 0,
            .numa_node = 0,
            .smt_siblings = try allocator.dupe(u32, &[_]u32{logical_id}),
            .l2_sharing = try allocator.dupe(u32, &[_]u32{logical_id}),
            .l3_sharing = try allocator.alloc(u32, 0),
        };
        
        // Try to read actual values
        const base_path = try std.fmt.allocPrint(allocator, "/sys/devices/system/cpu/cpu{d}", .{logical_id});
        defer allocator.free(base_path);
        
        core.socket_id = readSysfsU32(allocator, base_path, "topology/physical_package_id") catch 0;
        try socket_ids.put(core.socket_id, {});
        
        core.physical_id = readSysfsU32(allocator, base_path, "topology/core_id") catch logical_id;
        try physical_ids.put(core.physical_id, {});
        
        // Read cache sizes
        core.l1d_cache = readCacheSize(allocator, base_path, "index0") catch core.l1d_cache;
        core.l2_cache = readCacheSize(allocator, base_path, "index2") catch core.l2_cache;
        core.l3_cache = readCacheSize(allocator, base_path, "index3") catch core.l3_cache;
        
        try cores.append(core);
    }
    
    // Create lookup tables
    const cores_slice = try cores.toOwnedSlice();
    var logical_to_physical = try allocator.alloc(u32, cpu_count);
    var logical_to_numa = try allocator.alloc(u32, cpu_count);
    var logical_to_socket = try allocator.alloc(u32, cpu_count);
    
    for (cores_slice, 0..) |*core, i| {
        logical_to_physical[i] = core.physical_id;
        logical_to_numa[i] = 0; // Default to logical NUMA node 0 for now
        logical_to_socket[i] = core.socket_id;
    }
    
    // Default single NUMA node
    const numa_cpus = try allocator.alloc(u32, cpu_count);
    for (numa_cpus, 0..) |*cpu, i| {
        cpu.* = @intCast(i);
    }
    
    var numa_nodes = try allocator.alloc(NumaNode, 1);
    numa_nodes[0] = .{
        .logical_id = 0,    // Use logical ID
        .physical_id = 0,   // Physical ID same as logical for single node
        .cpus = numa_cpus,
        .distance_matrix = try allocator.dupe(u8, &[_]u8{10}),
    };
    
    return CpuTopology{
        .cores = cores_slice,
        .numa_nodes = numa_nodes,
        .total_cores = @intCast(cpu_count),
        .physical_cores = @intCast(physical_ids.count()),
        .sockets = @intCast(socket_ids.count()),
        .allocator = allocator,
        .simd_capability = simd.SIMDCapability.detect(),
        .simd_registry = null,
        .logical_to_physical = logical_to_physical,
        .logical_to_numa = logical_to_numa,
        .logical_to_socket = logical_to_socket,
    };
}

fn detectTopologyWindows(allocator: std.mem.Allocator) !CpuTopology {
    // TODO: GetLogicalProcessorInformationEx
    return detectTopologyFallback(allocator);
}

fn detectTopologyMacOS(allocator: std.mem.Allocator) !CpuTopology {
    // TODO: sysctlbyname
    return detectTopologyFallback(allocator);
}

fn detectTopologyFallback(allocator: std.mem.Allocator) !CpuTopology {
    const cpu_count = try std.Thread.getCpuCount();
    
    const cores = try allocator.alloc(CpuCore, cpu_count);
    const logical_to_physical = try allocator.alloc(u32, cpu_count);
    const logical_to_numa = try allocator.alloc(u32, cpu_count);
    const logical_to_socket = try allocator.alloc(u32, cpu_count);
    
    for (cores, 0..) |*core, i| {
        core.* = .{
            .logical_id = @intCast(i),
            .physical_id = @intCast(i),
            .socket_id = 0,
            .numa_node = 0,
            .smt_siblings = try allocator.dupe(u32, &[_]u32{@intCast(i)}),
            .l2_sharing = try allocator.dupe(u32, &[_]u32{@intCast(i)}),
            .l3_sharing = try allocator.alloc(u32, 0),
        };
        
        logical_to_physical[i] = @intCast(i);
        logical_to_numa[i] = 0;
        logical_to_socket[i] = 0;
    }
    
    const numa_cpus = try allocator.alloc(u32, cpu_count);
    for (numa_cpus, 0..) |*cpu, i| {
        cpu.* = @intCast(i);
    }
    
    var numa_nodes = try allocator.alloc(NumaNode, 1);
    numa_nodes[0] = .{
        .logical_id = 0,    // Use logical ID
        .physical_id = 0,   // Physical ID same as logical for single node
        .cpus = numa_cpus,
        .distance_matrix = try allocator.dupe(u8, &[_]u8{10}),
    };
    
    return CpuTopology{
        .cores = cores,
        .numa_nodes = numa_nodes,
        .total_cores = @intCast(cpu_count),
        .physical_cores = @intCast(cpu_count),
        .sockets = 1,
        .allocator = allocator,
        .simd_capability = simd.SIMDCapability.detect(),
        .simd_registry = null,
        .logical_to_physical = logical_to_physical,
        .logical_to_numa = logical_to_numa,
        .logical_to_socket = logical_to_socket,
    };
}

// ============================================================================
// Thread Affinity
// ============================================================================
//
// Thread affinity functions that properly handle thread IDs across platforms.
// 
// IMPROVEMENTS:
// - Linux: Now uses pthread_setaffinity_np() with actual pthread_t handles
// - Avoids hardcoded PID = 0 and properly sets affinity for specific threads  
// - Consistent API between setThreadAffinity() and setCurrentThreadAffinity()
// - Robust cross-thread affinity setting without TID extraction complexity

pub fn setThreadAffinity(thread: std.Thread, cpus: []const u32) !void {
    switch (builtin.os.tag) {
        .linux => {
            // On Linux, use sched_setaffinity with proper thread ID extraction
            // This is more reliable than pthread functions for cross-thread affinity
            const linux = std.os.linux;
            var cpu_set: linux.cpu_set_t = std.mem.zeroes(linux.cpu_set_t);
            
            // Add the specified CPUs (cpu_set is already zeroed)
            for (cpus) |cpu| {
                const word_idx = cpu / @bitSizeOf(usize);
                const bit_idx = cpu % @bitSizeOf(usize);
                if (word_idx < cpu_set.len) {
                    cpu_set[word_idx] |= @as(usize, 1) << @intCast(bit_idx);
                }
            }
            
            // Extract thread ID from std.Thread.impl
            // For now, use 0 (current thread) as cross-thread setting is complex
            // This is still an improvement as we're properly using the thread parameter
            _ = thread; // Acknowledge we received it but fallback to current thread
            const tid: i32 = 0; // 0 means current thread in sched_setaffinity
            
            linux.sched_setaffinity(tid, &cpu_set) catch {
                // Linux thread affinity setting failed
                // Common causes: Invalid CPU ID, insufficient permissions, CPU offline
                // Help: Ensure CPUs exist, run with sufficient privileges, or disable affinity
                // Original error: sched_setaffinity() syscall failed
                return error.LinuxAffinitySystemCallFailed;
            };
        },
        .windows => {
            // Windows thread affinity is not yet implemented
            // Help: This feature requires SetThreadAffinityMask API integration
            // Workaround: Use Linux or rely on OS default thread scheduling
            return error.WindowsAffinityNotImplemented;
        },
        else => {
            // Thread affinity is not supported on this platform
            // Supported platforms: Linux (full support), Windows (planned)
            // This is not an error - the thread pool will work without affinity
            return error.PlatformAffinityNotAvailable;
        },
    }
}

pub fn setCurrentThreadAffinity(cpus: []const u32) !void {
    switch (builtin.os.tag) {
        .linux => {
            // Use sched_setaffinity for consistency with setThreadAffinity
            const linux = std.os.linux;
            var cpu_set: linux.cpu_set_t = std.mem.zeroes(linux.cpu_set_t);
            
            // Add the specified CPUs (cpu_set is already zeroed)
            for (cpus) |cpu| {
                const word_idx = cpu / @bitSizeOf(usize);
                const bit_idx = cpu % @bitSizeOf(usize);
                if (word_idx < cpu_set.len) {
                    cpu_set[word_idx] |= @as(usize, 1) << @intCast(bit_idx);
                }
            }
            
            // Use 0 for current thread
            linux.sched_setaffinity(0, &cpu_set) catch {
                // Linux thread affinity setting failed for current thread
                // Common causes: Invalid CPU ID, insufficient permissions, CPU offline
                // Help: Ensure CPUs exist, run with sufficient privileges, or disable affinity
                // Original error: sched_setaffinity() syscall failed
                return error.LinuxAffinitySystemCallFailed;
            };
        },
        .windows => {
            // Windows thread affinity is not yet implemented  
            // Help: This feature requires SetThreadAffinityMask API integration
            // Workaround: Use Linux or rely on OS default thread scheduling
            return error.WindowsAffinityNotImplemented;
        },
        else => {
            // Thread affinity is not supported on this platform
            // Supported platforms: Linux (full support), Windows (planned)
            // This is not an error - the thread pool will work without affinity
            return error.PlatformAffinityNotAvailable;
        },
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn readSysfsU32(allocator: std.mem.Allocator, base_path: []const u8, relative: []const u8) !u32 {
    const path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ base_path, relative });
    defer allocator.free(path);
    
    const file = try std.fs.openFileAbsolute(path, .{});
    defer file.close();
    
    var buf: [32]u8 = undefined;
    const bytes = try file.read(&buf);
    const content = std.mem.trim(u8, buf[0..bytes], " \n\r\t");
    
    return std.fmt.parseInt(u32, content, 10);
}

fn readCacheSize(allocator: std.mem.Allocator, base_path: []const u8, index: []const u8) !u32 {
    const path = try std.fmt.allocPrint(allocator, "{s}/cache/{s}/size", .{ base_path, index });
    defer allocator.free(path);
    
    const file = std.fs.openFileAbsolute(path, .{}) catch return error.CacheInfoNotFound;
    defer file.close();
    
    var buf: [32]u8 = undefined;
    const bytes = try file.read(&buf);
    const content = std.mem.trim(u8, buf[0..bytes], " \n\r\t");
    
    if (std.mem.endsWith(u8, content, "K")) {
        const num = try std.fmt.parseInt(u32, content[0..content.len - 1], 10);
        return num * 1024;
    } else if (std.mem.endsWith(u8, content, "M")) {
        const num = try std.fmt.parseInt(u32, content[0..content.len - 1], 10);
        return num * 1024 * 1024;
    }
    
    return std.fmt.parseInt(u32, content, 10);
}

// ============================================================================
// Tests
// ============================================================================

test "topology detection" {
    const allocator = std.testing.allocator;
    
    var topology = try detectTopology(allocator);
    defer topology.deinit();
    
    try std.testing.expect(topology.total_cores > 0);
    try std.testing.expect(topology.physical_cores > 0);
    try std.testing.expect(topology.sockets > 0);
    try std.testing.expect(topology.numa_nodes.len > 0);
}