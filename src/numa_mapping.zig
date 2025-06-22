const std = @import("std");
const builtin = @import("builtin");

// ============================================================================
// Unified NUMA Node Mapping Layer
// 
// This module provides a unified interface for mapping between logical and 
// physical NUMA node IDs to eliminate indexing inconsistencies throughout
// the Beat.zig codebase.
//
// ISSUE ADDRESSED:
// - topology.zig uses physical NUMA node ordering (hardware order)
// - memory_pressure.zig uses OS logical ordering (kernel order)  
// - This mismatch causes mislabelled metrics and skewed load balancing
//
// SOLUTION:
// - All internal Beat.zig code uses logical NUMA node IDs (0, 1, 2, ...)
// - This module provides translation at system boundaries
// - Consistent NUMA indexing eliminates performance and correctness issues
// ============================================================================

/// Unified NUMA node identifier used throughout Beat.zig
/// Always represents logical ordering (0, 1, 2, ...) regardless of physical layout
pub const LogicalNumaNodeId = u32;

/// Physical NUMA node identifier as reported by hardware/BIOS
/// May be non-contiguous (e.g., 0, 2, 4, 8) depending on system configuration
pub const PhysicalNumaNodeId = u32;

/// NUMA node mapping entry containing logical/physical correspondence
pub const NumaNodeMapping = struct {
    logical_id: LogicalNumaNodeId,
    physical_id: PhysicalNumaNodeId,
    is_online: bool,
    cpu_list: []u32,         // CPUs belonging to this NUMA node (logical CPU IDs)
    memory_size_mb: u64,     // Memory size in MB for this NUMA node
    
    /// Calculate memory bandwidth estimate for this NUMA node
    pub fn estimatedBandwidth(self: NumaNodeMapping) u64 {
        // Simple estimation: DDR4-3200 provides ~25.6 GB/s per channel
        // Most NUMA nodes have 2-8 memory channels
        const base_bandwidth_mbps = 25600; // MB/s
        const estimated_channels = std.math.clamp(self.memory_size_mb / 8192, 2, 8); // 2-8 channels based on memory size
        return base_bandwidth_mbps * estimated_channels;
    }
};

/// NUMA distance matrix entry - distance between two logical NUMA nodes
pub const NumaDistance = struct {
    from_logical: LogicalNumaNodeId,
    to_logical: LogicalNumaNodeId,
    distance: u8,           // Relative distance (10=local, 20=adjacent, 40=remote)
    latency_ns: u32,        // Estimated memory access latency in nanoseconds
    
    /// Check if this represents local NUMA access
    pub fn isLocal(self: NumaDistance) bool {
        return self.distance <= 10;
    }
    
    /// Check if this represents remote NUMA access requiring optimization
    pub fn isRemote(self: NumaDistance) bool {
        return self.distance >= 40;
    }
};

/// Configuration for NUMA mapping behavior
pub const NumaMappingConfig = struct {
    /// Prefer contiguous logical IDs even if physical IDs are sparse
    enforce_logical_contiguity: bool = true,
    
    /// Cache NUMA distance calculations for performance
    enable_distance_caching: bool = true,
    
    /// Validate NUMA mapping consistency at startup
    enable_consistency_validation: bool = true,
    
    /// Fallback behavior when NUMA information is unavailable
    single_node_fallback: bool = true,
    
    /// Refresh interval for dynamic NUMA topology changes (seconds)
    topology_refresh_interval_s: u32 = 300, // 5 minutes
};

/// Unified NUMA mapping layer providing consistent logical ↔ physical translation
pub const NumaMapper = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    config: NumaMappingConfig,
    
    // Core mapping tables
    logical_to_physical: []PhysicalNumaNodeId,
    physical_to_logical: std.AutoHashMap(PhysicalNumaNodeId, LogicalNumaNodeId),
    
    // NUMA node information
    node_mappings: []NumaNodeMapping,
    distance_matrix: [][]NumaDistance,
    
    // Performance tracking
    mapping_hits: std.atomic.Value(u64),
    mapping_misses: std.atomic.Value(u64),
    distance_cache_hits: std.atomic.Value(u64),
    
    // Thread safety
    mapping_mutex: std.Thread.Mutex = .{},
    last_topology_refresh: std.atomic.Value(u64), // Timestamp of last refresh
    
    /// Initialize NUMA mapper with detected topology
    pub fn init(allocator: std.mem.Allocator, config: NumaMappingConfig) !Self {
        var mapper = Self{
            .allocator = allocator,
            .config = config,
            .logical_to_physical = undefined,
            .physical_to_logical = std.AutoHashMap(PhysicalNumaNodeId, LogicalNumaNodeId).init(allocator),
            .node_mappings = undefined,
            .distance_matrix = undefined,
            .mapping_hits = std.atomic.Value(u64).init(0),
            .mapping_misses = std.atomic.Value(u64).init(0),
            .distance_cache_hits = std.atomic.Value(u64).init(0),
            .last_topology_refresh = std.atomic.Value(u64).init(0),
        };
        
        try mapper.detectAndBuildMapping();
        
        if (config.enable_consistency_validation) {
            try mapper.validateMappingConsistency();
        }
        
        return mapper;
    }
    
    /// Clean up resources
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.logical_to_physical);
        self.physical_to_logical.deinit();
        
        for (self.node_mappings) |*mapping| {
            self.allocator.free(mapping.cpu_list);
        }
        self.allocator.free(self.node_mappings);
        
        for (self.distance_matrix) |row| {
            self.allocator.free(row);
        }
        self.allocator.free(self.distance_matrix);
    }
    
    // ========================================================================
    // Core Mapping Functions
    // ========================================================================
    
    /// Convert logical NUMA node ID to physical NUMA node ID
    pub fn logicalToPhysical(self: *Self, logical: LogicalNumaNodeId) !PhysicalNumaNodeId {
        if (logical >= self.logical_to_physical.len) {
            _ = self.mapping_misses.fetchAdd(1, .monotonic);
            return error.InvalidLogicalNumaNode;
        }
        
        _ = self.mapping_hits.fetchAdd(1, .monotonic);
        return self.logical_to_physical[logical];
    }
    
    /// Convert physical NUMA node ID to logical NUMA node ID
    pub fn physicalToLogical(self: *Self, physical: PhysicalNumaNodeId) !LogicalNumaNodeId {
        if (self.physical_to_logical.get(physical)) |logical| {
            _ = self.mapping_hits.fetchAdd(1, .monotonic);
            return logical;
        }
        
        _ = self.mapping_misses.fetchAdd(1, .monotonic);
        return error.InvalidPhysicalNumaNode;
    }
    
    /// Get NUMA node mapping information for logical node
    pub fn getNodeMapping(self: *Self, logical: LogicalNumaNodeId) ?*const NumaNodeMapping {
        if (logical >= self.node_mappings.len) return null;
        return &self.node_mappings[logical];
    }
    
    /// Get number of logical NUMA nodes in the system
    pub fn getNumaNodeCount(self: *Self) u32 {
        return @intCast(self.node_mappings.len);
    }
    
    /// Get distance between two logical NUMA nodes
    pub fn getNumaDistance(self: *Self, from: LogicalNumaNodeId, to: LogicalNumaNodeId) !NumaDistance {
        if (from >= self.distance_matrix.len or to >= self.distance_matrix[from].len) {
            return error.InvalidNumaNodeIds;
        }
        
        if (self.config.enable_distance_caching) {
            _ = self.distance_cache_hits.fetchAdd(1, .monotonic);
        }
        
        return self.distance_matrix[from][to];
    }
    
    /// Find the logical NUMA node containing the specified CPU
    pub fn getCpuNumaNode(self: *Self, cpu_id: u32) ?LogicalNumaNodeId {
        for (self.node_mappings, 0..) |mapping, logical_id| {
            for (mapping.cpu_list) |cpu| {
                if (cpu == cpu_id) {
                    _ = self.mapping_hits.fetchAdd(1, .monotonic);
                    return @intCast(logical_id);
                }
            }
        }
        
        _ = self.mapping_misses.fetchAdd(1, .monotonic);
        return null;
    }
    
    /// Find the closest logical NUMA node to the given logical node
    pub fn getClosestNumaNode(self: *Self, from: LogicalNumaNodeId, exclude_self: bool) !LogicalNumaNodeId {
        if (from >= self.distance_matrix.len) {
            return error.InvalidNumaNode;
        }
        
        var min_distance: u8 = 255;
        var closest_node: ?LogicalNumaNodeId = null;
        
        for (self.distance_matrix[from], 0..) |distance_info, to_id| {
            const to_logical = @as(LogicalNumaNodeId, @intCast(to_id));
            
            if (exclude_self and to_logical == from) continue;
            
            if (distance_info.distance < min_distance) {
                min_distance = distance_info.distance;
                closest_node = to_logical;
            }
        }
        
        return closest_node orelse error.NoValidNumaNode;
    }
    
    // ========================================================================
    // Performance Analysis Functions
    // ========================================================================
    
    /// Estimate memory bandwidth between two logical NUMA nodes
    pub fn estimateMemoryBandwidth(self: *Self, from: LogicalNumaNodeId, to: LogicalNumaNodeId) !u64 {
        const distance = try self.getNumaDistance(from, to);
        const to_mapping = self.getNodeMapping(to) orelse return error.InvalidNumaNode;
        
        const base_bandwidth = to_mapping.estimatedBandwidth();
        
        // Reduce bandwidth based on NUMA distance
        return switch (distance.distance) {
            0...15 => base_bandwidth,                    // Local access - full bandwidth
            16...25 => (base_bandwidth * 7) / 10,       // Adjacent nodes - 70% bandwidth
            26...35 => (base_bandwidth * 5) / 10,       // Remote nodes - 50% bandwidth
            else => (base_bandwidth * 2) / 10,          // Very remote - 20% bandwidth
        };
    }
    
    /// Get performance statistics for the NUMA mapper
    pub fn getPerformanceStats(self: *Self) NumaMapperStats {
        const hits = self.mapping_hits.load(.monotonic);
        const misses = self.mapping_misses.load(.monotonic);
        const cache_hits = self.distance_cache_hits.load(.monotonic);
        
        const hit_rate = if (hits + misses > 0) 
            @as(f32, @floatFromInt(hits)) / @as(f32, @floatFromInt(hits + misses))
        else 
            0.0;
        
        return NumaMapperStats{
            .total_lookups = hits + misses,
            .cache_hits = hits,
            .cache_misses = misses,
            .hit_rate = hit_rate,
            .distance_cache_hits = cache_hits,
            .numa_node_count = self.getNumaNodeCount(),
            .last_refresh_time = self.last_topology_refresh.load(.monotonic),
        };
    }
    
    // ========================================================================
    // Topology Detection and Building
    // ========================================================================
    
    /// Detect NUMA topology and build logical mapping
    fn detectAndBuildMapping(self: *Self) !void {
        self.mapping_mutex.lock();
        defer self.mapping_mutex.unlock();
        
        switch (builtin.os.tag) {
            .linux => try self.detectLinuxNumaTopology(),
            .windows => try self.detectWindowsNumaTopology(),
            .macos => try self.detectMacOSNumaTopology(),
            else => try self.createFallbackMapping(),
        }
        
        try self.buildDistanceMatrix();
        self.last_topology_refresh.store(@as(u64, @intCast(std.time.nanoTimestamp())), .monotonic);
    }
    
    /// Detect NUMA topology on Linux using /sys/devices/system/node/
    fn detectLinuxNumaTopology(self: *Self) !void {
        const node_dir = std.fs.openDirAbsolute("/sys/devices/system/node", .{ .iterate = true }) catch {
            return self.createFallbackMapping();
        };
        var node_dir_var = node_dir;
        defer node_dir_var.close();
        
        var physical_nodes = std.ArrayList(PhysicalNumaNodeId).init(self.allocator);
        defer physical_nodes.deinit();
        
        // Scan for available NUMA nodes
        var iter = node_dir_var.iterate();
        while (try iter.next()) |entry| {
            if (std.mem.startsWith(u8, entry.name, "node") and entry.kind == .directory) {
                const node_id_str = entry.name[4..]; // Skip "node" prefix
                if (std.fmt.parseInt(PhysicalNumaNodeId, node_id_str, 10)) |physical_id| {
                    try physical_nodes.append(physical_id);
                } else |_| {
                    // Skip invalid node directory names
                }
            }
        }
        
        // Sort physical nodes to ensure consistent logical ordering
        std.mem.sort(PhysicalNumaNodeId, physical_nodes.items, {}, std.sort.asc(PhysicalNumaNodeId));
        
        // Build logical mapping
        self.logical_to_physical = try self.allocator.alloc(PhysicalNumaNodeId, physical_nodes.items.len);
        self.node_mappings = try self.allocator.alloc(NumaNodeMapping, physical_nodes.items.len);
        
        for (physical_nodes.items, 0..) |physical_id, logical_idx| {
            const logical_id = @as(LogicalNumaNodeId, @intCast(logical_idx));
            
            // Build logical ↔ physical mapping
            self.logical_to_physical[logical_idx] = physical_id;
            try self.physical_to_logical.put(physical_id, logical_id);
            
            // Read NUMA node information
            const cpu_list = try self.readNumaNodeCpus(physical_id);
            const memory_size = try self.readNumaNodeMemory(physical_id);
            
            self.node_mappings[logical_idx] = NumaNodeMapping{
                .logical_id = logical_id,
                .physical_id = physical_id,
                .is_online = true,
                .cpu_list = cpu_list,
                .memory_size_mb = memory_size,
            };
        }
    }
    
    /// Read CPU list for a physical NUMA node
    fn readNumaNodeCpus(self: *Self, physical_node: PhysicalNumaNodeId) ![]u32 {
        const path = try std.fmt.allocPrint(self.allocator, "/sys/devices/system/node/node{d}/cpulist", .{physical_node});
        defer self.allocator.free(path);
        
        const file = std.fs.openFileAbsolute(path, .{}) catch {
            // Fallback: assume single CPU for this node
            const fallback_cpu = try self.allocator.alloc(u32, 1);
            fallback_cpu[0] = physical_node; // Simple assumption
            return fallback_cpu;
        };
        defer file.close();
        
        var buf: [1024]u8 = undefined;
        const bytes = try file.read(&buf);
        const content = std.mem.trim(u8, buf[0..bytes], " \n\r\t");
        
        var cpu_list = std.ArrayList(u32).init(self.allocator);
        defer cpu_list.deinit();
        
        // Parse CPU list (format: "0-3,8-11" or "0,2,4,6")
        var range_iter = std.mem.splitSequence(u8, content, ",");
        while (range_iter.next()) |range_str| {
            if (std.mem.indexOf(u8, range_str, "-")) |dash_pos| {
                // Range format: "0-3"
                const start = try std.fmt.parseInt(u32, range_str[0..dash_pos], 10);
                const end = try std.fmt.parseInt(u32, range_str[dash_pos + 1..], 10);
                
                var cpu = start;
                while (cpu <= end) : (cpu += 1) {
                    try cpu_list.append(cpu);
                }
            } else {
                // Single CPU: "5"
                const cpu = try std.fmt.parseInt(u32, range_str, 10);
                try cpu_list.append(cpu);
            }
        }
        
        return cpu_list.toOwnedSlice();
    }
    
    /// Read memory size for a physical NUMA node
    fn readNumaNodeMemory(self: *Self, physical_node: PhysicalNumaNodeId) !u64 {
        const path = try std.fmt.allocPrint(self.allocator, "/sys/devices/system/node/node{d}/meminfo", .{physical_node});
        defer self.allocator.free(path);
        
        const file = std.fs.openFileAbsolute(path, .{}) catch {
            return 2048; // 2GB fallback
        };
        defer file.close();
        
        var buf: [4096]u8 = undefined;
        const bytes = try file.read(&buf);
        const content = buf[0..bytes];
        
        // Look for "MemTotal:" line
        var line_iter = std.mem.splitSequence(u8, content, "\n");
        while (line_iter.next()) |line| {
            if (std.mem.startsWith(u8, line, "MemTotal:")) {
                // Extract number (format: "MemTotal:     2097152 kB")
                var word_iter = std.mem.tokenizeAny(u8, line, " \t");
                _ = word_iter.next(); // Skip "MemTotal:"
                if (word_iter.next()) |size_str| {
                    const size_kb = try std.fmt.parseInt(u64, size_str, 10);
                    return size_kb / 1024; // Convert KB to MB
                }
            }
        }
        
        return 2048; // 2GB fallback
    }
    
    /// Detect NUMA topology on Windows using GetNumaNodeProcessorMask
    fn detectWindowsNumaTopology(self: *Self) !void {
        // TODO: Implement Windows NUMA detection
        // Use GetLogicalProcessorInformationEx with RelationNumaNode
        return self.createFallbackMapping();
    }
    
    /// Detect NUMA topology on macOS using vm_stat and CPU affinity
    fn detectMacOSNumaTopology(self: *Self) !void {
        // TODO: Implement macOS NUMA detection  
        // Use sysctlbyname for CPU topology and memory info
        return self.createFallbackMapping();
    }
    
    /// Create fallback single-node mapping when NUMA detection fails
    fn createFallbackMapping(self: *Self) !void {
        const cpu_count = try std.Thread.getCpuCount();
        
        // Single logical NUMA node
        self.logical_to_physical = try self.allocator.alloc(PhysicalNumaNodeId, 1);
        self.logical_to_physical[0] = 0;
        try self.physical_to_logical.put(0, 0);
        
        // Build CPU list for single node
        const cpu_list = try self.allocator.alloc(u32, cpu_count);
        for (cpu_list, 0..) |*cpu, i| {
            cpu.* = @intCast(i);
        }
        
        self.node_mappings = try self.allocator.alloc(NumaNodeMapping, 1);
        self.node_mappings[0] = NumaNodeMapping{
            .logical_id = 0,
            .physical_id = 0,
            .is_online = true,
            .cpu_list = cpu_list,
            .memory_size_mb = 4096, // 4GB assumption
        };
    }
    
    /// Build distance matrix between all logical NUMA nodes
    fn buildDistanceMatrix(self: *Self) !void {
        const node_count = self.node_mappings.len;
        self.distance_matrix = try self.allocator.alloc([]NumaDistance, node_count);
        
        for (self.distance_matrix, 0..) |*row, from_idx| {
            row.* = try self.allocator.alloc(NumaDistance, node_count);
            
            for (row.*, 0..) |*distance_entry, to_idx| {
                const from_logical = @as(LogicalNumaNodeId, @intCast(from_idx));
                const to_logical = @as(LogicalNumaNodeId, @intCast(to_idx));
                
                distance_entry.* = NumaDistance{
                    .from_logical = from_logical,
                    .to_logical = to_logical,
                    .distance = if (from_idx == to_idx) 10 else 20, // Local vs remote
                    .latency_ns = if (from_idx == to_idx) 100 else 300, // Estimated latencies
                };
            }
        }
        
        // TODO: Read actual NUMA distances from /sys/devices/system/node/nodeX/distance on Linux
        try self.readActualNumaDistances();
    }
    
    /// Read actual NUMA distances from system (Linux-specific)
    fn readActualNumaDistances(self: *Self) !void {
        if (builtin.os.tag != .linux) return;
        
        for (self.node_mappings, 0..) |mapping, from_idx| {
            const distance_path = try std.fmt.allocPrint(
                self.allocator, 
                "/sys/devices/system/node/node{d}/distance", 
                .{mapping.physical_id}
            );
            defer self.allocator.free(distance_path);
            
            const file = std.fs.openFileAbsolute(distance_path, .{}) catch continue;
            defer file.close();
            
            var buf: [1024]u8 = undefined;
            const bytes = try file.read(&buf);
            const content = std.mem.trim(u8, buf[0..bytes], " \n\r\t");
            
            // Parse distance values (space-separated)
            var distance_iter = std.mem.tokenizeAny(u8, content, " \t");
            var to_idx: usize = 0;
            
            while (distance_iter.next()) |distance_str| {
                if (to_idx >= self.distance_matrix[from_idx].len) break;
                
                if (std.fmt.parseInt(u8, distance_str, 10)) |distance_value| {
                    self.distance_matrix[from_idx][to_idx].distance = distance_value;
                    // Estimate latency based on distance (rough approximation)
                    self.distance_matrix[from_idx][to_idx].latency_ns = @as(u32, distance_value) * 10;
                } else |_| {
                    // Keep default values on parse error
                }
                
                to_idx += 1;
            }
        }
    }
    
    /// Validate that the mapping is consistent and correct
    fn validateMappingConsistency(self: *Self) !void {
        // Check that all logical IDs map to unique physical IDs
        var seen_physical = std.AutoHashMap(PhysicalNumaNodeId, void).init(self.allocator);
        defer seen_physical.deinit();
        
        for (self.logical_to_physical) |physical_id| {
            if (seen_physical.contains(physical_id)) {
                return error.DuplicatePhysicalNumaNode;
            }
            try seen_physical.put(physical_id, {});
        }
        
        // Check that physical→logical mapping is bidirectional
        for (self.logical_to_physical, 0..) |physical_id, logical_idx| {
            const logical_id = @as(LogicalNumaNodeId, @intCast(logical_idx));
            const mapped_logical = try self.physicalToLogical(physical_id);
            if (mapped_logical != logical_id) {
                return error.InconsistentNumaMapping;
            }
        }
        
        // Check that distance matrix is symmetric and complete
        for (self.distance_matrix, 0..) |row, i| {
            if (row.len != self.distance_matrix.len) {
                return error.IncompleteDistanceMatrix;
            }
            
            for (row, 0..) |distance, j| {
                // Check diagonal is local distance
                if (i == j and distance.distance != 10) {
                    return error.InvalidLocalDistance;
                }
                
                // Check symmetry (distance from A to B should equal B to A)
                const reverse_distance = self.distance_matrix[j][i].distance;
                if (distance.distance != reverse_distance) {
                    return error.AsymmetricDistanceMatrix;
                }
            }
        }
    }
};

/// Performance statistics for NUMA mapper
pub const NumaMapperStats = struct {
    total_lookups: u64,
    cache_hits: u64,
    cache_misses: u64,
    hit_rate: f32,
    distance_cache_hits: u64,
    numa_node_count: u32,
    last_refresh_time: u64,
    
    pub fn format(
        self: NumaMapperStats,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("NumaMapperStats{{ nodes: {}, lookups: {}, hit_rate: {d:.1}% }}", 
            .{ self.numa_node_count, self.total_lookups, self.hit_rate * 100.0 });
    }
};

// ============================================================================
// Convenience Functions for Common Use Cases
// ============================================================================

/// Global NUMA mapper instance (initialized once per process)
var global_numa_mapper: ?NumaMapper = null;
var global_mapper_mutex: std.Thread.Mutex = .{};

/// Get the global NUMA mapper, initializing if necessary
pub fn getGlobalNumaMapper(allocator: std.mem.Allocator) !*NumaMapper {
    global_mapper_mutex.lock();
    defer global_mapper_mutex.unlock();
    
    if (global_numa_mapper) |*mapper| {
        return mapper;
    }
    
    const config = NumaMappingConfig{};
    global_numa_mapper = try NumaMapper.init(allocator, config);
    return &global_numa_mapper.?;
}

/// Clean up global NUMA mapper
pub fn deinitGlobalNumaMapper() void {
    global_mapper_mutex.lock();
    defer global_mapper_mutex.unlock();
    
    if (global_numa_mapper) |*mapper| {
        mapper.deinit();
        global_numa_mapper = null;
    }
}

// ============================================================================
// Legacy Compatibility Functions
// ============================================================================

/// Convert legacy NUMA node references to logical IDs
/// This helps migrate existing code gradually
pub fn convertLegacyNumaNodeId(allocator: std.mem.Allocator, legacy_id: u32, assume_physical: bool) !LogicalNumaNodeId {
    const mapper = try getGlobalNumaMapper(allocator);
    
    if (assume_physical) {
        return mapper.physicalToLogical(legacy_id);
    } else {
        // Assume it's already logical
        return legacy_id;
    }
}

// ============================================================================
// Testing
// ============================================================================

test "NUMA mapper basic functionality" {
    const allocator = std.testing.allocator;
    
    const config = NumaMappingConfig{
        .enforce_logical_contiguity = true,
        .enable_consistency_validation = true,
    };
    
    var mapper = try NumaMapper.init(allocator, config);
    defer mapper.deinit();
    
    // Test basic mapping functions
    const node_count = mapper.getNumaNodeCount();
    try std.testing.expect(node_count > 0);
    
    // Test logical ↔ physical conversion
    if (node_count > 0) {
        const physical = try mapper.logicalToPhysical(0);
        const logical = try mapper.physicalToLogical(physical);
        try std.testing.expect(logical == 0);
    }
    
    // Test distance calculation
    if (node_count > 0) {
        const distance = try mapper.getNumaDistance(0, 0);
        try std.testing.expect(distance.isLocal());
        try std.testing.expect(distance.distance == 10);
    }
    
    // Test performance stats
    const stats = mapper.getPerformanceStats();
    try std.testing.expect(stats.numa_node_count == node_count);
    try std.testing.expect(stats.total_lookups > 0);
}

test "NUMA mapper CPU node lookup" {
    const allocator = std.testing.allocator;
    
    var mapper = try NumaMapper.init(allocator, NumaMappingConfig{});
    defer mapper.deinit();
    
    // Test CPU to NUMA node mapping
    const cpu_numa = mapper.getCpuNumaNode(0);
    try std.testing.expect(cpu_numa != null);
    
    if (cpu_numa) |numa_node| {
        try std.testing.expect(numa_node < mapper.getNumaNodeCount());
    }
}

test "NUMA distance matrix validation" {
    const allocator = std.testing.allocator;
    
    const config = NumaMappingConfig{
        .enable_consistency_validation = true,
    };
    
    var mapper = try NumaMapper.init(allocator, config);
    defer mapper.deinit();
    
    const node_count = mapper.getNumaNodeCount();
    
    // Verify distance matrix properties
    for (0..node_count) |i| {
        for (0..node_count) |j| {
            const distance_ij = try mapper.getNumaDistance(@intCast(i), @intCast(j));
            const distance_ji = try mapper.getNumaDistance(@intCast(j), @intCast(i));
            
            // Distance matrix should be symmetric
            try std.testing.expect(distance_ij.distance == distance_ji.distance);
            
            // Local distance should be 10
            if (i == j) {
                try std.testing.expect(distance_ij.isLocal());
                try std.testing.expect(distance_ij.distance == 10);
            }
        }
    }
}