const std = @import("std");
const builtin = @import("builtin");
const core = @import("core.zig");
const topology = @import("topology.zig");

// SIMD Task Processing Module for Beat.zig (Phase 5.1.1)
//
// This module implements cross-platform SIMD support with:
// - Runtime feature detection and capability registry
// - Task suitability analysis and classification
// - SIMD-aligned memory management utilities
// - Integration with existing topology awareness
// - Type-safe vectorization with Zig's @Vector

// ============================================================================
// SIMD Capability Detection and Registry
// ============================================================================

/// Supported SIMD instruction sets with progressive capabilities
pub const SIMDInstructionSet = enum(u8) {
    none = 0,           // No SIMD support (scalar fallback)
    sse = 1,            // SSE 128-bit (x86_64)
    sse2 = 2,           // SSE2 128-bit with integer support
    sse3 = 3,           // SSE3 with horizontal operations
    sse41 = 4,          // SSE4.1 with enhanced operations
    sse42 = 5,          // SSE4.2 with string processing
    avx = 6,            // AVX 256-bit floating point
    avx2 = 7,           // AVX2 256-bit integer + FMA
    avx512f = 8,        // AVX-512 Foundation 512-bit
    avx512vl = 9,       // AVX-512 Vector Length extensions
    neon = 16,          // ARM NEON 128-bit (ARM64)
    sve = 17,           // ARM SVE scalable vectors
    
    /// Get vector width in bits for this instruction set
    pub fn getVectorWidth(self: SIMDInstructionSet) u16 {
        return switch (self) {
            .none => 0,
            .sse, .sse2, .sse3, .sse41, .sse42, .neon => 128,
            .avx, .avx2 => 256,
            .avx512f, .avx512vl => 512,
            .sve => 2048, // SVE can scale up to 2048-bit
        };
    }
    
    /// Check if this instruction set supports integer operations
    pub fn supportsInteger(self: SIMDInstructionSet) bool {
        return switch (self) {
            .none, .sse => false,
            .sse2, .sse3, .sse41, .sse42, .avx2, .avx512f, .avx512vl, .neon, .sve => true,
            .avx => false, // AVX1 was float-only
        };
    }
    
    /// Check if this instruction set supports fused multiply-add
    pub fn supportsFMA(self: SIMDInstructionSet) bool {
        return switch (self) {
            .avx2, .avx512f, .avx512vl, .neon, .sve => true,
            else => false,
        };
    }
};

/// SIMD data type support matrix
pub const SIMDDataType = enum(u8) {
    f32 = 0,    // 32-bit floating point
    f64 = 1,    // 64-bit floating point
    i8 = 2,     // 8-bit signed integer
    i16 = 3,    // 16-bit signed integer
    i32 = 4,    // 32-bit signed integer
    i64 = 5,    // 64-bit signed integer
    u8 = 6,     // 8-bit unsigned integer
    u16 = 7,    // 16-bit unsigned integer
    u32 = 8,    // 32-bit unsigned integer
    u64 = 9,    // 64-bit unsigned integer
    
    /// Get size in bytes for this data type
    pub fn getSize(self: SIMDDataType) u8 {
        return switch (self) {
            .i8, .u8 => 1,
            .i16, .u16 => 2,
            .f32, .i32, .u32 => 4,
            .f64, .i64, .u64 => 8,
        };
    }
    
    /// Check if this is a floating point type
    pub fn isFloat(self: SIMDDataType) bool {
        return switch (self) {
            .f32, .f64 => true,
            else => false,
        };
    }
};

/// Comprehensive SIMD capability information for a CPU/worker
pub const SIMDCapability = struct {
    // Instruction set support
    highest_instruction_set: SIMDInstructionSet,
    supported_instruction_sets: std.EnumSet(SIMDInstructionSet),
    
    // Vector characteristics
    max_vector_width_bits: u16,        // Maximum vector width (128, 256, 512)
    preferred_vector_width_bits: u16,  // Optimal vector width for this CPU
    max_vector_registers: u8,          // Number of vector registers
    
    // Data type support matrix
    supported_data_types: std.EnumSet(SIMDDataType),
    
    // Performance characteristics
    throughput_score: u8,              // 0-255, relative throughput capability
    latency_score: u8,                 // 0-255, lower is better latency
    power_efficiency_score: u8,        // 0-255, SIMD power efficiency
    
    // Advanced features
    supports_gather_scatter: bool,     // Supports gather/scatter operations
    supports_masked_operations: bool,  // Supports predicated/masked SIMD
    supports_horizontal_ops: bool,     // Supports horizontal add/mul operations
    supports_transcendental: bool,     // Hardware transcendental functions
    
    /// Initialize SIMD capability detection for current CPU
    pub fn detect() SIMDCapability {
        var capability = SIMDCapability{
            .highest_instruction_set = .none,
            .supported_instruction_sets = std.EnumSet(SIMDInstructionSet).init(.{}),
            .max_vector_width_bits = 0,
            .preferred_vector_width_bits = 0,
            .max_vector_registers = 0,
            .supported_data_types = std.EnumSet(SIMDDataType).init(.{}),
            .throughput_score = 0,
            .latency_score = 255,
            .power_efficiency_score = 128,
            .supports_gather_scatter = false,
            .supports_masked_operations = false,
            .supports_horizontal_ops = false,
            .supports_transcendental = false,
        };
        
        // Detect platform-specific SIMD capabilities
        switch (builtin.cpu.arch) {
            .x86_64 => detectX86_64Capabilities(&capability),
            .aarch64 => detectARM64Capabilities(&capability),
            else => {
                // Fallback to scalar processing for unsupported architectures
                capability.highest_instruction_set = .none;
            },
        }
        
        return capability;
    }
    
    /// Get optimal vector length for a given data type
    pub fn getOptimalVectorLength(self: *const SIMDCapability, data_type: SIMDDataType) u16 {
        const type_size = data_type.getSize();
        const optimal_width_bytes = self.preferred_vector_width_bits / 8;
        return optimal_width_bytes / type_size;
    }
    
    /// Check if a specific vector operation is supported
    pub fn supportsOperation(self: *const SIMDCapability, operation: SIMDOperation) bool {
        return switch (operation) {
            .arithmetic => true, // Basic arithmetic always supported
            .fma => self.supportsFMA(),
            .gather_scatter => self.supports_gather_scatter,
            .horizontal => self.supports_horizontal_ops,
            .masked => self.supports_masked_operations,
            .transcendental => self.supports_transcendental,
        };
    }
    
    /// Get relative performance score for SIMD vs scalar execution
    pub fn getPerformanceScore(self: *const SIMDCapability, data_type: SIMDDataType) f32 {
        if (self.highest_instruction_set == .none) return 1.0; // Scalar baseline
        
        const vector_width = self.getOptimalVectorLength(data_type);
        const base_speedup = @as(f32, @floatFromInt(vector_width));
        
        // Adjust for instruction set efficiency
        const efficiency_factor: f32 = switch (self.highest_instruction_set) {
            .none => 0.0,
            .sse, .sse2 => 0.7,
            .sse3, .sse41, .sse42 => 0.8,
            .avx => 0.85,
            .avx2 => 0.9,
            .avx512f, .avx512vl => 0.95,
            .neon => 0.8,
            .sve => 0.92,
        };
        
        return base_speedup * efficiency_factor;
    }
    
    /// Check if instruction set supports FMA operations
    fn supportsFMA(self: *const SIMDCapability) bool {
        return self.highest_instruction_set.supportsFMA();
    }
};

/// SIMD operation categories for capability checking
pub const SIMDOperation = enum {
    arithmetic,      // Basic add, sub, mul, div
    fma,            // Fused multiply-add
    gather_scatter, // Indirect memory access
    horizontal,     // Horizontal operations (sum across vector)
    masked,         // Predicated/masked operations
    transcendental, // Sin, cos, exp, log, etc.
};

// ============================================================================
// Platform-Specific SIMD Detection
// ============================================================================

/// Detect x86_64 SIMD capabilities using CPUID and feature flags
fn detectX86_64Capabilities(capability: *SIMDCapability) void {
    const features = builtin.cpu.features;
    
    // Add all supported data types for x86_64
    capability.supported_data_types.setPresent(.f32, true);
    capability.supported_data_types.setPresent(.f64, true);
    capability.supported_data_types.setPresent(.i8, true);
    capability.supported_data_types.setPresent(.i16, true);
    capability.supported_data_types.setPresent(.i32, true);
    capability.supported_data_types.setPresent(.i64, true);
    capability.supported_data_types.setPresent(.u8, true);
    capability.supported_data_types.setPresent(.u16, true);
    capability.supported_data_types.setPresent(.u32, true);
    capability.supported_data_types.setPresent(.u64, true);
    
    // Detect instruction sets in order of capability
    if (std.Target.x86.featureSetHas(features, .sse)) {
        capability.supported_instruction_sets.setPresent(.sse, true);
        capability.highest_instruction_set = .sse;
        capability.max_vector_width_bits = 128;
        capability.preferred_vector_width_bits = 128;
        capability.max_vector_registers = 16;
        capability.throughput_score = 100;
        capability.latency_score = 200;
    }
    
    if (std.Target.x86.featureSetHas(features, .sse2)) {
        capability.supported_instruction_sets.setPresent(.sse2, true);
        capability.highest_instruction_set = .sse2;
        capability.throughput_score = 120;
        capability.latency_score = 180;
        capability.supports_horizontal_ops = true;
    }
    
    if (std.Target.x86.featureSetHas(features, .sse3)) {
        capability.supported_instruction_sets.setPresent(.sse3, true);
        capability.highest_instruction_set = .sse3;
        capability.throughput_score = 140;
        capability.latency_score = 160;
    }
    
    if (std.Target.x86.featureSetHas(features, .sse4_1)) {
        capability.supported_instruction_sets.setPresent(.sse41, true);
        capability.highest_instruction_set = .sse41;
        capability.throughput_score = 160;
        capability.latency_score = 140;
    }
    
    if (std.Target.x86.featureSetHas(features, .sse4_2)) {
        capability.supported_instruction_sets.setPresent(.sse42, true);
        capability.highest_instruction_set = .sse42;
        capability.throughput_score = 180;
        capability.latency_score = 120;
    }
    
    if (std.Target.x86.featureSetHas(features, .avx)) {
        capability.supported_instruction_sets.setPresent(.avx, true);
        capability.highest_instruction_set = .avx;
        capability.max_vector_width_bits = 256;
        capability.preferred_vector_width_bits = 256;
        capability.max_vector_registers = 16;
        capability.throughput_score = 200;
        capability.latency_score = 100;
        capability.power_efficiency_score = 160;
    }
    
    if (std.Target.x86.featureSetHas(features, .avx2)) {
        capability.supported_instruction_sets.setPresent(.avx2, true);
        capability.highest_instruction_set = .avx2;
        capability.throughput_score = 220;
        capability.latency_score = 80;
        capability.power_efficiency_score = 180;
        capability.supports_gather_scatter = true;
    }
    
    if (std.Target.x86.featureSetHas(features, .avx512f)) {
        capability.supported_instruction_sets.setPresent(.avx512f, true);
        capability.highest_instruction_set = .avx512f;
        capability.max_vector_width_bits = 512;
        capability.preferred_vector_width_bits = 512;
        capability.max_vector_registers = 32;
        capability.throughput_score = 255;
        capability.latency_score = 60;
        capability.power_efficiency_score = 140; // AVX-512 uses more power
        capability.supports_gather_scatter = true;
        capability.supports_masked_operations = true;
        capability.supports_transcendental = true;
    }
    
    if (std.Target.x86.featureSetHas(features, .avx512vl)) {
        capability.supported_instruction_sets.setPresent(.avx512vl, true);
        capability.highest_instruction_set = .avx512vl;
        capability.throughput_score = 255;
        capability.latency_score = 50;
        capability.power_efficiency_score = 160; // VL improves efficiency
    }
}

/// Detect ARM64 SIMD capabilities using feature detection
fn detectARM64Capabilities(capability: *SIMDCapability) void {
    const features = builtin.cpu.features;
    
    // Add all supported data types for ARM64
    capability.supported_data_types.setPresent(.f32, true);
    capability.supported_data_types.setPresent(.f64, true);
    capability.supported_data_types.setPresent(.i8, true);
    capability.supported_data_types.setPresent(.i16, true);
    capability.supported_data_types.setPresent(.i32, true);
    capability.supported_data_types.setPresent(.i64, true);
    capability.supported_data_types.setPresent(.u8, true);
    capability.supported_data_types.setPresent(.u16, true);
    capability.supported_data_types.setPresent(.u32, true);
    capability.supported_data_types.setPresent(.u64, true);
    
    // ARM64 always has NEON support
    capability.supported_instruction_sets.setPresent(.neon, true);
    capability.highest_instruction_set = .neon;
    capability.max_vector_width_bits = 128;
    capability.preferred_vector_width_bits = 128;
    capability.max_vector_registers = 32;
    capability.throughput_score = 200;
    capability.latency_score = 80;
    capability.power_efficiency_score = 220; // ARM is very power efficient
    capability.supports_horizontal_ops = true;
    
    // Check for SVE support (ARMv8.2-A and later)
    if (std.Target.aarch64.featureSetHas(features, .sve)) {
        capability.supported_instruction_sets.setPresent(.sve, true);
        capability.highest_instruction_set = .sve;
        capability.max_vector_width_bits = 2048; // SVE can scale
        capability.preferred_vector_width_bits = 256; // Conservative default
        capability.throughput_score = 240;
        capability.latency_score = 60;
        capability.power_efficiency_score = 200;
        capability.supports_gather_scatter = true;
        capability.supports_masked_operations = true;
    }
}

// ============================================================================
// SIMD Capability Registry
// ============================================================================

/// Global SIMD capability registry for worker management
pub const SIMDCapabilityRegistry = struct {
    const Self = @This();
    
    // Per-worker SIMD capabilities (indexed by worker ID)
    worker_capabilities: []SIMDCapability,
    
    // System-wide SIMD information
    system_capability: SIMDCapability,
    numa_simd_topology: ?[]SIMDCapability, // Per-NUMA node capabilities
    
    // Thread-safe access
    mutex: std.Thread.Mutex,
    allocator: std.mem.Allocator,
    
    /// Initialize SIMD capability registry for thread pool
    pub fn init(allocator: std.mem.Allocator, num_workers: usize) !Self {
        const registry = Self{
            .worker_capabilities = try allocator.alloc(SIMDCapability, num_workers),
            .system_capability = SIMDCapability.detect(),
            .numa_simd_topology = null,
            .mutex = std.Thread.Mutex{},
            .allocator = allocator,
        };
        
        // Initialize all workers with system capabilities
        // In a real system, workers might have different capabilities
        for (registry.worker_capabilities) |*capability| {
            capability.* = registry.system_capability;
        }
        
        return registry;
    }
    
    /// Clean up registry resources
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.worker_capabilities);
        if (self.numa_simd_topology) |numa_topology| {
            self.allocator.free(numa_topology);
        }
    }
    
    /// Get SIMD capabilities for a specific worker (thread-safe)
    pub fn getWorkerCapability(self: *Self, worker_id: usize) SIMDCapability {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        if (worker_id < self.worker_capabilities.len) {
            return self.worker_capabilities[worker_id];
        }
        return self.system_capability; // Fallback to system capability
    }
    
    /// Update SIMD capabilities for a worker (thread-safe)
    pub fn updateWorkerCapability(self: *Self, worker_id: usize, capability: SIMDCapability) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        if (worker_id < self.worker_capabilities.len) {
            self.worker_capabilities[worker_id] = capability;
        }
    }
    
    /// Get optimal worker for SIMD task based on requirements
    pub fn selectOptimalWorker(self: *Self, required_width: u16, data_type: SIMDDataType) ?usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var best_worker: ?usize = null;
        var best_score: f32 = 0.0;
        
        for (self.worker_capabilities, 0..) |capability, worker_id| {
            if (capability.max_vector_width_bits >= required_width) {
                const score = capability.getPerformanceScore(data_type);
                if (score > best_score) {
                    best_score = score;
                    best_worker = worker_id;
                }
            }
        }
        
        return best_worker;
    }
    
    /// Get system-wide SIMD statistics
    pub fn getSystemStats(self: *Self) SIMDSystemStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var stats = SIMDSystemStats{
            .total_workers = self.worker_capabilities.len,
            .simd_capable_workers = 0,
            .max_system_width = self.system_capability.max_vector_width_bits,
            .average_throughput_score = 0,
            .instruction_set_distribution = std.EnumMap(SIMDInstructionSet, u32).init(.{}),
        };
        
        var total_throughput: u32 = 0;
        
        for (self.worker_capabilities) |capability| {
            if (capability.highest_instruction_set != .none) {
                stats.simd_capable_workers += 1;
                total_throughput += capability.throughput_score;
                
                const current_count = stats.instruction_set_distribution.get(capability.highest_instruction_set) orelse 0;
                stats.instruction_set_distribution.put(capability.highest_instruction_set, current_count + 1);
            }
        }
        
        if (stats.simd_capable_workers > 0) {
            stats.average_throughput_score = @as(u8, @intCast(total_throughput / stats.simd_capable_workers));
        }
        
        return stats;
    }
};

/// System-wide SIMD statistics
pub const SIMDSystemStats = struct {
    total_workers: usize,
    simd_capable_workers: usize,
    max_system_width: u16,
    average_throughput_score: u8,
    instruction_set_distribution: std.EnumMap(SIMDInstructionSet, u32),
};

// ============================================================================
// SIMD Memory Alignment Utilities
// ============================================================================

/// SIMD memory alignment requirements
pub const SIMDAlignment = enum(u8) {
    sse = 16,      // 128-bit alignment
    avx = 32,      // 256-bit alignment  
    avx512 = 64,   // 512-bit alignment
    
    /// Get alignment for instruction set
    pub fn forInstructionSet(instruction_set: SIMDInstructionSet) SIMDAlignment {
        return switch (instruction_set) {
            .none, .sse, .sse2, .sse3, .sse41, .sse42, .neon => .sse,
            .avx, .avx2, .sve => .avx,
            .avx512f, .avx512vl => .avx512,
        };
    }
    
    /// Get alignment value in bytes
    pub fn toBytes(self: SIMDAlignment) u8 {
        return @intFromEnum(self);
    }
};

/// SIMD-aligned memory allocator wrapper
pub fn SIMDAllocator(comptime alignment: SIMDAlignment) type {
    return struct {
        const Self = @This();
        const ALIGNMENT = alignment.toBytes();
        
        base_allocator: std.mem.Allocator,
        
        pub fn init(base_allocator: std.mem.Allocator) Self {
            return Self{ .base_allocator = base_allocator };
        }
        
        pub fn alloc(self: Self, comptime T: type, n: usize) ![]align(ALIGNMENT) T {
            return self.base_allocator.alignedAlloc(T, ALIGNMENT, n);
        }
        
        pub fn free(self: Self, slice: anytype) void {
            self.base_allocator.free(slice);
        }
        
        pub fn allocator(self: *Self) std.mem.Allocator {
            return .{
                .ptr = self,
                .vtable = &.{
                    .alloc = allocFn,
                    .resize = resizeFn,
                    .free = freeFn,
                },
            };
        }
        
        fn allocFn(ptr: *anyopaque, len: usize, log2_ptr_align: u8, ret_addr: usize) ?[*]u8 {
            const self: *Self = @ptrCast(@alignCast(ptr));
            const actual_align = @max(@as(u8, 1) << @as(u3, @intCast(log2_ptr_align)), ALIGNMENT);
            return self.base_allocator.rawAlloc(len, @intCast(std.math.log2(actual_align)), ret_addr);
        }
        
        fn resizeFn(ptr: *anyopaque, buf: []u8, log2_buf_align: u8, new_len: usize, ret_addr: usize) bool {
            _ = ptr;
            _ = buf;
            _ = log2_buf_align;
            _ = new_len;
            _ = ret_addr;
            return false; // Resize not supported for simplicity
        }
        
        fn freeFn(ptr: *anyopaque, buf: []u8, log2_buf_align: u8, ret_addr: usize) void {
            const self: *Self = @ptrCast(@alignCast(ptr));
            const actual_align = @max(@as(u8, 1) << @as(u3, @intCast(log2_buf_align)), ALIGNMENT);
            self.base_allocator.rawFree(buf, @intCast(std.math.log2(actual_align)), ret_addr);
        }
    };
}

/// Check if memory is properly aligned for SIMD operations
pub fn isAligned(ptr: *const anyopaque, alignment: SIMDAlignment) bool {
    const addr = @intFromPtr(ptr);
    return (addr & (alignment.toBytes() - 1)) == 0;
}

/// Align pointer to SIMD requirements
pub fn alignPointer(ptr: [*]u8, alignment: SIMDAlignment) [*]align(64) u8 {
    const align_bytes = alignment.toBytes();
    const addr = @intFromPtr(ptr);
    const aligned_addr = (addr + align_bytes - 1) & ~@as(usize, align_bytes - 1);
    return @ptrFromInt(aligned_addr);
}

// ============================================================================
// Tests
// ============================================================================

test "SIMD capability detection" {
    const capability = SIMDCapability.detect();
    
    // Should detect some SIMD capability on most modern systems
    try std.testing.expect(capability.max_vector_width_bits >= 128);
    try std.testing.expect(capability.supported_data_types.contains(.f32));
    try std.testing.expect(capability.throughput_score > 0);
    
    std.debug.print("Detected SIMD: {s}, Width: {}bits, Score: {}\n", .{
        @tagName(capability.highest_instruction_set),
        capability.max_vector_width_bits,
        capability.throughput_score,
    });
}

test "SIMD capability registry" {
    const allocator = std.testing.allocator;
    
    var registry = try SIMDCapabilityRegistry.init(allocator, 4);
    defer registry.deinit();
    
    const worker_cap = registry.getWorkerCapability(0);
    try std.testing.expect(worker_cap.max_vector_width_bits > 0);
    
    const stats = registry.getSystemStats();
    try std.testing.expect(stats.total_workers == 4);
    
    std.debug.print("SIMD Registry: {}/{} workers capable, Max width: {}bits\n", .{
        stats.simd_capable_workers,
        stats.total_workers,
        stats.max_system_width,
    });
}

test "SIMD memory alignment" {
    const allocator = std.testing.allocator;
    
    var simd_allocator = SIMDAllocator(.avx).init(allocator);
    
    const aligned_memory = try simd_allocator.alloc(f32, 16);
    defer simd_allocator.free(aligned_memory);
    
    try std.testing.expect(isAligned(aligned_memory.ptr, .avx));
    
    std.debug.print("SIMD Aligned memory at: 0x{X}\n", .{@intFromPtr(aligned_memory.ptr)});
}