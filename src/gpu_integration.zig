const std = @import("std");
const core = @import("core.zig");
const topology = @import("topology.zig");
const simd = @import("simd.zig");

// GPU Integration Module for Beat.zig (Task 3.1.1)
//
// This module provides Zig integration with the SYCL C++ wrapper for GPU acceleration.
// It implements device discovery, queue management, and task classification for GPU execution.
//
// Features:
// - SYCL device enumeration and capability detection
// - GPU-aware task classification and routing
// - Memory management for host-device transfers
// - Integration with existing worker selection algorithms
// - Fallback mechanisms for GPU unavailability

// Import mock SYCL implementation for testing
const mock_sycl = @import("mock_sycl.zig");

// ============================================================================
// SYCL C API Bindings
// ============================================================================

// Mock SYCL C API for compilation when SYCL not available
// In a full implementation, this would be replaced with actual C imports
pub const c = struct {
    // Mock types and constants for compilation
    pub const beat_sycl_result_t = enum(c_int) {
        BEAT_SYCL_SUCCESS = 0,
        BEAT_SYCL_ERROR_INVALID_PARAMETER = 1,
        BEAT_SYCL_ERROR_NO_DEVICE = 2,
        BEAT_SYCL_ERROR_OUT_OF_MEMORY = 4,
    };
    
    pub const beat_sycl_device_type_t = enum(c_int) {
        BEAT_SYCL_DEVICE_TYPE_CPU = 0,
        BEAT_SYCL_DEVICE_TYPE_GPU = 1,
        BEAT_SYCL_DEVICE_TYPE_ALL = 6,
    };
    
    pub const beat_sycl_backend_t = enum(c_int) {
        BEAT_SYCL_BACKEND_OPENCL = 0,
        BEAT_SYCL_BACKEND_LEVEL_ZERO = 1,
        BEAT_SYCL_BACKEND_CUDA = 2,
        BEAT_SYCL_BACKEND_HIP = 3,
    };
    
    pub const beat_sycl_device_info_t = extern struct {
        name: [256]u8,
        vendor: [128]u8,
        driver_version: [64]u8,
        type: beat_sycl_device_type_t,
        backend: beat_sycl_backend_t,
        max_compute_units: u32,
        max_work_group_size: u64,
        max_work_item_dimensions: u64,
        max_work_item_sizes: [3]u64,
        global_memory_size: u64,
        local_memory_size: u64,
        max_memory_allocation: u64,
        preferred_vector_width_float: u32,
        preferred_vector_width_double: u32,
        preferred_vector_width_int: u32,
        supports_double: u32,
        supports_half: u32,
        supports_unified_memory: u32,
        supports_sub_groups: u32,
        reserved: u32,
    };
    
    // Opaque handle types (mock)
    pub const beat_sycl_platform_t = ?*opaque {};
    pub const beat_sycl_device_t = ?*opaque {};
    pub const beat_sycl_context_t = ?*opaque {};
    pub const beat_sycl_queue_t = ?*opaque {};
    pub const beat_sycl_event_t = ?*opaque {};
    
    // Mock function declarations
    pub extern fn beat_sycl_is_available(available: *u32) beat_sycl_result_t;
    pub extern fn beat_sycl_get_version(version_string: [*]u8, buffer_size: usize) beat_sycl_result_t;
    pub extern fn beat_sycl_get_platform_count(count: *u32) beat_sycl_result_t;
    pub extern fn beat_sycl_get_platforms(platforms: [*]beat_sycl_platform_t, capacity: u32, count: *u32) beat_sycl_result_t;
    pub extern fn beat_sycl_get_device_count(platform: beat_sycl_platform_t, device_type: beat_sycl_device_type_t, count: *u32) beat_sycl_result_t;
    pub extern fn beat_sycl_get_devices(platform: beat_sycl_platform_t, device_type: beat_sycl_device_type_t, devices: [*]beat_sycl_device_t, capacity: u32, count: *u32) beat_sycl_result_t;
    pub extern fn beat_sycl_get_device_info(device: beat_sycl_device_t, info: *beat_sycl_device_info_t) beat_sycl_result_t;
    pub extern fn beat_sycl_create_context(devices: [*]beat_sycl_device_t, device_count: u32, context: *beat_sycl_context_t) beat_sycl_result_t;
    pub extern fn beat_sycl_create_queue(device: beat_sycl_device_t, context: beat_sycl_context_t, queue: *beat_sycl_queue_t) beat_sycl_result_t;
    pub extern fn beat_sycl_malloc_device(queue: beat_sycl_queue_t, size: usize, alignment: usize, ptr: *?*anyopaque) beat_sycl_result_t;
    pub extern fn beat_sycl_free(ptr: *anyopaque, queue: beat_sycl_queue_t) beat_sycl_result_t;
    pub extern fn beat_sycl_destroy_queue(queue: beat_sycl_queue_t) beat_sycl_result_t;
    pub extern fn beat_sycl_destroy_context(context: beat_sycl_context_t) beat_sycl_result_t;
    
    // Constants
    pub const BEAT_SYCL_SUCCESS = beat_sycl_result_t.BEAT_SYCL_SUCCESS;
    pub const BEAT_SYCL_ERROR_NO_DEVICE = beat_sycl_result_t.BEAT_SYCL_ERROR_NO_DEVICE;
    pub const BEAT_SYCL_ERROR_OUT_OF_MEMORY = beat_sycl_result_t.BEAT_SYCL_ERROR_OUT_OF_MEMORY;
    pub const BEAT_SYCL_DEVICE_TYPE_GPU = beat_sycl_device_type_t.BEAT_SYCL_DEVICE_TYPE_GPU;
    pub const BEAT_SYCL_DEVICE_TYPE_CPU = beat_sycl_device_type_t.BEAT_SYCL_DEVICE_TYPE_CPU;
    pub const BEAT_SYCL_DEVICE_TYPE_ALL = beat_sycl_device_type_t.BEAT_SYCL_DEVICE_TYPE_ALL;
    pub const BEAT_SYCL_BACKEND_OPENCL = beat_sycl_backend_t.BEAT_SYCL_BACKEND_OPENCL;
    pub const BEAT_SYCL_BACKEND_LEVEL_ZERO = beat_sycl_backend_t.BEAT_SYCL_BACKEND_LEVEL_ZERO;
    pub const BEAT_SYCL_BACKEND_CUDA = beat_sycl_backend_t.BEAT_SYCL_BACKEND_CUDA;
    pub const BEAT_SYCL_BACKEND_HIP = beat_sycl_backend_t.BEAT_SYCL_BACKEND_HIP;
};

// Re-export SYCL types and constants for convenience
pub const SyclResult = c.beat_sycl_result_t;
pub const SyclDeviceType = c.beat_sycl_device_type_t;
pub const SyclBackend = c.beat_sycl_backend_t;
pub const SyclDeviceInfo = c.beat_sycl_device_info_t;

// Opaque handle types
pub const SyclPlatform = c.beat_sycl_platform_t;
pub const SyclDevice = c.beat_sycl_device_t;
pub const SyclContext = c.beat_sycl_context_t;
pub const SyclQueue = c.beat_sycl_queue_t;
pub const SyclEvent = c.beat_sycl_event_t;

// Result constants
pub const SYCL_SUCCESS = c.BEAT_SYCL_SUCCESS;
pub const SYCL_ERROR_NO_DEVICE = c.BEAT_SYCL_ERROR_NO_DEVICE;
pub const SYCL_ERROR_OUT_OF_MEMORY = c.BEAT_SYCL_ERROR_OUT_OF_MEMORY;

// Device type constants
pub const DEVICE_TYPE_GPU = c.BEAT_SYCL_DEVICE_TYPE_GPU;
pub const DEVICE_TYPE_CPU = c.BEAT_SYCL_DEVICE_TYPE_CPU;
pub const DEVICE_TYPE_ALL = c.BEAT_SYCL_DEVICE_TYPE_ALL;

// ============================================================================
// GPU Device Management
// ============================================================================

/// Comprehensive GPU device information for Beat.zig scheduling
pub const GPUDeviceInfo = struct {
    // Basic device information
    name: [256]u8,
    vendor: [128]u8,
    device_type: SyclDeviceType,
    backend: SyclBackend,
    
    // Performance characteristics
    compute_units: u32,
    max_work_group_size: u64,
    global_memory_gb: f64,
    local_memory_kb: u64,
    
    // SIMD and vectorization capabilities
    preferred_vector_width_float: u32,
    preferred_vector_width_int: u32,
    supports_double: bool,
    supports_half: bool,
    supports_unified_memory: bool,
    
    // Performance scoring for task selection
    compute_score: f32,          // 0-1 relative compute capability
    memory_bandwidth_score: f32, // 0-1 relative memory bandwidth
    power_efficiency_score: f32, // 0-1 power efficiency rating
    
    // Integration with Beat.zig topology
    numa_node: ?u32,            // NUMA node affinity if applicable
    worker_affinity: ?u32,      // Preferred worker for this GPU
    
    pub fn init(sycl_info: SyclDeviceInfo) GPUDeviceInfo {
        var info = GPUDeviceInfo{
            .name = std.mem.zeroes([256]u8),
            .vendor = std.mem.zeroes([128]u8),
            .device_type = sycl_info.type,
            .backend = sycl_info.backend,
            .compute_units = sycl_info.max_compute_units,
            .max_work_group_size = sycl_info.max_work_group_size,
            .global_memory_gb = @as(f64, @floatFromInt(sycl_info.global_memory_size)) / (1024.0 * 1024.0 * 1024.0),
            .local_memory_kb = sycl_info.local_memory_size / 1024,
            .preferred_vector_width_float = sycl_info.preferred_vector_width_float,
            .preferred_vector_width_int = sycl_info.preferred_vector_width_int,
            .supports_double = sycl_info.supports_double != 0,
            .supports_half = sycl_info.supports_half != 0,
            .supports_unified_memory = sycl_info.supports_unified_memory != 0,
            .compute_score = 0.0,
            .memory_bandwidth_score = 0.0,
            .power_efficiency_score = 0.0,
            .numa_node = null,
            .worker_affinity = null,
        };
        
        // Copy strings safely
        @memcpy(info.name[0..std.mem.len(@as([*:0]const u8, @ptrCast(&sycl_info.name)))], 
                @as([*:0]const u8, @ptrCast(&sycl_info.name)));
        @memcpy(info.vendor[0..std.mem.len(@as([*:0]const u8, @ptrCast(&sycl_info.vendor)))], 
                @as([*:0]const u8, @ptrCast(&sycl_info.vendor)));
        
        // Calculate performance scores
        info.compute_score = calculateComputeScore(&info);
        info.memory_bandwidth_score = calculateMemoryScore(&info);
        info.power_efficiency_score = calculatePowerScore(&info);
        
        return info;
    }
    
    /// Get overall GPU performance score for task routing decisions
    pub fn getOverallScore(self: *const GPUDeviceInfo) f32 {
        return (self.compute_score * 0.4 + 
                self.memory_bandwidth_score * 0.3 + 
                self.power_efficiency_score * 0.3);
    }
    
    /// Check if GPU is suitable for a specific task type
    pub fn isSuitableForTask(self: *const GPUDeviceInfo, task_type: GPUTaskType) bool {
        return switch (task_type) {
            .compute_intensive => self.compute_score > 0.6,
            .memory_intensive => self.memory_bandwidth_score > 0.5 and self.global_memory_gb > 2.0,
            .vectorizable => self.preferred_vector_width_float >= 4,
            .machine_learning => self.supports_half and self.compute_units >= 16,
            .general_purpose => self.getOverallScore() > 0.4,
        };
    }
};

/// GPU task classification for optimal device selection
pub const GPUTaskType = enum {
    compute_intensive,    // Heavy arithmetic operations
    memory_intensive,     // Large data processing
    vectorizable,        // SIMD-friendly operations
    machine_learning,    // ML/AI workloads
    general_purpose,     // Mixed workloads
};

// ============================================================================
// GPU Device Registry
// ============================================================================

/// Registry for managing available GPU devices in Beat.zig
pub const GPUDeviceRegistry = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    devices: std.ArrayList(GPUDevice),
    mutex: std.Thread.Mutex,
    
    // System configuration
    numa_topology: ?*anyopaque, // Placeholder for NUMA topology
    total_gpu_memory_gb: f64,
    best_device_index: ?usize,
    
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .devices = std.ArrayList(GPUDevice).init(allocator),
            .mutex = std.Thread.Mutex{},
            .numa_topology = null,
            .total_gpu_memory_gb = 0.0,
            .best_device_index = null,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        for (self.devices.items) |*device| {
            device.deinit();
        }
        self.devices.deinit();
    }
    
    /// Discover and initialize all available GPU devices
    pub fn discoverDevices(self: *Self) !void {
        // Check if SYCL is available
        var available: u32 = 0;
        if (c.beat_sycl_is_available(&available) != SYCL_SUCCESS or available == 0) {
            std.log.info("SYCL runtime not available, GPU integration disabled", .{});
            return;
        }
        
        // Get platform count
        var platform_count: u32 = 0;
        if (c.beat_sycl_get_platform_count(&platform_count) != SYCL_SUCCESS) {
            return error.PlatformDiscoveryFailed;
        }
        
        if (platform_count == 0) {
            std.log.info("No SYCL platforms found", .{});
            return;
        }
        
        // Allocate platform handles
        const platforms = try self.allocator.alloc(SyclPlatform, platform_count);
        defer self.allocator.free(platforms);
        
        var actual_platform_count: u32 = 0;
        if (c.beat_sycl_get_platforms(platforms.ptr, platform_count, &actual_platform_count) != SYCL_SUCCESS) {
            return error.PlatformRetrievalFailed;
        }
        
        // Discover devices on each platform
        for (platforms[0..actual_platform_count]) |platform| {
            try self.discoverDevicesOnPlatform(platform);
        }
        
        // Calculate system-wide GPU statistics
        self.calculateSystemStats();
        
        std.log.info("GPU device discovery completed: {} devices found", .{self.devices.items.len});
    }
    
    /// Get optimal GPU device for a specific task type
    pub fn selectOptimalDevice(self: *Self, task_type: GPUTaskType) ?*GPUDevice {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var best_device: ?*GPUDevice = null;
        var best_score: f32 = 0.0;
        
        for (self.devices.items) |*device| {
            if (!device.info.isSuitableForTask(task_type)) continue;
            
            const device_score = calculateTaskScore(device, task_type);
            if (device_score > best_score) {
                best_score = device_score;
                best_device = device;
            }
        }
        
        return best_device;
    }
    
    /// Get device count by type
    pub fn getDeviceCount(self: *Self, device_type: SyclDeviceType) u32 {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var count: u32 = 0;
        for (self.devices.items) |*device| {
            if (device.info.device_type == device_type) {
                count += 1;
            }
        }
        return count;
    }
    
    /// Get total GPU memory across all devices
    pub fn getTotalGPUMemory(self: *Self) f64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.total_gpu_memory_gb;
    }
    
    // Private helper methods
    
    fn discoverDevicesOnPlatform(self: *Self, platform: SyclPlatform) !void {
        // Get GPU device count
        var gpu_device_count: u32 = 0;
        if (c.beat_sycl_get_device_count(platform, DEVICE_TYPE_GPU, &gpu_device_count) != SYCL_SUCCESS) {
            return; // Skip this platform
        }
        
        if (gpu_device_count == 0) return;
        
        // Get GPU devices
        const gpu_devices = try self.allocator.alloc(SyclDevice, gpu_device_count);
        defer self.allocator.free(gpu_devices);
        
        var actual_gpu_count: u32 = 0;
        if (c.beat_sycl_get_devices(platform, DEVICE_TYPE_GPU, gpu_devices.ptr, 
                                   gpu_device_count, &actual_gpu_count) != SYCL_SUCCESS) {
            return;
        }
        
        // Create GPU device wrappers
        for (gpu_devices[0..actual_gpu_count]) |sycl_device| {
            var device_info: SyclDeviceInfo = undefined;
            if (c.beat_sycl_get_device_info(sycl_device, &device_info) != SYCL_SUCCESS) {
                continue; // Skip this device
            }
            
            const gpu_device = GPUDevice.init(self.allocator, sycl_device, device_info) catch continue;
            self.devices.append(gpu_device) catch continue;
        }
    }
    
    fn calculateSystemStats(self: *Self) void {
        self.total_gpu_memory_gb = 0.0;
        var best_score: f32 = 0.0;
        
        for (self.devices.items, 0..) |*device, index| {
            self.total_gpu_memory_gb += device.info.global_memory_gb;
            
            const device_score = device.info.getOverallScore();
            if (device_score > best_score) {
                best_score = device_score;
                self.best_device_index = index;
            }
        }
    }
};

// ============================================================================
// GPU Device Wrapper
// ============================================================================

/// Individual GPU device with queue management and memory tracking
pub const GPUDevice = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    sycl_device: SyclDevice,
    sycl_context: ?SyclContext,
    sycl_queue: ?SyclQueue,
    info: GPUDeviceInfo,
    
    // Resource tracking
    allocated_memory: u64,
    peak_memory_usage: u64,
    active_kernels: u32,
    total_kernel_submissions: u64,
    
    pub fn init(allocator: std.mem.Allocator, sycl_device: SyclDevice, device_info: SyclDeviceInfo) !Self {
        var device = Self{
            .allocator = allocator,
            .sycl_device = sycl_device,
            .sycl_context = null,
            .sycl_queue = null,
            .info = GPUDeviceInfo.init(device_info),
            .allocated_memory = 0,
            .peak_memory_usage = 0,
            .active_kernels = 0,
            .total_kernel_submissions = 0,
        };
        
        // Create context and queue
        var context: SyclContext = undefined;
        if (c.beat_sycl_create_context(@ptrCast(&device.sycl_device), 1, &context) == SYCL_SUCCESS) {
            device.sycl_context = context;
            
            var queue: SyclQueue = undefined;
            if (c.beat_sycl_create_queue(device.sycl_device, context, &queue) == SYCL_SUCCESS) {
                device.sycl_queue = queue;
            }
        }
        
        return device;
    }
    
    pub fn deinit(self: *Self) void {
        if (self.sycl_queue) |queue| {
            _ = c.beat_sycl_destroy_queue(queue);
        }
        if (self.sycl_context) |context| {
            _ = c.beat_sycl_destroy_context(context);
        }
    }
    
    /// Check if device is ready for task execution
    pub fn isReady(self: *Self) bool {
        return self.sycl_queue != null;
    }
    
    /// Get current memory utilization ratio (0.0 - 1.0)
    pub fn getMemoryUtilization(self: *Self) f32 {
        const total_memory = @as(u64, @intFromFloat(self.info.global_memory_gb * 1024.0 * 1024.0 * 1024.0));
        if (total_memory == 0) return 0.0;
        return @as(f32, @floatFromInt(self.allocated_memory)) / @as(f32, @floatFromInt(total_memory));
    }
    
    /// Allocate device memory
    pub fn allocateMemory(self: *Self, size: usize, alignment: usize) !?*anyopaque {
        if (self.sycl_queue == null) return error.DeviceNotReady;
        
        var ptr: ?*anyopaque = null;
        const result = c.beat_sycl_malloc_device(self.sycl_queue.?, size, alignment, &ptr);
        
        if (result == SYCL_SUCCESS and ptr != null) {
            self.allocated_memory += size;
            if (self.allocated_memory > self.peak_memory_usage) {
                self.peak_memory_usage = self.allocated_memory;
            }
            return ptr;
        }
        
        return if (result == SYCL_ERROR_OUT_OF_MEMORY) error.OutOfMemory else error.AllocationFailed;
    }
    
    /// Free device memory
    pub fn freeMemory(self: *Self, ptr: *anyopaque, size: usize) void {
        if (self.sycl_queue == null) return;
        
        _ = c.beat_sycl_free(ptr, self.sycl_queue.?);
        if (self.allocated_memory >= size) {
            self.allocated_memory -= size;
        }
    }
    
    /// Submit a basic kernel for execution (placeholder)
    pub fn submitKernel(self: *Self, global_size: usize, local_size: usize) !?SyclEvent {
        if (self.sycl_queue == null) return error.DeviceNotReady;
        
        var event: SyclEvent = undefined;
        const result = c.beat_sycl_submit_kernel_1d(
            self.sycl_queue.?, global_size, local_size, null, null, &event
        );
        
        if (result == SYCL_SUCCESS) {
            self.active_kernels += 1;
            self.total_kernel_submissions += 1;
            return event;
        }
        
        return null;
    }
    
    /// Wait for all operations to complete
    pub fn synchronize(self: *Self) !void {
        if (self.sycl_queue == null) return;
        
        const result = c.beat_sycl_queue_wait(self.sycl_queue.?);
        if (result != SYCL_SUCCESS) {
            return error.SynchronizationFailed;
        }
        
        self.active_kernels = 0;
    }
};

// ============================================================================
// Performance Scoring Functions
// ============================================================================

fn calculateComputeScore(info: *const GPUDeviceInfo) f32 {
    // Scoring based on compute units and work group size
    const cu_score = @min(@as(f32, @floatFromInt(info.compute_units)) / 128.0, 1.0);
    const wg_score = @min(@as(f32, @floatFromInt(info.max_work_group_size)) / 1024.0, 1.0);
    
    // Device type bonus
    const type_bonus: f32 = switch (info.device_type) {
        DEVICE_TYPE_GPU => 1.0,
        DEVICE_TYPE_CPU => 0.6,
        else => 0.4,
    };
    
    return (cu_score * 0.6 + wg_score * 0.4) * type_bonus;
}

fn calculateMemoryScore(info: *const GPUDeviceInfo) f32 {
    // Scoring based on memory size and bandwidth characteristics
    const memory_score = @min(info.global_memory_gb / 16.0, 1.0); // 16GB as reference
    const local_memory_score = @min(@as(f32, @floatFromInt(info.local_memory_kb)) / 64.0, 1.0);
    
    // Unified memory bonus
    const usm_bonus: f32 = if (info.supports_unified_memory) 1.2 else 1.0;
    
    return @as(f32, @floatCast((memory_score * 0.7 + local_memory_score * 0.3) * usm_bonus));
}

fn calculatePowerScore(info: *const GPUDeviceInfo) f32 {
    // Heuristic power efficiency scoring
    const efficiency_base: f32 = switch (info.backend) {
        c.BEAT_SYCL_BACKEND_LEVEL_ZERO => 0.9, // Intel GPUs tend to be efficient
        c.BEAT_SYCL_BACKEND_CUDA => 0.7,       // NVIDIA GPUs are powerful but consume more
        c.BEAT_SYCL_BACKEND_HIP => 0.8,        // AMD GPUs balance
        c.BEAT_SYCL_BACKEND_OPENCL => 0.6,     // Generic OpenCL
    };
    
    // Memory efficiency factor
    const memory_efficiency: f32 = @floatCast(1.0 / (1.0 + info.global_memory_gb / 32.0));
    
    return efficiency_base * memory_efficiency;
}

fn calculateTaskScore(device: *GPUDevice, task_type: GPUTaskType) f32 {
    const base_score = device.info.getOverallScore();
    
    // Task-specific scoring adjustments
    const task_bonus: f32 = switch (task_type) {
        .compute_intensive => device.info.compute_score * 1.5,
        .memory_intensive => device.info.memory_bandwidth_score * 1.3,
        .vectorizable => @as(f32, @floatFromInt(device.info.preferred_vector_width_float)) / 16.0,
        .machine_learning => if (device.info.supports_half) base_score * 1.4 else base_score * 0.8,
        .general_purpose => base_score,
    };
    
    // Utilization penalty (prefer less loaded devices)
    const utilization_penalty = 1.0 - (device.getMemoryUtilization() * 0.3);
    
    return base_score * task_bonus * utilization_penalty;
}

// ============================================================================
// Integration with Beat.zig Core
// ============================================================================

/// Integration interface for GPU capabilities with Beat.zig thread pool
pub const GPUIntegration = struct {
    const Self = @This();
    
    device_registry: *GPUDeviceRegistry,
    enabled: bool,
    fallback_to_cpu: bool,
    
    pub fn init(device_registry: *GPUDeviceRegistry) Self {
        return Self{
            .device_registry = device_registry,
            .enabled = device_registry.devices.items.len > 0,
            .fallback_to_cpu = true,
        };
    }
    
    /// Check if a task should be routed to GPU
    pub fn shouldUseGPU(self: *const Self, task_characteristics: TaskCharacteristics) bool {
        if (!self.enabled) return false;
        
        // Simple heuristics for GPU routing
        const data_size_mb = task_characteristics.data_size / (1024 * 1024);
        const is_parallel = task_characteristics.parallelizable;
        const compute_intensity = task_characteristics.compute_intensity;
        
        return (data_size_mb > 1 and is_parallel) or compute_intensity > 0.7;
    }
    
    /// Get optimal device for task execution
    pub fn selectDevice(self: *Self, task_type: GPUTaskType) ?*GPUDevice {
        if (!self.enabled) return null;
        return self.device_registry.selectOptimalDevice(task_type);
    }
    
    /// Get system GPU statistics for scheduling decisions
    pub fn getGPUStats(self: *const Self) GPUSystemStats {
        return GPUSystemStats{
            .available_devices = @intCast(self.device_registry.devices.items.len),
            .total_memory_gb = self.device_registry.getTotalGPUMemory(),
            .gpu_enabled = self.enabled,
            .fallback_enabled = self.fallback_to_cpu,
        };
    }
};

/// Task characteristics for GPU routing decisions
pub const TaskCharacteristics = struct {
    data_size: usize,           // Input data size in bytes
    parallelizable: bool,       // Can be parallelized
    compute_intensity: f32,     // 0.0-1.0 compute vs memory bound
    memory_access_pattern: MemoryAccessPattern,
};

pub const MemoryAccessPattern = enum {
    sequential,
    strided,
    random,
    hierarchical,
};

/// System-wide GPU statistics
pub const GPUSystemStats = struct {
    available_devices: u32,
    total_memory_gb: f64,
    gpu_enabled: bool,
    fallback_enabled: bool,
};

// ============================================================================
// Tests
// ============================================================================

test "GPU device registry initialization" {
    const allocator = std.testing.allocator;
    
    var registry = GPUDeviceRegistry.init(allocator);
    defer registry.deinit();
    
    // Test discovery (may not find devices in test environment)
    registry.discoverDevices() catch |err| {
        if (err == error.PlatformDiscoveryFailed) {
            std.debug.print("SYCL not available in test environment\n", .{});
            return;
        }
        return err;
    };
    
    const gpu_count = registry.getDeviceCount(DEVICE_TYPE_GPU);
    std.debug.print("Found {} GPU devices\n", .{gpu_count});
}

test "GPU integration interface" {
    const allocator = std.testing.allocator;
    
    var registry = GPUDeviceRegistry.init(allocator);
    defer registry.deinit();
    
    const integration = GPUIntegration.init(&registry);
    const stats = integration.getGPUStats();
    
    try std.testing.expect(stats.available_devices == 0); // No devices in test
    try std.testing.expect(!stats.gpu_enabled);
    try std.testing.expect(stats.fallback_enabled);
}