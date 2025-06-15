const std = @import("std");
const gpu_integration = @import("gpu_integration.zig");

// Mock SYCL Implementation for Testing and Development
//
// This provides mock implementations of SYCL functions for testing the GPU integration
// when SYCL runtime is not available. In a production environment, these would be
// replaced with actual SYCL C++ wrapper function implementations.

// ============================================================================
// Mock Implementation Functions
// ============================================================================

export fn beat_sycl_is_available(available: *u32) gpu_integration.c.beat_sycl_result_t {
    // Mock: Report SYCL as not available for testing
    available.* = 0;
    return gpu_integration.c.beat_sycl_result_t.BEAT_SYCL_SUCCESS;
}

export fn beat_sycl_get_version(version_string: [*]u8, buffer_size: usize) gpu_integration.c.beat_sycl_result_t {
    const mock_version = "Mock SYCL 2020 (Beat.zig wrapper v1.0.0)";
    const copy_len = @min(mock_version.len, buffer_size - 1);
    
    @memcpy(version_string[0..copy_len], mock_version[0..copy_len]);
    version_string[copy_len] = 0;
    
    return gpu_integration.c.beat_sycl_result_t.BEAT_SYCL_SUCCESS;
}

export fn beat_sycl_get_platform_count(count: *u32) gpu_integration.c.beat_sycl_result_t {
    // Mock: Report no platforms available
    count.* = 0;
    return gpu_integration.c.beat_sycl_result_t.BEAT_SYCL_SUCCESS;
}

export fn beat_sycl_get_platforms(platforms: [*]gpu_integration.c.beat_sycl_platform_t, capacity: u32, count: *u32) gpu_integration.c.beat_sycl_result_t {
    _ = platforms;
    _ = capacity;
    count.* = 0;
    return gpu_integration.c.beat_sycl_result_t.BEAT_SYCL_SUCCESS;
}

export fn beat_sycl_get_device_count(platform: gpu_integration.c.beat_sycl_platform_t, device_type: gpu_integration.c.beat_sycl_device_type_t, count: *u32) gpu_integration.c.beat_sycl_result_t {
    _ = platform;
    _ = device_type;
    count.* = 0;
    return gpu_integration.c.beat_sycl_result_t.BEAT_SYCL_SUCCESS;
}

export fn beat_sycl_get_devices(platform: gpu_integration.c.beat_sycl_platform_t, device_type: gpu_integration.c.beat_sycl_device_type_t, devices: [*]gpu_integration.c.beat_sycl_device_t, capacity: u32, count: *u32) gpu_integration.c.beat_sycl_result_t {
    _ = platform;
    _ = device_type;
    _ = devices;
    _ = capacity;
    count.* = 0;
    return gpu_integration.c.beat_sycl_result_t.BEAT_SYCL_SUCCESS;
}

export fn beat_sycl_get_device_info(device: gpu_integration.c.beat_sycl_device_t, info: *gpu_integration.c.beat_sycl_device_info_t) gpu_integration.c.beat_sycl_result_t {
    _ = device;
    _ = info;
    return gpu_integration.c.beat_sycl_result_t.BEAT_SYCL_ERROR_NO_DEVICE;
}

export fn beat_sycl_create_context(devices: [*]gpu_integration.c.beat_sycl_device_t, device_count: u32, context: *gpu_integration.c.beat_sycl_context_t) gpu_integration.c.beat_sycl_result_t {
    _ = devices;
    _ = device_count;
    _ = context;
    return gpu_integration.c.beat_sycl_result_t.BEAT_SYCL_ERROR_NO_DEVICE;
}

export fn beat_sycl_create_queue(device: gpu_integration.c.beat_sycl_device_t, context: gpu_integration.c.beat_sycl_context_t, queue: *gpu_integration.c.beat_sycl_queue_t) gpu_integration.c.beat_sycl_result_t {
    _ = device;
    _ = context;
    _ = queue;
    return gpu_integration.c.beat_sycl_result_t.BEAT_SYCL_ERROR_NO_DEVICE;
}

export fn beat_sycl_malloc_device(queue: gpu_integration.c.beat_sycl_queue_t, size: usize, alignment: usize, ptr: *?*anyopaque) gpu_integration.c.beat_sycl_result_t {
    _ = queue;
    _ = size;
    _ = alignment;
    ptr.* = null;
    return gpu_integration.c.beat_sycl_result_t.BEAT_SYCL_ERROR_NO_DEVICE;
}

export fn beat_sycl_free(ptr: *anyopaque, queue: gpu_integration.c.beat_sycl_queue_t) gpu_integration.c.beat_sycl_result_t {
    _ = ptr;
    _ = queue;
    return gpu_integration.c.beat_sycl_result_t.BEAT_SYCL_ERROR_NO_DEVICE;
}

export fn beat_sycl_destroy_queue(queue: gpu_integration.c.beat_sycl_queue_t) gpu_integration.c.beat_sycl_result_t {
    _ = queue;
    return gpu_integration.c.beat_sycl_result_t.BEAT_SYCL_SUCCESS;
}

export fn beat_sycl_destroy_context(context: gpu_integration.c.beat_sycl_context_t) gpu_integration.c.beat_sycl_result_t {
    _ = context;
    return gpu_integration.c.beat_sycl_result_t.BEAT_SYCL_SUCCESS;
}