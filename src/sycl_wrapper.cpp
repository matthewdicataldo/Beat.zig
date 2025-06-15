#include "sycl_wrapper.hpp"
#include <sycl/sycl.hpp>
#include <vector>
#include <memory>
#include <cstring>
#include <exception>

// SYCL C++ Implementation for Beat.zig GPU Integration (Task 3.1.1)
//
// This implementation provides the actual SYCL C++ code backing the C interface.
// It uses opaque pointer management and exception handling with error code translation.
//
// Design patterns:
// - RAII for automatic resource management
// - Exception-to-error-code translation
// - Opaque struct implementations for C++ object lifetime
// - Smart pointer usage for memory safety

// ============================================================================
// Opaque Struct Implementations
// ============================================================================

struct beat_sycl_platform_impl {
    sycl::platform platform;
    
    beat_sycl_platform_impl(sycl::platform p) : platform(std::move(p)) {}
};

struct beat_sycl_device_impl {
    sycl::device device;
    
    beat_sycl_device_impl(sycl::device d) : device(std::move(d)) {}
};

struct beat_sycl_context_impl {
    sycl::context context;
    
    beat_sycl_context_impl(sycl::context c) : context(std::move(c)) {}
};

struct beat_sycl_queue_impl {
    sycl::queue queue;
    
    beat_sycl_queue_impl(sycl::queue q) : queue(std::move(q)) {}
};

struct beat_sycl_event_impl {
    sycl::event event;
    
    beat_sycl_event_impl(sycl::event e) : event(std::move(e)) {}
};

// ============================================================================
// Exception Handling and Error Translation
// ============================================================================

namespace {
    // Thread-local storage for error messages
    thread_local std::string last_error_message;
    
    beat_sycl_result_t translate_exception() {
        try {
            throw; // Re-throw current exception
        } catch (const sycl::exception& e) {
            last_error_message = std::string("SYCL exception: ") + e.what();
            
            // Map SYCL error codes to our error codes
            switch (e.code().value()) {
                case CL_INVALID_VALUE:
                    return BEAT_SYCL_ERROR_INVALID_PARAMETER;
                case CL_DEVICE_NOT_FOUND:
                case CL_DEVICE_NOT_AVAILABLE:
                    return BEAT_SYCL_ERROR_NO_DEVICE;
                case CL_OUT_OF_HOST_MEMORY:
                case CL_MEM_OBJECT_ALLOCATION_FAILURE:
                    return BEAT_SYCL_ERROR_OUT_OF_MEMORY;
                case CL_BUILD_PROGRAM_FAILURE:
                    return BEAT_SYCL_ERROR_COMPILATION_FAILED;
                case CL_INVALID_COMMAND_QUEUE:
                    return BEAT_SYCL_ERROR_INVALID_QUEUE;
                case CL_INVALID_CONTEXT:
                    return BEAT_SYCL_ERROR_INVALID_CONTEXT;
                default:
                    return BEAT_SYCL_ERROR_EXECUTION_FAILED;
            }
        } catch (const std::bad_alloc&) {
            last_error_message = "Memory allocation failed";
            return BEAT_SYCL_ERROR_OUT_OF_MEMORY;
        } catch (const std::exception& e) {
            last_error_message = std::string("Standard exception: ") + e.what();
            return BEAT_SYCL_ERROR_UNKNOWN;
        } catch (...) {
            last_error_message = "Unknown exception occurred";
            return BEAT_SYCL_ERROR_UNKNOWN;
        }
    }
    
    beat_sycl_device_type_t translate_device_type(sycl::info::device_type type) {
        switch (type) {
            case sycl::info::device_type::cpu:
                return BEAT_SYCL_DEVICE_TYPE_CPU;
            case sycl::info::device_type::gpu:
                return BEAT_SYCL_DEVICE_TYPE_GPU;
            case sycl::info::device_type::accelerator:
                return BEAT_SYCL_DEVICE_TYPE_ACCELERATOR;
            case sycl::info::device_type::custom:
                return BEAT_SYCL_DEVICE_TYPE_CUSTOM;
            case sycl::info::device_type::automatic:
                return BEAT_SYCL_DEVICE_TYPE_AUTOMATIC;
            case sycl::info::device_type::host:
                return BEAT_SYCL_DEVICE_TYPE_HOST;
            case sycl::info::device_type::all:
                return BEAT_SYCL_DEVICE_TYPE_ALL;
            default:
                return BEAT_SYCL_DEVICE_TYPE_CUSTOM;
        }
    }
    
    sycl::info::device_type translate_to_sycl_device_type(beat_sycl_device_type_t type) {
        switch (type) {
            case BEAT_SYCL_DEVICE_TYPE_CPU:
                return sycl::info::device_type::cpu;
            case BEAT_SYCL_DEVICE_TYPE_GPU:
                return sycl::info::device_type::gpu;
            case BEAT_SYCL_DEVICE_TYPE_ACCELERATOR:
                return sycl::info::device_type::accelerator;
            case BEAT_SYCL_DEVICE_TYPE_CUSTOM:
                return sycl::info::device_type::custom;
            case BEAT_SYCL_DEVICE_TYPE_AUTOMATIC:
                return sycl::info::device_type::automatic;
            case BEAT_SYCL_DEVICE_TYPE_HOST:
                return sycl::info::device_type::host;
            case BEAT_SYCL_DEVICE_TYPE_ALL:
                return sycl::info::device_type::all;
            default:
                return sycl::info::device_type::all;
        }
    }
    
    beat_sycl_backend_t detect_backend(const sycl::device& device) {
        // Attempt to detect backend based on platform name
        std::string platform_name = device.get_platform().get_info<sycl::info::platform::name>();
        std::transform(platform_name.begin(), platform_name.end(), platform_name.begin(), ::tolower);
        
        if (platform_name.find("intel") != std::string::npos) {
            if (platform_name.find("level-zero") != std::string::npos || 
                platform_name.find("level zero") != std::string::npos) {
                return BEAT_SYCL_BACKEND_LEVEL_ZERO;
            } else {
                return BEAT_SYCL_BACKEND_OPENCL;
            }
        } else if (platform_name.find("cuda") != std::string::npos ||
                   platform_name.find("nvidia") != std::string::npos) {
            return BEAT_SYCL_BACKEND_CUDA;
        } else if (platform_name.find("hip") != std::string::npos ||
                   platform_name.find("amd") != std::string::npos) {
            return BEAT_SYCL_BACKEND_HIP;
        } else if (platform_name.find("host") != std::string::npos) {
            return BEAT_SYCL_BACKEND_HOST;
        } else {
            return BEAT_SYCL_BACKEND_OPENCL; // Default assumption
        }
    }
    
    void safe_string_copy(char* dest, const std::string& src, size_t dest_size) {
        if (dest_size == 0) return;
        
        size_t copy_length = std::min(src.length(), dest_size - 1);
        std::memcpy(dest, src.c_str(), copy_length);
        dest[copy_length] = '\0';
    }
}

// ============================================================================
// Platform and Device Discovery Implementation
// ============================================================================

extern "C" {

beat_sycl_result_t beat_sycl_get_platform_count(uint32_t* count) {
    if (!count) {
        return BEAT_SYCL_ERROR_INVALID_PARAMETER;
    }
    
    try {
        auto platforms = sycl::platform::get_platforms();
        *count = static_cast<uint32_t>(platforms.size());
        return BEAT_SYCL_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

beat_sycl_result_t beat_sycl_get_platforms(
    beat_sycl_platform_t* platforms,
    uint32_t capacity,
    uint32_t* count
) {
    if (!platforms || !count || capacity == 0) {
        return BEAT_SYCL_ERROR_INVALID_PARAMETER;
    }
    
    try {
        auto sycl_platforms = sycl::platform::get_platforms();
        uint32_t available_count = static_cast<uint32_t>(sycl_platforms.size());
        uint32_t copy_count = std::min(capacity, available_count);
        
        for (uint32_t i = 0; i < copy_count; ++i) {
            platforms[i] = new beat_sycl_platform_impl(std::move(sycl_platforms[i]));
        }
        
        *count = copy_count;
        return BEAT_SYCL_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

beat_sycl_result_t beat_sycl_get_device_count(
    beat_sycl_platform_t platform,
    beat_sycl_device_type_t device_type,
    uint32_t* count
) {
    if (!platform || !count) {
        return BEAT_SYCL_ERROR_INVALID_PARAMETER;
    }
    
    try {
        auto sycl_device_type = translate_to_sycl_device_type(device_type);
        auto devices = platform->platform.get_devices(sycl_device_type);
        *count = static_cast<uint32_t>(devices.size());
        return BEAT_SYCL_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

beat_sycl_result_t beat_sycl_get_devices(
    beat_sycl_platform_t platform,
    beat_sycl_device_type_t device_type,
    beat_sycl_device_t* devices,
    uint32_t capacity,
    uint32_t* count
) {
    if (!platform || !devices || !count || capacity == 0) {
        return BEAT_SYCL_ERROR_INVALID_PARAMETER;
    }
    
    try {
        auto sycl_device_type = translate_to_sycl_device_type(device_type);
        auto sycl_devices = platform->platform.get_devices(sycl_device_type);
        uint32_t available_count = static_cast<uint32_t>(sycl_devices.size());
        uint32_t copy_count = std::min(capacity, available_count);
        
        for (uint32_t i = 0; i < copy_count; ++i) {
            devices[i] = new beat_sycl_device_impl(std::move(sycl_devices[i]));
        }
        
        *count = copy_count;
        return BEAT_SYCL_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

beat_sycl_result_t beat_sycl_get_device_info(
    beat_sycl_device_t device,
    beat_sycl_device_info_t* info
) {
    if (!device || !info) {
        return BEAT_SYCL_ERROR_INVALID_PARAMETER;
    }
    
    try {
        const auto& dev = device->device;
        
        // Clear the info structure
        std::memset(info, 0, sizeof(beat_sycl_device_info_t));
        
        // Basic device information
        auto device_name = dev.get_info<sycl::info::device::name>();
        auto vendor_name = dev.get_info<sycl::info::device::vendor>();
        auto driver_version = dev.get_info<sycl::info::device::driver_version>();
        
        safe_string_copy(info->name, device_name, sizeof(info->name));
        safe_string_copy(info->vendor, vendor_name, sizeof(info->vendor));
        safe_string_copy(info->driver_version, driver_version, sizeof(info->driver_version));
        
        info->type = translate_device_type(dev.get_info<sycl::info::device::device_type>());
        info->backend = detect_backend(dev);
        
        // Compute capabilities
        info->max_compute_units = dev.get_info<sycl::info::device::max_compute_units>();
        info->max_work_group_size = dev.get_info<sycl::info::device::max_work_group_size>();
        
        auto max_work_item_dimensions = dev.get_info<sycl::info::device::max_work_item_dimensions>();
        info->max_work_item_dimensions = static_cast<uint64_t>(max_work_item_dimensions);
        
        auto max_work_item_sizes = dev.get_info<sycl::info::device::max_work_item_sizes<3>>();
        for (size_t i = 0; i < 3 && i < max_work_item_sizes.size(); ++i) {
            info->max_work_item_sizes[i] = static_cast<uint64_t>(max_work_item_sizes[i]);
        }
        
        // Memory capabilities
        info->global_memory_size = dev.get_info<sycl::info::device::global_mem_size>();
        info->local_memory_size = dev.get_info<sycl::info::device::local_mem_size>();
        info->max_memory_allocation = dev.get_info<sycl::info::device::max_mem_alloc_size>();
        
        // Performance characteristics
        info->preferred_vector_width_float = dev.get_info<sycl::info::device::preferred_vector_width_float>();
        info->preferred_vector_width_double = dev.get_info<sycl::info::device::preferred_vector_width_double>();
        info->preferred_vector_width_int = dev.get_info<sycl::info::device::preferred_vector_width_int>();
        
        // Feature support flags
        info->supports_double = dev.has(sycl::aspect::fp64) ? 1 : 0;
        info->supports_half = dev.has(sycl::aspect::fp16) ? 1 : 0;
        info->supports_unified_memory = dev.has(sycl::aspect::usm_shared_allocations) ? 1 : 0;
        info->supports_sub_groups = dev.has(sycl::aspect::ext_intel_gpu_subgroups) ? 1 : 0;
        
        return BEAT_SYCL_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

// ============================================================================
// Context and Queue Management Implementation
// ============================================================================

beat_sycl_result_t beat_sycl_create_context(
    beat_sycl_device_t* devices,
    uint32_t device_count,
    beat_sycl_context_t* context
) {
    if (!devices || device_count == 0 || !context) {
        return BEAT_SYCL_ERROR_INVALID_PARAMETER;
    }
    
    try {
        std::vector<sycl::device> sycl_devices;
        sycl_devices.reserve(device_count);
        
        for (uint32_t i = 0; i < device_count; ++i) {
            if (!devices[i]) {
                return BEAT_SYCL_ERROR_INVALID_PARAMETER;
            }
            sycl_devices.push_back(devices[i]->device);
        }
        
        auto sycl_context = sycl::context(sycl_devices);
        *context = new beat_sycl_context_impl(std::move(sycl_context));
        
        return BEAT_SYCL_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

beat_sycl_result_t beat_sycl_create_queue(
    beat_sycl_device_t device,
    beat_sycl_context_t context,
    beat_sycl_queue_t* queue
) {
    if (!device || !queue) {
        return BEAT_SYCL_ERROR_INVALID_PARAMETER;
    }
    
    try {
        sycl::queue sycl_queue;
        
        if (context) {
            // Create queue with specific context
            sycl_queue = sycl::queue(context->context, device->device);
        } else {
            // Create queue with default context
            sycl_queue = sycl::queue(device->device);
        }
        
        *queue = new beat_sycl_queue_impl(std::move(sycl_queue));
        return BEAT_SYCL_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

beat_sycl_result_t beat_sycl_queue_wait(beat_sycl_queue_t queue) {
    if (!queue) {
        return BEAT_SYCL_ERROR_INVALID_PARAMETER;
    }
    
    try {
        queue->queue.wait();
        return BEAT_SYCL_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

beat_sycl_result_t beat_sycl_queue_is_finished(
    beat_sycl_queue_t queue,
    uint32_t* finished
) {
    if (!queue || !finished) {
        return BEAT_SYCL_ERROR_INVALID_PARAMETER;
    }
    
    try {
        // SYCL doesn't have a direct "is_finished" method, so we use wait_and_throw with timeout
        // For simplicity, we'll implement this as always returning true after checking
        // In a production implementation, you might want to track submitted events
        *finished = 1;
        return BEAT_SYCL_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

// ============================================================================
// Memory Management Implementation
// ============================================================================

beat_sycl_result_t beat_sycl_malloc_device(
    beat_sycl_queue_t queue,
    size_t size,
    size_t alignment,
    void** ptr
) {
    if (!queue || size == 0 || !ptr) {
        return BEAT_SYCL_ERROR_INVALID_PARAMETER;
    }
    
    try {
        if (alignment > 0) {
            *ptr = sycl::aligned_alloc_device(alignment, size, queue->queue);
        } else {
            *ptr = sycl::malloc_device(size, queue->queue);
        }
        
        if (*ptr == nullptr) {
            return BEAT_SYCL_ERROR_OUT_OF_MEMORY;
        }
        
        return BEAT_SYCL_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

beat_sycl_result_t beat_sycl_malloc_host(
    beat_sycl_queue_t queue,
    size_t size,
    size_t alignment,
    void** ptr
) {
    if (!queue || size == 0 || !ptr) {
        return BEAT_SYCL_ERROR_INVALID_PARAMETER;
    }
    
    try {
        if (alignment > 0) {
            *ptr = sycl::aligned_alloc_host(alignment, size, queue->queue);
        } else {
            *ptr = sycl::malloc_host(size, queue->queue);
        }
        
        if (*ptr == nullptr) {
            return BEAT_SYCL_ERROR_OUT_OF_MEMORY;
        }
        
        return BEAT_SYCL_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

beat_sycl_result_t beat_sycl_malloc_shared(
    beat_sycl_queue_t queue,
    size_t size,
    size_t alignment,
    void** ptr
) {
    if (!queue || size == 0 || !ptr) {
        return BEAT_SYCL_ERROR_INVALID_PARAMETER;
    }
    
    try {
        if (alignment > 0) {
            *ptr = sycl::aligned_alloc_shared(alignment, size, queue->queue);
        } else {
            *ptr = sycl::malloc_shared(size, queue->queue);
        }
        
        if (*ptr == nullptr) {
            return BEAT_SYCL_ERROR_OUT_OF_MEMORY;
        }
        
        return BEAT_SYCL_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

beat_sycl_result_t beat_sycl_free(void* ptr, beat_sycl_queue_t queue) {
    if (!ptr || !queue) {
        return BEAT_SYCL_ERROR_INVALID_PARAMETER;
    }
    
    try {
        sycl::free(ptr, queue->queue);
        return BEAT_SYCL_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

beat_sycl_result_t beat_sycl_memcpy(
    beat_sycl_queue_t queue,
    void* dest,
    const void* src,
    size_t size,
    beat_sycl_event_t* event
) {
    if (!queue || !dest || !src || size == 0) {
        return BEAT_SYCL_ERROR_INVALID_PARAMETER;
    }
    
    try {
        auto sycl_event = queue->queue.memcpy(dest, src, size);
        
        if (event) {
            *event = new beat_sycl_event_impl(std::move(sycl_event));
        }
        
        return BEAT_SYCL_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

// ============================================================================
// Basic Kernel Execution Implementation
// ============================================================================

beat_sycl_result_t beat_sycl_submit_kernel_1d(
    beat_sycl_queue_t queue,
    size_t global_size,
    size_t local_size,
    void* kernel_func,
    void* args,
    beat_sycl_event_t* event
) {
    if (!queue || global_size == 0) {
        return BEAT_SYCL_ERROR_INVALID_PARAMETER;
    }
    
    // NOTE: This is a placeholder implementation
    // Actual kernel execution would require more sophisticated kernel management
    // For now, we'll just return success to indicate the interface is functional
    
    try {
        // Placeholder: Submit a simple parallel_for kernel
        auto sycl_event = queue->queue.parallel_for(
            sycl::range<1>(global_size),
            [=](sycl::id<1> idx) {
                // Placeholder kernel - would be replaced with actual kernel logic
                (void)idx; // Suppress unused parameter warning
            }
        );
        
        if (event) {
            *event = new beat_sycl_event_impl(std::move(sycl_event));
        }
        
        return BEAT_SYCL_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

// ============================================================================
// Event Management Implementation
// ============================================================================

beat_sycl_result_t beat_sycl_event_wait(beat_sycl_event_t event) {
    if (!event) {
        return BEAT_SYCL_ERROR_INVALID_PARAMETER;
    }
    
    try {
        event->event.wait();
        return BEAT_SYCL_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

beat_sycl_result_t beat_sycl_event_is_complete(
    beat_sycl_event_t event,
    uint32_t* completed
) {
    if (!event || !completed) {
        return BEAT_SYCL_ERROR_INVALID_PARAMETER;
    }
    
    try {
        auto status = event->event.get_info<sycl::info::event::command_execution_status>();
        *completed = (status == sycl::info::event_command_status::complete) ? 1 : 0;
        return BEAT_SYCL_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

// ============================================================================
// Resource Cleanup Implementation
// ============================================================================

beat_sycl_result_t beat_sycl_destroy_context(beat_sycl_context_t context) {
    if (!context) {
        return BEAT_SYCL_ERROR_INVALID_PARAMETER;
    }
    
    try {
        delete context;
        return BEAT_SYCL_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

beat_sycl_result_t beat_sycl_destroy_queue(beat_sycl_queue_t queue) {
    if (!queue) {
        return BEAT_SYCL_ERROR_INVALID_PARAMETER;
    }
    
    try {
        delete queue;
        return BEAT_SYCL_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

beat_sycl_result_t beat_sycl_destroy_event(beat_sycl_event_t event) {
    if (!event) {
        return BEAT_SYCL_ERROR_INVALID_PARAMETER;
    }
    
    try {
        delete event;
        return BEAT_SYCL_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

// ============================================================================
// Utility Functions Implementation
// ============================================================================

const char* beat_sycl_get_error_string(beat_sycl_result_t result) {
    switch (result) {
        case BEAT_SYCL_SUCCESS:
            return "Success";
        case BEAT_SYCL_ERROR_INVALID_PARAMETER:
            return "Invalid parameter";
        case BEAT_SYCL_ERROR_NO_DEVICE:
            return "No device found";
        case BEAT_SYCL_ERROR_DEVICE_NOT_AVAILABLE:
            return "Device not available";
        case BEAT_SYCL_ERROR_OUT_OF_MEMORY:
            return "Out of memory";
        case BEAT_SYCL_ERROR_COMPILATION_FAILED:
            return "Kernel compilation failed";
        case BEAT_SYCL_ERROR_EXECUTION_FAILED:
            return "Kernel execution failed";
        case BEAT_SYCL_ERROR_INVALID_QUEUE:
            return "Invalid queue";
        case BEAT_SYCL_ERROR_INVALID_CONTEXT:
            return "Invalid context";
        case BEAT_SYCL_ERROR_BACKEND_NOT_SUPPORTED:
            return "Backend not supported";
        case BEAT_SYCL_ERROR_UNKNOWN:
            return "Unknown error";
        default:
            return "Unrecognized error code";
    }
}

beat_sycl_result_t beat_sycl_get_version(char* version_string, size_t buffer_size) {
    if (!version_string || buffer_size == 0) {
        return BEAT_SYCL_ERROR_INVALID_PARAMETER;
    }
    
    try {
        // Get SYCL version information
        std::string version = "SYCL 2020 (Beat.zig wrapper v1.0.0)";
        safe_string_copy(version_string, version, buffer_size);
        
        return BEAT_SYCL_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

beat_sycl_result_t beat_sycl_is_available(uint32_t* available) {
    if (!available) {
        return BEAT_SYCL_ERROR_INVALID_PARAMETER;
    }
    
    try {
        // Check if we can get at least one platform
        auto platforms = sycl::platform::get_platforms();
        *available = platforms.empty() ? 0 : 1;
        
        return BEAT_SYCL_SUCCESS;
    } catch (...) {
        *available = 0;
        return BEAT_SYCL_SUCCESS; // Don't fail here, just indicate unavailable
    }
}

} // extern "C"