#pragma once

#include <cstddef>
#include <cstdint>

// SYCL C++ to C API Wrapper for Beat.zig GPU Integration (Task 3.1.1)
//
// This header provides a clean C interface for SYCL functionality using the
// hourglass design pattern for ABI stability and opaque pointer management.
// 
// Design principles:
// - Opaque pointers for C++ object lifetime management
// - Exception-safe C interface with error codes
// - Minimal C-compatible types for Zig interoperability
// - Resource management through explicit create/destroy pairs

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Error Handling and Status Codes
// ============================================================================

typedef enum {
    BEAT_SYCL_SUCCESS = 0,
    BEAT_SYCL_ERROR_INVALID_PARAMETER = 1,
    BEAT_SYCL_ERROR_NO_DEVICE = 2,
    BEAT_SYCL_ERROR_DEVICE_NOT_AVAILABLE = 3,
    BEAT_SYCL_ERROR_OUT_OF_MEMORY = 4,
    BEAT_SYCL_ERROR_COMPILATION_FAILED = 5,
    BEAT_SYCL_ERROR_EXECUTION_FAILED = 6,
    BEAT_SYCL_ERROR_INVALID_QUEUE = 7,
    BEAT_SYCL_ERROR_INVALID_CONTEXT = 8,
    BEAT_SYCL_ERROR_BACKEND_NOT_SUPPORTED = 9,
    BEAT_SYCL_ERROR_UNKNOWN = 999
} beat_sycl_result_t;

// ============================================================================
// Device Information and Capabilities
// ============================================================================

typedef enum {
    BEAT_SYCL_DEVICE_TYPE_CPU = 0,
    BEAT_SYCL_DEVICE_TYPE_GPU = 1,
    BEAT_SYCL_DEVICE_TYPE_ACCELERATOR = 2,
    BEAT_SYCL_DEVICE_TYPE_CUSTOM = 3,
    BEAT_SYCL_DEVICE_TYPE_AUTOMATIC = 4,
    BEAT_SYCL_DEVICE_TYPE_HOST = 5,
    BEAT_SYCL_DEVICE_TYPE_ALL = 6
} beat_sycl_device_type_t;

typedef enum {
    BEAT_SYCL_BACKEND_OPENCL = 0,
    BEAT_SYCL_BACKEND_LEVEL_ZERO = 1,
    BEAT_SYCL_BACKEND_CUDA = 2,
    BEAT_SYCL_BACKEND_HIP = 3,
    BEAT_SYCL_BACKEND_HOST = 4
} beat_sycl_backend_t;

typedef struct {
    char name[256];                    // Device name
    char vendor[128];                  // Device vendor
    char driver_version[64];           // Driver version
    beat_sycl_device_type_t type;      // Device type
    beat_sycl_backend_t backend;       // Backend type
    
    // Compute capabilities
    uint32_t max_compute_units;        // Maximum compute units
    uint64_t max_work_group_size;      // Maximum work group size
    uint64_t max_work_item_dimensions; // Maximum work item dimensions
    uint64_t max_work_item_sizes[3];   // Maximum work item sizes per dimension
    
    // Memory capabilities
    uint64_t global_memory_size;       // Global memory size in bytes
    uint64_t local_memory_size;        // Local memory size in bytes
    uint64_t max_memory_allocation;    // Maximum single allocation size
    
    // Performance characteristics
    uint32_t preferred_vector_width_float;  // Preferred vector width for float
    uint32_t preferred_vector_width_double; // Preferred vector width for double
    uint32_t preferred_vector_width_int;    // Preferred vector width for int
    
    // Feature support flags
    uint32_t supports_double : 1;      // Double precision support
    uint32_t supports_half : 1;        // Half precision support
    uint32_t supports_unified_memory : 1; // Unified Shared Memory support
    uint32_t supports_sub_groups : 1;  // Sub-group support
    uint32_t reserved : 28;            // Reserved for future use
} beat_sycl_device_info_t;

// ============================================================================
// Opaque Handle Types
// ============================================================================

// Opaque pointers for C++ object management
typedef struct beat_sycl_platform_impl* beat_sycl_platform_t;
typedef struct beat_sycl_device_impl* beat_sycl_device_t;
typedef struct beat_sycl_context_impl* beat_sycl_context_t;
typedef struct beat_sycl_queue_impl* beat_sycl_queue_t;
typedef struct beat_sycl_buffer_impl* beat_sycl_buffer_t;
typedef struct beat_sycl_kernel_impl* beat_sycl_kernel_t;
typedef struct beat_sycl_event_impl* beat_sycl_event_t;

// ============================================================================
// Platform and Device Discovery
// ============================================================================

/**
 * Get the number of available SYCL platforms
 * @param count Output parameter for platform count
 * @return Status code indicating success or failure
 */
beat_sycl_result_t beat_sycl_get_platform_count(uint32_t* count);

/**
 * Get available SYCL platforms
 * @param platforms Array to store platform handles (caller allocated)
 * @param capacity Maximum number of platforms to retrieve
 * @param count Actual number of platforms retrieved
 * @return Status code indicating success or failure
 */
beat_sycl_result_t beat_sycl_get_platforms(
    beat_sycl_platform_t* platforms,
    uint32_t capacity,
    uint32_t* count
);

/**
 * Get the number of devices for a specific platform
 * @param platform Platform handle
 * @param device_type Type of devices to count
 * @param count Output parameter for device count
 * @return Status code indicating success or failure
 */
beat_sycl_result_t beat_sycl_get_device_count(
    beat_sycl_platform_t platform,
    beat_sycl_device_type_t device_type,
    uint32_t* count
);

/**
 * Get devices for a specific platform
 * @param platform Platform handle
 * @param device_type Type of devices to retrieve
 * @param devices Array to store device handles (caller allocated)
 * @param capacity Maximum number of devices to retrieve
 * @param count Actual number of devices retrieved
 * @return Status code indicating success or failure
 */
beat_sycl_result_t beat_sycl_get_devices(
    beat_sycl_platform_t platform,
    beat_sycl_device_type_t device_type,
    beat_sycl_device_t* devices,
    uint32_t capacity,
    uint32_t* count
);

/**
 * Get detailed information about a device
 * @param device Device handle
 * @param info Output parameter for device information
 * @return Status code indicating success or failure
 */
beat_sycl_result_t beat_sycl_get_device_info(
    beat_sycl_device_t device,
    beat_sycl_device_info_t* info
);

// ============================================================================
// Context and Queue Management
// ============================================================================

/**
 * Create a SYCL context from devices
 * @param devices Array of device handles
 * @param device_count Number of devices
 * @param context Output parameter for context handle
 * @return Status code indicating success or failure
 */
beat_sycl_result_t beat_sycl_create_context(
    beat_sycl_device_t* devices,
    uint32_t device_count,
    beat_sycl_context_t* context
);

/**
 * Create a SYCL queue for a device
 * @param device Device handle
 * @param context Context handle (can be null for default context)
 * @param queue Output parameter for queue handle
 * @return Status code indicating success or failure
 */
beat_sycl_result_t beat_sycl_create_queue(
    beat_sycl_device_t device,
    beat_sycl_context_t context,
    beat_sycl_queue_t* queue
);

/**
 * Wait for all operations in a queue to complete
 * @param queue Queue handle
 * @return Status code indicating success or failure
 */
beat_sycl_result_t beat_sycl_queue_wait(beat_sycl_queue_t queue);

/**
 * Check if a queue has finished all operations
 * @param queue Queue handle
 * @param finished Output parameter indicating completion status
 * @return Status code indicating success or failure
 */
beat_sycl_result_t beat_sycl_queue_is_finished(
    beat_sycl_queue_t queue,
    uint32_t* finished
);

// ============================================================================
// Memory Management
// ============================================================================

/**
 * Allocate device memory
 * @param queue Queue handle
 * @param size Size in bytes
 * @param alignment Alignment requirement (0 for default)
 * @param ptr Output parameter for allocated pointer
 * @return Status code indicating success or failure
 */
beat_sycl_result_t beat_sycl_malloc_device(
    beat_sycl_queue_t queue,
    size_t size,
    size_t alignment,
    void** ptr
);

/**
 * Allocate host memory
 * @param queue Queue handle
 * @param size Size in bytes
 * @param alignment Alignment requirement (0 for default)
 * @param ptr Output parameter for allocated pointer
 * @return Status code indicating success or failure
 */
beat_sycl_result_t beat_sycl_malloc_host(
    beat_sycl_queue_t queue,
    size_t size,
    size_t alignment,
    void** ptr
);

/**
 * Allocate shared memory (if supported)
 * @param queue Queue handle
 * @param size Size in bytes
 * @param alignment Alignment requirement (0 for default)
 * @param ptr Output parameter for allocated pointer
 * @return Status code indicating success or failure
 */
beat_sycl_result_t beat_sycl_malloc_shared(
    beat_sycl_queue_t queue,
    size_t size,
    size_t alignment,
    void** ptr
);

/**
 * Free allocated memory
 * @param ptr Pointer to free
 * @param queue Queue handle
 * @return Status code indicating success or failure
 */
beat_sycl_result_t beat_sycl_free(void* ptr, beat_sycl_queue_t queue);

/**
 * Copy memory between host and device
 * @param queue Queue handle
 * @param dest Destination pointer
 * @param src Source pointer
 * @param size Size in bytes
 * @param event Output parameter for event handle (can be null)
 * @return Status code indicating success or failure
 */
beat_sycl_result_t beat_sycl_memcpy(
    beat_sycl_queue_t queue,
    void* dest,
    const void* src,
    size_t size,
    beat_sycl_event_t* event
);

// ============================================================================
// Basic Kernel Execution
// ============================================================================

/**
 * Submit a parallel_for kernel with 1D range
 * @param queue Queue handle
 * @param global_size Global work size
 * @param local_size Local work size (0 for automatic)
 * @param kernel_func Kernel function pointer (placeholder for future implementation)
 * @param args Kernel arguments (placeholder for future implementation)
 * @param event Output parameter for event handle (can be null)
 * @return Status code indicating success or failure
 */
beat_sycl_result_t beat_sycl_submit_kernel_1d(
    beat_sycl_queue_t queue,
    size_t global_size,
    size_t local_size,
    void* kernel_func,
    void* args,
    beat_sycl_event_t* event
);

// ============================================================================
// Event Management
// ============================================================================

/**
 * Wait for an event to complete
 * @param event Event handle
 * @return Status code indicating success or failure
 */
beat_sycl_result_t beat_sycl_event_wait(beat_sycl_event_t event);

/**
 * Check if an event has completed
 * @param event Event handle
 * @param completed Output parameter indicating completion status
 * @return Status code indicating success or failure
 */
beat_sycl_result_t beat_sycl_event_is_complete(
    beat_sycl_event_t event,
    uint32_t* completed
);

// ============================================================================
// Resource Cleanup
// ============================================================================

/**
 * Destroy a context and free associated resources
 * @param context Context handle
 * @return Status code indicating success or failure
 */
beat_sycl_result_t beat_sycl_destroy_context(beat_sycl_context_t context);

/**
 * Destroy a queue and free associated resources
 * @param queue Queue handle
 * @return Status code indicating success or failure
 */
beat_sycl_result_t beat_sycl_destroy_queue(beat_sycl_queue_t queue);

/**
 * Destroy an event and free associated resources
 * @param event Event handle
 * @return Status code indicating success or failure
 */
beat_sycl_result_t beat_sycl_destroy_event(beat_sycl_event_t event);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get string description of a result code
 * @param result Result code
 * @return Null-terminated string description (valid until next call)
 */
const char* beat_sycl_get_error_string(beat_sycl_result_t result);

/**
 * Get SYCL runtime version information
 * @param version_string Output buffer for version string
 * @param buffer_size Size of output buffer
 * @return Status code indicating success or failure
 */
beat_sycl_result_t beat_sycl_get_version(char* version_string, size_t buffer_size);

/**
 * Check if SYCL runtime is available and initialized
 * @param available Output parameter indicating availability
 * @return Status code indicating success or failure
 */
beat_sycl_result_t beat_sycl_is_available(uint32_t* available);

#ifdef __cplusplus
}
#endif