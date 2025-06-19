// ISPC kernel for SIMD-optimized memory operations
// Replaces Zig SIMDAllocator and memory alignment functions
// Target: 3-8x faster memory operations with vectorized copies and alignment

// SIMD alignment requirements (matches Zig enum)
enum SIMDAlignment {
    ALIGN_SSE = 16,      // 128-bit alignment for SSE
    ALIGN_AVX = 32,      // 256-bit alignment for AVX
    ALIGN_AVX512 = 64,   // 512-bit alignment for AVX-512
    ALIGN_CACHE = 64     // Cache line alignment (most common)
};

// Memory operation types
enum MemoryOpType {
    COPY = 0,
    ZERO = 1,
    SET = 2,
    COMPARE = 3
};

// Vectorized memory copy with SIMD optimization
export void ispc_simd_memory_copy(
    uniform uint8 dest[],
    uniform uint8 src[], 
    uniform uint64 size,
    uniform int alignment
) {
    // Use alignment to determine optimal SIMD width
    uniform uint64 vector_size = alignment;
    uniform uint64 elements_per_iteration = vector_size;
    
    // Calculate aligned and unaligned portions
    uniform uint64 aligned_size = (size / elements_per_iteration) * elements_per_iteration;
    uniform uint64 remaining = size - aligned_size;
    
    // Vectorized copy for aligned portion
    foreach (i = 0 ... aligned_size step elements_per_iteration) {
        // Load vector from source
        #if TARGET_WIDTH >= 16
        varying uint32 src_vec = *((varying uint32 * uniform)(src + i));
        *((varying uint32 * uniform)(dest + i)) = src_vec;
        #else
        // Fallback for smaller SIMD widths
        for (uniform int j = 0; j < elements_per_iteration; j++) {
            dest[i + j] = src[i + j];
        }
        #endif
    }
    
    // Handle remaining bytes (scalar)
    for (uniform uint64 i = aligned_size; i < size; i++) {
        dest[i] = src[i];
    }
}

// Vectorized memory zero with SIMD optimization  
export void ispc_simd_memory_zero(
    uniform uint8 dest[],
    uniform uint64 size,
    uniform int alignment
) {
    uniform uint64 vector_size = alignment;
    uniform uint64 elements_per_iteration = vector_size;
    
    uniform uint64 aligned_size = (size / elements_per_iteration) * elements_per_iteration;
    uniform uint64 remaining = size - aligned_size;
    
    // Vectorized zero for aligned portion
    foreach (i = 0 ... aligned_size step elements_per_iteration) {
        #if TARGET_WIDTH >= 16
        varying uint32 zero_vec = 0;
        *((varying uint32 * uniform)(dest + i)) = zero_vec;
        #else
        for (uniform int j = 0; j < elements_per_iteration; j++) {
            dest[i + j] = 0;
        }
        #endif
    }
    
    // Handle remaining bytes
    for (uniform uint64 i = aligned_size; i < size; i++) {
        dest[i] = 0;
    }
}

// Vectorized memory set with SIMD optimization
export void ispc_simd_memory_set(
    uniform uint8 dest[],
    uniform uint8 value,
    uniform uint64 size,
    uniform int alignment  
) {
    uniform uint64 vector_size = alignment;
    uniform uint64 elements_per_iteration = vector_size;
    
    uniform uint64 aligned_size = (size / elements_per_iteration) * elements_per_iteration;
    
    // Vectorized set for aligned portion
    foreach (i = 0 ... aligned_size step elements_per_iteration) {
        #if TARGET_WIDTH >= 16
        varying uint8 set_vec = value;
        *((varying uint8 * uniform)(dest + i)) = set_vec;
        #else
        for (uniform int j = 0; j < elements_per_iteration; j++) {
            dest[i + j] = value;
        }
        #endif
    }
    
    // Handle remaining bytes
    for (uniform uint64 i = aligned_size; i < size; i++) {
        dest[i] = value;
    }
}

// Vectorized memory compare with early exit
export uniform bool ispc_simd_memory_compare(
    uniform uint8 a[],
    uniform uint8 b[],
    uniform uint64 size,
    uniform int alignment
) {
    uniform uint64 vector_size = alignment;
    uniform uint64 elements_per_iteration = vector_size;
    
    uniform uint64 aligned_size = (size / elements_per_iteration) * elements_per_iteration;
    
    // Vectorized compare for aligned portion
    for (uniform uint64 i = 0; i < aligned_size; i += elements_per_iteration) {
        #if TARGET_WIDTH >= 16
        varying uint32 a_vec = *((varying uint32 * uniform)(a + i));
        varying uint32 b_vec = *((varying uint32 * uniform)(b + i));
        
        // Check if any elements differ
        varying bool diff = (a_vec != b_vec);
        if (any(diff)) {
            return false; // Early exit on difference
        }
        #else
        for (uniform int j = 0; j < elements_per_iteration; j++) {
            if (a[i + j] != b[i + j]) {
                return false;
            }
        }
        #endif
    }
    
    // Handle remaining bytes
    for (uniform uint64 i = aligned_size; i < size; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    
    return true;
}

// Check memory alignment with SIMD requirements
export uniform bool ispc_check_alignment(
    uniform uint8 ptr[],
    uniform int required_alignment
) {
    // Cast pointer to integer for alignment check
    uniform uint64 addr = (uniform uint64)ptr;
    return (addr % required_alignment) == 0;
}

// Align pointer to SIMD boundary
export uniform uint8* uniform ispc_align_pointer(
    uniform uint8 ptr[],
    uniform int alignment
) {
    uniform uint64 addr = (uniform uint64)ptr;
    uniform uint64 aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
    return (uniform uint8* uniform)aligned_addr;
}

// Calculate required padding for alignment
export uniform uint64 ispc_alignment_padding(
    uniform uint8 ptr[],
    uniform int alignment
) {
    uniform uint64 addr = (uniform uint64)ptr;
    uniform uint64 aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
    return aligned_addr - addr;
}

// Vectorized memory prefetch for SIMD operations
export void ispc_simd_prefetch(
    uniform uint8 addresses[],
    uniform uint64 count,
    uniform int cache_level
) {
    // Prefetch multiple cache lines in parallel
    foreach (i = 0 ... count) {
        // ISPC provides prefetch intrinsics
        prefetch_l1(addresses + i * 64); // Assume 64-byte cache lines
        
        if (cache_level >= 2) {
            prefetch_l2(addresses + i * 64);
        }
        if (cache_level >= 3) {
            prefetch_l3(addresses + i * 64);
        }
    }
}

// NUMA-aware memory operations (Linux-specific optimization)
export void ispc_numa_aware_copy(
    uniform uint8 dest[],
    uniform uint8 src[],
    uniform uint64 size,
    uniform int numa_node_src,
    uniform int numa_node_dest,
    uniform int alignment
) {
    // If source and destination are on the same NUMA node, use regular copy
    if (numa_node_src == numa_node_dest) {
        ispc_simd_memory_copy(dest, src, size, alignment);
        return;
    }
    
    // For cross-NUMA copies, use smaller chunks to minimize latency
    uniform uint64 chunk_size = 4096; // 4KB chunks for cross-NUMA
    uniform uint64 chunks = (size + chunk_size - 1) / chunk_size;
    
    for (uniform uint64 chunk = 0; chunk < chunks; chunk++) {
        uniform uint64 offset = chunk * chunk_size;
        uniform uint64 copy_size = min(chunk_size, size - offset);
        
        // Prefetch next chunk while copying current
        if (chunk + 1 < chunks) {
            uniform uint64 next_offset = (chunk + 1) * chunk_size;
            prefetch_l1(src + next_offset);
            prefetch_l1(dest + next_offset);
        }
        
        ispc_simd_memory_copy(dest + offset, src + offset, copy_size, alignment);
    }
}

// Optimized scatter/gather operations for SIMD data structures
export void ispc_simd_gather(
    uniform uint8 dest[],
    uniform uint8 src[],
    uniform uint64 indices[],
    uniform uint64 element_size,
    uniform uint64 count
) {
    foreach (i = 0 ... count) {
        uniform uint64 src_offset = indices[i] * element_size;
        uniform uint64 dest_offset = i * element_size;
        
        // Copy element (vectorized for larger elements)
        for (uniform uint64 j = 0; j < element_size; j++) {
            dest[dest_offset + j] = src[src_offset + j];
        }
    }
}

export void ispc_simd_scatter(
    uniform uint8 dest[],
    uniform uint8 src[],
    uniform uint64 indices[],
    uniform uint64 element_size,
    uniform uint64 count
) {
    foreach (i = 0 ... count) {
        uniform uint64 src_offset = i * element_size;
        uniform uint64 dest_offset = indices[i] * element_size;
        
        // Copy element (vectorized for larger elements)
        for (uniform uint64 j = 0; j < element_size; j++) {
            dest[dest_offset + j] = src[src_offset + j];
        }
    }
}

// Memory bandwidth testing for performance tuning
export uniform float ispc_measure_memory_bandwidth(
    uniform uint8 test_buffer[],
    uniform uint64 buffer_size,
    uniform int iterations,
    uniform int operation_type
) {
    uniform uint64 start_time = 0; // Would use system timer in real implementation
    
    for (uniform int iter = 0; iter < iterations; iter++) {
        if (operation_type == COPY) {
            // Copy first half to second half
            ispc_simd_memory_copy(
                test_buffer + buffer_size/2, 
                test_buffer, 
                buffer_size/2, 
                ALIGN_CACHE
            );
        } else if (operation_type == ZERO) {
            ispc_simd_memory_zero(test_buffer, buffer_size, ALIGN_CACHE);
        } else if (operation_type == SET) {
            ispc_simd_memory_set(test_buffer, 0xAA, buffer_size, ALIGN_CACHE);
        }
    }
    
    uniform uint64 end_time = 0; // Would use system timer in real implementation
    uniform float elapsed_ms = (float)(end_time - start_time) / 1000.0f;
    
    // Calculate bandwidth in GB/s
    uniform float bytes_transferred = (float)(buffer_size * iterations);
    uniform float bandwidth_gbs = (bytes_transferred / (1024.0f * 1024.0f * 1024.0f)) / (elapsed_ms / 1000.0f);
    
    return bandwidth_gbs;
}