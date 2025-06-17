// ISPC kernel for SIMD capability detection and management
// Replaces Zig SIMDCapability.detect() and related functions
// Target: 3-5x faster capability detection with parallel probing

// SIMD instruction set enumeration (matches Zig enum)
enum SIMDInstructionSet {
    SIMD_NONE = 0,
    SIMD_SSE = 1,
    SIMD_SSE2 = 2,  
    SIMD_SSE3 = 3,
    SIMD_SSE41 = 4,
    SIMD_SSE42 = 5,
    SIMD_AVX = 6,
    SIMD_AVX2 = 7,
    SIMD_AVX512F = 8,
    SIMD_AVX512VL = 9,
    SIMD_NEON = 16,
    SIMD_SVE = 17
};

// SIMD capability structure (matches Zig struct)
struct SIMDCapability {
    uniform int instruction_set;
    uniform int vector_width;
    uniform bool supports_integer;
    uniform bool supports_fma;
    uniform float performance_score;
    uniform int cache_line_size;
};

// Detect SIMD capabilities for current hardware
export void ispc_detect_simd_capabilities(
    uniform SIMDCapability capabilities[],
    uniform int max_capabilities,
    uniform int* uniform detected_count
) {
    // Initialize detected count
    *detected_count = 0;
    
    int cap_index = 0;
    
    // Detect x86_64 SIMD capabilities
    #if TARGET_ARCH == "x86-64" || TARGET_ARCH == "x86"
    
    // SSE detection
    #if TARGET_WIDTH >= 4
    if (cap_index < max_capabilities) {
        capabilities[cap_index].instruction_set = SIMD_SSE;
        capabilities[cap_index].vector_width = 128;
        capabilities[cap_index].supports_integer = false;
        capabilities[cap_index].supports_fma = false;
        capabilities[cap_index].performance_score = 2.0f;
        capabilities[cap_index].cache_line_size = 64;
        cap_index++;
    }
    #endif
    
    // SSE2 detection
    #if TARGET_WIDTH >= 4
    if (cap_index < max_capabilities) {
        capabilities[cap_index].instruction_set = SIMD_SSE2;
        capabilities[cap_index].vector_width = 128;
        capabilities[cap_index].supports_integer = true;
        capabilities[cap_index].supports_fma = false;
        capabilities[cap_index].performance_score = 3.0f;
        capabilities[cap_index].cache_line_size = 64;
        cap_index++;
    }
    #endif
    
    // AVX detection  
    #if TARGET_WIDTH >= 8
    if (cap_index < max_capabilities) {
        capabilities[cap_index].instruction_set = SIMD_AVX;
        capabilities[cap_index].vector_width = 256;
        capabilities[cap_index].supports_integer = false;
        capabilities[cap_index].supports_fma = false;
        capabilities[cap_index].performance_score = 5.0f;
        capabilities[cap_index].cache_line_size = 64;
        cap_index++;
    }
    #endif
    
    // AVX2 detection
    #if TARGET_WIDTH >= 8 && (defined(ISPC_TARGET_AVX2) || defined(ISPC_TARGET_AVX2_I32X8))
    if (cap_index < max_capabilities) {
        capabilities[cap_index].instruction_set = SIMD_AVX2;
        capabilities[cap_index].vector_width = 256;
        capabilities[cap_index].supports_integer = true;
        capabilities[cap_index].supports_fma = true;
        capabilities[cap_index].performance_score = 8.0f;
        capabilities[cap_index].cache_line_size = 64;
        cap_index++;
    }
    #endif
    
    // AVX-512 detection
    #if TARGET_WIDTH >= 16 && (defined(ISPC_TARGET_AVX512KNL) || defined(ISPC_TARGET_AVX512SKX))
    if (cap_index < max_capabilities) {
        capabilities[cap_index].instruction_set = SIMD_AVX512F;
        capabilities[cap_index].vector_width = 512;
        capabilities[cap_index].supports_integer = true;
        capabilities[cap_index].supports_fma = true;
        capabilities[cap_index].performance_score = 15.0f;
        capabilities[cap_index].cache_line_size = 64;
        cap_index++;
    }
    #endif
    
    #endif // x86_64
    
    // Detect ARM SIMD capabilities
    #if TARGET_ARCH == "aarch64" || TARGET_ARCH == "arm"
    
    // NEON detection
    #if TARGET_WIDTH >= 4
    if (cap_index < max_capabilities) {
        capabilities[cap_index].instruction_set = SIMD_NEON;
        capabilities[cap_index].vector_width = 128;
        capabilities[cap_index].supports_integer = true;
        capabilities[cap_index].supports_fma = true;
        capabilities[cap_index].performance_score = 6.0f;
        capabilities[cap_index].cache_line_size = 64;
        cap_index++;
    }
    #endif
    
    // SVE detection (if supported)
    #if defined(ISPC_TARGET_NEON) && TARGET_WIDTH >= 8
    if (cap_index < max_capabilities) {
        capabilities[cap_index].instruction_set = SIMD_SVE;
        capabilities[cap_index].vector_width = 2048; // Scalable up to 2048-bit
        capabilities[cap_index].supports_integer = true;
        capabilities[cap_index].supports_fma = true;
        capabilities[cap_index].performance_score = 12.0f;
        capabilities[cap_index].cache_line_size = 128; // ARM typically has larger cache lines
        cap_index++;
    }
    #endif
    
    #endif // ARM
    
    *detected_count = cap_index;
}

// Get optimal vector length for data type and capability
export uniform int ispc_get_optimal_vector_length(
    uniform int instruction_set,
    uniform int data_type_size,
    uniform int vector_width
) {
    // Calculate optimal vector length based on instruction set and data type
    uniform int elements_per_vector = vector_width / (data_type_size * 8);
    
    // Clamp to reasonable limits based on instruction set
    if (instruction_set <= SIMD_SSE42) {
        // SSE family: max 128-bit
        elements_per_vector = min(elements_per_vector, 128 / (data_type_size * 8));
    } else if (instruction_set <= SIMD_AVX2) {
        // AVX family: max 256-bit  
        elements_per_vector = min(elements_per_vector, 256 / (data_type_size * 8));
    } else if (instruction_set <= SIMD_AVX512VL) {
        // AVX-512 family: max 512-bit
        elements_per_vector = min(elements_per_vector, 512 / (data_type_size * 8));
    }
    
    // Ensure minimum of 1 element
    return max(elements_per_vector, 1);
}

// Check if instruction set supports specific operation
export uniform bool ispc_supports_operation(
    uniform int instruction_set,
    uniform int operation_type
) {
    // Operation types (matches Zig enum)
    uniform int OP_ADD = 0;
    uniform int OP_MUL = 1;
    uniform int OP_FMA = 2;
    uniform int OP_GATHER = 3;
    uniform int OP_SCATTER = 4;
    
    if (operation_type == OP_ADD || operation_type == OP_MUL) {
        // Basic operations supported by all SIMD
        return instruction_set > SIMD_NONE;
    } else if (operation_type == OP_FMA) {
        // FMA requires AVX2+, NEON, or SVE
        return instruction_set >= SIMD_AVX2 || instruction_set == SIMD_NEON || instruction_set == SIMD_SVE;
    } else if (operation_type == OP_GATHER || operation_type == OP_SCATTER) {
        // Gather/scatter requires AVX2+ or advanced ARM
        return instruction_set >= SIMD_AVX2 || instruction_set == SIMD_SVE;
    }
    
    return false;
}

// Parallel capability scoring for worker selection
export void ispc_score_worker_capabilities(
    uniform SIMDCapability worker_capabilities[],
    uniform float required_vector_widths[],
    uniform int data_type_sizes[],
    uniform float capability_scores[],
    uniform int worker_count
) {
    foreach (worker_id = 0 ... worker_count) {
        float score = 0.0f;
        
        // Score based on vector width match
        float width_match = (float)worker_capabilities[worker_id].vector_width / required_vector_widths[worker_id];
        width_match = clamp(width_match, 0.1f, 2.0f); // Penalize severe mismatches
        score += width_match * 0.4f;
        
        // Score based on performance capability
        score += worker_capabilities[worker_id].performance_score * 0.3f;
        
        // Score based on feature support
        if (worker_capabilities[worker_id].supports_fma) {
            score += 2.0f;
        }
        if (worker_capabilities[worker_id].supports_integer) {
            score += 1.0f;
        }
        
        // Bonus for newer instruction sets
        if (worker_capabilities[worker_id].instruction_set >= SIMD_AVX2) {
            score += 3.0f;
        } else if (worker_capabilities[worker_id].instruction_set >= SIMD_AVX) {
            score += 1.5f;
        }
        
        capability_scores[worker_id] = score;
    }
}

// Update worker capabilities in parallel
export void ispc_update_worker_capabilities(
    uniform SIMDCapability worker_capabilities[],
    uniform float performance_measurements[],
    uniform int measurement_counts[],
    uniform int worker_count
) {
    foreach (worker_id = 0 ... worker_count) {
        if (measurement_counts[worker_id] > 0) {
            // Update performance score with exponential moving average
            float alpha = 0.1f; // Learning rate
            float measured_perf = performance_measurements[worker_id] / (float)measurement_counts[worker_id];
            
            worker_capabilities[worker_id].performance_score = 
                alpha * measured_perf + (1.0f - alpha) * worker_capabilities[worker_id].performance_score;
        }
    }
}