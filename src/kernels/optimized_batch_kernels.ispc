// Ultra-optimized ISPC kernels with minimized function call overhead
// Strategies: Inlining, batching, combined operations, template specialization
// Target: Eliminate 90% of function call overhead for 10-50x total speedup

// ============================================================================
// INLINE HELPER FUNCTIONS (zero call overhead)
// ============================================================================

// Inline fingerprint similarity computation
inline float compute_fingerprint_similarity_inline(uint64 fp_a_low, uint64 fp_a_high, uint64 fp_b_low, uint64 fp_b_high) {
    uint64 diff_low = fp_a_low ^ fp_b_low;
    uint64 diff_high = fp_a_high ^ fp_b_high;
    int hamming_distance = popcnt(diff_low) + popcnt(diff_high);
    return 1.0f - ((float)hamming_distance / 128.0f);
}

// Inline One Euro Filter step
inline float one_euro_filter_step_inline(float raw_value, float dt, float x_prev, float dx_prev, float min_cutoff, float beta) {
    // Compute derivative
    float dx = (raw_value - x_prev) / dt;
    
    // Low-pass filter derivative
    float tau_d = 1.0f / (2.0f * 3.14159265359f * 1.0f); // Fixed derivative cutoff
    float alpha_d = 1.0f / (1.0f + tau_d / dt);
    float dx_filtered = alpha_d * dx + (1.0f - alpha_d) * dx_prev;
    
    // Adaptive cutoff
    float cutoff = min_cutoff + beta * abs(dx_filtered);
    float tau = 1.0f / (2.0f * 3.14159265359f * cutoff);
    float alpha = 1.0f / (1.0f + tau / dt);
    
    return alpha * raw_value + (1.0f - alpha) * x_prev;
}

// ============================================================================
// MEGA-BATCH OPERATIONS (single call processes everything)
// ============================================================================

// Combined fingerprint similarity + classification + scoring in one kernel
export void ispc_fingerprint_mega_batch(
    uniform uint64 fingerprints_a_low[],
    uniform uint64 fingerprints_a_high[],
    uniform uint64 fingerprints_b_low[],
    uniform uint64 fingerprints_b_high[],
    uniform float task_priorities[],
    uniform float similarities[],
    uniform float compatibility_scores[],
    uniform int classifications[],
    uniform uint32 hashes[],
    uniform int count
) {
    // Single foreach loop processes ALL operations
    foreach (i = 0 ... count) {
        uint64 fp_a_low = fingerprints_a_low[i];
        uint64 fp_a_high = fingerprints_a_high[i];
        uint64 fp_b_low = fingerprints_b_low[i];
        uint64 fp_b_high = fingerprints_b_high[i];
        
        // 1. Compute similarity (inlined)
        float similarity = compute_fingerprint_similarity_inline(fp_a_low, fp_a_high, fp_b_low, fp_b_high);
        similarities[i] = similarity;
        
        // 2. Compute compatibility score
        float priority = task_priorities[i];
        compatibility_scores[i] = similarity * priority;
        
        // 3. Classify based on bit patterns
        int total_popcount = popcnt(fp_a_low) + popcnt(fp_a_high);
        int classification;
        if (total_popcount < 32) classification = 0;
        else if (total_popcount < 64) classification = 1;
        else if (total_popcount < 96) classification = 2;
        else classification = 3;
        classifications[i] = classification;
        
        // 4. Compute hash
        uint32 hash = (uint32)(fp_a_low ^ (fp_a_low >> 32)) ^ 
                     (uint32)(fp_a_high ^ (fp_a_high >> 32));
        hash ^= hash >> 16;
        hash *= 0x85ebca6b;
        hash ^= hash >> 13;
        hash *= 0xc2b2ae35;
        hash ^= hash >> 16;
        hashes[i] = hash;
    }
}

// Combined prediction pipeline: One Euro Filter + confidence + scoring
export void ispc_prediction_mega_batch(
    uniform float raw_values[],
    uniform float timestamps[],
    uniform float filtered_values[],
    uniform float confidence_scores[],
    uniform float prediction_scores[],
    uniform float worker_loads[],
    uniform float numa_distances[],
    uniform int count
) {
    uniform float min_cutoff = 1.0f;
    uniform float beta = 0.1f;
    uniform int worker_count = count; // Assume worker arrays match count
    
    foreach (i = 0 ... count) {
        float raw_value = raw_values[i];
        float timestamp = timestamps[i];
        
        // 1. Simplified One Euro Filter (stateless per iteration)
        float filtered;
        if (i == 0) {
            filtered = raw_value;
        } else {
            float dt = max(timestamp - timestamps[i-1], 0.001f);
            float alpha = 0.1f; // Simplified constant alpha
            filtered = alpha * raw_value + (1.0f - alpha) * filtered_values[i-1];
        }
        filtered_values[i] = filtered;
        
        // 2. Compute prediction confidence
        float error = abs(filtered - raw_value);
        float magnitude = max(abs(raw_value), 1.0f);
        float normalized_error = error / magnitude;
        float confidence = exp(-2.0f * normalized_error);
        confidence_scores[i] = confidence;
        
        // 3. Multi-factor prediction scoring (use modulo directly in array access)
        float load = worker_loads[i % worker_count];
        float numa_dist = numa_distances[i % worker_count];
        
        float time_score = 1.0f / (1.0f + filtered * 0.001f);
        float confidence_score = confidence;
        float load_score = 1.0f - clamp(load, 0.0f, 1.0f);
        float numa_score = 1.0f / (1.0f + numa_dist * 0.5f);
        
        float combined_score = 0.3f * time_score + 0.3f * confidence_score + 
                              0.25f * load_score + 0.15f * numa_score;
        prediction_scores[i] = exp(2.0f * combined_score) - 1.0f;
    }
}

// Ultra-optimized similarity matrix with cache-friendly access patterns
export void ispc_similarity_matrix_mega_optimized(
    uniform uint64 fingerprints_low[],
    uniform uint64 fingerprints_high[],
    uniform float similarity_matrix[],
    uniform int count
) {
    // Process matrix in cache-friendly tiles
    uniform int tile_size = 64; // Optimize for cache line
    
    for (uniform int tile_i = 0; tile_i < count; tile_i += tile_size) {
        for (uniform int tile_j = 0; tile_j < count; tile_j += tile_size) {
            uniform int end_i = min(tile_i + tile_size, count);
            uniform int end_j = min(tile_j + tile_size, count);
            
            // Process tile with SPMD parallelism
            foreach (i = tile_i ... end_i) {
                uint64 fp_i_low = fingerprints_low[i];
                uint64 fp_i_high = fingerprints_high[i];
                
                for (uniform int j = tile_j; j < end_j; j++) {
                    if (i == j) {
                        similarity_matrix[i * count + j] = 1.0f;
                        continue;
                    }
                    
                    uniform uint64 fp_j_low = fingerprints_low[j];
                    uniform uint64 fp_j_high = fingerprints_high[j];
                    
                    float similarity = compute_fingerprint_similarity_inline(fp_i_low, fp_i_high, fp_j_low, fp_j_high);
                    similarity_matrix[i * count + j] = similarity;
                }
            }
        }
    }
}

// ============================================================================
// TEMPLATE-STYLE GENERIC KERNELS (reduce function proliferation)
// ============================================================================

// Generic data processing kernel - processes any float operation
export void ispc_generic_float_transform(
    uniform float input_data[],
    uniform float output_data[],
    uniform int operation_type,
    uniform float param1,
    uniform float param2,
    uniform int count
) {
    foreach (i = 0 ... count) {
        float value = input_data[i];
        float result;
        
        // Branch-free operation selection using uniform branching
        if (operation_type == 0) {
            // Sigmoid: 1 / (1 + exp(-param1 * (x - param2)))
            result = 1.0f / (1.0f + exp(-param1 * (value - param2)));
        } else if (operation_type == 1) {
            // Exponential scaling: exp(param1 * x) - param2
            result = exp(param1 * value) - param2;
        } else if (operation_type == 2) {
            // Power law: param1 * pow(x, param2)
            result = param1 * pow(value, param2);
        } else if (operation_type == 3) {
            // Smooth step: smoothstep(param1, param2, x)
            float t = clamp((value - param1) / (param2 - param1), 0.0f, 1.0f);
            result = t * t * (3.0f - 2.0f * t);
        } else {
            // Default: linear transform
            result = param1 * value + param2;
        }
        
        output_data[i] = result;
    }
}

// Generic reduction operations (sum, max, min, etc.)
export uniform float ispc_generic_reduction(
    uniform float data[],
    uniform int operation_type,
    uniform int count
) {
    // Initialize accumulator
    varying float result;
    if (operation_type == 0) result = 0.0f;      // SUM
    else if (operation_type == 1) result = -1e9f; // MAX
    else if (operation_type == 2) result = 1e9f;  // MIN
    else result = 1.0f;                           // PRODUCT
    
    foreach (i = 0 ... count) {
        float value = data[i];
        
        if (operation_type == 0) {
            result += value;                      // SUM
        } else if (operation_type == 1) {
            result = max(result, value);          // MAX
        } else if (operation_type == 2) {
            result = min(result, value);          // MIN
        } else {
            result *= value;                      // PRODUCT
        }
    }
    
    // Use ISPC reduction intrinsics for final result
    if (operation_type == 0) return reduce_add(result);
    else if (operation_type == 1) return reduce_max(result);
    else if (operation_type == 2) return reduce_min(result);
    else {
        // Manual product reduction since reduce_mul doesn't exist
        uniform float product_result = 1.0f;
        for (uniform int i = 0; i < count; i++) {
            product_result *= data[i];
        }
        return product_result;
    }
}

// ============================================================================
// STREAMING OPERATIONS (overlap computation and memory access)
// ============================================================================

// Streaming fingerprint processor with prefetching hints
export void ispc_streaming_fingerprint_processor(
    uniform uint64 input_stream_low[],
    uniform uint64 input_stream_high[],
    uniform float output_stream[],
    uniform int stream_length,
    uniform int window_size
) {
    // Process data in streaming windows to improve cache efficiency
    for (uniform int window_start = 0; window_start < stream_length; window_start += window_size) {
        uniform int window_end = min(window_start + window_size, stream_length);
        
        // Prefetch next window while processing current
        prefetch_l1(&input_stream_low[window_end]);
        prefetch_l1(&input_stream_high[window_end]);
        
        foreach (i = window_start ... window_end) {
            uint64 fp_low = input_stream_low[i];
            uint64 fp_high = input_stream_high[i];
            
            // Example processing: compute complexity score
            int popcount = popcnt(fp_low) + popcnt(fp_high);
            float complexity = (float)popcount / 128.0f;
            
            // Apply non-linear transformation
            output_stream[i] = 1.0f / (1.0f + exp(-6.0f * (complexity - 0.5f)));
        }
    }
}