// Optimized ISPC kernel for fingerprint similarity computation using Structure of Arrays (SoA)
// Eliminates gather operations for maximum SIMD efficiency
// Target: 4-8x speedup over AoS layout + 3-6x speedup over scalar = 12-48x total improvement

// SoA fingerprint similarity computation - SIMD-optimized memory access
export void ispc_compute_fingerprint_similarity_soa(
    uniform uint64 fingerprints_a_low[],
    uniform uint64 fingerprints_a_high[],
    uniform uint64 fingerprints_b_low[],
    uniform uint64 fingerprints_b_high[],
    uniform float results[],
    uniform int count
) {
    // Process fingerprints in parallel with optimal memory access
    foreach (i = 0 ... count) {
        // Load fingerprint components with vectorized access (no gather!)
        uint64 fp_a_low = fingerprints_a_low[i];
        uint64 fp_a_high = fingerprints_a_high[i];
        uint64 fp_b_low = fingerprints_b_low[i];
        uint64 fp_b_high = fingerprints_b_high[i];
        
        // Compute XOR to find differing bits
        uint64 diff_low = fp_a_low ^ fp_b_low;
        uint64 diff_high = fp_a_high ^ fp_b_high;
        
        // Count population (number of 1 bits) using ISPC intrinsics
        int hamming_distance = popcnt(diff_low) + popcnt(diff_high);
        
        // Convert to similarity score (higher = more similar)
        float similarity = 1.0f - ((float)hamming_distance / 128.0f);
        
        results[i] = similarity;
    }
}

// Vectorized similarity matrix computation with SoA layout
export void ispc_compute_similarity_matrix_soa(
    uniform uint64 fingerprints_low[],
    uniform uint64 fingerprints_high[],
    uniform float similarity_matrix[],
    uniform int count
) {
    // Compute similarity matrix with optimal memory access patterns
    foreach (i = 0 ... count) {
        uint64 fp_i_low = fingerprints_low[i];
        uint64 fp_i_high = fingerprints_high[i];
        
        // Vectorize comparison against all other fingerprints
        for (uniform int j = 0; j < count; j++) {
            if (i == j) {
                similarity_matrix[i * count + j] = 1.0f; // Self-similarity
                continue;
            }
            
            // Load comparison fingerprint (vectorized access)
            uniform uint64 fp_j_low = fingerprints_low[j];
            uniform uint64 fp_j_high = fingerprints_high[j];
            
            // Compute similarity
            uint64 diff_low = fp_i_low ^ fp_j_low;
            uint64 diff_high = fp_i_high ^ fp_j_high;
            int hamming_distance = popcnt(diff_low) + popcnt(diff_high);
            float similarity = 1.0f - ((float)hamming_distance / 128.0f);
            
            similarity_matrix[i * count + j] = similarity;
        }
    }
}

// Batch similarity scoring with SoA layout for optimal cache usage
export void ispc_batch_similarity_scoring_soa(
    uniform uint64 fingerprints_low[],
    uniform uint64 fingerprints_high[],
    uniform float task_priorities[],
    uniform float compatibility_scores[],
    uniform int count
) {
    // Compute compatibility scores using reference fingerprint (first element)
    uniform uint64 ref_low = fingerprints_low[0];
    uniform uint64 ref_high = fingerprints_high[0];
    
    foreach (i = 1 ... count) {
        uint64 fp_low = fingerprints_low[i];
        uint64 fp_high = fingerprints_high[i];
        
        // Compute similarity to reference
        uint64 diff_low = fp_low ^ ref_low;
        uint64 diff_high = fp_high ^ ref_high;
        int hamming_distance = popcnt(diff_low) + popcnt(diff_high);
        float similarity = 1.0f - ((float)hamming_distance / 128.0f);
        
        // Weight by task priority
        float priority = task_priorities[i];
        compatibility_scores[i] = similarity * priority;
    }
    
    // Reference task always has perfect compatibility
    compatibility_scores[0] = task_priorities[0];
}

// Fast fingerprint hash computation with SoA layout
export void ispc_compute_fingerprint_hashes_soa(
    uniform uint64 fingerprints_low[],
    uniform uint64 fingerprints_high[],
    uniform uint32 hashes[],
    uniform int count
) {
    foreach (i = 0 ... count) {
        uint64 fp_low = fingerprints_low[i];
        uint64 fp_high = fingerprints_high[i];
        
        // Combine both halves for hash computation
        uint32 hash = (uint32)(fp_low ^ (fp_low >> 32)) ^ 
                     (uint32)(fp_high ^ (fp_high >> 32));
        
        // Additional mixing for better distribution
        hash ^= hash >> 16;
        hash *= 0x85ebca6b;
        hash ^= hash >> 13;
        hash *= 0xc2b2ae35;
        hash ^= hash >> 16;
        
        hashes[i] = hash;
    }
}

// Parallel fingerprint classification for task routing
export void ispc_classify_fingerprints_soa(
    uniform uint64 fingerprints_low[],
    uniform uint64 fingerprints_high[],
    uniform int classifications[],
    uniform int count
) {
    foreach (i = 0 ... count) {
        uint64 fp_low = fingerprints_low[i];
        uint64 fp_high = fingerprints_high[i];
        
        // Classify based on bit patterns (simplified heuristic)
        int low_popcount = popcnt(fp_low);
        int high_popcount = popcnt(fp_high);
        int total_popcount = low_popcount + high_popcount;
        
        int classification;
        if (total_popcount < 32) {
            classification = 0; // LOW_COMPLEXITY
        } else if (total_popcount < 64) {
            classification = 1; // MEDIUM_COMPLEXITY
        } else if (total_popcount < 96) {
            classification = 2; // HIGH_COMPLEXITY
        } else {
            classification = 3; // VERY_HIGH_COMPLEXITY
        }
        
        classifications[i] = classification;
    }
}