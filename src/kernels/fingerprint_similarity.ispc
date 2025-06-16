// ISPC kernel for fingerprint similarity computation
// Optimizes Beat.zig's task fingerprinting with SPMD parallelism
// Target: 3-6x speedup over scalar implementation

// Compute Hamming distance between 128-bit fingerprints using ISPC SPMD
export void ispc_compute_fingerprint_similarity(
    uniform uint64 fingerprints_a[],
    uniform uint64 fingerprints_b[], 
    uniform float results[],
    uniform int count
) {
    // Process fingerprints in parallel across SIMD lanes
    foreach (i = 0 ... count) {
        // Load 128-bit fingerprints as two 64-bit halves
        uint64 fp_a_low = fingerprints_a[i * 2];
        uint64 fp_a_high = fingerprints_a[i * 2 + 1];
        uint64 fp_b_low = fingerprints_b[i * 2];
        uint64 fp_b_high = fingerprints_b[i * 2 + 1];
        
        // Compute XOR to find differing bits
        uint64 diff_low = fp_a_low ^ fp_b_low;
        uint64 diff_high = fp_a_high ^ fp_b_high;
        
        // Count population (number of 1 bits) using ISPC intrinsics
        int hamming_distance = popcnt(diff_low) + popcnt(diff_high);
        
        // Convert to similarity score (higher = more similar)
        // Similarity = 1.0 - (hamming_distance / 128.0)
        float similarity = 1.0f - ((float)hamming_distance / 128.0f);
        
        results[i] = similarity;
    }
}

// Compute similarity matrix for batch formation optimization
export void ispc_compute_similarity_matrix(
    uniform uint64 fingerprints[],
    uniform float similarity_matrix[],
    uniform int count
) {
    // Compute upper triangular similarity matrix with uniform loops
    for (uniform int i = 0; i < count; i++) {
        for (uniform int j = i + 1; j < count; j++) {
            // Load fingerprints as uniform values
            uniform uint64 fp_i_low = fingerprints[i * 2];
            uniform uint64 fp_i_high = fingerprints[i * 2 + 1];
            uniform uint64 fp_j_low = fingerprints[j * 2];
            uniform uint64 fp_j_high = fingerprints[j * 2 + 1];
            
            // Compute similarity
            uniform uint64 diff_low = fp_i_low ^ fp_j_low;
            uniform uint64 diff_high = fp_i_high ^ fp_j_high;
            uniform int hamming_distance = popcnt(diff_low) + popcnt(diff_high);
            uniform float similarity = 1.0f - ((float)hamming_distance / 128.0f);
            
            // Store in matrix (symmetric)
            similarity_matrix[i * count + j] = similarity;
            similarity_matrix[j * count + i] = similarity;
        }
        
        // Diagonal elements (self-similarity) are always 1.0
        similarity_matrix[i * count + i] = 1.0f;
    }
}

// Fast fingerprint hash computation for performance tracking
export void ispc_compute_fingerprint_hashes(
    uniform uint64 fingerprints[],
    uniform uint32 hashes[],
    uniform int count
) {
    foreach (i = 0 ... count) {
        uint64 fp_low = fingerprints[i * 2];
        uint64 fp_high = fingerprints[i * 2 + 1];
        
        // Simple but effective hash combining both halves
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