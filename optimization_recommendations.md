# ISPC Performance Optimization Recommendations

## 🚨 Critical Performance Issues Found

### 1. Modulus Operations with Varying Types

**Problem**: Modulus operations with varying types prevent efficient vectorization
```ispc
// ❌ Current (inefficient)
float load = worker_loads[i % worker_count];
uint32 cache_idx = (uint32)(hash % (uint64)cache_size);
```

**Solution**: Pre-compute indices or use bit masking for power-of-2 sizes
```ispc
// ✅ Optimized approach
uniform int worker_mask = worker_count - 1;  // If worker_count is power of 2
float load = worker_loads[i & worker_mask];

// Or for non-power-of-2, compute outside the loop
uniform int indices[PROGRAM_COUNT];
for (uniform int lane = 0; lane < PROGRAM_COUNT; lane++) {
    indices[lane] = lane % worker_count;
}
float load = worker_loads[indices[programIndex]];
```

### 2. Gather/Scatter Memory Access Patterns

**Problem**: Non-contiguous memory access requires expensive gather/scatter operations
```ispc
// ❌ Current (requires gather)
uint64 fp_a_low = fingerprints_a[i * 2];
uint64 fp_a_high = fingerprints_a[i * 2 + 1];
```

**Solution**: Restructure data layout to enable vectorized loads
```ispc
// ✅ Structure of Arrays (SoA) approach
uint64 fp_low = fingerprints_low[i];    // Contiguous access
uint64 fp_high = fingerprints_high[i];  // Contiguous access

// Or use streaming loads for better cache utilization
uniform uint64 * uniform fp_base = &fingerprints_a[programIndex * 2];
uint64 fp_low = fp_base[0];
uint64 fp_high = fp_base[1];
```

### 3. Conditional Array Access Optimization

**Problem**: Conditional array access breaks vectorization
```ispc
// ❌ Current (breaks SIMD coherence)
float prediction = hit ? cached_predictions[cache_idx] : 1000.0f;
```

**Solution**: Use masked operations or separate code paths
```ispc
// ✅ Optimized with masking
float cached_val = cached_predictions[cache_idx];
float prediction = select(hit, cached_val, 1000.0f);
```

### 4. Matrix Operations with Scatter

**Problem**: Matrix updates require scatter operations
```ispc
// ❌ Current (scatter required)
similarity_matrix[i * count + j] = similarity;
```

**Solution**: Batch matrix operations or use temporary arrays
```ispc
// ✅ Use temporary arrays and bulk copy
uniform float temp_results[CHUNK_SIZE];
temp_results[programIndex] = similarity;
// Bulk copy to final matrix
```

## 🎯 Performance Impact Estimates

| Optimization | Expected Speedup | Implementation Effort |
|--------------|------------------|---------------------|
| Fix modulus operations | 2-3x | Low |
| Optimize memory layout (SoA) | 1.5-2x | Medium |
| Reduce gather/scatter | 1.3-1.8x | Medium |
| Conditional optimization | 1.2-1.5x | Low |

## 🚀 Priority Recommendations

1. **High Priority**: Fix modulus operations in worker selection and cache indexing
2. **Medium Priority**: Restructure fingerprint storage to use SoA layout
3. **Low Priority**: Optimize conditional array access patterns

## 📊 Monitoring

Track ISPC performance improvements using:
- `ispc --emit-asm` to verify vectorization
- Performance counters for gather/scatter operations  
- Benchmark comparisons before/after optimizations