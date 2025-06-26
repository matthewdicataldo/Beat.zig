// ISPC Cleanup Stubs - Provides no-op implementations for builds without full ISPC runtime
// This allows the core library to safely call ISPC cleanup functions even when
// the comprehensive ISPC runtime isn't available (e.g., in cross-library benchmarks)

// Export stub functions that match the ISPC cleanup coordinator expectations
export fn ispc_free_prediction_caches() void {
    // No-op: safe fallback when full ISPC runtime unavailable
}

export fn ispc_free_one_euro_filter_pools() void {
    // No-op: safe fallback when full ISPC runtime unavailable
}

export fn ispc_free_multi_factor_confidence_state() void {
    // No-op: safe fallback when full ISPC runtime unavailable
}

export fn ispc_free_simd_capability_cache() void {
    // No-op: safe fallback when full ISPC runtime unavailable
}

export fn ispc_free_vectorized_queue_state() void {
    // No-op: safe fallback when full ISPC runtime unavailable
}

export fn ispc_free_alignment_buffers() void {
    // No-op: safe fallback when full ISPC runtime unavailable
}

export fn ispc_free_batch_fingerprint_state() void {
    // No-op: safe fallback when full ISPC runtime unavailable
}

export fn ispc_free_batch_worker_scoring_state() void {
    // No-op: safe fallback when full ISPC runtime unavailable
}

export fn ispc_free_batch_optimization_buffers() void {
    // No-op: safe fallback when full ISPC runtime unavailable
}

export fn ispc_cleanup_task_parallelism() void {
    // No-op: safe fallback when full ISPC runtime unavailable
}

export fn ispc_free_async_work_queues() void {
    // No-op: safe fallback when full ISPC runtime unavailable
}

export fn ispc_reset_launch_sync_state() void {
    // No-op: safe fallback when full ISPC runtime unavailable
}

export fn ispc_shutdown_runtime() void {
    // No-op: safe fallback when full ISPC runtime unavailable
}

export fn ispc_verify_no_leaks() bool {
    // No-op: always report success when full ISPC runtime unavailable
    return true;
}

export fn ispc_force_garbage_collection() void {
    // No-op: safe fallback when full ISPC runtime unavailable
}

// Additional ISPC functions called from other modules
export fn ispc_cleanup_all_simd_resources() void {
    // No-op: safe fallback when full ISPC runtime unavailable
}

export fn ispc_reset_simd_system() void {
    // No-op: safe fallback when full ISPC runtime unavailable
}

export fn ispc_free_optimized_fingerprint_caches() void {
    // No-op: safe fallback when full ISPC runtime unavailable
}

export fn ispc_free_batch_optimization_state() void {
    // No-op: safe fallback when full ISPC runtime unavailable
}

export fn ispc_free_similarity_matrix_cache() void {
    // No-op: safe fallback when full ISPC runtime unavailable
}

export fn ispc_free_worker_scoring_cache() void {
    // No-op: safe fallback when full ISPC runtime unavailable
}