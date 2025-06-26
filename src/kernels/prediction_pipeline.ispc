// Beat.zig Prediction Pipeline ISPC Kernels
// Comprehensive SPMD acceleration for prediction system with transparent API integration
// Optimized for maximum performance while maintaining full compatibility

// ============================================================================
// MULTI-FACTOR CONFIDENCE COMPUTATION
// ============================================================================

struct MultiFactorConfidence {
    float sample_size_confidence;
    float accuracy_confidence;
    float temporal_confidence;
    float variance_confidence;
    float overall_confidence;
};

// Batch multi-factor confidence computation with vectorized exponential operations
export void ispc_compute_multi_factor_confidence_batch(
    uniform uint32 execution_counts[],
    uniform float accuracy_scores[],
    uniform float temporal_scores[],
    uniform float variance_scores[],
    uniform MultiFactorConfidence confidence_results[],
    uniform int count
) {
    // Confidence calculation parameters (matching Beat.zig's intelligent_decision.zig)
    uniform float sample_scale = 20.0f;
    uniform float accuracy_threshold = 0.8f;
    uniform float temporal_decay = 0.1f;
    uniform float variance_penalty = 2.0f;
    
    foreach (i = 0 ... count) {
        float exec_count = (float)execution_counts[i];
        float accuracy = accuracy_scores[i];
        float temporal = temporal_scores[i];
        float variance = variance_scores[i];
        
        // Sample size confidence (exponential growth with saturation)
        float sample_confidence = 1.0f - exp(-exec_count / sample_scale);
        
        // Accuracy confidence (sigmoid around threshold)
        float accuracy_diff = accuracy - accuracy_threshold;
        float accuracy_confidence = 1.0f / (1.0f + exp(-accuracy_diff * 10.0f));
        
        // Temporal confidence (decay over time)
        float temporal_confidence = exp(-temporal * temporal_decay);
        
        // Variance confidence (penalty for high variance)
        float variance_confidence = exp(-variance * variance_penalty);
        
        // Overall confidence (geometric mean for conservative estimate)
        float overall = pow(
            sample_confidence * accuracy_confidence * temporal_confidence * variance_confidence,
            0.25f
        );
        
        // Write results with reduced scatter operations
        // Cache struct components to minimize memory traffic
        MultiFactorConfidence result;
        result.sample_size_confidence = sample_confidence;
        result.accuracy_confidence = accuracy_confidence;
        result.temporal_confidence = temporal_confidence;
        result.variance_confidence = variance_confidence;
        result.overall_confidence = overall;
        
        confidence_results[i] = result;  // Single struct write instead of 5 separate writes
    }
}

// ============================================================================
// WORKER SELECTION SCORING
// ============================================================================

// Vectorized worker scoring with multi-criteria optimization
export void ispc_score_workers_batch(
    uniform float worker_loads[],
    uniform float numa_distances[],
    uniform float prediction_accuracies[],
    uniform float worker_scores[],
    uniform int worker_count,
    uniform int task_numa_preference
) {
    // Scoring weights (matching Beat.zig's advanced_worker_selection.zig)
    uniform float load_weight = 0.4f;
    uniform float numa_weight = 0.3f;
    uniform float accuracy_weight = 0.3f;
    
    foreach (worker_id = 0 ... worker_count) {
        float load = worker_loads[worker_id];
        float numa_dist = numa_distances[worker_id];
        float accuracy = prediction_accuracies[worker_id];
        
        // Load score (lower load = higher score)
        float load_score = 1.0f - clamp(load, 0.0f, 1.0f);
        
        // NUMA score (closer = higher score, ignore if no preference)
        float numa_score = 1.0f;
        if (task_numa_preference >= 0) {
            numa_score = 1.0f - clamp(numa_dist, 0.0f, 1.0f);
        }
        
        // Accuracy score (higher accuracy = higher score)
        float accuracy_score = clamp(accuracy, 0.0f, 1.0f);
        
        // Weighted combination
        float final_score = (load_score * load_weight) + 
                           (numa_score * numa_weight) + 
                           (accuracy_score * accuracy_weight);
        
        worker_scores[worker_id] = final_score;
    }
}

// ============================================================================
// ENHANCED ONE EURO FILTER PROCESSING
// ============================================================================

struct OneEuroState {
    float x_prev;
    float dx_prev;
    float t_prev;
    bool initialized;
};

// Batch One Euro Filter processing with optimized floating-point operations
export void ispc_process_one_euro_filter_batch(
    uniform float measurements[],
    uniform uint64 timestamps[],
    uniform OneEuroState states[],
    uniform float results[],
    uniform int count,
    uniform float dt_scale,
    uniform float beta,
    uniform float fc_min
) {
    foreach (i = 0 ... count) {
        float measurement = measurements[i];
        uint64 timestamp = timestamps[i];
        OneEuroState state = states[i];
        
        float result;
        
        if (!state.initialized) {
            // First measurement - no filtering
            result = measurement;
            state.x_prev = measurement;
            state.dx_prev = 0.0f;
            state.t_prev = (float)timestamp * dt_scale;
            state.initialized = true;
        } else {
            // One Euro Filter algorithm
            float t_current = (float)timestamp * dt_scale;
            float dt = t_current - state.t_prev;
            
            // Prevent division by zero and handle negative time deltas
            dt = max(dt, 1e-6f);
            
            // Velocity estimation
            float dx = (measurement - state.x_prev) / dt;
            
            // Adaptive cutoff frequency based on velocity
            float fc = fc_min + beta * abs(dx);
            
            // Low-pass filter coefficients
            float alpha = 1.0f / (1.0f + (1.0f / (2.0f * 3.14159f * fc * dt)));
            
            // Apply filtering
            float dx_filtered = alpha * dx + (1.0f - alpha) * state.dx_prev;
            result = alpha * measurement + (1.0f - alpha) * state.x_prev;
            
            // Update state
            state.x_prev = result;
            state.dx_prev = dx_filtered;
            state.t_prev = t_current;
        }
        
        // Write back results
        results[i] = result;
        states[i] = state;
    }
}

// ============================================================================
// PREDICTION REGISTRY ACCELERATION
// ============================================================================

// Vectorized prediction lookup with interpolation
export void ispc_lookup_predictions_batch(
    uniform uint64 fingerprint_hashes[],
    uniform float cached_predictions[],
    uniform float confidence_scores[],
    uniform bool cache_hits[],
    uniform float prediction_results[],
    uniform float result_confidences[],
    uniform int count,
    uniform int cache_size
) {
    // Optimize cache indexing - use bit masking for power-of-2 cache sizes
    uniform bool is_power_of_2 = (cache_size & (cache_size - 1)) == 0;
    uniform uint32 cache_mask = cache_size - 1;
    
    foreach (i = 0 ... count) {
        uint64 hash = fingerprint_hashes[i];
        
        // Use bit masking for power-of-2 sizes, fallback to modulo for others
        uint32 cache_idx = (uint32)(hash & (uint64)cache_mask);
        if (!is_power_of_2) {
            // Use uniform modulo to avoid SIMD performance penalty
            uniform uint32 uniform_hash = extract(hash, 0);
            cache_idx = uniform_hash % cache_size;
        }
        
        bool hit = cache_hits[cache_idx];
        // Use ISPC select() for better vectorization
        float cached_pred = cached_predictions[cache_idx];
        float cached_conf = confidence_scores[cache_idx];
        float prediction = select(hit, cached_pred, 1000.0f);
        float confidence = select(hit, cached_conf, 0.1f);
        
        prediction_results[i] = prediction;
        result_confidences[i] = confidence;
    }
}

// ============================================================================
// TASK FINGERPRINT ENHANCEMENT
// ============================================================================

// SPMD fingerprint generation with parallel feature extraction
export void ispc_generate_fingerprints_batch(
    uniform uint32 task_sizes[],
    uniform uint32 memory_accesses[],
    uniform uint32 branch_counts[],
    uniform float computation_ratios[],
    uniform uint64 fingerprint_results[],
    uniform int count
) {
    foreach (i = 0 ... count) {
        uint32 size = task_sizes[i];
        uint32 memory = memory_accesses[i];
        uint32 branches = branch_counts[i];
        float comp_ratio = computation_ratios[i];
        
        // Feature extraction (matching Beat.zig's fingerprint.zig)
        uint64 size_feature = (uint64)size & 0xFFFF;
        uint64 memory_feature = ((uint64)memory & 0xFFFF) << 16;
        uint64 branch_feature = ((uint64)branches & 0xFFFF) << 32;
        uint64 comp_feature = ((uint64)(comp_ratio * 65535.0f) & 0xFFFF) << 48;
        
        // Combine features into fingerprint
        uint64 fingerprint = size_feature | memory_feature | branch_feature | comp_feature;
        
        fingerprint_results[i] = fingerprint;
    }
}

// ============================================================================
// PREDICTIVE WORK ESTIMATION
// ============================================================================

struct PredictiveWorkEstimate {
    float estimated_cycles;
    float confidence;
    float expected_duration_ns;
    bool should_promote;
};

// Batch predictive work estimation with confidence calculation
export void ispc_estimate_work_batch(
    uniform uint64 fingerprints[],
    uniform float base_predictions[],
    uniform float confidence_scores[],
    uniform float workload_factors[],
    uniform PredictiveWorkEstimate work_estimates[],
    uniform int count,
    uniform float promotion_threshold
) {
    foreach (i = 0 ... count) {
        uint64 fingerprint = fingerprints[i];
        float base_prediction = base_predictions[i];
        float confidence = confidence_scores[i];
        float workload_factor = workload_factors[i];
        
        // Adjust prediction based on current workload
        float estimated_cycles = base_prediction * workload_factor;
        
        // Expected duration (assuming 3GHz CPU)
        float expected_duration = estimated_cycles / 3000000000.0f * 1000000000.0f; // Convert to nanoseconds
        
        // Promotion decision based on estimated work
        bool should_promote = estimated_cycles > promotion_threshold && confidence > 0.5f;
        
        // Write results with reduced scatter operations
        PredictiveWorkEstimate estimate;
        estimate.estimated_cycles = estimated_cycles;
        estimate.confidence = confidence;
        estimate.expected_duration_ns = expected_duration;
        estimate.should_promote = should_promote;
        
        work_estimates[i] = estimate;  // Single struct write instead of 4 separate writes
    }
}

// ============================================================================
// SCHEDULING DECISION OPTIMIZATION
// ============================================================================

struct SchedulingDecision {
    int selected_worker;
    float decision_confidence;
    float expected_completion_time;
    bool use_work_stealing;
};

// Comprehensive scheduling decision with multi-criteria optimization
export void ispc_make_scheduling_decisions_batch(
    uniform float worker_scores[],
    uniform float worker_loads[],
    uniform float task_estimates[],
    uniform float confidence_scores[],
    uniform SchedulingDecision scheduling_decisions[],
    uniform int task_count,
    uniform int worker_count,
    uniform float load_balance_threshold
) {
    foreach (task_id = 0 ... task_count) {
        float task_estimate = task_estimates[task_id];
        float task_confidence = confidence_scores[task_id];
        
        // Find best worker for this task
        int best_worker = 0;
        float best_score = -1.0f;
        float best_load = 1.0f;
        
        for (uniform int worker_id = 0; worker_id < worker_count; worker_id++) {
            float worker_score = worker_scores[worker_id * task_count + task_id];
            float worker_load = worker_loads[worker_id];
            
            // Combined score: worker capability + load balancing
            float combined_score = worker_score * (1.0f - worker_load);
            
            if (combined_score > best_score) {
                best_score = combined_score;
                best_worker = worker_id;
                best_load = worker_load;
            }
        }
        
        // Decision confidence based on score difference and task confidence
        float decision_confidence = best_score * task_confidence;
        
        // Expected completion time
        float completion_time = task_estimate / (1.0f - best_load + 0.1f); // Avoid division by zero
        
        // Work stealing recommendation
        bool use_work_stealing = best_load > load_balance_threshold || task_confidence < 0.3f;
        
        // Write results
        // Write scheduling decision with reduced scatter operations
        SchedulingDecision decision;
        decision.selected_worker = best_worker;
        decision.decision_confidence = decision_confidence;
        decision.expected_completion_time = completion_time;
        decision.use_work_stealing = use_work_stealing;
        
        scheduling_decisions[task_id] = decision;  // Single struct write instead of 4 separate writes
    }
}

// ============================================================================
// PERFORMANCE ANALYTICS
// ============================================================================

// Real-time performance metrics computation
export void ispc_compute_performance_metrics(
    uniform float execution_times[],
    uniform float predicted_times[],
    uniform float confidence_scores[],
    uniform float accuracy_results[],
    uniform float mean_error_results[],
    uniform float confidence_calibration[],
    uniform int count
) {
    // Parallel computation of prediction accuracy metrics
    varying float total_error = 0.0f;
    varying float total_absolute_error = 0.0f;
    varying float total_confidence_weighted_error = 0.0f;
    varying float total_confidence = 0.0f;
    
    foreach (i = 0 ... count) {
        float actual = execution_times[i];
        float predicted = predicted_times[i];
        float confidence = confidence_scores[i];
        
        float error = actual - predicted;
        float abs_error = abs(error);
        float relative_error = abs_error / max(actual, 1.0f);
        
        // Accumulate metrics
        total_error += error;
        total_absolute_error += abs_error;
        total_confidence_weighted_error += abs_error * confidence;
        total_confidence += confidence;
        
        // Individual accuracy
        accuracy_results[i] = 1.0f - min(relative_error, 1.0f);
    }
    
    // Reduce metrics across SIMD lanes
    uniform float mean_error = reduce_add(total_error) / (float)count;
    uniform float mean_absolute_error = reduce_add(total_absolute_error) / (float)count;
    uniform float weighted_error = reduce_add(total_confidence_weighted_error) / max(reduce_add(total_confidence), 1.0f);
    
    // Write aggregated results
    mean_error_results[0] = mean_error;
    mean_error_results[1] = mean_absolute_error;
    confidence_calibration[0] = weighted_error;
}