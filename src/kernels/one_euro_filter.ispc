// ISPC-optimized One Euro Filter implementation for Beat.zig predictive scheduling
// Parallel processing of multiple predictions with SPMD efficiency
// Target: 10-15x speedup over scalar implementation (float-heavy math perfect for ISPC)

// Helper function for smoothing factor computation
inline float smoothing_factor(float dt, float cutoff) {
    float tau = 1.0f / (2.0f * 3.14159265359f * cutoff);
    return 1.0f / (1.0f + tau / dt);
}

// One Euro Filter state for SPMD processing
struct OneEuroState {
    float x_prev;           // Previous filtered value
    float dx_prev;          // Previous derivative
    bool initialized;       // Initialization flag
    float min_cutoff;       // Minimum cutoff frequency
    float beta;             // Derivative cutoff coefficient
    float derivate_cutoff;  // Derivative cutoff frequency
    float prev_timestamp;   // Previous timestamp for dt calculation
};

// SPMD One Euro Filter processing for multiple predictions
export void ispc_one_euro_filter_batch(
    uniform float raw_values[],
    uniform float timestamps[],
    uniform OneEuroState states[],
    uniform float filtered_values[],
    uniform int count
) {
    foreach (i = 0 ... count) {
        float raw_value = raw_values[i];
        float timestamp = timestamps[i];
        OneEuroState state = states[i];
        
        if (!state.initialized) {
            // Initialize filter state
            state.x_prev = raw_value;
            state.dx_prev = 0.0f;
            state.prev_timestamp = timestamp;
            state.initialized = true;
            filtered_values[i] = raw_value;
            states[i] = state;
            continue;
        }
        
        // Compute time delta (ensure positive)
        float dt = max(timestamp - state.prev_timestamp, 0.001f);
        
        // Compute derivative
        float dx = (raw_value - state.x_prev) / dt;
        
        // Apply low-pass filter to derivative
        float alpha_d = smoothing_factor(dt, state.derivate_cutoff);
        float dx_filtered = alpha_d * dx + (1.0f - alpha_d) * state.dx_prev;
        
        // Compute adaptive cutoff frequency
        float cutoff = state.min_cutoff + state.beta * abs(dx_filtered);
        
        // Apply low-pass filter to raw value
        float alpha = smoothing_factor(dt, cutoff);
        float filtered = alpha * raw_value + (1.0f - alpha) * state.x_prev;
        
        // Update state
        state.x_prev = filtered;
        state.dx_prev = dx_filtered;
        state.prev_timestamp = timestamp;
        
        // Store results
        filtered_values[i] = filtered;
        states[i] = state;
    }
}

// Vectorized prediction confidence computation
export void ispc_compute_prediction_confidence(
    uniform float predicted_values[],
    uniform float actual_values[],
    uniform float timestamps[],
    uniform float confidence_scores[],
    uniform int count
) {
    foreach (i = 0 ... count) {
        float predicted = predicted_values[i];
        float actual = actual_values[i];
        
        // Compute prediction error
        float error = abs(predicted - actual);
        
        // Normalize error based on value magnitude
        float magnitude = max(abs(actual), 1.0f);
        float normalized_error = error / magnitude;
        
        // Convert to confidence (0.0 = no confidence, 1.0 = perfect confidence)
        float confidence = exp(-2.0f * normalized_error);
        
        // Apply temporal decay for older predictions
        if (i > 0) {
            float time_decay = exp(-0.1f * (timestamps[i] - timestamps[i]));
            confidence *= time_decay;
        }
        
        confidence_scores[i] = clamp(confidence, 0.0f, 1.0f);
    }
}

// Multi-factor prediction scoring for worker selection
export void ispc_compute_prediction_scores(
    uniform float execution_times[],
    uniform float confidence_levels[],
    uniform float worker_loads[],
    uniform float numa_distances[],
    uniform float prediction_scores[],
    uniform int worker_count
) {
    foreach (worker_id = 0 ... worker_count) {
        float exec_time = execution_times[worker_id];
        float confidence = confidence_levels[worker_id];
        float load = worker_loads[worker_id];
        float numa_dist = numa_distances[worker_id];
        
        // Normalize factors
        float time_score = 1.0f / (1.0f + exec_time * 0.001f);  // Lower time = higher score
        float confidence_score = confidence;                      // Direct confidence mapping
        float load_score = 1.0f - clamp(load, 0.0f, 1.0f);      // Lower load = higher score
        float numa_score = 1.0f / (1.0f + numa_dist * 0.5f);    // Lower distance = higher score
        
        // Weighted combination
        float combined_score = 0.3f * time_score +
                              0.3f * confidence_score +
                              0.25f * load_score +
                              0.15f * numa_score;
        
        // Apply exponential scaling for differentiation
        prediction_scores[worker_id] = exp(2.0f * combined_score) - 1.0f;
    }
}

// Advanced prediction validation with temporal analysis
export void ispc_validate_predictions(
    uniform float predicted_times[],
    uniform float actual_times[],
    uniform float validation_weights[],
    uniform float validation_scores[],
    uniform int count
) {
    foreach (i = 0 ... count) {
        float predicted = predicted_times[i];
        float actual = actual_times[i];
        float weight = validation_weights[i];
        
        // Compute relative error
        float relative_error = abs(predicted - actual) / max(actual, 0.001f);
        
        // Score based on logarithmic error reduction
        float error_score = 1.0f / (1.0f + log(1.0f + relative_error));
        
        // Apply confidence weighting
        float weighted_score = error_score * weight;
        
        // Temporal stability bonus (reward consistent predictions)
        float stability_bonus = 1.0f;
        if (i > 0) {
            float prev_error = abs(predicted_times[i-1] - actual_times[i-1]) / max(actual_times[i-1], 0.001f);
            float error_stability = 1.0f - abs(relative_error - prev_error);
            stability_bonus = 1.0f + 0.2f * max(error_stability, 0.0f);
        }
        
        validation_scores[i] = weighted_score * stability_bonus;
    }
}

// Real-time filter parameter adaptation
export void ispc_adapt_filter_parameters(
    uniform float recent_errors[],
    uniform float system_loads[],
    uniform OneEuroState states[],
    uniform int count
) {
    foreach (i = 0 ... count) {
        float error = recent_errors[i];
        float load = system_loads[i];
        OneEuroState state = states[i];
        
        // Adapt min_cutoff based on recent prediction errors
        if (error > 0.1f) {
            // High error: increase responsiveness
            state.min_cutoff = min(state.min_cutoff * 1.1f, 2.0f);
        } else if (error < 0.05f) {
            // Low error: increase smoothing
            state.min_cutoff = max(state.min_cutoff * 0.95f, 0.5f);
        }
        
        // Adapt beta based on system load
        if (load > 0.8f) {
            // High load: reduce sensitivity to changes
            state.beta = max(state.beta * 0.9f, 0.1f);
        } else if (load < 0.3f) {
            // Low load: increase sensitivity
            state.beta = min(state.beta * 1.05f, 0.5f);
        }
        
        states[i] = state;
    }
}