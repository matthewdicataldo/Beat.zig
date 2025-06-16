// ISPC kernels for Beat.zig heartbeat scheduling system
// Optimizes parallel worker processing, token accounting, and memory pressure adaptation
// Target: 3-15x speedup for scheduling hot paths

// ============================================================================
// HEARTBEAT WORKER PROCESSING
// ============================================================================

// Parallel promotion decision processing for all workers
export void ispc_process_worker_heartbeats(
    uniform uint64 work_cycles[],
    uniform uint64 overhead_cycles[],
    uniform uint64 promotion_thresholds[],
    uniform uint64 min_work_cycles[],
    uniform bool should_promote[],
    uniform bool needs_reset[],
    uniform int worker_count
) {
    foreach (worker_id = 0 ... worker_count) {
        uint64 work = work_cycles[worker_id];
        uint64 overhead = overhead_cycles[worker_id];
        uint64 threshold = promotion_thresholds[worker_id];
        uint64 min_work = min_work_cycles[worker_id];
        
        // Check promotion criteria
        bool meets_work_threshold = work >= min_work;
        bool beats_overhead_ratio = (overhead == 0) ? true : (work > overhead * threshold);
        bool promote = meets_work_threshold && beats_overhead_ratio;
        
        should_promote[worker_id] = promote;
        needs_reset[worker_id] = promote;
    }
}

// Parallel work:overhead ratio calculation with adaptive thresholds
export void ispc_compute_worker_ratios(
    uniform uint64 work_cycles[],
    uniform uint64 overhead_cycles[],
    uniform float work_ratios[],
    uniform float efficiency_scores[],
    uniform float adaptive_thresholds[],
    uniform int worker_count
) {
    foreach (worker_id = 0 ... worker_count) {
        uint64 work = work_cycles[worker_id];
        uint64 overhead = overhead_cycles[worker_id];
        
        // Compute work:overhead ratio (safe division)
        float ratio = (overhead == 0) ? 1000.0f : (float)work / (float)overhead;
        work_ratios[worker_id] = ratio;
        
        // Efficiency score (0.0 = all overhead, 1.0 = no overhead)
        float total = (float)(work + overhead);
        float efficiency = (total == 0.0f) ? 0.0f : (float)work / total;
        efficiency_scores[worker_id] = efficiency;
        
        // Adaptive threshold based on historical performance
        float base_threshold = 2.0f; // Default 2:1 work:overhead ratio
        float adaptation = 1.0f + 0.5f * max(0.0f, efficiency - 0.5f);
        adaptive_thresholds[worker_id] = base_threshold * adaptation;
    }
}

// ============================================================================
// PREDICTIVE TOKEN ACCOUNTING
// ============================================================================

// Batch update of prediction accuracy with confidence weighting
export void ispc_update_prediction_accuracy(
    uniform float predicted_values[],
    uniform float actual_values[],
    uniform float timestamps[],
    uniform float confidence_weights[],
    uniform float accuracy_scores[],
    uniform float temporal_factors[],
    uniform float smoothed_accuracy[],
    uniform int count
) {
    uniform float alpha = 0.1f; // Exponential smoothing factor
    uniform float decay_constant = 10.0f; // Temporal decay rate (10ms)
    
    foreach (i = 0 ... count) {
        float predicted = predicted_values[i];
        float actual = actual_values[i];
        float confidence = confidence_weights[i];
        
        // Compute relative error (avoid division by zero)
        float magnitude = max(abs(predicted), 1.0f);
        float error = abs(predicted - actual) / magnitude;
        float accuracy = max(0.0f, 1.0f - error);
        
        // Apply confidence weighting
        float weighted_accuracy = accuracy * confidence;
        accuracy_scores[i] = weighted_accuracy;
        
        // Temporal relevance factor
        float time_diff = (i > 0) ? (timestamps[i] - timestamps[i-1]) : 0.0f;
        float temporal_weight = exp(-time_diff / decay_constant);
        temporal_factors[i] = temporal_weight;
        
        // Exponential smoothing for running accuracy
        float prev_smoothed = (i > 0) ? smoothed_accuracy[i-1] : weighted_accuracy;
        smoothed_accuracy[i] = alpha * weighted_accuracy + (1.0f - alpha) * prev_smoothed;
    }
}

// Confidence-weighted token accumulation for predictive accounting
export void ispc_accumulate_predicted_tokens(
    uniform float predicted_cycles[],
    uniform float confidence_scores[],
    uniform float base_costs[],
    uniform float accumulated_tokens[],
    uniform float confidence_weighted_tokens[],
    uniform float uncertainty_penalties[],
    uniform int count
) {
    foreach (i = 0 ... count) {
        float predicted = predicted_cycles[i];
        float confidence = confidence_scores[i];
        float base_cost = base_costs[i];
        
        // Conservative adjustment for low confidence predictions
        float uncertainty = 1.0f - confidence;
        float penalty_factor = 1.0f + uncertainty * 0.5f; // Up to 50% penalty
        float adjusted_prediction = predicted * penalty_factor;
        
        // Token accumulation
        float tokens = max(base_cost, adjusted_prediction);
        accumulated_tokens[i] = tokens;
        
        // Confidence-weighted accumulation
        confidence_weighted_tokens[i] = tokens * confidence;
        uncertainty_penalties[i] = penalty_factor;
    }
}

// ============================================================================
// MEMORY PRESSURE ADAPTATION
// ============================================================================

// Parallel memory pressure adaptation across workers
export void ispc_adapt_memory_pressure(
    uniform float memory_levels[],
    uniform float worker_loads[],
    uniform float numa_distances[],
    uniform float adaptation_factors[],
    uniform float batch_size_limits[],
    uniform float memory_scores[],
    uniform int worker_count
) {
    uniform float pressure_thresholds[5] = { 0.0f, 0.2f, 0.4f, 0.6f, 0.8f };
    uniform float adaptation_rates[5] = { 1.0f, 0.9f, 0.7f, 0.5f, 0.3f };
    
    foreach (worker_id = 0 ... worker_count) {
        float memory_level = memory_levels[worker_id];
        float worker_load = worker_loads[worker_id];
        float numa_distance = numa_distances[worker_id];
        
        // Determine pressure level (0-4)
        int pressure_level = 0;
        for (uniform int level = 0; level < 4; level++) {
            if (memory_level > pressure_thresholds[level + 1]) {
                pressure_level = level + 1;
            }
        }
        
        // Base adaptation factor
        float base_adaptation = adaptation_rates[pressure_level];
        
        // Adjust for worker load (higher load = more conservative)
        float load_factor = 1.0f - 0.3f * worker_load;
        
        // Adjust for NUMA distance (remote memory = more conservative)
        float numa_factor = max(0.5f, 1.0f - numa_distance * 0.2f);
        
        // Combined adaptation factor
        float combined_adaptation = base_adaptation * load_factor * numa_factor;
        adaptation_factors[worker_id] = clamp(combined_adaptation, 0.1f, 1.0f);
        
        // Batch size limits based on memory pressure
        float base_batch_size = 1000.0f;
        batch_size_limits[worker_id] = base_batch_size * combined_adaptation;
        
        // Memory efficiency score
        float memory_efficiency = 1.0f - memory_level;
        memory_scores[worker_id] = memory_efficiency * combined_adaptation;
    }
}

// ============================================================================
// NUMA TOPOLOGY OPTIMIZATION
// ============================================================================

// Optimized NUMA distance matrix computation for topology-aware scheduling
export void ispc_compute_numa_distances(
    uniform int numa_nodes_a[],
    uniform int numa_nodes_b[],
    uniform float base_distances[],
    uniform float memory_bandwidths[],
    uniform float topology_scores[],
    uniform float migration_costs[],
    uniform int pair_count
) {
    foreach (i = 0 ... pair_count) {
        int node_a = numa_nodes_a[i];
        int node_b = numa_nodes_b[i];
        float base_distance = base_distances[i];
        float bandwidth = memory_bandwidths[i];
        
        // Distance-based scoring (0.0 = remote, 1.0 = local)
        float distance_score;
        if (node_a == node_b) {
            distance_score = 1.0f; // Same NUMA node
        } else if (abs(node_a - node_b) == 1) {
            distance_score = 0.7f; // Adjacent NUMA nodes
        } else {
            distance_score = max(0.1f, 1.0f - base_distance * 0.1f); // Remote nodes
        }
        
        // Bandwidth factor (higher bandwidth = better score)
        float bandwidth_factor = clamp(bandwidth / 100.0f, 0.5f, 1.5f);
        
        // Combined topology score
        topology_scores[i] = distance_score * bandwidth_factor;
        
        // Migration cost (inverse of topology score)
        float base_migration_cost = 1000.0f; // Base cost in cycles
        migration_costs[i] = base_migration_cost / max(0.1f, distance_score);
    }
}

// ============================================================================
// LOAD BALANCING OPTIMIZATION
// ============================================================================

// Advanced load balancing with predictive workload distribution
export void ispc_compute_load_balance_targets(
    uniform float current_loads[],
    uniform float predicted_incoming[],
    uniform float worker_capacities[],
    uniform float numa_preferences[],
    uniform float target_loads[],
    uniform float balance_scores[],
    uniform float redistribution_amounts[],
    uniform int worker_count
) {
    // Compute global statistics
    varying float total_current = 0.0f;
    varying float total_predicted = 0.0f;
    varying float total_capacity = 0.0f;
    
    foreach (i = 0 ... worker_count) {
        total_current += current_loads[i];
        total_predicted += predicted_incoming[i];
        total_capacity += worker_capacities[i];
    }
    
    uniform float global_current = reduce_add(total_current);
    uniform float global_predicted = reduce_add(total_predicted);
    uniform float global_capacity = reduce_add(total_capacity);
    
    uniform float total_workload = global_current + global_predicted;
    uniform float avg_capacity = global_capacity / (float)worker_count;
    
    foreach (worker_id = 0 ... worker_count) {
        float current_load = current_loads[worker_id];
        float incoming = predicted_incoming[worker_id];
        float capacity = worker_capacities[worker_id];
        float numa_pref = numa_preferences[worker_id];
        
        // Ideal load based on capacity ratio
        float capacity_ratio = capacity / max(1.0f, avg_capacity);
        float ideal_load = (total_workload / (float)worker_count) * capacity_ratio;
        
        // Adjust for NUMA preferences
        ideal_load *= numa_pref;
        
        target_loads[worker_id] = ideal_load;
        
        // Balance score (1.0 = perfectly balanced)
        float actual_load = current_load + incoming;
        float load_ratio = (ideal_load == 0.0f) ? 0.0f : actual_load / ideal_load;
        balance_scores[worker_id] = 1.0f / (1.0f + abs(load_ratio - 1.0f));
        
        // Redistribution amount (positive = needs more work, negative = overloaded)
        redistribution_amounts[worker_id] = ideal_load - actual_load;
    }
}