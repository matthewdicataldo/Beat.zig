// ISPC kernel for advanced worker selection optimization
// Enhances Beat.zig's 15.3x worker selection improvement with SPMD parallelism
// Target: Additional 3-6x speedup on multi-criteria worker scoring

// Multi-criteria worker scoring using ISPC SPMD
export void ispc_compute_worker_scores(
    uniform float worker_loads[],
    uniform float numa_distances[],
    uniform float cache_affinities[],
    uniform float worker_scores[],
    uniform int worker_count
) {
    // Scoring weights optimized for Beat.zig workloads
    uniform float load_weight = 0.4f;
    uniform float numa_weight = 0.35f;
    uniform float cache_weight = 0.25f;
    
    foreach (worker_id = 0 ... worker_count) {
        float load_score = 1.0f - clamp(worker_loads[worker_id], 0.0f, 1.0f);
        float numa_score = 1.0f - clamp(numa_distances[worker_id], 0.0f, 1.0f);
        float cache_score = clamp(cache_affinities[worker_id], 0.0f, 1.0f);
        
        // Non-linear scaling for better differentiation
        load_score = load_score * load_score; // Quadratic penalty for high load
        numa_score = sqrt(numa_score); // Diminishing returns for NUMA proximity
        
        // Weighted combination
        float combined_score = load_weight * load_score +
                              numa_weight * numa_score +
                              cache_weight * cache_score;
        
        // Apply exponential scaling for worker differentiation
        worker_scores[worker_id] = exp(2.0f * combined_score) - 1.0f;
    }
}

// Advanced topology-aware worker selection
export void ispc_topology_aware_selection(
    uniform float worker_loads[],
    uniform int numa_nodes[],
    uniform int cpu_cores[],
    uniform float task_memory_intensity,
    uniform int preferred_numa_node,
    uniform float topology_scores[],
    uniform int worker_count
) {
    foreach (worker_id = 0 ... worker_count) {
        float base_score = 1.0f - clamp(worker_loads[worker_id], 0.0f, 1.0f);
        
        // NUMA topology scoring
        float numa_score = 1.0f;
        if (numa_nodes[worker_id] == preferred_numa_node) {
            numa_score = 1.0f; // Perfect NUMA match
        } else {
            // Distance-based penalty
            int numa_distance = abs(numa_nodes[worker_id] - preferred_numa_node);
            numa_score = 1.0f / (1.0f + (float)numa_distance * 0.5f);
        }
        
        // CPU core locality (simplified model) - optimized bit testing
        float core_score = 1.0f;
        int core_id = cpu_cores[worker_id];
        if (core_id >= 0) {
            // Prefer even-numbered cores for hyperthreading efficiency
            // Use bit testing instead of modulo for better vectorization
            if ((core_id & 1) == 0) {
                core_score = 1.1f;
            }
        }
        
        // Memory intensity weighting
        float memory_weight = clamp(task_memory_intensity, 0.0f, 1.0f);
        numa_score = (1.0f - memory_weight) + memory_weight * numa_score;
        
        // Combined topology score
        topology_scores[worker_id] = base_score * numa_score * core_score;
    }
}

// Work-stealing victim selection optimization
export uniform int ispc_select_steal_victims(
    uniform float worker_queue_sizes[],
    uniform int numa_distances[],
    uniform float steal_success_rates[],
    uniform int victim_preferences[],
    uniform int worker_count,
    uniform int max_victims
) {
    uniform int victim_count = 0;
    
    // Score all potential victims using uniform loop
    for (uniform int victim_id = 0; victim_id < worker_count; victim_id++) {
        uniform float queue_size = worker_queue_sizes[victim_id];
        uniform float numa_penalty = (float)numa_distances[victim_id] * 0.1f;
        uniform float success_rate = steal_success_rates[victim_id];
        
        // Composite victim score
        uniform float victim_score = queue_size * success_rate * (1.0f - numa_penalty);
        
        // Simple selection: pick victims with above-average scores
        if (victim_score > 0.5f && victim_count < max_victims) {
            victim_preferences[victim_count] = victim_id;
            victim_count++;
        }
    }
    
    return victim_count;
}

// Dynamic load balancing score computation
export void ispc_compute_load_balance_scores(
    uniform float current_loads[],
    uniform float predicted_loads[],
    uniform float migration_costs[],
    uniform float balance_scores[],
    uniform int worker_count
) {
    // Compute overall load statistics first
    uniform float total_current = 0.0f;
    uniform float total_predicted = 0.0f;
    
    for (uniform int i = 0; i < worker_count; i++) {
        total_current += current_loads[i];
        total_predicted += predicted_loads[i];
    }
    
    uniform float avg_current = total_current / (float)worker_count;
    uniform float avg_predicted = total_predicted / (float)worker_count;
    
    foreach (worker_id = 0 ... worker_count) {
        float current_deviation = abs(current_loads[worker_id] - avg_current);
        float predicted_deviation = abs(predicted_loads[worker_id] - avg_predicted);
        
        // Balance improvement potential
        float improvement = current_deviation - predicted_deviation;
        
        // Migration cost penalty
        float migration_penalty = migration_costs[worker_id];
        
        // Net benefit score
        float net_score = improvement - migration_penalty;
        
        // Apply sigmoid for bounded output
        balance_scores[worker_id] = 1.0f / (1.0f + exp(-4.0f * net_score));
    }
}

// Real-time worker performance profiling
export void ispc_profile_worker_performance(
    uniform uint64 task_completion_times[],
    uniform uint64 cache_miss_counts[],
    uniform uint64 context_switch_counts[],
    uniform float performance_scores[],
    uniform int worker_count,
    uniform int time_window_size
) {
    foreach (worker_id = 0 ... worker_count) {
        // Compute average metrics over time window
        uint64 total_time = 0;
        uint64 total_cache_misses = 0;
        uint64 total_context_switches = 0;
        
        int base_idx = worker_id * time_window_size;
        for (uniform int i = 0; i < time_window_size; i++) {
            total_time += task_completion_times[base_idx + i];
            total_cache_misses += cache_miss_counts[base_idx + i];
            total_context_switches += context_switch_counts[base_idx + i];
        }
        
        float avg_time = (float)total_time / (float)time_window_size;
        float avg_cache_misses = (float)total_cache_misses / (float)time_window_size;
        float avg_context_switches = (float)total_context_switches / (float)time_window_size;
        
        // Normalize and invert for scoring (lower is better)
        float time_score = 1.0f / (1.0f + avg_time * 0.001f);
        float cache_score = 1.0f / (1.0f + avg_cache_misses * 0.01f);
        float context_score = 1.0f / (1.0f + avg_context_switches * 0.1f);
        
        // Weighted combination
        performance_scores[worker_id] = 0.5f * time_score + 
                                       0.3f * cache_score + 
                                       0.2f * context_score;
    }
}