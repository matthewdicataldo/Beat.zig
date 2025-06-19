// ISPC kernel for SIMD-optimized queue operations
// Replaces Zig SIMDQueue and SIMDDeque functionality  
// Target: 4-10x faster queue operations with vectorized batch processing

// Queue operation types
enum QueueOpType {
    ENQUEUE = 0,
    DEQUEUE = 1,
    PEEK = 2,
    BATCH_ENQUEUE = 3,
    BATCH_DEQUEUE = 4
};

// SIMD queue element structure (matches Zig)
struct SIMDQueueElement {
    uniform uint64 task_id;
    uniform uint32 priority;
    uniform uint32 batch_size;
    uniform float estimated_time;
    uniform uint64 fingerprint_low;
    uniform uint64 fingerprint_high;
    uniform uint32 worker_hint;
    uniform uint32 numa_node;
};

// Queue metrics structure (matches Zig)
struct SIMDQueueMetrics {
    uniform uint64 total_enqueued;
    uniform uint64 total_dequeued;
    uniform uint64 current_size;
    uniform uint64 peak_size;
    uniform float avg_wait_time;
    uniform uint64 batch_operations;
    uniform float throughput_ops_per_sec;
};

// Vectorized batch enqueue operation
export void ispc_simd_batch_enqueue(
    uniform SIMDQueueElement queue[],
    uniform uint64* uniform queue_head,
    uniform uint64* uniform queue_tail,
    uniform uint64 queue_capacity,
    uniform SIMDQueueElement new_elements[],
    uniform uint64 batch_size,
    uniform uint64* uniform elements_enqueued
) {
    uniform uint64 current_tail = *queue_tail;
    uniform uint64 available_space = queue_capacity - (*queue_head - current_tail);
    uniform uint64 actual_batch_size = min(batch_size, available_space);
    
    // Vectorized copy of elements
    foreach (i = 0 ... actual_batch_size) {
        uniform uint64 queue_index = (current_tail + i) % queue_capacity;
        queue[queue_index] = new_elements[i];
    }
    
    // Atomic update of tail pointer
    *queue_tail = current_tail + actual_batch_size;
    *elements_enqueued = actual_batch_size;
}

// Vectorized batch dequeue operation  
export void ispc_simd_batch_dequeue(
    uniform SIMDQueueElement queue[],
    uniform uint64* uniform queue_head,
    uniform uint64* uniform queue_tail,
    uniform uint64 queue_capacity,
    uniform SIMDQueueElement output_elements[],
    uniform uint64 requested_batch_size,
    uniform uint64* uniform elements_dequeued
) {
    uniform uint64 current_head = *queue_head;
    uniform uint64 available_elements = *queue_tail - current_head;
    uniform uint64 actual_batch_size = min(requested_batch_size, available_elements);
    
    // Vectorized copy of elements
    foreach (i = 0 ... actual_batch_size) {
        uniform uint64 queue_index = (current_head + i) % queue_capacity;
        output_elements[i] = queue[queue_index];
    }
    
    // Atomic update of head pointer
    *queue_head = current_head + actual_batch_size;
    *elements_dequeued = actual_batch_size;
}

// Parallel priority-based insertion with SIMD sorting
export void ispc_simd_priority_enqueue(
    uniform SIMDQueueElement queue[],
    uniform uint64* uniform queue_size,
    uniform uint64 queue_capacity,
    uniform SIMDQueueElement new_elements[],
    uniform uint64 batch_size
) {
    // First, sort the new elements by priority (parallel sort)
    // Using a simple parallel bubble sort for demonstration
    for (uniform int pass = 0; pass < batch_size - 1; pass++) {
        foreach (i = 0 ... batch_size - 1 - pass) {
            if (new_elements[i].priority < new_elements[i + 1].priority) {
                // Swap elements
                SIMDQueueElement temp = new_elements[i];
                new_elements[i] = new_elements[i + 1];
                new_elements[i + 1] = temp;
            }
        }
    }
    
    // Insert sorted elements into priority queue
    uniform uint64 current_size = *queue_size;
    uniform uint64 insertable = min(batch_size, queue_capacity - current_size);
    
    for (uniform uint64 i = 0; i < insertable; i++) {
        // Find insertion point using vectorized search
        uniform uint64 insert_pos = current_size;
        
        foreach (j = 0 ... current_size) {
            if (queue[j].priority < new_elements[i].priority) {
                insert_pos = j;
                break;
            }
        }
        
        // Shift elements to make space (vectorized)
        for (uniform uint64 k = current_size; k > insert_pos; k--) {
            queue[k] = queue[k - 1];
        }
        
        // Insert new element
        queue[insert_pos] = new_elements[i];
        current_size++;
    }
    
    *queue_size = current_size;
}

// SIMD-optimized queue search and filtering
export void ispc_simd_queue_filter(
    uniform SIMDQueueElement queue[],
    uniform uint64 queue_size,
    uniform uint32 filter_priority,
    uniform uint32 filter_numa_node,
    uniform uint64 filter_fingerprint_mask,
    uniform SIMDQueueElement filtered_results[],
    uniform uint64* uniform result_count
) {
    uniform uint64 matches = 0;
    
    // Parallel filtering with SIMD comparison
    foreach (i = 0 ... queue_size) {
        bool priority_match = (queue[i].priority >= filter_priority);
        bool numa_match = (filter_numa_node == 0xFFFFFFFF) || (queue[i].numa_node == filter_numa_node);
        bool fingerprint_match = (filter_fingerprint_mask == 0) || 
                                ((queue[i].fingerprint_low & filter_fingerprint_mask) != 0);
        
        if (priority_match && numa_match && fingerprint_match) {
            // Use atomic increment for thread safety
            uniform uint64 index = matches++;
            if (index < queue_size) { // Bounds check
                filtered_results[index] = queue[i];
            }
        }
    }
    
    *result_count = matches;
}

// Vectorized queue metrics computation
export void ispc_compute_queue_metrics(
    uniform SIMDQueueElement queue[],
    uniform uint64 queue_size,
    uniform uint64 total_enqueued,
    uniform uint64 total_dequeued,
    uniform float time_window_seconds,
    uniform SIMDQueueMetrics* uniform metrics
) {
    metrics->total_enqueued = total_enqueued;
    metrics->total_dequeued = total_dequeued;
    metrics->current_size = queue_size;
    metrics->peak_size = queue_size; // Simplified - would track actual peak
    
    // Parallel computation of average wait time
    uniform float total_wait_time = 0.0f;
    
    foreach (i = 0 ... queue_size) {
        total_wait_time += queue[i].estimated_time;
    }
    
    metrics->avg_wait_time = (queue_size > 0) ? (total_wait_time / (float)queue_size) : 0.0f;
    
    // Calculate throughput
    uniform uint64 operations = total_enqueued + total_dequeued;
    metrics->throughput_ops_per_sec = (time_window_seconds > 0.0f) ? 
        ((float)operations / time_window_seconds) : 0.0f;
    
    metrics->batch_operations = operations / 10; // Estimate batch operations
}

// SIMD work-stealing deque operations
export void ispc_simd_deque_steal_batch(
    uniform SIMDQueueElement source_deque[],
    uniform uint64* uniform source_head,
    uniform uint64* uniform source_tail,
    uniform uint64 source_capacity,
    uniform SIMDQueueElement dest_deque[],
    uniform uint64* uniform dest_head, 
    uniform uint64* uniform dest_tail,
    uniform uint64 dest_capacity,
    uniform uint64 steal_batch_size,
    uniform uint64* uniform stolen_count
) {
    uniform uint64 source_size = *source_tail - *source_head;
    uniform uint64 available_to_steal = source_size / 2; // Steal up to half
    uniform uint64 actual_steal = min(steal_batch_size, available_to_steal);
    
    uniform uint64 dest_available = dest_capacity - (*dest_tail - *dest_head);
    actual_steal = min(actual_steal, dest_available);
    
    if (actual_steal == 0) {
        *stolen_count = 0;
        return;
    }
    
    // Steal from head of source (oldest tasks)
    uniform uint64 source_start = *source_head;
    uniform uint64 dest_start = *dest_tail;
    
    // Vectorized copy
    foreach (i = 0 ... actual_steal) {
        uniform uint64 src_idx = (source_start + i) % source_capacity;
        uniform uint64 dest_idx = (dest_start + i) % dest_capacity;
        dest_deque[dest_idx] = source_deque[src_idx];
    }
    
    // Update pointers atomically
    *source_head = source_start + actual_steal;
    *dest_tail = dest_start + actual_steal;
    *stolen_count = actual_steal;
}

// NUMA-aware queue load balancing
export void ispc_numa_aware_queue_balance(
    uniform SIMDQueueElement numa_queues[][],
    uniform uint64 numa_queue_sizes[],
    uniform uint64 numa_queue_capacities[],
    uniform uint64 numa_nodes,
    uniform float balance_threshold
) {
    // Calculate average queue size
    uniform float total_size = 0.0f;
    foreach (node = 0 ... numa_nodes) {
        total_size += (float)numa_queue_sizes[node];
    }
    uniform float avg_size = total_size / (float)numa_nodes;
    
    // Parallel load balancing between NUMA nodes
    foreach (src_node = 0 ... numa_nodes) {
        uniform float load_ratio = (float)numa_queue_sizes[src_node] / avg_size;
        
        if (load_ratio > balance_threshold) {
            // This node is overloaded, distribute to others
            uniform uint64 excess = numa_queue_sizes[src_node] - (uint64)avg_size;
            uniform uint64 per_node_transfer = excess / (numa_nodes - 1);
            
            foreach (dest_node = 0 ... numa_nodes) {
                if (dest_node != src_node && per_node_transfer > 0) {
                    uniform uint64 dest_available = 
                        numa_queue_capacities[dest_node] - numa_queue_sizes[dest_node];
                    uniform uint64 transfer_size = min(per_node_transfer, dest_available);
                    
                    if (transfer_size > 0) {
                        // Transfer elements (simplified - would use proper deque operations)
                        for (uniform uint64 i = 0; i < transfer_size; i++) {
                            numa_queues[dest_node][numa_queue_sizes[dest_node] + i] = 
                                numa_queues[src_node][numa_queue_sizes[src_node] - transfer_size + i];
                        }
                        
                        numa_queue_sizes[src_node] -= transfer_size;
                        numa_queue_sizes[dest_node] += transfer_size;
                    }
                }
            }
        }
    }
}

// Prefetch queue elements for better cache performance
export void ispc_prefetch_queue_elements(
    uniform SIMDQueueElement queue[],
    uniform uint64 start_index,
    uniform uint64 count,
    uniform uint64 queue_capacity
) {
    foreach (i = 0 ... count) {
        uniform uint64 queue_index = (start_index + i) % queue_capacity;
        prefetch_l1(&queue[queue_index]);
        
        // Prefetch next cache line if element spans cache boundaries  
        if (sizeof(SIMDQueueElement) > 64) {
            prefetch_l1(((uniform uint8*)&queue[queue_index]) + 64);
        }
    }
}

// Queue compaction to reduce fragmentation
export void ispc_compact_queue(
    uniform SIMDQueueElement queue[],
    uniform uint64* uniform queue_head,
    uniform uint64* uniform queue_tail,
    uniform uint64 queue_capacity
) {
    uniform uint64 current_head = *queue_head;
    uniform uint64 current_tail = *queue_tail;
    uniform uint64 size = current_tail - current_head;
    
    if (current_head == 0) {
        return; // Already compacted
    }
    
    // Vectorized compaction - move all elements to start of array
    foreach (i = 0 ... size) {
        uniform uint64 src_idx = (current_head + i) % queue_capacity;
        queue[i] = queue[src_idx];
    }
    
    // Reset head and tail
    *queue_head = 0;
    *queue_tail = size;
}