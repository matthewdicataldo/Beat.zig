# 1€ Filter vs Exponential Moving Average for Predictive Task Scheduling

## Executive Summary

The 1€ filter could be superior to simple exponential moving average (EMA) for ZigPulse's predictive task scheduling, particularly for workloads with variable execution times. Its adaptive nature aligns well with the dynamic characteristics of parallel workloads.

## Analysis

### Current Challenge in Task Scheduling

Predictive task scheduling needs to estimate task execution times to make intelligent scheduling decisions. The challenge is:
- Task execution times can vary significantly based on input data
- Outliers (cache misses, context switches) create noise
- Need responsive adaptation to changing workload patterns
- Must maintain low overhead (nanosecond-scale decisions)

### Exponential Moving Average (Current Proposal)

**Formula:** `estimate(t) = α × measurement(t) + (1-α) × estimate(t-1)`

**Pros:**
- Simple and fast (few operations)
- Well-understood behavior
- Constant memory usage

**Cons:**
- Fixed smoothing factor doesn't adapt to variance
- Trade-off between noise reduction and responsiveness is static
- Slow to adapt to step changes in execution time

### 1€ Filter (Alternative Proposal)

**Key Innovation:** Adaptive smoothing based on rate of change

**How it would work for task scheduling:**
```zig
const OneEuroFilter = struct {
    // Parameters
    min_cutoff: f64 = 1.0,    // Hz - minimum cutoff frequency
    beta: f64 = 0.007,         // Speed coefficient
    d_cutoff: f64 = 1.0,       // Derivative cutoff
    
    // State
    x_prev: ?f64 = null,
    dx_prev: f64 = 0,
    t_prev: ?u64 = null,       // Previous timestamp (ns)
    
    pub fn filter(self: *OneEuroFilter, x: f64, t_ns: u64) f64 {
        if (self.t_prev == null) {
            self.x_prev = x;
            self.t_prev = t_ns;
            return x;
        }
        
        const t_e = @as(f64, @floatFromInt(t_ns - self.t_prev.?)) / 1e9; // seconds
        self.t_prev = t_ns;
        
        // Estimate derivative
        const dx = (x - self.x_prev.?) / t_e;
        const dx_filtered = self.filterDerivative(dx, t_e);
        
        // Adapt cutoff frequency based on speed
        const cutoff = self.min_cutoff + self.beta * @abs(dx_filtered);
        
        // Filter the signal
        const alpha = computeAlpha(cutoff, t_e);
        const x_filtered = alpha * x + (1 - alpha) * self.x_prev.?;
        
        self.x_prev = x_filtered;
        self.dx_prev = dx_filtered;
        
        return x_filtered;
    }
    
    fn filterDerivative(self: *OneEuroFilter, dx: f64, t_e: f64) f64 {
        const alpha = computeAlpha(self.d_cutoff, t_e);
        return alpha * dx + (1 - alpha) * self.dx_prev;
    }
    
    fn computeAlpha(cutoff: f64, t_e: f64) f64 {
        const tau = 1.0 / (2 * std.math.pi * cutoff);
        return t_e / (t_e + tau);
    }
};
```

### Application to ZigPulse Task Scheduling

**Integration Points:**

1. **Per-Task-Type Tracking:**
```zig
const TaskProfile = struct {
    fingerprint: u64,              // Task type identifier
    execution_filter: OneEuroFilter,
    sample_count: u64,
    last_update: u64,              // Timestamp
};
```

2. **Scheduling Decision:**
```zig
fn shouldPromoteTask(profile: *TaskProfile, current_time: u64) bool {
    const predicted_time = profile.execution_filter.getCurrentEstimate();
    const overhead_threshold = 1000; // 1μs overhead acceptable
    
    // Promote if predicted execution > overhead
    return predicted_time > overhead_threshold;
}
```

3. **Adaptive Behavior Benefits:**
   - **Stable workloads**: High smoothing reduces noise from cache effects
   - **Changing workloads**: Fast adaptation when task characteristics change
   - **Bursty patterns**: Responsive to sudden changes in execution time

### Performance Comparison

| Aspect               | EMA      | 1€ Filter |
|----------------------|----------|-----------|
| Computational Cost   | ~3 ops   | ~15 ops   |
| Memory per Task Type | 16 bytes | 48 bytes  |
| Adaptation Speed     | Fixed    | Variable  |
| Noise Handling       | Fixed    | Adaptive  |
| Outlier Robustness   | Poor     | Good      |

### Specific Advantages for Task Scheduling

1. **Variable Workloads**: Tasks with data-dependent execution times
   - Example: Tree traversal where depth varies
   - 1€ filter adapts smoothing based on variance

2. **Phase Changes**: Applications with distinct execution phases
   - Example: Initialization → Processing → Cleanup
   - Fast adaptation to step changes

3. **Outlier Handling**: Occasional cache misses or page faults
   - Less affected by spikes due to derivative filtering

4. **Microarchitectural Effects**: CPU frequency scaling, thermal throttling
   - Adapts to gradual performance changes

## Recommendation

**Yes, implement 1€ filter for ZigPulse v3 predictive scheduling because:**

1. **Better Accuracy**: Adaptive smoothing will provide more accurate predictions for variable workloads
2. **Acceptable Overhead**: ~15 operations is still sub-nanosecond on modern CPUs
3. **Future-Proof**: As workloads become more dynamic (GPU offloading, heterogeneous systems), adaptive filtering becomes more valuable
4. **Proven Technology**: Successfully used in real-time systems with similar latency constraints

### Implementation Strategy

1. Start with hybrid approach:
   - Simple EMA for high-frequency tasks (>1000 Hz submission rate)
   - 1€ filter for lower-frequency, variable tasks

2. Tune parameters based on workload:
   ```zig
   const FilterConfig = struct {
       // Conservative: More smoothing, slower adaptation
       conservative: OneEuroConfig{ .min_cutoff = 0.5, .beta = 0.001 },
       
       // Balanced: Good for most workloads  
       balanced: OneEuroConfig{ .min_cutoff = 1.0, .beta = 0.007 },
       
       // Responsive: Fast adaptation, less smoothing
       responsive: OneEuroConfig{ .min_cutoff = 2.0, .beta = 0.05 },
   };
   ```

3. Add profiling to validate improvements:
   - Prediction accuracy metrics
   - Scheduling decision quality
   - Overall throughput impact

## Conclusion

The 1€ filter's adaptive nature makes it particularly well-suited for the dynamic environment of parallel task scheduling. While slightly more complex than EMA, the improved prediction accuracy justifies the minimal additional overhead. This aligns with ZigPulse's philosophy of using advanced techniques where they provide meaningful performance benefits.