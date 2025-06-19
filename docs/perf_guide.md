# beat.zig perf_guide

## Performance Characteristics

### Overhead Measurements

| Operation                | Time   | Notes                |
|--------------------------|--------|----------------------|
| Task submission (cached) | ~50ns  | Using memory pool    |
| Task submission (alloc)  | ~200ns | Heap allocation      |
| Work stealing            | ~100ns | Between threads      |
| Local pop                | ~20ns  | Owner thread only    |
| Memory pool alloc        | ~20ns  | Lock-free            |
| Thread migration         | +650%  | Avoided by affinity  |
| pcall inline             | 0-5ns  | Zero in release mode |
| pcall deferred           | ~400ns | Full task submission |

### Scalability

**Linear Scaling Conditions:**
- Independent tasks
- Sufficient task granularity (>10μs)
- Balanced work distribution
- NUMA-aware data placement

**Scaling Limitations:**
- Memory bandwidth saturation
- Cache coherence traffic
- Lock contention (mutex mode)
- OS scheduler interference

## Optimization Strategies

### Task Granularity

**Optimal Task Size:**
```zig
// Good: 10-100μs of work
fn processDataChunk(chunk: []const u8) void {
    // Process 64KB-1MB chunks
    var checksum: u32 = 0;
    for (chunk) |byte| {
        checksum = updateCRC32(checksum, byte);
    }
}

// Too fine: <1μs of work
fn processSingleByte(byte: u8) void {
    // Overhead dominates execution time
}

// Too coarse: >10ms of work
fn processEntireFile(file: []const u8) void {
    // Poor load balancing
}
```

### Memory Access Patterns

**Cache-Friendly Access:**
```zig
// Sequential access pattern
fn sumArray(data: []const f32) f32 {
    var sum: f32 = 0;
    // Prefetcher-friendly linear scan
    for (data) |value| {
        sum += value;
    }
    return sum;
}

// NUMA-aware allocation
fn allocateOnWorkerNode(worker: *Worker, size: usize) ![]u8 {
    const numa_node = worker.numa_node orelse 0;
    return allocator.allocAdvanced(u8, 64, size, .{
        .numa_node = numa_node,
    });
}
```

### Work Distribution

**Balanced Distribution:**
```zig
// Static partitioning with load balancing
fn distributeWork(data: []const Item, pool: *ThreadPool) !void {
    const n_workers = pool.workers.len;
    const chunk_size = (data.len + n_workers - 1) / n_workers;
    
    // Create tasks with affinity hints
    for (0..n_workers) |i| {
        const start = i * chunk_size;
        const end = @min((i + 1) * chunk_size, data.len);
        if (start >= end) break;
        
        const task = Task{
            .func = processChunk,
            .data = &data[start..end],
            .affinity_hint = @intCast(i),
        };
        try pool.submit(task);
    }
}

// Dynamic work stealing for irregular workloads
fn processTree(node: *TreeNode, pool: *ThreadPool) !void {
    if (node.size < threshold) {
        // Process locally if small enough
        processNodeLocal(node);
    } else {
        // Split and defer to pool
        for (node.children) |child| {
            const task = Task{
                .func = processTreeTask,
                .data = child,
            };
            try pool.submit(task);
        }
    }
}
```

## Benchmarking

### Microbenchmarks

**Task Submission Benchmark:**
```zig
fn benchmarkSubmission(pool: *ThreadPool) !void {
    const iterations = 1_000_000;
    var timer = try Timer.start();
    
    for (0..iterations) |_| {
        const task = Task{
            .func = emptyTask,
            .data = undefined,
        };
        try pool.submit(task);
    }
    
    pool.wait();
    const elapsed = timer.read();
    const ns_per_task = elapsed / iterations;
    std.debug.print("Task submission: {}ns\n", .{ns_per_task});
}
```

**Work-Stealing Benchmark:**
```zig
fn benchmarkStealing(pool: *ThreadPool) !void {
    // Saturate one worker's queue
    const worker_0 = &pool.workers[0];
    for (0..1000) |_| {
        try worker_0.queue.lockfree.pushBottom(&dummy_task);
    }
    
    // Measure steal operations
    var timer = try Timer.start();
    var stolen: u32 = 0;
    
    for (0..1000) |_| {
        if (pool.workers[1].stealFrom(worker_0)) |_| {
            stolen += 1;
        }
    }
    
    const elapsed = timer.read();
    std.debug.print("Steal rate: {}ns per steal\n", .{elapsed / stolen});
}
```

### Macrobenchmarks

**Parallel Sum Reduction:**
```zig
fn benchmarkParallelSum(pool: *ThreadPool) !void {
    const data = try allocator.alloc(f32, 100_000_000);
    defer allocator.free(data);
    
    // Initialize with random data
    for (data) |*value| {
        value.* = random.float(f32);
    }
    
    // Serial baseline
    var timer = try Timer.start();
    const serial_sum = sumArray(data);
    const serial_time = timer.lap();
    
    // Parallel version
    const parallel_sum = try parallelSum(pool, data);
    const parallel_time = timer.read();
    
    const speedup = @as(f64, @floatFromInt(serial_time)) / 
                    @as(f64, @floatFromInt(parallel_time));
    
    std.debug.print("Serial: {}ms, Parallel: {}ms\n", .{
        serial_time / 1_000_000,
        parallel_time / 1_000_000,
    });
    std.debug.print("Speedup: {d:.2}x\n", .{speedup});
    std.debug.print("Efficiency: {d:.1}%\n", .{
        speedup / @as(f64, @floatFromInt(pool.workers.len)) * 100.0
    });
}
```

## Performance Tuning

### Configuration Tuning

```zig
// High-throughput configuration
const high_throughput_config = Config{
    .num_workers = try std.Thread.getCpuCount(),
    .enable_lock_free = true,
    .enable_topology_aware = true,
    .task_queue_size = 4096,
    .enable_heartbeat = false, // Reduce overhead
};

// Low-latency configuration
const low_latency_config = Config{
    .num_workers = physical_cores, // No hyperthreading
    .enable_lock_free = true,
    .enable_topology_aware = true,
    .task_queue_size = 256, // Smaller for cache
    .heartbeat_interval_us = 50, // More responsive
};

// Battery-efficient configuration
const power_save_config = Config{
    .num_workers = physical_cores / 2,
    .enable_adaptive_sizing = true,
    .min_workers = 1,
    .enable_heartbeat = true,
    .heartbeat_interval_us = 1000, // Less frequent
};
```

### NUMA Optimization

```zig
// NUMA-aware data partitioning
fn setupNumaPartitions(pool: *ThreadPool, data: []u8) !void {
    const topo = pool.topology orelse return;
    const numa_nodes = topo.numa_nodes.len;
    const partition_size = data.len / numa_nodes;
    
    for (topo.numa_nodes, 0..) |node, i| {
        const start = i * partition_size;
        const end = if (i == numa_nodes - 1) data.len else (i + 1) * partition_size;
        
        // Migrate pages to target NUMA node
        if (builtin.os.tag == .linux) {
            const MPOL_BIND = 2;
            const addr = @ptrToInt(&data[start]);
            const len = end - start;
            _ = linux.mbind(addr, len, MPOL_BIND, &node.id, 1, 0);
        }
    }
}
```

### Lock-Free Optimizations

```zig
// Reduce contention with local combining
const LocalCombiner = struct {
    local_sum: f32 = 0,
    count: u32 = 0,
    
    pub fn add(self: *LocalCombiner, value: f32) void {
        self.local_sum += value;
        self.count += 1;
        
        // Flush to global periodically
        if (self.count >= 1000) {
            global_sum.fetchAdd(self.local_sum, .monotonic);
            self.local_sum = 0;
            self.count = 0;
        }
    }
};
```

## Profiling and Analysis

### Using Zig's Built-in Profiler

```bash
# Build with profiling
zig build -Drelease-safe -Dprofile

# Run and generate profile
./zig-out/bin/app
# Creates profile.json

# Analyze with Chrome DevTools
google-chrome chrome://tracing
# Load profile.json
```

### Linux Perf Integration

```bash
# Record performance data
perf record -g ./zig-out/bin/app

# Analyze CPU usage
perf report

# Check cache misses
perf stat -e cache-misses,cache-references ./app

# Monitor thread migration
perf stat -e migrations ./app
```

### Custom Performance Counters

```zig
const PerfCounter = struct {
    start_cycles: u64,
    total_cycles: u64 = 0,
    count: u64 = 0,
    
    pub fn begin(self: *PerfCounter) void {
        self.start_cycles = readCycleCounter();
    }
    
    pub fn end(self: *PerfCounter) void {
        const cycles = readCycleCounter() - self.start_cycles;
        self.total_cycles += cycles;
        self.count += 1;
    }
    
    pub fn report(self: *const PerfCounter) void {
        if (self.count == 0) return;
        const avg = self.total_cycles / self.count;
        std.debug.print("Average cycles: {}\n", .{avg});
    }
};

// Usage
var submit_counter = PerfCounter{};

// In hot path
submit_counter.begin();
try pool.submit(task);
submit_counter.end();

// Report results
defer submit_counter.report();
```

## Common Performance Issues

### False Sharing

**Problem:**
```zig
// Bad: Workers share cache line
const Workers = struct {
    count: [8]u32, // All in same cache line
};
```

**Solution:**
```zig
// Good: Padding prevents false sharing
const Workers = struct {
    count: [8]CacheAligned(u32),
};

fn CacheAligned(comptime T: type) type {
    return struct {
        value: T,
        _padding: [64 - @sizeOf(T)]u8 = undefined,
    };
}
```

### Lock Contention

**Detection:**
```zig
// Monitor mutex contention
var contention_count: u64 = 0;

fn contestedLock(mutex: *Mutex) void {
    if (!mutex.tryLock()) {
        contention_count += 1;
        mutex.lock();
    }
    defer mutex.unlock();
}
```

**Mitigation:**
- Use lock-free data structures
- Reduce critical section size
- Implement backoff strategies
- Partition data to reduce sharing

### Memory Bandwidth Saturation

**Detection:**
```bash
# Intel systems
pcm-memory

# AMD systems  
amd_uprof --memory-access
```

**Mitigation:**
- Improve cache reuse
- Compress data structures
- Use SIMD for bulk operations
- Distribute across NUMA nodes