# ZigPulse v3 API Reference

## Table of Contents
- [Core API](#core-api)
- [Configuration](#configuration)
- [Task Management](#task-management)
- [Lock-Free Data Structures](#lock-free-data-structures)
- [CPU Topology](#cpu-topology)
- [Memory Management](#memory-management)
- [Scheduling](#scheduling)
- [Statistics](#statistics)

## Core API

### Creating a Thread Pool

```zig
// Create with default configuration
const pool = try zigpulse.createPool(allocator);
defer pool.deinit();

// Create with custom configuration
const config = zigpulse.Config{
    .num_workers = 8,
    .enable_lock_free = true,
};
const pool = try zigpulse.createPoolWithConfig(allocator, config);
```

### Submitting Tasks

```zig
const task = zigpulse.Task{
    .func = myFunction,
    .data = @ptrCast(&my_data),
    .priority = .high,
    .affinity_hint = 0, // Prefer NUMA node 0
};
try pool.submit(task);
```

### Waiting for Completion

```zig
// Wait for all tasks to complete
pool.wait();
```

## Configuration

### Config Structure

```zig
pub const Config = struct {
    // Thread pool settings
    num_workers: ?usize = null,              // null = physical CPU count
    min_workers: usize = 2,                  
    max_workers: ?usize = null,              // null = 2x physical cores
    
    // V1 features
    enable_work_stealing: bool = true,       // Work-stealing between threads
    enable_adaptive_sizing: bool = false,    // Dynamic worker count
    
    // V2 features (heartbeat scheduling)
    enable_heartbeat: bool = true,           // Heartbeat scheduling
    heartbeat_interval_us: u32 = 100,        // Heartbeat interval
    promotion_threshold: u64 = 10,           // Work:overhead ratio
    min_work_cycles: u64 = 1000,            // Min cycles for promotion
    
    // V3 features (all enabled by default)
    enable_topology_aware: bool = true,      // CPU topology awareness
    enable_numa_aware: bool = true,          // NUMA-aware allocation
    enable_lock_free: bool = true,          // Lock-free data structures
    enable_predictive: bool = false,         // Predictive scheduling (TODO)
    
    // Performance tuning
    task_queue_size: u32 = 1024,            // Per-worker queue size
    cache_line_size: u32 = 64,
    
    // Statistics and debugging
    enable_statistics: bool = true,          
    enable_trace: bool = false,
};
```

## Task Management

### Task Structure

```zig
pub const Task = struct {
    func: *const fn (*anyopaque) void,
    data: *anyopaque,
    priority: Priority = .normal,
    affinity_hint: ?u32 = null,              // Preferred NUMA node
};
```

### Priority Levels

```zig
pub const Priority = enum(u8) {
    low = 0,
    normal = 1,
    high = 2,
};
```

### Task Status

```zig
pub const TaskStatus = enum(u8) {
    pending = 0,
    running = 1,
    completed = 2,
    failed = 3,
    cancelled = 4,
};
```

## Lock-Free Data Structures

### Work-Stealing Deque

```zig
// Create a work-stealing deque
var deque = try zigpulse.lockfree.WorkStealingDeque(T).init(allocator, capacity);
defer deque.deinit();

// Owner operations (single-threaded)
try deque.pushBottom(item);
const item = deque.popBottom();

// Thief operations (thread-safe)
const stolen = deque.steal();
```

### MPMC Queue

```zig
// Create a multi-producer multi-consumer queue
var queue = try zigpulse.lockfree.MpmcQueue(T).init(allocator, capacity);
defer queue.deinit();

// Producer operations
try queue.push(item);

// Consumer operations
const item = queue.pop();
```

## CPU Topology

### Detecting System Topology

```zig
const topo = try zigpulse.topology.detectTopology(allocator);
defer topo.deinit();

// Access topology information
std.debug.print("Physical cores: {}\n", .{topo.physical_cores});
std.debug.print("Total cores: {}\n", .{topo.total_cores});
std.debug.print("NUMA nodes: {}\n", .{topo.numa_nodes.len});
```

### Thread Affinity

```zig
// Set thread affinity to specific cores
const cores = [_]u32{0, 1, 2, 3};
try zigpulse.topology.setThreadAffinity(thread, &cores);

// Set current thread affinity
try zigpulse.topology.setCurrentThreadAffinity(&cores);
```

### Core Distance Calculation

```zig
// Get distance between two cores (0-100)
const distance = topo.getCoreDistance(core1, core2);

// Distance meanings:
// 0: Same core (SMT siblings)
// 10: Same L2 cache
// 20: Same L3 cache
// 50: Same socket
// 100: Different sockets
```

## Memory Management

### Task Pool Allocator

```zig
// Memory pool eliminates allocation overhead
var pool = zigpulse.memory.TaskPool.init(allocator);
defer pool.deinit();

// Allocate a task
const task = try pool.alloc();
defer pool.free(task);

// Get pool statistics
const stats = pool.getStats();
std.debug.print("Allocations: {}\n", .{stats.allocations});
std.debug.print("Cache hits: {}%\n", .{stats.hit_rate});
```

### NUMA-Aware Allocation

```zig
// Allocate memory on specific NUMA node
const ptr = try zigpulse.memory.allocOnNode(T, numa_node);
defer zigpulse.memory.freeOnNode(ptr, numa_node);
```

## Scheduling

### Heartbeat Scheduler

```zig
// The heartbeat scheduler automatically manages task promotion
// based on work/overhead ratio tracking

// Manual control (advanced usage)
const scheduler = pool.scheduler.?;
scheduler.updateThreshold(new_threshold);
scheduler.forcePromotion(true);
```

### Parallel Call (pcall)

```zig
// Potentially parallel function call
const result = try pool.pcall(ReturnType, myFunction);

// Check if executed inline or deferred
switch (result) {
    .immediate => |value| {
        // Function executed inline
        use(value);
    },
    .deferred => |future| {
        // Function executed in thread pool
        const value = try future.get();
        use(value);
    },
}
```

## Statistics

### Accessing Pool Statistics

```zig
const stats = &pool.stats;

// Read statistics
const submitted = stats.tasks_submitted.load(.acquire);
const completed = stats.tasks_completed.load(.acquire);
const stolen = stats.tasks_stolen.load(.acquire);
const cancelled = stats.tasks_cancelled.load(.acquire);

// Calculate metrics
const completion_rate = @as(f64, @floatFromInt(completed)) / 
                       @as(f64, @floatFromInt(submitted)) * 100.0;
const steal_rate = @as(f64, @floatFromInt(stolen)) / 
                  @as(f64, @floatFromInt(completed)) * 100.0;

std.debug.print("Completion rate: {d:.2}%\n", .{completion_rate});
std.debug.print("Work-stealing rate: {d:.2}%\n", .{steal_rate});
```

### Custom Statistics

```zig
// Extend statistics for your application
const MyStats = struct {
    base: zigpulse.ThreadPoolStats,
    custom_metric: std.atomic.Value(u64),
    
    pub fn recordCustom(self: *MyStats) void {
        _ = self.custom_metric.fetchAdd(1, .monotonic);
    }
};
```

## Error Handling

### Task Errors

```zig
pub const TaskError = error{
    TaskPanicked,
    TaskCancelled,
    TaskTimeout,
    QueueFull,
    PoolShutdown,
};
```

### Handling Errors

```zig
// Submit with error handling
pool.submit(task) catch |err| switch (err) {
    error.QueueFull => {
        // Queue is full, retry with backoff
        std.time.sleep(1_000_000); // 1ms
        try pool.submit(task);
    },
    error.PoolShutdown => {
        // Pool is shutting down
        return;
    },
    else => return err,
};
```

## Best Practices

### Task Granularity

```zig
// Good: Substantial work per task
fn processChunk(data: []u8) void {
    // Process 1MB of data
    for (data) |*byte| {
        byte.* = complexTransform(byte.*);
    }
}

// Bad: Too fine-grained
fn processByte(byte: *u8) void {
    byte.* = complexTransform(byte.*);
}
```

### NUMA Awareness

```zig
// Allocate data close to where it will be processed
const numa_node = worker.numa_node orelse 0;
const data = try allocator.allocAdvanced(
    u8, 
    alignment, 
    size, 
    .{ .numa_node = numa_node }
);
```

### Work Distribution

```zig
// Use affinity hints for data locality
const chunk_size = data.len / pool.workers.len;
for (pool.workers, 0..) |worker, i| {
    const start = i * chunk_size;
    const end = if (i == pool.workers.len - 1) data.len else (i + 1) * chunk_size;
    
    const task = Task{
        .func = processRange,
        .data = &data[start..end],
        .affinity_hint = worker.numa_node,
    };
    try pool.submit(task);
}
```