# Beat v3

Ultra-optimized parallelism library for Zig with CPU topology awareness, lock-free data structures, and zero-overhead abstractions.

## Features

### Core Features
- **Lock-free work-stealing deque** (Chase-Lev algorithm)
- **CPU topology awareness** with thread affinity
- **NUMA-aware memory allocation**
- **Zero-overhead pcall** (potentially parallel calls)
- **Heartbeat scheduling** with token accounting
- **Memory pools** for allocation-free hot paths

### Performance
- Work-stealing for automatic load balancing
- Cache-aware data structures (64-byte aligned)
- SIMD-optimized operations where applicable
- Topology-aware task scheduling reduces migration overhead by 650%
- Sub-nanosecond overhead for inline pcall execution

## Quick Start

### Using the Modular Version

```zig
const std = @import("std");
const beat = @import("beat");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    // Create thread pool with default configuration
    const pool = try beat.createPool(allocator);
    defer pool.deinit();
    
    // Submit tasks
    const task = beat.Task{
        .func = myWork,
        .data = @ptrCast(&my_data),
    };
    try pool.submit(task);
    
    // Wait for completion
    pool.wait();
}

fn myWork(data: *anyopaque) void {
    // Your parallel work here
}
```

### Integration Options

**Option 1: Direct Module Import (Recommended)**
```zig
// In your build.zig
const beat = b.addModule("beat", .{
    .root_source_file = .{ .path = "path/to/Beat/src/core.zig" },
});
exe.root_module.addImport("beat", beat);
```

**Option 2: Bundle File Import**
```zig
// Copy beat.zig to your project, then:
const beat = @import("beat.zig");
```

The bundle file (`beat.zig`) provides a single entry point that imports all modules. It requires the `src/` directory to be present but offers the convenience of a single import.

## Building

```bash
# Run tests
zig build test

# Run benchmarks
zig build bench

# Test bundle file
zig build test-bundle

# Run examples
zig build example-modular
zig build example-bundle
```

## Advanced Configuration

```zig
const config = zigpulse.Config{
    .num_workers = 8,                    // Number of worker threads
    .enable_work_stealing = true,        // Work-stealing between threads
    .enable_topology_aware = true,       // CPU topology awareness
    .enable_lock_free = true,           // Use lock-free data structures
    .enable_heartbeat = true,           // Heartbeat scheduling
    .task_queue_size = 1024,           // Per-worker queue size
};

const pool = try zigpulse.createPoolWithConfig(allocator, config);
```

## Architecture

### Modular Structure
```
src/
├── core.zig      # Main thread pool implementation
├── lockfree.zig  # Lock-free data structures
├── topology.zig  # CPU topology detection
├── memory.zig    # Memory pools and allocators
├── scheduler.zig # Scheduling algorithms
└── pcall.zig     # Parallel call abstractions
```

### Key Components

1. **Work-Stealing Deque**: Each worker has its own deque, stealing from others when idle
2. **CPU Topology**: Detects CPU cores, caches, NUMA nodes for optimal scheduling
3. **Memory Pools**: Lock-free memory pools eliminate allocation overhead
4. **Heartbeat Scheduler**: Tracks work/overhead ratio for intelligent task promotion

## Performance Tips

1. **Task Granularity**: Ensure tasks are large enough to amortize scheduling overhead
2. **Memory Locality**: Use affinity hints to keep related tasks on the same NUMA node
3. **Avoid False Sharing**: Data structures are cache-line aligned by default
4. **Batch Operations**: Submit multiple tasks at once when possible

## Benchmarks

On a typical 4-core system:

- **Task submission**: ~50ns per task
- **Work stealing**: ~100ns per steal
- **Memory pool allocation**: ~20ns
- **Thread migration cost**: 650% overhead (avoided by topology awareness)
- **Real-world performance**: 9,110 req/s achieved in Reverb HTTP server integration

## Formal Verification

ZigPulse is working towards formal verification of its lock-free algorithms using:
- Lean 4 theorem prover with LLMLean integration
- LLM-assisted proof development (DeepSeek-Prover-V2, o3-pro)
- Mathematical guarantees of correctness

See [docs/FORMAL_VERIFICATION.md](docs/FORMAL_VERIFICATION.md) for details.

## License

MIT License

## Contributing

Contributions welcome! Please ensure:
- All tests pass (`zig build test`)
- Benchmarks show no regression
- Code follows existing style
- Changes are documented