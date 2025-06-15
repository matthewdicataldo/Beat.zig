# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Beat (formerly ZigPulse) is an ultra-optimized parallelism library for Zig featuring CPU topology awareness, lock-free data structures, and zero-overhead abstractions. The library prioritizes performance through work-stealing deques, heartbeat scheduling, and NUMA-aware memory allocation.

## Common Commands

### Build and Test
```bash
# Run unit tests
zig build test

# Run benchmarks
zig build bench

# Test bundle file
zig build test-bundle

# Run examples
zig build examples
zig build example-modular
zig build example-bundle
```

### Specialized Tests
```bash
# COZ profiling benchmark (with profiling enabled)
zig build bench-coz -Dcoz=true

# Performance and stress tests
zig build test-simple
zig build test-minimal
zig build test-stress
zig build test-tls
zig build test-tls-intensive
zig build test-coz-conditions
zig build test-cpu
```

### Profiling
```bash
# COZ profiler (requires coz to be installed)
./profile_coz.sh
```

## Architecture

### Module Structure
- `src/core.zig` - Main thread pool implementation and API entry point
- `src/lockfree.zig` - Lock-free data structures (Chase-Lev deque, MPMC queue)  
- `src/topology.zig` - CPU topology detection and thread affinity
- `src/memory.zig` - Memory pools and NUMA-aware allocation
- `src/scheduler.zig` - Heartbeat and predictive scheduling algorithms
- `src/pcall.zig` - Zero-overhead potentially parallel calls
- `src/coz.zig` - COZ profiler integration
- `src/testing.zig` - Enhanced parallel testing framework with resource validation

### Bundle vs Modular Usage
- **Bundle**: Single file import via `beat.zig` (convenience)
- **Modular**: Direct import of `src/core.zig` (recommended for performance)

The bundle file re-exports all modules but requires the `src/` directory structure.

### Key Design Principles
- Data-oriented design with cache-line alignment (64 bytes)
- Lock-free algorithms for hot paths with mutex fallbacks
- CPU topology awareness for thread placement and NUMA optimization
- Heartbeat scheduling with work/overhead ratio tracking
- Memory pools to eliminate allocation overhead

### Performance Features
- Work-stealing deque with Chase-Lev algorithm
- CPU topology-aware task scheduling (650% migration overhead reduction)
- Sub-nanosecond overhead for inline pcall execution
- NUMA-aware memory allocation and thread affinity
- Lock-free data structures with hazard pointer memory reclamation
- One Euro Filter for adaptive task execution time prediction (superior to simple averaging)
  - Handles variable workloads and phase changes
  - Outlier resilient (cache misses, thermal throttling)
  - Configurable parameters for different workload characteristics

## Development Notes

### Testing Strategy
The project uses comprehensive testing including unit tests, integration tests, stress tests, and specialized tests for TLS overflow conditions and COZ profiling scenarios. The enhanced parallel testing framework (`src/testing.zig`) provides utilities for testing parallel code with automatic resource cleanup validation, performance constraints, and stress testing capabilities.

### Version Evolution
- **V1**: Basic work-stealing thread pool
- **V2**: Added heartbeat scheduling with token accounting
- **V3**: CPU topology awareness, NUMA optimization, One Euro Filter prediction, formal verification planning

### Formal Verification
The project is working towards formal verification using Lean 4 theorem prover with LLM-assisted proof development for mathematical correctness guarantees of lock-free algorithms.

### Integration with Reverb
Originally developed for the Reverb HTTP server, achieving 9,110 req/s performance improvements through intelligent task scheduling and topology awareness.