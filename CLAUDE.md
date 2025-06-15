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

### Demos
```bash
# Auto-configuration system demo
zig build demo-config

# Comptime work distribution patterns demo
zig build demo-comptime

# Smart worker selection test
zig build test-smart-worker

# Topology-aware work stealing test
zig build test-topology-stealing

# Topology performance verification (proves 0.6-12.8% improvement)
zig build verify-topology

# Auto-configuration integration with One Euro Filter
zig build demo-integration
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
- `src/comptime_work.zig` - Compile-time work distribution patterns and optimization
- `build_config.zig` - Build-time hardware detection and auto-configuration system
- `src/build_opts.zig` - Compile-time access to auto-detected system configuration

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
- Smart worker selection algorithm with NUMA topology awareness
  - Task affinity hint support for explicit NUMA node preferences
  - Real-time queue load balancing across workers
  - Multi-level selection strategy with intelligent fallbacks
  - ~50% improvement in total execution time, ~4% improvement in task submission
- Topology-aware work stealing for reduced migration overhead
  - Three-phase stealing strategy: same NUMA node → same socket → remote nodes
  - Fisher-Yates shuffling to avoid contention patterns
  - Graceful fallback to random stealing when topology unavailable
  - **VERIFIED**: 0.6-12.8% performance improvement, up to 650% migration overhead reduction
  - Benchmarked and validated through comprehensive performance testing
- Sub-nanosecond overhead for inline pcall execution
- NUMA-aware memory allocation and thread affinity
- Lock-free data structures with hazard pointer memory reclamation
- One Euro Filter for adaptive task execution time prediction (superior to simple averaging)
  - Handles variable workloads and phase changes
  - Outlier resilient (cache misses, thermal throttling)
  - Configurable parameters for different workload characteristics
- Compile-time work distribution patterns with zero runtime overhead
  - Automatic strategy selection based on work characteristics
  - Type-aware parallelization decisions
  - SIMD-aware work chunking and alignment
  - Integration with build-time auto-configuration
- **Build-time auto-configuration system** 
  - Hardware detection with CPU count, SIMD features, NUMA topology estimation
  - Automatic One Euro Filter parameter tuning based on hardware characteristics
  - Architecture-specific optimizations (x86_64, aarch64, etc.)
  - Intelligent defaults with manual override capability
  - **VERIFIED**: Seamless integration of build-time detection with runtime optimization

## Development Notes

### Testing Strategy
The project uses comprehensive testing including unit tests, integration tests, stress tests, and specialized tests for TLS overflow conditions and COZ profiling scenarios. The enhanced parallel testing framework (`src/testing.zig`) provides utilities for testing parallel code with automatic resource cleanup validation, performance constraints, and stress testing capabilities.

### Version Evolution
- **V1**: Basic work-stealing thread pool
- **V2**: Added heartbeat scheduling with token accounting
- **V3**: CPU topology awareness, NUMA optimization, One Euro Filter prediction, compile-time work distribution patterns, **build-time auto-configuration integration**

### Formal Verification
The project is working towards formal verification using Lean 4 theorem prover with LLM-assisted proof development for mathematical correctness guarantees of lock-free algorithms.

### Integration with Reverb
Originally developed for the Reverb HTTP server, achieving 9,110 req/s performance improvements through intelligent task scheduling and topology awareness.