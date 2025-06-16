# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Beat (formerly ZigPulse) is an ultra-optimized parallelism library for Zig featuring CPU topology awareness, lock-free data structures, memory-aware scheduling, and zero-overhead abstractions. The library prioritizes performance through work-stealing deques, heartbeat scheduling, NUMA-aware memory allocation, and intelligent memory pressure adaptation.

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

# Memory-aware scheduling and development mode
zig build test-memory-pressure
zig build test-development-mode

# Advanced predictive scheduling tests
zig build test-fingerprint
zig build test-enhanced-filter
zig build test-multi-factor-confidence
zig build test-intelligent-decision
zig build test-predictive-accounting
zig build test-advanced-worker-selection

# SIMD task processing tests
zig build test-simd
zig build test-simd-batch
zig build test-simd-queue
zig build test-simd-classification
zig build test-simd-benchmark

# GPU integration tests
zig build test-gpu-integration
zig build test-gpu-classifier
zig build test-sycl-detection
```

### Demos
```bash
# Auto-configuration system demo
zig build demo-config

# Comptime work distribution patterns demo
zig build demo-comptime

# Enhanced error messages demonstration
zig build test-errors

# Smart worker selection test
zig build test-smart-worker

# Topology-aware work stealing test
zig build test-topology-stealing

# Topology performance verification (proves 0.6-12.8% improvement)
zig build verify-topology

# Auto-configuration integration with One Euro Filter
zig build demo-integration

# Parallel work distribution runtime tests
zig build test-parallel-work

# Thread affinity improvements test
zig build test-affinity
```

### Souper Superoptimization
```bash
# Setup Souper toolchain (30-60 minutes, one-time)
./scripts/setup_souper.sh --background

# Monitor setup progress
./scripts/monitor_souper_progress.sh -i

# Run comprehensive analysis (after setup complete)
source souper_env.sh
./scripts/run_souper_analysis.sh

# Quick analysis (high-priority modules only)
./scripts/run_souper_analysis.sh -q

# Analyze specific module
./scripts/run_souper_analysis.sh -m fingerprint

# Note: Generated LLVM IR files are automatically organized in artifacts/llvm_ir/
# Setup logs and progress files are stored in artifacts/souper/
```

### Documentation
```bash
# Generate API documentation
zig build docs

# Generate bundle documentation  
zig build docs-bundle

# Generate comprehensive documentation (all modules)
zig build docs-all

# Generate and get helpful output about documentation locations
zig build docs-open
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
- `src/scheduler.zig` - Heartbeat and predictive scheduling algorithms with memory-aware features
- `src/memory_pressure.zig` - Memory pressure monitoring and adaptive scheduling
- `src/pcall.zig` - Zero-overhead potentially parallel calls
- `src/coz.zig` - COZ profiler integration
- `src/testing.zig` - Enhanced parallel testing framework with resource validation
- `src/comptime_work.zig` - Compile-time work distribution patterns and optimization
- `src/easy_api.zig` - Progressive feature adoption API with development mode support
- `src/enhanced_errors.zig` - Comprehensive error handling with actionable solutions
- `src/fingerprint.zig` - Advanced task fingerprinting and classification
- `src/intelligent_decision.zig` - Multi-criteria worker selection and decision making
- `src/predictive_accounting.zig` - One Euro Filter-based execution time prediction
- `src/advanced_worker_selection.zig` - Optimized worker selection algorithms
- `src/simd.zig` - Cross-platform SIMD support and feature detection
- `src/simd_classifier.zig` - Advanced SIMD task classification and batch formation
- `src/gpu_integration.zig` - SYCL GPU integration and device management
- `src/gpu_classifier.zig` - Automatic GPU suitability detection and task routing
- `src/simd_batch.zig` - SIMD task batching architecture with type-safe vectorization
- `src/simd_queue.zig` - Vectorized queue operations and work-stealing integration
- `build_config.zig` - Build-time hardware detection and auto-configuration system
- `src/build_opts_new.zig` - Enhanced compile-time access to auto-detected system configuration
- `src/simd_benchmark.zig` - Comprehensive SIMD benchmarking and validation framework
- `src/sycl_wrapper.hpp/cpp` - SYCL C++ to C API wrapper with opaque pointer management
- `src/mock_sycl.zig` - Mock SYCL implementation for testing when runtime unavailable

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
- **Ultra-Optimized Task Processing Pipeline**
  - **Fast Path Execution**: 100% immediate execution for small tasks (16ns average)
  - **Work-Stealing Efficiency**: Improved from 40% to >90% for mixed workloads
  - **Task Submission Streamlining**: 333% improvement for mixed workloads, 1,354% for medium tasks
  - **Cache-Line Optimized Memory Layout**: 40% improvement through elimination of false sharing
- **SIMD Task Classification and Batch Formation**
  - **Real SIMD Vectorization**: 6-23x speedup over scalar implementations
  - **Cross-Platform SIMD Support**: SSE, AVX, AVX2, AVX-512, NEON, SVE with 1.35-1.94x speedup
  - **Intelligent Batch Formation**: 720x improvement (36μs → 0.05μs) with pre-warmed templates
  - **Task Addition Pipeline**: 24.3x improvement (583μs → 24μs) with template optimization
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
  - **Runtime parallel execution**: `parallelMap`, `parallelReduce`, `parallelFilter`, `distributeWork`
- **Build-time auto-configuration system** 
  - Hardware detection with CPU count, SIMD features, NUMA topology estimation
  - Automatic One Euro Filter parameter tuning based on hardware characteristics
  - Architecture-specific optimizations (x86_64, aarch64, etc.)
  - Intelligent defaults with manual override capability
  - **VERIFIED**: Seamless integration of build-time detection with runtime optimization
- **Enhanced error message system** with descriptive context and helpful suggestions
  - Specific error types instead of generic errors (e.g., `ParallelMapArraySizeMismatch`, `WorkStealingDequeFull`)
  - Common causes and root cause analysis for faster debugging
  - Helpful suggestions and workarounds for issue resolution
  - Platform-specific guidance (Linux/Windows thread affinity)
  - Self-documenting error conditions reducing need for external documentation
- **Memory-aware task scheduling** with Linux PSI (Pressure Stall Information) integration
  - Real-time memory pressure monitoring with 100ms update intervals
  - Adaptive task scheduling based on memory conditions (15-30% improvement for memory-intensive workloads)
  - 5-level pressure classification (none/low/medium/high/critical) with intelligent response strategies
  - Cross-platform memory utilization detection (Linux PSI, Windows/macOS fallbacks)
  - Integration with heartbeat scheduler for seamless memory pressure adaptation
  - **VERIFIED**: 20-40% reduction in memory pressure incidents, 10-25% improvement in NUMA memory locality
- **Development mode configuration** with comprehensive debugging features
  - Progressive feature adoption API: Basic → Performance → Advanced → Development
  - Comprehensive debugging: verbose logging, task tracing, memory debugging, deadlock detection
  - Specialized configurations: testing, profiling, custom development pools
  - Configuration analysis and validation with optimization recommendations
  - Enhanced Easy API with development-focused pool creation functions
- **SIMD task processing system** with intelligent classification and adaptive batch formation
  - Cross-platform SIMD capability detection (SSE, AVX, AVX2, AVX-512, NEON, SVE)
  - Multi-layered task classification (static analysis, dynamic profiling, ML, batch formation)
  - Intelligent batch formation with multi-criteria optimization
  - Comprehensive benchmarking and validation framework
  - **VERIFIED**: Cross-platform compatibility and performance improvements
- **GPU integration foundation** with SYCL C++ wrapper
  - Extern "C" interface with opaque pointer management for C++ objects
  - Exception handling and error code translation
  - Device discovery, capability analysis, and performance scoring
  - Task classification for optimal GPU routing decisions
  - Memory management interface and fallback mechanisms
  - **VERIFIED**: Seamless integration with graceful fallback when GPU unavailable

## Development Notes

### Testing Strategy
The project uses comprehensive testing including unit tests, integration tests, stress tests, and specialized tests for TLS overflow conditions and COZ profiling scenarios. The enhanced parallel testing framework (`src/testing.zig`) provides utilities for testing parallel code with automatic resource cleanup validation, performance constraints, and stress testing capabilities.

### Version Evolution
- **V1**: Basic work-stealing thread pool
- **V2**: Added heartbeat scheduling with token accounting
- **V3**: CPU topology awareness, NUMA optimization, One Euro Filter prediction, formal verification planning
- **V3.0**: Advanced predictive scheduling with 15.3x worker selection optimization and memory-safe caching
- **V3.0.1**: **Memory-aware task scheduling**, **development mode configuration**, **progressive feature adoption API**, **enhanced error handling**, **comprehensive debugging features**, **SIMD task processing**, **GPU integration foundation**
- **V3.1**: **Ultra-Performance Optimization Release**
  - **SIMD Task Processing**: Real vectorization with 6-23x speedup, cross-platform SIMD support
  - **Intelligent Batch Formation**: 720x improvement with machine learning-based classification
  - **Fast Path Execution**: 100% immediate execution for small tasks, >90% work-stealing efficiency
  - **Memory Layout Optimization**: Cache-line isolation eliminating false sharing (40% improvement)
  - **Task Submission Streamlining**: 333% mixed workload improvement, 1,354% medium task improvement
  - **Production-Ready Performance**: Near-optimal efficiency across all workload types

### Formal Verification
The project is working towards formal verification using Lean 4 theorem prover with LLM-assisted proof development for mathematical correctness guarantees of lock-free algorithms.

### Integration with Reverb
Originally developed for the Reverb HTTP server, achieving 9,110 req/s performance improvements through intelligent task scheduling and topology awareness.