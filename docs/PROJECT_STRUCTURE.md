# Beat.zig Project Structure

This document provides a comprehensive overview of the repository organization and file structure.

## 📁 **Root Directory**

```
Beat.zig/
├── build.zig               # Primary build configuration
├── build_config.zig        # Hardware auto-detection and build options
├── beat.zig                # Single-file bundle for convenience import
├── CLAUDE.md               # Development instructions and project context
├── README.md               # Main project documentation
├── ZIG-ISPC.md             # ISPC integration guide
└── profile_coz.sh          # COZ profiler automation script
```

## 📁 **Core Implementation** (`src/`)

### **Main Modules**
- **`core.zig`** - Primary API and thread pool implementation
- **`lockfree.zig`** - Lock-free data structures (Chase-Lev deque, MPMC queue)
- **`topology.zig`** - CPU topology detection and thread affinity
- **`memory.zig`** - Memory pools and NUMA-aware allocation
- **`scheduler.zig`** - Heartbeat and predictive scheduling algorithms

### **Advanced Features**
- **`simd.zig`** - Cross-platform SIMD support and feature detection
- **`simd_batch.zig`** - SIMD task batching architecture
- **`simd_classifier.zig`** - Advanced SIMD task classification
- **`batch_optimizer.zig`** - Ultra-optimized batch formation system
- **`prefetch.zig`** - Memory prefetching optimization utilities
- **`advanced_worker_selection.zig`** - Optimized worker selection algorithms

### **Intelligence & Machine Learning**
- **`a3c.zig`** - A3C reinforcement learning scheduler
- **`intelligent_decision.zig`** - Multi-criteria worker selection
- **`predictive_accounting.zig`** - One Euro Filter-based execution prediction
- **`fingerprint.zig`** - Advanced task fingerprinting and classification

### **Acceleration & Optimization**
- **`ispc_integration.zig`** - ISPC SPMD acceleration integration
- **`souper_integration.zig`** - Formal superoptimization with Google Souper
- **`mathematical_optimizations.zig`** - Formally verified optimizations

### **Developer Experience**
- **`easy_api.zig`** - Progressive feature adoption API
- **`enhanced_errors.zig`** - Comprehensive error handling
- **`testing.zig`** - Enhanced parallel testing framework
- **`memory_pressure.zig`** - Memory pressure monitoring and adaptation

## 📁 **Performance Validation** (`benchmarks/`)

### **Core Performance Benchmarks**
- **`benchmark_cache_alignment.zig`** - Cache-line alignment optimization validation
- **`benchmark_prefetching.zig`** - Memory prefetching performance measurement
- **`benchmark_batch_formation.zig`** - Batch formation optimization validation
- **`benchmark_worker_selection_optimized.zig`** - Worker selection optimization testing

### **Specialized Benchmarks**
- **`benchmark_lockfree_contention.zig`** - Lock-free queue contention analysis
- **`benchmark_soa_optimization.zig`** - Structure-of-arrays optimization testing
- **`benchmark_coz.zig`** - COZ profiler integration
- **`benchmark_ispc_performance.zig`** - ISPC vs native Zig performance comparison

## 📁 **Test Suite** (`tests/`)

### **Core Functionality Tests**
- **`test_simd_benchmark.zig`** - SIMD benchmarking framework validation
- **`test_simd_batch_architecture.zig`** - SIMD batching system testing
- **`test_advanced_worker_selection.zig`** - Worker selection algorithm testing
- **`test_predictive_accounting.zig`** - One Euro Filter prediction testing

### **Advanced Feature Tests**
- **`test_ispc_integration.zig`** - ISPC acceleration integration testing
- **`test_souper_integration.zig`** - Souper superoptimization testing
- **`test_development_mode.zig`** - Development mode configuration testing
- **`test_enhanced_errors.zig`** - Enhanced error handling validation

### **Performance & Integration Tests**
- **`test_topology_work_stealing.zig`** - Topology-aware work stealing validation
- **`test_thread_affinity_improved.zig`** - Thread affinity optimization testing
- **`test_parallel_work_runtime.zig`** - Runtime parallel work distribution testing

## 📁 **Examples & Demonstrations** (`examples/`)

### **Basic Usage**
- **`basic_usage.zig`** - Progressive API adoption demonstration
- **`modular_usage.zig`** - Individual module usage patterns
- **`single_file_usage.zig`** - Bundle file convenience import

### **Demonstrations** (`examples/demos/`)
- **`auto_config_integration_demo.zig`** - Hardware auto-detection showcase
- **`build_config_demo.zig`** - Build-time configuration options
- **`comptime_work_demo.zig`** - Compile-time work distribution patterns

### **Advanced Examples** (`examples/advanced/`)
- **`a3c_demo.zig`** - A3C reinforcement learning scheduler demonstration
- **`ml_memory_analysis.zig`** - Machine learning memory analysis
- **`souper_test.zig`** - Formal superoptimization integration

## 📁 **Documentation** (`docs/`)

### **Performance Documentation**
- **`PERFORMANCE_SUMMARY.md`** - Comprehensive optimization achievement report
- **`ARCHITECTURE.md`** - System architecture and design decisions
- **`PERFORMANCE.md`** - Performance characteristics and benchmarks

### **Integration Guides**
- **`ISPC_ACCELERATION_GUIDE.md`** - ISPC SPMD acceleration integration
- **`SOUPER_INTEGRATION.md`** - Formal superoptimization setup and usage
- **`A3C.md`** - A3C reinforcement learning scheduler documentation

### **Archived Documentation** (`docs/archive/`)
- **`performance/`** - Historical performance analysis reports
- **Legacy documentation** and outdated guides

## 📁 **Build Automation** (`scripts/`)

### **Souper Superoptimization**
- **`setup_souper.sh`** - Automated Souper toolchain setup
- **`run_souper_analysis.sh`** - Comprehensive optimization analysis
- **`monitor_souper_progress.sh`** - Setup progress monitoring

### **Development Tools**
- **`amalgamate.zig`** - Single-file bundle generation

## 📁 **ISPC Acceleration** (`src/kernels/`)

### **SPMD Kernels**
- **`fingerprint_similarity.ispc`** - Vectorized fingerprint comparison
- **`batch_optimization.ispc`** - Optimized batch formation kernels
- **`worker_selection.ispc`** - SPMD worker selection algorithms
- **`prediction_pipeline.ispc`** - One Euro Filter prediction acceleration

### **Research Kernels**
- **`advanced_ispc_research.ispc`** - Experimental SPMD research
- **`optimized_batch_kernels.ispc`** - Ultra-optimized mega-batch kernels

## 📁 **Generated Artifacts** (`artifacts/`)

### **Souper Analysis**
- **`llvm_ir/`** - Generated LLVM IR for superoptimization
- **`souper/`** - Souper setup and analysis results

## 📁 **Third-party Dependencies** (`third_party/`)

### **Google Souper**
- **`souper/`** - Complete Souper superoptimization toolchain

## 📁 **Build Outputs** (`zig-cache/`, `zig-out/`)

### **ISPC Compilation**
- **`zig-cache/ispc/`** - Compiled ISPC object files and headers

### **Documentation Generation**
- **`zig-out/docs/`** - Generated API documentation

## 🎯 **Key Design Principles**

### **Organization Philosophy**
1. **Clear Separation**: Core implementation, tests, examples, and documentation are clearly separated
2. **Progressive Complexity**: Examples progress from basic to advanced usage patterns
3. **Performance Focus**: Dedicated benchmarking and validation infrastructure
4. **Developer Experience**: Comprehensive documentation and easy-to-follow examples

### **File Naming Conventions**
- **Core modules**: `snake_case.zig` in `src/`
- **Tests**: `test_*.zig` in `tests/`
- **Benchmarks**: `benchmark_*.zig` in `benchmarks/`
- **Examples**: Descriptive names in `examples/`
- **Documentation**: `UPPERCASE.md` for major docs, `lowercase.md` for guides

### **Import Strategies**
1. **Bundle Import**: `@import("beat.zig")` - Single-file convenience
2. **Module Import**: `@import("beat")` - Clean modular approach
3. **Direct Import**: `@import("beat/core.zig")` - Fine-grained control

This structure supports both ease of use and advanced optimization while maintaining clear organization and comprehensive validation.