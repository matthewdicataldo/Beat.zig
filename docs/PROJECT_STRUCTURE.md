# Beat.zig Project Structure

This document outlines the organization of the Beat.zig repository for optimal development and maintenance.

## Directory Structure

### Core Implementation
```
src/                          # Core library implementation
├── core.zig                  # Main thread pool and API entry point
├── lockfree.zig             # Lock-free data structures (Chase-Lev deque, MPMC queue)
├── topology.zig             # CPU topology detection and thread affinity
├── memory.zig               # Memory pools and NUMA-aware allocation
├── scheduler.zig            # Heartbeat and predictive scheduling algorithms
├── memory_pressure.zig      # Memory pressure monitoring and adaptive scheduling
├── pcall.zig                # Zero-overhead potentially parallel calls
├── coz.zig                  # COZ profiler integration
├── testing.zig              # Enhanced parallel testing framework
├── comptime_work.zig        # Compile-time work distribution patterns
├── easy_api.zig             # Progressive feature adoption API
├── enhanced_errors.zig      # Comprehensive error handling
├── fingerprint.zig          # Advanced task fingerprinting and classification
├── intelligent_decision.zig # Multi-criteria worker selection
├── predictive_accounting.zig # One Euro Filter-based execution time prediction
├── advanced_worker_selection.zig # Optimized worker selection algorithms
├── simd.zig                 # Cross-platform SIMD support
├── simd_classifier.zig      # Advanced SIMD task classification
├── simd_batch.zig          # SIMD task batching architecture
├── simd_queue.zig          # Vectorized queue operations
├── simd_benchmark.zig      # Comprehensive SIMD benchmarking
├── gpu_integration.zig     # SYCL GPU integration and device management
├── gpu_classifier.zig      # Automatic GPU suitability detection
├── sycl_wrapper.hpp/cpp    # SYCL C++ to C API wrapper
└── mock_sycl.zig           # Mock SYCL implementation for testing
```

### Build System
```
build.zig                    # Primary build configuration
build_config.zig            # Build-time hardware detection and auto-configuration
beat.zig                    # Bundle file (single import convenience)
```

### Tests
```
tests/                       # Comprehensive test suite
├── test_*.zig              # Individual component tests
└── ...                     # Performance validation and integration tests
```

### Examples and Demos
```
examples/                    # Usage examples and demonstrations
├── comprehensive_demo.zig   # Complete feature demonstration
├── modular_usage.zig       # Modular import examples
├── single_file_usage.zig   # Bundle usage examples
├── souper_tests/           # Superoptimization test cases
├── auto_config_integration_demo.zig # Build-time configuration demo
├── batch_formation_profile.zig     # SIMD batch formation profiling
├── comptime_work_demo.zig          # Compile-time work distribution
└── ...                     # Additional examples and analysis tools
```

### Benchmarks
```
benchmarks/                  # Performance measurement suite
├── benchmark.zig           # Primary benchmark suite
├── benchmark_coz.zig       # COZ profiler benchmarks
├── benchmark_lockfree_vs_mutex.zig # Lock-free vs mutex comparison
├── benchmark_topology_*.zig # Topology-aware scheduling benchmarks
└── ...                     # Specialized performance tests
```

### Documentation
```
docs/                       # Documentation and guides
├── ARCHITECTURE.md         # System architecture overview
├── FORMAL_VERIFICATION.md  # Formal verification framework
├── PERFORMANCE.md          # Performance characteristics and tuning
├── PROJECT_STRUCTURE.md    # This file
└── archive/                # Historical and archived documentation
    ├── sopuer!.md          # Original Souper integration design
    ├── INTEGRATION_GUIDE.md # Legacy integration guide
    ├── ROADMAP.md          # Historical roadmap
    └── ...                 # Other archived documents
```

### Scripts and Automation
```
scripts/                    # Build and analysis automation
├── setup_souper.sh        # Enhanced Souper toolchain setup
├── monitor_souper_progress.sh # Real-time setup monitoring
├── run_souper_analysis.sh # Automated superoptimization analysis
├── README_SOUPER.md       # Souper integration documentation
├── amalgamate.zig         # Code amalgamation utility
└── profile_coz.sh         # COZ profiler execution script
```

### Generated Artifacts
```
artifacts/                  # Generated files and build artifacts
├── llvm_ir/               # LLVM IR files from superoptimization
│   ├── beat_souper_*.ll   # Human-readable LLVM IR
│   ├── beat_souper_*.bc   # LLVM bitcode files
│   └── lib*.a             # Compiled library artifacts
├── souper/                # Souper setup and analysis artifacts
│   ├── souper_*.log      # Setup progress logs
│   ├── souper_*.txt      # Progress tracking files
│   └── souper_*.pid      # Process ID files
└── jan-win-x64-0.5.17.exe # Stray executable (cleanup)
```

### Third-Party Dependencies
```
third_party/               # External dependencies and tools
└── souper/               # Google Souper superoptimizer
    ├── third_party/      # Souper's dependencies (Z3, LLVM, Alive2)
    ├── build/            # Souper build artifacts
    └── ...               # Souper source and configuration
```

## File Naming Conventions

### Source Files
- **Core modules**: Descriptive names (`core.zig`, `lockfree.zig`, `scheduler.zig`)
- **Feature modules**: Feature-prefixed (`simd_*.zig`, `gpu_*.zig`, `memory_*.zig`)
- **Test files**: `test_` prefix matching module names
- **Example files**: Descriptive names ending with purpose

### Generated Files
- **LLVM IR**: `beat_souper_[module].[ll|bc]` format
- **Souper artifacts**: `souper_` prefix with timestamp suffixes
- **Build artifacts**: Standard library naming (`lib*.a`, `*.o`)

### Documentation
- **Core docs**: ALL_CAPS.md for primary documentation
- **Archive docs**: Preserved original naming in `docs/archive/`
- **Specialized docs**: Descriptive names for specific topics

## Maintenance Guidelines

### Regular Cleanup
1. **Generated artifacts**: Move to `artifacts/` directory
2. **Temporary files**: Remove or add to `.gitignore`
3. **Documentation updates**: Archive outdated docs to `docs/archive/`
4. **Example organization**: Group by functionality in `examples/`

### Version Control
- **Ignore patterns**: Comprehensive `.gitignore` for all generated files
- **Artifact tracking**: Keep `artifacts/` structure but ignore contents
- **Documentation versioning**: Archive major documentation changes

### Development Workflow
1. **Source changes**: Direct edits in `src/`
2. **Test addition**: Add to `tests/` with matching names
3. **Example creation**: Add to appropriate `examples/` subdirectory
4. **Documentation**: Update relevant docs and archive old versions
5. **Cleanup**: Regular artifact organization and `.gitignore` updates

This structure ensures maintainable development while preserving important artifacts and providing clear organization for contributors and users.