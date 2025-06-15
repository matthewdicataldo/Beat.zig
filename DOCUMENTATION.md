# Beat.zig Documentation Guide

Beat.zig provides comprehensive auto-generated API documentation for both modular and bundle usage patterns.

## 📚 Generating Documentation

### Quick Start
```bash
# Generate main API documentation
zig build docs

# Generate bundle API documentation
zig build docs-bundle

# Generate all documentation
zig build docs-all
```

### Available Documentation

#### 🔧 **Modular API Documentation** (`zig build docs`)
- **Location**: `zig-out/docs/index.html`
- **Content**: Full API documentation for `src/core.zig` and all submodules
- **Use Case**: Recommended for performance-critical applications
- **Features**: Complete module breakdown, auto-configuration integration

#### 📦 **Bundle API Documentation** (`zig build docs-bundle`)
- **Location**: `zig-out/docs-bundle/index.html`  
- **Content**: Documentation for the single-file `beat.zig` bundle
- **Use Case**: Convenient for simple integration and prototyping
- **Features**: Simplified API surface, single import

## 🌟 Key Documentation Features

### Auto-Configuration Integration
The documentation automatically includes:
- **Hardware-detected settings** for your build machine
- **Optimal configuration parameters** based on CPU features
- **Platform-specific optimizations** and recommendations

### Module Coverage
Complete documentation for all major modules:
- **Core Thread Pool** (`src/core.zig`) - Main API and configuration
- **Lock-Free Data Structures** (`src/lockfree.zig`) - Chase-Lev deque, MPMC queue
- **CPU Topology** (`src/topology.zig`) - Hardware detection and affinity
- **Scheduler** (`src/scheduler.zig`) - Heartbeat and One Euro Filter
- **Memory Management** (`src/memory.zig`) - NUMA-aware allocation
- **Parallel Calls** (`src/pcall.zig`) - Zero-overhead abstractions
- **Compile-Time Work** (`src/comptime_work.zig`) - Compile-time optimization
- **Testing Framework** (`src/testing.zig`) - Enhanced parallel testing

### Performance Insights
Documentation includes verified performance characteristics:
- **Topology-aware work stealing**: 0.6-12.8% improvement
- **Smart worker selection**: ~50% execution time improvement  
- **One Euro Filter**: Superior task prediction vs simple averaging
- **Auto-configuration**: Hardware-optimized parameters automatically

## 🚀 Viewing Documentation

### Local Browser
```bash
# Open main documentation
firefox zig-out/docs/index.html

# Open bundle documentation  
firefox zig-out/docs-bundle/index.html

# On macOS
open zig-out/docs/index.html

# On Windows
start zig-out/docs/index.html
```

### Documentation Structure
```
zig-out/
├── docs/                  # Modular API docs
│   ├── index.html        # Main entry point
│   ├── main.js           # Interactive features
│   ├── main.wasm         # Search functionality
│   └── sources.tar       # Source code archive
└── docs-bundle/          # Bundle API docs
    ├── index.html        # Bundle entry point
    ├── main.js           
    ├── main.wasm         
    └── sources.tar       
```

## 🔍 Navigation Tips

### Finding Key APIs
- **Thread Pool Creation**: Look for `createPool`, `createOptimalPool`
- **Configuration**: Search for `Config` struct and auto-tuned defaults
- **Advanced Features**: Browse individual modules like `scheduler`, `topology`
- **Examples**: Check function documentation for usage examples

### Performance Features
- **Auto-Configuration**: `build_opts` module for hardware detection
- **One Euro Filter**: `scheduler.TaskPredictor` for adaptive prediction
- **Topology Awareness**: `topology.CpuTopology` for NUMA optimization
- **Work Stealing**: Core thread pool implementation details

## 📋 Integration Examples

The documentation includes examples for:
- Basic thread pool setup with auto-configuration
- Custom configuration for specialized workloads  
- NUMA-aware task placement and affinity hints
- Performance monitoring and statistics collection
- Advanced scheduling with One Euro Filter tuning

## 🛠️ Development Workflow

For contributors:
```bash
# Generate docs during development
zig build docs && firefox zig-out/docs/index.html

# Verify documentation builds cleanly
zig build docs-all

# Check both API surfaces
zig build docs && zig build docs-bundle
```

---

**Beat.zig: Ultra-optimized parallelism with comprehensive documentation! 📚✨**

For the most up-to-date API information, always regenerate documentation from your local build as it includes auto-detected hardware optimizations specific to your system.