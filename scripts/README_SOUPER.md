# Enhanced Souper Integration Scripts

This directory contains comprehensive tooling for integrating Google's Souper superoptimizer with Beat.zig's performance-critical algorithms.

## Overview

The enhanced setup includes **progress monitoring**, **background execution**, and **automated analysis pipelines** to handle the 30-60 minute Souper build process efficiently.

## Scripts

### 1. `setup_souper.sh` - Enhanced Setup Script

**Main setup script with comprehensive progress monitoring and background execution support.**

#### Features
- ✅ **Real-time progress tracking** with percentage completion
- ✅ **Background execution** with `nohup` support  
- ✅ **Detailed logging** with colored output and timestamps
- ✅ **Progress file monitoring** for external status checks
- ✅ **Enhanced error handling** with actionable feedback
- ✅ **Disk space and memory validation** before build starts
- ✅ **Build progress extraction** from compiler output

#### Usage

```bash
# Interactive setup (blocks terminal for 30-60 minutes)
./scripts/setup_souper.sh

# Background setup with monitoring (recommended)
./scripts/setup_souper.sh --background

# Check progress of background setup
./scripts/setup_souper.sh --check-progress

# Show help
./scripts/setup_souper.sh --help
```

#### Background Execution Workflow

```bash
# 1. Start setup in background
./scripts/setup_souper.sh --background

# 2. Monitor progress (multiple options)
./scripts/monitor_souper_progress.sh -i          # Interactive monitoring
watch -n 5 './scripts/monitor_souper_progress.sh -q'  # Auto-refresh status
tail -f souper_setup_*.log                      # Live log viewing

# 3. Check completion
./scripts/setup_souper.sh --check-progress
```

#### Progress Tracking

The script generates two monitoring files:
- **`souper_progress.txt`** - Current step, percentage, status, timestamp
- **`souper_setup_*.log`** - Detailed execution log with colored output
- **`souper_setup.pid`** - Background process ID for monitoring

### 2. `monitor_souper_progress.sh` - Dedicated Progress Monitor

**Real-time monitoring tool with interactive and batch modes.**

#### Features
- ✅ **Interactive monitoring** with auto-refresh every 5 seconds
- ✅ **Quick status snapshots** for scripting and automation
- ✅ **Colored progress indicators** with visual progress bar
- ✅ **Process health monitoring** (CPU/memory usage)
- ✅ **Recent log analysis** with syntax highlighting
- ✅ **Disk space monitoring** during build process

#### Usage

```bash
# Interactive monitoring (recommended)
./scripts/monitor_souper_progress.sh -i

# Quick status snapshot
./scripts/monitor_souper_progress.sh -q

# View recent logs only
./scripts/monitor_souper_progress.sh -l

# Auto-refreshing status (alternative to interactive)
watch -n 5 './scripts/monitor_souper_progress.sh -q'
```

#### Sample Output

```
=== Beat.zig Souper Setup Progress Monitor ===
Updated: 2024-01-15 14:30:25

✓ Setup process is running (PID: 12345)
  CPU/Memory usage: 95.2 8.4

Progress: 4/8 (50%)
Current Task: Build Shared LLVM
Status: RUNNING
Last Update: 2024-01-15 14:30:20

[==========================---------------------] 50%

=== Disk Space Usage ===
Available: 45G (Used: 12G of 100G)

=== Recent Log Entries ===
[INFO] Building LLVM (using 8 parallel jobs)...
[INFO] [42/847] Building CXX object lib/Support/CMakeFiles/LLVMSupport.dir/APInt.cpp.o
[INFO] [43/847] Building CXX object lib/Support/CMakeFiles/LLVMSupport.dir/APSInt.cpp.o
```

### 3. `run_souper_analysis.sh` - Automated Analysis Pipeline

**Comprehensive analysis workflow for systematic superoptimization.**

#### Key Features
- ✅ **Prioritized module analysis** (high/medium priority)
- ✅ **Timeout management** for long-running analyses
- ✅ **Comprehensive reporting** with markdown output
- ✅ **Statistics collection** and optimization counting
- ✅ **Whole-program analysis** support
- ✅ **Flexible execution modes** (full/quick/single module)

#### Analysis Targets

**High Priority (5 minutes timeout each):**
- `fingerprint` - Task hashing algorithms
- `lockfree` - Work-stealing deque bit operations  
- `scheduler` - Token accounting logic
- `simd` - SIMD capability detection

**Medium Priority (2 minutes timeout each):**
- `simd_classifier` - Feature vector similarity
- `simd_batch` - Task compatibility scoring
- `advanced_worker_selection` - Worker selection algorithms
- `topology` - CPU topology distance calculations

#### Usage Examples

```bash
# Source the environment first
source souper_env.sh

# Full analysis (all modules + whole-program)
./scripts/run_souper_analysis.sh

# Quick analysis (high-priority modules only)
./scripts/run_souper_analysis.sh -q

# Analyze specific module
./scripts/run_souper_analysis.sh -m fingerprint

# Whole-program analysis only
./scripts/run_souper_analysis.sh -w
```

## File Structure

```
scripts/
├── setup_souper.sh              # Enhanced setup with progress monitoring
├── monitor_souper_progress.sh   # Dedicated progress monitor
├── run_souper_analysis.sh       # Automated analysis pipeline
└── README_SOUPER.md             # This documentation

# Generated files during setup:
├── souper_progress.txt          # Real-time progress tracking
├── souper_setup_*.log          # Detailed execution logs
├── souper_setup.pid            # Background process ID
└── souper_env.sh               # Environment setup script

# Generated during analysis:
souper_analysis_results/
├── analysis_*.log              # Analysis execution logs
├── souper_*_*.txt             # Individual module results
├── stats_*_*.txt              # Statistics and summaries
└── comprehensive_report_*.md   # Markdown analysis report
```

## Integration with Beat.zig Development

### 1. **Setup Phase** (One-time, 30-60 minutes)

```bash
# Start background setup
cd /path/to/Beat.zig
./scripts/setup_souper.sh --background

# Monitor in another terminal
./scripts/monitor_souper_progress.sh -i
```

### 2. **Analysis Phase** (5-15 minutes)

```bash
# Load environment
source souper_env.sh

# Generate LLVM IR for all targets
zig build souper-all

# Run comprehensive analysis
./scripts/run_souper_analysis.sh

# Review results
cat souper_analysis_results/comprehensive_report_*.md
```

### 3. **Implementation Phase** (Variable)

```bash
# Review specific optimizations
cat souper_analysis_results/souper_fingerprint_*.txt

# Implement changes in source code
# Re-run benchmarks to validate improvements
zig build bench

# Re-analyze to verify optimization integration
./scripts/run_souper_analysis.sh -m fingerprint
```

## Performance Expectations

### Setup Time
- **Interactive**: 30-60 minutes (blocks terminal)
- **Background**: 30-60 minutes (non-blocking with progress monitoring)
- **Prerequisites check**: < 1 minute
- **Environment generation**: < 10 seconds

### Analysis Time
- **Single module**: 30 seconds - 5 minutes
- **High-priority modules**: 15-20 minutes total
- **All modules**: 25-35 minutes total
- **Whole-program**: 5-10 minutes (varies significantly by program size)

### Resource Usage
- **CPU**: Near 100% during LLVM build (expected)
- **Memory**: 4-8GB peak during linking phases
- **Disk**: ~10GB total (LLVM build artifacts)

## Troubleshooting

### Setup Issues
```bash
# Check prerequisites
./scripts/setup_souper.sh --help

# Monitor background process
ps aux | grep souper
cat souper_setup_*.log | tail -20

# Disk space issues
df -h .
du -sh third_party/
```

### Analysis Issues
```bash
# Verify environment
which souper z3
echo $SOUPER_PREFIX

# Check LLVM IR generation
zig build souper-fingerprint
ls -la zig-out/lib/beat_souper_*.bc

# Test individual analysis
souper -z3-path=$(which z3) zig-out/lib/beat_souper_fingerprint.bc
```

### Process Management
```bash
# Kill stuck background setup
cat souper_setup.pid | xargs kill

# Clean up artifacts
rm -f souper_progress.txt souper_setup.pid souper_setup_*.log

# Fresh start
rm -rf third_party/souper third_party/shared-llvm
```

## Advanced Usage

### Custom Analysis Targets
```bash
# Add new module to build.zig souper targets
# Update run_souper_analysis.sh TARGETS array
# Run analysis on custom module
./scripts/run_souper_analysis.sh -m custom_module
```

### Integration with CI/CD
```bash
# Non-interactive setup for automation
./scripts/setup_souper.sh --background
./scripts/setup_souper.sh --check-progress

# Automated analysis with result validation
./scripts/run_souper_analysis.sh -q
grep "Total optimizations found:" souper_analysis_results/comprehensive_report_*.md
```

### Performance Integration
```bash
# Before optimization implementation
zig build bench > before_optimization.txt

# After implementing Souper-discovered optimizations  
zig build bench > after_optimization.txt

# Compare results
diff before_optimization.txt after_optimization.txt
```

This enhanced tooling provides comprehensive support for integrating formal superoptimization analysis into Beat.zig's development workflow, with robust progress monitoring and automation capabilities for long-running build processes.