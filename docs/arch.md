# ZigPulse v3 Architecture

## Overview

ZigPulse v3 is a high-performance parallelism library for Zig that combines traditional thread pool concepts with cutting-edge techniques from parallel computing research. The architecture prioritizes cache efficiency, minimal overhead, and hardware-aware scheduling.

## System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        Application Layer                       │
│                    (User Tasks and Workloads)                  │
└────────────────────────────────────────────────────────────────┘
                                 │
┌────────────────────────────────────────────────────────────────┐
│                         ZigPulse Core                          │
│  ┌─────────────┐  ┌───────────────┐  ┌────────────────────┐    │
│  │  ThreadPool │  │   Scheduler   │  │  Memory Manager    │    │
│  │             │  │               │  │                    │    │
│  │ • Workers   │  │ • Heartbeat   │  │ • Lock-free pools  │    │
│  │ • Queues    │  │ • Predictive  │  │ • NUMA-aware       │    │
│  │ • Statistics│  │ • Token mgmt  │  │ • Cache-aligned    │    │
│  └─────────────┘  └───────────────┘  └────────────────────┘    │
└────────────────────────────────────────────────────────────────┘
                                 │
┌────────────────────────────────────────────────────────────────┐
│                      Hardware Abstraction                      │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │   Topology   │  │  Lock-Free   │  │    Platform        │    │
│  │              │  │              │  │                    │    │
│  │ • CPU detect │  │ • WS Deque   │  │ • Thread affinity  │    │
│  │ • Cache info │  │ • MPMC Queue │  │ • Memory binding   │    │
│  │ • NUMA nodes │  │ • Atomics    │  │ • OS abstractions  │    │
│  └──────────────┘  └──────────────┘  └────────────────────┘    │
└────────────────────────────────────────────────────────────────┘
```

## Module Structure

### Core Module (`src/core.zig`)
The main entry point and coordinator for all subsystems.

**Responsibilities:**
- Thread pool lifecycle management
- Worker thread coordination
- Task submission and distribution
- Configuration management
- Statistics aggregation

**Key Design Decisions:**
- Hybrid queue support (mutex vs lock-free)
- Pluggable scheduler architecture
- Optional subsystem initialization

### Lock-Free Module (`src/lockfree.zig`)
High-performance concurrent data structures.

**Components:**
1. **Work-Stealing Deque** (Chase-Lev Algorithm)
   - Owner has exclusive bottom access
   - Thieves compete for top access
   - Grows dynamically with circular buffer
   - Memory reclamation via hazard pointers

2. **MPMC Queue**
   - Multiple producer, multiple consumer
   - Fixed-size circular buffer
   - Cache-padded for false sharing prevention

**Performance Characteristics:**
- Push/pop: O(1) amortized
- Steal: O(1) with low contention
- Memory overhead: 2x capacity for growth

### Topology Module (`src/topology.zig`)
Hardware detection and affinity management.

**Features:**
- CPU core enumeration
- Cache hierarchy detection
- NUMA node discovery
- SMT sibling identification
- Thread affinity control

**Platform Support:**
- Linux: Full support via /sys/devices/system/cpu
- Windows: Basic support (planned)
- macOS: Basic support (planned)

### Memory Module (`src/memory.zig`)
Allocation strategies for hot paths.

**Components:**
1. **Task Pool Allocator**
   - Pre-allocated task objects
   - Lock-free free list
   - Cache-line aligned blocks
   - Zero-overhead in steady state

2. **NUMA-Aware Allocator**
   - Per-node memory pools
   - Affinity-based allocation
   - Cross-node migration tracking

### Scheduler Module (`src/scheduler.zig`)
Task scheduling intelligence.

**Strategies:**
1. **Heartbeat Scheduling**
   - 100μs interval by default
   - Token-based promotion
   - Work/overhead ratio tracking

### PCAll Module (`src/pcall.zig`)
Potentially parallel function calls.

**Optimization Levels:**
1. **Standard**: ~5ns overhead
2. **Minimal**: 0 cycles in release mode
3. **Adaptive**: Runtime threshold adjustment

## Data Flow

### Task Submission Path
```
User Code
    │
    ├─> Task Creation
    │     └─> Affinity hint calculation
    │
    ├─> Worker Selection
    │     ├─> NUMA locality check
    │     ├─> Queue depth analysis
    │     └─> Load balancing
    │
    └─> Queue Insertion
          ├─> Priority ordering
          ├─> Memory pool allocation
          └─> Statistics update
```

### Task Execution Path
```
Worker Thread
    │
    ├─> Local Queue Check
    │     └─> LIFO pop for cache locality
    │
    ├─> Work Stealing (if empty)
    │     ├─> Victim selection (topology-aware)
    │     ├─> FIFO steal for fairness
    │     └─> Backoff on failure
    │
    └─> Task Execution
          ├─> Function invocation
          ├─> Error handling
          └─> Statistics recording
```

## Memory Layout

### Cache-Line Alignment
All hot data structures are aligned to 64-byte boundaries:

```zig
pub const Worker = struct {
    // Hot data (first cache line)
    id: u32,                    // 4 bytes
    running: bool,              // 1 byte
    _pad1: [3]u8,              // 3 bytes padding
    queue: WorkQueue,          // 8 bytes (pointer)
    stats: LocalStats,         // 48 bytes
    
    // Cold data (second cache line)
    thread: Thread,            // 8 bytes
    pool: *ThreadPool,         // 8 bytes
    cpu_affinity: CpuSet,      // 16 bytes
    // ...
} align(64);
```

### NUMA Considerations
Memory allocation follows these principles:
1. Task data allocated on submitter's node
2. Queue structures on worker's node
3. Shared data on interleaved pages
4. Statistics on local nodes

## Synchronization Strategies

### Lock-Free Algorithms
1. **Compare-and-Swap (CAS)**
   - Used for atomic updates
   - Retry loops with exponential backoff
   - ABA problem prevention via tagging

2. **Memory Ordering**
   - Acquire-Release for synchronization
   - Relaxed for statistics
   - Sequential consistency sparingly

3. **Hazard Pointers**
   - Safe memory reclamation
   - Grace period tracking
   - Bounded memory usage

### Mutual Exclusion (Fallback)
When lock-free isn't suitable:
- Complex multi-step operations
- Initialization/shutdown paths
- Error recovery scenarios

## Performance Optimizations

### CPU-Level
1. **Branch Prediction**
   - Hot paths use likely/unlikely hints
   - Jump tables for dispatch
   - Predictable loop structures

2. **Cache Optimization**
   - Data structure padding
   - Hot/cold data separation
   - Prefetch hints in loops

3. **SIMD Usage**
   - Bulk memory operations
   - Parallel comparisons
   - Future: Task batching

### System-Level
1. **NUMA Optimization**
   - Local memory allocation
   - Minimal cross-node traffic
   - Replication of read-only data

2. **Thread Affinity**
   - Pin workers to cores
   - Avoid migration overhead
   - Respect CPU topology

3. **System Call Reduction**
   - Batch operations
   - User-space spinning
   - Adaptive sleep strategies

## Extensibility Points

### Custom Schedulers
Implement the Scheduler interface:
```zig
pub const SchedulerVTable = struct {
    shouldPromote: *const fn (*Scheduler) bool,
    selectWorker: *const fn (*Scheduler, Task) u32,
    updateStats: *const fn (*Scheduler, Stats) void,
};
```

### Custom Memory Allocators
Implement the Allocator interface:
```zig
pub const AllocatorVTable = struct {
    alloc: *const fn (*Allocator, usize) ?*anyopaque,
    free: *const fn (*Allocator, *anyopaque) void,
    reset: *const fn (*Allocator) void,
};
```

### Platform Abstractions
Add new platforms by implementing:
```zig
pub const PlatformVTable = struct {
    detectTopology: *const fn () Topology,
    setAffinity: *const fn (Thread, []const u32) !void,
    getCurrentCpu: *const fn () u32,
};
```
