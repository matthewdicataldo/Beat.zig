# Beat.zig Integration Guide for External Projects

## 🚀 **Quick Start for External Projects**

Beat.zig V4+ has evolved significantly with advanced predictive scheduling and build-time auto-configuration. This guide addresses common integration challenges and provides clear patterns for consuming Beat as a dependency.

## 📋 **Build System Integration Answers**

### **1. Module Export Strategy**

**Recommended Approach**: Use the **"beat" module** (bundle) for most external projects:

```zig
// In your build.zig
const beat_dep = b.dependency("beat", .{
    .target = target,
    .optimize = optimize,
});

// Use the beat module (recommended)
exe.root_module.addImport("beat", beat_dep.module("beat"));
```

**Why beat module over zigpulse**:
- ✅ **Complete feature set** with all modules bundled
- ✅ **Cleaner API** with consistent imports
- ✅ **Better compatibility** with build options

### **2. Build Options Dependency - SOLUTION**

The `build_config` issue has a simple solution. Add this to your **project's build.zig**:

```zig
// In your project's build.zig
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    
    // SOLUTION: Create build_config for Beat dependency
    const beat_build_config = b.addModule("build_config", .{
        .root_source_file = b.path("beat_config.zig"), // See template below
    });
    
    const beat_dep = b.dependency("beat", .{
        .target = target,
        .optimize = optimize,
    });
    
    const exe = b.addExecutable(.{
        .name = "your-app",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Import Beat with build config
    exe.root_module.addImport("beat", beat_dep.module("beat"));
    exe.root_module.addImport("build_config", beat_build_config);
    
    b.installArtifact(exe);
}
```

### **3. Build Config Template**

Create `beat_config.zig` in your project root:

```zig
// beat_config.zig - Minimal configuration for Beat dependency
pub const BuildConfig = struct {
    // Auto-detected hardware defaults (fallbacks)
    pub const hardware = struct {
        pub const cpu_count: u32 = 8;           // Fallback CPU count
        pub const optimal_workers: u32 = 6;     // Typically cpu_count - 2
        pub const numa_nodes: u32 = 1;          // Conservative default
        pub const cache_line_size: u32 = 64;    // Standard x86_64/ARM64
        pub const has_avx2: bool = false;       // Conservative default
        pub const has_avx512: bool = false;     // Conservative default
        pub const memory_gb: u32 = 8;           // Conservative default
    };
    
    // One Euro Filter defaults (tuned for general workloads)
    pub const one_euro = struct {
        pub const frequency: f64 = 60.0;        // 60 Hz update rate
        pub const min_cutoff: f64 = 1.0;        // Minimum cutoff frequency
        pub const beta: f64 = 0.007;            // Speed coefficient
        pub const d_cutoff: f64 = 1.0;          // Derivative cutoff
    };
    
    // Performance settings
    pub const perf = struct {
        pub const optimal_queue_size: u32 = hardware.optimal_workers * 32;
        pub const enable_numa_optimization: bool = hardware.numa_nodes > 1;
        pub const enable_simd: bool = hardware.has_avx2 or hardware.has_avx512;
    };
};
```

## ⚙️ **API Compatibility Solutions**

### **4. Static Configuration Mode**

Yes! Beat supports static configuration without build-time detection:

```zig
const beat = @import("beat");

// Static configuration - no build-time detection needed
const config = beat.Config{
    .num_workers = 4,                        // Static worker count
    .enable_predictive = false,              // Disable advanced features
    .enable_advanced_selection = false,      // Use simple worker selection
    .enable_topology_aware = false,          // Disable NUMA optimization
    .enable_lock_free = true,               // Keep lock-free performance
    .task_queue_size = 128,                 // Static queue size
};

var pool = try beat.ThreadPool.init(allocator, config);
defer pool.deinit();
```

### **5. Backwards Compatibility Mode**

**Legacy API Pattern** for simple use cases:

```zig
const beat = @import("beat");

pub fn simpleBeatSetup(allocator: std.mem.Allocator, worker_count: u32) !*beat.ThreadPool {
    const simple_config = beat.Config{
        .num_workers = worker_count,
        .enable_predictive = false,          // ✅ Disable advanced features
        .enable_advanced_selection = false,  // ✅ Simple round-robin
        .enable_topology_aware = true,       // ✅ Keep basic NUMA awareness
        .enable_lock_free = true,           // ✅ Keep performance
        .enable_heartbeat = false,          // ✅ Disable heartbeat scheduling
    };
    
    return try beat.ThreadPool.init(allocator, simple_config);
}
```

### **6. Feature Flags at Compile Time**

Beat respects config flags to disable advanced features:

```zig
// Minimal Beat configuration - no advanced dependencies
const minimal_config = beat.Config{
    .num_workers = 4,
    .enable_predictive = false,              // No fingerprinting/prediction
    .enable_advanced_selection = false,      // No intelligent decisions
    .enable_topology_aware = false,          // No build_config dependency
    .enable_lock_free = true,               // Keep core performance
};
```

## 🔧 **Integration Patterns**

### **7. Recommended Git Submodule Integration**

**Best Practice Pattern**:

```bash
# Add Beat as submodule
git submodule add https://github.com/your-org/Beat.zig libs/beat
git submodule update --init --recursive
```

**build.zig.zon** (if using package manager):
```zig
.{
    .name = "your-project",
    .version = "1.0.0",
    .dependencies = .{
        .beat = .{
            .url = "https://github.com/your-org/Beat.zig/archive/main.tar.gz",
            .hash = "1234...", // Get with `zig fetch`
        },
    },
}
```

**build.zig** integration:
```zig
const beat_dep = b.dependency("beat", .{
    .target = target,
    .optimize = optimize,
});

exe.root_module.addImport("beat", beat_dep.module("beat"));
```

### **8. Complete Minimal Example**

**build.zig**:
```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    
    // Create minimal build config for Beat
    const beat_config = b.addModule("build_config", .{
        .root_source_file = b.path("beat_config.zig"),
    });
    
    const exe = b.addExecutable(.{
        .name = "my-app",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Option A: If using as dependency
    const beat_dep = b.dependency("beat", .{});
    exe.root_module.addImport("beat", beat_dep.module("beat"));
    
    // Option B: If using as submodule
    // const beat_module = b.addModule("beat", .{
    //     .root_source_file = b.path("libs/beat/beat.zig"),
    // });
    // exe.root_module.addImport("beat", beat_module);
    
    exe.root_module.addImport("build_config", beat_config);
    
    b.installArtifact(exe);
}
```

**src/main.zig**:
```zig
const std = @import("std");
const beat = @import("beat");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Simple Beat configuration - no advanced features
    const config = beat.Config{
        .num_workers = 4,
        .enable_predictive = false,          // Disable to avoid build_config issues
        .enable_advanced_selection = false,  // Use simple worker selection
        .enable_topology_aware = false,      // Disable NUMA detection
    };
    
    var pool = try beat.ThreadPool.init(allocator, config);
    defer pool.deinit();
    
    // Submit work as usual
    const task = beat.Task{
        .func = myWorkFunction,
        .data = @ptrCast(&myData),
    };
    
    try pool.submit(task);
    pool.wait();
}

fn myWorkFunction(data: *anyopaque) void {
    const value = @as(*u32, @ptrCast(@alignCast(data)));
    value.* *= 2;
}
```

## 🛠️ **Specific Error Resolution**

### **9. "no module named 'build_config'" Fix**

This error occurs when Beat's advanced features are enabled but `build_config` isn't provided. **Two solutions**:

**Solution A**: Provide build_config (recommended)
```zig
// Add to your build.zig
const beat_config = b.addModule("build_config", .{
    .root_source_file = b.path("beat_config.zig"), // Use template above
});
exe.root_module.addImport("build_config", beat_config);
```

**Solution B**: Disable advanced features
```zig
const config = beat.Config{
    .enable_predictive = false,       // ✅ Fixes build_config dependency
    .enable_advanced_selection = false,
    .enable_topology_aware = false,
    // ... other basic settings
};
```

### **10. Auto-Configuration Graceful Fallback**

Beat's auto-configuration **does** fallback gracefully when build-time detection fails:

```zig
// In Beat's core - automatic fallbacks are built-in
const hardware_defaults = struct {
    const cpu_count = 8;           // Conservative fallback
    const optimal_workers = 6;     // Safe default
    const cache_line_size = 64;    // Universal standard
    const numa_nodes = 1;          // Single NUMA fallback
};
```

**Enable graceful fallback mode**:
```zig
const config = beat.Config{
    .num_workers = null,           // Use auto-detected or fallback
    .enable_predictive = false,    // Disable complex features
    .task_queue_size = null,       // Use computed default
};
```

## 🎯 **Migration Checklist**

### **From Beat V1/V2 to V4+**:

1. ✅ **Add beat_config.zig** using template above
2. ✅ **Update build.zig** to provide build_config module  
3. ✅ **Set enable_predictive = false** for simple use cases
4. ✅ **Use beat module** instead of zigpulse module
5. ✅ **Test with minimal config** first, then add features

### **Recommended Feature Adoption Path**:

1. **Week 1**: Basic thread pool with `enable_predictive = false`
2. **Week 2**: Add topology awareness with `enable_topology_aware = true`
3. **Week 3**: Enable predictive scheduling with proper build_config
4. **Week 4**: Explore advanced worker selection and caching

## 📞 **Support & Common Issues**

### **Quick Troubleshooting**:

| Error | Solution |
|-------|----------|
| `no module named 'build_config'` | Add beat_config.zig and import build_config module |
| `undefined symbol: optimal_workers` | Set `num_workers` explicitly in Config |
| Complex build dependencies | Use `enable_predictive = false` mode |
| Performance issues | Enable `enable_lock_free = true` |

### **Performance Recommendations**:

- ✅ **Always enable** `enable_lock_free = true` for performance
- ✅ **Start simple** with predictive features disabled
- ✅ **Add topology awareness** after basic integration works
- ✅ **Monitor with** existing COZ profiler integration

## 🚀 **Next Steps**

1. **Try the minimal example** above first
2. **Gradually enable features** as integration stabilizes  
3. **Use static configuration** until auto-detection is needed
4. **Explore advanced features** after basic integration works

Beat.zig V4+ provides **significant performance improvements** (15.3x worker selection optimization) while maintaining **backwards compatibility** through configuration flags. The integration patterns above solve all common dependency issues while providing a clear upgrade path to advanced features.