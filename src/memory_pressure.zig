const std = @import("std");
const builtin = @import("builtin");
const core = @import("core.zig");
const numa_mapping = @import("numa_mapping.zig");
const cgroup_detection = @import("cgroup_detection.zig");

// Memory-Aware Task Scheduling with Pressure Detection
//
// This module provides comprehensive memory pressure monitoring and response
// mechanisms to optimize task scheduling under varying memory conditions.
//
// Key Features:
// - Linux PSI (Pressure Stall Information) integration for real-time pressure detection
// - Cross-platform memory utilization monitoring with Windows/macOS fallbacks
// - Atomic pressure state updates for lock-free access
// - Integration with heartbeat scheduler for adaptive behavior
// - Configurable pressure thresholds and response strategies

// ============================================================================
// Core Types and Constants
// ============================================================================

/// Memory pressure levels with specific thresholds for scheduling decisions
pub const MemoryPressureLevel = enum(u8) {
    none = 0,       // < 10% pressure - normal operation
    low = 1,        // 10-30% pressure - prefer local NUMA, reduce batching
    medium = 2,     // 30-60% pressure - defer non-critical tasks, aggressive locality
    high = 3,       // 60-80% pressure - significant task deferral, emergency cleanup
    critical = 4,   // > 80% pressure - minimal task acceptance, force cleanup
    
    pub fn fromPercentage(pressure_pct: f32) MemoryPressureLevel {
        if (pressure_pct < 10.0) return .none;
        if (pressure_pct < 30.0) return .low;
        if (pressure_pct < 60.0) return .medium;
        if (pressure_pct < 80.0) return .high;
        return .critical;
    }
    
    pub fn shouldDeferTasks(self: MemoryPressureLevel) bool {
        return @intFromEnum(self) >= @intFromEnum(MemoryPressureLevel.medium);
    }
    
    pub fn shouldPreferLocalNUMA(self: MemoryPressureLevel) bool {
        return @intFromEnum(self) >= @intFromEnum(MemoryPressureLevel.low);
    }
    
    pub fn getTaskBatchLimit(self: MemoryPressureLevel, default_limit: u32) u32 {
        return switch (self) {
            .none => default_limit,
            .low => (default_limit * 3) / 4,      // 75% of normal
            .medium => default_limit / 2,          // 50% of normal
            .high => default_limit / 4,            // 25% of normal
            .critical => 1,                        // Minimal batching
        };
    }
};

/// Per-NUMA node memory metrics using logical NUMA node IDs
pub const NumaMemoryMetrics = struct {
    logical_numa_node: numa_mapping.LogicalNumaNodeId,
    memory_used_mb: u64 = 0,        // Memory used on this NUMA node (MB)
    memory_total_mb: u64 = 0,       // Total memory on this NUMA node (MB)
    memory_free_mb: u64 = 0,        // Free memory on this NUMA node (MB)
    memory_used_pct: f32 = 0.0,     // % memory used on this NUMA node
    
    /// Check if this NUMA node is under memory pressure
    pub fn isUnderPressure(self: NumaMemoryMetrics, threshold_pct: f32) bool {
        return self.memory_used_pct > threshold_pct;
    }
    
    /// Get available memory bandwidth estimate for this NUMA node
    pub fn estimateAvailableBandwidth(self: NumaMemoryMetrics, base_bandwidth_mbps: u64) u64 {
        // Reduce available bandwidth based on memory utilization
        // High memory usage typically correlates with increased memory traffic
        const utilization_factor = 1.0 - (self.memory_used_pct / 100.0);
        return @intFromFloat(@as(f64, @floatFromInt(base_bandwidth_mbps)) * utilization_factor);
    }
};

/// Real-time memory pressure metrics from Linux PSI with NUMA awareness
pub const MemoryPressureMetrics = struct {
    // PSI pressure percentages (0.0-100.0)
    some_avg10: f32 = 0.0,    // % of time at least one task stalled (10s average)
    some_avg60: f32 = 0.0,    // % of time at least one task stalled (60s average)
    some_avg300: f32 = 0.0,   // % of time at least one task stalled (300s average)
    full_avg10: f32 = 0.0,    // % of time all tasks stalled (10s average)
    full_avg60: f32 = 0.0,    // % of time all tasks stalled (60s average)
    full_avg300: f32 = 0.0,   // % of time all tasks stalled (300s average)
    
    // System memory utilization (platform-specific)
    memory_used_pct: f32 = 0.0,      // % of total memory used
    memory_available_mb: u64 = 0,     // Available memory in MB
    swap_used_pct: f32 = 0.0,         // % of swap space used
    
    // NUMA-aware memory metrics (using logical NUMA node IDs)
    numa_metrics: []NumaMemoryMetrics = &[_]NumaMemoryMetrics{},
    numa_metrics_available: bool = false,  // Whether per-NUMA metrics are available
    
    // Timestamps
    last_update_ns: u64 = 0,          // When metrics were last updated
    psi_available: bool = false,      // Whether PSI is available on this system
    
    /// Calculate overall memory pressure level using weighted PSI metrics
    pub fn calculatePressureLevel(self: *const MemoryPressureMetrics) MemoryPressureLevel {
        if (!self.psi_available) {
            // Fallback to memory utilization only
            return MemoryPressureLevel.fromPercentage(self.memory_used_pct);
        }
        
        // Weighted combination of PSI metrics (emphasize recent pressure)
        // some_avg10 is most important (recent pressure affecting at least one task)
        // full_avg10 is critical (all tasks affected)
        const weighted_pressure = (self.some_avg10 * 0.4) + 
                                 (self.some_avg60 * 0.2) + 
                                 (self.full_avg10 * 0.3) + 
                                 (self.memory_used_pct * 0.1);
        
        return MemoryPressureLevel.fromPercentage(weighted_pressure);
    }
    
    /// Check if pressure is increasing (trend analysis)
    pub fn isPressureIncreasing(self: *const MemoryPressureMetrics) bool {
        return self.some_avg10 > self.some_avg60 and self.some_avg60 > self.some_avg300;
    }
    
    /// Get the NUMA node with highest memory pressure
    pub fn getHighestPressureNumaNode(self: *const MemoryPressureMetrics) ?numa_mapping.LogicalNumaNodeId {
        if (!self.numa_metrics_available or self.numa_metrics.len == 0) return null;
        
        var highest_pressure: f32 = 0.0;
        var highest_node: ?numa_mapping.LogicalNumaNodeId = null;
        
        for (self.numa_metrics) |numa_metric| {
            if (numa_metric.memory_used_pct > highest_pressure) {
                highest_pressure = numa_metric.memory_used_pct;
                highest_node = numa_metric.logical_numa_node;
            }
        }
        
        return highest_node;
    }
    
    /// Get memory pressure level for a specific logical NUMA node
    pub fn getNumaNodePressureLevel(self: *const MemoryPressureMetrics, logical_numa: numa_mapping.LogicalNumaNodeId) ?MemoryPressureLevel {
        if (!self.numa_metrics_available) return null;
        
        for (self.numa_metrics) |numa_metric| {
            if (numa_metric.logical_numa_node == logical_numa) {
                return MemoryPressureLevel.fromPercentage(numa_metric.memory_used_pct);
            }
        }
        
        return null;
    }
    
    /// Get human-readable description of current state
    pub fn getDescription(self: *const MemoryPressureMetrics, allocator: std.mem.Allocator) ![]u8 {
        const level = self.calculatePressureLevel();
        const trend = if (self.isPressureIncreasing()) "↑" else "→";
        
        if (self.numa_metrics_available and self.numa_metrics.len > 0) {
            const numa_info = if (self.getHighestPressureNumaNode()) |highest_node|
                try std.fmt.allocPrint(allocator, ", Highest NUMA: {d}", .{highest_node})
            else
                try allocator.dupe(u8, "");
            defer allocator.free(numa_info);
            
            return std.fmt.allocPrint(allocator, 
                "Memory Pressure: {s} {s} (PSI: {d:.1}% some, {d:.1}% full, Mem: {d:.1}%{s})",
                .{ @tagName(level), trend, self.some_avg10, self.full_avg10, self.memory_used_pct, numa_info }
            );
        } else {
            return std.fmt.allocPrint(allocator, 
                "Memory Pressure: {s} {s} (PSI: {d:.1}% some, {d:.1}% full, Mem: {d:.1}%)",
                .{ @tagName(level), trend, self.some_avg10, self.full_avg10, self.memory_used_pct }
            );
        }
    }
};

/// Configuration for memory pressure monitoring with cross-platform cgroup detection
pub const MemoryPressureConfig = struct {
    // Monitoring intervals
    update_interval_ms: u32 = 100,           // How often to check pressure (100ms)
    
    // Legacy file path (deprecated - use auto-detection instead)
    psi_file_path: []const u8 = "/proc/pressure/memory",  // PSI file location
    
    // Pressure thresholds (can be tuned per workload)
    low_threshold: f32 = 10.0,               // Start preferring NUMA locality
    medium_threshold: f32 = 30.0,            // Begin task deferral
    high_threshold: f32 = 60.0,              // Aggressive memory management
    critical_threshold: f32 = 80.0,          // Emergency mode
    
    // Response configuration
    enable_task_deferral: bool = true,       // Allow deferring tasks under pressure
    enable_numa_preference: bool = true,     // Prefer local NUMA under pressure
    enable_batch_limiting: bool = true,      // Reduce batch sizes under pressure
    enable_cleanup_triggering: bool = true,  // Trigger cleanup callbacks under pressure
    
    // Cross-platform and container detection
    enable_auto_detection: bool = true,      // Enable automatic cgroup/container detection
    enable_cgroup_v2: bool = true,          // Enable cgroup v2 support
    enable_cgroup_v1: bool = true,          // Enable cgroup v1 support (fallback)
    enable_container_detection: bool = true, // Enable Docker/K8s/LXC detection
    
    // Platform fallbacks (enhanced)
    enable_meminfo_fallback: bool = true,    // Use /proc/meminfo on Linux without PSI
    enable_windows_fallback: bool = true,    // Use Windows memory APIs
    enable_macos_fallback: bool = true,      // Use macOS vm_stat equivalent
    enable_freebsd_fallback: bool = true,    // Use FreeBSD sysctl
    enable_generic_unix_fallback: bool = true, // Use generic Unix /proc
    
    // Reliability thresholds
    min_source_reliability: f32 = 0.3,      // Minimum reliability to use a source
    prefer_container_sources: bool = true,   // Prefer container-specific sources
    
    // Advanced features
    enable_numa_aware_detection: bool = true, // Per-NUMA node pressure detection
    enable_multi_source_validation: bool = false, // Cross-validate multiple sources
};

// ============================================================================
// Memory Pressure Monitor
// ============================================================================

/// Thread-safe memory pressure monitor with cross-platform cgroup detection and NUMA awareness
pub const MemoryPressureMonitor = struct {
    allocator: std.mem.Allocator,
    config: MemoryPressureConfig,
    
    // Protected state (using mutex for complex struct)
    metrics_mutex: std.Thread.Mutex = .{},
    current_metrics: MemoryPressureMetrics = .{},
    current_level: std.atomic.Value(MemoryPressureLevel),
    is_running: std.atomic.Value(bool),
    
    // Monitoring thread
    monitor_thread: ?std.Thread = null,
    
    // Cross-platform cgroup detection
    cgroup_detector: ?*cgroup_detection.CGroupDetector = null,
    detected_sources: []const cgroup_detection.MemoryPressureSourceConfig = &[_]cgroup_detection.MemoryPressureSourceConfig{},
    primary_source: ?cgroup_detection.MemoryPressureSourceConfig = null,
    
    // Platform-specific handles
    psi_file: ?std.fs.File = null,
    cgroup_v2_file: ?std.fs.File = null,        // cgroup v2 memory.pressure
    cgroup_v1_file: ?std.fs.File = null,        // cgroup v1 memory.usage_in_bytes
    meminfo_file: ?std.fs.File = null,          // /proc/meminfo cache
    
    // NUMA mapping integration - uses logical NUMA node IDs throughout
    numa_mapper: ?*numa_mapping.NumaMapper = null,
    
    pub fn init(allocator: std.mem.Allocator, config: MemoryPressureConfig) !*MemoryPressureMonitor {
        const self = try allocator.create(MemoryPressureMonitor);
        self.* = .{
            .allocator = allocator,
            .config = config,
            .current_level = std.atomic.Value(MemoryPressureLevel).init(.none),
            .is_running = std.atomic.Value(bool).init(false),
        };
        
        // Initialize cgroup detection if enabled
        if (config.enable_auto_detection) {
            try self.initializeCGroupDetection();
        }
        
        return self;
    }
    
    /// Initialize cross-platform cgroup detection
    fn initializeCGroupDetection(self: *MemoryPressureMonitor) !void {
        // Create and initialize the cgroup detector
        const detector = try self.allocator.create(cgroup_detection.CGroupDetector);
        detector.* = cgroup_detection.CGroupDetector.init(self.allocator);
        
        // Detect available memory pressure sources
        try detector.detect();
        
        // Store detection results
        self.cgroup_detector = detector;
        self.detected_sources = detector.getAvailableSources();
        self.primary_source = detector.getPrimarySource();
        
        // Log detection results for debugging
        if (self.primary_source) |primary| {
            std.log.info("Memory pressure monitoring: Using {s} (reliability: {d:.1}%)", 
                .{primary.source.getDescription(), primary.reliability * 100.0});
            
            const container_env = detector.getContainerEnvironment();
            const cgroup_version = detector.getCGroupVersion();
            std.log.info("Container environment: {s}, CGroup version: {s}", 
                .{@tagName(container_env), @tagName(cgroup_version)});
        } else {
            std.log.warn("Memory pressure monitoring: No reliable sources detected, using fallback", .{});
        }
    }
    
    pub fn deinit(self: *MemoryPressureMonitor) void {
        // Ensure thread is always stopped and joined
        self.stop();
        
        // Double-check thread cleanup (defensive programming)
        if (self.monitor_thread != null) {
            std.log.warn("MemoryPressureMonitor: Thread handle still exists after stop(), forcing cleanup", .{});
            self.monitor_thread = null;
        }
        
        // Clean up file handles
        if (self.psi_file) |file| {
            file.close();
        }
        if (self.cgroup_v2_file) |file| {
            file.close();
        }
        if (self.cgroup_v1_file) |file| {
            file.close();
        }
        if (self.meminfo_file) |file| {
            file.close();
        }
        
        // Clean up cgroup detector
        if (self.cgroup_detector) |detector| {
            detector.deinit();
            self.allocator.destroy(detector);
        }
        
        self.allocator.destroy(self);
    }
    
    /// Start the memory pressure monitoring thread with cross-platform source detection
    pub fn start(self: *MemoryPressureMonitor) !void {
        if (self.is_running.load(.acquire)) {
            return; // Already running
        }
        
        // Initialize file handles based on detected sources
        try self.initializeFileHandles();
        
        // Update initial metrics
        try self.updateMetrics();
        
        // Create thread BEFORE setting running flag to avoid race condition
        const thread = try std.Thread.spawn(.{}, monitorLoop, .{self});
        
        // Only set running state after successful thread creation
        self.monitor_thread = thread;
        self.is_running.store(true, .release);
    }
    
    /// Initialize file handles based on detected memory pressure sources
    fn initializeFileHandles(self: *MemoryPressureMonitor) !void {
        if (self.primary_source) |primary| {
            // Open file handle for the primary source if it's file-based
            if (primary.file_path) |file_path| {
                switch (primary.source) {
                    .linux_psi_global, .linux_psi_cgroup_v2 => {
                        self.psi_file = std.fs.openFileAbsolute(file_path, .{}) catch |err| {
                            std.log.warn("Failed to open PSI file {s}: {}", .{file_path, err});
                            return;
                        };
                    },
                    .linux_cgroup_v2_memory => {
                        self.cgroup_v2_file = std.fs.openFileAbsolute(file_path, .{}) catch |err| {
                            std.log.warn("Failed to open cgroup v2 file {s}: {}", .{file_path, err});
                            return;
                        };
                    },
                    .linux_cgroup_v1_memory => {
                        self.cgroup_v1_file = std.fs.openFileAbsolute(file_path, .{}) catch |err| {
                            std.log.warn("Failed to open cgroup v1 file {s}: {}", .{file_path, err});
                            return;
                        };
                    },
                    .linux_meminfo, .generic_unix_proc => {
                        self.meminfo_file = std.fs.openFileAbsolute(file_path, .{}) catch |err| {
                            std.log.warn("Failed to open meminfo file {s}: {}", .{file_path, err});
                            return;
                        };
                    },
                    else => {
                        // Non-file-based sources don't need file handles
                    },
                }
            }
        } else {
            // Fallback: try to open legacy PSI file
            self.psi_file = std.fs.openFileAbsolute(self.config.psi_file_path, .{}) catch null;
        }
    }
    
    /// Stop the monitoring thread and ensure proper cleanup
    pub fn stop(self: *MemoryPressureMonitor) void {
        // Always try to stop, even if running flag is inconsistent
        self.is_running.store(false, .release);
        
        // Always join thread if it exists, regardless of running flag
        if (self.monitor_thread) |thread| {
            thread.join();
            self.monitor_thread = null;
        }
    }
    
    /// Get current memory pressure level (atomic, lock-free)
    pub fn getCurrentLevel(self: *const MemoryPressureMonitor) MemoryPressureLevel {
        return self.current_level.load(.acquire);
    }
    
    /// Get current memory pressure metrics (protected copy)
    pub fn getCurrentMetrics(self: *const MemoryPressureMonitor) MemoryPressureMetrics {
        // Safe to cast const to non-const for mutex lock since we're not modifying the data
        const self_mut = @constCast(self);
        self_mut.metrics_mutex.lock();
        defer self_mut.metrics_mutex.unlock();
        return self.current_metrics;
    }
    
    /// Check if memory pressure suggests deferring new tasks
    pub fn shouldDeferTasks(self: *const MemoryPressureMonitor) bool {
        return self.getCurrentLevel().shouldDeferTasks();
    }
    
    /// Check if memory pressure suggests preferring local NUMA placement
    pub fn shouldPreferLocalNUMA(self: *const MemoryPressureMonitor) bool {
        return self.getCurrentLevel().shouldPreferLocalNUMA();
    }
    
    /// Get recommended task batch limit based on current pressure
    pub fn getTaskBatchLimit(self: *const MemoryPressureMonitor, default_limit: u32) u32 {
        return self.getCurrentLevel().getTaskBatchLimit(default_limit);
    }
    
    /// Check if the monitoring thread is properly initialized and running
    pub fn isThreadHealthy(self: *const MemoryPressureMonitor) bool {
        const running = self.is_running.load(.acquire);
        const has_thread = self.monitor_thread != null;
        
        // Thread is healthy if:
        // 1. Not running and no thread handle (stopped state)
        // 2. Running and has thread handle (active state)
        return (running and has_thread) or (!running and !has_thread);
    }
    
    /// Set NUMA mapper for NUMA-aware memory monitoring
    /// This enables per-NUMA node memory pressure tracking using logical NUMA node IDs
    pub fn setNumaMapper(self: *MemoryPressureMonitor, mapper: *numa_mapping.NumaMapper) !void {
        self.numa_mapper = mapper;
        
        // Initialize NUMA-aware monitoring if metrics are available
        self.metrics_mutex.lock();
        defer self.metrics_mutex.unlock();
        
        try self.initializeNumaMetrics();
    }
    
    /// Initialize per-NUMA memory metrics using logical NUMA node IDs
    fn initializeNumaMetrics(self: *MemoryPressureMonitor) !void {
        const mapper = self.numa_mapper orelse return;
        
        const numa_node_count = mapper.getNumaNodeCount();
        if (numa_node_count == 0) return;
        
        // Allocate storage for per-NUMA metrics
        const numa_metrics = try self.allocator.alloc(NumaMemoryMetrics, numa_node_count);
        
        // Initialize each NUMA node's metrics
        for (numa_metrics, 0..) |*numa_metric, logical_idx| {
            const logical_numa = @as(numa_mapping.LogicalNumaNodeId, @intCast(logical_idx));
            const node_mapping = mapper.getNodeMapping(logical_numa) orelse continue;
            
            numa_metric.* = NumaMemoryMetrics{
                .logical_numa_node = logical_numa,
                .memory_total_mb = node_mapping.memory_size_mb,
                .memory_used_mb = 0,
                .memory_free_mb = node_mapping.memory_size_mb,
                .memory_used_pct = 0.0,
            };
        }
        
        // Update current metrics to include NUMA information
        self.current_metrics.numa_metrics = numa_metrics;
        self.current_metrics.numa_metrics_available = true;
    }
    
    /// Get memory pressure level for a specific logical NUMA node
    pub fn getNumaNodePressureLevel(self: *const MemoryPressureMonitor, logical_numa: numa_mapping.LogicalNumaNodeId) ?MemoryPressureLevel {
        const self_mut = @constCast(self);
        self_mut.metrics_mutex.lock();
        defer self_mut.metrics_mutex.unlock();
        
        return self.current_metrics.getNumaNodePressureLevel(logical_numa);
    }
    
    /// Get the logical NUMA node with highest memory pressure
    pub fn getHighestPressureNumaNode(self: *const MemoryPressureMonitor) ?numa_mapping.LogicalNumaNodeId {
        const self_mut = @constCast(self);
        self_mut.metrics_mutex.lock();
        defer self_mut.metrics_mutex.unlock();
        
        return self.current_metrics.getHighestPressureNumaNode();
    }
    
    /// Get cross-platform detection information for debugging
    pub fn getDetectionInfo(self: *const MemoryPressureMonitor, allocator: std.mem.Allocator) ![]u8 {
        if (self.cgroup_detector) |detector| {
            return detector.getDetectionSummary(allocator);
        } else {
            return std.fmt.allocPrint(allocator, "Cross-platform detection disabled (enable_auto_detection = false)\n", .{});
        }
    }
    
    /// Get container environment information
    pub fn getContainerEnvironment(self: *const MemoryPressureMonitor) ?cgroup_detection.ContainerEnvironment {
        if (self.cgroup_detector) |detector| {
            return detector.getContainerEnvironment();
        }
        return null;
    }
    
    /// Get cgroup version information
    pub fn getCGroupVersion(self: *const MemoryPressureMonitor) ?cgroup_detection.CGroupVersion {
        if (self.cgroup_detector) |detector| {
            return detector.getCGroupVersion();
        }
        return null;
    }
    
    /// Get primary memory pressure source information
    pub fn getPrimarySource(self: *const MemoryPressureMonitor) ?cgroup_detection.MemoryPressureSourceConfig {
        return self.primary_source;
    }
    
    /// Update memory pressure metrics using cross-platform detection (called by monitoring thread)
    fn updateMetrics(self: *MemoryPressureMonitor) !void {
        var metrics = MemoryPressureMetrics{
            .last_update_ns = @as(u64, @intCast(std.time.nanoTimestamp())),
        };
        
        // Use detected primary source if available
        if (self.primary_source) |primary| {
            if (self.readFromPrimarySource(&metrics, primary)) {
                // Successfully read from primary source
                if (primary.source == .linux_psi_global or primary.source == .linux_psi_cgroup_v2) {
                    metrics.psi_available = true;
                }
            } else {
                // Primary source failed, try fallback sources
                metrics.psi_available = false;
                self.readFromFallbackSources(&metrics);
            }
        } else {
            // No detected sources, use legacy method
            if (self.readPSIMemory(&metrics)) {
                metrics.psi_available = true;
            } else {
                metrics.psi_available = false;
            }
            
            // Always try to get basic memory utilization
            self.readMemoryUtilization(&metrics);
        }
        
        // Calculate and update pressure level
        const new_level = metrics.calculatePressureLevel();
        
        // Protected updates
        self.metrics_mutex.lock();
        self.current_metrics = metrics;
        self.metrics_mutex.unlock();
        
        self.current_level.store(new_level, .release);
    }
    
    /// Read memory pressure from the primary detected source
    fn readFromPrimarySource(self: *MemoryPressureMonitor, metrics: *MemoryPressureMetrics, primary: cgroup_detection.MemoryPressureSourceConfig) bool {
        switch (primary.source) {
            .linux_psi_global, .linux_psi_cgroup_v2 => {
                return self.readPSIMemory(metrics);
            },
            .linux_cgroup_v2_memory => {
                return self.readCGroupV2Memory(metrics);
            },
            .linux_cgroup_v1_memory => {
                return self.readCGroupV1Memory(metrics);
            },
            .linux_meminfo, .generic_unix_proc => {
                self.readLinuxMeminfo(metrics);
                return true; // meminfo reading doesn't fail
            },
            .windows_performance_counters => {
                self.readWindowsMemory(metrics);
                return true;
            },
            .macos_vm_stat => {
                self.readMacOSMemory(metrics);
                return true;
            },
            .freebsd_sysctl => {
                self.readFreeBSDMemory(metrics);
                return true;
            },
            else => {
                std.log.debug("Unsupported primary source: {s}", .{primary.source.getDescription()});
                return false;
            },
        }
    }
    
    /// Read memory pressure from fallback sources
    fn readFromFallbackSources(self: *MemoryPressureMonitor, metrics: *MemoryPressureMetrics) void {
        // Try available fallback sources in order of reliability
        for (self.detected_sources) |source| {
            if (source.enabled and source.reliability >= self.config.min_source_reliability) {
                if (self.readFromPrimarySource(metrics, source)) {
                    std.log.debug("Successfully read from fallback source: {s}", .{source.source.getDescription()});
                    return;
                }
            }
        }
        
        // Last resort: use platform-specific fallback
        self.readMemoryUtilization(metrics);
    }
    
    /// Read cgroup v2 memory information
    fn readCGroupV2Memory(self: *MemoryPressureMonitor, metrics: *MemoryPressureMetrics) bool {
        const file = self.cgroup_v2_file orelse return false;
        
        // Try to read memory.current and memory.max
        file.seekTo(0) catch return false;
        
        var buf: [128]u8 = undefined;
        const bytes_read = file.read(&buf) catch return false;
        const content = buf[0..bytes_read];
        
        // Parse memory usage (simple format: just a number in bytes)
        const memory_current = std.fmt.parseInt(u64, std.mem.trim(u8, content, " \n\t"), 10) catch return false;
        
        // Try to read memory.max (if available)
        const memory_max = self.readCGroupV2MemoryMax() orelse (8 * 1024 * 1024 * 1024); // 8GB default
        
        // Calculate percentage
        metrics.memory_used_pct = (@as(f32, @floatFromInt(memory_current)) / @as(f32, @floatFromInt(memory_max))) * 100.0;
        metrics.memory_available_mb = (memory_max - memory_current) / (1024 * 1024);
        
        return true;
    }
    
    /// Read cgroup v2 memory.max file
    fn readCGroupV2MemoryMax(self: *MemoryPressureMonitor) ?u64 {
        _ = self; // Remove unused parameter warning
        
        const file = std.fs.openFileAbsolute("/sys/fs/cgroup/memory.max", .{}) catch return null;
        defer file.close();
        
        var buf: [128]u8 = undefined;
        const bytes_read = file.read(&buf) catch return null;
        const content = std.mem.trim(u8, buf[0..bytes_read], " \n\t");
        
        // Handle "max" value (unlimited)
        if (std.mem.eql(u8, content, "max")) {
            // Use system memory as limit
            return null;
        }
        
        return std.fmt.parseInt(u64, content, 10) catch null;
    }
    
    /// Read cgroup v1 memory information
    fn readCGroupV1Memory(self: *MemoryPressureMonitor, metrics: *MemoryPressureMetrics) bool {
        const file = self.cgroup_v1_file orelse return false;
        
        file.seekTo(0) catch return false;
        
        var buf: [128]u8 = undefined;
        const bytes_read = file.read(&buf) catch return false;
        const content = buf[0..bytes_read];
        
        // Parse memory usage in bytes
        const memory_current = std.fmt.parseInt(u64, std.mem.trim(u8, content, " \n\t"), 10) catch return false;
        
        // Try to read limit
        const memory_limit = self.readCGroupV1MemoryLimit() orelse (8 * 1024 * 1024 * 1024); // 8GB default
        
        // Calculate percentage
        metrics.memory_used_pct = (@as(f32, @floatFromInt(memory_current)) / @as(f32, @floatFromInt(memory_limit))) * 100.0;
        metrics.memory_available_mb = (memory_limit - memory_current) / (1024 * 1024);
        
        return true;
    }
    
    /// Read cgroup v1 memory.limit_in_bytes
    fn readCGroupV1MemoryLimit(self: *MemoryPressureMonitor) ?u64 {
        _ = self; // Remove unused parameter warning
        
        const file = std.fs.openFileAbsolute("/sys/fs/cgroup/memory/memory.limit_in_bytes", .{}) catch return null;
        defer file.close();
        
        var buf: [128]u8 = undefined;
        const bytes_read = file.read(&buf) catch return null;
        const content = std.mem.trim(u8, buf[0..bytes_read], " \n\t");
        
        return std.fmt.parseInt(u64, content, 10) catch null;
    }
    
    /// Read FreeBSD memory information via sysctl
    fn readFreeBSDMemory(self: *MemoryPressureMonitor, metrics: *MemoryPressureMetrics) void {
        _ = self; // Remove unused parameter warning
        
        // FreeBSD sysctl implementation would go here
        // For now, provide estimated values
        metrics.memory_used_pct = 50.0; // Estimate
        metrics.memory_available_mb = 2048; // 2GB estimate
    }
    
    /// Read Linux PSI memory pressure information
    fn readPSIMemory(self: *MemoryPressureMonitor, metrics: *MemoryPressureMetrics) bool {
        const file = self.psi_file orelse return false;
        
        // Reset file position
        file.seekTo(0) catch return false;
        
        var buf: [512]u8 = undefined;
        const bytes_read = file.read(&buf) catch return false;
        const content = buf[0..bytes_read];
        
        // Parse PSI format:
        // some avg10=2.04 avg60=1.81 avg300=1.84 total=12345678
        // full avg10=0.00 avg60=0.00 avg300=0.00 total=0
        
        var lines = std.mem.splitScalar(u8, content, '\n');
        
        // Parse "some" line
        if (lines.next()) |some_line| {
            if (std.mem.startsWith(u8, some_line, "some ")) {
                metrics.some_avg10 = self.parsePSIValue(some_line, "avg10=") orelse 0.0;
                metrics.some_avg60 = self.parsePSIValue(some_line, "avg60=") orelse 0.0;
                metrics.some_avg300 = self.parsePSIValue(some_line, "avg300=") orelse 0.0;
            }
        }
        
        // Parse "full" line
        if (lines.next()) |full_line| {
            if (std.mem.startsWith(u8, full_line, "full ")) {
                metrics.full_avg10 = self.parsePSIValue(full_line, "avg10=") orelse 0.0;
                metrics.full_avg60 = self.parsePSIValue(full_line, "avg60=") orelse 0.0;
                metrics.full_avg300 = self.parsePSIValue(full_line, "avg300=") orelse 0.0;
            }
        }
        
        return true;
    }
    
    /// Parse a PSI value from a line (e.g., "avg10=2.04" -> 2.04)
    fn parsePSIValue(self: *MemoryPressureMonitor, line: []const u8, prefix: []const u8) ?f32 {
        _ = self; // Remove unused parameter warning
        
        const start_pos = std.mem.indexOf(u8, line, prefix) orelse return null;
        const value_start = start_pos + prefix.len;
        
        // Find end of value (space or end of line)
        var value_end = value_start;
        while (value_end < line.len and line[value_end] != ' ') {
            value_end += 1;
        }
        
        const value_str = line[value_start..value_end];
        return std.fmt.parseFloat(f32, value_str) catch null;
    }
    
    /// Read basic memory utilization (cross-platform)
    fn readMemoryUtilization(self: *MemoryPressureMonitor, metrics: *MemoryPressureMetrics) void {
        switch (builtin.os.tag) {
            .linux => self.readLinuxMeminfo(metrics),
            .windows => self.readWindowsMemory(metrics),
            .macos => self.readMacOSMemory(metrics),
            else => {
                // Fallback: assume moderate memory usage
                metrics.memory_used_pct = 50.0;
                metrics.memory_available_mb = 1024; // 1GB fallback
            },
        }
    }
    
    /// Read Linux /proc/meminfo for memory utilization
    fn readLinuxMeminfo(self: *MemoryPressureMonitor, metrics: *MemoryPressureMetrics) void {
        
        const file = std.fs.openFileAbsolute("/proc/meminfo", .{}) catch return;
        defer file.close();
        
        var buf: [2048]u8 = undefined;
        const bytes_read = file.read(&buf) catch return;
        const content = buf[0..bytes_read];
        
        var mem_total: ?u64 = null;
        var mem_available: ?u64 = null;
        var swap_total: ?u64 = null;
        var swap_free: ?u64 = null;
        
        var lines = std.mem.splitScalar(u8, content, '\n');
        while (lines.next()) |line| {
            if (std.mem.startsWith(u8, line, "MemTotal:")) {
                mem_total = self.parseMemInfoValue(line);
            } else if (std.mem.startsWith(u8, line, "MemAvailable:")) {
                mem_available = self.parseMemInfoValue(line);
            } else if (std.mem.startsWith(u8, line, "SwapTotal:")) {
                swap_total = self.parseMemInfoValue(line);
            } else if (std.mem.startsWith(u8, line, "SwapFree:")) {
                swap_free = self.parseMemInfoValue(line);
            }
        }
        
        // Calculate memory usage percentages
        if (mem_total != null and mem_available != null) {
            const used_kb = mem_total.? - mem_available.?;
            metrics.memory_used_pct = (@as(f32, @floatFromInt(used_kb)) / @as(f32, @floatFromInt(mem_total.?))) * 100.0;
            metrics.memory_available_mb = mem_available.? / 1024; // Convert KB to MB
        }
        
        if (swap_total != null and swap_free != null and swap_total.? > 0) {
            const swap_used = swap_total.? - swap_free.?;
            metrics.swap_used_pct = (@as(f32, @floatFromInt(swap_used)) / @as(f32, @floatFromInt(swap_total.?))) * 100.0;
        }
    }
    
    /// Parse a value from /proc/meminfo line (e.g., "MemTotal: 16384000 kB" -> 16384000)
    fn parseMemInfoValue(_: *MemoryPressureMonitor, line: []const u8) ?u64 {
        
        // Find the colon
        const colon_pos = std.mem.indexOf(u8, line, ":") orelse return null;
        
        // Skip whitespace after colon
        var value_start = colon_pos + 1;
        while (value_start < line.len and (line[value_start] == ' ' or line[value_start] == '\t')) {
            value_start += 1;
        }
        
        // Find end of number
        var value_end = value_start;
        while (value_end < line.len and std.ascii.isDigit(line[value_end])) {
            value_end += 1;
        }
        
        if (value_start >= value_end) return null;
        
        const value_str = line[value_start..value_end];
        return std.fmt.parseInt(u64, value_str, 10) catch null;
    }
    
    /// Read Windows memory information using Zig's built-in Windows API bindings
    /// This avoids DLL handle leaks by using std.os.windows.kernel32.GlobalMemoryStatusEx
    fn readWindowsMemory(self: *MemoryPressureMonitor, metrics: *MemoryPressureMetrics) void {
        _ = self;
        
        if (builtin.os.tag != .windows) {
            // Fallback for non-Windows platforms
            metrics.memory_used_pct = 60.0;
            metrics.memory_available_mb = 2048;
            return;
        }
        
        // Use Zig's built-in Windows API bindings - no DLL handle management required
        var mem_status: std.os.windows.MEMORYSTATUSEX = undefined;
        mem_status.dwLength = @sizeOf(std.os.windows.MEMORYSTATUSEX);
        
        // Call GlobalMemoryStatusEx through Zig's Windows API bindings
        const result = std.os.windows.kernel32.GlobalMemoryStatusEx(&mem_status);
        
        if (result == 0) {
            // API call failed, use conservative fallback estimates
            std.log.debug("GlobalMemoryStatusEx failed, using fallback estimates", .{});
            metrics.memory_used_pct = 60.0;
            metrics.memory_available_mb = 2048;
            return;
        }
        
        // Calculate memory usage percentages from Windows API data
        const total_memory = mem_status.ullTotalPhys;
        const available_memory = mem_status.ullAvailPhys;
        const used_memory = total_memory - available_memory;
        
        // Convert to percentages and MB
        metrics.memory_used_pct = if (total_memory > 0) 
            (@as(f32, @floatFromInt(used_memory)) / @as(f32, @floatFromInt(total_memory))) * 100.0
        else 
            0.0;
        
        metrics.memory_available_mb = available_memory / (1024 * 1024); // Convert bytes to MB
        
        // Handle virtual memory (swap) if available
        const total_virtual = mem_status.ullTotalVirtual;
        const available_virtual = mem_status.ullAvailVirtual;
        
        if (total_virtual > 0) {
            const used_virtual = total_virtual - available_virtual;
            metrics.swap_used_pct = (@as(f32, @floatFromInt(used_virtual)) / @as(f32, @floatFromInt(total_virtual))) * 100.0;
        } else {
            metrics.swap_used_pct = 0.0;
        }
    }
    
    /// Read macOS memory information (placeholder - needs macOS APIs)
    fn readMacOSMemory(self: *MemoryPressureMonitor, metrics: *MemoryPressureMetrics) void {
        _ = self;
        // TODO: Implement vm_stat equivalent or sysctl
        // For now, use conservative estimates  
        metrics.memory_used_pct = 55.0;  // Assume moderate usage
        metrics.memory_available_mb = 4096; // 4GB fallback
    }
    
    /// Main monitoring loop (runs in separate thread)
    fn monitorLoop(self: *MemoryPressureMonitor) void {
        const interval_ns = @as(u64, self.config.update_interval_ms) * 1_000_000;
        
        // Log thread start for debugging
        if (builtin.mode == .Debug) {
            std.debug.print("MemoryPressureMonitor: Monitoring thread started\n", .{});
        }
        
        while (self.is_running.load(.acquire)) {
            self.updateMetrics() catch |err| {
                if (builtin.mode == .Debug) {
                    std.debug.print("MemoryPressureMonitor: Failed to update metrics: {}\n", .{err});
                }
            };
            
            // Sleep with periodic checks for shutdown (improved responsiveness)
            const sleep_chunks = 10; // Check shutdown flag 10 times during sleep
            const chunk_ns = interval_ns / sleep_chunks;
            
            for (0..sleep_chunks) |_| {
                if (!self.is_running.load(.acquire)) break;
                std.time.sleep(chunk_ns);
            }
        }
        
        // Log thread exit for debugging
        if (builtin.mode == .Debug) {
            std.debug.print("MemoryPressureMonitor: Monitoring thread exiting\n", .{});
        }
    }
};

// ============================================================================
// Integration Helpers
// ============================================================================

/// Memory pressure callback function type
pub const PressureCallback = *const fn (level: MemoryPressureLevel, metrics: *const MemoryPressureMetrics) void;

/// Registry for memory pressure event callbacks
pub const PressureCallbackRegistry = struct {
    callbacks: std.ArrayList(PressureCallback),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) PressureCallbackRegistry {
        return .{
            .callbacks = std.ArrayList(PressureCallback).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *PressureCallbackRegistry) void {
        self.callbacks.deinit();
    }
    
    pub fn register(self: *PressureCallbackRegistry, callback: PressureCallback) !void {
        try self.callbacks.append(callback);
    }
    
    pub fn triggerCallbacks(self: *const PressureCallbackRegistry, level: MemoryPressureLevel, metrics: *const MemoryPressureMetrics) void {
        for (self.callbacks.items) |callback| {
            callback(level, metrics);
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "memory pressure level calculation" {
    const metrics = MemoryPressureMetrics{
        .some_avg10 = 30.0,  // 30 * 0.4 + 25 * 0.1 = 12 + 2.5 = 14.5% -> .low
        .memory_used_pct = 25.0,
        .psi_available = true,
    };
    
    const level = metrics.calculatePressureLevel();
    try std.testing.expect(level == .low);
    
    try std.testing.expect(level.shouldPreferLocalNUMA());
    try std.testing.expect(!level.shouldDeferTasks());
}

test "memory pressure monitor initialization" {
    const allocator = std.testing.allocator;
    
    const config = MemoryPressureConfig{
        .update_interval_ms = 50, // Faster for testing
    };
    
    var monitor = try MemoryPressureMonitor.init(allocator, config);
    defer monitor.deinit();
    
    // Test initial state
    const initial_level = monitor.getCurrentLevel();
    try std.testing.expect(initial_level == .none);
    
    // Test metrics update
    try monitor.updateMetrics();
    const metrics = monitor.getCurrentMetrics();
    try std.testing.expect(metrics.last_update_ns > 0);
}

test "PSI value parsing" {
    const allocator = std.testing.allocator;
    const config = MemoryPressureConfig{};
    var monitor = try MemoryPressureMonitor.init(allocator, config);
    defer monitor.deinit();
    
    const test_line = "some avg10=2.04 avg60=1.81 avg300=1.84 total=12345678";
    
    const avg10 = monitor.parsePSIValue(test_line, "avg10=").?;
    const avg60 = monitor.parsePSIValue(test_line, "avg60=").?;
    
    try std.testing.expectApproxEqAbs(@as(f32, 2.04), avg10, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.81), avg60, 0.01);
}

test "batch limit adjustment" {
    const none_level = MemoryPressureLevel.none;
    const high_level = MemoryPressureLevel.high;
    
    try std.testing.expectEqual(@as(u32, 100), none_level.getTaskBatchLimit(100));
    try std.testing.expectEqual(@as(u32, 25), high_level.getTaskBatchLimit(100));
}