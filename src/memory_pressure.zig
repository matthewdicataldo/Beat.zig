const std = @import("std");
const builtin = @import("builtin");
const core = @import("core.zig");

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

/// Real-time memory pressure metrics from Linux PSI
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
    
    /// Get human-readable description of current state
    pub fn getDescription(self: *const MemoryPressureMetrics, allocator: std.mem.Allocator) ![]u8 {
        const level = self.calculatePressureLevel();
        const trend = if (self.isPressureIncreasing()) "↑" else "→";
        
        return std.fmt.allocPrint(allocator, 
            "Memory Pressure: {s} {s} (PSI: {d:.1f}% some, {d:.1f}% full, Mem: {d:.1f}%)",
            .{ @tagName(level), trend, self.some_avg10, self.full_avg10, self.memory_used_pct }
        );
    }
};

/// Configuration for memory pressure monitoring
pub const MemoryPressureConfig = struct {
    // Monitoring intervals
    update_interval_ms: u32 = 100,           // How often to check pressure (100ms)
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
    
    // Platform fallbacks
    enable_meminfo_fallback: bool = true,    // Use /proc/meminfo on Linux without PSI
    enable_windows_fallback: bool = true,    // Use Windows memory APIs
    enable_macos_fallback: bool = true,      // Use macOS vm_stat equivalent
};

// ============================================================================
// Memory Pressure Monitor
// ============================================================================

/// Thread-safe memory pressure monitor with protected state updates
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
    
    // Platform-specific handles
    psi_file: ?std.fs.File = null,
    
    pub fn init(allocator: std.mem.Allocator, config: MemoryPressureConfig) !*MemoryPressureMonitor {
        const self = try allocator.create(MemoryPressureMonitor);
        self.* = .{
            .allocator = allocator,
            .config = config,
            .current_level = std.atomic.Value(MemoryPressureLevel).init(.none),
            .is_running = std.atomic.Value(bool).init(false),
        };
        
        return self;
    }
    
    pub fn deinit(self: *MemoryPressureMonitor) void {
        self.stop();
        
        if (self.psi_file) |file| {
            file.close();
        }
        
        self.allocator.destroy(self);
    }
    
    /// Start the memory pressure monitoring thread
    pub fn start(self: *MemoryPressureMonitor) !void {
        if (self.is_running.load(.acquire)) {
            return; // Already running
        }
        
        // Try to open PSI file for efficient monitoring
        self.psi_file = std.fs.openFileAbsolute(self.config.psi_file_path, .{}) catch null;
        
        // Update initial metrics
        try self.updateMetrics();
        
        self.is_running.store(true, .release);
        self.monitor_thread = try std.Thread.spawn(.{}, monitorLoop, .{self});
    }
    
    /// Stop the monitoring thread
    pub fn stop(self: *MemoryPressureMonitor) void {
        if (!self.is_running.load(.acquire)) {
            return; // Not running
        }
        
        self.is_running.store(false, .release);
        
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
    
    /// Update memory pressure metrics (called by monitoring thread)
    fn updateMetrics(self: *MemoryPressureMonitor) !void {
        var metrics = MemoryPressureMetrics{
            .last_update_ns = @as(u64, @intCast(std.time.nanoTimestamp())),
        };
        
        // Try to read PSI information first (Linux-specific)
        if (self.readPSIMemory(&metrics)) {
            metrics.psi_available = true;
        } else {
            metrics.psi_available = false;
        }
        
        // Always try to get basic memory utilization
        self.readMemoryUtilization(&metrics);
        
        // Calculate and update pressure level
        const new_level = metrics.calculatePressureLevel();
        
        // Protected updates
        self.metrics_mutex.lock();
        self.current_metrics = metrics;
        self.metrics_mutex.unlock();
        
        self.current_level.store(new_level, .release);
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
    
    /// Read Windows memory information (placeholder - needs Windows API)
    fn readWindowsMemory(self: *MemoryPressureMonitor, metrics: *MemoryPressureMetrics) void {
        _ = self;
        // TODO: Implement GlobalMemoryStatusEx() or GetPerformanceInfo()
        // For now, use conservative estimates
        metrics.memory_used_pct = 60.0;  // Assume moderate usage
        metrics.memory_available_mb = 2048; // 2GB fallback
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
        
        while (self.is_running.load(.acquire)) {
            self.updateMetrics() catch |err| {
                if (builtin.mode == .Debug) {
                    std.debug.print("MemoryPressureMonitor: Failed to update metrics: {}\n", .{err});
                }
            };
            
            std.time.sleep(interval_ns);
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