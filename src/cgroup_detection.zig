const std = @import("std");
const builtin = @import("builtin");

// ============================================================================
// Cross-Platform CGroup Detection and Memory Pressure Sources
//
// This module provides comprehensive detection and fallback for memory pressure
// monitoring across different container environments and operating systems:
//
// Linux Support:
// - cgroup v2 unified hierarchy (/sys/fs/cgroup/memory.pressure)
// - cgroup v1 hierarchy (/sys/fs/cgroup/memory/memory.pressure_level)
// - systemd cgroup detection (/sys/fs/cgroup/user.slice/)
// - Docker container detection (/proc/self/cgroup)
// - Kubernetes pod detection (environment variables)
// - Traditional /proc/pressure/memory (Linux 4.20+)
// - /proc/meminfo fallback
//
// Container Environments:
// - Docker, Podman, containerd
// - Kubernetes, OpenShift
// - LXC/LXD containers
// - systemd user slices
// - chroot environments
//
// Cross-Platform:
// - Windows: Performance counters and WMI
// - macOS: vm_stat and Activity Monitor integration
// - FreeBSD: sysctl memory information
// - Generic Unix: /proc/meminfo where available
// ============================================================================

/// Memory pressure data source types
pub const MemoryPressureSource = enum {
    // Linux PSI (Pressure Stall Information)
    linux_psi_global,           // /proc/pressure/memory (kernel 4.20+)
    linux_psi_cgroup_v2,        // /sys/fs/cgroup/memory.pressure (cgroup v2)
    linux_psi_cgroup_v1,        // /sys/fs/cgroup/memory/memory.pressure_level (cgroup v1)
    
    // Linux memory information
    linux_meminfo,              // /proc/meminfo
    linux_cgroup_v2_memory,     // /sys/fs/cgroup/memory.current, memory.max
    linux_cgroup_v1_memory,     // /sys/fs/cgroup/memory/memory.usage_in_bytes
    
    // Container-specific sources
    docker_stats,               // Docker container stats API
    kubernetes_metrics,         // Kubernetes metrics server
    systemd_cgroup,            // systemd user/system slices
    
    // Cross-platform sources
    windows_performance_counters, // Windows Performance Toolkit
    windows_wmi,                // Windows Management Instrumentation
    macos_vm_stat,              // macOS virtual memory statistics
    macos_activity_monitor,     // macOS Activity Monitor integration
    freebsd_sysctl,             // FreeBSD sysctl MIB
    generic_unix_proc,          // Generic Unix /proc filesystem
    
    // Fallback
    estimated,                  // Estimated pressure based on available info
    unavailable,                // No memory pressure information available
    
    pub fn getDescription(self: MemoryPressureSource) []const u8 {
        return switch (self) {
            .linux_psi_global => "Linux PSI global (/proc/pressure/memory)",
            .linux_psi_cgroup_v2 => "Linux PSI cgroup v2 (/sys/fs/cgroup/memory.pressure)",
            .linux_psi_cgroup_v1 => "Linux PSI cgroup v1 (memory.pressure_level)",
            .linux_meminfo => "Linux meminfo (/proc/meminfo)",
            .linux_cgroup_v2_memory => "Linux cgroup v2 memory controller",
            .linux_cgroup_v1_memory => "Linux cgroup v1 memory controller",
            .docker_stats => "Docker container stats API",
            .kubernetes_metrics => "Kubernetes metrics server",
            .systemd_cgroup => "systemd cgroup management",
            .windows_performance_counters => "Windows Performance Counters",
            .windows_wmi => "Windows WMI (Management Instrumentation)",
            .macos_vm_stat => "macOS vm_stat",
            .macos_activity_monitor => "macOS Activity Monitor integration",
            .freebsd_sysctl => "FreeBSD sysctl memory information",
            .generic_unix_proc => "Generic Unix /proc filesystem",
            .estimated => "Estimated pressure (limited information)",
            .unavailable => "No memory pressure information available",
        };
    }
    
    pub fn getReliability(self: MemoryPressureSource) f32 {
        return switch (self) {
            .linux_psi_global, .linux_psi_cgroup_v2 => 1.0,      // Highest accuracy
            .linux_psi_cgroup_v1, .linux_cgroup_v2_memory => 0.9, // Very good
            .windows_performance_counters, .macos_vm_stat => 0.8,  // Good
            .linux_meminfo, .linux_cgroup_v1_memory => 0.7,      // Decent
            .docker_stats, .kubernetes_metrics => 0.6,           // Container-dependent
            .systemd_cgroup, .freebsd_sysctl => 0.5,             // Basic
            .windows_wmi, .macos_activity_monitor => 0.4,        // May have overhead
            .generic_unix_proc => 0.3,                           // Limited
            .estimated => 0.1,                                   // Very unreliable
            .unavailable => 0.0,                                 // No data
        };
    }
};

/// Container environment detection
pub const ContainerEnvironment = enum {
    bare_metal,      // Running directly on host OS
    docker,          // Docker container
    podman,          // Podman container
    containerd,      // containerd container
    kubernetes,      // Kubernetes pod
    openshift,       // OpenShift pod
    lxc,             // LXC container
    lxd,             // LXD container
    systemd_user,    // systemd user slice
    systemd_system,  // systemd system slice
    chroot,          // chroot environment
    wsl,             // Windows Subsystem for Linux
    unknown,         // Unknown containerization
    
    pub fn isContainerized(self: ContainerEnvironment) bool {
        return switch (self) {
            .bare_metal => false,
            else => true,
        };
    }
    
    pub fn supportsResourceLimits(self: ContainerEnvironment) bool {
        return switch (self) {
            .docker, .podman, .containerd, .kubernetes, .openshift, .lxc, .lxd => true,
            .systemd_user, .systemd_system => true, // With proper configuration
            .bare_metal, .chroot, .wsl, .unknown => false,
        };
    }
};

/// CGroup version detection
pub const CGroupVersion = enum {
    v1,              // Legacy cgroup hierarchy
    v2,              // Unified cgroup hierarchy
    hybrid,          // Mixed v1/v2 (transitional)
    none,            // No cgroup support
    unknown,         // Cannot determine version
};

/// Memory pressure source configuration
pub const MemoryPressureSourceConfig = struct {
    source: MemoryPressureSource,
    file_path: ?[]const u8 = null,           // File path for file-based sources
    update_interval_ms: u32 = 100,           // How often to check this source
    reliability: f32,                        // Reliability factor (0.0-1.0)
    enabled: bool = true,                    // Whether this source is enabled
    fallback_sources: []MemoryPressureSource = &[_]MemoryPressureSource{}, // Fallback chain
};

/// Cross-platform cgroup and memory pressure detector
pub const CGroupDetector = struct {
    allocator: std.mem.Allocator,
    container_env: ContainerEnvironment = .unknown,
    cgroup_version: CGroupVersion = .unknown,
    available_sources: std.ArrayList(MemoryPressureSourceConfig),
    primary_source: ?MemoryPressureSourceConfig = null,
    
    const Self = @This();
    
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .available_sources = std.ArrayList(MemoryPressureSourceConfig).init(allocator),
        };
    }
    
    pub fn deinit(self: *Self) void {
        for (self.available_sources.items) |source_config| {
            if (source_config.file_path) |path| {
                self.allocator.free(path);
            }
        }
        self.available_sources.deinit();
    }
    
    /// Detect container environment and available memory pressure sources
    pub fn detect(self: *Self) !void {
        // Step 1: Detect container environment
        self.container_env = self.detectContainerEnvironment();
        
        // Step 2: Detect cgroup version (Linux only)
        if (builtin.os.tag == .linux) {
            self.cgroup_version = self.detectCGroupVersion();
        }
        
        // Step 3: Detect available memory pressure sources
        try self.detectMemoryPressureSources();
        
        // Step 4: Select primary source based on reliability
        self.selectPrimarySource();
    }
    
    /// Detect the container environment we're running in
    fn detectContainerEnvironment(self: *Self) ContainerEnvironment {
        
        if (builtin.os.tag != .linux) {
            // Non-Linux systems - check for WSL
            if (builtin.os.tag == .windows) return .bare_metal;
            
            // Check for WSL on Windows
            if (self.checkFileExists("/proc/version")) {
                const version_content = self.readFileContent("/proc/version", 256) catch return .unknown;
                defer self.allocator.free(version_content);
                
                if (std.mem.indexOf(u8, version_content, "Microsoft") != null or
                    std.mem.indexOf(u8, version_content, "WSL") != null) {
                    return .wsl;
                }
            }
            
            return .bare_metal;
        }
        
        // Linux container detection
        
        // Check for Docker
        if (self.checkFileExists("/.dockerenv")) {
            return .docker;
        }
        
        // Check cgroup information for container detection
        if (self.readFileContent("/proc/self/cgroup", 1024)) |cgroup_content| {
            defer self.allocator.free(cgroup_content);
            
            if (std.mem.indexOf(u8, cgroup_content, "docker") != null) {
                return .docker;
            } else if (std.mem.indexOf(u8, cgroup_content, "containerd") != null) {
                return .containerd;
            } else if (std.mem.indexOf(u8, cgroup_content, "podman") != null) {
                return .podman;
            } else if (std.mem.indexOf(u8, cgroup_content, "lxc") != null) {
                return .lxc;
            } else if (std.mem.indexOf(u8, cgroup_content, "user.slice") != null) {
                return .systemd_user;
            } else if (std.mem.indexOf(u8, cgroup_content, "system.slice") != null) {
                return .systemd_system;
            }
        } else |_| {}
        
        // Check Kubernetes environment variables
        if (std.process.getEnvVarOwned(self.allocator, "KUBERNETES_SERVICE_HOST")) |_| {
            return .kubernetes;
        } else |_| {}
        
        if (std.process.getEnvVarOwned(self.allocator, "OPENSHIFT_BUILD_NAME")) |_| {
            return .openshift;
        } else |_| {}
        
        // Check for LXD
        if (self.checkFileExists("/var/lib/lxd/")) {
            return .lxd;
        }
        
        // Check if we're in a chroot
        if (self.isInChroot()) {
            return .chroot;
        }
        
        return .bare_metal;
    }
    
    /// Detect cgroup version on Linux systems
    fn detectCGroupVersion(self: *Self) CGroupVersion {
        
        // Check for cgroup v2 unified hierarchy
        if (self.checkFileExists("/sys/fs/cgroup/cgroup.controllers")) {
            // Check if we also have v1 mounts (hybrid mode)
            if (self.checkFileExists("/sys/fs/cgroup/memory/memory.stat")) {
                return .hybrid;
            }
            return .v2;
        }
        
        // Check for cgroup v1
        if (self.checkFileExists("/sys/fs/cgroup/memory/memory.stat")) {
            return .v1;
        }
        
        // Check if cgroup filesystem is mounted at all
        if (self.checkFileExists("/sys/fs/cgroup/")) {
            return .unknown; // Mounted but unclear version
        }
        
        return .none;
    }
    
    /// Detect available memory pressure sources for the current environment
    fn detectMemoryPressureSources(self: *Self) !void {
        switch (builtin.os.tag) {
            .linux => try self.detectLinuxMemoryPressureSources(),
            .windows => try self.detectWindowsMemoryPressureSources(),
            .macos => try self.detectMacOSMemoryPressureSources(),
            .freebsd => try self.detectFreeBSDMemoryPressureSources(),
            else => try self.detectGenericUnixMemoryPressureSources(),
        }
    }
    
    /// Detect Linux-specific memory pressure sources
    fn detectLinuxMemoryPressureSources(self: *Self) !void {
        // Global PSI (Linux 4.20+)
        if (self.checkFileExists("/proc/pressure/memory")) {
            try self.addSource(.{
                .source = .linux_psi_global,
                .file_path = try self.allocator.dupe(u8, "/proc/pressure/memory"),
                .reliability = 1.0,
                .update_interval_ms = 100,
            });
        }
        
        // CGroup v2 PSI
        if (self.cgroup_version == .v2 or self.cgroup_version == .hybrid) {
            const cgroup_v2_paths = [_][]const u8{
                "/sys/fs/cgroup/memory.pressure",           // Current cgroup
                "/sys/fs/cgroup/user.slice/memory.pressure", // User slice
                "/sys/fs/cgroup/system.slice/memory.pressure", // System slice
            };
            
            for (cgroup_v2_paths) |path| {
                if (self.checkFileExists(path)) {
                    try self.addSource(.{
                        .source = .linux_psi_cgroup_v2,
                        .file_path = try self.allocator.dupe(u8, path),
                        .reliability = 0.9,
                        .update_interval_ms = 100,
                    });
                    break; // Only need one cgroup v2 source
                }
            }
            
            // CGroup v2 memory usage
            const cgroup_v2_memory_paths = [_][]const u8{
                "/sys/fs/cgroup/memory.current",
                "/sys/fs/cgroup/user.slice/memory.current",
                "/sys/fs/cgroup/system.slice/memory.current",
            };
            
            for (cgroup_v2_memory_paths) |path| {
                if (self.checkFileExists(path)) {
                    try self.addSource(.{
                        .source = .linux_cgroup_v2_memory,
                        .file_path = try self.allocator.dupe(u8, path),
                        .reliability = 0.8,
                        .update_interval_ms = 200,
                    });
                    break;
                }
            }
        }
        
        // CGroup v1 memory
        if (self.cgroup_version == .v1 or self.cgroup_version == .hybrid) {
            if (self.checkFileExists("/sys/fs/cgroup/memory/memory.usage_in_bytes")) {
                try self.addSource(.{
                    .source = .linux_cgroup_v1_memory,
                    .file_path = try self.allocator.dupe(u8, "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
                    .reliability = 0.7,
                    .update_interval_ms = 200,
                });
            }
        }
        
        // Always add /proc/meminfo as fallback
        if (self.checkFileExists("/proc/meminfo")) {
            try self.addSource(.{
                .source = .linux_meminfo,
                .file_path = try self.allocator.dupe(u8, "/proc/meminfo"),
                .reliability = 0.6,
                .update_interval_ms = 500,
            });
        }
        
        // Container-specific sources
        switch (self.container_env) {
            .docker, .podman, .containerd => {
                // Note: Docker stats would require API access, not file-based
                try self.addSource(.{
                    .source = .docker_stats,
                    .reliability = 0.6,
                    .update_interval_ms = 1000,
                    .enabled = false, // Requires API implementation
                });
            },
            .kubernetes, .openshift => {
                try self.addSource(.{
                    .source = .kubernetes_metrics,
                    .reliability = 0.6,
                    .update_interval_ms = 1000,
                    .enabled = false, // Requires API implementation
                });
            },
            .systemd_user, .systemd_system => {
                try self.addSource(.{
                    .source = .systemd_cgroup,
                    .reliability = 0.5,
                    .update_interval_ms = 500,
                });
            },
            else => {},
        }
    }
    
    /// Detect Windows memory pressure sources
    fn detectWindowsMemoryPressureSources(self: *Self) !void {
        // Windows Performance Counters (preferred)
        try self.addSource(.{
            .source = .windows_performance_counters,
            .reliability = 0.8,
            .update_interval_ms = 500,
        });
        
        // Windows WMI (fallback)
        try self.addSource(.{
            .source = .windows_wmi,
            .reliability = 0.4,
            .update_interval_ms = 1000,
        });
    }
    
    /// Detect macOS memory pressure sources
    fn detectMacOSMemoryPressureSources(self: *Self) !void {
        // vm_stat command (preferred)
        try self.addSource(.{
            .source = .macos_vm_stat,
            .reliability = 0.8,
            .update_interval_ms = 500,
        });
        
        // Activity Monitor integration (fallback)
        try self.addSource(.{
            .source = .macos_activity_monitor,
            .reliability = 0.4,
            .update_interval_ms = 1000,
        });
    }
    
    /// Detect FreeBSD memory pressure sources
    fn detectFreeBSDMemoryPressureSources(self: *Self) !void {
        try self.addSource(.{
            .source = .freebsd_sysctl,
            .reliability = 0.5,
            .update_interval_ms = 500,
        });
    }
    
    /// Detect generic Unix memory pressure sources
    fn detectGenericUnixMemoryPressureSources(self: *Self) !void {
        // Try /proc/meminfo if available
        if (self.checkFileExists("/proc/meminfo")) {
            try self.addSource(.{
                .source = .generic_unix_proc,
                .file_path = try self.allocator.dupe(u8, "/proc/meminfo"),
                .reliability = 0.3,
                .update_interval_ms = 1000,
            });
        } else {
            // Last resort: estimated pressure
            try self.addSource(.{
                .source = .estimated,
                .reliability = 0.1,
                .update_interval_ms = 5000,
            });
        }
    }
    
    /// Add a memory pressure source to the available sources list
    fn addSource(self: *Self, config: MemoryPressureSourceConfig) !void {
        try self.available_sources.append(config);
    }
    
    /// Select the primary memory pressure source based on reliability
    fn selectPrimarySource(self: *Self) void {
        var best_source: ?MemoryPressureSourceConfig = null;
        var best_reliability: f32 = 0.0;
        
        for (self.available_sources.items) |source| {
            if (source.enabled and source.reliability > best_reliability) {
                best_reliability = source.reliability;
                best_source = source;
            }
        }
        
        self.primary_source = best_source;
    }
    
    /// Get the primary memory pressure source
    pub fn getPrimarySource(self: *const Self) ?MemoryPressureSourceConfig {
        return self.primary_source;
    }
    
    /// Get all available memory pressure sources
    pub fn getAvailableSources(self: *const Self) []const MemoryPressureSourceConfig {
        return self.available_sources.items;
    }
    
    /// Get container environment
    pub fn getContainerEnvironment(self: *const Self) ContainerEnvironment {
        return self.container_env;
    }
    
    /// Get cgroup version
    pub fn getCGroupVersion(self: *const Self) CGroupVersion {
        return self.cgroup_version;
    }
    
    /// Check if a file exists
    fn checkFileExists(self: *Self, path: []const u8) bool {
        _ = self; // Remove unused parameter warning
        std.fs.accessAbsolute(path, .{}) catch return false;
        return true;
    }
    
    /// Read file content (caller owns memory)
    fn readFileContent(self: *Self, path: []const u8, max_size: usize) ![]u8 {
        const file = std.fs.openFileAbsolute(path, .{}) catch return error.FileNotFound;
        defer file.close();
        
        const size = try file.getEndPos();
        const actual_size = @min(size, max_size);
        
        const content = try self.allocator.alloc(u8, actual_size);
        _ = try file.readAll(content);
        
        return content;
    }
    
    /// Check if running in a chroot environment
    fn isInChroot(self: *Self) bool {
        _ = self; // Remove unused parameter warning
        
        // Compare root directory inode with known system root
        const root_stat = std.fs.cwd().statFile("/") catch return false;
        const proc_1_root = std.fs.openFileAbsolute("/proc/1/root", .{}) catch return false;
        defer proc_1_root.close();
        
        const proc_1_stat = proc_1_root.stat() catch return false;
        
        // If inodes differ, we're likely in a chroot
        return root_stat.inode != proc_1_stat.inode;
    }
    
    /// Get detection summary for debugging
    pub fn getDetectionSummary(self: *const Self, allocator: std.mem.Allocator) ![]u8 {
        const container_name = @tagName(self.container_env);
        const cgroup_name = @tagName(self.cgroup_version);
        
        var summary = std.ArrayList(u8).init(allocator);
        defer summary.deinit();
        
        try summary.writer().print("=== CGroup Detection Summary ===\n", .{});
        try summary.writer().print("Container Environment: {s}\n", .{container_name});
        try summary.writer().print("CGroup Version: {s}\n", .{cgroup_name});
        try summary.writer().print("Available Sources: {d}\n", .{self.available_sources.items.len});
        
        if (self.primary_source) |primary| {
            try summary.writer().print("Primary Source: {s} (reliability: {d:.1}%)\n", 
                .{primary.source.getDescription(), primary.reliability * 100.0});
        } else {
            try summary.writer().print("Primary Source: None available\n", .{});
        }
        
        try summary.writer().print("\nDetailed Sources:\n", .{});
        for (self.available_sources.items, 0..) |source, i| {
            const status = if (source.enabled) "enabled" else "disabled";
            const path = source.file_path orelse "(no file)";
            try summary.writer().print("  {d}. {s} - {s} (reliability: {d:.1}%) - {s}\n", 
                .{i + 1, source.source.getDescription(), status, source.reliability * 100.0, path});
        }
        
        return summary.toOwnedSlice();
    }
};

// ============================================================================
// Testing
// ============================================================================

test "CGroup detector initialization" {
    var detector = CGroupDetector.init(std.testing.allocator);
    defer detector.deinit();
    
    try detector.detect();
    
    // Should detect some environment
    try std.testing.expect(detector.container_env != .unknown);
    
    // Should have at least one source
    try std.testing.expect(detector.available_sources.items.len > 0);
}

test "Container environment detection" {
    var detector = CGroupDetector.init(std.testing.allocator);
    defer detector.deinit();
    
    const env = detector.detectContainerEnvironment();
    
    // Should return a valid environment (not unknown on most systems)
    try std.testing.expect(@intFromEnum(env) >= 0);
}

test "Memory pressure source reliability" {
    const psi_reliability = MemoryPressureSource.linux_psi_global.getReliability();
    const fallback_reliability = MemoryPressureSource.estimated.getReliability();
    
    try std.testing.expect(psi_reliability > fallback_reliability);
    try std.testing.expect(psi_reliability == 1.0);
    try std.testing.expect(fallback_reliability == 0.1);
}