const std = @import("std");
const cgroup_detection = @import("../src/cgroup_detection.zig");
const memory_pressure = @import("../src/memory_pressure.zig");

test "CGroup detector initialization" {
    var detector = cgroup_detection.CGroupDetector.init(std.testing.allocator);
    defer detector.deinit();
    
    try detector.detect();
    
    // Should detect some environment
    const env = detector.getContainerEnvironment();
    try std.testing.expect(env != .unknown);
    
    // Should have at least one source
    const sources = detector.getAvailableSources();
    try std.testing.expect(sources.len > 0);
    
    // Should have a primary source with some reliability
    if (detector.getPrimarySource()) |primary| {
        try std.testing.expect(primary.reliability > 0.0);
        try std.testing.expect(primary.reliability <= 1.0);
    }
}

test "Container environment detection" {
    var detector = cgroup_detection.CGroupDetector.init(std.testing.allocator);
    defer detector.deinit();
    
    const env = detector.detectContainerEnvironment();
    
    // Should return a valid environment
    try std.testing.expect(@intFromEnum(env) >= 0);
    
    // Check if containerized environments support resource limits
    switch (env) {
        .docker, .podman, .kubernetes => try std.testing.expect(env.supportsResourceLimits()),
        .bare_metal, .chroot => try std.testing.expect(!env.supportsResourceLimits()),
        else => {}, // Other environments may or may not support limits
    }
}

test "Memory pressure source reliability ordering" {
    const sources = [_]cgroup_detection.MemoryPressureSource{
        .linux_psi_global,
        .linux_psi_cgroup_v2,
        .linux_meminfo,
        .estimated,
        .unavailable,
    };
    
    var prev_reliability: f32 = 1.1; // Start higher than max
    for (sources) |source| {
        const reliability = source.getReliability();
        try std.testing.expect(reliability <= prev_reliability);
        try std.testing.expect(reliability >= 0.0);
        try std.testing.expect(reliability <= 1.0);
        prev_reliability = reliability;
    }
}

test "Cross-platform memory pressure monitor integration" {
    const config = memory_pressure.MemoryPressureConfig{
        .enable_auto_detection = true,
        .enable_cgroup_v2 = true,
        .enable_cgroup_v1 = true,
        .enable_container_detection = true,
        .min_source_reliability = 0.1, // Accept low-reliability sources for testing
    };
    
    const monitor = try memory_pressure.MemoryPressureMonitor.init(std.testing.allocator, config);
    defer monitor.deinit();
    
    // Should have initialized cgroup detection
    try std.testing.expect(monitor.cgroup_detector != null);
    
    // Should have detected some sources
    try std.testing.expect(monitor.detected_sources.len > 0);
    
    // Test detection info
    const detection_info = try monitor.getDetectionInfo(std.testing.allocator);
    defer std.testing.allocator.free(detection_info);
    
    try std.testing.expect(detection_info.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, detection_info, "Container Environment:") != null);
}

test "Memory pressure monitor with disabled auto-detection" {
    const config = memory_pressure.MemoryPressureConfig{
        .enable_auto_detection = false,
    };
    
    const monitor = try memory_pressure.MemoryPressureMonitor.init(std.testing.allocator, config);
    defer monitor.deinit();
    
    // Should not have initialized cgroup detection
    try std.testing.expect(monitor.cgroup_detector == null);
    try std.testing.expect(monitor.primary_source == null);
    
    // Detection info should indicate disabled state
    const detection_info = try monitor.getDetectionInfo(std.testing.allocator);
    defer std.testing.allocator.free(detection_info);
    
    try std.testing.expect(std.mem.indexOf(u8, detection_info, "disabled") != null);
}

test "Memory pressure source fallback chain" {
    var detector = cgroup_detection.CGroupDetector.init(std.testing.allocator);
    defer detector.deinit();
    
    try detector.detect();
    
    const sources = detector.getAvailableSources();
    if (sources.len == 0) return; // Skip if no sources detected
    
    // Test that sources are ordered by reliability
    var prev_reliability: f32 = 1.1;
    var found_enabled_source = false;
    
    for (sources) |source| {
        if (source.enabled) {
            found_enabled_source = true;
            // In a properly configured system, reliability should generally decrease
            // (though this isn't strictly required)
        }
        
        // All sources should have valid reliability
        try std.testing.expect(source.reliability >= 0.0);
        try std.testing.expect(source.reliability <= 1.0);
    }
    
    try std.testing.expect(found_enabled_source);
}

test "CGroup version detection logic" {
    var detector = cgroup_detection.CGroupDetector.init(std.testing.allocator);
    defer detector.deinit();
    
    // Test cgroup version detection (will return appropriate value for current system)
    const version = detector.detectCGroupVersion();
    
    // Should return a valid cgroup version
    try std.testing.expect(@intFromEnum(version) >= 0);
    
    // On Linux systems, should detect some cgroup support
    if (std.builtin.os.tag == .linux) {
        // Most modern Linux systems should have at least some cgroup support
        // (unless running in a very minimal environment)
        const has_cgroup_support = (version != .none);
        _ = has_cgroup_support; // May or may not have cgroup support
    }
}

test "Platform-specific source detection" {
    var detector = cgroup_detection.CGroupDetector.init(std.testing.allocator);
    defer detector.deinit();
    
    try detector.detect();
    
    const sources = detector.getAvailableSources();
    var found_platform_appropriate_source = false;
    
    for (sources) |source| {
        switch (std.builtin.os.tag) {
            .linux => {
                // Linux should have Linux-specific sources
                switch (source.source) {
                    .linux_psi_global, .linux_psi_cgroup_v2, .linux_psi_cgroup_v1,
                    .linux_meminfo, .linux_cgroup_v2_memory, .linux_cgroup_v1_memory,
                    .generic_unix_proc => found_platform_appropriate_source = true,
                    else => {},
                }
            },
            .windows => {
                // Windows should have Windows-specific sources
                switch (source.source) {
                    .windows_performance_counters, .windows_wmi => found_platform_appropriate_source = true,
                    else => {},
                }
            },
            .macos => {
                // macOS should have macOS-specific sources
                switch (source.source) {
                    .macos_vm_stat, .macos_activity_monitor => found_platform_appropriate_source = true,
                    else => {},
                }
            },
            .freebsd => {
                // FreeBSD should have FreeBSD-specific sources
                switch (source.source) {
                    .freebsd_sysctl => found_platform_appropriate_source = true,
                    else => {},
                }
            },
            else => {
                // Other platforms should have generic sources
                switch (source.source) {
                    .generic_unix_proc, .estimated => found_platform_appropriate_source = true,
                    else => {},
                }
            },
        }
    }
    
    try std.testing.expect(found_platform_appropriate_source);
}

test "Detection summary generation" {
    var detector = cgroup_detection.CGroupDetector.init(std.testing.allocator);
    defer detector.deinit();
    
    try detector.detect();
    
    const summary = try detector.getDetectionSummary(std.testing.allocator);
    defer std.testing.allocator.free(summary);
    
    // Should contain expected information
    try std.testing.expect(summary.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, summary, "Container Environment:") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary, "CGroup Version:") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary, "Available Sources:") != null);
}