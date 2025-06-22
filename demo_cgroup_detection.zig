const std = @import("std");
const memory_pressure = @import("src/memory_pressure.zig");
const cgroup_detection = @import("src/cgroup_detection.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("=== Beat.zig Cross-Platform CGroup Detection Demo ===\n\n", .{});
    
    // Test standalone cgroup detector
    std.debug.print("🔍 Standalone CGroup Detection:\n", .{});
    var detector = cgroup_detection.CGroupDetector.init(allocator);
    defer detector.deinit();
    
    try detector.detect();
    
    const detection_summary = try detector.getDetectionSummary(allocator);
    defer allocator.free(detection_summary);
    
    std.debug.print("{s}\n", .{detection_summary});
    
    // Test integrated memory pressure monitor
    std.debug.print("🎯 Integrated Memory Pressure Monitor:\n", .{});
    
    const config = memory_pressure.MemoryPressureConfig{
        .enable_auto_detection = true,
        .enable_cgroup_v2 = true,
        .enable_cgroup_v1 = true,
        .enable_container_detection = true,
        .min_source_reliability = 0.1, // Accept low-reliability sources for demo
        .update_interval_ms = 200,      // Slower updates for demo
    };
    
    const monitor = try memory_pressure.MemoryPressureMonitor.init(allocator, config);
    defer monitor.deinit();
    
    // Get detection information
    const monitor_detection_info = try monitor.getDetectionInfo(allocator);
    defer allocator.free(monitor_detection_info);
    
    std.debug.print("{s}\n", .{monitor_detection_info});
    
    // Display current configuration
    std.debug.print("📊 Memory Pressure Monitor Configuration:\n", .{});
    if (monitor.getContainerEnvironment()) |env| {
        std.debug.print("Container Environment: {s}\n", .{@tagName(env)});
        std.debug.print("Containerized: {}\n", .{env.isContainerized()});
        std.debug.print("Supports Resource Limits: {}\n", .{env.supportsResourceLimits()});
    }
    
    if (monitor.getCGroupVersion()) |version| {
        std.debug.print("CGroup Version: {s}\n", .{@tagName(version)});
    }
    
    if (monitor.getPrimarySource()) |primary| {
        std.debug.print("Primary Source: {s}\n", .{primary.source.getDescription()});
        std.debug.print("Reliability: {d:.1}%\n", .{primary.reliability * 100.0});
        std.debug.print("Update Interval: {}ms\n", .{primary.update_interval_ms});
        if (primary.file_path) |path| {
            std.debug.print("File Path: {s}\n", .{path});
        }
    }
    
    // Start monitoring and show current metrics
    std.debug.print("\n⚡ Starting Memory Pressure Monitoring:\n", .{});
    try monitor.start();
    defer monitor.stop();
    
    // Let it monitor for a short time
    std.time.sleep(500 * 1000 * 1000); // 500ms
    
    const current_level = monitor.getCurrentLevel();
    const current_metrics = monitor.getCurrentMetrics();
    
    std.debug.print("Current Pressure Level: {s}\n", .{@tagName(current_level)});
    std.debug.print("Memory Used: {d:.1}%\n", .{current_metrics.memory_used_pct});
    std.debug.print("Available Memory: {}MB\n", .{current_metrics.memory_available_mb});
    std.debug.print("PSI Available: {}\n", .{current_metrics.psi_available});
    
    if (current_metrics.psi_available) {
        std.debug.print("PSI some_avg10: {d:.2}%\n", .{current_metrics.some_avg10});
        std.debug.print("PSI full_avg10: {d:.2}%\n", .{current_metrics.full_avg10});
    }
    
    // Test pressure level features
    std.debug.print("\n🎛️  Pressure Level Features:\n", .{});
    std.debug.print("Should Defer Tasks: {}\n", .{current_level.shouldDeferTasks()});
    std.debug.print("Should Prefer Local NUMA: {}\n", .{current_level.shouldPreferLocalNUMA()});
    std.debug.print("Task Batch Limit (default 32): {}\n", .{current_level.getTaskBatchLimit(32)});
    
    // Test NUMA-aware features if available
    if (current_metrics.numa_metrics_available) {
        std.debug.print("\n🔗 NUMA-Aware Features:\n", .{});
        std.debug.print("NUMA Metrics Available: true\n", .{});
        std.debug.print("NUMA Nodes: {}\n", .{current_metrics.numa_metrics.len});
        
        if (monitor.getHighestPressureNumaNode()) |highest_node| {
            std.debug.print("Highest Pressure NUMA Node: {}\n", .{highest_node});
        }
    } else {
        std.debug.print("\n🔗 NUMA-Aware Features: Not available\n", .{});
    }
    
    // Show platform-specific information
    std.debug.print("\n🖥️  Platform-Specific Information:\n", .{});
    std.debug.print("Operating System: {s}\n", .{@tagName(@import("builtin").os.tag)});
    std.debug.print("Architecture: {s}\n", .{@tagName(@import("builtin").cpu.arch)});
    
    // Show fallback capabilities
    std.debug.print("\n🔄 Fallback Capabilities:\n", .{});
    const fallback_capabilities = [_]struct { name: []const u8, enabled: bool }{
        .{ .name = "meminfo fallback", .enabled = config.enable_meminfo_fallback },
        .{ .name = "Windows fallback", .enabled = config.enable_windows_fallback },
        .{ .name = "macOS fallback", .enabled = config.enable_macos_fallback },
        .{ .name = "FreeBSD fallback", .enabled = config.enable_freebsd_fallback },
        .{ .name = "Generic Unix fallback", .enabled = config.enable_generic_unix_fallback },
    };
    
    for (fallback_capabilities) |capability| {
        const status = if (capability.enabled) "✅" else "❌";
        std.debug.print("{s} {s}: {}\n", .{ status, capability.name, capability.enabled });
    }
    
    std.debug.print("\n✨ Cross-Platform CGroup Detection Complete!\n", .{});
    std.debug.print("   🚀 Production-ready memory pressure monitoring\n", .{});
    std.debug.print("   🐳 Container environment detection (Docker, K8s, LXC)\n", .{});
    std.debug.print("   📈 CGroup v1/v2 support with automatic fallbacks\n", .{});
    std.debug.print("   🌍 Cross-platform compatibility (Linux, Windows, macOS, FreeBSD)\n", .{});
}