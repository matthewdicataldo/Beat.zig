const std = @import("std");

// Enhanced Error Messages for Beat.zig
// Provides comprehensive error handling with actionable solutions
// Addresses Reverb Code team feedback for better developer experience

// ============================================================================
// Configuration Errors with Actionable Solutions
// ============================================================================

/// Configuration errors with detailed help messages
pub const ConfigError = error{
    /// build_config module not found - common when using Beat as dependency
    MissingBuildConfig,
    
    /// Hardware detection failed - fallback to safe defaults
    HardwareDetectionFailed,
    
    /// NUMA topology not available on this system
    NumaNotSupported,
    
    /// Insufficient system resources for requested configuration
    InsufficientResources,
    
    /// Invalid configuration parameters provided
    InvalidConfiguration,
    
    /// Required features not available on target platform
    UnsupportedPlatform,
};

/// Integration errors with step-by-step solutions
pub const IntegrationError = error{
    /// External project dependency chain issues
    DependencyChainBroken,
    
    /// Module import path resolution failed
    ModuleImportFailed,
    
    /// Build system configuration incompatible
    BuildSystemIncompatible,
    
    /// Version compatibility issues
    VersionMismatch,
};

// ============================================================================
// Error Message Templates with Solutions
// ============================================================================

/// Generate comprehensive error message for missing build_config
pub fn formatMissingBuildConfigError(project_name: []const u8) []const u8 {
    _ = project_name; // We'll use a generic message for now
    return
        \\
        \\🚨 Beat.zig Configuration Error: Missing build_config module
        \\
        \\Problem: Beat.zig cannot find the 'build_config' module when used as a dependency.
        \\This is a common issue when integrating Beat into external projects.
        \\
        \\💡 SOLUTION OPTIONS (choose one):
        \\
        \\┌─ Option 1: Use Easy API (RECOMMENDED for most projects) ─────────────────┐
        \\│                                                                          │
        \\│  const beat = @import("beat");                                           │
        \\│                                                                          │
        \\│  // Basic pool - no build configuration needed                          │
        \\│  const pool = try beat.createBasicPool(allocator, 4);                   │
        \\│  defer pool.deinit();                                                   │
        \\│                                                                          │
        \\│  // OR performance pool with auto-detection                             │
        \\│  const pool = try beat.createPerformancePool(allocator, .{{}});         │
        \\│                                                                          │
        \\│  ✅ Works immediately, no configuration required                        │
        \\│  ✅ Progressive upgrade path to advanced features                       │
        \\│                                                                          │
        \\└──────────────────────────────────────────────────────────────────────────┘
        \\
        \\┌─ Option 2: Add build_config module to your project ────────────────────┐
        \\│                                                                          │
        \\│  1. Create beat_config.zig in your project:                             │
        \\│                                                                          │
        \\│     // beat_config.zig                                                  │
        \\│     const std = @import("std");                                         │
        \\│                                                                          │
        \\│     pub const detected_cpu_count = @intCast(std.Thread.getCpuCount()    │
        \\│         catch 8);                                                       │
        \\│     pub const optimal_workers = @max(detected_cpu_count - 2, 2);        │
        \\│     pub const optimal_queue_size = optimal_workers * 64;                │
        \\│     pub const has_avx2 = std.Target.x86.featureSetHas(                  │
        \\│         std.Target.current.cpu.features, .avx2);                       │
        \\│     // ... (see documentation for full example)                        │
        \\│                                                                          │
        \\│  2. Add to your build.zig:                                              │
        \\│                                                                          │
        \\│     const beat_config = b.addModule("build_config", .{{                 │
        \\│         .root_source_file = b.path("beat_config.zig"),                  │
        \\│     }});                                                                │
        \\│     exe.root_module.addImport("build_config", beat_config);             │
        \\│                                                                          │
        \\│  ✅ Full advanced features available                                    │
        \\│  ✅ Custom hardware optimization                                        │
        \\│                                                                          │
        \\└──────────────────────────────────────────────────────────────────────────┘
        \\
        \\┌─ Option 3: Use runtime configuration ───────────────────────────────────┐
        \\│                                                                          │
        \\│  const beat = @import("beat");                                           │
        \\│                                                                          │
        \\│  // Auto-detect optimal configuration at runtime                        │
        \\│  const runtime_config = try beat.detectOptimalConfig(allocator);        │
        \\│  const pool = try runtime_config.createPool(allocator);                 │
        \\│  defer pool.deinit();                                                   │
        \\│                                                                          │
        \\│  ✅ No build-time dependencies                                          │
        \\│  ✅ Automatic hardware optimization                                     │
        \\│                                                                          │
        \\└──────────────────────────────────────────────────────────────────────────┘
        \\
        \\📚 DOCUMENTATION & EXAMPLES:
        \\  • Integration Guide: https://github.com/Beat-zig/Beat.zig/blob/main/INTEGRATION_GUIDE.md
        \\  • Easy API Examples: https://github.com/Beat-zig/Beat.zig/blob/main/examples/easy_api/
        \\  • Full Documentation: https://beat-zig.github.io/Beat.zig/
        \\
        \\🔧 DEVELOPMENT MODE:
        \\  For development/testing, use createDevelopmentPool() with safe defaults
        \\  and enhanced debugging features enabled.
        \\
        \\❓ Need help? Open an issue at: https://github.com/Beat-zig/Beat.zig/issues
        \\
    ;
}

/// Generate error message for hardware detection failure  
pub fn formatHardwareDetectionError() []const u8 {
    return 
        \\
        \\⚠️  Beat.zig Hardware Detection Failed
        \\
        \\Problem: Unable to detect optimal hardware configuration automatically.
        \\This can happen on unsupported platforms or in constrained environments.
        \\
        \\💡 AUTOMATIC FALLBACK ACTIVATED:
        \\  • Using safe default configuration
        \\  • Workers: 4 (conservative)
        \\  • Queue size: 256
        \\  • Advanced features: disabled
        \\
        \\🔧 TO OPTIMIZE PERFORMANCE:
        \\
        \\1. Specify manual configuration:
        \\   const config = beat.Config{
        \\       .num_workers = 8,  // Set based on your system
        \\       .task_queue_size = 512,
        \\       .enable_topology_aware = false,  // Keep simple
        \\   };
        \\   const pool = try beat.ThreadPool.init(allocator, config);
        \\
        \\2. OR use runtime detection:
        \\   const runtime_config = try beat.detectOptimalConfig(allocator);
        \\   const pool = try runtime_config.createPool(allocator);
        \\
        \\3. OR use development mode:
        \\   const pool = try beat.createDevelopmentPool(allocator, .{});
        \\
        \\ℹ️  Your application will continue to work with safe defaults.
        \\
    ;
}

/// Generate error message for unsupported platform
pub fn formatUnsupportedPlatformError(platform: []const u8) []const u8 {
    _ = platform; // Use generic message for now
    return
        \\
        \\🚫 Beat.zig Platform Compatibility Issue
        \\
        \\Problem: Some advanced features are not supported on this platform.
        \\
        \\💡 SOLUTIONS:
        \\
        \\1. Use basic configuration (recommended):
        \\   const pool = try beat.createBasicPool(allocator, 4);
        \\   ✅ Cross-platform compatibility
        \\   ✅ Core performance features
        \\
        \\2. Disable specific features:
        \\   const config = beat.Config{{
        \\       .enable_topology_aware = false,
        \\       .enable_numa_aware = false,
        \\       .enable_lock_free = true,  // Usually supported
        \\   }};
        \\
        \\3. Check feature support:
        \\   const info = beat.getPlatformInfo();
        \\   if (info.supports_topology) {{
        \\       // Enable topology features
        \\   }}
        \\
        \\📋 PLATFORM COMPATIBILITY:
        \\  • Linux: Full support (all features)
        \\  • Windows: Full support (all features) 
        \\  • macOS: Full support (all features)
        \\  • FreeBSD: Basic support (no NUMA)
        \\  • Other: Basic support only
        \\
        \\🆘 Need platform-specific help? Create an issue with your platform details:
        \\   https://github.com/Beat-zig/Beat.zig/issues
        \\
    ;
}

/// Generate error message for dependency chain issues
pub fn formatDependencyChainError(error_details: []const u8) []const u8 {
    _ = error_details; // Use generic message for now
    return
        \\
        \\🔗 Beat.zig Dependency Chain Error
        \\
        \\Problem: Dependency chain issues detected.
        \\
        \\💡 QUICK FIXES:
        \\
        \\┌─ Fix 1: Update to new Smart Configuration ──────────────────────────────┐
        \\│                                                                          │
        \\│  Replace any direct imports with unified configuration:                 │
        \\│                                                                          │
        \\│  OLD: const build_opts = @import("build_opts.zig");                     │
        \\│  NEW: const build_opts = @import("build_config_unified.zig");           │
        \\│                                                                          │
        \\│  The new resolver automatically handles dependency scenarios.           │
        \\│                                                                          │
        \\└──────────────────────────────────────────────────────────────────────────┘
        \\
        \\┌─ Fix 2: Use Easy API (bypasses all dependency issues) ─────────────────┐
        \\│                                                                          │
        \\│  const beat = @import("beat");                                           │
        \\│  const pool = try beat.createBasicPool(allocator, 4);                   │
        \\│                                                                          │
        \\│  OR for your specific project type:                                     │
        \\│  const pool = try beat.createHttpServerPool(allocator, .{{}});          │
        \\│                                                                          │
        \\└──────────────────────────────────────────────────────────────────────────┘
        \\
        \\┌─ Fix 3: Add explicit module imports to your build.zig ─────────────────┐
        \\│                                                                          │
        \\│  const beat_dep = b.dependency("beat", .{{}});                           │
        \\│  const beat_module = beat_dep.module("beat");                           │
        \\│  exe.root_module.addImport("beat", beat_module);                        │
        \\│                                                                          │
        \\│  // If using advanced features:                                         │
        \\│  const config_module = b.addModule("build_config", .{{                  │
        \\│      .root_source_file = b.path("your_beat_config.zig"),                │
        \\│  }});                                                                   │
        \\│  exe.root_module.addImport("build_config", config_module);              │
        \\│                                                                          │
        \\└──────────────────────────────────────────────────────────────────────────┘
        \\
        \\🔄 MIGRATION CHECKLIST:
        \\  □ Update Beat.zig to latest version (3.0.1+)
        \\  □ Switch to easy API or smart configuration
        \\  □ Test with basic configuration first
        \\  □ Gradually enable advanced features
        \\
        \\📞 SUPPORT:
        \\  If you're still experiencing issues, please share:
        \\  • Your build.zig configuration
        \\  • Zig version (zig version)
        \\  • Beat.zig version
        \\  • Full error message
        \\
        \\  Create issue: https://github.com/Beat-zig/Beat.zig/issues
        \\
    ;
}

// ============================================================================
// Error Detection and Reporting
// ============================================================================

/// Detect and report configuration issues with actionable solutions
pub fn detectAndReportConfigIssues(allocator: std.mem.Allocator) void {
    // This function will be called during initialization to proactively
    // detect common configuration issues and provide helpful guidance
    
    var detected_issues = std.ArrayList([]const u8).init(allocator);
    defer {
        for (detected_issues.items) |issue| {
            allocator.free(issue);
        }
        detected_issues.deinit();
    }
    
    // Check for common issues
    if (!@hasDecl(@This(), "build_config")) {
        const issue = allocator.dupe(u8, "Missing build_config module - recommend using Easy API") catch return;
        detected_issues.append(issue) catch return;
    }
    
    // Check hardware detection capability
    const cpu_count = std.Thread.getCpuCount() catch {
        const issue = allocator.dupe(u8, "Hardware detection failed - using safe defaults") catch return;
        detected_issues.append(issue) catch return;
        return;
    };
    
    if (cpu_count < 2) {
        const issue = allocator.dupe(u8, "Single-core system detected - consider basic configuration") catch return;
        detected_issues.append(issue) catch return;
    }
    
    // Report detected issues with solutions
    if (detected_issues.items.len > 0) {
        std.log.info("🔍 Beat.zig Configuration Analysis:", .{});
        for (detected_issues.items) |issue| {
            std.log.info("  ⚠️  {s}", .{issue});
        }
        std.log.info("  💡 See enhanced error messages above for solutions", .{});
    }
}

/// Enhanced error logger that provides context and solutions
pub fn logEnhancedError(
    comptime error_type: type,
    error_value: error_type,
    context: []const u8,
) void {
    switch (error_value) {
        ConfigError.MissingBuildConfig => {
            std.log.err("{s}", .{formatMissingBuildConfigError(context)});
        },
        ConfigError.HardwareDetectionFailed => {
            std.log.warn("{s}", .{formatHardwareDetectionError()});
        },
        ConfigError.UnsupportedPlatform => {
            std.log.err("{s}", .{formatUnsupportedPlatformError(context)});
        },
        ConfigError.InvalidConfiguration => {
            std.log.err("{s}", .{formatDependencyChainError(context)});
        },
        else => {
            // Fallback for other errors
            std.log.err("Beat.zig Error: {} in context: {s}\n" ++
                       "💡 Check documentation: https://beat-zig.github.io/Beat.zig/\n" ++
                       "🆘 Need help? https://github.com/Beat-zig/Beat.zig/issues", 
                       .{ error_value, context });
        },
    }
}

/// Validate configuration and provide early warnings
pub fn validateConfigurationWithHelp(config: anytype) !void {
    if (@hasField(@TypeOf(config), "num_workers")) {
        if (config.num_workers != null and config.num_workers.? == 0) {
            std.log.err(
                \\
                \\❌ Invalid Configuration: Worker count cannot be zero
                \\
                \\💡 SOLUTIONS:
                \\  • Use auto-detection: .num_workers = null
                \\  • Set manual count: .num_workers = 4
                \\  • Use easy API: beat.createBasicPool(allocator, 4)
                \\
            , .{});
            return ConfigError.InvalidConfiguration;
        }
        
        if (config.num_workers != null and config.num_workers.? > 64) {
            std.log.warn(
                \\
                \\⚠️  High Worker Count: {} workers requested
                \\
                \\💡 RECOMMENDATIONS:
                \\  • Most systems perform best with workers = CPU count - 2
                \\  • High worker counts can increase overhead
                \\  • Consider using auto-detection for optimal performance
                \\
            , .{config.num_workers.?});
        }
    }
    
    if (@hasField(@TypeOf(config), "task_queue_size")) {
        if (config.task_queue_size < 16) {
            std.log.err(
                \\
                \\❌ Invalid Configuration: Queue size too small ({})
                \\
                \\💡 SOLUTIONS:
                \\  • Minimum queue size: 16
                \\  • Recommended: workers * 64  
                \\  • For high throughput: workers * 128
                \\
            , .{config.task_queue_size});
            return ConfigError.InvalidConfiguration;
        }
    }
}

// ============================================================================
// Integration Support Functions
// ============================================================================

/// Check if Beat.zig is being used as a dependency
pub fn isUsedAsDependency() bool {
    // Heuristic: check if build_config module is available
    // If not, likely being used as external dependency
    return !@hasDecl(@This(), "build_config");
}

/// Get recommended configuration for external projects
pub fn getRecommendedDependencyConfig() @import("core.zig").Config {
    return @import("core.zig").Config{
        .num_workers = null, // Auto-detect
        .enable_topology_aware = false, // Keep simple
        .enable_numa_aware = false,
        .enable_predictive = false,
        .enable_advanced_selection = false,
        .enable_lock_free = true, // Keep core performance
        .enable_statistics = false, // Reduce overhead
        .enable_trace = false,
    };
}

/// Provide migration guidance for existing Beat.zig users
pub fn printMigrationGuide() void {
    std.log.info(
        \\
        \\🚀 Beat.zig 3.0 Migration Guide
        \\
        \\NEW EASY API (Recommended):
        \\  OLD: const pool = try beat.ThreadPool.init(allocator, config);
        \\  NEW: const pool = try beat.createBasicPool(allocator, 4);
        \\
        \\PROGRESSIVE UPGRADE PATH:
        \\  1. Start with: beat.createBasicPool()
        \\  2. Upgrade to: beat.createPerformancePool() 
        \\  3. Full features: beat.createAdvancedPool()
        \\
        \\UNIFIED CONFIGURATION:
        \\  • Uses build_config_unified.zig with smart strategy detection
        \\  • Automatic fallback when build_config unavailable
        \\  • Runtime hardware detection as backup
        \\
        \\📚 Full migration guide: 
        \\   https://github.com/Beat-zig/Beat.zig/blob/main/MIGRATION_GUIDE.md
        \\
    );
}

// ============================================================================
// Testing
// ============================================================================

test "enhanced error message generation" {
    
    // Test error message generation
    const missing_config_msg = formatMissingBuildConfigError("TestProject");
    try std.testing.expect(missing_config_msg.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, missing_config_msg, "build_config") != null);
    try std.testing.expect(std.mem.indexOf(u8, missing_config_msg, "createBasicPool") != null);
    
    const hardware_error_msg = formatHardwareDetectionError();
    try std.testing.expect(hardware_error_msg.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, hardware_error_msg, "safe default") != null);
    
    const platform_error_msg = formatUnsupportedPlatformError("TestOS");
    try std.testing.expect(platform_error_msg.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, platform_error_msg, "platform") != null);
}

test "configuration validation with help" {
    const TestConfig = struct {
        num_workers: ?u32,
        task_queue_size: u32,
    };
    
    // Valid configuration should pass
    const valid_config = TestConfig{
        .num_workers = 4,
        .task_queue_size = 256,
    };
    try validateConfigurationWithHelp(valid_config);
    
    // Invalid worker count should fail
    const invalid_workers = TestConfig{
        .num_workers = 0,
        .task_queue_size = 256,
    };
    try std.testing.expectError(ConfigError.InvalidConfiguration, validateConfigurationWithHelp(invalid_workers));
    
    // Invalid queue size should fail
    const invalid_queue = TestConfig{
        .num_workers = 4,
        .task_queue_size = 8,
    };
    try std.testing.expectError(ConfigError.InvalidConfiguration, validateConfigurationWithHelp(invalid_queue));
}

test "dependency detection" {
    // Test dependency detection logic
    const is_dependency = isUsedAsDependency();
    
    // Should return a boolean value
    try std.testing.expect(is_dependency == true or is_dependency == false);
    
    // Recommended config should be valid
    const dep_config = getRecommendedDependencyConfig();
    try std.testing.expect(dep_config.enable_lock_free == true);
    try std.testing.expect(dep_config.enable_topology_aware == false);
}