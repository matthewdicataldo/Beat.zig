const std = @import("std");

// SYCL SDK Detection for Beat.zig (Task 3.1.2)
//
// This module provides comprehensive SYCL SDK detection and configuration
// for cross-platform GPU integration. It supports multiple SYCL implementations
// including Intel oneAPI DPC++, hipSYCL (AdaptiveCpp), and ComputeCpp.
//
// Features:
// - Automatic SDK detection and configuration
// - Cross-platform support (Windows, Linux, macOS)
// - Multi-vendor backend support
// - Environment variable detection
// - Build system integration helpers

// ============================================================================
// SYCL Implementation Types and Detection
// ============================================================================

/// Supported SYCL implementations
pub const SyclImplementation = enum {
    none,           // No SYCL implementation found
    intel_dpcpp,    // Intel oneAPI DPC++
    hipsycl,        // hipSYCL/AdaptiveCpp
    computecpp,     // ComputeCpp
    triSYCL,        // triSYCL (experimental)
    
    pub fn getName(self: SyclImplementation) []const u8 {
        return switch (self) {
            .none => "None",
            .intel_dpcpp => "Intel oneAPI DPC++",
            .hipsycl => "hipSYCL/AdaptiveCpp",
            .computecpp => "ComputeCpp",
            .triSYCL => "triSYCL",
        };
    }
};

/// Supported SYCL backends
pub const SyclBackend = enum {
    none,
    opencl,
    level_zero,
    cuda,
    hip,
    native_cpu,
    
    pub fn getName(self: SyclBackend) []const u8 {
        return switch (self) {
            .none => "None",
            .opencl => "OpenCL",
            .level_zero => "Level Zero",
            .cuda => "CUDA",
            .hip => "HIP",
            .native_cpu => "Native CPU",
        };
    }
};

/// SYCL SDK configuration information
pub const SyclSDKConfig = struct {
    // Implementation details
    implementation: SyclImplementation,
    available_backends: std.EnumSet(SyclBackend),
    version: []const u8,
    
    // Paths and environment
    install_path: ?[]const u8,
    include_path: ?[]const u8,
    library_path: ?[]const u8,
    compiler_path: ?[]const u8,
    
    // Build configuration
    compile_flags: std.ArrayList([]const u8),
    link_flags: std.ArrayList([]const u8),
    libraries: std.ArrayList([]const u8),
    
    // Environment variables
    environment_vars: std.StringHashMap([]const u8),
    
    // Capabilities
    supports_unified_memory: bool,
    supports_device_global: bool,
    supports_sub_groups: bool,
    supports_pipes: bool,
    
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) SyclSDKConfig {
        return SyclSDKConfig{
            .implementation = .none,
            .available_backends = std.EnumSet(SyclBackend).init(.{}),
            .version = "",
            .install_path = null,
            .include_path = null,
            .library_path = null,
            .compiler_path = null,
            .compile_flags = std.ArrayList([]const u8).init(allocator),
            .link_flags = std.ArrayList([]const u8).init(allocator),
            .libraries = std.ArrayList([]const u8).init(allocator),
            .environment_vars = std.StringHashMap([]const u8).init(allocator),
            .supports_unified_memory = false,
            .supports_device_global = false,
            .supports_sub_groups = false,
            .supports_pipes = false,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *SyclSDKConfig) void {
        self.compile_flags.deinit();
        self.link_flags.deinit();
        self.libraries.deinit();
        self.environment_vars.deinit();
    }
    
    pub fn isAvailable(self: *const SyclSDKConfig) bool {
        return self.implementation != .none and self.compiler_path != null;
    }
    
    pub fn hasBackend(self: *const SyclSDKConfig, backend: SyclBackend) bool {
        return self.available_backends.contains(backend);
    }
    
    pub fn addCompileFlag(self: *SyclSDKConfig, flag: []const u8) !void {
        try self.compile_flags.append(flag);
    }
    
    pub fn addLinkFlag(self: *SyclSDKConfig, flag: []const u8) !void {
        try self.link_flags.append(flag);
    }
    
    pub fn addLibrary(self: *SyclSDKConfig, lib: []const u8) !void {
        try self.libraries.append(lib);
    }
    
    pub fn setEnvironmentVar(self: *SyclSDKConfig, key: []const u8, value: []const u8) !void {
        try self.environment_vars.put(key, value);
    }
};

// ============================================================================
// SYCL SDK Detection Implementation
// ============================================================================

/// Comprehensive SYCL SDK detector
pub const SyclSDKDetector = struct {
    allocator: std.mem.Allocator,
    config: SyclSDKConfig,
    
    pub fn init(allocator: std.mem.Allocator) SyclSDKDetector {
        return SyclSDKDetector{
            .allocator = allocator,
            .config = SyclSDKConfig.init(allocator),
        };
    }
    
    pub fn deinit(self: *SyclSDKDetector) void {
        self.config.deinit();
    }
    
    /// Detect available SYCL SDK and configure build settings
    pub fn detectSyclSDK(self: *SyclSDKDetector) !bool {
        // Try to detect each SYCL implementation in order of preference
        if (try self.detectIntelDPCPP()) {
            return true;
        }
        
        if (try self.detectHipSYCL()) {
            return true;
        }
        
        if (try self.detectComputeCpp()) {
            return true;
        }
        
        if (try self.detectTriSYCL()) {
            return true;
        }
        
        return false;
    }
    
    /// Detect Intel oneAPI DPC++ implementation
    fn detectIntelDPCPP(self: *SyclSDKDetector) !bool {
        const builtin = @import("builtin");
        
        // Check environment variables
        if (std.process.getEnvVarOwned(self.allocator, "ONEAPI_ROOT")) |oneapi_root| {
            defer self.allocator.free(oneapi_root);
            
            self.config.implementation = .intel_dpcpp;
            self.config.install_path = try self.allocator.dupe(u8, oneapi_root);
            
            // Construct paths based on platform
            switch (builtin.os.tag) {
                .windows => {
                    self.config.compiler_path = try std.fmt.allocPrint(self.allocator, "{s}\\compiler\\latest\\windows\\bin\\dpcpp.exe", .{oneapi_root});
                    self.config.include_path = try std.fmt.allocPrint(self.allocator, "{s}\\compiler\\latest\\windows\\include\\sycl", .{oneapi_root});
                    self.config.library_path = try std.fmt.allocPrint(self.allocator, "{s}\\compiler\\latest\\windows\\lib", .{oneapi_root});
                },
                .linux => {
                    self.config.compiler_path = try std.fmt.allocPrint(self.allocator, "{s}/compiler/latest/linux/bin/dpcpp", .{oneapi_root});
                    self.config.include_path = try std.fmt.allocPrint(self.allocator, "{s}/compiler/latest/linux/include/sycl", .{oneapi_root});
                    self.config.library_path = try std.fmt.allocPrint(self.allocator, "{s}/compiler/latest/linux/lib", .{oneapi_root});
                },
                .macos => {
                    self.config.compiler_path = try std.fmt.allocPrint(self.allocator, "{s}/compiler/latest/mac/bin/dpcpp", .{oneapi_root});
                    self.config.include_path = try std.fmt.allocPrint(self.allocator, "{s}/compiler/latest/mac/include/sycl", .{oneapi_root});
                    self.config.library_path = try std.fmt.allocPrint(self.allocator, "{s}/compiler/latest/mac/lib", .{oneapi_root});
                },
                else => return false,
            }
            
            // Configure DPC++ specific settings
            try self.configureDPCPP();
            return true;
        } else |_| {
            // Try alternative detection methods
            return try self.detectDPCPPAlternative();
        }
    }
    
    /// Alternative Intel DPC++ detection
    fn detectDPCPPAlternative(self: *SyclSDKDetector) !bool {
        // Try to find dpcpp compiler in PATH
        const result = std.process.Child.run(.{
            .allocator = self.allocator,
            .argv = &[_][]const u8{ "which", "dpcpp" },
        }) catch return false;
        defer self.allocator.free(result.stdout);
        defer self.allocator.free(result.stderr);
        
        if (result.term.Exited == 0 and result.stdout.len > 0) {
            self.config.implementation = .intel_dpcpp;
            self.config.compiler_path = try self.allocator.dupe(u8, std.mem.trim(u8, result.stdout, " \t\n\r"));
            try self.configureDPCPP();
            return true;
        }
        
        return false;
    }
    
    /// Configure Intel DPC++ specific settings
    fn configureDPCPP(self: *SyclSDKDetector) !void {
        self.config.version = "oneAPI 2024.x";
        
        // Add available backends for DPC++
        self.config.available_backends.insert(.opencl);
        self.config.available_backends.insert(.level_zero);
        self.config.available_backends.insert(.native_cpu);
        
        // Add compile flags
        try self.config.addCompileFlag("-fsycl");
        try self.config.addCompileFlag("-std=c++17");
        
        // Add backend-specific flags if available
        try self.config.addCompileFlag("-fsycl-targets=spir64");
        
        // Add libraries
        try self.config.addLibrary("sycl");
        
        // Set environment variables
        try self.config.setEnvironmentVar("ONEAPI_DEVICE_SELECTOR", "opencl:gpu");
        
        // Set capabilities
        self.config.supports_unified_memory = true;
        self.config.supports_device_global = true;
        self.config.supports_sub_groups = true;
        self.config.supports_pipes = false;
    }
    
    /// Detect hipSYCL/AdaptiveCpp implementation
    fn detectHipSYCL(self: *SyclSDKDetector) !bool {
        // Check for HIPSYCL_INSTALL_PREFIX or ACPP_ROOT
        if (std.process.getEnvVarOwned(self.allocator, "ACPP_ROOT")) |acpp_root| {
            defer self.allocator.free(acpp_root);
            
            self.config.implementation = .hipsycl;
            self.config.install_path = try self.allocator.dupe(u8, acpp_root);
            self.config.compiler_path = try std.fmt.allocPrint(self.allocator, "{s}/bin/acpp", .{acpp_root});
            
            try self.configureHipSYCL();
            return true;
        } else |_| {}
        
        if (std.process.getEnvVarOwned(self.allocator, "HIPSYCL_INSTALL_PREFIX")) |hipsycl_root| {
            defer self.allocator.free(hipsycl_root);
            
            self.config.implementation = .hipsycl;
            self.config.install_path = try self.allocator.dupe(u8, hipsycl_root);
            self.config.compiler_path = try std.fmt.allocPrint(self.allocator, "{s}/bin/syclcc", .{hipsycl_root});
            
            try self.configureHipSYCL();
            return true;
        } else |_| {}
        
        // Try alternative detection
        return try self.detectHipSYCLAlternative();
    }
    
    /// Alternative hipSYCL detection
    fn detectHipSYCLAlternative(self: *SyclSDKDetector) !bool {
        // Try to find acpp or syclcc compiler in PATH
        const compilers = [_][]const u8{ "acpp", "syclcc" };
        
        for (compilers) |compiler| {
            const result = std.process.Child.run(.{
                .allocator = self.allocator,
                .argv = &[_][]const u8{ "which", compiler },
            }) catch continue;
            defer self.allocator.free(result.stdout);
            defer self.allocator.free(result.stderr);
            
            if (result.term.Exited == 0 and result.stdout.len > 0) {
                self.config.implementation = .hipsycl;
                self.config.compiler_path = try self.allocator.dupe(u8, std.mem.trim(u8, result.stdout, " \t\n\r"));
                try self.configureHipSYCL();
                return true;
            }
        }
        
        return false;
    }
    
    /// Configure hipSYCL specific settings
    fn configureHipSYCL(self: *SyclSDKDetector) !void {
        self.config.version = "AdaptiveCpp/hipSYCL";
        
        // Add available backends for hipSYCL
        self.config.available_backends.insert(.opencl);
        self.config.available_backends.insert(.cuda);
        self.config.available_backends.insert(.hip);
        
        // Add compile flags
        try self.config.addCompileFlag("--acpp-targets=omp");
        try self.config.addCompileFlag("-std=c++17");
        
        // Set environment variables
        try self.config.setEnvironmentVar("HIPSYCL_PLATFORM", "cpu");
        
        // Set capabilities
        self.config.supports_unified_memory = true;
        self.config.supports_device_global = false;
        self.config.supports_sub_groups = true;
        self.config.supports_pipes = false;
    }
    
    /// Detect ComputeCpp implementation
    fn detectComputeCpp(self: *SyclSDKDetector) !bool {
        if (std.process.getEnvVarOwned(self.allocator, "COMPUTECPP_PACKAGE_ROOT_DIR")) |computecpp_root| {
            defer self.allocator.free(computecpp_root);
            
            self.config.implementation = .computecpp;
            self.config.install_path = try self.allocator.dupe(u8, computecpp_root);
            self.config.compiler_path = try std.fmt.allocPrint(self.allocator, "{s}/bin/compute++", .{computecpp_root});
            
            try self.configureComputeCpp();
            return true;
        } else |_| {}
        
        return false;
    }
    
    /// Configure ComputeCpp specific settings
    fn configureComputeCpp(self: *SyclSDKDetector) !void {
        self.config.version = "ComputeCpp";
        
        // Add available backends for ComputeCpp
        self.config.available_backends.insert(.opencl);
        
        // Add compile flags
        try self.config.addCompileFlag("-sycl");
        try self.config.addCompileFlag("-std=c++14");
        
        // Add libraries
        try self.config.addLibrary("ComputeCpp");
        
        // Set capabilities
        self.config.supports_unified_memory = false;
        self.config.supports_device_global = false;
        self.config.supports_sub_groups = true;
        self.config.supports_pipes = true;
    }
    
    /// Detect triSYCL implementation (experimental)
    fn detectTriSYCL(self: *SyclSDKDetector) !bool {
        if (std.process.getEnvVarOwned(self.allocator, "TRISYCL_INCLUDE_DIR")) |trisycl_include| {
            defer self.allocator.free(trisycl_include);
            
            self.config.implementation = .triSYCL;
            self.config.include_path = try self.allocator.dupe(u8, trisycl_include);
            
            try self.configureTriSYCL();
            return true;
        } else |_| {}
        
        return false;
    }
    
    /// Configure triSYCL specific settings
    fn configureTriSYCL(self: *SyclSDKDetector) !void {
        self.config.version = "triSYCL";
        
        // triSYCL is header-only CPU implementation
        self.config.available_backends.insert(.native_cpu);
        
        // Add compile flags
        try self.config.addCompileFlag("-std=c++17");
        try self.config.addCompileFlag("-DTRISYCL");
        
        // Set capabilities
        self.config.supports_unified_memory = true; // CPU only
        self.config.supports_device_global = false;
        self.config.supports_sub_groups = false;
        self.config.supports_pipes = false;
    }
    
    /// Get the detected SDK configuration
    pub fn getConfig(self: *const SyclSDKDetector) *const SyclSDKConfig {
        return &self.config;
    }
    
    /// Print detection summary
    pub fn printSummary(self: *const SyclSDKDetector) void {
        std.debug.print("\\n=== SYCL SDK Detection Summary ===\\n", .{});
        std.debug.print("Implementation: {s}\\n", .{self.config.implementation.getName()});
        std.debug.print("Available: {}\\n", .{self.config.isAvailable()});
        std.debug.print("Version: {s}\\n", .{self.config.version});
        
        if (self.config.install_path) |path| {
            std.debug.print("Install Path: {s}\\n", .{path});
        }
        
        if (self.config.compiler_path) |path| {
            std.debug.print("Compiler Path: {s}\\n", .{path});
        }
        
        std.debug.print("Backends: ", .{});
        var backend_iter = self.config.available_backends.iterator();
        while (backend_iter.next()) |backend| {
            std.debug.print("{s} ", .{backend.getName()});
        }
        std.debug.print("\\n", .{});
        
        std.debug.print("Capabilities:\\n", .{});
        std.debug.print("  Unified Memory: {}\\n", .{self.config.supports_unified_memory});
        std.debug.print("  Device Global: {}\\n", .{self.config.supports_device_global});
        std.debug.print("  Sub Groups: {}\\n", .{self.config.supports_sub_groups});
        std.debug.print("  Pipes: {}\\n", .{self.config.supports_pipes});
        
        std.debug.print("Compile Flags: ", .{});
        for (self.config.compile_flags.items) |flag| {
            std.debug.print("{s} ", .{flag});
        }
        std.debug.print("\\n", .{});
    }
    
    /// Test SDK functionality
    pub fn testSDK(self: *const SyclSDKDetector) !bool {
        if (!self.config.isAvailable()) {
            return false;
        }
        
        // Create a simple test program
        const test_code =
            \\#include <CL/sycl.hpp>
            \\#include <iostream>
            \\int main() {
            \\    try {
            \\        sycl::queue q;
            \\        std::cout << "SYCL test successful" << std::endl;
            \\        return 0;
            \\    } catch (const std::exception& e) {
            \\        std::cout << "SYCL test failed: " << e.what() << std::endl;
            \\        return 1;
            \\    }
            \\}
        ;
        
        // Write test file
        const test_file = "/tmp/sycl_test.cpp";
        var file = std.fs.cwd().createFile(test_file, .{}) catch return false;
        defer file.close();
        defer std.fs.cwd().deleteFile(test_file) catch {};
        
        try file.writeAll(test_code);
        
        // Compile and run test
        const output_file = "/tmp/sycl_test";
        defer std.fs.cwd().deleteFile(output_file) catch {};
        
        var compile_args = std.ArrayList([]const u8).init(self.allocator);
        defer compile_args.deinit();
        
        try compile_args.append(self.config.compiler_path.?);
        for (self.config.compile_flags.items) |flag| {
            try compile_args.append(flag);
        }
        try compile_args.append(test_file);
        try compile_args.append("-o");
        try compile_args.append(output_file);
        
        const compile_result = std.process.Child.run(.{
            .allocator = self.allocator,
            .argv = compile_args.items,
        }) catch return false;
        defer self.allocator.free(compile_result.stdout);
        defer self.allocator.free(compile_result.stderr);
        
        if (compile_result.term.Exited != 0) {
            return false;
        }
        
        // Run the test
        const run_result = std.process.Child.run(.{
            .allocator = self.allocator,
            .argv = &[_][]const u8{output_file},
        }) catch return false;
        defer self.allocator.free(run_result.stdout);
        defer self.allocator.free(run_result.stderr);
        
        return run_result.term.Exited == 0;
    }
};

// ============================================================================
// Build System Integration Helpers
// ============================================================================

/// Create Zig build integration for SYCL
pub fn addSyclSupport(
    b: *std.Build,
    artifact: *std.Build.Step.Compile,
    config: *const SyclSDKConfig,
) !void {
    if (!config.isAvailable()) {
        return;
    }
    
    // Add include paths
    if (config.include_path) |include_path| {
        artifact.addIncludePath(b.path(include_path));
    }
    
    // Add library paths
    if (config.library_path) |lib_path| {
        artifact.addLibraryPath(b.path(lib_path));
    }
    
    // Add libraries
    for (config.libraries.items) |lib| {
        artifact.linkSystemLibrary(lib);
    }
    
    // Add C++ support
    artifact.linkLibCpp();
    
    // Create C++ source compilation
    if (config.compiler_path) |compiler_path| {
        // Add SYCL wrapper sources for compilation
        const wrapper_sources = [_][]const u8{
            "src/sycl_wrapper.cpp",
        };
        
        for (wrapper_sources) |source| {
            // Create a custom compilation step for SYCL C++ sources
            const cpp_compile = b.addSystemCommand(&[_][]const u8{compiler_path});
            
            // Add compile flags
            for (config.compile_flags.items) |flag| {
                cpp_compile.addArg(flag);
            }
            
            // Add source and output
            cpp_compile.addFileArg(b.path(source));
            cpp_compile.addArg("-c");
            cpp_compile.addArg("-o");
            const obj_file = cpp_compile.addOutputFileArg("sycl_wrapper.o");
            
            // Link the object file
            artifact.addObjectFile(obj_file);
            
            // Make sure the C++ compilation happens before the main artifact
            artifact.step.dependOn(&cpp_compile.step);
        }
    }
}

/// Detect and configure SYCL for build
pub fn detectAndConfigureSycl(allocator: std.mem.Allocator) !?SyclSDKConfig {
    var detector = SyclSDKDetector.init(allocator);
    defer detector.deinit();
    
    if (try detector.detectSyclSDK()) {
        return detector.config;
    }
    
    return null;
}

// ============================================================================
// Public API for Build System Integration
// ============================================================================

/// Check if SYCL is available on the system
pub fn isSyclAvailable(allocator: std.mem.Allocator) bool {
    var detector = SyclSDKDetector.init(allocator);
    defer detector.deinit();
    
    return detector.detectSyclSDK() catch false;
}

/// Get SYCL configuration for build system
pub fn getSyclConfig(allocator: std.mem.Allocator) !?SyclSDKConfig {
    return detectAndConfigureSycl(allocator);
}