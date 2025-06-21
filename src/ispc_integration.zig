// ISPC Integration Module for Beat.zig
// Provides seamless integration between Zig and Intel SPMD Program Compiler (ISPC)
// Based on the comprehensive technical analysis in ZIG-ISPC.md

const std = @import("std");
const builtin = @import("builtin");

// ISPC extern function declarations
extern fn ispc_free_fingerprint_cache() void;
extern fn ispc_free_batch_formation_cache() void;
extern fn ispc_free_worker_selection_cache() void;
extern fn ispc_free_batch_optimization_state() void;
extern fn ispc_free_similarity_matrix_cache() void;
extern fn ispc_free_worker_scoring_cache() void;
extern fn ispc_initialize_runtime() void;

/// ISPC integration configuration and utilities
pub const ISPC = struct {
    /// Represents ISPC varying data - a collection of values processed in parallel
    pub fn Varying(comptime T: type) type {
        return struct {
            data: []T,
            
            const Self = @This();
            
            pub fn init(allocator: std.mem.Allocator, count: usize) !Self {
                return Self{
                    .data = try allocator.alloc(T, count),
                };
            }
            
            pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
                allocator.free(self.data);
            }
            
            pub fn ptr(self: *Self) *T {
                return self.data.ptr;
            }
            
            pub fn len(self: Self) usize {
                return self.data.len;
            }
        };
    }
    
    /// ISPC compilation targets matching hardware capabilities
    pub const Target = enum {
        host,           // Auto-detect host SIMD capabilities
        sse2_i32x4,     // SSE2 128-bit, 4-wide int32
        sse4_i32x4,     // SSE4.1 128-bit, 4-wide int32
        avx1_i32x8,     // AVX 256-bit, 8-wide int32
        avx2_i32x8,     // AVX2 256-bit, 8-wide int32
        avx512knl_i32x16,  // AVX-512 Knights Landing 512-bit, 16-wide int32
        avx512skx_i32x16,  // AVX-512 Skylake-X 512-bit, 16-wide int32
        neon_i32x4,     // ARM NEON 128-bit, 4-wide int32
        
        pub fn toString(self: Target) []const u8 {
            return switch (self) {
                .host => "host",
                .sse2_i32x4 => "sse2-i32x4",
                .sse4_i32x4 => "sse4-i32x4",
                .avx1_i32x8 => "avx1-i32x8",
                .avx2_i32x8 => "avx2-i32x8",
                .avx512knl_i32x16 => "avx512knl-i32x16",
                .avx512skx_i32x16 => "avx512skx-i32x16",
                .neon_i32x4 => "neon-i32x4",
            };
        }
    };
    
    /// Compilation options for ISPC kernels
    pub const CompileOptions = struct {
        target: Target = .host,
        optimization: enum { O0, O1, O2, O3 } = .O2,
        debug: bool = false,
        math_lib: enum { default, fast, precise } = .default,
        addressing: enum { @"32", @"64" } = if (@sizeOf(usize) == 8) .@"64" else .@"32",
        quiet: bool = true,
        
        pub fn toArgs(self: CompileOptions, allocator: std.mem.Allocator) ![][]const u8 {
            var args = std.ArrayList([]const u8).init(allocator);
            defer args.deinit();
            
            try args.append("--target");
            try args.append(self.target.toString());
            
            const opt_flag = switch (self.optimization) {
                .O0 => "-O0",
                .O1 => "-O1", 
                .O2 => "-O2",
                .O3 => "-O3",
            };
            try args.append(opt_flag);
            
            if (self.debug) {
                try args.append("-g");
            }
            
            const math_flag = switch (self.math_lib) {
                .default => null,
                .fast => "--math-lib=fast",
                .precise => "--math-lib=precise",
            };
            if (math_flag) |flag| {
                try args.append(flag);
            }
            
            const addr_flag = switch (self.addressing) {
                .@"32" => "--addressing=32",
                .@"64" => "--addressing=64",
            };
            try args.append(addr_flag);
            
            if (self.quiet) {
                try args.append("--quiet");
            }
            
            return args.toOwnedSlice();
        }
    };
    
    /// Auto-detect optimal ISPC target based on CPU capabilities
    pub fn detectOptimalTarget() Target {
        const features = std.Target.x86.featureSetHas;
        
        if (builtin.cpu.arch == .x86_64) {
            if (features(builtin.cpu.features, .avx512f)) {
                return .avx512skx_i32x16;
            } else if (features(builtin.cpu.features, .avx2)) {
                return .avx2_i32x8;
            } else if (features(builtin.cpu.features, .avx)) {
                return .avx1_i32x8;
            } else if (features(builtin.cpu.features, .sse4_1)) {
                return .sse4_i32x4;
            } else {
                return .sse2_i32x4;
            }
        } else if (builtin.cpu.arch == .aarch64) {
            return .neon_i32x4;
        } else {
            return .host;
        }
    }
    
    /// Get the SIMD width for a given target
    pub fn getSimdWidth(target: Target) u32 {
        return switch (target) {
            .host => 8, // Conservative default
            .sse2_i32x4, .sse4_i32x4, .neon_i32x4 => 4,
            .avx1_i32x8, .avx2_i32x8 => 8,
            .avx512knl_i32x16, .avx512skx_i32x16 => 16,
        };
    }
};

/// Build system integration for ISPC compilation
pub const BuildIntegration = struct {
    /// Create an ISPC compilation step in the build graph
    pub fn addISPCStep(
        b: *std.Build,
        source_path: []const u8,
        options: ISPC.CompileOptions,
    ) *std.Build.Step {
        const cache_dir = b.cache_root.join(b.allocator, &.{"ispc"}) catch @panic("OOM");
        std.fs.makeDirAbsolute(cache_dir) catch |err| switch (err) {
            error.PathAlreadyExists => {}, // Expected case - directory already exists
            else => {
                // Log unexpected errors but continue - build system should be resilient
                std.log.warn("Failed to create ISPC cache directory '{}': {}", .{std.fmt.fmtSliceHexLower(cache_dir), err});
            },
        };
        
        const basename = std.fs.path.basename(source_path);
        const name_without_ext = if (std.mem.endsWith(u8, basename, ".ispc"))
            basename[0..basename.len - 5]
        else
            basename;
            
        const obj_path = std.fs.path.join(b.allocator, &.{
            cache_dir, 
            b.fmt("{s}.o", .{name_without_ext})
        }) catch @panic("OOM");
        
        const header_path = std.fs.path.join(b.allocator, &.{
            cache_dir,
            b.fmt("{s}.h", .{name_without_ext})
        }) catch @panic("OOM");
        
        const args = options.toArgs(b.allocator) catch @panic("OOM");
        defer b.allocator.free(args);
        
        var cmd_args = std.ArrayList([]const u8).init(b.allocator);
        defer cmd_args.deinit();
        
        cmd_args.append("ispc") catch @panic("OOM");
        cmd_args.append(source_path) catch @panic("OOM");
        cmd_args.append("-o") catch @panic("OOM");
        cmd_args.append(obj_path) catch @panic("OOM");
        cmd_args.append("-h") catch @panic("OOM");
        cmd_args.append(header_path) catch @panic("OOM");
        cmd_args.appendSlice(args) catch @panic("OOM");
        
        const ispc_step = b.addSystemCommand(cmd_args.items);
        
        // Store paths for later retrieval
        ispc_step.addFileSourceArg(.{ .path = obj_path });
        ispc_step.addFileSourceArg(.{ .path = header_path });
        
        return &ispc_step.step;
    }
    
    /// Helper to integrate ISPC object files with executable
    pub fn linkISPCWithExecutable(
        exe: *std.Build.Step.Compile,
        ispc_step: *std.Build.Step,
        obj_path: []const u8,
        header_dir: []const u8,
    ) void {
        exe.addObjectFile(.{ .path = obj_path });
        exe.addIncludePath(.{ .path = header_dir });
        exe.step.dependOn(ispc_step);
    }
};

/// Performance-critical ISPC kernel interfaces for Beat.zig
pub const Kernels = struct {
    /// Fingerprint similarity computation using ISPC SPMD
    pub const FingerprintSimilarity = struct {
        pub extern "ispc_compute_fingerprint_similarity" fn ispc_compute_fingerprint_similarity(
            fingerprints_a: [*]const u128,
            fingerprints_b: [*]const u128,
            results: [*]f32,
            count: c_int,
        ) void;
        
        pub fn computeSimilarity(
            allocator: std.mem.Allocator,
            fingerprints_a: []const u128,
            fingerprints_b: []const u128,
        ) ![]f32 {
            std.debug.assert(fingerprints_a.len == fingerprints_b.len);
            
            const results = try allocator.alloc(f32, fingerprints_a.len);
            errdefer allocator.free(results);
            
            ispc_compute_fingerprint_similarity(
                fingerprints_a.ptr,
                fingerprints_b.ptr,
                results.ptr,
                @intCast(fingerprints_a.len),
            );
            
            return results;
        }
        
        /// Clean up any internal ISPC state for fingerprint operations
        pub fn cleanup() void {
            ispc_free_fingerprint_cache();
        }
    };
    
    /// Batch formation optimization using ISPC
    pub const BatchOptimization = struct {
        pub extern "ispc_optimize_batch_formation" fn ispc_optimize_batch_formation(
            task_scores: [*]const f32,
            similarity_matrix: [*]const f32,
            batch_indices: [*]u32,
            count: c_int,
            batch_size: c_int,
        ) c_int;
        
        pub fn optimizeBatchFormation(
            allocator: std.mem.Allocator,
            task_scores: []const f32,
            similarity_matrix: []const f32,
            max_batch_size: u32,
        ) ![]u32 {
            const batch_indices = try allocator.alloc(u32, max_batch_size);
            errdefer allocator.free(batch_indices);
            
            const actual_batch_size = ispc_optimize_batch_formation(
                task_scores.ptr,
                similarity_matrix.ptr,
                batch_indices.ptr,
                @intCast(task_scores.len),
                @intCast(max_batch_size),
            );
            
            return batch_indices[0..@intCast(actual_batch_size)];
        }
        
        /// Clean up batch optimization internal state
        pub fn cleanup() void {
            ispc_free_batch_formation_cache();
        }
    };
    
    /// Multi-criteria worker selection scoring using ISPC
    pub const WorkerSelection = struct {
        pub extern "ispc_compute_worker_scores" fn ispc_compute_worker_scores(
            worker_loads: [*]const f32,
            numa_distances: [*]const f32, 
            cache_affinities: [*]const f32,
            worker_scores: [*]f32,
            worker_count: c_int,
        ) void;
        
        pub fn computeWorkerScores(
            allocator: std.mem.Allocator,
            worker_loads: []const f32,
            numa_distances: []const f32,
            cache_affinities: []const f32,
        ) ![]f32 {
            std.debug.assert(worker_loads.len == numa_distances.len);
            std.debug.assert(worker_loads.len == cache_affinities.len);
            
            const scores = try allocator.alloc(f32, worker_loads.len);
            errdefer allocator.free(scores);
            
            ispc_compute_worker_scores(
                worker_loads.ptr,
                numa_distances.ptr,
                cache_affinities.ptr,
                scores.ptr,
                @intCast(worker_loads.len),
            );
            
            return scores;
        }
        
        /// Clean up worker selection internal caches
        pub fn cleanup() void {
            ispc_free_worker_selection_cache();
        }
    };
};

/// Testing utilities for ISPC integration
pub const Testing = struct {
    pub fn benchmarkISPCvsNative(
        allocator: std.mem.Allocator,
        comptime native_fn: anytype,
        comptime ispc_fn: anytype,
        input_data: anytype,
        iterations: u32,
    ) !struct { native_time: u64, ispc_time: u64, speedup: f64 } {
        _ = allocator; // Reserved for future temporary buffer allocation
        var timer = try std.time.Timer.start();
        
        // Benchmark native implementation
        timer.reset();
        for (0..iterations) |_| {
            _ = native_fn(input_data);
        }
        const native_time = timer.read();
        
        // Benchmark ISPC implementation  
        timer.reset();
        for (0..iterations) |_| {
            _ = ispc_fn(input_data);
        }
        const ispc_time = timer.read();
        
        const speedup = @as(f64, @floatFromInt(native_time)) / @as(f64, @floatFromInt(ispc_time));
        
        return .{
            .native_time = native_time,
            .ispc_time = ispc_time,
            .speedup = speedup,
        };
    }
};

/// ISPC Runtime Management - Missing free helpers for internal allocations
pub const RuntimeManagement = struct {
    /// External ISPC runtime cleanup functions
    extern "ispc_cleanup_task_system" fn ispc_cleanup_task_system() void;
    extern "ispc_free_internal_caches" fn ispc_free_internal_caches() void;
    extern "ispc_reset_async_state" fn ispc_reset_async_state() void;
    extern "ispc_deallocate_work_queues" fn ispc_deallocate_work_queues() void;
    
    /// Comprehensive ISPC runtime cleanup
    pub fn cleanupISPCRuntime() void {
        // Clean up task parallelism system (launch/sync allocations)
        ispc_cleanup_task_system();
        
        // Free internal prediction caches and lookup tables
        ispc_free_internal_caches();
        
        // Reset async task state and worker queues
        ispc_reset_async_state();
        
        // Deallocate work-stealing queue structures
        ispc_deallocate_work_queues();
    }
    
    /// Clean up specific batch operation allocations
    pub fn cleanupBatchAllocations() void {
        // These functions handle cleanup of internal batch processing state
        ispc_free_batch_optimization_state();
        ispc_free_similarity_matrix_cache();
        ispc_free_worker_scoring_cache();
    }
    
    /// Initialize ISPC runtime (should be called once)
    pub fn initializeISPCRuntime() void {
        ispc_initialize_runtime();
    }
};

/// Configuration and runtime detection
pub const Config = struct {
    pub fn detectISPCCapabilities() struct {
        target: ISPC.Target,
        simd_width: u32,
        estimated_speedup: f32,
    } {
        const optimal_target = ISPC.detectOptimalTarget();
        const simd_width = ISPC.getSimdWidth(optimal_target);
        
        // Conservative speedup estimates based on SIMD width
        const estimated_speedup = switch (simd_width) {
            4 => 3.0,   // SSE/NEON: ~3x speedup
            8 => 5.0,   // AVX/AVX2: ~5x speedup  
            16 => 8.0,  // AVX-512: ~8x speedup
            else => 2.0,
        };
        
        return .{
            .target = optimal_target,
            .simd_width = simd_width,
            .estimated_speedup = estimated_speedup,
        };
    }
};