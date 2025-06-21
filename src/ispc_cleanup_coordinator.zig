// ISPC Cleanup Coordinator - Ensures proper cleanup of all ISPC runtime allocations
// Addresses cross-language memory leaks by providing centralized cleanup management

const std = @import("std");

/// Centralized ISPC cleanup coordinator
pub const ISPCCleanupCoordinator = struct {
    allocator: std.mem.Allocator,
    cleanup_called: bool = false,
    
    const Self = @This();
    
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
        };
    }
    
    /// Comprehensive cleanup of all ISPC runtime allocations
    /// This must be called before program termination to prevent memory leaks
    pub fn cleanupAll(self: *Self) void {
        if (self.cleanup_called) {
            std.log.warn("ISPC cleanup already called, skipping duplicate cleanup", .{});
            return;
        }
        
        std.log.info("Starting comprehensive ISPC runtime cleanup...", .{});
        
        // Step 1: Clean up prediction acceleration system
        self.cleanupPredictionSystem();
        
        // Step 2: Clean up SIMD and optimization caches
        self.cleanupSIMDSystem();
        
        // Step 3: Clean up batch processing allocations
        self.cleanupBatchSystem();
        
        // Step 4: Clean up core ISPC runtime
        self.cleanupCoreRuntime();
        
        // Step 5: Final verification and cleanup
        self.finalCleanup();
        
        self.cleanup_called = true;
        std.log.info("ISPC runtime cleanup completed successfully", .{});
    }
    
    /// Clean up prediction acceleration system allocations
    fn cleanupPredictionSystem(self: *Self) void {
        _ = self;
        std.log.debug("Cleaning up ISPC prediction system...", .{});
        
        // Clean up prediction integration
        const ispc_prediction = @import("ispc_prediction_integration.zig");
        ispc_prediction.deinitGlobalAccelerator();
        
        // Clean up prediction-specific ISPC state
        extern "ispc_free_prediction_caches" fn ispc_free_prediction_caches() void;
        extern "ispc_free_one_euro_filter_pools" fn ispc_free_one_euro_filter_pools() void;
        extern "ispc_free_multi_factor_confidence_state" fn ispc_free_multi_factor_confidence_state() void;
        
        ispc_free_prediction_caches();
        ispc_free_one_euro_filter_pools();
        ispc_free_multi_factor_confidence_state();
    }
    
    /// Clean up SIMD system and worker management
    fn cleanupSIMDSystem(self: *Self) void {
        _ = self;
        std.log.debug("Cleaning up ISPC SIMD system...", .{});
        
        const ispc_simd = @import("ispc_simd_wrapper.zig");
        const ispc_optimized = @import("ispc_optimized.zig");
        
        // Clean up SIMD wrapper resources
        ispc_simd.cleanupAllISPCResources();
        
        // Clean up optimized data structures
        ispc_optimized.OptimizedFingerprints.cleanup();
        
        // Clean up SIMD-specific ISPC state
        extern "ispc_free_simd_capability_cache" fn ispc_free_simd_capability_cache() void;
        extern "ispc_free_vectorized_queue_state" fn ispc_free_vectorized_queue_state() void;
        extern "ispc_free_alignment_buffers" fn ispc_free_alignment_buffers() void;
        
        ispc_free_simd_capability_cache();
        ispc_free_vectorized_queue_state();
        ispc_free_alignment_buffers();
    }
    
    /// Clean up batch processing system allocations
    fn cleanupBatchSystem(self: *Self) void {
        _ = self;
        std.log.debug("Cleaning up ISPC batch system...", .{});
        
        const ispc_integration = @import("ispc_integration.zig");
        
        // Clean up batch-specific allocations
        ispc_integration.RuntimeManagement.cleanupBatchAllocations();
        
        // Clean up kernel-specific batch state
        extern "ispc_free_batch_fingerprint_state" fn ispc_free_batch_fingerprint_state() void;
        extern "ispc_free_batch_worker_scoring_state" fn ispc_free_batch_worker_scoring_state() void;
        extern "ispc_free_batch_optimization_buffers" fn ispc_free_batch_optimization_buffers() void;
        
        ispc_free_batch_fingerprint_state();
        ispc_free_batch_worker_scoring_state();
        ispc_free_batch_optimization_buffers();
    }
    
    /// Clean up core ISPC runtime system
    fn cleanupCoreRuntime(self: *Self) void {
        _ = self;
        std.log.debug("Cleaning up ISPC core runtime...", .{});
        
        const ispc_integration = @import("ispc_integration.zig");
        
        // Clean up core runtime allocations
        ispc_integration.RuntimeManagement.cleanupISPCRuntime();
        
        // Clean up task parallelism system (launch/sync)
        extern "ispc_cleanup_task_parallelism" fn ispc_cleanup_task_parallelism() void;
        extern "ispc_free_async_work_queues" fn ispc_free_async_work_queues() void;
        extern "ispc_reset_launch_sync_state" fn ispc_reset_launch_sync_state() void;
        
        ispc_cleanup_task_parallelism();
        ispc_free_async_work_queues();
        ispc_reset_launch_sync_state();
    }
    
    /// Final cleanup and verification
    fn finalCleanup(self: *Self) void {
        _ = self;
        std.log.debug("Performing final ISPC cleanup...", .{});
        
        // Final comprehensive cleanup
        extern "ispc_shutdown_runtime" fn ispc_shutdown_runtime() void;
        extern "ispc_verify_no_leaks" fn ispc_verify_no_leaks() bool;
        extern "ispc_force_garbage_collection" fn ispc_force_garbage_collection() void;
        
        // Force any remaining cleanup
        ispc_force_garbage_collection();
        
        // Shutdown the ISPC runtime
        ispc_shutdown_runtime();
        
        // Verify no leaks remain (debug builds only)
        if (std.debug.runtime_safety) {
            const no_leaks = ispc_verify_no_leaks();
            if (!no_leaks) {
                std.log.warn("ISPC cleanup verification detected potential remaining allocations", .{});
            } else {
                std.log.debug("ISPC cleanup verification passed - no leaks detected", .{});
            }
        }
    }
    
    /// Emergency cleanup - forces cleanup even if already called
    pub fn emergencyCleanup(self: *Self) void {
        std.log.warn("Performing emergency ISPC cleanup...", .{});
        self.cleanup_called = false; // Reset flag to allow emergency cleanup
        self.cleanupAll();
    }
    
    /// Check if cleanup has been called
    pub fn isCleanedUp(self: *const Self) bool {
        return self.cleanup_called;
    }
};

/// Global cleanup coordinator instance
var global_cleanup_coordinator: ?ISPCCleanupCoordinator = null;

/// Initialize global ISPC cleanup coordinator
pub fn initGlobalCleanupCoordinator(allocator: std.mem.Allocator) void {
    global_cleanup_coordinator = ISPCCleanupCoordinator.init(allocator);
}

/// Get global cleanup coordinator (creates one if not exists)
pub fn getGlobalCleanupCoordinator(allocator: std.mem.Allocator) *ISPCCleanupCoordinator {
    if (global_cleanup_coordinator == null) {
        initGlobalCleanupCoordinator(allocator);
    }
    return &global_cleanup_coordinator.?;
}

/// Perform comprehensive cleanup of all ISPC resources
pub fn cleanupAllISPCResources(allocator: std.mem.Allocator) void {
    var coordinator = getGlobalCleanupCoordinator(allocator);
    coordinator.cleanupAll();
}

/// Emergency cleanup for all ISPC resources
pub fn emergencyCleanupAllISPCResources(allocator: std.mem.Allocator) void {
    var coordinator = getGlobalCleanupCoordinator(allocator);
    coordinator.emergencyCleanup();
}