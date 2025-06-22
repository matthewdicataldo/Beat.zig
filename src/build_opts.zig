// ============================================================================
// Legacy Compatibility Shim for build_opts.zig
// Redirects to unified configuration to maintain backward compatibility
// ============================================================================

const unified = @import("build_config_unified.zig");

// Re-export everything from unified configuration
pub const hardware = unified.hardware;
pub const cpu_features = unified.cpu_features;
pub const performance = unified.performance;
pub const build_info = unified.build_info;

// Re-export utility functions
pub const getOptimalConfig = unified.getOptimalConfig;
pub const getTestConfig = unified.getTestConfig;
pub const getBenchmarkConfig = unified.getBenchmarkConfig;
pub const printSummary = unified.printSummary;
pub const validateConfiguration = unified.validateConfiguration;

// Re-export SIMD helpers
pub const OptimalVector = unified.OptimalVector;
pub const shouldVectorize = unified.shouldVectorize;
pub const getSimdAlignment = unified.getSimdAlignment;

// Note: This is a legacy compatibility shim. 
// Please migrate to build_config_unified.zig for improved configuration strategy.