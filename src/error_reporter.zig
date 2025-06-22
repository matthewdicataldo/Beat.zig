const std = @import("std");
const builtin = @import("builtin");
const enhanced_errors = @import("enhanced_errors.zig");

// ============================================================================
// Central Error Reporter for Beat.zig
// 
// This module provides centralized error handling that wraps enhanced_errors
// with structured logging, error categorization, and integration points for
// telemetry and monitoring systems.
//
// ISSUE ADDRESSED:
// - Error handling scattered across modules with inconsistent reporting
// - No central point for error analytics and monitoring integration
// - Silent error swallowing (`catch {}`) identified in code review
// - Lack of structured error data for debugging and observability
//
// SOLUTION:
// - Centralized error reporting with categorization and context
// - Structured error logging with machine-readable metadata
// - Integration hooks for telemetry and monitoring systems
// - Error frequency tracking and analytics for proactive issue detection
// - Standardized error handling patterns across all Beat.zig modules
// ============================================================================

/// Error severity levels for structured logging and alerting
pub const ErrorSeverity = enum(u8) {
    debug = 0,      // Debug information, verbose logging
    info = 1,       // Informational messages
    warning = 2,    // Warning conditions that don't affect functionality
    err = 3,        // Error conditions that affect functionality but are recoverable
    critical = 4,   // Critical errors that may cause system instability
    fatal = 5,      // Fatal errors that require immediate shutdown
    
    pub fn toString(self: ErrorSeverity) []const u8 {
        return switch (self) {
            .debug => "DEBUG",
            .info => "INFO",
            .warning => "WARNING", 
            .err => "ERROR",
            .critical => "CRITICAL",
            .fatal => "FATAL",
        };
    }
    
    pub fn toLogLevel(self: ErrorSeverity) std.log.Level {
        return switch (self) {
            .debug => .debug,
            .info => .info,
            .warning => .warn,
            .err => .err,
            .critical => .err,
            .fatal => .err,
        };
    }
};

/// Error categories for classification and analytics
pub const ErrorCategory = enum {
    // Configuration and setup errors
    configuration,          // Invalid configuration parameters
    initialization,         // System initialization failures
    hardware_detection,     // Hardware/platform detection issues
    
    // Resource management errors  
    memory_allocation,      // Memory allocation failures
    resource_exhaustion,    // System resource exhaustion
    file_system,           // File system access errors
    
    // Concurrency and threading errors
    thread_creation,       // Thread creation/management failures
    synchronization,       // Lock contention or deadlock issues
    race_condition,        // Detected race conditions
    
    // Performance and optimization errors
    performance_degradation,  // Performance below expected thresholds
    optimization_failure,     // Optimization system failures
    prediction_accuracy,      // Prediction system accuracy issues
    
    // Integration and compatibility errors
    platform_compatibility,  // Platform-specific compatibility issues
    library_integration,     // External library integration failures
    version_mismatch,        // Version compatibility problems
    
    // Runtime execution errors
    task_execution,         // Task execution failures
    worker_thread,          // Worker thread errors
    queue_operations,       // Queue operation failures
    
    // External system errors
    numa_topology,          // NUMA topology detection/access errors
    cpu_topology,           // CPU topology issues
    memory_pressure,        // Memory pressure monitoring errors
    
    pub fn toString(self: ErrorCategory) []const u8 {
        return switch (self) {
            .configuration => "Configuration",
            .initialization => "Initialization", 
            .hardware_detection => "Hardware Detection",
            .memory_allocation => "Memory Allocation",
            .resource_exhaustion => "Resource Exhaustion",
            .file_system => "File System",
            .thread_creation => "Thread Creation",
            .synchronization => "Synchronization",
            .race_condition => "Race Condition",
            .performance_degradation => "Performance Degradation",
            .optimization_failure => "Optimization Failure",
            .prediction_accuracy => "Prediction Accuracy",
            .platform_compatibility => "Platform Compatibility",
            .library_integration => "Library Integration",
            .version_mismatch => "Version Mismatch",
            .task_execution => "Task Execution",
            .worker_thread => "Worker Thread",
            .queue_operations => "Queue Operations", 
            .numa_topology => "NUMA Topology",
            .cpu_topology => "CPU Topology",
            .memory_pressure => "Memory Pressure",
        };
    }
};

/// Error context providing additional debugging information
pub const ErrorContext = struct {
    module: []const u8,           // Module where error occurred
    function: []const u8,         // Function where error occurred  
    line: ?u32 = null,           // Line number if available
    worker_id: ?u32 = null,      // Worker thread ID if applicable
    task_id: ?u64 = null,        // Task ID if applicable
    numa_node: ?u32 = null,      // NUMA node if applicable
    additional_data: std.StringHashMap([]const u8), // Additional key-value pairs
    
    pub fn init(allocator: std.mem.Allocator, module: []const u8, function: []const u8) ErrorContext {
        return ErrorContext{
            .module = module,
            .function = function,
            .additional_data = std.StringHashMap([]const u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *ErrorContext) void {
        var iterator = self.additional_data.iterator();
        while (iterator.next()) |entry| {
            self.additional_data.allocator.free(entry.key_ptr.*);
            self.additional_data.allocator.free(entry.value_ptr.*);
        }
        self.additional_data.deinit();
    }
    
    /// Add additional context information
    pub fn addData(self: *ErrorContext, key: []const u8, value: []const u8) !void {
        const owned_key = try self.additional_data.allocator.dupe(u8, key);
        const owned_value = try self.additional_data.allocator.dupe(u8, value);
        try self.additional_data.put(owned_key, owned_value);
    }
};

/// Structured error report combining all error information
pub const ErrorReport = struct {
    // Core error information
    error_code: anyerror,
    severity: ErrorSeverity,
    category: ErrorCategory, 
    message: []const u8,
    message_owned: bool, // Whether message should be freed
    context: ErrorContext,
    
    // Timing information
    timestamp_ns: u64,
    thread_id: std.Thread.Id,
    
    // Enhanced error integration
    enhanced_message: ?[]const u8 = null,
    suggested_actions: ?[]const u8 = null,
    
    // Error tracking
    error_id: u64, // Unique identifier for this error instance
    
    pub fn init(
        allocator: std.mem.Allocator,
        error_code: anyerror,
        severity: ErrorSeverity,
        category: ErrorCategory,
        message: []const u8,
        module: []const u8,
        function: []const u8,
    ) ErrorReport {
        return ErrorReport{
            .error_code = error_code,
            .severity = severity,
            .category = category,
            .message = message,
            .message_owned = false, // By default, don't own the message
            .context = ErrorContext.init(allocator, module, function),
            .timestamp_ns = @as(u64, @intCast(std.time.nanoTimestamp())),
            .thread_id = std.Thread.getCurrentId(),
            .error_id = generateErrorId(),
        };
    }
    
    pub fn initWithOwnedMessage(
        allocator: std.mem.Allocator,
        error_code: anyerror,
        severity: ErrorSeverity,
        category: ErrorCategory,
        message: []const u8,
        module: []const u8,
        function: []const u8,
    ) ErrorReport {
        var report = init(allocator, error_code, severity, category, message, module, function);
        report.message_owned = true;
        return report;
    }
    
    pub fn deinit(self: *ErrorReport, allocator: std.mem.Allocator) void {
        self.context.deinit();
        if (self.enhanced_message) |msg| {
            allocator.free(msg);
        }
        if (self.suggested_actions) |actions| {
            allocator.free(actions);
        }
        // Only free message if we own it
        if (self.message_owned) {
            allocator.free(self.message);
        }
    }
    
    /// Generate machine-readable JSON representation  
    pub fn toJson(self: *const ErrorReport, allocator: std.mem.Allocator) ![]u8 {
        // Create JSON object with all error information
        var json_obj = std.json.ObjectMap.init(allocator);
        defer json_obj.deinit();
        
        try json_obj.put("error_id", std.json.Value{ .integer = @as(i64, @bitCast(self.error_id)) });
        try json_obj.put("error_code", std.json.Value{ .string = @errorName(self.error_code) });
        try json_obj.put("severity", std.json.Value{ .string = self.severity.toString() });
        try json_obj.put("category", std.json.Value{ .string = self.category.toString() });
        try json_obj.put("message", std.json.Value{ .string = self.message });
        try json_obj.put("module", std.json.Value{ .string = self.context.module });
        try json_obj.put("function", std.json.Value{ .string = self.context.function });
        try json_obj.put("timestamp_ns", std.json.Value{ .integer = @intCast(self.timestamp_ns) });
        try json_obj.put("thread_id", std.json.Value{ .integer = @intCast(self.thread_id) });
        
        // Add optional context fields
        if (self.context.line) |line| {
            try json_obj.put("line", std.json.Value{ .integer = @intCast(line) });
        }
        if (self.context.worker_id) |worker_id| {
            try json_obj.put("worker_id", std.json.Value{ .integer = @intCast(worker_id) });
        }
        if (self.context.task_id) |task_id| {
            try json_obj.put("task_id", std.json.Value{ .integer = @intCast(task_id) });
        }
        if (self.context.numa_node) |numa_node| {
            try json_obj.put("numa_node", std.json.Value{ .integer = @intCast(numa_node) });
        }
        
        // Add additional context data
        if (self.context.additional_data.count() > 0) {
            var additional_obj = std.json.ObjectMap.init(allocator);
            var iterator = self.context.additional_data.iterator();
            while (iterator.next()) |entry| {
                try additional_obj.put(entry.key_ptr.*, std.json.Value{ .string = entry.value_ptr.* });
            }
            try json_obj.put("additional_data", std.json.Value{ .object = additional_obj });
        }
        
        const json_value = std.json.Value{ .object = json_obj };
        return std.json.stringifyAlloc(allocator, json_value, .{});
    }
    
    /// Generate human-readable formatted message
    pub fn toFormattedString(self: *const ErrorReport, allocator: std.mem.Allocator) ![]u8 {
        const timestamp_s = self.timestamp_ns / 1_000_000_000;
        const timestamp_ms = (self.timestamp_ns / 1_000_000) % 1000;
        
        var formatted = std.ArrayList(u8).init(allocator);
        defer formatted.deinit();
        
        // Header with severity and category
        try formatted.writer().print("[{s}] [{s}] {s}\n", .{ 
            self.severity.toString(), 
            self.category.toString(), 
            self.message 
        });
        
        // Context information
        try formatted.writer().print("  Location: {s}::{s}", .{ self.context.module, self.context.function });
        if (self.context.line) |line| {
            try formatted.writer().print(":{}", .{line});
        }
        try formatted.writer().print("\n", .{});
        
        // Timing and thread info
        try formatted.writer().print("  Time: {}.{:03}s (Thread: {})\n", .{ timestamp_s, timestamp_ms, self.thread_id });
        try formatted.writer().print("  Error: {} (ID: {})\n", .{ self.error_code, self.error_id });
        
        // Optional context
        if (self.context.worker_id) |worker_id| {
            try formatted.writer().print("  Worker: {}\n", .{worker_id});
        }
        if (self.context.task_id) |task_id| {
            try formatted.writer().print("  Task: {}\n", .{task_id});
        }
        if (self.context.numa_node) |numa_node| {
            try formatted.writer().print("  NUMA Node: {}\n", .{numa_node});
        }
        
        // Enhanced error message if available
        if (self.enhanced_message) |enhanced| {
            try formatted.writer().print("\n{s}\n", .{enhanced});
        }
        
        // Suggested actions if available
        if (self.suggested_actions) |actions| {
            try formatted.writer().print("\n💡 Suggested Actions:\n{s}\n", .{actions});
        }
        
        // Additional context data
        if (self.context.additional_data.count() > 0) {
            try formatted.writer().print("\nAdditional Context:\n", .{});
            var iterator = self.context.additional_data.iterator();
            while (iterator.next()) |entry| {
                try formatted.writer().print("  {s}: {s}\n", .{ entry.key_ptr.*, entry.value_ptr.* });
            }
        }
        
        return formatted.toOwnedSlice();
    }
    
    /// Generate unique error ID for tracking and correlation
    fn generateErrorId() u64 {
        // Simple implementation using timestamp + thread ID + random
        const timestamp = @as(u64, @intCast(std.time.nanoTimestamp()));
        const thread_id = std.Thread.getCurrentId();
        const random = std.crypto.random.int(u16);
        
        return timestamp ^ (@as(u64, thread_id) << 32) ^ (@as(u64, random) << 48);
    }
};

/// Error frequency tracking for analytics
const ErrorFrequencyTracker = struct {
    error_counts: std.HashMap(ErrorKey, ErrorStats, ErrorKeyContext, std.hash_map.default_max_load_percentage),
    total_errors: std.atomic.Value(u64),
    start_time_ns: u64,
    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex = .{},
    
    const ErrorKey = struct {
        category: ErrorCategory,
        error_code: anyerror,
    };
    
    const ErrorKeyContext = struct {
        pub fn hash(self: @This(), key: ErrorKey) u64 {
            _ = self;
            var hasher = std.hash.Wyhash.init(0);
            hasher.update(@tagName(key.category));
            hasher.update(@errorName(key.error_code));
            return hasher.final();
        }
        
        pub fn eql(self: @This(), a: ErrorKey, b: ErrorKey) bool {
            _ = self;
            return a.category == b.category and a.error_code == b.error_code;
        }
    };
    
    const ErrorStats = struct {
        count: u64,
        first_seen_ns: u64,
        last_seen_ns: u64,
        severity_distribution: [6]u64, // Count by severity level
    };
    
    fn init(allocator: std.mem.Allocator) ErrorFrequencyTracker {
        return ErrorFrequencyTracker{
            .error_counts = std.HashMap(ErrorKey, ErrorStats, ErrorKeyContext, std.hash_map.default_max_load_percentage).init(allocator),
            .total_errors = std.atomic.Value(u64).init(0),
            .start_time_ns = @as(u64, @intCast(std.time.nanoTimestamp())),
            .allocator = allocator,
        };
    }
    
    fn deinit(self: *ErrorFrequencyTracker) void {
        self.error_counts.deinit();
    }
    
    fn recordError(self: *ErrorFrequencyTracker, category: ErrorCategory, error_code: anyerror, severity: ErrorSeverity) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const key = ErrorKey{ .category = category, .error_code = error_code };
        const now = @as(u64, @intCast(std.time.nanoTimestamp()));
        
        if (self.error_counts.getPtr(key)) |stats| {
            stats.count += 1;
            stats.last_seen_ns = now;
            stats.severity_distribution[@intFromEnum(severity)] += 1;
        } else {
            var new_stats = ErrorStats{
                .count = 1,
                .first_seen_ns = now,
                .last_seen_ns = now,
                .severity_distribution = [_]u64{0} ** 6,
            };
            new_stats.severity_distribution[@intFromEnum(severity)] = 1;
            self.error_counts.put(key, new_stats) catch {
                // If we can't track this error, at least count it in total
                std.log.warn("ErrorReporter: Failed to track error statistics for {}", .{error_code});
            };
        }
        
        _ = self.total_errors.fetchAdd(1, .monotonic);
    }
    
    fn getTopErrors(self: *ErrorFrequencyTracker, allocator: std.mem.Allocator, limit: usize) ![]ErrorFrequencyReport {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var error_list = std.ArrayList(ErrorFrequencyReport).init(allocator);
        defer error_list.deinit();
        
        var iterator = self.error_counts.iterator();
        while (iterator.next()) |entry| {
            try error_list.append(ErrorFrequencyReport{
                .category = entry.key_ptr.category,
                .error_code = entry.key_ptr.error_code,
                .count = entry.value_ptr.count,
                .first_seen_ns = entry.value_ptr.first_seen_ns,
                .last_seen_ns = entry.value_ptr.last_seen_ns,
                .severity_distribution = entry.value_ptr.severity_distribution,
            });
        }
        
        // Sort by count (descending)
        std.sort.insertion(ErrorFrequencyReport, error_list.items, {}, compareErrorFrequency);
        
        // Return top N errors
        const actual_limit = @min(limit, error_list.items.len);
        const owned_slice = try error_list.toOwnedSlice();
        return owned_slice[0..actual_limit];
    }
    
    fn compareErrorFrequency(context: void, a: ErrorFrequencyReport, b: ErrorFrequencyReport) bool {
        _ = context;
        return a.count > b.count;
    }
};

/// Error frequency report for analytics
pub const ErrorFrequencyReport = struct {
    category: ErrorCategory,
    error_code: anyerror,
    count: u64,
    first_seen_ns: u64,
    last_seen_ns: u64,
    severity_distribution: [6]u64,
    
    pub fn getErrorRate(self: ErrorFrequencyReport, time_window_ns: u64) f64 {
        const duration = self.last_seen_ns - self.first_seen_ns;
        if (duration == 0) return 0.0;
        
        const window_duration = @min(duration, time_window_ns);
        return @as(f64, @floatFromInt(self.count)) / (@as(f64, @floatFromInt(window_duration)) / 1_000_000_000.0);
    }
};

/// Configuration for the central error reporter
pub const ErrorReporterConfig = struct {
    /// Enable error frequency tracking
    enable_analytics: bool = true,
    
    /// Enable structured logging output  
    enable_structured_logging: bool = true,
    
    /// Enable integration with enhanced_errors for detailed messages
    enable_enhanced_messages: bool = true,
    
    /// Maximum number of error types to track in analytics
    max_tracked_errors: u32 = 1000,
    
    /// Minimum severity level to report
    min_severity: ErrorSeverity = .warning,
    
    /// Log output destination
    log_output: LogOutput = .standard,
    
    /// Suppress logging output (useful for testing)
    suppress_logging: bool = false,
    
    /// Whether to include stack traces (when available)
    include_stack_traces: bool = false,
    
    pub const LogOutput = enum {
        standard,    // Standard Zig logging
        json_file,   // JSON structured logs to file
        both,        // Both standard and JSON
    };
};

/// Central error reporter managing all Beat.zig error handling
pub const ErrorReporter = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    config: ErrorReporterConfig,
    
    // Error tracking and analytics
    frequency_tracker: ?ErrorFrequencyTracker = null,
    
    // Integration hooks for telemetry (function pointers)
    telemetry_hook: ?*const fn (report: *const ErrorReport) void = null,
    monitoring_hook: ?*const fn (category: ErrorCategory, severity: ErrorSeverity) void = null,
    
    // Statistics
    total_reports: std.atomic.Value(u64),
    reports_by_severity: [6]std.atomic.Value(u64),
    
    pub fn init(allocator: std.mem.Allocator, config: ErrorReporterConfig) !Self {
        var self = Self{
            .allocator = allocator,
            .config = config,
            .total_reports = std.atomic.Value(u64).init(0),
            .reports_by_severity = [_]std.atomic.Value(u64){std.atomic.Value(u64).init(0)} ** 6,
        };
        
        if (config.enable_analytics) {
            self.frequency_tracker = ErrorFrequencyTracker.init(allocator);
        }
        
        return self;
    }
    
    pub fn deinit(self: *Self) void {
        if (self.frequency_tracker) |*tracker| {
            tracker.deinit();
        }
    }
    
    /// Report an error with full context and structured information
    pub fn reportError(
        self: *Self,
        error_code: anyerror,
        severity: ErrorSeverity,
        category: ErrorCategory,
        message: []const u8,
        module: []const u8,
        function: []const u8,
    ) !ErrorReport {
        // Check if error meets minimum severity threshold
        if (@intFromEnum(severity) < @intFromEnum(self.config.min_severity)) {
            return error.BelowMinimumSeverity; // Don't process low-severity errors
        }
        
        // Create error report
        var report = ErrorReport.initWithOwnedMessage(
            self.allocator,
            error_code,
            severity,
            category,
            try self.allocator.dupe(u8, message),
            module,
            function,
        );
        
        // Add enhanced error message if available and enabled
        if (self.config.enable_enhanced_messages) {
            try self.addEnhancedErrorMessage(&report);
        }
        
        // Update statistics
        _ = self.total_reports.fetchAdd(1, .monotonic);
        _ = self.reports_by_severity[@intFromEnum(severity)].fetchAdd(1, .monotonic);
        
        // Track error frequency if analytics enabled
        if (self.frequency_tracker) |*tracker| {
            tracker.recordError(category, error_code, severity);
        }
        
        // Log the error
        try self.logError(&report);
        
        // Call integration hooks
        if (self.telemetry_hook) |hook| {
            hook(&report);
        }
        if (self.monitoring_hook) |hook| {
            hook(category, severity);
        }
        
        return report;
    }
    
    /// Convenience method for reporting errors with context
    pub fn reportErrorWithContext(
        self: *Self,
        error_code: anyerror,
        severity: ErrorSeverity,
        category: ErrorCategory,
        message: []const u8,
        context: ErrorContext,
    ) !ErrorReport {
        var report = try self.reportError(
            error_code,
            severity,
            category,
            message,
            context.module,
            context.function,
        );
        
        // Copy context data
        report.context.line = context.line;
        report.context.worker_id = context.worker_id;
        report.context.task_id = context.task_id;
        report.context.numa_node = context.numa_node;
        
        // Copy additional data
        var iterator = context.additional_data.iterator();
        while (iterator.next()) |entry| {
            try report.context.addData(entry.key_ptr.*, entry.value_ptr.*);
        }
        
        return report;
    }
    
    /// Set telemetry integration hook
    pub fn setTelemetryHook(self: *Self, hook: *const fn (report: *const ErrorReport) void) void {
        self.telemetry_hook = hook;
    }
    
    /// Set monitoring integration hook  
    pub fn setMonitoringHook(self: *Self, hook: *const fn (category: ErrorCategory, severity: ErrorSeverity) void) void {
        self.monitoring_hook = hook;
    }
    
    /// Get error analytics report
    pub fn getAnalyticsReport(self: *Self, allocator: std.mem.Allocator) !ErrorAnalyticsReport {
        var report = ErrorAnalyticsReport{
            .total_errors = self.total_reports.load(.monotonic),
            .errors_by_severity = [_]u64{0} ** 6,
            .top_errors = &[_]ErrorFrequencyReport{},
        };
        
        // Get severity distribution
        for (&self.reports_by_severity, 0..) |*counter, i| {
            report.errors_by_severity[i] = counter.load(.monotonic);
        }
        
        // Get top errors if analytics enabled
        if (self.frequency_tracker) |*tracker| {
            report.top_errors = try tracker.getTopErrors(allocator, 10);
        }
        
        return report;
    }
    
    // Private methods
    
    fn addEnhancedErrorMessage(self: *Self, report: *ErrorReport) !void {
        // Generate enhanced error messages based on error type
        const enhanced_message = switch (report.error_code) {
            enhanced_errors.ConfigError.MissingBuildConfig => enhanced_errors.formatMissingBuildConfigError("Beat.zig"),
            enhanced_errors.ConfigError.HardwareDetectionFailed => enhanced_errors.formatHardwareDetectionError(),
            enhanced_errors.ConfigError.UnsupportedPlatform => enhanced_errors.formatUnsupportedPlatformError("Current Platform"),
            else => null,
        };
        
        if (enhanced_message) |msg| {
            report.enhanced_message = try self.allocator.dupe(u8, msg);
        }
    }
    
    fn logError(self: *Self, report: *const ErrorReport) !void {
        // Skip logging if suppressed (useful for testing)
        if (self.config.suppress_logging) return;
        
        switch (self.config.log_output) {
            .standard => {
                // Standard Zig logging
                const formatted = try report.toFormattedString(self.allocator);
                defer self.allocator.free(formatted);
                
                // Use appropriate log level based on severity
                switch (report.severity.toLogLevel()) {
                    .debug => std.log.debug("ErrorReporter: {s}", .{formatted}),
                    .info => std.log.info("ErrorReporter: {s}", .{formatted}),
                    .warn => std.log.warn("ErrorReporter: {s}", .{formatted}),
                    .err => std.log.err("ErrorReporter: {s}", .{formatted}),
                }
            },
            .json_file => {
                // JSON structured logging (implementation would write to file)
                const json = try report.toJson(self.allocator);
                defer self.allocator.free(json);
                
                // For now, log JSON to standard output
                // In production, this would write to a structured log file
                std.log.info("JSON_ERROR: {s}", .{json});
            },
            .both => {
                // Both standard and JSON
                try self.logError(report); // This will call standard
                
                const json = try report.toJson(self.allocator);
                defer self.allocator.free(json);
                std.log.info("JSON_ERROR: {s}", .{json});
            },
        }
    }
};

/// Analytics report summarizing error patterns
pub const ErrorAnalyticsReport = struct {
    total_errors: u64,
    errors_by_severity: [6]u64,
    top_errors: []ErrorFrequencyReport,
    
    pub fn deinit(self: *ErrorAnalyticsReport, allocator: std.mem.Allocator) void {
        allocator.free(self.top_errors);
    }
    
    pub fn generateSummary(self: *const ErrorAnalyticsReport, allocator: std.mem.Allocator) ![]u8 {
        return std.fmt.allocPrint(allocator,
            \\=== Beat.zig Error Analytics Report ===
            \\
            \\Total Errors: {}
            \\
            \\Errors by Severity:
            \\  • DEBUG: {}
            \\  • INFO: {}  
            \\  • WARNING: {}
            \\  • ERROR: {}
            \\  • CRITICAL: {}
            \\  • FATAL: {}
            \\
            \\Top {} Error Types:
        , .{
            self.total_errors,
            self.errors_by_severity[0],
            self.errors_by_severity[1], 
            self.errors_by_severity[2],
            self.errors_by_severity[3],
            self.errors_by_severity[4],
            self.errors_by_severity[5],
            self.top_errors.len,
        });
    }
};

// ============================================================================
// Global Error Reporter Management
// ============================================================================

/// Global error reporter instance
var global_error_reporter: ?ErrorReporter = null;
var global_error_reporter_mutex: std.Thread.Mutex = .{};

/// Initialize the global error reporter
pub fn initGlobalErrorReporter(allocator: std.mem.Allocator, config: ErrorReporterConfig) !*ErrorReporter {
    global_error_reporter_mutex.lock();
    defer global_error_reporter_mutex.unlock();
    
    if (global_error_reporter) |*reporter| {
        std.log.warn("ErrorReporter: Global error reporter already initialized", .{});
        return reporter;
    }
    
    global_error_reporter = try ErrorReporter.init(allocator, config);
    std.log.info("ErrorReporter: Global error reporter initialized", .{});
    return &global_error_reporter.?;
}

/// Get the global error reporter
pub fn getGlobalErrorReporter() !*ErrorReporter {
    global_error_reporter_mutex.lock();
    defer global_error_reporter_mutex.unlock();
    
    if (global_error_reporter) |*reporter| {
        return reporter;
    }
    
    return error.ErrorReporterNotInitialized;
}

/// Deinitialize the global error reporter
pub fn deinitGlobalErrorReporter() void {
    global_error_reporter_mutex.lock();
    defer global_error_reporter_mutex.unlock();
    
    if (global_error_reporter) |*reporter| {
        reporter.deinit();
        global_error_reporter = null;
        std.log.info("ErrorReporter: Global error reporter shut down", .{});
    }
}

// ============================================================================
// Convenience Macros and Functions
// ============================================================================

/// Convenience function for reporting errors (most common case)
pub fn reportError(
    error_code: anyerror,
    severity: ErrorSeverity,
    category: ErrorCategory,
    comptime message_fmt: []const u8,
    args: anytype,
    comptime module: []const u8,
    comptime function: []const u8,
) void {
    const reporter = getGlobalErrorReporter() catch {
        // Fallback to standard logging if error reporter not available
        std.log.err("ErrorReporter not initialized: {} - " ++ message_fmt, .{error_code} ++ args);
        return;
    };
    
    const allocator = reporter.allocator;
    const message = std.fmt.allocPrint(allocator, message_fmt, args) catch {
        std.log.err("Failed to format error message: {}", .{error_code});
        return;
    };
    defer allocator.free(message);
    
    const report = reporter.reportError(
        error_code,
        severity,
        category,
        message,
        module,
        function,
    ) catch |err| {
        std.log.err("Failed to report error: {} (original error: {})", .{ err, error_code });
        return;
    };
    
    // Clean up the report (in real usage, caller might want to keep it)
    var mutable_report = report;
    mutable_report.deinit(allocator);
}

/// Macro-like function for convenient error reporting with source location
pub fn REPORT_ERROR(
    error_code: anyerror,
    severity: ErrorSeverity,
    category: ErrorCategory,
    comptime message_fmt: []const u8,
    args: anytype,
    comptime src: std.builtin.SourceLocation,
) void {
    reportError(
        error_code,
        severity,
        category,
        message_fmt,
        args,
        src.file,
        src.fn_name,
    );
}

// ============================================================================
// Testing
// ============================================================================

const TestError = error{TestError};

test "ErrorReporter basic functionality" {
    const allocator = std.testing.allocator;
    
    const config = ErrorReporterConfig{
        .enable_analytics = true,
        .enable_structured_logging = true,
        .min_severity = .debug,
        .suppress_logging = true, // Quiet mode for testing
    };
    
    var reporter = try ErrorReporter.init(allocator, config);
    defer reporter.deinit();
    
    // Test error reporting
    var report = try reporter.reportError(
        TestError.TestError,
        .err,
        .configuration,
        "Test error message",
        "test_module",
        "test_function",
    );
    defer report.deinit(allocator);
    
    try std.testing.expect(report.error_code == TestError.TestError);
    try std.testing.expect(report.severity == .err);
    try std.testing.expect(report.category == .configuration);
    try std.testing.expect(std.mem.eql(u8, report.message, "Test error message"));
}

test "ErrorReport JSON serialization" {
    const allocator = std.testing.allocator;
    
    var report = ErrorReport.init(
        allocator,
        TestError.TestError,
        .warning,
        .memory_allocation,
        "Test JSON serialization",
        "test_module",
        "test_function",
    );
    defer report.deinit(allocator);
    
    const json = try report.toJson(allocator);
    defer allocator.free(json);
    
    try std.testing.expect(json.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, json, "TestError") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "WARNING") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "Memory Allocation") != null);
}

test "ErrorReporter analytics" {
    const allocator = std.testing.allocator;
    
    const config = ErrorReporterConfig{
        .enable_analytics = true,
        .min_severity = .debug,
        .suppress_logging = true, // Quiet mode for testing
    };
    
    var reporter = try ErrorReporter.init(allocator, config);
    defer reporter.deinit();
    
    // Report several errors
    for (0..5) |i| {
        var report = try reporter.reportError(
            TestError.TestError,
            .err,
            .configuration,
            "Test error",
            "test_module",
            "test_function",
        );
        report.deinit(allocator);
        _ = i;
    }
    
    // Get analytics report
    var analytics = try reporter.getAnalyticsReport(allocator);
    defer analytics.deinit(allocator);
    
    try std.testing.expect(analytics.total_errors == 5);
    try std.testing.expect(analytics.errors_by_severity[3] == 5); // .err level
}