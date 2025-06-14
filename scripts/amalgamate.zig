const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    
    if (args.len < 5) {
        std.debug.print("Usage: {s} --input <input.zig> --output <output.zig>\n", .{args[0]});
        return;
    }
    
    var input_path: ?[]const u8 = null;
    var output_path: ?[]const u8 = null;
    
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--input") and i + 1 < args.len) {
            input_path = args[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--output") and i + 1 < args.len) {
            output_path = args[i + 1];
            i += 1;
        }
    }
    
    if (input_path == null or output_path == null) {
        std.debug.print("Missing required arguments\n", .{});
        return;
    }
    
    var amalgamator = Amalgamator.init(allocator);
    defer amalgamator.deinit();
    
    try amalgamator.amalgamate(input_path.?, output_path.?);
}

const Amalgamator = struct {
    allocator: std.mem.Allocator,
    output: std.ArrayList(u8),
    processed_files: std.StringHashMap(void),
    module_order: std.ArrayList([]const u8),
    
    pub fn init(allocator: std.mem.Allocator) Amalgamator {
        return .{
            .allocator = allocator,
            .output = std.ArrayList(u8).init(allocator),
            .processed_files = std.StringHashMap(void).init(allocator),
            .module_order = std.ArrayList([]const u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *Amalgamator) void {
        self.output.deinit();
        var iter = self.processed_files.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.processed_files.deinit();
        for (self.module_order.items) |path| {
            self.allocator.free(path);
        }
        self.module_order.deinit();
    }
    
    pub fn amalgamate(self: *Amalgamator, input_path: []const u8, output_path: []const u8) !void {
        _ = input_path; // We use a fixed module order instead
        // Write header
        try self.output.appendSlice(
            \\// ZigPulse - Ultra-optimized parallelism library for Zig
            \\// This is an amalgamated single-file version
            \\// Generated from modular source - do not edit directly
            \\
            \\const std = @import("std");
            \\const builtin = @import("builtin");
            \\const assert = std.debug.assert;
            \\
            \\
        );
        
        // Define our module order to avoid dependency issues
        const modules = [_][]const u8{
            "lockfree.zig",
            "topology.zig", 
            "memory.zig",
            "scheduler.zig",
            "pcall.zig",
            "core.zig",
        };
        
        // Process each module in order
        for (modules) |module| {
            const module_path = try std.fmt.allocPrint(self.allocator, "src/{s}", .{module});
            defer self.allocator.free(module_path);
            
            try self.processModule(module_path);
        }
        
        // Write the main ZigPulse exports
        try self.output.appendSlice(
            \\
            \\// ============================================================================
            \\// ZigPulse Public API
            \\// ============================================================================
            \\
            \\pub const lockfree = lockfree;
            \\pub const topology = topology;
            \\pub const memory = memory;
            \\pub const scheduler = scheduler;
            \\pub const pcall = pcall;
            \\
            \\// Re-export core types and functions
            \\pub const Config = core.Config;
            \\pub const Priority = core.Priority;
            \\pub const TaskStatus = core.TaskStatus;
            \\pub const TaskError = core.TaskError;
            \\pub const Task = core.Task;
            \\pub const ThreadPoolStats = core.ThreadPoolStats;
            \\pub const ThreadPool = core.ThreadPool;
            \\pub const createPool = core.createPool;
            \\pub const createPoolWithConfig = core.createPoolWithConfig;
            \\pub const version = core.version;
            \\pub const cache_line_size = core.cache_line_size;
            \\
        );
        
        // Write output
        const output_file = try std.fs.cwd().createFile(output_path, .{});
        defer output_file.close();
        
        try output_file.writeAll(self.output.items);
        
        std.debug.print("Successfully amalgamated {} files into {s} ({} bytes)\n", .{
            self.processed_files.count(),
            output_path,
            self.output.items.len,
        });
    }
    
    fn processModule(self: *Amalgamator, path: []const u8) !void {
        // Check if already processed
        if (self.processed_files.contains(path)) return;
        
        // Read file
        const file = std.fs.cwd().openFile(path, .{}) catch |err| {
            std.debug.print("Error opening {s}: {}\n", .{ path, err });
            return err;
        };
        defer file.close();
        
        const content = try file.readToEndAlloc(self.allocator, 1024 * 1024);
        defer self.allocator.free(content);
        
        // Mark as processed
        try self.processed_files.put(try self.allocator.dupe(u8, path), {});
        
        // Write section header
        try self.output.writer().print(
            \\// ============================================================================
            \\// From: {s}
            \\// ============================================================================
            \\
        , .{path});
        
        // Get module name
        const basename = std.fs.path.basename(path);
        const module_name = basename[0..basename.len - 4]; // Remove .zig
        
        // Process line by line
        var lines = std.mem.tokenizeScalar(u8, content, '\n');
        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t");
            
            // Skip std/builtin imports (already at top)
            if (std.mem.startsWith(u8, trimmed, "const std = @import(\"std\")")) continue;
            if (std.mem.startsWith(u8, trimmed, "const builtin = @import(\"builtin\")")) continue;
            if (std.mem.startsWith(u8, trimmed, "const assert = std.debug.assert")) continue;
            
            // Skip local imports (we'll handle references internally)
            if (std.mem.indexOf(u8, line, "@import(\"") != null) {
                if (self.isLocalImport(line)) {
                    continue;
                }
            }
            
            // For core.zig, skip re-exports (we'll handle them at the end)
            if (std.mem.eql(u8, module_name, "core")) {
                if (std.mem.startsWith(u8, trimmed, "pub const lockfree =") or
                    std.mem.startsWith(u8, trimmed, "pub const topology =") or
                    std.mem.startsWith(u8, trimmed, "pub const memory =") or
                    std.mem.startsWith(u8, trimmed, "pub const scheduler =") or
                    std.mem.startsWith(u8, trimmed, "pub const pcall =")) {
                    continue;
                }
            }
            
            try self.output.appendSlice(line);
            try self.output.append('\n');
        }
        
        // Don't add self-referential constants - they cause duplicate symbol errors
        if (std.mem.eql(u8, module_name, "core")) {
            // For core, we need a namespace
            try self.output.appendSlice(
                \\
                \\const core = struct {
                \\    pub const Config = Config;
                \\    pub const Priority = Priority;
                \\    pub const TaskStatus = TaskStatus;
                \\    pub const TaskError = TaskError;
                \\    pub const Task = Task;
                \\    pub const ThreadPoolStats = ThreadPoolStats;
                \\    pub const ThreadPool = ThreadPool;
                \\    pub const createPool = createPool;
                \\    pub const createPoolWithConfig = createPoolWithConfig;
                \\    pub const version = version;
                \\    pub const cache_line_size = cache_line_size;
                \\};
                \\
            );
        }
    }
    
    fn isLocalImport(self: *Amalgamator, line: []const u8) bool {
        _ = self;
        const modules = [_][]const u8{
            "lockfree.zig",
            "topology.zig",
            "memory.zig",
            "scheduler.zig",
            "pcall.zig",
            "core.zig",
        };
        
        for (modules) |module| {
            if (std.mem.indexOf(u8, line, module) != null) {
                return true;
            }
        }
        
        return false;
    }
};