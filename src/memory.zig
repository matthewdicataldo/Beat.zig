const std = @import("std");
const builtin = @import("builtin");
const core = @import("core.zig");

// Memory management for ZigPulse

// ============================================================================
// Type-specific Memory Pool
// ============================================================================

pub fn TypedPool(comptime T: type) type {
    return struct {
        const Self = @This();
        const Node = struct {
            next: ?*Node = null,
            data: T = undefined,
        };
        
        head: if (builtin.single_threaded) ?*Node else std.atomic.Value(?*Node),
        allocator: std.mem.Allocator,
        allocated_count: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        free_count: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        
        pub fn init(allocator: std.mem.Allocator) Self {
            if (builtin.single_threaded) {
                return .{ .head = null, .allocator = allocator };
            } else {
                return .{ .head = std.atomic.Value(?*Node).init(null), .allocator = allocator };
            }
        }
        
        pub fn deinit(self: *Self) void {
            // Clean up any remaining nodes
            if (builtin.single_threaded) {
                while (self.head) |node| {
                    self.head = node.next;
                    self.allocator.destroy(node);
                }
            } else {
                while (self.head.load(.acquire)) |node| {
                    const next = node.next;
                    self.head.store(next, .release);
                    self.allocator.destroy(node);
                }
            }
        }
        
        pub fn alloc(self: *Self) !*T {
            if (builtin.single_threaded) {
                if (self.head) |node| {
                    self.head = node.next;
                    _ = self.free_count.fetchSub(1, .monotonic);
                    return &node.data;
                }
            } else {
                // Lock-free fast path
                while (true) {
                    const current = self.head.load(.acquire);
                    if (current) |node| {
                        if (self.head.cmpxchgWeak(current, node.next, .release, .acquire) == null) {
                            _ = self.free_count.fetchSub(1, .monotonic);
                            return &node.data;
                        }
                        continue; // CAS failed, retry
                    }
                    break;
                }
            }
            
            // Slow path: allocate new
            const node = try self.allocator.create(Node);
            _ = self.allocated_count.fetchAdd(1, .monotonic);
            return &node.data;
        }
        
        pub fn free(self: *Self, ptr: *T) void {
            const node: *Node = @alignCast(@fieldParentPtr("data", ptr));
            
            if (builtin.single_threaded) {
                node.next = self.head;
                self.head = node;
                _ = self.free_count.fetchAdd(1, .monotonic);
            } else {
                // Lock-free push
                while (true) {
                    const current = self.head.load(.acquire);
                    node.next = current;
                    if (self.head.cmpxchgWeak(current, node, .release, .acquire) == null) {
                        _ = self.free_count.fetchAdd(1, .monotonic);
                        return;
                    }
                }
            }
        }
        
        pub fn getStats(self: *const Self) PoolStats {
            return .{
                .allocated = self.allocated_count.load(.monotonic),
                .free = self.free_count.load(.monotonic),
            };
        }
        
        pub const PoolStats = struct {
            allocated: u64,
            free: u64,
        };
    };
}

// ============================================================================
// Task Pool (specialized for core.Task)
// ============================================================================

pub const TaskPool = TypedPool(core.Task);

// ============================================================================
// Slab Allocator
// ============================================================================

pub fn SlabAllocator(comptime T: type, comptime slab_size: usize) type {
    return struct {
        const Self = @This();
        const Slab = struct {
            memory: [slab_size]T align(@alignOf(T)) = undefined,
            free_mask: std.bit_set.IntegerBitSet(slab_size) = std.bit_set.IntegerBitSet(slab_size).initFull(),
            next: ?*Slab = null,
        };
        
        // Slab range tracking for O(1) pointer-to-slab lookup
        const SlabRange = struct {
            start_addr: usize,
            end_addr: usize,
            slab: *Slab,
        };
        
        current_slab: ?*Slab,
        allocator: std.mem.Allocator,
        slab_count: usize = 0,
        // O(1) lookup: store ranges in array for binary search
        slab_ranges: std.ArrayList(SlabRange),
        // Cache for most recently accessed slab (optimization)
        last_accessed_slab: ?*Slab = null,
        
        pub fn init(allocator: std.mem.Allocator) !Self {
            var self = Self{
                .current_slab = null,
                .allocator = allocator,
                .slab_ranges = std.ArrayList(SlabRange).init(allocator),
            };
            
            // Pre-allocate first slab
            const first_slab = try allocator.create(Slab);
            first_slab.* = Slab{};
            self.current_slab = first_slab;
            self.slab_count = 1;
            
            // Register first slab range for O(1) lookup
            try self.registerSlabRange(first_slab);
            
            return self;
        }
        
        pub fn deinit(self: *Self) void {
            var current = self.current_slab;
            while (current) |slab| {
                const next = slab.next;
                self.allocator.destroy(slab);
                current = next;
            }
            self.slab_ranges.deinit();
        }
        
        pub fn alloc(self: *Self) !*T {
            var slab = self.current_slab.?;
            
            while (true) {
                const free_idx = slab.free_mask.findFirstSet();
                if (free_idx) |idx| {
                    slab.free_mask.unset(idx);
                    return &slab.memory[idx];
                }
                
                // Current slab is full, try next
                if (slab.next) |next| {
                    slab = next;
                } else {
                    // Allocate new slab
                    const new_slab = try self.allocator.create(Slab);
                    new_slab.* = Slab{};
                    slab.next = new_slab;
                    self.slab_count += 1;
                    
                    // Register new slab range for O(1) lookup
                    try self.registerSlabRange(new_slab);
                    
                    slab = new_slab;
                }
            }
        }
        
        pub fn free(self: *Self, ptr: *T) void {
            const ptr_addr = @intFromPtr(ptr);
            
            // O(1) optimization: Check last accessed slab first (cache locality)
            if (self.last_accessed_slab) |last_slab| {
                if (self.isPointerInSlab(ptr_addr, last_slab)) {
                    const idx = self.getSlabIndex(ptr_addr, last_slab);
                    last_slab.free_mask.set(idx);
                    return;
                }
            }
            
            // O(log n) optimization: Binary search through slab ranges instead of O(n) traversal
            if (self.findSlabForPointer(ptr_addr)) |target_slab| {
                const idx = self.getSlabIndex(ptr_addr, target_slab);
                target_slab.free_mask.set(idx);
                self.last_accessed_slab = target_slab;  // Cache for next access
                return;
            }
            
            // Slab allocator corruption detected - pointer not found in any allocated slab
            // This indicates either memory corruption, double-free, or freeing unallocated memory
            // Help: Check for use-after-free bugs, double-free errors, or incorrect pointer arithmetic
            // Debug: Enable allocator debugging, check pointer validity before free()
            std.debug.panic("SlabAllocator.free: invalid pointer not found in any slab", .{});
        }
        
        // ========================================================================
        // O(1) Slab Lookup Optimization - Eliminates O(n) traversal
        // ========================================================================
        
        /// Register a new slab's memory range for O(1) lookup
        fn registerSlabRange(self: *Self, slab: *Slab) !void {
            const start_addr = @intFromPtr(&slab.memory[0]);
            const end_addr = start_addr + (@sizeOf(T) * slab_size);
            
            const range = SlabRange{
                .start_addr = start_addr,
                .end_addr = end_addr,
                .slab = slab,
            };
            
            try self.slab_ranges.append(range);
            
            // Keep ranges sorted by start address for binary search
            std.sort.pdq(SlabRange, self.slab_ranges.items, {}, compareSlabRanges);
        }
        
        /// Compare function for sorting slab ranges
        fn compareSlabRanges(_: void, a: SlabRange, b: SlabRange) bool {
            return a.start_addr < b.start_addr;
        }
        
        /// O(log n) binary search to find slab containing pointer
        fn findSlabForPointer(self: *Self, ptr_addr: usize) ?*Slab {
            var left: usize = 0;
            var right: usize = self.slab_ranges.items.len;
            
            while (left < right) {
                const mid = left + (right - left) / 2;
                const range = self.slab_ranges.items[mid];
                
                if (ptr_addr >= range.start_addr and ptr_addr < range.end_addr) {
                    return range.slab;  // Found it!
                } else if (ptr_addr < range.start_addr) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            }
            
            return null;  // Not found
        }
        
        /// Check if pointer belongs to specific slab (O(1))
        inline fn isPointerInSlab(self: *Self, ptr_addr: usize, slab: *Slab) bool {
            _ = self;
            const start_addr = @intFromPtr(&slab.memory[0]);
            const end_addr = start_addr + (@sizeOf(T) * slab_size);
            return ptr_addr >= start_addr and ptr_addr < end_addr;
        }
        
        /// Calculate slot index within slab (O(1))
        inline fn getSlabIndex(self: *Self, ptr_addr: usize, slab: *Slab) usize {
            _ = self;
            const start_addr = @intFromPtr(&slab.memory[0]);
            return (ptr_addr - start_addr) / @sizeOf(T);
        }
        
        // Helper function for safe slice containment checking
        fn isSliceContained(inner: []const u8, outer: []const u8) bool {
            const inner_start = @intFromPtr(inner.ptr);
            const outer_start = @intFromPtr(outer.ptr);
            
            // Use Zig's overflow-safe arithmetic to prevent wraparound on 32-bit platforms
            const inner_end = std.math.add(usize, inner_start, inner.len) catch return false;
            const outer_end = std.math.add(usize, outer_start, outer.len) catch return false;
            
            // Safe containment check without wraparound risk
            return inner_start >= outer_start and inner_end <= outer_end;
        }
        
        pub fn reset(self: *Self) void {
            // Reset all slabs for bulk deallocation
            var slab = self.current_slab;
            while (slab) |s| {
                s.free_mask = std.bit_set.IntegerBitSet(slab_size).initFull();
                slab = s.next;
            }
        }
    };
}

// ============================================================================
// NUMA-Aware Allocator
// ============================================================================

pub const NumaAllocator = struct {
    base_allocator: std.mem.Allocator,
    numa_node: u32,
    
    pub fn init(base_allocator: std.mem.Allocator, numa_node: u32) NumaAllocator {
        return .{
            .base_allocator = base_allocator,
            .numa_node = numa_node,
        };
    }
    
    pub fn allocator(self: *NumaAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .free = free,
            },
        };
    }
    
    fn alloc(ctx: *anyopaque, len: usize, ptr_align: u8, ret_addr: usize) ?[*]u8 {
        const self: *NumaAllocator = @ptrCast(@alignCast(ctx));
        
        // On Linux with NUMA support, we would use mbind() here
        // For now, just use base allocator
        return self.base_allocator.rawAlloc(len, ptr_align, ret_addr);
    }
    
    fn resize(ctx: *anyopaque, buf: []u8, buf_align: u8, new_len: usize, ret_addr: usize) bool {
        const self: *NumaAllocator = @ptrCast(@alignCast(ctx));
        return self.base_allocator.rawResize(buf, buf_align, new_len, ret_addr);
    }
    
    fn free(ctx: *anyopaque, buf: []u8, buf_align: u8, ret_addr: usize) void {
        const self: *NumaAllocator = @ptrCast(@alignCast(ctx));
        self.base_allocator.rawFree(buf, buf_align, ret_addr);
    }
};

// ============================================================================
// Cache-Aligned Allocation
// ============================================================================

pub fn alignedAlloc(allocator: std.mem.Allocator, comptime T: type, count: usize) ![]align(core.cache_line_size) T {
    const slice = try allocator.alignedAlloc(T, core.cache_line_size, count);
    return @alignCast(slice);
}

// ============================================================================
// Tests
// ============================================================================

test "typed pool" {
    const allocator = std.testing.allocator;
    
    var pool = TypedPool(u64).init(allocator);
    defer pool.deinit();
    
    // Allocate and free
    const a = try pool.alloc();
    a.* = 42;
    const b = try pool.alloc();
    b.* = 100;
    
    pool.free(a);
    pool.free(b);
    
    // Should reuse
    const c = try pool.alloc();
    const d = try pool.alloc();
    
    try std.testing.expect(c == b); // LIFO order
    try std.testing.expect(d == a);
    
    pool.free(c);
    pool.free(d);
}

test "slab allocator" {
    const allocator = std.testing.allocator;
    
    var slab = try SlabAllocator(u32, 16).init(allocator);
    defer slab.deinit();
    
    var ptrs: [20]*u32 = undefined;
    
    // Allocate more than one slab
    for (&ptrs) |*ptr| {
        ptr.* = try slab.alloc();
    }
    
    // Free half
    for (ptrs[0..10]) |ptr| {
        slab.free(ptr);
    }
    
    // Reset all
    slab.reset();
    
    // Should be able to allocate again
    for (0..5) |_| {
        _ = try slab.alloc();
    }
}