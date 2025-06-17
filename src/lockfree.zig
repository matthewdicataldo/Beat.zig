const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

// Lock-free data structures for ZigPulse

// ============================================================================
// Task Wrapper
// ============================================================================

pub fn Task(comptime T: type) type {
    return struct {
        data: T,
        next: ?*@This() = null, // For free list
    };
}

// ============================================================================
// Work-Stealing Deque (Chase-Lev Algorithm)
// ============================================================================

pub fn WorkStealingDeque(comptime T: type) type {
    return struct {
        const Self = @This();
        const AtomicU64 = std.atomic.Value(u64);
        
        // HOT PATH: Critical atomic indices (separate cache lines to avoid false sharing)
        hot_data: struct {
            top: AtomicU64,    // Thieves steal from top
            _pad1: [64 - @sizeOf(AtomicU64)]u8 = [_]u8{0} ** (64 - @sizeOf(AtomicU64)),
            bottom: AtomicU64, // Owner works from bottom  
            _pad2: [64 - @sizeOf(AtomicU64)]u8 = [_]u8{0} ** (64 - @sizeOf(AtomicU64)),
        } align(64) = .{
            .top = AtomicU64.init(0),
            .bottom = AtomicU64.init(0),
        },
        
        // WARM PATH: Core deque state (accessed during resize/init)
        buffer: []?T,
        capacity: u64,
        mask: u64,
        allocator: std.mem.Allocator,
        
        // COLD PATH: Statistics (separate cache line)
        stats: struct {
            push_count: AtomicU64 = AtomicU64.init(0),
            pop_count: AtomicU64 = AtomicU64.init(0),
            steal_count: AtomicU64 = AtomicU64.init(0),
            _pad: [64 - 3 * @sizeOf(AtomicU64)]u8 = [_]u8{0} ** (64 - 3 * @sizeOf(AtomicU64)),
        } align(64) = .{},
        
        pub fn init(allocator: std.mem.Allocator, capacity: u64) !Self {
            // Ensure capacity is power of 2
            const actual_capacity = std.math.ceilPowerOfTwo(u64, capacity) catch {
                // Deque capacity must be representable as a power of 2
                // Requested capacity: {}, Maximum supported: 2^63
                // Help: Use a smaller capacity value (e.g., 1024, 4096, 8192)
                // Common values: 256 (small), 1024 (medium), 4096 (large)
                return error.DequeCapacityTooLarge;
            };
            
            const buffer = try allocator.alloc(?T, actual_capacity);
            @memset(buffer, null);
            
            return Self{
                .buffer = buffer,
                .capacity = actual_capacity,
                .mask = actual_capacity - 1,
                .allocator = allocator,
                // hot_data and stats are initialized with defaults
            };
        }
        
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.buffer);
        }
        
        // Owner operations (single producer)
        
        pub fn pushBottom(self: *Self, item: T) !void {
            const b = self.hot_data.bottom.load(.monotonic);
            const t = self.hot_data.top.load(.acquire);
            
            // Check if full (handle wraparound)
            const current_size = if (b >= t) b - t else 0;
            if (current_size >= self.capacity) {
                // Work-stealing deque is full, cannot accept more tasks
                // Current size: {}, Capacity: {}
                // Help: Increase deque capacity, process existing tasks, or implement backpressure
                // Consider: Larger capacity for high-throughput workloads
                return error.WorkStealingDequeFull;
            }
            
            // Store item
            self.buffer[b & self.mask] = item;
            
            // Increment bottom with release ordering to ensure task is visible
            self.hot_data.bottom.store(b + 1, .release);
            
            _ = self.stats.push_count.fetchAdd(1, .monotonic);
        }
        
        pub fn popBottom(self: *Self) ?T {
            const b = self.hot_data.bottom.load(.monotonic);
            if (b == 0) return null;
            
            const new_b = b - 1;
            self.hot_data.bottom.store(new_b, .seq_cst);
            
            const t = self.hot_data.top.load(.monotonic);
            
            if (@as(i64, @intCast(new_b)) < @as(i64, @intCast(t))) {
                // Empty
                self.hot_data.bottom.store(t, .monotonic);
                return null;
            }
            
            const item = self.buffer[new_b & self.mask];
            
            if (new_b == t) {
                // Last element - race with thieves
                if (self.hot_data.top.cmpxchgWeak(t, t + 1, .seq_cst, .monotonic) == null) {
                    // Won the race
                    self.hot_data.bottom.store(t + 1, .monotonic);
                    _ = self.stats.pop_count.fetchAdd(1, .monotonic);
                    return item;
                } else {
                    // Lost the race
                    self.hot_data.bottom.store(t + 1, .monotonic);
                    return null;
                }
            } else {
                // No race
                _ = self.stats.pop_count.fetchAdd(1, .monotonic);
                return item;
            }
        }
        
        // Thief operations (multiple consumers)
        
        pub fn steal(self: *Self) ?T {
            const t = self.hot_data.top.load(.acquire);
            const b = self.hot_data.bottom.load(.seq_cst);
            
            if (@as(i64, @intCast(t)) >= @as(i64, @intCast(b))) {
                // Empty
                return null;
            }
            
            const item = self.buffer[t & self.mask];
            
            // Try to increment top
            if (self.hot_data.top.cmpxchgWeak(t, t + 1, .seq_cst, .monotonic) == null) {
                // Success
                _ = self.stats.steal_count.fetchAdd(1, .monotonic);
                return item;
            }
            
            // Failed - another thief got it
            return null;
        }
        
        pub fn size(self: *const Self) u64 {
            const b = self.hot_data.bottom.load(.monotonic);
            const t = self.hot_data.top.load(.monotonic);
            return if (b >= t) b - t else 0;
        }
        
        pub fn isEmpty(self: *const Self) bool {
            return self.size() == 0;
        }
        
        pub fn getStats(self: *const Self) DequeStats {
            return .{
                .pushes = self.stats.push_count.load(.monotonic),
                .pops = self.stats.pop_count.load(.monotonic),
                .steals = self.stats.steal_count.load(.monotonic),
                .current_size = self.size(),
            };
        }
        
        pub const DequeStats = struct {
            pushes: u64,
            pops: u64,
            steals: u64,
            current_size: u64,
        };
    };
}

// ============================================================================
// MPMC Queue (Multi-Producer Multi-Consumer)
// ============================================================================

pub fn MpmcQueue(comptime T: type, comptime capacity: usize) type {
    // Power of 2 capacity for fast modulo
    const actual_capacity = std.math.ceilPowerOfTwo(usize, capacity) catch unreachable;
    
    return struct {
        const Self = @This();
        const mask = actual_capacity - 1;
        
        buffer: [actual_capacity]Cell,
        enqueue_pos: std.atomic.Value(usize),
        dequeue_pos: std.atomic.Value(usize),
        
        const Cell = struct {
            sequence: std.atomic.Value(usize),
            data: T,
        };
        
        pub fn init() Self {
            var queue = Self{
                .buffer = undefined,
                .enqueue_pos = std.atomic.Value(usize).init(0),
                .dequeue_pos = std.atomic.Value(usize).init(0),
            };
            
            for (&queue.buffer, 0..) |*cell, i| {
                cell.sequence = std.atomic.Value(usize).init(i);
                cell.data = undefined;
            }
            
            return queue;
        }
        
        pub fn enqueue(self: *Self, item: T) bool {
            var pos = self.enqueue_pos.load(.monotonic);
            
            while (true) {
                const cell = &self.buffer[pos & mask];
                const seq = cell.sequence.load(.acquire);
                const diff = @as(i64, @intCast(seq)) - @as(i64, @intCast(pos));
                
                if (diff == 0) {
                    // Cell is ready for enqueue
                    if (self.enqueue_pos.cmpxchgWeak(pos, pos + 1, .monotonic, .monotonic)) |new_pos| {
                        pos = new_pos;
                        continue;
                    }
                    
                    // Success - write data
                    cell.data = item;
                    cell.sequence.store(pos + 1, .release);
                    return true;
                } else if (diff < 0) {
                    // Queue is full
                    return false;
                } else {
                    // Another thread is ahead
                    pos = self.enqueue_pos.load(.monotonic);
                }
            }
        }
        
        pub fn dequeue(self: *Self) ?T {
            var pos = self.dequeue_pos.load(.monotonic);
            
            while (true) {
                const cell = &self.buffer[pos & mask];
                const seq = cell.sequence.load(.acquire);
                const diff = @as(i64, @intCast(seq)) - @as(i64, @intCast(pos + 1));
                
                if (diff == 0) {
                    // Cell has data
                    if (self.dequeue_pos.cmpxchgWeak(pos, pos + 1, .monotonic, .monotonic)) |new_pos| {
                        pos = new_pos;
                        continue;
                    }
                    
                    // Success - read data
                    const data = cell.data;
                    cell.sequence.store(pos + mask + 1, .release);
                    return data;
                } else if (diff < 0) {
                    // Queue is empty
                    return null;
                } else {
                    // Another thread is ahead
                    pos = self.dequeue_pos.load(.monotonic);
                }
            }
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "work stealing deque" {
    const allocator = std.testing.allocator;
    
    var deque = try WorkStealingDeque(i32).init(allocator, 16);
    defer deque.deinit();
    
    // Push some values
    for (0..10) |i| {
        try deque.pushBottom(@intCast(i));
    }
    
    try std.testing.expectEqual(@as(u64, 10), deque.size());
    
    // Pop in LIFO order
    for (0..5) |i| {
        const value = deque.popBottom();
        try std.testing.expect(value != null);
        try std.testing.expectEqual(@as(i32, @intCast(9 - i)), value.?);
    }
    
    // Steal in FIFO order
    for (0..5) |i| {
        const value = deque.steal();
        try std.testing.expect(value != null);
        try std.testing.expectEqual(@as(i32, @intCast(i)), value.?);
    }
    
    try std.testing.expect(deque.isEmpty());
}

test "mpmc queue" {
    var queue = MpmcQueue(u32, 16).init();
    
    // Basic enqueue/dequeue
    try std.testing.expect(queue.enqueue(42));
    try std.testing.expectEqual(@as(u32, 42), queue.dequeue().?);
    
    // Fill queue
    for (0..16) |i| {
        try std.testing.expect(queue.enqueue(@intCast(i)));
    }
    
    // Should be full
    try std.testing.expect(!queue.enqueue(999));
    
    // Drain queue
    for (0..16) |i| {
        try std.testing.expectEqual(@as(u32, @intCast(i)), queue.dequeue().?);
    }
    
    // Should be empty
    try std.testing.expect(queue.dequeue() == null);
}