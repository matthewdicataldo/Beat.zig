const std = @import("std");
const builtin = @import("builtin");
const core = @import("core.zig");

// Scheduling algorithms for ZigPulse

// ============================================================================
// Token Accounting (v2 Heartbeat)
// ============================================================================

pub const TokenAccount = struct {
    work_cycles: u64 = 0,
    overhead_cycles: u64 = 0,
    promotion_threshold: u64,
    min_work_cycles: u64,
    should_promote_cached: bool = false,
    last_check_cycles: u64 = 0,
    
    const CHECK_INTERVAL = 1000;
    
    pub fn init(config: *const core.Config) TokenAccount {
        return .{
            .promotion_threshold = config.promotion_threshold,
            .min_work_cycles = config.min_work_cycles,
        };
    }
    
    pub inline fn update(self: *TokenAccount, work: u64, overhead: u64) void {
        self.work_cycles +%= work;
        self.overhead_cycles +%= overhead;
        
        if (self.overhead_cycles -% self.last_check_cycles > CHECK_INTERVAL) {
            self.last_check_cycles = self.overhead_cycles;
            self.should_promote_cached = self.work_cycles > (self.overhead_cycles * self.promotion_threshold);
        }
    }
    
    pub inline fn shouldPromote(self: *const TokenAccount) bool {
        return self.should_promote_cached;
    }
    
    pub fn reset(self: *TokenAccount) void {
        self.work_cycles = 0;
        self.overhead_cycles = 0;
        self.should_promote_cached = false;
        self.last_check_cycles = 0;
    }
};

// ============================================================================
// Heartbeat Scheduler
// ============================================================================

pub const Scheduler = struct {
    allocator: std.mem.Allocator,
    config: *const core.Config,
    heartbeat_thread: ?std.Thread = null,
    running: std.atomic.Value(bool) = std.atomic.Value(bool).init(true),
    
    // Per-worker token accounts
    worker_tokens: []TokenAccount,
    
    pub fn init(allocator: std.mem.Allocator, config: *const core.Config) !*Scheduler {
        const self = try allocator.create(Scheduler);
        self.* = .{
            .allocator = allocator,
            .config = config,
            .worker_tokens = try allocator.alloc(TokenAccount, config.num_workers orelse 1),
        };
        
        // Initialize token accounts
        for (self.worker_tokens) |*tokens| {
            tokens.* = TokenAccount.init(config);
        }
        
        // Start heartbeat thread if enabled
        if (config.enable_heartbeat) {
            self.heartbeat_thread = try std.Thread.spawn(.{}, heartbeatLoop, .{self});
        }
        
        return self;
    }
    
    pub fn deinit(self: *Scheduler) void {
        self.running.store(false, .release);
        
        if (self.heartbeat_thread) |thread| {
            thread.join();
        }
        
        self.allocator.free(self.worker_tokens);
        self.allocator.destroy(self);
    }
    
    fn heartbeatLoop(self: *Scheduler) void {
        const interval_ns = @as(u64, self.config.heartbeat_interval_us) * 1000;
        
        while (self.running.load(.acquire)) {
            std.time.sleep(interval_ns);
            
            // Periodic promotion check
            for (self.worker_tokens) |*tokens| {
                if (tokens.shouldPromote()) {
                    // TODO: Trigger work promotion
                    tokens.reset();
                }
            }
        }
    }
};

// ============================================================================
// Thread-Local Scheduler State
// ============================================================================

threadlocal var tls_tokens: TokenAccount = undefined;
threadlocal var tls_tokens_init: bool = false;
threadlocal var tls_worker_id: u32 = 0;

pub fn registerWorker(scheduler: *Scheduler, worker_id: u32) void {
    tls_worker_id = worker_id;
    tls_tokens = scheduler.worker_tokens[worker_id];
    tls_tokens_init = true;
}

pub fn getLocalTokens() ?*TokenAccount {
    if (tls_tokens_init) {
        return &tls_tokens;
    }
    return null;
}

// ============================================================================
// Work Prediction (v3 - TODO)
// ============================================================================

pub const TaskPredictor = struct {
    history: std.AutoHashMap(u64, TaskStats),
    allocator: std.mem.Allocator,
    
    const TaskStats = struct {
        total_cycles: u64 = 0,
        execution_count: u64 = 0,
        avg_cycles: f64 = 0,
        variance: f64 = 0,
        last_seen: i64 = 0,
    };
    
    pub fn init(allocator: std.mem.Allocator) TaskPredictor {
        return .{
            .history = std.AutoHashMap(u64, TaskStats).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *TaskPredictor) void {
        self.history.deinit();
    }
    
    pub fn recordExecution(self: *TaskPredictor, task_hash: u64, cycles: u64) !void {
        const result = try self.history.getOrPut(task_hash);
        const stats = &result.value_ptr.*;
        
        if (!result.found_existing) {
            stats.* = TaskStats{};
        }
        
        // Update statistics
        stats.total_cycles += cycles;
        stats.execution_count += 1;
        
        const old_avg = stats.avg_cycles;
        stats.avg_cycles = @as(f64, @floatFromInt(stats.total_cycles)) / @as(f64, @floatFromInt(stats.execution_count));
        
        // Update variance (Welford's algorithm)
        if (stats.execution_count > 1) {
            const delta = @as(f64, @floatFromInt(cycles)) - old_avg;
            const delta2 = @as(f64, @floatFromInt(cycles)) - stats.avg_cycles;
            stats.variance += delta * delta2;
        }
        
        stats.last_seen = std.time.timestamp();
    }
    
    pub fn predict(self: *TaskPredictor, task_hash: u64) ?PredictedStats {
        if (self.history.get(task_hash)) |stats| {
            if (stats.execution_count > 0) {
                return PredictedStats{
                    .expected_cycles = @as(u64, @intFromFloat(stats.avg_cycles)),
                    .confidence = @min(1.0, @as(f64, @floatFromInt(stats.execution_count)) / 100.0),
                    .variance = if (stats.execution_count > 1) 
                        stats.variance / @as(f64, @floatFromInt(stats.execution_count - 1))
                    else 
                        0,
                };
            }
        }
        return null;
    }
    
    pub const PredictedStats = struct {
        expected_cycles: u64,
        confidence: f64,
        variance: f64,
    };
};

// ============================================================================
// Performance Measurement
// ============================================================================

pub inline fn rdtsc() u64 {
    if (comptime builtin.cpu.arch == .x86_64) {
        var low: u32 = undefined;
        var high: u32 = undefined;
        asm volatile ("rdtsc" : [low] "={eax}" (low), [high] "={edx}" (high));
        return (@as(u64, high) << 32) | low;
    } else {
        return @as(u64, @intCast(std.time.nanoTimestamp()));
    }
}

// ============================================================================
// Tests
// ============================================================================

test "token accounting" {
    const config = core.Config{
        .promotion_threshold = 10,
        .min_work_cycles = 1000,
    };
    
    var tokens = TokenAccount.init(&config);
    
    // Simulate work
    tokens.update(5000, 100); // 50:1 ratio
    try std.testing.expect(tokens.shouldPromote());
    
    tokens.reset();
    tokens.update(500, 100); // 5:1 ratio
    try std.testing.expect(!tokens.shouldPromote());
}

test "task predictor" {
    const allocator = std.testing.allocator;
    
    var predictor = TaskPredictor.init(allocator);
    defer predictor.deinit();
    
    const task_hash: u64 = 0x12345678;
    
    // Record some executions
    try predictor.recordExecution(task_hash, 1000);
    try predictor.recordExecution(task_hash, 1100);
    try predictor.recordExecution(task_hash, 900);
    
    // Check prediction
    const prediction = predictor.predict(task_hash).?;
    try std.testing.expect(prediction.expected_cycles == 1000);
    try std.testing.expect(prediction.confidence > 0);
}