// ISPC-Optimized Data Structures and Algorithms for Beat.zig
// Structure of Arrays (SoA) layout for maximum SIMD efficiency
// Integration with One Euro Filter and advanced prediction systems

const std = @import("std");

/// Structure of Arrays fingerprint storage for ISPC optimization
/// Eliminates gather operations for 4-8x memory access improvement
pub const FingerprintSoA = struct {
    low_bits: []u64,
    high_bits: []u64,
    count: usize,
    allocator: std.mem.Allocator,
    
    const Self = @This();
    
    pub fn init(allocator: std.mem.Allocator, capacity: usize) !Self {
        return Self{
            .low_bits = try allocator.alloc(u64, capacity),
            .high_bits = try allocator.alloc(u64, capacity),
            .count = 0,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.low_bits);
        self.allocator.free(self.high_bits);
    }
    
    pub fn add(self: *Self, fingerprint: u128) void {
        std.debug.assert(self.count < self.low_bits.len);
        self.low_bits[self.count] = @as(u64, @truncate(fingerprint));
        self.high_bits[self.count] = @as(u64, @truncate(fingerprint >> 64));
        self.count += 1;
    }
    
    pub fn get(self: Self, index: usize) u128 {
        std.debug.assert(index < self.count);
        const low: u128 = self.low_bits[index];
        const high: u128 = self.high_bits[index];
        return (high << 64) | low;
    }
    
    pub fn clear(self: *Self) void {
        self.count = 0;
    }
    
    /// Convert from Array of Structures to Structure of Arrays
    pub fn fromAoS(allocator: std.mem.Allocator, fingerprints: []const u128) !Self {
        var soa = try Self.init(allocator, fingerprints.len);
        for (fingerprints) |fp| {
            soa.add(fp);
        }
        return soa;
    }
    
    /// Convert back to Array of Structures if needed
    pub fn toAoS(self: Self, allocator: std.mem.Allocator) ![]u128 {
        var result = try allocator.alloc(u128, self.count);
        for (0..self.count) |i| {
            result[i] = self.get(i);
        }
        return result;
    }
};

/// One Euro Filter state for ISPC batch processing
pub const OneEuroFilterState = struct {
    x_prev: f32 = 0.0,
    dx_prev: f32 = 0.0,
    initialized: bool = false,
    min_cutoff: f32 = 1.0,
    beta: f32 = 0.1,
    derivate_cutoff: f32 = 1.0,
    
    pub fn init(min_cutoff: f32, beta: f32) OneEuroFilterState {
        return OneEuroFilterState{
            .min_cutoff = min_cutoff,
            .beta = beta,
            .derivate_cutoff = 1.0,
        };
    }
};

/// ISPC-optimized prediction system
pub const ISPCPredictionSystem = struct {
    states: []OneEuroFilterState,
    raw_values: []f32,
    filtered_values: []f32,
    timestamps: []f32,
    confidence_scores: []f32,
    allocator: std.mem.Allocator,
    capacity: usize,
    
    const Self = @This();
    
    pub fn init(allocator: std.mem.Allocator, capacity: usize) !Self {
        return Self{
            .states = try allocator.alloc(OneEuroFilterState, capacity),
            .raw_values = try allocator.alloc(f32, capacity),
            .filtered_values = try allocator.alloc(f32, capacity),
            .timestamps = try allocator.alloc(f32, capacity),
            .confidence_scores = try allocator.alloc(f32, capacity),
            .allocator = allocator,
            .capacity = capacity,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.states);
        self.allocator.free(self.raw_values);
        self.allocator.free(self.filtered_values);
        self.allocator.free(self.timestamps);
        self.allocator.free(self.confidence_scores);
    }
    
    pub fn initializeStates(self: *Self, min_cutoff: f32, beta: f32) void {
        for (self.states) |*state| {
            state.* = OneEuroFilterState.init(min_cutoff, beta);
        }
    }
};

/// External ISPC function declarations for SoA operations
extern fn ispc_compute_fingerprint_similarity_soa(
    fingerprints_a_low: [*]u64,
    fingerprints_a_high: [*]u64,
    fingerprints_b_low: [*]u64,
    fingerprints_b_high: [*]u64,
    results: [*]f32,
    count: i32,
) void;

extern fn ispc_compute_similarity_matrix_soa(
    fingerprints_low: [*]u64,
    fingerprints_high: [*]u64,
    similarity_matrix: [*]f32,
    count: i32,
) void;

extern fn ispc_batch_similarity_scoring_soa(
    fingerprints_low: [*]u64,
    fingerprints_high: [*]u64,
    task_priorities: [*]f32,
    compatibility_scores: [*]f32,
    count: i32,
) void;

extern fn ispc_compute_fingerprint_hashes_soa(
    fingerprints_low: [*]u64,
    fingerprints_high: [*]u64,
    hashes: [*]u32,
    count: i32,
) void;

extern fn ispc_one_euro_filter_batch(
    raw_values: [*]f32,
    timestamps: [*]f32,
    states: [*]OneEuroFilterState,
    filtered_values: [*]f32,
    count: i32,
) void;

extern fn ispc_compute_prediction_confidence(
    predicted_values: [*]f32,
    actual_values: [*]f32,
    timestamps: [*]f32,
    confidence_scores: [*]f32,
    count: i32,
) void;

extern fn ispc_compute_prediction_scores(
    execution_times: [*]f32,
    confidence_levels: [*]f32,
    worker_loads: [*]f32,
    numa_distances: [*]f32,
    prediction_scores: [*]f32,
    worker_count: i32,
) void;

/// High-level interface for optimized fingerprint operations
pub const OptimizedFingerprints = struct {
    /// Compute similarity between two sets of fingerprints using SoA layout
    pub fn computeSimilarity(
        allocator: std.mem.Allocator,
        fingerprints_a: FingerprintSoA,
        fingerprints_b: FingerprintSoA,
    ) ![]f32 {
        std.debug.assert(fingerprints_a.count == fingerprints_b.count);
        
        const results = try allocator.alloc(f32, fingerprints_a.count);
        
        ispc_compute_fingerprint_similarity_soa(
            fingerprints_a.low_bits.ptr,
            fingerprints_a.high_bits.ptr,
            fingerprints_b.low_bits.ptr,
            fingerprints_b.high_bits.ptr,
            results.ptr,
            @intCast(fingerprints_a.count),
        );
        
        return results;
    }
    
    /// Compute full similarity matrix using SoA layout
    pub fn computeSimilarityMatrix(
        allocator: std.mem.Allocator,
        fingerprints: FingerprintSoA,
    ) ![]f32 {
        const matrix_size = fingerprints.count * fingerprints.count;
        const matrix = try allocator.alloc(f32, matrix_size);
        
        ispc_compute_similarity_matrix_soa(
            fingerprints.low_bits.ptr,
            fingerprints.high_bits.ptr,
            matrix.ptr,
            @intCast(fingerprints.count),
        );
        
        return matrix;
    }
    
    /// Compute batch compatibility scores
    pub fn computeBatchCompatibility(
        allocator: std.mem.Allocator,
        fingerprints: FingerprintSoA,
        priorities: []const f32,
    ) ![]f32 {
        std.debug.assert(fingerprints.count == priorities.len);
        
        const scores = try allocator.alloc(f32, fingerprints.count);
        
        ispc_batch_similarity_scoring_soa(
            fingerprints.low_bits.ptr,
            fingerprints.high_bits.ptr,
            @as([*]f32, @ptrCast(@constCast(priorities.ptr))),
            scores.ptr,
            @intCast(fingerprints.count),
        );
        
        return scores;
    }
    
    /// Fast hash computation for fingerprints
    pub fn computeHashes(
        allocator: std.mem.Allocator,
        fingerprints: FingerprintSoA,
    ) ![]u32 {
        const hashes = try allocator.alloc(u32, fingerprints.count);
        
        ispc_compute_fingerprint_hashes_soa(
            fingerprints.low_bits.ptr,
            fingerprints.high_bits.ptr,
            hashes.ptr,
            @intCast(fingerprints.count),
        );
        
        return hashes;
    }
};

/// High-level interface for optimized prediction operations
pub const OptimizedPredictions = struct {
    /// Process batch of predictions using One Euro Filter
    pub fn filterBatch(
        system: *ISPCPredictionSystem,
        raw_values: []const f32,
        timestamps: []const f32,
    ) void {
        std.debug.assert(raw_values.len == timestamps.len);
        std.debug.assert(raw_values.len <= system.capacity);
        
        @memcpy(system.raw_values[0..raw_values.len], raw_values);
        @memcpy(system.timestamps[0..timestamps.len], timestamps);
        
        ispc_one_euro_filter_batch(
            system.raw_values.ptr,
            system.timestamps.ptr,
            system.states.ptr,
            system.filtered_values.ptr,
            @intCast(raw_values.len),
        );
    }
    
    /// Compute prediction confidence scores
    pub fn computeConfidence(
        system: *ISPCPredictionSystem,
        predicted_values: []const f32,
        actual_values: []const f32,
        timestamps: []const f32,
    ) void {
        std.debug.assert(predicted_values.len == actual_values.len);
        std.debug.assert(predicted_values.len == timestamps.len);
        std.debug.assert(predicted_values.len <= system.capacity);
        
        ispc_compute_prediction_confidence(
            @as([*]f32, @ptrCast(@constCast(predicted_values.ptr))),
            @as([*]f32, @ptrCast(@constCast(actual_values.ptr))),
            @as([*]f32, @ptrCast(@constCast(timestamps.ptr))),
            system.confidence_scores.ptr,
            @intCast(predicted_values.len),
        );
    }
    
    /// Compute multi-factor prediction scores for worker selection
    pub fn computePredictionScores(
        allocator: std.mem.Allocator,
        execution_times: []const f32,
        confidence_levels: []const f32,
        worker_loads: []const f32,
        numa_distances: []const f32,
    ) ![]f32 {
        std.debug.assert(execution_times.len == confidence_levels.len);
        std.debug.assert(execution_times.len == worker_loads.len);
        std.debug.assert(execution_times.len == numa_distances.len);
        
        const scores = try allocator.alloc(f32, execution_times.len);
        
        ispc_compute_prediction_scores(
            @as([*]f32, @ptrCast(@constCast(execution_times.ptr))),
            @as([*]f32, @ptrCast(@constCast(confidence_levels.ptr))),
            @as([*]f32, @ptrCast(@constCast(worker_loads.ptr))),
            @as([*]f32, @ptrCast(@constCast(numa_distances.ptr))),
            scores.ptr,
            @intCast(execution_times.len),
        );
        
        return scores;
    }
};

/// Performance measurement utilities
pub const PerformanceMeasurement = struct {
    pub const BenchmarkResult = struct {
        native_time: u64,
        ispc_time: u64,
        speedup: f64,
        correctness_check: bool,
    };
    
    /// Benchmark SoA vs AoS fingerprint similarity computation
    pub fn benchmarkFingerprintSimilarity(
        allocator: std.mem.Allocator,
        count: usize,
        iterations: u32,
    ) !BenchmarkResult {
        // Generate test data
        var fingerprints_a = try allocator.alloc(u128, count);
        defer allocator.free(fingerprints_a);
        var fingerprints_b = try allocator.alloc(u128, count);
        defer allocator.free(fingerprints_b);
        
        for (0..count) |i| {
            fingerprints_a[i] = (@as(u128, @intCast(i)) * 0x123456789ABCDEF) ^ 0xFEDCBA9876543210;
            fingerprints_b[i] = (@as(u128, @intCast(i)) * 0xDEADBEEFCAFEBABE) ^ 0x0123456789ABCDEF;
        }
        
        // Convert to SoA
        var soa_a = try FingerprintSoA.fromAoS(allocator, fingerprints_a);
        defer soa_a.deinit();
        var soa_b = try FingerprintSoA.fromAoS(allocator, fingerprints_b);
        defer soa_b.deinit();
        
        var timer = try std.time.Timer.start();
        
        // Native benchmark
        timer.reset();
        var native_results = try allocator.alloc(f32, count);
        defer allocator.free(native_results);
        
        for (0..iterations) |_| {
            for (0..count) |i| {
                const fp_a = fingerprints_a[i];
                const fp_b = fingerprints_b[i];
                const diff = fp_a ^ fp_b;
                const hamming = @popCount(diff);
                native_results[i] = 1.0 - (@as(f32, @floatFromInt(hamming)) / 128.0);
            }
            std.mem.doNotOptimizeAway(&native_results);
        }
        const native_time = timer.read();
        
        // ISPC SoA benchmark
        timer.reset();
        var ispc_results = try allocator.alloc(f32, count);
        defer allocator.free(ispc_results);
        
        for (0..iterations) |_| {
            ispc_compute_fingerprint_similarity_soa(
                soa_a.low_bits.ptr,
                soa_a.high_bits.ptr,
                soa_b.low_bits.ptr,
                soa_b.high_bits.ptr,
                ispc_results.ptr,
                @intCast(count),
            );
            std.mem.doNotOptimizeAway(&ispc_results);
        }
        const ispc_time = timer.read();
        
        // Verify correctness
        var max_diff: f32 = 0.0;
        for (0..count) |i| {
            const diff = @abs(native_results[i] - ispc_results[i]);
            max_diff = @max(max_diff, diff);
        }
        
        return BenchmarkResult{
            .native_time = native_time,
            .ispc_time = ispc_time,
            .speedup = @as(f64, @floatFromInt(native_time)) / @as(f64, @floatFromInt(ispc_time)),
            .correctness_check = max_diff < 0.001,
        };
    }
};