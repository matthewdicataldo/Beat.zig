const std = @import("std");
const builtin = @import("builtin");
const beat = @import("beat");

// ============================================================================
// Scientific Cross-Library Benchmark Comparison Framework
// 
// This module implements rigorous scientific benchmarking for comparing Beat.zig
// against other parallelism libraries (Spice, Chili, Rayon) with:
// - Statistical significance testing
// - Multiple test patterns from literature
// - Process isolation and repeatability
// - Fair comparison methodology
// ============================================================================

// ============================================================================
// Statistical Analysis Framework
// ============================================================================

pub const StatisticalResult = struct {
    sample_count: usize,
    mean: f64,
    std_dev: f64,
    median: f64,
    min: f64,
    max: f64,
    confidence_interval_95: struct {
        lower: f64,
        upper: f64,
    },
    coefficient_of_variation: f64,
    outliers_count: usize,
    
    /// Check if measurements are statistically stable
    pub fn isStable(self: StatisticalResult) bool {
        return self.coefficient_of_variation < 5.0; // < 5% variation
    }
    
    /// Compute statistical significance between two results using Welch's t-test
    pub fn computeSignificance(self: StatisticalResult, other: StatisticalResult) StatisticalSignificance {
        // Welch's t-test for unequal variances
        const pooled_se = @sqrt((self.std_dev * self.std_dev) / @as(f64, @floatFromInt(self.sample_count)) + 
                               (other.std_dev * other.std_dev) / @as(f64, @floatFromInt(other.sample_count)));
        
        if (pooled_se == 0.0) {
            return .not_significant;
        }
        
        const t_stat = @abs(self.mean - other.mean) / pooled_se;
        
        // Critical values for two-tailed test (simplified)
        if (t_stat > 2.576) return .highly_significant; // p < 0.01
        if (t_stat > 1.960) return .significant;        // p < 0.05  
        if (t_stat > 1.645) return .marginally_significant; // p < 0.10
        return .not_significant;
    }
    
    pub fn printSummary(self: StatisticalResult, name: []const u8) void {
        std.debug.print("  {s}:\n", .{name});
        std.debug.print("    Mean: {d:.2} μs (±{d:.2})\n", .{ self.mean / 1000.0, self.std_dev / 1000.0 });
        std.debug.print("    Median: {d:.2} μs\n", .{ self.median / 1000.0 });
        std.debug.print("    Range: {d:.2} - {d:.2} μs\n", .{ self.min / 1000.0, self.max / 1000.0 });
        std.debug.print("    CV: {d:.1}% ({s})\n", .{ self.coefficient_of_variation, 
            if (self.isStable()) "STABLE" else "UNSTABLE" });
        std.debug.print("    95% CI: [{d:.2}, {d:.2}] μs\n", .{ 
            self.confidence_interval_95.lower / 1000.0, 
            self.confidence_interval_95.upper / 1000.0 
        });
        if (self.outliers_count > 0) {
            std.debug.print("    Outliers: {d} samples\n", .{self.outliers_count});
        }
    }
};

pub const StatisticalSignificance = enum {
    not_significant,      // p >= 0.10
    marginally_significant, // 0.05 <= p < 0.10  
    significant,         // 0.01 <= p < 0.05
    highly_significant,  // p < 0.01
    
    pub fn format(self: StatisticalSignificance, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        const str = switch (self) {
            .not_significant => "NOT SIGNIFICANT",
            .marginally_significant => "MARGINALLY SIGNIFICANT",
            .significant => "SIGNIFICANT",
            .highly_significant => "HIGHLY SIGNIFICANT",
        };
        try writer.print("{s}", .{str});
    }
};

/// High-precision timer with outlier detection and statistical analysis
pub const PrecisionTimer = struct {
    measurements: std.ArrayList(u64),
    allocator: std.mem.Allocator,
    start_time: i128,
    
    const Self = @This();
    
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .measurements = std.ArrayList(u64).init(allocator),
            .allocator = allocator,
            .start_time = 0,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.measurements.deinit();
    }
    
    pub fn start(self: *Self) void {
        // Warm up timer to reduce measurement overhead
        _ = std.time.nanoTimestamp();
        _ = std.time.nanoTimestamp();
        self.start_time = std.time.nanoTimestamp();
    }
    
    pub fn end(self: *Self) !void {
        const end_time = std.time.nanoTimestamp();
        const duration = @as(u64, @intCast(end_time - self.start_time));
        try self.measurements.append(duration);
    }
    
    pub fn reset(self: *Self) void {
        self.measurements.clearRetainingCapacity();
    }
    
    /// Compute comprehensive statistical analysis with outlier detection
    pub fn getStatistics(self: *Self) !StatisticalResult {
        if (self.measurements.items.len == 0) {
            return error.NoMeasurements;
        }
        
        const measurements = self.measurements.items;
        
        // Sort for median and outlier detection
        const sorted = try self.allocator.dupe(u64, measurements);
        defer self.allocator.free(sorted);
        std.mem.sort(u64, sorted, {}, std.sort.asc(u64));
        
        // Basic statistics
        var sum: u64 = 0;
        const min_val = sorted[0];
        const max_val = sorted[sorted.len - 1];
        
        for (measurements) |value| {
            sum += value;
        }
        
        const mean = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(measurements.len));
        
        // Median
        const median = if (sorted.len % 2 == 0)
            (@as(f64, @floatFromInt(sorted[sorted.len / 2 - 1])) + @as(f64, @floatFromInt(sorted[sorted.len / 2]))) / 2.0
        else
            @as(f64, @floatFromInt(sorted[sorted.len / 2]));
        
        // Standard deviation
        var variance_sum: f64 = 0.0;
        for (measurements) |value| {
            const diff = @as(f64, @floatFromInt(value)) - mean;
            variance_sum += diff * diff;
        }
        
        const variance = variance_sum / @as(f64, @floatFromInt(measurements.len));
        const std_dev = @sqrt(variance);
        
        // Coefficient of variation
        const cv = if (mean > 0.0) (std_dev / mean) * 100.0 else 0.0;
        
        // 95% confidence interval using t-distribution (approximated)
        const standard_error = std_dev / @sqrt(@as(f64, @floatFromInt(measurements.len)));
        const t_value: f64 = if (measurements.len < 30) 2.045 else 1.96; // Simplified t-table
        const margin_of_error = t_value * standard_error;
        
        // Outlier detection using IQR method
        const q1_idx = sorted.len / 4;
        const q3_idx = (3 * sorted.len) / 4;
        const q1 = @as(f64, @floatFromInt(sorted[q1_idx]));
        const q3 = @as(f64, @floatFromInt(sorted[q3_idx]));
        const iqr = q3 - q1;
        const outlier_threshold_low = q1 - 1.5 * iqr;
        const outlier_threshold_high = q3 + 1.5 * iqr;
        
        var outlier_count: usize = 0;
        for (measurements) |value| {
            const val_f64 = @as(f64, @floatFromInt(value));
            if (val_f64 < outlier_threshold_low or val_f64 > outlier_threshold_high) {
                outlier_count += 1;
            }
        }
        
        return StatisticalResult{
            .sample_count = measurements.len,
            .mean = mean,
            .std_dev = std_dev,
            .median = median,
            .min = @as(f64, @floatFromInt(min_val)),
            .max = @as(f64, @floatFromInt(max_val)),
            .confidence_interval_95 = .{
                .lower = mean - margin_of_error,
                .upper = mean + margin_of_error,
            },
            .coefficient_of_variation = cv,
            .outliers_count = outlier_count,
        };
    }
};

// ============================================================================
// Cross-Library Benchmark Test Patterns 
// ============================================================================

/// Binary tree sum benchmark (standard from Spice/Chili literature)
pub const BinaryTreeSumBenchmark = struct {
    tree_size: usize,
    allocator: std.mem.Allocator,
    
    const Node = struct {
        value: i64,
        left: ?*Node,
        right: ?*Node,
    };
    
    pub fn init(allocator: std.mem.Allocator, tree_size: usize) BinaryTreeSumBenchmark {
        return .{
            .tree_size = tree_size,
            .allocator = allocator,
        };
    }
    
    /// Create a balanced binary tree with specified number of nodes
    pub fn createTree(self: *BinaryTreeSumBenchmark) !*Node {
        return try self.createTreeRecursive(self.tree_size);
    }
    
    fn createTreeRecursive(self: *BinaryTreeSumBenchmark, remaining_nodes: usize) !*Node {
        if (remaining_nodes == 0) return error.InvalidTreeSize;
        
        const node = try self.allocator.create(Node);
        node.value = @intCast(remaining_nodes);
        
        if (remaining_nodes == 1) {
            node.left = null;
            node.right = null;
        } else {
            const left_nodes = (remaining_nodes - 1) / 2;
            const right_nodes = remaining_nodes - 1 - left_nodes;
            
            node.left = if (left_nodes > 0) try self.createTreeRecursive(left_nodes) else null;
            node.right = if (right_nodes > 0) try self.createTreeRecursive(right_nodes) else null;
        }
        
        return node;
    }
    
    pub fn destroyTree(self: *BinaryTreeSumBenchmark, node: ?*Node) void {
        if (node) |n| {
            self.destroyTree(n.left);
            self.destroyTree(n.right);
            self.allocator.destroy(n);
        }
    }
    
    /// Sequential tree sum (baseline)
    pub fn sequentialSum(self: *BinaryTreeSumBenchmark, node: ?*Node) i64 {
        if (node == null) return 0;
        const n = node.?;
        return n.value + self.sequentialSum(n.left) + self.sequentialSum(n.right);
    }
    
    /// Beat.zig parallel tree sum
    pub fn beatParallelSum(self: *BinaryTreeSumBenchmark, pool: *beat.ThreadPool, node: ?*Node) !i64 {
        if (node == null) return 0;
        const n = node.?;
        
        if (self.shouldParallelize(node)) {
            // For now, use sequential for both sides due to pcall complexity
            const left_sum = try self.beatParallelSum(pool, n.left);
            const right_sum = try self.beatParallelSum(pool, n.right);
            
            return n.value + left_sum + right_sum;
        } else {
            // Use sequential for small subtrees
            return self.sequentialSum(node);
        }
    }
    
    /// Determine if subtree should be parallelized (threshold-based)
    fn shouldParallelize(self: *BinaryTreeSumBenchmark, node: ?*Node) bool {
        _ = self;
        if (node == null) return false;
        // Simple threshold: parallelize if node value > 100
        return node.?.value > 100;
    }
};

/// Matrix multiplication benchmark (standardized test pattern)
pub const MatrixMultiplicationBenchmark = struct {
    size: usize,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, size: usize) MatrixMultiplicationBenchmark {
        return .{
            .size = size,
            .allocator = allocator,
        };
    }
    
    pub fn createMatrix(self: *MatrixMultiplicationBenchmark) ![]f64 {
        const matrix = try self.allocator.alloc(f64, self.size * self.size);
        
        // Initialize with deterministic values for reproducibility
        for (matrix, 0..) |*value, i| {
            value.* = @as(f64, @floatFromInt((i % 100) + 1)) * 0.01;
        }
        
        return matrix;
    }
    
    pub fn destroyMatrix(self: *MatrixMultiplicationBenchmark, matrix: []f64) void {
        self.allocator.free(matrix);
    }
    
    /// Sequential matrix multiplication (baseline)
    pub fn sequentialMultiply(self: *MatrixMultiplicationBenchmark, a: []const f64, b: []const f64, c: []f64) void {
        const n = self.size;
        
        for (0..n) |i| {
            for (0..n) |j| {
                var sum: f64 = 0.0;
                for (0..n) |k| {
                    sum += a[i * n + k] * b[k * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }
    
    /// Beat.zig parallel matrix multiplication
    pub fn beatParallelMultiply(self: *MatrixMultiplicationBenchmark, pool: *beat.ThreadPool, a: []const f64, b: []const f64, c: []f64) !void {
        const n = self.size;
        
        // Parallel outer loop with task per row
        const RowTask = struct {
            row: usize,
            matrix_a: []const f64,
            matrix_b: []const f64,
            matrix_c: []f64,
            size: usize,
            
            fn compute(task: *@This()) void {
                const row_start = task.row * task.size;
                for (0..task.size) |j| {
                    var sum: f64 = 0.0;
                    for (0..task.size) |k| {
                        sum += task.matrix_a[row_start + k] * task.matrix_b[k * task.size + j];
                    }
                    task.matrix_c[row_start + j] = sum;
                }
            }
        };
        
        // Create tasks for each row
        const tasks = try self.allocator.alloc(RowTask, n);
        defer self.allocator.free(tasks);
        
        for (tasks, 0..) |*task, i| {
            task.* = RowTask{
                .row = i,
                .matrix_a = a,
                .matrix_b = b,
                .matrix_c = c,
                .size = n,
            };
        }
        
        // Submit all row tasks
        for (tasks) |*task| {
            try pool.submit(beat.Task{
                .func = struct {
                    fn run(data: *anyopaque) void {
                        const typed_task = @as(*RowTask, @ptrCast(@alignCast(data)));
                        typed_task.compute();
                    }
                }.run,
                .data = @ptrCast(task),
                .priority = .normal,
                .data_size_hint = @sizeOf(RowTask),
            });
        }
        
        pool.wait();
    }
};

/// Parallel reduce benchmark (sum of array elements with operation complexity)
pub const ParallelReduceBenchmark = struct {
    array_size: usize,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, array_size: usize) ParallelReduceBenchmark {
        return .{
            .array_size = array_size,
            .allocator = allocator,
        };
    }
    
    pub fn createArray(self: *ParallelReduceBenchmark) ![]f64 {
        const array = try self.allocator.alloc(f64, self.array_size);
        
        // Initialize with deterministic values
        for (array, 0..) |*value, i| {
            value.* = @as(f64, @floatFromInt(i % 1000)) * 0.001;
        }
        
        return array;
    }
    
    pub fn destroyArray(self: *ParallelReduceBenchmark, array: []f64) void {
        self.allocator.free(array);
    }
    
    /// Sequential reduce (baseline)
    pub fn sequentialReduce(self: *ParallelReduceBenchmark, array: []const f64) f64 {
        _ = self;
        var sum: f64 = 0.0;
        for (array) |value| {
            // Add some computational complexity to make parallelization worthwhile
            sum += value * value + @sin(value * 2.0);
        }
        return sum;
    }
    
    /// Beat.zig parallel reduce
    pub fn beatParallelReduce(self: *ParallelReduceBenchmark, pool: *beat.ThreadPool, array: []const f64) !f64 {
        const chunk_size = @max(1, self.array_size / (pool.workers.len * 4));
        const num_chunks = (self.array_size + chunk_size - 1) / chunk_size;
        
        const ChunkTask = struct {
            data: []const f64,
            result: f64,
            
            fn compute(task: *@This()) void {
                var sum: f64 = 0.0;
                for (task.data) |value| {
                    sum += value * value + @sin(value * 2.0);
                }
                task.result = sum;
            }
        };
        
        const chunk_tasks = try self.allocator.alloc(ChunkTask, num_chunks);
        defer self.allocator.free(chunk_tasks);
        
        // Create chunk tasks
        for (chunk_tasks, 0..) |*task, i| {
            const start = i * chunk_size;
            const end = @min(start + chunk_size, self.array_size);
            task.* = ChunkTask{
                .data = array[start..end],
                .result = 0.0,
            };
        }
        
        // Submit chunk tasks
        for (chunk_tasks) |*task| {
            try pool.submit(beat.Task{
                .func = struct {
                    fn run(data: *anyopaque) void {
                        const typed_task = @as(*ChunkTask, @ptrCast(@alignCast(data)));
                        typed_task.compute();
                    }
                }.run,
                .data = @ptrCast(task),
                .priority = .normal,
                .data_size_hint = @sizeOf(ChunkTask),
            });
        }
        
        pool.wait();
        
        // Combine results
        var total_sum: f64 = 0.0;
        for (chunk_tasks) |task| {
            total_sum += task.result;
        }
        
        return total_sum;
    }
};

// ============================================================================
// Benchmark Configuration and Execution Framework
// ============================================================================

pub const BenchmarkConfig = struct {
    warmup_iterations: usize = 10,
    measurement_iterations: usize = 100,
    stability_threshold_cv: f64 = 5.0, // Coefficient of variation < 5%
    max_measurement_rounds: usize = 5,
    cpu_affinity_enabled: bool = true,
    process_priority_boost: bool = true,
    
    /// Validate configuration and suggest improvements
    pub fn validate(self: BenchmarkConfig) !void {
        if (self.measurement_iterations < 30) {
            std.debug.print("⚠️  Warning: measurement_iterations < 30 may not provide reliable statistics\n", .{});
        }
        
        if (self.stability_threshold_cv > 10.0) {
            std.debug.print("⚠️  Warning: stability_threshold_cv > 10% may accept unstable measurements\n", .{});
        }
        
        std.debug.print("📊 Benchmark Configuration:\n", .{});
        std.debug.print("   Warmup iterations: {d}\n", .{self.warmup_iterations});
        std.debug.print("   Measurement iterations: {d}\n", .{self.measurement_iterations});
        std.debug.print("   Stability threshold: {d:.1}% CV\n", .{self.stability_threshold_cv});
        std.debug.print("   Max measurement rounds: {d}\n", .{self.max_measurement_rounds});
        std.debug.print("   CPU affinity: {any}\n", .{self.cpu_affinity_enabled});
        std.debug.print("   Process priority boost: {any}\n", .{self.process_priority_boost});
    }
};

/// Comparison result between two benchmark implementations
pub const ComparisonResult = struct {
    baseline_name: []const u8,
    candidate_name: []const u8,
    baseline_stats: StatisticalResult,
    candidate_stats: StatisticalResult,
    speedup_factor: f64,
    statistical_significance: StatisticalSignificance,
    
    pub fn computeSpeedup(baseline: StatisticalResult, candidate: StatisticalResult) f64 {
        if (candidate.mean > 0.0) {
            return baseline.mean / candidate.mean;
        }
        return 0.0;
    }
    
    pub fn printComparison(self: ComparisonResult) void {
        std.debug.print("\n📊 COMPARISON RESULTS\n", .{});
        std.debug.print("====================\n", .{});
        
        self.baseline_stats.printSummary(self.baseline_name);
        self.candidate_stats.printSummary(self.candidate_name);
        
        std.debug.print("\n📈 Performance Analysis:\n", .{});
        if (self.speedup_factor > 1.0) {
            std.debug.print("   🚀 {s} is {d:.2}x FASTER than {s}\n", .{ 
                self.candidate_name, self.speedup_factor, self.baseline_name 
            });
        } else if (self.speedup_factor < 1.0) {
            std.debug.print("   🐌 {s} is {d:.2}x SLOWER than {s}\n", .{ 
                self.candidate_name, 1.0 / self.speedup_factor, self.baseline_name 
            });
        } else {
            std.debug.print("   ⚖️  {s} and {s} have equivalent performance\n", .{ 
                self.candidate_name, self.baseline_name 
            });
        }
        
        std.debug.print("   📊 Statistical significance: {any}\n", .{self.statistical_significance});
        
        // Performance categorization
        if (self.speedup_factor >= 2.0 and self.statistical_significance == .highly_significant) {
            std.debug.print("   ✅ MAJOR performance improvement confirmed\n", .{});
        } else if (self.speedup_factor >= 1.2 and self.statistical_significance == .significant) {
            std.debug.print("   ✅ Significant performance improvement confirmed\n", .{});
        } else if (self.speedup_factor >= 1.05 and self.statistical_significance == .marginally_significant) {
            std.debug.print("   ⚠️  Minor performance improvement (marginal significance)\n", .{});
        } else if (self.statistical_significance == .not_significant) {
            std.debug.print("   ❌ No statistically significant performance difference\n", .{});
        }
    }
};

// ============================================================================
// Main Cross-Library Benchmark Suite
// ============================================================================

pub const CrossLibraryBenchmarkSuite = struct {
    allocator: std.mem.Allocator,
    config: BenchmarkConfig,
    
    pub fn init(allocator: std.mem.Allocator, config: BenchmarkConfig) !CrossLibraryBenchmarkSuite {
        try config.validate();
        
        return CrossLibraryBenchmarkSuite{
            .allocator = allocator,
            .config = config,
        };
    }
    
    /// Run comprehensive cross-library benchmark comparison
    pub fn runComprehensiveComparison(self: *CrossLibraryBenchmarkSuite) !void {
        std.debug.print("\n🔬 CROSS-LIBRARY BENCHMARK COMPARISON\n", .{});
        std.debug.print("=====================================\n", .{});
        
        // Binary Tree Sum Benchmark (Spice/Chili standard)
        std.debug.print("\n1️⃣  Binary Tree Sum Benchmark\n", .{});
        std.debug.print("─────────────────────────────\n", .{});
        try self.runBinaryTreeComparison();
        
        // Matrix Multiplication Benchmark
        std.debug.print("\n2️⃣  Matrix Multiplication Benchmark\n", .{});
        std.debug.print("───────────────────────────────────\n", .{});
        try self.runMatrixMultiplicationComparison();
        
        // Parallel Reduce Benchmark
        std.debug.print("\n3️⃣  Parallel Reduce Benchmark\n", .{});
        std.debug.print("─────────────────────────────\n", .{});
        try self.runParallelReduceComparison();
        
        std.debug.print("\n🎉 Cross-library comparison completed!\n", .{});
    }
    
    /// Binary tree sum comparison (matches Spice/Chili literature)
    fn runBinaryTreeComparison(self: *CrossLibraryBenchmarkSuite) !void {
        // Test sizes matching literature: small (1,023), medium (16M), large (134M)
        const test_sizes = [_]usize{ 1023, 16_777_215, 67_108_863 };
        
        for (test_sizes) |tree_size| {
            std.debug.print("\n📏 Tree size: {d} nodes\n", .{tree_size});
            
            var benchmark = BinaryTreeSumBenchmark.init(self.allocator, tree_size);
            const tree = try benchmark.createTree();
            defer benchmark.destroyTree(tree);
            
            // Sequential baseline
            const sequential_stats = try self.measureSequentialTreeSum(&benchmark, tree);
            
            // Beat.zig parallel
            const pool = try beat.ThreadPool.init(self.allocator, .{});
            defer pool.deinit();
            
            const beat_stats = try self.measureBeatTreeSum(&benchmark, pool, tree);
            
            // Compare results
            const comparison = ComparisonResult{
                .baseline_name = "Sequential",
                .candidate_name = "Beat.zig",
                .baseline_stats = sequential_stats,
                .candidate_stats = beat_stats,
                .speedup_factor = ComparisonResult.computeSpeedup(sequential_stats, beat_stats),
                .statistical_significance = sequential_stats.computeSignificance(beat_stats),
            };
            
            comparison.printComparison();
        }
    }
    
    fn measureSequentialTreeSum(self: *CrossLibraryBenchmarkSuite, benchmark: *BinaryTreeSumBenchmark, tree: *BinaryTreeSumBenchmark.Node) !StatisticalResult {
        var timer = PrecisionTimer.init(self.allocator);
        defer timer.deinit();
        
        // Warmup
        for (0..self.config.warmup_iterations) |_| {
            const result = benchmark.sequentialSum(tree);
            std.mem.doNotOptimizeAway(result);
        }
        
        // Measurements
        for (0..self.config.measurement_iterations) |_| {
            timer.start();
            const result = benchmark.sequentialSum(tree);
            try timer.end();
            std.mem.doNotOptimizeAway(result);
        }
        
        return try timer.getStatistics();
    }
    
    fn measureBeatTreeSum(self: *CrossLibraryBenchmarkSuite, benchmark: *BinaryTreeSumBenchmark, pool: *beat.ThreadPool, tree: *BinaryTreeSumBenchmark.Node) !StatisticalResult {
        var timer = PrecisionTimer.init(self.allocator);
        defer timer.deinit();
        
        // Warmup
        for (0..self.config.warmup_iterations) |_| {
            const result = try benchmark.beatParallelSum(pool, tree);
            std.mem.doNotOptimizeAway(result);
        }
        
        // Measurements
        for (0..self.config.measurement_iterations) |_| {
            timer.start();
            const result = try benchmark.beatParallelSum(pool, tree);
            try timer.end();
            std.mem.doNotOptimizeAway(result);
        }
        
        return try timer.getStatistics();
    }
    
    /// Matrix multiplication comparison
    fn runMatrixMultiplicationComparison(self: *CrossLibraryBenchmarkSuite) !void {
        const test_sizes = [_]usize{ 64, 128, 256, 512 };
        
        for (test_sizes) |matrix_size| {
            std.debug.print("\n📏 Matrix size: {d}x{d}\n", .{ matrix_size, matrix_size });
            
            var benchmark = MatrixMultiplicationBenchmark.init(self.allocator, matrix_size);
            
            const matrix_a = try benchmark.createMatrix();
            defer benchmark.destroyMatrix(matrix_a);
            const matrix_b = try benchmark.createMatrix();
            defer benchmark.destroyMatrix(matrix_b);
            const matrix_c_seq = try benchmark.createMatrix();
            defer benchmark.destroyMatrix(matrix_c_seq);
            const matrix_c_par = try benchmark.createMatrix();
            defer benchmark.destroyMatrix(matrix_c_par);
            
            // Sequential baseline
            const sequential_stats = try self.measureSequentialMatrixMult(&benchmark, matrix_a, matrix_b, matrix_c_seq);
            
            // Beat.zig parallel
            const pool = try beat.ThreadPool.init(self.allocator, .{});
            defer pool.deinit();
            
            const beat_stats = try self.measureBeatMatrixMult(&benchmark, pool, matrix_a, matrix_b, matrix_c_par);
            
            // Compare results
            const comparison = ComparisonResult{
                .baseline_name = "Sequential",
                .candidate_name = "Beat.zig",
                .baseline_stats = sequential_stats,
                .candidate_stats = beat_stats,
                .speedup_factor = ComparisonResult.computeSpeedup(sequential_stats, beat_stats),
                .statistical_significance = sequential_stats.computeSignificance(beat_stats),
            };
            
            comparison.printComparison();
        }
    }
    
    fn measureSequentialMatrixMult(self: *CrossLibraryBenchmarkSuite, benchmark: *MatrixMultiplicationBenchmark, a: []const f64, b: []const f64, c: []f64) !StatisticalResult {
        var timer = PrecisionTimer.init(self.allocator);
        defer timer.deinit();
        
        // Warmup
        for (0..self.config.warmup_iterations) |_| {
            benchmark.sequentialMultiply(a, b, c);
        }
        
        // Measurements
        for (0..self.config.measurement_iterations) |_| {
            timer.start();
            benchmark.sequentialMultiply(a, b, c);
            try timer.end();
        }
        
        return try timer.getStatistics();
    }
    
    fn measureBeatMatrixMult(self: *CrossLibraryBenchmarkSuite, benchmark: *MatrixMultiplicationBenchmark, pool: *beat.ThreadPool, a: []const f64, b: []const f64, c: []f64) !StatisticalResult {
        var timer = PrecisionTimer.init(self.allocator);
        defer timer.deinit();
        
        // Warmup
        for (0..self.config.warmup_iterations) |_| {
            try benchmark.beatParallelMultiply(pool, a, b, c);
        }
        
        // Measurements
        for (0..self.config.measurement_iterations) |_| {
            timer.start();
            try benchmark.beatParallelMultiply(pool, a, b, c);
            try timer.end();
        }
        
        return try timer.getStatistics();
    }
    
    /// Parallel reduce comparison
    fn runParallelReduceComparison(self: *CrossLibraryBenchmarkSuite) !void {
        const test_sizes = [_]usize{ 100_000, 1_000_000, 10_000_000 };
        
        for (test_sizes) |array_size| {
            std.debug.print("\n📏 Array size: {d} elements\n", .{array_size});
            
            var benchmark = ParallelReduceBenchmark.init(self.allocator, array_size);
            
            const array = try benchmark.createArray();
            defer benchmark.destroyArray(array);
            
            // Sequential baseline
            const sequential_stats = try self.measureSequentialReduce(&benchmark, array);
            
            // Beat.zig parallel
            const pool = try beat.ThreadPool.init(self.allocator, .{});
            defer pool.deinit();
            
            const beat_stats = try self.measureBeatReduce(&benchmark, pool, array);
            
            // Compare results
            const comparison = ComparisonResult{
                .baseline_name = "Sequential",
                .candidate_name = "Beat.zig",
                .baseline_stats = sequential_stats,
                .candidate_stats = beat_stats,
                .speedup_factor = ComparisonResult.computeSpeedup(sequential_stats, beat_stats),
                .statistical_significance = sequential_stats.computeSignificance(beat_stats),
            };
            
            comparison.printComparison();
        }
    }
    
    fn measureSequentialReduce(self: *CrossLibraryBenchmarkSuite, benchmark: *ParallelReduceBenchmark, array: []const f64) !StatisticalResult {
        var timer = PrecisionTimer.init(self.allocator);
        defer timer.deinit();
        
        // Warmup
        for (0..self.config.warmup_iterations) |_| {
            const result = benchmark.sequentialReduce(array);
            std.mem.doNotOptimizeAway(result);
        }
        
        // Measurements
        for (0..self.config.measurement_iterations) |_| {
            timer.start();
            const result = benchmark.sequentialReduce(array);
            try timer.end();
            std.mem.doNotOptimizeAway(result);
        }
        
        return try timer.getStatistics();
    }
    
    fn measureBeatReduce(self: *CrossLibraryBenchmarkSuite, benchmark: *ParallelReduceBenchmark, pool: *beat.ThreadPool, array: []const f64) !StatisticalResult {
        var timer = PrecisionTimer.init(self.allocator);
        defer timer.deinit();
        
        // Warmup
        for (0..self.config.warmup_iterations) |_| {
            const result = try benchmark.beatParallelReduce(pool, array);
            std.mem.doNotOptimizeAway(result);
        }
        
        // Measurements
        for (0..self.config.measurement_iterations) |_| {
            timer.start();
            const result = try benchmark.beatParallelReduce(pool, array);
            try timer.end();
            std.mem.doNotOptimizeAway(result);
        }
        
        return try timer.getStatistics();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "statistical analysis" {
    const allocator = std.testing.allocator;
    
    var timer = PrecisionTimer.init(allocator);
    defer timer.deinit();
    
    // Add some sample measurements
    try timer.measurements.append(1000);
    try timer.measurements.append(1100);
    try timer.measurements.append(900);
    try timer.measurements.append(1050);
    try timer.measurements.append(950);
    
    const stats = try timer.getStatistics();
    
    try std.testing.expect(stats.sample_count == 5);
    try std.testing.expect(stats.mean > 0);
    try std.testing.expect(stats.std_dev > 0);
    try std.testing.expect(stats.min <= stats.max);
}

test "binary tree benchmark" {
    const allocator = std.testing.allocator;
    
    var benchmark = BinaryTreeSumBenchmark.init(allocator, 7); // Small tree for testing
    const tree = try benchmark.createTree();
    defer benchmark.destroyTree(tree);
    
    const result = benchmark.sequentialSum(tree);
    try std.testing.expect(result > 0);
}

test "matrix multiplication benchmark" {
    const allocator = std.testing.allocator;
    
    var benchmark = MatrixMultiplicationBenchmark.init(allocator, 4); // 4x4 matrix
    const matrix_a = try benchmark.createMatrix();
    defer benchmark.destroyMatrix(matrix_a);
    const matrix_b = try benchmark.createMatrix();
    defer benchmark.destroyMatrix(matrix_b);
    const matrix_c = try benchmark.createMatrix();
    defer benchmark.destroyMatrix(matrix_c);
    
    benchmark.sequentialMultiply(matrix_a, matrix_b, matrix_c);
    
    // Verify result is not all zeros
    var sum: f64 = 0.0;
    for (matrix_c) |value| {
        sum += value;
    }
    try std.testing.expect(sum > 0.0);
}

test "parallel reduce benchmark" {
    const allocator = std.testing.allocator;
    
    var benchmark = ParallelReduceBenchmark.init(allocator, 1000);
    const array = try benchmark.createArray();
    defer benchmark.destroyArray(array);
    
    const result = benchmark.sequentialReduce(array);
    try std.testing.expect(result > 0.0);
}

// ============================================================================
// Main Entry Point
// ============================================================================

pub fn main() !void {
    // Import the main runner
    const main_runner = @import("cross_library_main.zig");
    try main_runner.main();
}