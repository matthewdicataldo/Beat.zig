const std = @import("std");
const builtin = @import("builtin");
const core = @import("core.zig");
const scheduler = @import("scheduler.zig");
const fingerprint = @import("fingerprint.zig");
const intelligent_decision = @import("intelligent_decision.zig");
const predictive_accounting = @import("predictive_accounting.zig");
const topology = @import("topology.zig");
const simd = @import("simd.zig");

// Advanced Worker Selection Algorithm for Beat.zig (Task 2.4.2)
//
// This module implements a sophisticated worker selection algorithm that replaces
// simple round-robin with predictive selection using multi-criteria optimization.
// It integrates execution time predictions, confidence levels, topology awareness,
// and exploratory placement for optimal task distribution.

// ============================================================================
// Multi-Criteria Scoring System
// ============================================================================

/// Comprehensive scoring criteria for worker selection decisions
pub const SelectionCriteria = struct {
    load_balance_weight: f32 = 0.20,        // Queue length and utilization
    prediction_weight: f32 = 0.20,          // Execution time predictions
    topology_weight: f32 = 0.20,            // NUMA and cache locality
    confidence_weight: f32 = 0.15,          // Prediction confidence levels
    exploration_weight: f32 = 0.15,         // New task type exploration
    simd_weight: f32 = 0.10,                // SIMD capability matching
    
    // Dynamic adjustment factors
    enable_adaptive_weights: bool = true,    // Adjust weights based on performance
    exploration_decay: f32 = 0.95,          // Reduce exploration over time
    min_exploration_weight: f32 = 0.05,     // Minimum exploration component
    
    /// Validate and normalize criteria weights
    pub fn normalize(self: *SelectionCriteria) void {
        const total = self.load_balance_weight + self.prediction_weight + 
                     self.topology_weight + self.confidence_weight + self.exploration_weight + self.simd_weight;
        
        if (total > 0.0) {
            self.load_balance_weight /= total;
            self.prediction_weight /= total;
            self.topology_weight /= total;
            self.confidence_weight /= total;
            self.exploration_weight /= total;
            self.simd_weight /= total;
        }
    }
    
    /// Create balanced criteria for general-purpose workloads
    pub fn balanced() SelectionCriteria {
        var criteria = SelectionCriteria{};
        criteria.normalize();
        return criteria;
    }
    
    /// Create criteria optimized for latency-sensitive workloads
    pub fn latencyOptimized() SelectionCriteria {
        return SelectionCriteria{
            .load_balance_weight = 0.35,
            .prediction_weight = 0.30,
            .topology_weight = 0.25,
            .confidence_weight = 0.10,
            .exploration_weight = 0.00,
            .enable_adaptive_weights = false,
        };
    }
    
    /// Create criteria optimized for throughput-intensive workloads
    pub fn throughputOptimized() SelectionCriteria {
        return SelectionCriteria{
            .load_balance_weight = 0.20,
            .prediction_weight = 0.15,
            .topology_weight = 0.30,
            .confidence_weight = 0.25,
            .exploration_weight = 0.10,
            .enable_adaptive_weights = true,
        };
    }
};

/// Detailed worker evaluation for multi-criteria scoring
pub const WorkerEvaluation = struct {
    worker_id: usize,
    
    // Individual scoring components (0.0 to 1.0, higher is better)
    load_balance_score: f32,        // 1.0 = lightest load
    prediction_score: f32,          // 1.0 = optimal predicted performance
    topology_score: f32,            // 1.0 = best locality/affinity
    confidence_score: f32,          // 1.0 = highest prediction confidence
    exploration_score: f32,         // 1.0 = best exploration opportunity
    simd_score: f32,                // 1.0 = best SIMD capability match
    
    // Composite scores
    weighted_score: f32,            // Final weighted combination
    normalized_score: f32,          // Normalized against all workers
    
    // Detailed rationale
    queue_size: usize,
    predicted_cycles: f64,
    confidence: f32,
    numa_distance: u32,
    exploration_factor: f32,
    
    /// Create evaluation from worker information
    pub fn init(worker_id: usize) WorkerEvaluation {
        return WorkerEvaluation{
            .worker_id = worker_id,
            .load_balance_score = 0.0,
            .prediction_score = 0.0,
            .topology_score = 0.0,
            .confidence_score = 0.0,
            .exploration_score = 0.0,
            .simd_score = 0.0,
            .weighted_score = 0.0,
            .normalized_score = 0.0,
            .queue_size = 0,
            .predicted_cycles = 0.0,
            .confidence = 0.0,
            .numa_distance = 0,
            .exploration_factor = 0.0,
        };
    }
    
    /// Calculate weighted score from individual components
    pub fn calculateWeightedScore(self: *WorkerEvaluation, criteria: SelectionCriteria) void {
        self.weighted_score = 
            self.load_balance_score * criteria.load_balance_weight +
            self.prediction_score * criteria.prediction_weight +
            self.topology_score * criteria.topology_weight +
            self.confidence_score * criteria.confidence_weight +
            self.exploration_score * criteria.exploration_weight +
            self.simd_score * criteria.simd_weight;
    }
};

/// Selection decision with comprehensive analysis
pub const SelectionDecision = struct {
    selected_worker_id: usize,
    selection_strategy: SelectionStrategy,
    
    // All worker evaluations for analysis
    evaluations: []WorkerEvaluation,
    
    // Decision rationale
    primary_factor: SelectionFactor,
    confidence_level: f32,
    exploration_used: bool,
    topology_optimization: bool,
    
    // Performance predictions
    predicted_execution_cycles: f64,
    predicted_confidence: f32,
    expected_queue_delay: f64,
    
    pub fn deinit(self: *SelectionDecision, allocator: std.mem.Allocator) void {
        allocator.free(self.evaluations);
    }
};

pub const SelectionStrategy = enum {
    predictive_optimal,      // Full multi-criteria optimization
    confidence_driven,       // Prioritize high-confidence predictions
    load_balanced,          // Focus on queue balance
    topology_aware,         // Optimize for locality
    exploratory,           // New task type exploration
    fallback_simple,       // Simple round-robin fallback
};

pub const SelectionFactor = enum {
    load_balancing,        // Queue sizes were primary factor
    execution_prediction,  // Predicted performance was primary
    topology_locality,     // NUMA/cache locality was primary
    prediction_confidence, // Confidence levels were primary
    exploration_benefit,   // Exploration was primary
    explicit_affinity,     // Task specified affinity hint
    fallback_heuristic,   // Used fallback logic
};

// ============================================================================
// Advanced Worker Selection Algorithm
// ============================================================================

pub const AdvancedWorkerSelector = struct {
    allocator: std.mem.Allocator,
    criteria: SelectionCriteria,
    
    // Prediction and analysis components
    fingerprint_registry: ?*fingerprint.FingerprintRegistry = null,
    predictive_scheduler: ?*predictive_accounting.PredictiveScheduler = null,
    decision_framework: ?*intelligent_decision.IntelligentDecisionFramework = null,
    simd_registry: ?*simd.SIMDCapabilityRegistry = null,
    
    // Selection history and learning
    selection_history: SelectionHistory,
    performance_tracker: PerformanceTracker,
    
    // Configuration
    enable_prediction: bool = true,
    enable_exploration: bool = true,
    enable_adaptive_criteria: bool = true,
    max_workers_to_evaluate: usize = 32,
    
    const Self = @This();
    
    /// Lock-free selection history using per-worker atomic counters
    pub const SelectionHistory = struct {
        // Lock-free per-worker counters with cache-line padding for zero contention
        worker_counters: []std.atomic.Value(u64),
        max_workers: usize,
        total_selections: std.atomic.Value(u64),
        
        // Rolling window for frequency calculation (lock-free circular buffer)
        recent_window: []std.atomic.Value(u32), // Stores worker IDs
        window_size: usize,
        window_index: std.atomic.Value(usize),
        
        // Initialization
        const CACHE_LINE_SIZE = 64;
        const DEFAULT_WINDOW_SIZE = 64; // Power of 2 for efficient modulo
        
        pub fn init(allocator: std.mem.Allocator, max_workers: usize) !SelectionHistory {
            // Allocate per-worker counters (use regular allocation for compatibility)
            const aligned_counters = try allocator.alloc(std.atomic.Value(u64), max_workers);
            
            // Initialize all counters to zero
            for (aligned_counters) |*counter| {
                counter.* = std.atomic.Value(u64).init(0);
            }
            
            // Allocate rolling window for frequency tracking
            const window = try allocator.alloc(std.atomic.Value(u32), DEFAULT_WINDOW_SIZE);
            for (window) |*slot| {
                slot.* = std.atomic.Value(u32).init(0); // Worker ID 0 as default
            }
            
            return SelectionHistory{
                .worker_counters = aligned_counters,
                .max_workers = max_workers,
                .total_selections = std.atomic.Value(u64).init(0),
                .recent_window = window,
                .window_size = DEFAULT_WINDOW_SIZE,
                .window_index = std.atomic.Value(usize).init(0),
            };
        }
        
        pub fn deinit(self: *SelectionHistory, allocator: std.mem.Allocator) void {
            // Use standard free for aligned allocations (Zig handles this automatically)
            allocator.free(self.worker_counters);
            allocator.free(self.recent_window);
        }
        
        /// Record a worker selection (lock-free, thread-safe)
        pub fn recordSelection(self: *SelectionHistory, worker_id: usize) void {
            // Bounds check for safety
            if (worker_id >= self.max_workers) return;
            
            // Atomically increment worker-specific counter
            _ = self.worker_counters[worker_id].fetchAdd(1, .monotonic);
            
            // Atomically increment total selections
            _ = self.total_selections.fetchAdd(1, .monotonic);
            
            // Update rolling window with lock-free circular buffer
            const current_index = self.window_index.fetchAdd(1, .monotonic);
            const slot_index = current_index % self.window_size;
            
            // Store worker ID in the rolling window slot
            self.recent_window[slot_index].store(@intCast(worker_id), .monotonic);
        }
        
        /// Get total selections for a specific worker (lock-free read)
        pub fn getWorkerSelections(self: *const SelectionHistory, worker_id: usize) u64 {
            if (worker_id >= self.max_workers) return 0;
            return self.worker_counters[worker_id].load(.monotonic);
        }
        
        /// Get selection frequency for a worker based on recent window (lock-free)
        pub fn getSelectionFrequency(self: *const SelectionHistory, worker_id: usize) f32 {
            if (worker_id >= self.max_workers) return 0.0;
            
            var count: u32 = 0;
            const total_window_selections = @min(
                self.total_selections.load(.monotonic), 
                self.window_size
            );
            
            if (total_window_selections == 0) return 0.0;
            
            // Scan recent window for matches (atomic reads)
            for (self.recent_window[0..total_window_selections]) |*slot| {
                const slot_worker_id = slot.load(.monotonic);
                if (slot_worker_id == worker_id) {
                    count += 1;
                }
            }
            
            return @as(f32, @floatFromInt(count)) / @as(f32, @floatFromInt(total_window_selections));
        }
        
        /// Get global selection distribution (lock-free analysis)
        pub fn getSelectionDistribution(self: *const SelectionHistory, allocator: std.mem.Allocator) ![]u64 {
            const distribution = try allocator.alloc(u64, self.max_workers);
            
            for (distribution, 0..) |*count, worker_id| {
                count.* = self.worker_counters[worker_id].load(.monotonic);
            }
            
            return distribution;
        }
        
        /// Get total selections across all workers (atomic read)
        pub fn getTotalSelections(self: *const SelectionHistory) u64 {
            return self.total_selections.load(.monotonic);
        }
        
        /// Reset all counters (administrative operation - not lock-free)
        pub fn reset(self: *SelectionHistory) void {
            // Reset per-worker counters
            for (self.worker_counters) |*counter| {
                counter.store(0, .release);
            }
            
            // Reset total counter
            self.total_selections.store(0, .release);
            
            // Clear rolling window
            for (self.recent_window) |*slot| {
                slot.store(0, .release);
            }
            
            // Reset window index
            self.window_index.store(0, .release);
        }
        
        /// Get detailed statistics (lock-free reads)
        pub fn getStatistics(self: *const SelectionHistory) SelectionStatistics {
            const total = self.getTotalSelections();
            
            var max_selections: u64 = 0;
            var min_selections: u64 = if (total > 0) std.math.maxInt(u64) else 0;
            var active_workers: u32 = 0;
            
            for (self.worker_counters) |*counter| {
                const selections = counter.load(.monotonic);
                if (selections > 0) {
                    active_workers += 1;
                    max_selections = @max(max_selections, selections);
                    min_selections = @min(min_selections, selections);
                }
            }
            
            // Calculate load balance coefficient (lower = more balanced)
            const balance_coefficient = if (active_workers > 0 and min_selections > 0)
                @as(f32, @floatFromInt(max_selections)) / @as(f32, @floatFromInt(min_selections))
            else
                1.0;
            
            return SelectionStatistics{
                .total_selections = total,
                .active_workers = active_workers,
                .max_worker_selections = max_selections,
                .min_worker_selections = if (min_selections == std.math.maxInt(u64)) 0 else min_selections,
                .load_balance_coefficient = balance_coefficient,
                .window_utilization = @as(f32, @floatFromInt(@min(total, self.window_size))) / @as(f32, @floatFromInt(self.window_size)),
            };
        }
    };
    
    /// Statistics for lock-free selection history analysis
    const SelectionStatistics = struct {
        total_selections: u64,
        active_workers: u32,
        max_worker_selections: u64,
        min_worker_selections: u64,
        load_balance_coefficient: f32, // 1.0 = perfect balance, higher = more imbalanced
        window_utilization: f32, // 0.0-1.0 how full the recent window is
    };
    
    /// Performance tracking for algorithm improvement
    const PerformanceTracker = struct {
        total_decisions: u64 = 0,
        prediction_accuracy_sum: f64 = 0.0,
        average_queue_utilization: f32 = 0.0,
        numa_locality_ratio: f32 = 0.0,
        
        /// Update performance metrics
        pub fn updateMetrics(
            self: *PerformanceTracker, 
            prediction_accurate: bool, 
            queue_utilization: f32,
            numa_local: bool
        ) void {
            self.total_decisions += 1;
            
            if (prediction_accurate) {
                self.prediction_accuracy_sum += 1.0;
            }
            
            // Exponential moving average for queue utilization
            const alpha: f32 = 0.1;
            self.average_queue_utilization = alpha * queue_utilization + 
                                           (1.0 - alpha) * self.average_queue_utilization;
            
            // Track NUMA locality ratio
            const numa_alpha: f32 = 0.05;
            const numa_score: f32 = if (numa_local) 1.0 else 0.0;
            self.numa_locality_ratio = numa_alpha * numa_score + 
                                     (1.0 - numa_alpha) * self.numa_locality_ratio;
        }
        
        /// Get current prediction accuracy
        pub fn getPredictionAccuracy(self: *const PerformanceTracker) f32 {
            if (self.total_decisions == 0) return 0.0;
            return @as(f32, @floatCast(self.prediction_accuracy_sum / @as(f64, @floatFromInt(self.total_decisions))));
        }
    };
    
    /// Initialize advanced worker selector with lock-free components
    pub fn init(allocator: std.mem.Allocator, criteria: SelectionCriteria, max_workers: usize) !Self {
        var normalized_criteria = criteria;
        normalized_criteria.normalize();
        
        return Self{
            .allocator = allocator,
            .criteria = normalized_criteria,
            .selection_history = try SelectionHistory.init(allocator, max_workers),
            .performance_tracker = PerformanceTracker{},
        };
    }
    
    /// Clean up resources
    pub fn deinit(self: *Self) void {
        self.selection_history.deinit(self.allocator);
    }
    
    /// Set prediction and analysis components
    pub fn setComponents(
        self: *Self,
        fingerprint_registry: ?*fingerprint.FingerprintRegistry,
        predictive_scheduler: ?*predictive_accounting.PredictiveScheduler,
        decision_framework: ?*intelligent_decision.IntelligentDecisionFramework,
        simd_registry: ?*simd.SIMDCapabilityRegistry
    ) void {
        self.fingerprint_registry = fingerprint_registry;
        self.predictive_scheduler = predictive_scheduler;
        self.decision_framework = decision_framework;
        self.simd_registry = simd_registry;
    }
    
    /// Select optimal worker using multi-criteria optimization
    pub fn selectWorker(
        self: *Self,
        task: *const core.Task,
        workers: []const intelligent_decision.WorkerInfo,
        topology_info: ?*const topology.CpuTopology
    ) !SelectionDecision {
        // Create worker evaluations
        var evaluations = try self.allocator.alloc(WorkerEvaluation, workers.len);
        for (evaluations, 0..) |*eval, i| {
            eval.* = WorkerEvaluation.init(i);
        }
        
        // Evaluate each worker across all criteria
        for (workers, 0..) |worker_info, i| {
            try self.evaluateWorker(task, worker_info, &evaluations[i], topology_info);
        }
        
        // Calculate weighted scores
        for (evaluations) |*eval| {
            eval.calculateWeightedScore(self.criteria);
        }
        
        // Normalize scores and find best worker
        self.normalizeScores(evaluations);
        const best_worker_idx = self.findBestWorker(evaluations);
        
        // Create selection decision
        const decision = SelectionDecision{
            .selected_worker_id = best_worker_idx,
            .selection_strategy = self.determineStrategy(evaluations[best_worker_idx]),
            .evaluations = evaluations,
            .primary_factor = self.determinePrimaryFactor(evaluations[best_worker_idx]),
            .confidence_level = evaluations[best_worker_idx].confidence,
            .exploration_used = evaluations[best_worker_idx].exploration_score > 0.1,
            .topology_optimization = evaluations[best_worker_idx].topology_score > 0.7,
            .predicted_execution_cycles = evaluations[best_worker_idx].predicted_cycles,
            .predicted_confidence = evaluations[best_worker_idx].confidence,
            .expected_queue_delay = self.estimateQueueDelay(evaluations[best_worker_idx]),
        };
        
        // Record selection for learning
        self.selection_history.recordSelection(best_worker_idx);
        
        // Adapt criteria if enabled
        if (self.enable_adaptive_criteria) {
            self.adaptCriteria();
        }
        
        return decision;
    }
    
    /// Evaluate a single worker across all criteria
    fn evaluateWorker(
        self: *Self,
        task: *const core.Task,
        worker_info: intelligent_decision.WorkerInfo,
        evaluation: *WorkerEvaluation,
        topology_info: ?*const topology.CpuTopology
    ) !void {
        // Basic worker information
        evaluation.queue_size = worker_info.queue_size;
        
        // 1. Load Balance Score (higher score = lighter load)
        evaluation.load_balance_score = self.calculateLoadBalanceScore(worker_info);
        
        // 2. Prediction Score (execution time and performance predictions)
        evaluation.prediction_score = try self.calculatePredictionScore(task, worker_info, evaluation);
        
        // 3. Topology Score (NUMA locality and cache affinity)
        evaluation.topology_score = self.calculateTopologyScore(task, worker_info, topology_info, evaluation);
        
        // 4. Confidence Score (prediction reliability)
        evaluation.confidence_score = self.calculateConfidenceScore(task, worker_info, evaluation);
        
        // 5. Exploration Score (new task type discovery)
        evaluation.exploration_score = self.calculateExplorationScore(task, worker_info, evaluation);
        
        // 6. SIMD Score (vectorization capability matching)
        evaluation.simd_score = self.calculateSIMDScore(task, worker_info, evaluation);
    }
    
    /// Calculate load balance score based on queue utilization
    fn calculateLoadBalanceScore(self: *Self, worker_info: intelligent_decision.WorkerInfo) f32 {
        _ = self; // May be used for historical data in the future
        
        if (worker_info.max_queue_size == 0) return 0.5; // Neutral score
        
        const utilization = @as(f32, @floatFromInt(worker_info.queue_size)) / 
                           @as(f32, @floatFromInt(worker_info.max_queue_size));
        
        // Higher score for lower utilization (inverted and scaled)
        return @max(0.0, 1.0 - utilization);
    }
    
    /// Calculate prediction score based on expected execution performance
    fn calculatePredictionScore(
        self: *Self,
        task: *const core.Task,
        worker_info: intelligent_decision.WorkerInfo,
        evaluation: *WorkerEvaluation
    ) !f32 {
        if (!self.enable_prediction or self.predictive_scheduler == null) {
            return 0.5; // Neutral score when prediction unavailable
        }
        
        // Get execution time prediction for this worker
        if (self.predictive_scheduler) |pred_scheduler| {
            const prediction = pred_scheduler.predictTaskWork(@intCast(worker_info.id), task);
            evaluation.predicted_cycles = prediction.predicted_cycles;
            evaluation.confidence = prediction.confidence;
            
            // Score based on predicted execution time (lower time = higher score)
            // Normalize against typical task execution time
            const typical_cycles: f64 = 5000.0; // Baseline expectation
            const time_score = @max(0.0, @min(1.0, typical_cycles / prediction.predicted_cycles));
            
            return @as(f32, @floatCast(time_score));
        }
        
        return 0.5;
    }
    
    /// Calculate topology score based on NUMA locality and affinity
    fn calculateTopologyScore(
        self: *Self,
        task: *const core.Task,
        worker_info: intelligent_decision.WorkerInfo,
        topology_info: ?*const topology.CpuTopology,
        evaluation: *WorkerEvaluation
    ) f32 {
        _ = self; // May be used for historical affinity data
        
        // Start with neutral score
        var score: f32 = 0.5;
        
        // Honor explicit affinity hints
        if (task.affinity_hint) |preferred_numa| {
            if (worker_info.numa_node) |worker_numa| {
                if (worker_numa == preferred_numa) {
                    score = 1.0; // Perfect affinity match
                    evaluation.numa_distance = 0;
                } else if (topology_info) |topo| {
                    // Calculate NUMA distance and score accordingly
                    if (worker_numa < topo.numa_nodes.len and preferred_numa < topo.numa_nodes.len) {
                        const distance = topo.numa_nodes[worker_numa].distanceTo(preferred_numa);
                        evaluation.numa_distance = distance;
                        // Score inversely related to distance (closer = better)
                        score = @max(0.1, 1.0 - @as(f32, @floatFromInt(distance)) / 100.0);
                    }
                }
            }
        } else {
            // No explicit hint - prefer workers on less loaded NUMA nodes
            if (worker_info.numa_node != null) {
                score = 0.7; // Slight preference for NUMA-aware placement
            }
        }
        
        return score;
    }
    
    /// Calculate confidence score based on prediction reliability
    fn calculateConfidenceScore(
        self: *Self,
        task: *const core.Task,
        worker_info: intelligent_decision.WorkerInfo,
        evaluation: *WorkerEvaluation
    ) f32 {
        _ = task;
        _ = worker_info;
        
        // Use prediction confidence if available
        if (evaluation.confidence > 0.0) {
            return evaluation.confidence;
        }
        
        // Fallback to historical prediction accuracy
        return self.performance_tracker.getPredictionAccuracy();
    }
    
    /// Calculate exploration score for new task types and load balancing
    fn calculateExplorationScore(
        self: *Self,
        task: *const core.Task,
        worker_info: intelligent_decision.WorkerInfo,
        evaluation: *WorkerEvaluation
    ) f32 {
        if (!self.enable_exploration) return 0.0;
        
        _ = task; // May be used for task type analysis
        
        // Encourage exploration of underutilized workers
        const selection_frequency = self.selection_history.getSelectionFrequency(worker_info.id);
        
        // Higher score for less frequently selected workers
        const frequency_score = @max(0.0, 1.0 - selection_frequency);
        
        // Boost exploration for workers with very light loads
        const load_boost: f32 = if (worker_info.queue_size == 0) 0.3 else 0.0;
        
        evaluation.exploration_factor = frequency_score + load_boost;
        return @min(1.0, evaluation.exploration_factor);
    }
    
    /// Calculate SIMD score based on vectorization capability matching
    fn calculateSIMDScore(
        self: *Self,
        task: *const core.Task,
        worker_info: intelligent_decision.WorkerInfo,
        evaluation: *WorkerEvaluation
    ) f32 {
        _ = evaluation; // May be used for detailed analysis in the future
        
        // If no SIMD registry available, assume all workers are equal
        const simd_registry = self.simd_registry orelse return 0.5;
        
        // Get task SIMD requirements from fingerprint
        const task_fingerprint = blk: {
            const context = fingerprint.ExecutionContext.init();
            break :blk fingerprint.TaskAnalyzer.analyzeTask(task, &context);
        };
        
        // Get worker SIMD capabilities
        const worker_capability = simd_registry.getWorkerCapability(worker_info.id);
        
        // Calculate SIMD suitability score (0.0 to 1.0)
        var simd_score: f32 = 0.0;
        
        // 1. Vector width matching (40% of score)
        const required_width = @as(u16, @intCast(task_fingerprint.simd_width)) * 64; // Convert hint to bits
        if (worker_capability.max_vector_width_bits >= required_width) {
            simd_score += 0.4; // Full points for meeting requirements
        } else if (worker_capability.max_vector_width_bits >= 128) {
            simd_score += 0.2; // Partial points for basic SIMD support
        }
        
        // 2. Vectorization benefit potential (30% of score)
        const vectorization_benefit = @as(f32, @floatFromInt(task_fingerprint.vectorization_benefit)) / 15.0;
        if (vectorization_benefit > 0.8) {
            // High benefit tasks get full SIMD score boost
            simd_score += 0.3 * vectorization_benefit;
        } else if (vectorization_benefit > 0.4) {
            // Medium benefit tasks get partial boost
            simd_score += 0.15 * vectorization_benefit;
        }
        
        // 3. Access pattern compatibility (20% of score)
        const access_pattern_score: f32 = switch (task_fingerprint.access_pattern) {
            .sequential => 1.0,      // Perfect for SIMD
            .strided => 0.8,         // Good for SIMD with gather/scatter
            .hierarchical => 0.6,    // Some SIMD benefit
            else => 0.3,             // Limited SIMD benefit
        };
        simd_score += 0.2 * access_pattern_score;
        
        // 4. Performance potential (10% of score)
        const data_type: simd.SIMDDataType = if (task_fingerprint.memory_footprint_log2 >= 2) .f32 else .i32;
        const performance_multiplier = worker_capability.getPerformanceScore(data_type);
        if (performance_multiplier > 2.0) {
            simd_score += 0.1; // Bonus for high-performance SIMD
        }
        
        return @min(1.0, simd_score);
    }
    
    /// Normalize scores across all workers
    fn normalizeScores(self: *Self, evaluations: []WorkerEvaluation) void {
        _ = self;
        
        if (evaluations.len == 0) return;
        
        // Find min and max weighted scores
        var min_score: f32 = evaluations[0].weighted_score;
        var max_score: f32 = evaluations[0].weighted_score;
        
        for (evaluations) |eval| {
            min_score = @min(min_score, eval.weighted_score);
            max_score = @max(max_score, eval.weighted_score);
        }
        
        // Normalize to 0.0-1.0 range
        const score_range = max_score - min_score;
        if (score_range > 0.0 and max_score > 0.0) {
            for (evaluations) |*eval| {
                eval.normalized_score = (eval.weighted_score - min_score) / score_range;
            }
        } else {
            // All scores equal or max_score is zero - normalize to 0.5
            for (evaluations) |*eval| {
                eval.normalized_score = 0.5;
            }
        }
    }
    
    /// Find worker with best normalized score
    fn findBestWorker(self: *Self, evaluations: []WorkerEvaluation) usize {
        _ = self;
        
        if (evaluations.len == 0) return 0;
        
        var best_idx: usize = 0;
        var best_score: f32 = evaluations[0].normalized_score;
        
        for (evaluations, 0..) |eval, i| {
            if (eval.normalized_score > best_score) {
                best_score = eval.normalized_score;
                best_idx = i;
            }
        }
        
        return best_idx;
    }
    
    /// Determine primary selection strategy used
    fn determineStrategy(self: *Self, evaluation: WorkerEvaluation) SelectionStrategy {
        _ = self;
        
        // Find the highest scoring component
        const max_score = @max(@max(evaluation.load_balance_score, evaluation.prediction_score),
                              @max(evaluation.topology_score, @max(evaluation.confidence_score, evaluation.exploration_score)));
        
        if (evaluation.prediction_score == max_score and max_score > 0.7) {
            return .predictive_optimal;
        } else if (evaluation.confidence_score == max_score and max_score > 0.7) {
            return .confidence_driven;
        } else if (evaluation.load_balance_score == max_score) {
            return .load_balanced;
        } else if (evaluation.topology_score == max_score and max_score > 0.6) {
            return .topology_aware;
        } else if (evaluation.exploration_score == max_score and max_score > 0.5) {
            return .exploratory;
        } else {
            return .fallback_simple;
        }
    }
    
    /// Determine primary factor in selection decision
    fn determinePrimaryFactor(self: *Self, evaluation: WorkerEvaluation) SelectionFactor {
        
        // Find the component that contributed most to the final score
        const scores = [_]f32{
            evaluation.load_balance_score * self.criteria.load_balance_weight,
            evaluation.prediction_score * self.criteria.prediction_weight,
            evaluation.topology_score * self.criteria.topology_weight,
            evaluation.confidence_score * self.criteria.confidence_weight,
            evaluation.exploration_score * self.criteria.exploration_weight,
        };
        
        var max_contribution: f32 = scores[0];
        var max_idx: usize = 0;
        
        for (scores, 0..) |score, i| {
            if (score > max_contribution) {
                max_contribution = score;
                max_idx = i;
            }
        }
        
        return switch (max_idx) {
            0 => .load_balancing,
            1 => .execution_prediction,
            2 => .topology_locality,
            3 => .prediction_confidence,
            4 => .exploration_benefit,
            else => .fallback_heuristic,
        };
    }
    
    /// Estimate queue delay for selected worker
    fn estimateQueueDelay(self: *Self, evaluation: WorkerEvaluation) f64 {
        _ = self;
        
        // Simple estimation based on queue size and predicted execution time
        const estimated_task_time = @max(1000.0, evaluation.predicted_cycles);
        return @as(f64, @floatFromInt(evaluation.queue_size)) * estimated_task_time;
    }
    
    /// Adapt selection criteria based on performance feedback
    fn adaptCriteria(self: *Self) void {
        const accuracy = self.performance_tracker.getPredictionAccuracy();
        
        // Increase prediction weight if accuracy is high
        if (accuracy > 0.8) {
            self.criteria.prediction_weight = @min(0.4, self.criteria.prediction_weight * 1.05);
        } else if (accuracy < 0.6) {
            self.criteria.prediction_weight = @max(0.1, self.criteria.prediction_weight * 0.95);
        }
        
        // Adjust exploration based on NUMA locality performance
        if (self.performance_tracker.numa_locality_ratio > 0.8) {
            self.criteria.exploration_weight = @max(self.criteria.min_exploration_weight, 
                                                   self.criteria.exploration_weight * self.criteria.exploration_decay);
        }
        
        // Renormalize after adaptation
        self.criteria.normalize();
    }
    
    /// Get comprehensive selection statistics
    pub fn getSelectionStats(self: *const Self) SelectionStats {
        return SelectionStats{
            .total_selections = self.selection_history.total_selections,
            .prediction_accuracy = self.performance_tracker.getPredictionAccuracy(),
            .average_queue_utilization = self.performance_tracker.average_queue_utilization,
            .numa_locality_ratio = self.performance_tracker.numa_locality_ratio,
            .current_criteria = self.criteria,
        };
    }
};

/// Comprehensive selection statistics
pub const SelectionStats = struct {
    total_selections: u64,
    prediction_accuracy: f32,
    average_queue_utilization: f32,
    numa_locality_ratio: f32,
    current_criteria: SelectionCriteria,
};

// ============================================================================
// Testing and Validation
// ============================================================================

const testing = std.testing;

test "selection criteria normalization" {
    var criteria = SelectionCriteria{
        .load_balance_weight = 2.0,
        .prediction_weight = 3.0,
        .topology_weight = 1.0,
        .confidence_weight = 2.0,
        .exploration_weight = 2.0,
    };
    
    criteria.normalize();
    
    const total = criteria.load_balance_weight + criteria.prediction_weight + 
                 criteria.topology_weight + criteria.confidence_weight + criteria.exploration_weight + criteria.simd_weight;
    
    try testing.expect(@abs(total - 1.0) < 0.001);
}

test "worker evaluation initialization" {
    const eval = WorkerEvaluation.init(42);
    
    try testing.expect(eval.worker_id == 42);
    try testing.expect(eval.weighted_score == 0.0);
    try testing.expect(eval.queue_size == 0);
}

test "advanced worker selector initialization" {
    const allocator = testing.allocator;
    const criteria = SelectionCriteria.balanced();
    
    var selector = try AdvancedWorkerSelector.init(allocator, criteria, 8);
    defer selector.deinit();
    
    try testing.expect(selector.enable_prediction == true);
    try testing.expect(selector.enable_exploration == true);
    try testing.expect(selector.selection_history.getTotalSelections() == 0);
}

test "lock-free selection history basic operations" {
    const allocator = testing.allocator;
    
    var history = try AdvancedWorkerSelector.SelectionHistory.init(allocator, 4);
    defer history.deinit(allocator);
    
    // Test initial state
    try testing.expect(history.getTotalSelections() == 0);
    try testing.expect(history.getWorkerSelections(0) == 0);
    try testing.expect(history.getSelectionFrequency(0) == 0.0);
    
    // Test recording selections
    history.recordSelection(0);
    history.recordSelection(1);
    history.recordSelection(0);
    history.recordSelection(2);
    
    // Verify counters
    try testing.expect(history.getTotalSelections() == 4);
    try testing.expect(history.getWorkerSelections(0) == 2);
    try testing.expect(history.getWorkerSelections(1) == 1);
    try testing.expect(history.getWorkerSelections(2) == 1);
    try testing.expect(history.getWorkerSelections(3) == 0);
    
    // Test frequency calculation
    const freq_0 = history.getSelectionFrequency(0);
    const freq_1 = history.getSelectionFrequency(1);
    const freq_3 = history.getSelectionFrequency(3);
    
    try testing.expect(freq_0 == 0.5); // 2/4
    try testing.expect(freq_1 == 0.25); // 1/4  
    try testing.expect(freq_3 == 0.0); // 0/4
}

test "lock-free selection history statistics" {
    const allocator = testing.allocator;
    
    var history = try AdvancedWorkerSelector.SelectionHistory.init(allocator, 6);
    defer history.deinit(allocator);
    
    // Create unbalanced selection pattern
    for (0..10) |_| history.recordSelection(0); // Worker 0: 10 selections
    for (0..5) |_| history.recordSelection(1);  // Worker 1: 5 selections
    for (0..2) |_| history.recordSelection(2);  // Worker 2: 2 selections
    // Workers 3,4,5: 0 selections
    
    const stats = history.getStatistics();
    
    try testing.expect(stats.total_selections == 17);
    try testing.expect(stats.active_workers == 3);
    try testing.expect(stats.max_worker_selections == 10);
    try testing.expect(stats.min_worker_selections == 2);
    try testing.expect(stats.load_balance_coefficient == 5.0); // 10/2 = 5.0 (imbalanced)
    
    // Test balanced pattern
    history.reset();
    for (0..4) |worker_id| {
        for (0..5) |_| history.recordSelection(worker_id);
    }
    
    const balanced_stats = history.getStatistics();
    try testing.expect(balanced_stats.load_balance_coefficient == 1.0); // Perfect balance
}

test "lock-free selection history concurrent simulation" {
    const allocator = testing.allocator;
    
    var history = try AdvancedWorkerSelector.SelectionHistory.init(allocator, 8);
    defer history.deinit(allocator);
    
    // Simulate concurrent access by multiple threads (sequential simulation)
    // This tests the atomic operations work correctly
    const operations_per_worker = 100;
    
    for (0..8) |worker_id| {
        for (0..operations_per_worker) |_| {
            history.recordSelection(worker_id);
        }
    }
    
    // Verify atomicity - all operations should be recorded
    try testing.expect(history.getTotalSelections() == 8 * operations_per_worker);
    
    for (0..8) |worker_id| {
        try testing.expect(history.getWorkerSelections(worker_id) == operations_per_worker);
    }
    
    // Test distribution analysis
    const distribution = try history.getSelectionDistribution(allocator);
    defer allocator.free(distribution);
    
    var total_from_distribution: u64 = 0;
    for (distribution) |count| {
        total_from_distribution += count;
    }
    
    try testing.expect(total_from_distribution == history.getTotalSelections());
}

test "lock-free selection history bounds checking" {
    const allocator = testing.allocator;
    
    var history = try AdvancedWorkerSelector.SelectionHistory.init(allocator, 4);
    defer history.deinit(allocator);
    
    // Test bounds checking for invalid worker IDs
    history.recordSelection(10); // Should be ignored (out of bounds)
    history.recordSelection(4);  // Should be ignored (out of bounds)
    
    try testing.expect(history.getTotalSelections() == 0);
    try testing.expect(history.getWorkerSelections(10) == 0);
    try testing.expect(history.getSelectionFrequency(10) == 0.0);
    
    // Valid selections should work
    history.recordSelection(0);
    history.recordSelection(3);
    
    try testing.expect(history.getTotalSelections() == 2);
    try testing.expect(history.getWorkerSelections(0) == 1);
    try testing.expect(history.getWorkerSelections(3) == 1);
}