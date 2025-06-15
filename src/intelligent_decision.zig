const std = @import("std");
const builtin = @import("builtin");
const core = @import("core.zig");
const topology = @import("topology.zig");
const fingerprint = @import("fingerprint.zig");

// Intelligent Decision Framework for Beat.zig (Task 2.3.2)
//
// This module implements sophisticated scheduling decisions based on
// multi-factor confidence levels, providing:
// - Confidence thresholds for scheduling decisions
// - Conservative placement for low confidence tasks
// - NUMA optimization for high confidence long tasks  
// - Balanced approach for medium confidence tasks

// ============================================================================
// Decision Framework Configuration
// ============================================================================

/// Configuration for intelligent scheduling decisions
pub const DecisionConfig = struct {
    // Confidence thresholds for different strategies
    high_confidence_threshold: f32 = 0.8,      // Above this: aggressive optimization
    medium_confidence_threshold: f32 = 0.5,    // Above this: balanced approach
    low_confidence_threshold: f32 = 0.2,       // Above this: conservative placement
    // Below low_confidence_threshold: very conservative
    
    // Long task detection (cycles threshold for NUMA optimization)
    long_task_cycles_threshold: u64 = 10_000,  // Tasks predicted to run >10k cycles
    
    // Conservative placement parameters
    conservative_queue_fill_ratio: f32 = 0.7,  // Avoid queues >70% full for low confidence
    conservative_numa_locality_boost: f32 = 1.5, // Prefer local NUMA for uncertain tasks
    
    // Aggressive optimization parameters
    aggressive_numa_optimization: bool = true,  // Enable cross-NUMA placement for long tasks
    aggressive_load_balancing: bool = true,     // More aggressive load distribution
    
    // Balanced approach parameters
    balanced_confidence_weight: f32 = 0.6,     // Weight confidence vs load in decisions
    balanced_prediction_weight: f32 = 0.4,     // Weight predicted execution time
};

// ============================================================================
// Decision Strategies
// ============================================================================

/// Scheduling strategy based on confidence analysis
pub const SchedulingStrategy = enum {
    very_conservative,  // Minimal optimization, safe placement
    conservative,       // Basic optimization, prefer local placement
    balanced,          // Moderate optimization with confidence weighting
    aggressive,        // Full optimization, use all available information
    
    /// Get strategy from confidence category
    pub fn fromConfidence(confidence: fingerprint.MultiFactorConfidence.ConfidenceCategory) SchedulingStrategy {
        return switch (confidence) {
            .very_low => .very_conservative,
            .low => .conservative,
            .medium => .balanced,
            .high => .aggressive,
        };
    }
};

/// Comprehensive scheduling decision with rationale
pub const SchedulingDecision = struct {
    worker_id: usize,
    strategy: SchedulingStrategy,
    confidence: fingerprint.MultiFactorConfidence,
    predicted_cycles: ?f64,
    rationale: DecisionRationale,
    
    /// Rationale for the scheduling decision
    pub const DecisionRationale = struct {
        primary_factor: PrimaryFactor,
        numa_optimization: bool,
        load_balancing_applied: bool,
        confidence_influenced: bool,
        fallback_used: bool,
        
        pub const PrimaryFactor = enum {
            explicit_affinity,      // Task specified affinity hint
            confidence_driven,      // Decision driven by confidence level
            load_balancing,        // Purely load-based decision
            numa_optimization,     // NUMA-aware placement for long tasks
            conservative_safety,   // Safety-first placement for uncertain tasks
        };
    };
};

// ============================================================================
// Intelligent Decision Framework
// ============================================================================

/// Main intelligent decision framework for task scheduling
pub const IntelligentDecisionFramework = struct {
    config: DecisionConfig,
    fingerprint_registry: ?*fingerprint.FingerprintRegistry,
    
    const Self = @This();
    
    /// Initialize the decision framework
    pub fn init(config: DecisionConfig) Self {
        return Self{
            .config = config,
            .fingerprint_registry = null,
        };
    }
    
    /// Set the fingerprint registry for confidence-based decisions
    pub fn setFingerprintRegistry(self: *Self, registry: *fingerprint.FingerprintRegistry) void {
        self.fingerprint_registry = registry;
    }
    
    /// Make intelligent scheduling decision for a task
    pub fn makeSchedulingDecision(
        self: *Self, 
        task: *const core.Task, 
        workers: []const WorkerInfo, 
        pool_topology: ?topology.CpuTopology
    ) SchedulingDecision {
        // 1. Get confidence analysis if fingerprinting is available
        var confidence_opt: ?fingerprint.MultiFactorConfidence = null;
        var predicted_cycles_opt: ?f64 = null;
        
        if (self.fingerprint_registry) |registry| {
            // Generate fingerprint for the task
            var context = fingerprint.ExecutionContext.init();
            // pool_topology will be used for topology-aware fingerprinting in future
            
            const task_fingerprint = fingerprint.generateTaskFingerprint(task, &context);
            confidence_opt = registry.getMultiFactorConfidence(task_fingerprint);
            
            // Get prediction if we have sufficient confidence
            if (confidence_opt) |conf| {
                if (conf.overall_confidence > 0.1) { // Minimum confidence for prediction
                    predicted_cycles_opt = registry.getPredictedCycles(task_fingerprint);
                }
            }
        }
        
        // 2. Handle explicit affinity hint (highest priority)
        if (task.affinity_hint) |numa_node| {
            const worker_id = self.selectWorkerOnNumaNode(workers, pool_topology, numa_node);
            return SchedulingDecision{
                .worker_id = worker_id,
                .strategy = .balanced, // Respect user intent but apply some intelligence
                .confidence = confidence_opt orelse defaultConfidence(),
                .predicted_cycles = predicted_cycles_opt,
                .rationale = .{
                    .primary_factor = .explicit_affinity,
                    .numa_optimization = true,
                    .load_balancing_applied = true,
                    .confidence_influenced = false,
                    .fallback_used = false,
                },
            };
        }
        
        // 3. Confidence-driven decision making
        if (confidence_opt) |confidence| {
            const strategy = SchedulingStrategy.fromConfidence(confidence.getConfidenceCategory());
            
            const worker_id = switch (strategy) {
                .very_conservative => self.makeVeryConservativeDecision(workers, pool_topology),
                .conservative => self.makeConservativeDecision(workers, pool_topology, confidence),
                .balanced => self.makeBalancedDecision(workers, pool_topology, confidence, predicted_cycles_opt),
                .aggressive => self.makeAggressiveDecision(workers, pool_topology, confidence, predicted_cycles_opt),
            };
            
            return SchedulingDecision{
                .worker_id = worker_id,
                .strategy = strategy,
                .confidence = confidence,
                .predicted_cycles = predicted_cycles_opt,
                .rationale = .{
                    .primary_factor = .confidence_driven,
                    .numa_optimization = strategy == .aggressive,
                    .load_balancing_applied = true,
                    .confidence_influenced = true,
                    .fallback_used = false,
                },
            };
        }
        
        // 4. Fallback to topology-aware load balancing
        const worker_id = if (pool_topology) |topo|
            self.selectWorkerWithLoadBalancing(workers, topo)
        else
            self.selectLightestWorker(workers);
        
        return SchedulingDecision{
            .worker_id = worker_id,
            .strategy = .balanced,
            .confidence = defaultConfidence(),
            .predicted_cycles = null,
            .rationale = .{
                .primary_factor = .load_balancing,
                .numa_optimization = pool_topology != null,
                .load_balancing_applied = true,
                .confidence_influenced = false,
                .fallback_used = true,
            },
        };
    }
    
    // ========================================================================
    // Strategy Implementations  
    // ========================================================================
    
    /// Very conservative strategy: minimal optimization, safety first
    pub fn makeVeryConservativeDecision(
        self: *Self,
        workers: []const WorkerInfo,
        pool_topology: ?topology.CpuTopology
    ) usize {
        _ = self;
        _ = pool_topology;
        
        // For very low confidence, just pick the worker with smallest queue
        // Avoid any risky optimizations that might fail
        var best_worker: usize = 0;
        var min_queue_size: usize = std.math.maxInt(usize);
        
        for (workers, 0..) |worker, i| {
            if (worker.queue_size < min_queue_size) {
                min_queue_size = worker.queue_size;
                best_worker = i;
            }
        }
        
        return best_worker;
    }
    
    /// Conservative strategy: basic optimization, prefer local placement
    pub fn makeConservativeDecision(
        self: *Self,
        workers: []const WorkerInfo,
        pool_topology: ?topology.CpuTopology,
        confidence: fingerprint.MultiFactorConfidence
    ) usize {
        _ = confidence; // May be used for fine-tuning in the future
        
        if (pool_topology) |_| {
            // Prefer local NUMA node but avoid overloaded workers
            const current_numa = self.getCurrentNumaNode();
            
            var best_worker: usize = 0;
            var best_score: f32 = std.math.floatMax(f32);
            
            for (workers, 0..) |worker, i| {
                const queue_ratio = @as(f32, @floatFromInt(worker.queue_size)) / @as(f32, @floatFromInt(worker.max_queue_size));
                
                // Skip workers that are too full (conservative threshold)
                if (queue_ratio > self.config.conservative_queue_fill_ratio) continue;
                
                var score = @as(f32, @floatFromInt(worker.queue_size));
                
                // Boost local NUMA node preference
                if (worker.numa_node == current_numa) {
                    score /= self.config.conservative_numa_locality_boost;
                }
                
                if (score < best_score) {
                    best_score = score;
                    best_worker = i;
                }
            }
            
            return best_worker;
        } else {
            // No topology info, fall back to simple selection
            return self.selectLightestWorker(workers);
        }
    }
    
    /// Balanced strategy: moderate optimization with confidence weighting
    pub fn makeBalancedDecision(
        self: *Self,
        workers: []const WorkerInfo,
        pool_topology: ?topology.CpuTopology,
        confidence: fingerprint.MultiFactorConfidence,
        predicted_cycles: ?f64
    ) usize {
        if (pool_topology) |_| {
            var best_worker: usize = 0;
            var best_score: f32 = std.math.floatMax(f32);
            
            for (workers, 0..) |worker, i| {
                const queue_size = worker.queue_size;
                var score = @as(f32, @floatFromInt(queue_size));
                
                // Factor in predicted execution time if available
                if (predicted_cycles) |cycles| {
                    const normalized_cycles = @as(f32, @floatCast(cycles / 10000.0)); // Normalize to ~1.0 scale
                    score += normalized_cycles * self.config.balanced_prediction_weight;
                }
                
                // Apply confidence weighting
                const confidence_factor = 1.0 + (1.0 - confidence.overall_confidence) * self.config.balanced_confidence_weight;
                score *= confidence_factor;
                
                // NUMA node preference (moderate boost)
                const current_numa = self.getCurrentNumaNode();
                if (worker.numa_node == current_numa) {
                    score *= 0.8; // 20% preference for local NUMA
                }
                
                if (score < best_score) {
                    best_score = score;
                    best_worker = i;
                }
            }
            
            return best_worker;
        } else {
            return self.selectLightestWorker(workers);
        }
    }
    
    /// Aggressive strategy: full optimization using all available information
    pub fn makeAggressiveDecision(
        self: *Self,
        workers: []const WorkerInfo,
        pool_topology: ?topology.CpuTopology,
        confidence: fingerprint.MultiFactorConfidence,
        predicted_cycles: ?f64
    ) usize {
        if (pool_topology) |topo| {
            // For high confidence tasks, use sophisticated optimization
            
            // Check if this is a long-running task that benefits from NUMA optimization
            const is_long_task = if (predicted_cycles) |cycles|
                cycles > @as(f64, @floatFromInt(self.config.long_task_cycles_threshold))
            else
                false;
            
            if (is_long_task and self.config.aggressive_numa_optimization) {
                return self.optimizeForLongTask(workers, topo, confidence, predicted_cycles.?);
            } else {
                return self.optimizeForShortTask(workers, topo, confidence);
            }
        } else {
            return self.selectLightestWorker(workers);
        }
    }
    
    /// Optimize placement for long-running tasks
    fn optimizeForLongTask(
        self: *Self,
        workers: []const WorkerInfo,
        topo: topology.CpuTopology,
        confidence: fingerprint.MultiFactorConfidence,
        predicted_cycles: f64
    ) usize {
        _ = confidence;
        
        // For long tasks, minimize worker migration and maximize cache locality
        // Find the NUMA node with the most available capacity
        
        var numa_scores: [8]f32 = [_]f32{0.0} ** 8; // Support up to 8 NUMA nodes
        var numa_worker_counts: [8]usize = [_]usize{0} ** 8;
        
        // Calculate per-NUMA-node load
        for (workers, 0..) |worker, i| {
            _ = i;
            const numa_node = @min(worker.numa_node orelse 0, 7);
            const queue_load = @as(f32, @floatFromInt(worker.queue_size));
            numa_scores[numa_node] += queue_load;
            numa_worker_counts[numa_node] += 1;
        }
        
        // Find the NUMA node with lowest average load
        var best_numa: u32 = 0;
        var best_numa_score: f32 = std.math.floatMax(f32);
        
        for (0..@min(topo.numa_nodes.len, 8)) |numa_idx| {
            if (numa_worker_counts[numa_idx] == 0) continue;
            
            const avg_load = numa_scores[numa_idx] / @as(f32, @floatFromInt(numa_worker_counts[numa_idx]));
            
            // Factor in predicted execution time (longer tasks prefer less loaded nodes)
            const load_penalty = avg_load * @as(f32, @floatCast(predicted_cycles / 50000.0));
            
            if (load_penalty < best_numa_score) {
                best_numa_score = load_penalty;
                best_numa = @intCast(numa_idx);
            }
        }
        
        // Select the best worker on the chosen NUMA node
        return self.selectWorkerOnNumaNode(workers, topo, best_numa);
    }
    
    /// Optimize placement for short-running tasks
    fn optimizeForShortTask(
        self: *Self,
        workers: []const WorkerInfo,
        topo: topology.CpuTopology,
        confidence: fingerprint.MultiFactorConfidence
    ) usize {
        _ = self;
        _ = topo;
        
        // For short tasks, prioritize immediate availability
        var best_worker: usize = 0;
        var best_score: f32 = std.math.floatMax(f32);
        
        for (workers, 0..) |worker, i| {
            const queue_size = worker.queue_size;
            var score = @as(f32, @floatFromInt(queue_size));
            
            // High confidence allows more aggressive load balancing
            const aggressiveness = confidence.overall_confidence;
            score *= (1.0 + aggressiveness * 0.5); // Up to 50% more aggressive
            
            if (score < best_score) {
                best_score = score;
                best_worker = i;
            }
        }
        
        return best_worker;
    }
    
    // ========================================================================
    // Utility Functions
    // ========================================================================
    
    /// Select worker on specific NUMA node with load balancing
    fn selectWorkerOnNumaNode(
        self: *Self,
        workers: []const WorkerInfo,
        pool_topology: ?topology.CpuTopology,
        numa_node: u32
    ) usize {
        _ = pool_topology;
        
        var best_worker: usize = 0;
        var min_queue_size: usize = std.math.maxInt(usize);
        var found_on_numa = false;
        
        for (workers, 0..) |worker, i| {
            if (worker.numa_node == numa_node) {
                const queue_size = worker.queue_size;
                if (!found_on_numa or queue_size < min_queue_size) {
                    min_queue_size = queue_size;
                    best_worker = i;
                    found_on_numa = true;
                }
            }
        }
        
        // If no worker found on specified NUMA node, fall back
        if (!found_on_numa) {
            return self.selectLightestWorker(workers);
        }
        
        return best_worker;
    }
    
    /// Load balancing with topology awareness
    fn selectWorkerWithLoadBalancing(
        self: *Self,
        workers: []const WorkerInfo,
        topo: topology.CpuTopology
    ) usize {
        // Round-robin across NUMA nodes for initial distribution
        const numa_count = @min(topo.numa_nodes.len, 8);
        if (numa_count == 0) return self.selectLightestWorker(workers);
        
        // Simple distribution based on worker count modulo NUMA nodes
        const preferred_numa = @as(u32, @intCast(workers.len % numa_count));
        
        return self.selectWorkerOnNumaNode(workers, null, preferred_numa);
    }
    
    /// Simple worker selection by queue size
    fn selectLightestWorker(self: *Self, workers: []const WorkerInfo) usize {
        _ = self;
        
        var best_worker: usize = 0;
        var min_queue_size: usize = std.math.maxInt(usize);
        
        for (workers, 0..) |worker, i| {
            const queue_size = worker.queue_size;
            if (queue_size < min_queue_size) {
                min_queue_size = queue_size;
                best_worker = i;
            }
        }
        
        return best_worker;
    }
    
    /// Get current NUMA node (simplified implementation)
    fn getCurrentNumaNode(self: *Self) u32 {
        _ = self;
        // TODO: Implement actual NUMA node detection
        // For now, return 0 as a fallback
        return 0;
    }
};

// ============================================================================
// Worker Interface for Decision Framework
// ============================================================================

/// Simplified worker information for decision making
pub const WorkerInfo = struct {
    id: u32,
    numa_node: ?u32,
    queue_size: usize,
    max_queue_size: usize,
};

// ============================================================================
// Utility Functions
// ============================================================================

/// Default confidence for tasks without fingerprinting data
fn defaultConfidence() fingerprint.MultiFactorConfidence {
    return fingerprint.MultiFactorConfidence{
        .sample_size_confidence = 0.0,
        .accuracy_confidence = 0.0,
        .temporal_confidence = 0.0,
        .variance_confidence = 0.0,
        .overall_confidence = 0.0,
        .sample_count = 0,
        .recent_accuracy = 0.0,
        .time_since_last_ms = 0.0,
        .coefficient_of_variation = 0.0,
    };
}

// ============================================================================
// Testing and Validation
// ============================================================================

const testing = std.testing;

test "intelligent decision framework initialization" {
    const config = DecisionConfig{};
    const framework = IntelligentDecisionFramework.init(config);
    
    // Test basic initialization
    try testing.expect(framework.config.high_confidence_threshold == 0.8);
    try testing.expect(framework.fingerprint_registry == null);
}

test "scheduling strategy from confidence" {
    const very_low = fingerprint.MultiFactorConfidence.ConfidenceCategory.very_low;
    const low = fingerprint.MultiFactorConfidence.ConfidenceCategory.low;
    const medium = fingerprint.MultiFactorConfidence.ConfidenceCategory.medium;
    const high = fingerprint.MultiFactorConfidence.ConfidenceCategory.high;
    
    try testing.expect(SchedulingStrategy.fromConfidence(very_low) == .very_conservative);
    try testing.expect(SchedulingStrategy.fromConfidence(low) == .conservative);
    try testing.expect(SchedulingStrategy.fromConfidence(medium) == .balanced);
    try testing.expect(SchedulingStrategy.fromConfidence(high) == .aggressive);
}