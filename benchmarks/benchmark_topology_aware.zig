const std = @import("std");
const builtin = @import("builtin");
const topology = @import("zigpulse_v3_topology_aware.zig");
const lockfree = @import("zigpulse_v3_lockfree_deque.zig");

// Benchmark comparing topology-aware vs naive work distribution

const Task = struct {
    work_size: u64,
    data: []u8,
    result: u64 = 0,
    
    pub fn execute(self: *Task) void {
        // Simulate memory-intensive work
        var sum: u64 = 0;
        
        // Touch all memory to ensure cache effects
        for (0..self.work_size) |i| {
            const idx = i % self.data.len;
            sum +%= self.data[idx];
            self.data[idx] = @truncate(sum);
        }
        
        self.result = sum;
    }
};

const NaivePool = struct {
    workers: []Worker,
    next_worker: std.atomic.Value(u32),
    
    const Worker = struct {
        thread: std.Thread,
        queue: lockfree.WorkStealingDeque(*Task),
        running: *std.atomic.Value(bool),
    };
    
    pub fn init(allocator: std.mem.Allocator, num_workers: usize) !*NaivePool {
        const self = try allocator.create(NaivePool);
        self.* = .{
            .workers = try allocator.alloc(Worker, num_workers),
            .next_worker = std.atomic.Value(u32).init(0),
        };
        
        const running = try allocator.create(std.atomic.Value(bool));
        running.* = std.atomic.Value(bool).init(true);
        
        for (self.workers, 0..) |*worker, i| {
            worker.* = .{
                .thread = undefined,
                .queue = try lockfree.WorkStealingDeque(*Task).init(allocator, 1024),
                .running = running,
            };
            
            worker.thread = try std.Thread.spawn(.{}, workerLoop, .{worker});
            
            // No affinity - let OS schedule
            _ = i;
        }
        
        return self;
    }
    
    pub fn submit(self: *NaivePool, task: *Task) !void {
        // Round-robin distribution
        const worker_id = self.next_worker.fetchAdd(1, .monotonic) % self.workers.len;
        try self.workers[worker_id].queue.pushBottom(task);
    }
    
    fn workerLoop(worker: *Worker) void {
        while (worker.running.load(.acquire)) {
            if (worker.queue.popBottom()) |task| {
                task.execute();
            } else {
                std.time.sleep(1000);
            }
        }
    }
};

const TopologyAwarePool = struct {
    topo: topology.CpuTopology,
    workers: []Worker,
    numa_queues: []lockfree.WorkStealingDeque(*Task),
    
    const Worker = struct {
        thread: std.Thread,
        cpu_id: u32,
        numa_node: u32,
        queue: lockfree.WorkStealingDeque(*Task),
        pool: *TopologyAwarePool,
        running: *std.atomic.Value(bool),
    };
    
    pub fn init(allocator: std.mem.Allocator) !*TopologyAwarePool {
        const self = try allocator.create(TopologyAwarePool);
        self.* = .{
            .topo = try topology.detectTopology(allocator),
            .workers = undefined,
            .numa_queues = undefined,
        };
        
        // One worker per physical core
        self.workers = try allocator.alloc(Worker, self.topo.physical_cores);
        
        // NUMA queues
        self.numa_queues = try allocator.alloc(lockfree.WorkStealingDeque(*Task), self.topo.numa_nodes.len);
        for (self.numa_queues) |*q| {
            q.* = try lockfree.WorkStealingDeque(*Task).init(allocator, 1024);
        }
        
        const running = try allocator.create(std.atomic.Value(bool));
        running.* = std.atomic.Value(bool).init(true);
        
        // Create workers with affinity
        for (self.workers, 0..) |*worker, i| {
            const cpu_id = @as(u32, @intCast(i * 2)); // Map to physical cores (skip SMT)
            const numa_node = self.topo.logical_to_numa[@min(cpu_id, self.topo.logical_to_numa.len - 1)];
            
            worker.* = .{
                .thread = undefined,
                .cpu_id = cpu_id,
                .numa_node = numa_node,
                .queue = try lockfree.WorkStealingDeque(*Task).init(allocator, 256),
                .pool = self,
                .running = running,
            };
            
            worker.thread = try std.Thread.spawn(.{}, workerLoop, .{worker});
            
            // Set CPU affinity
            topology.setThreadAffinity(worker.thread, &[_]u32{cpu_id}) catch {
                // Ignore affinity errors
            };
        }
        
        return self;
    }
    
    pub fn submit(self: *TopologyAwarePool, task: *Task, hint: ?u32) !void {
        // Smart task placement based on data location
        const numa_node = hint orelse 0;
        
        // Find worker on preferred NUMA node
        for (self.workers) |*worker| {
            if (worker.numa_node == numa_node) {
                try worker.queue.pushBottom(task);
                return;
            }
        }
        
        // Fallback to NUMA queue
        try self.numa_queues[@min(numa_node, self.numa_queues.len - 1)].pushBottom(task);
    }
    
    fn workerLoop(worker: *Worker) void {
        while (worker.running.load(.acquire)) {
            // Local queue
            if (worker.queue.popBottom()) |task| {
                task.execute();
                continue;
            }
            
            // NUMA queue
            if (worker.pool.numa_queues[worker.numa_node].steal()) |task| {
                task.execute();
                continue;
            }
            
            // Topology-aware stealing
            const pool = worker.pool;
            
            // First: steal from same NUMA node
            for (pool.workers) |*victim| {
                if (victim.numa_node == worker.numa_node and victim != worker) {
                    if (victim.queue.steal()) |task| {
                        task.execute();
                        break;
                    }
                }
            }
            
            std.time.sleep(1000);
        }
    }
};

fn benchmarkPools(allocator: std.mem.Allocator, num_tasks: usize, task_size: usize) !void {
    std.debug.print("\n=== Benchmark: {} tasks, {}KB each ===\n", .{ num_tasks, task_size / 1024 });
    
    // Allocate task data (simulating NUMA allocation)
    const tasks = try allocator.alloc(Task, num_tasks);
    defer allocator.free(tasks);
    
    for (tasks) |*task| {
        task.* = .{
            .work_size = task_size,
            .data = try allocator.alloc(u8, 4096), // 4KB data per task
        };
        // Initialize with some data
        for (task.data) |*b| {
            b.* = 42;
        }
    }
    defer {
        for (tasks) |*task| {
            allocator.free(task.data);
        }
    }
    
    // Naive pool benchmark
    {
        const cpu_count = try std.Thread.getCpuCount();
        const pool = try NaivePool.init(allocator, cpu_count);
        
        var timer = std.time.Timer.start() catch unreachable;
        
        // Submit all tasks
        for (tasks) |*task| {
            try pool.submit(task);
        }
        
        // Wait for completion
        var completed = false;
        while (!completed) {
            completed = true;
            for (tasks) |*task| {
                if (task.result == 0) {
                    completed = false;
                    break;
                }
            }
            if (!completed) std.time.sleep(1_000_000); // 1ms
        }
        
        const elapsed = timer.read();
        std.debug.print("Naive pool: {}ms\n", .{elapsed / 1_000_000});
        
        // Cleanup
        pool.workers[0].running.store(false, .release);
        for (pool.workers) |*worker| {
            worker.thread.join();
            worker.queue.deinit();
        }
        allocator.destroy(pool.workers[0].running);
        allocator.free(pool.workers);
        allocator.destroy(pool);
    }
    
    // Reset results
    for (tasks) |*task| {
        task.result = 0;
    }
    
    // Topology-aware pool benchmark
    {
        const pool = try TopologyAwarePool.init(allocator);
        
        var timer = std.time.Timer.start() catch unreachable;
        
        // Submit all tasks with NUMA hints
        for (tasks, 0..) |*task, i| {
            const numa_hint = @as(u32, @intCast(i % pool.topo.numa_nodes.len));
            try pool.submit(task, numa_hint);
        }
        
        // Wait for completion
        var completed = false;
        while (!completed) {
            completed = true;
            for (tasks) |*task| {
                if (task.result == 0) {
                    completed = false;
                    break;
                }
            }
            if (!completed) std.time.sleep(1_000_000); // 1ms
        }
        
        const elapsed = timer.read();
        std.debug.print("Topology-aware pool: {}ms\n", .{elapsed / 1_000_000});
        
        // Print topology info
        std.debug.print("  Physical cores: {}, NUMA nodes: {}\n", .{
            pool.topo.physical_cores,
            pool.topo.numa_nodes.len,
        });
        
        // Cleanup
        pool.workers[0].running.store(false, .release);
        for (pool.workers) |*worker| {
            worker.thread.join();
            worker.queue.deinit();
        }
        for (pool.numa_queues) |*q| {
            q.deinit();
        }
        allocator.destroy(pool.workers[0].running);
        allocator.free(pool.workers);
        allocator.free(pool.numa_queues);
        pool.topo.deinit();
        allocator.destroy(pool);
    }
}

pub fn main() !void {
    std.debug.print("=== CPU Topology-Aware Benchmark ===\n", .{});
    std.debug.print("Build mode: {}\n", .{builtin.mode});
    
    const allocator = std.heap.page_allocator;
    
    // Different workload sizes
    try benchmarkPools(allocator, 1000, 1024);      // Small tasks
    try benchmarkPools(allocator, 100, 1024 * 100); // Medium tasks
    try benchmarkPools(allocator, 10, 1024 * 1000); // Large tasks
    
    std.debug.print("\n=== Benchmark Complete ===\n", .{});
}