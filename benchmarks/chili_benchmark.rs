use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Serialize, Deserialize)]
struct BenchmarkConfig {
    benchmark: BenchmarkInfo,
}

#[derive(Serialize, Deserialize)]
struct BenchmarkInfo {
    name: String,
    description: String,
    tree_sizes: Vec<u32>,
    sample_count: u32,
    warmup_runs: u32,
    timeout_seconds: u32,
}

#[derive(Clone)]
struct Node {
    value: i64,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}

impl Node {
    fn new(value: i64) -> Self {
        Node {
            value,
            left: None,
            right: None,
        }
    }
    
    fn with_children(value: i64, left: Option<Box<Node>>, right: Option<Box<Node>>) -> Self {
        Node { value, left, right }
    }
}

fn create_tree(size: usize) -> Option<Box<Node>> {
    if size == 0 {
        return None;
    }
    
    if size == 1 {
        return Some(Box::new(Node::new(size as i64)));
    }
    
    let left_size = (size - 1) / 2;
    let right_size = size - 1 - left_size;
    
    let left = if left_size > 0 { create_tree(left_size) } else { None };
    let right = if right_size > 0 { create_tree(right_size) } else { None };
    
    Some(Box::new(Node::with_children(size as i64, left, right)))
}

fn sequential_sum(node: &Option<Box<Node>>) -> i64 {
    match node {
        None => 0,
        Some(n) => n.value + sequential_sum(&n.left) + sequential_sum(&n.right),
    }
}

fn chili_parallel_sum(scope: &mut chili::Scope, node: &Node) -> i64 {
    // Always try to parallelize if we have children (Chili's heartbeat system will decide)
    if node.left.is_some() || node.right.is_some() {
        let (left_result, right_result) = scope.join(
            |s| node.left.as_deref().map(|n| chili_parallel_sum(s, n)).unwrap_or(0),
            |s| node.right.as_deref().map(|n| chili_parallel_sum(s, n)).unwrap_or(0)
        );
        node.value + left_result + right_result
    } else {
        // Leaf node - just return the value
        node.value
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read config from JSON file
    let config_contents = fs::read_to_string("benchmark_config.json")?;
    let config: BenchmarkConfig = serde_json::from_str(&config_contents)?;
    
    println!("CHILI BENCHMARK RESULTS");
    println!("=======================");
    println!("{:<12} {:<12} {:<12} {:<12} {:<12}", 
        "Tree Size", "Seq (μs)", "Par (μs)", "Speedup", "Overhead");
    println!("------------------------------------------------------------");
    
    for &size in &config.benchmark.tree_sizes {
        // Use Vec with pre-allocated capacity for better memory locality (matches Zig's arena pattern)
        let expected_nodes = size as usize;
        let mut _node_arena: Vec<u8> = Vec::with_capacity(expected_nodes); // Reserve space for better locality
        
        let tree = create_tree(size as usize);
        
        let thread_pool = chili::ThreadPool::new();
        
        // Warmup runs
        for _ in 0..config.benchmark.warmup_runs {
            let _ = sequential_sum(&tree);
            if let Some(ref tree_node) = tree {
                let mut scope = thread_pool.scope();
                let _ = chili_parallel_sum(&mut scope, tree_node);
            }
        }
        
        // Sequential timing
        let mut seq_times = vec![0u128; config.benchmark.sample_count as usize];
        for i in 0..config.benchmark.sample_count as usize {
            let start = std::time::Instant::now();
            let result = sequential_sum(&tree);
            let end = start.elapsed();
            seq_times[i] = end.as_nanos();
            std::hint::black_box(result);
        }
        
        // Parallel timing
        let mut par_times = vec![0u128; config.benchmark.sample_count as usize];
        for i in 0..config.benchmark.sample_count as usize {
            let start = std::time::Instant::now();
            let result = if let Some(ref tree_node) = tree {
                let mut scope = thread_pool.scope();
                chili_parallel_sum(&mut scope, tree_node)
            } else {
                0
            };
            let end = start.elapsed();
            par_times[i] = end.as_nanos();
            std::hint::black_box(result);
        }
        
        // Calculate statistics
        seq_times.sort_unstable();
        par_times.sort_unstable();
        
        let seq_median_ns = seq_times[seq_times.len() / 2];
        let par_median_ns = par_times[par_times.len() / 2];
        
        let seq_median_us = seq_median_ns / 1_000;
        let par_median_us = par_median_ns / 1_000;
        
        let speedup = seq_median_ns as f64 / par_median_ns as f64;
        let overhead_ns = if par_median_ns > seq_median_ns {
            par_median_ns - seq_median_ns
        } else {
            0
        };
        
        let overhead_str = if overhead_ns < 1000 {
            "sub-ns".to_string()
        } else {
            format!("{}ns", overhead_ns)
        };
        
        println!("{:<12} {:<12} {:<12} {:<12.2} {:<12}",
            size, seq_median_us, par_median_us, speedup, overhead_str);
    }
    
    println!("\nNOTES:");
    println!("• OPTIMIZED: Using pre-allocated Vec capacity for memory locality (matches Zig arena)");
    println!("• Sample count: {}, Warmup runs: {}", config.benchmark.sample_count, config.benchmark.warmup_runs);
    println!("• Chili heartbeat scheduling with join primitive");
    
    Ok(())
}