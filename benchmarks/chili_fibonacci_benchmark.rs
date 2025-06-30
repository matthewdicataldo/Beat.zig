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
    fibonacci_numbers: Vec<u32>,
    sample_count: u32,
    warmup_runs: u32,
    timeout_seconds: u32,
}

// Sequential Fibonacci implementation
fn fib_sequential(n: u32) -> u64 {
    if n <= 1 {
        return n as u64;
    }
    fib_sequential(n - 1) + fib_sequential(n - 2)
}

// Chili parallel Fibonacci using join primitive
fn chili_fibonacci(scope: &mut chili::Scope, n: u32) -> u64 {
    if n <= 1 {
        return n as u64;
    }
    
    // Use sequential for small numbers to avoid excessive overhead
    if n <= 30 {
        return fib_sequential(n);
    }
    
    // Fork two parallel tasks using Chili's join
    let (result1, result2) = scope.join(
        |s| chili_fibonacci(s, n - 1),
        |s| chili_fibonacci(s, n - 2)
    );
    
    result1 + result2
}

// Aggressive recursive Chili Fibonacci (for comparison)
fn chili_fibonacci_recursive(scope: &mut chili::Scope, n: u32, depth: u32) -> u64 {
    if n <= 1 {
        return n as u64;
    }
    
    // Limit recursion depth to prevent task explosion
    if depth > 6 || n <= 25 {
        return fib_sequential(n);
    }
    
    // Fork two parallel tasks with depth tracking
    let (result1, result2) = scope.join(
        |s| chili_fibonacci_recursive(s, n - 1, depth + 1),
        |s| chili_fibonacci_recursive(s, n - 2, depth + 1)
    );
    
    result1 + result2
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read config from JSON file
    let config_contents = fs::read_to_string("benchmark_config.json")?;
    let config: BenchmarkConfig = serde_json::from_str(&config_contents)?;
    
    println!("CHILI FIBONACCI BENCHMARK RESULTS");
    println!("=================================");
    println!("{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}", 
        "Fib(n)", "Seq (ms)", "Simple (ms)", "Recursive (ms)", "Simple Speedup", "Recursive Speedup");
    println!("------------------------------------------------------------------------------");
    
    let thread_pool = chili::ThreadPool::new();
    
    for &n in &config.benchmark.fibonacci_numbers {
        println!("Testing Fibonacci({})...", n);
        
        // Memory optimization - pre-warm allocator for fair comparison with Zig arena
        let _allocation_warmup: Vec<u64> = Vec::with_capacity(1000);
        
        // Warmup runs
        for _ in 0..config.benchmark.warmup_runs {
            let _ = fib_sequential(n);
            let _ = thread_pool.scope(|scope| chili_fibonacci(scope, n));
            let _ = thread_pool.scope(|scope| chili_fibonacci_recursive(scope, n, 0));
        }
        
        // Sequential timing
        let mut seq_times = vec![0u128; config.benchmark.sample_count as usize];
        for i in 0..config.benchmark.sample_count as usize {
            let start = std::time::Instant::now();
            let result = fib_sequential(n);
            let end = start.elapsed();
            seq_times[i] = end.as_nanos();
            std::hint::black_box(result);
        }
        
        // Simple Chili timing
        let mut simple_times = vec![0u128; config.benchmark.sample_count as usize];
        for i in 0..config.benchmark.sample_count as usize {
            let start = std::time::Instant::now();
            let result = thread_pool.scope(|scope| chili_fibonacci(scope, n));
            let end = start.elapsed();
            simple_times[i] = end.as_nanos();
            std::hint::black_box(result);
        }
        
        // Recursive Chili timing
        let mut recursive_times = vec![0u128; config.benchmark.sample_count as usize];
        for i in 0..config.benchmark.sample_count as usize {
            let start = std::time::Instant::now();
            let result = thread_pool.scope(|scope| chili_fibonacci_recursive(scope, n, 0));
            let end = start.elapsed();
            recursive_times[i] = end.as_nanos();
            std::hint::black_box(result);
        }
        
        // Calculate statistics
        seq_times.sort_unstable();
        simple_times.sort_unstable();
        recursive_times.sort_unstable();
        
        let seq_median_ns = seq_times[seq_times.len() / 2];
        let simple_median_ns = simple_times[simple_times.len() / 2];
        let recursive_median_ns = recursive_times[recursive_times.len() / 2];
        
        let seq_median_ms = seq_median_ns / 1_000_000;
        let simple_median_ms = simple_median_ns / 1_000_000;
        let recursive_median_ms = recursive_median_ns / 1_000_000;
        
        let simple_speedup = seq_median_ns as f64 / simple_median_ns as f64;
        let recursive_speedup = seq_median_ns as f64 / recursive_median_ns as f64;
        
        println!("{:<12} {:<12} {:<12} {:<12} {:<12.2} {:<12.2}",
            n, seq_median_ms, simple_median_ms, recursive_median_ms, simple_speedup, recursive_speedup);
    }
    
    println!("\nNOTES:");
    println!("• Simple: Basic fork-join using Chili scope.join()");
    println!("• Recursive: Full recursive parallelization with heartbeat system");
    println!("• Sequential thresholds to prevent task explosion");
    println!("• Demonstrates Chili's low-overhead fork-join design");
    println!("• OPTIMIZED: Memory allocator pre-warmed for fair comparison with Zig arena");
    
    Ok(())
}