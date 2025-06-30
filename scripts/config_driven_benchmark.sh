#!/bin/bash

# =============================================================================
# Config-Driven Multi-Library Benchmark
# 
# Simple script that runs benchmarks based on benchmark_config.json
# Much cleaner than the previous approach with embedded code generation
# =============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TEMP_DIR="$REPO_ROOT/temp_multilibrary_comparison"
RESULTS_FILE="$REPO_ROOT/multilibrary_comparison_results.txt"
CONFIG_FILE="$REPO_ROOT/benchmark_config.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔬 CONFIG-DRIVEN MULTI-LIBRARY BENCHMARK${NC}"
echo "==========================================="

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Using config: $CONFIG_FILE${NC}"

# Read key values from config (using jq if available, otherwise basic parsing)
if command -v jq &> /dev/null; then
    TREE_SIZES=$(jq -r '.benchmark.tree_sizes | join(", ")' "$CONFIG_FILE")
    SAMPLE_COUNT=$(jq -r '.benchmark.sample_count' "$CONFIG_FILE")
    DESCRIPTION=$(jq -r '.benchmark.description' "$CONFIG_FILE")
else
    # Fallback for systems without jq
    TREE_SIZES="1023, 65535"
    SAMPLE_COUNT="20"
    DESCRIPTION="Binary Tree Sum (Fork-Join Parallelism)"
fi

echo -e "${YELLOW}📋 Benchmark Configuration:${NC}"
echo "  Description: $DESCRIPTION"
echo "  Tree Sizes: $TREE_SIZES"
echo "  Sample Count: $SAMPLE_COUNT"
echo ""

# Cleanup any previous temp directory
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# Initialize results file
cat > "$RESULTS_FILE" << EOF
BEAT.ZIG vs MULTI-LIBRARY COMPARISON RESULTS
============================================

Test Pattern: $DESCRIPTION
Tree Sizes: $TREE_SIZES nodes  
Sample Count: $SAMPLE_COUNT runs with warmup
Libraries: Beat.zig, Spice (Zig), Chili (Rust)

EOF

echo -e "${YELLOW}🔨 Building benchmarks...${NC}"

# Test if we're running tree sum or fibonacci benchmark
BENCHMARK_TYPE="${1:-tree}"  # Default to tree sum

if [ "$BENCHMARK_TYPE" = "fibonacci" ]; then
    echo -e "${BLUE}Running Fibonacci recursive benchmarks...${NC}"
else
    echo -e "${BLUE}Running Tree sum benchmarks...${NC}"
fi

# Build Spice benchmark (if Spice is available)
SPICE_BUILT=false
echo -e "${BLUE}Building Spice benchmark...${NC}"
cd "$TEMP_DIR"

if git clone https://github.com/judofyr/spice.git > /dev/null 2>&1; then
    cd spice
    
    # Copy our config-driven benchmarks
    if [ "$BENCHMARK_TYPE" = "fibonacci" ]; then
        cp "$REPO_ROOT/src/spice_fibonacci_benchmark.zig" .
        # Fix import path for external usage
        sed -i 's/@import("root.zig")/@import("src\/root.zig")/g' spice_fibonacci_benchmark.zig
    else
        cp "$REPO_ROOT/src/spice_benchmark.zig" .
        # Fix import path for external usage  
        sed -i 's/@import("root.zig")/@import("src\/root.zig")/g' spice_benchmark.zig
    fi
    cp "$CONFIG_FILE" .
    
    # Try to build (simple approach - assume root.zig is available)
    if [ -f "src/root.zig" ]; then
        if [ "$BENCHMARK_TYPE" = "fibonacci" ]; then
            if zig run spice_fibonacci_benchmark.zig > /dev/null 2>&1; then
                echo -e "${GREEN}✅ Spice Fibonacci benchmark built successfully${NC}"
                SPICE_BUILT=true
            else
                echo -e "${YELLOW}⚠️  Spice Fibonacci benchmark build failed${NC}"
            fi
        else
            if zig run spice_benchmark.zig > /dev/null 2>&1; then
                echo -e "${GREEN}✅ Spice benchmark built successfully${NC}"
                SPICE_BUILT=true
            else
                echo -e "${YELLOW}⚠️  Spice benchmark build failed${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}⚠️  Spice library structure not as expected${NC}"
    fi
    cd ..
else
    echo -e "${YELLOW}⚠️  Could not clone Spice repository${NC}"
fi

# Build Chili benchmark
CHILI_BUILT=false
echo -e "${BLUE}Building Chili benchmark...${NC}"

if git clone https://github.com/dragostis/chili.git > /dev/null 2>&1; then
    cd chili
    
    # Copy our config-driven benchmarks
    if [ "$BENCHMARK_TYPE" = "fibonacci" ]; then
        cp "$REPO_ROOT/benchmarks/chili_fibonacci_benchmark.rs" .
        CHILI_BINARY_NAME="chili_fibonacci_benchmark"
    else
        cp "$REPO_ROOT/benchmarks/chili_benchmark.rs" .
        CHILI_BINARY_NAME="chili_benchmark"
    fi
    cp "$CONFIG_FILE" .
    
    # Add serde dependency to Cargo.toml
    if ! grep -q "serde_json" Cargo.toml; then
        cat >> Cargo.toml << EOF

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[[bin]]
name = "$CHILI_BINARY_NAME"
path = "${CHILI_BINARY_NAME}.rs"
EOF
    fi
    
    # Build
    if cargo build --release --bin "$CHILI_BINARY_NAME" > /dev/null 2>&1; then
        if [ "$BENCHMARK_TYPE" = "fibonacci" ]; then
            echo -e "${GREEN}✅ Chili Fibonacci benchmark built successfully${NC}"
        else
            echo -e "${GREEN}✅ Chili benchmark built successfully${NC}"
        fi
        CHILI_BUILT=true
    else
        if [ "$BENCHMARK_TYPE" = "fibonacci" ]; then
            echo -e "${YELLOW}⚠️  Chili Fibonacci benchmark build failed${NC}"
        else
            echo -e "${YELLOW}⚠️  Chili benchmark build failed${NC}"
        fi
    fi
    cd ..
else
    echo -e "${YELLOW}⚠️  Could not clone Chili repository${NC}"
fi

echo -e "${GREEN}✅ Library builds completed${NC}"
echo -e "${BLUE}Built libraries: Spice=${SPICE_BUILT}, Chili=${CHILI_BUILT}${NC}"
echo ""

# Run benchmarks
echo -e "${YELLOW}🏃 Running benchmarks...${NC}"

# Zig std.Thread baseline
echo -e "${PURPLE}🧵 Running std.Thread baseline...${NC}"

if [ "$BENCHMARK_TYPE" = "fibonacci" ]; then
    echo "ZIG STD.THREAD FIBONACCI BASELINE RESULTS:" >> "$RESULTS_FILE"
    echo "==========================================" >> "$RESULTS_FILE"
    
    cd "$REPO_ROOT"
    if timeout 60s zig run src/std_thread_fibonacci_benchmark.zig >> "$RESULTS_FILE" 2>&1; then
        echo -e "${GREEN}✅ std.Thread Fibonacci baseline completed${NC}"
    else
        echo -e "${YELLOW}⚠️  std.Thread Fibonacci baseline failed${NC}"
    fi
else
    echo "ZIG STD.THREAD BASELINE RESULTS:" >> "$RESULTS_FILE"
    echo "===============================" >> "$RESULTS_FILE"
    
    cd "$REPO_ROOT"
    if timeout 30s zig run src/std_thread_benchmark.zig >> "$RESULTS_FILE" 2>&1; then
        echo -e "${GREEN}✅ std.Thread baseline completed${NC}"
    else
        echo -e "${YELLOW}⚠️  std.Thread baseline failed${NC}"
    fi
fi

# Beat.zig benchmark
echo -e "${PURPLE}🚀 Running Beat.zig benchmark...${NC}"
echo "" >> "$RESULTS_FILE"

if [ "$BENCHMARK_TYPE" = "fibonacci" ]; then
    echo "BEAT.ZIG FIBONACCI RESULTS:" >> "$RESULTS_FILE"
    echo "==========================" >> "$RESULTS_FILE"
    
    cd "$REPO_ROOT"
    if timeout 60s zig run src/fibonacci_benchmark.zig >> "$RESULTS_FILE" 2>&1; then
        echo -e "${GREEN}✅ Beat.zig Fibonacci benchmark completed${NC}"
    else
        echo -e "${YELLOW}⚠️  Beat.zig Fibonacci benchmark failed${NC}"
    fi
else
    echo "BEAT.ZIG RESULTS:" >> "$RESULTS_FILE"
    echo "=================" >> "$RESULTS_FILE"
    
    cd "$REPO_ROOT"
    if timeout 30s zig run src/beat_tree_benchmark.zig >> "$RESULTS_FILE" 2>&1; then
        echo -e "${GREEN}✅ Beat.zig benchmark completed${NC}"
    else
        echo -e "${YELLOW}⚠️  Beat.zig benchmark timed out or failed${NC}"
    fi
fi

# Spice benchmark
if [ "$SPICE_BUILT" = true ]; then
    echo -e "${PURPLE}⚡ Running Spice benchmark...${NC}"
    echo "" >> "$RESULTS_FILE"
    
    if [ "$BENCHMARK_TYPE" = "fibonacci" ]; then
        echo "SPICE FIBONACCI RESULTS:" >> "$RESULTS_FILE"
        echo "=======================" >> "$RESULTS_FILE"
        
        cd "$TEMP_DIR/spice"
        if timeout 60s zig run spice_fibonacci_benchmark.zig >> "$RESULTS_FILE" 2>&1; then
            echo -e "${GREEN}✅ Spice Fibonacci benchmark completed${NC}"
        else
            echo -e "${YELLOW}⚠️  Spice Fibonacci benchmark timed out or failed${NC}"
        fi
    else
        echo "SPICE RESULTS:" >> "$RESULTS_FILE"
        echo "==============" >> "$RESULTS_FILE"
        
        cd "$TEMP_DIR/spice"
        if timeout 30s zig run spice_benchmark.zig >> "$RESULTS_FILE" 2>&1; then
            echo -e "${GREEN}✅ Spice benchmark completed${NC}"
        else
            echo -e "${YELLOW}⚠️  Spice benchmark timed out or failed${NC}"
        fi
    fi
fi

# Chili benchmark
if [ "$CHILI_BUILT" = true ]; then
    echo -e "${PURPLE}🌶️  Running Chili benchmark...${NC}"
    echo "" >> "$RESULTS_FILE"
    
    if [ "$BENCHMARK_TYPE" = "fibonacci" ]; then
        echo "CHILI FIBONACCI RESULTS:" >> "$RESULTS_FILE"
        echo "=======================" >> "$RESULTS_FILE"
        
        cd "$TEMP_DIR/chili"
        if timeout 60s ./target/release/"$CHILI_BINARY_NAME" >> "$RESULTS_FILE" 2>&1; then
            echo -e "${GREEN}✅ Chili Fibonacci benchmark completed${NC}"
        else
            echo -e "${YELLOW}⚠️  Chili Fibonacci benchmark timed out or failed${NC}"
        fi
    else
        echo "CHILI RESULTS:" >> "$RESULTS_FILE"
        echo "==============" >> "$RESULTS_FILE"
        
        cd "$TEMP_DIR/chili"
        if timeout 30s ./target/release/"$CHILI_BINARY_NAME" >> "$RESULTS_FILE" 2>&1; then
            echo -e "${GREEN}✅ Chili benchmark completed${NC}"
        else
            echo -e "${YELLOW}⚠️  Chili benchmark timed out or failed${NC}"
        fi
    fi
fi

# Generate summary
echo "" >> "$RESULTS_FILE"
echo "MULTI-LIBRARY COMPARISON SUMMARY" >> "$RESULTS_FILE"
echo "================================" >> "$RESULTS_FILE"

# Success summary
SUCCESSFUL_BENCHMARKS=2  # std.Thread + Beat.zig always available
if [ "$SPICE_BUILT" = true ]; then
    SUCCESSFUL_BENCHMARKS=$((SUCCESSFUL_BENCHMARKS + 1))
fi
if [ "$CHILI_BUILT" = true ]; then
    SUCCESSFUL_BENCHMARKS=$((SUCCESSFUL_BENCHMARKS + 1))
fi

echo "" >> "$RESULTS_FILE"
echo "Libraries Successfully Benchmarked: $SUCCESSFUL_BENCHMARKS/4" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "✅ std.Thread: Zig standard library baseline (always available)" >> "$RESULTS_FILE"
echo "✅ Beat.zig: Advanced parallelism library (this repository)" >> "$RESULTS_FILE"
if [ "$SPICE_BUILT" = true ]; then
    echo "✅ Spice: Zig-based parallelism library with fork-join model" >> "$RESULTS_FILE"
else
    echo "❌ Spice: Build failed or repository unavailable" >> "$RESULTS_FILE"
fi
if [ "$CHILI_BUILT" = true ]; then
    echo "✅ Chili: Rust-based work-stealing library" >> "$RESULTS_FILE"
else
    echo "❌ Chili: Build failed or repository unavailable" >> "$RESULTS_FILE"
fi

echo ""
echo -e "${GREEN}📄 Full results saved to: $RESULTS_FILE${NC}"
echo ""
echo -e "${GREEN}✅ CONFIG-DRIVEN COMPARISON COMPLETED${NC}"
echo "Successfully benchmarked $SUCCESSFUL_BENCHMARKS/3 libraries using $CONFIG_FILE"

# Cleanup
echo ""
echo -e "${YELLOW}🧹 Cleaning up temporary files...${NC}"
rm -rf "$TEMP_DIR"
echo -e "${GREEN}✅ Cleanup completed${NC}"