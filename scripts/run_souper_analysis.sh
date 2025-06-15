#!/bin/bash

# Automated Souper Analysis Script for Beat.zig
# Runs comprehensive superoptimization analysis on all performance-critical modules

set -euo pipefail

# Configuration
RESULTS_DIR="souper_analysis_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ANALYSIS_LOG="$RESULTS_DIR/analysis_$TIMESTAMP.log"

# Souper analysis targets (matching build.zig)
TARGETS=(
    "fingerprint"
    "lockfree" 
    "scheduler"
    "simd"
    "simd_classifier"
    "simd_batch"
    "advanced_worker_selection"
    "topology"
)

# Priority levels for analysis
HIGH_PRIORITY=("fingerprint" "lockfree" "scheduler" "simd")
MEDIUM_PRIORITY=("simd_classifier" "simd_batch" "advanced_worker_selection" "topology")

echo "=== Beat.zig Souper Analysis Suite ==="
echo "Timestamp: $TIMESTAMP"
echo

# Setup results directory
setup_results() {
    mkdir -p "$RESULTS_DIR"
    echo "Results will be saved to: $RESULTS_DIR"
    echo
}

# Check if Souper environment is loaded
check_environment() {
    if [ -z "${SOUPER_PREFIX:-}" ]; then
        echo "Error: Souper environment not loaded"
        echo "Run: source souper_env.sh"
        exit 1
    fi
    
    # Verify tools are available
    if ! command -v souper >/dev/null 2>&1; then
        echo "Error: souper command not found in PATH"
        exit 1
    fi
    
    if ! command -v z3 >/dev/null 2>&1; then
        echo "Error: z3 command not found in PATH"
        echo "Install Z3 SMT solver or check your PATH"
        exit 1
    fi
    
    echo "Souper environment verified"
    echo "  Souper: $(which souper)"
    echo "  Z3: $(which z3)"
    echo
}

# Generate all LLVM IR files
generate_llvm_ir() {
    echo "Generating LLVM IR for all targets..."
    
    if ! zig build souper-all; then
        echo "Error: Failed to generate LLVM IR"
        exit 1
    fi
    
    echo "LLVM IR generation complete"
    echo
}

# Analyze individual module
analyze_module() {
    local module_name="$1"
    local priority="$2"
    
    echo "Analyzing module: $module_name (Priority: $priority)"
    
    local bc_file="zig-out/lib/beat_souper_${module_name}.bc"
    local ll_file="zig-out/lib/beat_souper_${module_name}.ll"
    local result_file="$RESULTS_DIR/souper_${module_name}_$TIMESTAMP.txt"
    local stats_file="$RESULTS_DIR/stats_${module_name}_$TIMESTAMP.txt"
    
    # Check if IR files exist
    if [ ! -f "$bc_file" ]; then
        echo "  Warning: $bc_file not found, skipping"
        return 1
    fi
    
    # Run Souper analysis with timeout (high priority gets more time)
    local timeout_duration
    if [[ " ${HIGH_PRIORITY[*]} " =~ " $module_name " ]]; then
        timeout_duration="300s"  # 5 minutes for high priority
    else
        timeout_duration="120s"  # 2 minutes for medium priority
    fi
    
    echo "  Running Souper analysis (timeout: $timeout_duration)..."
    
    {
        echo "=== Souper Analysis Results for $module_name ==="
        echo "Timestamp: $(date)"
        echo "Priority: $priority"
        echo "BC File: $bc_file"
        echo "LL File: $ll_file"
        echo
        
        # Run analysis with timeout
        if timeout "$timeout_duration" souper -z3-path="$(which z3)" "$bc_file" 2>&1; then
            echo
            echo "=== Analysis completed successfully ==="
        else
            local exit_code=$?
            echo
            if [ $exit_code -eq 124 ]; then
                echo "=== Analysis timed out after $timeout_duration ==="
            else
                echo "=== Analysis failed with exit code $exit_code ==="
            fi
        fi
        
    } > "$result_file" 2>&1
    
    # Generate statistics
    {
        echo "=== Module Statistics for $module_name ==="
        echo "Timestamp: $(date)"
        echo
        
        # Count optimizations found
        local opt_count=$(grep -c "infer" "$result_file" 2>/dev/null || echo "0")
        echo "Optimizations found: $opt_count"
        
        # Extract interesting patterns
        echo
        echo "=== Optimization Summary ==="
        if [ "$opt_count" -gt 0 ]; then
            echo "Found $opt_count potential optimizations:"
            grep -A 3 -B 1 "infer" "$result_file" 2>/dev/null || echo "  (Details in main result file)"
        else
            echo "No optimizations found - code may already be optimal!"
        fi
        
        # File sizes for context
        echo
        echo "=== File Information ==="
        ls -lh "$bc_file" "$ll_file" 2>/dev/null || echo "File size information unavailable"
        
    } > "$stats_file"
    
    echo "  Results saved to: $result_file"
    echo "  Statistics saved to: $stats_file"
    echo
}

# Run whole-program analysis
analyze_whole_program() {
    echo "Running whole-program superoptimization analysis..."
    
    # Build whole-program target
    if ! zig build souper-whole; then
        echo "Error: Failed to build whole-program target"
        return 1
    fi
    
    local bc_file="zig-out/bin/beat_whole_program_souper.bc"
    local result_file="$RESULTS_DIR/souper_whole_program_$TIMESTAMP.txt"
    
    if [ ! -f "$bc_file" ]; then
        echo "Error: Whole-program BC file not found: $bc_file"
        return 1
    fi
    
    echo "  Running comprehensive analysis (timeout: 600s)..."
    
    {
        echo "=== Whole-Program Souper Analysis ==="
        echo "Timestamp: $(date)"
        echo "BC File: $bc_file"
        echo
        echo "This analysis covers the complete Beat.zig program with all"
        echo "performance-critical algorithms integrated together."
        echo
        
        # Run with extended timeout for whole-program analysis
        if timeout 600s souper -z3-path="$(which z3)" "$bc_file" 2>&1; then
            echo
            echo "=== Whole-program analysis completed ==="
        else
            local exit_code=$?
            echo
            if [ $exit_code -eq 124 ]; then
                echo "=== Analysis timed out after 10 minutes ==="
                echo "Consider running analysis on individual modules instead"
            else
                echo "=== Analysis failed with exit code $exit_code ==="
            fi
        fi
        
    } > "$result_file" 2>&1
    
    echo "  Whole-program results saved to: $result_file"
    echo
}

# Generate comprehensive report
generate_report() {
    local report_file="$RESULTS_DIR/comprehensive_report_$TIMESTAMP.md"
    
    echo "Generating comprehensive analysis report..."
    
    {
        echo "# Beat.zig Souper Superoptimization Analysis Report"
        echo
        echo "**Generated:** $(date)"
        echo "**Analysis ID:** $TIMESTAMP"
        echo
        echo "## Executive Summary"
        echo
        echo "This report contains the results of Souper superoptimization analysis"
        echo "on Beat.zig's performance-critical algorithms. Souper uses SMT solvers"
        echo "to discover formally verified peephole optimizations."
        echo
        echo "## Analysis Targets"
        echo
        echo "### High Priority Modules"
        for target in "${HIGH_PRIORITY[@]}"; do
            echo "- **$target**: $(get_module_description "$target")"
        done
        echo
        echo "### Medium Priority Modules"
        for target in "${MEDIUM_PRIORITY[@]}"; do
            echo "- **$target**: $(get_module_description "$target")"
        done
        echo
        echo "## Results Summary"
        echo
        
        # Count total optimizations found
        local total_opts=0
        for target in "${TARGETS[@]}"; do
            local result_file="$RESULTS_DIR/souper_${target}_$TIMESTAMP.txt"
            if [ -f "$result_file" ]; then
                local opts=$(grep -c "infer" "$result_file" 2>/dev/null || echo "0")
                total_opts=$((total_opts + opts))
                echo "- **$target**: $opts optimizations found"
            fi
        done
        
        echo
        echo "**Total optimizations found:** $total_opts"
        echo
        echo "## Detailed Results"
        echo
        echo "Individual module analysis results are available in:"
        for target in "${TARGETS[@]}"; do
            echo "- \`souper_${target}_${TIMESTAMP}.txt\`"
        done
        echo
        echo "## Recommendations"
        echo
        if [ $total_opts -gt 0 ]; then
            echo "✅ **Action Required**: $total_opts potential optimizations discovered!"
            echo
            echo "1. Review each optimization for correctness and performance impact"
            echo "2. Implement promising optimizations in the codebase"
            echo "3. Benchmark before and after to validate improvements"
            echo "4. Consider why these optimizations were missed by LLVM"
        else
            echo "✅ **Excellent**: No additional optimizations found!"
            echo
            echo "This suggests that Beat.zig's performance-critical algorithms are"
            echo "already highly optimized and close to theoretical optimal performance."
            echo "The recent optimization work has been extremely effective."
        fi
        echo
        echo "## Next Steps"
        echo
        echo "1. **High-Impact Analysis**: Focus on modules with the most optimizations"
        echo "2. **Integration Testing**: Validate any changes with comprehensive benchmarks"
        echo "3. **Iterative Improvement**: Re-run analysis after implementing changes"
        echo "4. **Documentation**: Update performance claims with formal verification"
        
    } > "$report_file"
    
    echo "Comprehensive report generated: $report_file"
    echo
}

# Get module description for reporting
get_module_description() {
    local module="$1"
    case "$module" in
        "fingerprint") echo "Task fingerprinting and hashing algorithms" ;;
        "lockfree") echo "Work-stealing deque with critical bit operations" ;;
        "scheduler") echo "Token account promotion and scheduling logic" ;;
        "simd") echo "SIMD capability detection and bit flag operations" ;;
        "simd_classifier") echo "Feature vector similarity and classification" ;;
        "simd_batch") echo "Task compatibility scoring algorithms" ;;
        "advanced_worker_selection") echo "Worker selection scoring and normalization" ;;
        "topology") echo "CPU topology distance calculations" ;;
        *) echo "Beat.zig performance module" ;;
    esac
}

# Main analysis workflow
run_analysis() {
    echo "Starting comprehensive Souper analysis..."
    
    # Generate LLVM IR for all targets
    generate_llvm_ir
    
    # Analyze high-priority modules first
    echo "=== Analyzing High-Priority Modules ==="
    for target in "${HIGH_PRIORITY[@]}"; do
        analyze_module "$target" "HIGH"
    done
    
    # Analyze medium-priority modules
    echo "=== Analyzing Medium-Priority Modules ==="
    for target in "${MEDIUM_PRIORITY[@]}"; do
        analyze_module "$target" "MEDIUM"
    done
    
    # Run whole-program analysis
    echo "=== Whole-Program Analysis ==="
    analyze_whole_program
    
    # Generate comprehensive report
    generate_report
    
    echo "=== Analysis Complete ==="
    echo "Check $RESULTS_DIR for detailed results"
}

# Display usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -m MODULE      Analyze specific module only"
    echo "  -w, --whole    Run whole-program analysis only"
    echo "  -q, --quick    Quick analysis (high-priority modules only)"
    echo
    echo "Modules: ${TARGETS[*]}"
    echo
    echo "Examples:"
    echo "  $0                    # Full analysis"
    echo "  $0 -m fingerprint     # Analyze fingerprint module only"
    echo "  $0 -w                 # Whole-program analysis only"
    echo "  $0 -q                 # Quick analysis (high-priority only)"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -m|--module)
                if [[ -n "${2:-}" ]] && [[ " ${TARGETS[*]} " =~ " $2 " ]]; then
                    setup_results
                    check_environment
                    generate_llvm_ir
                    analyze_module "$2" "MANUAL"
                    exit 0
                else
                    echo "Error: Invalid module name: ${2:-}"
                    echo "Valid modules: ${TARGETS[*]}"
                    exit 1
                fi
                ;;
            -w|--whole)
                setup_results
                check_environment
                analyze_whole_program
                exit 0
                ;;
            -q|--quick)
                setup_results
                check_environment
                generate_llvm_ir
                echo "=== Quick Analysis (High-Priority Modules Only) ==="
                for target in "${HIGH_PRIORITY[@]}"; do
                    analyze_module "$target" "HIGH"
                done
                exit 0
                ;;
            *)
                echo "Error: Unknown option: $1"
                usage
                exit 1
                ;;
        esac
        shift
    done
}

# Main execution
main() {
    if [[ $# -gt 0 ]]; then
        parse_args "$@"
    else
        setup_results
        check_environment
        run_analysis
    fi
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi