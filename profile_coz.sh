#!/bin/bash

set -e

echo "=== ZigPulse COZ Profiling ==="
echo "=============================="
echo ""

# Check for COZ
if ! command -v coz &> /dev/null; then
    echo "ERROR: COZ profiler not installed"
    echo "Install with: sudo apt install coz-profiler (Ubuntu/Debian)"
    echo "Or from source: https://github.com/plasma-umass/coz"
    exit 1
fi

echo "COZ profiler found ✓"
echo ""

# Build options
BUILD_MODE=${1:-ReleaseSafe}
echo "Building ZigPulse with mode: $BUILD_MODE"

# Build benchmark with COZ support
echo "Building benchmark..."
zig build bench-coz -Dcoz=true 2>&1 | (grep -v "warning" || true)

# Find the actual benchmark binary
BENCHMARK_PATH=$(find .zig-cache -name "benchmark_coz" -type f -executable 2>/dev/null | head -1)
if [ -z "$BENCHMARK_PATH" ]; then
    echo "Build failed! Could not find benchmark_coz executable"
    exit 1
fi

echo "Build successful ✓"
echo ""

# Run COZ profiling
echo "=== Running COZ Profiler ==="
echo "This will take about 2 minutes..."
echo ""

# Disable COZ web server to avoid port conflicts
export COZ_DISABLE_WEB_SERVER=1

# Run with COZ - using simpler approach for better compatibility
coz run \
    --binary benchmark_coz \
    --source-scope src/ \
    --end-to-end \
    --output zigpulse_profile.coz \
    --- $BENCHMARK_PATH

echo ""
echo "=== Profiling Complete ==="
echo ""

# Check if profile was created
if [ -f zigpulse_profile.coz ]; then
    echo "Profile saved to: zigpulse_profile.coz"
    echo ""
    echo "To view results:"
    echo "  1. Upload to https://plasma-umass.org/coz/"
    echo "  2. Or run locally: coz plot zigpulse_profile.coz"
    echo ""
    
    # Try to extract some basic info
    echo "Progress point hits:"
    strings zigpulse_profile.coz 2>/dev/null | grep -E "zigpulse_" | sort | uniq -c | sort -nr || echo "Unable to extract progress points"
else
    echo "ERROR: No profile generated. Check for errors above."
fi

# Generate a simple report
REPORT_FILE="coz_profiling_report.md"
{
    echo "# ZigPulse COZ Profiling Report"
    echo ""
    echo "Generated: $(date)"
    echo "Build mode: $BUILD_MODE"
    echo ""
    echo "## Key Findings"
    echo ""
    echo "### Progress Points"
    echo "- **zigpulse_task_completed**: Measures overall throughput"
    echo "- **zigpulse_task_execution**: Measures task latency"
    echo "- **zigpulse_task_stolen**: Measures work-stealing activity"
    echo "- **zigpulse_worker_idle**: Measures load balancing"
    echo ""
    echo "## Analysis Instructions"
    echo ""
    echo "1. Upload zigpulse_profile.coz to https://plasma-umass.org/coz/"
    echo "2. Look for lines with positive virtual speedup"
    echo "3. Focus optimization efforts on code with highest impact"
    echo ""
    echo "## Common Bottlenecks"
    echo ""
    echo "- **Task Submission**: If submit() shows high impact"
    echo "- **Work Stealing**: If steal operations show high impact"  
    echo "- **Memory Management**: If allocation/free shows high impact"
    echo "- **Queue Contention**: If mutex operations show high impact"
    echo ""
    echo "## Next Steps"
    echo ""
    echo "1. Identify hotspots from COZ visualization"
    echo "2. Implement targeted optimizations"
    echo "3. Re-run profiling to verify improvements"
} > $REPORT_FILE

echo ""
echo "Report saved to: $REPORT_FILE"

# Clean up
# Keep the built binary for reuse

echo ""
echo "Done! Upload zigpulse_profile.coz to view results."