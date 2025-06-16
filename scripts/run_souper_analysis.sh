#!/bin/bash

# Comprehensive Souper Analysis Script for Beat.zig
set -euo pipefail

PROJECT_ROOT="/mnt/c/Users/matth/Documents/Repos/Beat.zig"
LLVM_IR_DIR="$PROJECT_ROOT/artifacts/llvm_ir"
SOUPER_RESULTS="$PROJECT_ROOT/artifacts/souper/results"

mkdir -p "$SOUPER_RESULTS"

echo "======================================="
echo "Beat.zig Souper Superoptimization Analysis"
echo "======================================="

# Create a comprehensive Beat.zig fingerprint test
cat > "$LLVM_IR_DIR/beat_fingerprint.c" << 'EOF'
#include <stdint.h>

uint64_t computeFingerprintSimilarity(uint64_t hash1, uint64_t hash2) {
    uint64_t xor_result = hash1 ^ hash2;
    uint64_t temp = xor_result;
    uint32_t count = 0;
    while (temp > 0 && count < 64) {
        if (temp & 1) break;
        temp >>= 1;
        count += 1;
    }
    uint64_t similarity = (64 - count) * 100 / 64;
    similarity = similarity + 0;  // Redundant operation
    similarity = similarity * 1;  // Redundant operation
    return similarity;
}

uint32_t optimizeBitManipulation(uint32_t x) {
    x = x + 0;           // Should be optimized away
    x = x * 1;           // Should be optimized away
    x = x | 0;           // Should be optimized away
    x = x & 0xFFFFFFFF;  // Should be optimized away for 32-bit
    x = x ^ x;           // Always 0
    return x;
}
EOF

echo "Analyzing Beat.zig fingerprint algorithms..."

# Generate bitcode
docker run --rm -v "$LLVM_IR_DIR":/work slumps/souper:latest bash -c \
    "cd /work && clang -emit-llvm -c -O1 beat_fingerprint.c -o beat_fingerprint.bc"

# Run Souper analysis with timeout
echo "Running Souper instruction inference..."
timeout 120s docker run --rm -v "$LLVM_IR_DIR":/work slumps/souper:latest \
    /usr/src/souper-build/souper --souper-infer-inst /work/beat_fingerprint.bc > "$SOUPER_RESULTS/fingerprint_infer.txt" 2>&1 || {
    echo "Warning: Souper inference timed out or failed"
    echo "timeout" > "$SOUPER_RESULTS/fingerprint_infer.txt"
}

# Run optimization-specific checks
echo "Running optimization checks..."

timeout 30s docker run --rm -v "$LLVM_IR_DIR":/work slumps/souper:latest \
    /usr/src/souper-build/souper --infer-known-bits /work/beat_fingerprint.bc > "$SOUPER_RESULTS/fingerprint_known_bits.txt" 2>&1 || true

timeout 30s docker run --rm -v "$LLVM_IR_DIR":/work slumps/souper:latest \
    /usr/src/souper-build/souper --infer-demanded-bits /work/beat_fingerprint.bc > "$SOUPER_RESULTS/fingerprint_demanded_bits.txt" 2>&1 || true

echo ""
echo "======================================="
echo "Analysis completed! Results in: $SOUPER_RESULTS/"
echo "Generated files:"
ls -la "$SOUPER_RESULTS/" 2>/dev/null || echo "No files generated"
echo "======================================="