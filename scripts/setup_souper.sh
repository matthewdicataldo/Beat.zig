#!/bin/bash

# Souper Toolchain Setup Script for Beat.zig Integration
# This script implements the LLVM version compatibility strategy from sopuer!.md

set -euo pipefail

# Configuration
SOUPER_REPO="https://github.com/google/souper.git"
SOUPER_DIR="$PWD/third_party/souper"
LLVM_INSTALL_PREFIX="$PWD/third_party/shared-llvm"
BUILD_TYPE="Release"

echo "=== Beat.zig Souper Toolchain Setup ==="
echo "Following the compatibility strategy from sopuer!.md"
echo

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    
    # Check for required tools
    for tool in cmake git ninja make; do
        if ! command -v $tool >/dev/null 2>&1; then
            echo "Error: $tool is required but not installed"
            exit 1
        fi
    done
    
    # Check for zstd development headers
    if ! pkg-config --exists libzstd 2>/dev/null; then
        echo "Warning: libzstd-dev may not be installed"
        echo "On Ubuntu/Debian: sudo apt-get install libzstd-dev"
        echo "On macOS: brew install zstd"
    fi
    
    echo "Prerequisites check passed"
    echo
}

# Clone Souper repository
clone_souper() {
    echo "Cloning Souper repository..."
    
    if [ -d "$SOUPER_DIR" ]; then
        echo "Souper directory already exists, pulling latest changes..."
        cd "$SOUPER_DIR"
        git pull
        cd - >/dev/null
    else
        mkdir -p "$(dirname "$SOUPER_DIR")"
        git clone "$SOUPER_REPO" "$SOUPER_DIR"
    fi
    
    echo "Souper repository ready"
    echo
}

# Extract LLVM version from Souper's build_deps.sh
get_llvm_version() {
    echo "Extracting LLVM version from Souper's build_deps.sh..."
    
    cd "$SOUPER_DIR"
    if [ ! -f "build_deps.sh" ]; then
        echo "Error: build_deps.sh not found in Souper repository"
        exit 1
    fi
    
    # Extract LLVM commit/version
    LLVM_COMMIT=$(grep -o 'llvm_commit=.*' build_deps.sh | cut -d'=' -f2 | tr -d '"' || true)
    LLVM_BRANCH=$(grep -o 'llvm_branch=.*' build_deps.sh | cut -d'=' -f2 | tr -d '"' || true)
    
    echo "Found LLVM configuration:"
    echo "  Commit: $LLVM_COMMIT"
    echo "  Branch: $LLVM_BRANCH"
    
    cd - >/dev/null
    echo
}

# Build shared LLVM installation
build_shared_llvm() {
    echo "Building shared LLVM installation..."
    echo "This will take 30-60 minutes depending on your system"
    
    # Run Souper's dependency script to get the exact LLVM version
    cd "$SOUPER_DIR"
    echo "Running Souper's build_deps.sh..."
    ./build_deps.sh $BUILD_TYPE
    
    # The dependencies are now in third_party/ subdirectory
    SOUPER_LLVM_BUILD="$SOUPER_DIR/third_party/llvm/$BUILD_TYPE"
    
    if [ ! -d "$SOUPER_LLVM_BUILD" ]; then
        echo "Error: LLVM build directory not found: $SOUPER_LLVM_BUILD"
        exit 1
    fi
    
    # Install to shared location
    echo "Installing LLVM to shared location: $LLVM_INSTALL_PREFIX"
    cd "$SOUPER_LLVM_BUILD"
    make install DESTDIR="" CMAKE_INSTALL_PREFIX="$LLVM_INSTALL_PREFIX"
    
    cd - >/dev/null
    echo "Shared LLVM installation complete"
    echo
}

# Build Souper against shared LLVM
build_souper() {
    echo "Building Souper against shared LLVM..."
    
    cd "$SOUPER_DIR"
    mkdir -p build
    cd build
    
    # Configure Souper to use the shared LLVM
    cmake \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DLLVM_DIR="$LLVM_INSTALL_PREFIX/lib/cmake/llvm" \
        -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_PREFIX" \
        ..
    
    # Build Souper
    make -j$(nproc)
    
    # Install Souper tools to shared location
    make install
    
    cd - >/dev/null
    echo "Souper build complete"
    echo
}

# Verify installation
verify_installation() {
    echo "Verifying Souper installation..."
    
    # Check if executables exist
    SOUPER_BIN="$LLVM_INSTALL_PREFIX/bin/souper"
    Z3_BIN="$LLVM_INSTALL_PREFIX/bin/z3"
    
    if [ ! -f "$SOUPER_BIN" ]; then
        echo "Error: souper executable not found at $SOUPER_BIN"
        exit 1
    fi
    
    if [ ! -f "$Z3_BIN" ]; then
        echo "Warning: z3 executable not found at $Z3_BIN"
        echo "You may need to install Z3 separately"
    fi
    
    # Run Souper's test suite
    echo "Running Souper test suite..."
    cd "$SOUPER_DIR/build"
    if make check; then
        echo "Souper test suite passed!"
    else
        echo "Warning: Some Souper tests failed, but installation may still be usable"
    fi
    
    cd - >/dev/null
    echo
}

# Generate environment setup script
generate_env_script() {
    echo "Generating environment setup script..."
    
    ENV_SCRIPT="$PWD/souper_env.sh"
    cat > "$ENV_SCRIPT" << EOF
#!/bin/bash
# Souper Environment Setup for Beat.zig
# Source this file to add Souper tools to your PATH

export SOUPER_PREFIX="$LLVM_INSTALL_PREFIX"
export PATH="\$SOUPER_PREFIX/bin:\$PATH"
export LD_LIBRARY_PATH="\$SOUPER_PREFIX/lib:\$LD_LIBRARY_PATH"

# Souper-specific aliases and functions
alias souper-analyze='souper -z3-path=\$(which z3)'
alias souper-opt='opt -load \$SOUPER_PREFIX/lib/libsouperPass.so -souper -z3-path=\$(which z3)'

# Function to run Souper on Beat.zig LLVM IR
analyze_beat_module() {
    local module_name=\$1
    local bc_file="zig-out/lib/beat_souper_\${module_name}.bc"
    
    if [ ! -f "\$bc_file" ]; then
        echo "Error: \$bc_file not found. Run 'zig build souper-\$module_name' first."
        return 1
    fi
    
    echo "Analyzing Beat.zig module: \$module_name"
    souper-analyze "\$bc_file" > "souper_results_\${module_name}.txt"
    echo "Results saved to souper_results_\${module_name}.txt"
}

# Function to run whole-program analysis
analyze_beat_whole() {
    local bc_file="zig-out/bin/beat_whole_program_souper.bc"
    
    if [ ! -f "\$bc_file" ]; then
        echo "Error: \$bc_file not found. Run 'zig build souper-whole' first."
        return 1
    fi
    
    echo "Running whole-program Souper analysis on Beat.zig..."
    souper-analyze "\$bc_file" > "souper_results_whole_program.txt"
    echo "Results saved to souper_results_whole_program.txt"
}

echo "Souper environment loaded!"
echo "Available commands:"
echo "  souper-analyze <file.bc>    - Run Souper analysis"
echo "  souper-opt <file.bc>        - Run Souper as LLVM pass"
echo "  analyze_beat_module <name>  - Analyze specific Beat.zig module"
echo "  analyze_beat_whole          - Whole-program analysis"
EOF

    chmod +x "$ENV_SCRIPT"
    echo "Environment script created: $ENV_SCRIPT"
    echo "Run 'source $ENV_SCRIPT' to set up your environment"
    echo
}

# Display final instructions
display_instructions() {
    echo "=== Souper Setup Complete ==="
    echo
    echo "Next steps:"
    echo "1. Source the environment: source souper_env.sh"
    echo "2. Generate LLVM IR: zig build souper-all"
    echo "3. Analyze specific modules: analyze_beat_module fingerprint"
    echo "4. Run whole-program analysis: analyze_beat_whole"
    echo
    echo "High-priority analysis targets:"
    echo "  - fingerprint: Task hashing algorithms"
    echo "  - lockfree: Work-stealing deque bit operations"
    echo "  - scheduler: Token accounting logic"
    echo "  - simd: Capability detection algorithms"
    echo
    echo "For more information, see sopuer!.md"
}

# Main execution
main() {
    check_prerequisites
    clone_souper
    get_llvm_version
    build_shared_llvm
    build_souper
    verify_installation
    generate_env_script
    display_instructions
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi