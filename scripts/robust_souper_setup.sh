#!/bin/bash
set -euo pipefail

# Robust Souper Setup Script with Atomic Operations and Resume Capability
# This script creates checkpoints and can resume from any failed step

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
THIRD_PARTY_DIR="$PROJECT_ROOT/third_party/souper"
ARTIFACTS_DIR="$PROJECT_ROOT/artifacts/souper"
CHECKPOINT_DIR="$ARTIFACTS_DIR/checkpoints"

# Create directories
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$ARTIFACTS_DIR/logs"

# Logging functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$ARTIFACTS_DIR/logs/setup.log"
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "$ARTIFACTS_DIR/logs/setup.log"
    exit 1
}

# Checkpoint management
create_checkpoint() {
    local step="$1"
    touch "$CHECKPOINT_DIR/$step.done"
    log "Checkpoint created: $step"
}

has_checkpoint() {
    local step="$1"
    [[ -f "$CHECKPOINT_DIR/$step.done" ]]
}

# Check if we can resume from a specific step
check_resume() {
    local step="$1"
    if has_checkpoint "$step"; then
        log "Resuming: $step already completed, skipping..."
        return 0
    else
        log "Starting: $step"
        return 1
    fi
}

# Step 1: Setup directories and initial dependencies
setup_directories() {
    if check_resume "setup_directories"; then return 0; fi
    
    log "Setting up directory structure..."
    mkdir -p "$THIRD_PARTY_DIR"
    cd "$THIRD_PARTY_DIR"
    
    # Install system dependencies if needed
    if ! command -v cmake &> /dev/null; then
        log "Installing cmake..."
        sudo apt-get update && sudo apt-get install -y cmake ninja-build
    fi
    
    if ! command -v git &> /dev/null; then
        log "Installing git..."
        sudo apt-get install -y git
    fi
    
    # Install HIREDIS dependency
    if ! pkg-config --exists hiredis; then
        log "Installing hiredis dependency..."
        if command -v apt-get &> /dev/null && [[ $EUID -eq 0 ]]; then
            apt-get update && apt-get install -y libhiredis-dev
        elif command -v apt-get &> /dev/null; then
            log "Note: Skipping system package install (no sudo). Building from source..."
        fi
        
        # Build from source if system install failed or not available
        if ! pkg-config --exists hiredis; then
            log "Building hiredis from source..."
            if [[ -d "hiredis-src" ]]; then
                rm -rf hiredis-src
            fi
            git clone https://github.com/redis/hiredis.git hiredis-src
            cd hiredis-src
            make -j$(nproc)
            # Install to local directory if we can't use sudo
            if [[ $EUID -eq 0 ]]; then
                make install
                ldconfig
            else
                make install PREFIX="$THIRD_PARTY_DIR/hiredis-install"
                export PKG_CONFIG_PATH="${PKG_CONFIG_PATH:-}:$THIRD_PARTY_DIR/hiredis-install/lib/pkgconfig"
                export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$THIRD_PARTY_DIR/hiredis-install/lib"
            fi
            cd ..
        fi
    fi
    
    create_checkpoint "setup_directories"
}

# Step 2: Clone Souper and dependencies (atomic)
clone_repositories() {
    if check_resume "clone_repositories"; then return 0; fi
    
    log "Cloning Souper repository..."
    if [[ ! -d "souper-src" ]]; then
        git clone --depth 1 https://github.com/google/souper.git souper-src || error "Failed to clone Souper"
    fi
    
    cd souper-src
    
    # Clone submodules atomically
    log "Initializing submodules..."
    git submodule update --init --recursive --depth 1 || error "Failed to initialize submodules"
    
    create_checkpoint "clone_repositories"
}

# Step 3: Build Souper dependencies using build_deps.sh (atomic)
build_dependencies() {
    if check_resume "build_dependencies"; then return 0; fi
    
    log "Building Souper dependencies using build_deps.sh..."
    cd "$THIRD_PARTY_DIR/souper-src"
    
    # Check if dependencies already exist
    if [[ ! -d "third_party" ]]; then
        log "Running Souper's build_deps.sh script (this will take 60-120 minutes)..."
        
        # Run the official Souper dependency build script
        ./build_deps.sh || error "Souper build_deps.sh failed"
        
        log "Souper dependencies built successfully"
    else
        log "Souper dependencies already built"
    fi
    
    cd ..
    create_checkpoint "build_dependencies"
}

# Step 4: Build Souper (atomic)
build_souper() {
    if check_resume "build_souper"; then return 0; fi
    
    log "Building Souper..."
    cd "$THIRD_PARTY_DIR"
    
    if [[ ! -d "souper-install" ]]; then
        mkdir -p souper-build
        cd souper-build
        
        # Set environment for hiredis  
        if [[ -d "$THIRD_PARTY_DIR/hiredis-install" ]]; then
            export PKG_CONFIG_PATH="$THIRD_PARTY_DIR/hiredis-install/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
            HIREDIS_LIB="$THIRD_PARTY_DIR/hiredis-install/lib/libhiredis.so"
            HIREDIS_INCLUDE="$THIRD_PARTY_DIR/hiredis-install/include"
        else
            # Use system hiredis or the one built by build_deps.sh
            HIREDIS_LIB="-lhiredis"
            HIREDIS_INCLUDE=""
        fi
        
        # Configure Souper using its built dependencies
        cmake -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX="$THIRD_PARTY_DIR/souper-install" \
            ${HIREDIS_LIB:+-DHIREDIS_LIBRARY="$HIREDIS_LIB"} \
            ${HIREDIS_INCLUDE:+-DHIREDIS_HEADER="$HIREDIS_INCLUDE"} \
            ../souper-src || error "Souper configuration failed"
        
        ninja -j$(nproc) || error "Souper build failed"
        ninja install || error "Souper install failed"
        cd ..
    fi
    
    create_checkpoint "build_souper"
}

# Step 5: Create environment setup (atomic)
create_environment() {
    if check_resume "create_environment"; then return 0; fi
    
    log "Creating Souper environment setup..."
    
    cat > "$PROJECT_ROOT/souper_env.sh" << 'EOF'
#!/bin/bash
# Souper Environment Setup
# Source this file to set up the Souper environment: source souper_env.sh

export SOUPER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/third_party/souper"

# Add Souper tools to PATH
if [[ -d "$SOUPER_ROOT/souper-install/bin" ]]; then
    export PATH="$SOUPER_ROOT/souper-install/bin:$PATH"
fi

# Add LLVM tools to PATH (from Souper's third_party)
if [[ -d "$SOUPER_ROOT/souper-src/third_party/llvm-install/bin" ]]; then
    export PATH="$SOUPER_ROOT/souper-src/third_party/llvm-install/bin:$PATH"
fi

# Add library paths
if [[ -d "$SOUPER_ROOT/souper-src/third_party/z3-install/lib" ]]; then
    export LD_LIBRARY_PATH="$SOUPER_ROOT/souper-src/third_party/z3-install/lib:${LD_LIBRARY_PATH:-}"
fi

echo "Souper environment configured!"
echo "Available tools:"
echo "  - souper: $(which souper 2>/dev/null || echo 'Not found')"
echo "  - souper-check: $(which souper-check 2>/dev/null || echo 'Not found')"
echo "  - llvm-config: $(which llvm-config 2>/dev/null || echo 'Not found')"
echo "  - z3: $(which z3 2>/dev/null || echo 'Not found')"
EOF
    
    chmod +x "$PROJECT_ROOT/souper_env.sh"
    create_checkpoint "create_environment"
}

# Step 6: Validation (atomic)
validate_installation() {
    if check_resume "validate_installation"; then return 0; fi
    
    log "Validating Souper installation..."
    
    # Source environment
    source "$PROJECT_ROOT/souper_env.sh"
    
    # Test basic functionality
    if ! command -v souper &> /dev/null; then
        error "Souper executable not found in PATH"
    fi
    
    # Create a simple test
    cat > "$ARTIFACTS_DIR/test.opt" << 'EOF'
; Simple test optimization
%0:i32 = var
%1:i32 = add %0, 0
cand %1
EOF
    
    if souper "$ARTIFACTS_DIR/test.opt" &> /dev/null; then
        log "Souper validation successful!"
    else
        error "Souper validation failed"
    fi
    
    create_checkpoint "validate_installation"
}

# Main execution
main() {
    log "Starting robust Souper setup..."
    log "Project root: $PROJECT_ROOT"
    log "Third party dir: $THIRD_PARTY_DIR"
    log "Artifacts dir: $ARTIFACTS_DIR"
    
    # Execute all steps
    setup_directories
    clone_repositories
    build_dependencies
    build_souper
    create_environment
    validate_installation
    
    log "Souper setup completed successfully!"
    log "To use Souper, run: source souper_env.sh"
    log "Then you can use: souper, souper-check, llvm-config, etc."
}

# Help function
show_help() {
    cat << EOF
Robust Souper Setup Script

Usage: $0 [OPTIONS]

OPTIONS:
    --resume            Resume from last successful checkpoint
    --clean             Clean all checkpoints and start fresh
    --status            Show current setup status
    --help              Show this help message

EXAMPLES:
    $0                  # Start fresh setup
    $0 --resume         # Resume from last checkpoint
    $0 --clean          # Clean and start over
    $0 --status         # Check what's been completed

The script creates atomic checkpoints for each step and can resume from failures.
Logs are stored in artifacts/souper/logs/setup.log
EOF
}

# Handle command line arguments
case "${1:-}" in
    --resume)
        log "Resuming setup from checkpoints..."
        main
        ;;
    --clean)
        log "Cleaning checkpoints and starting fresh..."
        rm -rf "$CHECKPOINT_DIR"
        main
        ;;
    --status)
        echo "Souper Setup Status:"
        echo "==================="
        for step in setup_directories clone_repositories build_z3 build_llvm build_alive2 build_souper create_environment validate_installation; do
            if has_checkpoint "$step"; then
                echo "✓ $step"
            else
                echo "✗ $step"
            fi
        done
        ;;
    --help)
        show_help
        ;;
    "")
        main
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
esac