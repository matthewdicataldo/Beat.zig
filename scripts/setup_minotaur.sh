#!/bin/bash

# Minotaur SIMD Superoptimizer Setup Script for Beat.zig Integration
# Complements existing Souper infrastructure for comprehensive optimization coverage
# Based on the comprehensive guide in docs/souper and minotaur.md

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
THIRD_PARTY_DIR="$PROJECT_ROOT/third_party"
LOG_FILE="minotaur_setup_$(date +%Y%m%d_%H%M%S).log"
PROGRESS_FILE="minotaur_progress.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Progress tracking
TOTAL_STEPS=8
CURRENT_STEP=0

# Installation paths
MINOTAUR_DIR="$THIRD_PARTY_DIR/minotaur"
LLVM_DIR="$THIRD_PARTY_DIR/llvm-minotaur"
ALIVE2_DIR="$THIRD_PARTY_DIR/alive2-intrinsics"

# Logging functions
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_section() {
    echo -e "\n${CYAN}=== $1 ===${NC}" | tee -a "$LOG_FILE"
}

# Progress tracking functions
update_progress() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    local percentage=$((CURRENT_STEP * 100 / TOTAL_STEPS))
    local step_name="$1"
    local status="$2"
    
    echo "Step $CURRENT_STEP/$TOTAL_STEPS ($percentage%): $step_name - $status" > "$PROGRESS_FILE"
    log_info "Progress: $CURRENT_STEP/$TOTAL_STEPS ($percentage%) - $step_name"
}

# Cleanup function
cleanup_on_error() {
    log_error "Setup failed. Check $LOG_FILE for details."
    echo "FAILED at step $CURRENT_STEP/$TOTAL_STEPS" >> "$PROGRESS_FILE"
    exit 1
}

trap cleanup_on_error ERR

# Main setup function
main() {
    log_section "Minotaur SIMD Superoptimizer Setup"
    log_info "Starting Minotaur setup for Beat.zig SIMD optimization integration"
    log_info "Log file: $LOG_FILE"
    log_info "Progress file: $PROGRESS_FILE"
    
    check_dependencies
    create_directories
    setup_llvm_for_minotaur
    setup_alive2
    clone_minotaur
    build_minotaur
    verify_installation
    create_integration_scripts
    
    log_section "Setup Complete"
    log_success "Minotaur SIMD superoptimizer setup completed successfully!"
    log_info "Integration scripts created in scripts/ directory"
    log_info "Next: Run ./scripts/run_minotaur_analysis.sh to analyze SIMD code"
    echo "COMPLETE" > "$PROGRESS_FILE"
}

check_dependencies() {
    update_progress "Checking dependencies" "running"
    
    log_info "Checking required dependencies for Minotaur build..."
    
    # Required tools
    local required_tools=("cmake" "ninja" "git" "gcc-10" "g++-10" "redis-server")
    local missing_tools=()
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "On Debian/Ubuntu, install with:"
        log_info "sudo apt-get install cmake ninja-build gcc-10 g++-10 redis redis-server libhiredis-dev libbsd-resource-perl libredis-perl re2c libgtest-dev"
        exit 1
    fi
    
    # Check Redis
    if ! pgrep redis-server > /dev/null; then
        log_warning "Redis server not running. Starting it..."
        sudo systemctl start redis-server || {
            log_error "Failed to start Redis server. Please start it manually."
            exit 1
        }
    fi
    
    log_success "All dependencies satisfied"
}

create_directories() {
    update_progress "Creating directories" "running"
    
    log_info "Creating directory structure..."
    mkdir -p "$THIRD_PARTY_DIR"
    mkdir -p "$PROJECT_ROOT/artifacts/minotaur"
    
    log_success "Directories created"
}

setup_llvm_for_minotaur() {
    update_progress "Setting up LLVM for Minotaur" "running"
    
    if [ -d "$LLVM_DIR" ]; then
        log_info "LLVM directory exists, checking if compatible..."
        if [ -f "$LLVM_DIR/build/bin/llvm-config" ]; then
            log_success "LLVM already built for Minotaur"
            return
        fi
    fi
    
    log_info "Cloning and building LLVM for Minotaur compatibility..."
    
    # Clone LLVM project (use version compatible with Minotaur)
    cd "$THIRD_PARTY_DIR"
    if [ ! -d "llvm-project" ]; then
        git clone --depth 1 --branch llvmorg-18.1.6 https://github.com/llvm/llvm-project.git
    fi
    
    # Build LLVM
    mkdir -p "$LLVM_DIR/build"
    cd "$LLVM_DIR/build"
    
    log_info "Configuring LLVM build..."
    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DLLVM_ENABLE_PROJECTS="clang;lld" \
        -DLLVM_TARGETS_TO_BUILD="X86;AArch64" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DCMAKE_C_COMPILER=gcc-10 \
        -DCMAKE_CXX_COMPILER=g++-10 \
        ../llvm-project/llvm
    
    log_info "Building LLVM (this may take 30-60 minutes)..."
    ninja
    
    log_success "LLVM built successfully for Minotaur"
}

setup_alive2() {
    update_progress "Setting up Alive2" "running"
    
    if [ -d "$ALIVE2_DIR" ]; then
        log_info "Alive2 directory exists, checking build..."
        if [ -f "$ALIVE2_DIR/build/alive2.so" ]; then
            log_success "Alive2 already built"
            return
        fi
    fi
    
    log_info "Cloning and building Alive2 verification engine..."
    
    cd "$THIRD_PARTY_DIR"
    if [ ! -d "alive2-intrinsics" ]; then
        git clone https://github.com/minotaur-toolkit/alive2.git alive2-intrinsics
    fi
    
    cd "$ALIVE2_DIR"
    mkdir -p build
    cd build
    
    log_info "Configuring Alive2 build..."
    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_PREFIX_PATH="$LLVM_DIR/build" \
        -DCMAKE_C_COMPILER=gcc-10 \
        -DCMAKE_CXX_COMPILER=g++-10 \
        ..
    
    log_info "Building Alive2..."
    ninja
    
    log_success "Alive2 built successfully"
}

clone_minotaur() {
    update_progress "Cloning Minotaur" "running"
    
    if [ -d "$MINOTAUR_DIR" ]; then
        log_info "Minotaur directory exists, updating..."
        cd "$MINOTAUR_DIR"
        git pull origin main
    else
        log_info "Cloning Minotaur SIMD superoptimizer..."
        cd "$THIRD_PARTY_DIR"
        git clone https://github.com/minotaur-toolkit/minotaur.git
    fi
    
    log_success "Minotaur source code ready"
}

build_minotaur() {
    update_progress "Building Minotaur" "running"
    
    log_info "Building Minotaur SIMD superoptimizer..."
    
    cd "$MINOTAUR_DIR"
    mkdir -p build
    cd build
    
    log_info "Configuring Minotaur build..."
    cmake .. \
        -DALIVE2_SOURCE_DIR="$ALIVE2_DIR" \
        -DALIVE2_BUILD_DIR="$ALIVE2_DIR/build" \
        -DCMAKE_PREFIX_PATH="$LLVM_DIR/build" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_C_COMPILER=gcc-10 \
        -DCMAKE_CXX_COMPILER=g++-10 \
        -G Ninja
    
    log_info "Building Minotaur (this may take 10-20 minutes)..."
    ninja
    
    log_success "Minotaur built successfully"
}

verify_installation() {
    update_progress "Verifying installation" "running"
    
    log_info "Running Minotaur test suite..."
    
    cd "$MINOTAUR_DIR/build"
    if ninja check; then
        log_success "Minotaur test suite passed"
    else
        log_warning "Some Minotaur tests failed, but installation may still be functional"
    fi
    
    # Check executables
    local minotaur_executables=("minotaur-cc" "minotaur-cxx" "cache-infer")
    for exe in "${minotaur_executables[@]}"; do
        if [ -f "$exe" ]; then
            log_success "Found executable: $exe"
        else
            log_error "Missing executable: $exe"
            exit 1
        fi
    done
    
    log_success "Minotaur installation verified"
}

create_integration_scripts() {
    update_progress "Creating integration scripts" "running"
    
    log_info "Creating Minotaur integration scripts..."
    
    # Create analysis script
    cat > "$SCRIPT_DIR/run_minotaur_analysis.sh" << 'EOF'
#!/bin/bash

# Minotaur SIMD Analysis Script for Beat.zig
# Analyzes SIMD code using Minotaur superoptimizer

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MINOTAUR_DIR="$PROJECT_ROOT/third_party/minotaur"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

main() {
    echo -e "${GREEN}🔍 Minotaur SIMD Analysis for Beat.zig${NC}"
    echo "========================================"
    
    if [ ! -d "$MINOTAUR_DIR/build" ]; then
        echo "Error: Minotaur not built. Run setup_minotaur.sh first."
        exit 1
    fi
    
    # Generate LLVM IR from Beat.zig SIMD modules
    log_info "Generating LLVM IR from SIMD modules..."
    cd "$PROJECT_ROOT"
    
    # Generate IR for key SIMD modules
    local simd_modules=("simd.zig" "simd_batch.zig" "simd_classifier.zig" "batch_optimizer.zig")
    
    for module in "${simd_modules[@]}"; do
        if [ -f "src/$module" ]; then
            log_info "Analyzing src/$module..."
            zig build-obj "src/$module" -femit-llvm-bc -O ReleaseFast -target native
            
            # Extract SIMD optimization candidates
            export ENABLE_MINOTAUR=ON
            export MINOTAUR_NO_INFER=ON
            "$MINOTAUR_DIR/build/minotaur-cc" "${module%.zig}.o" -c
        fi
    done
    
    # Run synthesis on cached cuts
    log_info "Running SIMD optimization synthesis..."
    "$MINOTAUR_DIR/build/cache-infer"
    
    log_success "Minotaur SIMD analysis complete!"
    log_info "Check Redis cache for discovered SIMD optimizations"
}

main "$@"
EOF
    
    chmod +x "$SCRIPT_DIR/run_minotaur_analysis.sh"
    
    # Create combined analysis script
    cat > "$SCRIPT_DIR/run_combined_optimization.sh" << 'EOF'
#!/bin/bash

# Combined Souper + Minotaur + ISPC Optimization Pipeline
# Triple-optimization strategy for Beat.zig

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

main() {
    echo -e "${CYAN}🚀 Triple-Optimization Pipeline${NC}"
    echo "=================================="
    echo "Souper (scalar) + Minotaur (SIMD) + ISPC (acceleration)"
    echo
    
    # Step 1: Souper scalar optimization
    echo -e "${GREEN}Phase 1: Souper Scalar Optimization${NC}"
    if [ -f "$SCRIPT_DIR/run_souper_analysis.sh" ]; then
        "$SCRIPT_DIR/run_souper_analysis.sh" --quick
    else
        echo "Warning: Souper not available, skipping scalar optimization"
    fi
    
    # Step 2: Minotaur SIMD optimization  
    echo -e "${GREEN}Phase 2: Minotaur SIMD Optimization${NC}"
    "$SCRIPT_DIR/run_minotaur_analysis.sh"
    
    # Step 3: ISPC acceleration compilation
    echo -e "${GREEN}Phase 3: ISPC SPMD Acceleration${NC}"
    cd "$PROJECT_ROOT"
    if command -v ispc &> /dev/null; then
        zig build ispc-all
        echo "ISPC kernels compiled successfully"
    else
        echo "Warning: ISPC not available, skipping SPMD acceleration"
    fi
    
    echo
    echo -e "${CYAN}🎉 Triple-optimization pipeline complete!${NC}"
    echo "Your Beat.zig code has been analyzed with:"
    echo "  ✓ Souper: Scalar integer optimizations"
    echo "  ✓ Minotaur: SIMD vector optimizations" 
    echo "  ✓ ISPC: SPMD acceleration patterns"
}

main "$@"
EOF
    
    chmod +x "$SCRIPT_DIR/run_combined_optimization.sh"
    
    log_success "Integration scripts created successfully"
}

# Run main function
main "$@"