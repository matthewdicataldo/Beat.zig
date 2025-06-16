#!/bin/bash

# Souper Toolchain Setup Script for Beat.zig Integration
# This script implements the LLVM version compatibility strategy from sopuer!.md

set -euo pipefail

# Logging configuration
LOG_FILE="souper_setup_$(date +%Y%m%d_%H%M%S).log"
PROGRESS_FILE="souper_progress.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Progress tracking  
TOTAL_STEPS=9
CURRENT_STEP=0

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

# Progress tracking functions
update_progress() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    local percentage=$((CURRENT_STEP * 100 / TOTAL_STEPS))
    local step_name="$1"
    local status="$2"
    
    echo "STEP: $CURRENT_STEP/$TOTAL_STEPS" > "$PROGRESS_FILE"
    echo "PERCENTAGE: $percentage%" >> "$PROGRESS_FILE"
    echo "CURRENT: $step_name" >> "$PROGRESS_FILE"
    echo "STATUS: $status" >> "$PROGRESS_FILE"
    echo "TIMESTAMP: $(date)" >> "$PROGRESS_FILE"
    
    log_info "Progress: [$CURRENT_STEP/$TOTAL_STEPS] ($percentage%) - $step_name: $status"
}

# Progress bar function
show_progress_bar() {
    local percentage=$1
    local width=50
    local filled=$((percentage * width / 100))
    local empty=$((width - filled))
    
    printf "\r${BLUE}["
    printf "%*s" $filled | tr ' ' '='
    printf "%*s" $empty | tr ' ' '-'
    printf "] %d%%${NC}" $percentage
}

# Enhanced command execution with progress monitoring
run_with_progress() {
    local cmd="$1"
    local step_name="$2"
    local progress_pattern="$3"
    
    log_info "Starting: $step_name"
    update_progress "$step_name" "RUNNING"
    
    # Run command in background and monitor
    if [[ -n "$progress_pattern" ]]; then
        $cmd 2>&1 | while IFS= read -r line; do
            echo "$line" >> "$LOG_FILE"
            
            # Extract progress if pattern matches
            if echo "$line" | grep -qE "$progress_pattern"; then
                local progress_info=$(echo "$line" | grep -oE "$progress_pattern" | head -1)
                echo -ne "\r${YELLOW}$step_name: $progress_info${NC}"
            fi
        done
    else
        $cmd >> "$LOG_FILE" 2>&1
    fi
    
    if [[ $? -eq 0 ]]; then
        update_progress "$step_name" "COMPLETED"
        log_success "$step_name completed successfully"
    else
        update_progress "$step_name" "FAILED"
        log_error "$step_name failed! Check $LOG_FILE for details"
        return 1
    fi
}

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
    update_progress "Prerequisites Check" "RUNNING"
    log_info "Checking prerequisites..."
    
    # Check for required tools
    local missing_tools=()
    for tool in cmake git ninja make python3; do
        if ! command -v $tool >/dev/null 2>&1; then
            missing_tools+=("$tool")
        else
            log_info "✓ Found $tool: $(which $tool)"
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Install with: sudo apt-get install ${missing_tools[*]}"
        update_progress "Prerequisites Check" "FAILED"
        exit 1
    fi
    
    # Check for zstd development headers
    if ! pkg-config --exists libzstd 2>/dev/null; then
        log_warning "libzstd-dev may not be installed"
        log_warning "Install with: sudo apt-get install libzstd-dev"
        log_warning "On macOS: brew install zstd"
    else
        log_info "✓ Found libzstd development headers"
    fi
    
    # Check available disk space
    local available_space=$(df . | tail -1 | awk '{print $4}')
    local required_space=10485760  # 10GB in KB
    if [[ $available_space -lt $required_space ]]; then
        log_warning "Low disk space detected. Souper build requires ~10GB"
        log_warning "Available: $(($available_space / 1024 / 1024))GB, Required: 10GB"
    else
        log_info "✓ Sufficient disk space available"
    fi
    
    # Check memory
    local memory_gb=$(free -g | awk 'NR==2{print $2}')
    if [[ $memory_gb -lt 8 ]]; then
        log_warning "Low memory detected. Build may be slow with <8GB RAM"
        log_warning "Available: ${memory_gb}GB, Recommended: 8GB+"
    else
        log_info "✓ Sufficient memory available: ${memory_gb}GB"
    fi
    
    update_progress "Prerequisites Check" "COMPLETED"
    log_success "Prerequisites check passed"
}

# Clone Souper repository
clone_souper() {
    update_progress "Clone Souper Repository" "RUNNING"
    log_info "Cloning Souper repository..."
    
    if [ -d "$SOUPER_DIR" ]; then
        log_info "Souper directory already exists, pulling latest changes..."
        cd "$SOUPER_DIR"
        run_with_progress "git pull" "Git Pull" "Receiving objects.*([0-9]+%)"
        cd - >/dev/null
    else
        log_info "Creating directory structure..."
        mkdir -p "$(dirname "$SOUPER_DIR")"
        run_with_progress "git clone $SOUPER_REPO $SOUPER_DIR" "Git Clone" "Receiving objects.*([0-9]+%)"
    fi
    
    update_progress "Clone Souper Repository" "COMPLETED"
    log_success "Souper repository ready"
}

# Extract LLVM version from Souper's build_deps.sh
get_llvm_version() {
    update_progress "Extract LLVM Version" "RUNNING"
    log_info "Extracting LLVM version from Souper's build_deps.sh..."
    
    cd "$SOUPER_DIR"
    if [ ! -f "build_deps.sh" ]; then
        log_error "build_deps.sh not found in Souper repository"
        update_progress "Extract LLVM Version" "FAILED"
        exit 1
    fi
    
    # Extract LLVM commit/version
    LLVM_COMMIT=$(grep -o 'llvm_commit=.*' build_deps.sh | cut -d'=' -f2 | tr -d '"' || true)
    LLVM_BRANCH=$(grep -o 'llvm_branch=.*' build_deps.sh | cut -d'=' -f2 | tr -d '"' || true)
    
    log_info "Found LLVM configuration:"
    log_info "  Commit: $LLVM_COMMIT"
    log_info "  Branch: $LLVM_BRANCH"
    
    cd - >/dev/null
    update_progress "Extract LLVM Version" "COMPLETED"
    log_success "LLVM version configuration extracted"
}

# Build shared LLVM installation
build_shared_llvm() {
    update_progress "Build Shared LLVM" "RUNNING"
    log_info "Building shared LLVM installation..."
    log_warning "This will take 30-60 minutes depending on your system"
    
    # Run Souper's dependency script to get the exact LLVM version
    cd "$SOUPER_DIR"
    log_info "Running Souper's build_deps.sh..."
    run_with_progress "./build_deps.sh $BUILD_TYPE" "LLVM Build" "\[[0-9]+/[0-9]+\]|Building.*([0-9]+%)|\[[0-9]+%\]"
    
    # The dependencies are now in third_party/ subdirectory
    SOUPER_LLVM_BUILD="$SOUPER_DIR/third_party/llvm/$BUILD_TYPE"
    
    if [ ! -d "$SOUPER_LLVM_BUILD" ]; then
        log_error "LLVM build directory not found: $SOUPER_LLVM_BUILD"
        update_progress "Build Shared LLVM" "FAILED"
        exit 1
    fi
    
    # Install to shared location
    log_info "Installing LLVM to shared location: $LLVM_INSTALL_PREFIX"
    cd "$SOUPER_LLVM_BUILD"
    run_with_progress "make install DESTDIR=\"\" CMAKE_INSTALL_PREFIX=\"$LLVM_INSTALL_PREFIX\"" "LLVM Install" "Installing.*([0-9]+%)|\[[0-9]+/[0-9]+\]"
    
    cd - >/dev/null
    update_progress "Build Shared LLVM" "COMPLETED"
    log_success "Shared LLVM installation complete"
}

# Build Souper against shared LLVM
build_souper() {
    update_progress "Build Souper" "RUNNING"
    log_info "Building Souper against shared LLVM..."
    
    cd "$SOUPER_DIR"
    mkdir -p build
    cd build
    
    # Configure Souper to use the shared LLVM
    log_info "Configuring Souper build system..."
    run_with_progress "cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DLLVM_DIR=\"$LLVM_INSTALL_PREFIX/lib/cmake/llvm\" -DCMAKE_INSTALL_PREFIX=\"$LLVM_INSTALL_PREFIX\" .." "CMake Configure" "-- .*"
    
    # Build Souper
    log_info "Building Souper (using $(nproc) parallel jobs)..."
    run_with_progress "make -j$(nproc)" "Souper Build" "\[[0-9]+/[0-9]+\]|Building.*([0-9]+%)|\[[0-9]+%\]"
    
    # Install Souper tools to shared location
    log_info "Installing Souper tools..."
    run_with_progress "make install" "Souper Install" "Installing.*([0-9]+%)|\[[0-9]+/[0-9]+\]"
    
    cd - >/dev/null
    update_progress "Build Souper" "COMPLETED"
    log_success "Souper build complete"
}

# Verify installation
verify_installation() {
    update_progress "Verify Installation" "RUNNING"
    log_info "Verifying Souper installation..."
    
    # Check if executables exist
    SOUPER_BIN="$LLVM_INSTALL_PREFIX/bin/souper"
    Z3_BIN="$LLVM_INSTALL_PREFIX/bin/z3"
    
    if [ ! -f "$SOUPER_BIN" ]; then
        log_error "souper executable not found at $SOUPER_BIN"
        update_progress "Verify Installation" "FAILED"
        exit 1
    else
        log_success "✓ Found souper executable: $SOUPER_BIN"
    fi
    
    if [ ! -f "$Z3_BIN" ]; then
        log_warning "z3 executable not found at $Z3_BIN"
        log_warning "You may need to install Z3 separately"
    else
        log_success "✓ Found z3 executable: $Z3_BIN"
    fi
    
    # Run Souper's test suite
    log_info "Running Souper test suite..."
    cd "$SOUPER_DIR/build"
    if run_with_progress "make check" "Souper Test Suite" "Running.*test|PASS|FAIL"; then
        log_success "Souper test suite passed!"
    else
        log_warning "Some Souper tests failed, but installation may still be usable"
    fi
    
    cd - >/dev/null
    update_progress "Verify Installation" "COMPLETED"
    log_success "Installation verification complete"
}

# Generate environment setup script
generate_env_script() {
    update_progress "Generate Environment Script" "RUNNING"
    log_info "Generating environment setup script..."
    
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
    update_progress "Generate Environment Script" "COMPLETED"
    log_success "Environment script created: $ENV_SCRIPT"
    log_info "Run 'source $ENV_SCRIPT' to set up your environment"
}

# Background execution wrapper
run_in_background() {
    local script_name="$0"
    local log_file="$LOG_FILE"
    local progress_file="$PROGRESS_FILE"
    
    log_info "Starting Souper setup in background..."
    log_info "Log file: $log_file"
    log_info "Progress file: $progress_file"
    
    # Run main setup in background
    nohup bash -c "
        source '$script_name'
        main_execution
    " >> "$log_file" 2>&1 &
    
    local bg_pid=$!
    echo "$bg_pid" > "souper_setup.pid"
    
    log_success "Souper setup started in background (PID: $bg_pid)"
    log_info "Monitor progress with: watch -n 5 'cat $progress_file'"
    log_info "View live logs with: tail -f $log_file"
    log_info "Check if running with: ps -p $bg_pid"
    
    echo
    echo "=== Background Setup Commands ==="
    echo "Monitor progress:  watch -n 5 'cat $progress_file'"
    echo "View live logs:    tail -f $log_file"
    echo "Check status:      ps -p $bg_pid"
    echo "Kill if needed:    kill $bg_pid"
    echo
}

# Periodic progress checker
check_progress() {
    local max_checks=720  # 60 minutes with 5-second intervals
    local check_count=0
    
    if [ ! -f "souper_setup.pid" ]; then
        log_error "No background setup process found (souper_setup.pid missing)"
        return 1
    fi
    
    local bg_pid=$(cat "souper_setup.pid")
    
    log_info "Monitoring background setup process (PID: $bg_pid)..."
    log_info "Will check for up to 60 minutes"
    
    while [ $check_count -lt $max_checks ]; do
        # Check if process is still running
        if ! ps -p "$bg_pid" > /dev/null 2>&1; then
            if [ -f "$PROGRESS_FILE" ]; then
                local final_status=$(grep "STATUS:" "$PROGRESS_FILE" | tail -1 | cut -d: -f2 | tr -d ' ')
                if [ "$final_status" = "COMPLETED" ]; then
                    log_success "Setup completed successfully!"
                    display_instructions
                    return 0
                else
                    log_error "Setup process ended unexpectedly"
                    log_error "Check $LOG_FILE for details"
                    return 1
                fi
            else
                log_error "Setup process ended without progress information"
                return 1
            fi
        fi
        
        # Display current progress
        if [ -f "$PROGRESS_FILE" ]; then
            local current_step=$(grep "STEP:" "$PROGRESS_FILE" | tail -1 | cut -d: -f2 | tr -d ' ')
            local percentage=$(grep "PERCENTAGE:" "$PROGRESS_FILE" | tail -1 | cut -d: -f2 | tr -d ' ')
            local current_task=$(grep "CURRENT:" "$PROGRESS_FILE" | tail -1 | cut -d: -f2- | sed 's/^ *//')
            local status=$(grep "STATUS:" "$PROGRESS_FILE" | tail -1 | cut -d: -f2 | tr -d ' ')
            
            echo -ne "\r${BLUE}Progress: $current_step ($percentage) - $current_task: $status${NC}"
            
            if [ "$status" = "COMPLETED" ] && [ "$current_step" = "$TOTAL_STEPS/$TOTAL_STEPS" ]; then
                echo
                log_success "Setup completed successfully!"
                display_instructions
                return 0
            fi
        fi
        
        sleep 5
        check_count=$((check_count + 1))
    done
    
    log_warning "Progress monitoring timed out after 60 minutes"
    log_info "Setup may still be running. Check manually with: ps -p $bg_pid"
    return 1
}

# Display final instructions
display_instructions() {
    echo
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
    echo "For more information, see SOUPER_INTEGRATION.md"
}

# Core setup execution (for background use)
main_execution() {
    check_prerequisites
    clone_souper
    get_llvm_version
    build_shared_llvm
    build_souper
    verify_installation
    generate_env_script
    
    # Final progress update
    update_progress "Setup Complete" "COMPLETED"
    log_success "=== Souper Setup Complete ==="
}

# Main execution with argument parsing
main() {
    case "${1:-}" in
        --background|-b)
            run_in_background
            ;;
        --check-progress|-p)
            check_progress
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --background, -b     Run setup in background with progress monitoring"
            echo "  --check-progress, -p Check progress of background setup"
            echo "  --help, -h           Show this help message"
            echo
            echo "Example usage:"
            echo "  $0                   # Run setup interactively"
            echo "  $0 --background      # Run setup in background"
            echo "  $0 --check-progress  # Monitor background progress"
            ;;
        "")
            # Interactive execution
            main_execution
            display_instructions
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
}

# Progress monitoring function for external use
monitor_progress() {
    if [ ! -f "$PROGRESS_FILE" ]; then
        echo "No progress file found. Setup may not be running."
        return 1
    fi
    
    echo "=== Souper Setup Progress ==="
    cat "$PROGRESS_FILE"
    echo
    
    # Show recent log entries
    if [ -f "$LOG_FILE" ]; then
        echo "=== Recent Log Entries ==="
        tail -10 "$LOG_FILE"
    fi
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi