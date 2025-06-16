#!/bin/bash

# Souper Setup Progress Monitor
# Monitors the robust setup script progress and provides real-time updates

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ARTIFACTS_DIR="$PROJECT_ROOT/artifacts/souper"
CHECKPOINT_DIR="$ARTIFACTS_DIR/checkpoints"
LOG_FILE="$ARTIFACTS_DIR/logs/setup.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Progress monitoring function
monitor_progress() {
    local interval=${1:-5}  # Default 5 second interval
    
    echo -e "${BLUE}Souper Setup Progress Monitor${NC}"
    echo -e "${BLUE}=============================${NC}"
    echo
    
    while true; do
        clear
        echo -e "${BLUE}Souper Setup Progress Monitor - $(date)${NC}"
        echo -e "${BLUE}================================================${NC}"
        echo
        
        # Check overall status
        local total_steps=6
        local completed_steps=0
        
        # Define steps with descriptions
        declare -A steps=(
            ["setup_directories"]="📁 Setting up directories and dependencies"
            ["clone_repositories"]="📥 Cloning Souper repository"
            ["build_dependencies"]="🏗️  Building dependencies (LLVM, Z3, Alive2) - 60-120 min"
            ["build_souper"]="⚡ Building Souper optimization framework"
            ["create_environment"]="🌟 Creating environment setup"
            ["validate_installation"]="✅ Validating installation"
        )
        
        echo -e "${YELLOW}Step Progress:${NC}"
        echo "=============="
        
        for step in setup_directories clone_repositories build_dependencies build_souper create_environment validate_installation; do
            if [[ -f "$CHECKPOINT_DIR/$step.done" ]]; then
                echo -e "${GREEN}✓${NC} ${steps[$step]}"
                ((completed_steps++))
            else
                echo -e "${RED}○${NC} ${steps[$step]}"
            fi
        done
        
        echo
        echo -e "${YELLOW}Overall Progress: ${completed_steps}/${total_steps} ($(( completed_steps * 100 / total_steps ))%)${NC}"
        
        # Progress bar
        local bar_length=50
        local filled_length=$(( completed_steps * bar_length / total_steps ))
        local bar=""
        for ((i=0; i<filled_length; i++)); do bar+="█"; done
        for ((i=filled_length; i<bar_length; i++)); do bar+="░"; done
        echo -e "${GREEN}${bar}${NC}"
        
        echo
        
        # Show recent log entries
        if [[ -f "$LOG_FILE" ]]; then
            echo -e "${YELLOW}Recent Activity:${NC}"
            echo "==============="
            tail -n 10 "$LOG_FILE" | while IFS= read -r line; do
                if [[ $line == *"ERROR"* ]]; then
                    echo -e "${RED}${line}${NC}"
                elif [[ $line == *"Checkpoint created"* ]]; then
                    echo -e "${GREEN}${line}${NC}"
                else
                    echo "$line"
                fi
            done
        else
            echo -e "${YELLOW}Log file not found. Setup may not have started yet.${NC}"
        fi
        
        echo
        
        # Check if setup is complete
        if [[ $completed_steps -eq $total_steps ]]; then
            echo -e "${GREEN}🎉 Setup Complete! 🎉${NC}"
            echo -e "${GREEN}Run 'source souper_env.sh' to activate the environment.${NC}"
            break
        fi
        
        # Check if setup failed
        if [[ -f "$LOG_FILE" ]] && tail -n 1 "$LOG_FILE" | grep -q "ERROR"; then
            echo -e "${RED}❌ Setup Failed! Check the log above for details.${NC}"
            echo -e "${YELLOW}💡 Run './scripts/robust_souper_setup.sh --resume' to continue from last checkpoint.${NC}"
        fi
        
        echo -e "${BLUE}Refreshing in ${interval} seconds... (Ctrl+C to exit)${NC}"
        sleep "$interval"
    done
}

# Log tail function
tail_logs() {
    if [[ -f "$LOG_FILE" ]]; then
        echo -e "${BLUE}Tailing Souper setup logs (Ctrl+C to exit):${NC}"
        echo -e "${BLUE}===========================================${NC}"
        tail -f "$LOG_FILE"
    else
        echo -e "${RED}Log file not found: $LOG_FILE${NC}"
        exit 1
    fi
}

# Status check function
check_status() {
    echo -e "${BLUE}Souper Setup Status${NC}"
    echo -e "${BLUE}===================${NC}"
    echo
    
    if [[ ! -d "$CHECKPOINT_DIR" ]]; then
        echo -e "${YELLOW}No setup has been started yet.${NC}"
        echo -e "${YELLOW}Run './scripts/robust_souper_setup.sh' to begin.${NC}"
        return
    fi
    
    local total_steps=6
    local completed_steps=0
    
    declare -A steps=(
        ["setup_directories"]="Setting up directories and dependencies"
        ["clone_repositories"]="Cloning Souper repository"
        ["build_dependencies"]="Building dependencies (LLVM, Z3, Alive2)"
        ["build_souper"]="Building Souper optimization framework"
        ["create_environment"]="Creating environment setup"
        ["validate_installation"]="Validating installation"
    )
    
    for step in setup_directories clone_repositories build_dependencies build_souper create_environment validate_installation; do
        if [[ -f "$CHECKPOINT_DIR/$step.done" ]]; then
            echo -e "${GREEN}✓${NC} ${steps[$step]}"
            ((completed_steps++))
        else
            echo -e "${RED}○${NC} ${steps[$step]}"
        fi
    done
    
    echo
    echo -e "${YELLOW}Progress: ${completed_steps}/${total_steps} steps completed${NC}"
    
    if [[ $completed_steps -eq $total_steps ]]; then
        echo -e "${GREEN}✅ Setup is complete!${NC}"
        if [[ -f "$PROJECT_ROOT/souper_env.sh" ]]; then
            echo -e "${GREEN}Run 'source souper_env.sh' to activate the environment.${NC}"
        fi
    elif [[ $completed_steps -gt 0 ]]; then
        echo -e "${YELLOW}Setup is in progress. Run with --resume to continue.${NC}"
    else
        echo -e "${YELLOW}Setup has not been started.${NC}"
    fi
}

# Help function
show_help() {
    cat << EOF
Souper Setup Progress Monitor

Usage: $0 [OPTIONS]

OPTIONS:
    -i, --interval N    Set refresh interval in seconds (default: 5)
    -t, --tail          Tail the setup logs in real-time
    -s, --status        Show current setup status and exit
    -h, --help          Show this help message

EXAMPLES:
    $0                  # Monitor with 5-second intervals
    $0 -i 2             # Monitor with 2-second intervals
    $0 --tail           # Tail logs in real-time
    $0 --status         # Check status once and exit

This script monitors the robust_souper_setup.sh progress and provides
real-time updates on the build status.
EOF
}

# Parse command line arguments
INTERVAL=5
ACTION="monitor"

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--interval)
            INTERVAL="$2"
            shift 2
            ;;
        -t|--tail)
            ACTION="tail"
            shift
            ;;
        -s|--status)
            ACTION="status"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Execute the requested action
case $ACTION in
    monitor)
        monitor_progress "$INTERVAL"
        ;;
    tail)
        tail_logs
        ;;
    status)
        check_status
        ;;
esac