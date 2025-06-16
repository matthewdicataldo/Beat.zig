#!/bin/bash

# Souper Setup Progress Monitor
# Real-time monitoring of the Souper toolchain setup process

set -euo pipefail

# Configuration
PROGRESS_FILE="souper_progress.txt"
LOG_FILE_PATTERN="souper_setup_*.log"
PID_FILE="souper_setup.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Display header
display_header() {
    clear
    echo -e "${CYAN}=== Beat.zig Souper Setup Progress Monitor ===${NC}"
    echo -e "${CYAN}Updated: $(date)${NC}"
    echo
}

# Show current progress
show_progress() {
    if [ ! -f "$PROGRESS_FILE" ]; then
        echo -e "${RED}No progress file found. Setup may not be running.${NC}"
        return 1
    fi
    
    local current_step=$(grep "STEP:" "$PROGRESS_FILE" | tail -1 | cut -d: -f2 | tr -d ' ')
    local percentage=$(grep "PERCENTAGE:" "$PROGRESS_FILE" | tail -1 | cut -d: -f2 | tr -d ' ')
    local current_task=$(grep "CURRENT:" "$PROGRESS_FILE" | tail -1 | cut -d: -f2- | sed 's/^ *//')
    local status=$(grep "STATUS:" "$PROGRESS_FILE" | tail -1 | cut -d: -f2 | tr -d ' ')
    local timestamp=$(grep "TIMESTAMP:" "$PROGRESS_FILE" | tail -1 | cut -d: -f2- | sed 's/^ *//')
    
    echo -e "${BLUE}Progress: $current_step ($percentage)${NC}"
    echo -e "${YELLOW}Current Task: $current_task${NC}"
    echo -e "${GREEN}Status: $status${NC}"
    echo -e "${CYAN}Last Update: $timestamp${NC}"
    echo
    
    # Progress bar
    local perc_num=$(echo "$percentage" | tr -d '%')
    if [[ "$perc_num" =~ ^[0-9]+$ ]]; then
        local width=50
        local filled=$((perc_num * width / 100))
        local empty=$((width - filled))
        
        printf "${BLUE}["
        printf "%*s" $filled | tr ' ' '='
        printf "%*s" $empty | tr ' ' '-'
        printf "] %s${NC}\n" "$percentage"
    fi
    echo
}

# Check process status
check_process_status() {
    if [ ! -f "$PID_FILE" ]; then
        echo -e "${RED}No PID file found. Setup may not be running.${NC}"
        return 1
    fi
    
    local bg_pid=$(cat "$PID_FILE")
    
    if ps -p "$bg_pid" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Setup process is running (PID: $bg_pid)${NC}"
        
        # Show CPU and memory usage
        local cpu_mem=$(ps -p "$bg_pid" -o %cpu,%mem --no-headers 2>/dev/null || echo "N/A N/A")
        echo -e "${CYAN}  CPU/Memory usage: $cpu_mem${NC}"
    else
        echo -e "${RED}✗ Setup process is not running${NC}"
        return 1
    fi
    echo
}

# Show recent log entries
show_recent_logs() {
    local log_file=$(ls -t $LOG_FILE_PATTERN 2>/dev/null | head -1)
    
    if [ -z "$log_file" ]; then
        echo -e "${RED}No log files found${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}=== Recent Log Entries ($log_file) ===${NC}"
    tail -15 "$log_file" | while IFS= read -r line; do
        # Color-code log levels
        if echo "$line" | grep -q "\[ERROR\]"; then
            echo -e "${RED}$line${NC}"
        elif echo "$line" | grep -q "\[WARNING\]"; then
            echo -e "${YELLOW}$line${NC}"
        elif echo "$line" | grep -q "\[SUCCESS\]"; then
            echo -e "${GREEN}$line${NC}"
        elif echo "$line" | grep -q "\[INFO\]"; then
            echo -e "${BLUE}$line${NC}"
        else
            echo "$line"
        fi
    done
    echo
}

# Show disk space
show_disk_space() {
    echo -e "${YELLOW}=== Disk Space Usage ===${NC}"
    df -h . | tail -1 | awk '{printf "Available: %s (Used: %s of %s)\n", $4, $3, $2}'
    echo
}

# Interactive monitoring mode
interactive_monitor() {
    local refresh_interval=5
    
    echo -e "${CYAN}Starting interactive monitoring (refreshing every ${refresh_interval}s)${NC}"
    echo -e "${CYAN}Press Ctrl+C to exit${NC}"
    echo
    
    while true; do
        display_header
        
        if check_process_status; then
            show_progress
            show_disk_space
            show_recent_logs
            
            # Check if completed
            if [ -f "$PROGRESS_FILE" ]; then
                local status=$(grep "STATUS:" "$PROGRESS_FILE" | tail -1 | cut -d: -f2 | tr -d ' ')
                local current_step=$(grep "STEP:" "$PROGRESS_FILE" | tail -1 | cut -d: -f2 | tr -d ' ')
                
                if [ "$status" = "COMPLETED" ]; then
                    echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
                    echo -e "${CYAN}You can now run: source souper_env.sh${NC}"
                    break
                elif [ "$status" = "FAILED" ]; then
                    echo -e "${RED}❌ Setup failed. Check the logs above for details.${NC}"
                    break
                fi
            fi
        else
            echo -e "${RED}Setup process is not running.${NC}"
            break
        fi
        
        echo -e "${CYAN}Refreshing in ${refresh_interval}s... (Ctrl+C to exit)${NC}"
        sleep $refresh_interval
    done
}

# Show quick status
quick_status() {
    display_header
    check_process_status
    show_progress
    show_disk_space
}

# Show help
show_help() {
    echo "Souper Setup Progress Monitor"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -i, --interactive    Interactive monitoring with auto-refresh"
    echo "  -q, --quick         Show quick status snapshot"
    echo "  -l, --logs          Show recent log entries only"
    echo "  -h, --help          Show this help message"
    echo
    echo "Examples:"
    echo "  $0                  # Quick status (default)"
    echo "  $0 -i               # Interactive monitoring"
    echo "  $0 -l               # View recent logs"
    echo
    echo "Tip: Use 'watch -n 5 $0 -q' for auto-refreshing quick status"
}

# Main execution
main() {
    case "${1:-}" in
        -i|--interactive)
            interactive_monitor
            ;;
        -q|--quick)
            quick_status
            ;;
        -l|--logs)
            show_recent_logs
            ;;
        -h|--help)
            show_help
            ;;
        "")
            quick_status
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
}

# Trap Ctrl+C gracefully
trap 'echo -e "\n${CYAN}Monitoring stopped.${NC}"; exit 0' INT

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi