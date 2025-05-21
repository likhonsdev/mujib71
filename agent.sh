#!/bin/bash

# Configuration
PROJECT_DIR="$HOME/bd-model-generations"
STATUS_DIR="$PROJECT_DIR/status"
LOG_FILE="$PROJECT_DIR/logs/actions.log"

# Ensure directories exist
mkdir -p "$STATUS_DIR" "$PROJECT_DIR/logs"

# Log function for errors
log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >> "$LOG_FILE"
}

# Live status display function
display_status() {
    while true; do
        clear
        echo -e "\033[1;34m=== Live Agent Status (Bengali Language Model Generation) ===\033[0m"
        echo -e "\033[1;36mTime: $(date '+%H:%M:%S')\033[0m"
        echo ""

        # Data Collector Status
        if [ -f "$STATUS_DIR/data_collector.status" ]; then
            echo -e "\033[1;32mData Collector:\033[0m $(cat "$STATUS_DIR/data_collector.status")"
        else
            echo -e "\033[1;32mData Collector:\033[0m Not started or completed"
        fi

        # Model Trainer Status
        if [ -f "$STATUS_DIR/model_trainer.status" ]; then
            echo -e "\033[1;33mModel Trainer:\033[0m $(cat "$STATUS_DIR/model_trainer.status")"
        else
            echo -e "\033[1;33mModel Trainer:\033[0m Not started or completed"
        fi

        # Model Evaluator Status
        if [ -f "$STATUS_DIR/model_evaluator.status" ]; then
            echo -e "\033[1;31mModel Evaluator:\033[0m $(cat "$STATUS_DIR/model_evaluator.status")"
        else
            echo -e "\033[1;31mModel Evaluator:\033[0m Not started or completed"
        fi

        # Check if all agents are done
        if [ ! -f "$STATUS_DIR/data_collector.status" ] && \
           [ ! -f "$STATUS_DIR/model_trainer.status" ] && \
           [ ! -f "$STATUS_DIR/model_evaluator.status" ]; then
            echo ""
            echo -e "\033[1;34mAll agents have completed their tasks.\033[0m"
            break
        fi
        sleep 2
    done
}

# Main process
echo "Starting Bengali language model generation..." | tee -a "$LOG_FILE"

# Launch agents in background
for agent in data_collector model_trainer model_evaluator; do
    if [ -f "$PROJECT_DIR/$agent.sh" ]; then
        echo "Starting $agent..." | tee -a "$LOG_FILE"
        bash "$PROJECT_DIR/$agent.sh" &>> "$LOG_FILE" || log_error "$agent failed to execute"
    else
        log_error "$agent.sh not found in $PROJECT_DIR"
    fi
done

# Display live status
display_status

echo "Process completed. Check logs in $LOG_FILE for details." | tee -a "$LOG_FILE"
