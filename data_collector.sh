#!/bin/bash
STATUS_FILE="$HOME/bd-model-generations/status/data_collector.status"
LOG_FILE="$HOME/bd-model-generations/logs/actions.log"

echo "Collecting data..." > "$STATUS_FILE"
# Simulate data collection (replace with actual logic)
sleep 5
echo "Data collection complete." > "$STATUS_FILE"
sleep 1
rm -f "$STATUS_FILE" || echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to remove $STATUS_FILE" >> "$LOG_FILE"