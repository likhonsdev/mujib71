#!/bin/bash
STATUS_FILE="$HOME/bd-model-generations/status/model_trainer.status"
LOG_FILE="$HOME/bd-model-generations/logs/actions.log"

echo "Training model..." > "$STATUS_FILE"
# Simulate model training (replace with actual logic)
sleep 5
echo "Model training complete." > "$STATUS_FILE"
sleep 1
rm -f "$STATUS_FILE" || echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to remove $STATUS_FILE" >> "$LOG_FILE"