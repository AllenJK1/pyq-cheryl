#!/bin/bash
BASE_DIR="$HOME/pyq-cheryl"
CONNECT="$BASE_DIR/connect.py"
FINALLY="$BASE_DIR/finally3.py"
SESSION_FILE="$BASE_DIR/session.json"
IDLE_LIMIT=120  # seconds
ALLOW_FILE="$BASE_DIR/connect.allow"
IDLE_FILE="$BASE_DIR/idle.lck"

# Remove session.json so we always start fresh
if [[ -f "$SESSION_FILE" ]]; then
    echo "[*] Removing existing session.json..."
    rm -f "$SESSION_FILE"
fi

echo "[*] Starting connect.py..."
# Run connect.py unbuffered and read line by line
python3 -u "$CONNECT" | while IFS= read -r line; do
    echo "$line"  # show output in tmux

    if [[ "$line" == *"Login successful"* ]]; then
        echo "[*] Login successful detected! Terminating connect.py..."
        pkill -f "$CONNECT"
        break
    fi
done

# Short delay to ensure process is gone
sleep 1

# Wait for connect.allow file before proceeding
echo "[*] Waiting for connect.allow file..."
while [[ ! -f "$ALLOW_FILE" ]]; do
    sleep 1
done
echo "[*] connect.allow detected. Proceeding to finally3.py..."

echo "[*] Starting finally3.py with idleness detection..."
# Track last output timestamp
LAST_OUTPUT=$(date +%s)

# Run finally3.py line by line in background
python3 -u "$FINALLY" | while IFS= read -r line; do
    echo "$line"
    LAST_OUTPUT=$(date +%s)
done &

FINALLY_PID=$!

# Monitor idleness
while kill -0 "$FINALLY_PID" 2>/dev/null; do
    NOW=$(date +%s)
    DIFF=$(( NOW - LAST_OUTPUT ))
    if (( DIFF > IDLE_LIMIT )); then
        echo "[!] Idleness detected (> $IDLE_LIMIT seconds). Terminating finally3.py..."
        # Create idle.lck as a flag for the daemon
        touch "$IDLE_FILE"
        kill "$FINALLY_PID"
        wait "$FINALLY_PID" 2>/dev/null
        break
    fi
    sleep 1
done

echo "[*] Runner finished."

