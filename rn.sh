#!/bin/bash
BASE_DIR="$HOME/pyq-cheryl"
CONNECT="$BASE_DIR/connect.py"
FINALLY="$BASE_DIR/finally3.py"
SESSION_FILE="$BASE_DIR/session.json"
IDLE_LIMIT=120  # seconds
ALLOW_FILE="$BASE_DIR/connect.allow"
IDLE_FILE="$BASE_DIR/idle.lck"
KILL_FILE="$BASE_DIR/kill.switch"

# Remove session.json so we always start fresh
[[ -f "$SESSION_FILE" ]] && echo "[*] Removing existing session.json..." && rm -f "$SESSION_FILE"

# Function to terminate any running scripts
terminate_all() {
    echo "[!] Terminating all runner processes..."
    pkill -f "$CONNECT" 2>/dev/null
    pkill -f "$FINALLY" 2>/dev/null
    exit 0
}

# Trap kill.switch globally in the background
(
    while true; do
        if [[ -f "$KILL_FILE" ]]; then
            echo "[!] Kill switch detected. Terminating everything..."
            rm -f "$KILL_FILE"  # optional: reset
            terminate_all
        fi
        sleep 1
    done
) &

KILL_WATCH_PID=$!

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
    [[ -f "$KILL_FILE" ]] && terminate_all
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

# Monitor idleness and kill switch for finally3.py
while kill -0 "$FINALLY_PID" 2>/dev/null; do
    NOW=$(date +%s)
    DIFF=$(( NOW - LAST_OUTPUT ))

    if (( DIFF > IDLE_LIMIT )); then
        echo "[!] Idleness detected (> $IDLE_LIMIT seconds). Terminating finally3.py..."
        touch "$IDLE_FILE"
        kill "$FINALLY_PID"
        wait "$FINALLY_PID" 2>/dev/null
        break
    fi

    [[ -f "$KILL_FILE" ]] && terminate_all

    sleep 1
done

# Clean up kill watch
kill "$KILL_WATCH_PID" 2>/dev/null

echo "[*] Runner finished."

