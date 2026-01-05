#!/bin/bash
# wrapper.sh - run inside tmux session

BASE_DIR="$HOME/pyq-cheryl"
CONNECT="$BASE_DIR/connect.py"
FINALLY="$BASE_DIR/finally3.py"
LOG_CONNECT="$BASE_DIR/connect.log"
LOG_FINALLY="$BASE_DIR/finally3.log"

# Clean old logs
rm -f "$LOG_CONNECT" "$LOG_FINALLY"

# Step 1: Run connect.py
echo "[*] Starting connect.py..."
python3 "$CONNECT" | tee "$LOG_CONNECT" &
CONNECT_PID=$!

# Step 2: Wait for "Login successful"
echo "[*] Waiting for login success..."
while true; do
    if grep -q "Connected successfully!" "$LOG_CONNECT"; then
        echo "[*] Login successful detected!"
        kill "$CONNECT_PID" 2>/dev/null
        wait "$CONNECT_PID" 2>/dev/null
        break
    fi
    # Optional: timeout if connect.py takes too long
    sleep 1
done

# Step 3: Run finally3.py
echo "[*] Starting finally3.py..."
python3 "$FINALLY" | tee "$LOG_FINALLY" &
FINALLY_PID=$!

# Step 4: Monitor output for stalls (120s)
LAST_LINE_TIME=$(date +%s)
while kill -0 "$FINALLY_PID" 2>/dev/null; do
    if [ -s "$LOG_FINALLY" ]; then
        NEW_LAST_LINE_TIME=$(stat -c %Y "$LOG_FINALLY")
        if [ "$NEW_LAST_LINE_TIME" -ne "$LAST_LINE_TIME" ]; then
            LAST_LINE_TIME=$NEW_LAST_LINE_TIME
        fi
    fi

    NOW=$(date +%s)
    DIFF=$((NOW - LAST_LINE_TIME))
    if [ "$DIFF" -ge 120 ]; then
        echo "[!] finally3.py idle for 120s, terminating..."
        kill "$FINALLY_PID" 2>/dev/null
        break
    fi

    sleep 2
done

echo "[*] Wrapper finished."

