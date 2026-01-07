#!/bin/bash
# wrapper.sh - foreground, live terminal output

BASE_DIR="$HOME/pyq-cheryl"
CONNECT="$BASE_DIR/connect.py"
FINALLY="$BASE_DIR/finally3.py"
LOG_CONNECT="$BASE_DIR/connect.log"
LOG_FINALLY="$BASE_DIR/finally3.log"

rm -f "$LOG_CONNECT" "$LOG_FINALLY"

# Step 1: Run connect.py and log output live
echo "[*] Starting connect.py..."
stdbuf -oL python3 "$CONNECT" | tee -a "$LOG_CONNECT" &
CONNECT_PID=$!

# Step 2: Wait for "Connected successfully!" live
echo "[*] Waiting for login success..."
while kill -0 "$CONNECT_PID" 2>/dev/null; do
    if tail -n 10 "$LOG_CONNECT" | grep -q "Connected successfully!"; then
        echo "[*] Login successful detected!"
        kill "$CONNECT_PID" 2>/dev/null
        wait "$CONNECT_PID" 2>/dev/null
        break
    fi
    sleep 1
done

# Step 3: Run finally3.py in foreground with live output
echo "[*] Starting finally3.py..."
stdbuf -oL python3 "$FINALLY" | tee -a "$LOG_FINALLY" &
FINALLY_PID=$!

# Step 4: Monitor output for idle (120s)
LAST_LINE_TIME=$(date +%s)
while kill -0 "$FINALLY_PID" 2>/dev/null; do
    NEW_LAST_LINE_TIME=$(stat -c %Y "$LOG_FINALLY")
    if [ "$NEW_LAST_LINE_TIME" -ne "$LAST_LINE_TIME" ]; then
        LAST_LINE_TIME=$NEW_LAST_LINE_TIME
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

