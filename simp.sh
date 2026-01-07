#!/bin/bash
BASE_DIR="$HOME/pyq-cheryl"
CONNECT="$BASE_DIR/connect.py"
FINALLY="$BASE_DIR/finally3.py"

echo "[*] Starting connect.py..."
# Run connect.py in foreground, read stdout line by line
python3 "$CONNECT" | while IFS= read -r line; do
    echo "$line"  # show output in terminal

    if [[ "$line" == *"Login successful"* ]]; then
        echo "[*] Login successful detected! Terminating connect.py..."
        pkill -f "$CONNECT"
        break
    fi
done

# Short delay to ensure process is gone
sleep 1

echo "[*] Starting finally3.py..."
# Run finally3.py in foreground
python3 "$FINALLY"

