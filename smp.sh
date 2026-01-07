#!/bin/bash
BASE_DIR="$HOME/pyq-cheryl"
CONNECT="$BASE_DIR/connect.py"
FINALLY="$BASE_DIR/finally3.py"

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

echo "[*] Starting finally3.py..."
python3 -u "$FINALLY"

