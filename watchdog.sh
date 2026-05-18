#!/bin/bash
# Watchdog: kills training if the log hasn't been updated for TIMEOUT seconds.
# Usage: bash watchdog.sh <log_file> <timeout_seconds> <pid>

LOG=$1
TIMEOUT=${2:-600}   # default 10 minutes
PID=$3

# Kill any OTHER train_uda.py processes that aren't ours or children of ours (orphans from old sessions)
OWN_CHILDREN=$(pgrep -P "$PID" 2>/dev/null | tr '\n' '|')
for OPID in $(pgrep -f 'train_uda.py' 2>/dev/null); do
    if [ "$OPID" = "$PID" ]; then continue; fi
    # Skip dataloader workers (children of our training process)
    PPID_OF=$(ps -o ppid= -p "$OPID" 2>/dev/null | tr -d ' ')
    if [ "$PPID_OF" = "$PID" ]; then continue; fi
    echo "[watchdog] killing orphan training process $OPID"
    kill "$OPID" 2>/dev/null
done

echo "[watchdog] monitoring $LOG (pid=$PID, timeout=${TIMEOUT}s)"

while true; do
    sleep 60

    # Stop if the training process is already gone
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "[watchdog] training process $PID exited — done"
        exit 0
    fi

    LAST_MOD=$(stat -c %Y "$LOG" 2>/dev/null)
    NOW=$(date +%s)
    AGE=$(( NOW - LAST_MOD ))

    echo "[watchdog] log last updated ${AGE}s ago"

    if [ "$AGE" -ge "$TIMEOUT" ]; then
        echo "[watchdog] STALE for ${AGE}s — killing training (pid=$PID) and its children"
        pkill -P "$PID" 2>/dev/null
        kill "$PID" 2>/dev/null
        echo "[watchdog] killed — machine should stay responsive"
        exit 1
    fi
done
