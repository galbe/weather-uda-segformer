#!/bin/bash
# Chain: wait for step1 → run step2 → run eval. All auto-resume on crash.
set -e
cd "$(dirname "$0")"

LOG=outputs/chain.log
PY=~/venvs/uda/bin/python

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== CHAIN START ==="

# ── Wait for step1 to finish ─────────────────────────────────────────────────
STEP1_LOG=outputs/step1_gta2cs_b2_full/train.log
STEP1_BEST=outputs/step1_gta2cs_b2_full/best.pth

log "Waiting for step1 to finish (polling $STEP1_LOG)..."
while true; do
    if grep -q "^Done\." "$STEP1_LOG" 2>/dev/null; then
        log "Step1 done."
        break
    fi
    sleep 60
done

if [ ! -f "$STEP1_BEST" ]; then
    log "ERROR: $STEP1_BEST not found after step1 finished. Aborting."
    exit 1
fi
log "Step1 best checkpoint: $STEP1_BEST"

# ── Step 2: CS → ACDC ────────────────────────────────────────────────────────
STEP2_OUT=outputs/step2_cs2acdc_b2_full
STEP2_RESUME="$STEP2_OUT/resume.pth"

log "Starting step2 (CS→ACDC, full-data)..."
mkdir -p "$STEP2_OUT"
while true; do
    if [ -f "$STEP2_RESUME" ]; then
        # auto-resume handles it — just launch without --resume flag
        $PY train_uda.py --config configs/uda_cs2acdc_b2_full.yaml \
            2>&1 | tee -a "$STEP2_OUT/train_chain.log"
    else
        # first run — initialize from step1 best
        $PY train_uda.py --config configs/uda_cs2acdc_b2_full.yaml \
            --resume "$STEP1_BEST" \
            2>&1 | tee -a "$STEP2_OUT/train_chain.log"
    fi

    if grep -q "^Done\." "$STEP2_OUT/train.log" 2>/dev/null; then
        log "Step2 done."
        break
    fi
    log "Step2 exited before finishing — will retry in 30s (auto-resume will pick up)..."
    sleep 30
done

# ── Final eval ───────────────────────────────────────────────────────────────
log "Running final eval..."
$PY evaluate.py \
    --before "$STEP1_BEST" \
    --after  "$STEP2_OUT/best.pth" \
    2>&1 | tee -a "$LOG"

log "=== ALL DONE ==="
