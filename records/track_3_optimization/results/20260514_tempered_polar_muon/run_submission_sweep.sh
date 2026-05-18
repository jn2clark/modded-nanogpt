#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-0 1 2 3 4 5 6}"
TRAIN_STEPS="${TRAIN_STEPS:-2985}"
SCHEDULE_STEPS="${SCHEDULE_STEPS:-3085}"

for seed in $SEEDS; do
  echo "=== seed ${seed} / train_steps ${TRAIN_STEPS} / schedule_steps ${SCHEDULE_STEPS} ==="
  SEED="$seed" \
  TRAIN_STEPS="$TRAIN_STEPS" \
  SCHEDULE_STEPS="$SCHEDULE_STEPS" \
    records/track_3_optimization/results/20260514_tempered_polar_muon/run_submission_candidate.sh
done
