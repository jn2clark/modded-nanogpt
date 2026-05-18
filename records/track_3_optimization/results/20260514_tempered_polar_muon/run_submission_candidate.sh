#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-2985}"
SCHEDULE_STEPS="${SCHEDULE_STEPS:-3085}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-127.0.0.1:29501}"
LOG_DIR="${LOG_DIR:-logs/track3_submission}"
SCRIPT="records/track_3_optimization/results/20260514_tempered_polar_muon/train_gpt_tempered_polar_muon.py"

mkdir -p "$LOG_DIR"

TRAIN_STEPS="$TRAIN_STEPS" \
SCHEDULE_STEPS="$SCHEDULE_STEPS" \
DENSE_EVAL_START="${DENSE_EVAL_START:-2850}" \
DENSE_EVAL_END="${DENSE_EVAL_END:-$TRAIN_STEPS}" \
DENSE_EVAL_STRIDE="${DENSE_EVAL_STRIDE:-25}" \
TEMPERED_POLAR_RHO="${TEMPERED_POLAR_RHO:-0.10}" \
TEMPERED_POLAR_MAX_DELTA="${TEMPERED_POLAR_MAX_DELTA:-0.25}" \
torchrun \
  --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$RDZV_ENDPOINT" \
  "$SCRIPT" \
  --seed "$SEED" \
  2>&1 | tee "$LOG_DIR/tempered_polar_submission_step${TRAIN_STEPS}_seed${SEED}.log"
