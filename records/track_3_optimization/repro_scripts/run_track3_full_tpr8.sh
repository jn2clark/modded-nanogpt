#!/usr/bin/env bash
set -euo pipefail

# Full-horizon generalization check for the same four Track 3 optimizer variants.
# Architecture is the official Track 3 script architecture:
#   GPT(vocab_size=50304, num_layers=12, model_dim=768)
# The scripts use global batch_size = 8 * 64 * 1024 = 524,288 tokens.
# Target-param-ratio 8 gives 988,545,024 tokens, or 1,885 steps at that batch.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/track3_full_tpr8"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SEED="${SEED:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-1885}"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

run_one() {
  local name="$1"
  local script="$2"
  local log_file="${LOG_DIR}/${name}_seed${SEED}.log"

  echo "=== ${name} ==="
  echo "script=${script}"
  echo "train_steps=${TRAIN_STEPS}"
  echo "seed=${SEED}"
  echo "nproc_per_node=${NPROC_PER_NODE}"
  echo "log=${log_file}"

  TRAIN_STEPS="${TRAIN_STEPS}" torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${script}" --seed "${SEED}" 2>&1 | tee "${log_file}"
}

run_one "pr294_radial_dampen_tpr8" "records/track_3_optimization/repro_scripts/pr294_radial_dampen.py"
run_one "pr291_contra_soft_muon_tpr8" "records/track_3_optimization/repro_scripts/pr291_contra_soft_muon.py"
run_one "our_muonplus_eqrow_control_tpr8" "records/track_3_optimization/repro_scripts/our_muonplus_eqrow_control.py"
run_one "our_muonplus_eqrow_hist4_lb_taper_tpr8" "records/track_3_optimization/repro_scripts/our_muonplus_eqrow_hist4_lb_taper.py"
