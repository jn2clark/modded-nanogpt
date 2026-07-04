#!/bin/bash
# Official runs (single GH200; math is world-size independent):
for SEED in 0 1 2 3 4 5 6 7; do
  MODDED_NANOGPT_SEED=$SEED MODDED_NANOGPT_STEP_CAP=2750 \
  MODDED_NANOGPT_FINE_VAL_START_STEP=2600 MODDED_NANOGPT_FINE_VAL_STEP_FREQ=5 \
  torchrun --standalone --nproc_per_node=1 train_gpt_uwramp_tailema_2650.py
done
