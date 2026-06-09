# Track 3: 2830 Steps H5a Delta-Cosine Shrink

This candidate is a small live-optimizer change on top of the previous
2830-step H5a late-blend result. It keeps the PR311 / Circuit-Muon stack and
the existing endpoint controller, but makes the H5a Muon-history correction
coherence-aware.

The only method change from the locked 20260605 H5a late-blend recipe is:

```text
MUON_HISTORY_DELTA_COS_SHRINK=1
MUON_HISTORY_DELTA_COS_MIN_SCALE=0.0
```

H5a still predicts a short-horizon pre-polar Muon direction from recent
pre-polar matrices. With this change, the predicted residual
`delta = M_pred - M_t` is shrunk by its cosine agreement with the current
pre-polar direction before it is blended:

```text
scale = clamp((1 + cos(delta, M_t)) / 2, 0, 1)
M_mix = M_t + blend_t * scale * clipped(delta)
U_t = polar(M_mix)
```

This keeps the history correction active where recent Muon motion is coherent
and suppresses it where the predicted residual points against the current
direction. The change is applied continuously inside the optimizer; it is not an
extra training step or a validation-selected endpoint operation.

## Result

Clean full runs from ordered seeds `0..8`:

| Seed | 2830 val | Previous locked H5a | Delta |
| -: | -: | -: | -: |
| 0 | 3.27828 | 3.27806 | +0.00022 |
| 1 | 3.27805 | 3.27823 | -0.00018 |
| 2 | 3.27862 | 3.27873 | -0.00011 |
| 3 | 3.27930 | 3.27843 | +0.00087 |
| 4 | 3.27866 | 3.27801 | +0.00065 |
| 5 | 3.27764 | 3.27969 | -0.00205 |
| 6 | 3.27850 | 3.27970 | -0.00120 |
| 7 | 3.27793 | 3.27757 | +0.00036 |
| 8 | 3.27991 | 3.28031 | -0.00040 |

Ordered-prefix precision:

```text
n=7 mean=3.27843571
(3.28 - mean) * sqrt(7) = 0.00413871

n=8 mean=3.27837250
(3.28 - mean) * sqrt(8) = 0.00460327

n=9 mean=3.27854333
(3.28 - mean) * sqrt(9) = 0.00437000

required = 0.00400000
```

The best clean stopping point is ordered `n=8` seeds `0..7`, with seed `8`
also passing when included.

Additional hard-seed checks:

| Seed | 2830 val | Previous locked H5a | Delta |
| -: | -: | -: | -: |
| 11 | 3.27861 | 3.28013 | -0.00152 |
| 12 | 3.27860 | 3.28067 | -0.00207 |

## Configuration

Frozen trainer:

```text
records/track_3_optimization/results/20260609_2830_h5a_delta_cos_shrink/train_gpt_pr311_h5a_delta_cos_shrink_2830.py
```

Runner:

```text
records/track_3_optimization/results/20260609_2830_h5a_delta_cos_shrink/run.sh
```

Core settings:

```text
TRAIN_STEPS=2830
SCHEDULE_STEPS=2950
EMA_NESTEROV_REST_STEPS=1880

TAIL_SCHEDULE_SWITCH_STEP=2400
TAIL_SCHEDULE_STEPS=3050

BROAD_DELTA_SCHEDULE="500:1000:-0.005;2000:2400:-0.005"
BROAD_DELTA_SCOPE=nonembed

TRAIL_DELTA_START_STEP=2630
TRAIL_DELTA_END_STEP=2830
TRAIL_DELTA_INTERVAL=100
TRAIL_DELTA_GAMMA=-0.08
TRAIL_DELTA_GAMMA_FIRST=-0.04
TRAIL_DELTA_GAMMA_FINAL=-0.12
TRAIL_DELTA_MOMENTUM_BETA=0.75
TRAIL_DELTA_SCOPE=nonembed

FINAL_READOUT_ANCHOR_STEP=2400
FINAL_READOUT_APPLY_STEP=2830
FINAL_READOUT_ALPHA=0.14
FINAL_READOUT_SCOPE=nonembed

MUON_HISTORY_STEPS=4
MUON_HISTORY_BLEND=0.25
MUON_HISTORY_BLEND_LATE=0.15
MUON_HISTORY_BLEND_SWITCH_STEP=2300
MUON_HISTORY_LOOKAHEAD=1.0
MUON_HISTORY_MIN_COS=0.25
MUON_HISTORY_MAX_DELTA=0.25
MUON_HISTORY_DECAY_START_STEP=2300
MUON_HISTORY_DECAY_END_STEP=2750
MUON_HISTORY_FINAL_BLEND_MULT=0.0
MUON_HISTORY_DELTA_COS_SHRINK=1
MUON_HISTORY_DELTA_COS_MIN_SCALE=0.0
```

## Step Count Check

The logs are full from-scratch runs:

```text
save_checkpoint_steps=[]
load_checkpoint=
align_dataloader_on_load=False
exit_after_save_checkpoint=False
```

The training loop validates at `step == train_steps` and then breaks before the
next batch fetch/backward/optimizer step. The final log lines are all
`step:2830/2830 val_loss:...`, so these are 2830 optimizer steps plus the
configured deterministic final readout/validation.

## Reproduction

Run ordered seeds `0..8`:

```bash
records/track_3_optimization/results/20260609_2830_h5a_delta_cos_shrink/run.sh
```

Run a subset:

```bash
SEEDS="0 1 2 3 4 5 6 7" records/track_3_optimization/results/20260609_2830_h5a_delta_cos_shrink/run.sh
```

The logs included here are:

```text
h5a_delta_cos_shrink_seed0.txt
...
h5a_delta_cos_shrink_seed8.txt
h5a_delta_cos_shrink_seed11.txt
h5a_delta_cos_shrink_seed12.txt
```
