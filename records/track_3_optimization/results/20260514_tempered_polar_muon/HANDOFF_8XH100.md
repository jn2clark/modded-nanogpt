# 8xH100 Handoff: Exact-2900 Late-TP Tail-Velocity Candidate

## Objective

Run the current exact-2900 Track 3 candidate to submission-grade evidence.
Do not rerun the baseline. The candidate is already selected; the remaining
work is seed extension and final statistics.

The candidate is:

```text
PR #300 Aurora-on-mlp.proj base
+ schedule phase alignment for 2900
+ Contra-Muon extended to step 2750
+ late radial retune
+ very-late Tempered Polar ramp
+ final Muon-matrix tail-velocity weight transform
```

## Fixed Candidate Config

Use these exact settings:

```text
TRAIN_STEPS=2900
SCHEDULE_STEPS=3020
CONTRA_TO_NORMAL_END_STEP=2750

TEMPERED_POLAR_RHO=0.05
TEMPERED_POLAR_MAX_DELTA=0.25
TEMPERED_POLAR_RAMP_START_STEP=2600
TEMPERED_POLAR_RAMP_END_STEP=2800

RADIAL_OUTWARD_SCALE=0.5
RADIAL_OUTWARD_SCALE_LATE=0.4
RADIAL_DECAY_START_STEP=2400
RADIAL_DECAY_END_STEP=2900

TAIL_EMA_START_STEP=0
TAIL_EMA_GAMMA=0.0
TAIL_VEL_START_STEP=2500
TAIL_VEL_BETA=0.90
TAIL_VEL_GAMMA=-6.0
TAIL_VEL_MAX_DELTA_RATIO=0.01
```

Do not enable `TAIL_VEL_EVAL_GAMMAS` for the final run. The earlier gamma
probes were exploratory; the fixed method is `TAIL_VEL_GAMMA=-6.0`.

## Why This Candidate

The exact-2900 radial retune alone was research-positive but submission-negative:

```text
n=6 mean = 3.279428
precision = (3.28 - mean) * sqrt(6) = 0.001400
required = 0.004000
```

The useful move came from reducing trajectory perturbation and moving the
remaining adjustment to the tail:

- full-run Tempered Polar was not robust enough across seeds;
- late-only Tempered Polar, ramped `2600 -> 2800`, improved the endpoint mean;
- tail velocity applies a fixed final weight-space move on Muon-managed matrix
  weights only, using an EMA of late parameter updates from step 2500 onward.

## Current Evidence

Current completed local evidence for the fixed `gamma=-6` read at step 2900:

| Seed | Val loss |
|---:|---:|
| 0 | 3.27799 |
| 1 | 3.27890 |
| 2 | 3.27891 |
| 3 | 3.27788 |
| 4 | 3.27949 |
| 5 | 3.27850 |
| 6 | 3.27985 |

Aggregate:

```text
n=7
mean=3.278789
track3_precision=(3.28 - mean) * sqrt(7)=0.003205
required=0.004000
passes_track3_precision=False
```

This is not passing yet, but it is strong enough to extend. For an `n=16`
submission, the remaining 9 seeds need to average about `3.27916` or better.

Seed 7 was running locally at the time this handoff was written. If that run is
not complete or you want a clean same-hardware set, start the 8xH100 run from
seed 7. If seed 7 completes locally and you are comfortable mixing runs, start
from seed 8.

## Run Commands

Run from the repo root.

For a clean full 16-seed replay on one 8xH100 node:

```bash
SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15" \
NPROC_PER_NODE=8 \
BASE_MASTER_PORT=29600 \
records/track_3_optimization/results/20260514_tempered_polar_muon/run_pr300_2900_late_tp_tail_velocity.sh
```

For the practical continuation using existing local seeds 0-6:

```bash
SEEDS="7 8 9 10 11 12 13 14 15" \
NPROC_PER_NODE=8 \
BASE_MASTER_PORT=29600 \
records/track_3_optimization/results/20260514_tempered_polar_muon/run_pr300_2900_late_tp_tail_velocity.sh
```

If local seed 7 has already completed and is accepted into the aggregate:

```bash
SEEDS="8 9 10 11 12 13 14 15" \
NPROC_PER_NODE=8 \
BASE_MASTER_PORT=29600 \
records/track_3_optimization/results/20260514_tempered_polar_muon/run_pr300_2900_late_tp_tail_velocity.sh
```

The script supports `NPROC_PER_NODE=1,2,4,8`; it keeps the global batch size
fixed and asserts that `world_size` divides 8.

## Logs

The runner writes tee logs to:

```text
logs/track3_submission/
```

The train script also prints a `logs/<uuid>.txt` path at startup. That UUID log
contains the full self-logged train script and should be preserved for final PR
evidence.

The clean-run tee log pattern is:

```text
logs/track3_submission/pr300_tempered_polar_radial_xewa2900_s0_b0p97_g0p0_md0p02_vels2500_vb0p90_vg-6p0_vmd0p01_tpr2600_2800_rada0_rade0_rk0p0_rm0p0_rt1p0_ags0_ab0p95_ac0p20_aa0p0_seed${SEED}.log
```

## Stats

After a clean 16-seed run, compute:

```bash
python3 records/track_3_optimization/results/20260514_tempered_polar_muon/summarize_submission_stats.py \
  --step 2900 \
  --eval-kind tail_vel \
  --gamma -6 \
  logs/track3_submission/pr300_tempered_polar_radial_xewa2900_s0_b0p97_g0p0_md0p02_vels2500_vb0p90_vg-6p0_vmd0p01_tpr2600_2800_rada0_rade0_rk0p0_rm0p0_rt1p0_ags0_ab0p95_ac0p20_aa0p0_seed{0..15}.log
```

Track 3 pass condition:

```text
(3.28 - mean) * sqrt(n) >= 0.004
```

Useful mean cutoffs:

| n | Required mean |
|---:|---:|
| 8 | <= 3.278586 |
| 10 | <= 3.278735 |
| 12 | <= 3.278845 |
| 14 | <= 3.278931 |
| 16 | <= 3.279000 |

If the candidate passes before seed 15, keep the later completed logs anyway;
they are useful evidence and may improve the reported p-value.

## Submission Notes

The submission writeup should emphasize:

- fixed checkpoint: `step=2900`;
- no baseline rerun;
- dataset, model architecture, batch size, and loss computation unchanged;
- one forward/backward pass per step;
- no validation-feedback early stopping or seed-specific decisions;
- final tail-velocity transform is fixed by config and applied uniformly to all
  Muon-managed matrix weights;
- extra gamma probes from exploratory logs are not part of the frozen candidate.

