# 2850 Tail Readout Beta 0.75

Author: Jesse Clark

This Track 3 submission keeps the benchmark data stream, global batch size, and
model architecture unchanged. It uses the PR311-style 2850-step TrailDelta and
final-readout recipe, with the TrailDelta momentum coefficient set to `0.75`.

The final fixed validation point is step `2850`.

## Result

The submitted logs are clean full runs from seeds 0 through 12. The runs were
launched sequentially on one node with 8 local processes. No run was stopped
early, dropped, or selected based on validation loss.

Under the Track 3 precision rule:

```text
n=13
mean=3.278719
(3.28 - mean) * sqrt(13) = 0.004618
required = 0.004000
```

Using the README's `sigma=0.0013` normal approximation:

```text
z = 3.552215
p = 0.000191002
```

| Seed | Log | 2850 val | Train time | Step avg |
| -: | - | -: | -: | -: |
| 0 | [tail_readout_beta075_2850_seed0.txt](tail_readout_beta075_2850_seed0.txt) | 3.27711 | 812.903s | 285.23ms |
| 1 | [tail_readout_beta075_2850_seed1.txt](tail_readout_beta075_2850_seed1.txt) | 3.27894 | 806.382s | 282.94ms |
| 2 | [tail_readout_beta075_2850_seed2.txt](tail_readout_beta075_2850_seed2.txt) | 3.27806 | 805.868s | 282.76ms |
| 3 | [tail_readout_beta075_2850_seed3.txt](tail_readout_beta075_2850_seed3.txt) | 3.27789 | 809.434s | 284.01ms |
| 4 | [tail_readout_beta075_2850_seed4.txt](tail_readout_beta075_2850_seed4.txt) | 3.27833 | 810.459s | 284.37ms |
| 5 | [tail_readout_beta075_2850_seed5.txt](tail_readout_beta075_2850_seed5.txt) | 3.27795 | 813.212s | 285.34ms |
| 6 | [tail_readout_beta075_2850_seed6.txt](tail_readout_beta075_2850_seed6.txt) | 3.27911 | 807.496s | 283.33ms |
| 7 | [tail_readout_beta075_2850_seed7.txt](tail_readout_beta075_2850_seed7.txt) | 3.27851 | 813.939s | 285.59ms |
| 8 | [tail_readout_beta075_2850_seed8.txt](tail_readout_beta075_2850_seed8.txt) | 3.27979 | 829.170s | 290.94ms |
| 9 | [tail_readout_beta075_2850_seed9.txt](tail_readout_beta075_2850_seed9.txt) | 3.27846 | 812.397s | 285.05ms |
| 10 | [tail_readout_beta075_2850_seed10.txt](tail_readout_beta075_2850_seed10.txt) | 3.27834 | 804.917s | 282.43ms |
| 11 | [tail_readout_beta075_2850_seed11.txt](tail_readout_beta075_2850_seed11.txt) | 3.28001 | 810.111s | 284.25ms |
| 12 | [tail_readout_beta075_2850_seed12.txt](tail_readout_beta075_2850_seed12.txt) | 3.28085 | 810.867s | 284.51ms |
| **Mean** |  | **3.278719** |  |  |

## Configuration

Frozen runner:

```text
records/track_3_optimization/results/20260529_tail_readout_beta075_2850/run.sh
```

Frozen trainer:

```text
records/track_3_optimization/results/20260529_tail_readout_beta075_2850/train_gpt_tail_readout_beta075_2850.py
```

Core settings:

```text
TRAIN_STEPS=2850
SCHEDULE_STEPS=2950
TAIL_SCHEDULE_SWITCH_STEP=2600
TAIL_SCHEDULE_STEPS=3010

TRAIL_DELTA_START_STEP=2650
TRAIL_DELTA_END_STEP=2850
TRAIL_DELTA_INTERVAL=100
TRAIL_DELTA_GAMMA=-0.08
TRAIL_DELTA_GAMMA_FIRST=-0.04
TRAIL_DELTA_GAMMA_FINAL=-0.12
TRAIL_DELTA_MOMENTUM_BETA=0.75
TRAIL_DELTA_SCOPE=nonembed

FINAL_READOUT_ANCHOR_STEP=2400
FINAL_READOUT_ALPHA=0.08
FINAL_READOUT_SCOPE=nonembed

BROAD_DELTA_GAMMA=0
PHASE_READOUT_KAPPA=0
```

## Method

This candidate is built on the PR311-style optimizer stack with PR309
EMA-Nesterov plus Aurora and Circuit-Muon on the attention V/O pair.

The fixed endpoint mechanics are:

- Capture the final readout anchor at step `2400`.
- Switch the late schedule at step `2600` to a `3010` horizon.
- Capture the TrailDelta reference at step `2650`.
- Apply the first TrailDelta pulse at step `2750` with `gamma=-0.04`.
- Apply the final TrailDelta momentum pulse at step `2850` with
  `gamma=-0.12` and `beta=0.75`.
- Evaluate after an 8% non-embedding readout toward the step-2400 anchor.

BroadDelta, phase readout, logit scale, MacroAccum, and split-scope final
readout are disabled in this locked candidate.

## Reproduction

Run the submitted seeds:

```bash
records/track_3_optimization/results/20260529_tail_readout_beta075_2850/run.sh
```

Run a shorter check:

```bash
SEEDS="8 12" records/track_3_optimization/results/20260529_tail_readout_beta075_2850/run.sh
```

Recompute the reported statistics:

```bash
python3 records/track_3_optimization/results/20260529_tail_readout_beta075_2850/summarize_submission_stats.py --step 2850 \
  records/track_3_optimization/results/20260529_tail_readout_beta075_2850/tail_readout_beta075_2850_seed0.txt \
  records/track_3_optimization/results/20260529_tail_readout_beta075_2850/tail_readout_beta075_2850_seed1.txt \
  records/track_3_optimization/results/20260529_tail_readout_beta075_2850/tail_readout_beta075_2850_seed2.txt \
  records/track_3_optimization/results/20260529_tail_readout_beta075_2850/tail_readout_beta075_2850_seed3.txt \
  records/track_3_optimization/results/20260529_tail_readout_beta075_2850/tail_readout_beta075_2850_seed4.txt \
  records/track_3_optimization/results/20260529_tail_readout_beta075_2850/tail_readout_beta075_2850_seed5.txt \
  records/track_3_optimization/results/20260529_tail_readout_beta075_2850/tail_readout_beta075_2850_seed6.txt \
  records/track_3_optimization/results/20260529_tail_readout_beta075_2850/tail_readout_beta075_2850_seed7.txt \
  records/track_3_optimization/results/20260529_tail_readout_beta075_2850/tail_readout_beta075_2850_seed8.txt \
  records/track_3_optimization/results/20260529_tail_readout_beta075_2850/tail_readout_beta075_2850_seed9.txt \
  records/track_3_optimization/results/20260529_tail_readout_beta075_2850/tail_readout_beta075_2850_seed10.txt \
  records/track_3_optimization/results/20260529_tail_readout_beta075_2850/tail_readout_beta075_2850_seed11.txt \
  records/track_3_optimization/results/20260529_tail_readout_beta075_2850/tail_readout_beta075_2850_seed12.txt
```

## Credits

This submission builds on the existing Track 3 optimizer lineage, including the
PR311 recipe, PR309 EMA-Nesterov, Aurora, Circuit-Muon, and prior TrailDelta and
final-readout experiments.
