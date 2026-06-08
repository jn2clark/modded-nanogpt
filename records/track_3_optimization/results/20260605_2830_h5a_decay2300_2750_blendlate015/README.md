# Track 3: 2830 Steps H5a Late-Blend Muon History

This candidate targets an exact `2830` training-step endpoint on top of the
PR311 / Circuit-Muon line. It keeps the existing macro-tail recipe and changes
the continuous optimizer behavior by retiming the H5a Muon-history taper.

The key optimizer-history change is:

```text
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
```

In words: H5a is applied continuously to eligible Muon matrices, using a
4-direction pre-polar history. It runs at full blend until step `2300`, then
uses a weaker `0.15` blend while tapering to zero by step `2750`. This keeps the
history effect as a training-time optimizer modification while reducing its
tail interference before the final TrailDelta/readout controls.

## Result

Final-only validation logs from seeds `0..15`:

| Seed | 2830 val |
| -: | -: |
| 0 | 3.27806 |
| 1 | 3.27823 |
| 2 | 3.27873 |
| 3 | 3.27843 |
| 4 | 3.27801 |
| 5 | 3.27969 |
| 6 | 3.27970 |
| 7 | 3.27757 |
| 8 | 3.28031 |
| 9 | 3.27860 |
| 10 | 3.27809 |
| 11 | 3.28013 |
| 12 | 3.28067 |
| 13 | 3.27726 |
| 14 | 3.27794 |
| 15 | 3.27838 |
| **Mean** | **3.27873750** |

Precision rule:

```text
n = 16
mean = 3.27873750
(3.28 - mean) * sqrt(16) = 0.00505000
required = 0.00400000
```

This passes the Track 3 precision condition at 2830 steps.

## Full Configuration

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
```

Frozen trainer:

```text
records/track_3_optimization/results/20260605_2830_h5a_decay2300_2750_blendlate015/train_gpt_pr311_h5a_lateblend_2830.py
```

Seed logs are in sibling directories:

```text
records/track_3_optimization/results/20260605_2830_h5a_decay2300_2750_blendlate015_seed0
...
records/track_3_optimization/results/20260605_2830_h5a_decay2300_2750_blendlate015_seed15
```

## Method Summary

The base is PR311 / Circuit-Muon:

- EMA-Nesterov and Aurora from the PR309/PR311 line.
- Circuit-Muon on attention `v` and `attn.proj`.
- Existing Contra/Soft-Muon, SOAP, NorMuon-lite, radial/u-w-floor machinery.

The winning change is a retimed H5a Muon-history schedule. H5a modifies the
Muon pre-polar matrix every eligible Muon update:

```text
M_t = current Muon pre-polar matrix
M_hist = short linear prediction from recent pre-polar matrices
M_mix = M_t + blend * clipped(M_hist - M_t)
U_t = polar(M_mix)
```

The late blend/taper matters. Plain taper `2300..2750` helped hard seeds but
made seed1 very bad. Keeping the taper while lowering the late blend to `0.15`
fixed seed1 and produced a passing n=16 mean.

## Recent Negative Probes

The following continuous-history variants did not promote:

- Circuit-history on both V/O, V-only, O-only, delayed-start, and agreement-shrink variants.
- Contra-history `1500..2400`.
- SOAP-history `1800..2400`.
- Muon-output history and Muon-output EMA.
- H5a lookahead `0.5`.
- H5a lower blend/max-delta `0.20/0.20`.
- H5a earlier/later tapers `2200..2700` and `2400..2750`.
- H5a block-limited history on early or late blocks only.
- Attention trust scalar history.

The current read is that broad all-Muon pre-polar history is useful, but it
must be weakened before the endpoint controller takes over.
