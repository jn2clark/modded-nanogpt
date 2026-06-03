# Parked 2840-Step H5a Macro-Delta Recipe

This note parks the current 2840-step recipe so subsequent work can pivot away
from one-off weight-delta/readout methods and back toward continuous
history-style optimizer changes.

## Base

The trainer is:

```text
/tmp/train_gpt_pr311_tail_readout_2850_multipulse.py
```

The method uses the PR311 / Circuit-Muon stack:

- EMA-Nesterov + Aurora from PR309.
- Circuit-Muon on attention `v` / `attn.proj`.
- Existing Contra/Soft-Muon, SOAP, NorMuon-lite, radial/u-w-floor machinery.

## Configuration

```text
TRAIN_STEPS=2840
SCHEDULE_STEPS=2950
TAIL_SCHEDULE_SWITCH_STEP=2400
TAIL_SCHEDULE_STEPS=3030

MUON_HISTORY_STEPS=4
MUON_HISTORY_BLEND=0.25
MUON_HISTORY_LOOKAHEAD=1.0
MUON_HISTORY_MIN_COS=0.25
MUON_HISTORY_MAX_DELTA=0.25
MUON_HISTORY_DECAY_START_STEP=2400
MUON_HISTORY_DECAY_END_STEP=2840
MUON_HISTORY_FINAL_BLEND_MULT=0.0

BROAD_DELTA_SCHEDULE="500:1000:-0.005;2000:2400:-0.005"
BROAD_DELTA_SCOPE=nonembed

TRAIL_DELTA_START_STEP=2640
TRAIL_DELTA_END_STEP=2840
TRAIL_DELTA_INTERVAL=100
TRAIL_DELTA_GAMMA=-0.08
TRAIL_DELTA_GAMMA_FIRST=-0.04
TRAIL_DELTA_GAMMA_FINAL=-0.12
TRAIL_DELTA_MOMENTUM_BETA=1.0
TRAIL_DELTA_SCOPE=nonembed

FINAL_READOUT_ANCHOR_STEP=2400
FINAL_READOUT_APPLY_STEP=2840
FINAL_READOUT_ALPHA=0.12
FINAL_READOUT_SCOPE=nonembed
```

## Method

H5a is the continuous optimizer-side component. Each eligible Muon-managed
matrix keeps a short history of recent pre-polar directions. On every eligible
Muon update, it blends a small aligned and capped history/lookahead correction
into the current pre-polar matrix, then applies the usual Muon polar projection.
The history correction is tapered from step 2400 to 2840 so it is effectively
zero at the exact endpoint.

The rest of the parked recipe is endpoint/trajectory control:

- Two small BroadDelta pullbacks at `500 -> 1000` and `2000 -> 2400`.
- A late schedule horizon switch at step 2400.
- Two TrailDelta Momentum tail pulses at steps 2740 and 2840.
- A final deterministic interpolation toward the step-2400 anchor.

## Clean Results

Counted clean full runs through seed10:

| Seed | Val @ 2840 |
| -: | -: |
| 0 | 3.27799 |
| 1 | 3.27939 |
| 2 | 3.27828 |
| 3 | 3.27855 |
| 4 | 3.27774 |
| 5 | 3.27891 |
| 6 | 3.27841 |
| 7 | 3.27724 |
| 8 | 3.27878 |
| 9 | 3.27845 |
| 10 | 3.27744 |

```text
n = 11
mean = 3.27828909
(3.28 - mean) * sqrt(11) = 0.00567444
required = 0.00400000
```

Seed11 was started by the queued loop and interrupted after the seed10 result
was captured. It is a partial run and is not counted.

Earlier seed12 tail-resume probing showed `alpha=0.12` at `3.28066`, but that
was not a clean full fixed-alpha run and is not included in this parked result.

## Next Direction

Do not keep tuning this parked recipe with more one-off weight-delta/readout
changes unless a reproduction issue appears.

Next work should test continuous history-style optimizer extensions inspired by
H5a:

- Circuit-Muon V/O history before paired projection/rebalance.
- SOAP/preconditioned-direction history.
- Adam preconditioned-update history only with early taper.
- Radial/u-w signal history as a lower-risk scalar controller smoother.
