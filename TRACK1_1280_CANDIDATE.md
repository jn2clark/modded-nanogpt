# Track 1 1280-Step Candidate

Base: PR315 Track 1 branch.

Candidate:

```bash
NUM_SCHEDULED_ITERATIONS=1255
NUM_EXTENSION_ITERATIONS=25
TRACK1_EMA_DECAY=0.7
TRACK1_EMA_START=1250
TRACK1_EMA_EVERY=1
TRACK1_EMA_SCOPE=readout
TRACK1_EMA_DTYPE=bf16
ADAM_COPTIM_SCALE=0
NORMUON_NOISE_SIGMA0=0
MUON_HISTORY_STEPS=0
```

Local screen used `DISABLE_FUSED_CE=1` because the local GH200 fused CE path produced NaNs. Do not include that flag for the official 8x H100 reproduction unless fused CE also fails there.

Single-GPU local fixed-seed results:

| seed | raw | ema |
| ---: | ---: | ---: |
| 1 | 3.27829 | 3.27790 |
| 2 | 3.27890 | 3.27850 |
| 3 | 3.27895 | 3.27847 |
| 4 | 3.27757 | 3.27718 |
| 5 | 3.27931 | 3.27886 |
| 6 | 3.27894 | 3.27851 |
| 7 | 3.27850 | 3.27809 |
| 8 | 3.27738 | 3.27694 |
| 9 | 3.28035 | 3.27995 |
| 10 | 3.28113 | 3.28073 |

Mean:

```text
n=10 raw_mean = 3.278932
n=10 ema_mean = 3.278513
```

Reference: PR315 reports `1275 + 15 = 1290` steps, n=10 mean around `3.27851`. This candidate matches that mean locally at `1255 + 25 = 1280`, 10 fewer steps.

8x reproduction command skeleton:

```bash
DATA_PATH=/path/to/data \
NUM_SCHEDULED_ITERATIONS=1255 \
NUM_EXTENSION_ITERATIONS=25 \
TRACK1_EMA_DECAY=0.7 \
TRACK1_EMA_START=1250 \
TRACK1_EMA_EVERY=1 \
TRACK1_EMA_SCOPE=readout \
TRACK1_EMA_DTYPE=bf16 \
ADAM_COPTIM_SCALE=0 \
NORMUON_NOISE_SIGMA0=0 \
MUON_HISTORY_STEPS=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Negative branches checked before freezing this candidate:

- 1270 schedule variants were unstable or undertrained on fixed seed2.
- `LR_FINAL_MULT=0.25` at 1275 regressed seed2 badly.
- PR322-style NorMuon exploration noise gave only a tiny 1270 seed2 improvement.
- H5/H5b Muon history was schedule-sensitive and not promotion-ready for Track 1.
- C-Optim on bigram/value embeddings regressed the hard 1270 seed.
