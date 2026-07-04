# Record: Track 3 Optimization — Tail heat ramp + retuned Tail-EMA window — 2650 steps (n=8)

## TL;DR

This is the PR [#331](https://github.com/KellerJordan/modded-nanogpt/pull/331) /
[#328](https://github.com/KellerJordan/modded-nanogpt/pull/328) SOAP-Muon + Tail-EMA lineage with a
**17-line diff** and no new optimizer: two of the existing components are re-tuned and two small tail
mechanisms are added. On n = 8 seeds (0–7) the formal Track 3 statistic first passes at **2650 steps**
(PR #331: 2660):

```text
mean val_loss at 2650 = 3.278535   (3.28 - mean) * sqrt(8) = 0.00414 >= 0.004
mean val_loss at 2645 = 3.278795   (3.28 - mean) * sqrt(8) = 0.00341  (fails)
```

All runs on a single GH200 (world-size-independent batch; matches 8-GPU math up to FP summation
order). The same recipe beats PR #331's published per-seed H200 values by −0.0008 on average on
every shared seed, including their worst (seed 4).

## Changes vs PR #331

| field | PR #331 | this PR | effect (paired same-seed tails) |
|---|---:|---:|---:|
| `TAILEMA_START` / `TAILEMA_TAU` / `TAILEMA_LAMBDA` | 2400 / 120 / 0.65 | **2000 / 150 / 0.75** | ≈ −0.0007 @2650 |
| `TARGET_UW` | 0.3825 const | **ramp 0.3825 → 0.75 over 2350→2660** | ≈ −0.0006 |
| `SOAP_DENOM_POWER` | 0.50 const | **ramp 0.50 → 0.72 (same window)** | ≈ −0.00005 |
| attn.proj trust gate | adaptive throughout | **forced to 1.0 (full SOAP) from step 2400** | ≈ −0.00008 |

Everything else (SOAP-f1 on all hidden matrices, RowFloor, post-pin Cautious WD 0.025, radius pin,
EMA-Nesterov lookahead, PowerCool LR with Muon horizon 2875 / power 1.183, μ schedule, init scheme)
is unchanged from PR #331.

## Mechanism

1. **Tail-EMA window fix.** PR #331's readout EMA starts at 2400 with τ=120, so at the 2660 claim
   the EMA has only 2.2τ of history — `(1−1/120)^250 ≈ 12%` of the readout is still the stale
   step-2400 snapshot. The τ 150→120 retune in #331 was compensating for this init bias. Starting
   at 2000 (>4τ) removes the bias; τ=150 then beats τ=120 at every claim step, and exact algebraic
   debiasing of the init term makes things *worse* — the window wants more history, not less.
2. **Heat through the floor.** With RowFloor active, the tail is LR-invariant: shifting the Muon
   schedule horizon by ±30 steps changes raw val loss by <2e-5. The per-row u/w floor is the
   effective step-size (temperature) knob. Ramping `TARGET_UW` to 0.75 makes the **raw** trajectory
   ≈ +0.006 *worse* (wider orbit) while the blended readout gets ≈ −0.0006 *better*: the EMA removes
   the added oscillation and keeps the faster orbit-center descent that larger steps buy. Dose
   response is monotone to 0.75 and saturates (0.85 ≡ 0.75); the saturation is global — reallocating
   heat between MLP and attention families, or per-row via a boost cap, is neutral or worse.
3. **Tail whitening + gate release** (small): stronger SOAP whitening (denom power → 0.72) points the
   enlarged orbit into flatter directions; and the attn.proj trust gate — early-training machinery —
   is over-conservative once the preconditioner has converged (forced full-SOAP < adaptive <
   momentum-only, on both probe seeds).

## Result (n = 8, seeds 0–7, dense val every 5 steps)

| seed | 2640 | 2645 | 2650 | 2655 | 2660 |
|---:|---:|---:|---:|---:|---:|
| 0 | 3.27826 | 3.27795 | 3.27770 | 3.27738 | 3.27710 |
| 1 | 3.27910 | 3.27881 | 3.27854 | 3.27821 | 3.27793 |
| 2 | 3.27903 | 3.27870 | 3.27845 | 3.27812 | 3.27785 |
| 3 | 3.27818 | 3.27786 | 3.27760 | 3.27727 | 3.27699 |
| 4 | 3.28104 | 3.28073 | 3.28047 | 3.28016 | 3.27987 |
| 5 | 3.27819 | 3.27787 | 3.27761 | 3.27730 | 3.27704 |
| 6 | 3.27862 | 3.27831 | 3.27805 | 3.27772 | 3.27744 |
| 7 | 3.28045 | 3.28013 | 3.27986 | 3.27955 | 3.27927 |
| **mean** | **3.279109** | **3.278795** | **3.278535** | **3.278214** | **3.277936** |
| **sig** | 0.00252 | 0.00341 | **0.00414** | 0.00505 | 0.00584 |

**First passing step = 2650.**

## Ablations / negative results (all paired same-seed tails from shared checkpoints, ±2–3e-5 resolution)

Tested and neutral-or-worse on this stack: Muon LR horizon ±30 (raw <2e-5 — the floor pins the
tail), earlier μ-cooldown and heat-then-quench floor schedules (raw improves, readout worse — the
readout already extracts the orbit center), μ_max 0.97, CWD 0.030 late, RowFloor ρ 1.1, boost-cap
(clamping the RowFloor multiplier), per-family heat splits, radial-scale asymmetry retunes,
per-coordinate Wiener readout, velocity-corrected (per-coordinate BEMA) readout, init-debiased EMA,
two-timescale EMA mixes, split-λ (head vs blocks), embedding-EMA readout, head/embed LR horizon
shifts.

## Files

- `train_gpt_uwramp_tailema_2650.py` — self-contained solution (17-line diff vs PR #331's script).
- `run.sh` — launch command used for the official runs.
- `official_seed{0..7}.stdout`, `uuid_seed{0..7}.txt` — full logs with embedded source (verified
  to match the artifact).
- `claim_stats.txt`, `summary.tsv` — formal statistic table.

## Credits

Direct retune/extension of PR #331 (@didiforgithub) and PR #328 (@ypwang61), which build on the
Tail-EMA eval readout from PR #325 (this account) and the SOAP-Muon clean base of PR #321
(@ypwang61). The heat-ramp observation extends #325's finding that weight-averaging makes tail
temperature nearly free.
