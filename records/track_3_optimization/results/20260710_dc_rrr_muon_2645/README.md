# Track 3: projected Tail-EMA, 2645 steps

This is a 15-step improvement over the open 2660-step result in PR #331.

## Result

- 19 consecutive seeds (`0-18`) on one GH200
- Step 2645 mean validation loss: `3.27903474`
- Formal statistic: `(3.28 - mean) * sqrt(19) = 0.00420748`
- Required statistic: `>= 0.004`
- Step 2640 fails with `0.00279887`; step 2645 passes

## Change

The optimizer stack from PR #331 is retained. The late readout now uses a
Veit Elser-style nonconvex projection idea: form a parameter-group-weighted
consensus between the current weights and Tail-EMA, then project each 2D
parameter row back onto its EMA row-norm sphere. Tail-EMA is initialized from
the late optimizer velocity. The final learning-rate phase is also retarded
linearly by up to 180 schedule steps.

The dataset, batch size, architecture, and one-forward/backward-pass-per-step
rule are unchanged.

## Reproduce

```bash
bash records/track_3_optimization/results/20260710_dc_rrr_muon_2645/run_formal.sh
```

`summary.tsv` contains the formal aggregate. `GH200_seed0.txt` through
`GH200_seed18.txt` are the full logs with embedded source.
