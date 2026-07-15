# Bi-Maxwell + output KFAC, 2625 steps

This is the Track 3 trainer from PR #339 plus an output-head activation-covariance preconditioner.

## Result

Twelve fixed GH200 seeds give a first passing step of **2625**, 10 steps earlier than the 2635-step base result.

The scoring margin is:

`margin = (3.28 - mean_loss) * sqrt(number_of_seeds)`

A step passes when `margin >= 0.004`.

| Step | Mean loss | Margin | Result |
|---:|---:|---:|:---|
| 2615 | 3.27934333 | 0.00227476 | fail |
| 2620 | 3.27898667 | 0.00351029 | fail |
| 2625 | 3.27860583 | 0.00482954 | pass |
| 2630 | 3.27826500 | 0.00601022 | pass |
| 2635 | 3.27788583 | 0.00732369 | pass |
| 2640 | 3.27752417 | 0.00857654 | pass |

Per-seed results are in `summary.tsv`. Raw logs are `GH200_seed0.txt` through `GH200_seed11.txt`.

## Method

The base dual-timescale momentum is unchanged from PR #339:

`m = 0.4385 * m_fast + 0.5615 * m_slow`

From step 2000 through 2639, the output-head input covariance is updated with:

`C = 0.9 * C + 0.1 * (X.T @ X / N)`

Every eight steps, the output gradient is transformed by a damped inverse square root of `C`, norm matched, mixed at strength `0.75`, and norm matched again:

`g_used = norm_match(0.25 * g + 0.75 * norm_match(g * P, g), g)`

`P = (C + 0.05 * mean_eigenvalue(C) * I)^(-1/2)`

Activations are subsampled by 16 for the covariance estimate. The transform changes direction but preserves the output-gradient norm.

The submitted trainer SHA-256 is `a0d1d9d41e0e87544d3a8bf02be867ea278080bc95be0f9a2ee8fdc6327d5ad7`.
