# Head eigen-Adam + anchor-extrapolated gradient + fp32 master embedding, 2560 steps

## Result

Twelve fixed GH200 seeds pass Track 3 at **2560 optimizer steps**.

The score is:

`margin = (3.28 - mean_loss) * sqrt(number_of_seeds)`

| Step | Seeds | Mean loss | Margin | Required margin | Result |
|---:|---:|---:|---:|---:|:---|
| 2555 | 12 | 3.27911000 | 0.00308305 | 0.00400000 | fail |
| 2560 | 12 | 3.27878583 | 0.00420600 | 0.00400000 | pass |

The maximum passing mean for 12 seeds is `3.27884530`, so the measured mean
clears it by `0.00005947`.

Per-seed results are in `summary.tsv`. Raw logs are `GH200_seed0.txt` through
`GH200_seed11.txt`.

The result is pairwise significant against PR #341. At the same step (2600),
the mean is `3.27639000` versus `3.27877000` (n=12), giving:

`(3.27877000 - 3.27639000) / sqrt(1/12 + 1/12) = 0.00582979`

## Method

The trainer is the PR #341 trainer with the following changes.

**Head eigen-Adam (one-sided SOAP).** From step 1000 the output head is
trained with Adam in the eigenbasis of an EMA of its gradient's right
covariance:

`C = EMA_0.95(G.T @ G / V)`

The eigenbasis `Q` of `C` is refreshed every 8 steps, rotating the Adam moments
into the new basis. Adam moments are kept for `G @ Q`, and the update is mapped
back by `Q.T`. This replaces the output-KFAC gradient mix and the head's
diagonal Adam.

**Plain SOAP for attn.proj.** The attn.proj SOAP trust gate is used only
before step 1000. After that, attn.proj uses plain SOAP like the other
matrices.

**Anchor-extrapolated gradient.** Let `e` be the Tail-EMA of the weights
(`e = e + (w - e) / 100`, from step 2040). From step 2100, each step's single
forward-backward pass is taken at

`y = w + g_t * (w - e)`

and the resulting gradient is applied to `w` by the unchanged optimizer. `g_t`
ramps linearly from `0` at step 2100 to `0.45` at step 2200.

**fp32 master embedding.** From step 1000, the optimizer keeps an fp32 master
copy and fp32 Adam state for the bf16 token embedding and writes the bf16
rounding of the master to the parameter. Late in training most embedding Adam
updates are otherwise below half a bf16 ulp and round away. The parameter dtype
and forward pass are unchanged. The embedding is included in the anchor `e`
(readout blend `0`).

**Readout.** Tail-EMA `tau` is `100`. The fixed blends are `0.80` for
first-block matrices, `0.55` for other block matrices, `1.00` for auxiliary
parameters, and `0.50` for the output projection.

The submitted trainer SHA-256 is
`c299e84768cbc4fb34359d60c24fdd0de6da103ff5c219a794c07d5dcbb10b49`.

## Reproduce

```bash
STOP_STEP=2560 torchrun --standalone --nproc_per_node=1 \
  records/track_3_optimization/results/20261005_headsoap_anchor_2560/train_gpt_headsoap_anchor_2560.py \
  --seed 0
```
