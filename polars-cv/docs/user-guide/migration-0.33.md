# Migrating from 0.32 to 0.33

0.33 corrects the bootstrap intervals of **weighted** tables and adds
`weight_scheme=` to choose how a resample treats the weights. Unweighted
(unit-weight) intervals are bit-identical to 0.32, and no point estimate
changes. This page lists what to check coming from 0.32; the
[changelog](../changelog.md) has the full list.

## Requirements

- **Python `>=3.11`** (was `>=3.10`). Python 3.10 has reached end of life;
  on it, `pip` now resolves polars-cv 0.32.x instead.

## Results that change

**Weighted intervals are wider, and now the right width.** 0.32 stratified the
image-level resample on `gt_label` *crossed with* the weight cells, so every
replicate kept each cell's positive count. When the weights are estimated over
all images (`p / q̂` reweighting a vendor or prevalence mix to a target) that
count is random, and the weighted statistics depend on it: the bootstrap SE
came out 7-17% low and 95% intervals covered 0.88-0.93. The default draw
(`weight_scheme="reestimate"`) no longer crosses the label with the cells. In
a vendor-reweighting simulation through `lroc_auc_ci_lazy` the SE went from
0.93x to 1.02x the estimate's actual spread, and coverage from 0.920 to 0.940.

**Small weighted groups can report null bounds.** A draw that no longer fixes
the `gt_label` counts can, in a small group, draw no positive image (or, for
Mann-Whitney, no negative one). That replicate's statistic still had a value
(`NaN`, `0.5` or `0.0`) and was scored; it now nulls the group's bounds, as an
undefined replicate does. Each replicate draws none with probability about
`(1 − prevalence)^n`, so this needs a group of a few dozen images or fewer.
`sample_col` draws could already make such replicates in 0.32 and scored them
the same way; they now null too.

**A weight-0 image no longer nulls its group's bounds.** It formed a singleton
weight cell; cells whose weights are all zero are now exempt from the
singleton rule.

## Choosing a `weight_scheme`

| Your weights are | Use |
|---|---|
| estimated from the sample: `p / q̂` over target distributions (conditional ones included), post-stratification, raking | `"reestimate"` (default) |
| as above, and each cell's positive count was fixed by the study design | `"stratified"` (0.32's draw) |
| known in advance (design weights), or continuous (e.g. from a propensity model) | `"fixed"` |

```python
# 0.32's intervals, unchanged:
froc_auc_ci_lazy(table, method="mann_whitney", weight_scheme="stratified")

# A continuous weight: 0.32 nulled every bound (all singleton cells).
lroc_auc_ci_lazy(table, group_by="group_id", weight_scheme="fixed")

# Report.ci takes the same arguments:
report.ci("map", weight_scheme="fixed")
```

- **`weight_rtol=1e300` workarounds** (one cell for every weight, to opt out of
  weight cells) become `weight_scheme="fixed"`. `"fixed"` forms no cells, so it
  raises if given `strata` or `weight_rtol`.
- **`weight_rtol`'s default is now `None`**, meaning `1e-6` for the schemes that
  form cells. Passing `1e-6` explicitly still works the same.
- An unknown `weight_scheme` raises `ValueError`.
