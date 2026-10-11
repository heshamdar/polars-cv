# Migrating from 0.36 to 0.37

0.37 is a detection-metrics release. Nothing is removed and no signature loses
a parameter. Two results can change, both in `polars_cv.metrics`; this page
lists them. The [changelog](../changelog.md) has the full list.

## Results that change

**A null group is evaluated.** A null group key (an unrecorded site, an
unknown scanner) was already a group in curves, operating points and AUCs,
but the joins behind them did not match nulls, so its FROC/LROC curve came
out empty: sensitivity null and AUC 0 or null. It is now evaluated exactly as
it would be under any other name. If you relied on the empty row to drop
those images, filter them out before matching:

```python
data = data.filter(pl.col("scanner").is_not_null())
```

In `match_detections`, a null `image_id` is now one image.

**`match_detections(images=...)` no longer checks for duplicate images.**
With `weight=` or `group=`, a repeated `image_id` in the `images` frame used
to fail the query. Checking it ran on polars' in-memory engine and cost a
pass over `images`, so it is now the caller's contract: the frame holds one
row per image. A repeated `image_id` repeats that image's objects. If your
`images` frame might repeat ids, deduplicate it yourself:

```python
images = images.unique("image_id", keep="first")
```

## New

**`carry=` on every matcher and on `match_detections`.** It copies columns,
or expressions over them, into `image_metadata`, computed in the matcher's
own pass over the image rows. A per-image measure of the same heatmap then
needs no second query over the images, which a streaming cache would
otherwise hold whole:

```python
from polars_cv.metrics import segmentation_measures

seg = segmentation_measures("heatmap", "gt_mask")
table = matcher.match(
    data,
    pred_col="heatmap",
    gt_col="gt_mask",
    carry=["scanner", seg.struct.field("dice")],
)
FROCSensitivity(1.0).by_group(table, "scanner")
```

A carried column is a grouping key like `group_id`. Missing columns, unnamed
literals, repeated names and names `image_metadata` defines itself are
refused. See [Metrics](operations/metrics.md).

## Behaviour you get for free

- **The detection metrics stay on the streaming engine.** FROC/LROC
  interpolation, ungrouped scans, 11/101-point AP, the weighted bootstrap and
  `match_detections(images=...)` no longer fall back to the in-memory engine.
  Only `group_objects`' per-(image, class) object lists still do, because
  polars builds a list only in memory. Results are unchanged.
