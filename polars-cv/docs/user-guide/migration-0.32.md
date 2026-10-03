# Migrating from 0.31 to 0.32

0.32 is mostly the detection-metrics layer: one-call evaluation
(`evaluate_detections`, `evaluate_heatmaps`, `evaluate_segmentation`), object
tables in (`match_detections`), IoU sweeps, COCO 101-point AP, and one
bootstrap engine for any `Statistic`. The breaking changes are a higher polars
floor, a smaller `to_per_image()` frame, and metric values that were wrong
before and are now computed as documented. This page lists what to check
coming from 0.31; the [changelog](../changelog.md) has the full list.

## Requirements

- **polars `>=1.44.2`** (was `>=1.43.2`). On 1.43.2 a cached lazy frame read
  under different projections returned wrong rows, which the bootstrap
  intervals now rely on.

## Results that change

**Multi-class matching no longer duplicates detections.** `BBoxMatcher` and
`ContourMatcher` attached `class_id` to detections by `image_id` alone, so on
an image with *k* classes every detection appeared *k* times. Per-class AP,
mAP and FROC from any multi-class evaluation change, and are now correct.
Single-class results are unchanged.

**All-points AP integrates the step rule.** `Σ (Rₖ − Rₖ₋₁)·P̂ₖ`, as Pascal
VOC 2010+ and COCO do, where it used to average neighbouring precisions (a
trapezoid). The two agree when scores are distinct. When a block of equal
scores mixes TPs and FPs, AP is now lower: TP@0.9 then {TP, FP, FP}@0.5
against 2 GTs read 0.875 and is now 0.75. Heatmap-derived scores tie often.

**The 11-point grid is `i · 0.1`** (the VOC devkit's), not `i / 10`. A recall
landing exactly on 0.3, 0.6 or 0.7 can compare differently.

**Precision-recall metrics are weighted.** `precision_recall_curve`,
`average_precision`, `mean_average_precision`, the `*_at_threshold`
functions, `confusion_at_threshold` and `average_precision_ci_lazy` now read
`image_metadata.weight`, as FROC and LROC already did. Unit weights give the
same values as before. `ConfusionResult.precision`/`recall`/`f1` read the
weighted counts; the raw `tp`/`fp`/`fn` are unchanged.

**Bootstrap intervals.**

- Multi-class and grouped tables drew wrongly: an image positive for one
  class and negative for another had two draw chances per replicate, and a
  grouped draw brought the image's other groups' rows. Intervals for those
  tables change. Single-class, ungrouped intervals are unchanged.
- The resample is stratified on weight cells. A group in which one weight
  cell holds a single image (a continuous weight, or a one-image group) now
  reports null `ci_lower`/`ci_upper`, keeping its point estimate, where it
  used to report a zero-width interval.

**`.contour.boundary_distances(frame=)` is continuous at the frame.** A
sample just inside the frame is now measured to the other side's whole
boundary. It used to be measured across the region, so annotations drawn a
pixel inside the image edge read about 30 px instead of 1 px in the reported
case. `mean_*`, `hd` and `hd95` near the frame drop to their true values.

**`DetectionTable.to_per_image()`'s `top_is_tp`** breaks a tie for the top
score by the lowest `det_idx` (the matcher's ranking). It used to depend on
input order, and so did LROC's `top_scoring` variant.

## Changed shapes

- **`DetectionTable.to_per_image()`** returns `max_score`, `top_is_tp` and
  `best_tp_score`, and no longer the `detections` list of structs. Read those
  columns instead of unpacking the list.
- **`BBoxMatcher` tables** no longer hold a null-score row for an image
  without predictions. The image stays in `image_metadata`, so the
  denominators are unchanged.
- **`PrecisionRecallResult.curve`** gains `cum_weighted_tp` and
  `cum_weighted_fp`; `PrecisionRecallResult` gains `weighted_gts`;
  `ConfusionResult` gains `weighted_tp`/`weighted_fp`/`weighted_fn`.

## Newly refused

- **A sweep pooled across thresholds.** A table matched at several IoU
  thresholds (`iou_threshold=[...]`) refuses any evaluation that would count
  each ground truth once per threshold. Group by `iou_threshold`, average over
  it (`mean_ap()`), or select one with `.at_iou_threshold(t)`.
- **An instance mask with several regions** (`match_detections` with a mask
  geometry) now fails with a `polars.exceptions.ComputeError` raised by the
  plugin (`.contour.single`), where it used to be a `ValueError` from a Python
  UDF. Catch the new type if you relied on the old one.
- **A repeated `image_id` in `match_detections(images=...)`** with `weight=`
  or `group=` fails the query. It used to repeat that image's detections and
  ground truth.
