# Metrics API Reference

## Core Types

::: polars_cv.metrics.DetectionTable
    options:
      show_root_heading: true
      members:
        - from_matched
        - with_group
        - filter_class
        - class_ids
        - at_iou_threshold
        - to_per_image
        - collect

::: polars_cv.metrics.MetricResult
    options:
      show_root_heading: true
      members:
        - auc
        - partial_auc
        - interpolate
        - summary_table

## Matchers

::: polars_cv.metrics.ContourMatcher
    options:
      show_root_heading: true
      members:
        - match

::: polars_cv.metrics.BBoxMatcher
    options:
      show_root_heading: true
      members:
        - match

::: polars_cv.metrics.PreMatchedAdapter
    options:
      show_root_heading: true
      members:
        - match

## Metric Functions

::: polars_cv.metrics.precision_recall_curve
    options:
      show_root_heading: true

::: polars_cv.metrics.average_precision
    options:
      show_root_heading: true

::: polars_cv.metrics.mean_average_precision
    options:
      show_root_heading: true

::: polars_cv.metrics.precision_at_threshold
    options:
      show_root_heading: true

::: polars_cv.metrics.recall_at_threshold
    options:
      show_root_heading: true

::: polars_cv.metrics.f1_at_threshold
    options:
      show_root_heading: true

::: polars_cv.metrics.froc_auc
    options:
      show_root_heading: true

::: polars_cv.metrics.froc_curve_lazy
    options:
      show_root_heading: true

::: polars_cv.metrics.froc_sensitivity_at_fp
    options:
      show_root_heading: true

::: polars_cv.metrics.froc_summary_table
    options:
      show_root_heading: true

::: polars_cv.metrics.lroc_auc
    options:
      show_root_heading: true

::: polars_cv.metrics.lroc_curve_lazy
    options:
      show_root_heading: true

::: polars_cv.metrics.lroc_sensitivity_at_fpf
    options:
      show_root_heading: true

::: polars_cv.metrics.confusion_at_threshold
    options:
      show_root_heading: true

## Bootstrap confidence intervals (lazy, group-aware)

::: polars_cv.metrics.froc_auc_ci_lazy
    options:
      show_root_heading: true

::: polars_cv.metrics.lroc_auc_ci_lazy
    options:
      show_root_heading: true

::: polars_cv.metrics.average_precision_ci_lazy
    options:
      show_root_heading: true
