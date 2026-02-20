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
        - image_ids_and_strata
        - collect

::: polars_cv.metrics.MetricResult
    options:
      show_root_heading: true
      members:
        - auc
        - partial_auc
        - interpolate
        - summary_table

::: polars_cv.metrics.BootstrapResult
    options:
      show_root_heading: true

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

::: polars_cv.metrics.froc_curve
    options:
      show_root_heading: true

::: polars_cv.metrics.lroc_curve
    options:
      show_root_heading: true

::: polars_cv.metrics.confusion_at_threshold
    options:
      show_root_heading: true

## Bootstrap

::: polars_cv.metrics.bootstrap_metric_sequential
    options:
      show_root_heading: true

::: polars_cv.metrics.bootstrap_pr_auc
    options:
      show_root_heading: true
