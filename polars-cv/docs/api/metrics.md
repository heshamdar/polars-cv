# Metrics API Reference

## Evaluation

::: polars_cv.metrics.evaluate_detections
    options:
      show_root_heading: true

::: polars_cv.metrics.evaluate_heatmaps
    options:
      show_root_heading: true

::: polars_cv.metrics.evaluate_segmentation
    options:
      show_root_heading: true

::: polars_cv.metrics.segmentation_measures
    options:
      show_root_heading: true

::: polars_cv.metrics.DetectionReport
    options:
      show_root_heading: true
      members:
        - summary
        - per_class
        - per_threshold
        - ci
        - matches
        - pr_curve
        - froc
        - confusion
        - at

::: polars_cv.metrics.SegmentationReport
    options:
      show_root_heading: true
      members:
        - summary
        - ci

::: polars_cv.metrics.match_detections
    options:
      show_root_heading: true

::: polars_cv.metrics.group_objects
    options:
      show_root_heading: true

## Statistics

::: polars_cv.metrics.Statistic
    options:
      show_root_heading: true
      members:
        - by_group
        - value
        - support

::: polars_cv.metrics.AP
    options:
      show_root_heading: true

::: polars_cv.metrics.Recall
    options:
      show_root_heading: true

::: polars_cv.metrics.PrecisionAt
    options:
      show_root_heading: true

::: polars_cv.metrics.RecallAt
    options:
      show_root_heading: true

::: polars_cv.metrics.F1At
    options:
      show_root_heading: true

::: polars_cv.metrics.FROCSensitivity
    options:
      show_root_heading: true

::: polars_cv.metrics.CPM
    options:
      show_root_heading: true

::: polars_cv.metrics.FROCAUC
    options:
      show_root_heading: true

::: polars_cv.metrics.LROCAUC
    options:
      show_root_heading: true

::: polars_cv.metrics.LROCSensitivity
    options:
      show_root_heading: true

::: polars_cv.metrics.MeanOver
    options:
      show_root_heading: true
      members:
        - per_facet

::: polars_cv.metrics.mean_ap
    options:
      show_root_heading: true

::: polars_cv.metrics.bootstrap_ci
    options:
      show_root_heading: true

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
        - iou_thresholds
        - frames
        - stack
        - filter_images
        - to_per_image
        - collect

::: polars_cv.metrics.MetricResult
    options:
      show_root_heading: true
      members:
        - auc

::: polars_cv.metrics.PrecisionRecallResult
    options:
      show_root_heading: true
      members:
        - auc
        - precision_at
        - recall_at

::: polars_cv.metrics.ConfusionResult
    options:
      show_root_heading: true
      members:
        - precision
        - recall
        - f1
        - to_dict

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

::: polars_cv.metrics.froc_operating_range
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
