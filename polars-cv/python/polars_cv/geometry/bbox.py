"""Bounding-box operations namespace for Polars expressions.

Provides the ``.bbox`` accessor for operations on ``List[BBOX_SCHEMA]`` columns.
"""

from __future__ import annotations

import polars as pl

from polars_cv._namespace import _ArgBinder, _GeomNullPolicy, _PluginNamespace


@pl.api.register_expr_namespace("bbox")
class BBoxNamespace(_GeomNullPolicy, _PluginNamespace):
    """Namespace for bounding-box operations on ``List[BBOX_SCHEMA]`` columns.

    Example::

        df.with_columns(
            iou_matrix=pl.col("pred_bboxes").bbox.pairwise_iou(pl.col("gt_bboxes")),
            pairs=pl.col("pred_bboxes").bbox.correspond(
                pl.col("gt_bboxes"), threshold=0.5
            ),
        )
    """

    def pairwise_iou(self, other: pl.Expr) -> pl.Expr:
        """Compute pairwise IoU matrix between two sets of bounding boxes.

        Args:
            other: Ground-truth bboxes column (``List[BBOX_SCHEMA]``).

        Returns:
            ``List[List[Float64]]`` IoU matrix expression.
        """
        return self._plugin("bbox_pairwise_iou", args=[other])

    def correspond(
        self,
        other: pl.Expr,
        *,
        threshold: float | pl.Expr = 0.5,
        order: pl.Expr | None = None,
    ) -> pl.Expr:
        """Pair each bbox with at most one bbox in *other*, by overlap.

        The bounding-box form of :meth:`polars_cv.geometry.ContourNamespace.correspond`,
        running the same engine rule, so the two agree on ordering, exclusivity,
        ties and the threshold bound.

        Args:
            other: Bbox-set expression to pair against (``List[BBOX_SCHEMA]``).
            threshold: Minimum IoU for a pairing. Accepts a Polars expression
                for a per-row threshold.
            order: Optional per-row list of indices giving the visit sequence,
                a permutation of ``0..n``. Defaults to natural order.

        Returns:
            A struct matching :data:`polars_cv.CORRESPONDENCE_SCHEMA`.
        """
        binder = _ArgBinder()
        binder.add_data("other", other)
        binder.add_data("order", order)
        binder.add_param("threshold", threshold)
        return binder.call(self, "bbox_correspond")
