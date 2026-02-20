"""Matcher protocol — the contract all input adapters implement."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import polars as pl

    from .._types import DetectionTable


@runtime_checkable
class Matcher(Protocol):
    """Protocol for detection matchers.

    A matcher takes a raw data frame with prediction and ground-truth columns
    and produces a canonical :class:`DetectionTable`.
    """

    def match(
        self,
        data: pl.LazyFrame | pl.DataFrame,
        *,
        pred_col: str,
        gt_col: str,
        score_col: str | None = None,
        class_col: str | None = None,
        image_id_col: str | None = None,
        weight_col: str | None = None,
        group_col: str | None = None,
    ) -> DetectionTable:
        """Produce a canonical ``DetectionTable`` from raw predictions and GTs.

        Args:
            data: Input frame with one image/sample per row.
            pred_col: Prediction column name (format depends on matcher).
            gt_col: Ground-truth column name (format depends on matcher).
            score_col: Optional confidence score column.
            class_col: Optional class label column.
            image_id_col: Optional image identifier column.
            weight_col: Optional sample weight column.
            group_col: Optional grouping column.

        Returns:
            A validated ``DetectionTable``.
        """
        ...
