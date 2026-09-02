"""Contour-based matcher: heatmap + binary mask -> DetectionTable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from ..._types import dtype_name_for
from ...geometry.schemas import CONTOUR_SET_SCHEMA, CORRESPONDENCE_SCHEMA
from ...pipeline import Pipeline
from .._types import (
    COL_CLASS_ID,
    COL_DET_IDX,
    COL_GROUP_ID,
    COL_GT_IDX,
    COL_GT_LABEL,
    COL_IMAGE_ID,
    COL_IOU,
    COL_IS_TP,
    COL_N_GTS,
    COL_SCORE,
    COL_WEIGHT,
    DEFAULT_CLASS,
    DetectionTable,
    ensure_columns_exist,
    to_lazy,
)

if TYPE_CHECKING:
    pass

#: Field names read off the published schema rather than spelled again here.
#: A private copy of this struct's layout is exactly what the correspondence
#: refactor removed; re-typing the names would restore it in miniature.
_RIGHT_IDX, _OVERLAP = (f.name for f in CORRESPONDENCE_SCHEMA.fields)

# ---------------------------------------------------------------------------
# Source format detection
# ---------------------------------------------------------------------------


@dataclass
class _SourceInfo:
    """The source a mask column needs, as far as *Polars* can say.

    Metrics does not choose the source **format** — `source("auto")` does, and
    the decision is Rust's (`resolve_auto_format`), taken from the column dtype
    and, for Binary, from the VIEW magic bytes. This carries only the one thing
    the Polars schema settles that the format cannot: the element dtype of a
    nested `List`/`Array`, which the planner needs before any data moves.
    """

    kwargs: dict[str, Any]

    def build_source(self) -> Pipeline:
        """Create a ``Pipeline().source("auto", ...)`` for this column."""
        return Pipeline().source("auto", **self.kwargs)


def _leaf_dtype(dtype: pl.DataType) -> pl.DataType:
    """Unwrap nested List/Array to reach the leaf element type."""
    while isinstance(dtype, (pl.List, pl.Array)):
        dtype = dtype.inner  # type: ignore[union-attr]
    return dtype


def _detect_source_info(schema: dict[str, pl.DataType], col: str) -> _SourceInfo:
    """Read the one source detail the Polars schema settles: the leaf dtype.

    This is a **planning-time** operation — it inspects the Polars schema
    (available via ``collect_schema()``) and does not touch data.

    It deliberately does **not** choose a source format. That decision belongs
    to `resolve_auto_format` in Rust, which `source("auto")` — the default —
    already makes, and which is strictly better informed: for a `Binary` column
    it inspects the VIEW magic bytes and routes a PNG/JPEG to `image_bytes`
    rather than the blob decoder. This function used to map every `Binary`
    column to `"blob"`, so a `ContourMatcher` over an image-bytes mask failed
    with "Invalid blob magic bytes" while the same column read fine through
    `source("auto")`.

    Args:
        schema: Polars schema mapping column names to dtypes.
        col: Column name to inspect.

    Returns:
        ``_SourceInfo`` carrying the source kwargs, which is the element dtype
        for a nested column and nothing at all otherwise.

    Raises:
        ValueError: If a nested column's leaf type has no buffer meaning.
    """
    dtype = schema[col]

    # Nested columns are the only case where Polars knows something Rust cannot
    # infer at plan time: the leaf element dtype, which a typed source needs
    # before any data moves. `dtype_name_for` is polars-cv's public naming of
    # that hop and raises for leaves with no buffer meaning (String, Duration);
    # metrics used to keep its own copy of the table it reads.
    if isinstance(dtype, (pl.List, pl.Array)):
        leaf = _leaf_dtype(dtype)
        try:
            return _SourceInfo(kwargs={"dtype": dtype_name_for(leaf)})
        except ValueError as exc:
            # Name the column: the caller supplied it and can change it, and
            # the shared accessor only knows about the type.
            raise ValueError(f"Column {col!r}: {exc}") from None

    # Everything else — Binary above all — goes to `auto` unqualified. An
    # unroutable dtype is rejected there, by the same code that rejects it for
    # every other caller, rather than by a second list maintained here.
    return _SourceInfo(kwargs={})


def _add_gt_shape_columns(
    lf: pl.LazyFrame,
    gt_col: str,
    gt_source: _SourceInfo,
) -> pl.LazyFrame:
    """Add ``_gt_h`` and ``_gt_w`` columns from the GT mask column.

    These are used as the dynamic resize target when ``auto_resize``
    is enabled. The strategy is chosen from the **Polars dtype**, which is what
    decides whether a cheaper native path exists — not from a source format,
    which is Rust's decision to make:

    - **List**: native ``.list.len()`` expressions.
    - **Array**: literal values from the Polars type metadata.
    - **anything else** (`Binary`, whether a VIEW blob or image bytes):
      ``extract_shape()`` through the pipeline, which works for every source
      ``auto`` can resolve.

    Args:
        lf: Input lazy frame.
        gt_col: Ground-truth mask column name.
        gt_source: Source spec for the GT column (its leaf dtype, if nested).

    Returns:
        LazyFrame with ``_gt_h`` and ``_gt_w`` columns added.
    """
    dtype = dict(lf.collect_schema())[gt_col]

    if isinstance(dtype, pl.List):
        return lf.with_columns(
            _gt_h=pl.col(gt_col).list.len().cast(pl.Int64),
            _gt_w=pl.col(gt_col).list.first().list.len().cast(pl.Int64),
        )

    if isinstance(dtype, pl.Array):
        h = dtype.size
        inner = dtype.inner
        w = inner.size if isinstance(inner, pl.Array) else 1
        return lf.with_columns(
            _gt_h=pl.lit(h, dtype=pl.Int64),
            _gt_w=pl.lit(w, dtype=pl.Int64),
        )

    shape_pipe = gt_source.build_source().extract_shape()
    return (
        lf.with_columns(_gt_shape=pl.col(gt_col).cv.pipe(shape_pipe).sink("native"))
        .with_columns(
            _gt_h=pl.col("_gt_shape").list.get(0).cast(pl.Int64),
            _gt_w=pl.col("_gt_shape").list.get(1).cast(pl.Int64),
        )
        .drop("_gt_shape")
    )


# ---------------------------------------------------------------------------
# Shared pipeline helpers
# ---------------------------------------------------------------------------


def _extract_with_fused_resize(
    lf: pl.LazyFrame,
    *,
    pred_col: str,
    threshold: float,
    min_area: float,
    source_info: _SourceInfo,
) -> pl.LazyFrame:
    """Fuse resize + extract into a single pipeline with multi-output sink.

    Produces both ``_pred_contours`` (extracted contours) and
    ``_pred_heatmap_aligned`` (resized heatmap as blob) in one
    ``vb_graph`` execution, eliminating the intermediate list-sink
    round-trip.

    Args:
        lf: LazyFrame with ``_gt_h``, ``_gt_w`` dimension columns.
        pred_col: Prediction heatmap column.
        threshold: Binary threshold for contour extraction.
        min_area: Minimum contour area filter.
        source_info: Auto-detected source format for the prediction column.

    Returns:
        LazyFrame with ``_pred_contours`` and ``_pred_heatmap_aligned``.
    """
    resize_pipe = source_info.build_source().resize(
        height=pl.col("_gt_h"), width=pl.col("_gt_w")
    )
    lazy_resized = pl.col(pred_col).cv.pipe(resize_pipe).alias("resized_heatmap")

    extract_builder = Pipeline().threshold(value=threshold)
    if min_area > 0.0:
        extract_pipe = extract_builder.extract_contours(
            mode="external", method="simple", min_area=min_area
        )
    else:
        extract_pipe = extract_builder.extract_contours(
            mode="external", method="simple"
        )

    fused = lazy_resized.pipe(extract_pipe).alias("extracted_contours")
    multi_out = fused.sink({"extracted_contours": "native", "resized_heatmap": "blob"})

    # Multi-output sink returns a Struct column; unnest to get
    # individual columns, then rename to internal names.
    return (
        lf.with_columns(_fused_out=multi_out)
        .unnest("_fused_out")
        .with_columns(
            _pred_contours=pl.col("extracted_contours").cast(CONTOUR_SET_SCHEMA),
            _pred_heatmap_aligned=pl.col("resized_heatmap"),
        )
        .drop("resized_heatmap", "extracted_contours")
    )


def _extract_contours_from_col(
    lf: pl.LazyFrame,
    *,
    source_col: str,
    threshold: float,
    min_area: float,
    output_col: str,
    source_info: _SourceInfo,
) -> pl.LazyFrame:
    """Extract contours from a buffer column of any supported format.

    Args:
        lf: Input lazy frame.
        source_col: Buffer column (blob, nested list, or array).
        threshold: Binary threshold used prior to extraction.
        min_area: Minimum contour area.
        output_col: Name of output contour-set column.
        source_info: Auto-detected source format for the column.

    Returns:
        LazyFrame with ``output_col`` as a list of contours.
    """
    pipe_builder = source_info.build_source().threshold(value=threshold)
    if min_area > 0.0:
        extract_pipe = pipe_builder.extract_contours(
            mode="external", method="simple", min_area=min_area
        )
    else:
        extract_pipe = pipe_builder.extract_contours(mode="external", method="simple")

    return lf.with_columns(
        pl.col(source_col)
        .cv.pipe(extract_pipe)
        .sink("native")
        .cast(CONTOUR_SET_SCHEMA)
        .alias(output_col)
    )


def _score_contours_from_heatmap(
    lf: pl.LazyFrame,
    *,
    contour_col: str,
    heatmap_col: str,
    output_col: str = "_pred_scores",
    source_info: _SourceInfo,
) -> pl.LazyFrame:
    """Score contour sets from heatmaps using max interior value.

    Args:
        lf: Input lazy frame.
        contour_col: Contour-set column.
        heatmap_col: Heatmap column.
        output_col: Output score list column.
        source_info: Auto-detected source format for the heatmap column.

    Returns:
        LazyFrame with score column added.
    """
    score_pipe = source_info.build_source().label_reduce(
        contours=pl.col(contour_col), reduction="max", region_mode="interior"
    )
    return lf.with_columns(
        pl.col(heatmap_col).cv.pipe(score_pipe).sink("native").alias(output_col)
    )


def _confidence_order(scores_col: str) -> pl.Expr:
    """Visit order for `correspond`: highest confidence first, ties by index.

    The engine takes a permutation, not scores -- deriving one from confidence
    is a detection-evaluation choice and belongs here rather than in the CV
    layer. ``rank(method="ordinal")`` assigns *distinct* ranks in order of
    appearance, so the ``arg_sort`` that inverts it into a visit order has no
    ties left to resolve. Sorting the scores directly instead would lean on
    ``arg_sort`` being stable, which this Polars neither documents nor exposes
    a ``maintain_order`` flag for -- and the tie-break is exactly what decides
    which of two equally-confident detections claims a target.
    """
    return (
        pl.col(scores_col)
        .list.eval(pl.element().rank(method="ordinal", descending=True).arg_sort())
        .cast(pl.List(pl.UInt32))
    )


def _filter_zero_score_detections(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Remove zero-score contours *before* matching.

    A detection that scores 0.0 against the heatmap carries no evidence, so
    letting it into greedy IoU assignment would allow it to claim a GT object
    ahead of a detection that does.

    This filter was introduced when such contours were mostly artifacts: the
    boundary tracer collapsed every region into degenerate 2x2 walks with no
    interior to score. That defect is fixed, so what reaches here now is genuinely
    unevidenced rather than malformed — the filter is kept for the matching reason
    above, not the tracing one.

    Computes the indices of positive scores, then gathers from both the
    score and contour lists to keep them aligned.

    Args:
        lf: LazyFrame with ``_pred_scores_raw`` and ``_pred_contours``.

    Returns:
        LazyFrame with filtered ``_pred_scores`` and ``_pred_contours``.
    """
    return (
        lf.with_columns(
            _keep_idx=pl.col("_pred_scores_raw").list.eval(
                pl.arg_where(pl.element() > 0.0).cast(pl.UInt32)
            ),
        )
        .with_columns(
            _pred_scores=pl.col("_pred_scores_raw").list.gather(pl.col("_keep_idx")),
            _pred_contours=pl.col("_pred_contours").list.gather(pl.col("_keep_idx")),
        )
        .drop("_keep_idx", "_pred_scores_raw")
    )


def _explode_match_to_detections(
    image_level: pl.LazyFrame,
    *,
    image_id_col: str,
    scores_col: str,
    gt_idx_col: str,
    iou_col: str,
    class_id: str,
) -> pl.LazyFrame:
    """Explode per-image match results into per-detection rows.

    Args:
        image_level: One row per image with list columns.
        image_id_col: Image identifier column.
        scores_col: Score list column.
        gt_idx_col: Matched-index list column.
        iou_col: IoU list column.
        class_id: Class label to assign.

    Returns:
        Per-detection lazy frame with canonical column names.
    """
    # A row whose ground-truth column was null gets a null payload, because
    # `correspond` declines to answer when an operand is null. Metrics has
    # already decided what a null GT column means -- `_n_gts` reads it as zero
    # -- so the payload follows: no pairings, one per prediction. Without this
    # the predictions on a ground-truth-free image would vanish instead of
    # counting as false positives, which is exactly what they are.
    unpaired = pl.col(gt_idx_col).is_null()
    payload = image_level.select(
        image_id_col,
        _scores=pl.col(scores_col),
        _gt_idx=pl.when(unpaired)
        .then(pl.col(scores_col).list.eval(pl.lit(None, dtype=pl.UInt32)))
        .otherwise(pl.col(gt_idx_col)),
        _iou=pl.when(unpaired)
        .then(pl.col(scores_col).list.eval(pl.lit(0.0)))
        .otherwise(pl.col(iou_col)),
        _det_ord=pl.int_ranges(0, pl.col(scores_col).list.len()),
    )

    # One explode, no join. The payload is positionally aligned with the
    # scores, so the ordinal is just the position -- the join this replaced
    # existed only to match a `pred_idx` column back up with it, and that
    # column was `0..n` spelled out.
    return (
        payload.explode(
            "_scores", "_gt_idx", "_iou", "_det_ord", empty_as_null=True
        ).with_columns(_det_ord=pl.col("_det_ord").cast(pl.UInt32))
    ).select(
        pl.col(image_id_col).alias(COL_IMAGE_ID),
        pl.lit(class_id).alias(COL_CLASS_ID),
        pl.col("_scores").alias(COL_SCORE),
        pl.col("_gt_idx").is_not_null().alias(COL_IS_TP),
        pl.col("_gt_idx").cast(pl.UInt32).alias(COL_GT_IDX),
        pl.col("_iou").fill_null(0.0).alias(COL_IOU),
        pl.col("_det_ord").alias(COL_DET_IDX),
    )


# ---------------------------------------------------------------------------
# ContourMatcher
# ---------------------------------------------------------------------------


class ContourMatcher:
    """Match detections from heatmaps + binary masks via contour extraction.

    This is the refactored version of the original ``prepare_detection_table``
    from ``_prepare.py``.  It extracts contours from both predictions and GT
    masks, scores predictions against the heatmap, and runs greedy IoU matching
    via ``.contour.correspond()``.

    Args:
        iou_threshold: IoU threshold for TP matching.
        extraction_threshold: Threshold for contour extraction from heatmaps.
        min_contour_area: Minimum extracted contour area for predictions.
        auto_resize: Whether to resize heatmaps to mask shapes automatically.
        gt_min_contour_area: Separate min area for GT contours (defaults to
            ``min_contour_area``).
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        extraction_threshold: float = 0.1,
        min_contour_area: float = 1.0,
        auto_resize: bool = True,
        gt_min_contour_area: float | None = None,
    ) -> None:
        if not (0.0 < iou_threshold <= 1.0):
            raise ValueError("`iou_threshold` must be in (0, 1].")
        self._iou_threshold = iou_threshold
        self._extraction_threshold = extraction_threshold
        self._min_contour_area = min_contour_area
        self._auto_resize = auto_resize
        self._gt_min_contour_area = gt_min_contour_area

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
        """Produce a ``DetectionTable`` from heatmap + binary mask data.

        Accepts any column format supported by polars-cv sources: nested
        ``List[List[...]]``, VIEW protocol ``Binary`` (blob), or fixed-size
        ``Array[...]``.  The source format is auto-detected from the column
        dtype.

        Args:
            data: Input frame with one image/sample per row.
            pred_col: Prediction heatmap column (any supported format).
            gt_col: Ground-truth binary mask column (any supported format).
            score_col: Unused for contour matching (scores are derived from
                heatmap peaks).
            class_col: Optional class label column for multi-class metrics.
            image_id_col: Optional image identifier column (defaults to row index).
            weight_col: Optional sample weight column.
            group_col: Optional grouping column.

        Returns:
            Validated ``DetectionTable``.
        """
        lf = to_lazy(data)
        schema = lf.collect_schema()
        schema_names = list(schema.names())
        ensure_columns_exist(schema_names, [pred_col, gt_col])
        if class_col is not None:
            ensure_columns_exist(schema_names, [class_col])
        if image_id_col is not None:
            ensure_columns_exist(schema_names, [image_id_col])
        if weight_col is not None:
            ensure_columns_exist(schema_names, [weight_col])
        if group_col is not None:
            ensure_columns_exist(schema_names, [group_col])

        # Auto-detect source format from column dtypes (planning-time only)
        schema_dict = dict(schema)
        pred_source = _detect_source_info(schema_dict, pred_col)
        gt_source = _detect_source_info(schema_dict, gt_col)

        # Assign image_id
        if image_id_col is None:
            prepared = lf.with_row_index(name="_metric_row_idx").with_columns(
                pl.col("_metric_row_idx").cast(pl.String).alias(COL_IMAGE_ID)
            )
        else:
            prepared = lf.with_columns(
                pl.col(image_id_col).cast(pl.String).alias(COL_IMAGE_ID)
            )

        # Assign weight
        prepared = prepared.with_columns(
            (
                pl.col(weight_col).cast(pl.Float64)
                if weight_col is not None
                else pl.lit(1.0, dtype=pl.Float64)
            ).alias(COL_WEIGHT)
        )

        # Contour extraction and scoring
        if self._auto_resize:
            # Resize prediction heatmaps to GT mask dimensions via a fused
            # pipeline.  If shapes already match the resize is a no-op.
            prepared = _add_gt_shape_columns(prepared, gt_col, gt_source)
            prepared = _extract_with_fused_resize(
                prepared,
                pred_col=pred_col,
                threshold=self._extraction_threshold,
                min_area=self._min_contour_area,
                source_info=pred_source,
            )
            aligned_pred_col = "_pred_heatmap_aligned"
            # `_pred_heatmap_aligned` is a VIEW blob this pipeline just
            # emitted, so `auto` recognises it by magic bytes — no need to name
            # the format, and naming it would be the second authority again.
            aligned_source = _SourceInfo(kwargs={})
        else:
            # Trust the user: shapes are assumed to match.
            prepared = _extract_contours_from_col(
                prepared,
                source_col=pred_col,
                threshold=self._extraction_threshold,
                min_area=self._min_contour_area,
                output_col="_pred_contours",
                source_info=pred_source,
            )
            aligned_pred_col = pred_col
            aligned_source = pred_source

        gt_area = (
            self._gt_min_contour_area
            if self._gt_min_contour_area is not None
            else self._min_contour_area
        )
        prepared = _extract_contours_from_col(
            prepared,
            source_col=gt_col,
            threshold=0.5,
            min_area=gt_area,
            output_col="_gt_contours",
            source_info=gt_source,
        )

        # Score predictions against the (possibly resized) heatmap
        prepared = _score_contours_from_heatmap(
            prepared,
            contour_col="_pred_contours",
            heatmap_col=aligned_pred_col,
            output_col="_pred_scores_raw",
            source_info=aligned_source,
        )

        # Drop detections that score 0.0 against the heatmap, so an unevidenced
        # contour cannot claim a GT object during greedy assignment.
        prepared = _filter_zero_score_detections(prepared)

        # Pair predictions with GT contours. The order is ours to choose;
        # `correspond` only knows about overlap.
        prepared = prepared.with_columns(
            _match=pl.col("_pred_contours").contour.correspond(
                pl.col("_gt_contours"),
                threshold=self._iou_threshold,
                order=_confidence_order("_pred_scores"),
            ),
            _n_gts=pl.col("_gt_contours").list.len().fill_null(0).cast(pl.Int64),
        )

        # Build image-level frame with match results
        image_level = prepared.select(
            COL_IMAGE_ID,
            COL_WEIGHT,
            "_pred_scores",
            "_n_gts",
            gt_idx=pl.col("_match").struct.field(_RIGHT_IDX),
            iou=pl.col("_match").struct.field(_OVERLAP),
            _gt_label=(pl.col("_gt_contours").list.len().fill_null(0) > 0),
            **(
                {COL_CLASS_ID: pl.col(class_col).cast(pl.String)}
                if class_col is not None
                else {COL_CLASS_ID: pl.lit(DEFAULT_CLASS)}
            ),
            **(
                {COL_GROUP_ID: pl.col(group_col).cast(pl.String)}
                if group_col is not None
                else {}
            ),
        )

        # Cache the shared upstream so the two derived branches (detections and
        # image metadata) run the extraction/correspond graph once — not twice —
        # under a single collect at the caller's boundary. This replaces an
        # eager `.collect()` that materialized here only to split the frame; the
        # explode below enforces prediction/payload alignment structurally, so
        # the former eager alignment guard is no longer needed. An empty input
        # now flows through as an empty table rather than a short-circuit.
        image_level = image_level.cache()

        # Explode into per-detection rows and drop null-score rows (images
        # with no predictions).  Zero-score artifacts are already removed
        # upstream by _filter_zero_score_detections.
        detections_lf = _explode_match_to_detections(
            image_level,
            image_id_col=COL_IMAGE_ID,
            scores_col="_pred_scores",
            gt_idx_col="gt_idx",
            iou_col="iou",
            class_id=DEFAULT_CLASS,
        ).filter(pl.col(COL_SCORE).is_not_null())

        # When a class column was provided, replace the placeholder class with
        # the per-image class from the image-level frame.
        if class_col is not None:
            detections_lf = detections_lf.drop(COL_CLASS_ID).join(
                image_level.select(COL_IMAGE_ID, COL_CLASS_ID).unique(),
                on=COL_IMAGE_ID,
                how="left",
            )

        # Build image metadata
        group_cols = (
            [COL_GROUP_ID]
            if COL_GROUP_ID in image_level.collect_schema().names()
            else []
        )
        meta_lf = image_level.select(
            COL_IMAGE_ID,
            COL_CLASS_ID,
            pl.col("_n_gts").alias(COL_N_GTS),
            COL_WEIGHT,
            pl.col("_gt_label").alias(COL_GT_LABEL),
            *group_cols,
        )

        return DetectionTable.from_matched(
            detections_lf,
            meta_lf,
            matching_iou_threshold=self._iou_threshold,
        )
