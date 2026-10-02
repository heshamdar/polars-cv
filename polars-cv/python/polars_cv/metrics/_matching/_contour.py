"""Contour-based matcher: heatmap + binary mask -> DetectionTable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import polars as pl

from ..._types import FloatOrExpr, LabelReduction, LabelRegionMode, dtype_name_for
from ...geometry.schemas import CONTOUR_SET_SCHEMA, CORRESPONDENCE_SCHEMA
from ...lazy import LazyPipelineExpr
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

#: Field names read off the published schema rather than spelled again here.
#: A private copy of this struct's layout is exactly what the correspondence
#: refactor removed; re-typing the names would restore it in miniature.
_RIGHT_IDX, _OVERLAP, _DUPLICATE = (f.name for f in CORRESPONDENCE_SCHEMA.fields)

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
        dtype = dtype.inner  # type: ignore[union-attr]  # ty: ignore[invalid-assignment]
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


@dataclass
class _SourceHandle:
    """How to obtain a decoded buffer expression for a matcher operand.

    Either a *named column* — decoded with ``source("auto")`` exactly as the
    matcher always has — or a caller-supplied *pre-decoded* ``LazyPipelineExpr``
    that the matcher extends with its own ops. The pre-decoded form lets the
    decode be shared with the caller's own graph (e.g. a segmentation pipeline):
    the matcher appends ``threshold``/``extract_contours``/``label_reduce`` onto
    the node the caller already built, so common-subexpression elimination
    collapses the two into a single decode.

    ``apply`` is the one entry point; a column and an expr differ only in the
    head of the expression they produce, never in the ops appended to it.
    """

    _column: str | None = None
    _kwargs: dict[str, Any] | None = None
    _expr: LazyPipelineExpr | None = None

    @classmethod
    def from_column(cls, col: str, source_info: _SourceInfo) -> _SourceHandle:
        """A handle that decodes *col* via ``source("auto", …)``."""
        return cls(_column=col, _kwargs=source_info.kwargs)

    @classmethod
    def from_expr(cls, expr: LazyPipelineExpr) -> _SourceHandle:
        """A handle that reuses a caller's already-decoded pipeline expr."""
        return cls(_expr=expr)

    @property
    def column(self) -> str | None:
        """The backing column name, or ``None`` for a pre-decoded expr."""
        return self._column

    def apply(self, build_ops: Callable[[Pipeline], Pipeline]) -> LazyPipelineExpr:
        """Decode (or reuse the decoded expr) and append *build_ops*' operations.

        For a column, ``build_ops`` extends ``source("auto", …)`` and the result
        is piped over the column — byte-identical to the pre-handle code. For a
        pre-decoded expr, ``build_ops`` extends a source-less continuation
        pipeline applied onto that expr.
        """
        if self._expr is not None:
            return self._expr.pipe(build_ops(Pipeline()))
        return pl.col(self._column).cv.pipe(  # ty: ignore[invalid-argument-type, unresolved-attribute]
            build_ops(Pipeline().source("auto", **(self._kwargs or {})))
        )


def _add_gt_shape_columns(
    lf: pl.LazyFrame,
    gt_handle: _SourceHandle,
    gt_dtype: pl.DataType | None,
) -> pl.LazyFrame:
    """Add ``_gt_h`` and ``_gt_w`` columns from the GT mask.

    These are the dynamic resize target when ``auto_resize`` is enabled. The
    cheap paths are chosen from the **Polars column dtype**, which is what
    decides whether a native path exists — not from a source format, which is
    Rust's decision:

    - **List**: native ``.list.len()`` expressions.
    - **Array**: literal values from the Polars type metadata.
    - **anything else** (`Binary`, or a pre-decoded GT expr with dtype ``None``):
      ``extract_shape()`` through the pipeline, which works for every source.

    Args:
        lf: Input lazy frame.
        gt_handle: Source handle for the GT mask.
        gt_dtype: The GT column's Polars dtype, or ``None`` for a pre-decoded
            expr (which has no column dtype, so it takes the ``extract_shape``
            path).

    Returns:
        LazyFrame with ``_gt_h`` and ``_gt_w`` columns added.
    """
    if isinstance(gt_dtype, pl.List):
        col = gt_handle.column
        return lf.with_columns(
            _gt_h=pl.col(col).list.len().cast(pl.Int64),  # ty: ignore[invalid-argument-type]
            _gt_w=pl.col(col).list.first().list.len().cast(pl.Int64),  # ty: ignore[invalid-argument-type]
        )

    if isinstance(gt_dtype, pl.Array):
        h = gt_dtype.size
        inner = gt_dtype.inner
        w = inner.size if isinstance(inner, pl.Array) else 1
        return lf.with_columns(
            _gt_h=pl.lit(h, dtype=pl.Int64),
            _gt_w=pl.lit(w, dtype=pl.Int64),
        )

    shape_expr = gt_handle.apply(lambda p: p.extract_shape()).sink("native")
    return (
        lf.with_columns(_gt_shape=shape_expr)
        .with_columns(
            _gt_h=pl.col("_gt_shape").list.get(0).cast(pl.Int64),
            _gt_w=pl.col("_gt_shape").list.get(1).cast(pl.Int64),
        )
        .drop("_gt_shape")
    )


# ---------------------------------------------------------------------------
# Shared pipeline helpers
# ---------------------------------------------------------------------------


def _extract_ops(
    p: Pipeline,
    *,
    threshold: FloatOrExpr,
    min_area: FloatOrExpr,
    min_area_fraction: FloatOrExpr | None = None,
) -> Pipeline:
    """Threshold, then extract the external contours.

    A literal ``min_area`` of 0 keeps every region and is left off the op, so
    the op — and with it CSE and the compiled-graph cache — is the one a
    caller writing ``extract_contours()`` builds. An expression is passed as
    it is, read per row.
    """
    filters: dict[str, FloatOrExpr] = {}
    if isinstance(min_area, pl.Expr) or min_area > 0.0:
        filters["min_area"] = min_area
    if min_area_fraction is not None:
        filters["min_area_fraction"] = min_area_fraction
    return p.threshold(value=threshold).extract_contours(
        mode="external", method="simple", **filters
    )


def _extract_with_fused_resize(
    lf: pl.LazyFrame,
    *,
    pred_handle: _SourceHandle,
    threshold: FloatOrExpr,
    min_area: FloatOrExpr,
    min_area_fraction: FloatOrExpr | None,
) -> pl.LazyFrame:
    """Fuse resize + extract into a single pipeline with multi-output sink.

    Produces both ``_pred_contours`` (extracted contours) and
    ``_pred_heatmap_aligned`` (resized heatmap as blob) in one
    ``vb_graph`` execution, eliminating the intermediate list-sink
    round-trip.

    Args:
        lf: LazyFrame with ``_gt_h``, ``_gt_w`` dimension columns.
        pred_handle: Source handle for the prediction heatmap.
        threshold: Binary threshold for contour extraction.
        min_area: Minimum contour area filter.
        min_area_fraction: Minimum contour area as a fraction of the image.

    Returns:
        LazyFrame with ``_pred_contours`` and ``_pred_heatmap_aligned``.
    """
    lazy_resized = pred_handle.apply(
        lambda p: p.resize(height=pl.col("_gt_h"), width=pl.col("_gt_w"))
    ).alias("resized_heatmap")

    extract_pipe = _extract_ops(
        Pipeline(),
        threshold=threshold,
        min_area=min_area,
        min_area_fraction=min_area_fraction,
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


def _extract_contours_via(
    lf: pl.LazyFrame,
    handle: _SourceHandle,
    *,
    threshold: FloatOrExpr,
    min_area: FloatOrExpr,
    output_col: str,
    min_area_fraction: FloatOrExpr | None = None,
) -> pl.LazyFrame:
    """Extract contours from a buffer handle (column or pre-decoded expr).

    Args:
        lf: Input lazy frame.
        handle: Source handle for the buffer to threshold and extract from.
        threshold: Binary threshold used prior to extraction.
        min_area: Minimum contour area.
        output_col: Name of output contour-set column.
        min_area_fraction: Minimum contour area as a fraction of the image.

    Returns:
        LazyFrame with ``output_col`` as a list of contours.
    """
    return lf.with_columns(
        handle.apply(
            lambda p: _extract_ops(
                p,
                threshold=threshold,
                min_area=min_area,
                min_area_fraction=min_area_fraction,
            )
        )
        .sink("native")
        .cast(CONTOUR_SET_SCHEMA)
        .alias(output_col)
    )


def _score_contours_via(
    lf: pl.LazyFrame,
    handle: _SourceHandle,
    *,
    contour_col: str,
    reduction: str,
    region_mode: str,
    output_col: str = "_pred_scores",
) -> pl.LazyFrame:
    """Score contour sets against a heatmap handle with ``label_reduce``.

    Args:
        lf: Input lazy frame.
        handle: Source handle for the heatmap to score against.
        contour_col: Contour-set column.
        reduction: ``label_reduce`` reduction over each contour's pixels.
        region_mode: ``label_reduce`` region mode.
        output_col: Output score list column.

    Returns:
        LazyFrame with score column added.
    """
    return lf.with_columns(
        handle.apply(
            lambda p: p.label_reduce(
                contours=pl.col(contour_col),
                reduction=reduction,
                region_mode=region_mode,
            )
        )
        .sink("native")
        .alias(output_col)
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
    ignore_duplicates_col: str | None = None,
) -> pl.LazyFrame:
    """Explode per-image match results into per-detection rows.

    Args:
        image_level: One row per image with list columns.
        image_id_col: Image identifier column.
        scores_col: Score list column.
        gt_idx_col: Matched-index list column.
        iou_col: IoU list column.
        class_id: Class label to assign.
        ignore_duplicates_col: The correspondence's ``duplicate`` list column,
            when duplicate hits are to be dropped rather than kept as false
            positives; ``None`` keeps every detection.

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
        _dup=(
            pl.when(unpaired)
            .then(pl.col(scores_col).list.eval(pl.lit(False)))
            .otherwise(pl.col(ignore_duplicates_col))
            if ignore_duplicates_col is not None
            else pl.col(scores_col).list.eval(pl.lit(False))
        ),
    )

    # One explode, no join. The payload is positionally aligned with the
    # scores, so the ordinal is just the position -- the join this replaced
    # existed only to match a `pred_idx` column back up with it, and that
    # column was `0..n` spelled out.
    return (
        payload.explode(
            "_scores", "_gt_idx", "_iou", "_det_ord", "_dup", empty_as_null=True
        )
        .with_columns(_det_ord=pl.col("_det_ord").cast(pl.UInt32))
        # A repeated hit on a claimed target is dropped, not scored as an FP.
        .filter(~pl.col("_dup").fill_null(False))
    ).select(
        pl.col(image_id_col).alias(COL_IMAGE_ID),
        pl.lit(class_id).alias(COL_CLASS_ID),
        pl.col("_scores").alias(COL_SCORE),
        pl.col("_gt_idx").is_not_null().alias(COL_IS_TP),
        pl.col("_gt_idx").cast(pl.UInt32).alias(COL_GT_IDX),
        pl.col("_iou").fill_null(0.0).alias(COL_IOU),
        pl.col("_det_ord").alias(COL_DET_IDX),
    )


def _is_contour_dtype(dtype: pl.DataType) -> bool:
    """Whether a column holds contours (a contour struct, or a list of them)
    rather than a mask: a struct with an ``exterior`` field, at either level."""
    inner = dtype.inner if isinstance(dtype, pl.List) else dtype
    return isinstance(inner, pl.Struct) and any(
        f.name == "exterior" for f in inner.fields
    )


def _contour_predictions(
    lf: pl.LazyFrame, pred_col: str, score_col: str, schema: dict[str, pl.DataType]
) -> pl.LazyFrame:
    """Use a contour (or contour-set) column as the detections, scored by
    ``score_col``, as ``_pred_contours`` / ``_pred_scores``.

    A single contour per row becomes a one-element set and its float score a
    one-element list, so matching sees one shape. The two lists are each read
    at the other's positions: a count mismatch, either way, is an
    out-of-bounds gather that fails the query, rather than detections or
    scores quietly dropped.

    Raises:
        ValueError: If ``score_col``'s dtype does not fit ``pred_col``'s.
    """
    is_set = isinstance(schema[pred_col], pl.List)
    score_dtype = schema[score_col]
    fits = (
        isinstance(score_dtype, pl.List) and score_dtype.inner.is_numeric()
        if is_set
        else score_dtype.is_numeric()
    )
    if not fits:
        want = "List[float], one per contour" if is_set else "a float"
        msg = (
            f"score_col {score_col!r} must be {want} for the "
            f"{'contour-set' if is_set else 'contour'} column {pred_col!r}, "
            f"got {score_dtype}"
        )
        raise ValueError(msg)
    if is_set:
        contours = pl.col(pred_col)
        scores = pl.col(score_col).cast(pl.List(pl.Float64))
    else:
        present = pl.col(pred_col).is_not_null()
        contours = pl.when(present).then(pl.concat_list(pl.col(pred_col)))
        scores = pl.when(present).then(
            pl.concat_list(pl.col(score_col).cast(pl.Float64))
        )
    return lf.with_columns(
        _pred_contours=contours.list.gather(pl.int_ranges(0, scores.list.len())).cast(
            CONTOUR_SET_SCHEMA
        ),
        _pred_scores=scores.list.gather(pl.int_ranges(0, contours.list.len())),
    )


# ---------------------------------------------------------------------------
# ContourMatcher
# ---------------------------------------------------------------------------


class ContourMatcher:
    """Match detections from heatmaps (or contours) and GT masks (or contours).

    It extracts contours from both predictions and GT masks, scores
    predictions against the heatmap, and runs greedy IoU matching via
    ``.contour.correspond()``. Predictions given as contours with their own
    scores skip the extraction and scoring (see :meth:`match`).

    ``iou_threshold``, ``extraction_threshold``, ``min_contour_area``,
    ``min_contour_area_fraction`` and ``coverage_tolerance`` may each be a
    Polars expression, read per row — e.g. a physical tolerance,
    ``coverage_tolerance=5.0 / pl.col("spacing_mm")``. A literal is checked
    here; an expression is checked per row when the query runs.

    Args:
        iou_threshold: IoU threshold for TP matching.
        extraction_threshold: Threshold for contour extraction from heatmaps.
        min_contour_area: Minimum polygon area of an extracted prediction
            contour. Contours are traced along pixel edges, so a region's area
            is its pixel count (for a region with holes, including the hole
            pixels). The default ``0.0`` keeps every region.
        auto_resize: Whether to resize heatmaps to mask shapes automatically.
            Ground truth given as contours has no mask to take a size from, so
            it needs ``auto_resize=False`` (the heatmap already in the
            contours' coordinates).
        gt_min_contour_area: Minimum polygon area of a ground-truth contour,
            independent of ``min_contour_area``. The smallest region, one
            pixel, has area 1, so the default ``1.0`` keeps every region.
        match_by: ``"iou"`` (default) pairs by overlap
            (``.contour.correspond``). ``"coverage"`` pairs by the fraction of
            each GT contour's boundary inside a prediction or within
            ``coverage_tolerance`` of it (``.contour.correspond_by_coverage``)
            — the rule for line-shaped GT (polylines), whose IoU with any
            region is 0. ``iou_threshold`` is then the minimum coverage, and
            the table's ``iou`` column holds coverage.
        coverage_tolerance: Pixels a GT sample may lie from a prediction and
            still count as covered; required with ``match_by="coverage"`` and
            refused otherwise. For a physical tolerance, divide by the pixel
            spacing.
        duplicates: ``"false_positive"`` (default) counts a prediction that
            hits only an already-matched GT as a false positive;
            ``"ignore"`` drops it (the LUNA16/CAMELYON convention).
        score_reduction: How a detection is scored from the heatmap pixels
            in its region (``label_reduce``'s ``reduction``): ``"max"``
            (default), ``"mean"`` or ``"sum"``. A heatmap that saturates
            scores most detections at its peak, so they tie under ``"max"``.
        score_region_mode: Which pixels are a detection's region
            (``label_reduce``'s ``region_mode``): ``"interior"`` (default),
            ``"boundary"`` or ``"bbox"``.
        min_contour_area_fraction: Minimum area of an extracted prediction
            contour as a fraction of the image's pixel count
            (``extract_contours(min_area_fraction=...)``); ``None`` (default)
            applies none.
    """

    def __init__(
        self,
        iou_threshold: FloatOrExpr = 0.5,
        extraction_threshold: FloatOrExpr = 0.1,
        min_contour_area: FloatOrExpr = 0.0,
        auto_resize: bool = True,
        gt_min_contour_area: float = 1.0,
        match_by: str = "iou",
        coverage_tolerance: FloatOrExpr | None = None,
        duplicates: str = "false_positive",
        score_reduction: str = "max",
        score_region_mode: str = "interior",
        min_contour_area_fraction: FloatOrExpr | None = None,
    ) -> None:
        if not isinstance(iou_threshold, pl.Expr) and not (0.0 < iou_threshold <= 1.0):
            raise ValueError("`iou_threshold` must be in (0, 1].")
        if match_by not in ("iou", "coverage"):
            msg = f"match_by must be 'iou' or 'coverage', got {match_by!r}"
            raise ValueError(msg)
        if (match_by == "coverage") != (coverage_tolerance is not None):
            msg = (
                "coverage_tolerance is required with match_by='coverage' and "
                "means nothing with match_by='iou'"
            )
            raise ValueError(msg)
        if (
            coverage_tolerance is not None
            and not isinstance(coverage_tolerance, pl.Expr)
            and not coverage_tolerance >= 0.0
        ):
            msg = f"coverage_tolerance must be >= 0, got {coverage_tolerance}"
            raise ValueError(msg)
        if duplicates not in ("false_positive", "ignore"):
            msg = f"duplicates must be 'false_positive' or 'ignore', got {duplicates!r}"
            raise ValueError(msg)
        # The accepted spellings are the engine's own enums, not a copy here.
        for name, value, enum in (
            ("score_reduction", score_reduction, LabelReduction),
            ("score_region_mode", score_region_mode, LabelRegionMode),
        ):
            allowed = tuple(v.value for v in enum)
            if value not in allowed:
                msg = f"{name} must be one of {allowed}, got {value!r}"
                raise ValueError(msg)
        self._match_by = match_by
        self._coverage_tolerance = coverage_tolerance
        self._duplicates = duplicates
        self._iou_threshold = iou_threshold
        self._extraction_threshold = extraction_threshold
        self._min_contour_area = min_contour_area
        self._min_contour_area_fraction = min_contour_area_fraction
        self._auto_resize = auto_resize
        self._gt_min_contour_area = gt_min_contour_area
        self._score_reduction = score_reduction
        self._score_region_mode = score_region_mode

    def match(
        self,
        data: pl.LazyFrame | pl.DataFrame,
        *,
        pred_col: str | LazyPipelineExpr,
        gt_col: str | LazyPipelineExpr,
        score_col: str | None = None,
        class_col: str | None = None,
        image_id_col: str | None = None,
        weight_col: str | None = None,
        group_col: str | None = None,
    ) -> DetectionTable:
        """Produce a ``DetectionTable`` from heatmap (or contour) + mask data.

        ``pred_col`` / ``gt_col`` accept either a **column name** (any format a
        polars-cv source supports: nested ``List[List[...]]``, VIEW ``Binary``
        blob, or fixed-size ``Array[...]`` — the format is auto-detected) **or a
        pre-decoded ``LazyPipelineExpr``**. Passing a pre-decoded expr lets the
        matcher extend the caller's own decoded buffer instead of decoding a
        column itself, so a segmentation graph and the contour extraction can
        share one decode (collapsed by CSE) and stream from a single collect.

        ``pred_col`` may instead name a **contour** or **contour-set** column —
        for a model that emits masks or polygons with its own per-object
        scores. Those contours are the detections as they are (no extraction,
        no heatmap scoring), each scored by ``score_col``; ``auto_resize``
        must then be ``False`` (the contours are in the GT's coordinates).

        Args:
            data: Input frame with one image/sample per row.
            pred_col: Prediction heatmap column name, or a pre-decoded
                ``LazyPipelineExpr`` producing the heatmap buffer.
            gt_col: Ground-truth mask column name, or a pre-decoded
                ``LazyPipelineExpr`` producing the mask buffer.
            score_col: Each prediction contour's score, for contour
                predictions: ``List[float]`` aligned with a contour-set
                ``pred_col`` (a count mismatch fails the query), or a float
                for a single-contour one. Every contour is kept, a 0.0 score
                included. Refused for heatmap predictions, whose scores come
                from the heatmap (``score_reduction``).
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
        pred_is_expr = isinstance(pred_col, LazyPipelineExpr)
        gt_is_expr = isinstance(gt_col, LazyPipelineExpr)
        if not pred_is_expr:
            ensure_columns_exist(schema_names, [pred_col])
        if not gt_is_expr:
            ensure_columns_exist(schema_names, [gt_col])
        if class_col is not None:
            ensure_columns_exist(schema_names, [class_col])
        if image_id_col is not None:
            ensure_columns_exist(schema_names, [image_id_col])
        if weight_col is not None:
            ensure_columns_exist(schema_names, [weight_col])
        if group_col is not None:
            ensure_columns_exist(schema_names, [group_col])

        schema_dict = dict(schema)
        # Predictions given as contours are the detections themselves, scored
        # by the caller; anything else is a heatmap to extract and score them
        # from, whose scores are the heatmap's.
        contour_preds: tuple[str, str] | None = None
        if isinstance(pred_col, str) and _is_contour_dtype(schema_dict[pred_col]):
            if score_col is None:
                msg = (
                    f"pred_col {pred_col!r} holds contours, which carry no "
                    "score: pass score_col with each contour's score"
                )
                raise ValueError(msg)
            ensure_columns_exist(schema_names, [score_col])
            if self._auto_resize:
                msg = (
                    f"pred_col {pred_col!r} holds contours, which auto_resize "
                    "cannot resize: pass auto_resize=False, with the contours "
                    "in the ground truth's coordinates"
                )
                raise ValueError(msg)
            contour_preds = (pred_col, score_col)
        elif score_col is not None:
            msg = (
                f"score_col {score_col!r} is for contour predictions; a "
                "heatmap's detections are scored from the heatmap (see "
                "score_reduction)"
            )
            raise ValueError(msg)

        # A column is decoded via source("auto") (its leaf dtype read at plan
        # time); a pre-decoded expr is reused as-is. Either way the same ops are
        # appended through the handle.
        # The predictions: the caller's contours and scores, or a heatmap.
        pred_source: tuple[str, str] | _SourceHandle
        if contour_preds is not None:
            pred_source = contour_preds
        elif isinstance(pred_col, LazyPipelineExpr):
            pred_source = _SourceHandle.from_expr(pred_col)
        else:
            pred_source = _SourceHandle.from_column(
                pred_col, _detect_source_info(schema_dict, pred_col)
            )
        # Ground truth given as contours (a contour or contour-set column) is
        # used as it is, by name; anything else is a mask to extract them from.
        gt_source: _SourceHandle | str
        if isinstance(gt_col, LazyPipelineExpr):
            gt_source = _SourceHandle.from_expr(gt_col)
        elif _is_contour_dtype(schema_dict[gt_col]):
            gt_source = gt_col
        else:
            gt_source = _SourceHandle.from_column(
                gt_col, _detect_source_info(schema_dict, gt_col)
            )
        gt_dtype = None if gt_is_expr else schema_dict[gt_col]

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

        # Contour extraction (or the caller's contours)
        aligned_handle: _SourceHandle | None = None
        if isinstance(pred_source, tuple):
            contours_col, scores_col = pred_source
            prepared = _contour_predictions(
                prepared, contours_col, scores_col, schema_dict
            )
        elif self._auto_resize:
            # Resize prediction heatmaps to GT mask dimensions via a fused
            # pipeline.  If shapes already match the resize is a no-op.
            if isinstance(gt_source, str):
                msg = (
                    f"gt_col {gt_source!r} holds contours, which have no mask "
                    "size for auto_resize to resize the heatmap to: pass "
                    "auto_resize=False, with the heatmap in the contours' "
                    "coordinates"
                )
                raise ValueError(msg)
            prepared = _add_gt_shape_columns(prepared, gt_source, gt_dtype)
            prepared = _extract_with_fused_resize(
                prepared,
                pred_handle=pred_source,
                threshold=self._extraction_threshold,
                min_area=self._min_contour_area,
                min_area_fraction=self._min_contour_area_fraction,
            )
            # `_pred_heatmap_aligned` is a VIEW blob this pipeline just emitted,
            # so `auto` recognises it by magic bytes — scoring reads it back as a
            # column regardless of how the prediction was supplied.
            aligned_handle = _SourceHandle.from_column(
                "_pred_heatmap_aligned", _SourceInfo(kwargs={})
            )
        else:
            # Trust the user: shapes are assumed to match.
            prepared = _extract_contours_via(
                prepared,
                pred_source,
                threshold=self._extraction_threshold,
                min_area=self._min_contour_area,
                min_area_fraction=self._min_contour_area_fraction,
                output_col="_pred_contours",
            )
            aligned_handle = pred_source

        if isinstance(gt_source, str):
            prepared = prepared.with_columns(_gt_contours=pl.col(gt_source))
        else:
            prepared = _extract_contours_via(
                prepared,
                gt_source,
                threshold=0.5,
                min_area=self._gt_min_contour_area,
                output_col="_gt_contours",
            )

        if aligned_handle is not None:
            # Score predictions against the (possibly resized) heatmap
            prepared = _score_contours_via(
                prepared,
                aligned_handle,
                contour_col="_pred_contours",
                reduction=self._score_reduction,
                region_mode=self._score_region_mode,
                output_col="_pred_scores_raw",
            )

            # Drop detections that score 0.0 against the heatmap, so an
            # unevidenced contour cannot claim a GT object during greedy
            # assignment. A caller's own 0.0 is a confidence, and is kept.
            prepared = _filter_zero_score_detections(prepared)

        # Pair predictions with GT contours. The order is ours to choose;
        # `correspond` only knows about overlap.
        preds = pl.col("_pred_contours").contour  # ty: ignore[unresolved-attribute]
        match = (
            preds.correspond(
                pl.col("_gt_contours"),
                threshold=self._iou_threshold,
                order=_confidence_order("_pred_scores"),
            )
            if self._match_by == "iou"
            else preds.correspond_by_coverage(
                pl.col("_gt_contours"),
                tolerance=self._coverage_tolerance,
                threshold=self._iou_threshold,
                order=_confidence_order("_pred_scores"),
            )
        )
        prepared = prepared.with_columns(
            _match=match,
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
            duplicate=pl.col("_match").struct.field(_DUPLICATE),
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
            ignore_duplicates_col="duplicate" if self._duplicates == "ignore" else None,
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
            # A per-row threshold has no single value to warn against.
            matching_iou_threshold=(
                None
                if isinstance(self._iou_threshold, pl.Expr)
                else self._iou_threshold
            ),
        )
