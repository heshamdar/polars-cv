"""Every expression-eligible operation parameter, resolved per row.

``tests/test_expression_params.py`` covers the ``.contour``/``.point``/``.bbox``
namespaces, which carry their parameters as extra plugin *inputs*. This file
covers the other half: the parameters that ride through the ``vb_graph`` graph
engine as ``ParamValue``s, which is nearly every operation on ``Pipeline``.

The sweep is table-driven (``tests/_expr_param_cases.py``) and each case is
checked three ways by ``tests/_expr_param_runner.py`` — against the literal
pipeline, against its own neighbours, and across rows — because none of those
three implies the others. On top of that every case is run through the
streaming engine and its planned schema compared with what execution produced,
since the lazy/streaming path is the one users are pointed at and the one where
a parameter resolved from "the first row" would survive the eager tests.

``test_every_expression_parameter_has_a_case`` is the ratchet: it reads the
expression-eligible parameters off ``Pipeline``'s live signatures and fails if
one has neither a case nor a documented exemption.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv import Pipeline
from polars_cv.geometry.schemas import CONTOUR_SET_SCHEMA
from tests._expr_param_cases import (
    CASES,
    CONTOUR_SET,
    CONTOURS,
    DIAMOND,
    DIAMOND_SET,
    IMAGE,
    NOT_SWEPT,
    RECT,
    RING,
    ExprCase,
    covered_keys,
    expression_eligible_parameters,
    literal_cases,
    varying_cases,
)
from tests._expr_param_runner import (
    assert_matches_per_row_literals,
    assert_rows_are_independent,
    assert_values_vary,
    run,
    sink_for,
)
from tests._schema_parity import assert_plan_equals_exec
from tests.conftest import make_image_png, make_rect_png, make_ring_png, plugin_required

#: 16x16 so every case's crops, pads and resizes stay inside it, and noisy so a
#: filter, a channel permutation or a threshold actually changes the result.
_SIDE = 16

PARAM = "p"


def _images() -> dict[str, bytes]:
    """The input columns, built once per test that needs them."""
    return {
        IMAGE: make_image_png(_SIDE, _SIDE, 3, seed=7),
        RECT: make_rect_png(_SIDE, _SIDE, 3),
        RING: make_ring_png(_SIDE, _SIDE, 3),
    }


def _frame(case: ExprCase, values: "tuple | list") -> pl.DataFrame:
    """A frame carrying every input column plus the parameter column.

    All rows hold the *same* image, so the only thing varying down the frame is
    the parameter. A case whose rows differ because their images differ would
    pass ``assert_values_vary`` without the parameter doing anything.
    """
    rows = len(values)
    images = _images()
    data: dict[str, list] = {name: [blob] * rows for name, blob in images.items()}
    data[CONTOURS] = [CONTOUR_SET] * rows
    data[DIAMOND] = [DIAMOND_SET] * rows
    data[PARAM] = list(values)
    overrides: dict[str, pl.DataType] = {
        CONTOURS: CONTOUR_SET_SCHEMA,
        DIAMOND: CONTOUR_SET_SCHEMA,
    }
    if case.dtype is not None:
        overrides[PARAM] = case.dtype
    return pl.DataFrame(data, schema_overrides=overrides)


def _ids(cases: "list[ExprCase]") -> list[str]:
    return [case.key for case in cases]


@plugin_required
class TestExpressionParameterSweep:
    """The table, driven end to end."""

    @pytest.mark.parametrize("case", literal_cases(), ids=_ids(literal_cases()))
    def test_expression_matches_the_literal(self, case: ExprCase) -> None:
        """Each row equals the pipeline built with that row's value inline."""
        df = _frame(case, case.values)
        assert_matches_per_row_literals(
            df,
            input_column=case.column,
            param_column=PARAM,
            build=case.build,
            values=case.values,
            label=f"{case.key}: ",
        )

    @pytest.mark.parametrize("case", varying_cases(), ids=_ids(varying_cases()))
    def test_the_value_reaches_the_kernel(self, case: ExprCase) -> None:
        """Distinct parameter values must produce distinct outputs."""
        df = _frame(case, case.values)
        outputs = run(df, case.column, case.build(pl.col(PARAM)))
        assert_values_vary(outputs, label=f"{case.key}: ")

    @pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
    def test_rows_are_independent(self, case: ExprCase) -> None:
        """A row's result is the same alone as it is inside a batch.

        This is the leg that covers the parameters with no literal spelling,
        and the one a morsel-boundary or compiled-graph-cache fault breaks.
        """
        df = _frame(case, case.values)
        assert_rows_are_independent(
            df,
            input_column=case.column,
            pipe=case.build(pl.col(PARAM)),
            label=f"{case.key}: ",
        )

    @pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
    def test_streaming_agrees_with_in_memory_and_with_the_plan(
        self, case: ExprCase
    ) -> None:
        """Lazy streaming is the documented execution mode; it must agree.

        ``assert_plan_equals_exec`` also pins the planned dtype against what
        each engine produced, which is the invariant an expression-valued
        parameter puts under most pressure: the planner cannot know the value,
        so it must publish a dtype that holds for every row.
        """
        df = _frame(case, case.values)
        pipe = case.build(pl.col(PARAM))
        expr = pl.col(case.column).cv.pipe(pipe).sink(sink_for(pipe))
        assert_plan_equals_exec(df, expr)

        streamed = run(df, case.column, pipe, engine="streaming")
        in_memory = run(df, case.column, pipe, engine="in-memory")
        assert streamed == in_memory, (
            f"{case.key}: the streaming and in-memory engines disagree"
        )


class TestExpressionParameterCoverage:
    """The ratchet. Runs without the compiled plugin."""

    def test_the_sweep_is_not_vacuous(self) -> None:
        """A table that emptied out would parametrize to nothing and read green.

        Zero-case parametrization is reported as "no tests ran" for that id,
        not as a failure, so the sweep's size is asserted rather than assumed.
        """
        assert len(CASES) > 80, f"only {len(CASES)} cases — the table has lost coverage"
        assert len(literal_cases()) > 75
        assert len(varying_cases()) > 70

    def test_every_expression_parameter_has_a_case(self) -> None:
        """A parameter that accepts an expression must be swept or exempted."""
        eligible = expression_eligible_parameters()
        assert len(eligible) > 90, (
            f"only {len(eligible)} expression-eligible parameters found — the "
            "signature scan is broken, not the table"
        )
        missing = {
            key: annotation
            for key, annotation in eligible.items()
            if key not in covered_keys()
        }
        assert not missing, (
            "these Pipeline parameters accept a Polars expression but no case "
            f"in tests/_expr_param_cases.py exercises them: {missing}"
        )

    def test_the_table_names_only_real_parameters(self) -> None:
        """A case for a parameter that no longer exists is a stale case.

        Without this the ratchet could be satisfied by a case whose op was
        renamed — it would keep passing while covering nothing.
        """
        eligible = expression_eligible_parameters()
        stale = sorted(key for key in covered_keys() if key not in eligible)
        assert not stale, (
            "these keys are covered but are no longer expression-eligible "
            f"parameters of Pipeline: {stale}"
        )

    def test_exemptions_carry_a_reason(self) -> None:
        """An exemption is a decision, so it has to say what it decided."""
        blank = [key for key, reason in NOT_SWEPT.items() if not reason.strip()]
        assert not blank, f"NOT_SWEPT entries without a reason: {blank}"

    def test_cases_are_unique(self) -> None:
        """Two cases for one parameter hide which one the ratchet accepted."""
        keys = [case.key for case in CASES]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        assert not duplicates, f"duplicated cases: {duplicates}"


@plugin_required
class TestDerivedExpressions:
    """A parameter takes any expression, not only a bare column reference.

    Expression parameters are keyed on ``str(expr)`` when they cross the wire
    (``ParamValue.to_dict``), so two derived expressions sharing a root column
    are the case that key exists to keep apart: before it, ``col("h").max()``
    and ``col("h").min()`` hashed to the same slot.
    """

    @staticmethod
    def _frame() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "image": [make_image_png(16, 16, 3, seed=8)] * 3,
                "p": pl.Series([2, 3, 4], dtype=pl.Int32),
            }
        )

    def test_two_derived_expressions_on_one_op_stay_distinct(self) -> None:
        """``height`` and ``width`` share a root column but not a value."""
        pipe = (
            Pipeline()
            .source("image_bytes", dtype="u8")
            .resize(height=pl.col("p") * 2, width=pl.col("p") + 1)
        )
        out = self._frame().with_columns(r=pl.col("image").cv.pipe(pipe).sink("list"))[
            "r"
        ]
        rows = out.to_list()
        assert [len(row) for row in rows] == [4, 6, 8]
        assert [len(row[0]) for row in rows] == [3, 4, 5]

    def test_two_aggregations_over_one_column_stay_distinct(self) -> None:
        """The collision the string key was introduced for: min vs max."""
        pipe = (
            Pipeline()
            .source("image_bytes", dtype="u8")
            .resize(height=pl.col("p").min(), width=pl.col("p").max())
        )
        out = self._frame().with_columns(r=pl.col("image").cv.pipe(pipe).sink("list"))[
            "r"
        ]
        rows = out.to_list()
        assert [len(row) for row in rows] == [2, 2, 2]
        assert [len(row[0]) for row in rows] == [4, 4, 4]

    def test_a_cast_inside_the_expression_is_honoured(self) -> None:
        pipe = (
            Pipeline()
            .source("image_bytes", dtype="u8")
            .resize(height=(pl.col("p") / 2 + 3).cast(pl.Int32), width=4)
        )
        out = self._frame().with_columns(r=pl.col("image").cv.pipe(pipe).sink("list"))[
            "r"
        ]
        assert [len(row) for row in out.to_list()] == [4, 4, 5]


@plugin_required
class TestParameterColumnDtypes:
    """A parameter column may be any numeric dtype, not just the obvious one.

    Parameter values are read through ``ParamCol``, which spans every numeric
    Polars dtype; nothing requires a user to hand an ``Int64`` to an ``int``
    parameter or a ``Float64`` to a ``float`` one. A resolver that matched on
    one dtype and fell through for the rest would only show up here.
    """

    @staticmethod
    def _resize_to(heights: pl.Series) -> list[int]:
        df = pl.DataFrame({"image": [make_image_png(16, 16, 3, seed=1)] * len(heights)})
        df = df.with_columns(h=heights)
        pipe = (
            Pipeline()
            .source("image_bytes", dtype="u8")
            .resize(height=pl.col("h"), width=8)
        )
        out = df.with_columns(r=pl.col("image").cv.pipe(pipe).sink("list"))
        return [len(row) for row in out["r"].to_list()]

    @pytest.mark.parametrize(
        "dtype",
        [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32],
    )
    def test_integer_parameter_column_dtypes(self, dtype: pl.DataType) -> None:
        heights = pl.Series("h", [4, 8, 12], dtype=dtype)
        assert self._resize_to(heights) == [4, 8, 12]

    @pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
    def test_float_column_feeds_an_integer_parameter(self, dtype: pl.DataType) -> None:
        heights = pl.Series("h", [4.0, 8.0, 12.0], dtype=dtype)
        assert self._resize_to(heights) == [4, 8, 12]

    def test_boolean_column_feeds_a_flag_parameter(self) -> None:
        df = pl.DataFrame(
            {
                "image": [make_image_png(16, 16, 1, seed=2)] * 2,
                "norm": pl.Series([True, False], dtype=pl.Boolean),
            }
        )
        pipe = (
            Pipeline()
            .source("image_bytes", dtype="u8")
            .convolve2d([1.0] * 9, 3, normalize=pl.col("norm"))
        )
        out = df.with_columns(r=pl.col("image").cv.pipe(pipe).sink("list"))["r"]
        assert out.to_list()[0] != out.to_list()[1]

    def test_string_column_feeds_an_enum_parameter(self) -> None:
        df = pl.DataFrame(
            {
                "image": [make_image_png(16, 16, 3, seed=3)] * 2,
                "f": pl.Series(["nearest", "lanczos3"], dtype=pl.String),
            }
        )
        pipe = (
            Pipeline()
            .source("image_bytes", dtype="u8")
            .resize(height=7, width=7, filter=pl.col("f"))
        )
        out = df.with_columns(r=pl.col("image").cv.pipe(pipe).sink("list"))["r"]
        assert out.to_list()[0] != out.to_list()[1]

    def test_an_unknown_enum_string_is_rejected(self) -> None:
        """An unrecognised value must fail the query, not fall back to a default."""
        df = pl.DataFrame(
            {
                "image": [make_image_png(16, 16, 3, seed=4)] * 2,
                "f": ["lanczos3", "not-a-filter"],
            }
        )
        pipe = (
            Pipeline()
            .source("image_bytes", dtype="u8")
            .resize(height=7, width=7, filter=pl.col("f"))
        )
        with pytest.raises(Exception, match="not-a-filter"):
            df.with_columns(r=pl.col("image").cv.pipe(pipe).sink("list"))


@plugin_required
class TestAssertShapeExpressions:
    """``assert_shape`` feeds the planner, so its parameters show up in the plan.

    The sweep marks these ``varies=False``: the op does not touch the buffer,
    so no output comparison can see the value. What the value does change is
    what the planner is willing to *promise* — a literal publishes a concrete
    shape, and an expression must not, because the planner has no way to know
    it. That is the eligibility rule for per-row parameters read from the
    other end, and it is the observable difference between the two spellings.
    """

    @staticmethod
    def _frame(heights: list[int]) -> pl.DataFrame:
        return pl.DataFrame(
            {"image": [make_image_png(16, 16, 3, seed=5)] * len(heights), "h": heights}
        )

    def test_an_expression_assertion_executes_and_leaves_the_buffer_alone(
        self,
    ) -> None:
        df = self._frame([16, 16])
        pipe = (
            Pipeline()
            .source("image_bytes", dtype="u8")
            .assert_shape(height=pl.col("h"))
        )
        out = df.with_columns(r=pl.col("image").cv.pipe(pipe).sink("list"))
        assert out["r"].null_count() == 0
        assert [len(row) for row in out["r"].to_list()] == [16, 16]

    def test_a_literal_assertion_publishes_a_fixed_shape(self) -> None:
        df = self._frame([16, 16])
        pipe = (
            Pipeline()
            .source("image_bytes", dtype="u8")
            .assert_shape(height=16, width=16, channels=3)
        )
        expr = pl.col("image").cv.pipe(pipe).sink("array")
        planned = df.lazy().with_columns(r=expr).collect_schema()["r"]
        assert planned == pl.Array(pl.UInt8, shape=(16, 16, 3))

    def test_an_expression_assertion_refuses_to_publish_one(self) -> None:
        """The fixed-shape sink is refused rather than given a guess."""
        pipe = (
            Pipeline()
            .source("image_bytes", dtype="u8")
            .assert_shape(height=pl.col("h"), width=16, channels=3)
        )
        with pytest.raises(ValueError, match="shape is required"):
            pl.col("image").cv.pipe(pipe).sink("array")


@plugin_required
class TestConvolveKsizeExpression:
    """``convolve2d(ksize=)`` is expression-valued but pinned to the kernel.

    The kernel's *length* is structural, so a per-row ``ksize`` can only
    restate the side it implies. That makes the interesting case the
    disagreeing one: it must be rejected at execution rather than silently
    reading past the kernel or truncating it.
    """

    @staticmethod
    def _frame(ksizes: list[int]) -> pl.DataFrame:
        return pl.DataFrame(
            {"image": [make_image_png(16, 16, 1, seed=6)] * len(ksizes), "k": ksizes}
        )

    def test_a_consistent_expression_ksize_executes(self) -> None:
        df = self._frame([3, 3])
        pipe = (
            Pipeline()
            .source("image_bytes", dtype="u8")
            .convolve2d([0.0] * 4 + [1.0] + [0.0] * 4, pl.col("k"))
        )
        out = df.with_columns(r=pl.col("image").cv.pipe(pipe).sink("list"))
        assert out["r"].null_count() == 0

    def test_a_ksize_disagreeing_with_the_kernel_is_rejected(self) -> None:
        df = self._frame([3, 5])
        pipe = (
            Pipeline()
            .source("image_bytes", dtype="u8")
            .convolve2d([0.0] * 4 + [1.0] + [0.0] * 4, pl.col("k"))
        )
        with pytest.raises(Exception):
            df.with_columns(r=pl.col("image").cv.pipe(pipe).sink("list"))
