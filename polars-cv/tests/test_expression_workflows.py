"""Multi-stage workflows where one pipeline's output parameterises the next.

The sweep in ``test_expression_op_params.py`` proves each parameter resolves
per row. This file covers what people actually build with that: a parameter
that is *computed*, by a Polars expression over another pipeline's output —
"measure every image, then resize the batch to the smallest one" — and run
lazily, including under the streaming engine.

Those two properties interact. An aggregation like ``pl.col("h").min()``
produces a length-1 Series that Polars broadcasts, and the streaming engine
splits a column into morsels; a parameter resolved from "this morsel's first
row" would give the right answer eagerly and a per-morsel answer under
streaming. Every workflow here is therefore checked on both engines and
against a value computed independently in Python, not just for internal
consistency.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv import Pipeline
from tests._schema_parity import ENGINES
from tests.conftest import make_image_png, plugin_required

#: (height, width) per row. Deliberately unsorted, with the minimum height in
#: the middle and the minimum width in a different row, so a workflow that
#: silently used row 0 — or that confused the two dimensions — lands on a
#: different answer than the correct one.
SIZES: list[tuple[int, int]] = [(20, 24), (12, 32), (28, 16), (16, 20)]


def _images() -> list[bytes]:
    return [make_image_png(h, w, 3, seed=i) for i, (h, w) in enumerate(SIZES)]


def _frame() -> pl.DataFrame:
    return pl.DataFrame({"image": _images()})


def _measure() -> Pipeline:
    """A pipeline that measures a buffer: ``[height, width, channels]``."""
    return Pipeline().source("image_bytes").extract_shape()


def _decode() -> Pipeline:
    """The base for pipelines whose result is compared as nested lists."""
    return Pipeline().source("image_bytes", dtype="u8")


def _heights(rows: list) -> list[int]:
    """Row heights of a ``list``-sink result (outermost nesting level)."""
    return [len(row) for row in rows]


def _widths(rows: list) -> list[int]:
    return [len(row[0]) for row in rows]


@plugin_required
class TestMeasureThenResize:
    """The headline workflow: measure the batch, then resize it to fit."""

    @staticmethod
    def _resize_all_to_the_smallest(engine: str) -> pl.DataFrame:
        """Stage 1 measures every image; stage 2 resizes to the smallest height.

        Both stages are pipelines in one lazy query — nothing is collected in
        between, so the target height is a Polars expression over stage 1's
        output rather than a Python value read out of an eager frame.
        """
        target = pl.col("shape").list.get(0).cast(pl.Int32).min()
        return (
            _frame()
            .lazy()
            .with_columns(shape=pl.col("image").cv.pipe(_measure()).sink("native"))
            .with_columns(
                out=pl.col("image")
                .cv.pipe(_decode().resize_to_height(target))
                .sink("list")
            )
            .collect(engine=engine)
        )

    @pytest.mark.parametrize("engine", ENGINES)
    def test_every_row_lands_on_the_batch_minimum(self, engine: str) -> None:
        smallest = min(height for height, _ in SIZES)
        out = self._resize_all_to_the_smallest(engine)
        assert _heights(out["out"].to_list()) == [smallest] * len(SIZES)

    @pytest.mark.parametrize("engine", ENGINES)
    def test_aspect_ratio_is_preserved_per_row(self, engine: str) -> None:
        """``resize_to_height`` derives the width from each row's own aspect.

        The height comes from a batch-wide aggregate and the width from the
        row: if the two were resolved from the same place, every row would come
        back the same width.
        """
        smallest = min(height for height, _ in SIZES)
        out = self._resize_all_to_the_smallest(engine)
        expected = [round(width * smallest / height) for height, width in SIZES]
        assert len(set(expected)) > 1, "SIZES no longer distinguishes the rows"
        assert _widths(out["out"].to_list()) == expected

    def test_the_streaming_and_in_memory_results_are_identical(self) -> None:
        """A per-morsel minimum would differ from the whole-column one."""
        streamed = self._resize_all_to_the_smallest("streaming")["out"].to_list()
        in_memory = self._resize_all_to_the_smallest("in-memory")["out"].to_list()
        assert streamed == in_memory

    def test_the_plan_matches_what_execution_produced(self) -> None:
        target = pl.col("shape").list.get(0).cast(pl.Int32).min()
        lf = (
            _frame()
            .lazy()
            .with_columns(shape=pl.col("image").cv.pipe(_measure()).sink("native"))
            .with_columns(
                out=pl.col("image")
                .cv.pipe(_decode().resize_to_height(target))
                .sink("list")
            )
        )
        assert lf.collect_schema()["out"] == lf.collect().schema["out"]

    def test_header_metadata_can_supply_the_target_instead(self) -> None:
        """``.cv.height()`` reads the header only — no decode, same answer.

        The two routes to a row's height are independent implementations, so
        agreeing is evidence rather than a tautology.
        """
        out = (
            _frame()
            .lazy()
            .with_columns(
                shape=pl.col("image").cv.pipe(_measure()).sink("native"),
                header_height=pl.col("image").cv.height(),
            )
            .collect(engine="streaming")
        )
        decoded = [int(row[0]) for row in out["shape"].to_list()]
        assert decoded == out["header_height"].to_list()
        assert decoded == [height for height, _ in SIZES]


@plugin_required
class TestPerGroupTargets:
    """A window function makes the target per group rather than per batch."""

    @staticmethod
    def _frame_with_groups() -> pl.DataFrame:
        return _frame().with_columns(group=pl.Series("group", ["a", "b", "a", "b"]))

    @pytest.mark.parametrize("engine", ENGINES)
    def test_each_group_resizes_to_its_own_minimum(self, engine: str) -> None:
        out = (
            self._frame_with_groups()
            .lazy()
            .with_columns(target=pl.col("image").cv.height().min().over("group"))
            .with_columns(
                out=pl.col("image")
                .cv.pipe(_decode().resize(height=pl.col("target"), width=8))
                .sink("list")
            )
            .collect(engine=engine)
        )
        # Groups are (rows 0, 2) -> heights 20, 28 and (rows 1, 3) -> 12, 16.
        assert _heights(out["out"].to_list()) == [20, 12, 20, 12]


@plugin_required
class TestParametersFromElsewhere:
    """The expression need not mention the image column at all."""

    def test_a_joined_column_parameterises_the_pipeline(self) -> None:
        sizes = pl.DataFrame(
            {"group": ["a", "b"], "target": pl.Series([6, 10], dtype=pl.Int32)}
        )
        frame = _frame().with_columns(group=pl.Series(["a", "b", "a", "b"]))
        out = (
            frame.lazy()
            .join(sizes.lazy(), on="group", how="left")
            .with_columns(
                out=pl.col("image")
                .cv.pipe(_decode().resize(height=pl.col("target"), width=4))
                .sink("list")
            )
            .collect(engine="streaming")
        )
        # Compared against the joined column rather than a fixed list: the
        # streaming join does not promise input order, and what matters is that
        # each row kept *its own* target through the reordering.
        assert _heights(out["out"].to_list()) == out["target"].to_list()
        assert sorted(out["target"].to_list()) == [6, 6, 10, 10]

    def test_a_conditional_expression_chooses_per_row(self) -> None:
        """``when/then`` is just another expression to the parameter binder."""
        target = pl.when(pl.col("image").cv.height() > 18).then(8).otherwise(4)
        out = (
            _frame()
            .lazy()
            .with_columns(
                out=pl.col("image")
                .cv.pipe(_decode().resize(height=target, width=4))
                .sink("list")
            )
            .collect(engine="streaming")
        )
        assert _heights(out["out"].to_list()) == [8, 4, 8, 4]

    def test_arithmetic_over_two_pipelines_feeds_one_parameter(self) -> None:
        """Half the mean of each row's own height and the batch maximum."""
        own = pl.col("shape").list.get(0)
        target = ((own + own.max()) / 4).cast(pl.Int32)
        out = (
            _frame()
            .lazy()
            .with_columns(shape=pl.col("image").cv.pipe(_measure()).sink("native"))
            .with_columns(
                out=pl.col("image")
                .cv.pipe(_decode().resize(height=target, width=4))
                .sink("list")
            )
            .collect(engine="streaming")
        )
        tallest = max(height for height, _ in SIZES)
        expected = [int((height + tallest) / 4) for height, _ in SIZES]
        assert _heights(out["out"].to_list()) == expected


@plugin_required
class TestMorselBoundaries:
    """Many rows, in many chunks, so the streaming engine has seams to get wrong.

    A parameter resolved once per batch rather than once per row is invisible
    at four rows in one chunk. The frame here is assembled from several
    unrechunked pieces (asserted, not assumed) and the parameter values are
    laid out so that no chunk sees a constant value — whichever row starts a
    morsel, taking its value for the rest of the morsel gives a wrong answer.
    """

    CHUNKS = 8
    PER_CHUNK = 50

    def _frame(self) -> pl.DataFrame:
        base = make_image_png(16, 16, 3, seed=11)
        pieces = [
            pl.DataFrame(
                {
                    "image": [base] * self.PER_CHUNK,
                    "h": pl.Series(
                        [
                            4 + ((chunk * self.PER_CHUNK + i) % 9)
                            for i in range(self.PER_CHUNK)
                        ],
                        dtype=pl.Int32,
                    ),
                }
            )
            for chunk in range(self.CHUNKS)
        ]
        df = pl.concat(pieces, rechunk=False)
        assert df.n_chunks() > 1, "the frame collapsed into one chunk"
        return df

    @pytest.mark.parametrize("engine", ENGINES)
    def test_every_row_keeps_its_own_parameter(self, engine: str) -> None:
        df = self._frame()
        out = (
            df.lazy()
            .with_columns(
                out=pl.col("image")
                .cv.pipe(_decode().resize(height=pl.col("h"), width=4))
                .sink("list")
            )
            .collect(engine=engine)
        )
        assert df["h"].n_unique() > 1, "a constant parameter proves nothing here"
        assert _heights(out["out"].to_list()) == df["h"].to_list()

    def test_a_batch_wide_aggregate_is_the_same_for_every_morsel(self) -> None:
        df = self._frame()
        out = (
            df.lazy()
            .with_columns(
                out=pl.col("image")
                .cv.pipe(_decode().resize(height=pl.col("h").max(), width=4))
                .sink("list")
            )
            .collect(engine="streaming")
        )
        assert set(_heights(out["out"].to_list())) == {df["h"].max()}


@plugin_required
class TestComposedPipelines:
    """Expression parameters survive graph composition.

    ``.pipe()`` on a ``LazyPipelineExpr`` continues one graph into another and
    common subexpressions are merged, both of which rewrite the node the
    parameter is attached to.
    """

    def test_a_continuation_carries_its_own_parameter(self) -> None:
        df = _frame().with_columns(
            crop=pl.Series([4, 4, 4, 4], dtype=pl.Int32),
            side=pl.Series([6, 8, 10, 12], dtype=pl.Int32),
        )
        first = pl.col("image").cv.pipe(_decode().resize(height=12, width=12))
        # A `Pipeline` with no `source()` continues from the upstream node.
        second = first.pipe(
            Pipeline().crop(top=pl.col("crop"), left=0, height=pl.col("side"), width=4)
        )
        out = (
            df.lazy().with_columns(out=second.sink("list")).collect(engine="streaming")
        )
        assert _heights(out["out"].to_list()) == [6, 8, 8, 8]

    def test_two_sinks_sharing_one_parameter_agree(self) -> None:
        """Common subexpression elimination must not merge different values."""
        df = _frame().with_columns(t=pl.Series([4, 6, 8, 10], dtype=pl.Int32))
        out = (
            df.lazy()
            .with_columns(
                a=pl.col("image")
                .cv.pipe(_decode().resize(height=pl.col("t"), width=4))
                .sink("list"),
                b=pl.col("image")
                .cv.pipe(_decode().resize(height=pl.col("t") * 2, width=4))
                .sink("list"),
            )
            .collect(engine="streaming")
        )
        assert _heights(out["a"].to_list()) == [4, 6, 8, 10]
        assert _heights(out["b"].to_list()) == [8, 12, 16, 20]
