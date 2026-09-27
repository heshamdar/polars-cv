"""Targeted cases for the paths the adapter scenarios do not reach.

``single_ops``/``pipelines``/``e2e`` time image pipelines through the framework
adapters, which always decode PNG and sink to numpy. That leaves whole
subsystems unmeasured: the geometry accessors (``.contour``/``.point``), the
tensor sinks (``array``/``list``), JPEG, re-encoding, and the ``blob`` source.
Each case here times one of them directly, eager.

Case names are grouped by prefix so a selector can take a subsystem at once
(``targeted:geom_*``): ``codec_`` (decode/encode), ``sink_`` (tensor sinks),
``blob_`` (the blob source), ``geom_`` (geometry accessors).

Image cases run once per suite (count, size); geometry cases once per count,
over ``count * GEOMETRY_ROWS_PER_IMAGE`` rows, and report ``image_size`` as
``(0, 0)`` — a geometry row has no image size.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl
from PIL import Image

import polars_cv  # noqa: F401  (registers the namespaces)
from benchmarks.utils.timing import timed_result
from polars_cv import Pipeline

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    from benchmarks.frameworks import BenchmarkResult

# A geometry row is far cheaper than an image, so more of them make one call.
GEOMETRY_ROWS_PER_IMAGE = 100


@dataclass(frozen=True)
class _Inputs:
    """The frames one (count, size) runs over, built once and shared by cases."""

    count: int
    height: int
    width: int

    @property
    def png(self) -> pl.DataFrame:
        return _encoded(self.count, self.height, self.width, "PNG")

    @property
    def jpeg(self) -> pl.DataFrame:
        return _encoded(self.count, self.height, self.width, "JPEG")

    @property
    def blobs(self) -> pl.DataFrame:
        return _blobs(self.count, self.height, self.width)

    @property
    def geometry(self) -> pl.DataFrame:
        return _geometry(self.count * GEOMETRY_ROWS_PER_IMAGE)


@cache
def _encoded(count: int, height: int, width: int, fmt: str) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    out = []
    for _ in range(count):
        # Quantised noise compresses like a photo rather than like noise.
        arr = (rng.integers(0, 255, (height, width, 3)) // 64 * 64).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format=fmt)
        out.append(buf.getvalue())
    return pl.DataFrame({"img": out})


@cache
def _blobs(count: int, height: int, width: int) -> pl.DataFrame:
    png = _encoded(count, height, width, "PNG")
    pipe = Pipeline().source("image_bytes").cast("f32")
    return png.select(pl.col("img").cv.pipe(pipe).sink("blob").alias("img"))


@cache
def _geometry(rows: int) -> pl.DataFrame:
    ts = np.linspace(0, 6.2, 40)
    ring = [
        {"x": float(np.cos(t) * 10 + 10), "y": float(np.sin(t) * 10 + 10)} for t in ts
    ]
    contour = {"exterior": ring, "holes": [], "is_closed": True}
    return pl.DataFrame(
        {
            "c": [contour] * rows,
            "d": [contour] * rows,
            "p": [{"x": 10.0, "y": 10.0}] * rows,
            "q": [{"x": 3.0, "y": 4.0}] * rows,
        }
    )


def _pipe(
    frame: str, pipe: Callable[[], Pipeline], sink: str
) -> Callable[[_Inputs], Callable[[], Any]]:
    """Run ``pipe`` over one of the image frames into ``sink``."""

    def build(i: _Inputs) -> Callable[[], Any]:
        df = getattr(i, frame)
        expr = pl.col("img").cv.pipe(pipe()).sink(sink)
        return lambda: df.select(expr)

    return build


def _array(
    pipe: Callable[[], Pipeline], shape: Callable[[_Inputs], list[int]]
) -> Callable[[_Inputs], Callable[[], Any]]:
    """Run ``pipe`` over the PNG frame into a fixed-shape ``array`` sink."""

    def build(i: _Inputs) -> Callable[[], Any]:
        df = i.png
        expr = pl.col("img").cv.pipe(pipe()).sink("array", shape=shape(i))
        return lambda: df.select(expr)

    return build


def _geom(expr: Callable[[], pl.Expr]) -> Callable[[_Inputs], Callable[[], Any]]:
    def build(i: _Inputs) -> Callable[[], Any]:
        df, e = i.geometry, expr()
        return lambda: df.select(e)

    return build


@dataclass(frozen=True)
class Case:
    name: str
    kind: Literal["image", "geometry"]
    build: Callable[[_Inputs], Callable[[], Any]]


# Pipelines are built when a case runs, not at import: listing the cases (for
# `--select`) must not need the compiled plugin.
def _src() -> Pipeline:
    return Pipeline().source("image_bytes")


def _u8() -> Pipeline:
    return Pipeline().source("image_bytes", dtype="u8")


CASES: tuple[Case, ...] = (
    Case("codec_png_decode", "image", _pipe("png", _src, "numpy")),
    Case("codec_jpeg_decode", "image", _pipe("jpeg", _src, "numpy")),
    Case("codec_png_roundtrip", "image", _pipe("png", _src, "png")),
    Case("codec_jpeg_roundtrip", "image", _pipe("jpeg", _src, "jpeg")),
    Case("sink_array_u8", "image", _array(_u8, lambda i: [i.height, i.width, 3])),
    Case("sink_list_u8", "image", _pipe("png", _u8, "list")),
    Case(
        "sink_array_transposed_u8",
        "image",
        _array(
            lambda: _u8().transpose(axes=[1, 0, 2]), lambda i: [i.width, i.height, 3]
        ),
    ),
    Case(
        "blob_f32_to_numpy",
        "image",
        _pipe("blobs", lambda: Pipeline().source("blob"), "numpy"),
    ),
    Case(
        "blob_f32_scale_to_numpy",
        "image",
        _pipe("blobs", lambda: Pipeline().source("blob").scale(2.0), "numpy"),
    ),
    Case("geom_contour_area", "geometry", _geom(lambda: pl.col("c").contour.area())),
    Case(
        "geom_contour_translate",
        "geometry",
        _geom(lambda: pl.col("c").contour.translate(1.0, 2.0)),
    ),
    Case(
        "geom_contour_iou",
        "geometry",
        _geom(lambda: pl.col("c").contour.iou(pl.col("d"))),
    ),
    Case(
        "geom_contour_contains_point",
        "geometry",
        _geom(lambda: pl.col("c").contour.contains_point(pl.col("p"))),
    ),
    Case(
        "geom_contour_rasterize",
        "geometry",
        _geom(
            lambda: (
                pl.col("c")
                .cv.pipe(Pipeline().source("contour").rasterize(width=32, height=32))
                .sink("numpy")
            )
        ),
    ),
    Case(
        "geom_point_translate",
        "geometry",
        _geom(lambda: pl.col("p").point.translate(1.0, 2.0)),
    ),
    Case(
        "geom_point_distance",
        "geometry",
        _geom(lambda: pl.col("p").point.distance(pl.col("q"))),
    ),
    Case(
        "geom_point_distance_to_contour",
        "geometry",
        _geom(lambda: pl.col("p").point.distance_to_contour(pl.col("c"))),
    ),
)


def case_names() -> list[str]:
    """Every case, by the ``operation`` name its result carries."""
    return [c.name for c in CASES]


def run_all_targeted(
    image_counts: list[int],
    image_sizes: list[tuple[int, int]],
    warmup_iterations: int = 3,
    benchmark_iterations: int = 10,
    names: Collection[str] | None = None,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """Run the selected cases (all when ``names`` is None)."""
    cases = [c for c in CASES if names is None or c.name in names]
    results: list[BenchmarkResult] = []
    for count in image_counts:
        runs: list[tuple[Case, _Inputs, tuple[int, int], int]] = []
        for h, w in image_sizes:
            runs += [
                (c, _Inputs(count, h, w), (h, w), count)
                for c in cases
                if c.kind == "image"
            ]
        rows = count * GEOMETRY_ROWS_PER_IMAGE
        geo_inputs = _Inputs(count, 0, 0)
        runs += [(c, geo_inputs, (0, 0), rows) for c in cases if c.kind == "geometry"]
        for case, inputs, size, n in runs:
            if verbose:
                print(f"    targeted/{case.name} n={n} size={size}", flush=True)
            results.append(
                timed_result(
                    case.name,
                    case.build(inputs),
                    n,
                    size,
                    warmup_iterations,
                    benchmark_iterations,
                )
            )
    return results
