"""Targeted cases for the paths the adapter scenarios do not reach.

``single_ops``/``pipelines``/``e2e`` time image pipelines through the framework
adapters, which always decode PNG and sink to numpy. That leaves whole
subsystems unmeasured: the geometry accessors (``.contour``/``.point``), the
tensor sinks (``array``/``list``), JPEG, re-encoding, and the ``blob`` source.
Each case here times one of them directly, eager.

Case names are grouped by prefix so a selector can take a subsystem at once
(``targeted:geom_*``): ``codec_`` (decode/encode), ``sink_`` (tensor sinks),
``blob_`` (the blob source), ``geom_`` (geometry accessors), ``split_`` (how
calls spread over the plugin's pool across engines and morsel shapes).

The ``split_`` cases only measure something with more than one thread: run
them with ``--threads`` above 1 (the suite pins 1 by default). One iteration
is a whole query, and they run several in a process on purpose: whether a
call spread once depended on how the previous call had run.

Image cases run once per suite (count, size). Geometry cases run once per
count, over ``count * ROWS_PER_IMAGE[kind]`` rows, and report ``image_size`` as
``(0, 0)`` — a geometry row has no image size. Point-only cases (``points``)
run over a 10× longer frame of points alone: at the contour frame's length a
call takes ~5 ms, and a same-binary self-check spread them ±7.6%.

Known noise: ``geom_contour_translate`` allocates its whole nested output per
call, and which of two allocator states a process lands in moves it ~15%
between runs of one binary (the others hold within ~5%). Confirm a verdict on
it by rerunning both sides.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path
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

# A geometry row is far cheaper than an image, so more of them make one call:
# enough that each call runs tens of milliseconds or more.
ROWS_PER_IMAGE: dict[str, int] = {"geometry": 300, "points": 3000}


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
        return _geometry(self.count * ROWS_PER_IMAGE["geometry"])

    @property
    def points(self) -> pl.DataFrame:
        return _points(self.count * ROWS_PER_IMAGE["points"])


def _stored(build: Callable[..., pl.DataFrame]) -> Callable[..., pl.DataFrame]:
    """Build an input frame once per machine, then read it back.

    Each case runs in its own process (``run_all_targeted``), and building the
    inputs is most of a case's cost — encoding 300 PNGs takes ~14 s. The frames
    are deterministic, so they are written once as Arrow IPC under the temp
    directory and read back after — into memory, not memory-mapped, so a case
    measures heap-resident inputs as before. The key covers this module's
    source, so changing how an input is made cannot reuse the old one.
    """
    root = _INPUT_ROOT

    @cache
    def load(*args: Any) -> pl.DataFrame:
        path = root / f"{build.__name__}-{'-'.join(map(str, args))}.arrow"
        if not path.exists():
            root.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            build(*args).write_ipc(tmp)
            tmp.replace(path)
        return pl.read_ipc(path)

    return load


_SOURCE_KEY = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
#: Where the inputs live, per machine and per version of this module.
_INPUT_ROOT = Path(tempfile.gettempdir()) / "polars-cv-bench-inputs" / _SOURCE_KEY


@_stored
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


@_stored
def _blobs(count: int, height: int, width: int) -> pl.DataFrame:
    png = _encoded(count, height, width, "PNG")
    pipe = Pipeline().source("image_bytes").cast("f32")
    return png.select(pl.col("img").cv.pipe(pipe).sink("blob").alias("img"))


@_stored
def _points(rows: int) -> pl.DataFrame:
    return pl.DataFrame(
        {"p": [{"x": 10.0, "y": 10.0}] * rows, "q": [{"x": 3.0, "y": 4.0}] * rows}
    )


@_stored
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


def _heavy() -> Pipeline:
    """Enough work per row that how a call's rows spread decides its time."""
    return _src().resize(height=128, width=128).blur(sigma=2.0)


def _streaming_then_eager(i: _Inputs) -> Callable[[], Any]:
    """A streaming run (concurrent morsel calls), then the same pipeline eager
    (one lone call), as in a notebook: the eager call must use the pool."""
    df = i.png
    expr = pl.col("img").cv.pipe(_heavy()).sink("numpy")

    def run() -> Any:
        df.lazy().select(expr).collect(engine="streaming")
        return df.select(expr)

    return run


def _uneven_row_groups(i: _Inputs) -> Callable[[], Any]:
    """A streaming scan of Parquet row groups of two thirds and one third of
    the rows: one call per row group, so a large call runs beside small ones
    (the last row group is split across the pipelines)."""
    path = _INPUT_ROOT / f"uneven-{i.count}-{i.height}-{i.width}.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        i.png.write_parquet(tmp, row_group_size=max(1, 2 * i.count // 3))
        tmp.replace(path)
    expr = pl.col("img").cv.pipe(_heavy()).sink("numpy")
    return lambda: pl.scan_parquet(path).select(expr).collect(engine="streaming")


def _geom(
    expr: Callable[[], pl.Expr], frame: str = "geometry"
) -> Callable[[_Inputs], Callable[[], Any]]:
    def build(i: _Inputs) -> Callable[[], Any]:
        df, e = getattr(i, frame), expr()
        return lambda: df.select(e)

    return build


@dataclass(frozen=True)
class Case:
    name: str
    kind: Literal["image", "geometry", "points"]
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
    Case("split_streaming_then_eager", "image", _streaming_then_eager),
    Case("split_streaming_uneven_row_groups", "image", _uneven_row_groups),
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
        "points",
        _geom(lambda: pl.col("p").point.translate(1.0, 2.0), "points"),
    ),
    Case(
        "geom_point_distance",
        "points",
        _geom(lambda: pl.col("p").point.distance(pl.col("q")), "points"),
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


def run_case(
    name: str,
    count: int,
    size: tuple[int, int],
    warmup_iterations: int,
    benchmark_iterations: int,
) -> BenchmarkResult:
    """Time one case in this process."""
    case = next(c for c in CASES if c.name == name)
    n = count if case.kind == "image" else count * ROWS_PER_IMAGE[case.kind]
    inputs = _Inputs(count, *size)
    return timed_result(
        name, case.build(inputs), n, size, warmup_iterations, benchmark_iterations
    )


def run_all_targeted(
    image_counts: list[int],
    image_sizes: list[tuple[int, int]],
    warmup_iterations: int = 3,
    benchmark_iterations: int = 10,
    names: Collection[str] | None = None,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """Run the selected cases (all when ``names`` is None), each in its own process.

    In one process a case's timing depended on the allocator state the cases
    before it left: ``geom_contour_translate`` measured 588-597k rows/s alone
    and 545-704k inside the suite. ``--select`` changes which cases precede
    which, so a shared process would make a case's number depend on what else
    was selected. The child inherits the parent's environment, thread pins
    included; a child that fails raises here.
    """
    from benchmarks.frameworks import BenchmarkResult

    cases = [c for c in CASES if names is None or c.name in names]
    runs: list[tuple[str, int, tuple[int, int]]] = []
    for count in image_counts:
        for size in image_sizes:
            runs += [(c.name, count, size) for c in cases if c.kind == "image"]
        runs += [(c.name, count, (0, 0)) for c in cases if c.kind != "image"]

    results: list[BenchmarkResult] = []
    for name, count, size in runs:
        if verbose:
            print(f"    targeted/{name} count={count} size={size}", flush=True)
        spec = [name, count, list(size), warmup_iterations, benchmark_iterations]
        proc = subprocess.run(
            [sys.executable, "-m", "benchmarks.scenarios.targeted", json.dumps(spec)],
            cwd=_PACKAGE_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            msg = f"targeted case {name} failed:\n{proc.stderr}"
            raise RuntimeError(msg)
        record = json.loads(proc.stdout.strip().splitlines()[-1])
        record["image_size"] = tuple(record["image_size"])
        results.append(BenchmarkResult(**record))
    return results


# The directory `benchmarks` is importable from, for the per-case child.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    """Child entry point: time one case, print its result as one JSON line."""
    name, count, size, warmup, iterations = json.loads((argv or sys.argv[1:])[0])
    result = run_case(name, count, tuple(size), warmup, iterations)
    print(json.dumps(dataclasses.asdict(result)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
