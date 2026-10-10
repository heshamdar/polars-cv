"""
Multi-branch workflow benchmarks.

The pipeline scenarios chain operations one after another; real preprocessing
graphs branch. One decoded image feeds a model tensor, a thumbnail and a
statistic; a mask derived from an image gates statistics over the same image;
a blurred copy is subtracted from the original. polars-cv expresses these as
one expression graph (shared nodes run once), every other library as a few
calls on a decoded image.

Each workflow is written once per library, in that library's idiom, and
starts from encoded PNG bytes, so decoding is part of every measurement.
``tests/test_benchmark_workflows.py`` holds each implementation to the
polars-cv result, output by output.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from benchmarks.frameworks import BaseFrameworkAdapter, BenchmarkResult
from benchmarks.utils.data_gen import generate_image_set
from benchmarks.utils.memory import run_timed_with_memory

if TYPE_CHECKING:
    from collections.abc import Collection

#: One image's outputs: name -> array or scalar.
Outputs = dict[str, Any]

UNSHARP_SIGMA = 2.0
UNSHARP_AMOUNT = 1.5
MASK_THRESHOLD = 127


@dataclass(frozen=True)
class Workflow:
    """A multi-branch workflow and its implementations.

    Attributes:
        name: Result name.
        description: One line for reports.
        pattern: The ``data_gen`` image pattern it runs on.
        polars_cv: Builds the expression over the ``images`` column; its
            result is a struct (or list) column ``extract`` unpacks.
        extract: One output row of the polars-cv result -> :data:`Outputs`.
        libraries: Library adapter name -> ``fn(adapter, png_bytes)`` giving
            one image's :data:`Outputs`. A library without an entry has no
            call for the workflow.
    """

    name: str
    description: str
    pattern: str
    polars_cv: Callable[[], pl.Expr]
    extract: Callable[[Any], Outputs]
    libraries: dict[str, Callable[[Any, bytes], Outputs]] = field(default_factory=dict)


def _source() -> Any:
    from polars_cv import Pipeline

    return pl.col("images").cv.pipe(Pipeline().source("image_bytes")).alias("src")


def _arr(struct: dict[str, Any]) -> np.ndarray:
    from polars_cv import numpy_from_struct

    arr = numpy_from_struct(struct)
    return arr[..., 0] if arr.ndim == 3 and arr.shape[2] == 1 else arr


# --- multi_output_etl ------------------------------------------------------
# One decode feeds three outputs: a letterboxed, min-max normalised model
# tensor, a 64x64 grayscale thumbnail and the mean grey level.


def _etl_polars() -> pl.Expr:
    from polars_cv import Pipeline

    src = _source()
    tensor = src.pipe(
        Pipeline()
        .letterbox(height=224, width=224, filter="bilinear")
        .normalize(method="minmax")
    ).alias("tensor")
    gray = src.pipe(Pipeline().grayscale()).alias("gray")
    thumb = gray.pipe(Pipeline().resize(height=64, width=64, filter="bilinear")).alias(
        "thumb"
    )
    mean = gray.pipe(Pipeline().reduce_mean()).alias("mean")
    return tensor.merge_pipe(thumb, mean).sink(
        {"tensor": "numpy", "thumb": "numpy", "mean": "native"}
    )


def _etl_extract(row: dict[str, Any]) -> Outputs:
    return {
        "tensor": _arr(row["tensor"]),
        "thumb": _arr(row["thumb"]),
        "mean": row["mean"],
    }


def _etl_opencv(a: Any, data: bytes) -> Outputs:
    import cv2

    img = a.load_from_bytes(data)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return {
        "tensor": a.normalize(a.letterbox(img, 224, 224)),
        # INTER_AREA: OpenCV's antialiased shrink.
        "thumb": cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA),
        "mean": cv2.mean(gray)[0],
    }


def _etl_pillow(a: Any, data: bytes) -> Outputs:
    from PIL import Image, ImageStat

    img = a.load_from_bytes(data)
    gray = img.convert("L")
    return {
        "tensor": a.normalize(a.letterbox(img, 224, 224)),
        "thumb": np.asarray(gray.resize((64, 64), Image.Resampling.BILINEAR)),
        "mean": ImageStat.Stat(gray).mean[0],
    }


def _etl_pyvips(a: Any, data: bytes) -> Outputs:
    # Three branches read the decoded image: decode it once.
    img = a.load_from_bytes(data).copy_memory()
    gray = a.grayscale(img).copy_memory()
    return {
        "tensor": a.normalize(a.letterbox(img, 224, 224)).numpy(),
        "thumb": a.resize(gray, 64, 64).numpy(),
        "mean": gray.avg(),
    }


# --- unsharp_mask ----------------------------------------------------------
# Two branches of one image recombined: img + amount * (img - blur(img)),
# clamped back to u8.


def _unsharp_polars() -> pl.Expr:
    from polars_cv import Pipeline

    f = _source().pipe(Pipeline().cast(dtype="f32")).alias("f")
    blurred = f.pipe(Pipeline().blur(sigma=UNSHARP_SIGMA)).alias("blurred")
    detail = f.subtract(blurred).pipe(Pipeline().scale(UNSHARP_AMOUNT))
    out = f.add(detail).pipe(Pipeline().clamp(0, 255).cast(dtype="u8"))
    return out.sink("numpy")


def _unsharp_extract(struct: dict[str, Any]) -> Outputs:
    return {"image": _arr(struct)}


def _unsharp_opencv(a: Any, data: bytes) -> Outputs:
    import cv2

    img = a.load_from_bytes(data)
    blurred = a.blur(img, UNSHARP_SIGMA)
    # Saturating u8 weighted sum: (1 + amount) * img - amount * blurred.
    sharp = cv2.addWeighted(img, 1 + UNSHARP_AMOUNT, blurred, -UNSHARP_AMOUNT, 0)
    return {"image": sharp}


def _unsharp_pillow(a: Any, data: bytes) -> Outputs:
    from PIL import ImageFilter

    img = a.load_from_bytes(data)
    # With threshold 0 Pillow's UnsharpMask is exactly this formula.
    mask = ImageFilter.UnsharpMask(
        radius=UNSHARP_SIGMA, percent=int(UNSHARP_AMOUNT * 100), threshold=0
    )
    return {"image": np.asarray(img.filter(mask))}


def _unsharp_pyvips(a: Any, data: bytes) -> Outputs:
    img = a.load_from_bytes(data)
    blurred = a.blur(img, UNSHARP_SIGMA)
    sharp = img * (1 + UNSHARP_AMOUNT) - blurred * UNSHARP_AMOUNT
    return {"image": sharp.clamp(min=0, max=255).cast("uchar").numpy()}


# --- masked_stats ----------------------------------------------------------
# A mask derived from the image gates statistics over the same image: the
# mean and max of the grayscale image with everything at or below the
# threshold zeroed.


def _masked_polars() -> pl.Expr:
    from polars_cv import Pipeline

    gray = _source().pipe(Pipeline().grayscale()).alias("gray")
    mask = gray.pipe(Pipeline().threshold(value=MASK_THRESHOLD)).alias("mask")
    return gray.apply_mask(mask).statistics(include=["mean", "max"])


def _masked_extract(row: dict[str, Any]) -> Outputs:
    return {"mean": row["mean"], "max": row["max"]}


def _masked_opencv(a: Any, data: bytes) -> Outputs:
    import cv2

    gray = cv2.cvtColor(a.load_from_bytes(data), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    _, hi, _, _ = cv2.minMaxLoc(masked)
    return {"mean": cv2.mean(masked)[0], "max": hi}


def _masked_pillow(a: Any, data: bytes) -> Outputs:
    from PIL import Image, ImageStat

    gray = a.load_from_bytes(data).convert("L")
    mask = gray.point(lambda p: 255 if p > MASK_THRESHOLD else 0)
    masked = Image.new("L", gray.size, 0)
    masked.paste(gray, mask=mask)
    stat = ImageStat.Stat(masked)
    return {"mean": stat.mean[0], "max": stat.extrema[0][1]}


def _masked_pyvips(a: Any, data: bytes) -> Outputs:
    gray = a.grayscale(a.load_from_bytes(data))
    # Two statistics read it: materialise once.
    masked = (gray > MASK_THRESHOLD).ifthenelse(gray, 0).copy_memory()
    return {"mean": masked.avg(), "max": masked.max()}


# --- mask_to_contours ------------------------------------------------------
# Segment, clean and trace: grayscale -> blur -> threshold -> opening ->
# external contours -> how many and their total area.


def _contours_polars() -> pl.Expr:
    from polars_cv import Pipeline

    pipe = (
        Pipeline()
        .source("image_bytes")
        .grayscale()
        .blur(sigma=1.5)
        .threshold(value=MASK_THRESHOLD)
        .morphology_open(ksize=3)
        .extract_contours(mode="external", method="simple")
        .area()
    )
    return pl.col("images").cv.pipe(pipe).sink("native")


def _contours_extract(areas: list[float]) -> Outputs:
    return {"count": len(areas), "total_area": float(sum(areas))}


def _contours_opencv(a: Any, data: bytes) -> Outputs:
    import cv2

    gray = cv2.cvtColor(a.load_from_bytes(data), cv2.COLOR_RGB2GRAY)
    gray = a.blur(gray, 1.5)
    _, mask = cv2.threshold(gray, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return {
        "count": len(contours),
        "total_area": float(sum(cv2.contourArea(c) for c in contours)),
    }


WORKFLOWS: dict[str, Workflow] = {
    w.name: w
    for w in [
        Workflow(
            name="multi_output_etl",
            description=(
                "One decode -> letterboxed 224 tensor + 64x64 gray thumbnail "
                "+ mean grey level"
            ),
            pattern="mixed",
            polars_cv=_etl_polars,
            extract=_etl_extract,
            libraries={
                "opencv": _etl_opencv,
                "pillow": _etl_pillow,
                "pyvips": _etl_pyvips,
            },
        ),
        Workflow(
            name="unsharp_mask",
            description="img + 1.5 * (img - gaussian(img, 2)), two branches",
            pattern="mixed",
            polars_cv=_unsharp_polars,
            extract=_unsharp_extract,
            libraries={
                "opencv": _unsharp_opencv,
                "pillow": _unsharp_pillow,
                "pyvips": _unsharp_pyvips,
            },
        ),
        Workflow(
            name="masked_stats",
            description="Threshold mask of the image gates its mean and max",
            pattern="mixed",
            polars_cv=_masked_polars,
            extract=_masked_extract,
            libraries={
                "opencv": _masked_opencv,
                "pillow": _masked_pillow,
                "pyvips": _masked_pyvips,
            },
        ),
        Workflow(
            name="mask_to_contours",
            description=(
                "Gray -> blur -> threshold -> opening -> external contours -> areas"
            ),
            pattern="blobs",
            polars_cv=_contours_polars,
            extract=_contours_extract,
            libraries={"opencv": _contours_opencv},
        ),
    ]
}


def _collect(workflow: Workflow, df: pl.DataFrame, *, streaming: bool) -> pl.Series:
    lf = df.lazy().select(out=workflow.polars_cv())
    return lf.collect(engine="streaming" if streaming else "auto")["out"]


def run_polars_cv(
    workflow: Workflow, images: list[bytes], *, streaming: bool = False
) -> list[Outputs]:
    """Run ``workflow`` through polars-cv and unpack every row."""
    out = _collect(workflow, pl.DataFrame({"images": images}), streaming=streaming)
    return [workflow.extract(row) for row in out.to_list()]


def run_library(
    workflow: Workflow, adapter: BaseFrameworkAdapter, images: list[bytes]
) -> list[Outputs]:
    """Run ``workflow`` through the library ``adapter`` wraps."""
    fn = workflow.libraries.get(adapter.name)
    if fn is None:
        msg = f"{adapter.name} has no implementation of {workflow.name}"
        raise NotImplementedError(msg)
    return [fn(adapter, data) for data in images]


def _timed(run: Callable[[], Any], warmup: int, iterations: int) -> tuple[float, float]:
    for _ in range(warmup):
        run()
    total, peak = 0.0, 0.0
    for _ in range(iterations):
        _, elapsed, mem = run_timed_with_memory(run)
        total += elapsed
        peak = max(peak, mem.peak_memory_mb)
    return total / iterations, peak


def _runner(
    workflow: Workflow,
    adapter: BaseFrameworkAdapter,
    df: pl.DataFrame,
    images: list[bytes],
) -> Callable[[], Any] | None:
    """What one timed run of ``workflow`` on ``adapter`` calls, or ``None``
    when the adapter has no implementation of it."""
    streaming = getattr(adapter, "streaming", None)
    if streaming is not None:
        return lambda: _collect(workflow, df, streaming=streaming)
    if adapter.name in workflow.libraries:
        return lambda: run_library(workflow, adapter, images)
    return None


def run_all_workflows(
    adapters: list[BaseFrameworkAdapter],
    image_counts: list[int],
    image_sizes: list[tuple[int, int]],
    warmup_iterations: int = 3,
    benchmark_iterations: int = 10,
    verbose: bool = True,
    names: Collection[str] | None = None,
) -> list[BenchmarkResult]:
    """
    Run every workflow on every adapter that implements it.

    polars-cv adapters run the expression graph (eager or streaming as the
    adapter says) over a prebuilt DataFrame of PNG bytes; library adapters
    run their implementation over the same bytes. Adapters without an
    implementation (torchvision, and libraries in a workflow's gap) are
    skipped.

    Args:
        adapters: Framework adapters to benchmark.
        image_counts: Image counts to test.
        image_sizes: ``(width, height)`` sizes to test.
        warmup_iterations: Untimed runs before measuring.
        benchmark_iterations: Timed runs, averaged.
        verbose: Print progress.
        names: If set, only these workflows.

    Returns:
        One result per (workflow, adapter, size, count) that ran.
    """
    results: list[BenchmarkResult] = []
    selected = [w for w in WORKFLOWS.values() if names is None or w.name in names]
    for width, height in image_sizes:
        for count in image_counts:
            if verbose:
                print(f"\n  {width}x{height}, {count} images", flush=True)
            for workflow in selected:
                images = generate_image_set(
                    count=count,
                    height=height,
                    width=width,
                    channels=3,
                    pattern=workflow.pattern,
                ).image_bytes
                df = pl.DataFrame({"images": images})
                for adapter in adapters:
                    run = _runner(workflow, adapter, df, images)
                    if run is None:
                        continue
                    if verbose:
                        print(
                            f"    {adapter.name}/{workflow.name}...", end="", flush=True
                        )
                    try:
                        avg, peak = _timed(run, warmup_iterations, benchmark_iterations)
                    except NotImplementedError:
                        print(" unsupported", flush=True)
                        continue
                    except Exception as e:  # noqa: BLE001
                        print(f" ERROR: {e}", flush=True)
                        continue
                    results.append(
                        BenchmarkResult(
                            framework=adapter.name,
                            operation=f"workflow_{workflow.name}",
                            image_count=count,
                            image_size=(width, height),
                            total_time_seconds=avg,
                            throughput_images_per_second=count / avg,
                            latency_ms_per_image=avg / count * 1000,
                            peak_memory_mb=peak,
                        )
                    )
                    if verbose:
                        print(f" {count / avg:.1f} img/s", flush=True)
    return results
