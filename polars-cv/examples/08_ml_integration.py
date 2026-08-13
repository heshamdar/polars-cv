"""ML integration patterns: numpy/torch sinks and dataset wrapping.

Run:
    uv run python polars-cv/examples/08_ml_integration.py
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image
from polars_cv import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUMPY_OUTPUT_SCHEMA,
    Pipeline,
    numpy_from_struct,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"


def make_image(seed: int, size: int = 112) -> np.ndarray:
    """Create a synthetic RGB image."""
    rng = np.random.default_rng(seed)
    y, x = np.indices((size, size))
    rgb = np.stack([(x * 2) % 255, (y * 2) % 255, ((x + y) * 3) % 255], axis=2).astype(
        np.uint8
    )
    rgb ^= rng.integers(0, 12, size=rgb.shape, dtype=np.uint8)
    return rgb


def to_png_bytes(image: np.ndarray) -> bytes:
    """Encode image as PNG bytes."""
    buf = BytesIO()
    Image.fromarray(image, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


class PreprocessedPolarsDataset:
    """Tiny dataset wrapper around preprocessed Polars results."""

    def __init__(self, frame: pl.DataFrame, image_col: str, label_col: str) -> None:
        """Store the frame and column names."""
        self._frame = frame
        self._image_col = image_col
        self._label_col = label_col

    def __len__(self) -> int:
        """Return number of samples."""
        return self._frame.height

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        """Return one sample as `(numpy_image, label)`."""
        row = self._frame.row(index, named=True)
        image = numpy_from_struct(row[self._image_col], copy=False)
        label = int(row[self._label_col])
        return image, label


def build_demo_frame() -> pl.DataFrame:
    """Build a small labeled image DataFrame."""
    images = [to_png_bytes(make_image(seed)) for seed in (1, 3, 5, 7)]
    labels = [0, 1, 1, 0]
    return pl.DataFrame({"image": images, "label": labels})


def preprocess_for_numpy(df: pl.DataFrame) -> pl.DataFrame:
    """Create a CHW float32 tensor-like preprocessing output."""
    # `assert_shape` names its dimensions `height`/`width`/`channels`, i.e. it
    # describes an [H, W, C] buffer. So it belongs *before* the transpose to
    # CHW, not after: asserting HWC dimensions on an already-transposed buffer
    # tells the planner to publish [96, 96, 3] for something that executes as
    # [3, 96, 96], and the query fails at collect() with a plan/exec mismatch.
    pipe = (
        Pipeline()
        .source("image_bytes")
        .resize(height=96, width=96)
        .normalize(method="preset", mean=IMAGENET_MEAN, std=IMAGENET_STD)
        .assert_shape(channels=3, height=96, width=96)
        .transpose([2, 0, 1])
    )
    return df.with_columns(processed=pl.col("image").cv.pipe(pipe).sink("numpy"))


def demonstrate_sink_formats(df: pl.DataFrame) -> pl.DataFrame:
    """Show multiple sink formats from one source expression."""
    img = (
        pl.col("image")
        .cv.pipe(Pipeline().source("image_bytes").resize(height=64, width=64))
        .alias("img")
    )
    sink_df = df.with_columns(
        numpy_struct=img.sink("numpy"),
        png_bytes=img.sink("png"),
        jpeg_bytes=img.sink("jpeg"),
        blob_bytes=img.sink("blob"),
        native=img.pipe(Pipeline().grayscale().reduce_mean()).sink("native"),
    )
    return sink_df


def demonstrate_array_and_list_sinks() -> pl.DataFrame:
    """Demonstrate deterministic list/array sinks from list source."""
    tensor = np.arange(4 * 4, dtype=np.float32).reshape(4, 4, 1).tolist()
    df = pl.DataFrame({"tensor": [tensor]})
    pipe = Pipeline().source("list", dtype="f32")
    # The `array` sink needs its shape at planning time. For an image source
    # `.assert_shape()` supplies it, but not for a `list` source: the plan-time
    # rank stays unresolved, so the hints never become an `expected_shape` and
    # the sink refuses. Pass the shape here instead.
    return df.with_columns(
        tensor_array=pl.col("tensor").cv.pipe(pipe).sink("array", shape=[4, 4, 1]),
        tensor_list=pl.col("tensor").cv.pipe(pipe).sink("list"),
    )


def maybe_torch_demo(processed: pl.DataFrame) -> None:
    """Try `sink('torch')` and report shape when torch is available."""
    try:
        import torch  # noqa: F401
    except Exception:
        print("\nTorch is not installed; skipping `sink('torch')` section.")
        return

    torch_df = processed.with_columns(
        torch_tensor=pl.col("image")
        .cv.pipe(
            Pipeline()
            .source("image_bytes")
            .resize(height=64, width=64)
            .normalize(method="preset", mean=IMAGENET_MEAN, std=IMAGENET_STD),
        )
        .sink("torch"),
    )
    print("\nTorch sink sample:")
    print(torch_df.select("torch_tensor").head(1))


def main() -> None:
    """Run ML integration examples."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_demo_frame()
    processed = preprocess_for_numpy(df)

    print("NUMPY_OUTPUT_SCHEMA:")
    print(NUMPY_OUTPUT_SCHEMA)
    print("\nPreprocessed schema:")
    print(processed.schema)

    first = numpy_from_struct(processed["processed"][0], copy=False)
    print("First sample CHW shape:", first.shape, "dtype:", first.dtype)

    dataset = PreprocessedPolarsDataset(
        processed.with_columns(label=pl.col("label")), "processed", "label"
    )
    x0, y0 = dataset[0]
    print("Dataset sample:", x0.shape, y0)

    sinks = demonstrate_sink_formats(df)
    print("\nSink formats summary:")
    print(
        sinks.select(
            pl.col("png_bytes").bin.size().alias("png_size"),
            pl.col("jpeg_bytes").bin.size().alias("jpeg_size"),
            pl.col("blob_bytes").bin.size().alias("blob_size"),
            "native",
            "numpy_struct",
        ).head(1)
    )
    array_list_demo = demonstrate_array_and_list_sinks()
    print("\nArray/list sink demo:")
    print(array_list_demo)

    maybe_torch_demo(processed)

    # Save one processed sample as image for quick visual verification.
    out_img = (np.clip((first.transpose(1, 2, 0) * 40.0) + 128.0, 0, 255)).astype(
        np.uint8
    )
    Image.fromarray(out_img, mode="RGB").save(OUTPUT_DIR / "08_ml_processed_sample.png")
    print("Saved:", OUTPUT_DIR / "08_ml_processed_sample.png")


if __name__ == "__main__":
    main()
