"""Sources and sinks: how an array becomes a column, and how a column comes back.

Each :class:`SourceSpec` knows how to put an ``[H, W, C]`` array into a Polars
column the engine can read, and which dtypes and channel counts that format
can carry. Each :class:`SinkSpec` knows how to read an output column back into
NumPy **without going through polars-cv** wherever a third-party decoder
exists (OpenCV for image codecs, Polars itself for ``list``/``array``) — a
round trip decoded by the code under test would agree with itself whatever it
did.

Both registries are keyed by name and checked against the engine's I/O
catalogue (``meta/test_parity_ratchets.py``): a source or sink the catalogue
names must appear here or in :data:`SOURCE_EXEMPT`/:data:`SINK_EXEMPT` with a
reason, so a new format cannot ship outside the invariance sweeps.

The PNG/TIFF/WebP/JPEG encoders here are not conftest's ``make_*_png``
helpers: those build fixed 8-bit test patterns, while a source encoder must
carry an arbitrary array of any dtype and channel count its format admits.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib.util
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import polars as pl
from PIL import Image

from polars_cv import Pipeline, numpy_from_column, numpy_from_struct
from tests.parity.framework.images import DTYPES, dtype_name

#: Polars element type per engine dtype spelling.
POLARS_DTYPES: dict[str, pl.DataType] = {
    "u8": pl.UInt8,
    "i8": pl.Int8,
    "u16": pl.UInt16,
    "i16": pl.Int16,
    "u32": pl.UInt32,
    "i32": pl.Int32,
    "u64": pl.UInt64,
    "i64": pl.Int64,
    "f32": pl.Float32,
    "f64": pl.Float64,
}

ALL_DTYPES = tuple(DTYPES)
ALL_CHANNELS = (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Image codecs (encoders for the byte sources, decoders for the codec sinks)
# ---------------------------------------------------------------------------


def _to_bgr(arr: np.ndarray) -> np.ndarray:
    """OpenCV's channel order for a 3/4-channel array (2-D passes through)."""
    if arr.ndim == 3 and arr.shape[2] == 3:
        return np.ascontiguousarray(arr[..., ::-1])
    if arr.ndim == 3 and arr.shape[2] == 4:
        return np.ascontiguousarray(arr[..., [2, 1, 0, 3]])
    return arr


def _squeeze_gray(arr: np.ndarray) -> np.ndarray:
    return arr[..., 0] if arr.ndim == 3 and arr.shape[2] == 1 else arr


def encode_codec(arr: np.ndarray, fmt: str) -> bytes:
    """Encode ``[H, W, C]`` *arr* as *fmt* (``png``/``tiff``/``webp``/``jpeg``)."""
    channels = arr.shape[2]
    if fmt == "webp":
        # exact=True: without it libwebp rewrites the colour of fully
        # transparent pixels, and the source would not be lossless.
        return _pil_save(arr, "WEBP", lossless=True, quality=100, exact=True)
    if fmt == "jpeg":
        return _pil_save(arr, "JPEG", quality=95)
    if channels == 2:
        # Gray+alpha: OpenCV has no two-channel writer; Pillow's "LA" is 8-bit.
        if arr.dtype != np.uint8:
            msg = f"no encoder for a 2-channel {arr.dtype} {fmt}"
            raise ValueError(msg)
        return _pil_save(arr, fmt.upper())
    ok, data = cv2.imencode(f".{'tif' if fmt == 'tiff' else fmt}", _to_bgr(arr))
    if not ok:  # pragma: no cover - the carries() tables admit only what encodes
        msg = f"OpenCV could not encode {arr.dtype}{arr.shape} as {fmt}"
        raise ValueError(msg)
    return data.tobytes()


def _pil_save(arr: np.ndarray, fmt: str, **kwargs: Any) -> bytes:
    import io

    image = Image.fromarray(_squeeze_gray(arr))
    buf = io.BytesIO()
    image.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def decode_codec(data: bytes, channels: int) -> np.ndarray:
    """Decode codec bytes with OpenCV into RGB(A) order.

    Pillow cannot hold 16-bit colour (it silently reduces it to 8 bits), so
    OpenCV is the decoder. OpenCV has no gray+alpha mode: it expands one to
    BGRA, which is folded back to two channels after checking the colour
    planes really are equal; or, for a gray+alpha TIFF, it drops the alpha,
    and Pillow (whose ``LA`` mode holds 8-bit gray+alpha) decodes it instead.
    """
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        msg = "OpenCV could not decode the sink's bytes"
        raise AssertionError(msg)
    if channels == 2 and decoded.ndim == 2:
        import io

        image = Image.open(io.BytesIO(data))
        if image.mode != "LA":
            msg = f"a gray+alpha output decoded as {image.mode}"
            raise AssertionError(msg)
        return np.asarray(image)
    decoded = _to_bgr(decoded)  # BGR<->RGB is its own inverse
    if channels == 2 and decoded.ndim == 3 and decoded.shape[2] == 4:
        gray = decoded[..., 0]
        if not (
            np.array_equal(gray, decoded[..., 1])
            and np.array_equal(gray, decoded[..., 2])
        ):
            msg = "a gray+alpha output decoded with unequal colour planes"
            raise AssertionError(msg)
        decoded = np.stack([gray, decoded[..., 3]], axis=2)
    return decoded


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpec:
    """One way of feeding an array to the engine.

    Attributes:
        name: Label for this variant (several may share one catalogue format).
        format: The ``Pipeline.source`` format it uses.
        carries: ``(dtype, channels) -> bool`` — what the format can hold.
        lossless: Whether the engine decodes exactly the array encoded. A
            lossy source's reference input is the engine's own decode.
        heterogeneous: Whether rows may differ in shape within one column.
        declares_dtype: Whether the pipeline states the dtype up front
            (``source(..., dtype=)``); without it the plan carries ``auto``.
    """

    name: str
    format: str
    carries: Callable[[str, int], bool]
    encode_row: Callable[[np.ndarray], Any]
    column_type: Callable[[Sequence[np.ndarray]], pl.DataType]
    lossless: bool = True
    heterogeneous: bool = True
    declares_dtype: bool = True
    admits: Callable[[np.ndarray], bool] = lambda a: True
    #: Whether rows travel as an encoded image file, which is ``[H, W, C]``
    #: by construction (the array formats carry any rank).
    encodes_image: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def can_carry(self, arrays: Sequence[np.ndarray]) -> bool:
        """Whether every array in *arrays* fits this source."""
        if not arrays:
            return True
        if not all(self.admits_rank(a.ndim) for a in arrays):
            return False
        if not all(self.carries(dtype_name(a.dtype), _channels(a)) for a in arrays):
            return False
        if not all(self.admits(a) for a in arrays):
            return False
        if not self.heterogeneous and len({a.shape for a in arrays}) > 1:
            return False
        return len({a.dtype for a in arrays}) == 1

    def admits_rank(self, ndim: int) -> bool:
        """Image codecs carry ``[H, W, C]``; the array formats any rank."""
        return ndim == 3 if self.encodes_image else ndim >= 1

    def pipeline(self, dtype: str) -> Pipeline:
        """The ``Pipeline`` that reads this source, with a known *dtype*."""
        kwargs = dict(self.extra.get("source_kwargs", {}))
        if self.declares_dtype:
            kwargs["dtype"] = dtype
        return Pipeline().source(self.format, **kwargs)

    def frame_column(
        self, rows: Sequence[np.ndarray | None]
    ) -> tuple[list[Any], pl.DataType]:
        """The column values and type for *rows* (``None`` for a null row)."""
        present = [r for r in rows if r is not None]
        values = [None if r is None else self.encode_row(r) for r in rows]
        return values, self.column_type(present)


def _channels(a: np.ndarray) -> int:
    """The channel count of an image-shaped array (1 for anything else)."""
    return a.shape[2] if a.ndim == 3 else 1


def _codec_source(
    name: str, fmt: str, carries: Callable[[str, int], bool], **kw: Any
) -> SourceSpec:
    return SourceSpec(
        name=name,
        format="image_bytes",
        carries=carries,
        encode_row=lambda a: encode_codec(a, fmt),
        column_type=lambda _: pl.Binary,
        encodes_image=True,
        **kw,
    )


def _nested_list_type(arrays: Sequence[np.ndarray]) -> pl.DataType:
    leaf = POLARS_DTYPES[dtype_name(arrays[0].dtype)] if arrays else pl.UInt8
    nested: pl.DataType = leaf
    for _ in range(arrays[0].ndim if arrays else 3):
        nested = pl.List(nested)
    return nested


def _array_type(arrays: Sequence[np.ndarray]) -> pl.DataType:
    first = arrays[0]
    return pl.Array(POLARS_DTYPES[dtype_name(first.dtype)], first.shape)


class _FileStore:
    """Content-addressed temp files for the ``file_path`` source."""

    def __init__(self) -> None:
        self._dir: Path | None = None

    def path_for(self, data: bytes) -> str:
        if self._dir is None:
            self._dir = Path(tempfile.mkdtemp(prefix="polars-cv-parity-"))
            atexit.register(shutil.rmtree, self._dir, ignore_errors=True)
        path = self._dir / f"{hashlib.sha256(data).hexdigest()[:24]}.png"
        if not path.exists():
            path.write_bytes(data)
        return str(path)


_FILES = _FileStore()


def _blob_row(arr: np.ndarray) -> bytes:
    """VIEW-protocol bytes for *arr*, produced by the engine's own blob sink.

    The one encoder that has to go through the code under test: the protocol
    has no third-party writer. What the ``blob`` source is checked for is that
    it hands back what the ``blob`` sink wrote, which this exercises.
    """
    df = pl.DataFrame(
        {"x": [arr]},
        schema={"x": pl.Array(POLARS_DTYPES[dtype_name(arr.dtype)], arr.shape)},
    )
    out = df.select(pl.col("x").cv.pipe(Pipeline().source("array")).sink("blob"))
    return out["x"][0]


def _png_carries(dtype: str, channels: int) -> bool:
    return (dtype == "u8") or (dtype == "u16" and channels != 2)


SOURCES: dict[str, SourceSpec] = {
    spec.name: spec
    for spec in (
        _codec_source("png", "png", _png_carries),
        # The same bytes with no declared dtype: the plan carries "auto" and
        # every dtype-dependent decision is made at run time instead.
        _codec_source("png_auto_dtype", "png", _png_carries, declares_dtype=False),
        _codec_source(
            "tiff",
            "tiff",
            # Gray+alpha is u8 only (Pillow's "LA"); OpenCV writes the rest.
            lambda d, c: (
                d == "u8" or (d == "u16" and c != 2) or (d == "f32" and c == 1)
            ),
        ),
        # libwebp drops an alpha channel that is opaque everywhere, so such
        # an image comes back with three channels, not four.
        _codec_source(
            "webp",
            "webp",
            lambda d, c: d == "u8" and c in (3, 4),
            admits=lambda a: a.shape[2] != 4 or bool((a[..., 3] != 255).any()),
        ),
        _codec_source(
            "jpeg", "jpeg", lambda d, c: d == "u8" and c in (1, 3), lossless=False
        ),
        SourceSpec(
            name="file_path",
            encodes_image=True,
            format="file_path",
            carries=_png_carries,
            encode_row=lambda a: _FILES.path_for(encode_codec(a, "png")),
            column_type=lambda _: pl.String,
        ),
        SourceSpec(
            name="auto_bytes",
            encodes_image=True,
            format="auto",
            carries=_png_carries,
            encode_row=lambda a: encode_codec(a, "png"),
            column_type=lambda _: pl.Binary,
        ),
        SourceSpec(
            name="list",
            format="list",
            carries=lambda d, c: True,
            encode_row=lambda a: a.tolist(),
            column_type=_nested_list_type,
        ),
        SourceSpec(
            name="auto_list",
            format="auto",
            carries=lambda d, c: True,
            encode_row=lambda a: a.tolist(),
            column_type=_nested_list_type,
        ),
        SourceSpec(
            name="array",
            format="array",
            carries=lambda d, c: True,
            encode_row=lambda a: a,
            column_type=_array_type,
            heterogeneous=False,
        ),
        SourceSpec(
            name="raw",
            format="raw",
            carries=lambda d, c: True,
            encode_row=lambda a: np.ascontiguousarray(a).tobytes(),
            column_type=lambda _: pl.Binary,
            heterogeneous=False,
        ),
        SourceSpec(
            name="blob",
            format="blob",
            carries=lambda d, c: True,
            encode_row=_blob_row,
            column_type=lambda _: pl.Binary,
        ),
    )
}

#: Catalogue sources this harness does not feed images through, and why.
SOURCE_EXEMPT: dict[str, str] = {
    "contour": (
        "decodes to the contour domain, not an image; contour geometry has its "
        "own differential suite (test_contour_raster_crosscheck.py)"
    ),
}

#: Source labels whose rows need a per-source shape prefix (``raw`` is a flat
#: 1-D buffer until reshaped).
RESHAPED_SOURCES = frozenset({"raw"})


def source_pipeline(source: SourceSpec, sample: np.ndarray) -> Pipeline:
    """The pipeline prefix that reads *source* back as ``[H, W, C]``."""
    pipe = source.pipeline(dtype_name(sample.dtype))
    if source.name in RESHAPED_SOURCES:
        pipe = pipe.reshape(list(sample.shape))
    return pipe


def sources_for(arrays: Sequence[np.ndarray]) -> list[str]:
    """Names of every source that can carry all of *arrays* exactly as given."""
    return [name for name, spec in SOURCES.items() if spec.can_carry(arrays)]


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputInfo:
    """What the plan says about an output, which decides the usable sinks."""

    domain: str
    dtype: str  # the engine spelling, or "auto" when unknown at plan time
    shapes: tuple[tuple[int, ...], ...]  # every non-null row's reference shape


def _torch_installed() -> bool:
    return importlib.util.find_spec("torch") is not None


def _decode_numpy(series: pl.Series, info: OutputInfo) -> list[Any]:
    return [None if v is None else numpy_from_struct(v) for v in series]


def _decode_ndarray(series: pl.Series, info: OutputInfo) -> list[Any]:
    return list(numpy_from_column(series))


def _decode_torch(series: pl.Series, info: OutputInfo) -> list[Any]:  # pragma: no cover
    from polars_cv import torch_from_struct  # type: ignore[attr-defined]

    return [None if v is None else torch_from_struct(v).numpy() for v in series]


def _decode_list(series: pl.Series, info: OutputInfo) -> list[Any]:
    dtype = DTYPES[info.dtype]
    return [None if v is None else np.array(v.to_list(), dtype=dtype) for v in series]


def _decode_array(series: pl.Series, info: OutputInfo) -> list[Any]:
    dtype = DTYPES[info.dtype]
    return [None if v is None else np.asarray(v, dtype=dtype) for v in series]


def _decode_blob(series: pl.Series, info: OutputInfo) -> list[Any]:
    # The one sink whose bytes only the engine can read (see _blob_row).
    out = series.to_frame("b").select(
        pl.col("b").cv.pipe(Pipeline().source("blob")).sink("numpy")
    )
    return _decode_numpy(out["b"], info)


def _decode_codec(series: pl.Series, info: OutputInfo) -> list[Any]:
    rows = []
    for value, shape in zip(series, _shapes_with_nulls(series, info)):
        if value is None:
            rows.append(None)
            continue
        channels = shape[2] if len(shape) == 3 else 1
        decoded = decode_codec(value, channels)
        # An image codec cannot say [H, W] from [H, W, 1]; that difference is
        # the format's, not the engine's.
        if decoded.size == int(np.prod(shape)) and decoded.ndim != len(shape):
            decoded = decoded.reshape(shape)
        rows.append(decoded)
    return rows


def _shapes_with_nulls(series: pl.Series, info: OutputInfo) -> list[tuple[int, ...]]:
    shapes = iter(info.shapes)
    return [() if v is None else next(shapes) for v in series]


def _decode_native(series: pl.Series, info: OutputInfo) -> list[Any]:
    if info.domain == "scalar":
        return [None if v is None else np.float64(v) for v in series]
    if info.domain == "vector":
        # to_numpy keeps the element dtype (a u64 count stays u64); a detour
        # through Python ints would re-type it and hide a dtype disagreement.
        return [None if v is None else v.to_numpy() for v in series]
    return series.to_list()


@dataclass(frozen=True)
class SinkSpec:
    """One way of reading an output back.

    Attributes:
        name: The ``sink()`` format.
        domains: Output domains it applies to.
        decode: ``(series, info) -> rows`` in NumPy (or Python for contours).
        exact: Whether the round trip is lossless.
        needs_dtype: Whether the plan must know the dtype (``list``/``array``).
        codec: For image codecs, ``(dtype, channels) -> bool`` it can hold.
        available: Whether its optional dependency is installed.
    """

    name: str
    domains: frozenset[str]
    decode: Callable[[pl.Series, OutputInfo], list[Any]]
    exact: bool = True
    needs_dtype: bool = False
    homogeneous: bool = False
    codec: Callable[[str, int], bool] | None = None
    available: Callable[[], bool] = lambda: True

    def applies(self, info: OutputInfo) -> bool:
        """Whether this sink can carry an output described by *info*."""
        if info.domain not in self.domains or not self.available():
            return False
        if self.needs_dtype and info.dtype == "auto":
            return False
        if self.homogeneous and len(set(info.shapes)) > 1:
            return False
        if self.codec is not None:
            if info.dtype == "auto" or not info.shapes:
                return False
            if any(len(s) not in (2, 3) for s in info.shapes):
                return False
            channels = {s[2] if len(s) == 3 else 1 for s in info.shapes}
            return all(self.codec(info.dtype, c) for c in channels)
        return True

    def kwargs(self, info: OutputInfo) -> dict[str, Any]:
        """The ``sink()`` keyword arguments for this output."""
        if self.name == "array" and info.domain == "buffer":
            return {"shape": list(info.shapes[0])}
        return {}


BUFFER = frozenset({"buffer"})

SINKS: dict[str, SinkSpec] = {
    spec.name: spec
    for spec in (
        SinkSpec("numpy", BUFFER, _decode_numpy),
        SinkSpec("ndarray", BUFFER, _decode_ndarray),
        SinkSpec("torch", BUFFER, _decode_torch, available=_torch_installed),
        SinkSpec(
            "list", frozenset({"buffer", "vector"}), _decode_list, needs_dtype=True
        ),
        SinkSpec("array", BUFFER, _decode_array, needs_dtype=True, homogeneous=True),
        SinkSpec("blob", BUFFER, _decode_blob),
        # Image codecs hold at most four channels (gray, gray+alpha, RGB, RGBA).
        SinkSpec(
            "png",
            BUFFER,
            _decode_codec,
            codec=lambda d, c: d in ("u8", "u16") and c <= 4,
        ),
        SinkSpec(
            "tiff",
            BUFFER,
            _decode_codec,
            # A 16-bit gray+alpha TIFF has no independent decoder here (OpenCV
            # drops its alpha, Pillow cannot read it); the engine's round trip
            # is held by `gray_alpha_tiff_round_trips_as_two_channels`.
            codec=lambda d, c: (
                (d == "u8" and c <= 4)
                or (d == "u16" and c in (1, 3, 4))
                or (d == "f32" and c == 1)
            ),
        ),
        # WebP has no gray mode: a 1/2-channel image comes back as 3/4 channels.
        SinkSpec(
            "webp", BUFFER, _decode_codec, codec=lambda d, c: d == "u8" and c in (3, 4)
        ),
        SinkSpec(
            "jpeg",
            BUFFER,
            _decode_codec,
            exact=False,
            codec=lambda d, c: d == "u8" and c in (1, 3),
        ),
        SinkSpec(
            "native",
            frozenset({"scalar", "vector", "contour"}),
            _decode_native,
        ),
    )
}

#: Catalogue sinks this harness does not read, and why.
SINK_EXEMPT: dict[str, str] = {}


def sinks_for(info: OutputInfo, *, exact_only: bool = True) -> list[str]:
    """Names of every sink that can carry an output described by *info*."""
    return [
        name
        for name, spec in SINKS.items()
        if spec.applies(info) and (spec.exact or not exact_only)
    ]
