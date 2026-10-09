"""
Pytest configuration and fixtures for polars-cv tests.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl
import pytest

# Add the python source to the path for testing without installation
python_src = Path(__file__).parent.parent / "python"
sys.path.insert(0, str(python_src))

# Every test sees the `.cv`/`.contour`/`.point`/`.bbox` namespaces, which
# importing the package registers: no test file may depend on another having
# imported it first (`test_expression_params.py` did, and failed run alone).
# Pure Python; importing it loads no compiled code.
import polars_cv  # noqa: E402, F401

# Streaming is the project's default execution engine (see
# docs/user-guide/concepts/streaming.md), and polars' too for lazy queries since
# 2.0: the plugin only runs multi-threaded when the streaming engine slices input
# into morsels, and the two engines chunk a plugin's inputs differently, so bugs
# at chunk boundaries hide under whichever engine a bare `.collect()` happens to
# pick. The suite pins every bare `.collect()` to streaming explicitly (not
# relying on polars' default), and a second CI lane runs it under `in-memory`
# for the dual-path guarantee.
#
# `setdefault` lets a lane that exports POLARS_ENGINE_AFFINITY explicitly win
# (the in-memory lane sets it before pytest starts), and `set_engine_affinity`
# reloads polars' cached view of the variable in case polars was imported before
# this ran. Tests that pass `engine=` explicitly (the streaming-vs-eager
# equivalence checks) are unaffected — the variable only changes the *default*.
_ENGINE_AFFINITY = os.environ.setdefault("POLARS_ENGINE_AFFINITY", "streaming")
pl.Config.set_engine_affinity(_ENGINE_AFFINITY)  # type: ignore[arg-type]

if TYPE_CHECKING:
    from collections.abc import Iterator


def _plugin_available() -> bool:
    """Check if the compiled plugin is available."""
    lib_path = Path(__file__).parent.parent / "python" / "polars_cv"
    so_files = list(lib_path.glob("*.so")) + list(lib_path.glob("*.pyd"))
    return len(so_files) > 0


# Skip, rather than fail, when the compiled extension is absent. This is a
# `skipif` and not a named marker, so it cannot be selected with `-k`/`-m`;
# tests carrying it drop out on their own when the plugin is not built.
plugin_required = pytest.mark.skipif(
    not _plugin_available(),
    reason="Requires compiled plugin (run maturin develop first)",
)


@pytest.fixture
def in_memory_engine() -> Iterator[None]:
    """Pin the default engine to ``in-memory`` for the duration of one test.

    For a test whose assertion is only well-defined under the in-memory engine —
    an error that names the *absolute* row index of an offending parameter. A
    per-row plugin under the streaming engine sees one morsel at a time and
    cannot know its global offset, so it reports a morsel-local row number
    instead. The suite defaults to streaming (see the affinity set above), so a
    test asserting the absolute index must opt back into in-memory explicitly;
    this mirrors the explicit ``engine=`` legs of the streaming-vs-eager
    equivalence tests.
    """
    previous = os.environ.get("POLARS_ENGINE_AFFINITY")
    pl.Config.set_engine_affinity("in-memory")
    try:
        yield
    finally:
        pl.Config.set_engine_affinity(previous)  # type: ignore[arg-type]  # None clears it


def make_test_png(
    width: int = 10, height: int = 10, color: tuple[int, int, int] = (255, 0, 0)
) -> bytes:
    """
    Create a test PNG image (module-level; importable by test files that
    build images outside a fixture context).

    Args:
        width: Image width.
        height: Image height.
        color: RGB color tuple.

    Returns:
        PNG bytes.
    """
    try:
        from PIL import Image

        img = Image.new("RGB", (width, height), color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        pytest.skip("PIL/Pillow required for this test")
        return b""


#: PIL mode per channel count. 2 channels is grayscale+alpha, which is what the
#: ``ColorChannels`` shape produces from RGBA and which nothing
#: fed through a sink before the schema-parity matrix existed.
_MODE_FOR_CHANNELS = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}


def make_image_png(
    height: int = 8,
    width: int = 8,
    channels: int = 3,
    *,
    sixteen_bit: bool = False,
    seed: int = 0,
) -> bytes:
    """Encode a deterministic PNG with an exact channel count.

    ``create_test_png``/``make_test_png`` only make flat RGB images. The schema
    matrix needs every channel count the alpha rules distinguish (1, 2, 3, 4)
    and a 16-bit path for the ``u16`` decode, at sizes it chooses, with varying
    pixel content so operations like ``equalize_histogram`` and ``canny`` have
    something to act on.
    """
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL/Pillow required for this test")
        return b""

    buf = io.BytesIO()
    if sixteen_bit:
        rng = np.random.default_rng(seed)
        arr = rng.integers(0, 65535, size=(height, width), dtype=np.uint16)
        Image.fromarray(arr, mode="I;16").save(buf, format="PNG")
        return buf.getvalue()

    mode = _MODE_FOR_CHANNELS.get(channels)
    if mode is None:
        raise ValueError(f"unsupported channel count for a PNG: {channels}")

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(height, width, channels), dtype=np.uint8)
    if channels == 1:
        arr = arr[:, :, 0]
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def make_rect_png(height: int = 100, width: int = 200, channels: int = 3) -> bytes:
    """A black image with one white filled rectangle.

    Contour pipelines need an image that thresholds into a small, predictable
    number of regions. Noise thresholds into hundreds of one-pixel contours,
    which is slow and makes any downstream assertion depend on the RNG.
    """
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL/Pillow required for this test")
        return b""

    arr = np.zeros((height, width, channels), dtype=np.uint8)
    arr[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 255
    if channels == 1:
        arr = arr[:, :, 0]
    buf = io.BytesIO()
    Image.fromarray(arr, mode=_MODE_FOR_CHANNELS[channels]).save(buf, format="PNG")
    return buf.getvalue()


def make_ring_png(height: int = 100, width: int = 200, channels: int = 3) -> bytes:
    """A black image with one white rectangle that has a rectangular hole.

    ``make_rect_png``'s solid block cannot tell ``extract_contours(mode=)``
    apart: with no enclosed background region, "external" and "all" find the
    same single border. A ring has a second border to find, so the mode
    genuinely changes the result.
    """
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL/Pillow required for this test")
        return b""

    arr = np.zeros((height, width, channels), dtype=np.uint8)
    arr[height // 8 : 7 * height // 8, width // 8 : 7 * width // 8] = 255
    arr[3 * height // 8 : 5 * height // 8, 3 * width // 8 : 5 * width // 8] = 0
    if channels == 1:
        arr = arr[:, :, 0]
    buf = io.BytesIO()
    Image.fromarray(arr, mode=_MODE_FOR_CHANNELS[channels]).save(buf, format="PNG")
    return buf.getvalue()


@dataclass(frozen=True)
class TiffFixture:
    """A TIFF written by :func:`write_tiled_tiff` and the pixels it holds.

    Attributes:
        path: Where the file was written.
        levels: The decoded truth of every pyramid level, level 0 first, each
            ``[H, W, C]`` (a single channel keeps its axis, as polars-cv decodes
            it). For a lossless file it is exactly what was written; for JPEG
            it is tifffile's own decode of the file.
        lossless: Whether ``levels`` is the written data (not a codec's decode).
    """

    path: Path
    levels: tuple[np.ndarray, ...]
    lossless: bool


_TIFF_DTYPES = {"u8": np.uint8, "u16": np.uint16, "f32": np.float32}
_TIFF_COMPRESSION = {
    None: None,
    "lzw": "lzw",
    "deflate": "adobe_deflate",
    "jpeg": "jpeg",
}


def _tiff_content(
    height: int, width: int, channels: int, dtype: str, seed: int
) -> np.ndarray:
    """Smooth gradients plus seeded low-amplitude noise, ``[H, W, C]``.

    Gradients make every tile distinct (a misplaced tile cannot compare equal)
    and stay JPEG-friendly; the noise keeps lossless codecs honest.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    planes = [
        (
            xx / max(width - 1, 1) * (0.6 + 0.1 * c)
            + yy / max(height - 1, 1) * (0.3 - 0.05 * c)
        )
        % 1.0
        for c in range(channels)
    ]
    unit = (
        np.stack(planes, axis=-1) * 0.9 + rng.random((height, width, channels)) * 0.03
    )
    if dtype == "f32":
        return unit.astype(np.float32)
    top = np.iinfo(_TIFF_DTYPES[dtype]).max
    return np.clip(np.round(unit * top), 0, top).astype(_TIFF_DTYPES[dtype])


def write_tiled_tiff(
    path: Path,
    *,
    height: int,
    width: int,
    channels: int = 3,
    dtype: str = "u8",
    tile: tuple[int, int] | None = (32, 32),
    levels: int = 1,
    compression: str | None = None,
    jpeg_photometric: str = "ycbcr",
    bigtiff: bool = False,
    description: str | None = None,
    svs_extras: bool = False,
    resolution_cm: float | None = None,
    seed: int = 0,
) -> TiffFixture:
    """Write a deterministic tiled (or strip) TIFF, optionally a pyramid.

    The tiled, pyramidal and JPEG-tiled TIFFs a whole-slide image is made of
    are what polars-cv's own encoder (strips only) cannot write, so tests build
    them with tifffile. Levels are top-level IFDs, SVS-style: level ``k`` is
    level 0 subsampled by ``2**k`` and flagged reduced-resolution
    (SubfileType 1).

    Args:
        path: Output file.
        height: Level-0 height.
        width: Level-0 width.
        channels: 1 (gray), 3 (RGB) or 4 (RGBA).
        dtype: ``"u8"``, ``"u16"`` or ``"f32"``.
        tile: ``(tile_height, tile_width)``, each a multiple of 16 (a TIFF
            rule), or ``None`` for strips.
        levels: Number of pyramid levels.
        compression: ``None``, ``"lzw"``, ``"deflate"`` or ``"jpeg"`` (u8, 1 or
            3 channels).
        jpeg_photometric: For JPEG: ``"ycbcr"`` (libtiff style: YCbCr-coded
            JPEG, Photometric=YCbCr) or ``"rgb"`` (RGB-coded JPEG marked by an
            Adobe APP14 transform 0, Photometric=RGB).
        bigtiff: Write a BigTIFF.
        description: ImageDescription of level 0 (e.g. an Aperio header).
        svs_extras: Interleave the images an Aperio SVS carries besides its
            levels: a strip thumbnail right after level 0 (same aspect, not
            tiled, not flagged reduced), and a label and a macro image at the
            end (strips, another aspect). None of them is a pyramid level.
        resolution_cm: Pixels per centimetre, written as level 0's X/Y
            resolution (ResolutionUnit 3).
        seed: Noise seed.

    Returns:
        The file and its per-level truth.
    """
    tifffile = pytest.importorskip("tifffile")
    if compression not in _TIFF_COMPRESSION:
        raise ValueError(f"unknown compression {compression!r}")
    if channels not in (1, 3, 4):
        raise ValueError(f"unsupported channel count for a TIFF fixture: {channels}")
    if compression == "jpeg" and (dtype != "u8" or channels == 4):
        raise ValueError("jpeg fixtures hold u8 samples in 1 or 3 channels only")

    base = _tiff_content(height, width, channels, dtype, seed)
    written = [base[:: 2**k, :: 2**k] for k in range(levels)]
    photometric = "minisblack" if channels == 1 else "rgb"
    kwargs: dict[str, object] = {
        "compression": _TIFF_COMPRESSION[compression],
        "photometric": photometric,
    }
    if tile is not None:
        kwargs["tile"] = tile
    if channels == 4:
        kwargs["extrasamples"] = ("unassalpha",)
    if compression == "jpeg" and channels == 3:
        # tifffile converts RGB input to YCbCr and tags it so by default (the
        # libtiff style); `photometric="ycbcr"` would instead declare the input
        # already YCbCr. `outcolorspace="RGB"` codes the JPEG itself in RGB.
        if jpeg_photometric == "rgb":
            kwargs["compressionargs"] = {"outcolorspace": "RGB"}
        elif jpeg_photometric != "ycbcr":
            raise ValueError(f"unknown jpeg_photometric {jpeg_photometric!r}")

    def extra(data: np.ndarray, subfiletype: int) -> None:
        plain = data[..., 0] if channels == 1 else data
        tw.write(
            np.ascontiguousarray(plain),
            subfiletype=subfiletype,
            photometric=photometric,
            compression=None,
            metadata=None,
        )

    with tifffile.TiffWriter(path, bigtiff=bigtiff) as tw:
        for k, level in enumerate(written):
            data = level[..., 0] if channels == 1 else level
            extras: dict[str, object] = {}
            if k == 0 and resolution_cm is not None:
                extras = {
                    "resolution": (resolution_cm, resolution_cm),
                    "resolutionunit": "CENTIMETER",
                }
            tw.write(
                data,
                subfiletype=1 if k else 0,
                description=description if k == 0 else None,
                metadata=None,
                **kwargs,
                **extras,
            )
            if k == 0 and svs_extras:
                extra(base[::8, ::8], 0)
        if svs_extras:
            extra(base[: max(height // 4, 1), : max(height // 4, 1)], 1)
            extra(base[: max(height // 3, 1), : max(width // 6, 1)], 9)

    if compression == "jpeg":
        # Level k's page: the thumbnail, when written, follows level 0.
        pages = [k + (1 if svs_extras and k else 0) for k in range(levels)]
        with tifffile.TiffFile(path) as tif:
            truth = tuple(
                tif.pages[i].asarray().reshape(lv.shape)
                for i, lv in zip(pages, written, strict=True)
            )
        return TiffFixture(path=path, levels=truth, lossless=False)
    return TiffFixture(
        path=path,
        levels=tuple(np.ascontiguousarray(lv) for lv in written),
        lossless=True,
    )


@pytest.fixture
def tiled_tiff(tmp_path: Path) -> Callable[..., TiffFixture]:
    """Fixture form of :func:`write_tiled_tiff`, writing into ``tmp_path``.

    Returns:
        ``make(name="slide.tif", **kwargs)``; ``kwargs`` as ``write_tiled_tiff``.
    """

    def _make(name: str = "slide.tif", **kwargs: object) -> TiffFixture:
        return write_tiled_tiff(tmp_path / name, **kwargs)  # type: ignore[arg-type]

    return _make


#: OpenSlide's public Aperio sample: JPEG tiles, a pyramid, and the
#: thumbnail/label/macro IFDs real slides interleave with their levels.
SVS_SAMPLE_URL = "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs"


@pytest.fixture(scope="session")
def svs_sample(pytestconfig: pytest.Config) -> Path:
    """A real Aperio SVS slide, downloaded once into the pytest cache.

    For ``network``-marked tests only. Skips when the download fails. The file
    is checked by parsing it as an Aperio SVS (a truncated or substituted
    download does not parse as one) rather than by a pinned digest.
    """
    tifffile = pytest.importorskip("tifffile")
    import urllib.request

    cache = getattr(pytestconfig, "cache", None)  # absent under -p no:cacheprovider
    directory = cache.mkdir("svs") if cache is not None else Path(tempfile.mkdtemp())
    target = directory / "CMU-1-Small-Region.svs"
    if not target.exists():
        partial = target.with_suffix(".part")
        try:
            with urllib.request.urlopen(SVS_SAMPLE_URL, timeout=60) as resp:  # noqa: S310
                partial.write_bytes(resp.read())
        except OSError as exc:
            pytest.skip(f"cannot download the SVS sample: {exc}")
        partial.rename(target)
    with tifffile.TiffFile(target) as tif:
        if not tif.is_svs:
            target.unlink()
            pytest.fail("the downloaded SVS sample does not parse as an Aperio SVS")
    return target


@pytest.fixture
def image_png() -> Callable[..., bytes]:
    """Fixture form of :func:`make_image_png`."""
    return make_image_png


@pytest.fixture
def ring_png() -> Callable[..., bytes]:
    """Fixture form of :func:`make_ring_png`."""
    return make_ring_png


@pytest.fixture
def rect_png() -> Callable[..., bytes]:
    """Fixture form of :func:`make_rect_png`."""
    return make_rect_png


@pytest.fixture
def create_test_png() -> Callable[[int, int, tuple[int, int, int]], bytes]:
    """
    Factory fixture for creating test PNG images.

    Returns:
        A callable that creates PNG bytes for a given width, height, and color
        (default 100x100 gray, kept for existing fixture users).
    """

    def _create(
        width: int = 100,
        height: int = 100,
        color: tuple[int, int, int] = (128, 128, 128),
    ) -> bytes:
        return make_test_png(width, height, color)

    return _create


@pytest.fixture
def encode_png() -> Callable[[np.ndarray], bytes]:
    """
    Encode a numpy array as PNG bytes.

    Returns:
        A callable that encodes a numpy array as PNG bytes.
    """

    def _encode(arr: np.ndarray) -> bytes:
        """
        Encode numpy array as PNG bytes.

        Args:
            arr: NumPy array with shape (H, W, 3) or (H, W) and dtype uint8.

        Returns:
            PNG bytes.
        """
        try:
            from PIL import Image

            img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except ImportError:
            pytest.skip("PIL/Pillow required for this test")
            return b""

    return _encode


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Create minimal valid PNG bytes for testing."""
    # Minimal 1x1 red PNG
    # This is a valid PNG that can be decoded by image libraries
    return bytes(
        [
            0x89,
            0x50,
            0x4E,
            0x47,
            0x0D,
            0x0A,
            0x1A,
            0x0A,  # PNG signature
            0x00,
            0x00,
            0x00,
            0x0D,
            0x49,
            0x48,
            0x44,
            0x52,  # IHDR chunk
            0x00,
            0x00,
            0x00,
            0x01,
            0x00,
            0x00,
            0x00,
            0x01,  # 1x1
            0x08,
            0x02,
            0x00,
            0x00,
            0x00,  # 8-bit RGB
            0x90,
            0x77,
            0x53,
            0xDE,  # CRC
            0x00,
            0x00,
            0x00,
            0x0C,
            0x49,
            0x44,
            0x41,
            0x54,  # IDAT chunk
            0x08,
            0xD7,
            0x63,
            0xF8,
            0xCF,
            0xC0,
            0x00,
            0x00,  # Compressed data
            0x00,
            0x03,
            0x00,
            0x01,  # Compressed data cont.
            0x00,
            0x18,
            0xDD,
            0x8D,
            0xB4,  # CRC
            0x00,
            0x00,
            0x00,
            0x00,
            0x49,
            0x45,
            0x4E,
            0x44,  # IEND chunk
            0xAE,
            0x42,
            0x60,
            0x82,  # CRC
        ]
    )
