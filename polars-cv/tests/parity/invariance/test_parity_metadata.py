"""Header-only metadata and ``read_bytes`` against what the bytes really hold.

``.cv.width()`` / ``height()`` / ``channels()`` / ``image_dtype()`` /
``image_info()`` answer from an image's header without decoding it. The claim
that makes them trustworthy is that the header agrees with a decode, so for
drawn images, each encoded by a third-party codec (OpenCV or Pillow, never
polars-cv), the four answers must equal all of:

* the shape and dtype of the array that was encoded,
* the layout an independent decoder reads (``cv2.imdecode`` with
  ``IMREAD_UNCHANGED``, reconciled with Pillow's mode where OpenCV has no
  gray+alpha), and
* the shape and dtype of the same bytes decoded by a polars-cv pipeline
  (``source("image_bytes")`` into ``sink("numpy")``).

One column mixes codecs, sizes, dtypes, null rows and unrecognisable bytes
(which must be null, not an error), and the answers must not depend on the
engine, the chunk layout or the neighbouring rows. The same corpus written to
disk must answer identically from a path column, and ``read_bytes`` must hand
back exactly the bytes of the file, whatever they are.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl
import pytest
from hypothesis import event
from hypothesis import strategies as st
from PIL import Image

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required
from tests.parity.framework.budget import property_lanes
from tests.parity.framework.images import image_specs
from tests.parity.framework.io import SOURCES, encode_codec

pytestmark = plugin_required

ENGINES = ("in-memory", "streaming")

EXTENSION = {"png": ".png", "tiff": ".tif", "webp": ".webp", "jpeg": ".jpg"}

#: Garbage the header reader must call "unrecognised" (null), not fail on.
GARBAGE = (b"", b"not an image", b"\x00" * 7, b"\x89PNG\r\n\x1a\n", b"II*\x00")

PIPELINE = Pipeline().source("image_bytes")


# ---------------------------------------------------------------------------
# Drawn corpora
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One row of a bytes column: an encoded image, a null, or garbage."""

    kind: str  # "image" | "null" | "garbage"
    codec: str = ""
    array: np.ndarray | None = None
    data: bytes | None = None

    def __repr__(self) -> str:
        if self.kind == "image":
            assert self.array is not None
            return f"Row({self.codec} {self.array.dtype}{self.array.shape})"
        return f"Row({self.kind} {self.data!r})"


#: The (dtypes, channels) each codec is drawn over. Everything the codec's
#: ``SOURCES`` entry says it can carry. TIFF includes float32 and gray+alpha,
#: whose headers the reader answered null for until it read TIFF the way the
#: decoder does (``ImageAdapter::tiff_layout``).
REGIONS: dict[str, dict[str, tuple]] = {
    "png": {"dtypes": ("u8", "u16"), "channels": (1, 2, 3, 4)},
    "tiff": {"dtypes": ("u8", "u16", "f32"), "channels": (1, 2, 3, 4)},
    "webp": {"dtypes": ("u8",), "channels": (3, 4)},
    "jpeg": {"dtypes": ("u8",), "channels": (1, 3)},
}


def coded_images(
    codecs: tuple[str, ...], regions: dict[str, dict[str, tuple]] = REGIONS
) -> st.SearchStrategy[Row]:
    """An image of a drawn codec, encoded by OpenCV or Pillow. What a codec can
    carry is ``SOURCES``' own answer (``carries`` and ``admits``), so a cell
    the encoder cannot write is never drawn."""

    @st.composite
    def build(draw: st.DrawFn) -> Row:
        codec = draw(st.sampled_from(codecs), label="codec")
        source = SOURCES[codec]
        spec = draw(
            image_specs(max_side=40, **regions[codec]).filter(
                lambda s: source.can_carry([s.render()])
            ),
            label="image",
        )
        array = spec.render()
        return Row("image", codec, array, encode_codec(array, codec))

    return build()


def rows(
    codecs: tuple[str, ...] = tuple(REGIONS),
    *,
    garbage: bool = True,
    max_rows: int = 5,
    regions: dict[str, dict[str, tuple]] = REGIONS,
) -> st.SearchStrategy[list[Row]]:
    """1..max_rows rows: mostly images (mixed codecs and sizes), some null,
    some garbage."""
    other = st.one_of(
        st.just(Row("null")),
        st.sampled_from(GARBAGE).map(lambda b: Row("garbage", data=b))
        if garbage
        else st.just(Row("null")),
    )
    return st.lists(
        st.one_of(coded_images(codecs, regions), coded_images(codecs, regions), other),
        min_size=1,
        max_size=max_rows,
    )


def column(name: str, corpus: list[Row]) -> pl.Series:
    return pl.Series(
        name,
        [None if r.kind == "null" else (r.data) for r in corpus],
        dtype=pl.Binary,
    )


def chunked(corpus: list[Row], split: int) -> pl.DataFrame:
    """The column as two chunks split at *split*: a boundary mid-column."""
    cut = split % (len(corpus) + 1)
    parts = [column("i", corpus[:cut]), column("i", corpus[cut:])]
    return pl.concat([pl.DataFrame([p]) for p in parts], rechunk=False)


# ---------------------------------------------------------------------------
# What the bytes hold, from three independent places
# ---------------------------------------------------------------------------


def layout_of(array: np.ndarray) -> tuple[int, int, int, str]:
    """``(height, width, channels, dtype)`` of the array that was encoded."""
    return array.shape[0], array.shape[1], array.shape[2], array.dtype.name


def independent_layout(data: bytes) -> tuple[int, int, int, str]:
    """``(height, width, channels, dtype)`` as OpenCV decodes *data*.

    OpenCV has no gray+alpha mode: it expands a PNG's to BGRA and drops a
    TIFF's alpha, so its channel count is wrong for exactly those files.
    Pillow's mode (``LA``) is right for them and wrong for 16-bit colour,
    which it reduces to 8 bits; so each decoder answers for what it holds,
    and they must agree on the size.
    """
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded is not None, "OpenCV cannot decode what the harness encoded"
    height, width = decoded.shape[:2]
    channels = 1 if decoded.ndim == 2 else decoded.shape[2]
    with Image.open(io.BytesIO(data)) as pil:
        assert pil.size == (width, height), "Pillow and OpenCV disagree on size"
        if pil.mode == "LA":
            channels = 2
    return height, width, channels, decoded.dtype.name


def header_columns(
    frame: pl.DataFrame, engine: str, col: str = "i"
) -> dict[str, list[Any]]:
    """The four header answers and ``image_info``, as Python lists."""
    c = pl.col(col)
    out = (
        frame.lazy()
        .select(
            width=c.cv.width(),
            height=c.cv.height(),
            channels=c.cv.channels(),
            dtype=c.cv.image_dtype(),
            info=c.cv.image_info(),
        )
        .collect(engine=engine)
    )
    return {name: out[name].to_list() for name in out.columns}


def decoded_arrays(frame: pl.DataFrame, engine: str, col: str = "i") -> list[Any]:
    """Each row decoded by a polars-cv pipeline (``None`` for a null row)."""
    out = (
        frame.lazy()
        .select(pl.col(col).cv.pipe(PIPELINE).sink("numpy"))
        .collect(engine=engine)
    )
    return [None if r is None else numpy_from_struct(r) for r in out[col].to_list()]


def answer(headers: dict[str, list[Any]], i: int) -> tuple | None:
    """Row *i*'s ``(height, width, channels, dtype)`` from the header columns,
    or ``None`` when every answer is null (checking they are null together)."""
    values = (
        headers["height"][i],
        headers["width"][i],
        headers["channels"][i],
        headers["dtype"][i],
    )
    if all(v is None for v in values):
        assert headers["info"][i] is None, f"row {i}: image_info answers, others null"
        return None
    info = headers["info"][i]
    assert info is not None, f"row {i}: image_info is null, the others are not"
    assert (info["height"], info["width"], info["channels"], info["dtype"]) == values, (
        f"row {i}: image_info {info} differs from {values}"
    )
    return values


def assert_headers_match_decoders(
    corpus: list[Row], headers: dict[str, list[Any]], decoded: list[Any]
) -> None:
    for i, row in enumerate(corpus):
        got = answer(headers, i)
        if row.kind != "image":
            assert got is None, f"{row.kind} row {i} answered {got}"
            assert decoded[i] is None
            continue
        assert row.array is not None and row.data is not None
        encoded = layout_of(row.array)
        independent = independent_layout(row.data)
        assert independent == encoded, (
            f"harness: {row.codec} wrote {independent}, drew {encoded}"
        )
        assert got == encoded, (
            f"row {i} ({row.codec}): header says {got}, bytes hold {encoded}"
        )
        arr = decoded[i]
        assert arr is not None, f"row {i}: the pipeline decoded a null"
        assert (arr.shape[0], arr.shape[1], arr.shape[2], arr.dtype.name) == encoded, (
            f"row {i} ({row.codec}): the pipeline decoded {arr.shape} {arr.dtype}, "
            f"the header says {got}"
        )


# ---------------------------------------------------------------------------
# Header-only metadata from a bytes column
# ---------------------------------------------------------------------------


@property_lanes(
    weight=3,
    corpus=rows(),
    engine=st.sampled_from(ENGINES),
    split=st.integers(0, 8),
)
def test_header_answers_equal_every_decoder(corpus, engine, split) -> None:
    """For PNG (8- and 16-bit, gray to RGBA), TIFF, WebP and JPEG in one column
    with null and unrecognisable rows: width, height, channels, dtype and
    ``image_info`` equal the encoded array, an independent decoder and a
    polars-cv decode; null and garbage rows answer null on every method."""
    event(f"codecs={sorted({r.codec for r in corpus if r.kind == 'image'})}")
    event(f"nonimage={sum(r.kind != 'image' for r in corpus)}")
    frame = chunked(corpus, split)
    present = frame.with_columns(
        pl.when(pl.Series([r.kind == "image" or r.kind == "null" for r in corpus]))
        .then(pl.col("i"))
        .otherwise(None)
        .alias("i")
    )
    headers = header_columns(frame, engine)
    # The pipeline refuses bytes it cannot decode, so garbage is nulled for it.
    decoded = decoded_arrays(present, engine)
    assert_headers_match_decoders(corpus, headers, decoded)


@property_lanes(weight=2, corpus=rows(max_rows=4), engine=st.sampled_from(ENGINES))
def test_a_header_answer_does_not_depend_on_the_other_rows(corpus, engine) -> None:
    """Each row answers as it does alone, and reversing the column reverses the
    answers: no row's header is read from a neighbour's bytes."""
    frame = pl.DataFrame([column("i", corpus)])
    whole = header_columns(frame, engine)
    for i in range(len(corpus)):
        alone = header_columns(frame.slice(i, 1), engine)
        for name in whole:
            assert alone[name] == [whole[name][i]], f"row {i} {name} differs alone"
    flipped = header_columns(frame.reverse(), engine)
    for name in whole:
        assert flipped[name] == whole[name][::-1], f"{name} changed under reverse"


# ---------------------------------------------------------------------------
# Paths and read_bytes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("parity-metadata")


def write(directory: Path, data: bytes, extension: str = ".bin") -> str:
    """*data* as a file named for its content (so equal bytes share one file)."""
    path = directory / f"{hashlib.sha256(data).hexdigest()[:24]}{extension}"
    if not path.exists():
        path.write_bytes(data)
    return str(path)


def path_column(directory: Path, corpus: list[Row], prefix: str = "") -> pl.Series:
    return pl.Series(
        "p",
        [
            None
            if r.kind == "null"
            else prefix
            + write(directory, r.data or b"", EXTENSION.get(r.codec, ".bin"))
            for r in corpus
        ],
        dtype=pl.String,
    )


@property_lanes(
    weight=2,
    corpus=rows(),
    engine=st.sampled_from(ENGINES),
    prefix=st.sampled_from(["", "file://"]),
)
def test_a_path_column_answers_as_its_bytes_do(
    corpus_dir, corpus, engine, prefix
) -> None:
    """The corpus written to disk answers, from bare and ``file://`` paths,
    exactly as the same bytes in a column do (garbage files included: null)."""
    event(f"prefix={prefix!r}")
    paths = pl.DataFrame([path_column(corpus_dir, corpus, prefix)])
    by_path = header_columns(paths, engine, "p")
    by_bytes = header_columns(pl.DataFrame([column("i", corpus)]), engine)
    assert by_path == by_bytes


@property_lanes(
    weight=2,
    corpus=rows(garbage=False),
    engine=st.sampled_from(ENGINES),
    where=st.integers(0, 5),
)
def test_a_missing_file_is_one_row_and_follows_on_error(
    corpus_dir, corpus, engine, where
) -> None:
    """A path that does not exist raises under the default ``on_error`` and is
    a null row under ``"null"``, leaving every other row's answer alone."""
    good = path_column(corpus_dir, corpus).to_list()
    at = where % (len(good) + 1)
    paths = [*good[:at], str(corpus_dir / "does-not-exist.png"), *good[at:]]
    frame = pl.DataFrame({"p": paths}, schema={"p": pl.String})
    expected = header_columns(
        pl.DataFrame({"p": good}, schema={"p": pl.String}), engine, "p"
    )
    out = (
        frame.lazy()
        .select(
            width=pl.col("p").cv.width(on_error="null"),
            channels=pl.col("p").cv.channels(on_error="null"),
        )
        .collect(engine=engine)
    )
    width = list(expected["width"])
    channels = list(expected["channels"])
    width.insert(at, None)
    channels.insert(at, None)
    assert out["width"].to_list() == width
    assert out["channels"].to_list() == channels
    with pytest.raises(pl.exceptions.ComputeError):
        frame.lazy().select(pl.col("p").cv.width()).collect(engine=engine)


@st.composite
def payloads(draw: st.DrawFn) -> list[bytes | None]:
    """Files that are not images at all, or are images, or are empty."""
    return draw(
        st.lists(
            st.one_of(
                st.none(),
                st.just(b""),
                st.binary(min_size=1, max_size=3000),
                coded_images(tuple(REGIONS)).map(lambda r: r.data),
            ),
            min_size=1,
            max_size=5,
        )
    )


@property_lanes(
    weight=2,
    files=payloads(),
    engine=st.sampled_from(ENGINES),
    prefix=st.sampled_from(["", "file://"]),
    split=st.integers(0, 6),
)
def test_read_bytes_returns_the_exact_bytes(
    corpus_dir, files, engine, prefix, split
) -> None:
    """``read_bytes`` hands back every file byte for byte — arbitrary payloads,
    an empty file and real images alike — as ``Binary``, null for a null path,
    under both engines, over a column split into two chunks, with the path
    sandbox open and with ``allowed_roots`` naming the directory."""
    paths = [None if f is None else prefix + write(corpus_dir, f) for f in files]
    cut = split % (len(paths) + 1)
    frame = pl.concat(
        [
            pl.DataFrame({"p": paths[:cut]}, schema={"p": pl.String}),
            pl.DataFrame({"p": paths[cut:]}, schema={"p": pl.String}),
        ],
        rechunk=False,
    )
    for kwargs in ({}, {"allowed_roots": [str(corpus_dir)]}):
        out = (
            frame.lazy()
            .select(b=pl.col("p").cv.read_bytes(**kwargs))
            .collect(engine=engine)
        )
        assert out.schema["b"] == pl.Binary
        assert out["b"].to_list() == files, f"read_bytes{kwargs} changed a file"
    for f, p in zip(files, paths):
        if f is not None:
            assert Path(p.removeprefix(prefix)).read_bytes() == f


@property_lanes(
    weight=1,
    files=st.lists(st.binary(min_size=1, max_size=200), min_size=1, max_size=3),
    engine=st.sampled_from(ENGINES),
    where=st.integers(0, 3),
)
def test_read_bytes_sandbox_and_missing_files(corpus_dir, files, engine, where) -> None:
    """A path outside ``allowed_roots`` is refused (or a null row under
    ``on_error="null"``), as is a missing file; the other rows read."""
    good = [write(corpus_dir, f) for f in files]
    at = where % (len(good) + 1)
    paths = [*good[:at], str(corpus_dir / "missing.bin"), *good[at:]]
    frame = pl.DataFrame({"p": paths}, schema={"p": pl.String})
    expected = [*files[:at], None, *files[at:]]
    soft = frame.lazy().select(b=pl.col("p").cv.read_bytes(on_error="null"))
    assert soft.collect(engine=engine)["b"].to_list() == expected
    with pytest.raises(pl.exceptions.ComputeError):
        frame.lazy().select(pl.col("p").cv.read_bytes()).collect(engine=engine)
    outside = frame.lazy().select(
        b=pl.col("p").cv.read_bytes(allowed_roots=["/nonexistent-root"])
    )
    with pytest.raises(pl.exceptions.ComputeError):
        outside.collect(engine=engine)


@property_lanes(weight=2, corpus=rows(garbage=False), engine=st.sampled_from(ENGINES))
def test_every_route_to_a_file_decodes_the_same(corpus_dir, corpus, engine) -> None:
    """The bytes in a column, ``read_bytes`` of the path, and the
    ``file_path`` source are three routes to one decode: header answers, and
    decoded arrays, are identical."""
    frame = pl.DataFrame([column("i", corpus), path_column(corpus_dir, corpus)])
    lazy = frame.lazy()
    direct = (
        lazy.select(i=pl.col("i").cv.pipe(PIPELINE).sink("numpy"))
        .collect(engine=engine)["i"]
        .to_list()
    )
    via_read = (
        lazy.select(i=pl.col("p").cv.read_bytes().cv.pipe(PIPELINE).sink("numpy"))
        .collect(engine=engine)["i"]
        .to_list()
    )
    via_source = (
        lazy.select(i=pl.col("p").cv.pipe(Pipeline().source("file_path")).sink("numpy"))
        .collect(engine=engine)["i"]
        .to_list()
    )
    for i in range(len(corpus)):
        arrays = [
            None if r[i] is None else numpy_from_struct(r[i])
            for r in (direct, via_read, via_source)
        ]
        if corpus[i].kind == "null":
            assert arrays == [None, None, None]
            continue
        assert all(a is not None for a in arrays)
        for other in arrays[1:]:
            assert other.dtype == arrays[0].dtype
            np.testing.assert_array_equal(other, arrays[0], err_msg=f"row {i}")
    heads = (
        lazy.select(
            a=pl.col("i").cv.width(),
            b=pl.col("p").cv.width(),
            c=pl.col("p").cv.read_bytes().cv.width(),
        )
        .collect(engine=engine)
        .to_dict(as_series=False)
    )
    assert heads["a"] == heads["b"] == heads["c"]
