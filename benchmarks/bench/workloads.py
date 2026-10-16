"""The four workloads, implemented once per library.

``prepare(library, workload, source, manifest, threads)`` does the untimed
setup: it loads the inputs into the library's own container (a Polars frame,
a Python list, a ``datasets.Dataset``, a Daft frame) and returns a zero-argument
callable. That callable is the timed job. It runs from those inputs to a
fully materialised result, including any worker pool it starts. ``source`` is
``"bytes"`` (encoded images already in memory) or ``"path"`` (each job reads
the files).

Workloads (fixed parameters; see benchmarks/README.md):

- ``w1``: decode, bilinear resize to 224x224, grayscale, one uint8 array.
- ``w2``: decode, thumbnail (long side 256, never upscaled, bilinear), encode
  JPEG at quality 85.
- ``w3``: width, height, channels and a 64-bit perceptual hash per image, then
  group images with the same hash (near-duplicates).
- ``w4``: crop a per-row box, then resize to a per-row size (bilinear), the
  parameters read from the manifest's columns.

Each workload uses the library's idiomatic way to do it. Where a library has no
way to do the operation, it raises ``NotApplicable`` and gives the reason; a
different operation is never put in its place.
"""

from __future__ import annotations

import io
import multiprocessing as mp
import os
import tempfile
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

WORKLOADS = ("w1", "w2", "w3", "w4")
LIBRARIES = (
    "polars-cv-eager",
    "polars-cv-streaming",
    "polars-cv-streaming-nopool",
    "pil",
    "pil-mp",
    "opencv",
    "opencv-mp",
    "torchvision",
    "daft",
    "hf-datasets",
)
#: A plain loop is one process by construction.
SINGLE_THREAD_ONLY = frozenset({"pil", "opencv"})
#: A process pool of one is the loop plus overhead.
MULTI_THREAD_ONLY = frozenset({"pil-mp", "opencv-mp"})

#: Each polars-cv library's engine, and whether the plugin splits a call's
#: rows over its own pool. ``nopool`` turns that off
#: (``polars_cv.set_row_parallelism(False)``), so the streaming engine's
#: morsels are the only parallelism.
POLARS_CV_MODES: dict[str, tuple[str, bool]] = {
    "polars-cv-eager": ("in-memory", True),
    "polars-cv-streaming": ("streaming", True),
    "polars-cv-streaming-nopool": ("streaming", False),
}


#: The installed polars-cv's (major, minor): what it can do decides which
#: variants run and which tolerances hold.
PCV_VERSION = tuple(int(x) for x in metadata.version("polars-cv").split(".")[:2])
#: ``set_row_parallelism`` arrived in polars-cv 0.38.0.
PCV_HAS_ROW_PARALLELISM_SWITCH = PCV_VERSION >= (0, 38)

SIZE = 224
THUMB = 256
QUALITY = 85

NOT_APPLICABLE: dict[tuple[str, str], str] = {
    ("torchvision", "w3"): (
        "torchvision has no perceptual hash; a torchvision user would call "
        "imagehash on PIL images, which is the PIL benchmark"
    ),
    ("daft", "w2"): (
        "daft resize() takes literal width and height, so it cannot keep each "
        "image's aspect ratio, and encode_image() has no JPEG quality setting"
    ),
    ("daft", "w4"): (
        "daft resize() takes literal width and height; a per-row target size "
        "needs a Python UDF around PIL, which is the PIL benchmark"
    ),
}


if not PCV_HAS_ROW_PARALLELISM_SWITCH:
    for _w in WORKLOADS:
        NOT_APPLICABLE[("polars-cv-streaming-nopool", _w)] = (
            "polars-cv < 0.38 has no set_row_parallelism, so its row "
            "parallelism cannot be turned off"
        )


class NotApplicable(Exception):
    """This library has no way to run this workload as specified."""


Job = Callable[[], Any]
PARAMS = ("crop_top", "crop_left", "crop_h", "crop_w", "target_h", "target_w")


def load_items(manifest: pl.DataFrame, source: str) -> list:
    paths = manifest["path"].to_list()
    if source == "path":
        return paths
    if source == "bytes":
        out = []
        for p in paths:
            with open(p, "rb") as f:
                out.append(f.read())
        return out
    raise ValueError(f"unknown source {source!r}")


def thumb_size(w: int, h: int, limit: int = THUMB) -> tuple[int, int]:
    """PIL's ``Image.thumbnail`` target size: fit in ``limit``, never upscale.

    The same rounding as Pillow's ``preserve_aspect_ratio``: of floor and
    ceil, the one whose aspect ratio is closer to the original's.
    """
    x, y = min(w, limit), min(h, limit)
    aspect = w / h
    if x / y >= aspect:
        lo, hi = int(y * aspect), int(y * aspect) + 1
        x = max(1, min((lo, hi), key=lambda n: abs(aspect - n / y)))
    else:
        lo, hi = int(x / aspect), int(x / aspect) + 1
        y = max(1, min((lo, hi), key=lambda n: abs(aspect - x / n) if n else 1e9))
    return x, y


# --- polars-cv ----------------------------------------------------------------


def _polars_cv(workload: str, source: str, m: pl.DataFrame, library: str) -> Job:
    from polars_cv import Pipeline, numpy_from_column

    engine, row_parallelism = POLARS_CV_MODES[library]
    if not row_parallelism:
        # Process-wide, and each cell is its own process.
        import polars_cv

        polars_cv.set_row_parallelism(False)

    items = load_items(m, source)
    df = pl.DataFrame({"id": m["id"], "src": items}).hstack(m.select(PARAMS))
    fmt = "image_bytes" if source == "bytes" else "file_path"
    src = pl.col("src")

    def run(build: Callable[[pl.LazyFrame], pl.LazyFrame]) -> pl.DataFrame:
        return build(df.lazy()).collect(engine=engine)

    if workload == "w1":
        pipe = (
            Pipeline()
            .source(fmt, dtype="u8")
            .resize(height=SIZE, width=SIZE, filter="bilinear")
            .grayscale()
        )
        expr = src.cv.pipe(pipe).sink("array", shape=[SIZE, SIZE, 1])
        return lambda: run(lambda lf: lf.select(x=expr))["x"].to_numpy()

    if workload == "w2":
        long_side = pl.max_horizontal(src.cv.width(), src.cv.height())
        pipe = (
            Pipeline()
            # No .thumbnail(): PIL's thumbnail() calls draft() with
            # reducing_gap=2, which at COCO's sizes (<= 640 px) keeps the full
            # decode, so a scaled decode here would do less work than PIL.
            .source(fmt)
            .resize_max(pl.min_horizontal(THUMB, long_side), filter="bilinear")
        )
        expr = src.cv.pipe(pipe).sink("jpeg", quality=QUALITY)
        return lambda: run(lambda lf: lf.select(thumb=expr))["thumb"]

    if workload == "w3":
        phash = src.cv.pipe(Pipeline().source(fmt).perceptual_hash()).sink("list")

        def job() -> tuple[pl.DataFrame, pl.DataFrame]:
            meta = df.lazy().select(
                "id",
                width=src.cv.width(),
                height=src.cv.height(),
                channels=src.cv.channels(),
                hash=phash,
            )
            dupes = (
                meta.group_by("hash")
                .agg(ids=pl.col("id").sort(), n=pl.len())
                .filter(pl.col("n") > 1)
            )
            return tuple(pl.collect_all([meta, dupes], engine=engine))

        return job

    if workload == "w4":
        pipe = (
            Pipeline()
            .source(fmt)
            .crop(
                top=pl.col("crop_top"),
                left=pl.col("crop_left"),
                height=pl.col("crop_h"),
                width=pl.col("crop_w"),
            )
            .resize(
                height=pl.col("target_h"), width=pl.col("target_w"), filter="bilinear"
            )
        )
        expr = src.cv.pipe(pipe).sink("numpy")
        return lambda: numpy_from_column(run(lambda lf: lf.select(x=expr))["x"])

    raise ValueError(workload)


# --- PIL ------------------------------------------------------------------------

_STATE: dict[str, Any] = {}


def _pil_open(item):
    from PIL import Image

    return Image.open(io.BytesIO(item) if isinstance(item, bytes) else item)


def pil_w1(item) -> np.ndarray:
    from PIL import Image

    with _pil_open(item) as im:
        return np.asarray(
            im.resize((SIZE, SIZE), Image.Resampling.BILINEAR).convert("L")
        )


def pil_w2(item) -> bytes:
    from PIL import Image

    with _pil_open(item) as im:
        im.thumbnail((THUMB, THUMB), Image.Resampling.BILINEAR)  # calls draft()
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=QUALITY)
        return buf.getvalue()


def pil_w3(item) -> tuple[int, int, int, int]:
    import imagehash

    with _pil_open(item) as im:
        h = imagehash.phash(im, hash_size=8)
        return im.width, im.height, len(im.getbands()), int(str(h), 16)


def pil_w4(item, p) -> np.ndarray:
    from PIL import Image

    top, left, ch, cw, th, tw = p
    with _pil_open(item) as im:
        crop = im.crop((left, top, left + cw, top + ch))
        return np.asarray(crop.resize((tw, th), Image.Resampling.BILINEAR))


def _w3_frames(m: pl.DataFrame, rows: list) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-image rows into the same frames polars-cv produces, then group."""
    w, h, c, hsh = zip(*rows, strict=True)
    meta = pl.DataFrame(
        {"id": m["id"], "width": w, "height": h, "channels": c, "hash": hsh},
        schema_overrides={"hash": pl.UInt64},
    )
    dupes = (
        meta.group_by("hash")
        .agg(ids=pl.col("id").sort(), n=pl.len())
        .filter(pl.col("n") > 1)
    )
    return meta, dupes


def _state_call(i: int):
    fn, items, params = _STATE["fn"], _STATE["items"], _STATE["params"]
    return fn(items[i], params[i]) if params is not None else fn(items[i])


def _loop_or_pool(fn, items, params, processes: int | None) -> list:
    """Map ``fn`` over the items in this process, or in a forked pool.

    The pool inherits the inputs through ``fork`` and is sent indices, so only
    results are pickled, which is what an expert would write.
    """
    if processes is None:
        if params is None:
            return [fn(x) for x in items]
        return [fn(x, p) for x, p in zip(items, params, strict=True)]
    _STATE.update(fn=fn, items=items, params=params)
    n = len(items)
    chunk = max(1, n // (processes * 8))
    with mp.get_context("fork").Pool(processes, initializer=_pin_worker) as pool:
        return pool.map(_state_call, range(n), chunksize=chunk)


def _pin_worker() -> None:
    try:
        import cv2

        cv2.setNumThreads(1)
    except ImportError:
        pass


def _params(m: pl.DataFrame) -> list[tuple[int, ...]]:
    return list(m.select(PARAMS).iter_rows())


def _per_image(library: str, fns: dict, workload: str, source: str, m, threads) -> Job:
    items = load_items(m, source)
    processes = threads if library.endswith("-mp") else None
    params = _params(m) if workload == "w4" else None
    fn = fns[workload]
    if workload == "w1":
        return lambda: np.stack(_loop_or_pool(fn, items, params, processes))
    if workload == "w3":
        return lambda: _w3_frames(m, _loop_or_pool(fn, items, params, processes))
    return lambda: _loop_or_pool(fn, items, params, processes)


# --- OpenCV -----------------------------------------------------------------------


def _cv_read(item, flags: int):
    import cv2

    if isinstance(item, bytes):
        return cv2.imdecode(np.frombuffer(item, np.uint8), flags)
    return cv2.imread(item, flags)


def _cv_color() -> int:
    import cv2

    # PIL and polars-cv do not apply EXIF orientation; neither does this.
    return cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION


def cv_w1(item) -> np.ndarray:
    import cv2

    img = _cv_read(item, _cv_color())
    small = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def cv_w2(item) -> bytes:
    import cv2

    img = _cv_read(item, _cv_color())
    h, w = img.shape[:2]
    tw, th = thumb_size(w, h)
    if (tw, th) != (w, h):
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    assert ok
    return buf.tobytes()


def cv_w3(item) -> tuple[int, int, int, int]:
    import cv2

    img = _cv_read(item, cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
    h, w = img.shape[:2]
    c = 1 if img.ndim == 2 else img.shape[2]
    digest = cv2.img_hash.pHash(img)  # (1, 8) uint8, opencv-contrib
    return w, h, c, int.from_bytes(digest.tobytes(), "big")


def cv_w4(item, p) -> np.ndarray:
    import cv2

    top, left, ch, cw, th, tw = p
    img = _cv_read(item, _cv_color())
    crop = img[top : top + ch, left : left + cw]
    out = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


# --- torchvision ------------------------------------------------------------------


def _torchvision(workload: str, source: str, m: pl.DataFrame, threads: int) -> Job:
    import torch
    import torchvision.io as tio
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import v2

    torch.set_num_threads(1 if threads == 1 else threads)
    items = load_items(m, source)
    params = _params(m)

    def read(item, mode):
        data = (
            torch.frombuffer(bytearray(item), dtype=torch.uint8)
            if isinstance(item, bytes)
            else tio.read_file(item)
        )
        return tio.decode_image(data, mode=mode)

    w1 = v2.Compose([v2.Resize((SIZE, SIZE), antialias=True), v2.Grayscale()])

    class Images(Dataset):
        def __len__(self) -> int:
            return len(items)

        def __getitem__(self, i: int):
            if workload == "w1":
                return w1(read(items[i], tio.ImageReadMode.RGB))
            if workload == "w2":
                img = read(items[i], tio.ImageReadMode.UNCHANGED)
                h, w = img.shape[-2:]
                tw, th = thumb_size(w, h)
                if (tw, th) != (w, h):
                    img = v2.functional.resize(img, [th, tw], antialias=True)
                return tio.encode_jpeg(img, quality=QUALITY).numpy().tobytes()
            top, left, ch, cw, th, tw = params[i]
            img = read(items[i], tio.ImageReadMode.RGB)
            img = v2.functional.crop(img, top, left, ch, cw)
            img = v2.functional.resize(img, [th, tw], antialias=True)
            return img.permute(1, 2, 0).numpy()

    workers = 0 if threads == 1 else threads
    if workload == "w1":

        def job():
            loader = DataLoader(Images(), batch_size=64, num_workers=workers)
            return torch.cat(list(loader)).numpy()

        return job

    def job():
        loader = DataLoader(
            Images(), batch_size=64, num_workers=workers, collate_fn=list
        )
        return [x for batch in loader for x in batch]

    return job


# --- Daft -------------------------------------------------------------------------

_DAFT_THREADS: int | None = None


def _daft(workload: str, source: str, m: pl.DataFrame, threads: int) -> Job:
    global _DAFT_THREADS
    import daft
    from daft import col
    from daft import functions as F

    if _DAFT_THREADS is None:
        daft.set_runner_native(num_threads=threads)
        _DAFT_THREADS = threads
    items = load_items(m, source)
    df = daft.from_pydict({"id": m["id"].to_list(), "src": items})
    raw = col("src") if source == "bytes" else F.download(col("src"))

    if workload == "w1":
        img = F.convert_image(F.resize(F.decode_image(raw), SIZE, SIZE), "L")
        q = df.select(F.image_to_tensor(img).alias("x"))

        def job():
            arr = q.to_arrow()["x"].combine_chunks()
            return np.stack(arr.to_numpy(zero_copy_only=False))

        return job

    if workload == "w3":
        img = F.decode_image(raw, mode=None)

        def job():
            meta = df.select(
                col("id"),
                F.image_width(img).alias("width"),
                F.image_height(img).alias("height"),
                F.image_attribute(img, "channel").alias("channels"),
                F.image_hash(img, method="phash", hash_size=8).alias("hash"),
            ).collect()
            dupes = (
                meta.groupby("hash")
                .agg(col("id").list_agg().alias("ids"), col("id").count().alias("n"))
                .where(col("n") > 1)
                .collect()
            )
            return meta, dupes

        return job

    raise ValueError(workload)


# --- Hugging Face datasets ------------------------------------------------------


def _hf_map_w1(batch):
    from PIL import Image

    return {
        "x": [
            np.asarray(im.resize((SIZE, SIZE), Image.Resampling.BILINEAR).convert("L"))
            for im in batch["image"]
        ]
    }


def _hf_map_w2(batch):
    out = []
    for im in batch["image"]:
        from PIL import Image

        im.thumbnail((THUMB, THUMB), Image.Resampling.BILINEAR)
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=QUALITY)
        out.append(buf.getvalue())
    return {"thumb": out}


def _hf_map_w3(batch):
    import imagehash

    cols: dict[str, list] = {"width": [], "height": [], "channels": [], "hash": []}
    for im in batch["image"]:
        cols["width"].append(im.width)
        cols["height"].append(im.height)
        cols["channels"].append(len(im.getbands()))
        # imagehash's own hex form: a uint64 overflows datasets' int path.
        cols["hash"].append(str(imagehash.phash(im, hash_size=8)))
    return cols


def _hf_map_w4(batch):
    from PIL import Image

    # Raw bytes plus a shape: datasets' numpy formatter walks variable-shape
    # nested lists element by element in Python (~0.3 s per image here), so
    # variable-size arrays are stored flat and rebuilt with np.frombuffer.
    data, shapes = [], []
    for im, *p in zip(batch["image"], *(batch[k] for k in PARAMS), strict=True):
        top, left, ch, cw, th, tw = p
        crop = im.crop((left, top, left + cw, top + ch))
        a = np.asarray(crop.resize((tw, th), Image.Resampling.BILINEAR))
        data.append(a.tobytes())
        shapes.append(list(a.shape))
    return {"x": data, "shape": shapes}


_KEEP_ALIVE: list = []
DATA_TMP = Path(__file__).resolve().parents[1] / "data" / "tmp"


def _hf(workload: str, source: str, m: pl.DataFrame, threads: int) -> Job:
    import datasets
    from datasets import (
        Array2D,
        Dataset,
        Features,
        Image,
        Sequence,
        Value,
        load_from_disk,
    )

    datasets.disable_progress_bars()
    datasets.disable_caching()
    DATA_TMP.mkdir(parents=True, exist_ok=True)
    items = load_items(m, source)
    images = [
        {"bytes": x, "path": None} if source == "bytes" else {"bytes": None, "path": x}
        for x in items
    ]
    cols = {
        "image": images,
        "id": m["id"].to_list(),
        **{k: m[k].to_list() for k in PARAMS},
    }
    feats = {
        "image": Image(),
        "id": Value("uint32"),
        **{k: Value("int64") for k in PARAMS},
    }
    # A dataset as it is normally used: Arrow on disk, memory-mapped. An
    # in-memory table from from_dict() is pickled to every num_proc worker
    # (measured: 135 img/s at num_proc=4 against 514 memory-mapped on w1).
    tmp = tempfile.TemporaryDirectory(dir=DATA_TMP)
    Dataset.from_dict(cols, features=Features(feats)).save_to_disk(tmp.name)
    ds = load_from_disk(tmp.name)
    _KEEP_ALIVE.append(tmp)
    num_proc = None if threads == 1 else threads
    keep = ["id"]
    fn, out_feats = {
        "w1": (_hf_map_w1, {"x": Array2D((SIZE, SIZE), "uint8")}),
        "w2": (_hf_map_w2, {"thumb": Value("binary")}),
        "w3": (
            _hf_map_w3,
            {
                "width": Value("int64"),
                "height": Value("int64"),
                "channels": Value("int64"),
                "hash": Value("string"),
            },
        ),
        "w4": (_hf_map_w4, {"x": Value("binary"), "shape": Sequence(Value("int64"))}),
    }[workload]
    features = Features({"id": Value("uint32"), **out_feats})

    def mapped():
        return ds.map(
            fn,
            batched=True,
            batch_size=64,
            num_proc=num_proc,
            remove_columns=[c for c in ds.column_names if c not in keep],
            features=features,
            keep_in_memory=True,
        )

    if workload == "w1":
        return lambda: mapped().with_format("numpy")["x"]
    if workload == "w2":
        return lambda: mapped()["thumb"]
    if workload == "w3":

        def job():
            meta = mapped().to_polars()
            dupes = (
                meta.group_by("hash")
                .agg(ids=pl.col("id").sort(), n=pl.len())
                .filter(pl.col("n") > 1)
            )
            return meta, dupes

        return job

    def job():
        out = mapped()
        return [
            np.frombuffer(x, np.uint8).reshape(shape)
            for x, shape in zip(out["x"], out["shape"], strict=True)
        ]

    return job


# --- Dispatch ---------------------------------------------------------------------

PIL_FNS = {"w1": pil_w1, "w2": pil_w2, "w3": pil_w3, "w4": pil_w4}
CV_FNS = {"w1": cv_w1, "w2": cv_w2, "w3": cv_w3, "w4": cv_w4}


def prepare(
    library: str, workload: str, source: str, manifest: pl.DataFrame, threads: int
) -> Job:
    if (library, workload) in NOT_APPLICABLE:
        raise NotApplicable(NOT_APPLICABLE[(library, workload)])
    if library in SINGLE_THREAD_ONLY and threads != 1:
        raise NotApplicable(f"{library} is a single-process loop; see {library}-mp")
    if library in MULTI_THREAD_ONLY and threads == 1:
        raise NotApplicable(f"{library} with one process is the {library[:-3]} loop")
    if library.startswith("polars-cv"):
        return _polars_cv(workload, source, manifest, library)
    if library in ("pil", "pil-mp"):
        return _per_image(library, PIL_FNS, workload, source, manifest, threads)
    if library in ("opencv", "opencv-mp"):
        import cv2

        cv2.setNumThreads(1)
        return _per_image(library, CV_FNS, workload, source, manifest, threads)
    if library == "torchvision":
        return _torchvision(workload, source, manifest, threads)
    if library == "daft":
        return _daft(workload, source, manifest, threads)
    if library == "hf-datasets":
        return _hf(workload, source, manifest, threads)
    raise ValueError(f"unknown library {library!r}")


# --- Normalised outputs, for the parity tests ----------------------------------


def _as_hwc(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return a[..., None] if a.ndim == 2 else a


def _hash_int(v) -> int:
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        return int(v, 16)
    if isinstance(v, (bytes, bytearray)):
        return int.from_bytes(v, "big")
    return int.from_bytes(bytes(list(v)), "big")


def to_common(workload: str, out) -> Any:
    """One representation per workload, whatever library produced it.

    w1: ``(N, 224, 224)`` uint8. w2: a list of JPEG bytes. w3: a frame of
    ``id, width, height, channels, hash`` (hash as an int) sorted by id, and
    the set of duplicate id groups. w4: a list of ``(H, W, C)`` uint8 arrays.
    """
    if workload == "w1":
        a = np.asarray(out)
        return a.reshape(a.shape[0], SIZE, SIZE)
    if workload == "w2":
        return [bytes(x) for x in (out.to_list() if hasattr(out, "to_list") else out)]
    if workload == "w3":
        meta, dupes = out
        if not isinstance(meta, pl.DataFrame):  # daft
            meta, dupes = (
                pl.from_arrow(meta.to_arrow()),
                pl.from_arrow(dupes.to_arrow()),
            )
        rows = (
            meta.select("id", "width", "height", "channels", "hash").sort("id").rows()
        )
        frame = pl.DataFrame(
            [(int(i), int(w), int(h), int(c), _hash_int(x)) for i, w, h, c, x in rows],
            schema={
                "id": pl.Int64,
                "width": pl.Int64,
                "height": pl.Int64,
                "channels": pl.Int64,
                "hash": pl.UInt64,
            },
            orient="row",
        )
        groups = {tuple(sorted(int(i) for i in ids)) for ids in dupes["ids"].to_list()}
        return frame, groups
    if workload == "w4":
        return [_as_hwc(a) for a in out]
    raise ValueError(workload)


def cpu_count() -> int:
    return (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    )
