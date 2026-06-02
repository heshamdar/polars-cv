"""
Notebook display utilities for image columns.

Provides ``show_images()`` for rendering images from Polars Binary columns
directly in Jupyter notebooks.  Outside of a notebook environment the
function prints a text summary instead.
"""

from __future__ import annotations

import base64
import io
import struct
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    pass

# VIEW protocol constants (must stay in sync with view-buffer/src/protocol.rs)
_VIEW_MAGIC = b"VIEW"
_VIEW_HEADER_SIZE = 64

# PNG/JPEG/WebP magic-byte signatures
_IMAGE_MAGIC: dict[str, bytes] = {
    "image/png": b"\x89PNG",
    "image/jpeg": b"\xff\xd8\xff",
    "image/webp": b"RIFF",
    "image/gif": b"GIF8",
    "image/bmp": b"BM",
    "image/tiff": b"II",
    "image/tiff_be": b"MM",
}


def _detect_mime(data: bytes) -> str | None:
    """Detect MIME type from the first few bytes of image data.

    Args:
        data: Raw bytes of the image.

    Returns:
        MIME type string, or ``None`` if the format is unrecognised.
    """
    if data[:4] == _VIEW_MAGIC:
        return "view"
    for mime, magic in _IMAGE_MAGIC.items():
        if data[: len(magic)] == magic:
            if mime.startswith("image/tiff"):
                return "image/tiff"
            return mime
    return None


def _view_to_png(data: bytes) -> bytes:
    """Convert a VIEW protocol blob to PNG bytes.

    Reconstructs the pixel array from the VIEW header, then encodes it as
    PNG via PIL.

    Args:
        data: VIEW protocol blob (header + shape + strides + pixel data).

    Returns:
        PNG-encoded bytes.
    """
    import struct

    import numpy as np

    dtype_code = data[6]
    rank = data[7]
    dtype_map = {
        1: np.uint8,
        2: np.int8,
        3: np.uint16,
        4: np.int16,
        5: np.uint32,
        6: np.int32,
        7: np.float32,
        8: np.float64,
        9: np.uint64,
        10: np.int64,
    }
    np_dtype = dtype_map.get(dtype_code, np.uint8)

    shape_start = _VIEW_HEADER_SIZE
    shape = []
    for i in range(rank):
        off = shape_start + i * 8
        dim = struct.unpack_from("<Q", data, off)[0]
        shape.append(dim)

    data_offset = struct.unpack_from("<Q", data, 8)[0]
    pixel_data = data[data_offset:]
    arr = np.frombuffer(pixel_data, dtype=np_dtype)
    arr = arr.reshape(shape)

    return _ndarray_to_png(arr)


def _ndarray_to_png(arr: "Any") -> bytes:
    """Normalise a NumPy array to uint8 and encode as PNG.

    Args:
        arr: NumPy array with 2-D or 3-D shape.

    Returns:
        PNG-encoded bytes.
    """
    import numpy as np
    from PIL import Image

    if arr.dtype != np.uint8:
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo > 0:
            arr = ((arr.astype(np.float64) - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _numpy_struct_to_png(row: dict[str, Any]) -> bytes | None:
    """Convert a numpy-sink struct to PNG bytes.

    Args:
        row: Dict with ``data``, ``dtype``, ``shape``, ``strides``, ``offset``
            fields as returned by the numpy/torch sink.

    Returns:
        PNG-encoded bytes, or ``None`` if fields are null.
    """
    import numpy as np

    raw = row.get("data")
    dtype_str = row.get("dtype")
    shape_list = row.get("shape")
    offset = row.get("offset", 0) or 0

    if raw is None or dtype_str is None or shape_list is None:
        return None

    if isinstance(shape_list, pl.Series):
        shape = tuple(int(x) for x in shape_list.to_list())
    else:
        shape = tuple(int(x) for x in shape_list)

    dt = np.dtype(dtype_str)
    arr = np.frombuffer(bytes(raw), dtype=dt, offset=int(offset)).reshape(shape)
    return _ndarray_to_png(arr)


def _is_notebook() -> bool:
    """Return ``True`` when running inside a Jupyter/IPython notebook."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        return "ZMQInteractiveShell" in type(shell).__name__
    except ImportError:
        return False


def show_images(
    df: pl.DataFrame,
    column: str,
    *,
    max_rows: int = 10,
    max_width: int = 200,
    format: str = "auto",
) -> None:
    """Display images from a binary column in a Jupyter notebook.

    Supports encoded images (PNG, JPEG, WebP, TIFF, BMP, GIF), VIEW
    protocol blobs, and numpy-sink struct columns.  Null rows are shown
    as a grey placeholder.

    Outside of a notebook, prints a text summary of each row (format,
    width, height).

    Args:
        df: Polars DataFrame containing the image column.
        column: Name of the column with image data.
        max_rows: Maximum number of rows to display (default 10).
        max_width: Maximum display width in pixels for each thumbnail
            (default 200).
        format: How to interpret the column.

            - ``"auto"`` (default): detect from magic bytes.
            - ``"numpy"``: column is a numpy-sink Struct.

    Example:
        ```python
        >>> import polars as pl
        >>> from polars_cv import Pipeline, show_images
        >>>
        >>> df = pl.DataFrame({"img": [png_bytes_1, png_bytes_2]})
        >>> show_images(df, "img")
        ```
    """
    if column not in df.columns:
        msg = f"Column '{column}' not found in DataFrame. Available: {df.columns}"
        raise KeyError(msg)

    n = min(max_rows, df.height)
    col = df[column]

    if _is_notebook():
        _show_notebook(col, n, max_width, format)
    else:
        _show_text(col, n, format)


def _show_notebook(
    col: pl.Series,
    n: int,
    max_width: int,
    fmt: str,
) -> None:
    """Render images in a Jupyter notebook using HTML."""
    from IPython.display import HTML, display

    html_parts: list[str] = [
        '<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:flex-start;">'
    ]

    for i in range(n):
        val = col[i]

        if val is None:
            html_parts.append(
                f'<div style="width:{max_width}px;height:{max_width}px;'
                "background:#eee;display:flex;align-items:center;"
                "justify-content:center;font-size:12px;color:#999;"
                f'border:1px solid #ddd;">null (row {i})</div>'
            )
            continue

        png_bytes = _to_png_bytes(val, fmt)
        if png_bytes is None:
            html_parts.append(
                f'<div style="width:{max_width}px;height:60px;'
                "background:#fdd;display:flex;align-items:center;"
                "justify-content:center;font-size:11px;color:#c00;"
                f'border:1px solid #dcc;">unsupported (row {i})</div>'
            )
            continue

        b64 = base64.b64encode(png_bytes).decode("ascii")
        html_parts.append(
            f'<img src="data:image/png;base64,{b64}" '
            f'style="max-width:{max_width}px;max-height:{max_width}px;'
            f'object-fit:contain;border:1px solid #ddd;" '
            f'title="row {i}" />'
        )

    html_parts.append("</div>")
    display(HTML("".join(html_parts)))


def _show_text(col: pl.Series, n: int, fmt: str) -> None:
    """Print a text summary outside of notebook environments."""
    for i in range(n):
        val = col[i]
        if val is None:
            print(f"[{i}] null")
            continue

        if fmt == "numpy" and isinstance(val, dict):
            shape = val.get("shape")
            dtype_str = val.get("dtype", "?")
            if shape is not None:
                if isinstance(shape, pl.Series):
                    shape = shape.to_list()
                print(f"[{i}] numpy struct: shape={list(shape)}, dtype={dtype_str}")
            else:
                print(f"[{i}] numpy struct: null fields")
            continue

        if isinstance(val, bytes):
            mime = _detect_mime(val)
            if mime == "view":
                print(f"[{i}] VIEW blob ({len(val)} bytes)")
            elif mime:
                print(f"[{i}] {mime} ({len(val)} bytes)")
            else:
                print(f"[{i}] unknown format ({len(val)} bytes)")
        else:
            print(f"[{i}] {type(val).__name__}")


def _to_png_bytes(val: Any, fmt: str) -> bytes | None:
    """Convert a cell value to PNG bytes, or None if not possible.

    Args:
        val: Cell value (bytes, dict, etc.).
        fmt: Requested format hint (``"auto"`` or ``"numpy"``).

    Returns:
        PNG-encoded bytes, or ``None`` if conversion failed.
    """
    if fmt == "numpy" and isinstance(val, dict):
        return _numpy_struct_to_png(val)

    if not isinstance(val, bytes):
        return None

    mime = _detect_mime(val)
    if mime is None:
        return None

    if mime == "view":
        # Malformed/unsupported blobs fail at header parsing or array reshape;
        # let unexpected errors (programming bugs) propagate rather than hide them.
        try:
            return _view_to_png(val)
        except (struct.error, ValueError, IndexError, OSError):
            return None

    if mime in ("image/png", "image/jpeg", "image/webp", "image/gif"):
        return val

    # TIFF/BMP: re-encode to PNG via PIL
    if mime in ("image/tiff", "image/bmp"):
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(val))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except (OSError, ValueError):
            return None

    return None
