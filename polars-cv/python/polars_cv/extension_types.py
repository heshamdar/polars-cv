"""polars-cv's Arrow extension types.

An extension type tags a column with *what it is* — a point, a contour, a
bounding box, an N-D array — on top of the plain struct that stores it. A
consumer then identifies the column by its type instead of sniffing field
names, and a tagged column keeps that identity through Parquet and IPC.

========================  ============================================
type                      storage
========================  ============================================
``polars_cv.ndarray``     :data:`NUMPY_OUTPUT_SCHEMA` (the numpy/torch sink struct)
``polars_cv.point``       :data:`~polars_cv.geometry.schemas.POINT_SCHEMA`
``polars_cv.contour``     :data:`~polars_cv.geometry.schemas.CONTOUR_SCHEMA`
``polars_cv.bbox``        :data:`~polars_cv.geometry.schemas.BBOX_SCHEMA`
========================  ============================================

Producing and consuming them:

- ``.sink("ndarray")`` emits a ``polars_cv.ndarray`` column; ``sink("numpy")``
  keeps emitting the plain struct.
- Any column with the right storage can be tagged with ``.ext.to(PointType())``
  and friends — a zero-copy relabel.
- Every polars-cv expression accepts a tagged column wherever it accepts the
  plain struct. Geometry outputs are plain structs.
- ``.ext.storage()`` recovers the plain struct. Polars' struct operations
  (``.struct.field``, ``unnest``, casts) need it: they do not see through a tag.

**An instance of one of these classes always has canonical storage.** polars
rebuilds a registered type through :meth:`~_PolarsCvType.ext_from_params`, and
a column that carries one of these names over any other storage (or with
metadata) comes back as polars' generic :class:`polars.Extension` instead — so
``isinstance(dtype, PointType)`` is a complete check, never a guess.

**Where the facts live.** The storage layouts belong to the Rust plugin
(``geom_schema``, ``output::numpy_output_dtype``); ``ExtType`` in
``src/ext_types.rs`` names the four types there and ``_lib.extension_types()``
publishes them. They are restated here only so ``import polars_cv`` can register
the types without loading the compiled extension (registering late would let a
Parquet read decay a tagged column to its storage). The parity test
``test_python_types_match_the_rust_declaration`` holds the two together.

.. warning::
    Polars documents its extension-type API as unstable. Everything polars-cv
    does with it goes through this module and ``src/ext_types.rs``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import polars as pl

from polars_cv.geometry.schemas import BBOX_SCHEMA, CONTOUR_SCHEMA, POINT_SCHEMA

#: Schema of the numpy/torch sink output struct, and the storage of
#: :class:`NdArrayType`. Matches ``output::numpy_output_dtype`` in the Rust
#: plugin (held to it by the extension-type parity test).
NUMPY_OUTPUT_SCHEMA = pl.Struct(
    {
        "data": pl.Binary,
        "dtype": pl.String,
        "shape": pl.List(pl.UInt64),
        "strides": pl.List(pl.Int64),
        "offset": pl.UInt64,
    }
)


class _PolarsCvType(pl.datatypes.BaseExtension):
    """A polars-cv extension type: one name over one fixed storage, no metadata."""

    NAME: ClassVar[str]
    STORAGE: ClassVar[pl.DataType]
    #: Short form shown in a DataFrame header, e.g. ``ext[point]``.
    DISPLAY: ClassVar[str]

    def __init__(self) -> None:
        super().__init__(name=self.NAME, storage=self.STORAGE, metadata=None)

    @classmethod
    def ext_from_params(
        cls, name: str, storage: Any, metadata: str | None
    ) -> pl.datatypes.BaseExtension:
        """Rebuild the type polars read, refusing to vouch for a malformed one.

        Returns an instance of this class only for the canonical storage and no
        metadata. Anything else under this name is returned as the generic
        ``pl.Extension``: raising here is not an option, because polars turns
        an exception from this hook into a panic.
        """
        if name == cls.NAME and storage == cls.STORAGE and metadata is None:
            return cls()
        return pl.Extension(name, storage, metadata)

    def _string_repr(self) -> str:
        return self.DISPLAY


class NdArrayType(_PolarsCvType):
    """``polars_cv.ndarray``: a strided N-D array, as the numpy/torch sink struct.

    Emitted by ``.sink("ndarray")``; read back with
    :func:`polars_cv.numpy_from_struct`.
    """

    NAME = "polars_cv.ndarray"
    STORAGE = NUMPY_OUTPUT_SCHEMA
    DISPLAY = "ndarray"


class PointType(_PolarsCvType):
    """``polars_cv.point``: a 2-D ``{x, y}`` point."""

    NAME = "polars_cv.point"
    STORAGE = POINT_SCHEMA
    DISPLAY = "point"


class ContourType(_PolarsCvType):
    """``polars_cv.contour``: a polygon ``{exterior, holes, is_closed}``."""

    NAME = "polars_cv.contour"
    STORAGE = CONTOUR_SCHEMA
    DISPLAY = "contour"


class BBoxType(_PolarsCvType):
    """``polars_cv.bbox``: an axis-aligned box ``{x, y, width, height}``."""

    NAME = "polars_cv.bbox"
    STORAGE = BBOX_SCHEMA
    DISPLAY = "bbox"


#: Every polars-cv extension type, in the order ``ExtType::ALL`` declares them.
#: Registering reads this tuple and the parity test reads the same tuple, so a
#: type cannot be registered without being checked against Rust.
EXTENSION_TYPES: tuple[type[_PolarsCvType], ...] = (
    NdArrayType,
    PointType,
    ContourType,
    BBoxType,
)


def _is_same_class(a: object, b: type) -> bool:
    """True for *b* itself, or a re-execution of its module (``importlib.reload``)."""
    return a is b or (
        isinstance(a, type)
        and a.__module__ == b.__module__
        and a.__qualname__ == b.__qualname__
    )


def register_extension_types() -> None:
    """Register every polars-cv type with polars. Idempotent.

    Called once at ``import polars_cv``. A name already registered to a class
    that is not ours is refused: registering over it would change how the other
    code's columns decode, and leaving it would change how ours do.
    """
    for cls in EXTENSION_TYPES:
        existing = pl.get_extension_type(cls.NAME)
        if existing is cls:
            continue
        if existing is not None and not _is_same_class(existing, cls):
            msg = (
                f"extension type {cls.NAME!r} is already registered to "
                f"{existing!r}; the 'polars_cv.' names belong to polars-cv"
            )
            raise RuntimeError(msg)
        if existing is not None:
            # A reloaded module: replace the stale class with the current one.
            pl.unregister_extension_type(cls.NAME)
        pl.register_extension_type(cls.NAME, cls)
