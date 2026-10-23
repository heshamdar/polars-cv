"""``this``: op parameters derived from the buffer the op reads.

``crop(height=this.height // 2)`` halves every image whatever its size, and
``blur(sigma=this.height / 320)`` scales the blur with it. Each op reads the
buffer it is given, so after ``resize(height=512, ...)`` ``this.height`` is
512. The expression is a tree sent to the plugin (``{"$this": tree}``) and
evaluated per row; nothing is computed here. An integer parameter takes the
value rounded to nearest.

``this`` (generated into ``_ops_generated``: one property per Rust
``ThisAttr``) has the attributes known before an op runs: ``height``,
``width``, ``channels`` and ``ndim``. Only a parameter that may be a per-row expression
takes one: a size that is then unknown when the query is planned, as for a
column.
"""

from __future__ import annotations

import math
from typing import Any, Union

#: A wire tree: an attribute name, a number, or ``[op, a, b]``.
_Tree = Union[str, int, float, list[Any]]


class ThisExpr:
    """A number computed, per row, from the buffer an op reads: an attribute
    of :data:`this`, combined with numbers by ``+ - * / //``. Pass it as any
    numeric parameter that takes an expression."""

    __slots__ = ("_tree",)

    def __init__(self, tree: _Tree) -> None:
        self._tree = tree

    def _wire(self) -> dict[str, _Tree]:
        """The parameter's wire value."""
        return {"$this": self._tree}

    def _op(self, op: str, other: Any, *, reflected: bool = False) -> ThisExpr:
        operand = _operand(other)
        if operand is None:
            # Raised, not `NotImplemented`: that would let the other side's
            # reflected operator (a `pl.Expr`'s) take the expression as a value.
            msg = (
                "`this` attributes combine with finite numbers and other `this` "
                f"expressions, got {other!r}"
            )
            raise TypeError(msg)
        a, b = (operand, self._tree) if reflected else (self._tree, operand)
        return ThisExpr([op, a, b])

    def __add__(self, other: Any) -> ThisExpr:
        return self._op("+", other)

    def __radd__(self, other: Any) -> ThisExpr:
        return self._op("+", other, reflected=True)

    def __sub__(self, other: Any) -> ThisExpr:
        return self._op("-", other)

    def __rsub__(self, other: Any) -> ThisExpr:
        return self._op("-", other, reflected=True)

    def __mul__(self, other: Any) -> ThisExpr:
        return self._op("*", other)

    def __rmul__(self, other: Any) -> ThisExpr:
        return self._op("*", other, reflected=True)

    def __truediv__(self, other: Any) -> ThisExpr:
        return self._op("/", other)

    def __rtruediv__(self, other: Any) -> ThisExpr:
        return self._op("/", other, reflected=True)

    def __floordiv__(self, other: Any) -> ThisExpr:
        return self._op("//", other)

    def __rfloordiv__(self, other: Any) -> ThisExpr:
        return self._op("//", other, reflected=True)

    def __neg__(self) -> ThisExpr:
        return ThisExpr(["-", 0, self._tree])

    def __bool__(self) -> bool:
        msg = "a `this` expression has no value until a row runs; it cannot be tested"
        raise TypeError(msg)

    def __repr__(self) -> str:
        # The engine's text for the tree, which its errors quote too: one
        # renderer, so the two cannot read differently.
        import json

        from polars_cv import _lib

        return _lib._this_repr(json.dumps(self._tree))


def _operand(value: Any) -> _Tree | None:
    """*value* as a tree operand; ``None`` for one that is not a finite
    number or a :class:`ThisExpr`."""
    if isinstance(value, ThisExpr):
        return value._tree
    # A numpy scalar is the number it holds, as for any literal.
    from polars_cv._types import _to_python

    value = _to_python(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return value
