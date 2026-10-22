"""``H`` and ``W``: op parameters derived from the shape of the buffer the op
reads.

``crop(height=H // 2)`` halves every image whatever its size, and
``blur(sigma=H / 320)`` scales the blur with it. Each op reads the buffer it is
given, so after ``resize(height=512, ...)`` ``H`` is 512. The expression is a
tree sent to the plugin (``{"$dim": tree}``) and evaluated per row; nothing is
computed here. An integer parameter takes the value rounded to nearest.

Only a parameter that may be a per-row expression takes one: a size that is
then unknown when the query is planned, as for a column.
"""

from __future__ import annotations

import math
from typing import Any, Union

#: A wire tree: ``"H"``, ``"W"``, a number, or ``[op, a, b]``.
_Tree = Union[str, int, float, list[Any]]

#: How tightly each operator binds, as in Python (for ``repr``).
_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "//": 2}


class Dim:
    """A number computed, per row, from the height ``H`` and width ``W`` of
    the buffer an op reads. Build one from :data:`H` and :data:`W` with
    ``+ - * / //``; pass it as any numeric parameter that takes an
    expression."""

    __slots__ = ("_tree",)

    def __init__(self, tree: _Tree) -> None:
        self._tree = tree

    def _wire(self) -> dict[str, _Tree]:
        """The parameter's wire value."""
        return {"$dim": self._tree}

    def _op(self, op: str, other: Any, *, reflected: bool = False) -> Dim:
        operand = _operand(other)
        if operand is None:
            # Raised, not `NotImplemented`: that would let the other side's
            # reflected operator (a `pl.Expr`'s) take the expression as a value.
            msg = (
                "H/W combine with finite numbers and other H/W expressions, "
                f"got {other!r}"
            )
            raise TypeError(msg)
        a, b = (operand, self._tree) if reflected else (self._tree, operand)
        return Dim([op, a, b])

    def __add__(self, other: Any) -> Dim:
        return self._op("+", other)

    def __radd__(self, other: Any) -> Dim:
        return self._op("+", other, reflected=True)

    def __sub__(self, other: Any) -> Dim:
        return self._op("-", other)

    def __rsub__(self, other: Any) -> Dim:
        return self._op("-", other, reflected=True)

    def __mul__(self, other: Any) -> Dim:
        return self._op("*", other)

    def __rmul__(self, other: Any) -> Dim:
        return self._op("*", other, reflected=True)

    def __truediv__(self, other: Any) -> Dim:
        return self._op("/", other)

    def __rtruediv__(self, other: Any) -> Dim:
        return self._op("/", other, reflected=True)

    def __floordiv__(self, other: Any) -> Dim:
        return self._op("//", other)

    def __rfloordiv__(self, other: Any) -> Dim:
        return self._op("//", other, reflected=True)

    def __neg__(self) -> Dim:
        return Dim(["-", 0, self._tree])

    def __bool__(self) -> bool:
        msg = "H/W have no value until a row runs; they cannot be tested"
        raise TypeError(msg)

    def __repr__(self) -> str:
        return _render(self._tree)


def _operand(value: Any) -> _Tree | None:
    """*value* as a tree operand; ``None`` for one that is not a finite
    number or a :class:`Dim`."""
    if isinstance(value, Dim):
        return value._tree
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return value


def _render(tree: _Tree, parent: str | None = None, *, right: bool = False) -> str:
    if not isinstance(tree, list):
        return str(tree)
    op, a, b = tree
    text = f"{_render(a, op)} {op} {_render(b, op, right=True)}"
    if parent is None:
        return text
    child, outer = _PRECEDENCE[op], _PRECEDENCE[parent]
    loose = child < outer or (
        right and child == outer and not (op == parent and op in ("+", "*"))
    )
    return f"({text})" if loose else text


#: The height of the buffer an op reads (axis 0).
H = Dim("H")
#: The width of the buffer an op reads (axis 1).
W = Dim("W")
