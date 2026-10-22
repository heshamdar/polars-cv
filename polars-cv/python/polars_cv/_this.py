"""``this``: op parameters derived from the buffer the op reads.

``crop(height=this.height // 2)`` halves every image whatever its size, and
``blur(sigma=this.height / 320)`` scales the blur with it. Each op reads the
buffer it is given, so after ``resize(height=512, ...)`` ``this.height`` is
512. The expression is a tree sent to the plugin (``{"$this": tree}``) and
evaluated per row; nothing is computed here. An integer parameter takes the
value rounded to nearest.

``this`` has the attributes known before an op runs: ``height``, ``width``,
``channels`` and ``ndim``. Only a parameter that may be a per-row expression
takes one: a size that is then unknown when the query is planned, as for a
column.
"""

from __future__ import annotations

import math
from typing import Any, Union

#: A wire tree: an attribute name, a number, or ``[op, a, b]``.
_Tree = Union[str, int, float, list[Any]]

#: How tightly each operator binds, as in Python (for ``repr``).
_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "//": 2}


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
        return _render(self._tree)


class This:
    """The buffer an op reads, as its parameters see it. Use the instance
    :data:`this`; each attribute is a :class:`ThisExpr`."""

    __slots__ = ()

    @property
    def height(self) -> ThisExpr:
        """Axis 0 of an ``[H, W]`` or ``[H, W, C]`` buffer."""
        return ThisExpr("height")

    @property
    def width(self) -> ThisExpr:
        """Axis 1 of an ``[H, W]`` or ``[H, W, C]`` buffer."""
        return ThisExpr("width")

    @property
    def channels(self) -> ThisExpr:
        """Axis 2 of an ``[H, W, C]`` buffer; 1 for an ``[H, W]`` one."""
        return ThisExpr("channels")

    @property
    def ndim(self) -> ThisExpr:
        """The rank, of any buffer."""
        return ThisExpr("ndim")

    def __repr__(self) -> str:
        return "this"


def _operand(value: Any) -> _Tree | None:
    """*value* as a tree operand; ``None`` for one that is not a finite
    number or a :class:`ThisExpr`."""
    if isinstance(value, ThisExpr):
        return value._tree
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return value


def _render(tree: _Tree, parent: str | None = None, *, right: bool = False) -> str:
    if isinstance(tree, str):
        return f"this.{tree}"
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


#: The buffer an op reads: ``this.height``, ``this.width``, ``this.channels``,
#: ``this.ndim``.
this = This()
