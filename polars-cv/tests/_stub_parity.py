"""Hold a hand-written ``.pyi`` stub to the module it describes, in both directions.

``polars_cv/_lib.pyi`` is a second copy of the compiled extension's surface, and
nothing but this comparison keeps it honest: ``ty`` reads the stub, never the
built ``.so``, so a stub that disagrees with Rust type-checks cleanly. This is
the comparison; ``test_lib_stub_parity.py`` runs it against the real
``polars_cv._lib`` and ``test_stub_parity_fixtures.py`` holds it to known-bad
and known-good pairs.

mypy's ``stubtest`` would do most of this, but it is not a dependency of the
project and the shared dev environment does not carry it, so the comparison is
written here (``ast`` for the stub, ``inspect`` for the module).

What is checked, for every stub entry, against the object of the same name:

* a module-level name exists, and a simple annotation (``str``, ``bool``, ...)
  matches the value's type;
* a ``def`` exists, is callable and is not a class, and its parameters match the
  runtime signature in **name, kind (positional-only, keyword-only, ``*args``,
  ...), order and whether a default exists**. Default *values* are not compared:
  a stub writes ``...`` where Rust has ``None`` or a struct;
* a ``class`` exists and is a class; each member in its body exists with the
  same kind (method, ``property``, ``staticmethod``, ``classmethod``) and, for a
  method, the same signature; ``__init__`` is compared to the constructor's
  signature (pyo3 spells it ``__new__``), so ``__init__(self)`` means
  ``Cls()``.

And the reverse: every name the module (or a class) defines must be in the stub,
or, for module-level names, in ``unstubbed`` with a reason. A stale ``unstubbed``
entry (one the stub now declares, or the module no longer has) fails too, so the
list cannot outlive its cause.

Limits, stated because a guard's blind spots are the thing to know about it:

* Only what ``inspect.signature`` can read is compared. A pyo3 function exposes
  ``__text_signature__`` and is covered; a callable that raises from
  ``inspect.signature`` (a builtin with no text signature) is reported as
  *unintrospectable*, never skipped -- a callable that cannot be checked has to
  be allowlisted by someone who has read why.
* Types are not compared: a parameter's or a return's annotation is not checked
  against Rust. Names, kinds, order and defaults are the part that drifts
  silently when a ``#[pyfunction]`` signature changes; the type spelling is not
  recoverable from the extension at all.
* A stub construct this does not model (an ``@overload``, an assignment other
  than ``name: T``, a nested class) is an error, not an ignore.
* Reverse direction for classes skips the dunders every object has
  (``_PLUMBING_DUNDERS``); any other dunder the class defines must be stubbed.
"""

from __future__ import annotations

import ast
import builtins
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any

#: Names every module carries that describe the import machinery, not the
#: extension's surface. ``__all__`` is the module's own export list (Rust's
#: ``#[pymodule]`` sets it) and is what the stub is compared against, not a name
#: to declare.
_MODULE_PLUMBING = frozenset(
    {
        "__all__",
        "__builtins__",
        "__cached__",
        "__doc__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__path__",
        "__spec__",
    }
)

#: Dunders every pyo3 ``#[pyclass]`` carries however it was declared (pickling,
#: copying, the constructor spelled ``__new__``, the comparison and hash
#: protocols derived by ``eq``/``ord``/``hash``). They are not part of what a
#: stub author chooses to expose; any *other* dunder must be stubbed.
_PLUMBING_DUNDERS = frozenset(
    {
        # Bookkeeping CPython adds to classes defined in Python (the fixtures).
        "__annotate__",
        "__annotate_func__",
        "__annotations__",
        "__annotations_cache__",
        "__firstlineno__",
        "__static_attributes__",
        "__copy__",
        "__deepcopy__",
        "__dict__",
        "__doc__",
        "__eq__",
        "__ge__",
        "__gt__",
        "__hash__",
        "__le__",
        "__lt__",
        "__module__",
        "__ne__",
        "__new__",
        "__qualname__",
        "__reduce__",
        "__repr__",
        "__weakref__",
    }
)

_KIND_NAMES = {
    inspect.Parameter.POSITIONAL_ONLY: "positional-only",
    inspect.Parameter.POSITIONAL_OR_KEYWORD: "positional-or-keyword",
    inspect.Parameter.VAR_POSITIONAL: "*args",
    inspect.Parameter.KEYWORD_ONLY: "keyword-only",
    inspect.Parameter.VAR_KEYWORD: "**kwargs",
}


@dataclass(frozen=True)
class Param:
    """One parameter as the stub or the runtime declares it."""

    name: str
    kind: inspect._ParameterKind
    has_default: bool

    def render(self) -> str:
        """The parameter as written in a signature, for messages."""
        text = self.name
        if self.kind is inspect.Parameter.VAR_POSITIONAL:
            text = "*" + text
        elif self.kind is inspect.Parameter.VAR_KEYWORD:
            text = "**" + text
        if self.has_default:
            text += "=..."
        if self.kind is inspect.Parameter.KEYWORD_ONLY:
            text = "kw:" + text
        return text


def render_params(params: list[Param]) -> str:
    """A parameter list as ``(a, b=..., kw:c)``, positional-only ones before ``/``."""
    parts: list[str] = []
    for index, p in enumerate(params):
        parts.append(p.render())
        nxt = params[index + 1] if index + 1 < len(params) else None
        if p.kind is inspect.Parameter.POSITIONAL_ONLY and (
            nxt is None or nxt.kind is not inspect.Parameter.POSITIONAL_ONLY
        ):
            parts.append("/")
    return "(" + ", ".join(parts) + ")"


def _stub_params(args: ast.arguments) -> list[Param]:
    """The stub ``def``'s parameters, in declaration order."""
    positional = [*args.posonlyargs, *args.args]
    first_default = len(positional) - len(args.defaults)
    out: list[Param] = []
    for index, a in enumerate(positional):
        kind = (
            inspect.Parameter.POSITIONAL_ONLY
            if index < len(args.posonlyargs)
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        out.append(Param(a.arg, kind, index >= first_default))
    if args.vararg is not None:
        out.append(Param(args.vararg.arg, inspect.Parameter.VAR_POSITIONAL, False))
    for a, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        out.append(Param(a.arg, inspect.Parameter.KEYWORD_ONLY, default is not None))
    if args.kwarg is not None:
        out.append(Param(args.kwarg.arg, inspect.Parameter.VAR_KEYWORD, False))
    return out


def _runtime_params(signature: inspect.Signature) -> list[Param]:
    return [
        Param(p.name, p.kind, p.default is not inspect.Parameter.empty)
        for p in signature.parameters.values()
    ]


def _decorator_names(node: ast.FunctionDef) -> list[str]:
    names: list[str] = []
    for d in node.decorator_list:
        if isinstance(d, ast.Name):
            names.append(d.id)
        elif isinstance(d, ast.Attribute):
            names.append(d.attr)
        else:
            names.append(ast.unparse(d))
    return names


def _stub_member_kind(node: ast.FunctionDef) -> tuple[str | None, str | None]:
    """``(kind, problem)`` for a method: method / property / staticmethod / classmethod."""
    decorators = _decorator_names(node)
    known = {"property", "staticmethod", "classmethod"}
    unknown = [d for d in decorators if d not in known]
    if unknown:
        return None, f"decorator(s) {unknown} are not modelled by the stub check"
    present = [d for d in decorators if d in known]
    if len(present) > 1:
        return None, f"decorators {present} cannot combine"
    return (present[0] if present else "method"), None


def _runtime_member_kind(cls: type, name: str) -> str:
    """How ``cls`` defines ``name``, read without triggering descriptors."""
    static = inspect.getattr_static(cls, name)
    if isinstance(static, staticmethod):
        return "staticmethod"
    if isinstance(static, classmethod):
        return "classmethod"
    if isinstance(static, property) or inspect.isgetsetdescriptor(static):
        return "property"
    if inspect.ismemberdescriptor(static):
        return "attribute"
    return "method" if callable(static) else "attribute"


def _compare_params(
    entry: str, stub: list[Param], built: list[Param], out: list[str]
) -> None:
    if stub == built:
        return
    detail = ""
    if [p.name for p in stub] != [p.name for p in built]:
        gone = [p.name for p in stub if p.name not in {q.name for q in built}]
        new = [p.name for p in built if p.name not in {q.name for q in stub}]
        if gone or new:
            detail = (
                f" (the stub has {gone or 'nothing'} the built module lacks; "
                f"the built module has {new or 'nothing'} the stub lacks)"
            )
        else:
            detail = " (same names, different order)"
    out.append(
        f"stub entry `{entry}` disagrees with the built module: "
        f"stub {render_params(stub)} vs built {render_params(built)}{detail}"
    )


def _signature_of(entry: str, obj: Any, out: list[str]) -> inspect.Signature | None:
    try:
        return inspect.signature(obj)
    except (TypeError, ValueError) as exc:
        out.append(
            f"stub entry `{entry}` cannot be checked: the built object has no "
            f"introspectable signature ({exc}); give it a `#[pyo3(signature = ...)]`/"
            "text signature or allowlist it with a reason"
        )
        return None


def _check_function(
    entry: str,
    node: ast.FunctionDef,
    obj: Any,
    out: list[str],
    *,
    member_kind: str = "function",
) -> None:
    """Compare one stub ``def`` with ``obj``'s signature.

    ``member_kind`` says how ``self``/``cls`` figure: a plain ``function`` has
    neither; an instance ``method`` has ``self`` in both the stub and the
    runtime signature; a ``classmethod`` has ``cls`` only in the stub (the
    runtime attribute is already bound); ``constructor`` compares ``__init__``
    minus ``self`` to the class's own signature.
    """
    sig = _signature_of(entry, obj, out)
    if sig is None:
        return
    stub = _stub_params(node.args)
    built = _runtime_params(sig)
    if member_kind in ("method", "constructor", "classmethod"):
        if not stub or stub[0].name not in ("self", "cls"):
            out.append(
                f"stub entry `{entry}` is a method whose first parameter is not "
                "`self`/`cls`"
            )
            return
        stub = stub[1:]
    if member_kind == "method":
        if not built:
            out.append(f"stub entry `{entry}`: built method takes no `self`")
            return
        built = built[1:]
    _compare_params(entry, stub, built, out)


def _check_class(name: str, node: ast.ClassDef, cls: Any, out: list[str]) -> None:
    if not inspect.isclass(cls):
        out.append(f"stub entry `{name}` is a class but the built object is not")
        return
    declared: set[str] = set()
    for item in node.body:
        if (
            isinstance(item, ast.Expr)
            and isinstance(item.value, ast.Constant)
            and isinstance(item.value.value, str)
        ):
            continue  # the class docstring
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            member = item.target.id
            declared.add(member)
            entry = f"{name}.{member}"
            if not hasattr(cls, member):
                out.append(f"stub entry `{entry}` is missing from the built class")
            continue
        if not isinstance(item, ast.FunctionDef):
            out.append(
                f"stub entry `{name}` has a construct the check does not model: "
                f"`{ast.unparse(item).splitlines()[0]}`"
            )
            continue
        member = item.name
        declared.add(member)
        entry = f"{name}.{member}"
        kind, problem = _stub_member_kind(item)
        if problem is not None or kind is None:
            out.append(f"stub entry `{entry}`: {problem}")
            continue
        if member == "__init__":
            if kind != "method":
                out.append(f"stub entry `{entry}` must be a plain method")
                continue
            _check_function(entry, item, cls, out, member_kind="constructor")
            continue
        if not hasattr(cls, member):
            out.append(f"stub entry `{entry}` is missing from the built class")
            continue
        built_kind = _runtime_member_kind(cls, member)
        if built_kind != kind:
            out.append(
                f"stub entry `{entry}` is a {kind} but the built class defines a "
                f"{built_kind}"
            )
            continue
        if kind == "property":
            continue  # a property has no signature; its kind is the whole claim
        _check_function(
            entry,
            item,
            getattr(cls, member),
            out,
            member_kind={"staticmethod": "function"}.get(kind, kind),
        )
    for member in vars(cls):
        if member in declared or member in _PLUMBING_DUNDERS:
            continue
        if member == "__init__":
            continue
        out.append(
            f"`{name}.{member}` is defined by the built class but is not in the stub"
        )


def _check_value(name: str, node: ast.AnnAssign, value: Any, out: list[str]) -> None:
    ann = node.annotation
    expected = getattr(builtins, ann.id, None) if isinstance(ann, ast.Name) else None
    if not isinstance(expected, type):
        out.append(
            f"stub entry `{name}` is annotated `{ast.unparse(ann)}`, which the "
            "check cannot compare; only builtin types are modelled"
        )
    elif not isinstance(value, expected):
        out.append(
            f"stub entry `{name}` is annotated `{expected.__name__}` but the built "
            f"value is a `{type(value).__name__}`"
        )


def stub_problems(
    stub_source: str,
    module: ModuleType,
    *,
    unstubbed: Mapping[str, str] | None = None,
) -> list[str]:
    """Every disagreement between ``stub_source`` and ``module``; empty when they agree.

    ``unstubbed`` maps each module-level name deliberately absent from the stub
    to the reason it is. A blank reason, a name the stub declares, or a name the
    module lacks is itself a problem.
    """
    unstubbed = dict(unstubbed or {})
    out: list[str] = []
    tree = ast.parse(stub_source)
    declared: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # the module docstring
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            declared.add(name)
            if not hasattr(module, name):
                out.append(f"stub entry `{name}` is missing from the built module")
            else:
                _check_value(name, node, getattr(module, name), out)
        elif isinstance(node, ast.ClassDef):
            declared.add(node.name)
            if not hasattr(module, node.name):
                out.append(f"stub entry `{node.name}` is missing from the built module")
            else:
                _check_class(node.name, node, getattr(module, node.name), out)
        elif isinstance(node, ast.FunctionDef):
            declared.add(node.name)
            if _decorator_names(node):
                out.append(
                    f"stub entry `{node.name}` is decorated; the check does not "
                    "model decorated module-level functions"
                )
            elif not hasattr(module, node.name):
                out.append(f"stub entry `{node.name}` is missing from the built module")
            elif inspect.isclass(getattr(module, node.name)) or not callable(
                getattr(module, node.name)
            ):
                out.append(
                    f"stub entry `{node.name}` is a function but the built object is "
                    "not a plain callable"
                )
            else:
                _check_function(node.name, node, getattr(module, node.name), out)
        else:
            out.append(
                "the stub has a construct the check does not model: "
                f"`{ast.unparse(node).splitlines()[0]}`"
            )
    for name in sorted(vars(module)):
        if name in declared or name in _MODULE_PLUMBING or name in unstubbed:
            continue
        out.append(
            f"`{name}` is exported by the built module but is not in the stub "
            "and not in the allowlist of deliberately unstubbed names"
        )
    for name, reason in sorted(unstubbed.items()):
        if not reason.strip():
            out.append(f"allowlisted name `{name}` has no reason")
        if name in declared:
            out.append(
                f"allowlisted name `{name}` is declared in the stub; drop it from "
                "the allowlist"
            )
        if not hasattr(module, name):
            out.append(
                f"allowlisted name `{name}` is not in the built module; drop it "
                "from the allowlist"
            )
    return out
