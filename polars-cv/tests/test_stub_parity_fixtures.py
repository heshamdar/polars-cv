"""Fixtures for ``tests/_stub_parity.py``, the stub-vs-extension comparison.

``test_lib_stub_parity.py`` runs the comparison on the real extension. These run
it on small synthetic modules, so they hold without the plugin: each known-bad
pair is a drift the check must name, and each known-good pair a stub the check
must accept (a check that fires on a correct stub gets weakened by whoever hits
it next). They call the same ``stub_problems`` the real test calls.
"""

from __future__ import annotations

import types

import pytest

from tests._stub_parity import stub_problems

pytestmark = pytest.mark.structural


def _module(source: str) -> types.ModuleType:
    module = types.ModuleType("fake_lib")
    exec(compile(source, "fake_lib", "exec"), module.__dict__)  # noqa: S102
    return module


MODULE = _module(
    """
__version__ = "1"
FLAG = True

def f(a, b=None): ...
def g(a, /, b, *, c, d=1): ...
def h(*args, **kwargs): ...

class C:
    def __init__(self, x): ...
    def m(self, a, b=None): ...
    @staticmethod
    def s(a): ...
    @classmethod
    def k(cls, a): ...
    @property
    def p(self): ...
    def __len__(self): ...
"""
)

GOOD_STUB = '''
"""Doc."""
from __future__ import annotations
import polars as pl
__version__: str
FLAG: bool
def f(a: int, b: int | None = None) -> None: ...
def g(a: int, /, b: int, *, c: int, d: int = 1) -> None: ...
def h(*args: int, **kwargs: int) -> None: ...
class C:
    """Doc."""
    def __init__(self, x: int) -> None: ...
    def m(self, a: int, b: int | None = None) -> None: ...
    @staticmethod
    def s(a: int) -> None: ...
    @classmethod
    def k(cls, a: int) -> None: ...
    @property
    def p(self) -> int: ...
    def __len__(self) -> int: ...
'''

F_SIG = "def f(a: int, b: int | None = None)"


def _edit(old: str, new: str) -> str:
    assert old in GOOD_STUB, old
    return GOOD_STUB.replace(old, new)


def test_a_faithful_stub_is_accepted() -> None:
    """The known-good stub raises nothing (the baseline every bad one departs from)."""
    assert stub_problems(GOOD_STUB, MODULE) == []


def test_default_values_and_annotations_are_not_compared() -> None:
    """Rust's ``None``/struct defaults are spelled ``...`` in a stub; only presence counts."""
    stub = _edit(F_SIG + " -> None: ...", "def f(a, b=...): ...")
    assert stub_problems(stub, MODULE) == []


BAD = {
    "missing function": (
        GOOD_STUB + "def gone(a: int) -> None: ...\n",
        "stub entry `gone` is missing from the built module",
    ),
    "missing class": (
        GOOD_STUB + "class Gone: ...\n",
        "stub entry `Gone` is missing from the built module",
    ),
    "missing value": (
        GOOD_STUB + "NOPE: str\n",
        "stub entry `NOPE` is missing from the built module",
    ),
    "wrong value type": (
        _edit("FLAG: bool", "FLAG: str"),
        "stub entry `FLAG` is annotated `str` but the built value is a `bool`",
    ),
    "renamed parameter": (
        _edit("def f(a: int, b:", "def f(a: int, bee:"),
        "stub entry `f` disagrees",
    ),
    "extra parameter": (
        _edit(F_SIG, "def f(a: int, b: int | None = None, z: int = 0)"),
        "stub entry `f` disagrees",
    ),
    "dropped parameter": (
        _edit(F_SIG, "def f(a: int)"),
        "stub entry `f` disagrees",
    ),
    "reordered parameters": (
        _edit(F_SIG, "def f(b: int | None = None, a: int = 0)"),
        "same names, different order",
    ),
    "default dropped": (
        _edit(F_SIG, "def f(a: int, b: int | None)"),
        "stub entry `f` disagrees",
    ),
    "kind: keyword-only declared positional": (
        _edit("b: int, *, c: int, d: int = 1", "b: int, c: int, d: int = 1"),
        "stub entry `g` disagrees",
    ),
    "kind: positional-only declared plain": (
        _edit("def g(a: int, /, b", "def g(a: int, b"),
        "stub entry `g` disagrees",
    ),
    "kind: varargs dropped": (
        _edit("def h(*args: int, **kwargs: int)", "def h(**kwargs: int)"),
        "stub entry `h` disagrees",
    ),
    "method parameter renamed": (
        _edit("def m(self, a: int,", "def m(self, aa: int,"),
        "stub entry `C.m` disagrees",
    ),
    "constructor parameter renamed": (
        _edit("def __init__(self, x: int)", "def __init__(self, y: int)"),
        "stub entry `C.__init__` disagrees",
    ),
    "constructor takes none": (
        _edit("def __init__(self, x: int)", "def __init__(self)"),
        "stub entry `C.__init__` disagrees",
    ),
    "staticmethod stubbed as method": (
        _edit("    @staticmethod\n    def s(a: int)", "    def s(self, a: int)"),
        "stub entry `C.s` is a method but the built class defines a staticmethod",
    ),
    "property stubbed as method": (
        _edit("    @property\n    def p(self)", "    def p(self)"),
        "stub entry `C.p` is a method but the built class defines a property",
    ),
    "missing method": (
        _edit(
            "def __len__(self) -> int: ...",
            "def __len__(self) -> int: ...\n    def gone(self) -> None: ...",
        ),
        "stub entry `C.gone` is missing from the built class",
    ),
    "class stubbed as function": (
        _edit("class C:", "def C(x: int) -> None: ...\nclass C0:"),
        "stub entry `C` is a function but the built object is not a plain callable",
    ),
    "unmodelled construct": (
        GOOD_STUB + "X = 1\n",
        "the stub has a construct the check does not model: `X = 1`",
    ),
}


@pytest.mark.parametrize("label", sorted(BAD))
def test_a_drifted_stub_is_rejected_for_the_reason_it_claims(label: str) -> None:
    """Each known-bad stub produces a problem naming the entry and the drift."""
    stub, expected = BAD[label]
    problems = stub_problems(stub, MODULE)
    assert any(expected in p for p in problems), (label, problems)


@pytest.mark.parametrize(
    ("dropped", "name"),
    [
        (F_SIG + " -> None: ...\n", "f"),
        ("__version__: str\n", "__version__"),
        ("    def m(self, a: int, b: int | None = None) -> None: ...\n", "C.m"),
        ("    def __len__(self) -> int: ...\n", "C.__len__"),
    ],
)
def test_a_name_the_stub_omits_is_rejected(dropped: str, name: str) -> None:
    """The reverse direction: the module has it, the stub does not."""
    problems = stub_problems(_edit(dropped, ""), MODULE)
    assert any(f"`{name}`" in p and "not in the stub" in p for p in problems), problems


def test_an_allowlisted_omission_is_accepted_and_a_stale_one_is_not() -> None:
    """The allowlist excuses a real omission, and only a real one, with a reason."""
    stub = _edit("def h(*args: int, **kwargs: int) -> None: ...\n", "")
    assert stub_problems(stub, MODULE, unstubbed={"h": "test-only"}) == []
    assert any(
        "no reason" in p for p in stub_problems(stub, MODULE, unstubbed={"h": " "})
    )
    assert any(
        "declared in the stub" in p
        for p in stub_problems(GOOD_STUB, MODULE, unstubbed={"h": "x"})
    )
    assert any(
        "not in the built module" in p
        for p in stub_problems(GOOD_STUB, MODULE, unstubbed={"nope": "x"})
    )


def test_a_callable_without_a_signature_is_reported_not_skipped() -> None:
    """``inspect.signature`` failing must surface; skipping would read as covered."""

    class Opaque:
        @property
        def __signature__(self) -> object:
            raise ValueError("no signature found")

        def __call__(self) -> None: ...

    module = types.ModuleType("opaque")
    module.f = Opaque()  # type: ignore[attr-defined]
    problems = stub_problems("def f(a: int) -> None: ...\n", module)
    assert any("stub entry `f` cannot be checked" in p for p in problems), problems
