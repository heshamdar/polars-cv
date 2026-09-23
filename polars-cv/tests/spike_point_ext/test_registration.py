"""SPIKE (throwaway): ``ensure_registered`` fails loudly instead of degrading.

Pure-Python checks of the shared registration helper: a missing plugin, a plugin
built without the spike feature, and a host registration error must each raise,
and a failed registration must not be recorded as done. Needs no compiled
extension — ``polars_cv._lib`` is replaced in ``sys.modules``.
"""

from __future__ import annotations

import sys
import types

import polars as pl
import pytest

from tests.spike_point_ext import _ext


class _Dummy(pl.datatypes.BaseExtension):
    def __init__(self) -> None:
        super().__init__(name="polars_cv.test_dummy", storage=pl.Int8, metadata=None)


@pytest.fixture
def fresh_registry(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, type]]:
    """Isolate the helper's module state: one pending type, nothing registered."""
    pending = [("polars_cv.test_dummy", _Dummy)]
    monkeypatch.setattr(_ext, "_PENDING", pending)
    monkeypatch.setattr(_ext, "_REGISTERED_NAMES", set())
    return pending


def _fake_lib(monkeypatch: pytest.MonkeyPatch, *, spike: bool) -> None:
    lib = types.ModuleType("polars_cv._lib")
    if spike:
        lib.__spike_ext_types__ = True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "polars_cv._lib", lib)


def test_missing_plugin_raises(monkeypatch: pytest.MonkeyPatch, fresh_registry) -> None:
    # A None entry makes `import polars_cv._lib` raise ImportError.
    monkeypatch.setitem(sys.modules, "polars_cv._lib", None)

    with pytest.raises(ImportError, match="compiled plugin"):
        _ext.ensure_registered()
    assert _ext._REGISTERED_NAMES == set()


def test_plugin_without_spike_feature_raises(
    monkeypatch: pytest.MonkeyPatch, fresh_registry
) -> None:
    _fake_lib(monkeypatch, spike=False)

    with pytest.raises(RuntimeError, match="spike-ext-types"):
        _ext.ensure_registered()
    assert _ext._REGISTERED_NAMES == set()


def test_host_registration_error_propagates(
    monkeypatch: pytest.MonkeyPatch, fresh_registry
) -> None:
    _fake_lib(monkeypatch, spike=True)

    def _boom(*_a: object, **_k: object) -> None:
        raise ValueError("registry rejected it")

    monkeypatch.setattr(pl, "register_extension_type", _boom)

    with pytest.raises(ValueError, match="registry rejected it"):
        _ext.ensure_registered()
    assert _ext._REGISTERED_NAMES == set()


def test_successful_registration_is_recorded_once(
    monkeypatch: pytest.MonkeyPatch, fresh_registry
) -> None:
    _fake_lib(monkeypatch, spike=True)
    calls: list[str] = []
    monkeypatch.setattr(
        pl, "register_extension_type", lambda name, _cls: calls.append(name)
    )

    _ext.ensure_registered()
    _ext.ensure_registered()

    assert calls == ["polars_cv.test_dummy"]
    assert _ext._REGISTERED_NAMES == {"polars_cv.test_dummy"}
