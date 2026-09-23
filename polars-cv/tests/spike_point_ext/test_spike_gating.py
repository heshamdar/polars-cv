"""SPIKE (throwaway): the extension-type spike never reaches the shipped wheel.

The Rust side compiles only under the opt-in ``spike-ext-types`` Cargo feature
and the Python side lives in this test package, outside ``python/polars_cv``.
These guards fail if either is undone — the feature defaulted on, added to the
maturin build features (which is what the release wheels are built with), or a
spike module moved back into the package.

Limits: reads ``Cargo.toml`` / ``pyproject.toml`` as TOML and lists the package
modules (via ``tests._discovery``); it does not inspect a built wheel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests._discovery import package_modules

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - 3.10 lane
    import tomli as tomllib

pytestmark = pytest.mark.structural

_PROJECT = Path(__file__).resolve().parents[2]
FEATURE = "spike-ext-types"


def _toml(name: str) -> dict:
    return tomllib.loads((_PROJECT / name).read_text())


def test_spike_feature_exists_and_is_opt_in() -> None:
    features = _toml("Cargo.toml")["features"]
    assert FEATURE in features, f"`{FEATURE}` feature missing from Cargo.toml"
    assert FEATURE not in features.get("default", [])


def test_spike_feature_not_in_maturin_build_features() -> None:
    maturin_features = _toml("pyproject.toml")["tool"]["maturin"].get("features", [])
    assert FEATURE not in maturin_features


def test_polars_extension_dtype_only_enabled_by_the_spike_feature() -> None:
    cargo = _toml("Cargo.toml")
    assert "dtype-extension" not in cargo["dependencies"]["polars"]["features"]
    assert "dtype-extension" not in cargo["dev-dependencies"]["polars"]["features"]
    assert "polars/dtype-extension" in cargo["features"][FEATURE]


def test_no_spike_python_in_the_package() -> None:
    stray = sorted(p.name for p in package_modules() if "spike" in p.name)
    assert stray == [], f"spike modules inside the shipped package: {stray}"
