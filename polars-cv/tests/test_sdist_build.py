"""The sdist ``publish.yml`` builds passes ``scripts/check_sdist.py``.

``test_sdist_metadata.py`` pins the check with fixtures; this builds the real
sdist (``maturin sdist`` lists the crate's files through cargo) and checks
what PyPI checks: every ``License-File`` in ``PKG-INFO`` is in the archive.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import tarfile
import tomllib
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]


def _check_sdist():
    spec = importlib.util.spec_from_file_location(
        "check_sdist", _PKG / "scripts" / "check_sdist.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_built_sdist_carries_its_declared_licenses(tmp_path: Path) -> None:
    maturin = shutil.which("maturin")
    assert maturin is not None, "maturin is in the dev group; run `uv sync --group dev`"
    subprocess.run(
        [maturin, "sdist", "--out", str(tmp_path)],
        cwd=_PKG,
        check=True,
        capture_output=True,
    )
    version = tomllib.loads((_PKG / "pyproject.toml").read_text())["project"]["version"]
    sdist = tmp_path / f"polars_cv-{version}.tar.gz"
    assert _check_sdist().missing_license_files(sdist) == []
    with tarfile.open(sdist, "r:gz") as archive:
        member = archive.extractfile(f"polars_cv-{version}/PKG-INFO")
        assert member is not None
        pkg_info = member.read().decode()
    assert "License-File: LICENSE-MIT" in pkg_info
    assert "License-File: LICENSE-APACHE" in pkg_info
