"""The sdist PyPI receives carries every license file its metadata declares.

PyPI refuses an sdist whose ``License-File`` is not in the archive, and
``publish.yml`` uploads the wheels before the sdist, so the 0.38.0 publish
left its wheels on PyPI and its sdist rejected: maturin packed the
auto-discovered licenses under ``polars-cv/`` while ``PKG-INFO`` named them at
the top. ``scripts/check_sdist.py`` is the check; ``publish.yml`` runs it on
the sdist before any upload, and ``test_sdist_build.py`` builds one and runs
it there (it needs cargo, so it is not in this structural module).
"""

from __future__ import annotations

import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

from tests._discovery import repo_file

pytestmark = pytest.mark.structural

_PKG = Path(__file__).resolve().parents[1]


def _check_sdist():
    spec = importlib.util.spec_from_file_location(
        "check_sdist", _PKG / "scripts" / "check_sdist.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sdist(tmp_path: Path, files: dict[str, str]) -> Path:
    path = tmp_path / "pkg-1.0.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(f"pkg-1.0/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


_PKG_INFO = (
    "Metadata-Version: 2.4\nName: pkg\nVersion: 1.0\nLicense-File: LICENSE-MIT\n\n"
)


class TestCheckSdist:
    def test_refuses_a_license_packed_under_a_member_directory(
        self, tmp_path: Path
    ) -> None:
        # The 0.38.0 layout: declared at the top, packed under the member.
        sdist = _sdist(
            tmp_path, {"PKG-INFO": _PKG_INFO, "polars-cv/LICENSE-MIT": "MIT"}
        )
        assert _check_sdist().missing_license_files(sdist) == ["LICENSE-MIT"]
        assert _check_sdist().main([str(sdist)]) == 1

    def test_accepts_a_license_at_the_declared_path(self, tmp_path: Path) -> None:
        sdist = _sdist(tmp_path, {"PKG-INFO": _PKG_INFO, "LICENSE-MIT": "MIT"})
        assert _check_sdist().missing_license_files(sdist) == []
        assert _check_sdist().main([str(sdist)]) == 0

    def test_refuses_an_archive_without_pkg_info(self, tmp_path: Path) -> None:
        sdist = _sdist(tmp_path, {"LICENSE-MIT": "MIT"})
        with pytest.raises(ValueError, match="no pkg-1.0/PKG-INFO"):
            _check_sdist().missing_license_files(sdist)

    def test_publish_checks_the_sdist_before_uploading(self) -> None:
        workflow = repo_file(".github/workflows/publish.yml").read_text()
        build_sdist = workflow.split("  build-sdist:", 1)[1].split(
            "\n  publish-pypi:", 1
        )[0]
        build = build_sdist.index("maturin sdist")
        check = build_sdist.index("scripts/check_sdist.py")
        upload = build_sdist.index("actions/upload-artifact")
        assert build < check < upload
