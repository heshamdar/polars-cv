"""Refuse an sdist whose ``License-File`` entries are not in the archive.

PyPI rejects such an sdist at upload ("License-File ... does not exist in
distribution file"), and ``publish.yml`` uploads the wheels first, so the
release is left half-published: wheels on PyPI under a version whose sdist
can never follow. Run on the built sdist before anything is uploaded::

    python scripts/check_sdist.py dist/polars_cv-*.tar.gz

Each ``License-File`` path is relative to the archive's top directory
(``polars_cv-<version>/``), which is where PyPI looks for it. A maturin that
auto-discovers the license files of a workspace member records them by name
but packs them under the member's directory (``polars-cv/LICENSE-MIT``);
``pyproject.toml``'s ``license-files`` places them at the top.
"""

from __future__ import annotations

import sys
import tarfile
from email.parser import HeaderParser
from pathlib import Path


def missing_license_files(sdist: Path) -> list[str]:
    """The ``License-File`` entries of ``sdist``'s ``PKG-INFO`` it does not contain."""
    with tarfile.open(sdist, "r:gz") as archive:
        names = set(archive.getnames())
        tops = {name.split("/", 1)[0] for name in names}
        if len(tops) != 1:
            msg = (
                f"{sdist.name}: expected one top-level directory, found {sorted(tops)}"
            )
            raise ValueError(msg)
        (top,) = tops
        member = (
            archive.extractfile(f"{top}/PKG-INFO")
            if f"{top}/PKG-INFO" in names
            else None
        )
        if member is None:
            msg = f"{sdist.name}: no {top}/PKG-INFO"
            raise ValueError(msg)
        metadata = HeaderParser().parsestr(member.read().decode())
    declared = metadata.get_all("License-File") or []
    return [path for path in declared if f"{top}/{path}" not in names]


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: check_sdist.py SDIST [SDIST ...]", file=sys.stderr)
        return 2
    failed = False
    for arg in argv:
        missing = missing_license_files(Path(arg))
        for path in missing:
            print(f"{arg}: License-File {path} is not in the archive", file=sys.stderr)
        failed |= bool(missing)
        if not missing:
            print(f"{arg}: every License-File is in the archive")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
