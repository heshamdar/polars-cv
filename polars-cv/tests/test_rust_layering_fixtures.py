"""Known-bad and known-good crate trees for ``tests/_rust_layering.py``.

Each case writes a small crate under ``tmp_path`` and checks the cycles it finds.
The bad cases must be reported; the good ones must not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._rust_layering import rust_cycles

pytestmark = pytest.mark.structural


def _crate(root: Path, files: dict[str, str]) -> tuple[dict[str, Path], list[Path]]:
    src = root / "src"
    written = []
    for rel, text in files.items():
        path = src / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return {"demo": src}, written


CASES = {
    "two_modules_use_each_other": (
        {
            "lib.rs": "",
            "a.rs": "use crate::b::X;\n",
            "b.rs": "use crate::a::Y;\n",
        },
        ["demo::a <-> demo::b"],
    ),
    "cross_branch_cycle_as_in_issue_192": (
        {
            "lib.rs": "",
            "kernels/mod.rs": "",
            "kernels/morph.rs": "use crate::ops::util::minimum;\n",
            "ops/mod.rs": "",
            "ops/util.rs": "use crate::kernels::morph::run;\n",
        },
        ["demo::kernels::morph <-> demo::ops::util"],
    ),
    "siblings_through_super": (
        {
            "lib.rs": "",
            "ops/mod.rs": "pub mod a;\npub mod b;\n",
            "ops/a.rs": "use super::b::X;\n",
            "ops/b.rs": "use super::a::Y;\n",
        },
        ["demo::ops::a <-> demo::ops::b"],
    ),
    "inline_path_without_a_use": (
        {
            "lib.rs": "",
            "a.rs": "pub fn f() { crate::b::run(); }\n",
            "b.rs": "use super::a::Z;\n",
        },
        ["demo::a <-> demo::b"],
    ),
    "one_way_dependency": (
        {
            "lib.rs": "",
            "a.rs": "use crate::b::X;\n",
            "b.rs": "pub struct X;\n",
        },
        [],
    ),
    "child_and_parent_name_each_other": (
        {
            "lib.rs": "",
            "ops/mod.rs": "pub mod child;\nuse crate::ops::child::X;\n",
            "ops/child.rs": "use crate::ops::Helper;\n",
        },
        [],
    ),
    "back_edge_only_in_cfg_test": (
        {
            "lib.rs": "",
            "a.rs": "use crate::b::X;\n",
            "b.rs": "#[cfg(test)]\nmod tests {\n    use crate::a::Y;\n}\n",
        },
        [],
    ),
    "back_edge_only_in_a_comment": (
        {
            "lib.rs": "",
            "a.rs": "// use crate::b::X;\n",
            "b.rs": "use crate::a::Y;\n",
        },
        [],
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_scanner_reports_the_expected_cycles(name: str, tmp_path: Path) -> None:
    files, expected = CASES[name]
    roots, written = _crate(tmp_path, files)
    assert rust_cycles(roots, written) == expected
