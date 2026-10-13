"""Every Python block in the README runs, and its images render off GitHub.

The README is the first thing a new user runs, and it is not built by MkDocs
(``--strict`` only checks ``docs/``), so nothing else would notice a snippet
calling a renamed method. Each fenced ``python`` block is executed on its own,
in a fresh namespace, as a user pasting it would: a block may not lean on a
name an earlier block defined.

The blocks run in a scratch directory holding what the README tells the reader
to bring: a ``photos/`` folder of JPEGs (the README's example globs it). Nothing
else is provided, so a block that needs more must create it itself.

The README is also PyPI's ``long_description``, where relative links do not
resolve, so every image must be an absolute ``https://`` URL.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

from .conftest import plugin_required

README = Path(__file__).resolve().parents[1] / "README.md"

#: A fenced block opened with ``python`` (or ``py``) as its info string. The
#: closing fence must use the same fence characters at the same indent, which
#: is how GitHub and PyPI read it; an indented fence inside ``<details>`` or a
#: list item counts too.
_FENCE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>[^\n`]*)\n"
    r"(?P<body>.*?)"
    r"^(?P=indent)(?P=fence)[ \t]*$",
    re.MULTILINE | re.DOTALL,
)

#: Markdown images ``![alt](url)`` and HTML ``<img src="url">``.
_IMAGE = re.compile(
    r"!\[[^\]]*\]\(\s*(?P<md>[^)\s]+)|<img\b[^>]*\bsrc=\"(?P<html>[^\"]+)\""
)


def python_blocks(markdown: str) -> list[tuple[int, str]]:
    """Return ``(line, source)`` for each fenced Python block, in order."""
    blocks = []
    for m in _FENCE.finditer(markdown):
        lang = m.group("info").strip().split(maxsplit=1)
        if not lang or lang[0] not in ("python", "py"):
            continue
        indent = len(m.group("indent"))
        body = "\n".join(line[indent:] for line in m.group("body").splitlines())
        blocks.append((markdown.count("\n", 0, m.start()) + 1, body))
    return blocks


def image_urls(markdown: str) -> list[str]:
    """Every image URL the markdown embeds."""
    return [m.group("md") or m.group("html") for m in _IMAGE.finditer(markdown)]


# --- The extractor's own fixtures: it must find blocks and nothing else. ----

_GOOD = """\
Intro.

```python
x = 1
```

```bash
pip install polars-cv
```

<details>
<summary>More</summary>

  ```py
  y = 2
  ```

</details>

````python
z = '```'
````
"""


@pytest.mark.structural
def test_extractor_finds_python_blocks_only() -> None:
    blocks = [body for _, body in python_blocks(_GOOD)]
    assert blocks == ["x = 1", "y = 2", "z = '```'"]


@pytest.mark.structural
def test_extractor_reports_the_opening_line() -> None:
    assert [line for line, _ in python_blocks(_GOOD)] == [3, 14, 20]


@pytest.mark.structural
def test_image_urls_sees_markdown_and_html() -> None:
    md = '![a](https://x/a.png) and <img alt="b" src="docs/b.svg" width="5">'
    assert image_urls(md) == ["https://x/a.png", "docs/b.svg"]


# --- The README itself. ------------------------------------------------------

_README_TEXT = README.read_text(encoding="utf-8")
_BLOCKS = python_blocks(_README_TEXT)


@pytest.mark.structural
def test_readme_has_python_blocks() -> None:
    """The sweep below must have something to sweep.

    Without this, a change to the fence syntax that the extractor misses would
    leave the parametrized test with no cases: green, and testing nothing.
    """
    assert len(_BLOCKS) >= 4, f"found only {len(_BLOCKS)} python blocks in README"


@pytest.mark.structural
def test_readme_images_are_absolute() -> None:
    """PyPI renders the same file, and resolves no relative path."""
    urls = image_urls(_README_TEXT)
    assert urls, "the README embeds no image"
    relative = [u for u in urls if not u.startswith("https://")]
    assert not relative, f"relative image links break on PyPI: {relative}"


def _write_photos(directory: Path) -> None:
    """A few JPEGs of different sizes and modes, as a user's folder would hold."""
    from PIL import Image

    directory.mkdir()
    sizes = [(640, 480), (480, 640), (800, 600), (320, 240)]
    for i, (w, h) in enumerate(sizes):
        img = Image.radial_gradient("L").resize((w, h)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        (directory / f"img_{i}.jpg").write_bytes(buf.getvalue())


@plugin_required
@pytest.mark.parametrize(
    ("line", "source"), _BLOCKS, ids=[f"README.md:{line}" for line, _ in _BLOCKS]
)
def test_readme_block_runs(
    line: int, source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_photos(tmp_path / "photos")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    # Padded so a traceback names the README's own line numbers.
    code = compile("\n" * line + source, str(README), "exec")
    exec(code, {"__name__": "__readme__"})
