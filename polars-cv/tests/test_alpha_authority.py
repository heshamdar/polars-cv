"""The alpha rule has one authority: ``view_buffer::ops::color::has_alpha``.

A 2-channel image is gray + alpha and a 4-channel one RGBA (decision D5). The
resamplers' premultiply, the shape rule's carried alpha, grayscale's gray
channel and the colour conversions read that one declaration
(``has_alpha``/``color_channels``). Each used to test the channel count itself
(``matches!(c, 2 | 4)``, ``C == 2``), and fast_image_resize premultiplied by
its own pixel-type default.

This is a source scan (the property is "nothing else restates the rule",
which neither the compiler nor a runtime check can express). Its limits: it
recognises a channel-count comparison written against a channel-ish name
(``c``, ``C``, ``ch``, ``channels``, ...) or a ``matches!(x, 2 | 4)``; a count
compared under another name is not seen. The fixtures below pin what it must
and must not report.
"""

from __future__ import annotations

import re

import pytest

from tests._discovery import REPO_ROOT as ROOT
from tests._discovery import requires_checkout, rust_sources

pytestmark = pytest.mark.structural

AUTHORITY = ROOT / "view-buffer/src/ops/color.rs"

_CHANNEL_NAME = r"(?:c|C|ch|channels|num_channels|n_channels|channel_count)"
RESTATEMENT = re.compile(
    rf"\b{_CHANNEL_NAME}\s*==\s*[24]\b"
    r"|matches!\(\s*[\w.()]+\s*,\s*2\s*\|\s*4\s*\)"
    r"|\b2\s*\|\s*4\s*=>"
)


def restatements(source: str) -> list[str]:
    """Lines of Rust *source* that test for alpha by channel count, outside
    comments."""
    hits = []
    for line in source.splitlines():
        code = line.split("//", 1)[0]
        if RESTATEMENT.search(code):
            hits.append(line.strip())
    return hits


@pytest.mark.parametrize(
    "snippet",
    [
        "let alpha = matches!(c, 2 | 4);",
        "let alpha = |c: usize| usize::from(matches!(c, 2 | 4));",
        "d.write(if C == 2 { p[0] } else { luma(p) });",
        "if channels == 4 { premultiply(&mut px) }",
        "match c { 2 | 4 => alpha(), _ => plain() }",
    ],
)
def test_a_restated_alpha_rule_is_reported(snippet: str) -> None:
    assert restatements(snippet) == [snippet]


@pytest.mark.parametrize(
    "snippet",
    [
        "let alpha = crate::ops::color::has_alpha(c);",
        "if color_channels(C) == 1 { p[0] }",
        "shape.len() == 2 || shape.len() == 3",
        "let mask = flags | 4;",
        "// a 4-channel image: C == 4 is RGBA",
        "(DType::U8, 4) => resize_pixels::<U8x4>(buf, w, h, filter),",
    ],
)
def test_other_code_is_not_reported(snippet: str) -> None:
    assert restatements(snippet) == []


@requires_checkout
def test_only_the_declaration_tests_for_alpha() -> None:
    crates = ("view-buffer/src", "polars-cv/src")
    sources = [
        p
        for p in rust_sources()
        if p != AUTHORITY and str(p.relative_to(ROOT)).startswith(crates)
    ]
    assert len(sources) > 50, "the scan found too few Rust files to mean anything"
    found = {
        str(p.relative_to(ROOT)): hits
        for p in sources
        if (hits := restatements(p.read_text()))
    }
    assert not found, (
        "these test for alpha by channel count; read "
        f"`ops::color::has_alpha`/`color_channels` instead: {found}"
    )
    assert restatements(AUTHORITY.read_text()), "the declaration itself was not seen"
