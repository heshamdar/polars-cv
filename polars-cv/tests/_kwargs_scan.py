"""Which deserialized plugin structs accept fields they do not declare.

A struct that crosses the plugin boundary without
``#[serde(deny_unknown_fields)]`` silently discards any key Python sends that
Rust no longer reads. That is not hypothetical here: ``sink("jpeg",
qualtiy=50)`` encoded at the default quality with nothing said, and
``match_detections(strategy=)`` rode the wire unread for releases.

This is a **source scan**, the weakest of the three guard kinds `CLAUDE.md`
ranks, and it is used because neither of the stronger two can express the
property: serde offers no way to *require* the attribute at compile time, and a
runtime probe can only check structs someone remembered to list. The scan's
compensating virtue is that it covers a struct nobody remembered to add.

Two things keep it from rotting into a checker that matches nothing and reads
as green forever:

* the caller asserts a floor on how many structs were seen, so a regex that
  stops matching fails instead of passing empty, and
* ``tests/test_kwargs_scan_fixtures.py`` feeds it known-bad and known-good
  snippets, so the matching itself is pinned.

Enums are deliberately out of scope: serde already rejects an unknown variant,
so a ``Deserialize`` enum is closed without the attribute and has nothing to
declare. Only structs can silently swallow a key.

``test_plugin_kwargs_reject_unknown_fields`` covers the other half — that the
attribute, once present, actually rejects at the real entry point.
"""

from __future__ import annotations

import re

__all__ = ["OPEN_STRUCT_EXEMPT", "all_deserialized_structs", "open_structs"]

# Keyed ``file.rs::StructName``, never a bare name: `Probe` is generic enough
# that a bare-name exemption would silently cover an unrelated future struct
# that happens to reuse it, which is the kind of quiet blanket this module
# exists to prevent.
OPEN_STRUCT_EXEMPT = {
    # Its params ride on `#[serde(flatten)]`, which serde documents as
    # incompatible with `deny_unknown_fields`. The one documented, permanent
    # exception on the wire format — not a precedent, see `CLAUDE.md`.
    "pipeline.rs::OpSpec": "params ride on #[serde(flatten)]",
    # A deliberate *partial* parse of a Google ADC file: it reads only `type`
    # so that `service_account` files, whose bodies differ, do not trip a
    # full-schema parse. Closing it would break that by design. It also reads
    # a file on disk rather than a kwarg from Python, so it is not a
    # plugin-boundary struct at all.
    "cloud_auth.rs::Probe": "intentional partial parse of an ADC file",
}

_DESERIALIZED_STRUCT = re.compile(
    r"#\[derive\([^)]*\bDeserialize\b[^)]*\)\]\s*"  # the derive
    r"((?:#\[[^\]]*\]\s*)*)"  # any attributes between it and the struct
    r"(?:pub(?:\s*\([^)]*\))?\s+)?struct\s+(\w+)",  # pub, pub(crate), or private
)


def open_structs(source: str, filename: str) -> list[str]:
    """Return the names of deserialized structs in ``source`` that stay open.

    ``filename`` is the bare file name (``"pipeline.rs"``); it qualifies the
    exemption lookup so an exemption cannot leak to a same-named struct
    elsewhere. Ordering follows the source, so a caller can report the first
    offender without sorting.
    """
    found = []
    for match in _DESERIALIZED_STRUCT.finditer(source):
        attrs, name = match.group(1), match.group(2)
        if "deny_unknown_fields" in attrs:
            continue
        if f"{filename}::{name}" in OPEN_STRUCT_EXEMPT:
            continue
        found.append(name)
    return found


def all_deserialized_structs(source: str) -> list[str]:
    """Every deserialized struct in ``source``, open or closed.

    The caller uses this for the floor assertion: it counts what the regex
    *saw*, which is what distinguishes "nothing is open" from "nothing matched".
    """
    return [m.group(2) for m in _DESERIALIZED_STRUCT.finditer(source)]
