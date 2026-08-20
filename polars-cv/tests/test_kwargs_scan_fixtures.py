"""Fixtures for the open-struct scan in :mod:`tests._kwargs_scan`.

The scan is a regex over Rust source, so its failure mode is silence: match
nothing and it reports nothing open. These snippets pin both directions — what
it must flag and what it must not — so a change to the regex that stops it
matching shows up here rather than as a permanently green checker.
"""

from __future__ import annotations

import pytest

from tests._kwargs_scan import all_deserialized_structs, open_structs

# --- known-bad: the scan must report each of these -------------------------

_MUST_FLAG = {
    "plain open struct": """
        #[derive(Debug, Deserialize)]
        pub struct Kwargs { pub a: Option<f64> }
    """,
    "open with an unrelated attribute between": """
        #[derive(Debug, Deserialize)]
        #[serde(rename_all = "snake_case")]
        pub struct Kwargs { pub a: Option<f64> }
    """,
    "open with a doc comment in the derive list": """
        #[derive(Clone, Debug, Default, Deserialize, Serialize)]
        pub struct Kwargs { pub a: Option<f64> }
    """,
    "private struct, still crosses serde": """
        #[derive(Deserialize)]
        struct Kwargs { a: Option<f64> }
    """,
    "pub(crate) struct": """
        #[derive(Deserialize)]
        pub(crate) struct Kwargs { a: Option<f64> }
    """,
}

# --- known-good: the scan must stay silent on each of these -----------------

_MUST_NOT_FLAG = {
    "closed directly after the derive": """
        #[derive(Debug, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct Kwargs { pub a: Option<f64> }
    """,
    "closed alongside another serde attribute": """
        #[derive(Debug, Deserialize)]
        #[serde(rename_all = "snake_case", deny_unknown_fields)]
        pub struct Kwargs { pub a: Option<f64> }
    """,
    "closed with attributes on either side": """
        #[derive(Debug, Deserialize)]
        #[serde(deny_unknown_fields)]
        #[non_exhaustive]
        pub struct Kwargs { pub a: Option<f64> }
    """,
    # keyed to its real file — see OPEN_STRUCT_EXEMPT
    "the documented OpSpec exemption": """
        #[derive(Debug, Deserialize)]
        pub struct OpSpec { #[serde(flatten)] pub params: Map }
    """,
    "serialize-only struct never crosses inbound": """
        #[derive(Debug, Serialize)]
        pub struct Report { pub a: f64 }
    """,
}


# Cases whose exemption is keyed to a specific file. Anything absent uses a
# neutral name, so an exemption that leaked across files would show up as a
# known-bad case silently passing.
_FIXTURE_FILE = {"the documented OpSpec exemption": "pipeline.rs"}


def test_an_exemption_does_not_leak_to_another_file() -> None:
    """`OpSpec` is exempt in pipeline.rs only, not wherever the name appears.

    Bare-name exemptions are how a blanket quietly widens: an unrelated struct
    reusing a generic name inherits a pass nobody granted it.
    """
    snippet = _MUST_NOT_FLAG["the documented OpSpec exemption"]
    assert open_structs(snippet, "pipeline.rs") == []
    assert open_structs(snippet, "somewhere_else.rs") == ["OpSpec"]


@pytest.mark.parametrize("label", sorted(_MUST_FLAG))
def test_the_scan_flags_an_open_struct(label: str) -> None:
    assert open_structs(_MUST_FLAG[label], "fixture.rs"), (
        f"the scan missed a struct that accepts undeclared fields: {label!r}"
    )


@pytest.mark.parametrize("label", sorted(_MUST_NOT_FLAG))
def test_the_scan_stays_silent_on_a_closed_struct(label: str) -> None:
    assert not open_structs(
        _MUST_NOT_FLAG[label], _FIXTURE_FILE.get(label, "fixture.rs")
    ), f"the scan flagged a struct that is already closed: {label!r}"


def test_the_floor_helper_counts_closed_structs_too() -> None:
    """The floor counts what the regex *saw*, not what it flagged.

    Without this the floor would fall as structs get closed, and reach zero
    exactly when every struct is correct — turning the anti-vacuity check into
    a check that fires only while the bug is present.
    """
    closed = _MUST_NOT_FLAG["closed directly after the derive"]
    assert all_deserialized_structs(closed) == ["Kwargs"]
    assert open_structs(closed, "fixture.rs") == []
