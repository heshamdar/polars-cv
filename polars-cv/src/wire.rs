//! The one parser of JSON the Python builder wrote: a graph, a plan, an op, a
//! source, a geometry call's arguments.
//!
//! `serde_json::from_str` ends a failure's message with `at line 1 column
//! 203`. That is a position in text a machine wrote and the user never sees,
//! printed after a diagnosis that is already complete, so [`from_str`] leaves
//! it out. Clippy refuses `serde_json::from_str` anywhere else in the
//! workspace (`clippy.toml`), so a new parse site cannot bring the position
//! back.

/// `json` parsed as a `T`; an error is the diagnosis without the position
/// `serde_json` appends to it.
pub(crate) fn from_str<T: serde::de::DeserializeOwned>(json: &str) -> Result<T, String> {
    #[expect(
        clippy::disallowed_methods,
        reason = "the one call the rest of the crate goes through"
    )]
    serde_json::from_str(json).map_err(|e| message(&e))
}

/// A `serde_json` error's text without its trailing ` at line L column C`.
fn message(e: &serde_json::Error) -> String {
    let text = e.to_string();
    // `Display` appends the position exactly when the line is non-zero.
    if e.line() == 0 {
        return text;
    }
    let position = format!(" at line {} column {}", e.line(), e.column());
    match text.strip_suffix(&position) {
        Some(diagnosis) => diagnosis.to_string(),
        None => text,
    }
}

#[cfg(test)]
mod tests {
    use super::from_str;

    #[derive(Debug, serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    #[allow(dead_code)]
    struct Probe {
        quality: u8,
    }

    #[test]
    fn a_refusal_is_the_diagnosis_without_a_position() {
        // A data error (the field), a syntax error and an early end: every
        // category `serde_json` positions.
        for (json, says) in [
            (r#"{"quality": 1, "speed": 2}"#, "unknown field `speed`"),
            (r#"{"quality": "high"}"#, "invalid type"),
            (r#"{"quality" 1}"#, "expected `:`"),
            (r#"{"quality": 1"#, "EOF"),
        ] {
            let err = from_str::<Probe>(json).unwrap_err();
            assert!(err.contains(says), "{err}");
            assert!(!err.contains(" at line "), "{err}");
            assert!(!err.contains("column"), "{err}");
        }
    }

    #[test]
    fn a_message_that_itself_ends_in_a_position_keeps_its_own_words() {
        // Only the suffix `serde_json` added is removed: a diagnosis that
        // quotes the same words earlier is left alone.
        let err = from_str::<Probe>(r#"{"at line 1 column 5": 1}"#).unwrap_err();
        assert!(err.contains("unknown field `at line 1 column 5`"), "{err}");
        assert!(!err.ends_with("column 24"), "{err}");
    }
}
