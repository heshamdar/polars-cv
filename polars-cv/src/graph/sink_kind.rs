//! What a `(domain, sink format)` pair produces — decided once.
//!
//! The sink contract has four halves, and each used to key on the
//! `(expected_domain, sink.format)` string pair for itself:
//!
//! | half | what it decides |
//! |------|-----------------|
//! | [`dtype_for_output`](super::decode::dtype_for_output) | the Polars dtype the planner publishes |
//! | [`encode_node_output`](super::encode::encode_node_output) | the value each row encodes to |
//! | [`null_row_result_for_spec`](super::decode::null_row_result_for_spec) | the typed null a skipped row pushes |
//! | [`build_series_from_spec`](super::decode::build_series_from_spec) | the Series the rows assemble into |
//!
//! Four copies of one table, and only the first two ended in an error. The
//! other two ended in `_ => Binary`, so a pair added to the first two and
//! forgotten in the last two published (say) `List(Float64)` at plan time and
//! produced an all-null **Binary** column at execution — with
//! `validate_output_schema` unable to see it, because it returns early for any
//! non-buffer output. That is not hypothetical: the `("vector", "array")` pair
//! was once added to the schema half without the encode half, and the comment
//! on `encode_node_output` records that "the pairs that did work did so by
//! coincidence of the two dispatches agreeing, not by construction."
//!
//! [`SinkKind`] is that table, resolved from the pair exactly once. The four
//! halves match on the *enum*, so a new kind is a compile error in all four at
//! once and there is no string pair left for a `_` arm to swallow. Adding a
//! `(domain, format)` pair means adding a [`SinkKind::resolve`] arm, and the
//! compiler then names every site that has to answer for it.

use polars::prelude::*;

use view_buffer::ImageCodec;

use super::types::OutputSpec;
use crate::formats::sink::Sink;
use view_buffer::ops::Domain;

/// The output shape a `(domain, format)` pair resolves to.
///
/// Deliberately finer-grained than the Polars dtype: `EncodedImage` and `Blob`
/// both produce `Binary`, but only the former carries a codec precondition the
/// planner has to check, and collapsing them would put that check back behind a
/// runtime `if`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SinkKind {
    /// `buffer` × `numpy`/`torch` — the zero-copy struct.
    NumpyStruct,
    /// `buffer` × `ndarray` — the same struct, tagged `polars_cv.ndarray`.
    ///
    /// Its own kind rather than a flag on `NumpyStruct` because it publishes a
    /// different dtype: every half has to answer for it, and the compiler says
    /// so. The rows and the assembled struct are `NumpyStruct`'s exactly; only
    /// the finished column is relabelled ([`crate::ext_types::ExtType::tag`]).
    NdArray,
    /// `buffer` × `png`/`jpeg`/`webp`/`tiff` — re-encoded through an image codec.
    EncodedImage,
    /// `buffer` × `blob` — the self-describing VIEW protocol, no precondition.
    Blob,
    /// `buffer` × `list` — nested `List`, one level per rank.
    BufferList,
    /// `buffer` × `array` — fixed-shape `Array`.
    BufferArray,
    /// `scalar` × `native` — `Float64`.
    Scalar,
    /// `vector` × `native`/`list` — nested `List`.
    VectorList,
    /// `vector` × `array` — fixed-shape `Array`, e.g. a perceptual hash.
    VectorArray,
    /// `contour` × `native` — `List(Struct)` matching `CONTOUR_SCHEMA`.
    Contours,
    /// Any domain whose spec asks for the histogram-bucket encoding.
    ///
    /// Encoding outranks the pair: histogram buckets are a `vector`-domain
    /// output with their own struct schema. All four halves used to carry an
    /// early return for this; now it is one more kind.
    HistogramBuckets,
}

impl SinkKind {
    /// Resolve the pair, rejecting anything the table does not name.
    ///
    /// The one place a `(domain, format)` string is interpreted. Its error is
    /// the *only* answer for an unknown pair — there is no fallback kind,
    /// because a fallback is exactly what let a mismatched pair reach execution
    /// disguised as a Binary column.
    pub(crate) fn resolve(spec: &OutputSpec) -> PolarsResult<Self> {
        if spec.histogram_buckets {
            return Ok(Self::HistogramBuckets);
        }
        match (spec.expected_domain, &spec.sink) {
            (Domain::Buffer, Sink::Numpy(_) | Sink::Torch(_)) => Ok(Self::NumpyStruct),
            (Domain::Buffer, Sink::NdArray(_)) => Ok(Self::NdArray),
            (Domain::Buffer, Sink::Png(_) | Sink::Jpeg(_) | Sink::WebP(_) | Sink::Tiff(_)) => {
                Ok(Self::EncodedImage)
            }
            (Domain::Buffer, Sink::Blob(_)) => Ok(Self::Blob),
            (Domain::Buffer, Sink::List(_)) => Ok(Self::BufferList),
            (Domain::Buffer, Sink::Array(_)) => Ok(Self::BufferArray),
            (Domain::Scalar, Sink::Native(_)) => Ok(Self::Scalar),
            (Domain::Vector, Sink::Native(_) | Sink::List(_)) => Ok(Self::VectorList),
            (Domain::Vector, Sink::Array(_)) => Ok(Self::VectorArray),
            (Domain::Contour, Sink::Native(_)) => Ok(Self::Contours),
            // Named separately from the catch-all so the message can say what
            // to do instead; the generic one cannot.
            (Domain::Buffer, Sink::Native(_)) => polars_bail!(ComputeError:
                "'native' sink is not defined for buffer outputs; use an explicit \
                 format (numpy, png, list, array, blob, ...)"
            ),
            (domain, sink) => polars_bail!(ComputeError:
                "Unsupported output combination: domain '{}' with sink format '{}'",
                domain.name(), sink.name()
            ),
        }
    }

    /// The image codec, for the kind that encodes through one.
    ///
    /// Lets the planner's codec precondition read the codec off the kind
    /// rather than re-testing the sink it was resolved from.
    pub(crate) fn image_codec(self, spec: &OutputSpec) -> Option<ImageCodec> {
        matches!(self, Self::EncodedImage)
            .then(|| spec.sink.image_codec())
            .flatten()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    fn spec(domain: &str, format: &str) -> OutputSpec {
        OutputSpec {
            node: "n".to_string(),
            sink: serde_json::from_value(serde_json::json!({"format": format})).unwrap(),
            expected_domain: view_buffer::naming::lookup(Domain::NAMED, domain).unwrap(),
            expected_dtype: view_buffer::PlannedDType::Known(view_buffer::DType::U8),
            expected_shape: None,
            shape_asserted: false,
            expected_ndim: None,
            histogram_buckets: false,
        }
    }

    /// Every pair the table names, and the kind it must resolve to.
    ///
    /// Written out rather than derived: this is the correspondence itself, and
    /// a test that re-derived it from `resolve` would agree with any answer.
    const PAIRS: &[(&str, &str, SinkKind)] = &[
        ("buffer", "numpy", SinkKind::NumpyStruct),
        ("buffer", "torch", SinkKind::NumpyStruct),
        ("buffer", "ndarray", SinkKind::NdArray),
        ("buffer", "png", SinkKind::EncodedImage),
        ("buffer", "jpeg", SinkKind::EncodedImage),
        ("buffer", "webp", SinkKind::EncodedImage),
        ("buffer", "tiff", SinkKind::EncodedImage),
        ("buffer", "blob", SinkKind::Blob),
        ("buffer", "list", SinkKind::BufferList),
        ("buffer", "array", SinkKind::BufferArray),
        ("scalar", "native", SinkKind::Scalar),
        ("vector", "native", SinkKind::VectorList),
        ("vector", "list", SinkKind::VectorList),
        ("vector", "array", SinkKind::VectorArray),
        ("contour", "native", SinkKind::Contours),
    ];

    /// A spec each kind can actually build a column from (the array kinds
    /// need a shape, the list kinds a rank).
    fn buildable_spec(domain: &str, format: &str) -> OutputSpec {
        let mut s = spec(domain, format);
        if let Sink::Array(array) = &mut s.sink {
            array.shape = Some(vec![crate::ops::Literal(1)]);
        }
        s.expected_ndim = Some(1);
        s
    }

    /// A row of the wrong kind is an error, never a null (CR-38).
    ///
    /// `encode_node_output` and `SinkKind` are two halves of one contract; if
    /// they ever disagree, `build_series_from_spec` used to publish the row as
    /// null through a `_ => None` arm, which reads as data rather than a bug.
    #[test]
    fn a_row_of_another_kind_is_an_error_not_a_null() {
        use crate::graph::decode::{build_series_from_spec, null_row_result_for_spec};
        use crate::graph::types::RowResult;
        for &(domain, format, kind) in PAIRS {
            let spec = buildable_spec(domain, format);
            let foreign = if kind == SinkKind::Scalar {
                RowResult::Binary(Some(vec![1]))
            } else {
                RowResult::Scalar(Some(1.0))
            };
            let result = build_series_from_spec("o".into(), &spec, vec![foreign]);
            assert!(
                result.is_err(),
                "({domain}, {format}) published a foreign row instead of failing"
            );
            // The kind's own null row is still a null, not an error.
            let null = null_row_result_for_spec(&spec).unwrap();
            let series = build_series_from_spec("o".into(), &spec, vec![null])
                .unwrap_or_else(|e| panic!("({domain}, {format}) null row failed: {e}"));
            assert_eq!(series.null_count(), 1, "({domain}, {format})");
        }
    }

    #[test]
    fn every_named_pair_resolves_to_its_kind() {
        for &(domain, format, expected) in PAIRS {
            let resolved = SinkKind::resolve(&spec(domain, format))
                .unwrap_or_else(|e| panic!("({domain}, {format}) failed to resolve: {e}"));
            assert_eq!(resolved, expected, "({domain}, {format})");
        }
    }

    /// The name of a kind.
    ///
    /// Exhaustive, so adding a `SinkKind` fails to compile here. On its own
    /// that only forces *this match* to grow, which is the gap
    /// `every_kind_is_produced_by_some_pair` closes: it reads these arms back
    /// out of the source, so acknowledging a kind and proving it reachable are
    /// the same act. A hand-written `[SinkKind; N]` array beside this match was
    /// the first spelling, and a new variant escaped it silently.
    fn kind_name(kind: SinkKind) -> &'static str {
        match kind {
            SinkKind::NumpyStruct => "NumpyStruct",
            SinkKind::NdArray => "NdArray",
            SinkKind::EncodedImage => "EncodedImage",
            SinkKind::Blob => "Blob",
            SinkKind::BufferList => "BufferList",
            SinkKind::BufferArray => "BufferArray",
            SinkKind::Scalar => "Scalar",
            SinkKind::VectorList => "VectorList",
            SinkKind::VectorArray => "VectorArray",
            SinkKind::Contours => "Contours",
            SinkKind::HistogramBuckets => "HistogramBuckets",
        }
    }

    /// The kinds `kind_name` acknowledges, parsed from this file.
    ///
    /// Rust cannot enumerate an enum's variants without a derive or a second
    /// list, and a second list is exactly what this replaces. The parse asserts
    /// it found a plausible match rather than silently matching nothing — the
    /// failure mode a source scan has to be protected from. Same shape as
    /// `acknowledged_variants` in `view-buffer/tests/apply_op_coverage.rs`.
    fn acknowledged_kinds() -> Vec<String> {
        let src = include_str!("sink_kind.rs");
        let body = src
            .split("fn kind_name(kind: SinkKind) -> &'static str {")
            .nth(1)
            .expect("kind_name's definition moved — this scan reads nothing");
        let body = body
            .split("\n    }")
            .next()
            .expect("kind_name's body has no closing brace");
        let names: Vec<String> = body
            .lines()
            .filter_map(|line| line.trim().strip_prefix("SinkKind::"))
            .filter_map(|rest| rest.split(' ').next())
            .map(str::to_string)
            .collect();
        assert!(
            names.len() >= 11,
            "parsed {} arms from kind_name; the scan is out of date",
            names.len()
        );
        names
    }

    /// Every kind is reachable from some pair.
    ///
    /// A kind no pair produces is dead vocabulary that still has to be answered
    /// for in four matches. Driven from the acknowledgment match rather than a
    /// sibling array, so a variant cannot be acknowledged without being shown
    /// constructible.
    #[test]
    fn every_kind_is_produced_by_some_pair() {
        let mut produced: BTreeSet<&str> = PAIRS.iter().map(|&(_, _, k)| kind_name(k)).collect();
        let mut buckets = spec("vector", "list");
        buckets.histogram_buckets = true;
        produced.insert(kind_name(SinkKind::resolve(&buckets).unwrap()));

        let missing: Vec<String> = acknowledged_kinds()
            .into_iter()
            .filter(|name| !produced.contains(name.as_str()))
            .collect();
        assert!(
            missing.is_empty(),
            "these SinkKinds are named by no (domain, format) pair — dead \
             vocabulary that four matches still have to answer for: {missing:?}"
        );
    }

    #[test]
    fn the_encoding_outranks_the_pair() {
        let mut s = spec("buffer", "numpy");
        s.histogram_buckets = true;
        assert_eq!(SinkKind::resolve(&s).unwrap(), SinkKind::HistogramBuckets);
    }

    #[test]
    fn an_unknown_pair_is_rejected_not_defaulted() {
        // The failure this whole module exists to prevent: a pair nothing
        // names used to arrive as a Binary column of nulls.
        let err = SinkKind::resolve(&spec("contour", "png")).unwrap_err();
        assert!(
            err.to_string().contains("Unsupported output combination"),
            "unexpected error: {err}"
        );
        let err = SinkKind::resolve(&spec("scalar", "list")).unwrap_err();
        assert!(err.to_string().contains("Unsupported output combination"));
    }

    #[test]
    fn a_buffer_native_sink_says_what_to_use_instead() {
        let err = SinkKind::resolve(&spec("buffer", "native")).unwrap_err();
        assert!(
            err.to_string().contains("use an explicit"),
            "unexpected error: {err}"
        );
    }

    /// `"binary"` was carried as a dead arm in two of the four halves.
    ///
    /// No Python path emits it, `SinkFormat` has no such member, and the schema
    /// half never had an arm for it — so a graph carrying it was already
    /// rejected before those arms could run. It must stay rejected: now by the
    /// typed sink itself, before any kind is resolved.
    #[test]
    fn the_dead_binary_format_stays_rejected() {
        let err = serde_json::from_value::<Sink>(serde_json::json!({"format": "binary"}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown sink format 'binary'"), "{err}");
    }
}
