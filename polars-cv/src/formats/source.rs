//! Source formats: how a node's input column is decoded.

use std::collections::HashMap;

use polars::prelude::*;
use polars_cv_macros::Op;
use serde::{Deserialize, Serialize};
use view_buffer::DType;

use super::formats;
use crate::fetch::FetchErrorPolicy;
use crate::ops::geometry::RasterSize;
use crate::ops::{Literal, Param};
use crate::params::ParamCtx;

/// Infer the decode path from the column's Polars dtype: String → file_path,
/// List/Array → list/array, Binary → blob if VIEW-tagged else image_bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct AutoSource {
    /// Asserted element dtype (a decoded image is cast to it).
    pub dtype: Option<Literal<DType>>,
    /// Require rectangular data when the column is a List/Array.
    pub require_contiguous: Option<Literal<bool>>,
    /// Cloud-storage credentials when the column is a path.
    pub cloud_options: Option<HashMap<String, String>>,
    /// Locations a path column may read from (unrestricted when absent).
    pub allowed_roots: Option<Vec<String>>,
    /// Decode only enough pixels for this long side (JPEG IDCT scaling).
    pub decode_max_size: Option<Literal<u32>>,
    /// "raise" (default) or "null": what a row that cannot be decoded does.
    pub on_error: Option<Literal<FetchErrorPolicy>>,
}

/// Encoded image bytes (PNG/JPEG/TIFF/...), always decoded to `[H, W, C]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct ImageBytesSource {
    /// Asserted element dtype: a decoded image with another dtype is cast.
    pub dtype: Option<Literal<DType>>,
    /// Decode only enough pixels for this long side (JPEG IDCT scaling).
    pub decode_max_size: Option<Literal<u32>>,
    /// "raise" (default) or "null": what a row that cannot be decoded does.
    pub on_error: Option<Literal<FetchErrorPolicy>>,
}

/// A path (local, s3://, gs://, az://, http://) whose contents decode like
/// image bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct FilePathSource {
    /// Asserted element dtype: a decoded image with another dtype is cast.
    pub dtype: Option<Literal<DType>>,
    /// Cloud-storage credentials (see `cloud::CloudOptions::from_map`).
    pub cloud_options: Option<HashMap<String, String>>,
    /// Locations the path column may read from (unrestricted when absent).
    pub allowed_roots: Option<Vec<String>>,
    /// Decode only enough pixels for this long side (JPEG IDCT scaling).
    pub decode_max_size: Option<Literal<u32>>,
    /// "raise" (default) or "null": what a row that cannot be read or decoded
    /// does.
    pub on_error: Option<Literal<FetchErrorPolicy>>,
}

/// The self-describing VIEW protocol.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct BlobSource {
    /// Declared element dtype. A blob carries its own, so a declaration is
    /// checked at decode: a blob of another dtype is a row error.
    pub dtype: Option<Literal<DType>>,
    /// "raise" (default) or "null": what a row that cannot be decoded does.
    pub on_error: Option<Literal<FetchErrorPolicy>>,
}

/// Raw bytes, decoded as a flat 1-D buffer of `dtype`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct RawSource {
    /// The element dtype: raw bytes carry no type metadata, so it is required.
    pub dtype: Literal<DType>,
    /// "raise" (default) or "null": what a row that cannot be decoded does.
    pub on_error: Option<Literal<FetchErrorPolicy>>,
}

/// A Polars nested `List` or fixed-size `Array` column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct NestedSource {
    /// Element dtype; inferred from the column when absent.
    pub dtype: Option<Literal<DType>>,
    /// Require rectangular data (zero-copy); jagged rows are then an error.
    pub require_contiguous: Option<Literal<bool>>,
    /// "raise" (default) or "null": what a row that cannot be decoded does.
    pub on_error: Option<Literal<FetchErrorPolicy>>,
}

/// Contour geometry, rasterized to an `[H, W, 1]` u8 mask — the same contract
/// the `rasterize` op publishes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct ContourSource {
    /// ``[height, width]`` of the mask (each may be a Polars expression), or
    /// the node whose buffer's height and width the mask takes.
    pub size: RasterSize,
    /// Inside value (default 255). Accepts a Polars expression for per-row
    /// values.
    pub fill_value: Option<Param<u8>>,
    /// Outside value (default 0). Accepts a Polars expression for per-row
    /// values.
    pub background: Option<Param<u8>>,
    /// "raise" (default) or "null": what a row that cannot be decoded does.
    pub on_error: Option<Literal<FetchErrorPolicy>>,
}

impl ContourSource {
    /// `(fill_value, background)` at `row`; absent is 255 and 0.
    pub fn fill(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<(u8, u8)> {
        let at =
            |p: &Option<Param<u8>>, absent| p.as_ref().map_or(Ok(absent), |p| p.resolve(row, ctx));
        Ok((at(&self.fill_value, 255)?, at(&self.background, 0)?))
    }
}

formats! {
    /// How a node's input column is decoded (see the module docs).
    Source("source") {
        "array" => Array(NestedSource) {"require_contiguous": true},
        "auto" => Auto(AutoSource) {"allowed_roots": ["/srv"], "decode_max_size": 64},
        "blob" => Blob(BlobSource) {},
        "contour" => Contour(ContourSource)
            {"size": [8, 6], "fill_value": {"$slot": 1}, "background": 0, "on_error": "null"},
        "file_path" => FilePath(FilePathSource)
            {"cloud_options": {"aws_region": "eu-west-1"}, "dtype": "u16"},
        "image_bytes" => ImageBytes(ImageBytesSource) {"decode_max_size": 64},
        "list" => List(NestedSource) {"dtype": "f32"},
        "raw" => Raw(RawSource) {"dtype": "u8"},
    }
}

impl Source {
    /// Whether the element dtype and rank are resolved from the input column's
    /// Polars type when the query is planned with its input (a `list`/`array`
    /// column, or `auto` routing to one), rather than at build time.
    pub fn resolves_from_column(&self) -> bool {
        match self {
            Source::List(_) | Source::Array(_) | Source::Auto(_) => true,
            Source::Blob(_)
            | Source::Contour(_)
            | Source::FilePath(_)
            | Source::ImageBytes(_)
            | Source::Raw(_) => false,
        }
    }

    /// Whether a row that cannot be decoded is nulled rather than failing the
    /// query.
    pub fn nulls_on_error(&self) -> bool {
        let on_error = match self {
            Source::Auto(s) => s.on_error,
            Source::ImageBytes(s) => s.on_error,
            Source::FilePath(s) => s.on_error,
            Source::Blob(s) => s.on_error,
            Source::Raw(s) => s.on_error,
            Source::List(s) | Source::Array(s) => s.on_error,
            Source::Contour(s) => s.on_error,
        };
        on_error.is_some_and(|p| p.get().nulls_the_row())
    }

    /// The declared element dtype, if any.
    pub fn dtype(&self) -> Option<DType> {
        let dtype = match self {
            Source::Auto(s) => s.dtype,
            Source::ImageBytes(s) => s.dtype,
            Source::FilePath(s) => s.dtype,
            Source::Blob(s) => s.dtype,
            Source::Raw(s) => Some(s.dtype),
            Source::List(s) | Source::Array(s) => s.dtype,
            Source::Contour(_) => None,
        };
        dtype.map(|d| d.get())
    }

    /// Whether a nested-column decode must be zero-copy.
    pub fn require_contiguous(&self) -> bool {
        let flag = match self {
            Source::Auto(s) => s.require_contiguous,
            Source::List(s) | Source::Array(s) => s.require_contiguous,
            Source::ImageBytes(_)
            | Source::FilePath(_)
            | Source::Blob(_)
            | Source::Raw(_)
            | Source::Contour(_) => None,
        };
        flag.is_some_and(|f| f.get())
    }

    /// The decode-scale assertion for an image decode.
    pub fn decode_max_size(&self) -> Option<u32> {
        let size = match self {
            Source::Auto(s) => s.decode_max_size,
            Source::ImageBytes(s) => s.decode_max_size,
            Source::FilePath(s) => s.decode_max_size,
            Source::Blob(_)
            | Source::Raw(_)
            | Source::List(_)
            | Source::Array(_)
            | Source::Contour(_) => None,
        };
        size.map(|s| s.get())
    }

    /// Cloud credentials and the path sandbox, for a source that reads paths.
    pub fn path_settings(&self) -> (Option<&HashMap<String, String>>, Option<&[String]>) {
        match self {
            Source::Auto(s) => (s.cloud_options.as_ref(), s.allowed_roots.as_deref()),
            Source::FilePath(s) => (s.cloud_options.as_ref(), s.allowed_roots.as_deref()),
            Source::ImageBytes(_)
            | Source::Blob(_)
            | Source::Raw(_)
            | Source::List(_)
            | Source::Array(_)
            | Source::Contour(_) => (None, None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(v: serde_json::Value) -> Result<Source, String> {
        serde_json::from_value(v).map_err(|e| e.to_string())
    }

    #[test]
    fn samples_round_trip_through_the_wire() {
        for source in Source::samples() {
            let wire = serde_json::to_value(&source).unwrap();
            assert_eq!(wire["format"], source.name());
            assert_eq!(parse(wire).unwrap(), source);
        }
    }

    #[test]
    fn absent_settings_take_their_defaults() {
        let source = parse(serde_json::json!({"format": "image_bytes"})).unwrap();
        assert!(!source.nulls_on_error());
        assert_eq!(source.dtype(), None);
        assert_eq!(source.decode_max_size(), None);
    }

    /// Every parameter used to be accepted by every format on the wire, and
    /// dropped by the formats that do not read it.
    #[test]
    fn a_field_the_format_does_not_read_is_rejected_naming_where_it_applies() {
        let err = parse(serde_json::json!({"format": "image_bytes", "allowed_roots": ["/srv"]}))
            .unwrap_err();
        assert!(
            err.contains("source 'image_bytes'")
                && err.contains("'allowed_roots' does not apply")
                && err.contains("it applies to: auto, file_path"),
            "{err}"
        );
        let err = parse(serde_json::json!({"format": "contour", "size": [4, 4],
                                           "fill_value": 1, "background": 0, "dtype": "u8"}))
        .unwrap_err();
        assert!(
            err.contains("'dtype' does not apply to the 'contour'"),
            "{err}"
        );
        let err = parse(serde_json::json!({"format": "png"})).unwrap_err();
        assert!(err.contains("unknown source format 'png'"), "{err}");
    }

    /// `on_error` was a string checked by a separate parser, and a raw source
    /// without a dtype failed per row.
    #[test]
    fn values_are_typed() {
        let err = parse(serde_json::json!({"format": "blob", "on_error": "skip"})).unwrap_err();
        assert!(err.contains("'on_error'") && err.contains("raise"), "{err}");
        let err = parse(serde_json::json!({"format": "raw"})).unwrap_err();
        assert!(err.contains("dtype"), "{err}");
        let err = parse(serde_json::json!({"format": "list", "dtype": "f128"})).unwrap_err();
        assert!(err.contains("'dtype'"), "{err}");
    }
}
