//! Source formats: how a node's input column is decoded.

use std::collections::HashMap;
use std::num::NonZeroU32;

use polars::prelude::{DataType, Series};
use polars_cv_macros::Ops;
use view_buffer::DType;

use crate::fetch::FetchErrorPolicy;
use crate::ops::Literal;

/// Define the input source format.
///
/// The default ``"auto"`` infers the decode path from the column's Polars
/// dtype at runtime. Pass an explicit format to override the inference (or
/// when the column dtype cannot be routed, such as a plain numeric column).
///
/// Image sources (``"image_bytes"`` and ``"file_path"``) auto-detect the
/// encoding and preserve its dtype: PNG/JPEG decode to u8, 16-bit PNG to u16,
/// and TIFF may produce u8, u16, f32 or f64. Decoded images are always 3D
/// ``[H, W, C]``. Until then the dtype is ``"auto"``: a ``list``/``array``
/// sink needs it known at planning time, from ``dtype=`` here, a ``cast()``,
/// or an operation that fixes it.
///
/// A ``"contour"`` source decodes to the contour domain; rasterize it with
/// :meth:`rasterize`.
///
/// Each keyword applies to some formats and not others, and defaults to
/// ``None`` (the format's own default). One that does not apply to the chosen
/// format is **rejected**, naming the formats it applies to.
///
/// ``decode_max_size`` asserts the pipeline needs at most this many pixels on
/// the decoded long side, so JPEG decoding uses IDCT scaling (1/8, 1/4 or
/// 1/2) to skip work. The long side never drops below
/// ``min(decode_max_size, original)``, so a downstream resize to that size
/// never upscales; other encodings decode at full size. A scaled decode
/// followed by a resize is not bit-identical to a full decode and the same
/// resize, hence the explicit opt-in.
///
/// ``allowed_roots`` restricts which locations a path column may read from.
/// An entry that parses as a remote URI (``"s3://bucket/public/"``) is
/// matched as a URI prefix, anything else (``"/srv/images"``) as a local
/// directory. Local paths are canonicalized first, so ``..`` and symlinks
/// cannot escape, and matching is component-wise (``"/srv/images"`` does not
/// admit ``"/srv/images-private"``). A path matching no entry is refused, and
/// the refusal is subject to ``on_error``.
///
/// Example:
///     ```python
///     >>> # Decode PNG/JPEG bytes from a column
///     >>> pipe = Pipeline().source("image_bytes").resize(height=224, width=224)
///     >>>
///     >>> # Read from file paths or URLs, sandboxed
///     >>> pipe = Pipeline().source("file_path", allowed_roots=["/srv/images"])
///     >>>
///     >>> # Assert dtype for a list sink (cast if needed at runtime)
///     >>> pipe = Pipeline().source("image_bytes", dtype="f32")
///     >>>
///     >>> # Gracefully handle corrupt images as null
///     >>> pipe = Pipeline().source("image_bytes", on_error="null")
///     >>>
///     >>> # Rasterize a contour column to a mask
///     >>> pipe = Pipeline().source("contour").rasterize(width=64, height=64)
///     ```
#[derive(Debug, Clone, PartialEq, Ops)]
pub enum Source {
    /// A Polars fixed-size `Array` column, one nesting level per axis.
    #[op(name = "array", sample = {"require_contiguous": true})]
    Array {
        /// Element dtype; inferred from the column when absent.
        dtype: Option<Literal<DType>>,
        /// Require rectangular data (zero-copy); jagged rows are then an error.
        #[param(default = false)]
        require_contiguous: Literal<bool>,
        /// "raise" or "null": what a row that cannot be decoded does.
        #[param(default = "raise")]
        on_error: Literal<FetchErrorPolicy>,
    },
    /// Infer the decode path from the column's Polars dtype: String → file_path,
    /// List/Array → list/array, Binary → blob if VIEW-tagged else image_bytes.
    #[op(name = "auto", sample = {"allowed_roots": ["/srv"], "decode_max_size": 64})]
    Auto {
        /// Asserted element dtype (a decoded image is cast to it).
        dtype: Option<Literal<DType>>,
        /// Require rectangular data when the column is a List/Array.
        #[param(default = false)]
        require_contiguous: Literal<bool>,
        /// Cloud-storage credentials: a ``CloudOptions`` or a dict.
        cloud_options: Option<HashMap<String, String>>,
        /// Locations a path column may read from (unrestricted when absent;
        /// see above).
        allowed_roots: Option<Vec<String>>,
        /// Decode only enough pixels for this long side (JPEG IDCT scaling;
        /// see above).
        decode_max_size: Option<Literal<NonZeroU32>>,
        /// "raise" or "null": what a row that cannot be decoded does.
        #[param(default = "raise")]
        on_error: Literal<FetchErrorPolicy>,
    },
    /// The self-describing VIEW protocol.
    #[op(name = "blob", sample = {})]
    Blob {
        /// Declared element dtype. A blob carries its own, so a declaration is
        /// checked at decode: a blob of another dtype is a row error.
        dtype: Option<Literal<DType>>,
        /// "raise" or "null": what a row that cannot be decoded does.
        #[param(default = "raise")]
        on_error: Literal<FetchErrorPolicy>,
    },
    /// A contour column (one contour or a set per row), decoded as the contour
    /// set `extract_contours` produces. A mask is the `rasterize` op's.
    #[op(name = "contour", sample = {"on_error": "null"})]
    Contour {
        /// "raise" or "null": what a row that cannot be decoded does.
        #[param(default = "raise")]
        on_error: Literal<FetchErrorPolicy>,
    },
    /// A path (local, s3://, gs://, az://, http://) whose contents decode like
    /// image bytes.
    #[op(name = "file_path",
         sample = {"cloud_options": {"aws_region": "eu-west-1"}, "dtype": "u16"})]
    FilePath {
        /// Asserted element dtype: a decoded image with another dtype is cast.
        dtype: Option<Literal<DType>>,
        /// Cloud-storage credentials: a ``CloudOptions`` or a dict.
        cloud_options: Option<HashMap<String, String>>,
        /// Locations a path column may read from (unrestricted when absent;
        /// see above).
        allowed_roots: Option<Vec<String>>,
        /// Decode only enough pixels for this long side (JPEG IDCT scaling;
        /// see above).
        decode_max_size: Option<Literal<NonZeroU32>>,
        /// "raise" or "null": what a row that cannot be read or decoded does.
        #[param(default = "raise")]
        on_error: Literal<FetchErrorPolicy>,
    },
    /// Encoded image bytes (PNG/JPEG/TIFF/...), always decoded to `[H, W, C]`.
    #[op(name = "image_bytes", sample = {"decode_max_size": 64})]
    ImageBytes {
        /// Asserted element dtype: a decoded image with another dtype is cast.
        dtype: Option<Literal<DType>>,
        /// Decode only enough pixels for this long side (JPEG IDCT scaling;
        /// see above).
        decode_max_size: Option<Literal<NonZeroU32>>,
        /// "raise" or "null": what a row that cannot be decoded does.
        #[param(default = "raise")]
        on_error: Literal<FetchErrorPolicy>,
    },
    /// A Polars nested `List` column, one nesting level per axis; its sizes
    /// may differ from row to row.
    #[op(name = "list", sample = {"dtype": "f32"})]
    List {
        /// Element dtype; inferred from the column when absent.
        dtype: Option<Literal<DType>>,
        /// Require rectangular data (zero-copy); jagged rows are then an error.
        #[param(default = false)]
        require_contiguous: Literal<bool>,
        /// "raise" or "null": what a row that cannot be decoded does.
        #[param(default = "raise")]
        on_error: Literal<FetchErrorPolicy>,
    },
    /// Raw bytes, decoded as a flat 1-D buffer of `dtype`.
    #[op(name = "raw", sample = {"dtype": "u8"})]
    Raw {
        /// The element dtype: raw bytes carry no type metadata, so it is required.
        dtype: Literal<DType>,
        /// "raise" or "null": what a row that cannot be decoded does.
        #[param(default = "raise")]
        on_error: Literal<FetchErrorPolicy>,
    },
}

impl Source {
    /// The format `source()` reads when none is named.
    pub const DEFAULT_FORMAT: &'static str = "auto";
}

impl super::Format for Source {
    const KIND: &'static str = "source";
    fn formats() -> &'static [view_buffer::mode::OpDesc] {
        static CATALOG: std::sync::LazyLock<Vec<view_buffer::mode::OpDesc>> =
            std::sync::LazyLock::new(Source::catalog);
        &CATALOG
    }
}

super::tagged_serde!(Source);

impl Source {
    /// The concrete source an `auto` source reads `column` as: the format the
    /// column's Polars dtype routes to, carrying the auto settings that format
    /// reads. `None` for a source that is already concrete.
    ///
    /// The dtype is constant across rows, so the route is taken once per
    /// batch. `Binary` columns are sniffed for the VIEW protocol magic to tell
    /// self-describing blobs apart from encoded image bytes (the image decoder
    /// auto-detects PNG/JPEG/TIFF internally, so `image_bytes` covers all
    /// non-VIEW binary).
    pub fn route(&self, column: &Series) -> Option<Result<Source, String>> {
        let Source::Auto {
            dtype,
            require_contiguous,
            cloud_options,
            allowed_roots,
            decode_max_size,
            on_error,
        } = self
        else {
            return None;
        };
        let (dtype, require_contiguous, decode_max_size, on_error) =
            (*dtype, *require_contiguous, *decode_max_size, *on_error);
        Some(match column.dtype() {
            DataType::String => Ok(Source::FilePath {
                dtype,
                cloud_options: cloud_options.clone(),
                allowed_roots: allowed_roots.clone(),
                decode_max_size,
                on_error,
            }),
            DataType::List(_) => Ok(Source::List {
                dtype,
                require_contiguous,
                on_error,
            }),
            DataType::Array(_, _) => Ok(Source::Array {
                dtype,
                require_contiguous,
                on_error,
            }),
            // The first present row decides: blobs carry the magic, images
            // don't. An all-null column reads as image bytes (null rows).
            DataType::Binary => {
                let blob = column.binary().ok().and_then(|ca| {
                    (0..ca.len())
                        .find_map(|i| ca.get(i))
                        .map(|bytes| bytes.starts_with(&view_buffer::protocol::MAGIC_BYTES))
                });
                Ok(if blob == Some(true) {
                    Source::Blob { dtype, on_error }
                } else {
                    Source::ImageBytes {
                        dtype,
                        decode_max_size,
                        on_error,
                    }
                })
            }
            other => Err(format!(
                "auto source cannot infer a decode path for column dtype {other:?}; \
                 specify an explicit source format (e.g. source(\"image_bytes\"), \
                 source(\"list\"), source(\"blob\"))."
            )),
        })
    }

    /// Whether the element dtype and rank are resolved from the input column's
    /// Polars type when the query is planned with its input (a `list`/`array`
    /// column, or `auto` routing to one), rather than at build time.
    pub fn resolves_from_column(&self) -> bool {
        match self {
            Source::List { .. } | Source::Array { .. } | Source::Auto { .. } => true,
            Source::Blob { .. }
            | Source::Contour { .. }
            | Source::FilePath { .. }
            | Source::ImageBytes { .. }
            | Source::Raw { .. } => false,
        }
    }

    /// Whether the input column's type may fix the sizes too: a fixed-size
    /// `Array` column states every one (`auto` may route to one); a `List`
    /// column's vary per row.
    pub fn column_may_fix_sizes(&self) -> bool {
        match self {
            Source::Array { .. } | Source::Auto { .. } => true,
            Source::List { .. }
            | Source::Blob { .. }
            | Source::Contour { .. }
            | Source::FilePath { .. }
            | Source::ImageBytes { .. }
            | Source::Raw { .. } => false,
        }
    }

    /// Whether a row that cannot be decoded is nulled rather than failing the
    /// query.
    pub fn nulls_on_error(&self) -> bool {
        match self {
            Source::Array { on_error, .. }
            | Source::Auto { on_error, .. }
            | Source::Blob { on_error, .. }
            | Source::Contour { on_error }
            | Source::FilePath { on_error, .. }
            | Source::ImageBytes { on_error, .. }
            | Source::List { on_error, .. }
            | Source::Raw { on_error, .. } => on_error.get().nulls_the_row(),
        }
    }

    /// The declared element dtype, if any.
    pub fn dtype(&self) -> Option<DType> {
        match self {
            Source::Array { dtype, .. }
            | Source::Auto { dtype, .. }
            | Source::Blob { dtype, .. }
            | Source::FilePath { dtype, .. }
            | Source::ImageBytes { dtype, .. }
            | Source::List { dtype, .. } => dtype.map(|d| d.get()),
            Source::Raw { dtype, .. } => Some(dtype.get()),
            Source::Contour { .. } => None,
        }
    }

    /// Whether a nested-column decode must be zero-copy.
    pub fn require_contiguous(&self) -> bool {
        match self {
            Source::Array {
                require_contiguous, ..
            }
            | Source::Auto {
                require_contiguous, ..
            }
            | Source::List {
                require_contiguous, ..
            } => require_contiguous.get(),
            Source::Blob { .. }
            | Source::Contour { .. }
            | Source::FilePath { .. }
            | Source::ImageBytes { .. }
            | Source::Raw { .. } => false,
        }
    }

    /// The decode-scale assertion for an image decode.
    pub fn decode_max_size(&self) -> Option<u32> {
        match self {
            Source::Auto {
                decode_max_size, ..
            }
            | Source::FilePath {
                decode_max_size, ..
            }
            | Source::ImageBytes {
                decode_max_size, ..
            } => decode_max_size.map(|s| s.get().get()),
            Source::Array { .. }
            | Source::Blob { .. }
            | Source::Contour { .. }
            | Source::List { .. }
            | Source::Raw { .. } => None,
        }
    }

    /// Cloud credentials and the path sandbox, for a source that reads paths.
    pub fn path_settings(&self) -> (Option<&HashMap<String, String>>, Option<&[String]>) {
        match self {
            Source::Auto {
                cloud_options,
                allowed_roots,
                ..
            }
            | Source::FilePath {
                cloud_options,
                allowed_roots,
                ..
            } => (cloud_options.as_ref(), allowed_roots.as_deref()),
            Source::Array { .. }
            | Source::Blob { .. }
            | Source::Contour { .. }
            | Source::ImageBytes { .. }
            | Source::List { .. }
            | Source::Raw { .. } => (None, None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::Format as _;

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

    /// An auto source reads a column as the format its dtype routes to,
    /// keeping the auto settings that format reads.
    #[test]
    fn auto_routes_by_column_dtype_keeping_its_settings() {
        use polars::prelude::*;
        let auto = parse(serde_json::json!({"format": "auto", "on_error": "null",
                                            "decode_max_size": 64, "dtype": "u8"}))
        .unwrap();
        let route = |s: Series| auto.route(&s).unwrap();
        let image = route(Series::new("b".into(), [Some(&b"\x89PNG"[..])])).unwrap();
        assert_eq!(image.name(), "image_bytes");
        assert_eq!(image.decode_max_size(), Some(64));
        assert!(image.nulls_on_error());
        assert_eq!(image.dtype(), Some(DType::U8));
        let blob = view_buffer::ViewBuffer::from_vec(vec![1u8, 2]).to_blob();
        assert_eq!(
            route(Series::new("b".into(), [&blob[..]])).unwrap().name(),
            "blob"
        );
        assert_eq!(
            route(Series::new("p".into(), ["/a.png"])).unwrap().name(),
            "file_path"
        );
        let err = route(Series::new("i".into(), [1i32])).unwrap_err();
        assert!(err.contains("cannot infer a decode path"), "{err}");
        let concrete = parse(serde_json::json!({"format": "raw", "dtype": "u8"})).unwrap();
        assert!(concrete.route(&Series::new("b".into(), [1i32])).is_none());
    }

    /// A decode scale of zero pixels is no size: refused by the field's
    /// type, where Python used to check it beside the definition.
    #[test]
    fn decode_max_size_is_positive() {
        let err =
            parse(serde_json::json!({"format": "image_bytes", "decode_max_size": 0})).unwrap_err();
        assert!(
            err.contains("'decode_max_size'") && err.contains("positive"),
            "{err}"
        );
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
        let err = parse(serde_json::json!({"format": "contour", "dtype": "u8"})).unwrap_err();
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
