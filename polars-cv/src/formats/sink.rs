//! Sink formats: how an output node is encoded.

use polars_cv_macros::Op;
use serde::{Deserialize, Serialize};
use view_buffer::ImageCodec;

use super::formats;
pub use super::sink_dtype::SinkDType;
use crate::ops::Literal;

/// A zero-copy tensor struct (`numpy`, `torch`, or the tagged `ndarray`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct TensorSink {
    /// Downcast the elements to this dtype at encode time: only half precision
    /// (``"f16"``/``"float16"``); use ``.cast(...)`` for any other dtype.
    pub dtype: Option<Literal<SinkDType>>,
}

/// Re-encoded as JPEG.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct JpegSink {
    /// JPEG quality, 1-100 (default 85). Only the JPEG encoder takes one.
    pub quality: Option<Literal<u8>>,
}

impl JpegSink {
    /// The quality when none is given.
    pub const DEFAULT_QUALITY: u8 = 85;
}

/// A fixed-shape Polars `Array`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct ArraySink {
    /// The output shape; inferred from the planned shape when absent.
    pub shape: Option<Vec<Literal<u32>>>,
}

/// A format with no parameters: re-encoded by an image codec (``png``,
/// ``webp``, ``tiff``), the self-describing VIEW ``blob``, a nested ``list``,
/// or the domain's ``native`` Polars type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct PlainSink {}

formats! {
    /// How an output node is encoded (see the module docs).
    Sink("sink") {
        "array" => Array(ArraySink) {"shape": [2, 2]},
        "blob" => Blob(PlainSink) {},
        "jpeg" => Jpeg(JpegSink) {"quality": 90},
        "list" => List(PlainSink) {},
        "native" => Native(PlainSink) {},
        "ndarray" => NdArray(TensorSink) {"dtype": "f16"},
        "numpy" => Numpy(TensorSink) {},
        "png" => Png(PlainSink) {},
        "tiff" => Tiff(PlainSink) {},
        "torch" => Torch(TensorSink) {"dtype": "float16"},
        "webp" => WebP(PlainSink) {},
    }
}

impl Sink {
    /// The image codec a re-encoding sink writes through.
    pub fn image_codec(&self) -> Option<ImageCodec> {
        match self {
            Sink::Png(_) => Some(ImageCodec::Png),
            Sink::Jpeg(_) => Some(ImageCodec::Jpeg),
            Sink::WebP(_) => Some(ImageCodec::WebP),
            Sink::Tiff(_) => Some(ImageCodec::Tiff),
            Sink::Array(_)
            | Sink::Blob(_)
            | Sink::List(_)
            | Sink::Native(_)
            | Sink::NdArray(_)
            | Sink::Numpy(_)
            | Sink::Torch(_) => None,
        }
    }

    /// The JPEG quality; the default for every other format, which ignores it.
    pub fn quality(&self) -> u8 {
        match self {
            Sink::Jpeg(JpegSink { quality }) => quality
                .map(|q| q.get())
                .unwrap_or(JpegSink::DEFAULT_QUALITY),
            _ => JpegSink::DEFAULT_QUALITY,
        }
    }

    /// The `array` sink's explicit shape.
    pub fn shape(&self) -> Option<Vec<usize>> {
        match self {
            Sink::Array(ArraySink { shape }) => shape
                .as_ref()
                .map(|s| s.iter().map(|d| d.get() as usize).collect()),
            _ => None,
        }
    }

    /// Whether a tensor sink downcasts to half precision.
    pub fn as_f16(&self) -> bool {
        match self {
            Sink::Numpy(t) | Sink::Torch(t) | Sink::NdArray(t) => {
                matches!(t.dtype, Some(Literal(SinkDType::F16)))
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(v: serde_json::Value) -> Result<Sink, String> {
        serde_json::from_value(v).map_err(|e| e.to_string())
    }

    #[test]
    fn samples_round_trip_through_the_wire() {
        for sink in Sink::samples() {
            let wire = serde_json::to_value(&sink).unwrap();
            assert_eq!(wire["format"], sink.name());
            assert_eq!(parse(wire).unwrap(), sink);
        }
    }

    #[test]
    fn an_absent_field_takes_its_default() {
        let sink = parse(serde_json::json!({"format": "jpeg"})).unwrap();
        assert_eq!(sink.quality(), 85);
        assert_eq!(
            parse(serde_json::json!({"format": "array"}))
                .unwrap()
                .shape(),
            None
        );
    }

    /// A field another format reads is refused naming where it applies —
    /// `quality` on webp was accepted and dropped (the WebP encoder takes none).
    #[test]
    fn a_field_the_format_does_not_read_is_rejected_naming_where_it_applies() {
        let err = parse(serde_json::json!({"format": "webp", "quality": 50})).unwrap_err();
        assert!(
            err.contains("sink 'webp'")
                && err.contains("'quality' does not apply")
                && err.contains("it applies to: jpeg"),
            "{err}"
        );
        let err = parse(serde_json::json!({"format": "jpeg", "qualtiy": 50})).unwrap_err();
        assert!(
            err.contains("'qualtiy' is not a sink parameter") && err.contains("quality"),
            "{err}"
        );
        let err = parse(serde_json::json!({"format": "jpg"})).unwrap_err();
        assert!(err.contains("unknown sink format 'jpg'"), "{err}");
    }

    /// The sink dtype used to be any string, of which only "f16"/"float16"
    /// did anything; the rest were ignored in Rust.
    #[test]
    fn a_sink_dtype_other_than_half_precision_is_rejected() {
        let err = parse(serde_json::json!({"format": "numpy", "dtype": "f32"})).unwrap_err();
        assert!(err.contains("'dtype'") && err.contains("f16"), "{err}");
        for spelling in ["f16", "float16"] {
            let sink = parse(serde_json::json!({"format": "torch", "dtype": spelling})).unwrap();
            assert!(sink.as_f16());
        }
    }

    #[test]
    fn every_codec_is_reached_by_exactly_one_sink() {
        let codecs: Vec<ImageCodec> = Sink::samples()
            .iter()
            .filter_map(Sink::image_codec)
            .collect();
        for codec in [
            ImageCodec::Png,
            ImageCodec::Jpeg,
            ImageCodec::WebP,
            ImageCodec::Tiff,
        ] {
            assert_eq!(
                codecs.iter().filter(|c| **c == codec).count(),
                1,
                "{codec:?}"
            );
        }
    }
}
