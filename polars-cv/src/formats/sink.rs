//! Sink formats: how an output node is encoded.

use polars_cv_macros::Ops;
use view_buffer::ImageCodec;

pub use super::sink_dtype::SinkDType;
use crate::ops::Literal;

/// How an output node is encoded (see the module docs).
#[derive(Debug, Clone, PartialEq, Ops)]
pub enum Sink {
    /// A fixed-shape Polars `Array`.
    #[op(name = "array", sample = {"shape": [2, 2]})]
    Array {
        /// The output shape; inferred from the planned shape when absent.
        shape: Option<Vec<Literal<u32>>>,
    },
    /// The self-describing VIEW protocol, as `Binary`.
    #[op(name = "blob", sample = {})]
    Blob,
    /// Re-encoded as JPEG.
    #[op(name = "jpeg", sample = {"quality": 90})]
    Jpeg {
        /// JPEG quality, 1-100. Only the JPEG encoder takes one.
        #[param(default = 85)]
        quality: Literal<u8>,
    },
    /// A nested Polars `List`.
    #[op(name = "list", sample = {})]
    List,
    /// The domain's own Polars type: `Float64` for a scalar, a `List` for a
    /// vector, the contour struct for contours.
    #[op(name = "native", sample = {})]
    Native,
    /// The `polars_cv.ndarray` extension type: a zero-copy tensor struct.
    #[op(name = "ndarray", sample = {"dtype": "f16"})]
    NdArray {
        /// Downcast the elements to this dtype at encode time: only half
        /// precision (``"f16"``/``"float16"``); use ``.cast(...)`` for any
        /// other dtype.
        dtype: Option<Literal<SinkDType>>,
    },
    /// A zero-copy tensor struct `numpy_from_struct` reads.
    #[op(name = "numpy", sample = {})]
    Numpy {
        /// Downcast the elements to this dtype at encode time: only half
        /// precision (``"f16"``/``"float16"``); use ``.cast(...)`` for any
        /// other dtype.
        dtype: Option<Literal<SinkDType>>,
    },
    /// Re-encoded as PNG.
    #[op(name = "png", sample = {})]
    Png,
    /// Re-encoded as TIFF.
    #[op(name = "tiff", sample = {})]
    Tiff,
    /// A zero-copy tensor struct `torch_from_struct` reads.
    #[op(name = "torch", sample = {"dtype": "float16"})]
    Torch {
        /// Downcast the elements to this dtype at encode time: only half
        /// precision (``"f16"``/``"float16"``); use ``.cast(...)`` for any
        /// other dtype.
        dtype: Option<Literal<SinkDType>>,
    },
    /// Re-encoded as WebP (lossless).
    #[op(name = "webp", sample = {})]
    WebP,
}

impl super::Format for Sink {
    const KIND: &'static str = "sink";
    fn formats() -> &'static [view_buffer::mode::OpDesc] {
        static CATALOG: std::sync::LazyLock<Vec<view_buffer::mode::OpDesc>> =
            std::sync::LazyLock::new(Sink::catalog);
        &CATALOG
    }
}

super::tagged_serde!(Sink);

impl Sink {
    /// The image codec a re-encoding sink writes through.
    pub fn image_codec(&self) -> Option<ImageCodec> {
        match self {
            Sink::Png => Some(ImageCodec::Png),
            Sink::Jpeg { .. } => Some(ImageCodec::Jpeg),
            Sink::WebP => Some(ImageCodec::WebP),
            Sink::Tiff => Some(ImageCodec::Tiff),
            Sink::Array { .. }
            | Sink::Blob
            | Sink::List
            | Sink::Native
            | Sink::NdArray { .. }
            | Sink::Numpy { .. }
            | Sink::Torch { .. } => None,
        }
    }

    /// The `array` sink's explicit shape.
    pub fn shape(&self) -> Option<Vec<usize>> {
        match self {
            Sink::Array { shape } => shape
                .as_ref()
                .map(|s| s.iter().map(|d| d.get() as usize).collect()),
            Sink::Blob
            | Sink::Jpeg { .. }
            | Sink::List
            | Sink::Native
            | Sink::NdArray { .. }
            | Sink::Numpy { .. }
            | Sink::Png
            | Sink::Tiff
            | Sink::Torch { .. }
            | Sink::WebP => None,
        }
    }

    /// Whether a tensor sink downcasts to half precision.
    pub fn as_f16(&self) -> bool {
        match self {
            Sink::NdArray { dtype } | Sink::Numpy { dtype } | Sink::Torch { dtype } => {
                matches!(dtype, Some(Literal(SinkDType::F16)))
            }
            Sink::Array { .. }
            | Sink::Blob
            | Sink::Jpeg { .. }
            | Sink::List
            | Sink::Native
            | Sink::Png
            | Sink::Tiff
            | Sink::WebP => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::Format as _;

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
        assert_eq!(
            sink,
            Sink::Jpeg {
                quality: Literal(85)
            }
        );
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
