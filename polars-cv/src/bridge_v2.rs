//! Bridge between Polars and the v2 graph subsystem (`pcv-graph`).
//!
//! This module owns the `vb_graph_v2` Polars expression plugin and the
//! corresponding output-dtype callback. It is the only place where pcv-graph
//! meets Polars Series; pcv-graph itself stays Polars-free.
//!
//! Scope of this initial commit (step 4c of the rewrite):
//!
//! - Single-source pipelines reading from a `Binary` column.
//! - Single-sink pipelines that emit bytes (e.g. the `png` sink).
//! - All-literal parameters; expression-bound params land in a follow-up.
//!
//! More sophisticated sinks (numpy struct, lists/arrays) inherit the
//! existing zero-copy path in `crate::output` and will be wired through here
//! as their pcv-graph Sink adapters land.

use pcv_graph::sink::SinkRowOutput;
use pcv_graph::wire::{self, WireFormat};
use pcv_graph::CompiledGraph;
use polars::prelude::*;
use serde::Deserialize;

/// Kwargs for the v2 plugin entry point.
///
/// Intentionally a strict superset of `GraphKwargs` (the v1 shape) so the
/// Python builder can emit the same key shape with a different `graph_json`
/// payload. The `expr_column_names` field is reserved for the
/// expression-bound-params commit.
#[derive(Debug, Deserialize)]
pub struct V2Kwargs {
    /// JSON-serialized `pcv_graph::Graph` (wire-format v2).
    pub graph_json: String,
    /// Names of expression columns (resolved per row; unused until the
    /// expression-bound-params commit).
    #[serde(default)]
    #[allow(dead_code)]
    pub expr_column_names: Vec<String>,
}

/// Execute a v2 graph against the given inputs.
///
/// `inputs[0]` must be a `Binary` column whose rows are the bytes the source
/// reads. Returns a `Binary` Series of the sink's emitted bytes.
pub(crate) fn execute_v2(inputs: &[Series], kwargs: &V2Kwargs) -> PolarsResult<Series> {
    if inputs.is_empty() {
        return Err(polars_err!(ComputeError: "vb_graph_v2 requires at least one input"));
    }

    let graph = wire::decode(kwargs.graph_json.as_bytes(), WireFormat::Json)
        .map_err(|e| polars_err!(ComputeError: "v2 wire decode failed: {e}"))?;
    let compiled = CompiledGraph::compile(graph)
        .map_err(|e| polars_err!(ComputeError: "v2 compile failed: {e}"))?;

    let binary = inputs[0].binary().map_err(|_| {
        polars_err!(ComputeError: "vb_graph_v2 currently requires a Binary input column, got {:?}", inputs[0].dtype())
    })?;

    let mut out: Vec<Option<Vec<u8>>> = Vec::with_capacity(binary.len());
    for (row_idx, bytes_opt) in binary.into_iter().enumerate() {
        match bytes_opt {
            Some(bytes) => {
                let sink_outputs = compiled
                    .execute_row_simple(row_idx, bytes)
                    .map_err(|e| polars_err!(ComputeError: "v2 row {row_idx} failed: {e}"))?;
                let bytes_out = match sink_outputs.into_iter().next() {
                    Some(SinkRowOutput::Bytes(b)) => b,
                    Some(_) => {
                        return Err(polars_err!(
                            ComputeError: "v2 bridge currently supports only byte-emitting sinks; non-byte sink came back"
                        ))
                    }
                    None => return Err(polars_err!(ComputeError: "v2 graph emitted no output")),
                };
                out.push(Some(bytes_out));
            }
            None => out.push(None),
        }
    }

    let name = inputs[0].name().clone();
    Ok(BinaryChunked::from_iter_options(
        name,
        out.into_iter().map(|opt| opt.map(|v| v.into_iter().collect::<Vec<u8>>())),
    )
    .into_series())
}

#[cfg(test)]
mod tests {
    //! Parity test: identical PNG in → identical PNG out for v1 vs v2.
    //!
    //! v1 calls view-buffer's ImageOp through ViewExpr; v2 does the same via
    //! the pcv-graph registered `grayscale` adapter that wraps the same call.
    //! If they disagree on bytes, that's a bug in the bridge.

    use super::*;
    use image::{ImageBuffer, Rgb};
    use pcv_graph::ir::{
        Graph, Inputs, Node, OutputBinding, SerializedParams, SinkSpec, SourceSpec,
    };
    use view_buffer::interop::image::ImageAdapter;
    use view_buffer::ops::{ImageOp, ImageOpKind, ViewDto};

    fn make_test_png() -> Vec<u8> {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(4, 4, |x, y| {
            Rgb([(x * 60) as u8, (y * 60) as u8, ((x + y) * 30) as u8])
        });
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        buf
    }

    fn v1_grayscale_png(png_in: &[u8]) -> Vec<u8> {
        // Mirror v1's flow at polars-cv/src/graph/types.rs:676-681 — same
        // method calls the v1 executor makes, just driven by hand here so the
        // test isn't coupled to v1's private API.
        let buf = ImageAdapter::decode(png_in).unwrap();
        let expr = view_buffer::ViewExpr::new_source(buf);
        let expr = expr.apply_op(ViewDto::Image(ImageOp {
            kind: ImageOpKind::Grayscale,
        }));
        let result = expr.plan().execute();
        ImageAdapter::encode(&result, image::ImageFormat::Png).unwrap()
    }

    fn build_v2_graph_json() -> String {
        let mut g = Graph::new();
        g.nodes.insert(
            "src".into(),
            Node::Source {
                spec: SourceSpec {
                    name: "image_bytes".into(),
                    params: SerializedParams::new(),
                    input_column: Some("img".into()),
                },
            },
        );
        g.nodes.insert(
            "gray".into(),
            Node::Op {
                op_id: "grayscale".into(),
                params: SerializedParams::new(),
                inputs: Inputs::Single { node: "src".into() },
            },
        );
        g.outputs.push(OutputBinding {
            alias: "_output".into(),
            node: "gray".into(),
            sink: SinkSpec {
                name: "png".into(),
                params: SerializedParams::new(),
            },
            planned: None,
        });
        serde_json::to_string(&g).unwrap()
    }

    #[test]
    fn v1_v2_grayscale_byte_parity() {
        let png_in = make_test_png();

        let v1_out = v1_grayscale_png(&png_in);

        let graph_json = build_v2_graph_json();
        let kwargs = V2Kwargs {
            graph_json,
            expr_column_names: vec![],
        };
        let series = BinaryChunked::from_iter_options(
            "img".into(),
            std::iter::once(Some(png_in.iter().copied().collect::<Vec<u8>>())),
        )
        .into_series();
        let v2_series = execute_v2(&[series], &kwargs).expect("v2 execute");
        let v2_chunked = v2_series.binary().expect("v2 returns Binary");
        let v2_out = v2_chunked
            .get(0)
            .expect("first row is non-null")
            .to_vec();

        assert_eq!(
            v1_out, v2_out,
            "v1 and v2 must produce byte-identical PNG output for the same input"
        );
    }

    #[test]
    fn v2_returns_binary_field() {
        let graph_json = build_v2_graph_json();
        let kwargs = V2Kwargs {
            graph_json,
            expr_column_names: vec![],
        };
        let field = output_field_v2(&[Field::new("img".into(), DataType::Binary)], &kwargs)
            .expect("output field");
        assert_eq!(*field.dtype(), DataType::Binary);
    }

    #[test]
    fn v2_propagates_null_rows() {
        let graph_json = build_v2_graph_json();
        let kwargs = V2Kwargs {
            graph_json,
            expr_column_names: vec![],
        };
        let png_in = make_test_png();
        // One null row, one valid row — null must propagate, valid must execute.
        let series = BinaryChunked::from_iter_options(
            "img".into(),
            [None, Some(png_in.iter().copied().collect::<Vec<u8>>())].into_iter(),
        )
        .into_series();
        let out = execute_v2(&[series], &kwargs).expect("execute");
        let bin = out.binary().expect("binary");
        assert_eq!(bin.len(), 2);
        assert!(bin.get(0).is_none(), "null row should propagate");
        assert!(bin.get(1).is_some(), "valid row should produce bytes");
    }
}

/// Decide the output dtype for `vb_graph_v2`.
///
/// For now, the bridge only supports byte-emitting sinks, so the field
/// dtype is `Binary` regardless of which sink is named. As list/array/numpy
/// sinks land, this will inspect the graph's first output binding to choose
/// the appropriate Polars dtype.
pub(crate) fn output_field_v2(
    input_fields: &[Field],
    _kwargs: &V2Kwargs,
) -> PolarsResult<Field> {
    let name = input_fields
        .first()
        .map(|f| f.name().clone())
        .unwrap_or_else(|| PlSmallStr::from_static("output"));
    Ok(Field::new(name, DataType::Binary))
}
