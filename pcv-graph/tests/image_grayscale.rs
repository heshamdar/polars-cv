//! End-to-end test: PNG bytes → image_bytes source → grayscale op → png sink.
//!
//! Builds a tiny 2x2 RGB PNG in-memory, drives the full executor, and
//! asserts the result is a valid PNG that decodes back to a 1-channel image.
//! This is the v2 mirror of the v1 behavior we'll parity-test against in
//! step 4c (the Polars wire-up commit).

use image::{ImageBuffer, Rgb};
use pcv_graph::ir::{
    Graph, Inputs, Node, OutputBinding, SerializedParams, SinkSpec, SourceSpec,
};
use pcv_graph::sink::SinkRowOutput;
use pcv_graph::CompiledGraph;

fn make_png_2x2() -> Vec<u8> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(2, 2, |x, y| {
        match (x, y) {
            (0, 0) => Rgb([255, 0, 0]),
            (1, 0) => Rgb([0, 255, 0]),
            (0, 1) => Rgb([0, 0, 255]),
            (1, 1) => Rgb([255, 255, 255]),
            _ => unreachable!(),
        }
    });
    let mut buf = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .expect("encode initial PNG");
    buf
}

fn build_graph() -> Graph {
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
    g
}

#[test]
fn png_in_grayscale_png_out_round_trip() {
    let png_in = make_png_2x2();
    let compiled = CompiledGraph::compile(build_graph()).expect("compile");

    let outputs = compiled
        .execute_row_simple(0, &png_in)
        .expect("execute row");

    assert_eq!(outputs.len(), 1, "single output binding");
    let out_bytes = match &outputs[0] {
        SinkRowOutput::Bytes(b) => b.clone(),
        _ => panic!("expected PNG bytes from `png` sink"),
    };

    // Decode the sink's PNG; verify it's a valid 2x2 single-channel image.
    let decoded =
        image::load_from_memory(&out_bytes).expect("png sink emitted a decodable PNG");
    let gray = decoded.to_luma8();
    assert_eq!(gray.dimensions(), (2, 2));

    // BT.601 luma for the input row 0: R=255 -> ~76, G=255 -> ~150.
    let p00 = gray.get_pixel(0, 0)[0];
    let p10 = gray.get_pixel(1, 0)[0];
    assert!(
        (70..=82).contains(&p00),
        "red pixel grayscale should be ~76, got {p00}"
    );
    assert!(
        (140..=160).contains(&p10),
        "green pixel grayscale should be ~150, got {p10}"
    );
    let p11 = gray.get_pixel(1, 1)[0];
    assert_eq!(p11, 255, "white pixel grayscale should be 255");
}
