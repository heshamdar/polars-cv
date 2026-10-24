//! Geometry execution: the one runner of a geometry step, shared by the
//! pipeline graph and every op-backed `.contour` accessor.

use view_buffer::geometry::contour::Outline;
use view_buffer::geometry::{
    extract::extract_contours, measures, rasterize::rasterize, transforms, Contour,
};
use view_buffer::{GeometryOp, Op};

use crate::graph::node_output::NodeOutput;

/// Execute a geometry step: the pipeline's, and every op-backed `.contour`
/// accessor's (`crate::geometry::accessor`), so the two cannot answer or refuse
/// differently.
///
/// Every refusal is named here, once, by the step (`Offset: ...`), as the
/// executor names every step; the engine functions it calls do not name
/// themselves.
pub(crate) fn execute_geometry_op(
    input: NodeOutput,
    op: &GeometryOp,
) -> Result<NodeOutput, String> {
    run_geometry_op(input, op).map_err(|e| format!("{}: {e}", op.name()))
}

/// [`execute_geometry_op`]'s dispatch, its refusals unnamed.
fn run_geometry_op(input: NodeOutput, op: &GeometryOp) -> Result<NodeOutput, String> {
    let shapes: Vec<&[usize]> = input.as_buffer().map(|b| b.shape()).into_iter().collect();
    let dtypes: Vec<view_buffer::DType> =
        input.as_buffer().map(|b| b.dtype()).into_iter().collect();
    view_buffer::ops::validation::validate_concrete(op, &shapes, &dtypes)
        .map_err(|e| e.to_string())?;
    let expected_domain = op.input_domain();
    let actual_domain = input.domain();
    if expected_domain != actual_domain {
        return Err(format!(
            "expects {} input but received {}. Add a domain-converting operation.",
            expected_domain.name(),
            actual_domain.name()
        ));
    }
    match op {
        GeometryOp::ExtractContours {
            mode,
            method,
            min_area,
            min_area_fraction,
        } => {
            let buffer = input
                .as_buffer()
                .ok_or_else(|| "requires Buffer input".to_string())?;
            // The fraction is of this image's own H x W; a contour must pass
            // both thresholds, so the larger absolute one is the filter. Its
            // range, (0, 1], is `GeometryOp::check`'s, run as the step resolves.
            let relative = min_area_fraction.map(|f| {
                let shape = buffer.shape();
                let pixels =
                    shape.first().copied().unwrap_or(0) * shape.get(1).copied().unwrap_or(0);
                f * pixels as f64
            });
            let threshold = match (*min_area, relative) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            };
            let contours = extract_contours(buffer, *mode, *method, threshold);
            Ok(NodeOutput::from_contours(
                contours.into_iter().map(Outline::Closed).collect(),
            ))
        }
        GeometryOp::Rasterize {
            fill_value,
            background,
            supersample,
            ..
        } => {
            let (height, width) = op
                .canvas()
                .ok_or_else(|| "rasterize reached execution without its canvas".to_string())?;
            // The whole set in one call: `rasterize` paints the regions'
            // union, which is order-independent and honours an inverted
            // fill/background pair. Folding per-contour masks with `max`
            // here did neither.
            let regions = regions(outlines(&input)?)?;
            Ok(NodeOutput::from_buffer(rasterize(
                &regions,
                width,
                height,
                *fill_value,
                *background,
                *supersample,
            )))
        }
        GeometryOp::Area { signed } => {
            let regions = regions(outlines(&input)?)?;
            Ok(NodeOutput::from_vector(
                regions.iter().map(|c| measures::area(c, *signed)).collect(),
            ))
        }
        // A line's length has no closing edge (`outline_length`), as the
        // accessor measures it.
        GeometryOp::Perimeter => Ok(NodeOutput::from_vector(
            outlines(&input)?
                .iter()
                .map(measures::outline_length)
                .collect(),
        )),
        GeometryOp::Centroid => {
            // Flat interleaved: [cx₀, cy₀, cx₁, cy₁, ...]
            let regions = regions(outlines(&input)?)?;
            let mut coords = Vec::with_capacity(regions.len() * 2);
            for c in &regions {
                let pt = measures::centroid(c);
                coords.push(pt.x);
                coords.push(pt.y);
            }
            Ok(NodeOutput::from_vector(coords))
        }
        GeometryOp::BoundingBox => {
            // Flat interleaved: [x₀, y₀, w₀, h₀, x₁, y₁, w₁, h₁, ...]. A
            // contour with no points has no extent: NaN, which a measure
            // vector can hold where it cannot hold a null (never a 0x0 box
            // at the origin, which is not there).
            let all = outlines(&input)?;
            let mut coords = Vec::with_capacity(all.len() * 4);
            for o in all.iter() {
                coords.extend_from_slice(&match o.bounding_box() {
                    Some(bb) => [bb.x, bb.y, bb.width, bb.height],
                    None => [f64::NAN; 4],
                });
            }
            Ok(NodeOutput::from_vector(coords))
        }
        GeometryOp::Translate { dx, dy } => Ok(NodeOutput::from_contours(
            outlines(&input)?
                .iter()
                .map(|o| o.map_points(|c| transforms::translate(c, *dx, *dy)))
                .collect(),
        )),
        GeometryOp::Scale { sx, sy, origin } => outlines(&input)?
            .iter()
            .map(|o| transforms::scale_outline(o, *sx, *sy, *origin))
            .collect::<Result<Vec<_>, _>>()
            .map(NodeOutput::from_contours)
            .map_err(str::to_string),
        GeometryOp::Simplify { tolerance } => Ok(NodeOutput::from_contours(
            outlines(&input)?
                .iter()
                .map(|o| transforms::simplify_outline(o, *tolerance))
                .collect(),
        )),
        // `k >= 1` is `GeometryOp::check`'s, run as every step resolves.
        GeometryOp::Largest { k } => {
            let regions = regions(outlines(&input)?)?;
            Ok(NodeOutput::from_contours(
                transforms::largest(&regions, *k as usize)
                    .into_iter()
                    .map(Outline::Closed)
                    .collect(),
            ))
        }
        // The hull reads only the points, so a line has one too: a region.
        GeometryOp::ConvexHull => Ok(NodeOutput::from_contours(
            outlines(&input)?
                .iter()
                .map(|o| Outline::Closed(transforms::convex_hull_outline(o)))
                .collect(),
        )),
        GeometryOp::Boundary => {
            let mut lines = Vec::new();
            for o in outlines(&input)?.iter() {
                lines.extend(transforms::boundary_outline(o)?);
            }
            Ok(NodeOutput::from_contours(lines))
        }
        GeometryOp::Offset { distance, join } => {
            let regions = regions(outlines(&input)?)?;
            transforms::offset(&regions, *distance, *join)
                .map(|c| NodeOutput::from_contours(c.into_iter().map(Outline::Closed).collect()))
        }
        GeometryOp::Stroke { width, cap, join } => {
            let style = transforms::Stroke {
                width: *width,
                cap: *cap,
                join: *join,
            };
            transforms::stroke(outlines(&input)?, &style)
                .map(|c| NodeOutput::from_contours(c.into_iter().map(Outline::Closed).collect()))
        }
        GeometryOp::UnionAll => {
            let regions = regions(outlines(&input)?)?;
            Ok(NodeOutput::from_contours(
                transforms::union_all(&regions)
                    .into_iter()
                    .map(Outline::Closed)
                    .collect(),
            ))
        }
        GeometryOp::Smooth { iterations } => {
            transforms::smooth(outlines(&input)?, *iterations).map(NodeOutput::from_contours)
        }
        GeometryOp::Densify { max_segment_length } => {
            transforms::densify(outlines(&input)?, *max_segment_length)
                .map(NodeOutput::from_contours)
        }
        GeometryOp::Close => outlines(&input)?
            .iter()
            .map(transforms::close_outline)
            .collect::<Result<Vec<_>, _>>()
            .map(NodeOutput::from_contours)
            .map_err(str::to_string),
    }
}

/// A boolean combination of two contour sets (`union`, `intersection`, …):
/// the step's own and the operand node's, each read as its regions'
/// union — the pipeline form of the `.contour` accessors, over the same
/// engine function.
pub(crate) fn execute_region_op(
    op: transforms::RegionOp,
    input: &NodeOutput,
    other: &NodeOutput,
) -> Result<NodeOutput, String> {
    fn side(output: &NodeOutput) -> Result<Vec<&Contour>, String> {
        let outlines = output
            .as_contours()
            .ok_or_else(|| "requires Contour input".to_string())?;
        regions(outlines)
    }
    let named = |e: String| format!("{}: {e}", op.name());
    let (a, b) = (side(input).map_err(named)?, side(other).map_err(named)?);
    Ok(NodeOutput::from_contours(
        transforms::boolean(&a, &b, op)
            .into_iter()
            .map(Outline::Closed)
            .collect(),
    ))
}

/// The step's contour set, or the domain error.
fn outlines(input: &NodeOutput) -> Result<&[Outline], String> {
    input
        .as_contours()
        .map(|c| c.as_slice())
        .ok_or_else(|| "requires Contour input".to_string())
}

/// Every member's region, borrowed, or the one refusal an open contour gets
/// from a region op (`Outline::region`); the step names it.
fn regions(outlines: &[Outline]) -> Result<Vec<&Contour>, String> {
    outlines
        .iter()
        .map(|o| o.region().map_err(str::to_string))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::execute_geometry_op;

    /// Structural coverage: every geometry op the graph builder can construct
    /// by resolving must actually execute. This is the geometry analog of
    /// view-buffer's `apply_op_coverage` probe.
    ///
    /// `GeometryOp` carries only variants the graph routes, so a variant
    /// `execute_geometry_op` cannot handle is a non-exhaustive-match compile
    /// error rather than a runtime string. What remains for this test is the
    /// other direction: that resolving and running each op *works*. Every
    /// registered op carries a sample, so a new geometry op is covered by
    /// registering it.
    #[test]
    fn every_graph_geometry_op_executes() {
        use crate::graph::node_output::NodeOutput;
        use crate::graph::step::GraphStep;
        use crate::params::ParamCtx;
        use view_buffer::geometry::contour::Outline;
        use view_buffer::geometry::Contour;
        use view_buffer::ops::Domain;
        use view_buffer::ViewBuffer;

        let sample_contours = || {
            NodeOutput::from_contours(vec![Outline::Closed(Contour::from_tuples(&[
                (0.0, 0.0),
                (10.0, 0.0),
                (10.0, 10.0),
                (0.0, 10.0),
            ]))])
        };
        let sample_buffer = || {
            NodeOutput::from_buffer(ViewBuffer::from_vec_with_shape(
                vec![0u8, 255, 255, 0],
                vec![2, 2, 1],
            ))
        };

        let mut executed = 0;
        for op in crate::ops::TypedOp::samples() {
            let name = op.name();
            let step = op
                .resolve(0, &ParamCtx::empty())
                .expect("a registered sample resolves");
            if let GraphStep::Geometry(geo) = step {
                let input = if geo.input_domain() == Domain::Buffer {
                    sample_buffer()
                } else if matches!(geo, view_buffer::GeometryOp::Close) {
                    // `close` takes a line: the region's boundary.
                    NodeOutput::from_contours(vec![Outline::Open(vec![
                        view_buffer::geometry::Point::new(0.0, 0.0),
                        view_buffer::geometry::Point::new(10.0, 0.0),
                        view_buffer::geometry::Point::new(10.0, 10.0),
                    ])])
                } else {
                    sample_contours()
                };
                if let Err(err) = execute_geometry_op(input, &geo) {
                    panic!("op '{name}' resolves to {geo:?} but does not execute: {err}");
                }
                executed += 1;
            }
        }
        // extract_contours, rasterize, four measures, four transforms.
        assert!(executed >= 10, "only {executed} geometry ops executed");
    }
}
