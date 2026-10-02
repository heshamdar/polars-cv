//! The one declaration of each geometry struct — point, contour, bbox — the
//! geometry surfaces publish.
//!
//! `contour.rs`, `point.rs` and `geometry/schemas.py` each used to spell these
//! out — the `{x, y}` point in eight places, the contour `{exterior, holes,
//! is_closed}` layout in two, and a per-row bbox parser in two — with nothing
//! relating the copies. Dtypes, op names and enum variants all have a named
//! authority and a bidirectional parity guard; these surfaces had neither, so a
//! rename could land in one module and not the others, and the mismatch would
//! only show as a confusing dtype error at query time.
//!
//! Rust reads [`point_fields`] / [`contour_fields`] / [`bbox_fields`]; Python
//! holds `POINT_SCHEMA` / `CONTOUR_SCHEMA` / `BBOX_SCHEMA` to them through the
//! `point_schema` / `contour_schema` / `bbox_schema` FFIs — a runtime
//! accessor plus a parity test, rather than a generated file, so `polars_cv.geometry` still
//! imports with no compiled extension present.
//!
//! A contour *is* built from [`point_fields`] (its rings are lists of points),
//! so [`contour_fields`] composes them. A bbox is deliberately *not*: it begins
//! `x, y` by coincidence of meaning, not by sharing the point schema, and
//! chaining them would silently drag a point rename into the bbox wire format.

use polars::prelude::*;
use view_buffer::geometry::contour::BoundingBox;

/// The field names of an `{x, y}` point, in wire order.
///
/// Separate from [`point_fields`] so the FFI can publish the names without
/// constructing polars types, and so a rename has exactly one site.
pub(crate) const POINT_FIELD_NAMES: [&str; 2] = ["x", "y"];

/// The spellings an *input* point struct may give each coordinate, in axis
/// order, the canonical [`POINT_FIELD_NAMES`] first.
///
/// **The single authority for "is this a point?".** Read by the geometry
/// column readers (`geom_columns`), which parse by these names, and by
/// [`is_point_dtype`](crate::geom_arity::is_point_dtype), which decides from
/// the dtype whether a `List` is one contour's ring or a set of contours.
pub(crate) const POINT_FIELD_SPELLINGS: [&[&str]; 2] =
    [&[POINT_FIELD_NAMES[0], "X"], &[POINT_FIELD_NAMES[1], "Y"]];

/// The `{x, y}` fields a point-valued result publishes.
///
/// **This is the declaration.** Every other point struct in the plugin is
/// built from it — see the module docs for why a second one is a bug.
pub(crate) fn point_fields() -> Vec<Field> {
    vec![
        Field::new(PlSmallStr::from_static("x"), DataType::Float64),
        Field::new(PlSmallStr::from_static("y"), DataType::Float64),
    ]
}

/// The `{x, y}` struct dtype a point-valued result publishes.
pub(crate) fn point_struct_dtype() -> DataType {
    DataType::Struct(point_fields())
}

/// One point as a struct value matching [`point_struct_dtype`].
pub(crate) fn point_anyvalue(x: f64, y: f64) -> AnyValue<'static> {
    AnyValue::StructOwned(Box::new((
        vec![AnyValue::Float64(x), AnyValue::Float64(y)],
        point_fields(),
    )))
}

/// The field names of a contour `{exterior, holes, is_closed}`, in wire order.
///
/// Separate from [`contour_fields`] so the FFI can publish the names without
/// constructing polars types, and so a rename has exactly one site.
pub(crate) const CONTOUR_FIELD_NAMES: [&str; 3] = ["exterior", "holes", "is_closed"];

/// The `{exterior, holes, is_closed}` fields a contour-valued result publishes.
///
/// **This is the declaration.** `contour_array` (the column builder),
/// `contour_to_anyvalue` (its test oracle) and the test helper that
/// used to re-spell this layout both read it.
///
/// Composed from [`point_fields`] — a contour genuinely *is* built of points, so
/// `exterior` is a list of them and each hole a ring of them. A bbox, by
/// contrast, is deliberately *not* built from the point fields (see the module
/// docs); the composition here is honest, chaining a bbox to them would not be.
pub(crate) fn contour_fields() -> Vec<Field> {
    let point = point_struct_dtype();
    vec![
        Field::new(
            PlSmallStr::from_static("exterior"),
            DataType::List(Box::new(point.clone())),
        ),
        Field::new(
            PlSmallStr::from_static("holes"),
            DataType::List(Box::new(DataType::List(Box::new(point)))),
        ),
        Field::new(PlSmallStr::from_static("is_closed"), DataType::Boolean),
    ]
}

/// A column of contours, built straight into Arrow from the [`contour_fields`]
/// declaration: one flat `x`/`y` buffer for every point, with offsets for the
/// exterior rings, the holes and each hole's ring.
///
/// Replaces building one `AnyValue` per point and a sub-`Series` per ring, which
/// cost ~30 ms per ~90k points in the contour sink (CR-36). Each declared
/// field is matched by name to the array that fills it; a field this does not
/// know is an error, so a schema change cannot be half-applied here.
pub(crate) fn contour_array<'a, C: ContourParts + 'a>(
    contours: impl ExactSizeIterator<Item = &'a C> + Clone,
) -> PolarsResult<Box<dyn polars_arrow::array::Array>> {
    use polars_arrow::array::{Array, BooleanArray, ListArray, PrimitiveArray, StructArray};
    use polars_arrow::datatypes::ArrowDataType;
    use polars_arrow::offset::Offsets;

    let arrow = |dt: &DataType| dt.to_arrow(CompatLevel::newest());
    let points = |pts: &mut dyn Iterator<Item = &'a view_buffer::geometry::Point>| {
        let (xs, ys): (Vec<f64>, Vec<f64>) = pts.map(|p| (p.x, p.y)).unzip();
        let len = xs.len();
        StructArray::try_new(
            arrow(&point_struct_dtype()),
            len,
            vec![
                PrimitiveArray::from_vec(xs).boxed(),
                PrimitiveArray::from_vec(ys).boxed(),
            ],
            None,
        )
    };
    let list = |dtype: &DataType, lengths: Vec<usize>, values: Box<dyn Array>| {
        let offsets = Offsets::<i64>::try_from_lengths(lengths.into_iter())?;
        ListArray::<i64>::try_new(arrow(dtype), offsets.into(), values, None).map(|a| a.boxed())
    };

    let n = contours.len();
    let fields = contour_fields();
    let mut values: Vec<Box<dyn Array>> = Vec::with_capacity(fields.len());
    for field in &fields {
        let array = match field.name().as_str() {
            "exterior" => {
                let exterior = points(&mut contours.clone().flat_map(|c| c.exterior().iter()))?;
                let lengths = contours.clone().map(|c| c.exterior().len()).collect();
                list(field.dtype(), lengths, exterior.boxed())?
            }
            "holes" => {
                let ring_points =
                    points(&mut contours.clone().flat_map(|c| c.holes().iter().flatten()))?;
                let ring_dtype = match field.dtype() {
                    DataType::List(inner) => inner.as_ref().clone(),
                    other => polars_bail!(ComputeError:
                        "contour field 'holes' is declared as {other:?}, not a list of rings"),
                };
                let ring_lengths = contours
                    .clone()
                    .flat_map(|c| c.holes().iter().map(Vec::len))
                    .collect();
                let rings = list(&ring_dtype, ring_lengths, ring_points.boxed())?;
                let hole_counts = contours.clone().map(|c| c.holes().len()).collect();
                list(field.dtype(), hole_counts, rings)?
            }
            "is_closed" => BooleanArray::from_slice(
                contours.clone().map(|c| c.is_closed()).collect::<Vec<_>>(),
            )
            .boxed(),
            other => polars_bail!(ComputeError:
                "contour field '{other}' is declared in contour_fields but \
                 contour_array does not build it"),
        };
        values.push(array);
    }
    let dtype: ArrowDataType = arrow(&DataType::Struct(fields));
    Ok(StructArray::try_new(dtype, n, values, None)?.boxed())
}

/// What [`contour_array`] writes of a contour: its rings and whether it closes.
///
/// A [`Contour`](view_buffer::geometry::Contour) is a closed region; an
/// [`Outline`](view_buffer::geometry::contour::Outline) may be an open
/// polyline, which is published with `is_closed = false`.
pub(crate) trait ContourParts {
    fn exterior(&self) -> &[view_buffer::geometry::Point];
    fn holes(&self) -> &[Vec<view_buffer::geometry::Point>];
    fn is_closed(&self) -> bool;
}

impl ContourParts for view_buffer::geometry::Contour {
    fn exterior(&self) -> &[view_buffer::geometry::Point] {
        &self.exterior
    }
    fn holes(&self) -> &[Vec<view_buffer::geometry::Point>] {
        &self.holes
    }
    fn is_closed(&self) -> bool {
        true
    }
}

impl ContourParts for view_buffer::geometry::contour::Outline {
    fn exterior(&self) -> &[view_buffer::geometry::Point] {
        self.exterior()
    }
    fn holes(&self) -> &[Vec<view_buffer::geometry::Point>] {
        self.holes()
    }
    fn is_closed(&self) -> bool {
        self.is_closed()
    }
}

/// The field names of a bbox `{x, y, width, height}`, in wire order.
///
/// Independent of [`POINT_FIELD_NAMES`] on purpose: a bbox begins `x, y` by
/// coincidence of meaning, not by sharing the point schema, and chaining the
/// two would drag a point rename into the bbox wire format.
pub(crate) const BBOX_FIELD_NAMES: [&str; 4] = ["x", "y", "width", "height"];

/// The `{x, y, width, height}` fields a bbox-valued result publishes.
pub(crate) fn bbox_fields() -> Vec<Field> {
    vec![
        Field::new(PlSmallStr::from_static("x"), DataType::Float64),
        Field::new(PlSmallStr::from_static("y"), DataType::Float64),
        Field::new(PlSmallStr::from_static("width"), DataType::Float64),
        Field::new(PlSmallStr::from_static("height"), DataType::Float64),
    ]
}

/// The `{x, y, width, height}` struct dtype a bbox-valued result publishes.
pub(crate) fn bbox_struct_dtype() -> DataType {
    DataType::Struct(bbox_fields())
}

/// One bbox as a struct value matching [`bbox_struct_dtype`], or null.
pub(crate) fn bbox_anyvalue(bbox: Option<BoundingBox>) -> AnyValue<'static> {
    let Some(bbox) = bbox else {
        return AnyValue::Null;
    };
    AnyValue::StructOwned(Box::new((
        vec![
            AnyValue::Float64(bbox.x),
            AnyValue::Float64(bbox.y),
            AnyValue::Float64(bbox.width),
            AnyValue::Float64(bbox.height),
        ],
        bbox_fields(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_names_and_the_fields_agree() {
        // The FFI publishes `POINT_FIELD_NAMES` while Rust callers build
        // `point_fields()`. Nothing else compares the two, so a rename applied
        // to one alone would let Python check a schema Rust does not produce.
        let fields = point_fields();
        let built: Vec<&str> = fields.iter().map(|f| f.name().as_str()).collect();
        assert_eq!(built, POINT_FIELD_NAMES.to_vec());
    }

    #[test]
    fn a_point_value_matches_the_published_dtype() {
        let DataType::Struct(fields) = point_struct_dtype() else {
            panic!("point_struct_dtype must be a struct");
        };
        let AnyValue::StructOwned(payload) = point_anyvalue(1.0, 2.0) else {
            panic!("point_anyvalue must be an owned struct");
        };
        assert_eq!(payload.1, fields);
        assert_eq!(payload.0.len(), fields.len());
    }

    #[test]
    fn contour_names_and_fields_agree() {
        // Same guard the point struct has: the FFI publishes
        // `CONTOUR_FIELD_NAMES`, Rust builds `contour_fields()`, and nothing
        // else compares the two.
        let fields = contour_fields();
        let built: Vec<&str> = fields.iter().map(|f| f.name().as_str()).collect();
        assert_eq!(built, CONTOUR_FIELD_NAMES.to_vec());
    }

    #[test]
    fn a_contour_value_matches_the_published_dtype() {
        // `contour_to_anyvalue` (in `contour.rs`) builds its struct from
        // `contour_fields()`; hold its output to that declaration so a future
        // divergence fails here.
        use view_buffer::geometry::contour::{Contour, Point};
        let fields = contour_fields();
        let contour = Contour::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
        ]);
        let AnyValue::StructOwned(payload) = crate::contour::contour_to_anyvalue(&contour) else {
            panic!("contour_to_anyvalue must be an owned struct");
        };
        assert_eq!(payload.1, fields);
    }

    #[test]
    fn bbox_names_and_fields_agree() {
        let fields = bbox_fields();
        let built: Vec<&str> = fields.iter().map(|f| f.name().as_str()).collect();
        assert_eq!(built, BBOX_FIELD_NAMES.to_vec());
    }

    #[test]
    fn a_bbox_value_matches_the_published_dtype() {
        let DataType::Struct(fields) = bbox_struct_dtype() else {
            panic!("bbox_struct_dtype must be a struct");
        };
        let AnyValue::StructOwned(payload) =
            bbox_anyvalue(Some(BoundingBox::new(1.0, 2.0, 3.0, 4.0)))
        else {
            panic!("bbox_anyvalue must be an owned struct");
        };
        assert_eq!(payload.1, fields);
        assert_eq!(payload.0.len(), fields.len());
    }

    #[test]
    fn an_emitted_bbox_reads_back() {
        // Emit via the authority, read via the one bbox reader: the two agree.
        let source = BoundingBox::new(5.0, 6.0, 7.0, 8.0);
        let column = Series::from_any_values_and_dtype(
            "b".into(),
            &[bbox_anyvalue(Some(source)), bbox_anyvalue(None)],
            &bbox_struct_dtype(),
            true,
        )
        .unwrap();
        let reader = crate::geom_columns::BBoxColumn::new(&column);
        assert_eq!(reader.single(0).unwrap(), Some(source));
        assert_eq!(reader.single(1).unwrap(), None);
    }
}
