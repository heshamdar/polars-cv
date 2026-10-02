//! Geometry columns read straight from their Arrow arrays.
//!
//! **The one reader per geometry type.** Every consumer of a contour, point
//! or bbox column — the `.contour`/`.point`/`.bbox` functions, the
//! pipeline's `contour` source and `label_reduce` — reads rows through
//! [`ContourColumn`], [`PointColumn`] or [`BBoxColumn`], so the accepted
//! forms, null handling and error text cannot diverge between them.
//!
//! A value's numeric fields are found by name (a struct's field order means
//! nothing) and must be `Float64`; a null, NaN or infinite field in a
//! non-null value is an error naming the row, never a stand-in `0.0` or a
//! position nothing has.
//!
//! It replaced a parser over `AnyValue`s, which built a `Series` per ring and
//! per point list: 98 allocations for a two-contour row that reading the
//! arrays does in 6 (`reading_a_row_allocates_only_its_contours`).
//!
//! Accepted forms, per row:
//! - a contour struct: an `exterior: List[{x, y}]` field, an optional
//!   `holes: List[List[{x, y}]]` and an optional `is_closed: Boolean` (absent
//!   or null reads as closed, as absent or null `holes` reads as none; only an
//!   explicit `false` opens it). A struct without `exterior` is refused, not
//!   guessed at;
//! - a bare `List[{x, y}]`, one contour without holes;
//! - a `List` of either, a contour set ([`Arity::Set`]), whose null elements
//!   are skipped.
//!
//! An open contour (`is_closed = false`) is a polyline: it is read as an
//! [`Outline::Open`] by [`ContourColumn::outlines`], for the functions that
//! measure a boundary, and refused by [`ContourColumn::row`], which hands the
//! region functions (area, overlap, containment, rasterizing) a [`Contour`] —
//! a closed region — or nothing. An open contour with holes is refused by both.
//!
//! A point's coordinates are its `x`/`X` and `y`/`Y` fields
//! ([`POINT_FIELD_SPELLINGS`]). A bbox's are [`BBOX_FIELD_NAMES`]. The
//! column's layout is
//! resolved once, but a layout it cannot read is reported only for a
//! non-null row, so an all-null column of any dtype reads as nulls.

use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray, ListArray, PrimitiveArray, StructArray};
use view_buffer::geometry::contour::{Contour, Outline, Point};

use view_buffer::geometry::contour::BoundingBox;

use crate::geom_arity::{is_point_dtype, Arity, ReadContour};
use crate::geom_schema::{BBOX_FIELD_NAMES, POINT_FIELD_SPELLINGS};

/// A contour column's rows, each as the contours it holds.
pub(crate) struct ContourColumn<'a> {
    arity: Arity,
    /// Per chunk: the array and its reader (or why it cannot be read).
    chunks: Vec<(&'a dyn Array, Result<Rows<'a>, String>)>,
}

/// How one chunk's rows hold their contours.
enum Rows<'a> {
    /// One contour per row, the chunk's `index`th.
    Single(&'a dyn Array, Contours<'a>),
    /// A list of contours per row.
    Set(&'a ListArray<i64>, &'a dyn Array, Contours<'a>),
}

/// An array of contours, read by index.
enum Contours<'a> {
    Struct {
        exterior: Rings<'a>,
        /// Per contour, a list of hole rings (or why those rings cannot be
        /// read, which matters only to a contour that has one).
        holes: Option<(&'a ListArray<i64>, Result<Rings<'a>, String>)>,
        /// The `is_closed` flags, when the struct has the field.
        is_closed: Option<&'a BooleanArray>,
    },
    Ring(Rings<'a>),
}

/// An array of point rings, read by index.
///
/// The ring offsets are always readable; the points are checked only when a
/// ring that holds some is read, so an empty ring of any type (a `List(Null)`
/// that Polars inferred from `[]`) is simply empty.
struct Rings<'a> {
    list: &'a ListArray<i64>,
    points: Result<Points<'a>, String>,
}

/// A point struct array and its `x`/`y` coordinate arrays.
type Points<'a> = Fields<'a, 2>;

/// A struct array and its named `Float64` field arrays, with their names.
type Fields<'a, const N: usize> = (
    &'a StructArray,
    [(&'static str, &'a PrimitiveArray<f64>); N],
);

impl<'a> ContourColumn<'a> {
    /// Resolve `series`'s layout, chunk by chunk. Reads no rows.
    pub(crate) fn new(series: &'a Series) -> Self {
        let dtype = series.dtype();
        let arity = Arity::of(dtype);
        let chunks = series
            .chunks()
            .iter()
            .map(|chunk| {
                let chunk = chunk.as_ref();
                let rows = match (arity, dtype) {
                    (Arity::Set, DataType::List(elem)) => {
                        downcast::<ListArray<i64>>(chunk, "a contour set").and_then(|list| {
                            let values = list.values().as_ref();
                            Ok(Rows::Set(list, values, contours(values, elem)?))
                        })
                    }
                    _ => contours(chunk, dtype).map(|c| Rows::Single(chunk, c)),
                };
                (chunk, rows)
            })
            .collect();
        ContourColumn { arity, chunks }
    }

    /// Whether the column holds a set of contours per row.
    pub(crate) fn arity(&self) -> Arity {
        self.arity
    }

    /// Row `i`'s contours — exactly one for a single-contour column — or
    /// `None` for a null row. Every one is a closed region: an open contour is
    /// refused, since the region functions that read rows this way have
    /// nothing to measure on a polyline. Boundary functions read
    /// [`Self::outlines`] instead.
    pub(crate) fn row(&self, row: usize) -> PolarsResult<Option<Vec<Contour>>> {
        let Some(outlines) = self.outlines(row)? else {
            return Ok(None);
        };
        outlines
            .into_iter()
            .map(|outline| match outline {
                Outline::Closed(contour) => Ok(contour),
                Outline::Open(_) => Err(polars_err!(ComputeError:
                    "an open contour (is_closed = false) has no region, so this \
                     function — an area, overlap, containment or rasterizing one — \
                     cannot measure it (row {}). Close it first, or use a boundary \
                     measure (perimeter, distance, nearest point, hausdorff)",
                    row
                )),
            })
            .collect::<PolarsResult<_>>()
            .map(Some)
    }

    /// Row `i`'s contours as outlines — closed regions and open polylines
    /// alike — or `None` for a null row.
    pub(crate) fn outlines(&self, row: usize) -> PolarsResult<Option<Vec<Outline>>> {
        let ((array, rows), i) = locate(&self.chunks, row)?;
        let rows = match rows {
            Ok(rows) => rows,
            Err(_) if !array.is_valid(i) => return Ok(None),
            Err(msg) => polars_bail!(ComputeError: "{}", msg),
        };
        let contours = match rows {
            Rows::Single(array, contours) if array.is_valid(i) => {
                contours.get(i).map(|c| Some(vec![c]))
            }
            Rows::Set(list, values, contours) if list.is_valid(i) => {
                let (start, end) = list.offsets().start_end(i);
                (start..end)
                    .filter(|&j| values.is_valid(j))
                    .map(|j| contours.get(j))
                    .collect::<Result<_, _>>()
                    .map(Some)
            }
            _ => Ok(None),
        };
        contours.map_err(|msg| polars_err!(ComputeError: "{} (row {})", msg, row))
    }

    /// Row `i`'s one contour, for a function of a single contour, or `None`
    /// for a null row. A contour set is refused: which of its contours is
    /// meant has no default.
    ///
    /// Read as `T`: a region ([`Contour`]) or a boundary ([`Outline`]).
    pub(crate) fn single_as<T: ReadContour>(&self, i: usize) -> PolarsResult<Option<T>> {
        if self.arity == Arity::Set {
            polars_bail!(ComputeError:
                "expected one contour per row, got a contour set: .explode() it first"
            );
        }
        Ok(T::read(self, i)?.and_then(|mut v| v.pop()))
    }
}

impl Contours<'_> {
    /// The `j`th contour.
    fn get(&self, j: usize) -> Result<Outline, String> {
        match self {
            Contours::Ring(rings) => rings.get(j).map(|ring| Outline::Closed(Contour::new(ring))),
            Contours::Struct {
                exterior,
                holes,
                is_closed,
            } => {
                let holes = match holes {
                    None => Vec::new(),
                    Some((per_contour, rings)) => {
                        let (start, end) = per_contour.offsets().start_end(j);
                        let mut present = (start..end)
                            .filter(|&r| per_contour.values().is_valid(r))
                            .peekable();
                        match (present.peek(), rings) {
                            (None, _) => Vec::new(),
                            (Some(_), Err(msg)) => return Err(msg.clone()),
                            (Some(_), Ok(rings)) => {
                                present.map(|r| rings.get(r)).collect::<Result<_, _>>()?
                            }
                        }
                    }
                };
                // Unspecified — no field, or a null one, which is what a dict
                // without the key becomes under CONTOUR_SCHEMA — is closed, as
                // unspecified `holes` is none: only an explicit `false` opens.
                let closed = match is_closed {
                    Some(flags) if flags.is_valid(j) => flags.value(j),
                    _ => true,
                };
                match (closed, holes.is_empty()) {
                    (true, _) => Ok(Outline::Closed(Contour::with_holes(
                        exterior.get(j)?,
                        holes,
                    ))),
                    (false, true) => Ok(Outline::Open(exterior.get(j)?)),
                    (false, false) => Err(
                        "an open contour (is_closed = false) has holes, but a polyline \
                         bounds no region for a hole to be cut from"
                            .to_string(),
                    ),
                }
            }
        }
    }
}

impl Rings<'_> {
    /// The `j`th ring's points. A null point, or a point with a null
    /// coordinate, is refused: it has no position, and reading one as the
    /// origin (as this once did) moves the ring silently.
    fn get(&self, j: usize) -> Result<Vec<Point>, String> {
        let (start, end) = self.list.offsets().start_end(j);
        if start == end {
            return Ok(Vec::new());
        }
        let points = self.points.as_ref().map_err(Clone::clone)?;
        // Sized up front: collecting `Result`s loses the length hint, and a
        // growing vector reallocates as it goes.
        let mut ring = Vec::with_capacity(end - start);
        for k in start..end {
            if !points.0.is_valid(k) {
                return Err("a contour point is null".to_string());
            }
            let [x, y] = values(points, k, "contour point")?;
            ring.push(Point::new(x, y));
        }
        Ok(ring)
    }
}

fn downcast<'a, T: 'static>(array: &'a dyn Array, what: &str) -> Result<&'a T, String> {
    array
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| format!("internal: {what} is not stored as expected"))
}

/// An array of `dtype` contours: contour structs or bare rings.
fn contours<'a>(array: &'a dyn Array, dtype: &DataType) -> Result<Contours<'a>, String> {
    match dtype {
        DataType::Struct(fields) => {
            let st = downcast::<StructArray>(array, "a contour struct")?;
            let field = |name: &str| fields.iter().position(|f| f.name() == name);
            let exterior_idx = field("exterior")
                .ok_or_else(|| "Contour struct has no 'exterior' field".to_string())?;
            let DataType::List(point) = fields[exterior_idx].dtype() else {
                return Err("exterior field must be List[Point]".to_string());
            };
            let exterior = rings(st.values()[exterior_idx].as_ref(), point)?;
            let holes = match field("holes") {
                None => None,
                Some(idx) => {
                    let DataType::List(ring) = fields[idx].dtype() else {
                        return Err("holes field must be List[List[Point]]".to_string());
                    };
                    let per_contour =
                        downcast::<ListArray<i64>>(st.values()[idx].as_ref(), "holes")?;
                    let rings = match ring.as_ref() {
                        DataType::List(point) => rings(per_contour.values().as_ref(), point),
                        _ => Err("holes field must be List[List[Point]]".to_string()),
                    };
                    Some((per_contour, rings))
                }
            };
            let is_closed = match field("is_closed") {
                None => None,
                Some(idx) => match fields[idx].dtype() {
                    DataType::Boolean => Some(downcast::<BooleanArray>(
                        st.values()[idx].as_ref(),
                        "is_closed",
                    )?),
                    other => return Err(format!("is_closed field must be Boolean, got {other}")),
                },
            };
            Ok(Contours::Struct {
                exterior,
                holes,
                is_closed,
            })
        }
        DataType::List(point) if is_point_dtype(point) => Ok(Contours::Ring(rings(array, point)?)),
        other => Err(format!("Expected Struct or List for contour, got {other}")),
    }
}

/// An array of rings of `point` structs. Fails only when `array` is not a
/// list; a point layout it cannot read is kept for a ring that has points.
fn rings<'a>(array: &'a dyn Array, point: &DataType) -> Result<Rings<'a>, String> {
    let list = downcast::<ListArray<i64>>(array, "a point ring")?;
    Ok(Rings {
        list,
        points: coordinates(list.values().as_ref(), point),
    })
}

/// The `x` and `y` arrays of an array of `point` structs, by field name.
fn coordinates<'a>(array: &'a dyn Array, point: &DataType) -> Result<Points<'a>, String> {
    named_f64(array, point, POINT_FIELD_SPELLINGS, "Point")
}

/// The `Float64` field arrays of an array of `dtype` structs, the `i`th
/// found under any of `spellings[i]` (the first being its name in errors).
fn named_f64<'a, const N: usize>(
    array: &'a dyn Array,
    dtype: &DataType,
    spellings: [&[&'static str]; N],
    what: &str,
) -> Result<Fields<'a, N>, String> {
    let DataType::Struct(fields) = dtype else {
        return Err(format!("Expected Struct for {}", what.to_lowercase()));
    };
    let st = downcast::<StructArray>(array, what)?;
    let arrays = spellings.map(|names| {
        let name = names[0];
        let idx = fields
            .iter()
            .position(|f| names.contains(&f.name().as_str()))
            .ok_or_else(|| format!("{what} struct missing '{name}' field"))?;
        if fields[idx].dtype() != &DataType::Float64 {
            return Err(format!("{name} field must be f64"));
        }
        downcast::<PrimitiveArray<f64>>(st.values()[idx].as_ref(), "a coordinate")
            .map(|array| (name, array))
    });
    let mut found = Vec::with_capacity(N);
    for array in arrays {
        found.push(array?);
    }
    let found: [_; N] = found
        .try_into()
        .map_err(|_| "internal: field count".to_string())?;
    Ok((st, found))
}

/// The fields of struct `k`: a null field is an error, not a stand-in 0.0,
/// and so is a NaN or infinite one, which has no position either.
fn values<const N: usize>(
    fields: &Fields<'_, N>,
    k: usize,
    what: &str,
) -> Result<[f64; N], String> {
    let mut out = [0.0; N];
    for (slot, (name, array)) in out.iter_mut().zip(&fields.1) {
        let v = array
            .get(k)
            .ok_or_else(|| format!("a {what} has a null {name}"))?;
        if !v.is_finite() {
            return Err(format!("a {what} has a non-finite {name} ({v})"));
        }
        *slot = v;
    }
    Ok(out)
}

/// The chunk holding column row `row`, and the row's index within it.
fn locate<'c, 'a, T>(
    chunks: &'c [(&'a dyn Array, T)],
    row: usize,
) -> PolarsResult<(&'c (&'a dyn Array, T), usize)> {
    let mut i = row;
    for chunk in chunks {
        if i < chunk.0.len() {
            return Ok((chunk, i));
        }
        i -= chunk.0.len();
    }
    polars_bail!(OutOfBounds: "geometry row {} out of bounds", row)
}

/// A point column's rows, each as a [`Point`].
pub(crate) struct PointColumn<'a> {
    chunks: Vec<(&'a dyn Array, Result<Points<'a>, String>)>,
}

impl<'a> PointColumn<'a> {
    /// Resolve `series`'s layout, chunk by chunk. Reads no rows.
    pub(crate) fn new(series: &'a Series) -> Self {
        let chunks = series
            .chunks()
            .iter()
            .map(|chunk| (chunk.as_ref(), coordinates(chunk.as_ref(), series.dtype())))
            .collect();
        PointColumn { chunks }
    }

    /// Row `row`'s point, or `None` for a null row.
    pub(crate) fn get(&self, row: usize) -> PolarsResult<Option<Point>> {
        let ((array, points), i) = locate(&self.chunks, row)?;
        if !array.is_valid(i) {
            return Ok(None);
        }
        points
            .as_ref()
            .map_err(Clone::clone)
            .and_then(|points| values(points, i, "point"))
            .map(|[x, y]| Some(Point::new(x, y)))
            .map_err(|msg| polars_err!(ComputeError: "{} (row {})", msg, row))
    }
}

/// Bbox `k` of an array of bbox structs.
fn bbox_at(fields: &Fields<'_, 4>, k: usize) -> Result<BoundingBox, String> {
    values(fields, k, "bbox").map(|[x, y, w, h]| BoundingBox::new(x, y, w, h))
}

/// A bbox column's rows: one `{x, y, width, height}` struct per row, or a
/// list of them ([`Arity::Set`]), whose null elements are skipped.
pub(crate) struct BBoxColumn<'a> {
    arity: Arity,
    chunks: Vec<(&'a dyn Array, Result<BBoxRows<'a>, String>)>,
}

enum BBoxRows<'a> {
    Single(Fields<'a, 4>),
    Set(&'a ListArray<i64>, Fields<'a, 4>),
}

impl<'a> BBoxColumn<'a> {
    /// Resolve `series`'s layout, chunk by chunk. Reads no rows.
    pub(crate) fn new(series: &'a Series) -> Self {
        let spellings: [&[&'static str]; 4] =
            std::array::from_fn(|i| std::slice::from_ref(&BBOX_FIELD_NAMES[i]));
        let dtype = series.dtype();
        // A bbox is never a list, so any list is a set of them (unlike a
        // contour column, whose list may be one contour's ring).
        let arity = match dtype {
            DataType::List(_) => Arity::Set,
            _ => Arity::Single,
        };
        let chunks = series
            .chunks()
            .iter()
            .map(|chunk| {
                let chunk = chunk.as_ref();
                let rows = match (arity, dtype) {
                    (Arity::Set, DataType::List(elem)) => {
                        downcast::<ListArray<i64>>(chunk, "a bbox list").and_then(|list| {
                            named_f64(list.values().as_ref(), elem, spellings, "BBox")
                                .map(|f| BBoxRows::Set(list, f))
                        })
                    }
                    _ => named_f64(chunk, dtype, spellings, "BBox").map(BBoxRows::Single),
                };
                (chunk, rows)
            })
            .collect();
        BBoxColumn { arity, chunks }
    }

    /// Row `row`'s bboxes — exactly one for a single-bbox column — or `None`
    /// for a null row.
    pub(crate) fn row(&self, row: usize) -> PolarsResult<Option<Vec<BoundingBox>>> {
        let ((array, rows), i) = locate(&self.chunks, row)?;
        if !array.is_valid(i) {
            return Ok(None);
        }
        let read = match rows.as_ref().map_err(Clone::clone) {
            Err(msg) => Err(msg),
            Ok(BBoxRows::Single(fields)) => bbox_at(fields, i).map(|b| vec![b]),
            Ok(BBoxRows::Set(list, fields)) => {
                let (start, end) = list.offsets().start_end(i);
                (start..end)
                    .filter(|&k| fields.0.is_valid(k))
                    .map(|k| bbox_at(fields, k))
                    .collect()
            }
        };
        read.map(Some)
            .map_err(|msg| polars_err!(ComputeError: "{} (row {})", msg, row))
    }

    /// Row `row`'s one bbox, or `None` for a null row. A set is refused.
    pub(crate) fn single(&self, row: usize) -> PolarsResult<Option<BoundingBox>> {
        if self.arity == Arity::Set {
            polars_bail!(ComputeError: "expected one bbox per row, got a list of them");
        }
        Ok(self.row(row)?.and_then(|mut v| v.pop()))
    }
}

#[cfg(test)]
mod tests {
    use polars::prelude::*;
    use view_buffer::geometry::contour::{Contour, Point};

    use super::ContourColumn;
    use crate::geom_arity::{Arity, ContourOutput};

    fn square(x0: f64, with_hole: bool) -> Contour {
        let exterior = vec![
            Point::new(x0, 0.0),
            Point::new(x0 + 10.0, 0.0),
            Point::new(x0 + 10.0, 10.0),
            Point::new(x0, 10.0),
        ];
        if with_hole {
            Contour::with_holes(
                exterior,
                vec![
                    vec![
                        Point::new(x0 + 4.0, 4.0),
                        Point::new(x0 + 6.0, 4.0),
                        Point::new(x0 + 6.0, 6.0),
                    ],
                    vec![
                        Point::new(x0 + 1.0, 1.0),
                        Point::new(x0 + 2.0, 1.0),
                        Point::new(x0 + 2.0, 2.0),
                    ],
                ],
            )
        } else {
            Contour::new(exterior)
        }
    }

    /// A column in the canonical schema, written by the contour writer.
    fn written(rows: Vec<Option<Vec<Contour>>>, arity: Arity) -> Series {
        let elem = DataType::Struct(crate::geom_schema::contour_fields());
        Contour::column("c".into(), rows, arity, &elem).unwrap()
    }

    fn read_all(series: &Series) -> Vec<Option<Vec<Contour>>> {
        let column = ContourColumn::new(series);
        (0..series.len()).map(|i| column.row(i).unwrap()).collect()
    }

    #[test]
    fn single_contours_round_trip() {
        let rows = vec![
            Some(vec![square(0.0, true)]),
            None,
            Some(vec![square(20.0, false)]),
        ];
        assert_eq!(read_all(&written(rows.clone(), Arity::Single)), rows);
    }

    #[test]
    fn contour_sets_round_trip() {
        let rows = vec![
            Some(vec![square(0.0, true), square(20.0, false)]),
            None,
            Some(vec![]),
            Some(vec![square(40.0, true)]),
        ];
        assert_eq!(read_all(&written(rows.clone(), Arity::Set)), rows);
    }

    #[test]
    fn sliced_and_multi_chunk_columns_read_the_right_rows() {
        let rows: Vec<Option<Vec<Contour>>> = (0..6)
            .map(|i| Some(vec![square(i as f64 * 20.0, i % 2 == 0)]))
            .collect();
        let whole = written(rows.clone(), Arity::Set);
        assert_eq!(read_all(&whole.slice(2, 3)), rows[2..5].to_vec());

        let mut chunked = written(rows[..2].to_vec(), Arity::Single);
        chunked
            .append(&written(rows[2..].to_vec(), Arity::Single))
            .unwrap();
        assert_eq!(chunked.chunks().len(), 2);
        assert_eq!(read_all(&chunked), rows);
    }

    /// A struct Series of points with the given field names and values.
    fn points(names: [&str; 2], xs: &[Option<f64>], ys: &[Option<f64>]) -> Series {
        let a = Series::new(names[0].into(), xs);
        let b = Series::new(names[1].into(), ys);
        StructChunked::from_series("p".into(), xs.len(), [a, b].iter())
            .unwrap()
            .into_series()
    }

    /// A column of bare rings (one `List[{x, y}]` per row).
    fn rings(rows: Vec<Option<Series>>) -> Series {
        let dtype = rows.iter().flatten().next().unwrap().dtype().clone();
        let values: Vec<AnyValue> = rows
            .into_iter()
            .map(|r| r.map_or(AnyValue::Null, AnyValue::List))
            .collect();
        Series::from_any_values_and_dtype(
            "r".into(),
            &values,
            &DataType::List(Box::new(dtype)),
            true,
        )
        .unwrap()
    }

    #[test]
    fn a_bare_ring_is_one_contour_with_its_points_read_by_name() {
        let ring = |names| points(names, &[Some(1.0), Some(2.0)], &[Some(3.0), Some(4.0)]);
        let expected = Contour::new(vec![Point::new(1.0, 3.0), Point::new(2.0, 4.0)]);
        for names in [["x", "y"], ["X", "Y"]] {
            let col = rings(vec![Some(ring(names)), None]);
            assert_eq!(
                read_all(&col),
                vec![Some(vec![expected.clone()]), None],
                "{names:?}"
            );
        }
        // Field order does not decide which coordinate is which.
        let swapped = points(["y", "x"], &[Some(3.0), Some(4.0)], &[Some(1.0), Some(2.0)]);
        assert_eq!(
            read_all(&rings(vec![Some(swapped)])),
            vec![Some(vec![expected])]
        );
    }

    #[test]
    fn a_set_of_bare_rings_and_its_null_elements() {
        let ring = points(["x", "y"], &[Some(1.0), Some(2.0)], &[Some(3.0), Some(4.0)]);
        let set_dtype = DataType::List(Box::new(ring.dtype().clone()));
        let inner = Series::from_any_values_and_dtype(
            "s".into(),
            &[
                AnyValue::List(ring.clone()),
                AnyValue::Null,
                AnyValue::List(ring),
            ],
            &set_dtype,
            true,
        )
        .unwrap();
        let col = Series::from_any_values_and_dtype(
            "c".into(),
            &[AnyValue::List(inner)],
            &DataType::List(Box::new(set_dtype)),
            true,
        )
        .unwrap();
        let one = Contour::new(vec![Point::new(1.0, 3.0), Point::new(2.0, 4.0)]);
        // A null contour inside a set is skipped, as a set of what is there.
        assert_eq!(read_all(&col), vec![Some(vec![one.clone(), one])]);
    }

    #[test]
    fn a_struct_without_an_exterior_is_refused_on_a_non_null_row() {
        let ring = points(["x", "y"], &[Some(1.0)], &[Some(2.0)]);
        let renamed = StructChunked::from_series(
            "c".into(),
            1,
            [Series::new("points".into(), &[AnyValue::List(ring)])].iter(),
        )
        .unwrap()
        .into_series();
        let err = ContourColumn::new(&renamed).row(0).unwrap_err().to_string();
        assert!(err.contains("'exterior'"), "{err}");
        // A null row of an unreadable column is still just null.
        let nulls = Series::full_null("c".into(), 2, renamed.dtype());
        assert_eq!(read_all(&nulls), vec![None, None]);
    }

    /// A point with no coordinate is not a point at the origin: a null `x`
    /// or `y`, or a null point in a ring, is refused, naming the row.
    #[test]
    fn a_null_coordinate_is_refused() {
        let ok = points(["x", "y"], &[Some(1.0), Some(2.0)], &[Some(3.0), Some(4.0)]);
        for (xs, ys) in [
            ([Some(1.0), None], [Some(3.0), Some(4.0)]),
            ([Some(1.0), Some(2.0)], [None, Some(4.0)]),
        ] {
            let col = rings(vec![Some(ok.clone()), Some(points(["x", "y"], &xs, &ys))]);
            let column = ContourColumn::new(&col);
            assert!(column.row(0).is_ok());
            let err = column.row(1).unwrap_err().to_string();
            assert!(err.contains("null") && err.contains("row 1"), "{err}");
        }
        // A null point struct in the ring.
        let null_point =
            Series::from_any_values_and_dtype("p".into(), &[AnyValue::Null], ok.dtype(), true)
                .unwrap();
        let mut ring = ok.clone();
        ring.append(&null_point).unwrap();
        let err = ContourColumn::new(&rings(vec![Some(ring)]))
            .row(0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("null"), "{err}");
    }

    /// A coordinate that is NaN or infinite has no position, so it is
    /// refused like a null one, by every consumer: a NaN vertex panicked
    /// `convex_hull`, `iou`, `dice` and `simplify` in the geometry library,
    /// made `hausdorff_distance` `f64::MAX`, and left the bbox, winding and
    /// convexity finite but wrong.
    #[test]
    fn a_non_finite_coordinate_is_refused() {
        let ok = points(["x", "y"], &[Some(1.0), Some(2.0)], &[Some(3.0), Some(4.0)]);
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            // The bad coordinate is in the second point (row 1 as points).
            for (xs, ys) in [
                ([Some(1.0), Some(bad)], [Some(3.0), Some(4.0)]),
                ([Some(1.0), Some(2.0)], [Some(3.0), Some(bad)]),
            ] {
                let col = rings(vec![Some(ok.clone()), Some(points(["x", "y"], &xs, &ys))]);
                let column = ContourColumn::new(&col);
                assert!(column.row(0).is_ok());
                let err = column.row(1).unwrap_err().to_string();
                assert!(err.contains("non-finite") && err.contains("row 1"), "{err}");
                let pts = points(["x", "y"], &xs, &ys);
                let err = PointColumn::new(&pts).get(1).unwrap_err().to_string();
                assert!(err.contains("non-finite"), "{bad}: {err}");
            }
        }
    }

    #[test]
    fn coordinates_that_are_not_f64_are_refused() {
        let ring = Series::new("x".into(), &[1i64]);
        let pts = StructChunked::from_series(
            "p".into(),
            1,
            [ring, Series::new("y".into(), &[2i64])].iter(),
        )
        .unwrap()
        .into_series();
        let err = ContourColumn::new(&rings(vec![Some(pts)]))
            .row(0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("f64"), "{err}");
    }

    /// A contour struct built from empty lists (`{"holes": []}` in Python)
    /// gets a `holes: List(Null)` field: an empty ring list of no particular
    /// type. That reads as no holes; a ring layout is checked only against a
    /// ring that has points to read.
    #[test]
    fn empty_rings_of_any_type_read_as_empty() {
        let ring = points(["x", "y"], &[Some(1.0), Some(2.0)], &[Some(3.0), Some(4.0)]);
        let empty = Series::new_empty("".into(), &DataType::Null);
        let contour = |exterior: Series, holes: Series| {
            StructChunked::from_series(
                "c".into(),
                1,
                [
                    Series::new("exterior".into(), &[AnyValue::List(exterior)]),
                    Series::new("holes".into(), &[AnyValue::List(holes)]),
                ]
                .iter(),
            )
            .unwrap()
            .into_series()
        };
        let col = contour(ring.clone(), empty.clone());
        assert!(
            matches!(col.dtype(), DataType::Struct(f) if f[1].dtype() == &DataType::List(Box::new(DataType::Null)))
        );
        let expected = Contour::new(vec![Point::new(1.0, 3.0), Point::new(2.0, 4.0)]);
        assert_eq!(read_all(&col), vec![Some(vec![expected])]);

        let col = contour(empty.clone(), empty);
        assert_eq!(read_all(&col), vec![Some(vec![Contour::new(vec![])])]);

        // A hole that does hold something must be a ring of points.
        let bad_hole = Series::new(
            "".into(),
            &[AnyValue::List(Series::new("".into(), &[1i64]))],
        );
        let err = ContourColumn::new(&contour(ring, bad_hole))
            .row(0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Expected Struct for point"), "{err}");
    }

    /// Reading a row allocates the contours and nothing else: the row's
    /// `Vec<Contour>`, and one `Vec<Point>` per ring (plus the `holes` vector
    /// of a contour that has holes). No per-row `Series`/`AnyValue` scaffolding.
    #[test]
    fn reading_a_row_allocates_only_its_contours() {
        // A 40-point exterior, so a ring vector that grew instead of being
        // sized up front would show as extra reallocations.
        let big = Contour::new(
            (0..40)
                .map(|i| Point::new(i as f64, (i * i) as f64))
                .collect(),
        );
        let rows = vec![Some(vec![big, square(20.0, true)])];
        let col = written(rows, Arity::Set);
        let column = ContourColumn::new(&col);
        let (row, allocations) = crate::test_alloc::large_allocations(1, || column.row(0));
        assert!(row.unwrap().is_some());
        // 1 (row) + 1 (first exterior) + 1 (second exterior) + 1 (its holes)
        // + 2 (two hole rings).
        assert_eq!(allocations, 6);
    }

    // ---- points and bboxes -------------------------------------------------

    use super::{BBoxColumn, PointColumn};
    use view_buffer::geometry::contour::BoundingBox;

    fn read_points(series: &Series) -> PolarsResult<Vec<Option<Point>>> {
        let column = PointColumn::new(series);
        (0..series.len()).map(|i| column.get(i)).collect()
    }

    #[test]
    fn points_are_read_by_field_name() {
        let expected = vec![Some(Point::new(1.0, 3.0)), Some(Point::new(2.0, 4.0))];
        for names in [["x", "y"], ["X", "Y"]] {
            let col = points(names, &[Some(1.0), Some(2.0)], &[Some(3.0), Some(4.0)]);
            assert_eq!(read_points(&col).unwrap(), expected, "{names:?}");
        }
        // Position does not decide the axis: `{y, x}` is not `{x, y}`.
        let swapped = points(["y", "x"], &[Some(3.0), Some(4.0)], &[Some(1.0), Some(2.0)]);
        assert_eq!(read_points(&swapped).unwrap(), expected);
    }

    #[test]
    fn a_null_point_row_is_none_and_a_null_coordinate_an_error() {
        let col = points(["x", "y"], &[Some(1.0), Some(2.0)], &[Some(3.0), Some(4.0)]);
        let nulls = Series::full_null("p".into(), 1, col.dtype());
        let mut with_null_row = col.clone();
        with_null_row.append(&nulls).unwrap();
        assert_eq!(with_null_row.chunks().len(), 2);
        assert_eq!(
            read_points(&with_null_row).unwrap(),
            vec![Some(Point::new(1.0, 3.0)), Some(Point::new(2.0, 4.0)), None]
        );
        let col = points(["x", "y"], &[Some(1.0), None], &[Some(3.0), Some(4.0)]);
        let err = PointColumn::new(&col).get(1).unwrap_err().to_string();
        assert!(err.contains("null x") && err.contains("row 1"), "{err}");
    }

    #[test]
    fn a_point_needs_f64_x_and_y() {
        let ints = StructChunked::from_series(
            "p".into(),
            1,
            [
                Series::new("x".into(), &[1i64]),
                Series::new("y".into(), &[2i64]),
            ]
            .iter(),
        )
        .unwrap()
        .into_series();
        let err = PointColumn::new(&ints).get(0).unwrap_err().to_string();
        assert!(err.contains("f64"), "{err}");
        let only_x =
            StructChunked::from_series("p".into(), 1, [Series::new("x".into(), &[1.0])].iter())
                .unwrap()
                .into_series();
        let err = PointColumn::new(&only_x).get(0).unwrap_err().to_string();
        assert!(err.contains("'y'"), "{err}");
    }

    /// A bbox struct column with fields in the given order.
    fn bboxes(names: [&str; 4], values: [&[Option<f64>]; 4]) -> Series {
        let fields: Vec<Series> = names
            .iter()
            .zip(values)
            .map(|(n, v)| Series::new((*n).into(), v))
            .collect();
        StructChunked::from_series("b".into(), values[0].len(), fields.iter())
            .unwrap()
            .into_series()
    }

    #[test]
    fn bboxes_are_read_by_field_name_single_or_as_a_set() {
        let col = bboxes(
            ["height", "width", "y", "x"],
            [&[Some(4.0)], &[Some(3.0)], &[Some(2.0)], &[Some(1.0)]],
        );
        let expected = BoundingBox::new(1.0, 2.0, 3.0, 4.0);
        let column = BBoxColumn::new(&col);
        assert_eq!(column.single(0).unwrap(), Some(expected));
        assert_eq!(column.row(0).unwrap(), Some(vec![expected]));

        let set = Series::new("s".into(), &[AnyValue::List(col.clone()), AnyValue::Null]);
        let column = BBoxColumn::new(&set);
        assert_eq!(column.row(0).unwrap(), Some(vec![expected]));
        assert_eq!(column.row(1).unwrap(), None);
        assert!(column.single(0).is_err(), "a set has no single bbox");
    }

    #[test]
    fn a_bbox_with_a_null_or_missing_field_is_refused() {
        let col = bboxes(
            ["x", "y", "width", "height"],
            [&[Some(1.0)], &[Some(2.0)], &[None], &[Some(4.0)]],
        );
        let err = BBoxColumn::new(&col).single(0).unwrap_err().to_string();
        assert!(err.contains("null width") && err.contains("row 0"), "{err}");
        let three = StructChunked::from_series(
            "b".into(),
            1,
            ["x", "y", "width"]
                .iter()
                .map(|n| Series::new((*n).into(), &[1.0]))
                .collect::<Vec<_>>()
                .iter(),
        )
        .unwrap()
        .into_series();
        let err = BBoxColumn::new(&three).single(0).unwrap_err().to_string();
        assert!(err.contains("'height'"), "{err}");
    }
}
