//! A contour column read straight from its Arrow arrays.
//!
//! **The one contour reader of the plugin.** Every consumer of a contour
//! column — the `.contour`/`.point` accessors, the set-level functions, the
//! pipeline's `contour` source and `label_reduce` — reads rows through
//! [`ContourColumn::row`], so the accepted forms, hole handling and error text
//! cannot diverge between them.
//!
//! It replaced a parser over `AnyValue`s, which built a `Series` per ring and
//! per point list: 98 allocations for a two-contour row that reading the
//! arrays does in 6 (`reading_a_row_allocates_only_its_contours`).
//!
//! Accepted forms, per row:
//! - a contour struct: an `exterior: List[{x, y}]` field and an optional
//!   `holes: List[List[{x, y}]]` (other fields, such as `is_closed`, are not
//!   read). A struct without `exterior` is refused, not guessed at;
//! - a bare `List[{x, y}]`, one contour without holes;
//! - a `List` of either, a contour set ([`Arity::Set`]), whose null elements
//!   are skipped.
//!
//! A point's coordinates are its `x`/`X` and `y`/`Y` fields, by name
//! ([`point_dtype_fields`]), and must be `Float64`. The column's layout is
//! resolved once, but a layout it cannot read is reported only for a
//! non-null row, so an all-null column of any dtype reads as nulls.

use polars::prelude::*;
use polars_arrow::array::{Array, ListArray, PrimitiveArray, StructArray};
use view_buffer::geometry::contour::{Contour, Point};

use crate::contour::point_dtype_fields;
use crate::geom_arity::{is_point_dtype, Arity};

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
    points: Result<(&'a PrimitiveArray<f64>, &'a PrimitiveArray<f64>), String>,
}

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
    /// `None` for a null row.
    pub(crate) fn row(&self, mut i: usize) -> PolarsResult<Option<Vec<Contour>>> {
        let Some((array, rows)) = self.chunks.iter().find(|(array, _)| {
            let here = i < array.len();
            if !here {
                i -= array.len();
            }
            here
        }) else {
            polars_bail!(OutOfBounds: "contour row out of bounds");
        };
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
        contours.map_err(|msg| polars_err!(ComputeError: "{}", msg))
    }

    /// Row `i`'s one contour, for a function of a single contour, or `None`
    /// for a null row. A contour set is refused: which of its contours is
    /// meant has no default.
    pub(crate) fn single(&self, i: usize) -> PolarsResult<Option<Contour>> {
        if self.arity == Arity::Set {
            polars_bail!(ComputeError:
                "expected one contour per row, got a contour set: .explode() it first"
            );
        }
        Ok(self.row(i)?.and_then(|mut v| v.pop()))
    }
}

impl Contours<'_> {
    /// The `j`th contour.
    fn get(&self, j: usize) -> Result<Contour, String> {
        match self {
            Contours::Ring(rings) => rings.get(j).map(Contour::new),
            Contours::Struct { exterior, holes } => {
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
                Ok(Contour::with_holes(exterior.get(j)?, holes))
            }
        }
    }
}

impl Rings<'_> {
    /// The `j`th ring's points. A null coordinate reads as 0.0, as it always
    /// has.
    fn get(&self, j: usize) -> Result<Vec<Point>, String> {
        let (start, end) = self.list.offsets().start_end(j);
        if start == end {
            return Ok(Vec::new());
        }
        let (x, y) = self.points.as_ref().map_err(Clone::clone)?;
        Ok((start..end)
            .map(|k| Point::new(x.get(k).unwrap_or(0.0), y.get(k).unwrap_or(0.0)))
            .collect())
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
            Ok(Contours::Struct { exterior, holes })
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
fn coordinates<'a>(
    array: &'a dyn Array,
    point: &DataType,
) -> Result<(&'a PrimitiveArray<f64>, &'a PrimitiveArray<f64>), String> {
    let DataType::Struct(fields) = point else {
        return Err("Expected Struct for point".to_string());
    };
    let st = downcast::<StructArray>(array, "a point struct")?;
    let [x, y] = point_dtype_fields().map(|names| {
        let axis = names[0];
        let idx = fields
            .iter()
            .position(|f| names.contains(&f.name().as_str()))
            .ok_or_else(|| format!("Point struct missing '{axis}' field"))?;
        if fields[idx].dtype() != &DataType::Float64 {
            return Err(format!("{axis} field must be f64"));
        }
        downcast::<PrimitiveArray<f64>>(st.values()[idx].as_ref(), "a coordinate")
    });
    Ok((x?, y?))
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
        let rows = vec![Some(vec![square(0.0, false), square(20.0, true)])];
        let col = written(rows, Arity::Set);
        let column = ContourColumn::new(&col);
        let (row, allocations) = crate::test_alloc::large_allocations(1, || column.row(0));
        assert!(row.unwrap().is_some());
        // 1 (row) + 1 (first exterior) + 1 (second exterior) + 1 (its holes)
        // + 2 (two hole rings).
        assert_eq!(allocations, 6);
    }
}
