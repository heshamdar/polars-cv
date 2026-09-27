//! One contour per row, or a set of them — decided once, for both halves.
//!
//! A geometry column carries either a single `Struct` matching `CONTOUR_SCHEMA`
//! or a `List` of them (`CONTOUR_SET_SCHEMA`, what `extract_contours()` emits).
//! Every `.contour` accessor has to work over both, and each one has *two*
//! halves that must agree about which it is looking at:
//!
//! | half | what it decides |
//! |------|-----------------|
//! | the `output_type_func` | the Polars dtype the accessor publishes at plan time |
//! | the function body | the Series it actually produces |
//!
//! Nothing forces those two to agree. Before this module the accessors declared
//! a *constant* output type (`#[polars_expr(output_type=Float64)]`, fifteen
//! times) and called [`crate::contour::parse_contour`] directly, which rejects a
//! set outright — so the whole question was answered by failing. Answering it
//! per accessor instead would be fifteen opportunities to declare `Float64` and
//! build a `List(Float64)`, and `test_schema_parity_namespaces` exists because
//! exactly that class of divergence has shipped here before.
//!
//! So the arity is one value, read from the **column dtype** — never from a row
//! — and it drives both halves:
//!
//! - [`Arity::of`] reads it, using the same point-vs-contour test the value-level
//!   parser uses, so the dispatch cannot admit something the parser then rejects.
//! - [`elementwise_field`] / [`binary_field`] wrap the per-contour element type
//!   for the declaration.
//! - [`map_contours`] / [`zip_contours`] wrap the per-contour *results* with the
//!   same [`Arity::wrap`], and are the only decode path the accessors use.
//!
//! The [`contour_accessor!`](crate::contour_accessor) macro then emits both
//! halves from a single `-> <elem>` declaration, so an accessor cannot state one
//! and mean the other.
//!
//! Reading the arity from the dtype rather than the value is what keeps plan ==
//! exec: the `output_type_func` is only handed [`Field`]s, so a row-level
//! decision would be one the declaration could not have made.

use polars::prelude::*;

use view_buffer::geometry::contour::Contour;

use crate::geom_columns::ContourColumn;
use crate::geom_params::GeomParams;
use crate::geom_schema::POINT_FIELD_SPELLINGS;
use crate::row_split::CallTracker;

/// Whether a geometry column holds one contour per row or a set per row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Arity {
    /// One `Struct` (or bare ring) per row.
    Single,
    /// A `List` of contours per row — `CONTOUR_SET_SCHEMA`.
    Set,
}

impl Arity {
    /// Read the arity from a column's dtype.
    ///
    /// A `List` whose elements are *point* structs is one contour's ring, not a
    /// set; anything else in a `List` is a set. Told apart by the element dtype
    /// rather than by trying one and falling back, because a fallback has to
    /// guess — and guessing wrong on a contour set is what used to surface as
    /// `Point struct missing 'x' field`.
    pub(crate) fn of(dtype: &DataType) -> Self {
        match dtype {
            DataType::List(inner) if !is_point_dtype(inner) => Arity::Set,
            _ => Arity::Single,
        }
    }

    /// The dtype of a single contour within a column of this arity.
    ///
    /// What a transform has to build its elements as: for a set that is the
    /// *inner* struct, not the column's own `List(...)`. Passing the outer dtype
    /// is what `build_contour_series(…, series.dtype())` did, and it is why the
    /// transforms could not have been made list-aware by relaxing the parser
    /// alone.
    pub(crate) fn elem_dtype(dtype: &DataType) -> DataType {
        match Arity::of(dtype) {
            Arity::Set => match dtype {
                DataType::List(inner) => (**inner).clone(),
                // `Arity::of` only answers `Set` for a `List`.
                other => other.clone(),
            },
            Arity::Single => dtype.clone(),
        }
    }

    /// Wrap a per-contour result type for a column of this arity.
    pub(crate) fn wrap(self, elem: DataType) -> DataType {
        match self {
            Arity::Set => DataType::List(Box::new(elem)),
            Arity::Single => elem,
        }
    }

    /// The arity of a result computed from two operands.
    ///
    /// Broadcasting: a set on either side makes the result a set, so
    /// `dets.iou(gt)` and `gt.iou(dets)` agree. Set × set is refused by
    /// [`zip_contours`] before this is reached.
    fn combine(self, other: Arity) -> Arity {
        match (self, other) {
            (Arity::Single, Arity::Single) => Arity::Single,
            _ => Arity::Set,
        }
    }
}

/// Does this dtype describe a point (`{x, y}`) rather than a contour?
///
/// Reads the field names from [`POINT_FIELD_SPELLINGS`] — the same names the point
/// parser reads — so the dispatch above cannot admit something the parser then
/// rejects.
pub(crate) fn is_point_dtype(dtype: &DataType) -> bool {
    let DataType::Struct(fields) = dtype else {
        return false;
    };
    POINT_FIELD_SPELLINGS
        .iter()
        .all(|wanted| fields.iter().any(|f| wanted.contains(&f.name().as_str())))
}

/// The declared output type of a one-operand accessor whose per-contour result
/// is `elem`.
///
/// Pairs with [`map_contours`]: both call [`Arity::wrap`], so the declaration
/// and the data cannot disagree about the nesting.
pub(crate) fn elementwise_field(
    input_fields: &[Field],
    name: &'static str,
    elem: DataType,
) -> PolarsResult<Field> {
    let input = input_fields
        .first()
        .ok_or_else(|| polars_err!(ComputeError: "{} takes a contour column", name))?;
    Ok(Field::new(
        input.name().clone(),
        Arity::of(input.dtype()).wrap(elem),
    ))
}

/// The declared output type of a two-operand accessor. See [`zip_contours`].
pub(crate) fn binary_field(
    input_fields: &[Field],
    name: &'static str,
    elem: DataType,
) -> PolarsResult<Field> {
    let [a, b, ..] = input_fields else {
        polars_bail!(ComputeError: "{} takes two contour columns", name);
    };
    let arity = Arity::of(a.dtype()).combine(Arity::of(b.dtype()));
    Ok(Field::new(a.name().clone(), arity.wrap(elem)))
}

/// A per-contour result type, and how a column of them is assembled.
///
/// `rows` holds one entry per row: `None` for a null row, otherwise that row's
/// per-contour results (exactly one under [`Arity::Single`]). The column must
/// have dtype `arity.wrap(elem)`, the type the accessor declared.
pub(crate) trait ContourOutput: Sized {
    fn column(
        name: PlSmallStr,
        rows: Vec<Option<Vec<Self>>>,
        arity: Arity,
        elem: &DataType,
    ) -> PolarsResult<Series>;
}

/// Measures, points, boxes and lists: small per-contour values.
impl ContourOutput for AnyValue<'static> {
    fn column(
        name: PlSmallStr,
        rows: Vec<Option<Vec<Self>>>,
        arity: Arity,
        elem: &DataType,
    ) -> PolarsResult<Series> {
        let values = rows
            .into_iter()
            .map(|row| match row {
                None => Ok(AnyValue::Null),
                // `Single` ran exactly one contour, so its result *is* the row.
                Some(results) if arity == Arity::Single => {
                    Ok(results.into_iter().next().unwrap_or(AnyValue::Null))
                }
                Some(results) => Series::from_any_values_and_dtype(
                    PlSmallStr::from_static("item"),
                    &results,
                    elem,
                    true,
                )
                .map(AnyValue::List),
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        Series::from_any_values_and_dtype(name, &values, &arity.wrap(elem.clone()), true)
    }
}

/// Transforms: a contour per contour, built straight into Arrow through
/// [`crate::geom_schema::contour_array`] rather than as an `AnyValue` with a
/// sub-`Series` per ring, which made `.contour.translate()` cost as much as
/// extracting the contours in the first place (CR-36).
impl ContourOutput for Contour {
    fn column(
        name: PlSmallStr,
        rows: Vec<Option<Vec<Self>>>,
        arity: Arity,
        elem: &DataType,
    ) -> PolarsResult<Series> {
        use polars_arrow::array::{Array, ListArray};
        use polars_arrow::bitmap::Bitmap;
        use polars_arrow::offset::Offsets;

        let validity: Option<Bitmap> = rows
            .iter()
            .any(Option::is_none)
            .then(|| rows.iter().map(Option::is_some).collect());
        let array: Box<dyn Array> = match arity {
            Arity::Single => {
                // One slot per row; a null row's slot holds an empty contour
                // under a cleared validity bit.
                let empty = Contour::new(Vec::new());
                let slots: Vec<&Contour> = rows
                    .iter()
                    .map(|row| match row {
                        Some(results) if results.len() == 1 => Ok(&results[0]),
                        Some(results) => Err(polars_err!(ComputeError:
                            "internal: a single-contour row produced {} results",
                            results.len()
                        )),
                        None => Ok(&empty),
                    })
                    .collect::<PolarsResult<_>>()?;
                crate::geom_schema::contour_array(slots.iter().copied())?.with_validity(validity)
            }
            Arity::Set => {
                let all: Vec<&Contour> = rows.iter().flatten().flatten().collect();
                let values = crate::geom_schema::contour_array(all.iter().copied())?;
                let lengths = rows.iter().map(|r| r.as_ref().map_or(0, Vec::len));
                let offsets = Offsets::<i64>::try_from_lengths(lengths)?;
                let dtype = ListArray::<i64>::default_datatype(values.dtype().clone());
                ListArray::<i64>::try_new(dtype, offsets.into(), values, validity)?.boxed()
            }
        };
        // The declared type governs; a column whose element layout differs
        // from the canonical contour (e.g. one without holes) is cast exactly as
        // the `AnyValue` path's strict construction would have been.
        let declared = arity.wrap(elem.clone());
        let series = Series::from_arrow(name, array)?;
        if series.dtype() == &declared {
            Ok(series)
        } else {
            series.strict_cast(&declared)
        }
    }
}

/// Run `compute` for every contour in every row, in the column's own arity,
/// each row's work under the call's
/// [`NullParamPolicy`](crate::params::NullParamPolicy).
///
/// **The one decode-and-assemble path for the single-column `.contour`
/// accessors.** Rows are read through [`ContourColumn`] in the arity the
/// column's dtype declares, so no accessor is free to disagree with what its
/// `output_type_func` declared.
///
/// The policy is a *row*-level decision, so an accessor with per-row parameters
/// wraps the row rather than each contour: `on_null("null")` nulls the whole
/// row, exactly as a null input contour already does. Routing it through here
/// keeps it from being re-implemented per accessor — the job `contour_row` did
/// for the single-contour accessors, which this replaces.
pub(crate) fn map_contours<R: ContourOutput + Send>(
    series: &Series,
    params: &GeomParams,
    calls: &CallTracker,
    elem: DataType,
    compute: impl Fn(&Contour, &GeomParams, usize) -> PolarsResult<R> + Sync,
) -> PolarsResult<Series> {
    let column = ContourColumn::new(series);
    let rows = params.map_rows(calls, series.len(), |params, i| {
        let Some(contours) = column.row(i)? else {
            return Ok(None);
        };
        contours
            .iter()
            .map(|contour| compute(contour, params, i))
            .collect::<PolarsResult<Vec<R>>>()
            .map(Some)
    })?;
    R::column(series.name().clone(), rows, column.arity(), &elem)
}

/// Run `compute` over two contour columns, broadcasting a single against a set.
///
/// `Set × Single` and `Single × Set` broadcast — one result per contour in the
/// set — so `dets.iou(gt)` and `gt.iou(dets)` mean the same thing. `Set × Set`
/// is **refused**: the two readings (an N×M matrix, or an index-wise pairing)
/// are both plausible and mean different things, and `pairwise_iou` already
/// provides the first. Guessing between them is the kind of silent choice this
/// crate's fallbacks have been removed for.
///
/// The refusal reads dtypes, so it fires before any row is parsed rather than
/// part-way through a batch.
pub(crate) fn zip_contours<R: ContourOutput + Send>(
    a: &Series,
    b: &Series,
    params: &GeomParams,
    calls: &CallTracker,
    name: &'static str,
    elem: DataType,
    compute: impl Fn(&Contour, &Contour, usize) -> PolarsResult<R> + Sync,
) -> PolarsResult<Series> {
    let (a_arity, b_arity) = (Arity::of(a.dtype()), Arity::of(b.dtype()));
    if a_arity == Arity::Set && b_arity == Arity::Set {
        polars_bail!(ComputeError:
            "{} received a contour set on both sides, which has two different \
             meanings and no default: use .contour.pairwise_iou() for the N x M \
             matrix over both sets, or .explode() one side to pair them row by \
             row. One side may be a set; both may not.",
            name
        );
    }
    let arity = a_arity.combine(b_arity);
    let (a_column, b_column) = (ContourColumn::new(a), ContourColumn::new(b));
    let row = |i: usize| -> PolarsResult<Option<Vec<R>>> {
        let (Some(left), Some(right)) = (a_column.row(i)?, b_column.row(i)?) else {
            return Ok(None);
        };
        // Exactly one side is a set, so the other side's single contour is
        // repeated against it; when neither is, both are one-element. A set
        // against an empty single side yields an empty set.
        let results = match (a_arity, b_arity) {
            (Arity::Set, _) => match right.first() {
                Some(single) => left.iter().map(|c| compute(c, single, i)).collect(),
                None => Ok(Vec::new()),
            },
            (_, Arity::Set) => match left.first() {
                Some(single) => right.iter().map(|c| compute(single, c, i)).collect(),
                None => Ok(Vec::new()),
            },
            (Arity::Single, Arity::Single) => compute(&left[0], &right[0], i).map(|r| vec![r]),
        }?;
        Ok(Some(results))
    };
    let rows = params.map_rows(calls, a.len(), |_, i| row(i))?;
    R::column(a.name().clone(), rows, arity, &elem)
}

/// Declare a `.contour` accessor's output type and body from one statement.
///
/// **The point of the macro is that `-> <elem>` is written once and used
/// twice** — once to build the `output_type_func` via [`elementwise_field`] /
/// [`binary_field`], once to drive [`map_contours`] / [`zip_contours`]. An
/// accessor therefore cannot publish one element type and produce another,
/// which is the divergence `test_schema_parity_namespaces` exists to catch and
/// which fifteen hand-written `#[polars_expr(output_type=...)]` attributes were
/// fifteen chances to introduce.
///
/// The element expression may read `input`, the primary input column's
/// `&DataType`. That is what lets the transforms say
/// `Arity::elem_dtype(input)` — "a contour of whatever shape came in" — in the
/// same breath as the measures say `DataType::Float64`.
///
/// Both function names are spelled by the caller because Rust cannot build an
/// identifier from another without a proc-macro dependency; the binding this
/// macro provides is over the *element type*, not the names.
///
/// Each form names the definition its arguments parse as (a `ContourFn`, or
/// the `GeometryOp` a pipeline op shares — see `geom_fns`), and destructures
/// it; the body reads a per-row field through `params.value(field, row)`:
///
/// - `map` — one contour column, per-row values resolved under
///   [`GeomParams::row`];
/// - `zip` — two contour columns (the second a `ColumnRef` operand of the
///   definition), broadcast by [`zip_contours`].
#[macro_export]
macro_rules! contour_accessor {
    (
        $(#[$meta:meta])*
        map fn $name:ident / $out_ty:ident -> |$ity:ident| $elem:expr;
        parse $fam:ident :: $var:ident $({ $($field:ident),* })?;
        |$c:ident, $params:ident, $row:ident| $body:expr
    ) => {
        fn $out_ty(input_fields: &[Field]) -> PolarsResult<Field> {
            let $ity = input_fields
                .first()
                .map(|f| f.dtype().clone())
                .unwrap_or(DataType::Null);
            let $ity = &$ity;
            $crate::geom_arity::elementwise_field(input_fields, stringify!($name), $elem)
        }
        $(#[$meta])*
        #[polars_expr(output_type_func=$out_ty)]
        fn $name(inputs: &[Series], kwargs: $crate::geom_params::GeomKwargs) -> PolarsResult<Series> {
            let (op, geom_params) = $crate::geom_params::GeomParams::parse::<
                $fam<::view_buffer::mode::Wire>,
            >(inputs, kwargs, stringify!($name))?;
            let $fam::$var $({ $($field),* })? = &op else {
                return Err($crate::geom_params::parsed_as_another(stringify!($name)));
            };
            let $ity = inputs[0].dtype();
            $crate::geom_arity::map_contours(
                &inputs[0],
                &geom_params,
                $crate::geom_calls!(),
                $elem,
                |$c, $params, $row| $body,
            )
        }
    };

    (
        $(#[$meta:meta])*
        zip fn $name:ident / $out_ty:ident -> $elem:expr;
        parse $fam:ident :: $var:ident { $other:ident };
        |$a:ident, $b:ident| $body:expr
    ) => {
        fn $out_ty(input_fields: &[Field]) -> PolarsResult<Field> {
            $crate::geom_arity::binary_field(input_fields, stringify!($name), $elem)
        }
        $(#[$meta])*
        #[polars_expr(output_type_func=$out_ty)]
        fn $name(inputs: &[Series], kwargs: $crate::geom_params::GeomKwargs) -> PolarsResult<Series> {
            let (op, params) = $crate::geom_params::GeomParams::parse::<
                $fam<::view_buffer::mode::Wire>,
            >(inputs, kwargs, stringify!($name))?;
            let $fam::$var { $other } = &op else {
                return Err($crate::geom_params::parsed_as_another(stringify!($name)));
            };
            $crate::geom_arity::zip_contours(
                &inputs[0],
                params.column($other),
                &params,
                $crate::geom_calls!(),
                stringify!($name),
                $elem,
                |$a, $b, _row| $body,
            )
        }
    };
}

/// A transform's per-contour result is assembled straight into Arrow (CR-36),
/// and must publish exactly what the per-contour `AnyValue` assembly did.
#[cfg(test)]
mod contour_output_tests {
    use super::{Arity, ContourOutput};
    use polars::prelude::*;
    use view_buffer::geometry::contour::{Contour, Point};

    fn contours() -> (Contour, Contour) {
        let p = Point::new;
        (
            Contour::with_holes(
                vec![p(0.0, 0.0), p(10.0, 0.0), p(10.0, 10.0), p(0.0, 10.0)],
                vec![vec![p(4.0, 4.0), p(6.0, 4.0), p(6.0, 6.0)]],
            ),
            Contour::new(vec![p(20.0, 20.0), p(30.0, 20.0), p(30.0, 30.0)]),
        )
    }

    fn elem() -> DataType {
        DataType::Struct(crate::geom_schema::contour_fields())
    }

    fn both(rows: Vec<Option<Vec<Contour>>>, arity: Arity) -> (Series, Series) {
        let as_anyvalues: Vec<Option<Vec<AnyValue<'static>>>> = rows
            .iter()
            .map(|r| {
                r.as_ref()
                    .map(|cs| cs.iter().map(crate::contour::contour_to_anyvalue).collect())
            })
            .collect();
        let expected =
            AnyValue::column("c".into(), as_anyvalues, arity, &elem()).expect("anyvalue path");
        let got = Contour::column("c".into(), rows, arity, &elem()).expect("arrow path");
        (got, expected)
    }

    #[test]
    fn single_arity_matches() {
        let (a, b) = contours();
        let (got, expected) = both(vec![Some(vec![a]), None, Some(vec![b])], Arity::Single);
        assert_eq!(got.dtype(), expected.dtype());
        assert!(got.equals_missing(&expected), "{got:?}\n{expected:?}");
        assert_eq!(got.null_count(), 1);
    }

    #[test]
    fn set_arity_matches_including_empty_sets() {
        let (a, b) = contours();
        let (got, expected) = both(
            vec![Some(vec![a, b.clone()]), None, Some(vec![]), Some(vec![b])],
            Arity::Set,
        );
        assert_eq!(got.dtype(), expected.dtype());
        assert!(got.equals_missing(&expected), "{got:?}\n{expected:?}");
    }
}

/// The single-column accessors spread a call's rows over the plugin's thread
/// pool, as the pipeline executor does (CR-32): without it a `.contour`
/// accessor used one core however many rows it held.
#[cfg(test)]
mod split_tests {
    use std::collections::HashSet;
    use std::sync::{Condvar, Mutex};
    use std::thread::ThreadId;
    use std::time::Duration;

    use pyo3_polars::export::polars_core::runtime::THREAD_POOL;
    use view_buffer::geometry::contour::Point;
    use view_buffer::mode::Wire;
    use view_buffer::GeometryOp;

    use crate::geom_fns::PointFn;

    use super::*;

    /// Threads seen, and a first row that waits (bounded) for a second
    /// thread: a call that runs on one thread waits out the timeout and
    /// reports one.
    struct Rendezvous {
        seen: Mutex<HashSet<ThreadId>>,
        arrived: Condvar,
        /// Whether row 0 waits for a second thread.
        wait: bool,
    }

    impl Rendezvous {
        fn new(wait: bool) -> Self {
            Rendezvous {
                seen: Mutex::new(HashSet::new()),
                arrived: Condvar::new(),
                wait,
            }
        }

        fn visit(&self, row: usize) {
            let mut seen = self.seen.lock().unwrap();
            seen.insert(std::thread::current().id());
            self.arrived.notify_all();
            if row == 0 && self.wait {
                let _ = self
                    .arrived
                    .wait_timeout_while(seen, Duration::from_secs(5), |s| s.len() < 2)
                    .unwrap();
            }
        }

        fn threads(&self) -> usize {
            self.seen.lock().unwrap().len()
        }
    }

    fn column(rows: usize) -> Series {
        let elem = DataType::Struct(crate::geom_schema::contour_fields());
        let rows = (0..rows)
            .map(|i| {
                let x = i as f64;
                Some(vec![Contour::new(vec![
                    Point::new(x, 0.0),
                    Point::new(x + 1.0, 0.0),
                    Point::new(x, 1.0),
                ])])
            })
            .collect();
        Contour::column("c".into(), rows, Arity::Single, &elem).unwrap()
    }

    fn params(inputs: &[Series]) -> GeomParams<'_> {
        let kwargs = serde_json::from_value(serde_json::json!({
            "args": {"signed": false},
            "on_null": "raise"
        }))
        .unwrap();
        GeomParams::parse::<GeometryOp<Wire>>(inputs, kwargs, "contour_area")
            .unwrap()
            .1
    }

    #[test]
    fn a_contour_accessor_call_runs_its_rows_on_several_threads() {
        if THREAD_POOL.current_num_threads() < 2 {
            eprintln!("skipped: the pool has a single thread");
            return;
        }
        let inputs = [column(256)];
        let params = params(&inputs);
        let rendezvous = Rendezvous::new(true);
        let calls = CallTracker::new();
        let out = map_contours(
            &inputs[0],
            &params,
            &calls,
            DataType::Float64,
            |c, _, row| {
                rendezvous.visit(row);
                Ok(AnyValue::Float64(c.exterior[0].x))
            },
        )
        .unwrap();
        // Rows come back in order.
        let xs: Vec<f64> = out.f64().unwrap().into_no_null_iter().collect();
        assert_eq!(xs, (0..256).map(f64::from).collect::<Vec<_>>());
        let threads = rendezvous.threads();
        assert!(threads > 1, "256 rows ran on {threads} thread(s)");
    }

    #[test]
    fn a_two_column_accessor_call_runs_its_rows_on_several_threads() {
        if THREAD_POOL.current_num_threads() < 2 {
            eprintln!("skipped: the pool has a single thread");
            return;
        }
        let inputs = [column(256), column(256)];
        let (a, b) = (&inputs[0], &inputs[1]);
        let params = params(&inputs[..1]);
        let rendezvous = Rendezvous::new(true);
        let out = zip_contours(
            a,
            b,
            &params,
            &CallTracker::new(),
            "test",
            DataType::Float64,
            |l, r, row| {
                rendezvous.visit(row);
                Ok(AnyValue::Float64(l.exterior[0].x + r.exterior[0].x))
            },
        )
        .unwrap();
        let sums: Vec<f64> = out.f64().unwrap().into_no_null_iter().collect();
        assert_eq!(
            sums,
            (0..256).map(|i| f64::from(i) * 2.0).collect::<Vec<_>>()
        );
        let threads = rendezvous.threads();
        assert!(threads > 1, "256 rows ran on {threads} thread(s)");
    }

    /// Under the streaming engine an accessor's morsels are concurrent calls,
    /// already parallel: a call that overlaps another runs its rows inline.
    #[test]
    fn a_call_overlapping_another_runs_inline() {
        let inputs = [column(256)];
        let params = params(&inputs);
        let calls = CallTracker::new();
        calls
            .running
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst); // another call
        let rendezvous = Rendezvous::new(false);
        let out = map_contours(
            &inputs[0],
            &params,
            &calls,
            DataType::Float64,
            |c, _, row| {
                rendezvous.visit(row);
                Ok(AnyValue::Float64(c.exterior[0].x))
            },
        )
        .unwrap();
        assert_eq!(out.len(), 256);
        assert_eq!(rendezvous.threads(), 1);
    }

    fn translate_params<'a>(
        inputs: &'a [Series],
        on_null: &str,
    ) -> (PointFn<Wire>, GeomParams<'a>) {
        let kwargs = serde_json::from_value(serde_json::json!({
            "args": {"dx": {"$slot": 1}, "dy": 0.0},
            "on_null": on_null
        }))
        .unwrap();
        GeomParams::parse::<PointFn<Wire>>(inputs, kwargs, "point_translate").unwrap()
    }

    /// `map_rows` is the one row loop of the geometry functions: it spreads,
    /// keeps rows in order, gives each range its own null-parameter flag, and
    /// reports the earliest failing row, as a sequential loop would.
    #[test]
    fn map_rows_spreads_and_keeps_row_semantics() {
        if THREAD_POOL.current_num_threads() < 2 {
            eprintln!("skipped: the pool has a single thread");
            return;
        }
        let dx: Vec<Option<f64>> = (0..256).map(|i| (i != 200).then_some(i as f64)).collect();
        let inputs = [column(256), Series::new("dx".into(), dx)];
        let (op, params) = translate_params(&inputs, "null");
        let PointFn::Translate { dx, .. } = &op else {
            unreachable!()
        };
        let rendezvous = Rendezvous::new(true);
        let rows = params
            .map_rows(&CallTracker::new(), 256, |params, i| {
                rendezvous.visit(i);
                params.value(dx, i).map(Some)
            })
            .unwrap();
        let expected: Vec<Option<f64>> = (0..256).map(|i| (i != 200).then_some(i as f64)).collect();
        assert_eq!(rows, expected, "only the null parameter's row is null");
        let threads = rendezvous.threads();
        assert!(threads > 1, "256 rows ran on {threads} thread(s)");

        let err = params
            .map_rows(&CallTracker::new(), 256, |_, i| match i {
                100 | 220 => Err(polars_err!(ComputeError: "row {} failed", i)),
                _ => Ok(Some(i)),
            })
            .unwrap_err();
        assert!(err.to_string().contains("row 100"), "{err}");
    }
}
