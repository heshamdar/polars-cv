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

use crate::contour::{parse_contour, point_dtype_fields};
use crate::geom_params::GeomParams;

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
/// Reads the field names from [`point_dtype_fields`] — the same names the point
/// parser reads — so the dispatch above cannot admit something the parser then
/// rejects.
pub(crate) fn is_point_dtype(dtype: &DataType) -> bool {
    let DataType::Struct(fields) = dtype else {
        return false;
    };
    point_dtype_fields()
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

/// The contours in one row, in whichever arity the column carries.
///
/// A single contour is repacked as a one-element slice, which is the whole of
/// the "small repacking to align them at the start": everything downstream sees
/// a slice and never asks again.
pub(crate) fn row_contours(value: &AnyValue, arity: Arity) -> PolarsResult<Vec<Contour>> {
    match arity {
        Arity::Single => Ok(vec![parse_contour(value)?]),
        Arity::Set => match value {
            AnyValue::List(series) => {
                let mut contours = Vec::with_capacity(series.len());
                for i in 0..series.len() {
                    let item = series.get(i)?;
                    if item.is_null() {
                        continue;
                    }
                    contours.push(parse_contour(&item)?);
                }
                Ok(contours)
            }
            AnyValue::Null => Ok(Vec::new()),
            other => Err(polars_err!(ComputeError: "Expected List[Contour], got {:?}", other)),
        },
    }
}

/// Assemble one row's per-contour results into the value that row contributes.
pub(crate) fn pack_row(
    results: Vec<AnyValue<'static>>,
    arity: Arity,
    elem: &DataType,
) -> PolarsResult<AnyValue<'static>> {
    match arity {
        // `Single` ran exactly one contour, so its single result *is* the row.
        Arity::Single => Ok(results.into_iter().next().unwrap_or(AnyValue::Null)),
        Arity::Set => {
            let series = Series::from_any_values_and_dtype(
                PlSmallStr::from_static("item"),
                &results,
                elem,
                true,
            )?;
            Ok(AnyValue::List(series))
        }
    }
}

/// Run `compute` for every contour in every row, in the column's own arity.
///
/// **The one decode-and-assemble path for the `.contour` accessors.** Nothing
/// else calls [`parse_contour`] directly, so no accessor is free to disagree
/// with what its `output_type_func` declared.
///
/// `compute` returns the per-contour result as an `AnyValue` of type `elem`; it
/// is never called for a null row, and its `row` argument is the *row* index —
/// per-row parameters vary by row, not by contour within a row, so a row's
/// parameter applies to every contour in its set.
pub(crate) fn map_contours(
    series: &Series,
    elem: DataType,
    mut compute: impl FnMut(&Contour, usize) -> PolarsResult<AnyValue<'static>>,
) -> PolarsResult<Series> {
    let arity = Arity::of(series.dtype());
    let len = series.len();
    let mut rows: Vec<AnyValue<'static>> = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        if value.is_null() {
            rows.push(AnyValue::Null);
            continue;
        }
        let contours = row_contours(&value, arity)?;
        let mut results = Vec::with_capacity(contours.len());
        for contour in &contours {
            results.push(compute(contour, i)?);
        }
        rows.push(pack_row(results, arity, &elem)?);
    }

    build_series(series.name().clone(), rows, arity.wrap(elem))
}

/// As [`map_contours`], but each row's work runs under the call's
/// [`NullParamPolicy`](crate::params::NullParamPolicy).
///
/// The policy is a *row*-level decision, so an accessor with per-row parameters
/// wraps the row rather than each contour: `on_null("null")` nulls the whole
/// row, exactly as a null input contour already does. Routing it through here
/// keeps it from being re-implemented per accessor — the job `contour_row` did
/// for the single-contour accessors, which this replaces.
pub(crate) fn map_contours_with_params(
    series: &Series,
    params: &GeomParams,
    elem: DataType,
    mut compute: impl FnMut(&Contour, usize) -> PolarsResult<AnyValue<'static>>,
) -> PolarsResult<Series> {
    let arity = Arity::of(series.dtype());
    let len = series.len();
    let mut rows: Vec<AnyValue<'static>> = Vec::with_capacity(len);

    for i in 0..len {
        let value = series.get(i)?;
        if value.is_null() {
            rows.push(AnyValue::Null);
            continue;
        }
        let row = params.row(|| {
            let contours = row_contours(&value, arity)?;
            let mut results = Vec::with_capacity(contours.len());
            for contour in &contours {
                results.push(compute(contour, i)?);
            }
            pack_row(results, arity, &elem)
        })?;
        rows.push(row.unwrap_or(AnyValue::Null));
    }

    build_series(series.name().clone(), rows, arity.wrap(elem))
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
pub(crate) fn zip_contours(
    a: &Series,
    b: &Series,
    name: &'static str,
    elem: DataType,
    mut compute: impl FnMut(&Contour, &Contour, usize) -> PolarsResult<AnyValue<'static>>,
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
    let len = a.len();
    let mut rows: Vec<AnyValue<'static>> = Vec::with_capacity(len);

    for i in 0..len {
        let (av, bv) = (a.get(i)?, b.get(i)?);
        if av.is_null() || bv.is_null() {
            rows.push(AnyValue::Null);
            continue;
        }
        let (left, right) = (row_contours(&av, a_arity)?, row_contours(&bv, b_arity)?);
        // Exactly one side is a set, so the other side's single contour is
        // repeated against it; when neither is, both are one-element.
        let mut results = Vec::with_capacity(left.len().max(right.len()));
        match (a_arity, b_arity) {
            (Arity::Set, _) => {
                let Some(single) = right.first() else {
                    rows.push(pack_row(Vec::new(), arity, &elem)?);
                    continue;
                };
                for contour in &left {
                    results.push(compute(contour, single, i)?);
                }
            }
            (_, Arity::Set) => {
                let Some(single) = left.first() else {
                    rows.push(pack_row(Vec::new(), arity, &elem)?);
                    continue;
                };
                for contour in &right {
                    results.push(compute(single, contour, i)?);
                }
            }
            (Arity::Single, Arity::Single) => {
                results.push(compute(&left[0], &right[0], i)?);
            }
        }
        rows.push(pack_row(results, arity, &elem)?);
    }

    build_series(a.name().clone(), rows, arity.wrap(elem))
}

fn build_series(
    name: PlSmallStr,
    rows: Vec<AnyValue<'static>>,
    dtype: DataType,
) -> PolarsResult<Series> {
    Series::from_any_values_and_dtype(name, &rows, &dtype, true)
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
/// Three forms:
///
/// - `map` — one contour column, no parameters.
/// - `map_params` — one contour column plus per-row parameters, resolved under
///   [`GeomParams::row`]; the body additionally sees `params`, `kwargs`, `row`.
/// - `zip` — two contour columns, broadcast by [`zip_contours`].
#[macro_export]
macro_rules! contour_accessor {
    (
        $(#[$meta:meta])*
        map fn $name:ident / $out_ty:ident -> |$ity:ident| $elem:expr;
        |$c:ident| $body:expr
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
        fn $name(inputs: &[Series]) -> PolarsResult<Series> {
            let $ity = inputs[0].dtype();
            $crate::geom_arity::map_contours(&inputs[0], $elem, |$c, _row| $body)
        }
    };

    (
        $(#[$meta:meta])*
        map_params fn $name:ident / $out_ty:ident -> |$ity:ident| $elem:expr;
        |$c:ident, $params:ident, $kwargs:ident, $row:ident| $body:expr
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
        fn $name(inputs: &[Series], kwargs: ContourKwargs) -> PolarsResult<Series> {
            let $params = GeomParams::new(inputs, &kwargs.input_slots, kwargs.on_null)?;
            let $kwargs = &kwargs;
            let $ity = inputs[0].dtype();
            $crate::geom_arity::map_contours_with_params(
                &inputs[0],
                &$params,
                $elem,
                |$c, $row| $body,
            )
        }
    };

    (
        $(#[$meta:meta])*
        zip fn $name:ident / $out_ty:ident -> $elem:expr;
        |$a:ident, $b:ident| $body:expr
    ) => {
        fn $out_ty(input_fields: &[Field]) -> PolarsResult<Field> {
            $crate::geom_arity::binary_field(input_fields, stringify!($name), $elem)
        }
        $(#[$meta])*
        #[polars_expr(output_type_func=$out_ty)]
        fn $name(inputs: &[Series]) -> PolarsResult<Series> {
            $crate::geom_arity::zip_contours(
                &inputs[0],
                &inputs[1],
                stringify!($name),
                $elem,
                |$a, $b, _row| $body,
            )
        }
    };
}
