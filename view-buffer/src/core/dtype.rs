//! Data type definitions for view-buffer.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Supported data types for buffer elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    F64,
    U64,
    I64,
}

/// Declare every representation of every dtype in one place.
///
/// A dtype is named three times across this workspace's boundaries: a short
/// name (`"u8"`) in the graph JSON, a VIEW protocol wire code (`1`) in the
/// binary header, and a numpy name (`"uint8"`) in the numpy/torch sink and
/// the header-only metadata accessors. Each used to be its own `match` — five
/// tables in four files — so adding a dtype meant finding all five.
///
/// Most of those were already exhaustive matches over `DType` and so were
/// compiler-enforced: forgetting one was a build error, not a silent gap. The
/// exception was the reverse map `u8_to_dtype`, whose `_ => None` arm meant a
/// new dtype would simply fail to decode from a VIEW header. So the win here
/// is locality — one row per dtype, all three spellings visible together —
/// plus closing that one silent case, not a wholesale gain in safety.
///
/// The generated accessors below all `match` on the listed variants, so the
/// compiler still rejects a `DType` variant that this table omits — the same
/// exhaustiveness guard `named_variants!` uses, extended to carry the wire
/// code and numpy name alongside the short name.
///
/// Wire codes are **not** the declaration order: they are fixed by the VIEW
/// binary format and must never be renumbered (`F32`/`F64` are 7/8, ahead of
/// `U64`/`I64` at 9/10). They are listed per row for that reason. Row order
/// itself sets `NAMED`'s order; no test depends on it — the Python parity
/// tests compare sets — but it is the order users see in `expected one of
/// {...}` errors, so rows keep their original order rather than being
/// reshuffled into wire-code order for cosmetics.
macro_rules! dtype_table {
    ($(($variant:ident, $short:literal, $code:literal, $numpy:literal)),+ $(,)?) => {
        crate::naming::named_variants!(DType: "Supported data types." { $($short => $variant),+ });

        impl DType {
            /// Every dtype, in `NAMED` declaration order.
            pub const ALL: &'static [DType] = &[$(DType::$variant),+];

            /// The canonical short name ("u8", "f32", …) of this dtype.
            pub const fn short_name(&self) -> &'static str {
                match self { $(DType::$variant => $short),+ }
            }

            /// This dtype's VIEW protocol wire code.
            ///
            /// Stable across releases — the binary format depends on it.
            pub const fn wire_code(&self) -> u8 {
                match self { $(DType::$variant => $code),+ }
            }

            /// This dtype's numpy name ("uint8", "float32", …).
            pub const fn numpy_name(&self) -> &'static str {
                match self { $(DType::$variant => $numpy),+ }
            }

            /// Parse a canonical short name back into a dtype.
            pub fn from_short_name(s: &str) -> Option<Self> {
                crate::naming::lookup(Self::NAMED, s)
            }

            /// Parse a VIEW protocol wire code back into a dtype.
            pub fn from_wire_code(code: u8) -> Option<Self> {
                match code { $($code => Some(DType::$variant),)+ _ => None }
            }

            /// Parse a numpy name back into a dtype.
            pub fn from_numpy_name(s: &str) -> Option<Self> {
                match s { $($numpy => Some(DType::$variant),)+ _ => None }
            }
        }
    };
}

dtype_table!(
    (U8, "u8", 1, "uint8"),
    (I8, "i8", 2, "int8"),
    (U16, "u16", 3, "uint16"),
    (I16, "i16", 4, "int16"),
    (U32, "u32", 5, "uint32"),
    (I32, "i32", 6, "int32"),
    (U64, "u64", 9, "uint64"),
    (I64, "i64", 10, "int64"),
    (F32, "f32", 7, "float32"),
    (F64, "f64", 8, "float64"),
);

/// Evaluate `$body` with `$T` bound to the Rust element type of `$dtype`.
///
/// **The one runtime match from a [`DType`] to its element type.** The match
/// is exhaustive, so a dtype added to `dtype_table!` fails to compile here
/// rather than falling through to a default arm. Nest it for dtype pairs:
/// `with_dtype!(src, S => with_dtype!(dst, D => convert::<S, D>(..)))`.
macro_rules! with_dtype {
    ($dtype:expr, $T:ident => $body:expr) => {
        match $dtype {
            $crate::core::dtype::DType::U8 => {
                type $T = u8;
                $body
            }
            $crate::core::dtype::DType::I8 => {
                type $T = i8;
                $body
            }
            $crate::core::dtype::DType::U16 => {
                type $T = u16;
                $body
            }
            $crate::core::dtype::DType::I16 => {
                type $T = i16;
                $body
            }
            $crate::core::dtype::DType::U32 => {
                type $T = u32;
                $body
            }
            $crate::core::dtype::DType::I32 => {
                type $T = i32;
                $body
            }
            $crate::core::dtype::DType::U64 => {
                type $T = u64;
                $body
            }
            $crate::core::dtype::DType::I64 => {
                type $T = i64;
                $body
            }
            $crate::core::dtype::DType::F32 => {
                type $T = f32;
                $body
            }
            $crate::core::dtype::DType::F64 => {
                type $T = f64;
                $body
            }
        }
    };
}
pub(crate) use with_dtype;

/// Categories of data types that operations can accept as input.
///
/// This enables operations to declare what types they can work with,
/// allowing the execution layer to handle automatic casting.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DTypeCategory {
    /// Accept all data types.
    #[default]
    Any,
    /// Accept all numeric types (all currently supported types).
    Numeric,
    /// Accept only integer types (u8, i8, u16, i16, u32, i32, u64, i64).
    Integer,
    /// Accept only floating-point types (f32, f64).
    Float,
    /// Accept only specific data types.
    Specific(Vec<DType>),
}

impl DTypeCategory {
    /// Check if a dtype is accepted by this category.
    pub fn accepts(&self, dtype: DType) -> bool {
        match self {
            DTypeCategory::Any => true,
            DTypeCategory::Numeric => true, // All current types are numeric
            DTypeCategory::Integer => matches!(
                dtype,
                DType::U8
                    | DType::I8
                    | DType::U16
                    | DType::I16
                    | DType::U32
                    | DType::I32
                    | DType::U64
                    | DType::I64
            ),
            DTypeCategory::Float => matches!(dtype, DType::F32 | DType::F64),
            DTypeCategory::Specific(allowed) => allowed.contains(&dtype),
        }
    }
}

/// Rules for determining output dtype of an operation.
///
/// This separates the semantic behavior of an operation from its
/// dtype mechanics, allowing for flexible and predictable pipelines.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum OutputDTypeRule {
    /// Output dtype matches input dtype.
    #[default]
    PreserveInput,
    /// Output is always a fixed dtype (e.g., always F32).
    Fixed(DType),
    /// A float: the input's [`DType::accumulator`] — f32 for the 8/16-bit
    /// integers, f64 for the 32/64-bit ones (which f32 cannot hold, as NumPy
    /// promotes them), the float dtypes preserved. The op computes in that
    /// same float, so what it computes in and what it stores agree.
    PromoteToFloat,
    /// Force output to F64 (for reductions that need precision).
    ForceF64,
    /// Force output to I64 (for argmax/argmin).
    ForceI64,
    /// Force output to U64 (for count-based operations).
    ForceU64,
    /// Force output to U32 (for bin indices).
    ForceU32,
}

/// What a *planner* knows about an element dtype it has not fully resolved.
///
/// The plan-time dtype string carries an `"auto"` sentinel for "the source's
/// decode dtype is not known until execution" — a PNG can decode u8 or u16, a
/// TIFF f32 or f64. Folding a rule over that used to collapse to `"auto"`
/// again, which throws away real information: whatever the input turns out to
/// be, `PromoteToFloat` of it is *a float*. That is enough to know a JPEG sink
/// cannot work, and losing it is why such a query planned successfully and
/// then failed in the encoder.
///
/// Three states, ordered by how much is known. `SomeFloat` is deliberately not
/// a dtype: there is no single answer (f32 or f64 by the input's width), and
/// inventing one would be a lie the runtime guard would catch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlannedDType {
    /// Fully resolved.
    Known(DType),
    /// Unknown, but guaranteed floating point.
    SomeFloat,
    /// Nothing known.
    Unknown,
}

impl PlannedDType {
    /// The wire spelling for "not known at all".
    pub const AUTO: &'static str = "auto";
    /// The wire spelling for "not known, but floating point".
    pub const AUTO_FLOAT: &'static str = "auto_float";

    /// Parse a plan-time dtype string.
    ///
    /// Returns `None` for a string that is neither a sentinel nor a dtype
    /// `dtype_table!` knows — an unrecognised spelling must fail loudly rather
    /// than default to anything.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            Self::AUTO => Some(Self::Unknown),
            Self::AUTO_FLOAT => Some(Self::SomeFloat),
            other => DType::from_short_name(other).map(Self::Known),
        }
    }

    /// The wire spelling of this state.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Known(d) => d.short_name(),
            Self::SomeFloat => Self::AUTO_FLOAT,
            Self::Unknown => Self::AUTO,
        }
    }

    /// Is this a single, fully-resolved dtype?
    pub fn is_concrete(self) -> bool {
        matches!(self, Self::Known(_))
    }

    /// Every dtype this could turn out to be; empty means "any".
    ///
    /// Callers that must decide something now — can this be a JPEG? — reject
    /// only when *every* candidate fails, so a less-resolved state is always
    /// more permissive.
    pub fn candidates(self) -> &'static [DType] {
        match self {
            Self::Known(DType::U8) => &[DType::U8],
            Self::Known(DType::I8) => &[DType::I8],
            Self::Known(DType::U16) => &[DType::U16],
            Self::Known(DType::I16) => &[DType::I16],
            Self::Known(DType::U32) => &[DType::U32],
            Self::Known(DType::I32) => &[DType::I32],
            Self::Known(DType::U64) => &[DType::U64],
            Self::Known(DType::I64) => &[DType::I64],
            Self::Known(DType::F32) => &[DType::F32],
            Self::Known(DType::F64) => &[DType::F64],
            Self::SomeFloat => &[DType::F32, DType::F64],
            Self::Unknown => &[],
        }
    }
}

impl OutputDTypeRule {
    /// Apply this rule to a partially-known input dtype.
    ///
    /// The symbolic twin of [`resolve`](Self::resolve), and the reason the
    /// planner can say anything at all about a pipeline over an `"auto"`
    /// source: every rule except `PreserveInput` either fixes the output or
    /// constrains it to a float, so an unknown input does not have to mean an
    /// unknown output.
    pub fn resolve_planned(&self, input: PlannedDType) -> PlannedDType {
        match (self, input) {
            // Fully known input: defer to the concrete rule.
            (_, PlannedDType::Known(dt)) => PlannedDType::Known(self.resolve(dt)),
            // Output follows an input we do not know.
            (OutputDTypeRule::PreserveInput, other) => other,
            // A float in, a float out; an integer in, f32 or f64 out. Either
            // way a float — which is the whole point of this state existing.
            (OutputDTypeRule::PromoteToFloat, _) => PlannedDType::SomeFloat,
            // The remaining rules ignore the input entirely, so an unknown
            // input is no obstacle. U8 is an arbitrary stand-in that `resolve`
            // is documented to discard for these.
            (rule, _) => PlannedDType::Known(rule.resolve(DType::U8)),
        }
    }

    /// Resolve the output dtype given an input dtype.
    pub fn resolve(&self, input_dtype: DType) -> DType {
        match self {
            OutputDTypeRule::PreserveInput => input_dtype,
            OutputDTypeRule::Fixed(dtype) => *dtype,
            OutputDTypeRule::PromoteToFloat => input_dtype.accumulator(),
            OutputDTypeRule::ForceF64 => DType::F64,
            OutputDTypeRule::ForceI64 => DType::I64,
            OutputDTypeRule::ForceU64 => DType::U64,
            OutputDTypeRule::ForceU32 => DType::U32,
        }
    }
}

impl DType {
    /// Returns the size in bytes of this data type.
    pub fn size_of(&self) -> usize {
        match self {
            DType::U8 | DType::I8 => 1,
            DType::U16 | DType::I16 => 2,
            DType::U32 | DType::I32 | DType::F32 => 4,
            DType::U64 | DType::I64 | DType::F64 => 8,
        }
    }

    /// Whether converting a value held in `self` to `target` (the crate's
    /// rule, `core::convert`) gives what converting that same value from a
    /// float gives. Only an integer-to-integer conversion that narrows
    /// differs: int -> int wraps (`as`), float -> int rounds and saturates.
    ///
    /// The one rule an optimizer pass reads before replacing a conversion
    /// from `self` with a float one: `cast_chain_collapse` dropping a float
    /// intermediate between integers, and `scalar_fusion` storing an integer
    /// chain's result through its f32 kernel.
    pub fn converts_like_float(&self, target: DType) -> bool {
        let int = |d: DType| DTypeCategory::Integer.accepts(d);
        !(int(*self) && int(target)) || target.losslessly_contains(*self)
    }

    /// The float an interpolating kernel (convolve, blur, resample, colour
    /// math) computes in for an input of this dtype: `F64` for `f64` and the
    /// 32/64-bit integers, whose values f32 cannot hold, else `F32`. The one
    /// rule, so no kernel decides it alone; the result is then stored in the
    /// op's declared output dtype.
    pub fn accumulator(&self) -> DType {
        match self {
            DType::F64 | DType::U32 | DType::I32 | DType::U64 | DType::I64 => DType::F64,
            DType::U8 | DType::I8 | DType::U16 | DType::I16 | DType::F32 => DType::F32,
        }
    }

    /// Whether every value representable in `other` is exactly representable in
    /// `self` — i.e. casting `other` → `self` loses no information.
    ///
    /// The plan-time authority for when an intermediate cast may be dropped from
    /// a cast chain: `grandchild → cast(inner) → cast(target)` collapses to
    /// `grandchild → cast(target)` only when `inner.losslessly_contains(grandchild)`,
    /// so the dropped intermediate cast provably changed no value. A narrowing
    /// intermediate (e.g. `f32 → cast(u8) → cast(f32)`, which quantizes) is not
    /// contained and must not be dropped.
    pub fn losslessly_contains(self, other: DType) -> bool {
        use DType::*;
        if self == other {
            return true;
        }
        // (is_signed, bit_width) for integer dtypes; None for floats.
        fn int_info(d: DType) -> Option<(bool, u32)> {
            match d {
                U8 => Some((false, 8)),
                U16 => Some((false, 16)),
                U32 => Some((false, 32)),
                U64 => Some((false, 64)),
                I8 => Some((true, 8)),
                I16 => Some((true, 16)),
                I32 => Some((true, 32)),
                I64 => Some((true, 64)),
                F32 | F64 => None,
            }
        }
        // Integers are exactly representable up to 2^mantissa; None for ints.
        fn mantissa(d: DType) -> Option<u32> {
            match d {
                F32 => Some(24),
                F64 => Some(53),
                _ => None,
            }
        }
        match (int_info(self), int_info(other)) {
            // integer → integer
            (Some((s_signed, s_bits)), Some((o_signed, o_bits))) => match (s_signed, o_signed) {
                (false, false) | (true, true) => s_bits >= o_bits,
                (true, false) => s_bits > o_bits, // signed needs one extra bit for the sign
                (false, true) => false,           // unsigned cannot hold negatives
            },
            // float destination, integer source: exact iff the magnitude fits the mantissa.
            (None, Some((o_signed, o_bits))) => {
                let need = if o_signed { o_bits - 1 } else { o_bits };
                need <= mantissa(self).expect("self is a float in this arm")
            }
            // float → float: the wider mantissa contains the narrower.
            (None, None) => mantissa(self).unwrap() >= mantissa(other).unwrap(),
            // integer destination, float source: never (fractionals / NaN / inf).
            (Some(_), None) => false,
        }
    }

    /// The dtype two operands of these dtypes combine in: **NumPy's
    /// promotion** (`np.result_type`). The first dtype, smallest first and
    /// unsigned before signed before float, that holds every value of both
    /// exactly ([`losslessly_contains`](Self::losslessly_contains)), else
    /// `F64` when none does (u64 with a signed integer, a 64-bit integer with
    /// a float). So u8 with i8 is i16 and u32 with f32 is f64, where the
    /// larger-integer rule this replaced lost values (u8 200 + i8 gave i8).
    pub fn promote(self, other: DType) -> DType {
        use DType::*;
        [U8, I8, U16, I16, U32, I32, U64, I64, F32, F64]
            .into_iter()
            .find(|d| d.losslessly_contains(self) && d.losslessly_contains(other))
            .unwrap_or(F64)
    }

    /// The value range's maximum: an integer dtype's largest value, 1.0 for
    /// the floats (whose image values are in [0, 1], as in OpenCV). The one
    /// rule for "what full scale means" that range-mapping ops read: gamma
    /// (`norm_range_max_f32`) and the colour conversions.
    pub fn value_range_max(&self) -> f64 {
        match self {
            DType::U8 => u8::MAX as f64,
            DType::I8 => i8::MAX as f64,
            DType::U16 => u16::MAX as f64,
            DType::I16 => i16::MAX as f64,
            DType::U32 => u32::MAX as f64,
            DType::I32 => i32::MAX as f64,
            DType::U64 => u64::MAX as f64,
            DType::I64 => i64::MAX as f64,
            DType::F32 | DType::F64 => 1.0,
        }
    }

    /// [`value_range_max`](Self::value_range_max) as f32 (approximate for the
    /// 32/64-bit integers), the normalization ceiling gamma uses.
    pub fn norm_range_max_f32(&self) -> f32 {
        self.value_range_max() as f32
    }
}

/// Trait to map Rust types to DType enum.
pub trait ViewType: 'static + Copy + Send + Sync + std::fmt::Debug {
    /// The corresponding DType for this Rust type.
    const DTYPE: DType;
}

macro_rules! impl_view_type {
    ($rust_type:ty, $dtype:expr) => {
        impl ViewType for $rust_type {
            const DTYPE: DType = $dtype;
        }
    };
}

impl_view_type!(u8, DType::U8);
impl_view_type!(i8, DType::I8);
impl_view_type!(u16, DType::U16);
impl_view_type!(i16, DType::I16);
impl_view_type!(u32, DType::U32);
impl_view_type!(i32, DType::I32);
impl_view_type!(f32, DType::F32);
impl_view_type!(f64, DType::F64);
impl_view_type!(u64, DType::U64);
impl_view_type!(i64, DType::I64);

#[cfg(test)]
mod losslessly_contains_tests {
    use super::DType::*;

    #[test]
    fn reflexive() {
        for d in super::DType::ALL {
            assert!(d.losslessly_contains(*d));
        }
    }

    #[test]
    fn integer_widening() {
        assert!(U16.losslessly_contains(U8));
        assert!(I16.losslessly_contains(U8)); // signed needs one extra bit
        assert!(!I8.losslessly_contains(U8)); // 255 > i8::MAX
        assert!(!U8.losslessly_contains(I8)); // unsigned can't hold negatives
        assert!(I32.losslessly_contains(I16));
        assert!(!U8.losslessly_contains(U16));
    }

    #[test]
    fn integer_to_float() {
        assert!(F32.losslessly_contains(U16)); // 16 <= 24 mantissa bits
        assert!(F32.losslessly_contains(I16));
        assert!(!F32.losslessly_contains(U32)); // 32 > 24
        assert!(F64.losslessly_contains(U32)); // 32 <= 53
        assert!(!F64.losslessly_contains(U64)); // 64 > 53
    }

    #[test]
    fn float_lattice_and_float_to_int() {
        assert!(F64.losslessly_contains(F32));
        assert!(!F32.losslessly_contains(F64));
        // A float source is never losslessly held by an integer (fractionals).
        assert!(!U8.losslessly_contains(F32));
        assert!(!I64.losslessly_contains(F64));
    }
}

#[cfg(test)]
mod planned_dtype_tests {
    use super::*;

    /// The ten dtypes, for the lattice tests below.
    const EVERY_DTYPE: &[DType] = &[
        DType::U8,
        DType::I8,
        DType::U16,
        DType::I16,
        DType::U32,
        DType::I32,
        DType::U64,
        DType::I64,
        DType::F32,
        DType::F64,
    ];

    const EVERY_RULE: &[OutputDTypeRule] = &[
        OutputDTypeRule::PreserveInput,
        OutputDTypeRule::Fixed(DType::U8),
        OutputDTypeRule::PromoteToFloat,
        OutputDTypeRule::ForceF64,
        OutputDTypeRule::ForceI64,
        OutputDTypeRule::ForceU64,
        OutputDTypeRule::ForceU32,
    ];

    /// `promote` is NumPy's table: every pair, as `np.result_type` gives it
    /// (numpy 2.x, recorded rather than derived).
    #[test]
    fn promote_is_numpys_result_type() {
        use DType::*;
        let numpy: &[(DType, DType, DType)] = &[
            (U8, U8, U8),
            (U8, I8, I16),
            (U8, U16, U16),
            (U8, I16, I16),
            (U8, U32, U32),
            (U8, I32, I32),
            (U8, U64, U64),
            (U8, I64, I64),
            (U8, F32, F32),
            (U8, F64, F64),
            (I8, I8, I8),
            (I8, U16, I32),
            (I8, I16, I16),
            (I8, U32, I64),
            (I8, I32, I32),
            (I8, U64, F64),
            (I8, I64, I64),
            (I8, F32, F32),
            (I8, F64, F64),
            (U16, U16, U16),
            (U16, I16, I32),
            (U16, U32, U32),
            (U16, I32, I32),
            (U16, U64, U64),
            (U16, I64, I64),
            (U16, F32, F32),
            (U16, F64, F64),
            (I16, I16, I16),
            (I16, U32, I64),
            (I16, I32, I32),
            (I16, U64, F64),
            (I16, I64, I64),
            (I16, F32, F32),
            (I16, F64, F64),
            (U32, U32, U32),
            (U32, I32, I64),
            (U32, U64, U64),
            (U32, I64, I64),
            (U32, F32, F64),
            (U32, F64, F64),
            (I32, I32, I32),
            (I32, U64, F64),
            (I32, I64, I64),
            (I32, F32, F64),
            (I32, F64, F64),
            (U64, U64, U64),
            (U64, I64, F64),
            (U64, F32, F64),
            (U64, F64, F64),
            (I64, I64, I64),
            (I64, F32, F64),
            (I64, F64, F64),
            (F32, F32, F32),
            (F32, F64, F64),
            (F64, F64, F64),
        ];
        assert_eq!(numpy.len(), 55, "every unordered pair");
        for &(a, b, want) in numpy {
            assert_eq!(a.promote(b), want, "{a:?} with {b:?}");
            assert_eq!(b.promote(a), want, "{b:?} with {a:?}");
        }
    }

    #[test]
    fn resolve_planned_agrees_with_resolve_on_a_known_input() {
        // The symbolic fold is only trustworthy if it is the same function as
        // the concrete one wherever both are defined. If these ever disagreed,
        // the planner would publish a dtype execution does not produce — the
        // exact failure the lattice was added to prevent.
        for rule in EVERY_RULE {
            for &dt in EVERY_DTYPE {
                assert_eq!(
                    rule.resolve_planned(PlannedDType::Known(dt)),
                    PlannedDType::Known(rule.resolve(dt)),
                    "{rule:?} on {dt:?}"
                );
            }
        }
    }

    #[test]
    fn promote_to_float_of_an_unknown_is_some_float() {
        // The whole reason the lattice exists: an unknown input still pins the
        // output to a float, which is enough to rule out an 8-bit codec.
        assert_eq!(
            OutputDTypeRule::PromoteToFloat.resolve_planned(PlannedDType::Unknown),
            PlannedDType::SomeFloat
        );
        // Stable under repetition, and carried through by PreserveInput.
        assert_eq!(
            OutputDTypeRule::PromoteToFloat.resolve_planned(PlannedDType::SomeFloat),
            PlannedDType::SomeFloat
        );
        assert_eq!(
            OutputDTypeRule::PreserveInput.resolve_planned(PlannedDType::SomeFloat),
            PlannedDType::SomeFloat
        );
    }

    #[test]
    fn a_rule_that_ignores_its_input_fully_resolves_an_unknown() {
        for rule in EVERY_RULE {
            if matches!(
                rule,
                OutputDTypeRule::PreserveInput | OutputDTypeRule::PromoteToFloat
            ) {
                continue;
            }
            assert!(
                rule.resolve_planned(PlannedDType::Unknown).is_concrete(),
                "{rule:?} ignores its input, so an unknown one is no obstacle"
            );
        }
    }

    #[test]
    fn candidates_are_monotonic_in_how_much_is_known() {
        // Callers reject only when every candidate fails, so a less-resolved
        // state must offer a superset of possibilities. If this inverted, the
        // planner would start refusing queries that execute perfectly.
        assert_eq!(
            PlannedDType::SomeFloat.candidates(),
            &[DType::F32, DType::F64]
        );
        assert!(
            PlannedDType::Unknown.candidates().is_empty(),
            "empty means 'any', the most permissive state"
        );
        for &dt in EVERY_DTYPE {
            assert_eq!(
                PlannedDType::Known(dt).candidates(),
                &[dt],
                "a known dtype is its own only candidate"
            );
            if matches!(dt, DType::F32 | DType::F64) {
                assert!(PlannedDType::SomeFloat.candidates().contains(&dt));
            }
        }
    }

    #[test]
    fn parse_and_as_str_round_trip_every_state() {
        for &dt in EVERY_DTYPE {
            let state = PlannedDType::Known(dt);
            assert_eq!(PlannedDType::parse(state.as_str()), Some(state));
        }
        for state in [PlannedDType::SomeFloat, PlannedDType::Unknown] {
            assert_eq!(PlannedDType::parse(state.as_str()), Some(state));
        }
        // An unrecognised spelling must not default to anything: the callers
        // treat `None` as "not concrete", and a silent fallback here would put
        // a bogus dtype into a published schema.
        assert_eq!(PlannedDType::parse("f128"), None);
        assert_eq!(PlannedDType::parse(""), None);
        assert_eq!(PlannedDType::parse("uint8"), None);
    }
}
