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
        crate::naming::named_variants!(DType { $($short => $variant),+ });

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
    /// Default to a specific dtype, but can be overridden via out_dtype parameter.
    Configurable(DType),
    /// Promote integers to float32, preserve float types.
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

impl OutputDTypeRule {
    /// Resolve the output dtype given an input dtype and optional override.
    pub fn resolve(&self, input_dtype: DType, out_dtype_override: Option<DType>) -> DType {
        // If there's an explicit override, use it
        if let Some(override_dtype) = out_dtype_override {
            return override_dtype;
        }

        match self {
            OutputDTypeRule::PreserveInput => input_dtype,
            OutputDTypeRule::Fixed(dtype) => *dtype,
            OutputDTypeRule::Configurable(default) => *default,
            OutputDTypeRule::PromoteToFloat => {
                if matches!(input_dtype, DType::F32 | DType::F64) {
                    input_dtype
                } else {
                    DType::F32
                }
            }
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

    /// The normalization ceiling range-mapping ops (gamma) use for this
    /// dtype: the maximum representable value for integers (as f32,
    /// approximate for the 64-bit types), 1.0 for floats.
    pub fn norm_range_max_f32(&self) -> f32 {
        match self {
            DType::U8 => u8::MAX as f32,
            DType::I8 => i8::MAX as f32,
            DType::U16 => u16::MAX as f32,
            DType::I16 => i16::MAX as f32,
            DType::U32 => u32::MAX as f32,
            DType::I32 => i32::MAX as f32,
            DType::U64 => u64::MAX as f32,
            DType::I64 => i64::MAX as f32,
            DType::F32 | DType::F64 => 1.0,
        }
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
