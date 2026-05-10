//! Plan-time op contracts.
//!
//! Mirrors `python/polars_cv/_types.py` enums (`DTypeEffect`, `NdimEffect`,
//! `AlphaMode`, `OpContract`). The Python side is regenerated from the Rust
//! registry by the `dump_schema` binary, so these definitions are the single
//! source of truth.

use serde::{Deserialize, Serialize};

/// How an operation affects the buffer dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DTypeEffect {
    /// Output dtype == input dtype (e.g. `resize`, `pad`, `crop`).
    Preserve,
    /// Output is always `u8` (e.g. `grayscale`, `threshold`).
    FixedU8,
    /// Output is always `f32`.
    FixedF32,
    /// Output is always `f64` (e.g. global reductions).
    FixedF64,
    /// Output is always `i64` (e.g. `argmax` / `argmin`).
    FixedI64,
    /// Output is always `u64` (e.g. histogram counts).
    FixedU64,
    /// Output is always `u32` (e.g. histogram quantized).
    FixedU32,
    /// Integer inputs become `f32`; float inputs are unchanged.
    PromoteToFloat,
    /// Default `f32`, but overridable via an `out_dtype` parameter.
    ConfigurableF32,
}

impl DTypeEffect {
    /// Stable string identifier matching the Python enum value.
    pub fn as_str(self) -> &'static str {
        match self {
            DTypeEffect::Preserve => "preserve",
            DTypeEffect::FixedU8 => "u8",
            DTypeEffect::FixedF32 => "f32",
            DTypeEffect::FixedF64 => "f64",
            DTypeEffect::FixedI64 => "i64",
            DTypeEffect::FixedU64 => "u64",
            DTypeEffect::FixedU32 => "u32",
            DTypeEffect::PromoteToFloat => "promote",
            DTypeEffect::ConfigurableF32 => "config_f32",
        }
    }

    /// Resolve the concrete output dtype given the input dtype string.
    ///
    /// `"auto"` propagates through `Preserve` and `PromoteToFloat`; all
    /// `Fixed*` variants resolve unconditionally. `ConfigurableF32` resolves
    /// to `f32` here — callers that honor the `out_dtype` parameter override
    /// must check it before calling this.
    pub fn resolve(self, input_dtype: &str) -> &'static str {
        if input_dtype == "auto" {
            return match self {
                DTypeEffect::FixedU8 => "u8",
                DTypeEffect::FixedF32 => "f32",
                DTypeEffect::FixedF64 => "f64",
                DTypeEffect::FixedI64 => "i64",
                DTypeEffect::FixedU64 => "u64",
                DTypeEffect::FixedU32 => "u32",
                DTypeEffect::ConfigurableF32 => "f32",
                DTypeEffect::Preserve | DTypeEffect::PromoteToFloat => "auto",
            };
        }
        match self {
            DTypeEffect::Preserve => match input_dtype {
                "u8" => "u8",
                "i8" => "i8",
                "u16" => "u16",
                "i16" => "i16",
                "u32" => "u32",
                "i32" => "i32",
                "u64" => "u64",
                "i64" => "i64",
                "f32" => "f32",
                "f64" => "f64",
                _ => "auto",
            },
            DTypeEffect::PromoteToFloat => match input_dtype {
                "f32" => "f32",
                "f64" => "f64",
                _ => "f32",
            },
            DTypeEffect::FixedU8 => "u8",
            DTypeEffect::FixedF32 => "f32",
            DTypeEffect::FixedF64 => "f64",
            DTypeEffect::FixedI64 => "i64",
            DTypeEffect::FixedU64 => "u64",
            DTypeEffect::FixedU32 => "u32",
            DTypeEffect::ConfigurableF32 => "f32",
        }
    }
}

/// How an operation affects the number of dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NdimEffect {
    /// Ndim unchanged (e.g. `resize`, `blur`, `pad`).
    Preserve,
    /// Ndim decreases by 1 (axis-based reduction).
    ReduceOne,
    /// Global reduction to scalar (ndim → 0).
    ToZero,
    /// Output is a 1-D vector (e.g. `perceptual_hash`, `extract_shape`).
    ToOne,
    /// Output is 3-D (e.g. `rasterize` → `[H, W, C]`).
    ToThree,
}

impl NdimEffect {
    pub fn as_str(self) -> &'static str {
        match self {
            NdimEffect::Preserve => "preserve",
            NdimEffect::ReduceOne => "reduce_one",
            NdimEffect::ToZero => "to_zero",
            NdimEffect::ToOne => "to_one",
            NdimEffect::ToThree => "to_three",
        }
    }
}

/// How an operation handles an alpha channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlphaMode {
    /// All channels (including alpha) processed uniformly.
    Passthrough,
    /// Alpha separated, op applied to color channels only, alpha restored.
    StripProcessRestore,
    /// Alpha discarded; output channel count is fixed by the operation.
    Drop,
    /// Non-image op; alpha handling is irrelevant.
    NotApplicable,
}

impl AlphaMode {
    pub fn as_str(self) -> &'static str {
        match self {
            AlphaMode::Passthrough => "passthrough",
            AlphaMode::StripProcessRestore => "strip_process_restore",
            AlphaMode::Drop => "drop",
            AlphaMode::NotApplicable => "not_applicable",
        }
    }
}

/// Plan-time declaration of an op's effects on dtype, ndim, and alpha.
///
/// Constructed as a `const` per op and exposed through the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpContract {
    pub dtype_effect: DTypeEffect,
    pub ndim_effect: NdimEffect,
    pub alpha_mode: AlphaMode,
}

impl OpContract {
    pub const fn new(
        dtype_effect: DTypeEffect,
        ndim_effect: NdimEffect,
        alpha_mode: AlphaMode,
    ) -> Self {
        Self {
            dtype_effect,
            ndim_effect,
            alpha_mode,
        }
    }

    pub const fn buffer(dtype: DTypeEffect, ndim: NdimEffect, alpha: AlphaMode) -> Self {
        Self::new(dtype, ndim, alpha)
    }

    pub const fn non_image(dtype: DTypeEffect, ndim: NdimEffect) -> Self {
        Self::new(dtype, ndim, AlphaMode::NotApplicable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_effect_strings_match_python() {
        // Spot check: must match the values in python/polars_cv/_types.py.
        assert_eq!(DTypeEffect::Preserve.as_str(), "preserve");
        assert_eq!(DTypeEffect::FixedU8.as_str(), "u8");
        assert_eq!(DTypeEffect::PromoteToFloat.as_str(), "promote");
        assert_eq!(DTypeEffect::ConfigurableF32.as_str(), "config_f32");
    }

    #[test]
    fn auto_dtype_resolution() {
        assert_eq!(DTypeEffect::Preserve.resolve("auto"), "auto");
        assert_eq!(DTypeEffect::PromoteToFloat.resolve("auto"), "auto");
        assert_eq!(DTypeEffect::FixedU8.resolve("auto"), "u8");
        assert_eq!(DTypeEffect::ConfigurableF32.resolve("auto"), "f32");
    }

    #[test]
    fn promote_to_float_preserves_floats() {
        assert_eq!(DTypeEffect::PromoteToFloat.resolve("u8"), "f32");
        assert_eq!(DTypeEffect::PromoteToFloat.resolve("f64"), "f64");
        assert_eq!(DTypeEffect::PromoteToFloat.resolve("f32"), "f32");
    }
}
