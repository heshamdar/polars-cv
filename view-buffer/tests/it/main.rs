//! view-buffer's integration tests, as one test binary.
//!
//! Each module was its own `tests/*.rs` crate, and so its own executable
//! statically linking all of view-buffer and its dependencies: 31 of them, about
//! 4 GB of `target/debug`, every one relinked on any change to the crate. One
//! binary links once. A module keeps its own `#![cfg(feature = ...)]`, and
//! `copy_counts`' counting allocator counts per thread, only inside its own
//! measurements, so it is unaffected by the other modules' tests.
//!
//! Run one module with `cargo test -p view-buffer --test it <module>::`.

mod affine_ref;
mod alignment_soundness;
mod apply_op_coverage;
mod binary_exact;
mod binary_views;
mod blob_write;
mod blur_dtype;
mod clamp_fusion;
mod convolve_ref;
mod copy_counts;
mod dense_rows_ops;
mod dtype_single_authority;
mod exact_data_movement;
mod f64_scalar_contract;
mod fused_ops;
mod gamma_dtype_range;
mod identity_rule_contract;
mod integration_plan;
mod luma_authority;
mod memory_effect_verification;
mod morph_ref;
mod nan_ordering;
mod opt_config;
mod png_decode_eval;
mod polars_interop;
mod resize_views;
mod rotation;
mod stride_contract_fixes;
mod strided_ops;
mod validation;
mod zero_copy;
