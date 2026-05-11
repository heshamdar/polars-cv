//! Built-in op adapters.
//!
//! Each submodule registers one op via `inventory::submit!`. Adding an op to
//! the registry is a single-file change and a single `submit!` call — no
//! match arms or central tables to update.

pub mod adjust_contrast;
pub mod adjust_gamma;
pub mod blur;
pub mod canny;
pub mod cast;
pub mod channel_select;
pub mod clamp;
pub mod common;
pub mod crop;
pub mod dilate;
pub mod equalize_histogram;
pub mod erode;
pub mod flip;
pub mod grayscale;
pub mod identity;
pub mod invert;
pub mod morphology_gradient;
pub mod normalize;
pub mod reduce_argmax;
pub mod reduce_argmin;
pub mod reduce_max;
pub mod reduce_mean;
pub mod reduce_min;
pub mod reduce_percentile;
pub mod reduce_popcount;
pub mod reduce_std;
pub mod reduce_sum;
pub mod relu;
pub mod reshape;
pub mod rotate;
pub mod scale;
pub mod threshold;
pub mod transpose;
