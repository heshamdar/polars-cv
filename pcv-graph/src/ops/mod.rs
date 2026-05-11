//! Built-in op adapters.
//!
//! Each submodule registers one op via `inventory::submit!`. Adding an op to
//! the registry is a single-file change and a single `submit!` call — no
//! match arms or central tables to update.

pub mod channel_select;
pub mod common;
pub mod crop;
pub mod flip;
pub mod grayscale;
pub mod identity;
pub mod reshape;
pub mod rotate;
pub mod transpose;
