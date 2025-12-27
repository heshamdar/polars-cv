//! Core types for the view-buffer framework.
//!
//! This module contains the fundamental types used throughout the crate:
//! - [`DType`] - Data type enumeration
//! - [`ViewType`] - Trait for mapping Rust types to DType
//! - [`Layout`], [`LayoutFacts`], [`LayoutReport`] - Memory layout tracking
//! - [`ViewBuffer`], [`BufferStorage`], [`BufferError`] - Buffer storage and views

pub mod buffer;
pub mod dtype;
pub mod layout;

pub use buffer::{BufferError, BufferStorage, ViewBuffer};
pub use dtype::{DType, ViewType};
pub use layout::{ExternalLayout, Layout, LayoutFacts, LayoutReport};
