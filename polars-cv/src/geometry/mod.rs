//! The geometry functions: the `.contour`/`.point`/`.bbox` namespaces'
//! definitions, readers and schemas, and the one runner of a geometry step
//! (`exec`), which the pipeline graph shares.

pub(crate) mod accessor;
pub(crate) mod arity;
pub(crate) mod columns;
pub(crate) mod construct;
pub(crate) mod contour;
pub(crate) mod exec;
pub(crate) mod fns;
pub(crate) mod params;
pub(crate) mod point;
pub(crate) mod schema;
