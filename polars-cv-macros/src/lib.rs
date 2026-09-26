//! The typed catalogue's derives: `#[derive(Ops)]` and `#[derive(Resolve)]`.
//!
//! Every wire vocabulary — the op families, the geometry accessors, the
//! sources and the sinks — is an enum deriving `Ops`. Each variant carrying
//! `#[op(name = ...)]` is one entry: its doc comment is the generated
//! docstring, each field's doc comment its `Args:` entry, and from the one
//! field list the derive emits the strict wire (unknown key and missing
//! required field refused, a declared default applied), the catalogue entry
//! the Python surface is generated from, the slot visitor and the samples the
//! registry-driven tests cover each entry with. A family with per-row values
//! is generic over its mode and also derives `Resolve` (`Wire` → `Exec`).
//!
//! Attributes:
//!
//! - variant: `#[op(name = "...", sample = {...})]`, plus `python = "..."`
//!   (the Python name when it differs) and `visibility = Internal | LazyOnly`
//!   (default `Public`).
//! - field: `#[param(default = <literal>)]` — the default the wire applies
//!   and the Python signature shows. An `Option<_>` field is absent as `None`
//!   and declares none. Which parameters are positional is not declared:
//!   `gen_ops.py` derives it from which fields are required.

use proc_macro::TokenStream;
use syn::{parse_macro_input, Attribute, DeriveInput, Expr, Lit};

mod modal;

/// The engine enums as the typed op catalogue: see `modal`.
#[proc_macro_derive(Ops, attributes(op, param))]
pub fn derive_ops(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match modal::derive_ops(&input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// `Wire` → `Exec`, field by field: see `modal`.
#[proc_macro_derive(Resolve)]
pub fn derive_resolve(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match modal::derive_resolve(&input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn doc_of(attrs: &[Attribute]) -> String {
    let lines: Vec<String> = attrs
        .iter()
        .filter(|a| a.path().is_ident("doc"))
        .filter_map(|a| match &a.meta {
            syn::Meta::NameValue(nv) => match &nv.value {
                Expr::Lit(syn::ExprLit {
                    lit: Lit::Str(s), ..
                }) => Some(s.value()),
                _ => None,
            },
            _ => None,
        })
        .map(|l| l.strip_prefix(' ').unwrap_or(&l).to_string())
        .collect();
    lines.join("\n").trim().to_string()
}
