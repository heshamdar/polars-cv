//! `#[derive(Op)]`: the per-op half of polars-cv's typed op catalogue.
//!
//! An operation is one struct in `polars-cv/src/ops/`. This derive reads that
//! struct — its doc comment, its fields, their doc comments and their
//! `#[param(...)]` markers — and emits the `OpFields` impl the catalogue is
//! built from: the description the Python builder is generated from, and the
//! slot visitor graph compilation uses to classify the op as static or
//! per-row. Both are derived from the one field list, so a field cannot be
//! declared and then left out of either.
//!
//! The derive also *rejects* what would let the wire and the definition drift:
//! a struct without `#[serde(deny_unknown_fields)]`, a field carrying its own
//! `#[serde(...)]` (a rename, default or skip would make the wire differ from
//! the catalogue) and a missing doc comment (it is the generated docstring).
//!
//! Attributes:
//!
//! - struct: `#[op(python = "name")]` — the Python method name when it differs
//!   from the wire name.
//! - field: `#[param(default = <literal>)]` — the Python signature default. An
//!   `Option<_>` field defaults to `None`. Which parameters are positional is
//!   not declared: `gen_ops.py` derives it from which fields are required.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, spanned::Spanned, Attribute, Data, DeriveInput, Expr, Fields, Lit};

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

#[proc_macro_derive(Op, attributes(op, param))]
pub fn derive_op(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand(&input) {
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

fn has_deny_unknown_fields(attrs: &[Attribute]) -> bool {
    attrs
        .iter()
        .filter(|a| a.path().is_ident("serde"))
        .any(|a| {
            let mut found = false;
            let _ = a.parse_nested_meta(|m| {
                if m.path.is_ident("deny_unknown_fields") {
                    found = true;
                } else if m.input.peek(syn::Token![=]) {
                    let _: Expr = m.value()?.parse()?;
                }
                Ok(())
            });
            found
        })
}

fn expand(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new(
            input.span(),
            "#[derive(Op)] needs a struct",
        ));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(syn::Error::new(
            input.span(),
            "#[derive(Op)] needs named fields (use `struct Abs {}` for no parameters)",
        ));
    };
    if !has_deny_unknown_fields(&input.attrs) {
        return Err(syn::Error::new(
            input.span(),
            "an op struct must be #[serde(deny_unknown_fields)]: an unknown key is \
             a parameter nothing reads",
        ));
    }
    let doc = doc_of(&input.attrs);
    if doc.is_empty() {
        return Err(syn::Error::new(
            input.span(),
            "an op needs a doc comment: it is the generated Python docstring",
        ));
    }

    let mut python: Option<String> = None;
    for attr in input.attrs.iter().filter(|a| a.path().is_ident("op")) {
        attr.parse_nested_meta(|m| {
            let value: syn::LitStr = m.value()?.parse()?;
            if m.path.is_ident("python") {
                python = Some(value.value());
            } else {
                return Err(m.error("unknown #[op] key (python)"));
            }
            Ok(())
        })?;
    }

    let mut descs = Vec::new();
    let mut visits = Vec::new();
    let mut idents = Vec::new();
    for field in &fields.named {
        let ident = field.ident.as_ref().expect("named field");
        let fname = ident.to_string();
        let ty = &field.ty;
        if let Some(attr) = field.attrs.iter().find(|a| a.path().is_ident("serde")) {
            return Err(syn::Error::new(
                attr.span(),
                "op fields take no #[serde] attribute: the wire must be exactly the \
                 catalogue (field name, required unless Option)",
            ));
        }
        let fdoc = doc_of(&field.attrs);
        if fdoc.is_empty() {
            return Err(syn::Error::new(
                field.span(),
                format!("field `{fname}` needs a doc comment: it is the generated Args: entry"),
            ));
        }
        let mut default: Option<Expr> = None;
        for attr in field.attrs.iter().filter(|a| a.path().is_ident("param")) {
            attr.parse_nested_meta(|m| {
                if m.path.is_ident("default") {
                    default = Some(m.value()?.parse()?);
                } else {
                    return Err(m.error("unknown #[param] key (default)"));
                }
                Ok(())
            })?;
        }
        let default_tokens = match default {
            Some(expr) => quote! { ::core::option::Option::Some(::serde_json::Value::from(#expr)) },
            None => quote! { ::core::option::Option::None },
        };
        descs.push(quote! {
            crate::ops::FieldDesc {
                name: #fname,
                doc: #fdoc,
                default: #default_tokens,
                ty: <#ty as crate::ops::FieldType>::describe(),
            }
        });
        visits.push(quote! {
            crate::ops::FieldType::visit_slots(#ident, &mut |slot| f(#fname, slot));
        });
        idents.push(ident.clone());
    }

    let python_tokens = match python {
        Some(p) => quote! { ::core::option::Option::Some(#p) },
        None => quote! { ::core::option::Option::None },
    };
    // `f` is unused for an op with no fields; the binding keeps one signature.
    Ok(quote! {
        impl crate::ops::OpFields for #name {
            const DOC: &'static str = #doc;
            const PYTHON_NAME: ::core::option::Option<&'static str> = #python_tokens;

            fn fields() -> ::std::vec::Vec<crate::ops::FieldDesc> {
                ::std::vec![#(#descs),*]
            }

            #[allow(unused_variables)]
            fn visit_slots(&self, f: &mut dyn FnMut(&'static str, usize)) {
                // Exhaustive by construction: the pattern is the field list.
                let #name { #(#idents),* } = self;
                #(#visits)*
            }
        }
    })
}
