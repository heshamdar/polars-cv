//! `#[derive(Ops)]` and `#[derive(Resolve)]`: the engine enums are the typed
//! op catalogue (see `view_buffer::mode`).
//!
//! A type deriving these is generic over its mode, the first type parameter
//! (`enum ImageOpKind<M: Mode = Exec>`). A field whose type names that
//! parameter (`M::V<u32>`, `Option<M::L<u32>>`, `RasterSize<M>`) changes with
//! the mode; any other field is the same in every mode.

use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{spanned::Spanned, Attribute, Data, DeriveInput, Expr, Fields, Ident, Type};

use crate::doc_of;

/// The mode parameter: the first type parameter.
fn mode_param(input: &DeriveInput) -> syn::Result<Ident> {
    input
        .generics
        .type_params()
        .next()
        .map(|p| p.ident.clone())
        .ok_or_else(|| {
            syn::Error::new(
                input.span(),
                "a moded op type is generic over its mode: `enum X<M: Mode = Exec>`",
            )
        })
}

/// Whether `ty` names the mode parameter anywhere.
fn names_mode(ty: &Type, mode: &Ident) -> bool {
    struct Finder<'a> {
        mode: &'a Ident,
        found: bool,
    }
    impl<'ast> Visit<'ast> for Finder<'_> {
        fn visit_ident(&mut self, i: &'ast Ident) {
            if i == self.mode {
                self.found = true;
            }
        }
    }
    let mut f = Finder { mode, found: false };
    f.visit_type(ty);
    f.found
}

/// `ty` with the mode parameter replaced by `Wire`.
fn wire_type(ty: &Type, mode: &Ident) -> Type {
    struct Replace<'a> {
        mode: &'a Ident,
    }
    impl VisitMut for Replace<'_> {
        fn visit_type_mut(&mut self, ty: &mut Type) {
            if let Type::Path(tp) = ty {
                if tp.qself.is_none()
                    && tp
                        .path
                        .segments
                        .first()
                        .is_some_and(|s| s.ident == *self.mode)
                {
                    let rest: Vec<_> = tp.path.segments.iter().skip(1).cloned().collect();
                    if rest.is_empty() {
                        *ty = syn::parse_quote!(::view_buffer::mode::Wire);
                        return;
                    }
                    let mut rest_tokens = TokenStream2::new();
                    for (i, seg) in rest.iter().enumerate() {
                        if i > 0 {
                            rest_tokens.extend(quote!(::));
                        }
                        let mut seg = seg.clone();
                        self.visit_path_segment_mut(&mut seg);
                        rest_tokens.extend(quote!(#seg));
                    }
                    *ty = syn::parse_quote!(
                        <::view_buffer::mode::Wire as ::view_buffer::mode::Mode>::#rest_tokens
                    );
                    return;
                }
            }
            syn::visit_mut::visit_type_mut(self, ty);
        }
    }
    let mut ty = ty.clone();
    Replace { mode }.visit_type_mut(&mut ty);
    ty
}

/// Whether `ty` is written `Option<...>`: an optional field (absent on the
/// wire means `None`).
fn is_option(ty: &Type) -> bool {
    matches!(ty, Type::Path(tp) if tp.qself.is_none()
        && tp.path.segments.last().is_some_and(|s| s.ident == "Option"))
}

/// The type with the mode parameter set to `mode_ty`, as `Name<Wire>`.
fn with_mode(input: &DeriveInput, mode_ty: TokenStream2) -> TokenStream2 {
    let name = &input.ident;
    quote!(#name<#mode_ty>)
}

/// `#[derive(Resolve)]`: `X<Wire>` → `X<Exec>`, field by field.
pub fn derive_resolve(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let mode = mode_param(input)?;
    let wire = with_mode(input, quote!(::view_buffer::mode::Wire));
    let exec = with_mode(input, quote!(::view_buffer::mode::Exec));
    let name = &input.ident;
    let convert = |fields: &Fields, ctor: TokenStream2| -> TokenStream2 {
        match fields {
            Fields::Unit => quote! { Self::Exec::#ctor },
            Fields::Named(named) => {
                let values = named.named.iter().map(|f| {
                    let id = f.ident.as_ref().unwrap();
                    if names_mode(&f.ty, &mode) {
                        quote! { #id: ::view_buffer::mode::Resolve::resolve(#id, values)? }
                    } else {
                        quote! { #id: ::core::clone::Clone::clone(#id) }
                    }
                });
                quote! { #name::#ctor { #(#values),* } }
            }
            Fields::Unnamed(unnamed) => {
                let values = unnamed.unnamed.iter().enumerate().map(|(i, f)| {
                    let id = format_ident!("f{i}");
                    if names_mode(&f.ty, &mode) {
                        quote! { ::view_buffer::mode::Resolve::resolve(#id, values)? }
                    } else {
                        quote! { ::core::clone::Clone::clone(#id) }
                    }
                });
                quote! { #name::#ctor(#(#values),*) }
            }
        }
    };
    let pattern = |fields: &Fields, ctor: TokenStream2| -> TokenStream2 {
        match fields {
            Fields::Unit => quote! { #name::#ctor },
            Fields::Named(named) => {
                let ids = named.named.iter().map(|f| f.ident.clone().unwrap());
                quote! { #name::#ctor { #(#ids),* } }
            }
            Fields::Unnamed(unnamed) => {
                let ids = (0..unnamed.unnamed.len()).map(|i| format_ident!("f{i}"));
                quote! { #name::#ctor(#(#ids),*) }
            }
        }
    };
    let body = match &input.data {
        Data::Enum(data) => {
            let arms = data.variants.iter().map(|v| {
                let ident = &v.ident;
                let pat = pattern(&v.fields, quote!(#ident));
                let out = match &v.fields {
                    Fields::Unit => quote! { #name::#ident },
                    _ => convert(&v.fields, quote!(#ident)),
                };
                quote! { #pat => #out, }
            });
            quote! {
                #[allow(unused_variables)]
                match self { #(#arms)* }
            }
        }
        Data::Struct(data) => {
            let pat = match &data.fields {
                Fields::Named(named) => {
                    let ids = named.named.iter().map(|f| f.ident.clone().unwrap());
                    quote! { #name { #(#ids),* } }
                }
                Fields::Unnamed(unnamed) => {
                    let ids = (0..unnamed.unnamed.len()).map(|i| format_ident!("f{i}"));
                    quote! { #name(#(#ids),*) }
                }
                Fields::Unit => quote! { #name },
            };
            let out = match &data.fields {
                Fields::Named(named) => {
                    let values = named.named.iter().map(|f| {
                        let id = f.ident.as_ref().unwrap();
                        if names_mode(&f.ty, &mode) {
                            quote! { #id: ::view_buffer::mode::Resolve::resolve(#id, values)? }
                        } else {
                            quote! { #id: ::core::clone::Clone::clone(#id) }
                        }
                    });
                    quote! { #name { #(#values),* } }
                }
                Fields::Unnamed(unnamed) => {
                    let values = unnamed.unnamed.iter().enumerate().map(|(i, f)| {
                        let id = format_ident!("f{i}");
                        if names_mode(&f.ty, &mode) {
                            quote! { ::view_buffer::mode::Resolve::resolve(#id, values)? }
                        } else {
                            quote! { ::core::clone::Clone::clone(#id) }
                        }
                    });
                    quote! { #name(#(#values),*) }
                }
                Fields::Unit => quote! { #name },
            };
            quote! {
                #[allow(unused_variables)]
                let #pat = self;
                #out
            }
        }
        Data::Union(_) => return Err(syn::Error::new(input.span(), "no unions")),
    };
    Ok(quote! {
        impl ::view_buffer::mode::Resolve for #wire {
            type Exec = #exec;
            fn resolve<V: ::view_buffer::mode::Values>(
                &self,
                values: &V,
            ) -> ::core::result::Result<Self::Exec, V::Error> {
                ::core::result::Result::Ok({ #body })
            }
        }
    })
}

struct OpAttrs {
    name: Option<String>,
    python: Option<String>,
    visibility: String,
    sample: Option<TokenStream2>,
}

fn op_attrs(attrs: &[Attribute]) -> syn::Result<OpAttrs> {
    let mut out = OpAttrs {
        name: None,
        python: None,
        visibility: "public".into(),
        sample: None,
    };
    for attr in attrs.iter().filter(|a| a.path().is_ident("op")) {
        attr.parse_nested_meta(|m| {
            if m.path.is_ident("sample") {
                let value = m.value()?;
                let group: proc_macro2::Group = value.parse()?;
                out.sample = Some(quote!(#group));
                return Ok(());
            }
            let value: syn::LitStr = m.value()?.parse()?;
            if m.path.is_ident("name") {
                out.name = Some(value.value());
            } else if m.path.is_ident("python") {
                out.python = Some(value.value());
            } else if m.path.is_ident("visibility") {
                let v = value.value();
                if !matches!(v.as_str(), "public" | "lazy_only" | "internal") {
                    return Err(m.error("visibility is public, lazy_only or internal"));
                }
                out.visibility = v;
            } else {
                return Err(m.error("unknown #[op] key (name, python, visibility, sample)"));
            }
            Ok(())
        })?;
    }
    Ok(out)
}

fn param_default(attrs: &[Attribute]) -> syn::Result<Option<Expr>> {
    let mut default = None;
    for attr in attrs.iter().filter(|a| a.path().is_ident("param")) {
        attr.parse_nested_meta(|m| {
            if m.path.is_ident("default") {
                default = Some(m.value()?.parse()?);
            } else {
                return Err(m.error("unknown #[param] key (default)"));
            }
            Ok(())
        })?;
    }
    Ok(default)
}

/// `#[derive(Ops)]`: each variant carrying `#[op(name = ...)]` is one wire
/// op. A variant without it is engine-internal (fusion's output, say): it has
/// no wire form, and the wire never produces it.
pub fn derive_ops(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let mode = mode_param(input)?;
    let name = &input.ident;
    let wire = with_mode(input, quote!(::view_buffer::mode::Wire));
    // A struct is one op: the same generation over a single "variant" whose
    // attributes are the struct's and whose constructor is the struct itself.
    let variants: Vec<(Option<&Ident>, &[Attribute], &Fields)> = match &input.data {
        Data::Enum(data) => data
            .variants
            .iter()
            .map(|v| (Some(&v.ident), v.attrs.as_slice(), &v.fields))
            .collect(),
        Data::Struct(data) => vec![(None, input.attrs.as_slice(), &data.fields)],
        Data::Union(_) => {
            return Err(syn::Error::new(
                input.span(),
                "#[derive(Ops)] needs an enum or struct",
            ))
        }
    };

    let mut names = Vec::new();
    let mut from_arms = Vec::new();
    let mut name_arms = Vec::new();
    let mut fields_arms = Vec::new();
    let mut visit_arms = Vec::new();
    let mut descs = Vec::new();
    let mut samples = Vec::new();

    for &(ident, v_attrs, v_fields) in &variants {
        // `Name::Variant` for an enum, `Name` for a struct.
        let path = match ident {
            Some(ident) => quote!(#name::#ident),
            None => quote!(#name),
        };
        let span = ident.map_or_else(|| input.span(), |i| i.span());
        let attrs = op_attrs(v_attrs)?;
        let Some(wire_name) = attrs.name else {
            if attrs.sample.is_some() || attrs.python.is_some() || ident.is_none() {
                return Err(syn::Error::new(
                    span,
                    "#[op(...)] keys need a `name`: a variant without one is engine-internal",
                ));
            }
            let pat = match v_fields {
                Fields::Unit => quote!(#path),
                Fields::Named(_) => quote!(#path { .. }),
                Fields::Unnamed(_) => quote!(#path(..)),
            };
            name_arms.push(quote! { #pat => ::core::option::Option::None, });
            fields_arms.push(quote! { #pat => ::core::option::Option::None, });
            visit_arms.push(quote! { #pat => {} });
            continue;
        };
        let doc = doc_of(v_attrs);
        if doc.is_empty() {
            return Err(syn::Error::new(
                span,
                "a wire op needs a doc comment: it is the generated Python docstring",
            ));
        }
        let Some(sample) = attrs.sample else {
            return Err(syn::Error::new(
                span,
                "a wire op needs `#[op(sample = {...})]`: one valid instance, which the \
                 registry-driven tests cover it with",
            ));
        };
        let named: Vec<&syn::Field> = match v_fields {
            Fields::Unit => Vec::new(),
            Fields::Named(n) => n.named.iter().collect(),
            Fields::Unnamed(_) => {
                return Err(syn::Error::new(
                    span,
                    "a wire op has named fields: the wire names each parameter",
                ))
            }
        };
        let python = attrs.python.unwrap_or_else(|| wire_name.clone());
        let visibility = attrs.visibility;
        let field_names: Vec<String> = named
            .iter()
            .map(|f| f.ident.as_ref().unwrap().to_string())
            .collect();
        let ids: Vec<&Ident> = named.iter().map(|f| f.ident.as_ref().unwrap()).collect();

        let mut field_descs = Vec::new();
        let mut takes = Vec::new();
        let mut puts = Vec::new();
        let mut visits = Vec::new();
        for f in &named {
            let id = f.ident.as_ref().unwrap();
            let fname = id.to_string();
            let fdoc = doc_of(&f.attrs);
            if fdoc.is_empty() {
                return Err(syn::Error::new(
                    f.span(),
                    format!("field `{fname}` needs a doc comment: it is the generated Args: entry"),
                ));
            }
            let wty = wire_type(&f.ty, &mode);
            let default = match param_default(&f.attrs)? {
                Some(expr) => {
                    quote! { ::core::option::Option::Some(::serde_json::Value::from(#expr)) }
                }
                None => quote! { ::core::option::Option::None },
            };
            field_descs.push(quote! {
                ::view_buffer::mode::FieldDesc {
                    name: #fname,
                    doc: #fdoc,
                    default: #default,
                    ty: <#wty as ::view_buffer::mode::FieldType>::describe(),
                }
            });
            if is_option(&f.ty) {
                takes.push(quote! {
                    let #id: #wty = ::view_buffer::mode::take_field(&mut fields, #fname)?;
                });
                puts.push(quote! {
                    if let ::core::option::Option::Some(value) = #id {
                        map.insert(#fname.into(), ::serde_json::to_value(value)
                            .expect("an op field serializes"));
                    }
                });
            } else {
                takes.push(quote! {
                    let #id: #wty = ::view_buffer::mode::take_field(&mut fields, #fname)?
                        .ok_or_else(|| ::view_buffer::mode::missing_field(#fname))?;
                });
                puts.push(quote! {
                    map.insert(#fname.into(), ::serde_json::to_value(#id)
                        .expect("an op field serializes"));
                });
            }
            visits.push(quote! {
                ::view_buffer::mode::FieldType::visit_slots(#id, &mut |slot| f(#fname, slot));
            });
        }

        let (pat, ctor) = if named.is_empty() {
            match v_fields {
                Fields::Unit => (quote!(#path), quote!(#path)),
                _ => (quote!(#path {}), quote!(#path {})),
            }
        } else {
            (quote!(#path { #(#ids),* }), quote!(#path { #(#ids),* }))
        };

        let rest = match v_fields {
            Fields::Unit => quote!(#path),
            _ => quote!(#path { .. }),
        };
        names.push(wire_name.clone());
        from_arms.push(quote! {
            #wire_name => ::core::option::Option::Some((|| {
                let mut fields = match fields {
                    ::serde_json::Value::Object(map) => map,
                    other => return ::core::result::Result::Err(
                        format!("an op's fields are an object, got {other}")),
                };
                #(#takes)*
                ::view_buffer::mode::refuse_unknown_fields(&fields, &[#(#field_names),*])?;
                ::core::result::Result::Ok(#ctor)
            })()),
        });
        name_arms.push(quote! { #rest => ::core::option::Option::Some(#wire_name), });
        fields_arms.push(quote! {
            #[allow(unused_variables)]
            #pat => {
                #[allow(unused_mut)]
                let mut map = ::serde_json::Map::new();
                #(#puts)*
                ::core::option::Option::Some(map)
            }
        });
        visit_arms.push(quote! {
            #[allow(unused_variables)]
            #pat => { #(#visits)* }
        });
        descs.push(quote! {
            ::view_buffer::mode::OpDesc {
                name: #wire_name,
                python: #python,
                visibility: #visibility,
                doc: #doc,
                fields: ::std::vec![#(#field_descs),*],
            }
        });
        samples.push(quote! {
            Self::from_wire(#wire_name, ::serde_json::json!(#sample))
                .expect("registered")
                .unwrap_or_else(|e| panic!("sample for '{}': {e}", #wire_name))
        });
    }

    Ok(quote! {
        impl #wire {
            /// Every wire op name this family defines.
            pub const WIRE_NAMES: &'static [&'static str] = &[#(#names),*];

            /// The op `name` from its wire fields (the object without
            /// `"op"`); `None` when this family has no op `name`.
            pub fn from_wire(
                name: &str,
                fields: ::serde_json::Value,
            ) -> ::core::option::Option<::core::result::Result<Self, ::std::string::String>> {
                match name {
                    #(#from_arms)*
                    _ => ::core::option::Option::None,
                }
            }

            /// The op's wire name; `None` for an engine-internal variant.
            pub fn wire_name(&self) -> ::core::option::Option<&'static str> {
                match self { #(#name_arms)* }
            }

            /// The op's wire fields (without `"op"`), an absent optional
            /// field left out; `None` for an engine-internal variant.
            pub fn wire_fields(
                &self,
            ) -> ::core::option::Option<::serde_json::Map<::std::string::String, ::serde_json::Value>>
            {
                match self { #(#fields_arms)* }
            }

            /// Call `f(field, slot)` for every slot a field reads.
            #[allow(unused_variables)]
            pub fn visit_slots(&self, f: &mut dyn FnMut(&'static str, usize)) {
                match self { #(#visit_arms)* }
            }

            /// Every op's catalogue entry, in declaration order.
            pub fn catalog() -> ::std::vec::Vec<::view_buffer::mode::OpDesc> {
                ::std::vec![#(#descs),*]
            }

            /// One valid instance of every op, in declaration order.
            pub fn samples() -> ::std::vec::Vec<Self> {
                ::std::vec![#(#samples),*]
            }
        }

        impl ::view_buffer::mode::WireOps for #wire {
            fn from_wire(
                name: &str,
                fields: ::serde_json::Value,
            ) -> ::core::option::Option<::core::result::Result<Self, ::std::string::String>> {
                Self::from_wire(name, fields)
            }
            fn wire_name(&self) -> ::core::option::Option<&'static str> {
                Self::wire_name(self)
            }
            fn visit_slots(&self, f: &mut dyn FnMut(&'static str, usize)) {
                Self::visit_slots(self, f)
            }
            fn catalog() -> ::std::vec::Vec<::view_buffer::mode::OpDesc> {
                Self::catalog()
            }
        }
    })
}
