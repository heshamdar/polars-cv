//! `dtype_table!` is the single authority for every name a dtype has.
//!
//! A dtype is spelled three ways across this workspace's boundaries: a short
//! name in the graph JSON, a wire code in the VIEW binary header, and a numpy
//! name in the numpy/torch sink and the metadata accessors. These tests pin
//! the properties that make one table safe to rely on — total coverage,
//! round-tripping, and uniqueness — so a table that grows a wrong or duplicate
//! entry fails here rather than at a user's dtype.
//!
//! The compiler already rejects a `DType` variant the table omits (the
//! generated accessors match exhaustively). What it cannot check is that the
//! *values* are distinct and reversible, which is what follows.

use std::collections::HashSet;
use view_buffer::protocol::{dtype_to_u8, u8_to_dtype};
use view_buffer::DType;

#[test]
fn wire_codes_are_unique() {
    let codes: HashSet<u8> = DType::ALL.iter().map(|d| d.wire_code()).collect();
    assert_eq!(
        codes.len(),
        DType::ALL.len(),
        "two dtypes share a VIEW wire code — every blob written with the \
         duplicate would decode as whichever variant matched first"
    );
}

#[test]
fn names_are_unique() {
    for (label, names) in [
        (
            "short",
            DType::ALL
                .iter()
                .map(|d| d.short_name())
                .collect::<Vec<_>>(),
        ),
        (
            "numpy",
            DType::ALL
                .iter()
                .map(|d| d.numpy_name())
                .collect::<Vec<_>>(),
        ),
    ] {
        let unique: HashSet<_> = names.iter().collect();
        assert_eq!(
            unique.len(),
            names.len(),
            "duplicate {label} name in dtype_table!: {names:?}"
        );
    }
}

#[test]
fn every_representation_round_trips() {
    for &dt in DType::ALL {
        assert_eq!(
            DType::from_wire_code(dt.wire_code()),
            Some(dt),
            "{dt:?} does not survive a wire-code round trip"
        );
        assert_eq!(
            DType::from_short_name(dt.short_name()),
            Some(dt),
            "{dt:?} does not survive a short-name round trip"
        );
        assert_eq!(
            DType::from_numpy_name(dt.numpy_name()),
            Some(dt),
            "{dt:?} does not survive a numpy-name round trip"
        );
    }
}

#[test]
fn protocol_helpers_agree_with_the_table() {
    // `dtype_to_u8`/`u8_to_dtype` are the protocol-facing spelling of the same
    // table. They delegate today; this fails if either regrows its own match.
    for &dt in DType::ALL {
        assert_eq!(dtype_to_u8(dt), dt.wire_code());
        assert_eq!(u8_to_dtype(dt.wire_code()), Some(dt));
    }
}

#[test]
fn wire_codes_match_the_frozen_view_format() {
    // The VIEW binary format is on disk in users' files: these codes are not
    // ours to renumber. Pinned literally so a reordering of dtype_table! that
    // shifts a code fails here instead of silently misreading old blobs.
    let expected: &[(DType, u8)] = &[
        (DType::U8, 1),
        (DType::I8, 2),
        (DType::U16, 3),
        (DType::I16, 4),
        (DType::U32, 5),
        (DType::I32, 6),
        (DType::F32, 7),
        (DType::F64, 8),
        (DType::U64, 9),
        (DType::I64, 10),
    ];
    for &(dt, code) in expected {
        assert_eq!(dt.wire_code(), code, "{dt:?} changed VIEW wire code");
    }
}

#[test]
fn unknown_encodings_are_rejected_not_guessed() {
    assert_eq!(u8_to_dtype(0), None);
    assert_eq!(u8_to_dtype(11), None);
    assert_eq!(u8_to_dtype(255), None);
    assert_eq!(DType::from_numpy_name("float16"), None);
    assert_eq!(DType::from_short_name("f16"), None);
}
