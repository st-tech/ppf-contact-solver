// File: crates/ppf-cts-formats/src/envelope.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Schema-version envelope wrapped around every cross-language payload.
// Bumped exactly when the inner schema changes incompatibly: an old
// reader sees a higher version and refuses with a typed error rather
// than silently mis-parsing. Producers and consumers always go through
// `to_cbor` / `from_cbor`, never raw serde, so the envelope can't be
// forgotten.

use serde::{de::DeserializeOwned, Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope<T> {
    pub version: u32,
    pub kind: String,
    pub payload: T,
}

impl<T> Envelope<T> {
    pub fn new(kind: impl Into<String>, payload: T) -> Self {
        Self {
            version: SCHEMA_VERSION,
            kind: kind.into(),
            payload,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FormatError {
    #[error("CBOR encode: {0}")]
    CborSer(String),
    #[error("CBOR decode: {0}")]
    CborDe(String),
    #[error(
        "schema version mismatch: payload has version {found}, this build expects {expected}"
    )]
    VersionMismatch { found: u32, expected: u32 },
    #[error("payload kind mismatch: expected {expected:?}, got {found:?}")]
    KindMismatch { expected: String, found: String },
}

impl<E: std::fmt::Debug> From<ciborium::ser::Error<E>> for FormatError {
    fn from(e: ciborium::ser::Error<E>) -> Self {
        FormatError::CborSer(format!("{e:?}"))
    }
}

impl<E: std::fmt::Debug> From<ciborium::de::Error<E>> for FormatError {
    fn from(e: ciborium::de::Error<E>) -> Self {
        FormatError::CborDe(format!("{e:?}"))
    }
}

pub fn to_cbor<T: Serialize>(kind: &str, payload: &T) -> Result<Vec<u8>, FormatError> {
    to_cbor_with_version(SCHEMA_VERSION, kind, payload)
}

pub fn from_cbor<T: DeserializeOwned>(kind: &str, bytes: &[u8]) -> Result<T, FormatError> {
    from_cbor_with_version(SCHEMA_VERSION, kind, bytes)
}

/// Encode a payload under an EXPLICIT envelope version instead of the
/// shared [`SCHEMA_VERSION`]. Used by payloads that evolve on their own
/// cadence (e.g. [`crate::status::RunStatus`]) so bumping their layout
/// does not invalidate every other CBOR file on disk. Cross-language
/// wire payloads (Scene / Param) keep using [`to_cbor`].
pub fn to_cbor_with_version<T: Serialize>(
    version: u32,
    kind: &str,
    payload: &T,
) -> Result<Vec<u8>, FormatError> {
    let env = Envelope {
        version,
        kind: kind.to_string(),
        payload,
    };
    let mut buf = Vec::new();
    ciborium::into_writer(&env, &mut buf)?;
    Ok(buf)
}

/// Decode a payload, refusing any envelope whose version does not match
/// `expected_version` (the per-kind counterpart of [`from_cbor`]). A
/// newer-versioned record yields [`FormatError::VersionMismatch`] rather
/// than a silent mis-parse.
pub fn from_cbor_with_version<T: DeserializeOwned>(
    expected_version: u32,
    kind: &str,
    bytes: &[u8],
) -> Result<T, FormatError> {
    let env: Envelope<T> = ciborium::from_reader(bytes)?;
    check_envelope_meta(&env.version, expected_version, &env.kind, kind)?;
    Ok(env.payload)
}

fn check_envelope_meta(
    version: &u32,
    expected_version: u32,
    found_kind: &str,
    expected_kind: &str,
) -> Result<(), FormatError> {
    if *version != expected_version {
        return Err(FormatError::VersionMismatch {
            found: *version,
            expected: expected_version,
        });
    }
    if found_kind != expected_kind {
        return Err(FormatError::KindMismatch {
            expected: expected_kind.to_string(),
            found: found_kind.to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Demo {
        a: u32,
        b: String,
    }

    #[test]
    fn cbor_roundtrip() {
        let v = Demo { a: 7, b: "hi".into() };
        let bytes = to_cbor("Demo", &v).unwrap();
        let back: Demo = from_cbor("Demo", &bytes).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn rejects_wrong_kind() {
        let v = Demo { a: 1, b: String::new() };
        let bytes = to_cbor("Demo", &v).unwrap();
        let err = from_cbor::<Demo>("Other", &bytes).unwrap_err();
        assert!(matches!(err, FormatError::KindMismatch { .. }));
    }
}
