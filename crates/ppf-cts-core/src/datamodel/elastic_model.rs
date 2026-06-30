// File: crates/ppf-cts-core/src/datamodel/elastic_model.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Single authoritative name<->id table for the elastic constitutive
// model. The Python exporter writes per-element model ids as u8 and the
// solver reads them back; both must agree on the exact numeric encoding.
// Defining the ordering once here removes the drift hazard of two
// hand-rolled mappings.
//
// The id is the position in `MODEL_NAMES` and must match the variant
// discriminant order of the solver's `repr(C)` `Model` enum (and the
// matching C++ `Model` enum), so do not reorder or insert in the middle:
// only append at the end.

/// Canonical ordering of the elastic model names. The index of each name
/// is its on-disk u8 id (arap=0, stvk=1, baraff-witkin=2, snhk=3, pdrd=4).
/// `pdrd` denotes an Painless Differentiable Rotation Dynamics face: it carries no per-element
/// elastic energy (the solver gates elastic dispatch off `Model::Pdrd`);
/// the id only exists so PDRD faces stay key-compatible with shells in the
/// per-face model stream.
pub const MODEL_NAMES: &[&str] = &["arap", "stvk", "baraff-witkin", "snhk", "pdrd"];

/// Map an elastic model name to its u8 id. Returns `None` for an
/// unknown name so callers can raise their own error.
pub fn model_name_to_id(name: &str) -> Option<u8> {
    MODEL_NAMES
        .iter()
        .position(|&n| n == name)
        .map(|i| i as u8)
}

/// Map a u8 id back to its elastic model name. Returns `None` for an
/// out-of-range id so callers can raise their own error.
pub fn model_id_to_name(id: u8) -> Option<&'static str> {
    MODEL_NAMES.get(id as usize).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_name_id_round_trip() {
        for &name in MODEL_NAMES {
            let id = model_name_to_id(name).expect("name should map to an id");
            let back = model_id_to_name(id).expect("id should map back to a name");
            assert_eq!(back, name);
        }
    }

    #[test]
    fn model_id_to_name_round_trip() {
        for id in 0..MODEL_NAMES.len() as u8 {
            let name = model_id_to_name(id).expect("id should map to a name");
            let back = model_name_to_id(name).expect("name should map back to an id");
            assert_eq!(back, id);
        }
    }

    #[test]
    fn model_encoding_is_stable() {
        assert_eq!(model_name_to_id("arap"), Some(0));
        assert_eq!(model_name_to_id("stvk"), Some(1));
        assert_eq!(model_name_to_id("baraff-witkin"), Some(2));
        assert_eq!(model_name_to_id("snhk"), Some(3));
        assert_eq!(model_name_to_id("pdrd"), Some(4));
        assert_eq!(model_name_to_id("unknown"), None);
        assert_eq!(model_id_to_name(5), None);
    }
}
