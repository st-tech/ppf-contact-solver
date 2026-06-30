// File: crates/ppf-cts-core/src/datamodel/param_manager.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// `ParamManager`: the user-facing simulation-parameter configurator.
// Direct port of frontend/_session_.py:ParamManager.
//
// What it adds on top of the bare `ParamHolder` from
// crates/ppf-cts-core/src/datamodel/params.rs:
//   * A `default_param` snapshot for `clear_all` to restore from.
//   * Dynamic-parameter overrides keyed by time. The time cursor
//     (`dyn(key)` / `time(t)` / `change(value)` / `hold()`) lives in
//     the frontend Python `ParamManager`; it assembles the per-key
//     `[(0.0, initial), (t, value), ...]` list and ships it here via
//     `inject_dyn_entries`. The Rust side stores and serializes that
//     list but does not host its own cursor.
//   * `to_toml_string()` produces the same `[param]` body the
//     Python `export(path)` writes to `param.toml`.
//
// Out of scope:
//   * Filesystem I/O: the Python `export(path)` writes files; we
//     return strings and let the caller persist (mirrors the
//     ppf-cts-server pattern of separating data from disk).
//   * `Utils.is_fast_check()` integration: fast-check forces
//     `frames=1` on export; the Rust port honors a `fast_check`
//     flag passed to `to_toml_string` instead of querying global
//     state.

use std::collections::BTreeMap;
use std::fmt::Write;

use super::params::{app_param, ParamHolder, ParamValue};

#[derive(Debug, thiserror::Error)]
pub enum ParamManagerError {
    #[error("Key cannot contain underscore. Use '-' instead.")]
    UnderscoreInKey,
    #[error("Key {0} does not exist")]
    UnknownKey(String),
    #[error("I/O failure: {0}")]
    Io(#[from] std::io::Error),
    #[error("Value must be float, int, bool, or list. {0:?} is given.")]
    UnsupportedDynValue(ParamValue),
}

#[derive(Debug, Clone)]
pub struct ParamManager {
    holder: ParamHolder,
    default: ParamHolder,
    dyn_param: BTreeMap<String, Vec<(f64, ParamValue)>>,
}

impl ParamManager {
    /// Construct with the default `app_param()` defaults.
    pub fn new() -> Self {
        let holder = app_param();
        Self {
            default: holder.clone(),
            holder,
            dyn_param: BTreeMap::new(),
        }
    }

    /// Snapshot copy.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Set a parameter. Refuses underscore-bearing keys (they're a
    /// common typo for hyphen-form keys; the Python source raises).
    /// `value = None` in Python becomes `Bool(true)` here, same
    /// semantics, just typed.
    pub fn set(
        &mut self,
        key: &str,
        value: Option<ParamValue>,
    ) -> Result<&mut Self, ParamManagerError> {
        if key.contains('_') {
            return Err(ParamManagerError::UnderscoreInKey);
        }
        let v = value.unwrap_or(ParamValue::Bool(true));
        self.holder
            .set(key, v)
            .map_err(|_| ParamManagerError::UnknownKey(key.to_string()))?;
        Ok(self)
    }

    /// Reset every parameter to its default and drop all dynamic
    /// overrides.
    pub fn clear_all(&mut self) {
        self.holder = self.default.clone();
        self.dyn_param.clear();
    }

    /// Reset a single key + drop its dynamic overrides.
    pub fn clear(&mut self, key: &str) -> Result<&mut Self, ParamManagerError> {
        let default_val = self
            .default
            .get(key)
            .map_err(|_| ParamManagerError::UnknownKey(key.to_string()))?
            .clone();
        self.holder
            .set(key, default_val)
            .map_err(|_| ParamManagerError::UnknownKey(key.to_string()))?;
        self.dyn_param.remove(key);
        Ok(self)
    }

    /// Read a current parameter value.
    pub fn get(&self, key: &str) -> Result<&ParamValue, ParamManagerError> {
        self.holder
            .get(key)
            .map_err(|_| ParamManagerError::UnknownKey(key.to_string()))
    }

    /// All parameter (key, value) pairs.
    pub fn items(&self) -> Vec<(&str, &ParamValue)> {
        self.holder.items()
    }

    /// Bulk-replace the dynamic-parameter entries for a single key. The
    /// caller has already serialized the `(time, value)` list (the
    /// frontend Python `ParamManager` owns the dyn/time/change/hold time
    /// cursor and ships the assembled `_dyn_param` dict here), so the
    /// pre-built list already starts at `(0.0, initial)`. This is the
    /// only path the frontend export takes; the Rust side does not host
    /// its own cursor state machine.
    pub fn inject_dyn_entries(
        &mut self,
        key: &str,
        entries: Vec<(f64, ParamValue)>,
    ) -> Result<(), ParamManagerError> {
        if !self.holder.key_list().contains(&key) {
            return Err(ParamManagerError::UnknownKey(key.to_string()));
        }
        self.dyn_param.insert(key.to_string(), entries);
        Ok(())
    }

    /// Render the static-param holder as `param.toml`. `fast_check =
    /// true` forces `frames = 1`.
    pub fn to_toml_string(&self, fast_check: bool) -> String {
        let mut effective = self.holder.clone();
        if fast_check {
            // Best-effort: only writes if the key exists. Matches
            // Python's `self._param.set("frames", 1)`.
            let _ = effective.set("frames", ParamValue::Int(1));
        }
        let mut out = String::new();
        if effective.key_list().is_empty() {
            return out;
        }
        out.push_str("[param]\n");
        for (key, val) in effective.items() {
            let key = key.replace('-', "_");
            match val {
                ParamValue::String(s) => {
                    let _ = writeln!(&mut out, r#"{key} = "{s}""#);
                }
                ParamValue::Bool(true) => {
                    let _ = writeln!(&mut out, "{key} = true");
                }
                ParamValue::Bool(false) => {
                    let _ = writeln!(&mut out, "{key} = false");
                }
                ParamValue::Vec3(v) => {
                    let _ = writeln!(
                        &mut out,
                        "{key} = [{}, {}, {}]",
                        format_float_python(v[0]),
                        format_float_python(v[1]),
                        format_float_python(v[2]),
                    );
                }
                ParamValue::Float(f) => {
                    let _ = writeln!(&mut out, "{key} = {}", format_float_python(*f));
                }
                ParamValue::Int(i) => {
                    let _ = writeln!(&mut out, "{key} = {i}");
                }
            }
        }
        out
    }

    /// Render the dynamic-parameter overrides as `dyn_param.txt`.
    /// Returns an empty string if no overrides are recorded.
    pub fn to_dyn_param_string(&self) -> Result<String, ParamManagerError> {
        if self.dyn_param.is_empty() {
            return Ok(String::new());
        }
        let mut out = String::new();
        for (key, vals) in &self.dyn_param {
            let _ = writeln!(&mut out, "[{key}]");
            for (time, value) in vals {
                match value {
                    ParamValue::Vec3(v) => {
                        let _ = writeln!(
                            &mut out,
                            "{} {} {} {}",
                            format_float_python(*time),
                            format_float_python(v[0]),
                            format_float_python(v[1]),
                            format_float_python(v[2]),
                        );
                    }
                    ParamValue::Float(f) => {
                        let _ = writeln!(
                            &mut out,
                            "{} {}",
                            format_float_python(*time),
                            format_float_python(*f),
                        );
                    }
                    ParamValue::Int(i) => {
                        // Int-typed keys (frames, min-newton-steps, etc.) can
                        // be seeded into the dyn list (the first entry is the
                        // current value), so emit them as floats to match the
                        // toml path and the solver's f64 dyn_param reader. Note
                        // the solver's apply_dyn_param has no arm for int keys,
                        // so a dynamic int override exports but is not honored
                        // at runtime; this arm only unblocks the export.
                        let _ = writeln!(
                            &mut out,
                            "{} {}",
                            format_float_python(*time),
                            format_float_python(*i as f64),
                        );
                    }
                    ParamValue::Bool(b) => {
                        let _ = writeln!(
                            &mut out,
                            "{} {}",
                            format_float_python(*time),
                            if *b {
                                format_float_python(1.0)
                            } else {
                                format_float_python(0.0)
                            },
                        );
                    }
                    other => {
                        return Err(ParamManagerError::UnsupportedDynValue(other.clone()));
                    }
                }
            }
        }
        Ok(out)
    }
}

impl Default for ParamManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Format a float the way Python's `str()` does for the export path:
/// `1.0` prints as `1.0`, `0.001` prints as `0.001`, very small / very
/// large numbers use scientific notation. Rust's default `{}` for
/// `f64` matches Python closely; whole numbers are the only divergence
/// (Rust prints `1`, Python prints `1.0`). Special-case that.
fn format_float_python(f: f64) -> String {
    if f.is_finite() && f.fract() == 0.0 && f.abs() < 1e16 {
        format!("{}.0", f as i64)
    } else {
        format!("{f}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_basic_value() {
        let mut p = ParamManager::new();
        p.set("frames", Some(ParamValue::Int(120))).unwrap();
        assert_eq!(p.get("frames").unwrap(), &ParamValue::Int(120));
    }

    #[test]
    fn set_underscore_rejected() {
        let mut p = ParamManager::new();
        let err = p.set("min_newton_steps", Some(ParamValue::Int(32))).unwrap_err();
        assert!(matches!(err, ParamManagerError::UnderscoreInKey));
    }

    #[test]
    fn set_unknown_key_rejected() {
        let mut p = ParamManager::new();
        let err = p.set("not-a-param", Some(ParamValue::Int(0))).unwrap_err();
        assert!(matches!(err, ParamManagerError::UnknownKey(_)));
    }

    #[test]
    fn set_none_becomes_true() {
        let mut p = ParamManager::new();
        p.set("disable-contact", None).unwrap();
        assert_eq!(p.get("disable-contact").unwrap(), &ParamValue::Bool(true));
    }

    #[test]
    fn clear_resets_one_key() {
        let mut p = ParamManager::new();
        p.set("dt", Some(ParamValue::Float(0.005))).unwrap();
        assert_eq!(p.get("dt").unwrap(), &ParamValue::Float(0.005));
        p.clear("dt").unwrap();
        assert_eq!(p.get("dt").unwrap(), &ParamValue::Float(1e-3));
    }

    #[test]
    fn clear_all_drops_dyn_param_too() {
        let mut p = ParamManager::new();
        p.set("dt", Some(ParamValue::Float(0.005))).unwrap();
        p.inject_dyn_entries(
            "playback",
            vec![(0.0, ParamValue::Float(1.0)), (1.0, ParamValue::Float(0.5))],
        )
        .unwrap();
        assert!(!p.to_dyn_param_string().unwrap().is_empty());

        p.clear_all();
        assert_eq!(p.get("dt").unwrap(), &ParamValue::Float(1e-3));
        assert!(p.to_dyn_param_string().unwrap().is_empty());
    }

    #[test]
    fn dyn_chain_replays_python_example() {
        // The frontend Python ParamManager owns the dyn/time/change/hold
        // cursor; it assembles this list (from the _session_.py docstring
        //   session.param.dyn("playback")
        //                .time(2.99).hold()
        //                .time(3.0).change(0.1)
        // ) and ships it here via inject_dyn_entries. Verify the injected
        // list round-trips through the serialization path.
        let mut p = ParamManager::new();
        p.inject_dyn_entries(
            "playback",
            vec![
                (0.0, ParamValue::Float(1.0)),
                (2.99, ParamValue::Float(1.0)),
                (3.0, ParamValue::Float(0.1)),
            ],
        )
        .unwrap();

        let dyn_str = p.to_dyn_param_string().unwrap();
        assert!(dyn_str.starts_with("[playback]"));
        assert!(dyn_str.contains("0.0 1.0\n"));
        assert!(dyn_str.contains("2.99 1.0\n"));
        assert!(dyn_str.contains("3.0 0.1\n"));
    }

    #[test]
    fn copy_is_independent() {
        let mut a = ParamManager::new();
        a.set("dt", Some(ParamValue::Float(0.005))).unwrap();
        let mut b = a.copy();
        b.set("dt", Some(ParamValue::Float(0.001))).unwrap();
        assert_eq!(a.get("dt").unwrap(), &ParamValue::Float(0.005));
        assert_eq!(b.get("dt").unwrap(), &ParamValue::Float(0.001));
    }

    #[test]
    fn toml_export_uses_underscore_keys() {
        let mut p = ParamManager::new();
        p.set("dt", Some(ParamValue::Float(0.005))).unwrap();
        p.set("frames", Some(ParamValue::Int(60))).unwrap();
        let toml = p.to_toml_string(false);
        assert!(toml.starts_with("[param]\n"));
        // Hyphen → underscore conversion happens on every key.
        assert!(toml.contains("dt = 0.005"));
        assert!(toml.contains("frames = 60"));
        assert!(toml.contains("min_newton_steps"));
    }

    #[test]
    fn toml_export_writes_vec3_for_gravity() {
        let p = ParamManager::new();
        let toml = p.to_toml_string(false);
        assert!(toml.contains("gravity = [0.0, -9.8, 0.0]"), "toml = {toml}");
    }

    #[test]
    fn toml_export_quotes_string_enums() {
        let p = ParamManager::new();
        let toml = p.to_toml_string(false);
        assert!(toml.contains(r#"barrier = "cubic""#));
        assert!(toml.contains(r#"friction_mode = "min""#));
    }

    #[test]
    fn toml_export_fast_check_forces_frames_1() {
        let mut p = ParamManager::new();
        p.set("frames", Some(ParamValue::Int(300))).unwrap();
        let toml = p.to_toml_string(true);
        assert!(toml.contains("frames = 1"), "got: {toml}");
    }

    #[test]
    fn dyn_export_handles_vec3() {
        let mut p = ParamManager::new();
        p.inject_dyn_entries(
            "gravity",
            vec![
                (0.0, ParamValue::Vec3([0.0, -9.8, 0.0])),
                (1.0, ParamValue::Vec3([0.0, 9.8, 0.0])),
            ],
        )
        .unwrap();
        let dyn_str = p.to_dyn_param_string().unwrap();
        assert!(dyn_str.contains("1.0 0.0 9.8 0.0\n"), "got: {dyn_str}");
    }

    #[test]
    fn dyn_export_handles_bool() {
        let mut p = ParamManager::new();
        p.inject_dyn_entries(
            "disable-contact",
            vec![
                (0.0, ParamValue::Bool(false)),
                (1.0, ParamValue::Bool(true)),
            ],
        )
        .unwrap();
        let dyn_str = p.to_dyn_param_string().unwrap();
        // bool serialises as 0.0 or 1.0, same as Python's float(val).
        assert!(dyn_str.contains("1.0 1.0\n"), "got: {dyn_str}");
    }

    #[test]
    fn dyn_export_handles_int() {
        let mut p = ParamManager::new();
        p.inject_dyn_entries(
            "min-newton-steps",
            vec![
                (0.0, ParamValue::Int(0)),
                (1.0, ParamValue::Int(32)),
            ],
        )
        .unwrap();
        let dyn_str = p.to_dyn_param_string().unwrap();
        // int serialises as a float, same as Python's float(val) and the
        // toml path; the solver's dyn_param reader parses scalars as f64.
        assert!(dyn_str.contains("1.0 32.0\n"), "got: {dyn_str}");
    }

    #[test]
    fn dyn_export_empty_when_no_overrides() {
        let p = ParamManager::new();
        assert_eq!(p.to_dyn_param_string().unwrap(), "");
    }

    #[test]
    fn inject_dyn_entries_replaces_list() {
        let mut p = ParamManager::new();
        // Pre-built list as the frontend's _dyn_param dict already is:
        // first entry at t=0.0 is the initial value, second is the
        // user-supplied change.
        let entries = vec![
            (0.0, ParamValue::Float(1.0)),
            (2.0, ParamValue::Float(0.5)),
        ];
        p.inject_dyn_entries("playback", entries).unwrap();
        let dyn_str = p.to_dyn_param_string().unwrap();
        assert!(dyn_str.contains("[playback]"));
        assert!(dyn_str.contains("0.0 1.0\n"));
        assert!(dyn_str.contains("2.0 0.5\n"));
    }

    #[test]
    fn inject_dyn_entries_rejects_unknown_key() {
        let mut p = ParamManager::new();
        let err = p.inject_dyn_entries("nonexistent-key", vec![]).unwrap_err();
        assert!(matches!(err, ParamManagerError::UnknownKey(_)));
    }
}
