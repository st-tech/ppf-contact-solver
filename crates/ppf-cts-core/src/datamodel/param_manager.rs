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
//   * Dynamic-parameter overrides keyed by time:
//       `dyn(key)` selects a key,
//       `time(t)` advances the cursor (strictly increasing),
//       `change(value)` / `hold()` records (t, value) into the
//       per-key list.
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
    #[error("Key is not set")]
    NoKeySelected,
    #[error("I/O failure: {0}")]
    Io(#[from] std::io::Error),
    #[error("Time must be increasing")]
    NonMonotonicTime,
    #[error("Value must be float, bool, or list. {0:?} is given.")]
    UnsupportedDynValue(ParamValue),
}

#[derive(Debug, Clone)]
pub struct ParamManager {
    holder: ParamHolder,
    default: ParamHolder,
    dyn_param: BTreeMap<String, Vec<(f64, ParamValue)>>,
    current_key: Option<String>,
    current_time: f64,
}

impl ParamManager {
    /// Construct with the default `app_param()` defaults.
    pub fn new() -> Self {
        let holder = app_param();
        Self {
            default: holder.clone(),
            holder,
            dyn_param: BTreeMap::new(),
            current_key: None,
            current_time: 0.0,
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

    /// Select the dynamic-parameter cursor key and reset the time
    /// cursor to 0.
    pub fn dyn_select(&mut self, key: &str) -> Result<&mut Self, ParamManagerError> {
        if !self.holder.key_list().contains(&key) {
            return Err(ParamManagerError::UnknownKey(key.to_string()));
        }
        self.current_time = 0.0;
        self.current_key = Some(key.to_string());
        Ok(self)
    }

    /// Advance the dynamic-parameter time cursor. Strictly monotonic;
    /// equal times are rejected.
    pub fn time(&mut self, time: f64) -> Result<&mut Self, ParamManagerError> {
        if time <= self.current_time {
            return Err(ParamManagerError::NonMonotonicTime);
        }
        self.current_time = time;
        Ok(self)
    }

    /// Record a `(time, value)` override at the cursor. If this is the
    /// first entry for the key, the initial-default
    /// `(0.0, current_value)` is seeded first so the entry list is
    /// `[(0.0, initial), (t, value)]`.
    pub fn change(&mut self, value: ParamValue) -> Result<&mut Self, ParamManagerError> {
        let key = self
            .current_key
            .clone()
            .ok_or(ParamManagerError::NoKeySelected)?;
        let entry = self.dyn_param.entry(key.clone()).or_default();
        if entry.is_empty() {
            let initial = self
                .holder
                .get(&key)
                .map_err(|_| ParamManagerError::UnknownKey(key.clone()))?
                .clone();
            entry.push((0.0, initial));
        }
        entry.push((self.current_time, value));
        Ok(self)
    }

    /// Snapshot the current value at the cursor. Identical to
    /// `change(current_value)` but spelled separately so the user's
    /// intent in `time(t).hold()` reads naturally.
    pub fn hold(&mut self) -> Result<&mut Self, ParamManagerError> {
        let key = self
            .current_key
            .clone()
            .ok_or(ParamManagerError::NoKeySelected)?;
        let value_to_hold: ParamValue = if let Some(entry) = self.dyn_param.get(&key) {
            entry.last().map(|(_, v)| v.clone()).unwrap_or_else(|| {
                self.holder
                    .get(&key).cloned()
                    .unwrap_or(ParamValue::Bool(false))
            })
        } else {
            self.holder
                .get(&key)
                .map_err(|_| ParamManagerError::UnknownKey(key.clone()))?
                .clone()
        };
        self.change(value_to_hold)
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

    /// Dynamic-param entries by key.
    pub fn dyn_param(&self) -> &BTreeMap<String, Vec<(f64, ParamValue)>> {
        &self.dyn_param
    }

    /// Bulk-replace the dynamic-parameter entries for a single key. The
    /// caller has already serialized the `(time, value)` list, so we
    /// skip the public `dyn_select + time + change` chain (which
    /// rejects `t == 0.0` for the first entry because it would equal
    /// the freshly-reset cursor). Used by the PyO3 binding when the
    /// frontend handed us a pre-built `_dyn_param` dict.
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
        p.dyn_select("playback").unwrap();
        p.time(1.0).unwrap();
        p.change(ParamValue::Float(0.5)).unwrap();
        assert!(!p.dyn_param().is_empty());

        p.clear_all();
        assert_eq!(p.get("dt").unwrap(), &ParamValue::Float(1e-3));
        assert!(p.dyn_param().is_empty());
    }

    #[test]
    fn dyn_change_seeds_initial_entry() {
        let mut p = ParamManager::new();
        p.dyn_select("playback").unwrap();
        p.time(2.0).unwrap();
        p.change(ParamValue::Float(0.5)).unwrap();
        let entries = &p.dyn_param()["playback"];
        // [(0.0, initial), (2.0, 0.5)]
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, 0.0);
        assert_eq!(entries[0].1, ParamValue::Float(1.0)); // playback default
        assert_eq!(entries[1], (2.0, ParamValue::Float(0.5)));
    }

    #[test]
    fn time_must_strictly_increase() {
        let mut p = ParamManager::new();
        p.dyn_select("dt").unwrap();
        p.time(1.0).unwrap();
        let err = p.time(1.0).unwrap_err();
        assert!(matches!(err, ParamManagerError::NonMonotonicTime));
        let err = p.time(0.5).unwrap_err();
        assert!(matches!(err, ParamManagerError::NonMonotonicTime));
    }

    #[test]
    fn dyn_chain_replays_python_example() {
        // From _session_.py docstring:
        //   session.param.dyn("playback")
        //                .time(2.99).hold()
        //                .time(3.0).change(0.1)
        let mut p = ParamManager::new();
        p.dyn_select("playback").unwrap();
        p.time(2.99).unwrap();
        p.hold().unwrap();
        p.time(3.0).unwrap();
        p.change(ParamValue::Float(0.1)).unwrap();

        let entries = &p.dyn_param()["playback"];
        // [(0.0, 1.0), (2.99, 1.0), (3.0, 0.1)]
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0], (0.0, ParamValue::Float(1.0)));
        assert_eq!(entries[1], (2.99, ParamValue::Float(1.0)));
        assert_eq!(entries[2], (3.0, ParamValue::Float(0.1)));
    }

    #[test]
    fn change_without_dyn_select_errors() {
        let mut p = ParamManager::new();
        let err = p.change(ParamValue::Float(0.5)).unwrap_err();
        assert!(matches!(err, ParamManagerError::NoKeySelected));
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
    fn dyn_export_produces_expected_format() {
        let mut p = ParamManager::new();
        p.dyn_select("playback").unwrap();
        p.time(2.99).unwrap();
        p.hold().unwrap();
        p.time(3.0).unwrap();
        p.change(ParamValue::Float(0.1)).unwrap();

        let dyn_str = p.to_dyn_param_string().unwrap();
        // Expected: [playback]\n0.0 1.0\n2.99 1.0\n3.0 0.1\n
        assert!(dyn_str.starts_with("[playback]"));
        assert!(dyn_str.contains("0.0 1.0\n"));
        assert!(dyn_str.contains("2.99 1.0\n"));
        assert!(dyn_str.contains("3.0 0.1\n"));
    }

    #[test]
    fn dyn_export_handles_vec3() {
        let mut p = ParamManager::new();
        p.dyn_select("gravity").unwrap();
        p.time(1.0).unwrap();
        p.change(ParamValue::Vec3([0.0, 9.8, 0.0])).unwrap();
        let dyn_str = p.to_dyn_param_string().unwrap();
        assert!(dyn_str.contains("1.0 0.0 9.8 0.0\n"), "got: {dyn_str}");
    }

    #[test]
    fn dyn_export_handles_bool() {
        let mut p = ParamManager::new();
        p.dyn_select("disable-contact").unwrap();
        p.time(1.0).unwrap();
        p.change(ParamValue::Bool(true)).unwrap();
        let dyn_str = p.to_dyn_param_string().unwrap();
        // bool serialises as 0.0 or 1.0, same as Python's float(val).
        assert!(dyn_str.contains("1.0 1.0\n"), "got: {dyn_str}");
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
