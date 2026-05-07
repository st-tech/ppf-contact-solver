// File: crates/ppf-cts-core/src/datamodel/session/format.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// User-facing formatters used by `summary()` / `display_log()` and the
// session-creation error message. Pure string assembly; no I/O.

/// Format a millisecond duration the way the Python helper does:
///   < 1s  -> "{int}ms"
///   < 60s -> "{:.2f}s"
///   else  -> "{:.2f}m"
pub fn convert_time(ms: f64) -> String {
    if ms < 1_000.0 {
        format!("{}ms", ms as i64)
    } else if ms < 60_000.0 {
        format!("{:.2}s", ms / 1_000.0)
    } else {
        format!("{:.2}m", ms / 60_000.0)
    }
}

/// Format an integer-valued count with a thousand-suffix.
pub fn convert_integer(n: f64) -> String {
    if n < 1_000.0 {
        format!("{}", n as i64)
    } else if n < 1_000_000.0 {
        format!("{:.2}k", n / 1_000.0)
    } else if n < 1_000_000_000.0 {
        format!("{:.2}M", n / 1_000_000.0)
    } else {
        format!("{:.2}B", n / 1_000_000_000.0)
    }
}

/// `convert_time` with a `None -> "N/A"` guard.
pub fn convert_time_optional(ms: Option<f64>) -> String {
    match ms {
        Some(v) => convert_time(v),
        None => "N/A".to_string(),
    }
}

/// `convert_integer` with two extra guards from the public Python
/// helper:
///   * `None -> "N/A"`
///   * Values `< 1000` go through the raw `str()` path so a stray
///     float like `42.5` renders as `"42.5"` rather than truncating
///     to `"42"`.
pub fn convert_integer_optional(n: Option<f64>) -> String {
    let Some(v) = n else {
        return "N/A".to_string();
    };
    if v >= 1_000.0 {
        convert_integer(v)
    } else if v.is_finite() && v.fract() == 0.0 {
        format!("{}", v as i64)
    } else {
        // Python's `str(42.5) == "42.5"`, which matches Rust `{}` for
        // most finite floats. Whole-number paths handled above.
        format!("{v}")
    }
}

/// Format the violation-summary string raised when
/// `scene.has_violations`: `"Cannot create session: <msg1>; <msg2>. "`.
pub fn session_violations_message(messages: &[String]) -> String {
    format!("Cannot create session: {}. ", messages.join("; "))
}

/// Strip trailing `\n` from each line, the way `display_log` does
/// before feeding the Jupyter widget body.
pub fn rstrip_newlines(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .map(|s| s.trim_end_matches('\n').to_string())
        .collect()
}
