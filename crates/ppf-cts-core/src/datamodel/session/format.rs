// File: crates/ppf-cts-core/src/datamodel/session/format.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// User-facing formatters used by `summary()` / `display_log()` and the
// session-creation error message. Pure string assembly; no I/O.

/// Whole-number test mirroring Python's `float_or_int` discrimination:
/// a finite value with no fractional part. The single source for the
/// `int`-vs-`float` predicate shared by the formatters and
/// `float_or_int_pair`.
pub(super) fn is_whole(v: f64) -> bool {
    v.is_finite() && v.fract() == 0.0
}

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
    } else if is_whole(v) {
        format!("{}", v as i64)
    } else {
        // Python's `str(42.5) == "42.5"`, which matches Rust `{}` for
        // most finite floats. Whole-number paths handled above.
        format!("{v}")
    }
}

/// Format the `stretch` summary value from a max-sigma reading as a
/// percentage `100 * (sigma - 1)`. Returns `None` when `max_sigma <= 0`
/// so the row is omitted, matching the Python version's behavior. The
/// single source of the stretch convention (precision, cutoff, and
/// formula) shared by the live and average summary builders.
pub fn format_stretch(max_sigma: f64) -> Option<String> {
    (max_sigma > 0.0).then(|| format!("{:.2}%", 100.0 * (max_sigma - 1.0)))
}

/// Format an averaged count channel (num-contact, pcg-iter) for the
/// summary panels. A value at or above 1000 gets a thousands-suffix
/// (`12.44k`, `1.50M`, `2.00B`) via [`convert_integer`] so a large count
/// reads as `12.44k` instead of a long digit run. Smaller values keep a
/// fixed two-decimal display (`40.00`, `4.33`) so a fractional per-step
/// average reads cleanly instead of truncating to a bare `4`.
pub fn convert_average_count(v: f64) -> String {
    if v >= 1_000.0 {
        convert_integer(v)
    } else {
        format!("{v:.2}")
    }
}

/// [`convert_average_count`] with a `None -> "N/A"` guard, for the live
/// summary's optional channel readings.
pub fn convert_average_count_optional(v: Option<f64>) -> String {
    match v {
        Some(v) => convert_average_count(v),
        None => "N/A".to_string(),
    }
}

/// Format a ratio in `[0, 1]` (e.g. the dynamic-Hessian memory-usage
/// fraction or the advanced fractional step size) as a percentage
/// `100 * v` with two decimals, matching the `stretch` row's precision.
pub fn convert_ratio(v: f64) -> String {
    format!("{:.2}%", 100.0 * v)
}

/// `convert_ratio` with a `None -> "N/A"` guard, for the live summary's
/// optional channel readings.
pub fn convert_ratio_optional(v: Option<f64>) -> String {
    match v {
        Some(v) => convert_ratio(v),
        None => "N/A".to_string(),
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
