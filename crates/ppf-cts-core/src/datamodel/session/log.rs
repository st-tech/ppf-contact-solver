// File: crates/ppf-cts-core/src/datamodel/session/log.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Log-file readers, formatters, and the solver-error analyzer:
//
//   * `read_log_tail`, `read_log_numbers`, `latest_log_value`: the
//     SimpleLog parsers that back `SessionLog.stdout/stderr/numbers`.
//   * `analyze_solver_error`: pattern-based scan of stdout+stderr
//     used by `FixedSession._analyze_solver_error`.
//   * `format_log_summary`, `format_log_average_summary`,
//     `read_log_average_metrics`, `average_summary_from_disk`: the
//     `summary()` / `average_summary()` builders polled per refresh.
//   * `LogStream`, `log_filename_path`, `log_tail_path`: path
//     composition helpers for the per-stream log files.
//   * `read_log_numbers_squashed`, `float_or_int_pair`: int/float
//     squash mirrors of the Python `float_or_int` helper.
//   * `read_lines_with_newlines`, `read_log_tail_joined`,
//     `solver_failed_short_message`, `solver_failed_to_start_message`:
//     the residual readers / formatters from `FixedSession.start`.

use std::path::{Path, PathBuf};

/// Read the file at `path`, return all lines (without trailing
/// newlines), keeping at most `n_lines` from the end. `None` for
/// `n_lines` returns the whole file. `Some(0)` also returns the whole
/// file (Python's `lines[-0:]` is `lines[0:]`, which we mirror).
/// Missing file → empty Vec.
pub fn read_log_tail(path: &Path, n_lines: Option<usize>) -> Vec<String> {
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let lines: Vec<&str> = raw.lines().collect();
    let start = match n_lines {
        Some(0) | None => 0,
        Some(n) if lines.len() > n => lines.len() - n,
        Some(_) => 0,
    };
    lines[start..].iter().map(|s| s.to_string()).collect()
}

/// `(x, y)` numeric pairs from a SimpleLog `<name>.out` file. Each
/// line is `"X Y"` separated by a single space; lines that don't
/// parse are skipped (looser than raising, easier on partial writes).
pub fn read_log_numbers(path: &Path) -> Vec<(f64, f64)> {
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    raw.lines()
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            let x: f64 = parts.next()?.parse().ok()?;
            let y: f64 = parts.next()?.parse().ok()?;
            Some((x, y))
        })
        .collect()
}

/// Latest `(x, y)` pair, or None.
pub fn latest_log_value(path: &Path) -> Option<f64> {
    read_log_numbers(path).last().map(|(_x, y)| *y)
}

// ---------------------------------------------------------------------------
// Solver error analysis. Mirrors
// `FixedSession._analyze_solver_error`. Pure CPU-bound string scan
// over the combined stdout+stderr lines; ported so the Python
// fallback can drop a few KB of pattern table boilerplate.

/// Pattern rules used by `analyze_solver_error`. Order matters:
/// matching is left-to-right and the first hit wins.
const ERROR_PATTERNS: &[(&str, &str)] = &[
    ("cuda: no device found", "No CUDA device found"),
    ("### ccd failed", "Continuous Collision Detection failed"),
    ("### cg failed", "Linear solver failed"),
    ("### intersection detected", "Intersection detected"),
    (
        "error: reduce buffer size is too small",
        "Insufficient GPU memory",
    ),
    ("stack overflow", "BVH traversal stack overflow"),
    ("overflow detected", "Numerical overflow"),
    (
        "e.squarednorm() > sqr(offset)",
        "Contact offset too large: a rod vertex is within offset distance \
         of a triangle face. Reduce contact-offset or increase separation.",
    ),
];

/// Scan combined log + err lines for a known failure mode and return
/// a single user-facing message. Returns `None` when nothing matched.
///
///   * exact-substring match against lowercased+trimmed lines
///   * panic / assertion-failed fallback returns a 7-line context
///     block (3 before + the hit + 3 after, blanks dropped)
pub fn analyze_solver_error(log_lines: &[String], err_lines: &[String]) -> Option<String> {
    let mut all: Vec<&str> = Vec::with_capacity(log_lines.len() + err_lines.len());
    for l in log_lines {
        all.push(l.as_str());
    }
    for l in err_lines {
        all.push(l.as_str());
    }

    for line in &all {
        let lower = line.to_lowercase();
        let lower = lower.trim();
        for (pat, msg) in ERROR_PATTERNS {
            if lower.contains(pat) {
                return Some((*msg).to_string());
            }
        }
    }

    // Panic / assertion-failed fallback with surrounding context.
    for (i, line) in all.iter().enumerate() {
        let lower = line.to_lowercase();
        if line.contains("panicked at") || lower.contains("assertion failed") {
            let start = i.saturating_sub(3);
            let end = (i + 4).min(all.len());
            let ctx: Vec<String> = all[start..end]
                .iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            return Some(ctx.join("\n"));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// SessionLog.summary / average_summary formatters.

use super::format::{convert_integer, convert_time};

/// Build the "latest values" summary dict from already-fetched raw
/// metric values. `None` for any metric leaves it unrepresented in
/// the same way the Python version does (it always emits the five
/// core metrics; `stretch` is added only when `max_sigma > 0`).
///
/// The output is a list of `(key, value)` pairs in insertion order so
/// the PyO3 binding can convert directly to a Python dict.
pub fn format_log_summary(
    time_per_frame_ms: f64,
    time_per_step_ms: f64,
    num_contact: f64,
    newton_steps: f64,
    pcg_iter: f64,
    max_sigma: Option<f64>,
) -> Vec<(String, String)> {
    let mut out = Vec::with_capacity(6);
    out.push(("time-per-frame".to_string(), convert_time(time_per_frame_ms)));
    out.push(("time-per-step".to_string(), convert_time(time_per_step_ms)));
    out.push(("num-contact".to_string(), convert_integer(num_contact)));
    out.push(("newton-steps".to_string(), convert_integer(newton_steps)));
    out.push(("pcg-iter".to_string(), convert_integer(pcg_iter)));
    if let Some(s) = max_sigma {
        if s > 0.0 {
            out.push(("stretch".to_string(), format!("{:.2}%", 100.0 * (s - 1.0))));
        }
    }
    out
}

/// Build the "average over run" summary dict. Each input is `Option`
/// because the underlying `.out` file may not exist (skip the metric).
pub fn format_log_average_summary(
    time_per_frame_ms_avg: Option<f64>,
    time_per_step_ms_avg: Option<f64>,
    num_contact_max: Option<f64>,
    newton_steps_avg: Option<f64>,
    pcg_iter_avg: Option<f64>,
    max_sigma_avg: Option<f64>,
) -> Vec<(String, String)> {
    let mut out = Vec::new();
    if let Some(v) = time_per_frame_ms_avg {
        out.push(("time-per-frame".to_string(), convert_time(v)));
    }
    if let Some(v) = time_per_step_ms_avg {
        out.push(("time-per-step".to_string(), convert_time(v)));
    }
    if let Some(v) = num_contact_max {
        out.push(("num-contact (max)".to_string(), convert_integer(v.round())));
    }
    if let Some(v) = newton_steps_avg {
        out.push(("newton-steps".to_string(), format!("{v:.2}")));
    }
    if let Some(v) = pcg_iter_avg {
        out.push(("pcg-iter".to_string(), format!("{v:.2}")));
    }
    if let Some(v) = max_sigma_avg {
        if v > 0.0 {
            out.push(("stretch".to_string(), format!("{:.2}%", 100.0 * (v - 1.0))));
        }
    }
    out
}

/// Read the metric values from a `data_dir` containing `<name>.out`
/// SimpleLog files. Returns `(time_per_frame_avg_ms,
/// time_per_step_avg_ms, num_contact_max, newton_steps_avg,
/// pcg_iter_avg, max_sigma_avg)`. Each is `None` when the
/// corresponding `<filename>` did not exist or contained no numeric
/// rows.
fn read_log_average_metrics(
    data_dir: &Path,
    log_filenames: &[(&str, &str)],
) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
    fn lookup<'a>(name: &str, log_filenames: &'a [(&'a str, &'a str)]) -> Option<&'a str> {
        log_filenames
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| *v)
    }
    fn values(data_dir: &Path, name: &str, log_filenames: &[(&str, &str)]) -> Option<Vec<f64>> {
        let fname = lookup(name, log_filenames)?;
        let pairs = read_log_numbers(&data_dir.join(fname));
        if pairs.is_empty() {
            return None;
        }
        Some(pairs.into_iter().map(|(_, y)| y).collect())
    }
    fn avg(data_dir: &Path, name: &str, log_filenames: &[(&str, &str)]) -> Option<f64> {
        let v = values(data_dir, name, log_filenames)?;
        if v.is_empty() {
            return None;
        }
        Some(v.iter().sum::<f64>() / (v.len() as f64))
    }
    fn maximum(data_dir: &Path, name: &str, log_filenames: &[(&str, &str)]) -> Option<f64> {
        let v = values(data_dir, name, log_filenames)?;
        v.into_iter().reduce(f64::max)
    }
    (
        avg(data_dir, "time-per-frame", log_filenames),
        avg(data_dir, "time-per-step", log_filenames),
        maximum(data_dir, "num-contact", log_filenames),
        avg(data_dir, "newton-steps", log_filenames),
        avg(data_dir, "pcg-iter", log_filenames),
        avg(data_dir, "max-sigma", log_filenames),
    )
}

/// All-in-one wrapper: read the on-disk `data_dir` and produce the
/// formatted average-summary dict. Returns an empty list when the
/// directory does not exist.
pub fn average_summary_from_disk(
    data_dir: &Path,
    log_filenames: &[(&str, &str)],
) -> Vec<(String, String)> {
    if !data_dir.is_dir() {
        return Vec::new();
    }
    let (tpf, tps, nc, ns, pi, ms) = read_log_average_metrics(data_dir, log_filenames);
    format_log_average_summary(tpf, tps, nc, ns, pi, ms)
}

/// Squash a (x, y) row into the most natural numeric type, preserving
/// the integer-or-float discrimination Python's `float_or_int` makes.
/// Returns `(x, y, x_is_int, y_is_int)`; the binding can map the
/// flags onto `int` vs `float` Python objects.
pub fn float_or_int_pair(x: f64, y: f64) -> (f64, f64, bool, bool) {
    let xi = x.is_finite() && x.fract() == 0.0;
    let yi = y.is_finite() && y.fract() == 0.0;
    (x, y, xi, yi)
}

/// Compose `<info_path>/output/data/<filename>` given a `name ->
/// filename` lookup. Returns `None` when `name` is not in the map
/// (Python returns `None`).
pub fn log_filename_path(
    info_path: &Path,
    name: &str,
    log_filenames: &[(&str, &str)],
) -> Option<PathBuf> {
    let filename = log_filenames
        .iter()
        .find(|(k, _)| *k == name)
        .map(|(_, v)| *v)?;
    Some(info_path.join("output").join("data").join(filename))
}

#[derive(Debug, Clone, Copy)]
pub enum LogStream {
    Stdout,
    Stderr,
}

/// Pick `stdout.log` or `error.log` under a session directory.
pub fn log_tail_path(info_path: &Path, stream: LogStream) -> PathBuf {
    let leaf = match stream {
        LogStream::Stdout => "stdout.log",
        LogStream::Stderr => "error.log",
    };
    info_path.join(leaf)
}

/// Read SimpleLog `(x, y)` pairs and squash to (x, y, x_is_int,
/// y_is_int) so the PyO3 wrapper can synthesize `int` vs `float`
/// objects per cell.
#[cfg(test)]
pub(super) fn read_log_numbers_squashed(path: &Path) -> Vec<(f64, f64, bool, bool)> {
    read_log_numbers(path)
        .into_iter()
        .map(|(x, y)| {
            let xi = x.is_finite() && x.fract() == 0.0;
            let yi = y.is_finite() && y.fract() == 0.0;
            (x, y, xi, yi)
        })
        .collect()
}

/// Read a log file as Vec<String> of lines that preserve their
/// trailing newlines (Python's `f.readlines()` semantics). Missing
/// file returns an empty Vec.
pub fn read_lines_with_newlines(path: &Path) -> Vec<String> {
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    let mut buf = String::new();
    for ch in raw.chars() {
        buf.push(ch);
        if ch == '\n' {
            out.push(std::mem::take(&mut buf));
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

/// Tail of an existing log file as a single string (Python's
/// `"".join(lines[-n:]).strip()`). Used by `FixedSession.stream`'s
/// per-poll widget refresh. Empty string when the file is missing.
pub fn read_log_tail_joined(path: &Path, n_lines: usize) -> String {
    let lines = read_log_tail(path, Some(n_lines));
    let joined = lines.join("\n");
    joined.trim().to_string()
}

/// Format the inline `f"Solver failed: {''.join(err_lines[:5])}"`
/// fallback raised when no specific error pattern matched but stderr
/// is non-empty.
pub fn solver_failed_short_message(err_lines: &[String]) -> String {
    let head: String = err_lines.iter().take(5).cloned().collect::<Vec<_>>().join("");
    format!("Solver failed: {head}")
}

/// Format the panicked-at-startup `f"Solver failed to start
/// (rc={rc})"` message.
pub fn solver_failed_to_start_message(rc: Option<i32>) -> String {
    match rc {
        Some(c) => format!("Solver failed to start (rc={c})"),
        None => "Solver failed to start (rc=None)".to_string(),
    }
}
