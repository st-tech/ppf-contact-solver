// File: crates/ppf-cts-solver/src/driver/log.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The per-step diagnostic channel, and the stream a human reads while a solve
//! is running.
//!
//! This module does two jobs at once, and they are separable only on paper.
//! It RECORDS: the solver's indicator streams are plain two-column text files,
//! one per scope and channel, appended to as the run goes:
//! `<dir>/<scope>.<channel>.out` holding `{sim_time} {value}` per line.
//! `advance.iter.out`, `advance.SL_toi.out` and `advance.max_sigma.out` are the
//! three a linear-solve diagnosis is read off: a collapsing `SL_toi` beside an
//! exploding `max_sigma` is what identifies a lost PCG residual denominator, and
//! neither is visible anywhere else. And it PRINTS: the same values go to stdout
//! as they are recorded, which is the running commentary the frontend tails into
//! `>>> Log:` and the addon shows in its log panel.
//!
//! WHY THE TWO JOBS ARE ONE MODULE. The C++ `SimpleLog`
//! (`src/kernels/simplelog/SimpleLog.cpp`) does both from one call: `mark`
//! appends to a per-channel vector AND prints `* <name>: <value>`, `pop` writes
//! the elapsed milliseconds AND prints `> <name>...<elapsed>`. Splitting them
//! here would give a caller two ways to disagree about a number, and it would
//! give the docstring harvester nothing to key on (below). One call, one
//! number, both destinations.
//!
//! THE FILE FORMAT IS C's, NOT RUST's, and the difference is not cosmetic. A
//! stream file carries `"%f %d\n"` for a value that is a whole number and
//! `"%f %e\n"` otherwise, which is what every reader of one parses. Rust's
//! `{:e}` writes `1.5e0` where
//! C writes `1.500000e+00`, so a stream written the Rust way is a different file
//! format wearing the same name, and every reader of it would have to learn
//! which backend produced it. The printed forms have the same problem one
//! decimal place shallower: `%.3e` and `%.2e`, both rebuilt here.
//!
//! THE PRINTED LINES CARRY NO TIMESTAMP, AND THAT IS A CONTRACT RATHER THAN AN
//! OMISSION. Two writers share one stdout stream: `message_text` below is a
//! bare `println!` with no prefix, while every Rust `info!` goes through log4rs
//! under the pattern `"[{d(%Y-%m-%d %H:%M:%S)}] {m}{n}"`. A transcript shows
//! the two in consecutive lines, `===== advance: 28 msec =====` bare and then
//! `[2026-09-08 07:26:36] GPU SM Clock: 2520 MHz` prefixed. A log line is a HOST
//! CONTRACT, read by the frontend's tail, by the addon and by a person, so the
//! split is kept rather than resolved. The dividing line is the WRITER, not the
//! content: a line belonging to the solver's own transcript goes through
//! `message!` and is bare; a line the driver emits on its own account stays on
//! `::log::` and keeps its prefix.
//!
//! NO EXPLICIT FLUSH IS NEEDED, which is worth saying because `fflush(stdout)`
//! has no counterpart below. Rust's `Stdout` is a `LineWriter` unconditionally,
//! so each `println!` flushes at its newline. log4rs's `ConsoleAppender` writes
//! through `io::stdout()` when stdout is not a tty, which is the shipped case
//! (the solver's stdout is redirected into `session/stdout.log`), so it shares
//! this handle and this lock and the interleaving is exact. When stdout IS a tty
//! it writes to fd 1 directly, but it flushes per record and `println!` flushes
//! per line, so neither writer ever leaves a partial line outstanding across the
//! other.
//!
//! WHY A CHANNEL IS SPELLED `log::mark(` AND NOTHING ELSE.
//! `ppf_cts_core::parsers::get_logging_docstrings` harvests the channel registry
//! by SCANNING THIS CRATE'S SOURCE, not by listing the output directory: it
//! keys a channel on the literal token `log::mark(` appearing on the call site's
//! own line, with a `// Name:` docstring block above it. A separate `phase()`
//! entry point would therefore write its stream and never appear in
//! `session.get.log.names()`. That is why a timed phase is `Marked::Elapsed`
//! passed to the same `mark`, and why the value's TYPE rather than the function
//! name selects the printed shape.
//!
//! THE LOCK IS RELEASED BEFORE ANY OUTPUT, AND THAT IS A REQUIREMENT RATHER THAN
//! A STYLE NOTE. `STATE` is a `std::sync::Mutex`, which is not reentrant, and
//! `message_text` reads `depth` from it. Any emitter that held the guard across
//! its own print would deadlock the solver on the first channel it wrote. Every
//! entry point below therefore locks once, copies out what it needs, drops the
//! guard, and only then touches a file or stdout. Reviewing a change here: no
//! `STATE.lock()` result may be alive across a `println!` or a file write.

// `mark` is called from `step.rs` and `set_path` from `mod.rs`. The allow covers
// `path` and `time`, the two accessors this module exports that no caller reads,
// and it is scoped to this module so a genuinely unused item elsewhere still
// surfaces.
#![allow(dead_code)]

use std::borrow::Cow;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// The directory the streams are written into, the clock they are stamped with,
/// and how deeply nested the section printing them is. All three are
/// process-global for the same reason `SimpleLog`'s are: they describe one run
/// and one stdout stream.
///
/// `depth` is deliberately NOT thread-local. It describes a position in ONE
/// output stream, not a position in a thread, so two threads emitting at once
/// with private depths would interleave two different indents into one stream,
/// which is less legible rather than more. In practice the question does not
/// arise: `initialize()`, `advance()` and `step::advance` are the only emitters
/// and all of them are the step thread, while the rayon workers inside a
/// dispatch do not log. If that ever changes, the mutex serializes the read and
/// the worst outcome is an indent taken from a concurrent scope, which is
/// cosmetic and cannot corrupt a stream file.
static STATE: Mutex<LogState> = Mutex::new(LogState {
    directory: None,
    time: 0.0,
    depth: 0,
});

struct LogState {
    directory: Option<PathBuf>,
    time: f64,
    depth: usize,
}

/// What a channel records, and how it prints.
///
/// THE TYPE PICKS THE SHAPE. It has to, because the docstring harvester keys a
/// channel on the literal token `log::mark(` on the call site's own line (see
/// the module comment), so a phase timer that wanted its own entry point would
/// be invisible to the registry. One entry point, three value kinds, and the
/// kind is chosen by what the caller hands over rather than by a flag it has to
/// remember to set.
pub enum Marked {
    /// Recorded, and printed as `* <channel>: <value>`.
    Number(f64),
    /// A phase timer. PRINTED as whole milliseconds, in the shape
    /// `> <channel>...<elapsed>`, and RECORDED at full resolution.
    ///
    /// THE PRINTED LINE AND THE RECORDED VALUE ANSWER DIFFERENT QUESTIONS, so
    /// they take different roundings. The STREAM is what a reader watches go
    /// by, and its timings are whole milliseconds. The `.out` FILE is what the
    /// addon's Matrix Assembly, PCG Solve and Line Search cells read and what a
    /// profiling session compares across builds, and truncating there throws
    /// away everything under a millisecond: measured on a small fixture, every
    /// timer row became literally `0`. Recording the fraction costs nothing and
    /// leaves the printed log unchanged.
    Elapsed { shown_millis: u64, recorded_millis: f64 },
    /// Recorded, never printed. A channel that feeds a later analysis rather
    /// than a reader watching the run takes this, so instrumentation can keep
    /// its stream without adding a line to the transcript.
    Quiet(f64),
}

impl From<f64> for Marked {
    fn from(value: f64) -> Self {
        Marked::Number(value)
    }
}

impl From<Duration> for Marked {
    /// Truncating to whole milliseconds for the printed line, and keeping the
    /// fraction for the file.
    fn from(elapsed: Duration) -> Self {
        Marked::Elapsed {
            shown_millis: elapsed.as_millis() as u64,
            recorded_millis: elapsed.as_secs_f64() * 1000.0,
        }
    }
}

impl Marked {
    /// A value recorded in the stream file and shown to nobody.
    pub fn quiet(value: f64) -> Self {
        Marked::Quiet(value)
    }

    /// The number that reaches `<scope>.<channel>.out`.
    fn recorded(&self) -> f64 {
        match *self {
            Marked::Number(value) | Marked::Quiet(value) => value,
            Marked::Elapsed { recorded_millis, .. } => recorded_millis,
        }
    }

    /// The line that reaches stdout, or `None` for a channel that only records.
    fn printed(&self, channel: &str) -> Option<String> {
        match *self {
            // The SAME predicate `format_line` uses, so the file and the
            // printed line can never disagree about which form a value takes.
            // Stating the rule twice is what would let the two drift.
            Marked::Number(value) => Some(if value.is_finite() && value.fract() == 0.0 {
                format!("* {channel}: {}", value as i32)
            } else {
                format!("* {channel}: {}", c_exponential(value, 3))
            }),
            Marked::Elapsed { shown_millis, .. } => {
                Some(format!("> {channel}...{}", tstr(shown_millis)))
            }
            Marked::Quiet(_) => None,
        }
    }
}

/// Print one line at the current section depth, unprefixed.
///
/// The only place this module writes to stdout. It is public so the `message!`
/// macro below can expand to it from another module; the emitters in this file,
/// which already hold a built string, call it directly.
pub fn message_text(text: &str) {
    let depth = current_depth();
    let mut line = String::with_capacity(text.len() + 3 * depth);
    // THE OUTERMOST SECTION IS FLUSH LEFT, so the indent is (depth - 1) * 3.
    //
    // Nothing in this solver nests a section inside another, so the loop never
    // runs today and every line comes out flush left. The counter is still
    // maintained rather than dropped: it is what a nested scope would need, and
    // a reader should not have to work out whether an indent was lost or was
    // never produced.
    for _ in 1..depth {
        line.push_str("   ");
    }
    line.push_str(text);
    println!("{line}");
}

/// Print one unprefixed line at the current section depth.
///
/// A macro rather than a function taking `&str` so a call site reads
/// `log::message!("------ newton step {step} ------")` instead of wrapping each
/// of its call sites in `&format!(...)`.
macro_rules! message {
    ($($arg:tt)*) => {
        $crate::driver::log::message_text(&format!($($arg)*))
    };
}
pub(crate) use message;

/// Record the directory the streams go in, creating it if needed.
///
/// Returns whether the directory is usable. A path that cannot be created is
/// reported and then dropped, which is the one place in this backend where
/// carrying on is right: diagnostics are telemetry, and a run that can solve
/// the scene must not be stopped because it cannot write a log beside it.
///
/// On success this prints the `* data_directory_path path = ...` line the
/// transcript carries. On failure it does not: the warning names the directory
/// and the reason, which is strictly more than that line would say about a
/// directory that is not there.
pub fn set_path(path: &Path) -> bool {
    if let Err(error) = std::fs::create_dir_all(path) {
        log::warn!(
            "solver driver: cannot create the log directory {}: {error}. The per-step indicator \
             streams will not be written for this run.",
            path.display()
        );
        return false;
    }
    if let Ok(mut state) = STATE.lock() {
        state.directory = Some(path.to_path_buf());
    }
    message!("* data_directory_path path = {}", path.display());
    true
}

/// The directory the streams go in, or `None` when no path was set.
pub fn path() -> Option<PathBuf> {
    STATE.lock().ok().and_then(|state| state.directory.clone())
}

/// Stamp the simulation time every subsequent line carries, and announce it.
///
/// The line is `* time = %g`, C's shortest-of-two-forms conversion, which is
/// neither of the shapes the rest of this module writes.
pub fn set_time(time: f64) {
    if let Ok(mut state) = STATE.lock() {
        state.time = time;
    }
    message!("* time = {}", c_general(time));
}

pub fn time() -> f64 {
    STATE.lock().map(|state| state.time).unwrap_or(0.0)
}

/// Record one value under `<scope>.<channel>` and, unless it is `Quiet`, print
/// it.
///
/// The file half appends one `{sim_time} {value}` line to
/// `<dir>/<scope>.<channel>.out`, and is a no-op when no path was set, which is
/// what a run with no output directory is: there is nowhere for the stream to
/// go, and inventing one would put files somewhere the caller did not ask for.
/// The printed half needs no directory and happens either way.
///
/// **THE PHASE TIMERS ARE `Marked::Elapsed`**, `linsolve`, `matrix_assembly`,
/// `asm_contact`, `line_search` and `check_intersection` among them, taken in
/// `step.rs` around the phase each one names. They are what locates a run's
/// time: measured on a 134 s `trapped` run they account for about 59 s, where a
/// driver recording only `toi`, `newton_steps` and `playback` leaves 45 s of a
/// 57 s gap with no row to land in.
///
/// Spaces in a channel name become underscores, as
/// `ppf_cts_core::parsers::extract_pair` does when it harvests the same name out
/// of this source. No channel in the tree carries a space today; keeping the
/// rule means a future one cannot name a file and a registry entry differently.
pub fn mark(scope: &str, channel: &str, value: impl Into<Marked>) {
    let marked = value.into();
    let channel = if channel.contains(' ') {
        Cow::Owned(channel.replace(' ', "_"))
    } else {
        Cow::Borrowed(channel)
    };
    if let Some((directory, time)) = sink() {
        let path = directory.join(format!("{scope}.{channel}.out"));
        append(&path, &format_line(time, marked.recorded()));
    }
    if let Some(text) = marked.printed(&channel) {
        message_text(&text);
    }
}

/// One `====== name ======` / `===== name: elapsed =====` bracket, closed by
/// the drop.
///
/// The only thing that moves the indent depth: `mark` never touches it. Its
/// drop also writes `<dir>/<name>.out`, the section's own total, which
/// `ppf-cts-server`'s response summary pins as the addon's `time-per-step`
/// cell.
pub struct Section {
    name: String,
    start: Instant,
}

impl Section {
    pub fn new(name: &str) -> Self {
        let name = name.replace(' ', "_");
        if let Ok(mut state) = STATE.lock() {
            state.depth += 1;
        }
        // The header prints AFTER the increment here, and the footer prints
        // BEFORE the decrement in the drop below, so the two sit at the same
        // indent.
        message_text(&format!("====== {name} ======"));
        Self {
            name,
            start: Instant::now(),
        }
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        let millis = self.start.elapsed().as_millis() as u64;
        message_text(&format!("===== {}: {} =====", self.name, tstr(millis)));
        record_section(&self.name, millis);
        if let Ok(mut state) = STATE.lock() {
            // Saturating because a poisoned mutex can leave the matching
            // increment unrecorded, and an underflow in a diagnostic would
            // panic a run that is otherwise healthy.
            state.depth = state.depth.saturating_sub(1);
        }
    }
}

/// Append the section's own total to `<dir>/<name>.out`.
///
/// A DIFFERENT FILE SHAPE from a channel's: `<name>.out`, with no channel
/// segment. `ppf-cts-server`'s summary reads `advance.out` as `time-per-step`,
/// so this is the file behind that cell.
fn record_section(name: &str, millis: u64) {
    let Some((directory, time)) = sink() else {
        return;
    };
    let path = directory.join(format!("{name}.out"));
    append(&path, &format_line(time, millis as f64));
}

/// The section depth, copied out and the guard dropped before the caller
/// prints.
fn current_depth() -> usize {
    STATE.lock().map(|state| state.depth).unwrap_or(0)
}

/// The directory and the clock a stream line needs, copied out and the guard
/// dropped before the caller opens anything.
///
/// `None` means no path was set, which is the case a run with no output
/// directory is in.
fn sink() -> Option<(PathBuf, f64)> {
    let state = STATE.lock().ok()?;
    let directory = state.directory.clone()?;
    Some((directory, state.time))
}

/// Append one already-formatted line, reporting rather than failing.
fn append(path: &Path, line: &str) {
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        Ok(mut file) => {
            if let Err(error) = file.write_all(line.as_bytes()) {
                log::warn!("solver driver: cannot append to {}: {error}", path.display());
            }
        }
        Err(error) => {
            log::warn!("solver driver: cannot open {}: {error}", path.display());
        }
    }
}

/// One stream line, in the C spelling every reader of these files parses.
fn format_line(time: f64, value: f64) -> String {
    let mut line = format!("{time:.6} ");
    // A WHOLE NUMBER TAKES THE INTEGER FORM and everything else takes the
    // exponential form. A non-finite value has no whole part, so it takes the
    // exponential branch
    // and is written rather than silently dropped: a NaN in an indicator stream
    // is the most interesting line in the file.
    if value.is_finite() && value.fract() == 0.0 {
        let _ = write!(line, "{}", value as i32);
    } else {
        line.push_str(&c_exponential(value, 6));
    }
    line.push('\n');
    line
}

/// An elapsed duration in words, the bounds `tstr` in
/// `src/kernels/simplelog/SimpleLog.cpp` prints with.
///
/// The bounds are transcribed rather than tidied. The fourth one is
/// `1000 * 60 * 60 * 60 * 3`, which is 648,000,000 msec, or 7.5 days, and not
/// the 3 days that the pattern of its neighbors suggests it was meant to be. It
/// decides which word a duration is printed with, so correcting it here would
/// change the output of every run between 3 and 7.5 days long. It also fits in
/// a C `int` (the maximum is 2,147,483,647), so that bound is not an
/// overflow.
fn tstr(msec: u64) -> String {
    if msec < 1_000 {
        format!("{msec} msec")
    } else if msec < 180_000 {
        format!("{:.3} sec", msec as f64 / 1_000.0)
    } else if msec < 10_800_000 {
        format!("{:.3} minutes", msec as f64 / 60_000.0)
    } else if msec < 648_000_000 {
        format!("{:.3} hours", msec as f64 / 3_600_000.0)
    } else {
        format!("{:.3} days", msec as f64 / 86_400_000.0)
    }
}

/// `value` in C's `%e` at `precision` mantissa decimals, with a signed exponent
/// of at least two digits.
///
/// Rust's own `{:.6e}` gives `1.500000e0`, so the exponent is rebuilt here.
/// Three precisions are in use and all three go through this one function: 6
/// for the stream files (`%e`), 3 for a printed mark (`%.3e`), and 2 for the
/// handful of message lines that spell a `toi` (`%.2e`).
///
/// `pub(super)` rather than private, because the third of those callers is in
/// `step.rs`: a message line that spells a `toi` is written by the site that
/// knows what the number means, and it must reach C's spelling of `%.2e`
/// through this function rather than through a second copy of the exponent
/// rebuild. Keeping it private would put that copy in the tree.
pub(super) fn c_exponential(value: f64, precision: usize) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-inf"
        } else {
            "inf"
        }
        .to_string();
    }
    let formatted = format!("{value:.precision$e}");
    let (mantissa, exponent) = match formatted.split_once('e') {
        Some(parts) => parts,
        // Unreachable for a finite float, and reported as itself rather than
        // guessed at if the formatter ever changes shape.
        None => return formatted,
    };
    let (sign, digits) = match exponent.strip_prefix('-') {
        Some(rest) => ('-', rest),
        None => ('+', exponent),
    };
    format!("{mantissa}e{sign}{digits:0>2}")
}

/// `value` in C's `%g` at the default precision of 6 significant digits, which
/// is what `set_time` above prints the simulation time with.
///
/// The rule, from the C standard's description of `%g` with precision P = 6:
/// take the decimal exponent X the value has once rounded to P significant
/// digits; use `%e` with precision P - 1 when X < -4 or X >= P, and `%f` with
/// precision P - 1 - X otherwise; then strip trailing zeros from the fractional
/// part, and the decimal point with them if nothing is left after it.
///
/// ROUNDING FIRST IS LOAD-BEARING, and it is the one part a reimplementation
/// gets wrong. 999999.5 rounds to 1000000, whose exponent is 6 rather than 5,
/// so C prints `1e+06` and not `1000000`. Reading the exponent off the rounded
/// `%e` form, which is what happens below, gets that for free.
fn c_general(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-inf"
        } else {
            "inf"
        }
        .to_string();
    }
    /// C's default precision for `%g`.
    const P: i32 = 6;
    let rounded = format!("{value:.precision$e}", precision = (P - 1) as usize);
    let (mantissa, exponent_text) = match rounded.split_once('e') {
        Some(parts) => parts,
        None => return rounded,
    };
    let exponent: i32 = match exponent_text.parse() {
        Ok(exponent) => exponent,
        Err(_) => return rounded,
    };
    if exponent < -4 || exponent >= P {
        let sign = if exponent < 0 { '-' } else { '+' };
        let digits = exponent.abs();
        format!("{}e{sign}{digits:0>2}", strip_trailing_zeros(mantissa))
    } else {
        // P - 1 - X, which is at most 9 here because the branch above already
        // took everything below X = -4.
        let places = (P - 1 - exponent) as usize;
        strip_trailing_zeros(&format!("{value:.places$}")).to_string()
    }
}

/// Drop trailing fractional zeros, and the decimal point if it is then last.
///
/// A no-op on text with no decimal point, so `100000` survives whole rather
/// than losing its zeros.
fn strip_trailing_zeros(text: &str) -> &str {
    if !text.contains('.') {
        return text;
    }
    let trimmed = text.trim_end_matches('0');
    trimmed.strip_suffix('.').unwrap_or(trimmed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_whole_number_takes_the_integer_form() {
        // The integer form, `%f %d`: no exponent on a whole value.
        assert_eq!(format_line(0.5, 12.0), "0.500000 12\n");
    }

    #[test]
    fn a_fractional_value_takes_c_s_exponential_form() {
        // The exponent is what Rust and C disagree about: `{:.6e}` alone would
        // write `9.130000e-1`, which no reader of these files expects.
        assert_eq!(format_line(1.25, 0.913), "1.250000 9.130000e-01\n");
        assert_eq!(format_line(0.0, 1234.5), "0.000000 1.234500e+03\n");
        assert_eq!(format_line(0.0, -0.0001042), "0.000000 -1.042000e-04\n");
    }

    #[test]
    fn a_three_digit_exponent_keeps_all_three() {
        // The rule is a MINIMUM of two digits, not exactly two; zero-padding to
        // a fixed width would truncate a denormal-scale value.
        assert_eq!(format_line(0.0, 1.5e-120), "0.000000 1.500000e-120\n");
    }

    #[test]
    fn a_non_finite_value_is_written_rather_than_dropped() {
        assert_eq!(format_line(2.0, f64::NAN), "2.000000 nan\n");
        assert_eq!(format_line(2.0, f64::INFINITY), "2.000000 inf\n");
        assert_eq!(format_line(2.0, f64::NEG_INFINITY), "2.000000 -inf\n");
    }

    #[test]
    fn each_tstr_branch_takes_over_at_the_reference_s_own_bound() {
        assert_eq!(tstr(0), "0 msec");
        assert_eq!(tstr(999), "999 msec");
        assert_eq!(tstr(1_000), "1.000 sec");
        assert_eq!(tstr(179_999), "179.999 sec");
        assert_eq!(tstr(180_000), "3.000 minutes");
        // Not 179.999: 10,799,999 / 60,000 rounds up to 180.000 at three
        // decimals, which is what glibc prints too. The branch is still the
        // minutes one, and the next millisecond is what crosses over.
        assert_eq!(tstr(10_799_999), "180.000 minutes");
        assert_eq!(tstr(10_800_000), "3.000 hours");
        assert_eq!(tstr(647_999_999), "180.000 hours");
        // 7.5 days, the fourth bound, not 3 days.
        assert_eq!(tstr(648_000_000), "7.500 days");
    }

    #[test]
    fn the_printed_precisions_are_the_reference_s() {
        // `%.2e`, the message lines that spell a toi.
        assert_eq!(c_exponential(1.0, 2), "1.00e+00");
        assert_eq!(c_exponential(5.96e-9, 2), "5.96e-09");
        // `%.3e`, a printed mark.
        assert_eq!(c_exponential(9.812e-4, 3), "9.812e-04");
        assert_eq!(c_exponential(0.01, 3), "1.000e-02");
        // `%e`, the stream files, unchanged.
        assert_eq!(c_exponential(0.913, 6), "9.130000e-01");
    }

    #[test]
    fn c_general_matches_c_s_shortest_of_two_forms() {
        // Verified against glibc's printf("%g", ...).
        assert_eq!(c_general(1.0 / 60.0), "0.0166667");
        assert_eq!(c_general(0.01), "0.01");
        assert_eq!(c_general(0.0), "0");
        assert_eq!(c_general(1e-7), "1e-07");
        assert_eq!(c_general(1234567.0), "1.23457e+06");
        assert_eq!(c_general(100000.0), "100000");
        assert_eq!(c_general(1000000.0), "1e+06");
        assert_eq!(c_general(0.0001), "0.0001");
        assert_eq!(c_general(-0.5), "-0.5");
        assert_eq!(c_general(1.5), "1.5");
        // Rounding to six significant digits carries this one into the next
        // exponent, so it takes the `%e` branch and not the `%f` one.
        assert_eq!(c_general(999999.5), "1e+06");
    }

    #[test]
    fn an_elapsed_duration_prints_whole_milliseconds_and_records_the_fraction() {
        // THE PRINTED LINE AND THE RECORDED VALUE PART COMPANY HERE, ON
        // PURPOSE. The printed duration truncates to whole milliseconds, so the
        // STREAM says `7 msec`.
        //
        // The FILE keeps the fraction. Writing that same truncated integer
        // would put a literal `0` in every timer row of any scene whose phases
        // run under a millisecond, which is what the addon's Matrix Assembly,
        // PCG Solve and Line Search cells read. Measured on a small fixture,
        // every row became `0`. Nothing downstream requires the integer: every
        // reader parses the value column as `f64`.
        let marked: Marked = Duration::from_micros(7_900).into();
        assert_eq!(
            marked.printed("linsolve").as_deref(),
            Some("> linsolve...7 msec")
        );
        assert!(
            (marked.recorded() - 7.9).abs() < 1e-9,
            "recorded {}, want 7.9",
            marked.recorded()
        );
        // And the recorded fraction reaches the file in the exponential form,
        // because it is not a whole number.
        assert_eq!(format_line(0.5, marked.recorded()), "0.500000 7.900000e+00\n");
    }

    #[test]
    fn a_whole_millisecond_elapsed_still_records_the_integer_form() {
        // The complement of the test above: when the phase really did take a
        // whole number of milliseconds, the file keeps the integer spelling.
        let marked: Marked = Duration::from_millis(3).into();
        assert_eq!(
            marked.printed("linsolve").as_deref(),
            Some("> linsolve...3 msec")
        );
        assert_eq!(format_line(0.5, marked.recorded()), "0.500000 3\n");
    }

    #[test]
    fn a_quiet_channel_records_and_prints_nothing() {
        let marked = Marked::quiet(1.5);
        assert_eq!(marked.recorded(), 1.5);
        assert!(marked.printed("nt_head").is_none());
    }

    #[test]
    fn a_printed_mark_takes_the_same_branch_its_stream_line_does() {
        let whole: Marked = 199.0.into();
        assert_eq!(whole.printed("iter").as_deref(), Some("* iter: 199"));
        assert_eq!(format_line(0.0, whole.recorded()), "0.000000 199\n");
        let fractional: Marked = 9.812e-4.into();
        assert_eq!(
            fractional.printed("reresid").as_deref(),
            Some("* reresid: 9.812e-04")
        );
        assert_eq!(
            format_line(0.0, fractional.recorded()),
            "0.000000 9.812000e-04\n"
        );
    }
}
