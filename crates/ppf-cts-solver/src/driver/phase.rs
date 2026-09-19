// File: phase.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! Wall time per HOST phase, the counterpart to the backend's per-region tally.
//!
//! `PPF_REGION_STATS` measures the time spent INSIDE device boundaries, which
//! was measured at 31 percent of a `drape` run: the other 69 percent is host
//! work between the boundaries, which a per-region tally cannot see and which a
//! sampling profiler cannot reach on a host that restricts
//! `perf_event_paranoid`.
//!
//! This closes that gap from the driver's side. A phase brackets one top-level
//! call in the step, so its total includes the device boundaries that call
//! opens AND the host work around them; subtracting the region tally from a
//! phase says how much of it was the host.
//!
//! OFF UNLESS `PPF_PHASE_STATS` IS SET, and a nanosecond accumulator per label
//! rather than a sample, because a sampling profiler is what is unavailable.
//!
//! # A guard must bracket ONE call, and the obvious spelling does not
//!
//! `let _phase = start("x");` on the line above a call runs until the end of
//! the ENCLOSING SCOPE, so it charges that phase with everything after it too.
//! Instrumented that way on `step.rs`, eleven phases reported 28.4 seconds
//! inside a 10.2 second run and their times came out nearly equal and
//! descending, which is the signature: each row is the tail of the function
//! rather than one call. The readings are still usable, as consecutive
//! DIFFERENCES, but only if a reader knows that.
//!
//! Bracket the call itself, `{ let _phase = start("x"); f()?; }`, and check the
//! total against the run's wall clock: a total ABOVE it means the guards are
//! nesting. Wrapping the sites mechanically is not safe either, since several
//! sit inside `if`/`else` chains where an inserted block splits the chain.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::Instant;

static TALLY: OnceLock<Mutex<BTreeMap<&'static str, (u64, u128)>>> = OnceLock::new();

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("PPF_PHASE_STATS").is_some())
}

/// A phase that charges its own wall time on drop.
pub struct Phase {
    name: &'static str,
    at: Instant,
}

/// Bracket a host phase. Returns `None` when the tally is off, so a call site
/// costs one boolean.
pub fn start(name: &'static str) -> Option<Phase> {
    enabled().then(|| Phase { name, at: Instant::now() })
}

impl Drop for Phase {
    fn drop(&mut self) {
        let elapsed = self.at.elapsed().as_nanos();
        let map = TALLY.get_or_init(|| Mutex::new(BTreeMap::new()));
        if let Ok(mut guard) = map.lock() {
            let row = guard.entry(self.name).or_insert((0, 0));
            row.0 += 1;
            row.1 += elapsed;
        }
    }
}

/// Print the tally, longest first. Called once at the end of a solve.
pub fn dump() {
    if !enabled() {
        return;
    }
    let Some(map) = TALLY.get() else { return };
    let Ok(guard) = map.lock() else { return };
    if guard.is_empty() {
        eprintln!("\n=== PPF_PHASE_STATS: no host phases recorded ===");
        return;
    }
    let mut rows: Vec<_> = guard.iter().map(|(k, v)| (*k, v.0, v.1)).collect();
    rows.sort_by(|a, b| b.2.cmp(&a.2));
    let total: u128 = rows.iter().map(|r| r.2).sum();
    eprintln!("\n=== PPF_PHASE_STATS: host wall time by phase ===");
    eprintln!("{:<40} {:>10} {:>12} {:>8}", "phase", "calls", "seconds", "share");
    for (name, calls, nanos) in &rows {
        eprintln!(
            "{:<40} {:>10} {:>12.3} {:>7.1}%",
            name,
            calls,
            *nanos as f64 / 1e9,
            if total > 0 { 100.0 * *nanos as f64 / total as f64 } else { 0.0 }
        );
    }
    eprintln!("{:<40} {:>10} {:>12.3}", "TOTAL", "", total as f64 / 1e9);
}

extern "C" fn dump_at_exit() {
    dump();
}

/// Install the exit hook, once. `libc::atexit` is what `crate::status_writer`
/// already uses for the same reason: a solver run ends without unwinding
/// through a shutdown path that a `Drop` could hang on.
pub fn register() {
    if !enabled() {
        return;
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // Safety: the hook takes no arguments, touches only this module's
        // statics, and runs after main on the process's own thread.
        unsafe {
            libc::atexit(dump_at_exit);
        }
    });
}
