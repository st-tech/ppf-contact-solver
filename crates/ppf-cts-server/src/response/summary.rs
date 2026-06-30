// File: crates/ppf-cts-server/src/response/summary.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Channel tables and helpers that produce the live `summary` and the
// idle `average_summary` maps spliced into the status response.

use serde_json::{Map, Value};

use crate::hardware::runtime_usage;
use ppf_cts_core::datamodel::session::{
    average_summary_from_disk, convert_average_count_optional, convert_integer_optional,
    convert_ratio_optional, convert_time_optional, format_stretch, latest_log_value,
    latest_step_average, log_filename_path,
};

/// Channel keys rendered by the live `summary` and the baked-fallback
/// `average_summary`. Order is the panel row order.
const CHANNEL_NAMES: [&str; 12] = [
    "time-per-frame",
    "time-per-step",
    "num-contact",
    "newton-steps",
    "pcg-iter",
    "max-sigma",
    "matrix-assembly",
    "toi-advanced",
    "dyn-consumed",
    "pcg-linsolve",
    "line-search",
    "toi",
];

/// Channel-name -> log-filename pairs harvested at compile time by
/// `build.rs`. Used as the fallback when `EngineConfig::log_filenames`
/// is empty at runtime (e.g. a stripped release bundle whose binary
/// can't see `crates/ppf-cts-*/src/`). Generated; not hand-maintained.
const BAKED_LOG_CHANNELS: &[(&str, &str)] =
    &include!(concat!(env!("OUT_DIR"), "/log_channels_baked.rs"));

/// Hand-pinned filenames for the six channels rendered in the live
/// "Realtime Statistics" panel. The docstring harvester can mis-route
/// a panel cell to a non-existent file when configuration drifts;
/// pinning keeps the panel resilient to that. The names are stable
/// on-disk artefacts of `cpp/main/main.cu`: if any of them gets
/// renamed the constant has to follow, but that's a deliberate
/// change, not a parser inference.
pub const LIVE_SUMMARY_CHANNELS: [(&str, &str); 12] = [
    (CHANNEL_NAMES[0], "time_per_frame.out"),
    (CHANNEL_NAMES[1], "advance.out"),
    (CHANNEL_NAMES[2], "advance.num_contact.out"),
    (CHANNEL_NAMES[3], "advance.newton_steps.out"),
    (CHANNEL_NAMES[4], "advance.iter.out"),
    (CHANNEL_NAMES[5], "advance.max_sigma.out"),
    // Written by `logging.push("matrix assembly")` in cpp/main/main.cu
    // (m_name "advance", spaces -> underscores), one entry per Newton
    // iteration; averaged over the latest step.
    (CHANNEL_NAMES[6], "advance.matrix_assembly.out"),
    // Advanced fractional step size (ratio in (0, 1]) and the dynamic
    // contact-Hessian memory-usage ratio ([0, 1]); both written once per
    // step via `logging.mark` in cpp/main/main.cu.
    (CHANNEL_NAMES[7], "advance.toi_advanced.out"),
    (CHANNEL_NAMES[8], "advance.dyn_consumed.out"),
    // PCG solve wall-clock, one entry per Newton iteration; averaged
    // over the latest step.
    (CHANNEL_NAMES[9], "advance.linsolve.out"),
    // Line-search wall-clock (CCD + strain-limit CCD), written by
    // `logging.push("line search")` in cpp/main/main.cu, one entry per
    // Newton iteration; averaged over the latest step.
    (CHANNEL_NAMES[10], "advance.line_search.out"),
    // Line-search time of impact (ratio in (0, 1]), written once per
    // Newton iteration via `logging.mark("toi", ...)`; averaged over the
    // latest step.
    (CHANNEL_NAMES[11], "advance.toi.out"),
];

/// Read the live metric values for the pinned channels, render
/// them with the `_optional` formatters so unreadable channels surface
/// as `"N/A"`, then append GPU/VRAM/CPU/RAM utilization rows.
///
/// Always emits the fixed summary keys (frame, the timing rows including
/// matrix-assembly, the count rows; plus `stretch` when `max-sigma`
/// is readable and positive (max-sigma > 0)) so the panel renders the
/// same row layout in both healthy and degraded states. Host-capability
/// rows (GPU/VRAM/CPU/RAM) are omitted when the probe fails: those are not
/// solver rows, so suppressing the row is more honest than implying
/// the value is "currently" unknown.
pub fn build_live_summary(
    frame: i32,
    info_path: &std::path::Path,
    pairs: &[(&str, &str)],
) -> Map<String, Value> {
    fn latest(info_path: &std::path::Path, name: &str, pairs: &[(&str, &str)]) -> Option<f64> {
        log_filename_path(info_path, name, pairs).and_then(|p| latest_log_value(&p))
    }
    // Per-Newton-iteration channels (a step emits one row per iteration,
    // including the trailing error-reduction iteration): show the step's
    // average, not just the last iteration's value.
    fn step_avg(info_path: &std::path::Path, name: &str, pairs: &[(&str, &str)]) -> Option<f64> {
        log_filename_path(info_path, name, pairs).and_then(|p| latest_step_average(&p))
    }
    let tpf = latest(info_path, "time-per-frame", pairs);
    let tps = latest(info_path, "time-per-step", pairs);
    let ma = step_avg(info_path, "matrix-assembly", pairs);
    let nc = step_avg(info_path, "num-contact", pairs);
    let dc = latest(info_path, "dyn-consumed", pairs);
    let ns = latest(info_path, "newton-steps", pairs);
    let pi = step_avg(info_path, "pcg-iter", pairs);
    let pl = step_avg(info_path, "pcg-linsolve", pairs);
    let lsr = step_avg(info_path, "line-search", pairs);
    let toi = step_avg(info_path, "toi", pairs);
    let ta = latest(info_path, "toi-advanced", pairs);
    let ms = latest(info_path, "max-sigma", pairs);

    let mut obj: Map<String, Value> = Map::new();
    obj.insert("frame".into(), Value::String(frame.to_string()));
    obj.insert(
        "time-per-frame".into(),
        Value::String(convert_time_optional(tpf)),
    );
    obj.insert(
        "time-per-step".into(),
        Value::String(convert_time_optional(tps)),
    );
    // Latest step's average matrix-assembly wall-clock (per Newton
    // iteration), grouped with the other timing rows.
    obj.insert(
        "matrix-assembly".into(),
        Value::String(convert_time_optional(ma)),
    );
    // Latest step's average contact count over its Newton iterations.
    // Sub-thousand averages show a fixed XX.YY (the average is usually
    // whole but can be fractional); larger counts get a k/M suffix so the
    // cell reads `12.44k` instead of a long digit run.
    obj.insert(
        "num-contact".into(),
        Value::String(convert_average_count_optional(nc)),
    );
    // Dynamic contact-Hessian memory-usage ratio, grouped with the
    // contact row. Live-only (not in the average summary).
    obj.insert(
        "dyn-consumed".into(),
        Value::String(convert_ratio_optional(dc)),
    );
    obj.insert(
        "newton-steps".into(),
        Value::String(convert_integer_optional(ns)),
    );
    // Latest step's average PCG iteration count and solve wall-clock over
    // its Newton iterations. Sub-thousand averages show XX.YY; larger
    // counts get a k/M suffix for the same reason as num-contact.
    obj.insert(
        "pcg-iter".into(),
        Value::String(convert_average_count_optional(pi)),
    );
    obj.insert(
        "pcg-linsolve".into(),
        Value::String(convert_time_optional(pl)),
    );
    // Latest step's average line-search wall-clock (CCD + strain-limit
    // CCD) per Newton iteration, grouped with the other timing rows.
    obj.insert(
        "line-search".into(),
        Value::String(convert_time_optional(lsr)),
    );
    // Latest step's average line-search time of impact (ratio in (0, 1]),
    // next to the advanced-step ratio.
    obj.insert("toi".into(), Value::String(convert_ratio_optional(toi)));
    // Advanced fractional step size (ratio), next to the step metrics.
    obj.insert(
        "toi-advanced".into(),
        Value::String(convert_ratio_optional(ta)),
    );
    // `stretch` is the optional 7th row: present only when max-sigma
    // is readable AND positive, matching `format_log_summary`'s rule.
    // When the reading fails we omit it so the panel doesn't show a
    // misleading "N/A" row for a metric that may simply be inactive
    // (no shells / rods in the scene).
    if let Some(s) = ms {
        if let Some(v) = format_stretch(s) {
            obj.insert("stretch".into(), Value::String(v));
        }
    }

    let usage = runtime_usage();
    if let Some(v) = usage.gpu_util {
        obj.insert("GPU Util".into(), Value::String(v));
    }
    if let Some(v) = usage.vram_usage {
        obj.insert("VRAM Usage".into(), Value::String(v));
    }
    if let Some(v) = usage.cpu_usage {
        obj.insert("CPU Usage".into(), Value::String(v));
    }
    if let Some(v) = usage.ram_usage {
        obj.insert("RAM Usage".into(), Value::String(v));
    }
    obj
}

/// Resolve the channel-name -> log-filename pairs to use for
/// `average_summary`. Falls back to the baked table when the runtime
/// harvest produced nothing (e.g., release bundle without `crates/`).
pub fn average_summary_pairs(harvested: &[(String, String)]) -> Vec<(&str, &str)> {
    if harvested.is_empty() {
        BAKED_LOG_CHANNELS.to_vec()
    } else {
        harvested
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect()
    }
}

/// Build the `average_summary` map (channel -> formatted string) by
/// reading each pair's `.out` file. Returns an empty map when no
/// channel has any readable data, so the caller can omit the field.
pub fn build_average_summary(
    data_dir: &std::path::Path,
    pairs: &[(&str, &str)],
) -> Map<String, Value> {
    let mut obj: Map<String, Value> = Map::new();
    let avg = average_summary_from_disk(data_dir, pairs);
    for (k, v) in avg {
        obj.insert(k, Value::String(v));
    }
    obj
}
