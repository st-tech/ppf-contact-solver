// File: crates/ppf-cts-server/src/hardware.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// One-shot hardware probe used to populate the `hardware` field in
// the response JSON the addon panel renders as "Remote Hardware".
// Subprocess calls are best-effort: any failure leaves the
// corresponding field as "Unknown" instead of aborting the probe.

use std::process::Command;
use std::time::Duration;

use crate::config::HardwareInfo;

/// Per-subprocess wall-clock cap shared by every `run_with_timeout`
/// call site. A stuck nvidia-smi (bad driver state) is the main
/// motivation; the slow path is a Windows powershell cold start.
/// Defined once so a tuning change touches a single spot.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// MiB to GiB. Input is nvidia-smi output under `--format=...,nounits`,
/// which reports memory in MiB, so the divisor is 1024.
fn mib_to_gib(mib: u64) -> f64 {
    mib as f64 / 1024.0
}

/// Bytes to GiB. Input is sysinfo total_memory()/used_memory(), which
/// report bytes, so the divisor is 1024^3. Kept distinct from
/// `mib_to_gib` on purpose: the two source units must not be unified.
fn bytes_to_gib(b: u64) -> f64 {
    b as f64 / 1024_f64.powi(3)
}

/// Shared "<pct>% (<used>/<total> GB)" formatter for the two
/// used/total rows (VRAM, RAM). `pct` is passed in already computed
/// from the raw source units so the displayed percentage matches the
/// historical rounding rather than recomputing from the rounded GiB.
fn fmt_usage(pct: u64, used_gib: f64, total_gib: f64) -> String {
    format!("{}% ({:.1}/{:.1} GB)", pct, used_gib, total_gib)
}

/// Runtime utilization snapshot built fresh on every status
/// response, to populate the live "Realtime Statistics" rows (GPU
/// Util / VRAM Usage / CPU Usage / RAM Usage) the addon panel
/// renders alongside the log-summary metrics. Each field is an
/// already-formatted string so the response builder can drop them
/// straight into the summary dict.
#[derive(Debug, Default, Clone)]
pub struct RuntimeUsage {
    pub gpu_util: Option<String>,
    pub vram_usage: Option<String>,
    pub cpu_usage: Option<String>,
    pub ram_usage: Option<String>,
}

/// Probe nvidia-smi (GPU util + VRAM) and sysinfo (CPU % + RAM %).
/// Best-effort: every probe failure leaves the corresponding field
/// `None` so the response builder can omit it without blocking the
/// poll cycle. Caller is expected to clamp the cost: on hosts without
/// a GPU, nvidia-smi prints to stderr and returns quickly; on hosts
/// with a stuck driver, `run_with_timeout` kills the child after
/// PROBE_TIMEOUT.
pub fn runtime_usage() -> RuntimeUsage {
    let mut out = run_with_timeout(
        Command::new("nvidia-smi")
            .args([
                "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,name",
                "--format=csv,noheader,nounits",
            ])
            .env_remove("CUDA_VISIBLE_DEVICES"),
        PROBE_TIMEOUT,
    )
    .map(|s| {
        parse_runtime_gpu_usage(
            &s,
            std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref(),
        )
    })
    .unwrap_or_default();

    // `cpu_usage()` reports the delta between successive refreshes,
    // so we hold one `System` across calls and refresh once per
    // poll. Keeps `runtime_usage` non-blocking at the cost of the
    // first poll always reporting 0.0% CPU. The mutex is held for
    // microseconds (one refresh + a couple of reads), and status
    // polls are serialized at the wire layer anyway.
    let mutex = SYSINFO.get_or_init(|| std::sync::Mutex::new(sysinfo::System::new()));
    let mut sys = mutex.lock().expect("sysinfo mutex poisoned");
    sys.refresh_cpu_usage();
    let cpus = sys.cpus();
    if !cpus.is_empty() {
        let avg = cpus.iter().map(|c| c.cpu_usage()).sum::<f32>() / (cpus.len() as f32);
        out.cpu_usage = Some(format!("{}%", avg.round() as i64));
    }

    sys.refresh_memory();
    let total = sys.total_memory();
    let used = sys.used_memory();
    if total > 0 {
        let pct = (100.0 * used as f64 / total as f64).round() as u64;
        out.ram_usage = Some(fmt_usage(pct, bytes_to_gib(used), bytes_to_gib(total)));
    }

    out
}

fn parse_runtime_gpu_usage(stdout: &str, visible: Option<&str>) -> RuntimeUsage {
    let mut out = RuntimeUsage::default();
    let rows = ppf_cts_core::utils::parse_gpu_rows(stdout, 3);
    let pairs: Vec<(u32, &str)> = rows
        .iter()
        .map(|row| (row.index, row.uuid.as_str()))
        .collect();
    let Some(position) = ppf_cts_core::utils::visible_device_position(&pairs, visible) else {
        return out;
    };
    let selected = &rows[position];
    out.gpu_util = Some(format!("{}%", selected.fields[0]));
    if let (Ok(used), Ok(total)) = (
        selected.fields[1].parse::<u64>(),
        selected.fields[2].parse::<u64>(),
    ) {
        let pct = if total > 0 {
            (100.0 * used as f64 / total as f64).round() as u64
        } else {
            0
        };
        out.vram_usage = Some(fmt_usage(pct, mib_to_gib(used), mib_to_gib(total)));
    }
    out
}

static SYSINFO: std::sync::OnceLock<std::sync::Mutex<sysinfo::System>> =
    std::sync::OnceLock::new();

/// Run the hardware probe. Always returns a populated struct, with
/// "Unknown" for any field whose probe failed. Intended to be called
/// once at server startup; the result is then cloned into every
/// response.
pub fn probe() -> HardwareInfo {
    let mut hw = HardwareInfo::default();

    probe_gpu(&mut hw);
    probe_cpu_and_ram(&mut hw);

    // Compile-time backend: the `emulated` feature swaps the CUDA
    // backend for a CPU stub (see ppf-cts-solver/build.rs). The addon
    // warns before running on such a build.
    hw.emulated = cfg!(feature = "emulated");

    hw
}

/// `nvidia-smi --query-gpu=index,uuid,memory.total,compute_cap,name` parses the
/// CSV rows for GPU/VRAM/SM, then a plain `nvidia-smi` for the CUDA
/// driver line. Both calls run under a `run_with_timeout` capped at
/// PROBE_TIMEOUT so a stuck driver can't block startup.
///
/// The row reported is the one `CUDA_VISIBLE_DEVICES` selects, so what the
/// add-on's Remote Hardware block shows is the device the solver runs on,
/// named by index as well as model. It is the only place that names it, so a
/// machine holding two cards of the same model has to be readable here.
fn probe_gpu(hw: &mut HardwareInfo) {
    // Enumerate every device, then apply CUDA_VISIBLE_DEVICES in code.
    // Measured on a four-GPU Windows host, nvidia-smi ignores that variable
    // and lists every device anyway, but stripping it states the requirement
    // rather than resting on a driver behavior: a build that did filter would
    // renumber the surviving row and the selection would resolve to nothing.
    if let Some(out) = run_with_timeout(
        Command::new("nvidia-smi")
            .args([
                "--query-gpu=index,uuid,memory.total,compute_cap,name",
                "--format=csv,noheader,nounits",
            ])
            .env_remove("CUDA_VISIBLE_DEVICES"),
        PROBE_TIMEOUT,
    ) {
        let rows = ppf_cts_core::utils::parse_gpu_rows(&out, 2);
        match ppf_cts_core::utils::selected_gpu_row(&rows) {
            Ok(selected) => {
                hw.gpu_index = selected.index as i64;
                // Index first, always: this row is the one place that names
                // the device the solver is on, and two cards of the same model
                // are told apart by nothing else.
                hw.gpu = format!("{}: {}", selected.index, selected.name);
                if let Ok(mb) = selected.fields[0].parse::<u64>() {
                    hw.vram = format!("{:.1} GB", mib_to_gib(mb));
                }
                hw.sm = format!("sm_{}", selected.fields[1].replace('.', ""));
            }
            // A selection that names no device leaves CUDA with nothing to run
            // on. Report why in the field the add-on shows, rather than leaving
            // the default there for the user to interpret.
            Err(err) => hw.gpu = err.to_string(),
        }
    }

    if let Some(out) = run_with_timeout(&mut Command::new("nvidia-smi"), PROBE_TIMEOUT) {
        for line in out.lines() {
            if line.contains("CUDA Version") {
                for part in line.split_whitespace() {
                    if part.parse::<f64>().is_ok() {
                        hw.cuda = part.to_string();
                        break;
                    }
                }
                break;
            }
        }
    }
}

/// CPU brand string + total RAM via `sysinfo`. Falls back to an OS
/// subprocess (`lscpu` on Linux, `powershell` on Windows) only if
/// sysinfo cannot surface a brand string, which never happens on the
/// platforms we ship to but mirrors what the Python probe did.
fn probe_cpu_and_ram(hw: &mut HardwareInfo) {
    let mut sys = sysinfo::System::new();
    sys.refresh_cpu_specifics(sysinfo::CpuRefreshKind::new().with_frequency());
    sys.refresh_memory();

    let mut got_brand = false;
    if let Some(cpu) = sys.cpus().first() {
        let brand = cpu.brand().trim().to_string();
        if !brand.is_empty() {
            hw.cpu = brand;
            got_brand = true;
        }
    }
    // Gate on probe success, not the default sentinel string, so
    // changing HardwareInfo::default().cpu cannot silently disable
    // this OS-subprocess fallback.
    if !got_brand {
        if cfg!(target_os = "windows") {
            if let Some(out) = run_with_timeout(
                Command::new("powershell").args([
                    "-Command",
                    "(Get-CimInstance Win32_Processor).Name",
                ]),
                PROBE_TIMEOUT,
            ) {
                let trimmed = out.trim().to_string();
                if !trimmed.is_empty() {
                    hw.cpu = trimmed;
                }
            }
        } else if let Some(out) =
            run_with_timeout(&mut Command::new("lscpu"), PROBE_TIMEOUT)
        {
            for line in out.lines() {
                if let Some(rest) = line.strip_prefix("Model name:") {
                    hw.cpu = rest.trim().to_string();
                    break;
                }
            }
        }
    }

    let total_bytes = sys.total_memory();
    if total_bytes > 0 {
        hw.ram = format!("{:.1} GB", bytes_to_gib(total_bytes));
    }
}

/// `Command::output()` blocks indefinitely; nvidia-smi can hang when
/// the driver is in a bad state, so we cap each probe at `timeout`.
/// On timeout we kill the child and return `None`; the caller leaves
/// the corresponding HardwareInfo field as "Unknown".
///
/// stdout is drained on a dedicated thread so the pipe never
/// backpressures the child: if we only read after the process exited,
/// any command that writes more than the OS pipe buffer (~64 KB on
/// Linux) before exiting would block in write(), never exit, and get
/// killed at `timeout` with its output lost. stderr stays null, so
/// only stdout needs a draining thread. The blocking read returns as
/// soon as the child closes stdout (on exit or kill), so the join
/// after the poll loop is prompt.
fn run_with_timeout(cmd: &mut Command, timeout: Duration) -> Option<String> {
    let mut child = cmd
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;

    let stdout = child.stdout.take();
    let reader = std::thread::spawn(move || {
        let mut buf = String::new();
        if let Some(mut stdout) = stdout {
            use std::io::Read;
            let _ = stdout.read_to_string(&mut buf);
        }
        buf
    });

    let start = std::time::Instant::now();
    loop {
        match child.try_wait().ok()? {
            Some(status) if status.success() => {
                let _ = child.wait();
                return reader.join().ok();
            }
            Some(_) => return None,
            None => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }

    }
}

#[cfg(test)]
mod tests {
    use super::parse_runtime_gpu_usage;

    const TWO_GPUS: &str = "0, GPU-aaaa, 11, 1024, 4096, NVIDIA L40S\n\
1, GPU-bbbb, 73, 8192, 16384, NVIDIA RTX A4000, Inc.\n";

    #[test]
    fn runtime_usage_selects_the_visible_device() {
        let usage = parse_runtime_gpu_usage(TWO_GPUS, Some("GPU-bbbb"));
        assert_eq!(usage.gpu_util.as_deref(), Some("73%"));
        assert_eq!(usage.vram_usage.as_deref(), Some("50% (8.0/16.0 GB)"));
    }

    #[test]
    fn runtime_usage_honors_an_empty_mask() {
        let usage = parse_runtime_gpu_usage(TWO_GPUS, Some(""));
        assert!(usage.gpu_util.is_none());
        assert!(usage.vram_usage.is_none());
    }
}
