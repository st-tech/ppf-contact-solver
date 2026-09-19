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
    // ONLY A CUDA BUILD HAS AN nvidia-smi TO ASK. This runs on every status
    // poll, so on a Metal or CPU build the old unconditional call spawned a
    // process that does not exist several times a second, for a result that
    // could only ever be empty. Metal exposes no equivalent per-process
    // utilization counter, so those two backends report no GPU rows here, the
    // same answer the startup probe gives.
    let mut out = if ppf_cts_core::utils::backend() == ppf_cts_core::utils::Backend::Cuda {
        run_with_timeout(
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
        .unwrap_or_default()
    } else {
        #[cfg(target_os = "macos")]
        {
            metal_runtime_usage()
        }
        #[cfg(not(target_os = "macos"))]
        {
            RuntimeUsage::default()
        }
    };

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

/// The Metal counterpart of the nvidia-smi sample above, read from a sampler
/// this process owns rather than from the counter directly.
///
/// WHY A SAMPLER AND NOT A READ. IOAccelerator's "Device Utilization %" reports
/// utilization BETWEEN READS, so whoever reads it defines the window and
/// consumes it. Reading it here, on the status path, made the measurement
/// window whatever gap the add-on happened to leave between polls: a slow
/// poller measured seconds, a fast one measured milliseconds, and any other
/// reader on the machine, Activity Monitor included, silently took the interval
/// away. The number that produced was not wrong so much as undefined. A sampler
/// on a fixed cadence owns the window, so what is reported means the same thing
/// on every poll and whatever else is running.
///
/// WHY AN AVERAGE AND NOT THE LATEST SAMPLE. Measured against a Metal load with
/// a known 25.6% duty cycle, single samples at 250 ms read 24 to 29, which is
/// right; but the same measurement over the real fixture suite read 0 in 114 of
/// 120 samples with occasional excursions to 52, because those scenes are tiny
/// and host-bound. A row flickering between 0 and 52 says nothing a reader can
/// use. The mean over the window is the fraction of recent wall time the GPU
/// was busy, which is the question the row is asking.
///
/// IT IS ACCURATE WHERE IT MATTERS, and that was measured rather than assumed:
/// during `examples/bench_drape.py` on Metal, 5 res-128 sheets over a sphere,
/// twelve consecutive 250 ms samples read 86 to 99 with a mean of 92.2% and no
/// zeros. A quiet row on a small scene is the truth about that scene.
#[cfg(target_os = "macos")]
fn metal_runtime_usage() -> RuntimeUsage {
    let mut out = RuntimeUsage::default();
    if ppf_cts_core::utils::backend() != ppf_cts_core::utils::Backend::Metal {
        // The branch that reaches here is "not CUDA", so a CPU build on a Mac
        // lands in it. Reporting what the GPU is doing there would describe
        // hardware this solver is not using.
        return out;
    }
    let Some((util_pct, in_use_bytes)) = metal_sampler_snapshot() else {
        // No window yet, or no counter on this machine. An absent row rather
        // than a zero: the two mean different things and only one of them is
        // true here.
        return out;
    };
    out.gpu_util = Some(format!("{}%", util_pct));
    let total = *METAL_WORKING_SET.get_or_init(|| {
        ppf_cts_core::utils::metal_device_info()
            .map(|info| info.working_set_bytes)
            .unwrap_or(0)
    });
    if in_use_bytes > 0 && total > 0 {
        let pct = (100.0 * in_use_bytes as f64 / total as f64).round() as u64;
        out.vram_usage = Some(fmt_usage(pct, bytes_to_gib(in_use_bytes), bytes_to_gib(total)));
    }
    out
}

/// `recommendedMaxWorkingSetSize`, read once. A property of the machine, and
/// the denominator of the VRAM percentage above.
#[cfg(target_os = "macos")]
static METAL_WORKING_SET: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

/// How often the sampler reads the counter. 250 ms is the interval the
/// measurements above were taken at and the one the counter was accurate at:
/// sampling the same 25.6% load every 50 ms instead read a mean of 45.8%, so a
/// faster cadence is not a free improvement in resolution.
#[cfg(target_os = "macos")]
const METAL_SAMPLE_INTERVAL: Duration = Duration::from_millis(250);

/// How many samples the reported average covers, so the row describes about the
/// last two seconds. Long enough that a gap between two frames does not empty
/// it, short enough to follow a solve that has just started or stopped.
#[cfg(target_os = "macos")]
const METAL_WINDOW: usize = 8;

/// THE OnceLock HOLDS THE SAME Arc THE THREAD WRITES THROUGH. Storing a plain
/// Mutex here instead would need the Arc unwrapped to build it, which cannot
/// succeed while the worker holds its clone, so the reader would end up owning
/// a second, permanently empty window and the row would never appear.
#[cfg(target_os = "macos")]
static METAL_SAMPLES: std::sync::OnceLock<
    std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<(u32, u64)>>>,
> = std::sync::OnceLock::new();

/// Start the sampler once and return the window's current average, or `None`
/// while it holds nothing.
///
/// THE THREAD IS DETACHED AND LIVES AS LONG AS THE PROCESS, which is what makes
/// the cadence fixed. It is started on first use rather than at startup so a
/// server on a machine nobody asks about never spawns it.
#[cfg(target_os = "macos")]
fn metal_sampler_snapshot() -> Option<(u32, u64)> {
    let samples = METAL_SAMPLES.get_or_init(|| {
        let shared: std::sync::Arc<
            std::sync::Mutex<std::collections::VecDeque<(u32, u64)>>,
        > = std::sync::Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new()));
        let worker = std::sync::Arc::clone(&shared);
        // A failed spawn leaves the window empty forever, which reports an
        // absent row rather than a wrong one.
        let _ = std::thread::Builder::new()
            .name("metal-gpu-sampler".into())
            .spawn(move || loop {
                if let Some(usage) = ppf_cts_core::utils::metal_usage() {
                    if let Ok(mut window) = worker.lock() {
                        window.push_back((
                            usage.device_util_pct,
                            usage.in_use_bytes.unwrap_or(0),
                        ));
                        while window.len() > METAL_WINDOW {
                            window.pop_front();
                        }
                    }
                }
                std::thread::sleep(METAL_SAMPLE_INTERVAL);
            });
        shared
    });
    let window = samples.lock().ok()?;
    if window.is_empty() {
        return None;
    }
    let mean = window.iter().map(|(pct, _)| *pct as u64).sum::<u64>() / window.len() as u64;
    let latest_bytes = window.back().map(|(_, bytes)| *bytes).unwrap_or(0);
    Some((mean as u32, latest_bytes))
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

/// Run the hardware probe. Intended to be called once at server startup; the
/// result is then cloned into every response.
///
/// A field whose probe APPLIED and failed keeps "Unknown". A field that does
/// not apply to this backend is left absent, so the add-on draws no row for it
/// at all: see [`HardwareInfo`].
pub fn probe() -> HardwareInfo {
    let mut hw = HardwareInfo::default();

    // WHICH GPU QUESTION TO ASK IS THE BACKEND'S TO ANSWER, and the match is
    // exhaustive so a fourth backend cannot be added without deciding what it
    // reports here. Asking nvidia-smi on every host was the old shape, and on
    // macOS it spawned a binary that does not exist and then filled four rows
    // with "Unknown".
    match ppf_cts_core::utils::backend() {
        ppf_cts_core::utils::Backend::Cuda => probe_gpu_cuda(&mut hw),
        ppf_cts_core::utils::Backend::Metal => {
            #[cfg(target_os = "macos")]
            probe_gpu_metal(&mut hw);
        }
        // The bundled HIP runtime names the device and nothing else here is
        // ROCm's to report: the CUDA version, SM and index rows are CUDA's, and
        // the runtime offers no memory or utilization figure this probe reads.
        ppf_cts_core::utils::Backend::Rocm => {
            hw.gpu = ppf_cts_core::utils::rocm_device_arch();
        }
        // A CPU build has no GPU to describe, so every GPU row stays absent.
        // The CPU and RAM rows below are the whole of what it can report, which
        // is also the whole of what it uses.
        ppf_cts_core::utils::Backend::Cpu => {}
    }
    probe_cpu_and_ram(&mut hw);

    hw
}

/// Fill the GPU rows from the default Metal device.
///
/// It reports a name, a family and a working set. The remaining fields are
/// CUDA's: there is no index to disambiguate two cards when Metal opens the
/// system default device and offers no way to name another, and no CUDA version
/// exists to report. Those rows are left absent rather than filled with a
/// placeholder.
#[cfg(target_os = "macos")]
fn probe_gpu_metal(hw: &mut HardwareInfo) {
    let Some(info) = ppf_cts_core::utils::metal_device_info() else {
        // No Metal device. `check_gpu` is what refuses a run over that; this
        // only decides whether a row is drawn, so it says nothing rather than
        // claiming a device it could not see.
        return;
    };
    hw.gpu = Some(info.name);
    // The Metal analogue of the CUDA SM row: which feature set this device
    // implements, which is what decides whether the backend can run on it at
    // all. Absent when the device supports no Apple family, which is a device
    // `check_gpu` refuses anyway.
    hw.gpu_family = info.apple_family.map(|family| format!("Apple{family}"));
    if info.working_set_bytes > 0 {
        // NAMED "unified" IN THE VALUE because the number is not a card's
        // capacity. Apple Silicon shares one pool with the CPU, so the RAM row
        // below is describing the same memory, and a reader adding the two
        // together would be double counting.
        hw.vram = Some(format!(
            "{:.1} GB (unified)",
            bytes_to_gib(info.working_set_bytes)
        ));
    }
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
fn probe_gpu_cuda(hw: &mut HardwareInfo) {
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
                hw.gpu_index = Some(selected.index as i64);
                // Index first, always: this row is the one place that names
                // the device the solver is on, and two cards of the same model
                // are told apart by nothing else.
                hw.gpu = Some(format!("{}: {}", selected.index, selected.name));
                if let Ok(mb) = selected.fields[0].parse::<u64>() {
                    hw.vram = Some(format!("{:.1} GB", mib_to_gib(mb)));
                }
                hw.sm = Some(format!("sm_{}", selected.fields[1].replace('.', "")));
            }
            // A selection that names no device leaves CUDA with nothing to run
            // on. Report why in the field the add-on shows, and pair it with
            // the -1 index the add-on reads as "resolved nothing", so the two
            // cannot disagree.
            Err(err) => {
                hw.gpu = Some(err.to_string());
                hw.gpu_index = Some(-1);
            }
        }
    }

    if let Some(out) = run_with_timeout(&mut Command::new("nvidia-smi"), PROBE_TIMEOUT) {
        for line in out.lines() {
            if line.contains("CUDA Version") {
                for part in line.split_whitespace() {
                    if part.parse::<f64>().is_ok() {
                        hw.cuda = Some(part.to_string());
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
    use super::{parse_runtime_gpu_usage, probe};
    use ppf_cts_core::utils::{backend, Backend};

    /// The reported rows are the ones that APPLY, and this is the regression
    /// gate for the defect that motivated it: on macOS the panel showed a row
    /// headed `CUDA` and four rows reading `Unknown`, because the probe asked
    /// nvidia-smi on every host and filled the rest with a placeholder.
    ///
    /// IT ASSERTS THE ABSENCES, NOT THE VALUES. What a GPU is called is a fact
    /// about the machine the test runs on, and CI runs this where nvidia-smi
    /// may be missing and where the Metal device is a paravirtual one, so
    /// asserting a name would fail for reasons that are not defects. Which rows
    /// exist is a property of the code.
    #[test]
    fn the_payload_omits_rows_that_do_not_apply_to_this_backend() {
        let json = serde_json::to_value(probe()).expect("HardwareInfo serializes");
        let map = json.as_object().expect("a JSON object");

        // Always present: the backend names itself, and the two host rows apply
        // on every backend even when their probe fails.
        assert_eq!(map["Backend"], backend().name());
        assert!(map.contains_key("CPU"), "CPU row missing: {map:?}");
        assert!(map.contains_key("RAM"), "RAM row missing: {map:?}");

        match backend() {
            // The GPU rows here depend on nvidia-smi being present, which a
            // build host need not have, so nothing is asserted about them.
            Backend::Cuda => {}
            // No CUDA concept exists on any of these, so none may report one. A
            // reader seeing `CUDA: Unknown` under a Metal server cannot tell
            // "there is no such thing here" from "the probe failed".
            Backend::Metal | Backend::Rocm | Backend::Cpu => {
                for cuda_only in ["CUDA", "SM", "GPU Index"] {
                    assert!(
                        !map.contains_key(cuda_only),
                        "{} names a CUDA-only row: {map:?}",
                        backend().name(),
                    );
                }
            }
        }

        // "GPU Family" is Metal's own row, the analogue of SM, so a CUDA build
        // must not name it either. Its PRESENCE is not asserted anywhere: a
        // paravirtual Metal device supports no Apple family at all, which is
        // exactly what CI runs on, so requiring the row would fail there for a
        // reason that is not a defect.
        if backend() != Backend::Metal {
            assert!(
                !map.contains_key("GPU Family"),
                "{} named a Metal-only row: {map:?}",
                backend().name(),
            );
        }

        // A CPU build describes no device at all, so a GPU row would be naming
        // hardware it does not use.
        if backend() == Backend::Cpu {
            for gpu_row in ["GPU", "VRAM", "GPU Family"] {
                assert!(
                    !map.contains_key(gpu_row),
                    "the CPU backend named a GPU row: {map:?}",
                );
            }
        }
    }

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
