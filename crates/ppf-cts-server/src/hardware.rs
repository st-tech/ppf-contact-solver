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
/// with a stuck driver, `run_with_timeout` kills the child after 5s.
pub fn runtime_usage() -> RuntimeUsage {
    let mut out = RuntimeUsage::default();

    if let Some(s) = run_with_timeout(
        Command::new("nvidia-smi").args([
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]),
        Duration::from_secs(5),
    ) {
        if let Some(line) = s.lines().next() {
            let parts: Vec<&str> = line.split(',').map(str::trim).collect();
            if parts.len() == 3 {
                out.gpu_util = Some(format!("{}%", parts[0]));
                if let (Ok(used), Ok(total)) =
                    (parts[1].parse::<u64>(), parts[2].parse::<u64>())
                {
                    let pct = if total > 0 {
                        (100.0 * used as f64 / total as f64).round() as u64
                    } else {
                        0
                    };
                    out.vram_usage = Some(format!(
                        "{}% ({:.1}/{:.1} GB)",
                        pct,
                        used as f64 / 1024.0,
                        total as f64 / 1024.0
                    ));
                }
            }
        }
    }

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
        out.ram_usage = Some(format!(
            "{}% ({:.1}/{:.1} GB)",
            pct,
            used as f64 / 1024_f64.powi(3),
            total as f64 / 1024_f64.powi(3),
        ));
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

    hw
}

/// `nvidia-smi --query-gpu=name,memory.total,compute_cap` parses the
/// CSV row for GPU/VRAM/SM, then a plain `nvidia-smi` for the CUDA
/// driver line. Both calls run under a 5-second `run_with_timeout`
/// so a stuck driver can't block startup.
fn probe_gpu(hw: &mut HardwareInfo) {
    if let Some(out) = run_with_timeout(
        Command::new("nvidia-smi").args([
            "--query-gpu=name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]),
        Duration::from_secs(5),
    ) {
        if let Some(line) = out.lines().next() {
            let parts: Vec<&str> = line.split(',').map(str::trim).collect();
            if parts.len() == 3 {
                hw.gpu = parts[0].to_string();
                if let Ok(mb) = parts[1].parse::<u64>() {
                    hw.vram = format!("{:.1} GB", mb as f64 / 1024.0);
                }
                hw.sm = format!("sm_{}", parts[2].replace('.', ""));
            }
        }
    }

    if let Some(out) = run_with_timeout(&mut Command::new("nvidia-smi"), Duration::from_secs(5)) {
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

    if let Some(cpu) = sys.cpus().first() {
        let brand = cpu.brand().trim().to_string();
        if !brand.is_empty() {
            hw.cpu = brand;
        }
    }
    if hw.cpu == "Unknown" {
        if cfg!(target_os = "windows") {
            if let Some(out) = run_with_timeout(
                Command::new("powershell").args([
                    "-Command",
                    "(Get-CimInstance Win32_Processor).Name",
                ]),
                Duration::from_secs(5),
            ) {
                let trimmed = out.trim().to_string();
                if !trimmed.is_empty() {
                    hw.cpu = trimmed;
                }
            }
        } else if let Some(out) =
            run_with_timeout(&mut Command::new("lscpu"), Duration::from_secs(5))
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
        hw.ram = format!("{:.1} GB", total_bytes as f64 / 1024_f64.powi(3));
    }
}

/// `Command::output()` blocks indefinitely; nvidia-smi can hang when
/// the driver is in a bad state, so we cap each probe at `timeout`.
/// On timeout we kill the child and return `None`; the caller leaves
/// the corresponding HardwareInfo field as "Unknown".
fn run_with_timeout(cmd: &mut Command, timeout: Duration) -> Option<String> {
    let mut child = cmd
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;

    let start = std::time::Instant::now();
    loop {
        match child.try_wait().ok()? {
            Some(status) if status.success() => {
                let mut buf = String::new();
                if let Some(mut stdout) = child.stdout.take() {
                    use std::io::Read;
                    let _ = stdout.read_to_string(&mut buf);
                }
                return Some(buf);
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
