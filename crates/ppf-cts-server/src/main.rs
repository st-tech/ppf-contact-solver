// File: crates/ppf-cts-server/src/main.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// ppf-cts-server binary. Parses CLI args (host/port/debug), builds an
// engine + default executor, runs the tokio accept loop until the
// process receives SIGINT/SIGTERM.
//
// The binary speaks the wire protocol advertised by
// `PROTOCOL_VERSION` (see lib.rs) and follows the `progress.log`
// startup contract the launcher polls for: it appends
// `SERVER_STARTING` on entry and `SERVER_READY` once the listener is
// bound. The launcher script
// (`blender_addon/core/effect_runner.py`) greps for `SERVER_READY`.

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use ppf_cts_server::{
    config::EngineConfig,
    serve::{bind_listener, serve_with_listener},
    DefaultExecutor, EffectExecutor, ServerEngine,
};

/// Composed version string shown by `--version` and at the top of
/// `--help`. Embeds the wire protocol and schema versions so the
/// launcher (or a human) can sanity-check compatibility from the CLI
/// without speaking the wire protocol. Built at first access since
/// `concat!` only accepts literal tokens, not `pub const` items from
/// other crates.
fn version_line() -> &'static str {
    use std::sync::OnceLock;
    static V: OnceLock<String> = OnceLock::new();
    V.get_or_init(|| {
        format!(
            "{} (protocol v{}, schema v{})",
            env!("CARGO_PKG_VERSION"),
            ppf_cts_server::PROTOCOL_VERSION,
            ppf_cts_formats::SCHEMA_VERSION,
        )
    })
}

/// ppf-cts-server: tokio-based Rust solver host.
#[derive(Debug, Parser)]
#[command(
    name = "ppf-cts-server",
    version = version_line(),
    about = "ppf-cts-server (tokio-based Rust solver host)",
)]
struct Args {
    /// Bind address.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Listen port.
    #[arg(long, default_value_t = 9090)]
    port: u16,

    /// Enable verbose logging.
    #[arg(long, default_value_t = false)]
    debug: bool,

    /// Readiness marker file. The launcher polls this for
    /// `SERVER_READY`; default is `progress.log` relative to CWD.
    #[arg(long, default_value = "progress.log")]
    progress_file: PathBuf,
}

/// Append a single line to the progress file. Best-effort: a write
/// failure must not abort startup, since the launcher will fall back
/// to its overall timeout anyway.
fn write_progress(path: &std::path::Path, line: &str) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(f, "{line}");
    }
}

/// Truncate the progress file at startup so a previous run's
/// `SERVER_READY` line can't fool the launcher into thinking we're
/// already up.
fn clear_progress(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let log_level = if args.debug {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Info
    };
    setup_logger(log_level);

    clear_progress(&args.progress_file);
    write_progress(&args.progress_file, "SERVER_STARTING");

    let addr: std::net::SocketAddr = format!("{}:{}", args.host, args.port).parse()?;

    // Harvest log channel `(name, filename)` pairs at startup. The map
    // lets `response::build_response` populate the `summary` and
    // `average_summary` fields the addon panel reads. Best-effort:
    // an unreadable tree contributes nothing.
    //
    // Walk locations: the `// Name:` / `logging.push("...")` markers
    // live in CUDA + Rust source. Order matters only for diagnostic
    // logging; the merged map is keyed by channel name and later
    // entries with the same key overwrite earlier ones (channel names
    // do not collide across files in practice). Probed roots, in order:
    //   1. `<root>/crates/ppf-cts-solver/src/`        (CUDA solver,
    //                                                  owns most
    //                                                  channels)
    //   2. `<root>/crates/ppf-cts-core/src/`          (Rust kernels)
    // `<root>` is resolved as: `PPF_CTS_LOG_SRC_DIR` env var first
    // (colon-separated absolute paths used directly), then cwd, then a
    // `target/release/` walk-up from the binary's own location. The
    // walk-up matters for the debug-rig orchestrator, which spawns the
    // server with `cwd=<per_worker_tmp_dir>` — there's no `src/` or
    // `crates/` next to cwd, but the binary still lives at
    // `<repo_root>/target/release/ppf-cts-server`, so its parent chain
    // points at the right tree.
    let mut config = EngineConfig::default();
    // Populate `hardware` with a real probe. Without this, every
    // field stays "Unknown" and the addon's "Remote Hardware" panel
    // shows nothing useful.
    config.hardware = ppf_cts_server::hardware::probe();
    log::info!(
        target: "ppf::serve",
        "hardware: GPU={} VRAM={} CUDA={} SM={} CPU={} RAM={}",
        config.hardware.gpu,
        config.hardware.vram,
        config.hardware.cuda,
        config.hardware.sm,
        config.hardware.cpu,
        config.hardware.ram,
    );
    if config.hardware.emulated {
        log::warn!(
            target: "ppf::serve",
            "MODE: EMULATED build (CPU stub backend, no CUDA). Simulations \
             will NOT produce real physics; this binary is for the test \
             rig only. Rebuild without `--features emulated` for real runs."
        );
    }
    let src_roots: Vec<std::path::PathBuf> = if let Ok(override_paths) =
        std::env::var("PPF_CTS_LOG_SRC_DIR")
    {
        override_paths
            .split(':')
            .filter(|s| !s.is_empty())
            .map(std::path::PathBuf::from)
            .collect()
    } else {
        let mut probe_bases: Vec<std::path::PathBuf> = Vec::new();
        if let Ok(cwd) = std::env::current_dir() {
            probe_bases.push(cwd);
        }
        // Walk up from the binary: `<root>/target/release/<bin>` ->
        // `<root>`. Three parents (file -> release -> target -> root).
        if let Ok(exe) = std::env::current_exe() {
            if let Some(root) = exe.parent().and_then(|p| p.parent()).and_then(|p| p.parent()) {
                let root = root.to_path_buf();
                if !probe_bases.contains(&root) {
                    probe_bases.push(root);
                }
            }
        }
        let mut roots: Vec<std::path::PathBuf> = Vec::new();
        for base in probe_bases {
            roots.push(base.join("crates").join("ppf-cts-solver").join("src"));
            roots.push(base.join("crates").join("ppf-cts-core").join("src"));
        }
        roots
    };
    let mut merged: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::new();
    for root in &src_roots {
        if !root.is_dir() {
            continue;
        }
        let docstrings = ppf_cts_core::parsers::get_logging_docstrings(root);
        let added = docstrings.len();
        for (name, entry) in docstrings {
            merged.insert(name, entry.filename);
        }
        log::info!(
            target: "ppf::serve",
            "log_filenames: harvested {} channels from {}",
            added,
            root.display(),
        );
    }
    config.log_filenames = merged.into_iter().collect();
    log::info!(
        target: "ppf::serve",
        "log_filenames: {} channels total",
        config.log_filenames.len(),
    );

    let engine = ServerEngine::new(config);
    let executor: Arc<dyn EffectExecutor> = Arc::new(DefaultExecutor::new());

    // Bind first, then publish SERVER_READY: the launcher polls this
    // file and dispatches ServerLaunched only after seeing the marker
    // *and* a successful TCP query, so writing it before bind would
    // race the polling loop in the docker-misconfig branch.
    let (listener, bound) = bind_listener(addr).await?;
    log::info!(target: "ppf::serve", "ppf-cts-server listening on {bound}");
    write_progress(&args.progress_file, "SERVER_READY");

    // Cancel on SIGINT or SIGTERM (Unix). Windows only delivers
    // Ctrl-C through `tokio::signal::ctrl_c()`; SIGTERM doesn't apply.
    let cancel = async {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM handler");
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    log::info!(target: "ppf::serve", "Ctrl-C received, shutting down");
                }
                _ = sigterm.recv() => {
                    log::info!(target: "ppf::serve", "SIGTERM received, shutting down");
                }
            }
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
            log::info!(target: "ppf::serve", "Ctrl-C received, shutting down");
        }
    };

    serve_with_listener(listener, engine, executor, cancel).await;
    Ok(())
}

fn setup_logger(level: log::LevelFilter) {
    use log4rs::append::console::ConsoleAppender;
    use log4rs::config::{Appender, Config, Root};
    use log4rs::encode::pattern::PatternEncoder;

    let console = ConsoleAppender::builder()
        .encoder(Box::new(PatternEncoder::new("{d(%Y-%m-%dT%H:%M:%S%.3f)} {l} {t} - {m}{n}")))
        .build();
    let cfg = Config::builder()
        .appender(Appender::builder().build("console", Box::new(console)))
        .build(Root::builder().appender("console").build(level));
    if let Ok(c) = cfg {
        let _ = log4rs::init_config(c);
    }
}
