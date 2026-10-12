// File: crates/ppf-cts-server/src/serve.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Tokio TCP accept loop. Spawned per-connection handler runs in
// `wire::handle_connection`. The monitor task is bootstrapped here
// too. Embedders bind with `bind_listener` and then drive the loop
// with `serve_with_listener`.

use std::sync::Arc;

use tokio::net::TcpListener;

use crate::engine::ServerEngine;
use crate::executor::EffectExecutor;
use crate::monitor::spawn_monitor;
use crate::wire;

/// Clear `HANDLE_FLAG_INHERIT` on the listener's socket handle.
///
/// Without this, every child process the server spawns (notably
/// `ppf-contact-solver.exe` from the solver launcher) inherits the
/// listen socket on port 9090: Rust's `Command::spawn` defaults to
/// `bInheritHandles=TRUE` on Windows and tokio's `TcpListener`
/// doesn't mark sockets non-inheritable. If the parent dies while a
/// child is still alive, the kernel keeps the port bound -- netstat
/// shows it as `LISTENING` against a non-existent PID, the next
/// Connect probes the address and times out (the child isn't running
/// an accept loop), and the addon surfaces `Port 9090 is in use`.
///
/// SOCKET handles are valid HANDLEs for `SetHandleInformation`, per
/// MSDN's "Windows Sockets" notes. No-op on non-Windows.
#[cfg(windows)]
fn mark_listener_no_inherit(listener: &TcpListener) -> std::io::Result<()> {
    use std::os::raw::c_void;
    use std::os::windows::io::AsRawSocket;

    type Handle = *mut c_void;
    const HANDLE_FLAG_INHERIT: u32 = 0x0000_0001;
    extern "system" {
        fn SetHandleInformation(h: Handle, dwMask: u32, dwFlags: u32) -> i32;
    }

    let raw = listener.as_raw_socket();
    let ok = unsafe { SetHandleInformation(raw as Handle, HANDLE_FLAG_INHERIT, 0) };
    if ok == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(windows))]
fn mark_listener_no_inherit(_listener: &TcpListener) -> std::io::Result<()> {
    Ok(())
}

/// Bind on `addr` and return the actual bound address; handy for
/// tests using `0.0.0.0:0` to pick an ephemeral port.
pub async fn bind_listener(
    addr: std::net::SocketAddr,
) -> std::io::Result<(TcpListener, std::net::SocketAddr)> {
    let listener = TcpListener::bind(addr).await?;
    mark_listener_no_inherit(&listener)?;
    let bound = listener.local_addr()?;
    Ok((listener, bound))
}

/// Run the accept loop on an already-bound listener until cancelled.
/// The `cancel` future fires when the host wants shutdown (Ctrl-C,
/// signal handler). Tests use this so they can capture the ephemeral
/// port via `bind_listener` before driving the loop.
///
/// On accept failure, logs and continues; a flap on accept shouldn't
/// kill the whole server.
pub async fn serve_with_listener(
    listener: TcpListener,
    engine: ServerEngine,
    executor: Arc<dyn EffectExecutor>,
    cancel: impl std::future::Future<Output = ()>,
) {
    engine.attach_executor(&executor);
    let accept_backoff_ms = engine.config().accept_backoff_ms;
    let _monitor = spawn_monitor(engine.clone(), executor.clone());
    tokio::pin!(cancel);
    loop {
        tokio::select! {
            _ = &mut cancel => {
                // Stop accepting. In-flight `handle_connection` tasks are
                // detached and get aborted when the runtime drops; we do
                // not wait on them here, so this is not a graceful drain.
                // Detached solver/build child processes survive (they run
                // in their own process group).
                log::info!(target: "ppf::serve", "shutdown requested; stopping accept loop, in-flight connections will be aborted on runtime shutdown");
                return;
            }
            res = listener.accept() => {
                match res {
                    Ok((stream, peer)) => {
                        let engine = engine.clone();
                        let executor = executor.clone();
                        // Taken here rather than in the task, so a request is
                        // counted from the moment it is accepted.
                        let request = engine.request_started();
                        tokio::spawn(async move {
                            let _request = request;
                            wire::handle_connection(stream, peer, engine, executor).await;
                        });
                    }
                    Err(e) => {
                        log::warn!(target: "ppf::serve", "accept failed: {e}");
                        tokio::time::sleep(std::time::Duration::from_millis(accept_backoff_ms)).await;
                    }
                }
            }
        }
    }
}

/// Resolve once the server has gone `timeout` with no request open or
/// arriving and no solve or build in progress, so that the host can stop it.
///
/// The idle time counts from when the last request ended, and work in
/// progress restarts it, so a solve that runs for hours with no client
/// attached is never ended, and the server stops `timeout` after it finishes.
/// The add-on pings a server it is connected to (a status query every few
/// seconds), so a Blender that is connected and idle keeps its server up, and
/// one that closed or disconnected lets it go.
pub async fn stop_when_idle(engine: ServerEngine, timeout: std::time::Duration) {
    // While a request is open or work is running, look again this often.
    let recheck = timeout.min(std::time::Duration::from_secs(5));
    loop {
        let Some(idle) = engine.idle_for() else {
            tokio::time::sleep(recheck).await;
            continue;
        };
        if idle < timeout {
            tokio::time::sleep(timeout - idle).await;
            continue;
        }
        if engine.work_in_progress() {
            engine.mark_busy();
            tokio::time::sleep(recheck).await;
            continue;
        }
        log::info!(
            target: "ppf::serve",
            "idle stop: no request for {} s and no solve or build running; stopping",
            timeout.as_secs_f64(),
        );
        return;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EngineConfig;
    use ppf_cts_core::events::Event;
    use std::time::Duration;
    use tokio::time::{advance, Instant};

    const TIMEOUT: Duration = Duration::from_secs(3);

    fn engine() -> ServerEngine {
        ServerEngine::new(EngineConfig::default())
    }

    /// Let spawned tasks run at the current (paused) instant.
    async fn settle() {
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
    }

    #[tokio::test(start_paused = true)]
    async fn an_untouched_server_stops_at_the_timeout() {
        let start = Instant::now();
        stop_when_idle(engine(), TIMEOUT).await;
        assert_eq!(start.elapsed(), TIMEOUT);
    }

    #[tokio::test(start_paused = true)]
    async fn a_request_restarts_the_clock_when_it_ends() {
        let engine = engine();
        let stop = tokio::spawn(stop_when_idle(engine.clone(), TIMEOUT));
        advance(Duration::from_secs(2)).await;
        let request = engine.request_started();
        // Open far past the timeout: an open request is a client.
        advance(Duration::from_secs(30)).await;
        settle().await;
        assert!(!stop.is_finished(), "stopped with a request open");
        drop(request);
        let ended = Instant::now();
        advance(Duration::from_millis(2_900)).await;
        settle().await;
        assert!(!stop.is_finished(), "stopped before the timeout after the request ended");
        stop.await.unwrap();
        let waited = ended.elapsed();
        assert!(waited >= TIMEOUT, "stopped {waited:?} after the request ended");
    }

    #[tokio::test(start_paused = true)]
    async fn a_running_solve_holds_the_server_up() {
        let engine = engine();
        engine.set_project_context("p", "/tmp/p");
        engine.dispatch(Event::ExternalSolverAdopted { has_data_on_disk: true });
        assert!(engine.work_in_progress(), "the adopted solve should read as running");
        let stop = tokio::spawn(stop_when_idle(engine.clone(), TIMEOUT));
        advance(Duration::from_secs(600)).await;
        settle().await;
        assert!(!stop.is_finished(), "stopped under a running solve");
        engine.dispatch(Event::SolverFinished { resumable: false });
        assert!(!engine.work_in_progress());
        let finished = Instant::now();
        stop.await.unwrap();
        // Counted from the last time the solve was seen running, which is at
        // most one recheck before it finished.
        let waited = finished.elapsed();
        assert!(waited <= TIMEOUT, "stopped {waited:?} after the solve");
    }

    #[tokio::test(start_paused = true)]
    async fn a_build_in_progress_holds_the_server_up() {
        let engine = engine();
        engine.set_project_context("p", "/tmp/p");
        engine.dispatch(Event::upload_landed("uid"));
        engine.dispatch(Event::BuildRequested { preserve_output: false });
        assert!(engine.work_in_progress(), "the build should read as in progress");
        let stop = tokio::spawn(stop_when_idle(engine.clone(), TIMEOUT));
        advance(Duration::from_secs(600)).await;
        settle().await;
        assert!(!stop.is_finished(), "stopped under a build");
        engine.dispatch(Event::BuildCompleted { exemptions: Vec::new() });
        assert!(!engine.work_in_progress());
        stop.await.unwrap();
    }
}
