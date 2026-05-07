// File: crates/ppf-cts-server/src/serve.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Tokio TCP accept loop. Spawned per-connection handler runs in
// `wire::handle_connection`. The monitor task is bootstrapped here
// too so a single `serve()` call is everything an embedder needs.

use std::sync::Arc;

use tokio::net::TcpListener;

use crate::engine::ServerEngine;
use crate::executor::EffectExecutor;
use crate::monitor::spawn_monitor;
use crate::wire;

/// Bind the listener and run the accept loop until cancelled. The
/// `cancel` future fires when the host wants graceful shutdown
/// (Ctrl-C, signal handler).
///
/// On bind failure, returns `Err`. On accept failure, logs and
/// continues; a flap on accept shouldn't kill the whole server.
pub async fn serve(
    addr: std::net::SocketAddr,
    engine: ServerEngine,
    executor: Arc<dyn EffectExecutor>,
    cancel: impl std::future::Future<Output = ()>,
) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?;
    log::info!(target: "ppf::serve", "ppf-cts-server listening on {bound}");

    // Wire the executor onto the engine so re-entrant effect
    // handlers (build pipeline emitting `BuildCompleted`, solver
    // helpers emitting `ErrorOccurred`, etc.) dispatch through this
    // executor instead of fabricating a stateless `DefaultExecutor`.
    engine.attach_executor(&executor);
    let _monitor = spawn_monitor(engine.clone(), executor.clone());

    tokio::pin!(cancel);
    loop {
        tokio::select! {
            _ = &mut cancel => {
                log::info!(target: "ppf::serve", "shutdown requested, draining accept loop");
                return Ok(());
            }
            res = listener.accept() => {
                match res {
                    Ok((stream, peer)) => {
                        let engine = engine.clone();
                        let executor = executor.clone();
                        tokio::spawn(async move {
                            wire::handle_connection(stream, peer, engine, executor).await;
                        });
                    }
                    Err(e) => {
                        log::warn!(target: "ppf::serve", "accept failed: {e}");
                        // Brief backoff so we don't busy-loop on
                        // hard failures (e.g. fd exhaustion).
                        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    }
                }
            }
        }
    }
}

/// Bind on `addr` and return the actual bound address; handy for
/// tests using `0.0.0.0:0` to pick an ephemeral port.
pub async fn bind_listener(
    addr: std::net::SocketAddr,
) -> std::io::Result<(TcpListener, std::net::SocketAddr)> {
    let listener = TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?;
    Ok((listener, bound))
}

/// Variant of `serve` that takes an already-bound listener. Tests
/// use this so they can capture the ephemeral port before driving
/// the loop.
pub async fn serve_with_listener(
    listener: TcpListener,
    engine: ServerEngine,
    executor: Arc<dyn EffectExecutor>,
    cancel: impl std::future::Future<Output = ()>,
) {
    engine.attach_executor(&executor);
    let _monitor = spawn_monitor(engine.clone(), executor.clone());
    tokio::pin!(cancel);
    loop {
        tokio::select! {
            _ = &mut cancel => return,
            res = listener.accept() => {
                match res {
                    Ok((stream, peer)) => {
                        let engine = engine.clone();
                        let executor = executor.clone();
                        tokio::spawn(async move {
                            wire::handle_connection(stream, peer, engine, executor).await;
                        });
                    }
                    Err(e) => {
                        log::warn!(target: "ppf::serve", "accept failed: {e}");
                        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    }
                }
            }
        }
    }
}
