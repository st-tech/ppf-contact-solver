// File: crates/ppf-cts-server/tests/common/mod.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Shared helpers for the in-process integration tests
// (wire_integration.rs, wire_data_paths.rs, monitor_integration.rs).
//
// Cargo treats `.rs` files at the top of `tests/` as separate test
// binaries, but ignores files in subdirectories of `tests/` unless
// they are referenced via `mod`. By placing this module at
// `tests/common/mod.rs` and adding `mod common;` at the top of each
// test file, the helpers are compiled once per test binary without
// generating a phantom test binary of their own.
//
// `binary_smoke.rs` is intentionally not a consumer: it spawns the
// real `ppf-cts-server` cargo binary as a child process and uses
// blocking std I/O, while the helpers below assume tokio + the
// in-process `serve_with_listener` entry point.
//
// `dead_code` is suppressed at the module level: each test binary
// only consumes a subset of the helpers, so unconditional warnings
// would fire on every build despite all helpers being used somewhere.

#![allow(dead_code)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use ppf_cts_server::config::EngineConfig;
use ppf_cts_server::serve::{bind_listener, serve_with_listener};
use ppf_cts_server::{DefaultExecutor, EffectExecutor, ServerEngine};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Spin up an in-process server on an ephemeral 127.0.0.1 port and
/// return its bound address plus a oneshot cancel handle. The caller
/// drops the cancel handle (or sends `()` on it) to stop the server.
///
/// The engine is configured with short monitor / startup grace
/// intervals so tests don't sleep on production timing constants.
///
/// Replaces the per-file `spawn_server()` copies in the integration
/// tests, including the 20 ms post-bind grace sleep that was a known
/// flake source. Instead of sleeping, this helper polls the freshly
/// bound port with `TcpStream::connect` until a connection succeeds
/// (or 500 ms elapses, which is well past the worst case observed in
/// CI).
pub async fn spawn_server() -> (std::net::SocketAddr, tokio::sync::oneshot::Sender<()>) {
    spawn_server_with_data_root(None).await
}

/// Variant of [`spawn_server`] that pins the engine's `data_root` to a
/// caller-supplied directory. Tests that need to read the uploaded
/// `data.pickle` / `param.pickle` off disk plant a tempdir here so
/// `make_root("p")` resolves under the tempdir instead of the real
/// `~/.local/share/ppf-cts/...` path. Per-test isolation, no env-var
/// races.
pub async fn spawn_server_with_data_root(
    data_root: Option<std::path::PathBuf>,
) -> (std::net::SocketAddr, tokio::sync::oneshot::Sender<()>) {
    let cfg = EngineConfig {
        monitor_interval_ms: 50,
        solver_startup_grace_ms: 50,
        data_root,
        ..Default::default()
    };
    let engine = ServerEngine::new(cfg);
    let executor: Arc<dyn EffectExecutor> = Arc::new(DefaultExecutor::new());
    let (listener, addr) = bind_listener("127.0.0.1:0".parse().unwrap()).await.unwrap();
    let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        serve_with_listener(listener, engine, executor, async {
            let _ = cancel_rx.await;
        })
        .await;
    });
    // Probe the socket instead of sleeping: poll connect() until it
    // succeeds, capped at 500 ms. The first probe usually wins because
    // bind_listener returned a fully bound socket; the loop just
    // covers the brief gap before serve_with_listener reaches its
    // accept() call.
    let deadline = Instant::now() + Duration::from_millis(500);
    loop {
        match TcpStream::connect(addr).await {
            Ok(_probe) => break,
            Err(_) if Instant::now() < deadline => {
                tokio::task::yield_now().await;
            }
            Err(e) => panic!("server never became connectable on {addr}: {e}"),
        }
    }
    (addr, cancel_tx)
}

/// Send a TCMD request on `stream` using the current wire: the
/// `b"TCMD"` header, a 4-byte big-endian payload-length prefix, then
/// the `--key value` argument bytes. Protocol 0.04+ length-prefixes
/// the payload instead of relying on a `shutdown(SHUT_WR)` half-close,
/// so the server reads exactly `args.len()` bytes and never blocks
/// waiting for EOF. The caller then drains the reply with
/// [`read_to_eof`].
pub async fn send_tcmd(stream: &mut TcpStream, args: &[u8]) {
    stream.write_all(b"TCMD").await.unwrap();
    stream
        .write_all(&(args.len() as u32).to_be_bytes())
        .await
        .unwrap();
    stream.write_all(args).await.unwrap();
    stream.flush().await.unwrap();
}

/// Drain `stream` to EOF with a 2 s timeout and return the bytes.
/// Used by tests that send a request and expect the server to reply
/// then close the connection.
pub async fn read_to_eof(stream: &mut TcpStream) -> Vec<u8> {
    let mut buf = Vec::with_capacity(2048);
    let _ = tokio::time::timeout(Duration::from_secs(2), stream.read_to_end(&mut buf)).await;
    buf
}

/// Error returned when `wait_until` exhausts its budget without the
/// predicate ever returning `true`. The `Display` impl includes the
/// elapsed budget so test failures are self-describing.
#[derive(Debug)]
pub struct TimeoutError {
    pub waited: Duration,
}

impl std::fmt::Display for TimeoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "wait_until: predicate never became true within {:?}",
            self.waited
        )
    }
}

impl std::error::Error for TimeoutError {}

/// Poll `predicate` every 10 ms until it returns `true`, or until
/// `timeout` elapses. Replaces the `for _ in 0..50 { sleep(10ms);
/// if pred { break; } }` boilerplate that was littered across
/// `monitor_integration.rs`.
///
/// The first probe runs before any sleep so a predicate that is
/// already true returns immediately.
pub async fn wait_until<F>(mut predicate: F, timeout: Duration) -> Result<(), TimeoutError>
where
    F: FnMut() -> bool,
{
    let deadline = Instant::now() + timeout;
    loop {
        if predicate() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(TimeoutError { waited: timeout });
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
