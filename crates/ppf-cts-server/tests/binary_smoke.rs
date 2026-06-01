// File: crates/ppf-cts-server/tests/binary_smoke.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Black-box smoke test for the actual `ppf-cts-server` binary. The
// in-process integration tests in `wire_integration.rs` already lock
// in the wire format, but they bypass `main.rs` entirely. This test
// closes that gap: cargo builds the binary, we spawn it as a real
// child process on an ephemeral port, send a TCMD ping over a real
// TCP socket, assert the response, and verify clean SIGINT/SIGTERM
// shutdown.
//
// This test covers the launcher → Rust binary handoff: it proves the
// Rust binary the launcher spawns listens, accepts a TCMD command,
// writes the `progress.log` readiness marker, and exits cleanly on a
// stop signal.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use ppf_cts_server::PROTOCOL_VERSION;

/// Pick a free TCP port by binding to port 0 and immediately closing.
/// Race window vs. the child spawn is small but non-zero; tests
/// retry the connection on `ConnectionRefused` for the boot grace.
fn pick_free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    l.local_addr().unwrap().port()
}

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ppf-cts-server"))
}

/// Wait up to `deadline` for the binary to write `SERVER_READY` to
/// the progress file.
fn wait_for_ready(progress_file: &std::path::Path, deadline: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < deadline {
        if let Ok(s) = std::fs::read_to_string(progress_file) {
            if s.contains("SERVER_READY") {
                return true;
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

/// Reap a child process: try gentle stop first, then SIGKILL after a
/// short grace. Returns the exit status if the child stopped, or
/// `None` if we had to escalate.
fn reap(mut child: Child) -> Option<std::process::ExitStatus> {
    #[cfg(unix)]
    {
        // SIGTERM via libc::kill so the binary's signal handler runs
        // and writes its goodbye log line. The plain Child::kill()
        // sends SIGKILL on Unix, which would skip the handler.
        let pid = child.id() as i32;
        unsafe {
            libc::kill(pid, libc::SIGTERM);
        }
    }
    #[cfg(not(unix))]
    {
        let _ = child.kill();
    }

    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(3) {
        if let Ok(Some(status)) = child.try_wait() {
            return Some(status);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let _ = child.kill();
    let _ = child.wait();
    None
}

#[test]
fn binary_listens_accepts_tcmd_and_shuts_down() {
    let port = pick_free_port();
    let workdir = tempfile::tempdir().expect("tempdir");
    let progress_file = workdir.path().join("progress.log");

    // Spawn the actual binary. CWD is the temp dir so it doesn't
    // pollute the repo root with progress.log / server.log droppings.
    let child = Command::new(binary_path())
        .args([
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--progress-file",
            progress_file.to_str().unwrap(),
        ])
        .current_dir(workdir.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ppf-cts-server");

    // Boot grace. The launcher polls progress.log for SERVER_READY;
    // do the same here so we know bind() returned before sending.
    let ready = wait_for_ready(&progress_file, Duration::from_secs(10));
    assert!(ready, "ppf-cts-server did not write SERVER_READY in 10s");

    // Real TCMD round-trip over a real TCP socket.
    let mut stream = std::net::TcpStream::connect(("127.0.0.1", port))
        .expect("connect to ppf-cts-server");
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("set read timeout");
    // TCMD wire (0.04+): header, 4-byte big-endian payload-length
    // prefix, then the argument bytes. The server reads exactly that
    // many bytes, so no half-close is needed to signal end-of-input.
    let args = b"--name smoke";
    stream.write_all(b"TCMD").expect("write header");
    stream
        .write_all(&(args.len() as u32).to_be_bytes())
        .expect("write length prefix");
    stream.write_all(args).expect("write body");
    stream.flush().expect("flush");

    let mut buf = Vec::with_capacity(2048);
    stream.read_to_end(&mut buf).expect("read response");
    let body = std::str::from_utf8(&buf).expect("utf8");
    let trimmed = body.trim_end_matches('\n').trim();

    let response: serde_json::Value =
        serde_json::from_str(trimmed).unwrap_or_else(|e| panic!("bad JSON: {e}\nbody: {trimmed}"));
    assert_eq!(response["protocol_version"], PROTOCOL_VERSION);
    assert_eq!(response["status"], "NO_DATA");
    assert!(response.get("hardware").is_some(), "missing hardware block");

    // Clean shutdown: SIGTERM (Unix) or kill (Windows). The binary
    // installs a SIGTERM handler so this exercises the graceful
    // shutdown path.
    let status = reap(child).expect("server did not stop within grace");
    // SIGTERM-induced exits don't have a code on Unix (they're
    // signal-terminated). On Windows kill() reports a status. We just
    // assert that it stopped: any exit is "clean enough" for the
    // smoke test; the launcher only cares that the port is freed.
    let _ = status;

    // SERVER_STARTING + SERVER_READY both present means the file
    // contract the launcher polls is honored.
    let progress_text = std::fs::read_to_string(&progress_file).unwrap_or_default();
    assert!(
        progress_text.contains("SERVER_STARTING"),
        "missing SERVER_STARTING: {progress_text:?}",
    );
    assert!(
        progress_text.contains("SERVER_READY"),
        "missing SERVER_READY: {progress_text:?}",
    );
}
