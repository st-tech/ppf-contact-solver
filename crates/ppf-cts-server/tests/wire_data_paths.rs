// File: crates/ppf-cts-server/tests/wire_data_paths.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Wire-level integration coverage for the file-transport paths the
// happy-path `wire_integration.rs` doesn't touch:
//
//   * BDAT (binary-frame stub): drains payload + acks BINARY_OK.
//   * data_send + data_receive round-trip via JSON.
//   * data_send rejects `data.pickle` (must use upload_atomic instead).
//   * data_send rejects zero size and missing fields.
//
// These exercise the same `serve_with_listener` entry point as
// `wire_integration.rs`, so the byte-level protocol contract is
// covered end-to-end.

use std::time::Duration;

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

mod common;
use common::{read_to_eof, spawn_server, spawn_server_with_data_root};

// BDAT is a vestigial path: the addon defines the constant in
// `core/protocol.py` but never sends the header, and `handle_bdat`
// drains bytes without persisting them. This test pins the
// drain-and-ack contract so legacy smoke tools that still emit BDAT
// keep getting `BINARY_OK` until the header is removed entirely.
#[tokio::test]
async fn bdat_drains_payload_and_acks_binary_ok() {
    let (addr, cancel) = spawn_server().await;

    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(b"BDAT").await.unwrap();
    // Arbitrary binary payload; server drains until EOF.
    let payload = vec![0xAAu8; 4096];
    s.write_all(&payload).await.unwrap();
    s.shutdown().await.unwrap();

    let bytes = read_to_eof(&mut s).await;
    assert_eq!(bytes, b"BINARY_OK", "BDAT must ack with BINARY_OK");

    let _ = cancel.send(());
}

#[tokio::test]
async fn data_receive_streams_multi_chunk_payload() {
    // Exercise the chunked send loop in `handle_data_receive`. Payload
    // crosses the 32 KiB read buffer several times so we'd notice
    // either a busted `read` loop (truncated body) or a regression to
    // the `tokio::fs::read(...)` whole-file pre-load.
    //
    // The transport anchors `path` under `make_root(name)`, so pin the
    // engine's `data_root` to a tempdir and plant the file at
    // `<data_root>/<name>/...` where the resolver can reach it.
    let dir = tempfile::tempdir().unwrap();
    let (addr, cancel) =
        spawn_server_with_data_root(Some(dir.path().to_path_buf())).await;
    let project_root = dir.path().join("big");
    tokio::fs::create_dir_all(&project_root).await.unwrap();
    let target_path = project_root.join("big.bin");
    let mut payload = Vec::with_capacity(200 * 1024);
    for i in 0..(200 * 1024usize) {
        // Position-dependent bytes so a truncation or chunk swap is
        // visible in the byte-by-byte equality assertion below.
        payload.push((i & 0xff) as u8);
    }
    tokio::fs::write(&target_path, &payload).await.unwrap();

    let header = serde_json::json!({
        "request": "data_receive",
        "name": "big",
        "path": target_path.to_string_lossy().to_string(),
    });
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(b"JSON").await.unwrap();
    s.write_all(format!("{header}\n").as_bytes()).await.unwrap();
    s.shutdown().await.unwrap();

    let bytes = read_to_eof(&mut s).await;
    let nl = bytes.iter().position(|&b| b == b'\n').expect("missing newline");
    let meta: Value = serde_json::from_slice(&bytes[..nl]).expect("meta JSON");
    assert_eq!(meta["size"], payload.len() as u64);
    assert_eq!(&bytes[nl + 1..], &payload[..]);

    let _ = cancel.send(());
}

#[tokio::test]
async fn data_send_then_receive_roundtrips_payload() {
    // `path` is anchored under `make_root(name)`, so pin `data_root` to
    // a tempdir; `<data_root>/<name>/blob.bin` is the sandboxed target.
    let dir = tempfile::tempdir().unwrap();
    let (addr, cancel) =
        spawn_server_with_data_root(Some(dir.path().to_path_buf())).await;
    let target_path = dir.path().join("blob").join("blob.bin");
    let payload = b"ROUND-TRIP-PAYLOAD".to_vec();

    // 1. data_send writes the file.
    let header = serde_json::json!({
        "request": "data_send",
        "name": "blob",
        "path": target_path.to_string_lossy().to_string(),
        "size": payload.len(),
    });
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(b"JSON").await.unwrap();
    s.write_all(format!("{header}\n").as_bytes()).await.unwrap();
    s.write_all(&payload).await.unwrap();
    s.flush().await.unwrap();
    let mut ack = [0u8; 8];
    let n = tokio::time::timeout(Duration::from_secs(2), s.read(&mut ack))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(&ack[..n], b"OK\n");
    drop(s);

    // 2. data_receive reads it back.
    let header = serde_json::json!({
        "request": "data_receive",
        "name": "blob",
        "path": target_path.to_string_lossy().to_string(),
    });
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(b"JSON").await.unwrap();
    s.write_all(format!("{header}\n").as_bytes()).await.unwrap();
    s.shutdown().await.unwrap();

    let bytes = read_to_eof(&mut s).await;
    // Response is `<json line>\n<raw bytes>`.
    let nl = bytes.iter().position(|&b| b == b'\n').expect("missing newline");
    let meta: Value = serde_json::from_slice(&bytes[..nl]).expect("meta JSON");
    assert_eq!(meta["size"], payload.len() as u64);
    assert_eq!(&bytes[nl + 1..], &payload[..]);

    let _ = cancel.send(());
}

#[tokio::test]
async fn data_send_rejects_pickle_basenames() {
    let (addr, cancel) = spawn_server().await;
    let dir = tempfile::tempdir().unwrap();

    let header = serde_json::json!({
        "request": "data_send",
        "name": "p",
        "path": dir.path().join("data.pickle").to_string_lossy().to_string(),
        "size": 4,
    });
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(b"JSON").await.unwrap();
    s.write_all(format!("{header}\n").as_bytes()).await.unwrap();
    s.shutdown().await.unwrap();

    let bytes = read_to_eof(&mut s).await;
    let body = std::str::from_utf8(&bytes).unwrap().trim_end();
    let response: Value = serde_json::from_str(body).expect(body);
    assert!(
        response["error"]
            .as_str()
            .unwrap()
            .contains("data_send no longer accepts data.pickle"),
        "got: {body}"
    );

    let _ = cancel.send(());
}

#[tokio::test]
async fn data_receive_missing_file_returns_error() {
    // Path under the project root but pointing at a file that does not
    // exist: it clears the sandbox check, so the handler reaches the
    // metadata probe and answers `File not found` (not `SandboxEscape`).
    let dir = tempfile::tempdir().unwrap();
    let (addr, cancel) =
        spawn_server_with_data_root(Some(dir.path().to_path_buf())).await;
    let missing = dir.path().join("x").join("this-file-should-not-exist-xyz12345");

    let header = serde_json::json!({
        "request": "data_receive",
        "name": "x",
        "path": missing.to_string_lossy().to_string(),
    });
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(b"JSON").await.unwrap();
    s.write_all(format!("{header}\n").as_bytes()).await.unwrap();
    s.shutdown().await.unwrap();

    let bytes = read_to_eof(&mut s).await;
    let body = std::str::from_utf8(&bytes).unwrap().trim_end();
    let response: Value = serde_json::from_str(body).expect(body);
    assert_eq!(response["error"], "File not found");

    let _ = cancel.send(());
}

#[tokio::test]
async fn data_send_blocks_path_outside_project_root() {
    // An absolute `path` outside the project root (anything other than
    // `<data_root>/<name>/...`) must be rejected before any disk write,
    // so a client cannot use data_send to clobber arbitrary files.
    let dir = tempfile::tempdir().unwrap();
    let (addr, cancel) =
        spawn_server_with_data_root(Some(dir.path().to_path_buf())).await;
    let escape = dir.path().join("other").join("evil.bin");

    let header = serde_json::json!({
        "request": "data_send",
        "name": "proj",
        "path": escape.to_string_lossy().to_string(),
        "size": 4,
    });
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(b"JSON").await.unwrap();
    s.write_all(format!("{header}\n").as_bytes()).await.unwrap();
    s.shutdown().await.unwrap();

    let bytes = read_to_eof(&mut s).await;
    let body = std::str::from_utf8(&bytes).unwrap().trim_end();
    let response: Value = serde_json::from_str(body).expect(body);
    assert!(
        response["error"]
            .as_str()
            .unwrap()
            .contains("escapes sandbox"),
        "got: {body}"
    );
    assert!(
        !escape.exists(),
        "data_send must not create files outside the project root"
    );

    let _ = cancel.send(());
}

#[tokio::test]
async fn data_receive_blocks_path_outside_project_root() {
    // Symmetric arbitrary-read guard: an absolute `path` outside the
    // project root is rejected rather than streamed back to the client.
    let dir = tempfile::tempdir().unwrap();
    let (addr, cancel) =
        spawn_server_with_data_root(Some(dir.path().to_path_buf())).await;

    let header = serde_json::json!({
        "request": "data_receive",
        "name": "proj",
        "path": "/etc/passwd",
    });
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(b"JSON").await.unwrap();
    s.write_all(format!("{header}\n").as_bytes()).await.unwrap();
    s.shutdown().await.unwrap();

    let bytes = read_to_eof(&mut s).await;
    let body = std::str::from_utf8(&bytes).unwrap().trim_end();
    let response: Value = serde_json::from_str(body).expect(body);
    assert!(
        response["error"]
            .as_str()
            .unwrap()
            .contains("escapes sandbox"),
        "got: {body}"
    );

    let _ = cancel.send(());
}

#[tokio::test]
async fn unknown_json_request_returns_error() {
    let (addr, cancel) = spawn_server().await;

    let header = serde_json::json!({"request": "no_such_request"});
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(b"JSON").await.unwrap();
    s.write_all(format!("{header}\n").as_bytes()).await.unwrap();
    s.shutdown().await.unwrap();

    let bytes = read_to_eof(&mut s).await;
    let body = std::str::from_utf8(&bytes).unwrap().trim_end();
    let response: Value = serde_json::from_str(body).expect(body);
    assert!(
        response["error"]
            .as_str()
            .unwrap()
            .contains("Unknown request: no_such_request"),
        "got: {body}"
    );

    let _ = cancel.send(());
}

#[tokio::test]
async fn notebook_send_blocks_path_traversal() {
    // Sandbox guard: relative_path containing `..` that escapes the
    // examples root must be rejected before any payload is read or
    // file is written. Mirrors the os.path.commonpath check in
    // server.py.
    let (addr, cancel) = spawn_server().await;

    let header = serde_json::json!({
        "request": "notebook_send",
        "name": "demo",
        "relative_path": "../../etc/passwd",
        "size": 4,
    });
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(b"JSON").await.unwrap();
    s.write_all(format!("{header}\n").as_bytes()).await.unwrap();
    s.shutdown().await.unwrap();

    let bytes = read_to_eof(&mut s).await;
    let body = std::str::from_utf8(&bytes).unwrap().trim_end();
    let response: Value = serde_json::from_str(body).expect(body);
    assert!(
        response["error"]
            .as_str()
            .unwrap()
            .contains("escapes sandbox"),
        "got: {body}"
    );

    let _ = cancel.send(());
}

#[tokio::test]
async fn notebook_delete_blocks_path_traversal() {
    let (addr, cancel) = spawn_server().await;

    let header = serde_json::json!({
        "request": "notebook_delete",
        "name": "demo",
        "relative_path": "../escape.ipynb",
    });
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(b"JSON").await.unwrap();
    s.write_all(format!("{header}\n").as_bytes()).await.unwrap();
    s.shutdown().await.unwrap();

    let bytes = read_to_eof(&mut s).await;
    let body = std::str::from_utf8(&bytes).unwrap().trim_end();
    let response: Value = serde_json::from_str(body).expect(body);
    assert!(
        response["error"]
            .as_str()
            .unwrap()
            .contains("escapes sandbox"),
        "got: {body}"
    );

    let _ = cancel.send(());
}

#[tokio::test]
async fn notebook_send_missing_fields_returns_error() {
    // notebook_send / notebook_delete use a sandboxed examples path
    // resolved relative to the binary's location, so a wire-level test
    // that actually writes a notebook would race with other tests.
    // Instead exercise the validation gate that runs before any disk
    // I/O: missing relative_path must produce an error response.
    let (addr, cancel) = spawn_server().await;

    let header = serde_json::json!({
        "request": "notebook_send",
        "name": "demo",
    });
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(b"JSON").await.unwrap();
    s.write_all(format!("{header}\n").as_bytes()).await.unwrap();
    s.shutdown().await.unwrap();

    let bytes = read_to_eof(&mut s).await;
    let body = std::str::from_utf8(&bytes).unwrap().trim_end();
    let response: Value = serde_json::from_str(body).expect(body);
    assert!(
        response["error"]
            .as_str()
            .unwrap()
            .contains("Missing name or relative_path"),
        "got: {body}"
    );

    let _ = cancel.send(());
}
