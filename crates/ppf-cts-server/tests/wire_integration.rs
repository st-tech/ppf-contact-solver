// File: crates/ppf-cts-server/tests/wire_integration.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// End-to-end test for the wire protocol. Spawns the real
// tokio server on an ephemeral port, opens TcpStreams from the test,
// sends raw protocol bytes (TCMD / JSON / upload_atomic), and asserts
// the response shape + on-disk side effects.
//
// This is the closest analog to a Blender-addon ↔ server interaction
// we can run from cargo test, so it locks in protocol_version 0.03
// byte compatibility.

use std::time::Duration;

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

mod common;
use common::{read_to_eof, spawn_server, spawn_server_with_data_root};

#[tokio::test]
async fn tcmd_ping_returns_status_response() {
    let (addr, cancel) = spawn_server().await;

    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(b"TCMD").await.unwrap();
    s.write_all(b"--name demo").await.unwrap();
    s.shutdown().await.unwrap();

    let bytes = read_to_eof(&mut s).await;
    let body = std::str::from_utf8(&bytes).unwrap().trim_end();
    let response: Value = serde_json::from_str(body).unwrap_or_else(|e| {
        panic!("invalid JSON: {e}\n--- raw ---\n{body}");
    });

    assert_eq!(response["protocol_version"], "0.03");
    // No data uploaded yet, so status must be NO_DATA.
    assert_eq!(response["status"], "NO_DATA");
    assert_eq!(response["data"], "NO_DATA");
    // protocol_version + hardware + git_branch always present.
    assert!(response.get("hardware").is_some());
    assert!(response.get("git_branch").is_some());

    let _ = cancel.send(());
}

#[tokio::test]
async fn tcmd_no_id_returns_error() {
    let (addr, cancel) = spawn_server().await;

    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(b"TCMD").await.unwrap();
    // No --name → server returns NO_ID error.
    s.write_all(b"--request build").await.unwrap();
    s.shutdown().await.unwrap();

    let bytes = read_to_eof(&mut s).await;
    let body = std::str::from_utf8(&bytes).unwrap().trim_end();
    let response: Value = serde_json::from_str(body).expect(body);
    assert_eq!(response["error"], "NO_ID");
    assert_eq!(response["protocol_version"], "0.03");

    let _ = cancel.send(());
}

#[tokio::test]
async fn unknown_header_returns_error_json() {
    let (addr, cancel) = spawn_server().await;

    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(b"XXXX").await.unwrap();
    s.shutdown().await.unwrap();

    let bytes = read_to_eof(&mut s).await;
    let body = std::str::from_utf8(&bytes).unwrap().trim_end();
    let response: Value = serde_json::from_str(body).expect(body);
    let err = response["error"].as_str().unwrap();
    assert!(err.contains("Unknown header"), "got: {err}");

    let _ = cancel.send(());
}

#[tokio::test]
async fn upload_atomic_lands_data_and_param_files() {
    let dir = tempfile::tempdir().unwrap();
    let (addr, cancel) = spawn_server_with_data_root(Some(dir.path().to_path_buf())).await;
    // Server resolves the project root via make_root: it's
    // `<data_root>/git-<branch>/p`. The branch is whatever the test
    // process detects, which inside `cargo test` is "unknown" (no
    // branch_name.txt + the harness's cwd is not the repo). Glob
    // for the single git-* subdir under the tempdir to stay robust
    // either way.
    let data_payload = b"DATA-PICKLE-BYTES".to_vec();
    let param_payload = b"PARAM-PICKLE-BYTES-MORE".to_vec();

    let header = serde_json::json!({
        "request": "upload_atomic",
        "name": "p",
        "data_size": data_payload.len(),
        "param_size": param_payload.len(),
        "data_hash": "deadbeef",
        "param_hash": "feedface",
    });
    let header_bytes = format!("{header}\n");

    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(b"JSON").await.unwrap();
    s.write_all(header_bytes.as_bytes()).await.unwrap();
    s.write_all(&data_payload).await.unwrap();
    s.write_all(&param_payload).await.unwrap();
    s.flush().await.unwrap();

    let mut buf = [0u8; 32];
    let n = tokio::time::timeout(Duration::from_secs(2), s.read(&mut buf))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(&buf[..n], b"OK\n");

    // Resolve the actual project root the server picked. With
    // `data_root = <tempdir>` and a project named "p", make_root
    // composes `<tempdir>/git-<branch>/p` for some detected branch.
    let mut git_dir: Option<std::path::PathBuf> = None;
    let mut entries = tokio::fs::read_dir(dir.path()).await.unwrap();
    while let Some(entry) = entries.next_entry().await.unwrap() {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with("git-") {
            git_dir = Some(entry.path());
            break;
        }
    }
    let project_root = git_dir
        .expect("server should have created exactly one git-<branch>/ dir under data_root")
        .join("p");

    // Files must exist with the right contents and the right names.
    let on_data = tokio::fs::read(project_root.join("data.pickle")).await.unwrap();
    let on_param = tokio::fs::read(project_root.join("param.pickle")).await.unwrap();
    assert_eq!(on_data, data_payload);
    assert_eq!(on_param, param_payload);

    // Upload-id, hashes written.
    let uid = tokio::fs::read_to_string(project_root.join("upload_id.txt"))
        .await
        .unwrap();
    assert_eq!(uid.len(), 12, "upload_id should be 12-char hex");
    assert_eq!(
        tokio::fs::read_to_string(project_root.join("data_hash.txt"))
            .await
            .unwrap(),
        "deadbeef"
    );
    assert_eq!(
        tokio::fs::read_to_string(project_root.join("param_hash.txt"))
            .await
            .unwrap(),
        "feedface"
    );

    let _ = cancel.send(());
}

#[tokio::test]
async fn upload_atomic_then_tcmd_status_reflects_uploaded() {
    let dir = tempfile::tempdir().unwrap();
    let (addr, cancel) = spawn_server_with_data_root(Some(dir.path().to_path_buf())).await;

    let header = serde_json::json!({
        "request": "upload_atomic",
        "name": "proj",
        "data_size": 4,
        "param_size": 4,
    });
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(b"JSON").await.unwrap();
    s.write_all(format!("{header}\n").as_bytes()).await.unwrap();
    s.write_all(b"AAAA").await.unwrap();
    s.write_all(b"BBBB").await.unwrap();
    s.flush().await.unwrap();
    let mut ack = [0u8; 8];
    let _ = tokio::time::timeout(Duration::from_secs(2), s.read(&mut ack)).await;

    // Now poll status with TCMD --name proj. The same in-memory
    // engine handled the upload, so its state should now be
    // UPLOADED.
    drop(s);
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(b"TCMD").await.unwrap();
    s.write_all(b"--name proj").await.unwrap();
    s.shutdown().await.unwrap();

    let bytes = read_to_eof(&mut s).await;
    let body = std::str::from_utf8(&bytes).unwrap().trim_end();
    let response: Value = serde_json::from_str(body).expect(body);
    assert_eq!(response["data"], "READY", "data must be READY after upload");
    assert_eq!(
        response["status"], "NO_BUILD",
        "status must be NO_BUILD (uploaded but never built)",
    );
    assert_eq!(response["upload_id"].as_str().unwrap().len(), 12);

    let _ = cancel.send(());
}

#[tokio::test]
async fn upload_atomic_rejects_zero_sizes() {
    let (addr, cancel) = spawn_server().await;
    let dir = tempfile::tempdir().unwrap();

    let header = serde_json::json!({
        "request": "upload_atomic",
        "name": "p",
        "path": dir.path().to_string_lossy().to_string(),
        "data_size": 0,
        "param_size": 0,
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
            .contains("at least one of data_size or param_size"),
        "got: {body}",
    );

    let _ = cancel.send(());
}

#[tokio::test]
async fn tcmd_reconciles_existing_project_from_disk() {
    // Plant a fully-built project on disk: data.pickle + param.pickle
    // (Data::Uploaded), upload_id.txt + hash files, app_state.pickle
    // (Build::Built marker), and a state checkpoint
    // (project_resumable). Then start the server and TCMD-ping it.
    // The status response must reflect the restored state instead of
    // the default "NO_DATA" / "NO_BUILD" the engine starts at -- this
    // is what the addon panel reads to decide whether the Run button
    // is enabled.
    let dir = tempfile::tempdir().unwrap();
    let project_root = dir.path().join("p");
    std::fs::create_dir_all(project_root.join("session/output")).unwrap();
    std::fs::write(project_root.join("data.pickle"), b"DUMMY").unwrap();
    std::fs::write(project_root.join("param.pickle"), b"DUMMY").unwrap();
    std::fs::write(project_root.join("upload_id.txt"), b"abcdef012345").unwrap();
    std::fs::write(project_root.join("data_hash.txt"), b"deadbeef").unwrap();
    std::fs::write(project_root.join("param_hash.txt"), b"feedface").unwrap();
    std::fs::write(project_root.join("app_state.pickle"), b"DUMMY").unwrap();
    std::fs::write(
        project_root.join("session/output/state_1.bin.gz"),
        b"DUMMY",
    )
    .unwrap();

    let (addr, cancel) = spawn_server_with_data_root(Some(dir.path().to_path_buf())).await;
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(b"TCMD").await.unwrap();
    s.write_all(b"--name p").await.unwrap();
    s.shutdown().await.unwrap();
    let bytes = read_to_eof(&mut s).await;
    let body = std::str::from_utf8(&bytes).unwrap().trim_end();
    let response: Value = serde_json::from_str(body).expect(body);
    assert_eq!(response["data"], "READY", "data must be READY: {body}");
    assert_eq!(
        response["upload_id"], "abcdef012345",
        "upload_id should hydrate from upload_id.txt: {body}"
    );
    assert_eq!(response["data_hash"], "deadbeef", "got: {body}");
    assert_eq!(response["param_hash"], "feedface", "got: {body}");
    // Built + resumable means the status is RESUMABLE (the resumable
    // gate wins over READY when state.resumable is true).
    let status = response["status"].as_str().unwrap();
    assert!(
        status == "RESUMABLE" || status == "READY",
        "status should be a runnable state, got: {status} body={body}"
    );

    let _ = cancel.send(());
}
