// File: crates/ppf-cts-server/src/error.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Centralized wire-protocol error type. The JSON shape is owned in
// one place rather than scattered across `wire/*` handlers.
//
// Three shapes are emitted today, picked per call-site:
//
//   * "Bare":         `{"error": "<message>"}`
//                     JSON request handlers (data/upload/notebook).
//   * "TCMD-Full":    `{"error": "...", "protocol_version": ...,
//                       "upload_id": "", "status": ""}`
//                     Used for TCMD text-decode failures, where the
//                     addon recognizes `error != "" && status == ""`
//                     as "server raised, no status update".
//   * "TCMD-Brief":   `{"error": "...", "protocol_version": ...}`
//                     The NO_ID response (`name` missing on TCMD).
//                     The addon checks `error == "NO_ID"` literally.
//
// Tests assert on substring matches of `error`, so message text is
// preserved verbatim where existing assertions require it.
//
// Wire handlers can build a response with `err.into_response()` and
// can use `?` propagation by returning `Result<_, ServerError>` from
// inner closures (the outer handler keeps its `io::Result<()>` shape
// so writer errors stay distinct from protocol errors).

use serde_json::{json, Value};

use crate::upload::UploadError;
use crate::PROTOCOL_VERSION;

/// Wire-protocol error categories. Each variant carries the message
/// text that ends up in the response JSON's `"error"` field.
#[derive(Debug, thiserror::Error)]
pub(crate) enum ServerError {
    /// Generic 400-style: missing fields, invalid sizes, refusal of
    /// unsupported inputs (e.g. `data.pickle` over `data_send`).
    #[error("{0}")]
    BadRequest(String),

    /// 404-style: requested file or resource doesn't exist.
    #[error("{0}")]
    NotFound(String),

    /// `notebook_send` / `notebook_delete` path-traversal guard.
    #[error("Path escapes sandbox: {0}")]
    SandboxEscape(String),

    /// Conflict: state machine refuses the request right now (e.g.
    /// upload while a build is in progress).
    #[error("{0}")]
    Conflict(String),

    /// Unknown wire header (the 4-byte prefix isn't TCMD/JSON/BDAT).
    #[error("Unknown header: {0}")]
    UnknownHeader(String),

    /// Unknown JSON `request` field value.
    #[error("Unknown request: {0}")]
    UnknownRequest(String),

    /// TCMD payload arrived without a `--name` argument. The addon
    /// keys on `error == "NO_ID"` literally, so this variant carries
    /// no payload.
    #[error("NO_ID")]
    NoId,

    /// TCMD body wasn't valid UTF-8.
    #[error("Text decode error: {0}")]
    TextDecode(String),

    /// JSON header line couldn't be read.
    #[error("JSON header read: {0}")]
    JsonHeaderRead(String),

    /// JSON header line didn't parse as a JSON object.
    #[error("JSON error: {0}")]
    JsonParse(String),

    /// `io::Error` lift for `?` propagation. Re-emitted as
    /// `Internal` shape on the wire.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// Upload-id helper failure (see `upload::UploadError`).
    #[error(transparent)]
    Upload(#[from] UploadError),

    /// Catch-all internal failure (filesystem mkdir, scratch-file
    /// rename, etc.). Carries the formatted message verbatim so test
    /// substring assertions still match.
    #[error("{0}")]
    Internal(String),
}

/// Which envelope to wrap an error message in. Each existing wire
/// callsite produced one of three shapes; preserve them so the
/// switch is byte-compatible.
enum Shape {
    /// `{"error": "<msg>"}` -- bare JSON-path response.
    Bare,
    /// `{"error": "...", "protocol_version": ..., "upload_id": "",
    ///   "status": ""}` -- TCMD failure response with the full
    /// status-substitute envelope (mirrors `response::error_response`).
    TcmdFull,
    /// `{"error": "...", "protocol_version": ...}` -- TCMD `NO_ID`
    /// response, the addon's literal `error == "NO_ID"` check.
    TcmdBrief,
}

impl ServerError {
    fn shape(&self) -> Shape {
        match self {
            ServerError::NoId => Shape::TcmdBrief,
            ServerError::TextDecode(_) => Shape::TcmdFull,
            _ => Shape::Bare,
        }
    }

    /// Render the JSON response body the wire layer should write.
    pub(crate) fn into_response(self) -> Value {
        let shape = self.shape();
        let message = match self {
            ServerError::NoId => "NO_ID".to_string(),
            // For non-NoId variants we use the Display impl, which
            // already prefixes / formats the message.
            other => other.to_string(),
        };
        match shape {
            Shape::Bare => json!({"error": message}),
            Shape::TcmdBrief => json!({
                "error": message,
                "protocol_version": PROTOCOL_VERSION,
            }),
            Shape::TcmdFull => json!({
                "error": message,
                "protocol_version": PROTOCOL_VERSION,
                "upload_id": "",
                "status": "",
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bad_request_renders_bare_shape() {
        let r = ServerError::BadRequest("Missing path or name".into()).into_response();
        assert_eq!(r["error"], "Missing path or name");
        assert!(r.get("protocol_version").is_none());
    }

    #[test]
    fn no_id_renders_brief_tcmd_shape() {
        // NO_ID emits the 2-key brief shape so the addon's literal
        // `error == "NO_ID"` check passes.
        let r = ServerError::NoId.into_response();
        assert_eq!(r["error"], "NO_ID");
        assert_eq!(r["protocol_version"], PROTOCOL_VERSION);
        // No status / upload_id keys in the brief shape.
        assert!(r.get("status").is_none());
        assert!(r.get("upload_id").is_none());
    }

    #[test]
    fn text_decode_renders_full_tcmd_shape() {
        let r = ServerError::TextDecode("not utf-8".into()).into_response();
        assert!(r["error"].as_str().unwrap().contains("Text decode error"));
        assert_eq!(r["protocol_version"], PROTOCOL_VERSION);
        assert_eq!(r["upload_id"], "");
        assert_eq!(r["status"], "");
    }

    #[test]
    fn unknown_header_carries_hex_payload() {
        let r = ServerError::UnknownHeader("deadbeef".into()).into_response();
        assert!(r["error"].as_str().unwrap().contains("Unknown header"));
        assert!(r["error"].as_str().unwrap().contains("deadbeef"));
    }

    #[test]
    fn sandbox_escape_message_contains_relative_path() {
        let r = ServerError::SandboxEscape("../escape".into()).into_response();
        assert!(r["error"].as_str().unwrap().contains("escapes sandbox"));
        assert!(r["error"].as_str().unwrap().contains("../escape"));
    }

    #[test]
    fn unknown_request_message_contains_request_name() {
        let r = ServerError::UnknownRequest("no_such_request".into()).into_response();
        assert!(
            r["error"]
                .as_str()
                .unwrap()
                .contains("Unknown request: no_such_request"),
            "got: {r}"
        );
    }
}
