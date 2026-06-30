// File: crates/ppf-cts-server/src/protocol.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Wire-level protocol constants + framing helpers. Pinned to
// PROTOCOL_VERSION in lib.rs; the Blender addon expects every byte of
// these headers exactly as defined here.

use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// 4-byte text-command header. The TCMD path carries an
/// `--key val`-style argument string after the header.
pub const HEADER_TCMD: &[u8; 4] = b"TCMD";

/// 4-byte JSON header. Followed by a single newline-terminated JSON
/// document; `request` field selects the sub-handler (upload_atomic,
/// upload_notify, data_send, data_receive, notebook_send,
/// notebook_delete).
pub const HEADER_JSON: &[u8; 4] = b"JSON";

/// 4-byte binary-data header. Currently a stub; the addon doesn't
/// emit BDAT in production, but the path acknowledges with
/// `BINARY_OK` for legacy testing tools.
pub const HEADER_BDAT: &[u8; 4] = b"BDAT";

/// Read exactly 4 bytes from the connection. Returns `None` on
/// clean disconnect (e.g. client closed before sending the header).
pub async fn read_header<R>(reader: &mut R) -> std::io::Result<Option<[u8; 4]>>
where
    R: AsyncReadExt + Unpin,
{
    let mut buf = [0u8; 4];
    match reader.read_exact(&mut buf).await {
        Ok(_) => Ok(Some(buf)),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
        Err(e) => Err(e),
    }
}

/// Read until and including the first `\n` byte. Used for JSON
/// header lines.
pub async fn read_line<R>(reader: &mut R, max: usize) -> std::io::Result<Vec<u8>>
where
    R: AsyncReadExt + Unpin,
{
    let mut out = Vec::with_capacity(256);
    let mut byte = [0u8; 1];
    while out.len() < max {
        let n = reader.read(&mut byte).await?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "connection closed mid-line",
            ));
        }
        out.push(byte[0]);
        if byte[0] == b'\n' {
            return Ok(out);
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("line exceeds {max} bytes"),
    ))
}

/// Chunk size for `read_exact_n_chunked`. 32 KB.
const RECV_CHUNK: usize = 32 * 1024;

/// Cap on the up-front reservation in `read_exact_n_chunked`. The Vec
/// still grows via `extend_from_slice` as data arrives, so correctness
/// is unchanged; this only bounds the initial allocation so an
/// untrusted `n` cannot request a multi-terabyte buffer before the
/// first byte is read. Callers should still reject oversized declared
/// sizes up front (see MAX_PAYLOAD_BYTES); this is defense-in-depth.
const MAX_RECV_RESERVE: usize = 256 * 1024;

/// Read `n` bytes in 32 KB chunks. Avoids one giant allocation for
/// multi-MB pickles.
pub async fn read_exact_n_chunked<R>(reader: &mut R, n: usize) -> std::io::Result<Vec<u8>>
where
    R: AsyncReadExt + Unpin,
{
    let mut out = Vec::with_capacity(n.min(MAX_RECV_RESERVE));
    let mut remaining = n;
    let mut buf = [0u8; RECV_CHUNK];
    while remaining > 0 {
        let want = remaining.min(buf.len());
        let n_read = reader.read(&mut buf[..want]).await?;
        if n_read == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("upload truncated, {remaining} of {n} bytes missing"),
            ));
        }
        out.extend_from_slice(&buf[..n_read]);
        remaining -= n_read;
    }
    Ok(out)
}

/// Send a JSON value as a single line (newline-terminated).
pub async fn write_json_line<W>(
    writer: &mut W,
    value: &serde_json::Value,
) -> std::io::Result<()>
where
    W: AsyncWriteExt + Unpin,
{
    let s = serde_json::to_string(value)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    writer.write_all(s.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use tokio::io::BufReader;

    #[tokio::test]
    async fn read_header_returns_4_bytes() {
        let cur = Cursor::new(b"TCMD".to_vec());
        let mut r = BufReader::new(cur);
        let h = read_header(&mut r).await.unwrap().unwrap();
        assert_eq!(&h, HEADER_TCMD);
    }

    #[tokio::test]
    async fn read_header_handles_clean_eof() {
        let cur = Cursor::new(Vec::<u8>::new());
        let mut r = BufReader::new(cur);
        assert!(read_header(&mut r).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn read_line_collects_until_newline() {
        let cur = Cursor::new(b"hello\nworld".to_vec());
        let mut r = BufReader::new(cur);
        let line = read_line(&mut r, 64).await.unwrap();
        assert_eq!(line, b"hello\n");
    }

    #[tokio::test]
    async fn read_line_rejects_too_long() {
        let cur = Cursor::new(vec![b'a'; 100]);
        let mut r = BufReader::new(cur);
        let err = read_line(&mut r, 16).await.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    // A huge declared `n` must not trigger an up-front multi-terabyte
    // reservation (which would abort the process). The reservation is
    // capped at MAX_RECV_RESERVE, so with a short reader the call
    // simply streams what it can and then reports a truncated read.
    #[tokio::test]
    async fn read_exact_n_chunked_caps_upfront_reservation() {
        let cur = Cursor::new(vec![0u8; 64]);
        let mut r = BufReader::new(cur);
        // A petabyte-scale declared size: the old `with_capacity(n)`
        // would abort here; the capped reservation lets us reach the
        // EOF path instead.
        let huge = 1usize << 50;
        let err = read_exact_n_chunked(&mut r, huge).await.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
    }
}
