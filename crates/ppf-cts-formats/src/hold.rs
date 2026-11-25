// File: crates/ppf-cts-formats/src/hold.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! The caller's side of the frame-step hold: writing
//! [`HOLD_AT_FRAME`](crate::files::HOLD_AT_FRAME) and reading
//! [`HELD`](crate::files::HELD).
//!
//! BOTH FILES ARE TOUCHED BY TWO PROCESSES AT ONCE. A held solver reads
//! `hold_at_frame` every 20 ms and removes `held` when it continues, while
//! the frontend polls `held` and replaces `hold_at_frame`. On Windows that
//! works only when every handle on the file shares delete access, and when the
//! replace may land on a target another handle has open. Rust's std does both.
//! Python's file calls do neither:
//!
//! - Python's `open` shares no delete access, so the solver's removal of
//!   `held` fails while a Python poll has it open. The solver ignores that
//!   failure, and the old frame stays behind.
//! - Python's `os.replace` (MoveFileEx) refuses with "Access is denied"
//!   whenever the solver has `hold_at_frame` open for its poll, so a step
//!   request fails.
//!
//! Measured on Windows Server 2025 against a reader in a tight loop: 17742
//! of 19580 `os.replace` calls refused, against 0 of 12737 `std::fs::rename`
//! calls, and 1944 of 2414 removals of `held` failed under a Python reader,
//! against 0 of 6620 under a std reader. The frontend therefore goes through
//! these functions, and none of its own file calls.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use crate::files::{HELD, HOLD_AT_FRAME};

/// Ask the solver to hold once it has written `frame`, waiting there at most
/// `timeout` seconds, by writing `<output_dir>/hold_at_frame`. Returns the
/// path written.
///
/// The request is written to a sibling and renamed over the file, so the
/// solver reads the old request or the new one, never part of one.
pub fn write_hold_at_frame(output_dir: &Path, frame: i64, timeout: f64) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(output_dir)?;
    let path = output_dir.join(HOLD_AT_FRAME);
    let tmp = output_dir.join(format!("{HOLD_AT_FRAME}.tmp"));
    std::fs::write(&tmp, format!("{frame} {timeout}\n"))?;
    std::fs::rename(&tmp, &path)?;
    Ok(path)
}

/// The frame `<output_dir>/held` names, or `None` when the solver is not
/// held. That covers no file, and an empty one the solver has created but
/// not yet written.
///
/// A file whose removal is in progress also reads as `None`. On Windows,
/// opening one fails with access denied rather than not found, so that error
/// means the same thing here.
pub fn read_held_frame(output_dir: &Path) -> std::io::Result<Option<i64>> {
    let path = output_dir.join(HELD);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
        Err(e) if cfg!(windows) && e.kind() == ErrorKind::PermissionDenied => return Ok(None),
        Err(e) => return Err(e),
    };
    let text = text.trim();
    if text.is_empty() {
        return Ok(None);
    }
    text.parse::<i64>().map(Some).map_err(|e| {
        std::io::Error::new(
            ErrorKind::InvalidData,
            format!("{}: {text:?} is not a frame: {e}", path.display()),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hold_request_is_one_line_the_solver_parses() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("output");
        let path = write_hold_at_frame(&output, 11, 3600.0).unwrap();
        assert_eq!(path, output.join(HOLD_AT_FRAME));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "11 3600\n");
        write_hold_at_frame(&output, 12, 0.5).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "12 0.5\n");
        assert!(!output.join(format!("{HOLD_AT_FRAME}.tmp")).exists());
    }

    #[test]
    fn held_reads_the_frame_or_none() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read_held_frame(dir.path()).unwrap(), None);
        std::fs::write(dir.path().join(HELD), "").unwrap();
        assert_eq!(read_held_frame(dir.path()).unwrap(), None);
        std::fs::write(dir.path().join(HELD), "7\n").unwrap();
        assert_eq!(read_held_frame(dir.path()).unwrap(), Some(7));
        std::fs::write(dir.path().join(HELD), "seven\n").unwrap();
        assert_eq!(read_held_frame(dir.path()).unwrap_err().kind(), ErrorKind::InvalidData);
    }
}
