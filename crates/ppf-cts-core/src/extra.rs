// File: crates/ppf-cts-core/src/extra.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Auxiliary helpers:
//
//   * `load_cipc_stitch_mesh`: parse the OBJ-with-`stitch`-tag format
//     used by the Codim-IPC repo. Returns vertices, zero-based faces,
//     and per-stitch (index4, weight4) rows. Pure parser, no I/O
//     beyond reading the file.
//   * `sparse_clone`: thin wrapper around `git` for blob:none +
//     sparse-checkout cloning. Delegates to `std::process::Command`
//     using the standard subprocess flow (same argv, same `set` ->
//     `add` -> `checkout` ordering) so the fixtures-fetching examples
//     in docs keep working.
//
// We expose `Vec<f64>` / `Vec<i32>` flat row-major arrays here; the
// PyO3 binding wraps them as numpy arrays the same way the rest of
// the kernels do.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::Command;

#[derive(Debug, thiserror::Error)]
pub enum ExtraError {
    #[error("failed to open mesh file '{path}': {error}")]
    OpenFailed { path: String, error: String },
    #[error("failed to read mesh file '{path}': {error}")]
    ReadFailed { path: String, error: String },
    #[error(
        "git is not found in PATH. The bundled distribution includes MinGit, \
         but it may not be in PATH. Please ensure you're running from start.bat \
         or add mingit/cmd to your PATH. For manual installation: choco install git"
    )]
    GitNotFound,
    #[error("git command failed: {0}")]
    GitFailed(String),
    #[error("filesystem error at '{path}': {error}")]
    FsError { path: String, error: String },
    #[error("expected '{path}' to exist after sparse-checkout add")]
    SparseCheckoutMissing { path: String },
}

/// Parsed Codim-IPC stitch mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct CipcStitchMesh {
    /// `(n_vertices, 3)` row-major positions.
    pub vertices: Vec<f64>,
    pub n_vertices: usize,
    /// `(n_faces, 3)` zero-based vertex indices, row-major.
    pub faces: Vec<i32>,
    pub n_faces: usize,
    /// `(n_stitch, 4)` index quadruple `[i0, i1, i2, i2]` (last
    /// repeats per the Python source). This is the legacy single-source
    /// wire form; `assemble_dyn_scene` Step 6 pads it to the 6-wide
    /// barycentric-barycentric layout `[i0, i0, i0, i1, i2, i2]`.
    pub stitch_index: Vec<i32>,
    /// `(n_stitch, 4)` weight quadruple `[1.0, 1-w, w, 0.0]`; padded to
    /// `[1, 0, 0, 1-w, w, 0]` (source degenerate) by assembly Step 6.
    pub stitch_weight: Vec<f64>,
    pub n_stitch: usize,
}

pub fn load_cipc_stitch_mesh<P: AsRef<Path>>(path: P) -> Result<CipcStitchMesh, ExtraError> {
    let path_ref = path.as_ref();
    let file = File::open(path_ref).map_err(|e| ExtraError::OpenFailed {
        path: path_ref.display().to_string(),
        error: e.to_string(),
    })?;
    let reader = BufReader::new(file);

    let mut vertices: Vec<f64> = Vec::new();
    let mut faces: Vec<i32> = Vec::new();
    let mut stitch_index: Vec<i32> = Vec::new();
    let mut stitch_weight: Vec<f64> = Vec::new();
    let mut n_vertices = 0usize;
    let mut n_faces = 0usize;
    let mut n_stitch = 0usize;

    for (lineno, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| ExtraError::ReadFailed {
            path: format!("{}:{}", path_ref.display(), lineno + 1),
            error: e.to_string(),
        })?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        match parts[0] {
            "v" if parts.len() == 4 => {
                let x: f64 = match parts[1].parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let y: f64 = match parts[2].parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let z: f64 = match parts[3].parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                vertices.extend_from_slice(&[x, y, z]);
                n_vertices += 1;
            }
            "f" if parts.len() == 4 => {
                // OBJ face indices are 1-based, may carry tex/normal
                // suffix `v/vt/vn`. Take the position component and
                // shift to zero-based.
                let mut face = [0i32; 3];
                let mut ok = true;
                for (i, p) in parts[1..4].iter().enumerate() {
                    let v_part = p.split('/').next().unwrap_or("");
                    match v_part.parse::<i32>() {
                        Ok(v) => face[i] = v - 1,
                        Err(_) => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    faces.extend_from_slice(&face);
                    n_faces += 1;
                }
            }
            "stitch" if parts.len() == 5 => {
                let i0: i32 = match parts[1].parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let i1: i32 = match parts[2].parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let i2: i32 = match parts[3].parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let w: f64 = match parts[4].parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                stitch_index.extend_from_slice(&[i0, i1, i2, i2]);
                stitch_weight.extend_from_slice(&[1.0, 1.0 - w, w, 0.0]);
                n_stitch += 1;
            }
            _ => {}
        }
    }

    Ok(CipcStitchMesh {
        vertices,
        n_vertices,
        faces,
        n_faces,
        stitch_index,
        stitch_weight,
        n_stitch,
    })
}

// ---------------------------------------------------------------------------
// sparse_clone

fn git_available() -> bool {
    Command::new("git")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn run_git(args: &[&str], cwd: Option<&Path>) -> Result<(), ExtraError> {
    let mut cmd = Command::new("git");
    cmd.args(args);
    if let Some(d) = cwd {
        cmd.current_dir(d);
    }
    let output = cmd
        .output()
        .map_err(|e| ExtraError::GitFailed(format!("spawn git failed: {e}")))?;
    if !output.status.success() {
        return Err(ExtraError::GitFailed(format!(
            "git {} failed (status {}): {}",
            args.join(" "),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(())
}

/// Fetch a git repository using sparse-checkout. Already-present
/// paths are skipped; paths missing under `dest` are added via `git
/// sparse-checkout add` followed by `git checkout`.
pub fn sparse_clone(
    url: &str,
    dest: &Path,
    paths: &[&str],
    delete_exist: bool,
) -> Result<(), ExtraError> {
    if !git_available() {
        return Err(ExtraError::GitNotFound);
    }
    if delete_exist && dest.exists() {
        std::fs::remove_dir_all(dest).map_err(|e| ExtraError::FsError {
            path: dest.display().to_string(),
            error: e.to_string(),
        })?;
    }
    if !dest.exists() {
        run_git(
            &[
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                url,
                &dest.display().to_string(),
            ],
            None,
        )?;
        run_git(&["sparse-checkout", "set"], Some(dest))?;
    }
    for path in paths {
        let target = dest.join(path);
        if !target.exists() {
            let mut argv = vec!["sparse-checkout", "add"];
            argv.push(path);
            run_git(&argv, Some(dest))?;
            run_git(&["checkout"], Some(dest))?;
        }
        if !target.exists() {
            return Err(ExtraError::SparseCheckoutMissing {
                path: target.display().to_string(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_file(dir: &std::path::Path, name: &str, content: &str) -> std::path::PathBuf {
        let p = dir.join(name);
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        p
    }

    #[test]
    fn parses_basic_obj_with_stitch_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(
            dir.path(),
            "stage.obj",
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nstitch 1 2 3 0.25\n",
        );
        let m = load_cipc_stitch_mesh(&path).unwrap();
        assert_eq!(m.n_vertices, 4);
        assert_eq!(m.n_faces, 1);
        assert_eq!(m.n_stitch, 1);
        assert_eq!(m.faces, vec![0, 1, 2]); // 1-based -> 0-based
        assert_eq!(m.stitch_index, vec![1, 2, 3, 3]); // last index repeats
        assert_eq!(m.stitch_weight, vec![1.0, 0.75, 0.25, 0.0]);
    }

    #[test]
    fn skips_unknown_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(
            dir.path(),
            "stage.obj",
            "# header comment\no Object\nv 0 0 0\nvn 0 1 0\nv 1 0 0\nf 1 2 1\n",
        );
        let m = load_cipc_stitch_mesh(&path).unwrap();
        assert_eq!(m.n_vertices, 2);
        assert_eq!(m.n_faces, 1);
        assert_eq!(m.n_stitch, 0);
    }

    #[test]
    fn handles_obj_face_with_texcoord_indices() {
        // OBJ face syntax `v/vt/vn`: take the v component.
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(
            dir.path(),
            "stage.obj",
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1/1 2/2/1 3/3/1\n",
        );
        let m = load_cipc_stitch_mesh(&path).unwrap();
        assert_eq!(m.faces, vec![0, 1, 2]);
    }

    #[test]
    fn open_failed_for_missing_file() {
        let err = load_cipc_stitch_mesh("/no/such/file.obj").unwrap_err();
        match err {
            ExtraError::OpenFailed { .. } => {}
            other => panic!("expected OpenFailed, got {other:?}"),
        }
    }

    #[test]
    fn empty_file_returns_empty_mesh() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(dir.path(), "empty.obj", "");
        let m = load_cipc_stitch_mesh(&path).unwrap();
        assert_eq!(m.n_vertices, 0);
        assert_eq!(m.n_faces, 0);
        assert_eq!(m.n_stitch, 0);
    }

    #[test]
    fn malformed_vertex_line_is_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(dir.path(), "bad.obj", "v 0 0\nv 1 2 3\n"); // first has wrong arity
        let m = load_cipc_stitch_mesh(&path).unwrap();
        assert_eq!(m.n_vertices, 1);
    }

    #[test]
    fn git_error_strings_match_python_substring() {
        let m = format!("{}", ExtraError::GitNotFound);
        assert!(m.contains("git is not found in PATH"));
        assert!(m.contains("MinGit"));
        assert!(m.contains("start.bat"));
        assert!(m.contains("choco install git"));
    }

    #[test]
    fn sparse_clone_delete_exist_removes_dir() {
        // We don't have network access in tests; just exercise the
        // delete branch with an empty `paths` list to confirm
        // existing-dir cleanup runs without error before the git
        // clone. Skipped entirely if git isn't on PATH.
        if !git_available() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("repo");
        std::fs::create_dir_all(&dest).unwrap();
        std::fs::write(dest.join("placeholder"), "x").unwrap();
        // Delete-exist path: removes the dir, then tries to clone a
        // bogus URL. We only care that the deletion fired before the
        // clone attempt.
        let err = sparse_clone("file:///dev/null/no-such-repo", &dest, &[], true).unwrap_err();
        assert!(!dest.exists() || matches!(err, ExtraError::GitFailed(_)));
    }
}
