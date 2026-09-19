// File: crates/ppf-cts-core/src/datamodel/session/scripts.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Solver-launcher script body + persistence, ffmpeg invocation
// formatting, and the `subprocess.Popen` command-line builders. Pure
// formatting / pure I/O wrappers; no actual subprocess work happens
// here (the caller drives `subprocess.run` etc.).

use std::path::{Path, PathBuf};

use super::types::{FixedSession, Platform, Session};

/// The directories a Windows solver's backend DLL can sit in under
/// `proj_root`, in the order the launcher searches them: a checkout's build
/// output (`build-win-native/build.bat`'s `LIB_DIR`), then a bundle's `bin`.
///
/// ONE LIST FOR EVERY CALLER THAT RUNS THE SOLVER. The solver IMPORTS this
/// DLL, so it does not start at all unless one of these is on PATH. The
/// session launcher puts both there for every solve, and `ppf-cts-server`
/// puts the same pair there for the one direct exec it makes (asking the
/// solver `--backend`), so the next time the directory moves neither caller
/// can be left naming the old one.
pub fn windows_library_dirs(proj_root: &Path) -> [PathBuf; 2] {
    [
        proj_root.join("crates/ppf-cts-compute/cuda/build/lib"),
        proj_root.join("bin"),
    ]
}

/// Produce the body of the solver-launcher script.
///
/// Caller persists the returned string to `<session.path>/<filename>`
/// (use `Platform::script_filename`) and `chmod 0o755` on Unix.
pub fn shell_command_script(
    session: &Session,
    fixed_session: &FixedSession,
    platform: Platform,
    solver_dir: &Path,
) -> String {
    let session_path = fixed_session.info.path.display();
    let output_path = fixed_session.output.path.display();
    match platform {
        Platform::Windows => {
            // THE SOLVER IS THE CALLER'S TO NAME, AND USED TO BE GUESSED HERE.
            // This joined `<proj_root>/target/release`, which is right only
            // while CARGO_TARGET_DIR is unset. With it set, `frontend` loads
            // one directory's extension module and this script ran another
            // directory's solver, which is a SPLIT-BACKEND RUN that nothing
            // downstream reports: both binaries answer their own `--backend`
            // honestly and neither is asked about the other. `build.rs`
            // refuses to put two backends in one directory precisely so a
            // directory names exactly one, and this line defeated that.
            //
            // `solver_dir` is where the generating process loaded the cdylib
            // from, and the solver is its sibling on every layout: a dev tree
            // builds both into `target/<profile>`, and the Windows and macOS
            // bundles copy both into `target/release`.
            let program = solver_dir.join("ppf-contact-solver.exe");
            // libsimbackend_cuda.dll lives in two different paths:
            //   - dev tree: `<proj_root>/crates/ppf-cts-compute/cuda/build/lib`
            //   - bundle:   `<proj_root>/bin`
            // Cmd's PATH accepts both as semicolon-separated dirs, so
            // we set both. Whichever exists gets picked up; missing
            // dirs are tolerated by Windows. CUDA_PATH/bin is also
            // included for the cudart runtime.
            //
            // THE DEV PATH IS THE COMPUTE CRATE'S, and it must match
            // `build-win-native/build.bat`'s `LIB_DIR`, which is
            // `crates\ppf-cts-compute\cuda\build\lib`. It named the solver
            // crate's old tree until the backend moved, and nothing noticed
            // because the Windows solver did not LOAD the library at all: it
            // dispatched into its own host renderings, so a wrong PATH cost
            // nothing. Once the solver actually imported the DLL, the stale
            // value became an immediate `STATUS_DLL_NOT_FOUND` (exit
            // 0xC0000135) with no output, and every frontend-launched Windows
            // solve died at startup. A path that only matters once something
            // else is fixed is exactly the kind that rots unnoticed.
            let [lib_dev, lib_bundle] = windows_library_dirs(&session.proj_root);
            // Path interpolations are wrapped in double quotes so a
            // project root containing spaces (e.g. ``C:\New Folder\proj``)
            // reaches the solver as a single argument instead of being
            // word-split by clap. ``set FOO=...`` does not need quoting
            // (cmd reads the value to end-of-line literally), and use
            // sites already quote ``%SOLVER_PATH%`` / the run line.
            format!(
                r#"@echo off
set SOLVER_PATH={program}
set LIB_PATH_DEV={lib_dev}
set LIB_PATH_BUNDLE={lib_bundle}

REM CUDA_PATH should be set by start.bat or the environment
set PATH=%LIB_PATH_DEV%;%LIB_PATH_BUNDLE%;%CUDA_PATH%\bin;%PATH%

if not exist "%SOLVER_PATH%" (
    echo Error: Solver does not exist at %SOLVER_PATH% >&2
    exit /b 1
)

"%SOLVER_PATH%" --path "{session_path}" --output "{output_path}" %*
"#,
                program = program.display(),
                lib_dev = lib_dev.display(),
                lib_bundle = lib_bundle.display(),
            )
        }
        Platform::Unix => {
            let program = solver_dir.join("ppf-contact-solver");
            // A SELF-CONTAINED TREE KEEPS EVERY CACHE INSIDE ITSELF, and two
            // caches are written by the SOLVER process rather than the
            // frontend, each under $HOME by default: the Metal pipeline
            // archive, whose directory the backend takes from
            // PPF_METAL_ARCHIVE_DIR, and the NVIDIA driver's compute cache,
            // which the driver creates at ~/.nv/ComputeCache when CUDA
            // initializes unless CUDA_CACHE_PATH names another directory.
            // Without these a distribution would write under the user's home
            // however it was started. Every solve runs through this script,
            // from the launcher and from a server the add-on spawns alike, so
            // this is where the tree's own cache directory is handed down. The
            // directories come from the one resolver every other cache uses, and
            // a value the user exported is kept.
            let cache_exports = if crate::datamodel::app::is_selfcontained(&session.proj_root) {
                let cache = crate::datamodel::app::default_cache_dir(&session.proj_root, None);
                [
                    ("PPF_METAL_ARCHIVE_DIR", "metal-pipeline-archive"),
                    ("CUDA_CACHE_PATH", "nv-compute-cache"),
                ]
                .iter()
                .map(|(name, dir)| {
                    format!("export {name}=\"${{{name}:-{}}}\"\n", cache.join(dir).display())
                })
                .collect::<String>()
            } else {
                String::new()
            };
            // Same spaces-in-path concern as the Windows branch: quote
            // ``{session_path}`` / ``{output_path}`` so the POSIX shell
            // doesn't word-split them before exec hands them to clap.
            format!(
                r#"#!/bin/bash
SOLVER_PATH="{program}"
{cache_exports}
if [ ! -f "$SOLVER_PATH" ]; then
    echo "Error: Solver does not exist at $SOLVER_PATH" >&2
    exit 1
fi

"$SOLVER_PATH" --path "{session_path}" --output "{output_path}" "$@"
"#,
                program = program.display(),
            )
        }
    }
}

/// Persist the solver-launcher script under `<session_path>/<filename>`,
/// chmod 0o755 on Unix, return the absolute path. Caller drives
/// `param.export(...)` separately through `param_export_to_disk`.
pub fn write_shell_command_script(
    session_path: &Path,
    output_path: &Path,
    proj_root: &Path,
    platform: Platform,
    solver_dir: &Path,
) -> std::io::Result<PathBuf> {
    // AN EMPTY SOLVER DIRECTORY IS REFUSED HERE RATHER THAN WRITTEN. It reached
    // this function once: `frontend.artifact_dir()` answered `None` on the path
    // where the cdylib was already registered, the caller passed "", and the
    // launcher went to disk with a bare `ppf-contact-solver` as its
    // `SOLVER_PATH`. The only symptom was the script's own existence check
    // failing at run time, which names a path it never prints, so the caller
    // learned nothing about where the empty string came from.
    if solver_dir.as_os_str().is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "the solver directory is empty: the caller could not say where its \
             own build is. `frontend.artifact_dir()` answers it from the \
             directory the cdylib was loaded from, and returns None only when \
             nothing is built.",
        ));
    }
    // Re-use the existing templater. It expects two struct stand-ins
    // so the on-disk paths flow through the same formatting branch
    // the Python source did.
    let session_shim = Session::new(
        "py-shim",
        "py-shim",
        std::path::PathBuf::from("/unused/app"),
        proj_root.to_path_buf(),
        std::path::PathBuf::from("/unused/data"),
    );
    let mut fixed_shim = FixedSession::from_session(&session_shim);
    fixed_shim.info.path = session_path.to_path_buf();
    fixed_shim.output.path = output_path.to_path_buf();
    let body = shell_command_script(&session_shim, &fixed_shim, platform, solver_dir);

    let dest = session_path.join(platform.script_filename());
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&dest, body.as_bytes())?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if matches!(platform, Platform::Unix) {
            let mut perm = std::fs::metadata(&dest)?.permissions();
            perm.set_mode(0o755);
            std::fs::set_permissions(&dest, perm)?;
        }
    }
    Ok(dest)
}

/// Build the per-platform solver invocation string (the body passed
/// to `subprocess.Popen(shell=True)`):
///     win: `"{cmd_path}" --load {load}`
///     unix: `bash "{cmd_path}" --load {load}`
///
/// Both branches double-quote `cmd_path` so a project root containing
/// spaces (e.g. `~/My Project/...`) doesn't get word-split by the
/// shell before exec.
pub fn solver_subprocess_command(cmd_path: &Path, load: i64, platform: Platform) -> String {
    match platform {
        Platform::Windows => format!(r#""{}" --load {}"#, cmd_path.display(), load),
        Platform::Unix => format!(r#"bash "{}" --load {}"#, cmd_path.display(), load),
    }
}

/// Search for a bundled `ffmpeg` binary under `project_root`. Returns
/// the first existing candidate or `None`. The caller falls back to
/// `shutil.which("ffmpeg")` when this returns `None`.
pub fn locate_bundled_ffmpeg(project_root: &Path) -> Option<PathBuf> {
    let candidates = [
        project_root.join("bin").join("ffmpeg"),
        project_root.join("bin").join("ffmpeg.exe"),
        project_root
            .join("build-win-native")
            .join("ffmpeg")
            .join("ffmpeg.exe"),
    ];
    candidates.into_iter().find(|p| p.is_file())
}

/// Build the ffmpeg command line invoked via
/// `subprocess.run(..., shell=True)`. Caller still drives the
/// subprocess (we don't shell out from Rust). Both `ffmpeg_path` and
/// `vid_name` are double-quoted so paths or file names with spaces
/// reach ffmpeg as single arguments.
pub fn ffmpeg_video_command(ffmpeg_path: &Path, ext: &str, vid_name: &str) -> String {
    format!(
        "\"{ffmpeg}\" -hide_banner -loglevel error -y -r 60 -i frame_%d.{ext}.png \
         -pix_fmt yuv420p -c:v libx264 \"{vid_name}\"",
        ffmpeg = ffmpeg_path.display(),
    )
}
