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

/// Produce the body of the solver-launcher script.
///
/// Caller persists the returned string to `<session.path>/<filename>`
/// (use `Platform::script_filename`) and `chmod 0o755` on Unix.
pub fn shell_command_script(
    session: &Session,
    fixed_session: &FixedSession,
    platform: Platform,
) -> String {
    let session_path = fixed_session.info.path.display();
    let output_path = fixed_session.output.path.display();
    match platform {
        Platform::Windows => {
            // Resolve the solver binary across both layouts the
            // template supports:
            //   - dev tree: `<proj_root>/target/release/ppf-contact-solver.exe`
            //   - bundle:   `<proj_root>/target/release/ppf-contact-solver.exe`
            //               (bundle.bat copies it under the same path)
            let program = session
                .proj_root
                .join("target/release/ppf-contact-solver.exe");
            // libsimbackend_cuda.dll lives in two different paths:
            //   - dev tree: `<proj_root>/crates/ppf-cts-solver/src/cpp/build/lib`
            //   - bundle:   `<proj_root>/bin`
            // Cmd's PATH accepts both as semicolon-separated dirs, so
            // we set both. Whichever exists gets picked up; missing
            // dirs are tolerated by Windows. CUDA_PATH/bin is also
            // included for the cudart runtime.
            let lib_dev = session
                .proj_root
                .join("crates/ppf-cts-solver/src/cpp/build/lib");
            let lib_bundle = session.proj_root.join("bin");
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
            let program = session.proj_root.join("target/release/ppf-contact-solver");
            // Same spaces-in-path concern as the Windows branch: quote
            // ``{session_path}`` / ``{output_path}`` so the POSIX shell
            // doesn't word-split them before exec hands them to clap.
            format!(
                r#"#!/bin/bash
SOLVER_PATH="{program}"

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
) -> std::io::Result<PathBuf> {
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
    let body = shell_command_script(&session_shim, &fixed_shim, platform);

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
