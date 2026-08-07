// File: signal_sidecar.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! Fatal-signal handler that records WHICH signal killed the solver.
//!
//! A signal death is the one exit that cannot reach the terminal status
//! record: the handler must stay async-signal-safe, and serializing CBOR is
//! not. It writes one pre-formatted line to
//! `<output>/crash_signal` instead, which the server reads when it finds no
//! terminal outcome (`ppf_cts_formats::status::signal_sidecar`).
//!
//! This is deliberately incomplete and the incompleteness is structural:
//! `SIGKILL` cannot be caught, so an OOM-kill leaves no sidecar. The server
//! covers that case from the launcher's exit status, and reports it as a kill
//! it could not attribute rather than guessing at a reason.

use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::sync::atomic::{AtomicI32, AtomicPtr, Ordering};

use ppf_cts_formats::files;
use ppf_cts_formats::status::HANDLED_SIGNALS;

/// File descriptor of the open sidecar, or -1 before `install` runs. Read by
/// the handler, so it must be lock-free and never point at a closed
/// descriptor; the file stays open for the whole process.
static SIDECAR_FD: AtomicI32 = AtomicI32::new(-1);

/// Per-signal payload and previous disposition, one entry per
/// [`HANDLED_SIGNALS`] slot and in the same order. Leaked to `'static` at
/// install time so the handler only ever dereferences memory that outlives
/// every thread.
struct Installed {
    signo: i32,
    line: &'static [u8],
    previous: libc::sigaction,
}

/// Base of the leaked [`Installed`] array, or null until `install` publishes
/// it. A single atomic word, so a handler that runs during the store either
/// sees the whole table or sees nothing; a plain `&[Installed]` is a pointer
/// and a length written separately, and a signal landing between the two
/// halves would dereference a torn pair. The length is always
/// `HANDLED_SIGNALS.len()`, because `install` builds one entry per signal
/// whether or not the `sigaction` that arms it succeeds.
static INSTALLED: AtomicPtr<Installed> = AtomicPtr::new(std::ptr::null_mut());

/// Install the handler for every signal in [`HANDLED_SIGNALS`] and open the
/// sidecar the handler writes to.
///
/// Call once, after the liveness lock is acquired, so the sidecar and the
/// status record describe the same launch.
pub fn install(output_dir: &Path, launch_id: &str) {
    let path = output_dir.join(files::CRASH_SIGNAL);
    let c_path = match std::ffi::CString::new(path.as_os_str().as_bytes()) {
        Ok(p) => p,
        Err(_) => {
            log::warn!(
                "status: output path contains a NUL byte; a fatal signal will \
                 not be named in the crash report"
            );
            return;
        }
    };
    // SAFETY: `c_path` is a valid NUL-terminated path for the duration of
    // the call. The descriptor is intentionally never closed: the handler
    // uses it and the process owns it for its whole life.
    let fd = unsafe {
        libc::open(
            c_path.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC,
            0o644 as libc::c_int,
        )
    };
    if fd < 0 {
        log::warn!(
            "status: could not open the crash-signal sidecar; a fatal signal \
             will not be named in the crash report"
        );
        return;
    }
    SIDECAR_FD.store(fd, Ordering::SeqCst);

    // Two passes, and the order between them is the point. The first only
    // READS each signal's current disposition, so nothing is armed yet; the
    // table is published between the passes; the second arms the handlers. A
    // signal that arrives while a handler is armed but the table is not yet
    // published would find no entry for itself, and a handler that returns
    // from a synchronous fault resumes at the faulting instruction, so the
    // process would spin in the fault forever instead of dying.
    let mut installed: Vec<Installed> = Vec::with_capacity(HANDLED_SIGNALS.len());
    for (signo, name) in HANDLED_SIGNALS {
        // Format the whole line NOW. The handler may not allocate or format,
        // so what it writes has to already exist as bytes.
        let line: &'static [u8] = Box::leak(
            format!("{name} {launch_id}\n")
                .into_bytes()
                .into_boxed_slice(),
        );
        // SAFETY: a NULL `act` queries the current disposition and changes
        // nothing; `previous` receives it. A zeroed sigaction is a valid
        // output buffer.
        let mut previous: libc::sigaction = unsafe { std::mem::zeroed() };
        unsafe {
            libc::sigaction(*signo, std::ptr::null(), &mut previous);
        }
        installed.push(Installed {
            signo: *signo,
            line,
            previous,
        });
    }
    let table: &'static [Installed] = Box::leak(installed.into_boxed_slice());
    INSTALLED.store(table.as_ptr() as *mut Installed, Ordering::Release);

    for entry in table {
        // SAFETY: a zeroed sigaction is a valid empty template; the fields
        // below fill it in before use.
        let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
        action.sa_sigaction = handler as *const () as usize;
        // SA_ONSTACK is required, not decorative: a stack-overflow SIGSEGV
        // is delivered with no usable stack, and without an alternate stack
        // the handler faults again and the process dies with nothing
        // recorded. Rust's runtime installs the alternate stack itself.
        action.sa_flags = libc::SA_ONSTACK | libc::SA_SIGINFO;
        // SAFETY: `action` is fully initialized before the call. A failure
        // leaves that signal on its original disposition and its table entry
        // inert, which is the same outcome as never listing it.
        unsafe {
            libc::sigemptyset(&mut action.sa_mask);
            libc::sigaction(entry.signo, &action, std::ptr::null_mut());
        }
    }
}

/// The fatal-signal handler.
///
/// # Async-signal-safety audit
///
/// This body may call ONLY: `write(2)`, `sigaction(2)`, `raise(3)`, the
/// previous handler, an atomic load, and a walk over a `'static` slice.
/// Everything it needs is already formatted and already allocated.
///
/// It may NOT call: anything under `std::fs`, `format!`, `String`, `Vec`,
/// `log::`, `println!`, or any mutex. Allocation is the specific hazard:
/// `malloc` is not async-signal-safe, and a heap-corruption `SIGSEGV`
/// arrives with the allocator's own lock held, so any allocation here
/// deadlocks the process instead of reporting the crash.
///
/// # Why it hands the signal on rather than re-raising it
///
/// Recording the signal must not change how the process dies, and Rust's
/// runtime installs its own `SIGSEGV` / `SIGBUS` handler that turns a
/// guard-page fault into "thread 'main' has overflowed its stack". That
/// handler decides by reading `si_addr` out of the ORIGINAL `siginfo_t`, so
/// it has to receive that same `siginfo_t`. Calling it directly delivers it;
/// re-raising the signal manufactures a new one whose `si_addr` is not the
/// guard page, and a stack overflow then dies as a bare `SIGSEGV` with no
/// message.
///
/// `si_code` cannot be used to tell a hardware fault from a sent signal:
/// measured on macOS, `raise(SIGSEGV)` reports `si_code` 2 (`SEGV_ACCERR`),
/// the same value a real access violation reports, so a branch on it
/// misclassifies a sent signal and swallows it.
///
/// When the previous disposition is `SIG_DFL` or `SIG_IGN` there is nothing
/// to call, so the handler restores it and re-raises; the signal is blocked
/// inside its own handler, so it is delivered on return under that restored
/// disposition. Measured against a build with no handler installed, this
/// reproduces the exit status exactly for a stack overflow (134, with the
/// message), a null dereference (139), `SIGABRT` (134), `SIGTERM` (143) and
/// `SIGILL` (132).
///
/// # Why it cannot simply return when it finds no entry
///
/// Returning from a synchronous fault handler resumes at the faulting
/// instruction, so a handler that finds nothing to do turns a crash into an
/// endless fault loop: the process stays alive, the liveness lock stays
/// held, and the supervisor waits forever for a run that will never report.
/// The fall-through therefore restores `SIG_DFL` and re-raises, which is
/// reachable only in the window before `install` publishes the table.
extern "C" fn handler(signo: i32, info: *mut libc::siginfo_t, ctx: *mut libc::c_void) {
    let fd = SIDECAR_FD.load(Ordering::SeqCst);
    let base = INSTALLED.load(Ordering::Acquire);
    let count = if base.is_null() { 0 } else { HANDLED_SIGNALS.len() };
    for i in 0..count {
        // SAFETY: `base` is non-null here, and it points at a leaked array of
        // exactly HANDLED_SIGNALS.len() entries, published once with a
        // release store and never mutated afterwards.
        let entry = unsafe { &*base.add(i) };
        if entry.signo != signo {
            continue;
        }
        if fd >= 0 {
            // SAFETY: a plain write to a descriptor this process owns. A
            // short or failed write is ignored on purpose: there is nothing
            // a handler can do about it, and the server falls back to the
            // launcher's exit status.
            unsafe {
                libc::write(
                    fd,
                    entry.line.as_ptr() as *const libc::c_void,
                    entry.line.len(),
                );
            }
        }
        let previous = entry.previous.sa_sigaction;
        // SAFETY: restoring a disposition captured by the matching
        // `sigaction` call in `install`, then handing the signal to it. The
        // transmute picks the signature the previous disposition was
        // registered with, which SA_SIGINFO names.
        unsafe {
            libc::sigaction(signo, &entry.previous, std::ptr::null_mut());
            if previous == libc::SIG_DFL || previous == libc::SIG_IGN {
                libc::raise(signo);
            } else if entry.previous.sa_flags & libc::SA_SIGINFO != 0 {
                let chained: extern "C" fn(i32, *mut libc::siginfo_t, *mut libc::c_void) =
                    std::mem::transmute(previous);
                chained(signo, info, ctx);
            } else {
                let chained: extern "C" fn(i32) = std::mem::transmute(previous);
                chained(signo);
            }
        }
        return;
    }
    // No entry for this signal, so nothing recorded it and nothing will hand
    // it on. Restore the default disposition and deliver it under that: the
    // signal is blocked inside its own handler, so the raise lands on return.
    // SAFETY: a zeroed sigaction with SIG_DFL is a valid default disposition.
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = libc::SIG_DFL;
        libc::sigemptyset(&mut action.sa_mask);
        libc::sigaction(signo, &action, std::ptr::null_mut());
        libc::raise(signo);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    /// Environment variable naming what the re-executed child should do. The
    /// fault has to happen in a child: installing these handlers in the test
    /// process itself would replace the harness's own dispositions, and a
    /// deliberate crash would take the whole run with it.
    const CHILD_MODE: &str = "PPF_SIGNAL_SIDECAR_TEST_MODE";
    const CHILD_DIR: &str = "PPF_SIGNAL_SIDECAR_TEST_DIR";

    /// Runs in the re-executed child, before any assertion, because the test
    /// harness runs every `#[test]` and the child must not run the parent's.
    fn child_body_if_requested() {
        let mode = match std::env::var(CHILD_MODE) {
            Ok(m) => m,
            Err(_) => return,
        };
        let dir = std::path::PathBuf::from(std::env::var(CHILD_DIR).unwrap());
        install(&dir, "a1b2c3d4e5f6");
        match mode.as_str() {
            "abort" => unsafe {
                libc::raise(libc::SIGABRT);
            },
            "kill" => unsafe {
                libc::raise(libc::SIGKILL);
            },
            "unpublished" => {
                // Recreate exactly what a signal sees in the window between
                // arming a handler and publishing the table: the handler is
                // installed and `INSTALLED` is still null. The process must
                // still die of the signal. Returning instead would resume a
                // synchronous fault at the faulting instruction, so the
                // process would spin in the fault forever with its liveness
                // lock held and the supervisor waiting on a run that can
                // never report.
                INSTALLED.store(std::ptr::null_mut(), Ordering::Release);
                unsafe {
                    libc::raise(libc::SIGABRT);
                }
            }
            "overflow" => {
                // Recurse with a frame the optimizer cannot elide: the array
                // makes each frame large, and reading it back through
                // `black_box` on both the argument and the return value
                // defeats tail-call elimination, so the stack really grows.
                fn recurse(n: u64) -> u64 {
                    let pad = std::hint::black_box([n; 1024]);
                    if n == 0 {
                        return 0;
                    }
                    recurse(pad[0] + 1) + pad[1023]
                }
                println!("{}", recurse(1));
            }
            other => panic!("unknown child mode {other}"),
        }
        std::process::exit(0);
    }

    /// Re-exec this test binary with only this module's child hook selected,
    /// and return the child's output.
    fn run_child(mode: &str, dir: &std::path::Path) -> std::process::Output {
        Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("signal_sidecar::tests::child_entry")
            .arg("--nocapture")
            .env(CHILD_MODE, mode)
            .env(CHILD_DIR, dir)
            .output()
            .unwrap()
    }

    /// The child's entry point. In the parent this is a no-op test; in the
    /// child the environment variable is set and the body never returns.
    #[test]
    fn child_entry() {
        child_body_if_requested();
    }

    #[test]
    fn sidecar_written_on_fatal_signal() {
        let dir = tempfile::tempdir().unwrap();
        let out = run_child("abort", dir.path());
        // The signal must still kill the child exactly as it would with no
        // handler installed: recording the cause may not change the death.
        use std::os::unix::process::ExitStatusExt as _;
        assert_eq!(out.status.signal(), Some(libc::SIGABRT));
        assert_eq!(
            ppf_cts_formats::status::signal_sidecar::read(dir.path(), "a1b2c3d4e5f6"),
            Some("SIGABRT")
        );
    }

    #[test]
    fn sidecar_absent_for_sigkill() {
        // The documented limit, asserted rather than papered over: SIGKILL
        // cannot be caught, so no handler records it and the server has only
        // the launcher's exit status to go on.
        let dir = tempfile::tempdir().unwrap();
        let out = run_child("kill", dir.path());
        use std::os::unix::process::ExitStatusExt as _;
        assert_eq!(out.status.signal(), Some(libc::SIGKILL));
        assert_eq!(
            ppf_cts_formats::status::signal_sidecar::read(dir.path(), "a1b2c3d4e5f6"),
            None
        );
        // `install` creates the file so the handler can write without
        // allocating a path, so what proves nothing was recorded is that the
        // file is EMPTY, not that it is missing.
        assert_eq!(
            std::fs::read(dir.path().join(files::CRASH_SIGNAL)).unwrap(),
            Vec::<u8>::new()
        );
    }

    #[test]
    fn an_unmatched_signal_still_kills_the_process() {
        // The handler is armed before the table is published, so there is a
        // window in which it can run with nothing to match against. The
        // fall-through has to end the process anyway; a handler that merely
        // returns turns a crash into a hang, which is worse than the crash it
        // was recording.
        let dir = tempfile::tempdir().unwrap();
        let out = run_child("unpublished", dir.path());
        use std::os::unix::process::ExitStatusExt as _;
        assert_eq!(
            out.status.signal(),
            Some(libc::SIGABRT),
            "an unmatched signal must terminate, not resume: {:?}",
            out.status
        );
    }

    #[test]
    fn stack_overflow_message_preserved() {
        // The handler hands the signal to the previous disposition instead of
        // re-raising it, so Rust's guard-page handler still receives the
        // original siginfo and still names the overflow.
        let dir = tempfile::tempdir().unwrap();
        let out = run_child("overflow", dir.path());
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("has overflowed its stack"),
            "stack overflow reporting was lost; stderr was: {stderr}"
        );
        // Which signal a guard-page hit raises is the platform's choice:
        // SIGBUS on Darwin, SIGSEGV on Linux. Either is recorded.
        let recorded =
            ppf_cts_formats::status::signal_sidecar::read(dir.path(), "a1b2c3d4e5f6");
        assert!(
            matches!(recorded, Some("SIGSEGV") | Some("SIGBUS")),
            "expected the guard-page signal to be recorded, got {recorded:?}"
        );
    }
}
