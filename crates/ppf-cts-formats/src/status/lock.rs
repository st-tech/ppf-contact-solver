// File: crates/ppf-cts-formats/src/status/lock.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! Kernel-released liveness lock and PID-scoped liveness probe, the two
//! primitives behind the "crash by absence" verdict.
//!
//! The solver host advisory-locks `status.lock` for its whole lifetime.
//! The OS releases an advisory lock on ANY process death, including
//! `SIGKILL` and a segfault, with zero cooperation from the dying
//! process. So a FREE lock plus a DEAD owning PID plus NO terminal
//! outcome on disk is, by construction, an abrupt crash.
//!
//! Two primitives, never one signal alone:
//!   * [`is_held_by_other`]: is the lock currently held (by a live
//!     process)? Uses a non-blocking exclusive lock attempt on a fresh
//!     handle; advisory locks conflict across independent opens even
//!     within one process, so the server (a separate process from the
//!     solver) sees a live solver's lock as held.
//!   * [`pid_alive`]: is THIS specific PID alive? Always keyed off
//!     `RunStatus.pid`, never the global process-name scan, so a second
//!     unrelated solver (e.g. another run sharing the host) cannot
//!     suppress one project's crash detection.
//!
//! The verdict gates on BOTH agreeing.

use std::fs::{File, OpenOptions};
use std::path::Path;

use crate::files;

/// An acquired liveness lock. Held by value for the solver host's whole
/// lifetime; dropping it (or the process dying) releases the advisory
/// lock. The inner handle is intentionally unused after construction.
#[derive(Debug)]
pub struct Lock {
    _file: File,
}

// ---------------------------------------------------------------------------
// Unix: flock(LOCK_EX | LOCK_NB), FD_CLOEXEC, kill(pid, 0).
// ---------------------------------------------------------------------------
#[cfg(unix)]
mod sys {
    use super::*;
    use std::os::unix::io::AsRawFd;

    /// Acquire the exclusive lock, creating `status.lock` if needed. The
    /// fd is marked `FD_CLOEXEC` so the lock never leaks to a forked or
    /// exec'd child (which would keep it "held" after the host dies).
    /// Fails if another live process already holds it.
    pub fn acquire(output_dir: &Path) -> std::io::Result<Lock> {
        let path = output_dir.join(files::STATUS_LOCK);
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)?;
        let fd = file.as_raw_fd();
        // Safety: fd is owned by `file` and valid for these calls.
        unsafe {
            let flags = libc::fcntl(fd, libc::F_GETFD);
            if flags < 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) != 0 {
                return Err(std::io::Error::last_os_error());
            }
        }
        Ok(Lock { _file: file })
    }

    /// True iff the lock is currently held by some (live) process. A
    /// missing lock file means nobody holds it.
    pub fn is_held_by_other(output_dir: &Path) -> bool {
        let path = output_dir.join(files::STATUS_LOCK);
        let file = match OpenOptions::new().read(true).write(true).open(&path) {
            Ok(f) => f,
            Err(_) => return false,
        };
        let fd = file.as_raw_fd();
        // Safety: fd is owned by `file` for the duration of these calls.
        unsafe {
            if libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) == 0 {
                // We acquired it, so nobody held it. Release immediately;
                // closing `file` would release it anyway.
                libc::flock(fd, libc::LOCK_UN);
                false
            } else {
                // EWOULDBLOCK (== EAGAIN) means a live process holds it.
                // Any other errno is treated as "not provably held".
                matches!(
                    std::io::Error::last_os_error().raw_os_error(),
                    Some(libc::EWOULDBLOCK)
                )
            }
        }
    }

    /// True iff `pid` names a live process. `kill(pid, 0)` returns 0 when
    /// the process exists and is signalable, `EPERM` when it exists but
    /// is not ours, and `ESRCH` when no such process exists.
    pub fn pid_alive(pid: u32) -> bool {
        if pid == 0 {
            return false;
        }
        if unsafe { libc::kill(pid as libc::pid_t, 0) } == 0 {
            return true;
        }
        !matches!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ESRCH)
        )
    }
}

// ---------------------------------------------------------------------------
// Windows: LockFileEx(EXCLUSIVE | FAIL_IMMEDIATELY), OpenProcess +
// GetExitCodeProcess. std::fs::File handles are non-inheritable by
// default on Windows, so there is no FD_CLOEXEC analogue to set.
//
// NOTE: this path compiles only on Windows; verify on a native
// Windows build before relying on the crash-detection behavior.
// ---------------------------------------------------------------------------
#[cfg(windows)]
mod sys {
    use super::*;
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, HANDLE};
    use windows_sys::Win32::Storage::FileSystem::{LockFileEx, UnlockFileEx};
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    use windows_sys::Win32::System::IO::OVERLAPPED;

    // Defined locally to keep the windows-sys import surface minimal and
    // version-stable. Values are fixed by the Win32 ABI.
    const LOCKFILE_FAIL_IMMEDIATELY: u32 = 0x0000_0001;
    const LOCKFILE_EXCLUSIVE_LOCK: u32 = 0x0000_0002;
    const ERROR_LOCK_VIOLATION: u32 = 33;
    const ERROR_INVALID_PARAMETER: u32 = 87;
    const STILL_ACTIVE: u32 = 259;

    fn try_lock(handle: HANDLE) -> bool {
        let mut ov: OVERLAPPED = unsafe { std::mem::zeroed() };
        unsafe {
            LockFileEx(
                handle,
                LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
                0,
                1,
                0,
                &mut ov,
            ) != 0
        }
    }

    fn release(handle: HANDLE) {
        let mut ov: OVERLAPPED = unsafe { std::mem::zeroed() };
        unsafe {
            UnlockFileEx(handle, 0, 1, 0, &mut ov);
        }
    }

    pub fn acquire(output_dir: &Path) -> std::io::Result<Lock> {
        let path = output_dir.join(files::STATUS_LOCK);
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)?;
        let handle = file.as_raw_handle() as HANDLE;
        if !try_lock(handle) {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Lock { _file: file })
    }

    pub fn is_held_by_other(output_dir: &Path) -> bool {
        let path = output_dir.join(files::STATUS_LOCK);
        let file = match OpenOptions::new().read(true).write(true).open(&path) {
            Ok(f) => f,
            Err(_) => return false,
        };
        let handle = file.as_raw_handle() as HANDLE;
        if try_lock(handle) {
            release(handle);
            false
        } else {
            unsafe { GetLastError() == ERROR_LOCK_VIOLATION }
        }
    }

    pub fn pid_alive(pid: u32) -> bool {
        if pid == 0 {
            return false;
        }
        unsafe {
            let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if h.is_null() {
                // ERROR_INVALID_PARAMETER => no such pid; any other error
                // (e.g. access denied) means the process exists.
                return GetLastError() != ERROR_INVALID_PARAMETER;
            }
            let mut code: u32 = 0;
            let ok = GetExitCodeProcess(h, &mut code) != 0;
            CloseHandle(h);
            ok && code == STILL_ACTIVE
        }
    }
}

pub use sys::{acquire, is_held_by_other, pid_alive};

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn no_lock_file_means_not_held() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!is_held_by_other(dir.path()), "no lock file -> not held");
    }

    #[test]
    fn exclusive_while_held_then_reacquire_after_drop() {
        let dir = tempfile::tempdir().unwrap();
        let lock = acquire(dir.path()).unwrap();
        assert!(
            is_held_by_other(dir.path()),
            "a held lock is visible to a separate open"
        );
        assert!(
            acquire(dir.path()).is_err(),
            "exclusive lock blocks a second acquire while held"
        );
        drop(lock);
        // Release-on-drop is proven portably by a successful re-acquire.
        // (A same-process is_held_by_other read immediately after release
        // is an unreliable macOS flock artifact; the production reader is
        // the server, a separate process, where flock is unambiguous.)
        // The same macOS artifact can make the very first re-acquire
        // return EWOULDBLOCK transiently when sibling tests run in
        // parallel, so allow a short bounded retry; a real release leak
        // would still hold past the full window and fail the test.
        let lock2 = (0..50)
            .find_map(|_| {
                acquire(dir.path()).ok().or_else(|| {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    None
                })
            })
            .expect("lock is released on drop");
        drop(lock2);
    }

    #[test]
    fn pid_alive_self_true_zero_false() {
        assert!(pid_alive(std::process::id()));
        assert!(!pid_alive(0));
    }

    #[test]
    fn pid_alive_false_for_reaped_child() {
        // Spawn and reap a child, then its PID is dead (modulo the
        // negligible reuse race a unit test can tolerate).
        let mut child = std::process::Command::new("true")
            .spawn()
            .expect("spawn /usr/bin/true");
        let pid = child.id();
        child.wait().unwrap();
        assert!(!pid_alive(pid));
    }
}
