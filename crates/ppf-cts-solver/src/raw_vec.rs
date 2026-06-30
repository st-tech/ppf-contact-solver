// File: raw_vec.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

// Shared raw-buffer helpers for the C-vector FFI containers (CVec and CVecVec).
// Both types own heap buffers behind raw pointers and round-trip through serde,
// so the leak / borrow / reclaim boilerplate lives here once instead of being
// copy-pasted (and drifting) across the two files.

/// Consume a `Vec<T>` and hand back its raw parts as (pointer, length, capacity).
/// The buffer is leaked (no drop runs), so the caller now owns the allocation and
/// is responsible for freeing it later via [`reclaim`].
pub fn leak_vec<T>(mut v: Vec<T>) -> (*mut T, usize, usize) {
    let ptr = v.as_mut_ptr();
    let len = v.len();
    let cap = v.capacity();
    std::mem::forget(v);
    (ptr, len, cap)
}

/// Borrow a raw buffer as a slice for the serialize path without taking ownership.
/// Returns an empty slice when the pointer is null or the length is zero, which keeps
/// the guard logic identical for every container and never constructs a Vec/slice from
/// a null or dangling pointer.
///
/// # Safety
/// When `ptr` is non-null and `len > 0`, the caller must guarantee that `ptr` points to
/// `len` initialized, contiguous `T` values that outlive the returned slice.
pub unsafe fn borrow_slice<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if ptr.is_null() || len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    }
}

/// Free a buffer previously leaked via [`leak_vec`] (or any allocation whose len equals
/// its capacity). Returns early when `ptr` is null, since `Vec::from_raw_parts` requires a
/// non-null pointer even at capacity 0. Reconstructs the Vec with len == cap to preserve
/// the container behavior where the stored allocated count doubles as both length and
/// capacity.
///
/// # Safety
/// `ptr` must be either null or a pointer returned by [`leak_vec`] whose allocation has
/// capacity `cap` and has not yet been reclaimed.
pub unsafe fn reclaim<T>(ptr: *mut T, cap: usize) {
    if !ptr.is_null() {
        drop(Vec::from_raw_parts(ptr, cap, cap));
    }
}
