// File: crates/ppf-cts-compute/src/abi.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The C ABI target: one [`Device`] over any library exporting `be_*`.
//!
//! `crates/ppf-cts-solver/src/kernels/seam/backend_abi.h` is the ABI in
//! full, at version 2. It sits in the neutral kernel tree because it is the
//! seam's own declaration rather than any target's: every target compiles it,
//! and this file is the ONE Rust declaration of it that the header asks for. A
//! second one anywhere would be the mirror pair the header exists to prevent,
//! moved up a level.
//!
//! # There is one of these and not three
//!
//! Nothing below names CUDA, Metal or the host. A library implementing the
//! header is a library this file drives, and which one was loaded is
//! `be_backend_name`, reported through [`DeviceInfo::backend`]. That is what
//! makes the file admissible here at all: it is allocation, transfer, free and
//! launch over a C boundary, and it learns nothing about what is being
//! computed. An argument record crosses it as opaque bytes.
//!
//! # Where the symbols come from, and why the binding is gated
//!
//! No library in this workspace exports `be_*` yet. The extern block is
//! therefore compiled where its symbols resolve, and in exactly two places: a
//! build that links such a library, which its build script announces by
//! emitting `--cfg backend_abi`, and this crate's own test build, where
//! [`test_double`] defines them in Rust.
//!
//! The alternative, compiling the block unconditionally and letting the linker
//! drop the unreferenced calls, was measured to work on this host and was still
//! refused: whether an unreferenced call to an undefined symbol survives into
//! the link is a property of the linker and of how codegen units happened to
//! fall, so it would hold here and could fail on a platform this change cannot
//! be built on. A gate answers the same question with no measurement.
//!
//! The types, the constants and the layout comparison are OUTSIDE that gate,
//! because they are what a stale library is caught by and they must be readable
//! and testable wherever this crate builds.
//!
//! # What is bound, and what is deliberately not
//!
//! The header declares more than [`Device`] asks for. Bound here is what the
//! trait's methods need, plus the three calls that are legal before
//! `be_open` and are what make a stale library fail by name. NOT bound:
//! `be_read_scalars`, `be_encode_fill`, `be_diag_drain`,
//! `be_handle_label`, `be_arena_count`, `be_bytes_used`,
//! `be_encoder_length` and `be_shader_cache_report`. Each lands with the
//! trait method that calls it: a declaration nothing calls is a mirror pair with
//! no reader, which is the shape this ABI exists to remove.
//!
//! `be_kernel_id_by_name` is not bound for a different reason, and it is a
//! conclusion rather than a postponement: the table cross-check at open already
//! establishes the mapping in both directions. The counts are equal and every id
//! carries the same name on both sides, so the mapping is a bijection and asking
//! for the reverse lookup would re-derive what has been proved.

// What the mirrored declarations below need. The binding needs more, and takes
// it inside its own gate rather than here, so that the ungated half of this file
// carries no import that only the gated half reads.
use std::ffi::{c_char, c_void};

use crate::{Extent, Fault, Handle};

// ===========================================================================
// CONSTANTS, MIRRORED FROM THE HEADER
//
// Each is a `#define` there and a `const` here, and the two are compared at run
// time through `be_layout_probe`, which is what catches a driver and a
// library built from different revisions of the header.
// ===========================================================================

/// `PPF_BE_ABI_VERSION`. Bumped on any change to a declaration in the header.
pub const ABI_VERSION: u32 = 2;

/// `PPF_BE_MAX_ARGS_BYTES`. The portable per-argument cap, taken well short of
/// Metal's own 32752 B device cliff, which kills the process with SIGABRT and
/// nothing on stdout or stderr.
pub const MAX_ARGS_BYTES: u32 = 4096;

/// `PPF_BE_MAX_ARENAS`. Metal's buffer binding budget is compiler-enforced at
/// 31 slots, with the argument record at 29 and diagnostics at 30, so arenas
/// take 0 through 28. Every target honors the same number so a handle means one
/// thing everywhere.
pub const MAX_ARENAS: u32 = 29;

/// `PPF_BE_ERROR_DETAIL`, including the terminating NUL.
pub const ERROR_DETAIL: usize = 1024;

/// `BeStatus`, as the discriminants the header fixes.
///
/// A plain integer rather than a Rust enum: a value outside the set is what a
/// library built against a later header returns, and an enum would make reading
/// one undefined behavior rather than a named failure.
pub mod status {
    pub const OK: i32 = 0;
    pub const PLATFORM: i32 = 1;
    pub const MISUSE: i32 = 2;
    pub const NO_KERNEL: i32 = 3;
    pub const BAD_ALLOC: i32 = 4;
    pub const STALE_REGION: i32 = 5;
    pub const DEVICE_FAULT: i32 = 6;
    pub const DEVICE_LOST: i32 = 7;
    pub const NO_DEFERRAL: i32 = 8;
}

/// `BeExtentKind::EXTENT_ELEMENTS`.
pub const EXTENT_ELEMENTS: u32 = 0;
/// `BeExtentKind::EXTENT_GROUPS`. Declared for completeness; [`Extent`]
/// carries no groups variant yet, so nothing here emits it.
pub const EXTENT_GROUPS: u32 = 1;

/// `BeEncodeMode`.
pub const ENCODE_IMMEDIATE: u32 = 0;
/// `BeEncodeMode`.
pub const ENCODE_RECORD: u32 = 1;

// ===========================================================================
// THE SEAM'S DATA VOCABULARY, MIRRORED
//
// `BeHandle` is NOT mirrored here: `crate::Handle` already declares itself
// byte-identical to it, and a second declaration of one type is exactly what
// this ABI exists to prevent.
// ===========================================================================

/// `BeBackend`, opaque.
#[repr(C)]
pub struct BeBackend {
    _private: [u8; 0],
}

/// `BeEncoder`, opaque.
#[repr(C)]
pub struct BeEncoder {
    _private: [u8; 0],
}

/// `BeRegion`, opaque.
#[repr(C)]
pub struct BeRegion {
    _private: [u8; 0],
}

/// `BeLayoutProbe`: the sizes and alignments the LOADED library was compiled
/// against.
#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct BeLayoutProbe {
    pub abi_version: u32,
    pub handle_size: u32,
    pub handle_align: u32,
    pub diag_record_size: u32,
    pub diag_record_align: u32,
    pub extent_size: u32,
    pub error_size: u32,
    pub diag_summary_size: u32,
    pub counters_size: u32,
    pub device_info_size: u32,
    pub config_size: u32,
    pub region_info_size: u32,
    pub shader_cache_report_size: u32,
    pub max_args_bytes: u32,
    pub max_arenas: u32,
    pub reserved: [u32; 3],
}

/// `BeError`. Caller-owned storage, fixed size, never allocated by the
/// library, so there is no failure path inside a failure path.
#[repr(C)]
pub struct BeError {
    pub status: i32,
    pub truncated: i32,
    pub platform_code: i64,
    pub detail: [c_char; ERROR_DETAIL],
}

impl BeError {
    /// A slot to hand to a fallible call. Zeroed rather than uninitialized, so
    /// a library that returns non-OK without filling it reports an empty
    /// message instead of a stack fragment.
    pub fn slot() -> Self {
        BeError {
            status: 0,
            truncated: 0,
            platform_code: 0,
            detail: [0; ERROR_DETAIL],
        }
    }

    /// The message, however far it got. The header guarantees NUL termination,
    /// and a library that broke that guarantee is read only as far as the
    /// buffer, never past it.
    pub fn detail(&self) -> String {
        let bytes = &self.detail;
        let end = bytes.iter().position(|&c| c == 0).unwrap_or(bytes.len());
        // Safety: `bytes[..end]` holds no interior NUL by construction and is
        // inside the caller-owned array.
        let slice =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u8>(), end) };
        let text = String::from_utf8_lossy(slice).into_owned();
        if self.truncated != 0 {
            format!("{text} (truncated)")
        } else {
            text
        }
    }
}

/// `BeDiagRecord`. One device diagnostic record, fixed 32-byte layout shared
/// by every channel.
#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct BeDiagRecord {
    pub assert_id: u32,
    pub file_id: u32,
    pub line: u32,
    pub thread_id: u32,
    pub payload: [f32; 4],
}

/// `BeDiagSummary`. What a boundary reports back: a REPORT, never a verdict.
#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct BeDiagSummary {
    pub failures: u64,
    pub assert_hit: i32,
    pub reserved: i32,
    pub assert_record: BeDiagRecord,
    pub trace_written: u64,
    pub trace_dropped: u64,
    pub trace_count: u32,
    pub reserved2: u32,
}

/// `BeExtent`.
#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct BeExtent {
    pub kind: u32,
    pub count: u32,
    pub threads: u32,
    pub scratch_bytes: u32,
}

/// `BeDeviceInfo`. Reporting and limits only.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BeDeviceInfo {
    pub device_name: [c_char; 128],
    pub max_arenas: u32,
    pub max_arena_bytes: u64,
    pub max_threads_per_group: u32,
    pub max_group_scratch_bytes: u32,
    pub faults_on_oob: i32,
    pub supports_deferred_regions: i32,
}

impl Default for BeDeviceInfo {
    fn default() -> Self {
        BeDeviceInfo {
            device_name: [0; 128],
            max_arenas: 0,
            max_arena_bytes: 0,
            max_threads_per_group: 0,
            max_group_scratch_bytes: 0,
            faults_on_oob: 0,
            supports_deferred_regions: 0,
        }
    }
}

/// `BeCounters`. Counters a GATE asserts on, not diagnostics a human reads.
#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct BeCounters {
    pub syncs: u64,
    pub dispatches: u64,
    pub fills: u64,
    pub regions_recorded: u64,
    pub regions_deferred: u64,
    pub regions_fallback: u64,
    pub replays: u64,
    pub replay_repeats: u64,
    pub bytes_uploaded: u64,
    pub bytes_downloaded: u64,
}

/// `BeRegionInfo`.
#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct BeRegionInfo {
    pub deferred: i32,
    pub dispatch_count: u32,
    pub fill_count: u32,
    pub allocator_generation: u64,
}

/// `BeShaderCacheReport`. Mirrored because its size is one of the fields
/// `be_layout_probe` compares, not because anything here reads it yet.
#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct BeShaderCacheReport {
    pub libraries: u32,
    pub libraries_prebuilt: u32,
    pub caches_loaded: u32,
    pub caches_started: u32,
    pub caches_written: u32,
    pub pipeline_hits: u32,
    pub pipeline_misses: u32,
    pub pipelines_uncached: u32,
    pub compile_ms: f64,
    pub prebuilt_load_ms: f64,
    pub cache_open_ms: f64,
    pub pipeline_ms: f64,
    pub serialize_ms: f64,
}

/// `BeLogFn`. Must not unwind and must not call back into the library.
pub type BeLogFn = extern "C" fn(context: *mut c_void, level: i32, message: *const c_char);

/// `BeBackendConfig`.
#[repr(C)]
pub struct BeBackendConfig {
    pub abi_version: u32,
    pub poison_byte: i32,
    pub require_deferred_regions: i32,
    pub diag_ring_slots: u32,
    pub pipeline_cache_dir: *const c_char,
    pub library_dir: *const c_char,
    pub log: Option<BeLogFn>,
    pub log_context: *mut c_void,
}

// The five the header itself asserts, against the same literals, so a
// disagreement between the two compilations is a build failure on whichever
// side changed. The run-time half is `be_layout_probe`, which catches the
// case a compile-time assertion cannot: two builds that each agree with their
// own header and not with each other.
const _: () = assert!(std::mem::size_of::<Handle>() == 16);
const _: () = assert!(std::mem::size_of::<BeDiagRecord>() == 32);
const _: () = assert!(std::mem::size_of::<BeExtent>() == 16);
const _: () = assert!(std::mem::size_of::<BeDiagSummary>() == 72);
const _: () = assert!(std::mem::size_of::<BeCounters>() == 80);

// ===========================================================================
// THE LAYOUT CROSS-CHECK
// ===========================================================================

/// The probe this build expects a library to return.
///
/// Every field is read from Rust's own `size_of` and `align_of` rather than
/// written out, so a mirror type edited above moves the expectation with it and
/// there is nothing to keep in step by hand.
pub fn expected_layout() -> BeLayoutProbe {
    BeLayoutProbe {
        abi_version: ABI_VERSION,
        handle_size: std::mem::size_of::<Handle>() as u32,
        handle_align: std::mem::align_of::<Handle>() as u32,
        diag_record_size: std::mem::size_of::<BeDiagRecord>() as u32,
        diag_record_align: std::mem::align_of::<BeDiagRecord>() as u32,
        extent_size: std::mem::size_of::<BeExtent>() as u32,
        error_size: std::mem::size_of::<BeError>() as u32,
        diag_summary_size: std::mem::size_of::<BeDiagSummary>() as u32,
        counters_size: std::mem::size_of::<BeCounters>() as u32,
        device_info_size: std::mem::size_of::<BeDeviceInfo>() as u32,
        config_size: std::mem::size_of::<BeBackendConfig>() as u32,
        region_info_size: std::mem::size_of::<BeRegionInfo>() as u32,
        shader_cache_report_size: std::mem::size_of::<BeShaderCacheReport>() as u32,
        max_args_bytes: MAX_ARGS_BYTES,
        max_arenas: MAX_ARENAS,
        reserved: [0; 3],
    }
}

/// Compares a library's probe against this build's, and names the FIRST field
/// that disagrees.
///
/// A mismatch is never recoverable and must never be tolerated: a wrong
/// `BeHandle` size is a wrong arena id, and a wrong arena id is a silent wrong
/// answer with plausible floats on a target that never faults. Naming one field
/// rather than printing both structs is what makes the report actionable, since
/// the field that moved is the one whose declaration diverged.
///
/// `reserved` is not compared. It is padding for a later revision, so a library
/// that filled it is not disagreeing about anything this build reads.
pub fn compare_layout(probe: &BeLayoutProbe) -> Result<(), String> {
    let want = expected_layout();
    let fields: [(&str, u32, u32); 15] = [
        ("abi_version", want.abi_version, probe.abi_version),
        ("handle_size", want.handle_size, probe.handle_size),
        ("handle_align", want.handle_align, probe.handle_align),
        ("diag_record_size", want.diag_record_size, probe.diag_record_size),
        ("diag_record_align", want.diag_record_align, probe.diag_record_align),
        ("extent_size", want.extent_size, probe.extent_size),
        ("error_size", want.error_size, probe.error_size),
        ("diag_summary_size", want.diag_summary_size, probe.diag_summary_size),
        ("counters_size", want.counters_size, probe.counters_size),
        ("device_info_size", want.device_info_size, probe.device_info_size),
        ("config_size", want.config_size, probe.config_size),
        ("region_info_size", want.region_info_size, probe.region_info_size),
        (
            "shader_cache_report_size",
            want.shader_cache_report_size,
            probe.shader_cache_report_size,
        ),
        ("max_args_bytes", want.max_args_bytes, probe.max_args_bytes),
        ("max_arenas", want.max_arenas, probe.max_arenas),
    ];
    for (name, expected, reported) in fields {
        if expected != reported {
            return Err(format!(
                "the loaded backend library reports {name} = {reported} and this \
                 build was compiled against {expected}. The two were built from \
                 different revisions of backend_abi.h"
            ));
        }
    }
    Ok(())
}

/// The ABI spelling of an extent.
///
/// Every field is stated, including the two an ELEMENTS launch does not use,
/// because the library reads a whole `BeExtent` and a left-over group width is
/// a launch geometry nobody asked for.
pub fn extent_to_abi(extent: Extent) -> BeExtent {
    match extent {
        Extent::Elements { count } => BeExtent {
            kind: EXTENT_ELEMENTS,
            count,
            threads: 0,
            scratch_bytes: 0,
        },
        Extent::Groups {
            groups,
            threads,
            scratch_bytes,
        } => BeExtent {
            kind: EXTENT_GROUPS,
            count: groups,
            threads,
            scratch_bytes,
        },
    }
}

/// A non-OK status, as the driver reads it.
///
/// `call` names the ABI function, so a report says where the boundary refused
/// rather than only what it said. A status outside the header's set is reported
/// as itself: a library built against a later header is a real possibility and
/// naming the number is more useful than mapping it onto a neighbor.
pub fn fault_from(code: i32, err: &BeError, call: &'static str) -> Fault {
    let detail = err.detail();
    let named = match code {
        status::PLATFORM => "the platform refused the call",
        status::MISUSE => "a precondition of the ABI was violated, which is a driver defect",
        status::NO_KERNEL => "this library carries no such kernel",
        status::BAD_ALLOC => "the allocator refused the request on its own terms",
        status::STALE_REGION => "the region names allocations that have moved",
        status::DEVICE_FAULT => "the platform aborted the work and the context is unusable",
        status::DEVICE_LOST => "an earlier call faulted and the context has not been reopened",
        status::NO_DEFERRAL => "this library cannot record a deferred region",
        _ => "the library returned a status this build does not know",
    };
    Fault::Platform {
        call,
        detail: if detail.is_empty() {
            format!("status {code}, {named}")
        } else {
            format!("status {code}, {named}: {detail}")
        },
    }
}

// ===========================================================================
// THE BINDING
// ===========================================================================

/// The extern block and the [`Device`] over it.
///
/// Gated for the reason the module comment gives: nothing in this workspace
/// exports these symbols yet, so the block is compiled where they resolve.
#[cfg(any(backend_abi, test))]
mod bound {
    use super::*;

    extern "C" {
        // Callable before `be_open`, and the whole of what makes a stale
        // library fail by name rather than as a bus error.
        pub fn be_abi_version() -> u32;
        pub fn be_backend_name() -> *const c_char;
        pub fn be_layout_probe(out: *mut BeLayoutProbe);

        // Lifecycle.
        pub fn be_open(
            config: *const BeBackendConfig,
            out: *mut *mut BeBackend,
            err: *mut BeError,
        ) -> i32;
        pub fn be_close(be: *mut BeBackend);

        // Identity and limits.
        pub fn be_info(be: *mut BeBackend, out: *mut BeDeviceInfo);

        // The kernel table.
        pub fn be_kernel_count(be: *mut BeBackend) -> u32;
        pub fn be_kernel_name(be: *mut BeBackend, kernel_id: u32) -> *const c_char;
        pub fn be_kernel_args_bytes(be: *mut BeBackend, kernel_id: u32) -> u32;
        pub fn be_kernel_present(be: *mut BeBackend, kernel_id: u32) -> i32;
        pub fn be_prepare_kernels(
            be: *mut BeBackend,
            ids: *const u32,
            count: u32,
            err: *mut BeError,
        ) -> i32;

        // Memory.
        pub fn be_alloc(
            be: *mut BeBackend,
            count: usize,
            elem_size: usize,
            align: usize,
            label: *const c_char,
            out: *mut Handle,
            err: *mut BeError,
        ) -> i32;
        pub fn be_grow(
            be: *mut BeBackend,
            handle: *mut Handle,
            new_count: usize,
            elem_size: usize,
            align: usize,
            err: *mut BeError,
        ) -> i32;
        pub fn be_free(be: *mut BeBackend, handle: *mut Handle, err: *mut BeError) -> i32;
        pub fn be_write(
            be: *mut BeBackend,
            handle: Handle,
            byte_offset: usize,
            src: *const c_void,
            bytes: usize,
            err: *mut BeError,
        ) -> i32;
        pub fn be_read(
            be: *mut BeBackend,
            handle: Handle,
            byte_offset: usize,
            dst: *mut c_void,
            bytes: usize,
            err: *mut BeError,
        ) -> i32;
        pub fn be_copy(
            be: *mut BeBackend,
            dst: Handle,
            dst_byte_offset: usize,
            src: Handle,
            src_byte_offset: usize,
            bytes: usize,
            err: *mut BeError,
        ) -> i32;
        /// The host address of a window inside a live block, where the
        /// library has one.
        ///
        /// A NULL answer with `STATUS_OK` is what a target whose device memory
        /// the host cannot address reports, and it is a fact about the platform
        /// rather than a failure: the copy path through `be_write` and
        /// `be_read` is what serves such a target. The header states the two
        /// conditions a library accepts by returning non-NULL, and both are
        /// requirements on the library rather than advice to this caller.
        ///
        /// A REFUSAL IS A REFUSAL ON EVERY LIBRARY. The handle and the window
        /// are checked whether or not a pointer comes back, so an invented
        /// handle and a window past the block are an `Err` on a target that
        /// serves no view exactly as they are on one that serves a view. A
        /// caller therefore never has to ask which target it is on to know what
        /// a refusal means.
        pub fn be_host_ptr(
            be: *mut BeBackend,
            handle: Handle,
            byte_offset: usize,
            bytes: usize,
            out: *mut *mut c_void,
            err: *mut BeError,
        ) -> i32;
        pub fn be_allocator_generation(be: *mut BeBackend) -> u64;
        pub fn be_bytes_reserved(be: *mut BeBackend) -> u64;

        // Execution.
        pub fn be_encode_begin(
            be: *mut BeBackend,
            region: *const c_char,
            mode: u32,
            out: *mut *mut BeEncoder,
            err: *mut BeError,
        ) -> i32;
        /// The DEVICE-SIDE fill, which the header has always carried and
        /// this crate never called.
        ///
        /// Without it `Device::fill_zero` falls through to its default, which
        /// copies HOST zeros in 64 KB chunks: measured on `drape`, that was
        /// about four fifths of every host-to-device call and byte in the
        /// tree, spread across every buffer that gets sized. A backend fill is
        /// a `cudaMemsetAsync` or a Metal blit and moves nothing across the
        /// bus.
        pub fn be_encode_fill(
            enc: *mut BeEncoder,
            dst: Handle,
            byte_offset: usize,
            bytes: u64,
            value: u8,
            err: *mut BeError,
        ) -> i32;
        pub fn be_encode_dispatch(
            enc: *mut BeEncoder,
            kernel_id: u32,
            extent: *const BeExtent,
            args: *const c_void,
            args_bytes: u32,
            err: *mut BeError,
        ) -> i32;
        pub fn be_encode_submit(
            enc: *mut BeEncoder,
            out_diag: *mut BeDiagSummary,
            err: *mut BeError,
        ) -> i32;
        pub fn be_encode_record(
            enc: *mut BeEncoder,
            out: *mut *mut BeRegion,
            err: *mut BeError,
        ) -> i32;
        pub fn be_encode_abandon(enc: *mut BeEncoder);
        pub fn be_region_info(region: *mut BeRegion, out: *mut BeRegionInfo);
        pub fn be_replay(
            be: *mut BeBackend,
            region: *mut BeRegion,
            repeats: u32,
            out_diag: *mut BeDiagSummary,
            err: *mut BeError,
        ) -> i32;
        pub fn be_region_release(be: *mut BeBackend, region: *mut BeRegion);

        // The diagnostic channel. Only the file resolver is bound: the ring
        // drain lands with the trace oracle, which has nothing to compare
        // against while one target implements the trait.
        pub fn be_diag_file_path(
            be: *mut BeBackend,
            file_id: u32,
            out: *mut c_char,
            capacity: usize,
            err: *mut BeError,
        ) -> i32;

        // Counters.
        pub fn be_counters(be: *mut BeBackend, out: *mut BeCounters);
        pub fn be_counters_reset(be: *mut BeBackend);
    }
}

#[cfg(any(backend_abi, test))]
pub use device_impl::{AbiDevice, AbiRegion, OpenConfig};

#[cfg(any(backend_abi, test))]
mod device_impl {
    use std::collections::HashMap;
    use std::ffi::{c_int, CStr, CString};

    use super::bound::*;
    use super::*;
    use crate::{
        AllocLabel, Counters, Device, DeviceInfo, Diag, DiagFailure, Encoder, KernelDecl, KernelId,
    };

    /// Where a library's non-fatal messages go.
    ///
    /// A callback rather than a default, because the obvious default is a
    /// defect: anything this process writes to stderr lands in the session's
    /// error.log and the frontend reports a run with a non-empty error.log as a
    /// failure, so "the pipeline archive was stale, I recompiled" would turn a
    /// slow start into a reported crash. The driver decides; the library only
    /// chooses whether a message is worth emitting.
    pub type LogSink = fn(level: i32, message: &str);

    /// Boxed so its address is stable for the life of the backend, which is
    /// what `BeBackendConfig::log_context` requires.
    struct LogContext {
        emit: LogSink,
    }

    /// Must not unwind: the header states that no exception or panic may cross
    /// any of its functions, and this is the one direction that runs caller code
    /// on the library's stack.
    extern "C" fn log_trampoline(context: *mut c_void, level: i32, message: *const c_char) {
        if context.is_null() || message.is_null() {
            return;
        }
        let _ = std::panic::catch_unwind(|| {
            // Safety: `context` is the `LogContext` handed to `be_open` and
            // kept alive by the `AbiDevice` until after `be_close`.
            let sink = unsafe { &*context.cast::<LogContext>() };
            // Safety: the header guarantees a NUL-terminated message.
            let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
            (sink.emit)(level, &text);
        });
    }

    /// What the driver states when it opens a library.
    ///
    /// Every field is a DRIVER decision the header deliberately refuses to read
    /// from the environment, so that a run states its own configuration rather
    /// than inheriting it and one gate can drive every target the same way.
    pub struct OpenConfig {
        /// `Some(byte)` fills every fresh allocation with it. `None` leaves
        /// fresh bytes unspecified, which is the production setting.
        ///
        /// NOT `Some(0)` by default, and the reason is the defect it looks like
        /// it prevents: a buffer accumulated into but never cleared is correct
        /// only while the memory it happens to get is still zero, and fresh
        /// device memory frequently is. 0x5a is what the gates use.
        pub poison_byte: Option<u8>,
        /// Makes a library that cannot record a deferred region fail at record
        /// time rather than fall back, so a performance gate gets a named
        /// failure instead of a number that drifts inside the run-to-run
        /// envelope.
        pub require_deferred_regions: bool,
        /// Records the diagnostic ring holds before it starts DROPPING, never
        /// wrapping.
        pub diag_ring_slots: u32,
        pub pipeline_cache_dir: Option<CString>,
        pub library_dir: Option<CString>,
        pub log: LogSink,
    }

    impl OpenConfig {
        /// The production defaults: no poisoning, a fallback permitted and
        /// counted rather than refused, and messages dropped.
        ///
        /// Spelled out rather than derived, because two of the three have a
        /// wrong-looking right answer and `Default` would hide which.
        pub fn new(log: LogSink) -> Self {
            OpenConfig {
                poison_byte: None,
                require_deferred_regions: false,
                diag_ring_slots: 1024,
                pipeline_cache_dir: None,
                library_dir: None,
                log,
            }
        }
    }

    /// A recorded region, plus the label a fault names it by.
    pub struct AbiRegion {
        region: *mut BeRegion,
        label: &'static str,
    }

    impl AbiRegion {
        /// Whether the library realized this in its deferred form.
        ///
        /// ASSERT ON IT, do not merely log it: a region that quietly stopped
        /// being deferred still produces bit-identical numbers and shows up only
        /// as a slowdown, and a slowdown is inside this project's measured
        /// run-to-run envelope.
        pub fn info(&self) -> BeRegionInfo {
            let mut out = BeRegionInfo::default();
            // Safety: the region is live until `Device::release`.
            unsafe { be_region_info(self.region, &mut out) };
            out
        }
    }

    /// A [`Device`] over a linked `be_*` library.
    pub struct AbiDevice {
        backend: *mut BeBackend,
        info: DeviceInfo,
        /// The CALLER's declaration table, cross-checked against the library's
        /// at open. Both are generated, and both can be generated from
        /// different trees, which is the disagreement the check exists for.
        table: &'static [KernelDecl],
        /// Ids the library carries no implementation for. The DRIVER maps these
        /// to named features and refuses a scene by name and count; the library
        /// never decides what to refuse.
        missing: Vec<KernelId>,
        /// NUL-terminated copies of the `&'static str` labels the seam passes,
        /// because the ABI takes a `const char *` that must outlive the backend.
        /// The `CString`s own stable heap buffers, so growing this vector never
        /// moves a string the library is holding.
        labels: Vec<CString>,
        label_index: HashMap<&'static str, usize>,
        /// Kept alive because `be_open` was given its address.
        log_context: *mut LogContext,
    }

    // THE HANDLES ARE PROCESS-OWNED, NOT THREAD-OWNED, and this is where that
    // requirement on a backend library stops being implicit.
    //
    // It is forced by the driver's shape rather than by any wish to run the
    // seam concurrently: the device lives in a `static Mutex<Option<..>>`, a
    // `static` must be `Sync`, and `Mutex<T>: Sync` requires `T: Send`. So even
    // a driver that only ever touches it from one thread needs the bound, and
    // a build without it does not compile.
    //
    // Safety: what the struct owns across the seam is `*mut BeBackend` and a
    // `*mut LogContext`, both opaque handles the library returned and only this
    // value ever passes back. `backend_abi.h` requires a backend to accept its
    // own handles from whichever thread holds them, and the mutex serializes
    // every call, so no two threads are ever inside the library at once. That
    // is the same contract CUDA's runtime API and Metal's device objects
    // already meet.
    //
    // NOT `Sync`, deliberately. Sync would say two threads may call the library
    // AT ONCE, which the ABI does not promise and which no caller here needs.
    unsafe impl Send for AbiDevice {}

    impl AbiDevice {
        /// Opens the library and cross-checks it against this build.
        ///
        /// FOUR CHECKS BEFORE ANY WORK IS DONE, in the order a wrong answer gets
        /// quieter: the ABI version, the layout probe, the kernel table, and the
        /// per-id presence that becomes [`Device::missing`]. The first two are
        /// legal before `be_open` precisely so a library built from another
        /// revision of the header cannot be opened at all.
        ///
        /// # Safety
        /// The linked library must implement `backend_abi.h`, and `table`
        /// must be the caller's generated declaration table for the same tree
        /// the library's kernels were generated from. The second is not assumed:
        /// it is what the cross-check below establishes.
        pub unsafe fn open(
            table: &'static [KernelDecl],
            config: &OpenConfig,
        ) -> Result<Self, Fault> {
            let reported = be_abi_version();
            if reported != ABI_VERSION {
                return Err(Fault::Platform {
                    call: "be_abi_version",
                    detail: format!(
                        "the loaded backend library implements ABI version \
                         {reported} and this build was compiled against \
                         {ABI_VERSION}"
                    ),
                });
            }
            let mut probe = BeLayoutProbe::default();
            be_layout_probe(&mut probe);
            if let Err(detail) = compare_layout(&probe) {
                return Err(Fault::Platform {
                    call: "be_layout_probe",
                    detail,
                });
            }

            let log_context = Box::into_raw(Box::new(LogContext { emit: config.log }));
            let raw = BeBackendConfig {
                abi_version: ABI_VERSION,
                poison_byte: config.poison_byte.map_or(-1, i32::from),
                require_deferred_regions: i32::from(config.require_deferred_regions),
                diag_ring_slots: config.diag_ring_slots,
                pipeline_cache_dir: config
                    .pipeline_cache_dir
                    .as_ref()
                    .map_or(std::ptr::null(), |p| p.as_ptr()),
                library_dir: config
                    .library_dir
                    .as_ref()
                    .map_or(std::ptr::null(), |p| p.as_ptr()),
                log: Some(log_trampoline),
                log_context: log_context.cast::<c_void>(),
            };
            let mut backend: *mut BeBackend = std::ptr::null_mut();
            let mut err = BeError::slot();
            let code = be_open(&raw, &mut backend, &mut err);
            if code != status::OK || backend.is_null() {
                drop(Box::from_raw(log_context));
                return Err(fault_from(code, &err, "be_open"));
            }

            let mut device = AbiDevice {
                backend,
                info: DeviceInfo {
                    backend: "unopened",
                    device_name: String::new(),
                    max_arenas: 0,
                    faults_on_oob: false,
                    supports_deferred_regions: false,
                },
                table,
                missing: Vec::new(),
                labels: Vec::new(),
                label_index: HashMap::new(),
                log_context,
            };
            device.info = device.read_info();
            device.missing = device.cross_check_table()?;
            Ok(device)
        }

        /// The name the library reports for itself, which is the only place a
        /// target is named on this path.
        unsafe fn read_info(&self) -> DeviceInfo {
            let mut raw = BeDeviceInfo::default();
            be_info(self.backend, &mut raw);
            let name = be_backend_name();
            let backend: &'static str = if name.is_null() {
                "unnamed"
            } else {
                // Safety: the header states the pointer is a string literal in
                // the library and lives for the life of the process.
                CStr::from_ptr(name).to_str().unwrap_or("unnamed")
            };
            DeviceInfo {
                backend,
                device_name: CStr::from_ptr(raw.device_name.as_ptr())
                    .to_string_lossy()
                    .into_owned(),
                max_arenas: raw.max_arenas,
                faults_on_oob: raw.faults_on_oob != 0,
                supports_deferred_regions: raw.supports_deferred_regions != 0,
            }
        }

        /// Establishes that the caller's table and the library's are the same
        /// table, and returns the ids the library carries no implementation for.
        ///
        /// THE NAME CHECK IS THE ONE THAT MATTERS. Both tables are generated,
        /// and two trees generate two orders, so an id that names one kernel
        /// here and another there is a dispatch of the wrong kernel with the
        /// right bytes: correct types, plausible numbers, nothing to see. The
        /// size check catches the narrower case of one record having grown a
        /// field.
        unsafe fn cross_check_table(&self) -> Result<Vec<KernelId>, Fault> {
            let theirs = be_kernel_count(self.backend) as usize;
            if theirs != self.table.len() {
                return Err(Fault::Platform {
                    call: "be_kernel_count",
                    detail: format!(
                        "the library carries {theirs} kernels and this build \
                         declares {}. Ids are dense, so the two tables cannot be \
                         the same table",
                        self.table.len()
                    ),
                });
            }
            let mut missing = Vec::new();
            for decl in self.table {
                let id = decl.id.0 as u32;
                let name = be_kernel_name(self.backend, id);
                let theirs = if name.is_null() {
                    return Err(Fault::Platform {
                        call: "be_kernel_name",
                        detail: format!("the library names no kernel at id {id}"),
                    });
                } else {
                    CStr::from_ptr(name).to_string_lossy()
                };
                if theirs != decl.name {
                    return Err(Fault::Platform {
                        call: "be_kernel_name",
                        detail: format!(
                            "id {id} is {} in this build and {theirs} in the \
                             library, so a dispatch would run the wrong kernel \
                             with the right bytes",
                            decl.name
                        ),
                    });
                }
                let bytes = be_kernel_args_bytes(self.backend, id);
                if bytes != u32::from(decl.args_bytes) {
                    return Err(Fault::Platform {
                        call: "be_kernel_args_bytes",
                        detail: format!(
                            "the record for {} is {bytes} bytes in the library \
                             and {} bytes in this build",
                            decl.name, decl.args_bytes
                        ),
                    });
                }
                if be_kernel_present(self.backend, id) == 0 {
                    missing.push(decl.id);
                }
            }
            Ok(missing)
        }

        /// A NUL-terminated copy of a `&'static str`, interned so one label
        /// costs one allocation for the life of the backend.
        fn c_label(&mut self, label: &'static str) -> *const c_char {
            let index = match self.label_index.get(label) {
                Some(&index) => index,
                None => {
                    // A NUL inside a Rust label would truncate silently on the
                    // C side, so it is replaced rather than carried: the label
                    // is a name in a report, and a report naming half a label is
                    // worse than one naming an escaped label.
                    let owned = CString::new(label.replace('\0', "\\0"))
                        .expect("the interior NULs were removed above");
                    self.labels.push(owned);
                    let index = self.labels.len() - 1;
                    self.label_index.insert(label, index);
                    index
                }
            };
            self.labels[index].as_ptr()
        }

        /// Distinct labels interned so far, which is one per allocation site
        /// and one per region name rather than one per call.
        ///
        /// It is here because the ABI requires a label to outlive the backend
        /// and Rust's `&'static str` is not NUL-terminated, so the copies have
        /// to live somewhere; exposing the count is what lets a test show that
        /// somewhere is bounded by the SITES rather than by the calls.
        pub fn label_count(&self) -> usize {
            self.labels.len()
        }

        /// One boundary's diagnostic summary, as the driver reads it.
        ///
        /// The library owns the TRANSPORT, which includes resolving whatever the
        /// channel recorded into a file name, so a driver never learns whether
        /// that was a `#line` injection, a compile-time hash or a pointer.
        fn diag_from(&self, summary: &BeDiagSummary) -> Diag {
            if summary.failures == 0 && summary.assert_hit == 0 {
                return Diag::default();
            }
            let first = if summary.assert_hit != 0 {
                Some(DiagFailure {
                    file: self.diag_file(summary.assert_record.file_id),
                    line: summary.assert_record.line,
                    payload: summary.assert_record.payload,
                })
            } else {
                // A count with no claimant has nothing to name, and inventing a
                // location for it would be worse than reporting what the channel
                // actually recorded.
                None
            };
            Diag {
                failures: summary.failures,
                first,
            }
        }

        fn diag_file(&self, file_id: u32) -> String {
            let mut buffer = [0 as c_char; 512];
            let mut err = BeError::slot();
            // Safety: the buffer is caller-owned and its capacity is stated.
            let code = unsafe {
                be_diag_file_path(
                    self.backend,
                    file_id,
                    buffer.as_mut_ptr(),
                    buffer.len(),
                    &mut err,
                )
            };
            if code != status::OK {
                return format!("<file id {file_id}>");
            }
            // Safety: a successful call NUL-terminates within the capacity.
            unsafe { CStr::from_ptr(buffer.as_ptr()) }
                .to_string_lossy()
                .into_owned()
        }
    }

    impl std::fmt::Debug for AbiDevice {
        /// The library's own name and how much of the caller's table it carries.
        ///
        /// Written out rather than derived because the fields are raw pointers,
        /// and a report naming an address says nothing a reader can act on.
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "{} backend on {}, {} kernels declared, {} not carried",
                self.info.backend,
                self.info.device_name,
                self.table.len(),
                self.missing.len()
            )
        }
    }

    impl Drop for AbiDevice {
        fn drop(&mut self) {
            // Safety: the backend was opened by `open` and is closed once. The
            // log context is released only afterwards, because the library may
            // emit while closing.
            unsafe {
                be_close(self.backend);
                drop(Box::from_raw(self.log_context));
            }
        }
    }

    /// Appends work to the region being built.
    ///
    /// It holds the encoder and the caller's table and nothing else. It cannot
    /// read and cannot allocate, which is not a check here but a consequence of
    /// what [`Encoder`] carries: a host read inside a region is a host
    /// dependency no platform can express inside a captured graph or a committed
    /// command buffer, and an allocation inside one is what CUDA's capture
    /// forbids.
    pub struct AbiEncoder {
        encoder: *mut BeEncoder,
        table: &'static [KernelDecl],
        label: &'static str,
    }

    impl Encoder for AbiEncoder {
        unsafe fn dispatch_raw(
            &mut self,
            kernel: KernelId,
            extent: Extent,
            args: *const u8,
            args_bytes: usize,
        ) -> Result<(), Fault> {
            let decl = self
                .table
                .get(kernel.0 as usize)
                .ok_or(Fault::MissingKernel { kernel })?;
            if args_bytes != decl.args_bytes as usize {
                return Err(Fault::Shape {
                    kernel: decl.name,
                    detail: format!(
                        "the argument record is {args_bytes} bytes and the \
                         declaration says {}",
                        decl.args_bytes
                    ),
                });
            }
            if args_bytes > MAX_ARGS_BYTES as usize {
                return Err(Fault::Shape {
                    kernel: decl.name,
                    detail: format!(
                        "the record is {args_bytes} bytes, over the \
                         {MAX_ARGS_BYTES} the ABI carries"
                    ),
                });
            }
            // A RECORD NAMING BUFFERS BY HOST ADDRESS CANNOT BE BOUND BY A
            // DEVICE, and this is the earliest point at which that can be said.
            // A `HostRef` is an address; on a target with its own memory it is a
            // wild pointer, and on one that never faults it is plausible floats
            // with status Completed. Refusing by name here is what turns the
            // handle migration from a thing to remember into a precondition the
            // seam states. It costs nothing once a record's references are
            // handles, because then this list is empty.
            if !decl.host_refs.is_empty() {
                return Err(Fault::Shape {
                    kernel: decl.name,
                    detail: format!(
                        "the record names {} buffers by host address. A backend \
                         library resolves an (arena, offset) handle and cannot \
                         resolve an address, so this record has to carry handles \
                         before it can be dispatched here",
                        decl.host_refs.len()
                    ),
                });
            }
            let extent = extent_to_abi(extent);
            let mut err = BeError::slot();
            let code = be_encode_dispatch(
                self.encoder,
                u32::from(kernel.0),
                &extent,
                args.cast::<c_void>(),
                args_bytes as u32,
                &mut err,
            );
            if code != status::OK {
                return Err(fault_from(code, &err, "be_encode_dispatch"));
            }
            Ok(())
        }

        unsafe fn fill_raw(
            &mut self,
            handle: Handle,
            byte_offset: usize,
            bytes: usize,
            value: u8,
        ) -> Result<(), Fault> {
            if bytes == 0 {
                return Ok(());
            }
            let mut err = BeError::slot();
            // Safety: the caller owns the allocation for the region's lifetime,
            // and the encoder is open until this region is submitted.
            let code = unsafe {
                be_encode_fill(self.encoder, handle, byte_offset, bytes as u64, value, &mut err)
            };
            if code != status::OK {
                return Err(fault_from(code, &err, "be_encode_fill"));
            }
            Ok(())
        }

        fn region(&self) -> &'static str {
            self.label
        }
    }

    impl Device for AbiDevice {
        /// Zero a range ON THE DEVICE, not by copying host zeros to it.
        ///
        /// THE DEFAULT IMPLEMENTATION IS A HOST COPY IN 64 KB CHUNKS, and
        /// nothing overrode it, so every `Buffer::size` on a real backend
        /// walked its whole allocation across the bus. Measured on `drape` at
        /// three frames: of 29,248 host-to-device calls and 2,592 MB, the
        /// buffers being SIZED accounted for about four fifths, every one of
        /// them at ~62.5 KB a call, which is the chunk. `be_encode_fill` has
        /// been in the header and implemented by the CUDA backend the whole
        /// time; this crate mentioned it in a doc comment and never called it.
        ///
        /// AN IMMEDIATE ENCODER CARRYING ONE FILL is what the header prescribes
        /// for a standalone one, and the note there is worth repeating: on
        /// Metal the fill must ride the same command buffer rather than take
        /// its own and wait, or it is a hidden host round trip.
        fn fill_zero(&mut self, handle: Handle, bytes: usize) -> Result<(), Fault> {
            if bytes == 0 {
                return Ok(());
            }
            let label = self.c_label("fill_zero");
            let mut encoder: *mut BeEncoder = std::ptr::null_mut();
            let mut err = BeError::slot();
            // Safety: the label outlives the encoder, as the header requires.
            let code = unsafe {
                be_encode_begin(self.backend, label, ENCODE_IMMEDIATE, &mut encoder, &mut err)
            };
            if code != status::OK {
                return Err(fault_from(code, &err, "be_encode_begin"));
            }
            // Safety: the handle names a live allocation of at least `bytes`,
            // and the encoder is open.
            let code =
                unsafe { be_encode_fill(encoder, handle, 0, bytes as u64, 0, &mut err) };
            if code != status::OK {
                // Safety: the encoder is open and has not been closed.
                unsafe { be_encode_abandon(encoder) };
                return Err(fault_from(code, &err, "be_encode_fill"));
            }
            let mut summary = BeDiagSummary::default();
            // Safety: submit closes the encoder whatever it returns, so the
            // pointer is not used again.
            let code = unsafe { be_encode_submit(encoder, &mut summary, &mut err) };
            if code != status::OK {
                return Err(fault_from(code, &err, "be_encode_submit"));
            }
            Ok(())
        }

        type Region = AbiRegion;

        fn info(&self) -> &DeviceInfo {
            &self.info
        }

        fn kernels(&self) -> &'static [KernelDecl] {
            self.table
        }

        fn missing(&self) -> &[KernelId] {
            &self.missing
        }

        fn prepare_kernels(&mut self, ids: &[KernelId]) -> Result<(), Fault> {
            let raw: Vec<u32> = ids.iter().map(|id| u32::from(id.0)).collect();
            let mut err = BeError::slot();
            // Safety: `raw` addresses `raw.len()` ids for the call's duration.
            let code = unsafe {
                be_prepare_kernels(self.backend, raw.as_ptr(), raw.len() as u32, &mut err)
            };
            if code != status::OK {
                return Err(fault_from(code, &err, "be_prepare_kernels"));
            }
            Ok(())
        }

        fn alloc(
            &mut self,
            count: usize,
            elem_size: usize,
            align: usize,
            label: AllocLabel,
        ) -> Result<Handle, Fault> {
            let name = self.c_label(label.0);
            let mut handle = Handle::NONE;
            let mut err = BeError::slot();
            // Safety: the label outlives the backend, and the two out pointers
            // address caller-owned storage.
            let code = unsafe {
                be_alloc(
                    self.backend,
                    count,
                    elem_size,
                    align,
                    name,
                    &mut handle,
                    &mut err,
                )
            };
            if code != status::OK {
                return Err(Fault::Alloc {
                    label,
                    bytes: elem_size.saturating_mul(count),
                    detail: err.detail(),
                });
            }
            Ok(handle)
        }

        fn grow(
            &mut self,
            handle: &mut Handle,
            new_count: usize,
            elem_size: usize,
            align: usize,
        ) -> Result<(), Fault> {
            let mut err = BeError::slot();
            // Safety: the handle is caller-owned and is rewritten in place,
            // which is what makes every other copy of it stale.
            let code = unsafe {
                be_grow(self.backend, handle, new_count, elem_size, align, &mut err)
            };
            if code != status::OK {
                return Err(Fault::Alloc {
                    label: AllocLabel("grow"),
                    bytes: elem_size.saturating_mul(new_count),
                    detail: err.detail(),
                });
            }
            Ok(())
        }

        fn free(&mut self, handle: &mut Handle) -> Result<(), Fault> {
            // A HANDLE THAT ALREADY NAMES NOTHING NEVER REACHES THE LIBRARY. The
            // sentinel is `u32::MAX`, which every backend reads as an arena it
            // did not open, so passing one down would report a fault for a call
            // the seam defines as legal and does nothing.
            if handle.is_none() {
                return Ok(());
            }
            let mut err = BeError::slot();
            // Safety: the handle is caller-owned and is zeroed by the library.
            let code = unsafe { be_free(self.backend, handle, &mut err) };
            if code != status::OK {
                return Err(fault_from(code, &err, "be_free"));
            }
            // **NORMALIZED TO THE SENTINEL, and this is not cosmetic.** The
            // library ZEROES the handle, which is right for the library: its own
            // bounds checks then reject it. But zero is a REAL arena, so
            // `is_none()` on the zeroed handle is false, and the next
            // `Buffer::size` would take the grow path against a block that has
            // just been released. The host target leaves `Handle::NONE`, so this
            // is where a handle freed through a C ABI is given the same spelling
            // rather than a second meaning. Removing this line leaves every
            // other test on every host green, so it is covered by name:
            // `crate::device::assert_free_leaves_the_sentinel` reads the handle
            // back and
            // `assert_sizing_after_a_free_spares_the_first_allocation` reads the
            // block a zeroed handle would have grown.
            *handle = Handle::NONE;
            Ok(())
        }

        /// ON THE DEVICE, through `be_copy`.
        ///
        /// This went through a host bounce until the C ABI grew the verb: a
        /// `read` into a `Vec` and a `write` back out, a download and an upload
        /// to do no work. It was the LARGEST device-to-host row this tree
        /// measured, because the per-Newton-step matrix snapshot in `step.rs`
        /// is a copy: on `drape` it moved 17,510,760 bytes each way, 84 times
        /// in 12 frames, recorded under the label of the buffer it read FROM,
        /// which is why reading the driver for a `download` never found it.
        ///
        /// CUDA takes `cudaMemcpyDeviceToDevice`; Metal memcpys between two
        /// host-visible arenas. Neither counts a transfer, because nothing
        /// crosses a bus.
        fn copy(
            &mut self,
            dst: Handle,
            dst_byte_offset: usize,
            src: Handle,
            src_byte_offset: usize,
            bytes: usize,
        ) -> Result<(), Fault> {
            if bytes == 0 {
                return Ok(());
            }
            let mut err = BeError::slot();
            // Safety: both handles name live blocks and the library checks both
            // windows, as it checks `be_write`'s and `be_read`'s.
            let code = unsafe {
                be_copy(
                    self.backend,
                    dst,
                    dst_byte_offset,
                    src,
                    src_byte_offset,
                    bytes,
                    &mut err,
                )
            };
            if code != status::OK {
                return Err(fault_from(code, &err, "be_copy"));
            }
            Ok(())
        }

        fn write(&mut self, handle: Handle, byte_offset: usize, src: &[u8]) -> Result<(), Fault> {
            let mut err = BeError::slot();
            // Safety: `src` addresses its own length for the call's duration and
            // the window is checked against the block by the library.
            let code = unsafe {
                be_write(
                    self.backend,
                    handle,
                    byte_offset,
                    src.as_ptr().cast::<c_void>(),
                    src.len(),
                    &mut err,
                )
            };
            if code != status::OK {
                return Err(fault_from(code, &err, "be_write"));
            }
            Ok(())
        }

        fn read(&mut self, handle: Handle, byte_offset: usize, dst: &mut [u8]) -> Result<(), Fault> {
            let mut err = BeError::slot();
            // Safety: as `write`, in the other direction.
            let code = unsafe {
                be_read(
                    self.backend,
                    handle,
                    byte_offset,
                    dst.as_mut_ptr().cast::<c_void>(),
                    dst.len(),
                    &mut err,
                )
            };
            if code != status::OK {
                return Err(fault_from(code, &err, "be_read"));
            }
            Ok(())
        }

        /// THE ADDRESS, WHERE THE LIBRARY HAS ONE. A NULL answer is not a
        /// failure: it is what a target whose device memory the host cannot
        /// address reports, and the caller's copy path is what serves it.
        unsafe fn host_view(
            &mut self,
            handle: Handle,
            byte_offset: usize,
            bytes: usize,
        ) -> Result<Option<std::ptr::NonNull<u8>>, Fault> {
            let mut out: *mut c_void = std::ptr::null_mut();
            let mut err = BeError::slot();
            // Safety: the library checks the window as it checks `be_read`'s,
            // and refuses a call made where a view may not be handed out.
            let code = unsafe {
                be_host_ptr(self.backend, handle, byte_offset, bytes, &mut out, &mut err)
            };
            if code != status::OK {
                return Err(fault_from(code, &err, "be_host_ptr"));
            }
            Ok(std::ptr::NonNull::new(out.cast::<u8>()))
        }

        fn allocator_generation(&self) -> u64 {
            // Safety: the backend is live for the life of this value.
            unsafe { be_allocator_generation(self.backend) }
        }

        fn bytes_reserved(&self) -> u64 {
            // Safety: as above.
            unsafe { be_bytes_reserved(self.backend) }
        }

        fn run<F>(&mut self, region: &'static str, body: F) -> Result<Diag, Fault>
        where
            F: FnOnce(&mut dyn Encoder) -> Result<(), Fault>,
        {
            let label = self.c_label(region);
            let mut encoder: *mut BeEncoder = std::ptr::null_mut();
            let mut err = BeError::slot();
            // Safety: the label outlives the encoder, which is what the header
            // requires of it.
            let code = unsafe {
                be_encode_begin(
                    self.backend,
                    label,
                    ENCODE_IMMEDIATE,
                    &mut encoder,
                    &mut err,
                )
            };
            if code != status::OK {
                return Err(fault_from(code, &err, "be_encode_begin"));
            }
            let mut sink = AbiEncoder {
                encoder,
                table: self.table,
                label: region,
            };
            if let Err(fault) = body(&mut sink) {
                // The encoder is closed on every path, including this one. An
                // abandoned encoder is what the header provides for exactly this
                // case, and leaving one open would refuse the next
                // `encode_begin` as misuse.
                // Safety: the encoder is open and has not been closed.
                unsafe { be_encode_abandon(encoder) };
                return Err(fault);
            }
            let mut summary = BeDiagSummary::default();
            // Safety: submit closes the encoder, whatever it returns, so the
            // pointer is not used again.
            let code = unsafe { be_encode_submit(encoder, &mut summary, &mut err) };
            let diag = self.diag_from(&summary);
            if code != status::OK {
                // A FAILING BOUNDARY THAT ALSO RECORDED SOMETHING REPORTS BOTH,
                // and it stays a Platform fault while doing so.
                //
                // The status names the call that reported the failure; the
                // record names the check that failed, its file, its line and
                // four captured values, and it is the one of the two that says
                // WHY. Returning the record INSTEAD would drop the status, the
                // call name and the library's detail, and would change the
                // fault's VARIANT, which callers map to a crash kind: a device
                // fault would start being reported as a failed assertion. So
                // the record is appended to the detail the platform fault
                // already carries.
                let mut fault = fault_from(code, &err, "be_encode_submit");
                if !diag.is_clean() {
                    if let Fault::Platform { detail, .. } = &mut fault {
                        detail.push_str(&format!(
                            "; the device also recorded {} failing check(s)",
                            diag.failures
                        ));
                        if let Some(first) = &diag.first {
                            detail.push_str(&format!(", first at {first}"));
                        }
                    }
                }
                return Err(fault);
            }
            if diag.is_clean() {
                Ok(diag)
            } else {
                Err(Fault::Device { region, diag })
            }
        }

        fn record<F>(&mut self, region: &'static str, body: F) -> Result<AbiRegion, Fault>
        where
            F: FnOnce(&mut dyn Encoder) -> Result<(), Fault>,
        {
            let label = self.c_label(region);
            let mut encoder: *mut BeEncoder = std::ptr::null_mut();
            let mut err = BeError::slot();
            // Safety: as `run`, in the mode the header requires to be stated at
            // OPEN rather than at close, because CUDA must begin stream capture
            // before the first launch is issued.
            let code = unsafe {
                be_encode_begin(self.backend, label, ENCODE_RECORD, &mut encoder, &mut err)
            };
            if code != status::OK {
                return Err(fault_from(code, &err, "be_encode_begin"));
            }
            let mut sink = AbiEncoder {
                encoder,
                table: self.table,
                label: region,
            };
            if let Err(fault) = body(&mut sink) {
                // Safety: the encoder is open and has not been closed.
                unsafe { be_encode_abandon(encoder) };
                return Err(fault);
            }
            let mut out: *mut BeRegion = std::ptr::null_mut();
            // Safety: `encode_record` closes the encoder whatever it returns.
            let code = unsafe { be_encode_record(encoder, &mut out, &mut err) };
            if code != status::OK {
                return Err(fault_from(code, &err, "be_encode_record"));
            }
            Ok(AbiRegion {
                region: out,
                label: region,
            })
        }

        fn replay(&mut self, region: &AbiRegion, repeats: u32) -> Result<Diag, Fault> {
            let mut summary = BeDiagSummary::default();
            let mut err = BeError::slot();
            // Safety: the region is live until `release`.
            let code = unsafe {
                be_replay(self.backend, region.region, repeats, &mut summary, &mut err)
            };
            if code == status::STALE_REGION {
                // The one status that maps onto a fault carrying its own
                // evidence, and the two generations are what say how far apart
                // they are rather than only that they differ.
                return Err(Fault::StaleRegion {
                    region: region.label,
                    recorded: region.info().allocator_generation,
                    now: self.allocator_generation(),
                });
            }
            let diag = self.diag_from(&summary);
            if code != status::OK {
                // The same rule as `submit`: a failing replay that also
                // recorded something reports both, and stays the fault variant
                // its status names.
                let mut fault = fault_from(code, &err, "be_replay");
                if !diag.is_clean() {
                    if let Fault::Platform { detail, .. } = &mut fault {
                        detail.push_str(&format!(
                            "; the device also recorded {} failing check(s)",
                            diag.failures
                        ));
                        if let Some(first) = &diag.first {
                            detail.push_str(&format!(", first at {first}"));
                        }
                    }
                }
                return Err(fault);
            }
            if diag.is_clean() {
                Ok(diag)
            } else {
                Err(Fault::Device {
                    region: region.label,
                    diag,
                })
            }
        }

        fn release(&mut self, region: AbiRegion) {
            // Safety: the region is released once, because it is taken by value.
            unsafe { be_region_release(self.backend, region.region) };
        }

        fn counters(&self) -> Counters {
            let mut raw = BeCounters::default();
            // Safety: `raw` is caller-owned.
            unsafe { be_counters(self.backend, &mut raw) };
            // The ABI carries ten and the seam reads four. The other six are a
            // library's own reporting; a gate asserts on what the trait carries,
            // and adding a field here would be adding one nothing reads.
            Counters {
                syncs: raw.syncs,
                dispatches: raw.dispatches,
                regions_deferred: raw.regions_deferred,
                regions_fallback: raw.regions_fallback,
            }
        }

        fn counters_reset(&mut self) {
            // Safety: the backend is live.
            unsafe { be_counters_reset(self.backend) };
        }
    }

    // `AbiDevice` holds raw pointers, so it is neither `Send` nor `Sync`, and
    // that is left as the compiler's own conclusion rather than asserted: the
    // library's state is not synchronized by anything this file can see, so a
    // device stays on the thread that opened it. It costs nothing, because the
    // driver holds one and drives it from the thread that runs the step loop.

    /// The header spells every status `int32_t`, and a platform where `c_int`
    /// is not that width would reinterpret every one of them.
    const _: () = assert!(std::mem::size_of::<c_int>() == std::mem::size_of::<i32>());
}

// ===========================================================================
// A TEST DOUBLE FOR THE LIBRARY SIDE
// ===========================================================================

/// A Rust implementation of the C ABI, for this crate's own tests.
///
/// **IT IS A TEST DOUBLE AND NOT A BACKEND, and the difference is what makes it
/// admissible.** A stub backend is forbidden outright: one that computes no
/// physics while writing to the path a live session reads is how a run gets fake
/// results from a binary that looks right. This one is compiled under
/// `cfg(test)` alone, so it exists only inside `cargo test` for this crate, it
/// is never linked into any solver binary, and it computes nothing at all. What
/// it does is hold the far side of a boundary so the near side can be exercised:
/// it allocates, copies, counts dispatches and reports what it is told to
/// report.
///
/// It is also what makes the extern block above compile on a host with no GPU
/// and no Mac, which is the property that lets this binding land where it is
/// written rather than where it will first run.
#[cfg(test)]
pub mod test_double {
    // EVERY FUNCTION HERE IS UNSAFE FOR ONE REASON AND THE SAME REASON: it is an
    // entry point of a C ABI, so it dereferences pointers the caller supplies
    // and its preconditions are the ones
    // `crates/ppf-cts-solver/src/kernels/seam/backend_abi.h` states at each
    // declaration. That contract is stated once, there, and a per-function
    // restatement here would be 29 copies of it that can disagree with the
    // header, which is the mirror pair this whole ABI exists to remove.
    #![allow(clippy::missing_safety_doc)]

    use super::*;
    use std::ffi::CStr;
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
    use std::sync::{Mutex, MutexGuard};

    /// The knobs below are process-wide because the pre-open calls take no
    /// backend, so a test that turns one must hold this for its duration.
    /// Poisoning is ignored: a test that panicked while holding it has already
    /// failed, and refusing every later test would report the same defect many
    /// times over.
    static LOCK: Mutex<()> = Mutex::new(());

    /// What `be_abi_version` reports.
    pub static ABI_VERSION_KNOB: AtomicU32 = AtomicU32::new(super::ABI_VERSION);

    /// Which layout-probe field to report wrongly. `NO_CORRUPTION` reports the
    /// truth.
    pub static LAYOUT_KNOB: AtomicU32 = AtomicU32::new(NO_CORRUPTION);
    pub const NO_CORRUPTION: u32 = u32::MAX;
    pub const CORRUPT_HANDLE_SIZE: u32 = 0;
    pub const CORRUPT_MAX_ARENAS: u32 = 1;

    /// How the library's kernel table disagrees with the caller's.
    pub static TABLE_KNOB: AtomicU32 = AtomicU32::new(TABLE_MATCHES);
    pub const TABLE_MATCHES: u32 = 0;
    pub const TABLE_WRONG_COUNT: u32 = 1;
    pub const TABLE_WRONG_NAME: u32 = 2;
    pub const TABLE_WRONG_ARGS_BYTES: u32 = 3;

    /// Whether the double reports a host view of its blocks.
    ///
    /// ABSENT BY DEFAULT, so every assertion written before this knob existed
    /// keeps the meaning it was written with, and the mapped representation is
    /// reached only by a test that asks for it. The double's blocks are host
    /// `Vec<u8>`, so it can serve a REAL view and both representations are
    /// therefore exercised on any host, including one with no GPU.
    pub static HOST_PTR_KNOB: AtomicU32 = AtomicU32::new(HOST_PTR_ABSENT);
    pub const HOST_PTR_ABSENT: u32 = 0;
    pub const HOST_PTR_SERVED: u32 = 1;

    /// Failing threads the next `submit` or `replay` reports. The channel is a
    /// REPORT rather than a verdict, so this leaves the status OK and puts the
    /// count in the summary, which is the distinction the header draws.
    pub static FAILING_THREADS: AtomicU64 = AtomicU64::new(0);

    /// Takes the knobs and returns them to their honest settings first, so a
    /// test never inherits the previous one's.
    pub fn acquire() -> MutexGuard<'static, ()> {
        let guard = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        ABI_VERSION_KNOB.store(super::ABI_VERSION, Ordering::SeqCst);
        LAYOUT_KNOB.store(NO_CORRUPTION, Ordering::SeqCst);
        TABLE_KNOB.store(TABLE_MATCHES, Ordering::SeqCst);
        HOST_PTR_KNOB.store(HOST_PTR_ABSENT, Ordering::SeqCst);
        FAILING_THREADS.store(0, Ordering::SeqCst);
        guard
    }

    /// The two kernels this library claims, and the second is deliberately
    /// absent so `Device::missing` has something to report.
    pub const PRESENT_KERNEL: &str = "stub_present_entry";
    pub const ABSENT_KERNEL: &str = "stub_absent_entry";
    pub const KERNEL_ARGS_BYTES: u32 = 16;

    const PRESENT_C: &CStr = c"stub_present_entry";
    const ABSENT_C: &CStr = c"stub_absent_entry";
    const RENAMED_C: &CStr = c"stub_renamed_entry";
    const DIAG_FILE_C: &CStr = c"stub_kernel.kernel.cpp";

    struct Op {
        kernel: u32,
        count: u32,
        args: Vec<u8>,
    }

    struct Region {
        ops: Vec<Op>,
        generation: u64,
    }

    struct Encoder {
        backend: *mut Backend,
        recording: bool,
        ops: Vec<Op>,
    }

    struct Backend {
        arenas: Vec<Option<Vec<u8>>>,
        generation: u64,
        counters: BeCounters,
        log: Option<BeLogFn>,
        log_context: *mut c_void,
    }

    impl Backend {
        /// Exercises the log callback, which is the one direction that runs
        /// caller code on the library's stack and therefore the one the header
        /// forbids to unwind.
        fn say(&self, message: &[u8]) {
            if let Some(log) = self.log {
                log(self.log_context, 0, message.as_ptr().cast::<c_char>());
            }
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_abi_version() -> u32 {
        ABI_VERSION_KNOB.load(Ordering::SeqCst)
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_backend_name() -> *const c_char {
        c"stub".as_ptr()
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_layout_probe(out: *mut BeLayoutProbe) {
        // A library fills this from its OWN sizes. This one is Rust compiled
        // against the same mirrors, so it reports the truth unless a knob says
        // otherwise, and the knob is what makes the comparison testable at all.
        let mut probe = expected_layout();
        probe.abi_version = ABI_VERSION_KNOB.load(Ordering::SeqCst);
        match LAYOUT_KNOB.load(Ordering::SeqCst) {
            CORRUPT_HANDLE_SIZE => probe.handle_size = 8,
            CORRUPT_MAX_ARENAS => probe.max_arenas = 64,
            _ => {}
        }
        unsafe { *out = probe };
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_open(
        config: *const BeBackendConfig,
        out: *mut *mut BeBackend,
        err: *mut BeError,
    ) -> i32 {
        let config = unsafe { &*config };
        if config.abi_version != super::ABI_VERSION {
            unsafe { (*err).status = status::MISUSE };
            return status::MISUSE;
        }
        let backend = Box::into_raw(Box::new(Backend {
            arenas: Vec::new(),
            generation: 0,
            counters: BeCounters::default(),
            log: config.log,
            log_context: config.log_context,
        }));
        unsafe { (*backend).say(b"the stub library is open\0") };
        unsafe { *out = backend.cast::<BeBackend>() };
        status::OK
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_close(be: *mut BeBackend) {
        if be.is_null() {
            return;
        }
        drop(unsafe { Box::from_raw(be.cast::<Backend>()) });
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_info(_be: *mut BeBackend, out: *mut BeDeviceInfo) {
        let mut info = BeDeviceInfo {
            max_arenas: super::MAX_ARENAS,
            max_arena_bytes: 1 << 30,
            max_threads_per_group: 1024,
            max_group_scratch_bytes: 32768,
            faults_on_oob: 1,
            supports_deferred_regions: 1,
            ..BeDeviceInfo::default()
        };
        for (slot, byte) in info.device_name.iter_mut().zip(b"stub device\0") {
            *slot = *byte as c_char;
        }
        unsafe { *out = info };
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_kernel_count(_be: *mut BeBackend) -> u32 {
        if TABLE_KNOB.load(Ordering::SeqCst) == TABLE_WRONG_COUNT {
            1
        } else {
            2
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_kernel_name(_be: *mut BeBackend, kernel_id: u32) -> *const c_char {
        let renamed = TABLE_KNOB.load(Ordering::SeqCst) == TABLE_WRONG_NAME;
        match kernel_id {
            0 if renamed => RENAMED_C.as_ptr(),
            0 => PRESENT_C.as_ptr(),
            1 => ABSENT_C.as_ptr(),
            _ => std::ptr::null(),
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_kernel_args_bytes(_be: *mut BeBackend, kernel_id: u32) -> u32 {
        if kernel_id > 1 {
            return 0;
        }
        if TABLE_KNOB.load(Ordering::SeqCst) == TABLE_WRONG_ARGS_BYTES {
            KERNEL_ARGS_BYTES + 4
        } else {
            KERNEL_ARGS_BYTES
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_kernel_present(_be: *mut BeBackend, kernel_id: u32) -> i32 {
        i32::from(kernel_id == 0)
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_prepare_kernels(
        _be: *mut BeBackend,
        ids: *const u32,
        count: u32,
        err: *mut BeError,
    ) -> i32 {
        let ids = unsafe { std::slice::from_raw_parts(ids, count as usize) };
        // An id this library cannot supply fails HERE, by name, rather than
        // mid-step or by quietly substituting something else.
        if ids.iter().any(|&id| be_kernel_present(std::ptr::null_mut(), id) == 0) {
            fill(err, status::NO_KERNEL, b"no implementation for a requested id");
            return status::NO_KERNEL;
        }
        status::OK
    }

    fn fill(err: *mut BeError, code: i32, message: &[u8]) {
        if err.is_null() {
            return;
        }
        let slot = unsafe { &mut *err };
        slot.status = code;
        slot.truncated = 0;
        for (target, byte) in slot.detail.iter_mut().zip(message.iter().chain(&[0])) {
            *target = *byte as c_char;
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_alloc(
        be: *mut BeBackend,
        count: usize,
        elem_size: usize,
        align: usize,
        _label: *const c_char,
        out: *mut Handle,
        err: *mut BeError,
    ) -> i32 {
        let backend = unsafe { &mut *be.cast::<Backend>() };
        if align < 4 || !align.is_power_of_two() {
            fill(err, status::BAD_ALLOC, b"alignment below 4 or not a power of two");
            return status::BAD_ALLOC;
        }
        if backend.arenas.len() as u32 >= super::MAX_ARENAS {
            fill(err, status::BAD_ALLOC, b"every arena slot is bound");
            return status::BAD_ALLOC;
        }
        let bytes = elem_size.max(1) * count.max(1);
        backend.arenas.push(Some(vec![0u8; bytes]));
        backend.generation += 1;
        unsafe {
            *out = Handle {
                arena: (backend.arenas.len() - 1) as u32,
                off: 0,
                // ELEMENTS, as on every real target. This stand-in packs one
                // allocation per arena, which a real one does not, but a handle
                // means the same thing here as there or the tests below would be
                // exercising a second vocabulary.
                size: count as u32,
                allocated: count as u32,
            }
        };
        status::OK
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_grow(
        be: *mut BeBackend,
        handle: *mut Handle,
        new_count: usize,
        elem_size: usize,
        _align: usize,
        err: *mut BeError,
    ) -> i32 {
        let backend = unsafe { &mut *be.cast::<Backend>() };
        let current = unsafe { *handle };
        let Some(Some(block)) = backend.arenas.get_mut(current.arena as usize) else {
            fill(err, status::MISUSE, b"handle names no live block");
            return status::MISUSE;
        };
        block.resize(elem_size.max(1) * new_count.max(1), 0);
        backend.generation += 1;
        unsafe {
            // The capacity moves and the LOGICAL size is left where it was, as
            // on every real target: a caller that wants the length to follow
            // sets it.
            (*handle).allocated = new_count as u32;
        }
        status::OK
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_free(
        be: *mut BeBackend,
        handle: *mut Handle,
        err: *mut BeError,
    ) -> i32 {
        let backend = unsafe { &mut *be.cast::<Backend>() };
        let current = unsafe { *handle };
        let Some(slot) = backend.arenas.get_mut(current.arena as usize) else {
            fill(err, status::MISUSE, b"handle names no live block");
            return status::MISUSE;
        };
        *slot = None;
        backend.generation += 1;
        // ZEROED, which is what the header says a library leaves behind: arena 0
        // with `allocated` 0, so the library's own checks reject it. Spelled out
        // rather than value-initialized, because zero is a real arena and the
        // sentinel is not zero; `AbiDevice::free` is what gives the caller's
        // copy the sentinel.
        unsafe {
            *handle = Handle {
                arena: 0,
                off: 0,
                size: 0,
                allocated: 0,
            }
        };
        status::OK
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_write(
        be: *mut BeBackend,
        handle: Handle,
        byte_offset: usize,
        src: *const c_void,
        bytes: usize,
        err: *mut BeError,
    ) -> i32 {
        let backend = unsafe { &mut *be.cast::<Backend>() };
        let Some(Some(block)) = backend.arenas.get_mut(handle.arena as usize) else {
            fill(err, status::MISUSE, b"handle names no live block");
            return status::MISUSE;
        };
        if byte_offset + bytes > block.len() {
            fill(err, status::MISUSE, b"the window is outside the block");
            return status::MISUSE;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.cast::<u8>(),
                block.as_mut_ptr().add(byte_offset),
                bytes,
            )
        };
        backend.counters.bytes_uploaded += bytes as u64;
        status::OK
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_read(
        be: *mut BeBackend,
        handle: Handle,
        byte_offset: usize,
        dst: *mut c_void,
        bytes: usize,
        err: *mut BeError,
    ) -> i32 {
        let backend = unsafe { &mut *be.cast::<Backend>() };
        let Some(Some(block)) = backend.arenas.get(handle.arena as usize) else {
            fill(err, status::MISUSE, b"handle names no live block");
            return status::MISUSE;
        };
        if byte_offset + bytes > block.len() {
            fill(err, status::MISUSE, b"the window is outside the block");
            return status::MISUSE;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                block.as_ptr().add(byte_offset),
                dst.cast::<u8>(),
                bytes,
            )
        };
        backend.counters.bytes_downloaded += bytes as u64;
        status::OK
    }

    /// THE HANDLE AND THE WINDOW ARE CHECKED AHEAD OF THE KNOB, WHICH IS THE
    /// ORDER THE HEADER ASKS FOR AND NOT AN ACCIDENT OF THIS BODY. A library
    /// that serves no view refuses an invented handle and a window past the
    /// block exactly as one that serves a view does, so this double answers a
    /// given call the same way under both settings of the knob and the same way
    /// the CUDA library answers it. Checking only on the served arm would make
    /// the double agree with no shipped library on the absent one.
    #[no_mangle]
    pub unsafe extern "C" fn be_host_ptr(
        be: *mut BeBackend,
        handle: Handle,
        byte_offset: usize,
        bytes: usize,
        out: *mut *mut c_void,
        err: *mut BeError,
    ) -> i32 {
        if be.is_null() || out.is_null() {
            fill(err, status::MISUSE, b"be_host_ptr needs a backend and an output slot");
            return status::MISUSE;
        }
        unsafe { *out = std::ptr::null_mut() };
        if bytes == 0 {
            return status::OK;
        }
        let backend = unsafe { &mut *be.cast::<Backend>() };
        let Some(Some(block)) = backend.arenas.get_mut(handle.arena as usize) else {
            fill(err, status::MISUSE, b"handle names no live block");
            return status::MISUSE;
        };
        // SPELLED SO IT CANNOT WRAP, unlike `be_write` and `be_read` above:
        // `byte_offset + bytes` overflows to a small number in release for a
        // window near the top of the address space, and the check then passes.
        // This is the one call whose result is a raw pointer the caller
        // dereferences with no further check, so the bound is written the way
        // the Metal library's `check_window` and the CUDA library's `window`
        // write it.
        if byte_offset > block.len() || bytes > block.len() - byte_offset {
            fill(err, status::MISUSE, b"the window is outside the block");
            return status::MISUSE;
        }
        if HOST_PTR_KNOB.load(Ordering::SeqCst) != HOST_PTR_SERVED {
            return status::OK;
        }
        unsafe { *out = block.as_mut_ptr().add(byte_offset).cast::<c_void>() };
        status::OK
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_allocator_generation(be: *mut BeBackend) -> u64 {
        unsafe { (*be.cast::<Backend>()).generation }
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_bytes_reserved(be: *mut BeBackend) -> u64 {
        let backend = unsafe { &*be.cast::<Backend>() };
        backend
            .arenas
            .iter()
            .filter_map(|slot| slot.as_ref())
            .map(|block| block.len() as u64)
            .sum()
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_encode_begin(
        be: *mut BeBackend,
        _region: *const c_char,
        mode: u32,
        out: *mut *mut BeEncoder,
        _err: *mut BeError,
    ) -> i32 {
        let encoder = Box::into_raw(Box::new(Encoder {
            backend: be.cast::<Backend>(),
            recording: mode == super::ENCODE_RECORD,
            ops: Vec::new(),
        }));
        unsafe { *out = encoder.cast::<BeEncoder>() };
        status::OK
    }

    /// The stub's device-side fill: it zeroes the window in place.
    ///
    /// A STUB THAT DID NOTHING WOULD HIDE THE THING THE OVERRIDE EXISTS FOR.
    /// `Device::fill_zero` promises the range reads back as zeros, and the
    /// tests that size a buffer and then read it depend on that, so this
    /// honours the promise the way a real backend's memset does.
    #[no_mangle]
    pub unsafe extern "C" fn be_encode_fill(
        enc: *mut BeEncoder,
        dst: Handle,
        byte_offset: usize,
        bytes: u64,
        value: u8,
        err: *mut BeError,
    ) -> i32 {
        if enc.is_null() {
            fill(err, status::MISUSE, b"be_encode_fill needs an encoder");
            return status::MISUSE;
        }
        let encoder = unsafe { &mut *enc.cast::<Encoder>() };
        let backend = unsafe { &mut *encoder.backend };
        let Some(Some(block)) = backend.arenas.get_mut(dst.arena as usize) else {
            fill(err, status::MISUSE, b"handle names no live block");
            return status::MISUSE;
        };
        let span = bytes as usize;
        if byte_offset + span > block.len() {
            fill(err, status::MISUSE, b"the fill runs past the allocation");
            return status::MISUSE;
        }
        block[byte_offset..byte_offset + span].fill(value);
        // NOT COUNTED AS AN UPLOAD, because it is not one: that is the whole
        // point of routing `fill_zero` here rather than through `be_write`.
        status::OK
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_encode_dispatch(
        enc: *mut BeEncoder,
        kernel_id: u32,
        extent: *const BeExtent,
        args: *const c_void,
        args_bytes: u32,
        err: *mut BeError,
    ) -> i32 {
        let encoder = unsafe { &mut *enc.cast::<Encoder>() };
        let extent = unsafe { &*extent };
        if extent.kind != super::EXTENT_ELEMENTS {
            fill(err, status::MISUSE, b"this kernel takes an ELEMENTS extent");
            return status::MISUSE;
        }
        if args_bytes != be_kernel_args_bytes(std::ptr::null_mut(), kernel_id) {
            fill(err, status::MISUSE, b"the record is not the declared width");
            return status::MISUSE;
        }
        // The bytes are COPIED here, so the caller's buffer may be reused
        // immediately and a recorded region owns its own copy.
        let mut bytes = vec![0u8; args_bytes as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(args.cast::<u8>(), bytes.as_mut_ptr(), bytes.len())
        };
        encoder.ops.push(Op {
            kernel: kernel_id,
            count: extent.count,
            args: bytes,
        });
        status::OK
    }

    fn summary_now() -> BeDiagSummary {
        let failures = FAILING_THREADS.load(Ordering::SeqCst);
        BeDiagSummary {
            failures,
            assert_hit: i32::from(failures > 0),
            assert_record: BeDiagRecord {
                assert_id: 1,
                file_id: 7,
                line: 314,
                thread_id: 2,
                payload: [1.0, 2.0, 3.0, 4.0],
            },
            ..BeDiagSummary::default()
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_encode_submit(
        enc: *mut BeEncoder,
        out_diag: *mut BeDiagSummary,
        err: *mut BeError,
    ) -> i32 {
        let encoder = unsafe { Box::from_raw(enc.cast::<Encoder>()) };
        if encoder.recording {
            fill(err, status::MISUSE, b"this encoder was opened for recording");
            return status::MISUSE;
        }
        let backend = unsafe { &mut *encoder.backend };
        backend.counters.syncs += 1;
        backend.counters.dispatches += encoder.ops.len() as u64;
        // OK FOR WORK THAT RAN, EVEN WITH FAILING THREADS. The counts land in
        // the summary and the driver renders the verdict.
        unsafe { *out_diag = summary_now() };
        status::OK
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_encode_record(
        enc: *mut BeEncoder,
        out: *mut *mut BeRegion,
        err: *mut BeError,
    ) -> i32 {
        let encoder = unsafe { Box::from_raw(enc.cast::<Encoder>()) };
        if !encoder.recording {
            fill(err, status::MISUSE, b"this encoder was opened for immediate use");
            return status::MISUSE;
        }
        let backend = unsafe { &mut *encoder.backend };
        backend.counters.regions_recorded += 1;
        let region = Box::into_raw(Box::new(Region {
            ops: encoder.ops,
            generation: backend.generation,
        }));
        unsafe { *out = region.cast::<BeRegion>() };
        status::OK
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_encode_abandon(enc: *mut BeEncoder) {
        if enc.is_null() {
            return;
        }
        drop(unsafe { Box::from_raw(enc.cast::<Encoder>()) });
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_region_info(region: *mut BeRegion, out: *mut BeRegionInfo) {
        let region = unsafe { &*region.cast::<Region>() };
        unsafe {
            *out = BeRegionInfo {
                deferred: 1,
                dispatch_count: region.ops.len() as u32,
                fill_count: 0,
                allocator_generation: region.generation,
            }
        };
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_replay(
        be: *mut BeBackend,
        region: *mut BeRegion,
        repeats: u32,
        out_diag: *mut BeDiagSummary,
        err: *mut BeError,
    ) -> i32 {
        let backend = unsafe { &mut *be.cast::<Backend>() };
        let region = unsafe { &*region.cast::<Region>() };
        if region.generation != backend.generation {
            fill(err, status::STALE_REGION, b"the allocator has moved");
            return status::STALE_REGION;
        }
        // The recorded bytes are replayed UNCHANGED, so a region whose entries
        // no longer match their declarations is a library defect rather than a
        // caller's. Reading them back here is what makes the copy at
        // `encode_dispatch` a property this double actually holds.
        for op in &region.ops {
            let declared = be_kernel_args_bytes(std::ptr::null_mut(), op.kernel) as usize;
            if op.args.len() != declared || op.count == 0 {
                fill(err, status::MISUSE, b"a recorded entry lost its shape");
                return status::MISUSE;
            }
        }
        backend.counters.syncs += 1;
        backend.counters.replays += 1;
        backend.counters.replay_repeats += u64::from(repeats);
        backend.counters.regions_deferred += 1;
        backend.counters.dispatches += region.ops.len() as u64 * u64::from(repeats);
        unsafe { *out_diag = summary_now() };
        status::OK
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_region_release(_be: *mut BeBackend, region: *mut BeRegion) {
        if region.is_null() {
            return;
        }
        drop(unsafe { Box::from_raw(region.cast::<Region>()) });
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_diag_file_path(
        _be: *mut BeBackend,
        file_id: u32,
        out: *mut c_char,
        capacity: usize,
        err: *mut BeError,
    ) -> i32 {
        if file_id != 7 {
            fill(err, status::MISUSE, b"this library assigned no such file id");
            return status::MISUSE;
        }
        if capacity < DIAG_FILE_C.to_bytes_with_nul().len() {
            fill(err, status::MISUSE, b"the buffer is shorter than the path");
            return status::MISUSE;
        }
        for (index, byte) in DIAG_FILE_C.to_bytes_with_nul().iter().enumerate() {
            unsafe { *out.add(index) = *byte as c_char };
        }
        status::OK
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_counters(be: *mut BeBackend, out: *mut BeCounters) {
        unsafe { *out = (*be.cast::<Backend>()).counters };
    }

    #[no_mangle]
    pub unsafe extern "C" fn be_counters_reset(be: *mut BeBackend) {
        unsafe { (*be.cast::<Backend>()).counters = BeCounters::default() };
    }
}

#[cfg(test)]
mod tests {
    use super::test_double as stub;
    use super::*;
    use crate::{
        AllocLabel, Buffer, Counters, Device, EncoderExt, KernelArgs, KernelDecl, KernelId, Scatter,
    };
    use device_impl::{AbiDevice, OpenConfig};
    use std::sync::atomic::Ordering;

    /// The caller's table, as a generated one would look: one kernel this
    /// library carries and one it does not.
    static TABLE: [KernelDecl; 2] = [
        KernelDecl {
            id: KernelId(0),
            name: stub::PRESENT_KERNEL,
            scatter: Scatter::Disjoint,
            nanos_per_item: 1.0,
            args_bytes: stub::KERNEL_ARGS_BYTES as u16,
            host_refs: &[],
            diag: true,
            generated: true,
        },
        KernelDecl {
            id: KernelId(1),
            name: stub::ABSENT_KERNEL,
            scatter: Scatter::Disjoint,
            nanos_per_item: 1.0,
            args_bytes: stub::KERNEL_ARGS_BYTES as u16,
            host_refs: &[],
            diag: true,
            generated: true,
        },
    ];

    /// A record naming no buffer, which is the only shape this seam can carry
    /// to a device today.
    #[repr(C, packed(4))]
    #[derive(Clone, Copy)]
    struct StubArgs {
        count: u32,
        scale: f32,
        first: u32,
        seam_arena_count: u32,
    }
    const _: () = assert!(std::mem::size_of::<StubArgs>() == stub::KERNEL_ARGS_BYTES as usize);

    unsafe impl KernelArgs for StubArgs {
        const KERNEL: KernelId = KernelId(0);
        const NAME: &'static str = stub::PRESENT_KERNEL;
        fn guard_count(&self) -> Option<u32> {
            Some(self.count)
        }
    }

    /// A record naming a buffer by host ADDRESS, which is what the driver's
    /// records carry today and what a device cannot resolve.
    ///
    /// Laid out as `crate::HostRef` is, so the declaration below can name its
    /// offset without this test reaching into that type's private fields, and
    /// sized to the same 16 bytes the library declares for this id, so the
    /// refusal under test is the one about ADDRESSES and not about width.
    #[repr(C, packed(4))]
    #[derive(Clone, Copy, Default)]
    struct AddressedArgs {
        addr: u64,
        bytes: u64,
    }
    const _: () = assert!(std::mem::size_of::<AddressedArgs>() == stub::KERNEL_ARGS_BYTES as usize);

    /// The same two kernels, with the first declaring that its record names one
    /// buffer by address.
    static ADDRESSED: [KernelDecl; 2] = [
        KernelDecl {
            host_refs: &[0],
            ..TABLE[0]
        },
        TABLE[1],
    ];

    unsafe impl KernelArgs for AddressedArgs {
        const KERNEL: KernelId = KernelId(0);
        const NAME: &'static str = stub::PRESENT_KERNEL;
    }

    fn silent(_level: i32, _message: &str) {}

    fn open() -> AbiDevice {
        // Safety: the linked library is the test double above, and TABLE is the
        // table it was written against.
        unsafe { AbiDevice::open(&TABLE, &OpenConfig::new(silent)) }
            .expect("the double agrees with this build")
    }

    // ---- the layout cross-check, which needs no library at all -------------

    #[test]
    fn a_library_built_from_this_header_agrees_field_for_field() {
        assert_eq!(compare_layout(&expected_layout()), Ok(()));
    }

    #[test]
    fn a_layout_disagreement_names_the_field_that_moved() {
        // Naming one field rather than printing both structs is what makes the
        // report actionable: the field that moved is the declaration that
        // diverged.
        let mut probe = expected_layout();
        probe.handle_size = 8;
        let message = compare_layout(&probe).expect_err("a wrong handle size is fatal");
        assert!(message.contains("handle_size"), "{message}");
        assert!(message.contains('8') && message.contains("16"), "{message}");
    }

    #[test]
    fn the_reserved_words_are_not_compared() {
        // They are padding for a later revision of the header, so a library that
        // filled them is not disagreeing about anything this build reads.
        let mut probe = expected_layout();
        probe.reserved = [1, 2, 3];
        assert_eq!(compare_layout(&probe), Ok(()));
    }

    #[test]
    fn the_five_sizes_the_header_asserts_are_asserted_here_too() {
        // The compile-time half is the `const _` block above; this is the same
        // five read back, so a reader can see the numbers the header names
        // without opening it.
        assert_eq!(std::mem::size_of::<Handle>(), 16);
        assert_eq!(std::mem::size_of::<BeDiagRecord>(), 32);
        assert_eq!(std::mem::size_of::<BeExtent>(), 16);
        assert_eq!(std::mem::size_of::<BeDiagSummary>(), 72);
        assert_eq!(std::mem::size_of::<BeCounters>(), 80);
    }

    #[test]
    fn an_extent_states_every_field_it_has() {
        // A left-over group width would be a launch geometry nobody asked for,
        // so the two fields an ELEMENTS launch does not use are written rather
        // than left at whatever the caller's stack held.
        assert_eq!(
            extent_to_abi(Extent::Elements { count: 41 }),
            BeExtent {
                kind: EXTENT_ELEMENTS,
                count: 41,
                threads: 0,
                scratch_bytes: 0,
            }
        );
    }

    #[test]
    fn an_unknown_status_is_reported_as_itself() {
        // A library built against a later header returns a code this build does
        // not know, and naming the number is more useful than mapping it onto a
        // neighbor.
        let err = BeError::slot();
        let Fault::Platform { call, detail } = fault_from(97, &err, "be_open") else {
            panic!("a status maps to a platform fault");
        };
        assert_eq!(call, "be_open");
        assert!(detail.contains("97"), "{detail}");
        assert!(detail.contains("does not know"), "{detail}");
    }

    #[test]
    fn an_error_detail_stops_at_the_nul_and_reports_truncation() {
        let mut err = BeError::slot();
        for (slot, byte) in err.detail.iter_mut().zip(b"out of device memory\0") {
            *slot = *byte as std::ffi::c_char;
        }
        assert_eq!(err.detail(), "out of device memory");
        err.truncated = 1;
        assert!(err.detail().ends_with("(truncated)"));
    }

    // ---- opening, which is where a stale library is caught ------------------

    #[test]
    fn an_abi_version_mismatch_refuses_before_the_library_is_opened() {
        let _guard = stub::acquire();
        stub::ABI_VERSION_KNOB.store(ABI_VERSION + 1, Ordering::SeqCst);
        // Safety: the double is linked and TABLE is its table.
        let fault = unsafe { AbiDevice::open(&TABLE, &OpenConfig::new(silent)) }
            .expect_err("a library from another revision cannot be opened");
        let Fault::Platform { call, detail } = fault else {
            panic!("a version mismatch is a platform fault");
        };
        assert_eq!(call, "be_abi_version");
        // DERIVED FROM THE CONSTANT rather than written out, because the
        // knob stores `ABI_VERSION + 1` and a literal here stops naming the
        // reported version the next time the ABI moves.
        let reported = format!("version {}", ABI_VERSION + 1);
        assert!(detail.contains(&reported), "{detail}");
    }

    #[test]
    fn a_layout_mismatch_refuses_before_the_library_is_opened() {
        // A wrong handle size is a wrong arena id, and a wrong arena id is a
        // silent wrong answer with plausible floats on a target that never
        // faults, so this may never be tolerated.
        let _guard = stub::acquire();
        stub::LAYOUT_KNOB.store(stub::CORRUPT_MAX_ARENAS, Ordering::SeqCst);
        // Safety: as above.
        let fault = unsafe { AbiDevice::open(&TABLE, &OpenConfig::new(silent)) }
            .expect_err("a moved field cannot be tolerated");
        let Fault::Platform { call, detail } = fault else {
            panic!("a layout mismatch is a platform fault");
        };
        assert_eq!(call, "be_layout_probe");
        assert!(detail.contains("max_arenas"), "{detail}");
    }

    #[test]
    fn a_kernel_table_of_a_different_length_is_not_the_same_table() {
        let _guard = stub::acquire();
        stub::TABLE_KNOB.store(stub::TABLE_WRONG_COUNT, Ordering::SeqCst);
        // Safety: as above.
        let fault = unsafe { AbiDevice::open(&TABLE, &OpenConfig::new(silent)) }
            .expect_err("dense ids cannot be dense in two different lengths");
        assert!(format!("{fault}").contains("kernels"), "{fault}");
    }

    #[test]
    fn an_id_naming_two_different_kernels_is_refused_by_both_names() {
        // THE CHECK THAT MATTERS. Both tables are generated, two trees generate
        // two orders, and an id that names one kernel here and another there is
        // a dispatch of the wrong kernel with the right bytes: correct types,
        // plausible numbers, nothing to see.
        let _guard = stub::acquire();
        stub::TABLE_KNOB.store(stub::TABLE_WRONG_NAME, Ordering::SeqCst);
        // Safety: as above.
        let fault = unsafe { AbiDevice::open(&TABLE, &OpenConfig::new(silent)) }
            .expect_err("an id must name one kernel");
        let text = format!("{fault}");
        assert!(text.contains(stub::PRESENT_KERNEL), "{text}");
        assert!(text.contains("stub_renamed_entry"), "{text}");
    }

    #[test]
    fn a_record_that_grew_a_field_on_one_side_is_refused() {
        let _guard = stub::acquire();
        stub::TABLE_KNOB.store(stub::TABLE_WRONG_ARGS_BYTES, Ordering::SeqCst);
        // Safety: as above.
        let fault = unsafe { AbiDevice::open(&TABLE, &OpenConfig::new(silent)) }
            .expect_err("a record has one width");
        assert!(format!("{fault}").contains("20 bytes"), "{fault}");
    }

    #[test]
    fn what_the_library_lacks_becomes_the_drivers_refusal_list() {
        // The library reports absence; the DRIVER maps it to a named feature and
        // refuses a scene by name and count. Nothing here decides what to
        // refuse.
        let _guard = stub::acquire();
        let device = open();
        assert_eq!(device.missing(), &[KernelId(1)]);
        assert_eq!(device.info().backend, "stub");
        assert_eq!(device.info().device_name, "stub device");
        assert_eq!(device.info().max_arenas, MAX_ARENAS);
    }

    #[test]
    fn preparing_a_kernel_the_library_lacks_fails_by_name_at_prepare_time() {
        // An id that cannot be supplied fails HERE rather than mid-step, which
        // is the whole reason `prepare_kernels` exists.
        let _guard = stub::acquire();
        let mut device = open();
        assert!(device.prepare_kernels(&[KernelId(0)]).is_ok());
        let fault = device
            .prepare_kernels(&[KernelId(0), KernelId(1)])
            .expect_err("the second id has no implementation");
        assert!(format!("{fault}").contains("be_prepare_kernels"), "{fault}");
    }

    // ---- the memory half ---------------------------------------------------

    #[test]
    fn bytes_written_through_the_abi_come_back() {
        let _guard = stub::acquire();
        let mut device = open();
        let handle = device
            .alloc(4, 4, 4, AllocLabel("test.buffer"))
            .expect("four floats");
        let source = [1.0f32, 2.0, 3.0, 4.0];
        device
            .write(handle, 0, bytemuck_slice(&source))
            .expect("a whole-block write");
        let mut read = [0.0f32; 4];
        device
            .read(handle, 0, bytemuck_slice_mut(&mut read))
            .expect("a whole-block read");
        assert_eq!(read, source);
        assert_eq!(device.bytes_reserved(), 16);
    }

    #[test]
    fn an_alignment_the_allocator_refuses_names_the_label() {
        // Metal silently FLOORS a misaligned buffer offset to a multiple of 4
        // and returns the wrong data with status Completed, so the question is
        // asked at allocation, which is the earliest point it can be answered
        // loudly.
        let _guard = stub::acquire();
        let mut device = open();
        let fault = device
            .alloc(4, 4, 2, AllocLabel("test.misaligned"))
            .expect_err("an alignment below 4 is refused");
        let Fault::Alloc { label, bytes, .. } = fault else {
            panic!("an allocator refusal is an alloc fault");
        };
        assert_eq!(label.0, "test.misaligned");
        assert_eq!(bytes, 16);
    }

    #[test]
    fn a_freed_handle_comes_back_from_the_library_naming_nothing() {
        // THE ONE ASSERTION THAT COVERS `AbiDevice::free`'s NORMALIZATION. The
        // library zeroes the handle, as the header says it must; without the
        // line that rewrites the caller's copy to the sentinel, this fails here
        // rather than as a wrong answer four phases later.
        let _guard = stub::acquire();
        let mut device = open();
        crate::device::assert_free_leaves_the_sentinel(&mut device);
        // THE FOUR BUFFER ASSERTIONS RUN UNDER BOTH REPRESENTATIONS, because a
        // staged or readback buffer holds either its own array or the
        // allocation itself, and the accounting they assert must be the same
        // under both. The double serves a real view of its host blocks, so this
        // host reaches the mapped arm with no GPU and no Mac; without the loop
        // the arm would be compiled and read by nothing here.
        for served in [stub::HOST_PTR_ABSENT, stub::HOST_PTR_SERVED] {
            stub::HOST_PTR_KNOB.store(served, Ordering::SeqCst);
            crate::device::assert_a_prefix_upload_names_only_what_it_carried(&mut device);
            crate::device::assert_a_readback_answers_only_after_a_download(&mut device);
            crate::device::assert_a_redundant_upload_transfers_nothing(&mut device);
            crate::device::assert_a_host_view_aliases_the_allocation(&mut device);
        }
        stub::HOST_PTR_KNOB.store(stub::HOST_PTR_ABSENT, Ordering::SeqCst);
    }

    #[test]
    fn a_backend_that_serves_no_host_view_says_so_rather_than_failing() {
        // NULL IS AN ANSWER AND NOT A FAILURE. A target whose device memory the
        // host cannot address reports it this way, and the caller's copy path
        // through `be_write` and `be_read` is what serves such a target, so an
        // error here would refuse every buffer on it.
        let _guard = stub::acquire();
        let mut device = open();
        let mut buffer = Buffer::<u32>::none();
        buffer
            .size(&mut device, 4, AllocLabel("test.no_view"))
            .expect("four words");
        // Safety: the handle names the live allocation and the window is its
        // own length.
        let view = unsafe { device.host_view(buffer.handle(), 0, 16) }
            .expect("a target that serves no view still answers");
        assert!(view.is_none(), "the double reports no view unless it is asked");
        buffer.free(&mut device).expect("the allocation is released");
    }

    #[test]
    fn a_served_host_view_addresses_the_block() {
        // THE VIEW MUST BE THE BLOCK, NOT A COPY OF IT. A library that handed
        // back a staging buffer would satisfy every accounting assertion above
        // while every write through it landed where no dispatch reads, and on a
        // target that never faults on an out-of-bounds access that is silent.
        let _guard = stub::acquire();
        stub::HOST_PTR_KNOB.store(stub::HOST_PTR_SERVED, Ordering::SeqCst);
        let mut device = open();
        let mut buffer = Buffer::<u32>::none();
        buffer
            .size(&mut device, 4, AllocLabel("test.view"))
            .expect("four words");
        let bytes = 4 * std::mem::size_of::<u32>();
        // Safety: the handle names the live allocation and the window is its
        // own length.
        let view = unsafe { device.host_view(buffer.handle(), 0, bytes) }
            .expect("the double answers")
            .expect("and serves a view when it is asked");
        let source = [7u32, 8, 9, 10];
        // Safety: the pointer addresses the block's sixteen bytes, the source
        // is a distinct local, and no device work is in flight here.
        unsafe {
            std::ptr::copy_nonoverlapping(source.as_ptr().cast::<u8>(), view.as_ptr(), bytes)
        };
        let mut read = [0u32; 4];
        buffer
            .read(&mut device, 0, &mut read)
            .expect("a whole-block read");
        assert_eq!(
            read, source,
            "a write through the view must reach the bytes `read` returns"
        );
        buffer.free(&mut device).expect("the allocation is released");
    }

    #[test]
    fn a_host_view_of_an_invented_handle_is_refused() {
        // A HANDLE THE ALLOCATOR NEVER HANDED OUT NAMES NO BLOCK, and a window
        // running past one is outside it. Both are refused rather than
        // answered, so this call is not a way to manufacture an address the
        // transfer calls would have rejected.
        //
        // ASSERTED UNDER BOTH SETTINGS OF THE KNOB, because the header puts the
        // refusal on every library and not only on one with a pointer to hand
        // back: a driver that invents a handle is refused on CUDA, which serves
        // no view, exactly as it is on Metal, which serves one. Asserting it on
        // the served arm alone would leave the double free to answer STATUS_OK
        // on the absent arm, which is the behavior no shipped library has.
        let _guard = stub::acquire();
        let mut device = open();
        for served in [stub::HOST_PTR_ABSENT, stub::HOST_PTR_SERVED] {
            stub::HOST_PTR_KNOB.store(served, Ordering::SeqCst);
            let mut buffer = Buffer::<u32>::none();
            buffer
                .size(&mut device, 4, AllocLabel("test.invented"))
                .expect("four words");

            let mut invented = buffer.handle();
            invented.arena += 7;
            // Safety: the refusal is what is under test, and the library reads
            // the handle without dereferencing anything the caller owns.
            let fault = unsafe { device.host_view(invented, 0, 16) }
                .expect_err("a handle naming no live block is refused");
            let Fault::Platform { call, .. } = fault else {
                panic!("a violated precondition of the ABI is a platform fault");
            };
            assert_eq!(call, "be_host_ptr");

            // AGAINST THE ALLOCATOR'S CAPACITY AND NOT THE LOGICAL LENGTH,
            // which is the rule [`Handle`] states: `allocated` is what the
            // arena holds, and a window inside it is legitimate however short
            // the buffer's own `size` is. So the window that must be refused is
            // one past the capacity, derived from the handle rather than
            // written out, because the headroom `Buffer::size` asks for is a
            // sizing policy and would make a literal here a test of that policy
            // instead.
            let past = (buffer.handle().allocated as usize + 1) * std::mem::size_of::<u32>();
            // Safety: as above, with a window that runs past the block.
            let fault = unsafe { device.host_view(buffer.handle(), 0, past) }
                .expect_err("a window outside the block is refused");
            let Fault::Platform { call, .. } = fault else {
                panic!("a violated precondition of the ABI is a platform fault");
            };
            assert_eq!(call, "be_host_ptr");

            // AND AN OFFSET THAT WOULD WRAP ITS OWN SUM IS OUTSIDE THE BLOCK
            // TOO. `byte_offset + bytes` overflows in release for a window near
            // the top of the address space, and a bound written that way admits
            // it, which is the one refusal whose result would be a raw pointer.
            let fault = unsafe { device.host_view(buffer.handle(), usize::MAX, 16) }
                .expect_err("a window whose sum wraps is refused");
            let Fault::Platform { call, .. } = fault else {
                panic!("a violated precondition of the ABI is a platform fault");
            };
            assert_eq!(call, "be_host_ptr");

            buffer.free(&mut device).expect("the allocation is released");
        }
        stub::HOST_PTR_KNOB.store(stub::HOST_PTR_ABSENT, Ordering::SeqCst);
    }

    #[test]
    fn sizing_a_released_buffer_does_not_grow_the_first_allocation() {
        // The same defect read through the DATA, so it is caught even if the
        // handle assertion above is weakened. The double packs one allocation
        // per arena and never reuses a slot, so a zeroed handle names arena 0,
        // which is the block the first buffer holds.
        let _guard = stub::acquire();
        let mut device = open();
        crate::device::assert_sizing_after_a_free_spares_the_first_allocation(&mut device);
    }

    #[test]
    fn a_label_is_interned_once_however_often_it_is_used() {
        let _guard = stub::acquire();
        let mut device = open();
        for _ in 0..8 {
            device
                .alloc(1, 4, 4, AllocLabel("test.repeated"))
                .expect("one float");
        }
        assert_eq!(device.label_count(), 1);
    }

    // ---- the execution half ------------------------------------------------

    #[test]
    fn one_dispatch_is_one_sync_and_one_dispatch() {
        // A dispatch that did not happen is a counter that did not move, and
        // unlike a fixture this cannot pass while the production path calls
        // something else.
        let _guard = stub::acquire();
        let mut device = open();
        let args = StubArgs {
            count: 64,
            scale: 0.5,
            first: 0,
            seam_arena_count: 0,
        };
        // Safety: the record names no buffer, so there is nothing to outlive
        // the call.
        let diag = unsafe { device.launch("test.phase", &args, 64) }.expect("a clean boundary");
        assert!(diag.is_clean());
        assert_eq!(
            device.counters(),
            Counters {
                syncs: 1,
                dispatches: 1,
                regions_deferred: 0,
                regions_fallback: 0,
            }
        );
    }

    #[test]
    fn a_record_naming_a_buffer_by_address_is_refused_by_name() {
        // THE PRECONDITION THE HANDLE MIGRATION EXISTS FOR, stated by the seam
        // rather than remembered. A `HostRef` is an address; on a target with
        // its own memory it is a wild pointer, and on one that never faults it
        // is plausible floats with status Completed.
        let _guard = stub::acquire();
        // Safety: ADDRESSED is the same table the double serves, with the first
        // record declaring one buffer reference.
        let mut device = unsafe { AbiDevice::open(&ADDRESSED, &OpenConfig::new(silent)) }
            .expect("the double serves this table");
        let args = AddressedArgs::default();
        // Safety: the dispatch is refused before the record is read.
        let fault = unsafe { device.launch("test.addressed", &args, 1) }
            .expect_err("an address cannot be bound by a device");
        let Fault::Shape { kernel, detail } = fault else {
            panic!("a record the seam cannot carry is a shape fault");
        };
        assert_eq!(kernel, stub::PRESENT_KERNEL);
        assert!(detail.contains("by host address"), "{detail}");
        assert_eq!(device.counters().dispatches, 0, "nothing was encoded");
    }

    #[test]
    fn a_failing_check_becomes_a_device_fault_naming_its_source() {
        // The channel is a REPORT and the driver renders the verdict, so the
        // library returns OK for work that ran and the failure surfaces here.
        let _guard = stub::acquire();
        let mut device = open();
        stub::FAILING_THREADS.store(3, Ordering::SeqCst);
        let args = StubArgs {
            count: 8,
            scale: 1.0,
            first: 0,
            seam_arena_count: 0,
        };
        // Safety: the record names no buffer.
        let fault =
            unsafe { device.launch("test.checked", &args, 8) }.expect_err("three threads failed");
        let Fault::Device { region, diag } = fault else {
            panic!("a failing check is a device fault");
        };
        assert_eq!(region, "test.checked");
        assert_eq!(diag.failures, 3);
        let first = diag.first.expect("a claimant was recorded");
        // The library owns the transport, which includes resolving whatever the
        // channel recorded into a file name.
        assert_eq!(first.file, "stub_kernel.kernel.cpp");
        assert_eq!(first.line, 314);
    }

    #[test]
    fn a_body_that_fails_closes_its_encoder() {
        // One encoder is open at a time, so an error path between begin and
        // close has to abandon it or the next boundary is refused as misuse.
        let _guard = stub::acquire();
        let mut device = open();
        let fault = device
            .run("test.abandoned", |_| {
                Err(Fault::MissingKernel {
                    kernel: KernelId(9),
                })
            })
            .expect_err("the body refused");
        assert!(matches!(fault, Fault::MissingKernel { .. }));
        let args = StubArgs {
            count: 1,
            scale: 1.0,
            first: 0,
            seam_arena_count: 0,
        };
        // Safety: the record names no buffer.
        unsafe { device.launch("test.after", &args, 1) }.expect("the encoder was released");
    }

    #[test]
    fn a_recorded_region_replays_without_re_encoding() {
        let _guard = stub::acquire();
        let mut device = open();
        let args = StubArgs {
            count: 32,
            scale: 1.0,
            first: 0,
            seam_arena_count: 0,
        };
        let region = device
            .record("test.loop", |e| {
                // Safety: the record names no buffer, so nothing has to outlive
                // the replays.
                unsafe { e.elements(&args, 32) }
            })
            .expect("a region of one entry");
        assert_eq!(region.info().deferred, 1, "the region is held deferred");
        assert_eq!(region.info().dispatch_count, 1);
        device.replay(&region, 10).expect("ten repeats");
        let counters = device.counters();
        assert_eq!(counters.dispatches, 10, "one entry, ten repeats");
        assert_eq!(counters.syncs, 1, "one host round trip for the whole batch");
        assert_eq!(counters.regions_fallback, 0, "the region was never re-issued");
        device.release(region);
    }

    #[test]
    fn a_region_recorded_before_an_allocation_is_refused_rather_than_replayed() {
        // `grow` may relocate a block, so a recorded argument record can name a
        // stale handle, and Metal does not fault on addressing freed memory.
        let _guard = stub::acquire();
        let mut device = open();
        let args = StubArgs {
            count: 4,
            scale: 1.0,
            first: 0,
            seam_arena_count: 0,
        };
        let region = device
            .record("test.stale", |e| {
                // Safety: the record names no buffer.
                unsafe { e.elements(&args, 4) }
            })
            .expect("a region");
        let recorded = device.allocator_generation();
        device
            .alloc(1, 4, 4, AllocLabel("test.after.record"))
            .expect("one float");
        let fault = device
            .replay(&region, 1)
            .expect_err("the allocator moved underneath the region");
        let Fault::StaleRegion {
            region: label,
            recorded: was,
            now,
        } = fault
        else {
            panic!("a moved allocator is a stale region");
        };
        assert_eq!(label, "test.stale");
        assert_eq!(was, recorded);
        assert!(now > was, "{now} is not later than {was}");
        device.release(region);
    }

    #[test]
    fn a_dispatch_whose_extent_disagrees_with_the_record_is_refused() {
        // The count exists TWICE at a dispatch, once in the record and once in
        // the extent, and the entry point clamps to the record, so a record left
        // at zero would run no elements and report nothing.
        let _guard = stub::acquire();
        let mut device = open();
        let args = StubArgs {
            count: 7,
            scale: 1.0,
            first: 0,
            seam_arena_count: 0,
        };
        // Safety: the record names no buffer.
        let fault = unsafe { device.launch("test.mismatched", &args, 8) }
            .expect_err("the two counts disagree");
        assert!(format!("{fault}").contains("silently not done"), "{fault}");
    }

    // Small helpers, so the tests above read as what they assert.

    fn bytemuck_slice(values: &[f32]) -> &[u8] {
        // Safety: `f32` has no padding and no invalid bit pattern, and the
        // result borrows the same allocation.
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) }
    }

    fn bytemuck_slice_mut(values: &mut [f32]) -> &mut [u8] {
        // Safety: as above.
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), values.len() * 4) }
    }
}

/// The name the LOADED backend library reports for itself.
///
/// Callable before `be_open`, which `kernels/seam/backend_abi.h` states
/// explicitly and which is what lets a `--backend` query answer without
/// touching a device. It exists beside `DeviceInfo::backend` because that one
/// requires an opened device, and the question "what library did this binary
/// actually link" has to be answerable on a machine with no GPU at all.
#[cfg(backend_abi)]
pub fn linked_backend_name() -> &'static str {
    // Safety: the header states the pointer is a string literal in the library
    // and lives for the life of the process.
    unsafe {
        let name = bound::be_backend_name();
        if name.is_null() {
            "unnamed"
        } else {
            std::ffi::CStr::from_ptr(name).to_str().unwrap_or("unnamed")
        }
    }
}
