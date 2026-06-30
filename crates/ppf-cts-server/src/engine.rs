// File: crates/ppf-cts-server/src/engine.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// `ServerEngine`: tokio-friendly state holder + dispatcher.
// Replaces server/engine.py's `ServerEngine` class.
//
// Concurrency model:
//   * `state` is held under a `parking_lot::RwLock` so multi-reader
//     queries (status responses) don't block each other.
//   * `dispatch(event)` acquires the write lock for the duration of
//     the pure transition, replaces the snapshot, releases the lock,
//     and returns the produced effects to the caller.
//   * Effect *execution* is the executor's responsibility, NOT the
//     engine's. This keeps the engine cancellation-token-free and
//     makes it trivially testable in pure Rust.
//   * `cancel_token` holds the cooperative-cancellation handle for an
//     in-flight build. `DoSpawnBuild` mints + stores a fresh token;
//     `DoCancelBuild` calls `.cancel()` on it. Replaces the
//     ctypes.PyThreadState_SetAsyncExc thread-kill in the Python
//     reference (see CancelHandle in ppf-cts-core::cancel).

use std::sync::{Arc, OnceLock, Weak};

use parking_lot::{Mutex, RwLock};
use ppf_cts_core::cancel::CancelHandle;
use ppf_cts_core::effects::Effect;
use ppf_cts_core::events::Event;
use ppf_cts_core::state::ServerState;
use ppf_cts_core::transitions::transition;

use crate::config::EngineConfig;
use crate::executor::EffectExecutor;

#[derive(Clone)]
pub struct ServerEngine {
    inner: Arc<EngineInner>,
}

struct EngineInner {
    state: RwLock<ServerState>,
    config: EngineConfig,
    /// Cancellation control for the in-flight build, kept under a
    /// single mutex so install/cancel/clear are atomic with respect
    /// to each other. The `pending` flag latches a cancel that
    /// arrives in the gap between `BuildRequested` dispatch and the
    /// executor's call to `install_cancel_handle`, so the next build
    /// observes the cancel instead of silently ignoring it.
    cancel: Mutex<CancelState>,
    /// Weak handle to the active executor, set once at startup by
    /// `attach_executor`. Re-entrant effect handlers (build pipeline,
    /// solver launch, save/quit) upgrade this to dispatch follow-up
    /// events through the same executor that processed the original
    /// effect, instead of minting a stateless `DefaultExecutor`. Held
    /// as `Weak` to avoid the engine→executor→... cycle when an
    /// executor implementation captures an `Arc<ServerEngine>`
    /// internally; the embedder owns the `Arc<dyn EffectExecutor>`
    /// for the process lifetime so the upgrade is expected to
    /// succeed during normal operation.
    executor: OnceLock<Weak<dyn EffectExecutor>>,
}

#[derive(Default)]
struct CancelState {
    /// Active build cancellation token (if any).
    handle: Option<CancelHandle>,
    /// Sticky "cancel pending" latch: set when `cancel_active_build`
    /// runs while no handle is installed yet. The next
    /// `install_cancel_handle` consumes the latch and returns an
    /// already-cancelled handle.
    pending: bool,
}

impl ServerEngine {
    pub fn new(config: EngineConfig) -> Self {
        Self {
            inner: Arc::new(EngineInner {
                state: RwLock::new(ServerState::default()),
                config,
                cancel: Mutex::new(CancelState::default()),
                executor: OnceLock::new(),
            }),
        }
    }

    /// Bind the executor used for re-entrant effect dispatch (build
    /// pipeline emitting `BuildCompleted`, solver helpers emitting
    /// `ErrorOccurred`, etc.). Idempotent: subsequent calls are
    /// ignored so tests can leave it unset and fall back to a fresh
    /// `DefaultExecutor` via `executor_or_default`. Stored as `Weak`
    /// so the embedder keeps ownership and can drop the executor at
    /// shutdown without the engine pinning it alive.
    pub fn attach_executor(&self, executor: &Arc<dyn EffectExecutor>) {
        let _ = self.inner.executor.set(Arc::downgrade(executor));
    }

    /// Upgrade the attached executor to a concrete `Arc`, or return
    /// `None` if no executor has been attached or the embedder has
    /// already dropped it. Re-entrant handlers use this to thread
    /// effects back through the user's executor instead of building
    /// a fresh `DefaultExecutor` each time.
    pub fn executor(&self) -> Option<Arc<dyn EffectExecutor>> {
        self.inner.executor.get().and_then(Weak::upgrade)
    }

    pub fn config(&self) -> &EngineConfig {
        &self.inner.config
    }

    /// Snapshot the current state. Cheap (read lock + clone of a
    /// ~14-field POD).
    pub fn state(&self) -> ServerState {
        self.inner.state.read().clone()
    }

    /// Run the pure transition under the write lock and return the
    /// produced effects. The caller forwards these effects to an
    /// `EffectExecutor`. Re-entrance from inside an effect (e.g. a
    /// build task dispatching `BuildCompleted` back) is handled
    /// because the lock is dropped before this returns.
    pub fn dispatch(&self, event: Event) -> Vec<Effect> {
        let mut guard = self.inner.state.write();
        let snapshot = guard.clone();
        let (new_state, effects) = transition(snapshot, event);
        *guard = new_state;
        effects
    }

    /// Replace the project name + root without going through a full
    /// `ProjectSelected` transition. Used by request handlers that
    /// need to set context before dispatching guarded events.
    pub fn set_project_context(&self, name: &str, root: &str) {
        let mut guard = self.inner.state.write();
        guard.name = name.to_string();
        guard.root = root.to_string();
    }

    // --- Cancellation token plumbing --------------------------------

    /// Install a cancel handle for an in-flight build. Returns a
    /// clone the build task can use to poll `is_cancelled()`.
    /// Cooperative cancellation only; no ctypes thread-kill.
    ///
    /// If a cancel arrived between `BuildRequested` dispatch and this
    /// call (e.g. another connection raced a `cancel_build` against
    /// the just-issued `build`), the returned handle is already
    /// cancelled so the build observes it on the first checkpoint
    /// instead of running to completion.
    pub fn install_cancel_handle(&self) -> CancelHandle {
        let h = CancelHandle::new();
        let mut guard = self.inner.cancel.lock();
        if guard.pending {
            h.cancel();
            guard.pending = false;
        }
        guard.handle = Some(h.clone());
        h
    }

    /// Trip the active cancel token if there is one. If no build is
    /// in flight yet, latches the cancel so the next
    /// `install_cancel_handle` returns an already-cancelled handle.
    pub fn cancel_active_build(&self) {
        let mut guard = self.inner.cancel.lock();
        match guard.handle.as_ref() {
            Some(h) => h.cancel(),
            None => guard.pending = true,
        }
    }

    /// Drop the cancel handle when the build finishes (success,
    /// failure, or cancellation observed). Also clears any latched
    /// pending-cancel: by the time a build has run to a terminal
    /// state, any earlier latched cancel has been resolved (either
    /// observed via the cancelled handle, or the build completed
    /// before the next `install_cancel_handle` could consume it; in
    /// the latter case the latch was meant for *that* build, not a
    /// future one, so we drop it here rather than letting it bleed
    /// into the next launch).
    pub fn clear_cancel_handle(&self) {
        let mut guard = self.inner.cancel.lock();
        guard.handle = None;
        guard.pending = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ppf_cts_core::events::Event;
    use ppf_cts_core::state::{Build, Data};

    #[test]
    fn dispatch_advances_state_and_emits_effects() {
        let engine = ServerEngine::new(EngineConfig::default());
        engine.set_project_context("p", "/tmp/p");
        // Mark data uploaded so BuildRequested isn't rejected.
        engine.dispatch(Event::upload_landed("uid"));

        let effects = engine.dispatch(Event::BuildRequested { preserve_output: false });
        let s = engine.state();
        assert_eq!(s.build, Build::Building);
        assert_eq!(s.data, Data::Uploaded);
        assert!(effects.iter().any(|e| matches!(e, Effect::DoSpawnBuild { .. })));
    }

    #[test]
    fn cancel_handle_lifecycle() {
        let engine = ServerEngine::new(EngineConfig::default());
        let h = engine.install_cancel_handle();
        assert!(!h.is_cancelled());

        engine.cancel_active_build();
        assert!(h.is_cancelled(), "cancel propagates to held handle");

        engine.clear_cancel_handle();
        // Subsequent cancel without an installed handle latches a
        // pending cancel; clear_cancel_handle wipes it so the next
        // build starts fresh.
        engine.cancel_active_build();
        engine.clear_cancel_handle();
        let h2 = engine.install_cancel_handle();
        assert!(!h2.is_cancelled(), "cleared latch should not bleed forward");
    }

    #[test]
    fn cancel_before_install_latches_into_next_handle() {
        // Simulates the TOCTOU window the audit called out: a cancel
        // arrives between `BuildRequested` dispatch and the executor's
        // call to `install_cancel_handle`. The next install must
        // observe the cancel rather than silently dropping it.
        let engine = ServerEngine::new(EngineConfig::default());
        engine.cancel_active_build();
        let h = engine.install_cancel_handle();
        assert!(
            h.is_cancelled(),
            "cancel latched before install must apply to the new handle"
        );
    }

    #[test]
    fn set_project_context_persists() {
        let engine = ServerEngine::new(EngineConfig::default());
        engine.set_project_context("demo", "/tmp/demo");
        let s = engine.state();
        assert_eq!(s.name, "demo");
        assert_eq!(s.root, "/tmp/demo");
    }
}
