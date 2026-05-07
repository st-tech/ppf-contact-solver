// File: crates/ppf-cts-core/src/cancel.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Cooperative cancellation primitive that replaces the
// ctypes.PyThreadState_SetAsyncExc() thread-kill in
// server/engine.py:215. The build task receives a clone, polls
// `is_cancelled()` at well-defined checkpoints (pre-decoder,
// post-decoder, pre-tetrahedralize, per-object loop, post-FixedScene),
// and unwinds on observation. Wraps tokio_util::sync::CancellationToken
// so both the sync build path (spawn_blocking) and any future async
// build path can share the same handle.

use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Default)]
pub struct CancelHandle(CancellationToken);

impl CancelHandle {
    pub fn new() -> Self {
        Self(CancellationToken::new())
    }

    pub fn cancel(&self) {
        self.0.cancel();
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.is_cancelled()
    }

    pub fn token(&self) -> &CancellationToken {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancel_propagates_to_clones() {
        let h = CancelHandle::new();
        assert!(!h.is_cancelled());
        let h2 = h.clone();
        h.cancel();
        assert!(h.is_cancelled());
        assert!(h2.is_cancelled());
    }
}
