// File: crates/ppf-cts-compute/cpu/mem.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
//! The scratch pool.
//!
//! Allocation computes no values, so this is skeleton by definition and has no
//! shared body. Two properties of this pool are deliberate, and both rule out a
//! float-backed design that rounds every request up to whole floats:
//!
//! 1. **`len` and `capacity` are both in elements of `T`.** A float-backed
//!    block would carry its length in elements of `T` and its capacity in
//!    floats. Two numbers in one struct measuring different things is a
//!    subtraction waiting to happen.
//! 2. **Alignment is asserted, not assumed.** A float-backed block is correctly
//!    aligned only for types whose `align_of` is 4, which is a property of the
//!    call sites rather than of the pool: the first `f64` or 16-byte type would
//!    read misaligned. This pool asserts natural alignment at ALLOCATION time,
//!    which is the earliest point the question can be asked.
//!
//! Two allocation policies follow from what the solver asks of the pool: reuse
//! the smallest free block that fits, and grow a free block in place rather
//! than orphaning it and adding another. Orphaning instead of growing makes the
//! pool grow monotonically under a sequence of increasing nnz-sized requests,
//! which is what Schwarz issues, and that exhausts device memory rather than
//! merely slowing the run down.

// These are P1 components, landed ahead of the Newton driver that will call
// them. Until `advance()` is built they have no caller outside their own tests,
// which is dead code by the compiler's reckoning and deliberate by the plan's:
// each slice lands with its own gates rather than waiting for a driver that
// would then arrive untested. The allow is removed in the change that wires the
// driver up, and it is scoped per module so a genuinely unused item elsewhere
// still surfaces.
#![allow(dead_code)]

use std::alloc::{alloc, dealloc, Layout};
use std::cell::RefCell;

/// One pooled block, sized in bytes and aligned to at least `align`.
struct Block {
    ptr: *mut u8,
    bytes: usize,
    align: usize,
    in_use: bool,
}

// A block's pointer never crosses a thread boundary through this type: the pool
// is per-step host state, and a checked-out block becomes a slice whose lifetime
// is tied to the borrow of the pool. The parallel work inside a step operates on
// slices already carved out, not on the pool itself.
unsafe impl Send for Block {}

/// A scratch allocator for the per-step temporaries.
///
/// Blocks are recycled across steps: the pool reaches the scene's high-water
/// mark and then performs no further allocation, which is the property that
/// makes it worth having rather than calling the system allocator per step.
pub struct Pool {
    /// Interior mutability is required rather than convenient: the solver holds
    /// several temporaries at once (`eval_x` and `target` and a scratch buffer
    /// in the same scope), so `get` must take `&self`. A `&mut self` signature
    /// compiles and then makes the pool useless for the thing it exists for,
    /// which is what the first version of this file did until a test asked for
    /// two checkouts.
    blocks: RefCell<Vec<Block>>,
}

impl Default for Pool {
    fn default() -> Self {
        Self::new()
    }
}

impl Pool {
    pub const fn new() -> Self {
        Pool {
            blocks: RefCell::new(Vec::new()),
        }
    }

    /// Blocks currently held, whether checked out or free.
    ///
    /// Exposed so a test can assert the pool stops growing, which is the whole
    /// claim above.
    pub fn block_count(&self) -> usize {
        self.blocks.borrow().len()
    }

    /// Check out a zeroed block of `count` elements of `T`.
    ///
    /// # Panics
    /// If `T`'s alignment cannot be satisfied, which is asserted here rather
    /// than discovered as a misaligned read later.
    pub fn get<T: Copy + Default>(&self, count: usize) -> PooledSlice<'_, T> {
        let align = std::mem::align_of::<T>();
        let bytes = std::mem::size_of::<T>().max(1) * count.max(1);
        let index = self.take_block(bytes, align);
        let ptr = self.block_base(index) as *mut T;
        // Safety: the block is at least `bytes` long and aligned for `T`, which
        // `take_block` required and `allocate` asserted.
        let slice = unsafe {
            std::ptr::write_bytes(ptr, 0, count);
            std::slice::from_raw_parts_mut(ptr, count)
        };
        PooledSlice {
            slice,
            index,
            pool: self,
        }
    }

    /// Check out a block of at least `bytes`, aligned to at least `align`, and
    /// hand back its index rather than a slice.
    ///
    /// This is the form the backend seam allocates through
    /// ([`crate::HostDevice`]), where an allocation is addressed by a handle that
    /// outlives any borrow and is released explicitly rather than by a drop. The
    /// block reuse policy is the same one and it is stated once, here, because
    /// two copies of an allocator policy drift the way two copies of a kernel do.
    ///
    /// The bytes are NOT cleared: a freshly allocated block is undefined on every
    /// backend, and a pool that happens to zero is exactly what hides a missing
    /// clear.
    pub fn take_block(&self, bytes: usize, align: usize) -> usize {
        let bytes = bytes.max(1);
        // Smallest free block that fits, in bytes AND in alignment. Alignment is
        // part of "fits": a block carved for f32 does not satisfy an f64 request
        // merely by being long enough.
        let mut blocks = self.blocks.borrow_mut();
        let mut best: Option<usize> = None;
        for (i, block) in blocks.iter().enumerate() {
            if block.in_use || block.bytes < bytes || block.align < align {
                continue;
            }
            match best {
                Some(b) if blocks[b].bytes <= block.bytes => {}
                _ => best = Some(i),
            }
        }

        let index = match best {
            Some(i) => i,
            None => {
                // Nothing fits. Grow the largest free block with a compatible
                // alignment in place rather than orphaning it, so the block
                // COUNT stays bounded by peak concurrency instead of by the
                // number of distinct sizes ever requested.
                let grow = blocks
                    .iter()
                    .enumerate()
                    .filter(|(_, b)| !b.in_use && b.align >= align)
                    .max_by_key(|(_, b)| b.bytes)
                    .map(|(i, _)| i);
                match grow {
                    Some(i) => {
                        Self::free_block_at(&mut blocks, i);
                        blocks[i] = Self::allocate(bytes, align);
                        i
                    }
                    None => {
                        blocks.push(Self::allocate(bytes, align));
                        blocks.len() - 1
                    }
                }
            }
        };

        blocks[index].in_use = true;
        index
    }

    /// The base address of a checked-out block.
    ///
    /// # Panics
    /// If the index names no block, which is a caller defect: a handle whose
    /// arena is not bound must be refused above this, where the fault can name
    /// the handle.
    pub fn block_base(&self, index: usize) -> *mut u8 {
        let blocks = self.blocks.borrow();
        let ptr = blocks[index].ptr;
        assert!(!ptr.is_null(), "pool: block {index} has been deallocated");
        ptr
    }

    /// Return a block taken by [`Pool::take_block`].
    pub fn release_block(&self, index: usize) {
        self.release(index);
    }

    /// Bytes the pool holds, checked out or free.
    ///
    /// The high-water mark a scene reaches, which is the number that says the
    /// pool stopped growing.
    pub fn reserved_bytes(&self) -> u64 {
        self.blocks.borrow().iter().map(|b| b.bytes as u64).sum()
    }

    fn allocate(bytes: usize, align: usize) -> Block {
        let layout = Layout::from_size_align(bytes, align)
            .expect("pool: size and alignment do not form a valid layout");
        // Safety: `bytes` is nonzero because `get` clamps both factors to 1.
        let ptr = unsafe { alloc(layout) };
        assert!(!ptr.is_null(), "pool: allocation of {bytes} bytes failed");
        assert_eq!(
            ptr as usize % align,
            0,
            "pool: the allocator returned a block not aligned to {align}"
        );
        Block {
            ptr,
            bytes,
            align,
            in_use: true,
        }
    }

    fn free_block_at(blocks: &mut [Block], index: usize) {
        let block = &blocks[index];
        let layout = Layout::from_size_align(block.bytes, block.align)
            .expect("pool: layout was valid at allocation and must still be");
        // Safety: the pointer came from `alloc` with this exact layout and is
        // not aliased, because the block is free.
        unsafe { dealloc(block.ptr, layout) };
        blocks[index].ptr = std::ptr::null_mut();
        blocks[index].bytes = 0;
    }

    fn release(&self, index: usize) {
        self.blocks.borrow_mut()[index].in_use = false;
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        let mut blocks = self.blocks.borrow_mut();
        for i in 0..blocks.len() {
            if !blocks[i].ptr.is_null() {
                Self::free_block_at(&mut blocks, i);
            }
        }
    }
}

/// A checked-out block, released back to the pool when dropped.
pub struct PooledSlice<'a, T> {
    slice: &'a mut [T],
    index: usize,
    pool: &'a Pool,
}

impl<T> std::ops::Deref for PooledSlice<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.slice
    }
}

impl<T> std::ops::DerefMut for PooledSlice<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.slice
    }
}

impl<T> Drop for PooledSlice<'_, T> {
    fn drop(&mut self) {
        self.pool.release(self.index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_block_is_reused_rather_than_reallocated() {
        let pool = Pool::new();
        {
            let a = pool.get::<f32>(1024);
            assert_eq!(a.len(), 1024);
        }
        let before = pool.block_count();
        for _ in 0..50 {
            let b = pool.get::<f32>(1024);
            assert_eq!(b.len(), 1024);
        }
        assert_eq!(
            pool.block_count(),
            before,
            "a repeated request of the same size must not grow the pool"
        );
    }

    #[test]
    fn the_pool_stops_growing_at_the_high_water_mark() {
        let pool = Pool::new();
        // Rising sizes, one at a time: each should grow the free block in place
        // rather than orphaning it, so the count stays at one.
        for size in [16usize, 64, 256, 1024, 4096, 16384] {
            let s = pool.get::<f32>(size);
            assert_eq!(s.len(), size);
        }
        assert_eq!(
            pool.block_count(),
            1,
            "rising one-at-a-time requests must grow one block, not accumulate; \
             this is the shape that caused an OOM on the CUDA side"
        );
    }

    #[test]
    fn simultaneous_checkouts_do_not_alias() {
        // The property the pool exists for, and the one a `&mut self` signature
        // silently removes: the solver holds several temporaries in one scope.
        let pool = Pool::new();
        let mut a = pool.get::<f32>(128);
        let mut b = pool.get::<f32>(128);
        let mut c = pool.get::<u32>(64);
        assert_eq!(pool.block_count(), 3, "three live checkouts need three blocks");

        a.fill(1.0);
        b.fill(2.0);
        c.fill(7);
        assert!(a.iter().all(|v| *v == 1.0), "b's writes reached a: the blocks alias");
        assert!(b.iter().all(|v| *v == 2.0), "a's or c's writes reached b");
        assert!(c.iter().all(|v| *v == 7), "a float write reached the u32 block");

        // Spans must be disjoint in memory, which is the stronger statement.
        let (pa, pb) = (a.as_ptr() as usize, b.as_ptr() as usize);
        let (la, lb) = (a.len() * 4, b.len() * 4);
        assert!(
            pa + la <= pb || pb + lb <= pa,
            "two live checkouts overlap: [{pa}, {}) against [{pb}, {})",
            pa + la,
            pb + lb
        );
    }

    #[test]
    fn a_released_block_is_reused_by_the_next_checkout() {
        let pool = Pool::new();
        let first = {
            let a = pool.get::<f32>(256);
            a.as_ptr() as usize
        };
        let second = {
            let b = pool.get::<f32>(256);
            b.as_ptr() as usize
        };
        assert_eq!(
            first, second,
            "a released block must be handed back rather than a fresh one              allocated; otherwise the pool is a slow malloc"
        );
        assert_eq!(pool.block_count(), 1);
    }

    #[test]
    fn a_block_is_zeroed_on_checkout() {
        let pool = Pool::new();
        {
            let mut a = pool.get::<u32>(64);
            for v in a.iter_mut() {
                *v = 0xDEAD_BEEF;
            }
        }
        let b = pool.get::<u32>(64);
        assert!(
            b.iter().all(|v| *v == 0),
            "a recycled block must come back zeroed; handing back the previous \
             tenant's bytes is the uninitialized-read class of defect"
        );
    }

    #[test]
    fn alignment_is_satisfied_per_type_not_by_accident() {
        let pool = Pool::new();
        // Check out and release a float-backed block first, so a naive pool
        // would hand the same memory back for the wider type.
        {
            let _f = pool.get::<f32>(1024);
        }
        let wide = pool.get::<u64>(512);
        assert_eq!(
            wide.as_ptr() as usize % std::mem::align_of::<u64>(),
            0,
            "an 8-byte type must not be served from a 4-byte-aligned block"
        );
    }

    #[test]
    fn a_zero_length_request_is_valid_and_empty() {
        let pool = Pool::new();
        let s = pool.get::<f32>(0);
        assert_eq!(s.len(), 0);
    }
}
