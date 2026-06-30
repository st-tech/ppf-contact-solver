// File: cvec.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use super::raw_vec::{borrow_slice, leak_vec, reclaim};
use std::fmt;

#[repr(C)]
pub struct CVec<T> {
    pub data: *mut T,
    pub size: u32,
    pub allocated: u32,
}

unsafe impl<T> Send for CVec<T> where T: Send {}

impl<T> CVec<T> {
    pub fn new() -> Self {
        Self {
            data: std::ptr::null_mut(),
            size: 0,
            allocated: 0,
        }
    }
}

impl<T> From<&[T]> for CVec<T>
where
    T: Copy,
{
    fn from(slice: &[T]) -> Self {
        let size = slice.len() as u32;
        // with_capacity(size) keeps the real capacity equal to size, so allocated
        // doubles as both length and capacity on Drop.
        let mut data = Vec::with_capacity(size as usize);
        data.extend_from_slice(slice);
        let (ptr, _len, _cap) = leak_vec(data);
        CVec {
            data: ptr,
            size,
            allocated: size,
        }
    }
}

impl<T> Drop for CVec<T> {
    fn drop(&mut self) {
        // reclaim returns early when data is null (new() leaves it null with allocated == 0)
        // and otherwise rebuilds the Vec with len == cap == allocated to free it.
        unsafe {
            reclaim(self.data, self.allocated as usize);
        }
    }
}

impl<T: fmt::Display> fmt::Display for CVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for i in 0..self.size {
            if i > 0 {
                write!(f, ", ")?;
            }
            let item = unsafe { self.data.add(i as usize).read() };
            write!(f, "{item}")?;
        }
        write!(f, "]")
    }
}

pub struct CVecIter<'a, T> {
    ptr: *const T,
    remaining: usize,
    _marker: std::marker::PhantomData<&'a T>,
}

impl<T> CVec<T> {
    /// Borrow the contents as a slice. Empty when null or zero-length.
    pub fn as_slice(&self) -> &[T] {
        if self.data.is_null() || self.size == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.data, self.size as usize) }
        }
    }

    pub fn iter(&self) -> CVecIter<'_, T> {
        CVecIter {
            ptr: self.data,
            remaining: self.size as usize,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, T> Iterator for CVecIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining > 0 {
            let item = unsafe { &*self.ptr };
            self.ptr = unsafe { self.ptr.add(1) };
            self.remaining -= 1;
            Some(item)
        } else {
            None
        }
    }
}

impl<T: serde::Serialize> serde::Serialize for CVec<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Borrow the live buffer as a slice instead of materializing an owning Vec; the
        // serialized seq is byte-identical to a Vec under bincode, so the wire layout is
        // unchanged. borrow_slice centralizes the null/empty guard.
        #[derive(serde::Serialize)]
        struct Inner<'a, T> {
            pub data: &'a [T],
            pub size: u32,
            pub allocated: u32,
        }
        let inner = Inner {
            data: unsafe { borrow_slice(self.data, self.size as usize) },
            size: self.size,
            allocated: self.allocated,
        };
        inner.serialize(serializer)
    }
}

impl<'de, T: serde::Deserialize<'de>> serde::Deserialize<'de> for CVec<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Inner<T> {
            pub data: Vec<T>,
            pub size: u32,
            pub allocated: u32,
        }
        let inner: Inner<T> = Inner::deserialize(deserializer)?;
        // Drop reconstructs the Vec from `data`/`allocated`, so the stored capacity must
        // match the kept pointer's real allocation. Derive it from the decoded Vec's own
        // capacity (not the serialized `allocated` field, which only coincides under
        // bincode) and keep the null branch at capacity 0, since a null pointer with a
        // non-zero capacity would be undefined behavior in Drop.
        let (data_ptr, allocated) = if inner.size > 0 {
            let (ptr, _len, cap) = leak_vec(inner.data);
            (ptr, cap as u32)
        } else {
            (std::ptr::null_mut(), 0)
        };
        Ok(CVec {
            data: data_ptr,
            size: inner.size,
            allocated,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // I497: Drop on a null-pointer CVec (from new()) must not call
    // Vec::from_raw_parts(null, 0, 0), which aborts under current rustc.
    #[test]
    fn new_is_null_and_drops_without_abort() {
        let v: CVec<u32> = CVec::new();
        assert!(v.data.is_null());
        assert_eq!((v.size, v.allocated), (0, 0));
        drop(v);
    }

    // I496: from() keeps the real capacity equal to size, so Drop frees the
    // correct Layout; the data round-trips through as_slice().
    #[test]
    fn from_slice_roundtrips_and_capacity_matches() {
        let src = [10u32, 20, 30, 40];
        let v = CVec::from(&src[..]);
        assert_eq!((v.size, v.allocated), (4, 4));
        assert_eq!(v.as_slice(), &src);
        drop(v);
    }

    #[test]
    fn empty_from_slice_drops_safely() {
        let empty: [u32; 0] = [];
        let v = CVec::from(&empty[..]);
        assert_eq!(v.size, 0);
        assert!(v.as_slice().is_empty());
        drop(v);
    }

    // I498: Deserialize derives the allocated capacity from the decoded Vec
    // (not the serialized field), so Drop frees the right Layout.
    #[test]
    fn serde_roundtrip_preserves_data_and_drops_safely() {
        let src = [1u32, 2, 3, 4, 5];
        let v = CVec::from(&src[..]);
        let bytes = bincode::serialize(&v).unwrap();
        let back: CVec<u32> = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back.size, 5);
        assert_eq!(back.as_slice(), &src);
        drop(back);
        drop(v);
    }
}
