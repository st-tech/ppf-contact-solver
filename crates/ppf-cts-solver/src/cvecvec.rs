// File: cvecvec.rs
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

use super::raw_vec::{borrow_slice, leak_vec, reclaim};
use std::fmt;

#[repr(C)]
pub struct CVecVec<T> {
    pub data: *mut T,
    pub offset: *mut u32,
    pub size: u32,
    pub nnz: u32,
    pub nnz_allocated: u32,
    pub offset_allocated: u32,
}

unsafe impl<T> Send for CVecVec<T> where T: Send {}

impl<T> CVecVec<T> {
    pub fn new() -> Self {
        Self {
            data: std::ptr::null_mut(),
            offset: std::ptr::null_mut(),
            size: 0,
            nnz_allocated: 0,
            offset_allocated: 0,
            nnz: 0,
        }
    }
}

impl<T> From<&[Vec<T>]> for CVecVec<T>
where
    T: Copy,
{
    fn from(slice: &[Vec<T>]) -> Self {
        let size = slice.len() as u32;
        // Preallocate the exact data capacity so the real allocated capacity equals nnz.
        // Vec::new() + extend_from_slice grows geometrically (capacity > len), which would
        // leave nnz_allocated smaller than the true capacity and feed the wrong Layout to
        // the allocator on Drop. Sizing up front keeps capacity == len == nnz_allocated.
        let total: usize = slice.iter().map(|row| row.len()).sum();
        let mut data = Vec::with_capacity(total);
        let mut offset = Vec::with_capacity(size as usize + 1);
        offset.push(0);
        for row in slice {
            data.extend_from_slice(row);
            offset.push(data.len() as u32);
        }
        let nnz = data.len() as u32;
        let (data_ptr, _data_len, _data_cap) = leak_vec(data);
        let (offset_ptr, _offset_len, _offset_cap) = leak_vec(offset);
        CVecVec {
            data: data_ptr,
            offset: offset_ptr,
            size,
            nnz_allocated: nnz,
            offset_allocated: size + 1,
            nnz,
        }
    }
}

impl<T> Drop for CVecVec<T> {
    fn drop(&mut self) {
        // reclaim returns early for each null pointer (new() leaves data and offset
        // independently nullable with allocated counts == 0) and otherwise rebuilds the
        // Vec with len == cap == allocated to free it.
        unsafe {
            reclaim(self.data, self.nnz_allocated as usize);
            reclaim(self.offset, self.offset_allocated as usize);
        }
    }
}

impl<T: fmt::Display> fmt::Display for CVecVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for i in 0..self.size {
            if i > 0 {
                write!(f, ",\n ")?;
            } else {
                write!(f, "\n ")?;
            }
            let start = unsafe { *self.offset.add(i as usize) as usize };
            let end = unsafe { *self.offset.add(i as usize + 1) as usize };
            write!(f, "{start}: [")?;
            for j in start..end {
                if j > start {
                    write!(f, ", ")?;
                }
                let item = unsafe { self.data.add(j).read() };
                write!(f, "{item}")?;
            }
            write!(f, "] :{end}")?;
        }
        write!(f, "\n]")
    }
}

impl<T: serde::Serialize> serde::Serialize for CVecVec<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Borrow each live buffer as a slice instead of materializing owning Vecs; the
        // serialized seqs are byte-identical to Vecs under bincode, so the wire layout is
        // unchanged. borrow_slice centralizes the null/empty guard. The offset buffer holds
        // size + 1 entries (the prefix-sum bounds), so borrow that many.
        #[derive(serde::Serialize)]
        struct Inner<'a, T> {
            pub data: &'a [T],
            pub offset: &'a [u32],
            pub size: u32,
            pub nnz: u32,
            pub nnz_allocated: u32,
            pub offset_allocated: u32,
        }
        let offset_len = if self.size > 0 {
            self.size as usize + 1
        } else {
            0
        };
        let inner = Inner {
            data: unsafe { borrow_slice(self.data, self.nnz as usize) },
            offset: unsafe { borrow_slice(self.offset, offset_len) },
            size: self.size,
            nnz: self.nnz,
            nnz_allocated: self.nnz_allocated,
            offset_allocated: self.offset_allocated,
        };
        inner.serialize(serializer)
    }
}

impl<'de, T: serde::Deserialize<'de>> serde::Deserialize<'de> for CVecVec<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Inner<T> {
            pub data: Vec<T>,
            pub offset: Vec<u32>,
            pub size: u32,
            pub nnz: u32,
            pub nnz_allocated: u32,
            pub offset_allocated: u32,
        }
        let inner: Inner<T> = Inner::deserialize(deserializer)?;
        // Drop reconstructs each Vec from its pointer plus the stored allocated count, so
        // those counts must match the kept pointers' real allocations. Derive them from the
        // decoded Vecs' own capacities (not the serialized `*_allocated` fields, which only
        // coincide under bincode) and keep the null branches at capacity 0, since a null
        // pointer with a non-zero capacity would be undefined behavior in Drop.
        let (data_ptr, nnz_allocated) = if inner.nnz > 0 {
            let (ptr, _len, cap) = leak_vec(inner.data);
            (ptr, cap as u32)
        } else {
            (std::ptr::null_mut(), 0)
        };
        let (offset_ptr, offset_allocated) = if inner.size > 0 {
            let (ptr, _len, cap) = leak_vec(inner.offset);
            (ptr, cap as u32)
        } else {
            (std::ptr::null_mut(), 0)
        };
        Ok(CVecVec {
            data: data_ptr,
            offset: offset_ptr,
            size: inner.size,
            nnz: inner.nnz,
            nnz_allocated,
            offset_allocated,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // I497: Drop on a null-pointer CVecVec (from new(), independently nullable
    // data and offset) must not abort via Vec::from_raw_parts(null, 0, 0).
    #[test]
    fn new_is_null_and_drops_without_abort() {
        let v: CVecVec<u32> = CVecVec::new();
        assert!(v.data.is_null());
        assert!(v.offset.is_null());
        drop(v);
    }

    // I496: the data buffer's real capacity must equal nnz (and the offset
    // buffer's capacity == size + 1), so Drop frees the correct Layout. Also
    // locks the prefix-sum packing, including an empty middle row.
    #[test]
    fn from_rows_packs_offsets_and_drops_safely() {
        let rows = vec![vec![1u32, 2, 3], Vec::<u32>::new(), vec![4, 5]];
        let v = CVecVec::from(&rows[..]);
        assert_eq!((v.size, v.nnz), (3, 5));
        assert_eq!((v.nnz_allocated, v.offset_allocated), (5, 4));
        let offsets: Vec<u32> =
            (0..=v.size as usize).map(|i| unsafe { *v.offset.add(i) }).collect();
        assert_eq!(offsets, vec![0, 3, 3, 5]);
        let data: Vec<u32> =
            (0..v.nnz as usize).map(|i| unsafe { *v.data.add(i) }).collect();
        assert_eq!(data, vec![1, 2, 3, 4, 5]);
        drop(v);
    }

    // I498: serde round-trip preserves the packing and derives the drop
    // capacities from the decoded Vecs' real capacities.
    #[test]
    fn serde_roundtrip_preserves_and_drops_safely() {
        let rows = vec![vec![7u32, 8], vec![9]];
        let v = CVecVec::from(&rows[..]);
        let bytes = bincode::serialize(&v).unwrap();
        let back: CVecVec<u32> = bincode::deserialize(&bytes).unwrap();
        assert_eq!((back.size, back.nnz), (2, 3));
        let data: Vec<u32> =
            (0..back.nnz as usize).map(|i| unsafe { *back.data.add(i) }).collect();
        assert_eq!(data, vec![7, 8, 9]);
        drop(back);
        drop(v);
    }
}

