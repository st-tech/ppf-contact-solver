// File: cvecvec.rs
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

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

#[allow(unused)]
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
    pub fn with_capacity(size: u32, nnz: u32) -> Self {
        let mut data: Vec<T> = Vec::with_capacity(nnz as usize);
        let mut offset: Vec<u32> = vec![0; size as usize + 1];
        let data_ptr = data.as_mut_ptr();
        let offset_ptr = offset.as_mut_ptr();
        std::mem::forget(data);
        std::mem::forget(offset);
        CVecVec {
            data: data_ptr,
            offset: offset_ptr,
            size,
            nnz_allocated: nnz,
            offset_allocated: size + 1,
            nnz: 0,
        }
    }

    pub fn row_iter(&self, i: u32) -> RowIter<'_, T> {
        assert!(i < self.size, "Row index out of bounds");
        let start = unsafe { *self.offset.add(i as usize) as usize };
        let end = unsafe { *self.offset.add(i as usize + 1) as usize };
        RowIter {
            ptr: self.data,
            curr: start,
            end,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T> From<&[Vec<T>]> for CVecVec<T>
where
    T: Copy,
{
    fn from(slice: &[Vec<T>]) -> Self {
        let size = slice.len() as u32;
        let mut data = Vec::new();
        let mut offset = Vec::with_capacity(size as usize + 1);
        offset.push(0);
        for row in slice {
            data.extend_from_slice(row);
            offset.push(data.len() as u32);
        }
        let nnz = data.len() as u32;
        let data_ptr = data.as_mut_ptr();
        let offset_ptr = offset.as_mut_ptr();
        std::mem::forget(data);
        std::mem::forget(offset);
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
        unsafe {
            drop(Vec::from_raw_parts(
                self.data,
                self.nnz_allocated as usize,
                self.nnz_allocated as usize,
            ));
            drop(Vec::from_raw_parts(
                self.offset,
                self.offset_allocated as usize,
                self.offset_allocated as usize,
            ));
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
            write!(f, "{}: [", start)?;
            for j in start..end {
                if j > start {
                    write!(f, ", ")?;
                }
                let item = unsafe { self.data.add(j).read() };
                write!(f, "{}", item)?;
            }
            write!(f, "] :{}", end)?;
        }
        write!(f, "\n]")
    }
}

impl<T: serde::Serialize> serde::Serialize for CVecVec<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(serde::Serialize)]
        struct Inner<T> {
            pub data: Vec<T>,
            pub offset: Vec<u32>,
            pub size: u32,
            pub nnz: u32,
            pub nnz_allocated: u32,
            pub offset_allocated: u32,
        }
        let inner = Inner {
            data: unsafe {
                if self.nnz > 0 {
                    Vec::from_raw_parts(self.data, self.nnz as usize, self.nnz_allocated as usize)
                } else {
                    Vec::new()
                }
            },
            offset: unsafe {
                if self.size > 0 {
                    Vec::from_raw_parts(
                        self.offset,
                        self.size as usize + 1,
                        self.offset_allocated as usize,
                    )
                } else {
                    Vec::new()
                }
            },
            size: self.size,
            nnz: self.nnz,
            nnz_allocated: self.nnz_allocated,
            offset_allocated: self.offset_allocated,
        };
        let result = inner.serialize(serializer);
        std::mem::forget(inner.data);
        std::mem::forget(inner.offset);
        result
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
        let data_ptr = if inner.nnz > 0 {
            inner.data.as_ptr() as *mut T
        } else {
            std::ptr::null_mut()
        };
        let offset_ptr = if inner.size > 0 {
            inner.offset.as_ptr() as *mut u32
        } else {
            std::ptr::null_mut()
        };
        std::mem::forget(inner.data);
        std::mem::forget(inner.offset);
        Ok(CVecVec {
            data: data_ptr,
            offset: offset_ptr,
            size: inner.size,
            nnz: inner.nnz,
            nnz_allocated: inner.nnz_allocated,
            offset_allocated: inner.offset_allocated,
        })
    }
}

// Iterator over a row of CVecVec
pub struct RowIter<'a, T> {
    ptr: *mut T,
    curr: usize,
    end: usize,
    _marker: std::marker::PhantomData<&'a T>,
}

impl<T> Iterator for RowIter<'_, T>
where
    T: Copy,
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr < self.end {
            let item = unsafe { *self.ptr.add(self.curr) };
            self.curr += 1;
            Some(item)
        } else {
            None
        }
    }
}
