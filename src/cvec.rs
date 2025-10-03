// File: cvec.rs
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

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
        let mut data = Vec::with_capacity(size as usize);
        data.extend_from_slice(slice);
        let ptr = data.as_mut_ptr();
        std::mem::forget(data);
        CVec {
            data: ptr,
            size,
            allocated: size,
        }
    }
}

impl<T> Drop for CVec<T> {
    fn drop(&mut self) {
        unsafe {
            drop(Vec::from_raw_parts(
                self.data,
                self.allocated as usize,
                self.allocated as usize,
            ));
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
            write!(f, "{}", item)?;
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
        #[derive(serde::Serialize)]
        struct Inner<T> {
            pub data: Vec<T>,
            pub size: u32,
            pub allocated: u32,
        }
        let inner = Inner {
            data: unsafe {
                Vec::from_raw_parts(self.data, self.size as usize, self.allocated as usize)
            },
            size: self.size,
            allocated: self.allocated,
        };
        let result = inner.serialize(serializer);
        std::mem::forget(inner.data);
        result
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
        let data_ptr = if inner.size > 0 {
            inner.data.as_ptr() as *mut T
        } else {
            std::ptr::null_mut()
        };
        std::mem::forget(inner.data);
        Ok(CVec {
            data: data_ptr,
            size: inner.size,
            allocated: inner.allocated,
        })
    }
}
