#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
mod bindings;

use std::{ffi::c_void, os::raw::c_int};
use torch_sys::C_tensor;

pub unsafe fn upfirdn2d_raw(
    x: *const C_tensor,
    f: *const C_tensor,
    upx: c_int,
    upy: c_int,
    downx: c_int,
    downy: c_int,
    padx0: c_int,
    padx1: c_int,
    pady0: c_int,
    pady1: c_int,
    flip: bool,
    gain: f32,
) -> *mut C_tensor {
    bindings::upfirdn2d_ffi(
        x as *const c_void,
        f as *const c_void,
        upx,
        upy,
        downx,
        downy,
        padx0,
        padx1,
        pady0,
        pady1,
        flip,
        gain,
    ) as *mut C_tensor
}

#[cfg(feature = "tch")]
pub use with_tch::*;

#[cfg(feature = "tch")]
mod with_tch {
    use super::*;
    use tch::Tensor;

    pub fn upfirdn2d(
        x: &Tensor,
        f: &Tensor,
        upx: i64,
        upy: i64,
        downx: i64,
        downy: i64,
        padx0: i64,
        padx1: i64,
        pady0: i64,
        pady1: i64,
        flip: bool,
        gain: f32,
    ) -> Tensor {
        unsafe {
            let ptr = upfirdn2d_raw(
                x.as_ptr(),
                f.as_ptr(),
                upx as c_int,
                upy as c_int,
                downx as c_int,
                downy as c_int,
                padx0 as c_int,
                padx1 as c_int,
                pady0 as c_int,
                pady1 as c_int,
                flip,
                gain,
            );
            Tensor::from_ptr(ptr)
        }
    }
}
