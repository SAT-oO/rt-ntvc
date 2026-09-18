//! C ABI for the decode session (NVCR-style).

use std::ffi::{c_char, c_int, CStr};
use std::path::Path;
use std::ptr;
use std::slice;

use crate::session::Engine;
use crate::weights::{S, V};

#[no_mangle]
pub extern "C" fn rtnv_engine_load(weight_path: *const c_char, global_freq: *const f32, vocab: c_int) -> *mut Engine {
    if weight_path.is_null() || global_freq.is_null() || vocab != V as c_int {
        return ptr::null_mut();
    }
    let path = unsafe { CStr::from_ptr(weight_path) };
    let Ok(path) = path.to_str() else {
        return ptr::null_mut();
    };
    let global = unsafe { slice::from_raw_parts(global_freq, V) };
    match Engine::load(Path::new(path), global) {
        Ok(e) => Box::into_raw(Box::new(e)),
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn rtnv_engine_free(eng: *mut Engine) {
    if !eng.is_null() {
        unsafe {
            drop(Box::from_raw(eng));
        }
    }
}

#[no_mangle]
pub extern "C" fn rtnv_encode(
    eng: *mut Engine,
    tokens: *const i32,
    n_frames: c_int,
    s: c_int,
    out: *mut u8,
    out_cap: usize,
    out_len: *mut usize,
    adaptive: c_int,
) -> c_int {
    if eng.is_null() || tokens.is_null() || out.is_null() || out_len.is_null() || s != S as c_int {
        return -1;
    }
    let n = n_frames as usize;
    let tok = unsafe { slice::from_raw_parts(tokens, n * S) };
    let blob = unsafe { (*eng).encode(tok, n, adaptive != 0) };
    if blob.len() > out_cap {
        return -2;
    }
    unsafe {
        ptr::copy_nonoverlapping(blob.as_ptr(), out, blob.len());
        *out_len = blob.len();
    }
    0
}

#[no_mangle]
pub extern "C" fn rtnv_decode(
    eng: *mut Engine,
    bits: *const u8,
    bits_len: usize,
    n_frames: c_int,
    tokens_out: *mut i32,
) -> c_int {
    if eng.is_null() || bits.is_null() || tokens_out.is_null() {
        return -1;
    }
    let data = unsafe { slice::from_raw_parts(bits, bits_len) };
    let hint = if n_frames > 0 { Some(n_frames as usize) } else { None };
    match unsafe { (*eng).decode(data, hint) } {
        Ok(nf) => {
            let n = nf * S;
            unsafe {
                ptr::copy_nonoverlapping((*eng).tokens(nf).as_ptr(), tokens_out, n);
            }
            nf as c_int
        }
        Err(_) => -3,
    }
}

#[no_mangle]
pub extern "C" fn rtnv_hot_allocs() -> usize {
    crate::alloc::last()
}
