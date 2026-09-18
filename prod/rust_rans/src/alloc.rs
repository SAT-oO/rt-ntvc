//! Count Rust heap ops on the hot decode path (R1).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub struct CountingAlloc;

static HOT: AtomicBool = AtomicBool::new(false);
static COUNT: AtomicUsize = AtomicUsize::new(0);
static LAST: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump();
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        bump();
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump();
        System.realloc(ptr, layout, new_size)
    }
}

#[inline]
fn bump() {
    if HOT.load(Ordering::Relaxed) {
        COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn begin_hot() {
    COUNT.store(0, Ordering::Relaxed);
    HOT.store(true, Ordering::Relaxed);
}

pub fn end_hot() -> usize {
    HOT.store(false, Ordering::Relaxed);
    let n = COUNT.load(Ordering::Relaxed);
    LAST.store(n, Ordering::Relaxed);
    n
}

pub fn last() -> usize {
    LAST.load(Ordering::Relaxed)
}
