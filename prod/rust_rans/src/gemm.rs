//! Row-major f32 GEMM + Accelerate vDSP helpers. No heap.

#[cfg(target_os = "macos")]
pub mod accel {
    pub const ROW_MAJOR: i32 = 101;
    pub const NO_TRANS: i32 = 111;
    pub const TRANS: i32 = 112;

    unsafe extern "C" {
        pub fn cblas_sgemm(
            order: i32,
            transa: i32,
            transb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
        pub fn vDSP_maxv(a: *const f32, ia: isize, c: *mut f32, n: usize);
        pub fn vDSP_sve(a: *const f32, ia: isize, c: *mut f32, n: usize);
        pub fn vDSP_vsadd(a: *const f32, ia: isize, b: *const f32, c: *mut f32, ic: isize, n: usize);
        pub fn vDSP_vsdiv(a: *const f32, ia: isize, b: *const f32, c: *mut f32, ic: isize, n: usize);
        pub fn vvexpf(y: *mut f32, x: *const f32, n: *const i32);
        pub fn vDSP_meanv(a: *const f32, ia: isize, c: *mut f32, n: usize);
        pub fn vDSP_vsq(a: *const f32, ia: isize, c: *mut f32, ic: isize, n: usize);
        pub fn vDSP_vsmul(a: *const f32, ia: isize, b: *const f32, c: *mut f32, ic: isize, n: usize);
        pub fn vDSP_vma(
            a: *const f32,
            ia: isize,
            b: *const f32,
            ib: isize,
            c: *const f32,
            ic: isize,
            d: *mut f32,
            id: isize,
            n: usize,
        );
        pub fn vDSP_vadd(
            a: *const f32,
            ia: isize,
            b: *const f32,
            ib: isize,
            c: *mut f32,
            ic: isize,
            n: usize,
        );
        pub fn cblas_saxpy(n: i32, alpha: f32, x: *const f32, incx: i32, y: *mut f32, incy: i32);
    }

    pub fn layer_norm_row(src: &[f32], dst: &mut [f32], w: &[f32], b: &[f32], tmp: &mut [f32]) {
        let n = src.len();
        let mut mean = 0.0f32;
        unsafe {
            vDSP_meanv(src.as_ptr(), 1, &mut mean, n);
            let neg = -mean;
            vDSP_vsadd(src.as_ptr(), 1, &neg, tmp.as_mut_ptr(), 1, n);
            vDSP_vsq(tmp.as_ptr(), 1, dst.as_mut_ptr(), 1, n);
            let mut msq = 0.0f32;
            vDSP_meanv(dst.as_ptr(), 1, &mut msq, n);
            let inv = 1.0 / (msq + 1e-5).sqrt();
            vDSP_vsmul(tmp.as_ptr(), 1, &inv, tmp.as_mut_ptr(), 1, n);
            // dst = tmp * w + b
            for i in 0..n {
                dst[i] = tmp[i] * w[i] + b[i];
            }
        }
    }

    pub fn sgemm(
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        trans_b: bool,
        beta: f32,
        c: &mut [f32],
    ) {
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(c.len(), m * n);
        debug_assert_eq!(b.len(), if trans_b { n * k } else { k * n });
        let transb = if trans_b { TRANS } else { NO_TRANS };
        let ldb = if trans_b { k as i32 } else { n as i32 };
        unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                NO_TRANS,
                transb,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                k as i32,
                b.as_ptr(),
                ldb,
                beta,
                c.as_mut_ptr(),
                n as i32,
            );
        }
    }

    pub fn scale_inplace(x: &mut [f32], s: f32) {
        unsafe {
            vDSP_vsmul(x.as_ptr(), 1, &s, x.as_mut_ptr(), 1, x.len());
        }
    }

    pub fn add_inplace(dst: &mut [f32], src: &[f32]) {
        debug_assert_eq!(dst.len(), src.len());
        unsafe {
            cblas_saxpy(
                dst.len() as i32,
                1.0,
                src.as_ptr(),
                1,
                dst.as_mut_ptr(),
                1,
            );
        }
    }

    pub fn add_bias_rows(y: &mut [f32], rows: usize, cols: usize, bias: &[f32]) {
        debug_assert_eq!(bias.len(), cols);
        for r in 0..rows {
            let row = &mut y[r * cols..(r + 1) * cols];
            unsafe {
                vDSP_vadd(row.as_ptr(), 1, bias.as_ptr(), 1, row.as_mut_ptr(), 1, cols);
            }
        }
    }

    pub fn sgemm_strided(
        m: usize,
        n: usize,
        k: usize,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        c: *mut f32,
        ldc: i32,
    ) {
        unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                NO_TRANS,
                TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a,
                lda,
                b,
                ldb,
                0.0,
                c,
                ldc,
            );
        }
    }

    pub fn sgemm_nn_strided(
        m: usize,
        n: usize,
        k: usize,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        c: *mut f32,
        ldc: i32,
    ) {
        unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                NO_TRANS,
                NO_TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a,
                lda,
                b,
                ldb,
                0.0,
                c,
                ldc,
            );
        }
    }

    pub fn softmax_row(row: &mut [f32]) {
        let n = row.len();
        let n_i = n as i32;
        let mut m = 0.0f32;
        unsafe {
            vDSP_maxv(row.as_ptr(), 1, &mut m, n);
            let neg = -m;
            vDSP_vsadd(row.as_ptr(), 1, &neg, row.as_mut_ptr(), 1, n);
            vvexpf(row.as_mut_ptr(), row.as_ptr(), &n_i);
            let mut sum = 0.0f32;
            vDSP_sve(row.as_ptr(), 1, &mut sum, n);
            let s = if sum.abs() < 1e-12 { 1e-12 } else { sum };
            vDSP_vsdiv(row.as_ptr(), 1, &s, row.as_mut_ptr(), 1, n);
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod accel {
    pub fn sgemm(
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        trans_b: bool,
        beta: f32,
        c: &mut [f32],
    ) {
        let rsb = if trans_b { 1isize } else { n as isize };
        let csb = if trans_b { k as isize } else { 1isize };
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as isize,
                1,
                b.as_ptr(),
                rsb,
                csb,
                beta,
                c.as_mut_ptr(),
                n as isize,
                1,
            );
        }
    }

    pub fn scale_inplace(x: &mut [f32], s: f32) {
        for v in x {
            *v *= s;
        }
    }

    pub fn add_inplace(dst: &mut [f32], src: &[f32]) {
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d += *s;
        }
    }

    pub fn add_bias_rows(y: &mut [f32], rows: usize, cols: usize, bias: &[f32]) {
        for r in 0..rows {
            let row = &mut y[r * cols..(r + 1) * cols];
            for (v, b) in row.iter_mut().zip(bias.iter()) {
                *v += *b;
            }
        }
    }
}

#[inline]
pub fn sgemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], trans_b: bool, c: &mut [f32]) {
    accel::sgemm(m, n, k, a, b, trans_b, 0.0, c);
}

#[inline]
pub fn sgemm_beta(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    trans_b: bool,
    beta: f32,
    c: &mut [f32],
) {
    accel::sgemm(m, n, k, a, b, trans_b, beta, c);
}

#[inline]
pub fn add_bias_rows(y: &mut [f32], rows: usize, cols: usize, bias: &[f32]) {
    accel::add_bias_rows(y, rows, cols, bias);
}

#[inline]
pub fn add_inplace(dst: &mut [f32], src: &[f32]) {
    accel::add_inplace(dst, src);
}

#[inline]
pub fn scale_inplace(x: &mut [f32], s: f32) {
    accel::scale_inplace(x, s);
}
