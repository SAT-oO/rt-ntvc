//! Zero-heap-after-init transformer encoder (matches PyTorch NextFramePredictor).

use crate::gemm::{add_bias_rows, scale_inplace, sgemm, sgemm_beta};
use crate::weights::{Weights, COLS, D, DH, FFN, H, L, N_LAYERS, ROWS, S, T, V};

const EPS: f32 = 1e-5;
const SCALE: f32 = 0.125; // 1/sqrt(64)

pub struct Work {
    pub x: Vec<f32>,      // L*D
    pub xn: Vec<f32>,     // L*D
    pub qkv: Vec<f32>,    // L*3D
    pub q: Vec<f32>,      // H*L*DH packed contiguous
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub scores: Vec<f32>, // H*L*L
    pub att: Vec<f32>,    // L*D
    pub h1: Vec<f32>,     // L*FFN
    pub head: Vec<f32>,   // L*DH
    pub logits: Vec<f32>, // S*V
    pub probs: Vec<f32>,  // S*V
    pub ln_tmp: Vec<f32>, // D
}

impl Work {
    pub fn new() -> Self {
        Self {
            x: vec![0.0; L * D],
            xn: vec![0.0; L * D],
            qkv: vec![0.0; L * 3 * D],
            q: vec![0.0; H * L * DH],
            k: vec![0.0; H * L * DH],
            v: vec![0.0; H * L * DH],
            scores: vec![0.0; H * L * L],
            att: vec![0.0; L * D],
            h1: vec![0.0; L * FFN],
            head: vec![0.0; L * DH],
            logits: vec![0.0; S * V],
            probs: vec![0.0; S * V],
            ln_tmp: vec![0.0; D],
        }
    }
}

fn layer_norm(src: &[f32], dst: &mut [f32], w: &[f32], b: &[f32], rows: usize, tmp: &mut [f32]) {
    for r in 0..rows {
        let s = &src[r * D..(r + 1) * D];
        let d = &mut dst[r * D..(r + 1) * D];
        #[cfg(target_os = "macos")]
        {
            crate::gemm::accel::layer_norm_row(s, d, w, b, tmp);
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = tmp;
            let mut mean = 0.0f32;
            for v in s {
                mean += *v;
            }
            mean /= D as f32;
            let mut var = 0.0f32;
            for v in s {
                let z = *v - mean;
                var += z * z;
            }
            var /= D as f32;
            let inv = 1.0 / (var + EPS).sqrt();
            for i in 0..D {
                d[i] = (s[i] - mean) * inv * w[i] + b[i];
            }
        }
    }
}

fn relu_inplace(x: &mut [f32]) {
    for v in x {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

fn softmax_rows(mat: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut mat[r * cols..(r + 1) * cols];
        #[cfg(target_os = "macos")]
        crate::gemm::accel::softmax_row(row);
        #[cfg(not(target_os = "macos"))]
        {
            let mut m = f32::NEG_INFINITY;
            for v in row.iter() {
                if *v > m {
                    m = *v;
                }
            }
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - m).exp();
                sum += *v;
            }
            let inv = 1.0 / sum.max(1e-12);
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }
}

fn embed(w: &Weights, tokens: &[i32], x: &mut [f32]) {
    debug_assert_eq!(tokens.len(), L);
    let te = w.token_embed();
    let re = w.row_embed();
    let ce = w.col_embed();
    let pe = w.temporal_embed();
    for t in 0..T {
        for s in 0..S {
            let id = tokens[t * S + s] as usize;
            let row = s / COLS;
            let col = s % COLS;
            debug_assert!(row < ROWS && col < COLS && id < V);
            let o = (t * S + s) * D;
            let to = id * D;
            let ro = row * D;
            let co = col * D;
            let po = t * D;
            for d in 0..D {
                x[o + d] = te[to + d] + re[ro + d] + ce[co + d] + pe[po + d];
            }
        }
    }
}

fn pack_qkv(qkv: &[f32], q: &mut [f32], k: &mut [f32], v: &mut [f32]) {
    for l in 0..L {
        let base = l * 3 * D;
        for h in 0..H {
            let dst = h * L * DH + l * DH;
            q[dst..dst + DH].copy_from_slice(&qkv[base + h * DH..base + h * DH + DH]);
            k[dst..dst + DH].copy_from_slice(&qkv[base + D + h * DH..base + D + h * DH + DH]);
            v[dst..dst + DH].copy_from_slice(&qkv[base + 2 * D + h * DH..base + 2 * D + h * DH + DH]);
        }
    }
}

fn attention(w: &Weights, layer: usize, ws: &mut Work) {
    sgemm(L, 3 * D, D, &ws.xn, w.in_proj_w(layer), true, &mut ws.qkv);
    add_bias_rows(&mut ws.qkv, L, 3 * D, w.in_proj_b(layer));
    pack_qkv(&ws.qkv, &mut ws.q, &mut ws.k, &mut ws.v);

    for h in 0..H {
        let qh = &ws.q[h * L * DH..(h + 1) * L * DH];
        let kh = &ws.k[h * L * DH..(h + 1) * L * DH];
        let vh = &ws.v[h * L * DH..(h + 1) * L * DH];
        let scs = &mut ws.scores[h * L * L..(h + 1) * L * L];
        sgemm(L, L, DH, qh, kh, true, scs);
        scale_inplace(scs, SCALE);
        softmax_rows(scs, L, L);
        sgemm(L, DH, L, scs, vh, false, &mut ws.head);
        for l in 0..L {
            let dst = l * D + h * DH;
            ws.att[dst..dst + DH].copy_from_slice(&ws.head[l * DH..(l + 1) * DH]);
        }
    }

    // x += att @ W + b
    sgemm_beta(L, D, D, &ws.att, w.out_proj_w(layer), true, 1.0, &mut ws.x);
    add_bias_rows(&mut ws.x, L, D, w.out_proj_b(layer));
}

fn ffn(w: &Weights, layer: usize, ws: &mut Work) {
    sgemm(L, FFN, D, &ws.xn, w.linear1_w(layer), true, &mut ws.h1);
    add_bias_rows(&mut ws.h1, L, FFN, w.linear1_b(layer));
    relu_inplace(&mut ws.h1);
    sgemm_beta(L, D, FFN, &ws.h1, w.linear2_w(layer), true, 1.0, &mut ws.x);
    add_bias_rows(&mut ws.x, L, D, w.linear2_b(layer));
}

/// tokens: T*S int ids. Writes ws.logits (S*V) un-normalized.
pub fn forward(w: &Weights, tokens: &[i32], ws: &mut Work) {
    embed(w, tokens, &mut ws.x);
    for layer in 0..N_LAYERS {
        layer_norm(
            &ws.x,
            &mut ws.xn,
            w.norm1_w(layer),
            w.norm1_b(layer),
            L,
            &mut ws.ln_tmp,
        );
        attention(w, layer, ws);
        layer_norm(
            &ws.x,
            &mut ws.xn,
            w.norm2_w(layer),
            w.norm2_b(layer),
            L,
            &mut ws.ln_tmp,
        );
        ffn(w, layer, ws);
    }
    layer_norm(
        &ws.x,
        &mut ws.xn,
        w.final_norm_w(),
        w.final_norm_b(),
        L,
        &mut ws.ln_tmp,
    );
    let last = &ws.xn[(L - S) * D..];
    sgemm(S, V, D, last, w.output_w(), true, &mut ws.logits);
}

pub fn softmax_logits(logits: &[f32], tau: f32, out: &mut [f32]) {
    debug_assert_eq!(logits.len(), S * V);
    debug_assert_eq!(out.len(), S * V);
    let inv_tau = 1.0 / tau;
    for r in 0..S {
        let src = &logits[r * V..(r + 1) * V];
        let dst = &mut out[r * V..(r + 1) * V];
        for i in 0..V {
            dst[i] = src[i] * inv_tau;
        }
        #[cfg(target_os = "macos")]
        crate::gemm::accel::softmax_row(dst);
        #[cfg(not(target_os = "macos"))]
        {
            let mut m = f32::NEG_INFINITY;
            for v in dst.iter() {
                if *v > m {
                    m = *v;
                }
            }
            let mut sum = 0.0f32;
            for v in dst.iter_mut() {
                *v = (*v - m).exp();
                sum += *v;
            }
            let inv = 1.0 / sum.max(1e-12);
            for v in dst.iter_mut() {
                *v *= inv;
            }
        }
    }
}
