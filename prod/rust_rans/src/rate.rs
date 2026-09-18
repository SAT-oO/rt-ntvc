//! Per-frame tau + mix + intra-frame histogram + copy-from-prev.

use crate::rans::{normalize_row_inplace, probs_to_cdf_into, rans_theoretical_bits, RansEncoder};
use crate::weights::{S, V};

pub const TAU_GRID: [f32; 4] = [0.85, 1.00, 1.20, 1.45];
pub const MIX_GRID: [f32; 4] = [0.55, 0.80, 0.92, 1.00];
pub const HIST_GRID: [f32; 4] = [0.00, 0.18, 0.36, 0.55];
pub const COPY_GRID: [f32; 4] = [0.00, 0.25, 0.50, 0.75];

pub const N_TAU: usize = 4;
pub const N_MIX: usize = 4;
pub const N_HIST: usize = 4;
pub const N_COPY: usize = 4;

/// Fixed neural baseline: tau=1, mix=1, hist=0, copy=0.
pub const GLOBAL_CODE: u8 = 1 | (3 << 2);

#[inline]
pub fn pack_code(tau_i: usize, mix_i: usize, hist_i: usize, copy_i: usize) -> u8 {
    (tau_i | (mix_i << 2) | (hist_i << 4) | (copy_i << 6)) as u8
}

#[inline]
pub fn unpack_code(c: u8) -> (usize, usize, usize, usize) {
    let x = c as usize;
    (x & 3, (x >> 2) & 3, (x >> 4) & 3, (x >> 6) & 3)
}

pub fn mix_global(probs: &mut [f32], global: &[f32], mix: f32) {
    if (mix - 1.0).abs() < 1e-6 {
        return;
    }
    let gmix = 1.0 - mix;
    for r in 0..S {
        let row = &mut probs[r * V..(r + 1) * V];
        let mut sum = 0.0f32;
        for i in 0..V {
            row[i] = mix * row[i] + gmix * global[i];
            sum += row[i];
        }
        let inv = 1.0 / sum.max(1e-12);
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

/// Mix toward a Dirac at the previous frame's same-position token.
pub fn apply_copy_on_row(row: &mut [f32], prev: i32, copy: f32) {
    if copy <= 1e-8 {
        return;
    }
    let keep = 1.0 - copy;
    for v in row.iter_mut() {
        *v *= keep;
    }
    let i = prev as usize;
    if i < row.len() {
        row[i] += copy;
    }
}

/// Causal intra-frame hist mix: p' = (1-h)*p + h*Dirichlet(counts).
pub fn apply_hist_on_row(row: &mut [f32], counts: &[f32], seen: f32, hist: f32) {
    if hist <= 1e-8 {
        return;
    }
    let inv = 1.0 / (seen + V as f32 * 0.5);
    let keep = 1.0 - hist;
    let mut sum = 0.0f32;
    for i in 0..V {
        let ph = (counts[i] + 0.5) * inv;
        row[i] = keep * row[i] + hist * ph;
        sum += row[i];
    }
    let s_inv = 1.0 / sum.max(1e-12);
    for v in row.iter_mut() {
        *v *= s_inv;
    }
}

/// Theoretical rANS bits for a frame (rank-preserving vs actual length).
pub fn score_frame_bits(
    symbols: &[i32],
    probs: &[f32],
    cdf: &mut [u32],
    row: &mut [f32],
    counts: &mut [f32],
    hist: f32,
    copy: f32,
    prev: Option<&[i32]>,
) -> f32 {
    counts.fill(0.0);
    let mut seen = 0.0f32;
    let mut bits = 0.0f32;
    for i in 0..S {
        row.copy_from_slice(&probs[i * V..(i + 1) * V]);
        if copy > 1e-8 {
            if let Some(p) = prev {
                apply_copy_on_row(row, p[i], copy);
            }
        }
        apply_hist_on_row(row, counts, seen, hist);
        normalize_row_inplace(row);
        probs_to_cdf_into(row, cdf);
        let s = symbols[i] as usize;
        let freq = cdf[s + 1] - cdf[s];
        bits += rans_theoretical_bits(freq);
        counts[s] += 1.0;
        seen += 1.0;
    }
    bits
}

/// Encode one frame (S symbols) into `scratch` (cleared, capacity kept). Returns byte length.
#[allow(dead_code)]
pub fn encode_frame_len(
    symbols: &[i32],
    probs: &[f32],
    cdf: &mut [u32],
    row: &mut [f32],
    scratch: &mut Vec<u8>,
) -> usize {
    scratch.clear();
    let mut enc = RansEncoder::from_vec(std::mem::take(scratch));
    for i in (0..S).rev() {
        row.copy_from_slice(&probs[i * V..(i + 1) * V]);
        normalize_row_inplace(row);
        probs_to_cdf_into(row, cdf);
        let s = symbols[i] as usize;
        let freq = cdf[s + 1] - cdf[s];
        enc.encode(freq, cdf[s]);
    }
    let out = enc.flush();
    let n = out.len();
    *scratch = out;
    n
}

#[allow(dead_code)]
pub fn nll_bits_row(symbol: usize, row: &[f32]) -> f32 {
    let p = row[symbol].max(1e-12);
    -p.log2()
}

#[allow(dead_code)]
pub fn nll_bits_frame(symbols: &[i32], probs: &[f32]) -> f32 {
    let mut b = 0.0f32;
    for i in 0..S {
        let s = symbols[i] as usize;
        b += nll_bits_row(s, &probs[i * V..(i + 1) * V]);
    }
    b
}
