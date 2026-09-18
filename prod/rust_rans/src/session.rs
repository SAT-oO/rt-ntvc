//! Decode/encode session: all scratch allocated at init.

use std::path::Path;

use crate::model::{forward, softmax_logits, Work};
use crate::rans::{normalize_row_inplace, probs_to_cdf_into, RansDecoder, RansEncoder};
use crate::rate::{
    apply_copy_on_row, apply_hist_on_row, mix_global, pack_code, score_frame_bits, unpack_code,
    COPY_GRID, GLOBAL_CODE, HIST_GRID, MIX_GRID, TAU_GRID,
};
use crate::weights::{Weights, S, T, V};

pub const MAGIC: &[u8; 4] = b"RTN1";
pub const FLAG_ADAPTIVE: u32 = 1;
pub const FLAG_NEURAL: u32 = 2;
pub const MAX_FRAMES: usize = 1200;
const BITS_CAP: usize = 4 * 1024 * 1024;
const CDF_STRIDE: usize = V + 1;

pub struct Engine {
    weights: Weights,
    global: Vec<f32>,
    work: Work,
    ctx: Vec<i32>,
    tokens: Vec<i32>,
    cdf: Vec<u32>,
    cdf_table: Vec<u32>,
    row: Vec<f32>,
    counts: Vec<f32>,
    codebook: Vec<u8>,
    bits: Vec<u8>,
    scratch: Vec<u8>,
    tmp_logits: Vec<f32>,
}

impl Engine {
    pub fn load(weight_path: &Path, global: &[f32]) -> std::io::Result<Self> {
        if global.len() != V {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "global_freq len",
            ));
        }
        let weights = Weights::load(weight_path)?;
        Ok(Self::from_weights(weights, global))
    }

    pub fn from_weights(weights: Weights, global: &[f32]) -> Self {
        let mut g = global.to_vec();
        let s: f32 = g.iter().sum();
        if s > 0.0 {
            for v in g.iter_mut() {
                *v /= s;
            }
        }
        let scratch = Vec::with_capacity(BITS_CAP);
        let bits = Vec::with_capacity(BITS_CAP);
        Self {
            weights,
            global: g,
            work: Work::new(),
            ctx: vec![0; T * S],
            tokens: vec![0; MAX_FRAMES * S],
            cdf: vec![0u32; CDF_STRIDE],
            cdf_table: vec![0u32; S * CDF_STRIDE],
            row: vec![0f32; V],
            counts: vec![0f32; V],
            codebook: vec![0u8; MAX_FRAMES],
            bits,
            scratch,
            tmp_logits: vec![0f32; S * V],
        }
    }

    fn build_ctx(&mut self, n_written: usize) {
        if n_written == 0 {
            self.ctx.fill(0);
            return;
        }
        let start = n_written.saturating_sub(T);
        let avail = n_written - start;
        let pad = T - avail;
        for i in 0..pad {
            self.ctx[i * S..(i + 1) * S].copy_from_slice(&self.tokens[0..S]);
        }
        for i in 0..avail {
            let src = (start + i) * S;
            let dst = (pad + i) * S;
            self.ctx[dst..dst + S].copy_from_slice(&self.tokens[src..src + S]);
        }
    }

    #[allow(dead_code)]
    fn predict_into_logits(&mut self, frame: usize) {
        if frame == 0 {
            return;
        }
        self.build_ctx(frame);
        forward(&self.weights, &self.ctx, &mut self.work);
    }

    fn probs_from_logits(&mut self, tau: f32, mix: f32) {
        softmax_logits(&self.work.logits, tau, &mut self.work.probs);
        mix_global(&mut self.work.probs, &self.global, mix);
    }

    fn tile_global(&mut self) {
        for r in 0..S {
            self.work.probs[r * V..(r + 1) * V].copy_from_slice(&self.global);
        }
    }

    fn pick_code(&mut self, frame: usize, symbols: &[i32]) -> u8 {
        let mut prev_buf = [0i32; S];
        let has_prev = frame > 0;
        if has_prev {
            prev_buf.copy_from_slice(&self.tokens[(frame - 1) * S..frame * S]);
        }
        let mut best = GLOBAL_CODE;
        let mut best_bits = f32::INFINITY;
        if !has_prev {
            self.tile_global();
            for hist_i in 0..HIST_GRID.len() {
                let bits = score_frame_bits(
                    symbols,
                    &self.work.probs,
                    &mut self.cdf,
                    &mut self.row,
                    &mut self.counts,
                    HIST_GRID[hist_i],
                    0.0,
                    None,
                );
                if bits < best_bits {
                    best_bits = bits;
                    best = pack_code(1, 3, hist_i, 0);
                }
            }
            return best;
        }
        self.tmp_logits.copy_from_slice(&self.work.logits);
        for mix_i in 0..MIX_GRID.len() {
            for tau_i in 0..TAU_GRID.len() {
                softmax_logits(&self.tmp_logits, TAU_GRID[tau_i], &mut self.work.probs);
                mix_global(&mut self.work.probs, &self.global, MIX_GRID[mix_i]);
                for hist_i in 0..HIST_GRID.len() {
                    for copy_i in 0..COPY_GRID.len() {
                        let bits = score_frame_bits(
                            symbols,
                            &self.work.probs,
                            &mut self.cdf,
                            &mut self.row,
                            &mut self.counts,
                            HIST_GRID[hist_i],
                            COPY_GRID[copy_i],
                            Some(&prev_buf),
                        );
                        if bits < best_bits {
                            best_bits = bits;
                            best = pack_code(tau_i, mix_i, hist_i, copy_i);
                        }
                    }
                }
            }
        }
        best
    }

    fn encode_frame_into(
        &mut self,
        symbols: &[i32],
        hist: f32,
        copy: f32,
        prev: Option<&[i32]>,
        enc: &mut RansEncoder,
    ) {
        self.counts.fill(0.0);
        let mut seen = 0.0f32;
        for i in 0..S {
            self.row.copy_from_slice(&self.work.probs[i * V..(i + 1) * V]);
            if let Some(p) = prev {
                apply_copy_on_row(&mut self.row, p[i], copy);
            }
            apply_hist_on_row(&mut self.row, &self.counts, seen, hist);
            normalize_row_inplace(&mut self.row);
            let dest = &mut self.cdf_table[i * CDF_STRIDE..(i + 1) * CDF_STRIDE];
            probs_to_cdf_into(&self.row, dest);
            self.counts[symbols[i] as usize] += 1.0;
            seen += 1.0;
        }
        for i in (0..S).rev() {
            let cdf = &self.cdf_table[i * CDF_STRIDE..(i + 1) * CDF_STRIDE];
            let s = symbols[i] as usize;
            let freq = cdf[s + 1] - cdf[s];
            enc.encode(freq, cdf[s]);
        }
    }

    pub fn encode(&mut self, tokens: &[i32], n_frames: usize, adaptive: bool) -> Vec<u8> {
        self.encode_ex(
            tokens,
            n_frames,
            adaptive,
            true,
            None as Option<fn(&[i32], &mut [f32])>,
        )
    }

    pub fn encode_ex<F>(
        &mut self,
        tokens: &[i32],
        n_frames: usize,
        adaptive: bool,
        neural: bool,
        mut pred: Option<F>,
    ) -> Vec<u8>
    where
        F: FnMut(&[i32], &mut [f32]),
    {
        assert!(n_frames <= MAX_FRAMES);
        assert_eq!(tokens.len(), n_frames * S);
        self.tokens[..n_frames * S].copy_from_slice(tokens);
        let use_nn = neural || adaptive;
        if adaptive {
            for t in 0..n_frames {
                if t == 0 {
                    self.tile_global();
                } else {
                    self.build_ctx(t);
                    if let Some(p) = pred.as_mut() {
                        p(&self.ctx, &mut self.work.logits);
                    } else {
                        forward(&self.weights, &self.ctx, &mut self.work);
                    }
                }
                let sym = &tokens[t * S..(t + 1) * S];
                self.codebook[t] = self.pick_code(t, sym);
            }
        } else {
            self.codebook[..n_frames].fill(GLOBAL_CODE);
        }

        let mut enc = RansEncoder::from_vec(std::mem::take(&mut self.scratch));
        enc.reserve(n_frames * S * 2 + 64);
        for t in (0..n_frames).rev() {
            let (tau_i, mix_i, hist_i, copy_i) = unpack_code(self.codebook[t]);
            if t == 0 || !use_nn {
                self.tile_global();
            } else {
                self.build_ctx(t);
                if let Some(p) = pred.as_mut() {
                    p(&self.ctx, &mut self.work.logits);
                } else {
                    forward(&self.weights, &self.ctx, &mut self.work);
                }
                self.probs_from_logits(TAU_GRID[tau_i], MIX_GRID[mix_i]);
            }
            let mut prev_buf = [0i32; S];
            let prev_ref = if t > 0 {
                prev_buf.copy_from_slice(&self.tokens[(t - 1) * S..t * S]);
                Some(&prev_buf[..])
            } else {
                None
            };
            let sym = &tokens[t * S..(t + 1) * S];
            self.encode_frame_into(sym, HIST_GRID[hist_i], COPY_GRID[copy_i], prev_ref, &mut enc);
        }
        let payload = enc.flush();

        let mut out = Vec::with_capacity(12 + n_frames + payload.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(n_frames as u32).to_le_bytes());
        let mut flags = 0u32;
        if adaptive {
            flags |= FLAG_ADAPTIVE;
        }
        if use_nn {
            flags |= FLAG_NEURAL;
        }
        out.extend_from_slice(&flags.to_le_bytes());
        if adaptive {
            out.extend_from_slice(&self.codebook[..n_frames]);
        }
        out.extend_from_slice(&payload);
        self.scratch.clear();
        if self.scratch.capacity() < BITS_CAP {
            self.scratch.reserve(BITS_CAP);
        }
        out
    }

    pub fn decode(&mut self, data: &[u8], n_frames_hint: Option<usize>) -> Result<usize, String> {
        self.decode_ex(data, n_frames_hint, None as Option<fn(&[i32], &mut [f32])>)
    }

    pub fn decode_ex<F>(
        &mut self,
        data: &[u8],
        n_frames_hint: Option<usize>,
        mut pred: Option<F>,
    ) -> Result<usize, String>
    where
        F: FnMut(&[i32], &mut [f32]),
    {
        crate::alloc::begin_hot();
        let r = self.decode_hot(data, n_frames_hint, &mut pred);
        crate::alloc::end_hot();
        r
    }

    fn decode_hot<F>(
        &mut self,
        data: &[u8],
        n_frames_hint: Option<usize>,
        pred: &mut Option<F>,
    ) -> Result<usize, String>
    where
        F: FnMut(&[i32], &mut [f32]),
    {
        if data.len() < 12 || &data[0..4] != MAGIC {
            return Err("bad bitstream magic".into());
        }
        let n_frames = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        if let Some(h) = n_frames_hint {
            if h != n_frames {
                return Err("n_frames mismatch".into());
            }
        }
        if n_frames > MAX_FRAMES {
            return Err("too many frames".into());
        }
        let flags = u32::from_le_bytes(data[8..12].try_into().unwrap());
        let use_nn = flags & FLAG_NEURAL != 0 || flags & FLAG_ADAPTIVE != 0;
        let mut pos = 12usize;
        if flags & FLAG_ADAPTIVE != 0 {
            if data.len() < pos + n_frames {
                return Err("truncated codebook".into());
            }
            self.codebook[..n_frames].copy_from_slice(&data[pos..pos + n_frames]);
            pos += n_frames;
        } else {
            self.codebook[..n_frames].fill(GLOBAL_CODE);
        }
        let payload = &data[pos..];
        // Decode from a borrowed slice: copy into pre-sized bits (no growth).
        if payload.len() > self.bits.capacity() {
            return Err("bitstream larger than session cap".into());
        }
        self.bits.clear();
        self.bits.extend_from_slice(payload);
        let mut dec = RansDecoder::from_vec(std::mem::take(&mut self.bits));

        for t in 0..n_frames {
            let (tau_i, mix_i, hist_i, copy_i) = unpack_code(self.codebook[t]);
            if t == 0 || !use_nn {
                self.tile_global();
            } else {
                self.build_ctx(t);
                if let Some(p) = pred {
                    p(&self.ctx, &mut self.work.logits);
                } else {
                    forward(&self.weights, &self.ctx, &mut self.work);
                }
                self.probs_from_logits(TAU_GRID[tau_i], MIX_GRID[mix_i]);
            }
            self.counts.fill(0.0);
            let mut seen = 0.0f32;
            let hist = HIST_GRID[hist_i];
            let copy = COPY_GRID[copy_i];
            for s in 0..S {
                self.row.copy_from_slice(&self.work.probs[s * V..(s + 1) * V]);
                if t > 0 {
                    apply_copy_on_row(&mut self.row, self.tokens[(t - 1) * S + s], copy);
                }
                apply_hist_on_row(&mut self.row, &self.counts, seen, hist);
                normalize_row_inplace(&mut self.row);
                probs_to_cdf_into(&self.row, &mut self.cdf);
                let tok = dec.decode(&self.cdf) as i32;
                self.tokens[t * S + s] = tok;
                self.counts[tok as usize] += 1.0;
                seen += 1.0;
            }
        }
        self.bits = dec.into_vec();
        Ok(n_frames)
    }

    pub fn tokens(&self, n_frames: usize) -> &[i32] {
        &self.tokens[..n_frames * S]
    }
}
