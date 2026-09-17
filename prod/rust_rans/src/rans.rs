//! rANS entropy coder — algorithm only, no Python types.

pub const LOG2_L: u32 = 23;
pub const L: u32 = 1 << LOG2_L;
pub const FREQ_BITS: u32 = 16;
pub const M: u32 = 1 << FREQ_BITS;

pub struct RansEncoder {
    state: u32,
    output: Vec<u8>,
}

impl RansEncoder {
    pub fn new() -> Self {
        Self {
            state: L,
            output: Vec::new(),
        }
    }

    #[inline]
    pub fn encode(&mut self, freq: u32, cum_low: u32) {
        assert!(freq > 0 && freq <= M);
        let upper_bound = freq << (LOG2_L - FREQ_BITS + 8);
        let mut x = self.state;
        while x >= upper_bound {
            self.output.push((x & 0xFF) as u8);
            x >>= 8;
        }
        self.state = (x / freq) * M + cum_low + (x % freq);
    }

    pub fn flush(mut self) -> Vec<u8> {
        let x = self.state;
        self.output.push(((x >> 24) & 0xFF) as u8);
        self.output.push(((x >> 16) & 0xFF) as u8);
        self.output.push(((x >> 8) & 0xFF) as u8);
        self.output.push((x & 0xFF) as u8);
        self.output.reverse();
        self.output
    }
}

pub struct RansDecoder {
    state: u32,
    data: Vec<u8>,
    pos: usize,
}

impl RansDecoder {
    pub fn new(data: &[u8]) -> Self {
        let state = (data[0] as u32)
            | ((data[1] as u32) << 8)
            | ((data[2] as u32) << 16)
            | ((data[3] as u32) << 24);
        Self {
            state,
            data: data.to_vec(),
            pos: 4,
        }
    }

    fn read_byte(&mut self) -> u8 {
        if self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;
            b
        } else {
            0
        }
    }

    #[inline]
    pub fn decode(&mut self, cdf: &[u32]) -> usize {
        let slot = self.state & (M - 1);
        let s = cdf.partition_point(|&c| c <= slot).saturating_sub(1);
        let s = s.min(cdf.len() - 2);
        let freq = cdf[s + 1] - cdf[s];
        self.state = freq * (self.state >> FREQ_BITS) + slot - cdf[s];
        while self.state < L {
            self.state = (self.state << 8) | (self.read_byte() as u32);
        }
        s
    }
}

/// Write CDF (length n+1, last == M) into `cdf`; every freq >= 1.
pub fn probs_to_cdf_into(probs: &[f32], cdf: &mut [u32]) {
    let n = probs.len();
    assert_eq!(cdf.len(), n + 1);
    let mut sum: u32 = 0;
    let mut max_idx = 0usize;
    let mut max_freq = 0u32;
    cdf[0] = 0;
    for i in 0..n {
        let f = ((probs[i] * M as f32).round() as u32).max(1);
        cdf[i + 1] = cdf[i] + f;
        sum += f;
        if f >= max_freq {
            max_freq = f;
            max_idx = i;
        }
    }
    if sum < M {
        cdf[n] += M - sum;
    } else if sum > M {
        let mut excess = sum - M;
        for i in (0..n).rev() {
            if excess == 0 {
                break;
            }
            let start = cdf[i];
            let f = cdf[i + 1] - start;
            if f <= 1 {
                continue;
            }
            let removed = (f - 1).min(excess);
            for j in (i + 1)..=n {
                cdf[j] -= removed;
            }
            excess -= removed;
        }
        if excess > 0 {
            cdf[max_idx + 1] -= excess;
        }
    }
}

pub fn probs_to_cdf(probs: &[f32]) -> Vec<u32> {
    let mut cdf = vec![0u32; probs.len() + 1];
    probs_to_cdf_into(probs, &mut cdf);
    cdf
}

fn build_cdf_table(probs_flat: &[f32], vocab: usize, n: usize) -> Vec<u32> {
    let stride = vocab + 1;
    let mut tables = vec![0u32; n * stride];
    let mut row = vec![0f32; vocab];
    for i in 0..n {
        let off = i * vocab;
        normalize_row(&probs_flat[off..off + vocab], &mut row);
        let base = i * stride;
        probs_to_cdf_into(&row, &mut tables[base..base + stride]);
    }
    tables
}

pub fn encode_symbols_with_probs_i32(
    symbols: &[i32],
    probs_flat: &[f32],
    vocab: usize,
) -> Vec<u8> {
    let n = symbols.len();
    let stride = vocab + 1;
    let tables = build_cdf_table(probs_flat, vocab, n);
    let mut enc = RansEncoder::new();
    for i in (0..n).rev() {
        let base = i * stride;
        let cdf = &tables[base..base + stride];
        let s = symbols[i] as usize;
        let freq = cdf[s + 1] - cdf[s];
        enc.encode(freq, cdf[s]);
    }
    enc.flush()
}

pub fn encode_symbols_with_probs(
    symbols: &[u32],
    probs_flat: &[f32],
    vocab: usize,
) -> Vec<u8> {
    let n = symbols.len();
    let mut enc = RansEncoder::new();
    let mut cdf = vec![0u32; vocab + 1];
    let mut row = vec![0f32; vocab];
    for i in (0..n).rev() {
        let off = i * vocab;
        normalize_row(&probs_flat[off..off + vocab], &mut row);
        probs_to_cdf_into(&row, &mut cdf);
        let s = symbols[i] as usize;
        let freq = cdf[s + 1] - cdf[s];
        enc.encode(freq, cdf[s]);
    }
    enc.flush()
}

pub fn decode_symbols_with_probs(data: &[u8], probs_flat: &[f32], vocab: usize) -> Vec<u32> {
    let n = probs_flat.len() / vocab;
    let stride = vocab + 1;
    let tables = build_cdf_table(probs_flat, vocab, n);
    let mut dec = RansDecoder::new(data);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * stride;
        out.push(dec.decode(&tables[base..base + stride]) as u32);
    }
    out
}

const MIN_PROB: f32 = 1e-6;

#[inline]
fn round7(x: f32) -> f32 {
    (x * 1.0e7).round() / 1.0e7
}

pub fn normalize_row_into(src: &[f32], dst: &mut [f32]) {
    normalize_row(src, dst);
}

pub fn normalize_row_inplace(dst: &mut [f32]) {
    let mut sum = 0.0f32;
    for i in 0..dst.len() {
        let v = round7(dst[i].max(MIN_PROB));
        dst[i] = v;
        sum += v;
    }
    let inv = 1.0 / sum;
    for i in 0..dst.len() {
        dst[i] = round7(dst[i] * inv);
    }
}

#[inline]
fn normalize_row(src: &[f32], dst: &mut [f32]) {
    let mut sum = 0.0f32;
    for i in 0..dst.len() {
        let v = round7(src[i].max(MIN_PROB));
        dst[i] = v;
        sum += v;
    }
    let inv = 1.0 / sum;
    for i in 0..dst.len() {
        dst[i] = round7(dst[i] * inv);
    }
}

pub fn encode_symbols(symbols: &[u32], cdf_tables: &[Vec<u32>]) -> Vec<u8> {
    assert_eq!(symbols.len(), cdf_tables.len());
    let mut enc = RansEncoder::new();
    for i in (0..symbols.len()).rev() {
        let s = symbols[i] as usize;
        let cdf = &cdf_tables[i];
        let freq = cdf[s + 1] - cdf[s];
        enc.encode(freq, cdf[s]);
    }
    enc.flush()
}

pub fn decode_symbols(data: &[u8], cdf_tables: &[Vec<u32>]) -> Vec<u32> {
    let mut dec = RansDecoder::new(data);
    let mut out = Vec::with_capacity(cdf_tables.len());
    for cdf in cdf_tables {
        out.push(dec.decode(cdf) as u32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng, rngs::SmallRng};

    fn uniform_cdf(n: usize) -> Vec<u32> {
        let f = M / n as u32;
        let mut cdf = vec![0u32; n + 1];
        for i in 0..n {
            cdf[i + 1] = cdf[i] + f;
        }
        cdf[n] = M;
        cdf
    }

    #[test]
    fn round_trip_uniform() {
        let mut rng = SmallRng::seed_from_u64(42);
        let cdf = uniform_cdf(256);
        let symbols: Vec<u32> = (0..1000).map(|_| rng.gen_range(0..256) as u32).collect();
        let freqs: Vec<u32> = (0..256).map(|i| cdf[i + 1] - cdf[i]).collect();
        let tables: Vec<Vec<u32>> = symbols
            .iter()
            .map(|&s| {
                let mut c = vec![0u32; 257];
                for i in 0..256 {
                    c[i + 1] = c[i] + freqs[i];
                }
                c[256] = M;
                let _ = s;
                c
            })
            .collect();
        let bytes = encode_symbols(&symbols, &tables);
        let got = decode_symbols(&bytes, &tables);
        assert_eq!(got, symbols);
    }

    #[test]
    fn probs_to_cdf_sums_to_m() {
        let probs: Vec<f32> = (0..1024).map(|i| (i + 1) as f32).collect();
        let s: f32 = probs.iter().sum();
        let probs_norm: Vec<f32> = probs.iter().map(|&p| p / s).collect();
        let cdf = probs_to_cdf(&probs_norm);
        assert_eq!(*cdf.last().unwrap(), M);
        for w in 1..1024 {
            assert!(cdf[w] - cdf[w - 1] >= 1);
        }
    }

    proptest! {
        #[test]
        fn prop_roundtrip_random(
            len in 1usize..20_000,
            vocab in 2usize..1025,
            seed in 0u64..u64::MAX,
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let probs: Vec<f32> = (0..vocab).map(|_| rng.gen_range(0.001f32..1.0)).collect();
            let sum: f32 = probs.iter().sum();
            let probs_norm: Vec<f32> = probs.iter().map(|&p| p / sum).collect();
            let cdf = probs_to_cdf(&probs_norm);
            assert_eq!(*cdf.last().unwrap(), M);

            let symbols: Vec<u32> = (0..len)
                .map(|_| rng.gen_range(0..vocab) as u32)
                .collect();
            let tables: Vec<Vec<u32>> = (0..len).map(|_| cdf.clone()).collect();
            let bytes = encode_symbols(&symbols, &tables);
            let got = decode_symbols(&bytes, &tables);
            prop_assert_eq!(got, symbols);
        }
    }
}
