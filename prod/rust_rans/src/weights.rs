//! Packed f32 weight blob (magic RTNVW001).

use std::fs;
use std::io::{self, Write};
use std::path::Path;

pub const MAGIC: &[u8; 8] = b"RTNVW001";
pub const D: usize = 256;
pub const V: usize = 1024;
pub const T: usize = 8;
pub const S: usize = 128;
pub const L: usize = T * S;
pub const H: usize = 4;
pub const DH: usize = D / H;
pub const FFN: usize = 768;
pub const N_LAYERS: usize = 6;
pub const ROWS: usize = 8;
pub const COLS: usize = 16;

fn layer_keys(i: usize) -> Vec<(String, usize)> {
    let p = format!("transformer.layers.{i}");
    vec![
        (format!("{p}.self_attn.in_proj_weight"), 3 * D * D),
        (format!("{p}.self_attn.in_proj_bias"), 3 * D),
        (format!("{p}.self_attn.out_proj.weight"), D * D),
        (format!("{p}.self_attn.out_proj.bias"), D),
        (format!("{p}.linear1.weight"), FFN * D),
        (format!("{p}.linear1.bias"), FFN),
        (format!("{p}.linear2.weight"), D * FFN),
        (format!("{p}.linear2.bias"), D),
        (format!("{p}.norm1.weight"), D),
        (format!("{p}.norm1.bias"), D),
        (format!("{p}.norm2.weight"), D),
        (format!("{p}.norm2.bias"), D),
    ]
}

pub fn tensor_specs() -> Vec<(String, usize)> {
    let mut v = vec![
        ("token_embed.weight".into(), V * D),
        ("row_embed.weight".into(), ROWS * D),
        ("col_embed.weight".into(), COLS * D),
        ("temporal_embed.weight".into(), T * D),
    ];
    for i in 0..N_LAYERS {
        v.extend(layer_keys(i));
    }
    v.push(("transformer.norm.weight".into(), D));
    v.push(("transformer.norm.bias".into(), D));
    v.push(("output_head.weight".into(), V * D));
    v
}

pub fn blob_floats() -> usize {
    tensor_specs().iter().map(|(_, n)| n).sum()
}

#[derive(Clone)]
pub struct Weights {
    data: Vec<f32>,
    off: Vec<usize>,
}

impl Weights {
    pub fn from_blob(data: Vec<f32>) -> io::Result<Self> {
        let specs = tensor_specs();
        let need = specs.iter().map(|(_, n)| n).sum::<usize>();
        if data.len() != need {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("weight blob {} floats, expected {need}", data.len()),
            ));
        }
        let mut off = Vec::with_capacity(specs.len() + 1);
        let mut o = 0usize;
        off.push(0);
        for (_, n) in &specs {
            o += *n;
            off.push(o);
        }
        Ok(Self { data, off })
    }

    pub fn load(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        if bytes.len() < 12 || &bytes[0..8] != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad weight magic"));
        }
        let n = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let need = blob_floats();
        if n != need {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("nfloats {n} != {need}"),
            ));
        }
        let body = &bytes[12..];
        if body.len() != n * 4 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "truncated weights"));
        }
        let mut data = vec![0f32; n];
        for i in 0..n {
            data[i] = f32::from_le_bytes(body[i * 4..i * 4 + 4].try_into().unwrap());
        }
        Self::from_blob(data)
    }

    #[inline]
    pub fn t(&self, idx: usize) -> &[f32] {
        &self.data[self.off[idx]..self.off[idx + 1]]
    }

    pub fn token_embed(&self) -> &[f32] {
        self.t(0)
    }
    pub fn row_embed(&self) -> &[f32] {
        self.t(1)
    }
    pub fn col_embed(&self) -> &[f32] {
        self.t(2)
    }
    pub fn temporal_embed(&self) -> &[f32] {
        self.t(3)
    }

    fn layer_base(i: usize) -> usize {
        4 + i * 12
    }

    pub fn in_proj_w(&self, i: usize) -> &[f32] {
        self.t(Self::layer_base(i))
    }
    pub fn in_proj_b(&self, i: usize) -> &[f32] {
        self.t(Self::layer_base(i) + 1)
    }
    pub fn out_proj_w(&self, i: usize) -> &[f32] {
        self.t(Self::layer_base(i) + 2)
    }
    pub fn out_proj_b(&self, i: usize) -> &[f32] {
        self.t(Self::layer_base(i) + 3)
    }
    pub fn linear1_w(&self, i: usize) -> &[f32] {
        self.t(Self::layer_base(i) + 4)
    }
    pub fn linear1_b(&self, i: usize) -> &[f32] {
        self.t(Self::layer_base(i) + 5)
    }
    pub fn linear2_w(&self, i: usize) -> &[f32] {
        self.t(Self::layer_base(i) + 6)
    }
    pub fn linear2_b(&self, i: usize) -> &[f32] {
        self.t(Self::layer_base(i) + 7)
    }
    pub fn norm1_w(&self, i: usize) -> &[f32] {
        self.t(Self::layer_base(i) + 8)
    }
    pub fn norm1_b(&self, i: usize) -> &[f32] {
        self.t(Self::layer_base(i) + 9)
    }
    pub fn norm2_w(&self, i: usize) -> &[f32] {
        self.t(Self::layer_base(i) + 10)
    }
    pub fn norm2_b(&self, i: usize) -> &[f32] {
        self.t(Self::layer_base(i) + 11)
    }
    pub fn final_norm_w(&self) -> &[f32] {
        self.t(4 + N_LAYERS * 12)
    }
    pub fn final_norm_b(&self) -> &[f32] {
        self.t(4 + N_LAYERS * 12 + 1)
    }
    pub fn output_w(&self) -> &[f32] {
        self.t(4 + N_LAYERS * 12 + 2)
    }
}

pub fn write_blob<W: Write>(mut w: W, data: &[f32]) -> io::Result<()> {
    w.write_all(MAGIC)?;
    w.write_all(&(data.len() as u32).to_le_bytes())?;
    for x in data {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}
