//! PyO3 bindings for batched rANS encode/decode (one FFI call per clip).

mod rans;

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rans::{decode_symbols_with_probs, encode_symbols_with_probs_i32, M};

fn encode_clip(symbols: &[i32], probs_flat: &[f32], vocab: usize) -> Vec<u8> {
    encode_symbols_with_probs_i32(symbols, probs_flat, vocab)
}

fn decode_clip(data: &[u8], probs_flat: &[f32], vocab: usize) -> Vec<i32> {
    decode_symbols_with_probs(data, probs_flat, vocab)
        .into_iter()
        .map(|v| v as i32)
        .collect()
}

#[pyfunction]
fn encode(symbols: PyReadonlyArray2<i32>, probs: PyReadonlyArray3<f32>) -> PyResult<Py<PyBytes>> {
    let sym = symbols.as_array();
    let pr = probs.as_array();
    let vocab = pr.shape()[2] as usize;
    let bytes = encode_clip(sym.as_slice().unwrap(), pr.as_slice().unwrap(), vocab);
    Python::with_gil(|py| Ok(PyBytes::new_bound(py, &bytes).unbind()))
}

#[pyfunction]
fn decode<'py>(
    py: Python<'py>,
    data: &[u8],
    probs: PyReadonlyArray3<f32>,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let pr = probs.as_array();
    let t = pr.shape()[0] as usize;
    let s = pr.shape()[1] as usize;
    let vocab = pr.shape()[2] as usize;
    let out = decode_clip(data, pr.as_slice().unwrap(), vocab);
    let arr = Array2::from_shape_vec((t, s), out).expect("shape");
    Ok(arr.into_pyarray_bound(py))
}

/// Stateful decoder for causal frame-by-frame decode.
#[pyclass]
struct ClipDecoder {
    inner: rans::RansDecoder,
    n_frames: usize,
    frames_done: usize,
    vocab: usize,
    cdf: Vec<u32>,
    row: Vec<f32>,
}

#[pymethods]
impl ClipDecoder {
    #[new]
    fn new(data: &[u8], n_frames: usize) -> Self {
        Self {
            inner: rans::RansDecoder::new(data),
            n_frames,
            frames_done: 0,
            vocab: 0,
            cdf: Vec::new(),
            row: Vec::new(),
        }
    }

    fn decode_frame<'py>(
        &mut self,
        py: Python<'py>,
        probs: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        if self.frames_done >= self.n_frames {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "decode beyond n_frames",
            ));
        }
        let pr = probs.as_array();
        let s = pr.shape()[0] as usize;
        let vocab = pr.shape()[1] as usize;
        if self.vocab == 0 {
            self.vocab = vocab;
            self.cdf = vec![0u32; vocab + 1];
            self.row = vec![0f32; vocab];
        }
        let mut out = Vec::with_capacity(s);
        for row in 0..s {
            let r = pr.row(row);
            for i in 0..vocab {
                self.row[i] = r[i];
            }
            rans::normalize_row_inplace(&mut self.row);
            rans::probs_to_cdf_into(&self.row, &mut self.cdf);
            out.push(self.inner.decode(&self.cdf) as i32);
        }
        self.frames_done += 1;
        Ok(Array1::from_vec(out).into_pyarray_bound(py))
    }
}

#[pymodule]
fn _rans(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_class::<ClipDecoder>()?;
    m.add("M", M)?;
    Ok(())
}
