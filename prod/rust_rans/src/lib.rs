//! PyO3 bindings: batched rANS + native neural session (one FFI per clip).

mod alloc;
mod c_api;
mod gemm;
mod model;
mod rans;
mod rate;
mod session;
mod weights;

#[global_allocator]
static ALLOC: alloc::CountingAlloc = alloc::CountingAlloc;

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rans::{decode_symbols_with_probs, encode_symbols_with_probs_i32, M};
use session::Engine as NativeEngine;
use std::path::Path;

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

/// Stateful decoder for causal frame-by-frame decode (legacy Python loop).
#[pyclass]
struct ClipDecoder {
    inner: rans::RansDecoder,
    n_frames: usize,
    frames_done: usize,
    vocab: usize,
    cdf: Vec<u32>,
    row: Vec<f32>,
    out: Vec<i32>,
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
            out: Vec::new(),
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
            self.out = vec![0i32; s];
        }
        if self.out.len() < s {
            self.out.resize(s, 0);
        }
        for row in 0..s {
            let r = pr.row(row);
            for i in 0..vocab {
                self.row[i] = r[i];
            }
            rans::normalize_row_inplace(&mut self.row);
            rans::probs_to_cdf_into(&self.row, &mut self.cdf);
            self.out[row] = self.inner.decode(&self.cdf) as i32;
        }
        self.frames_done += 1;
        Ok(Array1::from_vec(self.out.clone()).into_pyarray_bound(py))
    }
}

/// Native neural + rANS session. One encode/decode FFI per clip.
#[pyclass]
struct Engine {
    inner: NativeEngine,
    predictor: Option<Py<PyAny>>,
}

#[pymethods]
impl Engine {
    #[new]
    fn new(weight_path: &str, global_freq: PyReadonlyArray1<f32>) -> PyResult<Self> {
        let g = global_freq.as_slice()?;
        let inner = NativeEngine::load(Path::new(weight_path), g)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner,
            predictor: None,
        })
    }

    fn set_predictor(&mut self, predictor: Py<PyAny>) {
        self.predictor = Some(predictor);
    }

    #[pyo3(signature = (tokens, adaptive, cpu=false, neural=true))]
    fn encode<'py>(
        &mut self,
        py: Python<'py>,
        tokens: PyReadonlyArray2<i32>,
        adaptive: bool,
        cpu: bool,
        neural: bool,
    ) -> PyResult<Py<PyBytes>> {
        let t = tokens.as_array();
        let n_frames = t.shape()[0];
        let s = t.shape()[1];
        if s != 128 {
            return Err(PyValueError::new_err("S must be 128"));
        }
        let slice = t.as_slice().ok_or_else(|| PyValueError::new_err("tokens not C-contiguous"))?;
        let blob = if cpu || self.predictor.is_none() {
            py.allow_threads(|| {
                self.inner
                    .encode_ex(slice, n_frames, adaptive, neural, None as Option<fn(&[i32], &mut [f32])>)
            })
        } else {
            let pred_obj = self.predictor.as_ref().unwrap().clone_ref(py);
            let err = std::cell::Cell::new(None::<String>);
            let blob = self.inner.encode_ex(
                slice,
                n_frames,
                adaptive,
                neural,
                Some(|ctx: &[i32], logits: &mut [f32]| {
                    invoke_predictor(&pred_obj, ctx, logits, &err);
                }),
            );
            if let Some(e) = err.into_inner() {
                return Err(PyRuntimeError::new_err(e));
            }
            blob
        };
        Ok(PyBytes::new_bound(py, &blob).unbind())
    }

    #[pyo3(signature = (data, cpu=false))]
    fn decode<'py>(
        &mut self,
        py: Python<'py>,
        data: &[u8],
        cpu: bool,
    ) -> PyResult<Bound<'py, PyArray2<i32>>> {
        let n = if cpu || self.predictor.is_none() {
            py.allow_threads(|| self.inner.decode(data, None))
                .map_err(|e| PyRuntimeError::new_err(e))?
        } else {
            let pred_obj = self.predictor.as_ref().unwrap().clone_ref(py);
            let err = std::cell::Cell::new(None::<String>);
            let n = self
                .inner
                .decode_ex(
                    data,
                    None,
                    Some(|ctx: &[i32], logits: &mut [f32]| {
                        invoke_predictor(&pred_obj, ctx, logits, &err);
                    }),
                )
                .map_err(|e| PyRuntimeError::new_err(e))?;
            if let Some(e) = err.into_inner() {
                return Err(PyRuntimeError::new_err(e));
            }
            n
        };
        let tok = self.inner.tokens(n);
        let arr = Array2::from_shape_vec((n, 128), tok.to_vec()).expect("shape");
        Ok(arr.into_pyarray_bound(py))
    }
}

fn invoke_predictor(
    pred_obj: &Py<PyAny>,
    ctx: &[i32],
    logits: &mut [f32],
    err: &std::cell::Cell<Option<String>>,
) {
    Python::with_gil(|py| {
        let np = numpy::PyArray1::from_slice_bound(py, ctx);
        match pred_obj.bind(py).call1((np,)) {
            Ok(out) => match out.extract::<PyReadonlyArray2<f32>>() {
                Ok(view) => match view.as_slice() {
                    Ok(sl) if sl.len() == logits.len() => logits.copy_from_slice(sl),
                    _ => err.set(Some("predictor logits shape".into())),
                },
                Err(e) => err.set(Some(e.to_string())),
            },
            Err(e) => err.set(Some(e.to_string())),
        }
    });
}

#[pyfunction]
fn hot_allocs() -> usize {
    alloc::last()
}

#[pymodule]
fn _rans(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(hot_allocs, m)?)?;
    m.add_class::<ClipDecoder>()?;
    m.add_class::<Engine>()?;
    m.add("M", M)?;
    Ok(())
}
