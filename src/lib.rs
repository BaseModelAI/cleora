use std::borrow::Cow;
use std::ffi::CString;

use ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayDescrMethods, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray2, PyReadwriteArray2, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::{
    PyFileNotFoundError, PyIOError, PyRuntimeError, PyRuntimeWarning, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyBytes, PyTuple};

use crate::configuration::{resolve_num_workers, Configuration};
use crate::embedding::{thread_pool, MarkovType};
use crate::pipeline::{
    build_graph_from_files, build_graph_from_int_edges, build_graph_from_lines,
    build_graph_from_rows, build_graph_from_str_edges,
};
use crate::sparse_matrix::{Normalization, SparseMatrix};
use crate::sparse_matrix_builder::BuildStats;

pub mod configuration;
pub mod embedding;
pub mod entity;
pub mod pipeline;
pub mod sparse_matrix;
pub mod sparse_matrix_builder;

/// Turns the outcome of a build into a matrix, a warning (some lines unusable) or
/// an error (nothing usable).
fn finish_build(
    py: Python<'_>,
    result: Result<(SparseMatrix, BuildStats), String>,
) -> PyResult<SparseMatrix> {
    let (matrix, stats) = result.map_err(|message| {
        if message.starts_with("Cannot open file") || message.starts_with("Cannot read file") {
            PyIOError::new_err(message)
        } else {
            PyValueError::new_err(message)
        }
    })?;
    if stats.lines_skipped > 0 {
        let detail = match &stats.first_bad_line {
            Some(bad) => {
                let file = bad.pos >> 52;
                let line = ((bad.pos >> 16) & 0xF_FFFF_FFFF) + 1;
                let place = if file > 0 {
                    format!("file #{}, line {}", file + 1, line)
                } else {
                    format!("line {}", line)
                };
                format!(
                    " First problem at {}: {} -> {:?}",
                    place,
                    bad.error.describe(),
                    bad.text
                )
            }
            None => String::new(),
        };
        if stats.lines_used == 0 {
            return Err(PyValueError::new_err(format!(
                "None of the {} input line(s) could be used.{}",
                stats.lines_total, detail
            )));
        }
        let message = format!(
            "{} of {} input line(s) were skipped because they could not be parsed.{}",
            stats.lines_skipped, stats.lines_total, detail
        );
        let message = CString::new(message.replace('\0', " "))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        PyErr::warn(py, py.get_type::<PyRuntimeWarning>().as_any(), &message, 1)?;
    }
    Ok(matrix)
}

fn extract_weights(weights: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<f32>>> {
    let Some(weights) = weights else {
        return Ok(None);
    };
    if weights.is_none() {
        return Ok(None);
    }
    if let Ok(array) = weights.extract::<PyReadonlyArray1<f32>>() {
        return Ok(Some(array.as_array().iter().copied().collect()));
    }
    if let Ok(array) = weights.extract::<PyReadonlyArray1<f64>>() {
        return Ok(Some(array.as_array().iter().map(|w| *w as f32).collect()));
    }
    let values: Vec<f64> = weights.extract().map_err(|_| {
        PyTypeError::new_err("weights must be a sequence or a 1-dimensional numpy array of numbers")
    })?;
    Ok(Some(values.into_iter().map(|w| w as f32).collect()))
}

fn extract_strings(items: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<PyBackedStr>> {
    let mut strings: Vec<PyBackedStr> = match items.len() {
        Ok(len) => Vec::with_capacity(len),
        Err(_) => Vec::new(),
    };
    for item in items.try_iter()? {
        let item = item?;
        let string = item
            .extract::<PyBackedStr>()
            .map_err(|_| PyValueError::new_err(format!("{} must be strings", what)))?;
        strings.push(string);
    }
    Ok(strings)
}

/// A float32 matrix borrowed from numpy, as a row-major slice. Arrays that are not
/// C-contiguous are copied.
fn matrix_data<'a>(array: &'a PyReadonlyArray2<'_, f32>) -> Cow<'a, [f32]> {
    // `as_slice` also succeeds for Fortran-ordered arrays (and returns the columns
    // one after another), so the layout has to be checked explicitly.
    if array.is_c_contiguous() {
        if let Ok(slice) = array.as_slice() {
            return Cow::Borrowed(slice);
        }
    }
    Cow::Owned(array.as_array().iter().copied().collect())
}

/// The row-major data of an array that is written in place.
fn writable_matrix_data<'a>(
    array: &'a mut PyReadwriteArray2<'_, f32>,
    name: &str,
) -> PyResult<&'a mut [f32]> {
    let message = format!("{} must be C-contiguous", name);
    if !array.is_c_contiguous() {
        return Err(PyValueError::new_err(message));
    }
    array
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err(message))
}

/// Casts a Python object to a 2-dimensional float32 numpy array, with a readable error.
fn as_f32_matrix<'a, 'py>(
    object: &'a Bound<'py, PyAny>,
    name: &str,
) -> PyResult<&'a Bound<'py, PyArray2<f32>>> {
    object.cast::<PyArray2<f32>>().map_err(|_| {
        let found = match object.cast::<PyUntypedArray>() {
            Ok(array) => format!(
                "a {}-dimensional array of dtype {}",
                array.ndim(),
                array.dtype()
            ),
            Err(_) => "an object that is not a numpy array".to_string(),
        };
        PyTypeError::new_err(format!(
            "{} must be a 2-dimensional numpy array of dtype float32, got {} \
             (convert with numpy.asarray(x, dtype=numpy.float32))",
            name, found
        ))
    })
}

fn vec_to_pyarray(
    py: Python<'_>,
    data: Vec<f32>,
    rows: usize,
    columns: usize,
) -> PyResult<Bound<'_, PyArray2<f32>>> {
    let array = Array2::from_shape_vec((rows, columns), data)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(array.into_pyarray(py))
}

fn parse_propagation(propagation: &str) -> PyResult<MarkovType> {
    match propagation {
        "left" => Ok(MarkovType::Left),
        "symmetric" => Ok(MarkovType::Symmetric),
        _ => Err(PyValueError::new_err(format!(
            "Unknown propagation '{}'. Use 'left' or 'symmetric'.",
            propagation
        ))),
    }
}

/// Row indices given from Python (a numpy integer array or a sequence of ints).
fn extract_rows(rows: &Bound<'_, PyAny>, num_entities: usize) -> PyResult<Vec<usize>> {
    let raw: Vec<i64> = match rows.extract::<PyReadonlyArray1<i64>>() {
        Ok(array) => array.as_array().iter().copied().collect(),
        Err(_) => rows.extract::<Vec<i64>>().map_err(|_| {
            PyTypeError::new_err("rows must be a 1-dimensional sequence of integers")
        })?,
    };
    raw.into_iter()
        .map(|row| {
            if row < 0 || row as u64 >= num_entities as u64 {
                Err(PyValueError::new_err(format!(
                    "row index {} is out of range for a graph with {} entities",
                    row, num_entities
                )))
            } else {
                Ok(row as usize)
            }
        })
        .collect()
}

impl SparseMatrix {
    fn markov_propagate<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        markov_type: MarkovType,
        num_workers: Option<usize>,
        out: Option<&Bound<'py, PyAny>>,
        rows: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x = as_f32_matrix(x, "x")?;
        let (input_rows, dim) = (x.shape()[0], x.shape()[1]);
        if input_rows != self.num_entities() {
            return Err(PyValueError::new_err(format!(
                "Embedding matrix has {} rows but graph has {} entities",
                input_rows,
                self.num_entities()
            )));
        }
        let selected = match rows {
            Some(rows) if !rows.is_none() => Some(extract_rows(rows, self.num_entities())?),
            _ => None,
        };
        let output_rows = selected.as_ref().map_or(input_rows, |rows| rows.len());
        let input = x
            .try_readonly()
            .map_err(|e| PyRuntimeError::new_err(format!("x cannot be read: {}", e)))?;
        let input_data = matrix_data(&input);
        let pool = thread_pool(resolve_num_workers(num_workers));
        let run = |target: &mut [f32]| match &selected {
            Some(rows) => {
                embedding::propagate_selected(self, &input_data, rows, target, dim, markov_type)
            }
            None => embedding::propagate(self, &input_data, target, dim, markov_type),
        };

        match out {
            Some(out) if !out.is_none() => {
                let out = as_f32_matrix(out, "out")?;
                if out.shape() != [output_rows, dim] {
                    return Err(PyValueError::new_err(format!(
                        "out has shape {:?} but the result has shape {:?}",
                        out.shape(),
                        [output_rows, dim]
                    )));
                }
                let mut output = out.try_readwrite().map_err(|_| {
                    PyValueError::new_err(
                        "out must be a writeable array that does not share memory with x",
                    )
                })?;
                let output_data = writable_matrix_data(&mut output, "out")?;
                py.detach(|| pool.install(|| run(output_data)));
                Ok(out.clone())
            }
            _ => {
                let mut output = vec![0f32; output_rows * dim];
                py.detach(|| pool.install(|| run(&mut output)));
                vec_to_pyarray(py, output, output_rows, dim)
            }
        }
    }

    /// Builds a matrix from an iterator of lines (Rust API).
    pub fn from_rust_iterator<'a>(
        columns: &str,
        hyperedge_trim_n: usize,
        hyperedges: impl Iterator<Item = &'a str>,
        num_workers: Option<usize>,
    ) -> Result<SparseMatrix, String> {
        let config = Configuration::new(columns, hyperedge_trim_n, num_workers)?;
        let lines: Vec<&str> = hyperedges.collect();
        build_graph_from_lines(&config, &lines, None).map(|(matrix, _)| matrix)
    }

    fn propagate_normalized_inplace(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        markov_type: MarkovType,
        num_iterations: usize,
        num_workers: Option<usize>,
    ) -> PyResult<()> {
        let x = as_f32_matrix(x, "x")?;
        let (rows, dim) = (x.shape()[0], x.shape()[1]);
        if rows != self.num_entities() {
            return Err(PyValueError::new_err(format!(
                "Embedding matrix has {} rows but graph has {} entities",
                rows,
                self.num_entities()
            )));
        }
        let mut target = x
            .try_readwrite()
            .map_err(|_| PyValueError::new_err("x must be a writeable array"))?;
        let data = writable_matrix_data(&mut target, "x")?;
        let pool = thread_pool(resolve_num_workers(num_workers));
        py.detach(|| {
            pool.install(|| {
                let mut scratch = vec![0f32; data.len()];
                embedding::iterate(
                    self,
                    data,
                    &mut scratch,
                    dim,
                    markov_type,
                    num_iterations,
                    0.0,
                    0.0,
                );
            })
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn embed_classic<'py>(
        &self,
        py: Python<'py>,
        feature_dim: usize,
        num_iterations: usize,
        propagation: &str,
        seed: i64,
        residual_weight: f32,
        convergence_threshold: f32,
        num_workers: Option<usize>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, usize)> {
        let markov_type = parse_propagation(propagation)?;
        let pool = thread_pool(resolve_num_workers(num_workers));
        let rows = self.num_entities();
        let (embeddings, iterations) = py.detach(|| {
            pool.install(|| {
                let initial = embedding::initialize(&self.entity_ids, feature_dim, seed);
                embedding::embed(
                    self,
                    initial,
                    feature_dim,
                    markov_type,
                    num_iterations,
                    residual_weight,
                    convergence_threshold,
                )
            })
        });
        Ok((
            vec_to_pyarray(py, embeddings, rows, feature_dim)?,
            iterations,
        ))
    }
}

#[pymethods]
impl SparseMatrix {
    /// One propagation step with the row-stochastic (random walk) matrix.
    ///
    /// `x` is a float32 array with one row per entity. The result is written to
    /// `out` when given (it must not share memory with `x`), otherwise a new array
    /// is returned. With `rows` (entity indices) only those rows of the result are
    /// computed, in the given order; the work is proportional to their entries.
    #[pyo3(signature = (x, num_workers = None, out = None, rows = None))]
    pub fn left_markov_propagate<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        num_workers: Option<usize>,
        out: Option<&Bound<'py, PyAny>>,
        rows: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.markov_propagate(py, x, MarkovType::Left, num_workers, out, rows)
    }

    /// One propagation step with the symmetrically normalised matrix.
    /// Arguments as for `left_markov_propagate`.
    #[pyo3(signature = (x, num_workers = None, out = None, rows = None))]
    fn symmetric_markov_propagate<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        num_workers: Option<usize>,
        out: Option<&Bound<'py, PyAny>>,
        rows: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.markov_propagate(py, x, MarkovType::Symmetric, num_workers, out, rows)
    }

    /// `num_iterations` rounds of random-walk propagation, each followed by scaling
    /// every row to unit L2 norm, in place: `x` (float32, C-contiguous, one row per
    /// entity) is overwritten with the result. Needs one more matrix of working
    /// memory, whatever the number of rounds.
    #[pyo3(signature = (x, num_iterations, num_workers = None))]
    fn left_markov_propagate_normalized_inplace(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        num_iterations: usize,
        num_workers: Option<usize>,
    ) -> PyResult<()> {
        self.propagate_normalized_inplace(py, x, MarkovType::Left, num_iterations, num_workers)
    }

    /// Like `left_markov_propagate_normalized_inplace`, with the symmetrically
    /// normalised matrix.
    #[pyo3(signature = (x, num_iterations, num_workers = None))]
    fn symmetric_markov_propagate_normalized_inplace(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        num_iterations: usize,
        num_workers: Option<usize>,
    ) -> PyResult<()> {
        self.propagate_normalized_inplace(py, x, MarkovType::Symmetric, num_iterations, num_workers)
    }

    /// Builds the matrix from an iterable of cliques: every element is a sequence of
    /// entity names, and all entities of a clique are connected with each other.
    ///
    /// Names are taken as they are, so they may contain any character (spaces, tabs,
    /// commas). `weights` optionally gives one non-negative weight per clique.
    #[staticmethod]
    #[pyo3(signature = (cliques, hyperedge_trim_n = 16, num_workers = None, weights = None))]
    fn from_cliques(
        py: Python<'_>,
        cliques: &Bound<'_, PyAny>,
        hyperedge_trim_n: usize,
        num_workers: Option<usize>,
        weights: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<SparseMatrix> {
        let config = Configuration::new(
            "complex::reflexive::CliqueNode",
            hyperedge_trim_n,
            num_workers,
        )
        .map_err(PyValueError::new_err)?;
        let mut rows: Vec<Vec<Vec<PyBackedStr>>> = match cliques.len() {
            Ok(len) => Vec::with_capacity(len),
            Err(_) => Vec::new(),
        };
        for clique in cliques.try_iter()? {
            let clique = clique?;
            if clique.is_instance_of::<pyo3::types::PyString>() {
                return Err(PyValueError::new_err(
                    "every clique must be a sequence of entity names, not a single string",
                ));
            }
            let names = extract_strings(&clique, "Clique elements")?;
            if names.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "clique {} is empty",
                    rows.len()
                )));
            }
            rows.push(vec![names]);
        }
        let weights = extract_weights(weights)?;
        let result = py.detach(|| build_graph_from_rows(&config, &rows, weights.as_deref()));
        finish_build(py, result)
    }

    /// Indices of the entities that have no neighbour other than themselves.
    #[getter(isolated_entities)]
    fn py_isolated_entities(&self) -> Vec<usize> {
        self.isolated_entities()
    }

    /// Builds the matrix from an iterable of lines, one hyperedge per line.
    ///
    /// `weights` optionally gives one non-negative weight per line.
    #[staticmethod]
    #[pyo3(signature = (hyperedges, columns, hyperedge_trim_n = 16, num_workers = None, weights = None))]
    fn from_iterator(
        py: Python<'_>,
        hyperedges: &Bound<'_, PyAny>,
        columns: &str,
        hyperedge_trim_n: usize,
        num_workers: Option<usize>,
        weights: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<SparseMatrix> {
        let config = Configuration::new(columns, hyperedge_trim_n, num_workers)
            .map_err(PyValueError::new_err)?;
        let lines = extract_strings(hyperedges, "Iterator elements")?;
        let weights = extract_weights(weights)?;
        let result = py.detach(|| build_graph_from_lines(&config, &lines, weights.as_deref()));
        finish_build(py, result)
    }

    /// Builds the matrix from text files with one hyperedge per line.
    #[staticmethod]
    #[pyo3(signature = (filepaths, columns, hyperedge_trim_n = 16, num_workers = None))]
    fn from_files(
        py: Python<'_>,
        filepaths: Vec<String>,
        columns: &str,
        hyperedge_trim_n: usize,
        num_workers: Option<usize>,
    ) -> PyResult<SparseMatrix> {
        if filepaths.is_empty() {
            return Err(PyValueError::new_err("At least one file path is required"));
        }
        for filepath in &filepaths {
            let lower = filepath.to_ascii_lowercase();
            if [".gz", ".zst", ".bz2", ".xz", ".zip", ".parquet"]
                .iter()
                .any(|extension| lower.ends_with(extension))
            {
                return Err(PyValueError::new_err(format!(
                    "Unsupported file format: {}. Files must be plain text (tab-, comma- or \
                     space-separated); decompress or convert the file first.",
                    filepath
                )));
            }
            if let Err(error) = std::fs::File::open(filepath) {
                let message = format!("Cannot open file '{}': {}", filepath, error);
                return Err(if error.kind() == std::io::ErrorKind::NotFound {
                    PyFileNotFoundError::new_err(message)
                } else {
                    PyIOError::new_err(message)
                });
            }
        }
        let config = Configuration::new(columns, hyperedge_trim_n, num_workers)
            .map_err(PyValueError::new_err)?;
        let result = py.detach(|| build_graph_from_files(&config, &filepaths));
        finish_build(py, result)
    }

    /// Builds the matrix from two parallel arrays of node identifiers: `src[i]` and
    /// `dst[i]` are the two ends of edge `i`.
    ///
    /// The arrays are either int64 numpy arrays or sequences of str. With the
    /// default columns definition (one reflexive column) the result equals
    /// `from_iterator(f"{s} {d}" ...)`; with two columns it equals
    /// `from_iterator(f"{s}\t{d}" ...)`. No strings are built for integer input.
    #[staticmethod]
    #[pyo3(signature = (src, dst, columns = "complex::reflexive::node", weights = None, num_workers = None))]
    fn from_edges(
        py: Python<'_>,
        src: &Bound<'_, PyAny>,
        dst: &Bound<'_, PyAny>,
        columns: &str,
        weights: Option<&Bound<'_, PyAny>>,
        num_workers: Option<usize>,
    ) -> PyResult<SparseMatrix> {
        let config = Configuration::new(columns, 16, num_workers).map_err(PyValueError::new_err)?;
        let weights = extract_weights(weights)?;

        if let (Ok(sources), Ok(targets)) = (
            src.extract::<PyReadonlyArray1<i64>>(),
            dst.extract::<PyReadonlyArray1<i64>>(),
        ) {
            let sources: Cow<[i64]> = match sources.as_slice() {
                Ok(slice) => Cow::Borrowed(slice),
                Err(_) => Cow::Owned(sources.as_array().iter().copied().collect()),
            };
            let targets: Cow<[i64]> = match targets.as_slice() {
                Ok(slice) => Cow::Borrowed(slice),
                Err(_) => Cow::Owned(targets.as_array().iter().copied().collect()),
            };
            let result = py.detach(|| {
                build_graph_from_int_edges(&config, &sources, &targets, weights.as_deref())
            });
            return finish_build(py, result);
        }

        let type_error = || {
            PyTypeError::new_err(
                "src and dst must both be int64 numpy arrays or both be sequences of str",
            )
        };
        if src.cast::<PyUntypedArray>().is_ok() || dst.cast::<PyUntypedArray>().is_ok() {
            // numpy arrays of another dtype: convert explicitly rather than guess
            let is_str_array = |object: &Bound<'_, PyAny>| -> bool {
                object
                    .cast::<PyUntypedArray>()
                    .map(|array| {
                        let kind = array.dtype().kind();
                        kind == b'U' || kind == b'O' || kind == b'T'
                    })
                    .unwrap_or(true)
            };
            if !(is_str_array(src) && is_str_array(dst)) {
                return Err(type_error());
            }
        }
        let sources = extract_strings(src, "src elements").map_err(|_| type_error())?;
        let targets = extract_strings(dst, "dst elements").map_err(|_| type_error())?;
        let result = py
            .detach(|| build_graph_from_str_edges(&config, &sources, &targets, weights.as_deref()));
        finish_build(py, result)
    }

    /// Boolean mask of the entities that belong to the given column.
    fn get_entity_column_mask<'py>(
        &self,
        py: Python<'py>,
        column_name: &str,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let column = self
            .column_names
            .iter()
            .position(|name| name == column_name)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Column name '{}' not found. Available: {}",
                    column_name,
                    self.column_names
                        .iter()
                        .map(|name| format!("'{}'", name))
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })? as u8;
        let mask: Vec<bool> = self.column_ids.iter().map(|id| *id == column).collect();
        Ok(PyArray1::from_vec(py, mask))
    }

    /// Weighted degree of every entity (total hyperedge mass), as float32.
    #[getter]
    fn entity_degrees<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let degrees: Vec<f32> = self.row_sums.iter().map(|sum| *sum as f32).collect();
        PyArray1::from_vec(py, degrees)
    }

    #[getter(num_entities)]
    fn py_num_entities(&self) -> usize {
        self.num_entities()
    }

    /// Number of stored (directed) entries of the matrix.
    #[getter]
    fn num_edges(&self) -> usize {
        self.nnz()
    }

    /// Names of the entity columns the matrix was built from.
    #[getter(column_names)]
    fn py_column_names(&self) -> Vec<String> {
        self.column_names.clone()
    }

    /// Entity names, in matrix row order (an immutable tuple, built once).
    #[getter(entity_ids)]
    fn py_entity_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let ids = self
            .py_entity_ids
            .get_or_try_init(py, || PyTuple::new(py, &self.entity_ids).map(Bound::unbind))?;
        Ok(ids.bind(py).clone())
    }

    #[setter(entity_ids)]
    fn py_set_entity_ids(&mut self, entity_ids: Vec<String>) -> PyResult<()> {
        if entity_ids.len() != self.num_entities() {
            return Err(PyValueError::new_err(format!(
                "entity_ids must have {} elements, got {}",
                self.num_entities(),
                entity_ids.len()
            )));
        }
        self.replace_entity_ids(entity_ids);
        Ok(())
    }

    fn get_entity_index(&self, entity_id: &str) -> PyResult<usize> {
        self.index_of(entity_id)
            .ok_or_else(|| PyValueError::new_err(format!("Entity '{}' not found", entity_id)))
    }

    fn get_entity_indices(&self, entity_ids: Vec<PyBackedStr>) -> PyResult<Vec<usize>> {
        entity_ids
            .iter()
            .map(|id| {
                self.index_of(id)
                    .ok_or_else(|| PyValueError::new_err(format!("Entity '{}' not found", &**id)))
            })
            .collect()
    }

    /// Deterministic initial embeddings: a pure function of (entity name, column,
    /// seed), uniform in [-1, 1).
    #[pyo3(signature = (feature_dim, seed = 0))]
    fn initialize_deterministically<'py>(
        &self,
        py: Python<'py>,
        feature_dim: usize,
        seed: i64,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let rows = self.num_entities();
        let data = py.detach(|| embedding::initialize(&self.entity_ids, feature_dim, seed));
        vec_to_pyarray(py, data, rows, feature_dim)
    }

    /// The matrix in coordinate form: `(rows, cols, values, n_rows, n_cols)`.
    /// `markov_type` is 'left' (default), 'symmetric' or 'raw' (un-normalised weights).
    #[pyo3(signature = (markov_type = None))]
    #[allow(clippy::type_complexity)]
    fn to_sparse_csr<'py>(
        &self,
        py: Python<'py>,
        markov_type: Option<&str>,
    ) -> PyResult<(
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<f32>>,
        usize,
        usize,
    )> {
        let kind = Normalization::parse(markov_type).map_err(PyValueError::new_err)?;
        let n = self.num_entities();
        let nnz = self.nnz();
        let mut rows: Vec<u32> = Vec::with_capacity(nnz);
        let mut values: Vec<f32> = Vec::with_capacity(nnz);
        for row in 0..n {
            for k in self.row_range(row) {
                rows.push(row as u32);
                values.push(self.normalized_value(row, k, kind));
            }
        }
        Ok((
            PyArray1::from_vec(py, rows),
            PyArray1::from_slice(py, &self.indices),
            PyArray1::from_vec(py, values),
            n,
            n,
        ))
    }

    /// The matrix in CSR form: `(indptr, indices, data)`, ready for
    /// `scipy.sparse.csr_matrix((data, indices, indptr), shape=(n, n))`.
    #[pyo3(signature = (markov_type = None))]
    #[allow(clippy::type_complexity)]
    fn to_csr<'py>(
        &self,
        py: Python<'py>,
        markov_type: Option<&str>,
    ) -> PyResult<(
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
        Bound<'py, PyArray1<f32>>,
    )> {
        let kind = Normalization::parse(markov_type).map_err(PyValueError::new_err)?;
        let n = self.num_entities();
        let mut values: Vec<f32> = Vec::with_capacity(self.nnz());
        for row in 0..n {
            for k in self.row_range(row) {
                values.push(self.normalized_value(row, k, kind));
            }
        }
        let data = PyArray1::from_vec(py, values);
        let small = self.nnz() <= i32::MAX as usize && n <= i32::MAX as usize;
        if small {
            let indptr: Vec<i32> = self.indptr.iter().map(|v| *v as i32).collect();
            let indices: Vec<i32> = self.indices.iter().map(|v| *v as i32).collect();
            Ok((
                PyArray1::from_vec(py, indptr).into_any(),
                PyArray1::from_vec(py, indices).into_any(),
                data,
            ))
        } else {
            let indptr: Vec<i64> = self.indptr.iter().map(|v| *v as i64).collect();
            let indices: Vec<i64> = self.indices.iter().map(|v| *v as i64).collect();
            Ok((
                PyArray1::from_vec(py, indptr).into_any(),
                PyArray1::from_vec(py, indices).into_any(),
                data,
            ))
        }
    }

    /// Neighbours of an entity with their random-walk transition probabilities.
    fn get_neighbors(&self, entity_id: &str) -> PyResult<Vec<(String, f32)>> {
        let row = self.get_entity_index(entity_id)?;
        Ok(self
            .row_range(row)
            .map(|k| {
                (
                    self.entity_ids[self.indices[k] as usize].clone(),
                    self.normalized_value(row, k, Normalization::Left),
                )
            })
            .collect())
    }

    /// Classic Cleora in one call: deterministic initialisation followed by
    /// `num_iterations` rounds of propagation and row-wise L2 normalisation.
    #[pyo3(signature = (feature_dim, num_iterations, propagation = "left", seed = 0, residual_weight = 0.0, num_workers = None))]
    #[allow(clippy::too_many_arguments)]
    fn embed_fast<'py>(
        &self,
        py: Python<'py>,
        feature_dim: usize,
        num_iterations: usize,
        propagation: &str,
        seed: i64,
        residual_weight: f32,
        num_workers: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.embed_classic(
            py,
            feature_dim,
            num_iterations,
            propagation,
            seed,
            residual_weight,
            0.0,
            num_workers,
        )
        .map(|(embeddings, _)| embeddings)
    }

    /// Like `embed_fast`, but stops early once the RMSE between two consecutive
    /// rounds drops below `convergence_threshold`. Returns `(embeddings, rounds)`.
    #[pyo3(signature = (feature_dim, max_iterations, propagation = "left", seed = 0, residual_weight = 0.0, convergence_threshold = 0.0, num_workers = None))]
    #[allow(clippy::too_many_arguments)]
    fn embed_fast_convergence<'py>(
        &self,
        py: Python<'py>,
        feature_dim: usize,
        max_iterations: usize,
        propagation: &str,
        seed: i64,
        residual_weight: f32,
        convergence_threshold: f32,
        num_workers: Option<usize>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, usize)> {
        self.embed_classic(
            py,
            feature_dim,
            max_iterations,
            propagation,
            seed,
            residual_weight,
            convergence_threshold,
            num_workers,
        )
    }

    /// Returns a copy of `x` with every row scaled to unit L2 norm.
    #[pyo3(signature = (x, num_workers = None))]
    fn l2_normalize<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        num_workers: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x = as_f32_matrix(x, "x")?;
        let (rows, dim) = (x.shape()[0], x.shape()[1]);
        let input = x
            .try_readonly()
            .map_err(|e| PyRuntimeError::new_err(format!("x cannot be read: {}", e)))?;
        let mut data = matrix_data(&input).into_owned();
        let pool = thread_pool(resolve_num_workers(num_workers));
        py.detach(|| pool.install(|| embedding::l2_normalize_rows(&mut data, dim)));
        vec_to_pyarray(py, data, rows, dim)
    }

    fn __repr__(&self) -> String {
        let columns = match self.relations.as_slice() {
            [relation] => format!(
                "('{}', '{}')",
                self.column_names[relation.a as usize], self.column_names[relation.b as usize]
            ),
            _ => format!(
                "({})",
                self.column_names
                    .iter()
                    .map(|name| format!("'{}'", name))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        format!(
            "SparseMatrix(entities={}, edges={}, columns={})",
            self.num_entities(),
            self.nnz(),
            columns
        )
    }

    fn __len__(&self) -> usize {
        self.num_entities()
    }

    #[new]
    #[pyo3(signature = (*args))]
    fn new(args: &Bound<'_, PyTuple>) -> PyResult<Self> {
        match args.len() {
            0 => Ok(SparseMatrix::empty()),
            _ => Err(PyValueError::new_err(
                "SparseMatrix cannot be constructed directly. Use SparseMatrix.from_files(), \
                 SparseMatrix.from_iterator() or SparseMatrix.from_edges().",
            )),
        }
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.to_state_bytes().map_err(PyRuntimeError::new_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        *self =
            SparseMatrix::from_state_bytes(state.as_bytes()).map_err(PyRuntimeError::new_err)?;
        Ok(())
    }
}

/// Column means and unbiased covariance of a float32 matrix, computed in float64.
/// The result does not depend on the number of threads.
#[pyfunction]
#[pyo3(signature = (x, num_workers = None))]
#[allow(clippy::type_complexity)]
fn column_mean_cov<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    num_workers: Option<usize>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let x = as_f32_matrix(x, "x")?;
    let (rows, dim) = (x.shape()[0], x.shape()[1]);
    let input = x
        .try_readonly()
        .map_err(|e| PyRuntimeError::new_err(format!("x cannot be read: {}", e)))?;
    let data = matrix_data(&input);
    let pool = thread_pool(resolve_num_workers(num_workers));
    let (mean, covariance) =
        py.detach(|| pool.install(|| embedding::column_mean_and_covariance(&data, rows, dim)));
    let covariance = Array2::from_shape_vec((dim, dim), covariance)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok((PyArray1::from_vec(py, mean), covariance.into_pyarray(py)))
}

fn check_projection_arguments(
    dim: usize,
    mean: &PyReadonlyArray1<'_, f32>,
    transform: &PyReadonlyArray2<'_, f32>,
) -> PyResult<()> {
    if mean.len() != dim {
        return Err(PyValueError::new_err(format!(
            "mean has {} element(s) but x has {} column(s)",
            mean.len(),
            dim
        )));
    }
    if transform.shape()[0] != dim {
        return Err(PyValueError::new_err(format!(
            "transform has {} row(s) but x has {} column(s)",
            transform.shape()[0],
            dim
        )));
    }
    Ok(())
}

/// Returns `(x - mean) @ transform` as a new float32 array.
#[pyfunction]
#[pyo3(signature = (x, mean, transform, num_workers = None))]
fn center_project<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    mean: PyReadonlyArray1<'py, f32>,
    transform: PyReadonlyArray2<'py, f32>,
    num_workers: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let x = as_f32_matrix(x, "x")?;
    let (rows, dim) = (x.shape()[0], x.shape()[1]);
    check_projection_arguments(dim, &mean, &transform)?;
    let components = transform.shape()[1];
    let input = x
        .try_readonly()
        .map_err(|e| PyRuntimeError::new_err(format!("x cannot be read: {}", e)))?;
    let data = matrix_data(&input);
    let mean: Vec<f32> = mean.as_array().iter().copied().collect();
    let transform = transform.as_array();
    let mut output = vec![0f32; rows * components];
    let pool = thread_pool(resolve_num_workers(num_workers));
    py.detach(|| {
        pool.install(|| embedding::center_and_project(&data, dim, &mean, transform, &mut output))
    });
    vec_to_pyarray(py, output, rows, components)
}

/// Replaces `x` by `(x - mean) @ transform` for a square transform, without
/// allocating a second matrix.
#[pyfunction]
#[pyo3(signature = (x, mean, transform, num_workers = None))]
fn center_project_inplace<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    mean: PyReadonlyArray1<'py, f32>,
    transform: PyReadonlyArray2<'py, f32>,
    num_workers: Option<usize>,
) -> PyResult<()> {
    let x = as_f32_matrix(x, "x")?;
    let dim = x.shape()[1];
    check_projection_arguments(dim, &mean, &transform)?;
    if transform.shape()[1] != dim {
        return Err(PyValueError::new_err(
            "an in-place projection needs a square transform",
        ));
    }
    let mut target = x
        .try_readwrite()
        .map_err(|_| PyValueError::new_err("x must be a writeable array"))?;
    let data = writable_matrix_data(&mut target, "x")?;
    let mean: Vec<f32> = mean.as_array().iter().copied().collect();
    let transform = transform.as_array();
    let pool = thread_pool(resolve_num_workers(num_workers));
    py.detach(|| {
        pool.install(|| embedding::center_and_project_inplace(data, dim, &mean, transform))
    });
    Ok(())
}

/// The 64-bit hash that identifies an entity name (XXH64, seed 0). It decides the
/// entity's initial vector and breaks ties when oversized hyperedges are trimmed.
#[pyfunction]
fn entity_hash(entity_id: &str) -> u64 {
    entity::hash_entity(entity_id)
}

/// Scales every row of `x` to unit L2 norm, in place.
#[pyfunction]
#[pyo3(signature = (x, num_workers = None))]
fn l2_normalize_inplace<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    num_workers: Option<usize>,
) -> PyResult<()> {
    let x = as_f32_matrix(x, "x")?;
    let dim = x.shape()[1];
    let mut target = x
        .try_readwrite()
        .map_err(|_| PyValueError::new_err("x must be a writeable array"))?;
    let data = writable_matrix_data(&mut target, "x")?;
    let pool = thread_pool(resolve_num_workers(num_workers));
    py.detach(|| pool.install(|| embedding::l2_normalize_rows(data, dim)));
    Ok(())
}

#[pymodule]
#[pyo3(name = "pycleora")]
fn pycleora(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SparseMatrix>()?;
    m.add_function(wrap_pyfunction!(column_mean_cov, m)?)?;
    m.add_function(wrap_pyfunction!(center_project, m)?)?;
    m.add_function(wrap_pyfunction!(center_project_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(l2_normalize_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(entity_hash, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
