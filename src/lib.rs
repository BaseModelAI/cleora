use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::Hasher;

use bincode::{deserialize, serialize};
use ndarray::{Array1, Array2, ArrayViewMut2, Axis, Ix1, Ix2};
use numpy::{PyArray, PyArray2, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyIterator, PyString, PyTuple};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator};

use crate::configuration::Configuration;
use crate::embedding::{MarkovType, NdArrayMatrix};
use crate::entity::hash_entity;
use crate::pipeline::{build_graph_from_files, build_graph_from_iterator};
use crate::sparse_matrix::{create_sparse_matrix_descriptor, SparseMatrix, SparseMatrixDescriptor};

pub mod configuration;
pub mod embedding;
pub mod entity;
pub mod pipeline;
pub mod sparse_matrix;
pub mod sparse_matrix_builder;

impl SparseMatrix {
    fn markov_propagate<'py>(
        &self,
        x: &'py PyArray2<f32>,
        markov_type: MarkovType,
        num_workers: Option<usize>,
    ) -> PyResult<&'py PyArray<f32, Ix2>> {
        let array = unsafe { x.as_array() };
        let num_rows = array.shape()[0];
        let num_entities = self.entity_ids.len();
        if num_rows != num_entities {
            return Err(PyValueError::new_err(format!(
                "Embedding matrix has {} rows but graph has {} entities",
                num_rows, num_entities
            )));
        }
        let multiplication_workers: usize = num_workers.unwrap_or_else(num_cpus::get).max(1);
        let propagated = NdArrayMatrix::multiply(self, array, markov_type, multiplication_workers);
        Ok(propagated.to_pyarray(x.py()))
    }

    pub fn from_rust_iterator<'a>(
        columns: &str,
        hyperedge_trim_n: usize,
        hyperedges: impl Iterator<Item = &'a str>,
        num_workers: Option<usize>,
    ) -> Result<SparseMatrix, String> {
        let columns = configuration::parse_fields(columns)?;
        let matrix_desc = create_sparse_matrix_descriptor(&columns)
            .map_err(|e| e.to_string())?;
        let workers = num_workers.unwrap_or_else(num_cpus::get).max(1);
        let config = Configuration {
            seed: None,
            columns,
            matrix_desc,
            hyperedge_trim_n,
            num_workers_graph_building: workers,
        };

        Ok(build_graph_from_iterator(&config, hyperedges))
    }

    fn initialize_deterministically_rust(&self, mut vectors: ArrayViewMut2<f32>, seed: i64) {
        vectors
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(entity_ix, mut row)| {
                let entity_id_hash = hash_entity(self.entity_ids[entity_ix].as_str());
                row.indexed_iter_mut().for_each(|(col_ix, v)| {
                    let value = init_value(col_ix, entity_id_hash, seed);
                    *v = value
                });
            });
    }
}

#[pymethods]
impl SparseMatrix {
    #[pyo3(signature = (x, num_workers = None))]
    pub fn left_markov_propagate<'py>(
        &self,
        x: &'py PyArray2<f32>,
        num_workers: Option<usize>,
    ) -> PyResult<&'py PyArray<f32, Ix2>> {
        self.markov_propagate(x, MarkovType::Left, num_workers)
    }

    #[pyo3(signature = (x, num_workers = None))]
    fn symmetric_markov_propagate<'py>(
        &self,
        x: &'py PyArray2<f32>,
        num_workers: Option<usize>,
    ) -> PyResult<&'py PyArray<f32, Ix2>> {
        self.markov_propagate(x, MarkovType::Symmetric, num_workers)
    }

    #[staticmethod]
    #[pyo3(signature = (hyperedges, columns, hyperedge_trim_n = 16, num_workers = None))]
    fn from_iterator(
        hyperedges: &PyIterator,
        columns: &str,
        hyperedge_trim_n: usize,
        num_workers: Option<usize>,
    ) -> PyResult<SparseMatrix> {
        let hyperedges = hyperedges.map(|line| {
            let line = line.map_err(|e| {
                PyValueError::new_err(format!("Error reading iterator element: {}", e))
            })?;
            let line: &PyString = line
                .downcast()
                .map_err(|_| PyValueError::new_err("Iterator elements must be strings"))?;
            let line = line
                .to_str()
                .map_err(|_| PyValueError::new_err("Iterator elements must be valid UTF-8"))?;
            Ok::<&str, PyErr>(line)
        });

        let collected: Result<Vec<&str>, PyErr> = hyperedges.collect();
        let collected = collected?;

        SparseMatrix::from_rust_iterator(
            columns,
            hyperedge_trim_n,
            collected.iter().map(|s| *s),
            num_workers,
        )
        .map_err(PyValueError::new_err)
    }

    #[staticmethod]
    #[pyo3(signature = (filepaths, columns, hyperedge_trim_n = 16, num_workers = None))]
    fn from_files(
        filepaths: Vec<String>,
        columns: &str,
        hyperedge_trim_n: usize,
        num_workers: Option<usize>,
    ) -> PyResult<SparseMatrix> {
        if filepaths.is_empty() {
            return Err(PyValueError::new_err("At least one file path is required"));
        }
        for filepath in filepaths.iter() {
            if !filepath.ends_with(".tsv") && !filepath.ends_with(".csv") && !filepath.ends_with(".txt") {
                return Err(PyValueError::new_err(
                    format!("Unsupported file format: {}. Supported: .tsv, .csv, .txt", filepath)
                ));
            }
        }

        let columns = configuration::parse_fields(columns)
            .map_err(PyValueError::new_err)?;
        let matrix_desc =
            create_sparse_matrix_descriptor(&columns).map_err(PyValueError::new_err)?;

        let workers = num_workers.unwrap_or_else(num_cpus::get).max(1);
        let config = Configuration {
            seed: None,
            matrix_desc,
            columns,
            hyperedge_trim_n,
            num_workers_graph_building: workers,
        };
        Ok(build_graph_from_files(&config, filepaths))
    }

    fn get_entity_column_mask<'py>(
        &self,
        py: Python<'py>,
        column_name: String,
    ) -> PyResult<&'py PyArray<bool, Ix1>> {
        let column_id_by_name = HashMap::from([
            (&self.descriptor.col_a_name, self.descriptor.col_a_id),
            (&self.descriptor.col_b_name, self.descriptor.col_b_id),
        ]);
        let column_id = column_id_by_name
            .get(&column_name)
            .ok_or(PyValueError::new_err(format!(
                "Column name '{}' not found. Available: '{}', '{}'",
                column_name, self.descriptor.col_a_name, self.descriptor.col_b_name
            )))?;

        let mask: Vec<bool> = self
            .column_ids
            .par_iter()
            .map(|id| *id == *column_id)
            .collect();
        let mask = Array1::from_vec(mask);
        Ok(mask.to_pyarray(py))
    }

    #[getter]
    fn entity_degrees<'py>(&self, py: Python<'py>) -> &'py PyArray<f32, Ix1> {
        let entity_degrees: Vec<f32> = self.entities.par_iter().map(|e| e.row_sum).collect();
        Array1::from_vec(entity_degrees).to_pyarray(py)
    }

    #[getter]
    fn num_entities(&self) -> usize {
        self.entity_ids.len()
    }

    #[getter]
    fn num_edges(&self) -> usize {
        self.edges.len()
    }

    fn get_entity_index(&self, entity_id: &str) -> PyResult<usize> {
        self.entity_ids
            .iter()
            .position(|id| id == entity_id)
            .ok_or_else(|| PyValueError::new_err(format!("Entity '{}' not found", entity_id)))
    }

    fn get_entity_indices(&self, entity_ids: Vec<String>) -> PyResult<Vec<usize>> {
        let index_map: HashMap<&str, usize> = self
            .entity_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i))
            .collect();

        entity_ids
            .iter()
            .map(|id| {
                index_map
                    .get(id.as_str())
                    .copied()
                    .ok_or_else(|| PyValueError::new_err(format!("Entity '{}' not found", id)))
            })
            .collect()
    }

    #[pyo3(signature = (feature_dim, seed = 0))]
    fn initialize_deterministically<'py>(
        &self,
        py: Python<'py>,
        feature_dim: usize,
        seed: i64,
    ) -> &'py PyArray<f32, Ix2> {
        let mut vectors = Array2::zeros([self.entity_ids.len(), feature_dim]);
        self.initialize_deterministically_rust(vectors.view_mut(), seed);
        vectors.to_pyarray(py)
    }

    #[pyo3(signature = (markov_type = None))]
    fn to_sparse_csr<'py>(
        &self,
        py: Python<'py>,
        markov_type: Option<&str>,
    ) -> PyResult<(
        &'py PyArray<u32, Ix1>,
        &'py PyArray<u32, Ix1>,
        &'py PyArray<f32, Ix1>,
        usize,
        usize,
    )> {
        let n = self.entity_ids.len();
        let nnz = self.edges.len();
        let mut row_indices: Vec<u32> = Vec::with_capacity(nnz);
        let mut col_indices: Vec<u32> = Vec::with_capacity(nnz);
        let mut values: Vec<f32> = Vec::with_capacity(nnz);

        let mt = markov_type.unwrap_or("left");
        if mt != "left" && mt != "symmetric" {
            return Err(PyValueError::new_err(format!(
                "Unknown markov_type '{}'. Use 'left' or 'symmetric'.",
                mt
            )));
        }
        let use_symmetric = mt == "symmetric";

        for (row_ix, (start, end)) in self.slices.iter().enumerate() {
            for edge in &self.edges[*start..*end] {
                row_indices.push(row_ix as u32);
                col_indices.push(edge.other_entity_ix);
                values.push(if use_symmetric {
                    edge.symmetric_markov_value
                } else {
                    edge.left_markov_value
                });
            }
        }

        Ok((
            Array1::from_vec(row_indices).to_pyarray(py),
            Array1::from_vec(col_indices).to_pyarray(py),
            Array1::from_vec(values).to_pyarray(py),
            n,
            n,
        ))
    }

    fn get_neighbors(&self, entity_id: &str) -> PyResult<Vec<(String, f32)>> {
        let idx = self
            .entity_ids
            .iter()
            .position(|id| id == entity_id)
            .ok_or_else(|| PyValueError::new_err(format!("Entity '{}' not found", entity_id)))?;

        let (start, end) = self.slices[idx];
        let neighbors: Vec<(String, f32)> = self.edges[start..end]
            .iter()
            .map(|edge| {
                let neighbor_id = self.entity_ids[edge.other_entity_ix as usize].clone();
                (neighbor_id, edge.left_markov_value)
            })
            .collect();
        Ok(neighbors)
    }

    fn __repr__(&self) -> String {
        format!(
            "SparseMatrix(entities={}, edges={}, columns=('{}', '{}'))",
            self.entity_ids.len(),
            self.edges.len(),
            self.descriptor.col_a_name,
            self.descriptor.col_b_name
        )
    }

    fn __len__(&self) -> usize {
        self.entity_ids.len()
    }

    #[new]
    #[pyo3(signature = (*args))]
    fn new(args: &PyTuple) -> PyResult<Self> {
        match args.len() {
            0 => Ok(SparseMatrix {
                descriptor: SparseMatrixDescriptor {
                    col_a_id: 0,
                    col_a_name: "".to_string(),
                    col_b_id: 0,
                    col_b_name: "".to_string(),
                },
                entity_ids: vec![],
                entities: vec![],
                edges: vec![],
                slices: vec![],
                column_ids: vec![],
            }),
            _ => Err(PyValueError::new_err(
                "SparseMatrix cannot be constructed directly. Use SparseMatrix.from_files() or SparseMatrix.from_iterator()."
            )),
        }
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let bytes = serialize(self)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization failed: {}", e)))?;
        Ok(PyBytes::new(py, &bytes).to_object(py))
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        let bytes = state.extract::<&PyBytes>(py)?;
        let sm: SparseMatrix = deserialize(bytes.as_bytes())
            .map_err(|e| PyRuntimeError::new_err(format!("Deserialization failed: {}", e)))?;
        *self = sm;
        Ok(())
    }
}

fn init_value(col: usize, hsh: u64, fixed_random_value: i64) -> f32 {
    let hash = |num: i64| {
        let mut hasher = DefaultHasher::new();
        hasher.write_i64(num);
        hasher.finish() as i64
    };

    const MAX_HASH_I64: i64 = 8 * 1024 * 1024;
    const MAX_HASH_F32: f32 = MAX_HASH_I64 as f32;
    ((hash((hsh as i64) + (col as i64) + fixed_random_value) % MAX_HASH_I64) as f32) / MAX_HASH_F32
}

#[pymodule]
#[pyo3(name = "pycleora")]
fn pycleora(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SparseMatrix>()?;
    Ok(())
}
