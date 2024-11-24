use std::cmp::min;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::Hasher;

use bincode::{deserialize, serialize};
use ndarray::{Array1, Array2, ArrayViewMut2, Axis, Ix1, Ix2};
use numpy::{PyArray, PyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
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

// Methods not exposed to python
impl SparseMatrix {
    fn markov_propagate<'py>(
        &self,
        x: &'py PyArray2<f32>,
        markov_type: MarkovType,
        num_workers: Option<usize>,
    ) -> &'py PyArray<f32, Ix2> {
        let array = unsafe { x.as_array() };
        let multiplication_workers: usize = num_workers.unwrap_or_else(num_cpus::get);
        let propagated = NdArrayMatrix::multiply(self, array, markov_type, multiplication_workers);
        propagated.to_pyarray(x.py())
    }

    pub fn from_rust_iterator<'a>(
        columns: &str,
        hyperedge_trim_n: usize,
        hyperedges: impl Iterator<Item = &'a str>,
        num_workers: Option<usize>,
    ) -> Result<SparseMatrix, &'static str> {
        let columns = configuration::parse_fields(columns).expect("Columns should be valid");
        let matrix_desc = create_sparse_matrix_descriptor(&columns)?;
        let config = Configuration {
            seed: None,
            columns,
            matrix_desc,
            hyperedge_trim_n,
            num_workers_graph_building: num_workers.unwrap_or_else(|| min(num_cpus::get(), 8)),
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
    ) -> &'py PyArray<f32, Ix2> {
        self.markov_propagate(x, MarkovType::Left, num_workers)
    }

    #[pyo3(signature = (x, num_workers = None))]
    fn symmetric_markov_propagate<'py>(
        &self,
        x: &'py PyArray2<f32>,
        num_workers: Option<usize>,
    ) -> &'py PyArray<f32, Ix2> {
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
            let line = line.expect("Should be proper line");
            let line: &PyString = line
                .downcast()
                .expect("Iterator elements should be strings");
            let line = line.to_str().expect("Should be proper UTF-8 string");
            line
        });
        SparseMatrix::from_rust_iterator(columns, hyperedge_trim_n, hyperedges, num_workers)
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
        for filepath in filepaths.iter() {
            if !filepath.ends_with(".tsv") {
                return Err(PyValueError::new_err("Only .tsv files are supported"));
            }
        }

        let columns = configuration::parse_fields(columns).expect("Columns should be valid");
        let matrix_desc =
            create_sparse_matrix_descriptor(&columns).map_err(PyValueError::new_err)?;

        let config = Configuration {
            seed: None,
            matrix_desc,
            columns,
            hyperedge_trim_n,
            // TODO consider limiting to some maximum no of workers
            num_workers_graph_building: num_workers.unwrap_or_else(num_cpus::get),
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
            .ok_or(PyValueError::new_err("Column name invalid"))?;

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

    // Stuff needed for pickle to work (new, getstate, setstate)
    #[new]
    #[pyo3(signature = (*args))]
    fn new(args: &PyTuple) -> Self {
        match args.len() {
            0 => SparseMatrix {
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
            },
            _ => panic!("SparseMatrix::new never meant to be called by user. Only 0-arg implementation provided to make pickle happy"),
        }
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &serialize(self).unwrap()).to_object(py))
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                let sm: SparseMatrix = deserialize(s.as_bytes()).unwrap();
                *self = sm;
                Ok(())
            }
            Err(e) => Err(e),
        }
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
