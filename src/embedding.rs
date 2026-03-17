use crate::sparse_matrix::Edge;
use crate::sparse_matrix::SparseMatrix;
use ndarray::{Array, Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

pub enum MarkovType {
    Left,
    Symmetric,
}

pub struct NdArrayMatrix;

impl NdArrayMatrix {
    pub fn multiply(
        sparse_matrix_reader: &SparseMatrix,
        other: ArrayView2<f32>,
        markov_type: MarkovType,
        num_workers: usize,
    ) -> Array2<f32> {
        let mut new_matrix: Array2<f32> = Array::zeros(other.raw_dim());
        let dim = other.shape()[1];

        ThreadPoolBuilder::new()
            .num_threads(num_workers)
            .build()
            .unwrap()
            .install(|| {
                new_matrix
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .zip(sparse_matrix_reader.slices.par_iter())
                    .for_each(|(mut row, (start, end))| {
                        let edges = &sparse_matrix_reader.edges[*start..*end];

                        if edges.is_empty() {
                            return;
                        }

                        if edges.len() < 32 {
                            let mut acc = Array1::zeros(dim);
                            for edge in edges {
                                let Edge {
                                    left_markov_value,
                                    symmetric_markov_value,
                                    other_entity_ix,
                                } = edge;
                                let value = match markov_type {
                                    MarkovType::Left => left_markov_value,
                                    MarkovType::Symmetric => symmetric_markov_value,
                                };
                                let other_row = &other.row(*other_entity_ix as usize);
                                acc.scaled_add(*value, other_row);
                            }
                            row.assign(&acc);
                        } else {
                            let new_row: Array1<f32> = edges
                                .par_iter()
                                .fold(
                                    || Array1::zeros(dim),
                                    |mut acc, edge| {
                                        let Edge {
                                            left_markov_value,
                                            symmetric_markov_value,
                                            other_entity_ix,
                                        } = edge;
                                        let value = match markov_type {
                                            MarkovType::Left => left_markov_value,
                                            MarkovType::Symmetric => symmetric_markov_value,
                                        };
                                        let other_row = &other.row(*other_entity_ix as usize);
                                        acc.scaled_add(*value, other_row);
                                        acc
                                    },
                                )
                                .reduce_with(|v1, v2| v1 + v2)
                                .unwrap();

                            row.assign(&new_row);
                        }
                    });
            });
        new_matrix
    }
}
