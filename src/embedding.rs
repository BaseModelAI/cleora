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

                        let new_row: Array1<f32> = edges
                            .par_iter()
                            .fold(
                                || Array1::zeros(other.shape()[1]),
                                |mut row, edge| {
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
                                    row.scaled_add(*value, other_row);
                                    row
                                },
                            )
                            .reduce_with(|v1, v2| v1 + v2)
                            .expect("Must have at least one edge");

                        row.assign(&new_row);
                    });
            });
        new_matrix
    }
}
