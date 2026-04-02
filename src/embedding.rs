use crate::sparse_matrix::Edge;
use crate::sparse_matrix::SparseMatrix;
use ndarray::{Array, Array1, Array2, ArrayView2, ArrayViewMut2, Axis};
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

        let pool = ThreadPoolBuilder::new()
            .num_threads(num_workers)
            .build()
            .unwrap();

        pool.install(|| {
            Self::spmm_kernel(
                sparse_matrix_reader,
                other,
                &markov_type,
                dim,
                new_matrix.view_mut(),
            );
        });
        new_matrix
    }

    pub fn multiply_into(
        sparse_matrix_reader: &SparseMatrix,
        src: ArrayView2<f32>,
        markov_type: &MarkovType,
        dst: &mut Array2<f32>,
    ) {
        let dim = src.shape()[1];
        dst.fill(0.0);
        Self::spmm_kernel(sparse_matrix_reader, src, markov_type, dim, dst.view_mut());
    }

    fn spmm_kernel(
        sparse_matrix_reader: &SparseMatrix,
        other: ArrayView2<f32>,
        markov_type: &MarkovType,
        dim: usize,
        mut output: ArrayViewMut2<f32>,
    ) {
        output
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(sparse_matrix_reader.slices.par_iter())
            .for_each(|(mut row, (start, end))| {
                let edges = &sparse_matrix_reader.edges[*start..*end];

                if edges.is_empty() {
                    return;
                }

                let mut acc = Array1::zeros(dim);
                for edge in edges {
                    let value = match markov_type {
                        MarkovType::Left => &edge.left_markov_value,
                        MarkovType::Symmetric => &edge.symmetric_markov_value,
                    };
                    let other_row = other.row(edge.other_entity_ix as usize);
                    let acc_slice = acc.as_slice_mut().unwrap();
                    let src_slice = other_row.as_slice().unwrap();
                    let v = *value;
                    for j in 0..dim {
                        acc_slice[j] += v * src_slice[j];
                    }
                }
                row.assign(&acc);
            });
    }

    pub fn l2_normalize_inplace(matrix: &mut Array2<f32>) {
        matrix
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                let slice = row.as_slice_mut().unwrap();
                let mut sum_sq: f32 = 0.0;
                for v in slice.iter() {
                    sum_sq += v * v;
                }
                let norm = sum_sq.sqrt().max(1e-10);
                let inv_norm = 1.0 / norm;
                for v in slice.iter_mut() {
                    *v *= inv_norm;
                }
            });
    }

    pub fn embed_full(
        sparse_matrix_reader: &SparseMatrix,
        initial: Array2<f32>,
        markov_type: MarkovType,
        num_iterations: usize,
        residual_weight: f32,
    ) -> Array2<f32> {
        let shape = initial.raw_dim();
        let mut src = initial;
        let mut dst = Array2::<f32>::zeros(shape);
        let use_residual = residual_weight > 0.0 && residual_weight < 1.0;

        for _ in 0..num_iterations {
            Self::multiply_into(sparse_matrix_reader, src.view(), &markov_type, &mut dst);

            if use_residual {
                let alpha = 1.0 - residual_weight;
                let rw = residual_weight;
                let dst_slice = dst.as_slice_mut().unwrap();
                let src_slice = src.as_slice().unwrap();
                for j in 0..dst_slice.len() {
                    dst_slice[j] = alpha * dst_slice[j] + rw * src_slice[j];
                }
            }

            Self::l2_normalize_inplace(&mut dst);

            std::mem::swap(&mut src, &mut dst);
        }
        src
    }

    pub fn embed_full_with_convergence(
        sparse_matrix_reader: &SparseMatrix,
        initial: Array2<f32>,
        markov_type: MarkovType,
        max_iterations: usize,
        residual_weight: f32,
        convergence_threshold: f32,
    ) -> (Array2<f32>, usize) {
        let shape = initial.raw_dim();
        let mut src = initial;
        let mut dst = Array2::<f32>::zeros(shape);
        let use_residual = residual_weight > 0.0 && residual_weight < 1.0;
        let check_convergence = convergence_threshold > 0.0;
        let mut actual_iterations = max_iterations;
        let dim = shape[1];

        for iter in 0..max_iterations {
            Self::multiply_into(sparse_matrix_reader, src.view(), &markov_type, &mut dst);

            if use_residual {
                let alpha = 1.0 - residual_weight;
                let rw = residual_weight;
                let dst_slice = dst.as_slice_mut().unwrap();
                let src_slice = src.as_slice().unwrap();
                for j in 0..dst_slice.len() {
                    dst_slice[j] = alpha * dst_slice[j] + rw * src_slice[j];
                }
            }

            Self::l2_normalize_inplace(&mut dst);

            if check_convergence && iter > 0 {
                let dst_slice = dst.as_slice().unwrap();
                let src_slice = src.as_slice().unwrap();
                let mut diff: f32 = 0.0;
                for j in 0..dst_slice.len() {
                    let delta = dst_slice[j] - src_slice[j];
                    diff += delta * delta;
                }
                let rmse = (diff / (dst.shape()[0] * dim) as f32).sqrt();
                if rmse < convergence_threshold {
                    std::mem::swap(&mut src, &mut dst);
                    actual_iterations = iter + 1;
                    break;
                }
            }

            std::mem::swap(&mut src, &mut dst);
        }
        (src, actual_iterations)
    }
}
