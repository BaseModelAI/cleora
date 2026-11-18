//! Dense kernels: sparse propagation, normalisation, initialisation and the
//! building blocks of whitening.
//!
//! All kernels give the same result for any number of threads: parallel work is
//! either independent per row, or reduced in a fixed order.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::entity::hash_entity;
use crate::sparse_matrix::SparseMatrix;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkovType {
    Left,
    Symmetric,
}

/// Returns a thread pool with the requested number of threads, creating it on
/// first use. Pools are kept for the lifetime of the process.
pub fn thread_pool(num_threads: usize) -> Arc<ThreadPool> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<ThreadPool>>>> = OnceLock::new();
    let num_threads = num_threads.max(1);
    let mut pools = POOLS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    pools
        .entry(num_threads)
        .or_insert_with(|| {
            Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .thread_name(|i| format!("pycleora-{}", i))
                    .build()
                    .expect("failed to start worker threads"),
            )
        })
        .clone()
}

/// Rows handled by one parallel task.
const ROWS_PER_TASK: usize = 64;

/// `out = M * x` for the row-major `n x dim` matrices `x` and `out`, where `M` is
/// the left or symmetric Markov matrix of the graph.
/// Runs on the current rayon pool.
pub fn propagate(graph: &SparseMatrix, x: &[f32], out: &mut [f32], dim: usize, kind: MarkovType) {
    let n = graph.num_entities();
    assert_eq!(x.len(), n * dim, "input has the wrong size");
    assert_eq!(out.len(), n * dim, "output has the wrong size");
    if n == 0 || dim == 0 {
        return;
    }
    out.par_chunks_mut(dim * ROWS_PER_TASK)
        .enumerate()
        .for_each(|(task, chunk)| {
            propagate_rows(graph, x, chunk, Rows::From(task * ROWS_PER_TASK), dim, kind);
        });
}

/// The rows `rows` of `M * x`: `out` is a row-major `rows.len() x dim` matrix whose
/// i-th row is bit-identical to row `rows[i]` of what [`propagate`] computes.
/// Costs time proportional to the entries of the selected rows only.
pub fn propagate_selected(
    graph: &SparseMatrix,
    x: &[f32],
    rows: &[usize],
    out: &mut [f32],
    dim: usize,
    kind: MarkovType,
) {
    let n = graph.num_entities();
    assert_eq!(x.len(), n * dim, "input has the wrong size");
    assert_eq!(out.len(), rows.len() * dim, "output has the wrong size");
    assert!(rows.iter().all(|&row| row < n), "row index out of range");
    if rows.is_empty() || dim == 0 {
        return;
    }
    out.par_chunks_mut(dim * ROWS_PER_TASK)
        .zip(rows.par_chunks(ROWS_PER_TASK))
        .for_each(|(chunk, rows)| {
            propagate_rows(graph, x, chunk, Rows::List(rows), dim, kind);
        });
}

/// The rows one task computes: consecutive ones or an explicit list.
#[derive(Clone, Copy)]
enum Rows<'a> {
    From(usize),
    List(&'a [usize]),
}

#[inline]
fn propagate_rows(
    graph: &SparseMatrix,
    x: &[f32],
    chunk: &mut [f32],
    rows: Rows<'_>,
    dim: usize,
    kind: MarkovType,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // Safety: AVX2 availability was just checked.
            unsafe { propagate_rows_avx2(graph, x, chunk, rows, dim, kind) };
            return;
        }
    }
    propagate_rows_generic(graph, x, chunk, rows, dim, kind);
}

/// Same code as the generic version, compiled for AVX2. No fused multiply-add is
/// used, so both versions produce bit-identical results.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn propagate_rows_avx2(
    graph: &SparseMatrix,
    x: &[f32],
    chunk: &mut [f32],
    rows: Rows<'_>,
    dim: usize,
    kind: MarkovType,
) {
    propagate_rows_generic(graph, x, chunk, rows, dim, kind)
}

#[inline(always)]
fn propagate_rows_generic(
    graph: &SparseMatrix,
    x: &[f32],
    chunk: &mut [f32],
    rows: Rows<'_>,
    dim: usize,
    kind: MarkovType,
) {
    let symmetric = kind == MarkovType::Symmetric;
    for (offset, dst) in chunk.chunks_exact_mut(dim).enumerate() {
        let row = match rows {
            Rows::From(first_row) => first_row + offset,
            Rows::List(list) => list[offset],
        };
        dst.fill(0.0);
        for k in graph.row_range(row) {
            let column = graph.indices[k] as usize;
            let coefficient = if symmetric {
                graph.values[k] * graph.inv_sqrt_row[column]
            } else {
                graph.values[k]
            };
            let src = &x[column * dim..(column + 1) * dim];
            for (d, s) in dst.iter_mut().zip(src) {
                *d += coefficient * *s;
            }
        }
        let scale = if symmetric {
            graph.inv_sqrt_row[row]
        } else {
            graph.inv_row[row]
        };
        for d in dst.iter_mut() {
            *d *= scale;
        }
    }
}

/// Scales every row to unit L2 norm (rows of zeros stay zero).
pub fn l2_normalize_rows(matrix: &mut [f32], dim: usize) {
    if dim == 0 {
        return;
    }
    matrix
        .par_chunks_mut(dim * ROWS_PER_TASK)
        .for_each(|chunk| {
            for row in chunk.chunks_exact_mut(dim) {
                let mut sum_sq: f32 = 0.0;
                for v in row.iter() {
                    sum_sq += v * v;
                }
                let inv_norm = 1.0 / sum_sq.sqrt().max(1e-10);
                for v in row.iter_mut() {
                    *v *= inv_norm;
                }
            }
        });
}

/// `dst = (1 - residual_weight) * dst + residual_weight * src`
fn mix_residual(dst: &mut [f32], src: &[f32], residual_weight: f32) {
    let alpha = 1.0 - residual_weight;
    dst.par_chunks_mut(1 << 14)
        .zip(src.par_chunks(1 << 14))
        .for_each(|(d, s)| {
            for (d, s) in d.iter_mut().zip(s) {
                *d = alpha * *d + residual_weight * *s;
            }
        });
}

/// Root mean squared difference between two matrices, independent of the thread count.
fn rmse(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let partial: Vec<f64> = a
        .par_chunks(1 << 14)
        .zip(b.par_chunks(1 << 14))
        .map(|(a, b)| {
            let mut sum = 0.0f64;
            for (x, y) in a.iter().zip(b) {
                let delta = (*x - *y) as f64;
                sum += delta * delta;
            }
            sum
        })
        .collect();
    (partial.iter().sum::<f64>() / a.len() as f64).sqrt()
}

/// Classic Cleora: `num_iterations` rounds of propagation followed by row-wise L2
/// normalisation. Returns the embeddings and the number of rounds performed.
/// With `convergence_threshold > 0` iteration stops once the RMSE between two
/// consecutive rounds drops below the threshold.
pub fn embed(
    graph: &SparseMatrix,
    initial: Vec<f32>,
    dim: usize,
    kind: MarkovType,
    num_iterations: usize,
    residual_weight: f32,
    convergence_threshold: f32,
) -> (Vec<f32>, usize) {
    let mut src = initial;
    let mut dst = vec![0f32; src.len()];
    let use_residual = residual_weight > 0.0 && residual_weight < 1.0;
    let check_convergence = convergence_threshold > 0.0;

    for iteration in 0..num_iterations {
        propagate(graph, &src, &mut dst, dim, kind);
        if use_residual {
            mix_residual(&mut dst, &src, residual_weight);
        }
        l2_normalize_rows(&mut dst, dim);

        let converged =
            check_convergence && iteration > 0 && rmse(&dst, &src) < convergence_threshold as f64;
        std::mem::swap(&mut src, &mut dst);
        if converged {
            return (src, iteration + 1);
        }
    }
    (src, num_iterations)
}

#[inline(always)]
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Per-column keys of the initialisation for a given seed.
fn init_column_keys(dim: usize, seed: i64) -> Vec<u64> {
    let seed_key = splitmix64(seed as u64);
    (0..dim as u64)
        .map(|column| splitmix64(column ^ seed_key))
        .collect()
}

/// Initial value of one coordinate: uniform in `[-1, 1)`, a pure function of
/// (entity name, column, seed).
#[inline(always)]
fn init_value(entity_hash: u64, column_key: u64) -> f32 {
    let bits = splitmix64(entity_hash ^ column_key) >> 40; // 24 random bits
    (bits as f32) * (1.0 / 8_388_608.0) - 1.0
}

/// Deterministic pseudo-random initial embeddings.
///
/// Every entity gets a vector that depends only on its name, the seed and the
/// dimension index, so it is the same in every graph and on every machine, and
/// vectors of different entities (and of different seeds) are statistically
/// independent.
pub fn initialize(entity_ids: &[String], dim: usize, seed: i64) -> Vec<f32> {
    let mut out = vec![0f32; entity_ids.len() * dim];
    if dim == 0 {
        return out;
    }
    let keys = init_column_keys(dim, seed);
    out.par_chunks_mut(dim)
        .zip(entity_ids.par_iter())
        .for_each(|(row, id)| {
            let hash = hash_entity(id);
            for (value, key) in row.iter_mut().zip(&keys) {
                *value = init_value(hash, *key);
            }
        });
    out
}

/// Number of row groups used for the order-fixed reduction behind the column means.
/// Depends on the shape only, never on the number of threads.
fn mean_groups(n: usize) -> usize {
    n.div_ceil(4096).clamp(1, 64)
}

/// Rows that are centred, converted to f64 and multiplied at once.
const GRAM_BLOCK: usize = 256;

/// How the `d x d` matrix of centred second moments is accumulated in parallel.
/// Both plans are functions of the shape only, so the result never depends on the
/// number of threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GramPlan {
    /// The rows are split into this many groups; every group accumulates a matrix
    /// of its own and the matrices are added in order. Fastest, but needs one
    /// `d x d` matrix per group.
    RowGroups(usize),
    /// One matrix; for every block of rows its bands (this many matrix rows each)
    /// are independent products. Needs no extra matrices, which is what counts
    /// when `d` is large.
    Bands(usize),
}

fn gram_plan(n: usize, d: usize) -> GramPlan {
    // Per-group matrices may take 1/16 of the size of the input, and 16 MB at least.
    let budget = (16usize << 20).max(n * d * 4 / 16);
    let affordable = budget / (d * d * 8).max(1);
    let wanted = n.div_ceil(1024).clamp(1, 64);
    if d >= 256 && affordable < wanted.min(8) {
        GramPlan::Bands(d.div_ceil(32).max(64))
    } else {
        GramPlan::RowGroups(wanted.min(affordable).max(1))
    }
}

/// `block <- rows - mean` in f64.
fn centre_block(rows: &[f32], mean: &[f64], d: usize, block: &mut [f64]) {
    for (centred, row) in block.chunks_exact_mut(d).zip(rows.chunks_exact(d)) {
        for ((c, v), m) in centred.iter_mut().zip(row).zip(mean) {
            *c = *v as f64 - *m;
        }
    }
}

fn gram_by_row_groups(x: &[f32], n: usize, d: usize, mean: &[f64], groups: usize) -> Vec<f64> {
    let rows_per_group = n.div_ceil(groups);
    let partial: Vec<Array2<f64>> = x
        .par_chunks(rows_per_group * d)
        .map(|group| {
            let mut gram = Array2::<f64>::zeros((d, d));
            let mut block = vec![0.0f64; GRAM_BLOCK.min(group.len() / d) * d];
            for rows in group.chunks(GRAM_BLOCK * d) {
                let count = rows.len() / d;
                centre_block(rows, mean, d, &mut block[..count * d]);
                let view = ArrayView2::from_shape((count, d), &block[..count * d])
                    .expect("block has the shape it was allocated with");
                ndarray::linalg::general_mat_mul(1.0, &view.t(), &view, 1.0, &mut gram);
            }
            gram
        })
        .collect();
    let mut total = vec![0.0f64; d * d];
    for gram in &partial {
        for (t, g) in total.iter_mut().zip(gram.iter()) {
            *t += *g;
        }
    }
    total
}

fn gram_by_bands(x: &[f32], n: usize, d: usize, mean: &[f64], band_rows: usize) -> Vec<f64> {
    // about 4 MB of f64 values per block
    let block_rows = ((1usize << 19) / d).clamp(64, 4096).min(n);
    let mut total = vec![0.0f64; d * d];
    let mut block = vec![0.0f64; block_rows * d];
    for rows in x.chunks(block_rows * d) {
        let count = rows.len() / d;
        block[..count * d]
            .par_chunks_mut(GRAM_BLOCK * d)
            .zip(rows.par_chunks(GRAM_BLOCK * d))
            .for_each(|(centred, rows)| centre_block(rows, mean, d, centred));
        let view = ArrayView2::from_shape((count, d), &block[..count * d])
            .expect("block has the shape it was allocated with");
        total
            .par_chunks_mut(band_rows * d)
            .enumerate()
            .for_each(|(band, target)| {
                let first = band * band_rows;
                let height = target.len() / d;
                let mut target = ArrayViewMut2::from_shape((height, d), target)
                    .expect("band has the shape it was split into");
                let columns = view.slice(ndarray::s![.., first..first + height]);
                // rows [first, first + height) of block^T * block
                ndarray::linalg::general_mat_mul(1.0, &columns.t(), &view, 1.0, &mut target);
            });
    }
    total
}

/// Column means and the (unbiased) covariance matrix of an `n x d` row-major
/// matrix, both in f64. The covariance is returned row-major, `d x d`.
/// Runs on the current rayon pool; the result does not depend on its size.
pub fn column_mean_and_covariance(x: &[f32], n: usize, d: usize) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(x.len(), n * d);
    if n == 0 || d == 0 {
        return (vec![0.0; d], vec![0.0; d * d]);
    }
    let rows_per_group = n.div_ceil(mean_groups(n));
    let partial_sums: Vec<Vec<f64>> = x
        .par_chunks(rows_per_group * d)
        .map(|group| {
            let mut sums = vec![0.0f64; d];
            for row in group.chunks_exact(d) {
                for (s, v) in sums.iter_mut().zip(row) {
                    *s += *v as f64;
                }
            }
            sums
        })
        .collect();
    let mut mean = vec![0.0f64; d];
    for sums in &partial_sums {
        for (m, s) in mean.iter_mut().zip(sums) {
            *m += *s;
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f64;
    }

    let mut covariance = match gram_plan(n, d) {
        GramPlan::RowGroups(groups) => gram_by_row_groups(x, n, d, &mean, groups),
        GramPlan::Bands(band_rows) => gram_by_bands(x, n, d, &mean, band_rows),
    };
    let scale = 1.0 / (n.max(2) - 1) as f64;
    for value in covariance.iter_mut() {
        *value *= scale;
    }
    // exact symmetry, whatever rounding the products picked up
    for i in 0..d {
        for j in (i + 1)..d {
            let v = 0.5 * (covariance[i * d + j] + covariance[j * d + i]);
            covariance[i * d + j] = v;
            covariance[j * d + i] = v;
        }
    }
    (mean, covariance)
}

/// Rows projected at once.
const PROJECT_BLOCK: usize = 512;

/// `out = (x - mean) * transform` for an `n x d` matrix `x` and a `d x k`
/// transform; `out` is `n x k`.
pub fn center_and_project(
    x: &[f32],
    d: usize,
    mean: &[f32],
    transform: ArrayView2<f32>,
    out: &mut [f32],
) {
    let k = transform.ncols();
    assert_eq!(transform.nrows(), d);
    assert_eq!(mean.len(), d);
    if d == 0 || k == 0 || x.is_empty() {
        return;
    }
    assert_eq!(x.len() / d * k, out.len());
    out.par_chunks_mut(PROJECT_BLOCK * k)
        .zip(x.par_chunks(PROJECT_BLOCK * d))
        .for_each(|(dst, src)| {
            let rows = src.len() / d;
            let mut centered = Array2::<f32>::zeros((rows, d));
            for (c, s) in centered
                .as_slice_mut()
                .unwrap()
                .chunks_exact_mut(d)
                .zip(src.chunks_exact(d))
            {
                for ((c, s), m) in c.iter_mut().zip(s).zip(mean) {
                    *c = *s - *m;
                }
            }
            let projected = centered.dot(&transform);
            dst.copy_from_slice(projected.as_slice().unwrap());
        });
}

/// In-place variant of [`center_and_project`] for a square (`d x d`) transform.
pub fn center_and_project_inplace(
    x: &mut [f32],
    d: usize,
    mean: &[f32],
    transform: ArrayView2<f32>,
) {
    assert_eq!(transform.nrows(), d);
    assert_eq!(transform.ncols(), d);
    assert_eq!(mean.len(), d);
    if d == 0 || x.is_empty() {
        return;
    }
    x.par_chunks_mut(PROJECT_BLOCK * d).for_each(|chunk| {
        let rows = chunk.len() / d;
        let mut centered = Array2::<f32>::zeros((rows, d));
        for (c, s) in centered
            .as_slice_mut()
            .unwrap()
            .chunks_exact_mut(d)
            .zip(chunk.chunks_exact(d))
        {
            for ((c, s), m) in c.iter_mut().zip(s).zip(mean) {
                *c = *s - *m;
            }
        }
        let projected = centered.dot(&transform);
        chunk.copy_from_slice(projected.as_slice().unwrap());
    });
}

/// Kept for the Rust API and the snapshot tests.
pub struct NdArrayMatrix;

impl NdArrayMatrix {
    pub fn multiply(
        graph: &SparseMatrix,
        other: ArrayView2<f32>,
        markov_type: MarkovType,
        num_workers: usize,
    ) -> Array2<f32> {
        let dim = other.ncols();
        let input = other.as_standard_layout();
        let x = input.as_slice().expect("standard layout");
        let mut out = vec![0f32; x.len()];
        thread_pool(num_workers).install(|| propagate(graph, x, &mut out, dim, markov_type));
        Array2::from_shape_vec((other.nrows(), dim), out).expect("matching shape")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("entity_{}", i)).collect()
    }

    fn ring_graph(n: usize) -> SparseMatrix {
        let lines: Vec<String> = (0..n)
            .map(|i| format!("v{} v{} v{}", i, (i + 1) % n, (i * 7 + 3) % n))
            .collect();
        let config =
            crate::configuration::Configuration::new("complex::reflexive::v", 16, Some(2)).unwrap();
        crate::pipeline::build_graph_from_lines(&config, &lines, None)
            .unwrap()
            .0
    }

    #[test]
    fn selected_rows_match_the_full_propagation() {
        let graph = ring_graph(300);
        let n = graph.num_entities();
        let dim = 9;
        let x = initialize(&graph.entity_ids, dim, 4);
        let rows: Vec<usize> = (0..500).map(|i| (i * 37 + 11) % n).collect();
        for kind in [MarkovType::Left, MarkovType::Symmetric] {
            let mut full = vec![0f32; n * dim];
            propagate(&graph, &x, &mut full, dim, kind);
            let mut selected = vec![f32::NAN; rows.len() * dim];
            propagate_selected(&graph, &x, &rows, &mut selected, dim, kind);
            for (i, &row) in rows.iter().enumerate() {
                assert_eq!(
                    &selected[i * dim..(i + 1) * dim],
                    &full[row * dim..(row + 1) * dim]
                );
            }
        }
        let mut empty: Vec<f32> = Vec::new();
        propagate_selected(&graph, &x, &[], &mut empty, dim, MarkovType::Left);
    }

    #[test]
    fn init_is_uniform_and_in_range() {
        let dim = 256;
        let x = initialize(&ids(2000), dim, 0);
        let (mut min, mut max, mut sum, mut sum_sq) = (f32::MAX, f32::MIN, 0f64, 0f64);
        for v in &x {
            min = min.min(*v);
            max = max.max(*v);
            sum += *v as f64;
            sum_sq += (*v as f64) * (*v as f64);
        }
        let mean = sum / x.len() as f64;
        let variance = sum_sq / x.len() as f64 - mean * mean;
        assert!(min >= -1.0 && max < 1.0);
        assert!(mean.abs() < 0.01, "mean {}", mean);
        assert!((variance - 1.0 / 3.0).abs() < 0.01, "variance {}", variance);
    }

    /// Vectors of different entities must be (nearly) orthogonal: the cosine of a
    /// random pair has standard deviation 1/sqrt(dim).
    #[test]
    fn init_vectors_of_different_entities_are_uncorrelated() {
        for dim in [64usize, 256, 1024] {
            let n = 400;
            let x = initialize(&ids(n), dim, 0);
            let row = |i: usize| &x[i * dim..(i + 1) * dim];
            let norm = |r: &[f32]| r.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
            let mut sum_sq = 0f64;
            let mut count = 0usize;
            let mut worst = 0f64;
            for i in 0..n {
                for j in (i + 1)..n {
                    let dot: f64 = row(i)
                        .iter()
                        .zip(row(j))
                        .map(|(a, b)| *a as f64 * *b as f64)
                        .sum();
                    let cosine = dot / (norm(row(i)) * norm(row(j)));
                    sum_sq += cosine * cosine;
                    worst = worst.max(cosine.abs());
                    count += 1;
                }
            }
            let std = (sum_sq / count as f64).sqrt();
            let ideal = 1.0 / (dim as f64).sqrt();
            assert!(
                std < 1.15 * ideal,
                "dim {}: cosine std {} vs ideal {}",
                dim,
                std,
                ideal
            );
            assert!(worst < 8.0 * ideal, "dim {}: worst |cosine| {}", dim, worst);
        }
    }

    #[test]
    fn init_depends_on_name_seed_and_column_only() {
        let a = initialize(&ids(50), 32, 7);
        let b = initialize(&ids(50), 32, 7);
        assert_eq!(a, b);
        // a prefix of a wider initialisation is the narrower one
        let wide = initialize(&ids(50), 64, 7);
        for i in 0..50 {
            assert_eq!(&wide[i * 64..i * 64 + 32], &a[i * 32..(i + 1) * 32]);
        }
        // the vector of an entity does not depend on which other entities exist
        let single = initialize(&["entity_13".to_string()], 32, 7);
        assert_eq!(&a[13 * 32..14 * 32], &single[..]);
    }

    /// Seeds must give unrelated matrices (they used to be column shifts of each other).
    #[test]
    fn init_seeds_are_independent() {
        let dim = 128;
        let a = initialize(&ids(300), dim, 0);
        let b = initialize(&ids(300), dim, 1);
        for shift in 0..4 {
            let mut dot = 0f64;
            let mut count = 0usize;
            for i in 0..300 {
                for c in 0..(dim - shift) {
                    dot += a[i * dim + c + shift] as f64 * b[i * dim + c] as f64;
                    count += 1;
                }
            }
            let correlation = dot / count as f64 * 3.0;
            assert!(
                correlation.abs() < 0.03,
                "shift {}: correlation {}",
                shift,
                correlation
            );
        }
    }

    #[test]
    fn both_covariance_plans_agree() {
        let (n, d) = (700usize, 300usize);
        let x: Vec<f32> = (0..n * d)
            .map(|i| ((i as f32 * 0.13).sin() * 2.0) + ((i % 11) as f32) * 0.1)
            .collect();
        let mean: Vec<f64> = (0..d)
            .map(|c| (0..n).map(|r| x[r * d + c] as f64).sum::<f64>() / n as f64)
            .collect();
        let by_groups = thread_pool(3).install(|| gram_by_row_groups(&x, n, d, &mean, 5));
        let by_bands = thread_pool(3).install(|| gram_by_bands(&x, n, d, &mean, 64));
        for (a, b) in by_groups.iter().zip(&by_bands) {
            assert!((a - b).abs() <= 1e-9 * (1.0 + a.abs()), "{} vs {}", a, b);
        }
        // neither depends on the number of threads
        let single = thread_pool(1).install(|| gram_by_row_groups(&x, n, d, &mean, 5));
        assert_eq!(by_groups, single);
        let single = thread_pool(1).install(|| gram_by_bands(&x, n, d, &mean, 64));
        assert_eq!(by_bands, single);
    }

    #[test]
    fn covariance_plan_follows_the_memory_budget() {
        assert_eq!(gram_plan(100_000, 256), GramPlan::RowGroups(32));
        assert_eq!(gram_plan(13_471, 256), GramPlan::RowGroups(14));
        assert_eq!(gram_plan(500, 64), GramPlan::RowGroups(1));
        // one matrix per group would exceed the input itself here
        assert_eq!(gram_plan(30_000, 1024), GramPlan::Bands(64));
        assert_eq!(gram_plan(100_000, 4096), GramPlan::Bands(128));
        assert_eq!(gram_plan(10_000_000, 4096), GramPlan::RowGroups(64));
    }

    #[test]
    fn covariance_matches_direct_computation() {
        let (n, d) = (3000usize, 7usize);
        let x: Vec<f32> = (0..n * d)
            .map(|i| ((i as f32 * 0.37).sin() * 3.0) + (i % d) as f32)
            .collect();
        let (mean, covariance) = thread_pool(3).install(|| column_mean_and_covariance(&x, n, d));
        let single = thread_pool(1).install(|| column_mean_and_covariance(&x, n, d));
        assert_eq!(mean, single.0);
        assert_eq!(covariance, single.1);
        for c in 0..d {
            let direct: f64 = (0..n).map(|r| x[r * d + c] as f64).sum::<f64>() / n as f64;
            assert!((mean[c] - direct).abs() < 1e-9);
        }
        for a in 0..d {
            for b in 0..d {
                let direct: f64 = (0..n)
                    .map(|r| (x[r * d + a] as f64 - mean[a]) * (x[r * d + b] as f64 - mean[b]))
                    .sum::<f64>()
                    / (n - 1) as f64;
                assert!((covariance[a * d + b] - direct).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn projection_variants_agree() {
        let (n, d) = (1300usize, 5usize);
        let x: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.11).cos()).collect();
        let mean: Vec<f32> = (0..d).map(|c| c as f32 * 0.1).collect();
        let transform = Array2::from_shape_fn((d, d), |(i, j)| ((i * 3 + j) as f32 * 0.7).sin());
        let mut out = vec![0f32; n * d];
        center_and_project(&x, d, &mean, transform.view(), &mut out);
        let mut inplace = x.clone();
        center_and_project_inplace(&mut inplace, d, &mean, transform.view());
        assert_eq!(out, inplace);
        for r in [0usize, 511, 512, 1299] {
            for j in 0..d {
                let direct: f32 = (0..d)
                    .map(|c| (x[r * d + c] - mean[c]) * transform[[c, j]])
                    .sum();
                assert!((out[r * d + j] - direct).abs() < 1e-4);
            }
        }
    }
}
