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
///
/// A forked child inherits this table but not the worker threads behind it: handing
/// out one of the parent's pools there would wait forever for threads that do not
/// exist. The table therefore remembers which process filled it and starts afresh in
/// another one.
pub fn thread_pool(num_threads: usize) -> Arc<ThreadPool> {
    type Pools = (u32, HashMap<usize, Arc<ThreadPool>>);
    static POOLS: OnceLock<Mutex<Pools>> = OnceLock::new();
    let num_threads = num_threads.max(1);
    let process = std::process::id();
    let mut guard = POOLS
        .get_or_init(|| Mutex::new((process, HashMap::new())))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if guard.0 != process {
        // The parent's pools are never dropped here: their threads are not ours to stop.
        std::mem::forget(std::mem::take(&mut guard.1));
        guard.0 = process;
    }
    guard
        .1
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

/// Rows handled by one parallel task of the dense row-wise kernels (normalisation,
/// projection): there every row costs the same.
const ROWS_PER_TASK: usize = 64;

/// Matrix entries one propagation task handles. Propagation work is proportional to the
/// number of entries, not rows, and entities are numbered by first appearance, which
/// puts the busiest ones next to each other at the start: tasks of equal *row* counts
/// would leave most of the work to the one thread that got the first rows.
pub(crate) const TASK_NNZ: usize = 4096;
/// Upper bound on the rows of one propagation task (rows cost something even when empty).
const TASK_ROWS: usize = 1024;
/// A row with more entries than this is a task of its own, summed in parallel in
/// segments of `TASK_NNZ` entries that are combined in order. The segmentation depends
/// on the row only, so the result is the same for any number of threads.
pub(crate) const HUB_NNZ: usize = 4 * TASK_NNZ;
/// How many entries ahead the first cache line of a source row is requested.
const PREFETCH_DISTANCE: usize = 4;

/// What one propagation computes: `M * x` for the left or symmetric Markov matrix.
#[derive(Clone, Copy)]
struct Product<'a> {
    graph: &'a SparseMatrix,
    x: &'a [f32],
    dim: usize,
    symmetric: bool,
}

#[inline(always)]
fn prefetch(pointer: *const f32) {
    #[cfg(target_arch = "x86_64")]
    // Safety: prefetching is a hint; it never faults, whatever the address.
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(pointer as *const i8);
    }
    #[cfg(target_arch = "aarch64")]
    // Safety: as above (`prfm` is a hint).
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) pointer, options(nostack, preserves_flags, readonly));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = pointer;
}

impl<'a> Product<'a> {
    #[inline(always)]
    fn coefficient(&self, k: usize) -> f32 {
        self.graph.entry_factor(k, self.symmetric)
    }

    #[inline(always)]
    fn source(&self, k: usize) -> &'a [f32] {
        let column = self.graph.indices[k] as usize;
        &self.x[column * self.dim..(column + 1) * self.dim]
    }

    #[inline(always)]
    fn prefetch_source(&self, k: usize) {
        if let Some(&column) = self.graph.indices.get(k) {
            prefetch(self.x.as_ptr().wrapping_add(column as usize * self.dim));
        }
    }

    /// `dst += sum over entries k in [first, last) of coefficient(k) * source(k)`.
    ///
    /// Four source rows are added per pass over `dst`. Every element still receives
    /// its terms one after another in entry order, so the result is bit-identical to
    /// adding the rows one at a time; the passes over `dst` are fewer and four streams
    /// of source rows are in flight at once.
    #[inline(always)]
    fn accumulate(&self, first: usize, last: usize, dst: &mut [f32]) {
        let mut k = first;
        while k + 4 <= last {
            for ahead in 0..4 {
                self.prefetch_source(k + PREFETCH_DISTANCE + ahead);
            }
            let (c0, c1, c2, c3) = (
                self.coefficient(k),
                self.coefficient(k + 1),
                self.coefficient(k + 2),
                self.coefficient(k + 3),
            );
            let (s0, s1, s2, s3) = (
                self.source(k),
                self.source(k + 1),
                self.source(k + 2),
                self.source(k + 3),
            );
            for i in 0..dst.len() {
                let mut value = dst[i];
                value += c0 * s0[i];
                value += c1 * s1[i];
                value += c2 * s2[i];
                value += c3 * s3[i];
                dst[i] = value;
            }
            k += 4;
        }
        while k < last {
            self.prefetch_source(k + PREFETCH_DISTANCE);
            let coefficient = self.coefficient(k);
            for (d, s) in dst.iter_mut().zip(self.source(k)) {
                *d += coefficient * *s;
            }
            k += 1;
        }
    }

    /// Row `row` of the product, written to `dst`.
    #[inline(always)]
    fn row(&self, row: usize, dst: &mut [f32]) {
        let entries = self.graph.row_range(row);
        dst.fill(0.0);
        if entries.len() > HUB_NNZ {
            let segments: Vec<usize> = entries.clone().step_by(TASK_NNZ).collect();
            let partial: Vec<Vec<f32>> = segments
                .par_iter()
                .with_max_len(1)
                .map(|&first| {
                    let mut sum = vec![0f32; self.dim];
                    accumulate_dispatch(self, first, (first + TASK_NNZ).min(entries.end), &mut sum);
                    sum
                })
                .collect();
            for sum in &partial {
                for (d, s) in dst.iter_mut().zip(sum) {
                    *d += *s;
                }
            }
        } else {
            self.accumulate(entries.start, entries.end, dst);
        }
        let scale = self.graph.row_factor(row, self.symmetric);
        for d in dst.iter_mut() {
            *d *= scale;
        }
    }

    /// The rows of one task, each followed by `finish(row, values)` while it is still
    /// in the cache.
    #[inline(always)]
    fn rows<F: Fn(usize, &mut [f32])>(&self, rows: Rows<'_>, chunk: &mut [f32], finish: &F) {
        for (offset, dst) in chunk.chunks_exact_mut(self.dim).enumerate() {
            let row = match rows {
                Rows::From(first_row) => first_row + offset,
                Rows::List(list) => list[offset],
            };
            self.row(row, dst);
            finish(row, dst);
        }
    }
}

/// The rows one task computes: consecutive ones or an explicit list.
#[derive(Clone, Copy)]
enum Rows<'a> {
    From(usize),
    List(&'a [usize]),
}

/// [`Product::accumulate`], compiled for AVX2 when the CPU has it. No fused
/// multiply-add is used, so both versions produce bit-identical results.
fn accumulate_dispatch(product: &Product<'_>, first: usize, last: usize, dst: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx2")]
        unsafe fn avx2(product: &Product<'_>, first: usize, last: usize, dst: &mut [f32]) {
            product.accumulate(first, last, dst)
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            // Safety: AVX2 availability was just checked.
            return unsafe { avx2(product, first, last, dst) };
        }
    }
    product.accumulate(first, last, dst)
}

/// [`Product::rows`], compiled for AVX2 when the CPU has it (see [`accumulate_dispatch`]).
fn rows_dispatch<F: Fn(usize, &mut [f32])>(
    product: &Product<'_>,
    rows: Rows<'_>,
    chunk: &mut [f32],
    finish: &F,
) {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx2")]
        unsafe fn avx2<F: Fn(usize, &mut [f32])>(
            product: &Product<'_>,
            rows: Rows<'_>,
            chunk: &mut [f32],
            finish: &F,
        ) {
            product.rows(rows, chunk, finish)
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            // Safety: AVX2 availability was just checked.
            return unsafe { avx2(product, rows, chunk, finish) };
        }
    }
    product.rows(rows, chunk, finish)
}

/// Groups a sequence of rows, given by their numbers of entries, into consecutive
/// tasks of about `TASK_NNZ` entries (see the constants above). Returns the position
/// of the first row of every task, followed by the number of rows.
pub(crate) fn balanced_tasks(row_entries: impl Iterator<Item = usize>) -> Vec<usize> {
    let mut bounds = vec![0usize];
    let mut entries = 0usize;
    let mut count = 0usize;
    for (position, row_entries) in row_entries.enumerate() {
        let first = *bounds.last().expect("never empty");
        let hub = row_entries > HUB_NNZ;
        let full = entries + row_entries > TASK_NNZ || position - first >= TASK_ROWS;
        if position > first && (hub || full) {
            bounds.push(position);
            entries = 0;
        }
        entries += row_entries;
        if hub {
            bounds.push(position + 1);
            entries = 0;
        }
        count = position + 1;
    }
    if *bounds.last().expect("never empty") != count {
        bounds.push(count);
    }
    bounds
}

/// Cuts `data`, which holds `width` elements per row, into the pieces of the tasks given
/// by `bounds` (as [`balanced_tasks`] returns them).
pub(crate) fn split_tasks<'a, T>(
    data: &'a mut [T],
    bounds: &[usize],
    width: usize,
) -> Vec<&'a mut [T]> {
    let mut pieces = Vec::with_capacity(bounds.len() - 1);
    let mut rest = data;
    for task in bounds.windows(2) {
        let (piece, tail) = rest.split_at_mut((task[1] - task[0]) * width);
        pieces.push(piece);
        rest = tail;
    }
    pieces
}

/// Computes the rows `rows` of the product into `out` (one row of `out` per entry of
/// `rows`), in parallel tasks of about equal cost.
fn run_product<F>(product: &Product<'_>, rows: Rows<'_>, out: &mut [f32], finish: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    let dim = product.dim;
    let row_at = |position: usize| match rows {
        Rows::From(first_row) => first_row + position,
        Rows::List(list) => list[position],
    };
    let count = out.len() / dim;
    let bounds = balanced_tasks((0..count).map(|p| product.graph.row_range(row_at(p)).len()));
    // Every task may be stolen on its own: they are about equally expensive.
    split_tasks(out, &bounds, dim)
        .into_par_iter()
        .zip(bounds.par_windows(2))
        .with_max_len(1)
        .for_each(|(chunk, task)| {
            let task_rows = match rows {
                Rows::From(first_row) => Rows::From(first_row + task[0]),
                Rows::List(list) => Rows::List(&list[task[0]..task[1]]),
            };
            rows_dispatch(product, task_rows, chunk, &finish)
        });
}

/// `out = M * x` for the row-major `n x dim` matrices `x` and `out`, where `M` is
/// the left or symmetric Markov matrix of the graph.
/// Runs on the current rayon pool.
pub fn propagate(graph: &SparseMatrix, x: &[f32], out: &mut [f32], dim: usize, kind: MarkovType) {
    propagate_then(graph, x, out, dim, kind, |_, _| {});
}

/// [`propagate`], with `finish(row, values)` applied to every row of the result as
/// soon as it is computed, i.e. while it is still in the cache.
pub fn propagate_then<F>(
    graph: &SparseMatrix,
    x: &[f32],
    out: &mut [f32],
    dim: usize,
    kind: MarkovType,
    finish: F,
) where
    F: Fn(usize, &mut [f32]) + Sync,
{
    let n = graph.num_entities();
    assert_eq!(x.len(), n * dim, "input has the wrong size");
    assert_eq!(out.len(), n * dim, "output has the wrong size");
    if n == 0 || dim == 0 {
        return;
    }
    let product = Product {
        graph,
        x,
        dim,
        symmetric: kind == MarkovType::Symmetric,
    };
    run_product(&product, Rows::From(0), out, finish);
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
    let product = Product {
        graph,
        x,
        dim,
        symmetric: kind == MarkovType::Symmetric,
    };
    run_product(&product, Rows::List(rows), out, |_, _| {});
}

/// Scales one row to unit L2 norm (a row of zeros stays zero).
#[inline(always)]
pub fn l2_normalize_row(row: &mut [f32]) {
    let mut sum_sq: f32 = 0.0;
    for v in row.iter() {
        sum_sq += v * v;
    }
    let inv_norm = 1.0 / sum_sq.sqrt().max(1e-10);
    for v in row.iter_mut() {
        *v *= inv_norm;
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
                l2_normalize_row(row);
            }
        });
}

/// `dst = (1 - residual_weight) * dst + residual_weight * src`
#[inline(always)]
fn mix_residual_row(dst: &mut [f32], src: &[f32], residual_weight: f32) {
    let alpha = 1.0 - residual_weight;
    for (d, s) in dst.iter_mut().zip(src) {
        *d = alpha * *d + residual_weight * *s;
    }
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

/// Classic Cleora on a matrix in place: `num_iterations` rounds of propagation followed
/// by row-wise L2 normalisation of `current`, using `scratch` (same size) as the second
/// buffer. Returns the number of rounds performed.
/// With `convergence_threshold > 0` iteration stops once the RMSE between two
/// consecutive rounds drops below the threshold.
#[allow(clippy::too_many_arguments)]
pub fn iterate(
    graph: &SparseMatrix,
    current: &mut [f32],
    scratch: &mut [f32],
    dim: usize,
    kind: MarkovType,
    num_iterations: usize,
    residual_weight: f32,
    convergence_threshold: f32,
) -> usize {
    let use_residual = residual_weight > 0.0 && residual_weight < 1.0;
    let check_convergence = convergence_threshold > 0.0;
    // the two buffers take turns; `src` always holds the latest round
    let (mut src, mut dst) = (current, scratch);
    let mut in_scratch = false;
    let mut rounds = num_iterations;

    for iteration in 0..num_iterations {
        let previous: &[f32] = src;
        propagate_then(graph, previous, dst, dim, kind, |row, values| {
            if use_residual {
                mix_residual_row(
                    values,
                    &previous[row * dim..(row + 1) * dim],
                    residual_weight,
                );
            }
            l2_normalize_row(values);
        });

        let converged =
            check_convergence && iteration > 0 && rmse(dst, src) < convergence_threshold as f64;
        std::mem::swap(&mut src, &mut dst);
        in_scratch = !in_scratch;
        if converged {
            rounds = iteration + 1;
            break;
        }
    }
    if in_scratch {
        // `src` is the scratch buffer: the result belongs in `current`
        dst.par_chunks_mut(1 << 16)
            .zip(src.par_chunks(1 << 16))
            .for_each(|(target, source)| target.copy_from_slice(source));
    }
    rounds
}

/// What the vectors will be used for; see [`suggested_iterations`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    /// Nearest neighbours, link ranking, recommendation: local structure matters.
    Similarity,
    /// One embedding for several uses.
    General,
    /// Features for a classifier: the wider neighbourhood matters.
    Classification,
}

impl Purpose {
    /// Random-walk steps for a graph whose entities have one neighbour on average.
    fn steps(self) -> f64 {
        match self {
            Purpose::Similarity => 12.0,
            Purpose::General => 16.0,
            Purpose::Classification => 24.0,
        }
    }
}

/// The symmetric normalisation smooths more slowly and gets this many times more steps.
const AUTO_SYMMETRIC_FACTOR: f64 = 1.5;
/// With fewer steps than this, vectors of very dense graphs are still mostly noise.
const AUTO_MIN_ITERATIONS: f64 = 2.0;
/// Share of an entity's own initial (random) vector that may be left after the last
/// round. A diagonal weight share `s` leaves `s^rounds` of it.
const AUTO_SELF_REMAINDER: f64 = 1.0 / 256.0;
/// Largest number of rounds the rule was validated for.
const AUTO_MAX_ITERATIONS: usize = 64;

/// Number of rounds for [`embed`] when there are no labels to choose it with:
///
/// ```text
/// mixing = steps(purpose) / ((1 - s) * sqrt(mean neighbours))        (x 1.5 symmetric)
/// forget = ln(1/256) / ln(s)                                          (0 when s = 0)
/// rounds = max(mixing, forget, 2), at most 64
/// ```
///
/// where `s` is the share of the weight that lies on the diagonal.
///
/// Every round moves the vectors one step along a random walk. Too few rounds and
/// only the closest neighbourhood is visible; too many and all vectors converge to
/// the same direction. Denser graphs mix faster, hence the square root of the mean
/// number of neighbours (`mixing`). A matrix with weight on its diagonal - a reflexive
/// column - describes a lazy walk: it only moves with probability `1 - s`, and every
/// vector keeps a share `s` of what it was. `mixing` is scaled accordingly, and
/// `forget` is the number of rounds after which the entity's own random initial
/// vector has decayed to 1/256: 8 rounds for a reflexive column of two-entity
/// hyperedges, however dense the graph.
///
/// The form and the constants were fitted leave-one-graph-out on iteration sweeps
/// over graphs with 3 to 500 neighbours per node (`benchmarks/README.md`, "Number of
/// iterations").
pub fn suggested_iterations(graph: &SparseMatrix, kind: MarkovType, purpose: Purpose) -> usize {
    let staying = graph.self_weight_share();
    if staying >= 1.0 {
        // nothing but self-loops: further rounds change nothing
        return 1;
    }
    let factor = match kind {
        MarkovType::Left => 1.0,
        MarkovType::Symmetric => AUTO_SYMMETRIC_FACTOR,
    };
    let mixing =
        purpose.steps() * factor / ((1.0 - staying) * graph.mean_neighbours().max(1.0).sqrt());
    let forget = if staying > 0.0 {
        AUTO_SELF_REMAINDER.ln() / staying.ln()
    } else {
        0.0
    };
    let rounds = mixing.max(forget).max(AUTO_MIN_ITERATIONS);
    (rounds.round() as usize).clamp(1, AUTO_MAX_ITERATIONS)
}

/// Classic Cleora from the given initial vectors. Returns the embeddings and the
/// number of rounds performed (see [`iterate`]).
pub fn embed(
    graph: &SparseMatrix,
    initial: Vec<f32>,
    dim: usize,
    kind: MarkovType,
    num_iterations: usize,
    residual_weight: f32,
    convergence_threshold: f32,
) -> (Vec<f32>, usize) {
    let mut current = initial;
    let mut scratch = vec![0f32; current.len()];
    let rounds = iterate(
        graph,
        &mut current,
        &mut scratch,
        dim,
        kind,
        num_iterations,
        residual_weight,
        convergence_threshold,
    );
    (current, rounds)
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

    fn graph_of(columns: &str, lines: &[String]) -> SparseMatrix {
        let config = crate::configuration::Configuration::new(columns, 16, Some(2)).unwrap();
        crate::pipeline::build_graph_from_lines(&config, lines, None)
            .unwrap()
            .0
    }

    fn ring(nodes: usize, separator: &str) -> Vec<String> {
        (0..nodes)
            .map(|i| format!("v{}{}v{}", i, separator, (i + 1) % nodes))
            .collect()
    }

    fn clique(nodes: usize, separator: &str) -> Vec<String> {
        (0..nodes)
            .flat_map(|i| (i + 1..nodes).map(move |j| format!("v{}{}v{}", i, separator, j)))
            .collect()
    }

    #[test]
    fn suggested_iterations_follow_density_purpose_and_normalisation() {
        // a ring: every entity has exactly two neighbours
        let pair = graph_of("a b", &ring(50, "\t"));
        assert_eq!(pair.mean_neighbours(), 2.0);
        assert_eq!(pair.self_weight_share(), 0.0);
        let rounds = |purpose| suggested_iterations(&pair, MarkovType::Left, purpose);
        assert_eq!(rounds(Purpose::Similarity), 8); // 12 / sqrt(2) = 8.5
        assert_eq!(rounds(Purpose::General), 11); // 16 / sqrt(2) = 11.3
        assert_eq!(rounds(Purpose::Classification), 17); // 24 / sqrt(2) = 17.0
                                                         // symmetric: x 1.5
        assert_eq!(
            suggested_iterations(&pair, MarkovType::Symmetric, Purpose::General),
            17
        );
        // a clique of 26: 25 neighbours each, 16 / 5 = 3.2
        let dense = graph_of("a b", &clique(26, "\t"));
        assert_eq!(
            suggested_iterations(&dense, MarkovType::Left, Purpose::General),
            3
        );
    }

    #[test]
    fn suggested_iterations_of_a_lazy_walk() {
        // the ring as a reflexive column: half of every row is the self-loop
        let lazy = graph_of("complex::reflexive::v", &ring(50, " "));
        assert_eq!(lazy.mean_neighbours(), 2.0);
        assert!((lazy.self_weight_share() - 0.5).abs() < 1e-12);
        // twice the rounds of the plain ring: 2 * 16 / sqrt(2) = 22.6
        assert_eq!(
            suggested_iterations(&lazy, MarkovType::Left, Purpose::General),
            23
        );
        // a dense lazy walk still needs 8 rounds to forget the initial vectors (0.5^8 = 1/256)
        let dense = graph_of("complex::reflexive::v", &clique(40, " "));
        for purpose in [
            Purpose::Similarity,
            Purpose::General,
            Purpose::Classification,
        ] {
            assert_eq!(suggested_iterations(&dense, MarkovType::Left, purpose), 8);
        }
    }

    #[test]
    fn suggested_iterations_stay_in_range_on_degenerate_graphs() {
        let general =
            |graph: &SparseMatrix| suggested_iterations(graph, MarkovType::Left, Purpose::General);
        assert_eq!(general(&SparseMatrix::empty()), 16);
        // one hyperedge with a single entity: only a self-loop
        let alone = graph_of("complex::reflexive::v", &["x".to_string()]);
        assert_eq!(alone.self_weight_share(), 1.0);
        assert_eq!(general(&alone), 1);
        // a very dense plain graph gets the minimum of two rounds
        let dense = graph_of("a b", &clique(100, "\t"));
        assert_eq!(general(&dense), 2);
        assert_eq!(
            suggested_iterations(&dense, MarkovType::Symmetric, Purpose::Classification),
            4 // 24 * 1.5 / sqrt(99) = 3.6
        );
    }

    /// A star whose centre has more entries than `HUB_NNZ`, plus a ring among the leaves.
    fn hub_graph(leaves: usize) -> SparseMatrix {
        let mut lines: Vec<String> = (0..leaves).map(|i| format!("hub leaf{}", i)).collect();
        lines.extend((0..leaves).map(|i| format!("leaf{} leaf{}", i, (i + 1) % leaves)));
        let config =
            crate::configuration::Configuration::new("complex::reflexive::v", 16, Some(2)).unwrap();
        crate::pipeline::build_graph_from_lines(&config, &lines, None)
            .unwrap()
            .0
    }

    #[test]
    fn tasks_cover_all_rows_and_isolate_hubs() {
        let entries = [10usize, 5000, 3, HUB_NNZ + 1, 7, 7, TASK_NNZ, 1];
        let bounds = balanced_tasks(entries.iter().copied());
        assert_eq!(bounds.first(), Some(&0));
        assert_eq!(bounds.last(), Some(&entries.len()));
        assert!(bounds.windows(2).all(|w| w[0] < w[1]));
        // the hub (position 3) is a task of its own
        assert!(bounds.windows(2).any(|w| w == [3, 4]));
        // no other task exceeds the budget unless it is a single row
        for w in bounds.windows(2) {
            let total: usize = entries[w[0]..w[1]].iter().sum();
            assert!(
                total <= TASK_NNZ || w[1] - w[0] == 1,
                "{:?} holds {}",
                w,
                total
            );
        }
        assert_eq!(balanced_tasks(std::iter::empty()), vec![0]);
        // many empty rows are still split up
        let bounds = balanced_tasks(std::iter::repeat_n(0, 3 * TASK_ROWS + 5));
        assert_eq!(bounds.len(), 5);
    }

    /// Rows that are summed in segments (hubs) must not depend on the thread count,
    /// must agree with the plain definition of the product, and must come out the
    /// same through `propagate_selected`.
    #[test]
    fn hub_rows_are_exact_and_thread_independent() {
        let leaves = HUB_NNZ + 300;
        let graph = hub_graph(leaves);
        let n = graph.num_entities();
        let hub = graph.index_of("hub").unwrap();
        assert!(graph.row_range(hub).len() > HUB_NNZ);
        let dim = 5;
        let x = initialize(&graph.entity_ids, dim, 1);
        for kind in [MarkovType::Left, MarkovType::Symmetric] {
            let mut single = vec![0f32; n * dim];
            thread_pool(1).install(|| propagate(&graph, &x, &mut single, dim, kind));
            let mut many = vec![0f32; n * dim];
            thread_pool(7).install(|| propagate(&graph, &x, &mut many, dim, kind));
            assert_eq!(single, many);

            let normalization = match kind {
                MarkovType::Left => crate::sparse_matrix::Normalization::Left,
                MarkovType::Symmetric => crate::sparse_matrix::Normalization::Symmetric,
            };
            for row in [hub, 0, n - 1] {
                for column in 0..dim {
                    let direct: f64 = graph
                        .row_range(row)
                        .map(|k| {
                            graph.normalized_value(row, k, normalization) as f64
                                * x[graph.indices[k] as usize * dim + column] as f64
                        })
                        .sum();
                    let got = single[row * dim + column] as f64;
                    assert!(
                        (got - direct).abs() < 1e-5,
                        "row {}: {} vs {}",
                        row,
                        got,
                        direct
                    );
                }
            }

            let rows = [hub, 3, hub, n - 1];
            let mut selected = vec![f32::NAN; rows.len() * dim];
            thread_pool(3)
                .install(|| propagate_selected(&graph, &x, &rows, &mut selected, dim, kind));
            for (i, &row) in rows.iter().enumerate() {
                assert_eq!(
                    &selected[i * dim..(i + 1) * dim],
                    &single[row * dim..(row + 1) * dim]
                );
            }
        }
    }

    /// The fused pass must give exactly what separate passes give.
    #[test]
    fn finishing_rows_in_the_propagation_pass_changes_nothing() {
        let graph = ring_graph(500);
        let n = graph.num_entities();
        let dim = 12;
        let x = initialize(&graph.entity_ids, dim, 3);
        let mut separate = vec![0f32; n * dim];
        propagate(&graph, &x, &mut separate, dim, MarkovType::Left);
        l2_normalize_rows(&mut separate, dim);
        let mut fused = vec![0f32; n * dim];
        propagate_then(&graph, &x, &mut fused, dim, MarkovType::Left, |_, row| {
            l2_normalize_row(row)
        });
        assert_eq!(separate, fused);
    }

    /// The result of the in-place loop must end up in the caller's buffer whatever
    /// the number of rounds, and equal the rounds applied one by one.
    #[test]
    fn iterating_in_place_matches_separate_rounds() {
        let graph = ring_graph(200);
        let n = graph.num_entities();
        let dim = 6;
        for rounds in 0..5 {
            let mut expected = initialize(&graph.entity_ids, dim, 2);
            for _ in 0..rounds {
                let mut next = vec![0f32; n * dim];
                propagate(&graph, &expected, &mut next, dim, MarkovType::Symmetric);
                l2_normalize_rows(&mut next, dim);
                expected = next;
            }
            let mut current = initialize(&graph.entity_ids, dim, 2);
            let mut scratch = vec![f32::NAN; n * dim];
            let done = iterate(
                &graph,
                &mut current,
                &mut scratch,
                dim,
                MarkovType::Symmetric,
                rounds,
                0.0,
                0.0,
            );
            assert_eq!(done, rounds);
            assert_eq!(current, expected, "{} rounds", rounds);
        }
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
