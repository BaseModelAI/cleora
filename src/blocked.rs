//! Classic Cleora on a column-blocked embedding matrix.
//!
//! Concern of this module: running the rounds of [`crate::embedding::embed`] -
//! propagate, normalise every row - with less memory traffic than the row-major
//! kernel, and with bit-identical results.
//!
//! Why. A round of the row-major kernel reads one whole source row per matrix entry
//! (`dim` floats: 4 KB at 1024 dimensions) from wherever in memory that row is. The
//! embedding matrix is far larger than the caches, so nearly all of it comes from
//! main memory, and the kernel runs at the speed of the memory bus. Here the matrix
//! is kept as blocks of [`BLOCK`] columns, each block one contiguous slab with a
//! cache line per entity, and a round handles one block after the other. All threads
//! read the same slab, which fits the caches or nearly does, and what is streamed
//! from memory per entry and block is its 8 bytes of index and weight.
//!
//! Same numbers. Every coordinate goes through the operations of the row-major
//! kernel in the same order ([`Block::segment`], [`Block::rows`]), so a result does
//! not depend on which of the two computed it; a test pins that. Rows are normalised
//! after every round as always, but the factor is kept per entity and applied when a
//! line is read - `value * factor` is the very multiplication the normalisation
//! performs - which saves a pass over the matrix per round.

use rayon::prelude::*;

use crate::embedding::{balanced_tasks, prefetch, split_tasks, MarkovType, HUB_NNZ, TASK_NNZ};
use crate::sparse_matrix::SparseMatrix;

/// Columns per block: one cache line of `f32`.
const BLOCK: usize = 16;

/// Below this many rounds the conversion at the end costs more than the blocks save.
pub const MIN_ROUNDS: usize = 2;

/// How many entries ahead a source line is requested. The lines of a slab are read in
/// no order at all; asking early keeps as many requests in flight as the CPU allows.
const PREFETCH_DISTANCE: usize = 32;

/// Lines (or rows) one task of the passes that are not propagation handles.
const LINES_PER_TASK: usize = 4096;

/// [`BLOCK`] consecutive coordinates of one entity, on a cache line of their own.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Line([f32; BLOCK]);

/// A matrix in blocks: the line of block `b` and entity `i` is at `b * n + i`.
///
/// The memory is an ordinary zeroed vector of floats, which the allocator hands out
/// without writing to it (an over-aligned zeroed allocation would be cleared by one
/// thread, gigabyte after gigabyte); the lines start at its first cache-line boundary.
struct Lines {
    floats: Vec<f32>,
    len: usize,
}

impl Lines {
    fn zeroed(len: usize) -> Lines {
        Lines {
            floats: vec![0f32; (len + 1) * BLOCK],
            len,
        }
    }

    /// Floats before the first cache-line boundary of the allocation.
    fn skipped(&self) -> usize {
        let line = std::mem::align_of::<Line>();
        (line - self.floats.as_ptr() as usize % line) % line / std::mem::size_of::<f32>()
    }

    fn get(&self) -> &[Line] {
        // Safety: `skipped() < BLOCK` floats are passed over and the vector holds one line
        // more than `len`, so the lines are inside the allocation; they are aligned by
        // construction, and any bit pattern is a valid `Line`.
        unsafe {
            std::slice::from_raw_parts(
                self.floats.as_ptr().add(self.skipped()) as *const Line,
                self.len,
            )
        }
    }

    fn get_mut(&mut self) -> &mut [Line] {
        // Safety: as in `get`; the borrow of `self` is exclusive.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.floats.as_mut_ptr().add(self.skipped()) as *mut Line,
                self.len,
            )
        }
    }
}

/// The matrix whose coordinates are `initial(entity, column)`. The last block is
/// padded with zeros, which stay zeros.
fn initial_blocks<I>(n: usize, dim: usize, initial: &I) -> Lines
where
    I: Fn(usize, usize) -> f32 + Sync,
{
    let mut lines = Lines::zeroed(dim.div_ceil(BLOCK) * n);
    lines
        .get_mut()
        .par_iter_mut()
        .enumerate()
        .with_min_len(LINES_PER_TASK)
        .for_each(|(index, line)| {
            let (first, entity) = (index / n * BLOCK, index % n);
            for (offset, value) in line.0.iter_mut().take(dim - first).enumerate() {
                *value = initial(entity, first + offset);
            }
        });
    lines
}

/// The blocks as a row-major `n x dim` matrix, every row multiplied by its factor.
fn to_rows(lines: &[Line], factor: &[f32], n: usize, dim: usize) -> Vec<f32> {
    let mut x = vec![0f32; n * dim];
    x.par_chunks_mut(dim)
        .zip(factor)
        .enumerate()
        .with_min_len(LINES_PER_TASK / BLOCK)
        .for_each(|(entity, (row, &factor))| {
            for (block, part) in row.chunks_mut(BLOCK).enumerate() {
                let line = &lines[block * n + entity].0;
                for (value, source) in part.iter_mut().zip(line) {
                    *value = *source * factor;
                }
            }
        });
    x
}

/// What one round computes for one block.
#[derive(Clone, Copy)]
struct Block<'a> {
    graph: &'a SparseMatrix,
    symmetric: bool,
    /// The lines of this block after the previous round, not yet normalised.
    source: &'a [Line],
    /// What the previous round's normalisation multiplies every row by.
    factor: &'a [f32],
}

impl Block<'_> {
    /// Entry `k` as the kernel uses it: its factor, the normalisation factor of its
    /// column and the column's line.
    #[inline(always)]
    fn entry(&self, k: usize) -> (f32, f32, &[f32; BLOCK]) {
        let column = self.graph.indices[k] as usize;
        (
            self.graph.entry_factor(k, self.symmetric),
            self.factor[column],
            &self.source[column].0,
        )
    }

    #[inline(always)]
    fn prefetch_line(&self, k: usize) {
        if let Some(&column) = self.graph.indices.get(k) {
            prefetch(self.source.as_ptr().wrapping_add(column as usize) as *const f32);
        }
    }

    /// The sum over the entries `first..last` of `entry factor * (line * column factor)`.
    ///
    /// Four entries are added per pass, but every coordinate still receives its terms
    /// one after another in entry order: the result is the one of adding them singly.
    #[inline(always)]
    fn segment(&self, first: usize, last: usize) -> [f32; BLOCK] {
        let mut sum = [0f32; BLOCK];
        let mut k = first;
        while k + 4 <= last {
            for ahead in 0..4 {
                self.prefetch_line(k + PREFETCH_DISTANCE + ahead);
            }
            let (c0, f0, l0) = self.entry(k);
            let (c1, f1, l1) = self.entry(k + 1);
            let (c2, f2, l2) = self.entry(k + 2);
            let (c3, f3, l3) = self.entry(k + 3);
            for i in 0..BLOCK {
                let mut value = sum[i];
                value += c0 * (l0[i] * f0);
                value += c1 * (l1[i] * f1);
                value += c2 * (l2[i] * f2);
                value += c3 * (l3[i] * f3);
                sum[i] = value;
            }
            k += 4;
        }
        while k < last {
            let (coefficient, factor, line) = self.entry(k);
            for (sum, value) in sum.iter_mut().zip(line) {
                *sum += coefficient * (*value * factor);
            }
            k += 1;
        }
        sum
    }

    /// The lines of the rows `first_row..` of the product into `out`, their squares
    /// added to `squares` (one running sum per row, continued from block to block in
    /// coordinate order: what the row normalisation sums).
    #[inline(always)]
    fn rows(&self, first_row: usize, out: &mut [Line], squares: &mut [f32]) {
        for (offset, (line, square)) in out.iter_mut().zip(squares).enumerate() {
            let row = first_row + offset;
            let entries = self.graph.row_range(row);
            let sum = if entries.len() > HUB_NNZ {
                // as the row-major kernel sums such a row: in segments, combined in order
                let mut total = [0f32; BLOCK];
                for first in entries.clone().step_by(TASK_NNZ) {
                    let part = self.segment(first, (first + TASK_NNZ).min(entries.end));
                    for (total, part) in total.iter_mut().zip(part) {
                        *total += part;
                    }
                }
                total
            } else {
                self.segment(entries.start, entries.end)
            };
            let scale = self.graph.row_factor(row, self.symmetric);
            let mut running = *square;
            for (target, sum) in line.0.iter_mut().zip(sum) {
                let value = sum * scale;
                *target = value;
                running += value * value;
            }
            *square = running;
        }
    }
}

/// [`Block::rows`], compiled for AVX2 when the CPU has it. No fused multiply-add is
/// used, so both versions produce bit-identical results.
fn rows_dispatch(block: &Block<'_>, first_row: usize, out: &mut [Line], squares: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx2")]
        unsafe fn avx2(block: &Block<'_>, first_row: usize, out: &mut [Line], squares: &mut [f32]) {
            block.rows(first_row, out, squares)
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            // Safety: AVX2 availability was just checked.
            return unsafe { avx2(block, first_row, out, squares) };
        }
    }
    block.rows(first_row, out, squares)
}

/// `rounds` rounds of propagation and row normalisation of the `n x dim` matrix whose
/// coordinates are `initial(entity, column)`: exactly what
/// [`crate::embedding::iterate`] leaves in its buffer, as a row-major matrix.
/// Runs on the current rayon pool; needs two matrices of memory, as the row-major loop.
pub fn embed<I>(
    graph: &SparseMatrix,
    dim: usize,
    kind: MarkovType,
    rounds: usize,
    initial: I,
) -> Vec<f32>
where
    I: Fn(usize, usize) -> f32 + Sync,
{
    let n = graph.num_entities();
    if n == 0 || dim == 0 {
        return Vec::new();
    }
    let mut current = initial_blocks(n, dim, &initial);
    let mut next = Lines::zeroed(current.len);
    let mut factor = vec![1f32; n];
    let mut squares = vec![0f32; n];
    let bounds = balanced_tasks((0..n).map(|row| graph.row_range(row).len()));

    for _ in 0..rounds {
        squares.fill(0.0);
        for (source, out) in current
            .get()
            .chunks_exact(n)
            .zip(next.get_mut().chunks_exact_mut(n))
        {
            let block = Block {
                graph,
                symmetric: kind == MarkovType::Symmetric,
                source,
                factor: &factor,
            };
            split_tasks(out, &bounds, 1)
                .into_par_iter()
                .zip(split_tasks(&mut squares, &bounds, 1))
                .zip(bounds.par_iter())
                .with_max_len(1)
                .for_each(|((out, squares), &first_row)| {
                    rows_dispatch(&block, first_row, out, squares)
                });
        }
        for (factor, square) in factor.iter_mut().zip(&squares) {
            *factor = 1.0 / square.sqrt().max(1e-10);
        }
        std::mem::swap(&mut current, &mut next);
    }
    drop(next);
    to_rows(current.get(), &factor, n, dim)
}
