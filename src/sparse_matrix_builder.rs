//! Turns hyperedges into the sparse co-occurrence matrix.
//!
//! Everything that is summed here (edge weights, row sums, occurrence counts) is
//! accumulated in integer arithmetic. Integer addition is associative and
//! commutative, so the result does not depend on how the input is split between
//! threads or in which order lines happen to be processed: the same input always
//! yields bit-identical matrices, for any number of workers.

use std::cmp::Reverse;
use std::collections::hash_map::Entry;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::mapref::entry::Entry as DashEntry;
use dashmap::DashMap;
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap};
use smallvec::SmallVec;

use crate::configuration::Configuration;
use crate::entity::{hash_entity, LineError, Tokens};
use crate::sparse_matrix::SparseMatrix;

/// Fixed-point scale of the accumulators: 64 fractional bits.
const FIXED_ONE: f64 = 18_446_744_073_709_551_616.0; // 2^64

#[inline(always)]
fn to_fixed(value: f64) -> u128 {
    // `value` is finite and non-negative; the cast saturates on overflow.
    (value * FIXED_ONE).round() as u128
}

#[inline(always)]
fn from_fixed(value: u128) -> f64 {
    (value as f64) / FIXED_ONE
}

/// Position of a token in the input: (file, line, token). Used to give entities the
/// index they would get when reading the input sequentially.
#[inline(always)]
pub fn position(file: u32, line: u64, token: u64) -> u64 {
    ((file as u64 & 0xFFF) << 52) | ((line & 0xF_FFFF_FFFF) << 16) | token.min(0xFFFF)
}

pub const MAX_FILES: usize = 1 << 12;
pub const MAX_LINES: u64 = 1 << 36;

struct EntityEntry {
    /// Smallest input position at which the entity was seen.
    pos: u64,
    column: u8,
    id: Box<str>,
    occurrence: AtomicU64,
    sum_lo: AtomicU64,
    sum_hi: AtomicU64,
}

impl EntityEntry {
    /// Exact, lock-free 128-bit accumulation. The final value is the plain integer
    /// sum of all increments, whatever the interleaving of threads.
    #[inline(always)]
    fn accumulate(&self, occurrence: u64, sum: u128) {
        self.occurrence.fetch_add(occurrence, Ordering::Relaxed);
        let lo = sum as u64;
        let hi = (sum >> 64) as u64;
        let previous = self.sum_lo.fetch_add(lo, Ordering::Relaxed);
        let carry = (previous.wrapping_add(lo) < previous) as u64;
        self.sum_hi
            .fetch_add(hi.wrapping_add(carry), Ordering::Relaxed);
    }
}

/// Concurrent registry of entities: name, first position, and row accumulators.
pub struct Indexer {
    map: DashMap<u64, EntityEntry, FxBuildHasher>,
}

/// Entities in their final order, with the per-entity sums.
pub struct IndexedEntities {
    pub entity_ids: Vec<String>,
    pub column_ids: Vec<u8>,
    pub occurrences: Vec<u64>,
    pub row_sums: Vec<f64>,
    pub key_2_index: FxHashMap<u64, u32>,
}

impl Default for Indexer {
    fn default() -> Self {
        Indexer {
            map: DashMap::with_hasher(FxBuildHasher),
        }
    }
}

impl Indexer {
    #[inline]
    pub fn observe(&self, hash: u64, id: &str, column: u8, pos: u64, occurrence: u64, sum: u128) {
        // Fast path: a shared lock is enough for entities that are already known.
        if let Some(entry) = self.map.get(&hash) {
            if entry.pos <= pos {
                entry.accumulate(occurrence, sum);
                return;
            }
        }
        match self.map.entry(hash) {
            DashEntry::Occupied(mut occupied) => {
                let entry = occupied.get_mut();
                if pos < entry.pos {
                    entry.pos = pos;
                    entry.column = column;
                }
                entry.accumulate(occurrence, sum);
            }
            DashEntry::Vacant(vacant) => {
                vacant.insert(EntityEntry {
                    pos,
                    column,
                    id: id.into(),
                    occurrence: AtomicU64::new(occurrence),
                    sum_lo: AtomicU64::new(sum as u64),
                    sum_hi: AtomicU64::new((sum >> 64) as u64),
                });
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Orders entities by first appearance in the input (ties: by hash).
    pub fn finish(self) -> Result<IndexedEntities, String> {
        let mut entries: Vec<(u64, u64, EntityEntry)> = self
            .map
            .into_iter()
            .map(|(hash, entry)| (entry.pos, hash, entry))
            .collect();
        if entries.len() > u32::MAX as usize {
            return Err(format!(
                "Too many distinct entities ({}); at most {} are supported",
                entries.len(),
                u32::MAX
            ));
        }
        entries.par_sort_unstable_by_key(|(pos, hash, _)| (*pos, *hash));

        let n = entries.len();
        let mut entity_ids = Vec::with_capacity(n);
        let mut column_ids = Vec::with_capacity(n);
        let mut occurrences = Vec::with_capacity(n);
        let mut row_sums = Vec::with_capacity(n);
        let mut key_2_index = FxHashMap::with_capacity_and_hasher(n, FxBuildHasher);
        for (index, (_, hash, entry)) in entries.into_iter().enumerate() {
            let sum =
                ((entry.sum_hi.into_inner() as u128) << 64) | entry.sum_lo.into_inner() as u128;
            entity_ids.push(String::from(entry.id));
            column_ids.push(entry.column);
            occurrences.push(entry.occurrence.into_inner());
            row_sums.push(from_fixed(sum));
            key_2_index.insert(hash, index as u32);
        }
        Ok(IndexedEntities {
            entity_ids,
            column_ids,
            occurrences,
            row_sums,
            key_2_index,
        })
    }
}

type PairMap = FxHashMap<(u64, u64), u128>;

/// A hyperedge relation too large to expand in full; expanded once global
/// occurrence counts are known.
struct Deferred {
    a: Box<[u64]>,
    /// `None`: reflexive relation (both sides are `a`).
    b: Option<Box<[u64]>>,
    value: u128,
}

/// A line that could not be used, kept for the error report.
#[derive(Debug, Clone)]
pub struct BadLine {
    pub pos: u64,
    pub text: String,
    pub error: LineError,
}

/// Summary of what was read.
#[derive(Debug, Clone, Default)]
pub struct BuildStats {
    pub lines_total: u64,
    pub lines_used: u64,
    pub lines_skipped: u64,
    pub first_bad_line: Option<BadLine>,
}

/// Per-thread accumulation state.
#[derive(Default)]
pub struct WorkerBuffer {
    pairs: PairMap,
    deferred: Vec<Deferred>,
    pub stats: BuildStats,
}

/// Adds the contribution of the ordered element `(x, y)` of a cartesian product.
///
/// The matrix is symmetric, so only one orientation of every pair is stored. An
/// element with `x != y` adds `value` to both `(x, y)` and `(y, x)`; an element with
/// `x == y` adds `value` twice to the diagonal entry.
#[inline(always)]
fn add_element(pairs: &mut PairMap, x: u64, y: u64, value: u128) {
    let (key, amount) = if x == y {
        ((x, x), value << 1)
    } else if x < y {
        ((x, y), value)
    } else {
        ((y, x), value)
    };
    match pairs.entry(key) {
        Entry::Occupied(mut e) => {
            let slot = e.get_mut();
            *slot = slot.saturating_add(amount);
        }
        Entry::Vacant(e) => {
            e.insert(amount);
        }
    }
}

#[inline]
fn add_cross(pairs: &mut PairMap, a: &[u64], b: &[u64], value: u128) {
    for &x in a {
        for &y in b {
            add_element(pairs, x, y, value);
        }
    }
}

/// Same result as `add_cross(pairs, a, a, value)` with half the lookups.
#[inline]
fn add_reflexive(pairs: &mut PairMap, a: &[u64], value: u128) {
    for i in 0..a.len() {
        add_element(pairs, a[i], a[i], value);
        for j in (i + 1)..a.len() {
            // the two ordered elements (i, j) and (j, i)
            if a[i] == a[j] {
                add_element(pairs, a[i], a[i], value << 1);
            } else {
                let key = if a[i] < a[j] {
                    (a[i], a[j])
                } else {
                    (a[j], a[i])
                };
                let amount = value << 1;
                match pairs.entry(key) {
                    Entry::Occupied(mut e) => {
                        let slot = e.get_mut();
                        *slot = slot.saturating_add(amount);
                    }
                    Entry::Vacant(e) => {
                        e.insert(amount);
                    }
                }
            }
        }
    }
}

impl WorkerBuffer {
    /// Records a line that could not be parsed.
    pub fn skip_line(&mut self, pos: u64, text: &str, error: LineError) {
        self.stats.lines_total += 1;
        self.stats.lines_skipped += 1;
        let replace = match &self.stats.first_bad_line {
            Some(bad) => pos < bad.pos,
            None => true,
        };
        if replace {
            let mut sample: String = text.chars().take(200).collect();
            if sample.len() < text.len() {
                sample.push('…');
            }
            self.stats.first_bad_line = Some(BadLine {
                pos,
                text: sample,
                error,
            });
        }
    }

    /// Adds one hyperedge. `fields[c]` holds the entities of entity column `c`;
    /// `pos` is the position of the line (see [`position`]).
    pub fn add_row(
        &mut self,
        config: &Configuration,
        indexer: &Indexer,
        fields: &[Tokens<'_>],
        weight: f32,
        pos: u64,
    ) {
        self.stats.lines_total += 1;
        if weight <= 0.0 {
            // A zero weight contributes nothing; the line is treated as absent.
            return;
        }
        self.stats.lines_used += 1;

        let mut hashes: SmallVec<[u64; 16]> = SmallVec::new();
        let mut ranges: SmallVec<[(usize, usize); 4]> = SmallVec::new();
        for tokens in fields {
            let start = hashes.len();
            hashes.extend(tokens.iter().map(|t| hash_entity(t)));
            ranges.push((start, hashes.len()));
        }

        let mut occurrence: SmallVec<[u64; 16]> = SmallVec::from_elem(0, hashes.len());
        let mut row_sum: SmallVec<[u128; 16]> = SmallVec::from_elem(0, hashes.len());
        let weight = weight as f64;
        let trim = config.hyperedge_trim_n;

        for relation in &config.relations {
            let (a0, a1) = ranges[relation.a as usize];
            let (b0, b1) = ranges[relation.b as usize];
            let (len_a, len_b) = (a1 - a0, b1 - b0);

            let share_a = to_fixed(weight / len_b as f64);
            let share_b = to_fixed(weight / len_a as f64);
            for i in a0..a1 {
                occurrence[i] += len_b as u64;
                row_sum[i] = row_sum[i].saturating_add(share_a);
            }
            for i in b0..b1 {
                occurrence[i] += len_a as u64;
                row_sum[i] = row_sum[i].saturating_add(share_b);
            }

            let value = to_fixed(weight / (len_a as f64 * len_b as f64));
            let reflexive = relation.a == relation.b;
            if len_a > trim && len_b > trim {
                self.deferred.push(Deferred {
                    a: hashes[a0..a1].into(),
                    b: if reflexive {
                        None
                    } else {
                        Some(hashes[b0..b1].into())
                    },
                    value,
                });
            } else if reflexive {
                add_reflexive(&mut self.pairs, &hashes[a0..a1], value);
            } else {
                add_cross(&mut self.pairs, &hashes[a0..a1], &hashes[b0..b1], value);
            }
        }

        let mut token_index = 0u64;
        for (column, tokens) in fields.iter().enumerate() {
            let start = ranges[column].0;
            for (k, token) in tokens.iter().enumerate() {
                let i = start + k;
                indexer.observe(
                    hashes[i],
                    token,
                    column as u8,
                    pos | token_index.min(0xFFFF),
                    occurrence[i],
                    row_sum[i],
                );
                token_index += 1;
            }
        }
    }
}

/// Keeps the `trim` entities with the highest global occurrence (ties: smaller hash)
/// in front and returns `(high, low)`.
fn split_high_low<'a>(
    nodes: &'a mut [u64],
    trim: usize,
    entities: &IndexedEntities,
) -> (&'a [u64], &'a [u64]) {
    if nodes.len() > trim {
        nodes.select_nth_unstable_by_key(trim, |hash| {
            let occurrence = entities.occurrences[entities.key_2_index[hash] as usize];
            (Reverse(occurrence), *hash)
        });
        nodes.split_at(trim)
    } else {
        (nodes, &[])
    }
}

/// Expands an oversized hyperedge relation: every pair that involves at least one
/// of the `trim` most frequent entities of either side is kept.
fn expand_deferred(
    deferred: &Deferred,
    trim: usize,
    entities: &IndexedEntities,
    pairs: &mut PairMap,
) {
    let mut a = deferred.a.to_vec();
    let (a_high, a_low) = split_high_low(&mut a, trim, entities);
    match &deferred.b {
        None => {
            add_cross(pairs, a_high, a_high, deferred.value);
            add_cross(pairs, a_high, a_low, deferred.value);
            add_cross(pairs, a_low, a_high, deferred.value);
        }
        Some(b) => {
            let mut b = b.to_vec();
            let (b_high, b_low) = split_high_low(&mut b, trim, entities);
            add_cross(pairs, a_high, b_high, deferred.value);
            add_cross(pairs, a_high, b_low, deferred.value);
            add_cross(pairs, a_low, b_high, deferred.value);
        }
    }
}

#[derive(Clone, Copy)]
struct PairEntry {
    /// `(row << 32) | column` with `row <= column`.
    key: u64,
    lo: u64,
    hi: u64,
}

impl PairEntry {
    #[inline(always)]
    fn value(&self) -> u128 {
        ((self.hi as u128) << 64) | self.lo as u128
    }
}

fn merge_stats(buffers: &[WorkerBuffer]) -> BuildStats {
    let mut stats = BuildStats::default();
    for buffer in buffers {
        stats.lines_total += buffer.stats.lines_total;
        stats.lines_used += buffer.stats.lines_used;
        stats.lines_skipped += buffer.stats.lines_skipped;
        if let Some(bad) = &buffer.stats.first_bad_line {
            let replace = match &stats.first_bad_line {
                Some(current) => bad.pos < current.pos,
                None => true,
            };
            if replace {
                stats.first_bad_line = Some(bad.clone());
            }
        }
    }
    stats
}

/// Combines the per-thread buffers into the final matrix.
/// Must be called from within the rayon pool that should do the work.
pub fn reduce(
    config: &Configuration,
    indexer: Indexer,
    mut buffers: Vec<WorkerBuffer>,
) -> Result<(SparseMatrix, BuildStats), String> {
    let stats = merge_stats(&buffers);
    let entities = indexer.finish()?;
    let n = entities.entity_ids.len();

    // Oversized hyperedges: expanded now that global occurrence counts are known.
    let deferred: Vec<Deferred> = buffers
        .iter_mut()
        .flat_map(|buffer| std::mem::take(&mut buffer.deferred))
        .collect();
    let mut pair_maps: Vec<PairMap> = buffers.into_iter().map(|buffer| buffer.pairs).collect();
    if !deferred.is_empty() {
        let trim = config.hyperedge_trim_n;
        let chunk = deferred
            .len()
            .div_ceil(rayon::current_num_threads().max(1))
            .max(1);
        let expanded: Vec<PairMap> = deferred
            .par_chunks(chunk)
            .map(|part| {
                let mut pairs = PairMap::default();
                for item in part {
                    expand_deferred(item, trim, &entities, &mut pairs);
                }
                pairs
            })
            .collect();
        pair_maps.extend(expanded);
    }
    drop(deferred);

    // Entity hashes -> matrix indices, one orientation per pair.
    let parts: Vec<Vec<PairEntry>> = pair_maps
        .into_par_iter()
        .map(|pairs| {
            pairs
                .into_iter()
                .map(|((x, y), value)| {
                    let i = entities.key_2_index[&x] as u64;
                    let j = entities.key_2_index[&y] as u64;
                    let (row, column) = if i <= j { (i, j) } else { (j, i) };
                    PairEntry {
                        key: (row << 32) | column,
                        lo: value as u64,
                        hi: (value >> 64) as u64,
                    }
                })
                .collect()
        })
        .collect();
    let total: usize = parts.iter().map(|p| p.len()).sum();
    let mut entries: Vec<PairEntry> = Vec::with_capacity(total);
    for part in parts {
        entries.extend_from_slice(&part);
    }
    entries.par_sort_unstable_by_key(|entry| entry.key);

    // Sum the partial values of every pair (exact, so the order is irrelevant).
    let mut unique = 0usize;
    for read in 0..entries.len() {
        if unique > 0 && entries[unique - 1].key == entries[read].key {
            let sum = entries[unique - 1]
                .value()
                .saturating_add(entries[read].value());
            entries[unique - 1].lo = sum as u64;
            entries[unique - 1].hi = (sum >> 64) as u64;
        } else {
            entries[unique] = entries[read];
            unique += 1;
        }
    }
    entries.truncate(unique);

    // Symmetric CSR. Scanning the pairs in (row, column) order visits, for every row r,
    // first all its entries below the diagonal (as mirrors of earlier rows, in
    // increasing column order) and then its entries from the diagonal onwards, so
    // the columns of every row come out sorted.
    let mut indptr = vec![0u64; n + 1];
    for entry in &entries {
        let row = (entry.key >> 32) as usize;
        let column = (entry.key & 0xFFFF_FFFF) as usize;
        indptr[row + 1] += 1;
        if row != column {
            indptr[column + 1] += 1;
        }
    }
    for i in 0..n {
        indptr[i + 1] += indptr[i];
    }
    let nnz = indptr[n] as usize;
    let mut indices = vec![0u32; nnz];
    let mut values = vec![0f32; nnz];
    let mut cursor: Vec<u64> = indptr[..n].to_vec();
    for entry in &entries {
        let row = (entry.key >> 32) as usize;
        let column = (entry.key & 0xFFFF_FFFF) as usize;
        let value = from_fixed(entry.value()) as f32;
        let at = cursor[row] as usize;
        indices[at] = column as u32;
        values[at] = value;
        cursor[row] += 1;
        if row != column {
            let at = cursor[column] as usize;
            indices[at] = row as u32;
            values[at] = value;
            cursor[column] += 1;
        }
    }

    let matrix = SparseMatrix::from_parts(
        config.entity_column_names.clone(),
        config.relations.clone(),
        entities.entity_ids,
        entities.column_ids,
        entities.row_sums,
        indptr,
        indices,
        values,
    );
    Ok((matrix, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_point_round_trip() {
        for v in [0.0, 1.0, 0.5, 1.0 / 3.0, 1e-9, 12345.678, 1e9] {
            let back = from_fixed(to_fixed(v));
            assert!((back - v).abs() <= v * 1e-15 + 1e-19, "{} -> {}", v, back);
        }
        // sums are exact whatever the order
        let parts = [
            to_fixed(1.0 / 3.0),
            to_fixed(1.0 / 7.0),
            to_fixed(1e-8),
            to_fixed(5.0),
        ];
        let forward: u128 = parts.iter().sum();
        let backward: u128 = parts.iter().rev().sum();
        assert_eq!(forward, backward);
    }

    #[test]
    fn positions_order_like_sequential_reading() {
        assert!(position(0, 5, 3) < position(0, 6, 0));
        assert!(position(0, u32::MAX as u64, 9) < position(1, 0, 0));
        assert!(position(2, 7, 1) < position(2, 7, 2));
        assert_eq!(position(0, 1, 70_000), position(0, 1, 65_535));
    }

    #[test]
    fn reflexive_shortcut_matches_cross_product() {
        let nodes = [5u64, 9, 5, 2, 7, 9, 9];
        let mut a = PairMap::default();
        let mut b = PairMap::default();
        add_cross(&mut a, &nodes, &nodes, 3);
        add_reflexive(&mut b, &nodes, 3);
        assert_eq!(a, b);
    }

    #[test]
    fn concurrent_row_accumulation_is_exact() {
        let indexer = Indexer::default();
        let big = (1u128 << 64) - 3; // forces carries into the high word
        std::thread::scope(|scope| {
            for t in 0..8u64 {
                let indexer = &indexer;
                scope.spawn(move || {
                    for k in 0..2000u64 {
                        indexer.observe(42, "x", 0, position(0, t * 10_000 + k, 0), 1, big);
                    }
                });
            }
        });
        let entry = indexer.map.get(&42).unwrap();
        let sum = ((entry.sum_hi.load(Ordering::Relaxed) as u128) << 64)
            | entry.sum_lo.load(Ordering::Relaxed) as u128;
        assert_eq!(sum, big * 16_000);
        assert_eq!(entry.occurrence.load(Ordering::Relaxed), 16_000);
        assert_eq!(entry.pos, position(0, 0, 0));
    }
}
