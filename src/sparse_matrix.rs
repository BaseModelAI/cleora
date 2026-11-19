use std::sync::OnceLock;

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyTuple;
use rustc_hash::{FxBuildHasher, FxHashMap};
use serde::{Deserialize, Serialize};

use crate::configuration::Relation;
use crate::entity::hash_entity;

/// Symmetric sparse co-occurrence matrix of a hypergraph, in CSR form.
///
/// `values` holds the raw accumulated edge weights; the Markov normalisations are
/// applied on the fly from `row_sums`:
///
/// * left:      `P[i, j] = values[i, j] / row_sums[i]`
/// * symmetric: `S[i, j] = values[i, j] / sqrt(row_sums[i] * row_sums[j])`
#[pyclass(name = "SparseMatrix", module = "pycleora.pycleora")]
pub struct SparseMatrix {
    /// Names of the entity columns the matrix was built from.
    pub column_names: Vec<String>,
    /// Relations between those columns that were combined into the matrix.
    pub relations: Vec<Relation>,
    pub entity_ids: Vec<String>,
    /// Entity column in which every entity appeared first.
    pub column_ids: Vec<u8>,
    /// Total hyperedge mass of every entity (its weighted degree).
    pub row_sums: Vec<f64>,
    pub indptr: Vec<u64>,
    pub indices: Vec<u32>,
    pub values: Vec<f32>,

    /// `1 / row_sums` and `1 / sqrt(row_sums)`; zero for entities without mass.
    pub(crate) inv_row: Vec<f32>,
    pub(crate) inv_sqrt_row: Vec<f32>,
    /// Entity hash -> index, built on first lookup.
    lookup: OnceLock<FxHashMap<u64, u32>>,
    /// `entity_ids` as a Python tuple, built on first access.
    pub(crate) py_entity_ids: PyOnceLock<Py<PyTuple>>,
}

const STATE_MAGIC: &[u8] = b"PYCLEORA\x02";

/// Pickled form (format 2).
#[derive(Serialize, Deserialize)]
struct State {
    column_names: Vec<String>,
    relations: Vec<Relation>,
    entity_ids: Vec<String>,
    column_ids: Vec<u8>,
    row_sums: Vec<f64>,
    indptr: Vec<u64>,
    indices: Vec<u32>,
    values: Vec<f32>,
}

/// Pickled form written by pycleora <= 3.2.
#[derive(Deserialize)]
struct LegacyState {
    descriptor: LegacyDescriptor,
    entity_ids: Vec<String>,
    entities: Vec<LegacyEntity>,
    edges: Vec<LegacyEdge>,
    slices: Vec<(usize, usize)>,
    column_ids: Vec<u8>,
}

#[derive(Deserialize)]
struct LegacyDescriptor {
    col_a_id: u8,
    col_a_name: String,
    col_b_id: u8,
    col_b_name: String,
}

#[derive(Deserialize)]
struct LegacyEntity {
    row_sum: f32,
}

#[derive(Deserialize)]
struct LegacyEdge {
    other_entity_ix: u32,
    left_markov_value: f32,
    #[allow(dead_code)]
    symmetric_markov_value: f32,
}

impl SparseMatrix {
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        column_names: Vec<String>,
        relations: Vec<Relation>,
        entity_ids: Vec<String>,
        column_ids: Vec<u8>,
        row_sums: Vec<f64>,
        indptr: Vec<u64>,
        indices: Vec<u32>,
        values: Vec<f32>,
    ) -> SparseMatrix {
        let inv_row = row_sums
            .iter()
            .map(|&s| if s > 0.0 { (1.0 / s) as f32 } else { 0.0 })
            .collect();
        let inv_sqrt_row = row_sums
            .iter()
            .map(|&s| {
                if s > 0.0 {
                    (1.0 / s.sqrt()) as f32
                } else {
                    0.0
                }
            })
            .collect();
        SparseMatrix {
            column_names,
            relations,
            entity_ids,
            column_ids,
            row_sums,
            indptr,
            indices,
            values,
            inv_row,
            inv_sqrt_row,
            lookup: OnceLock::new(),
            py_entity_ids: PyOnceLock::new(),
        }
    }

    pub fn empty() -> SparseMatrix {
        SparseMatrix::from_parts(
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![0],
            vec![],
            vec![],
        )
    }

    #[inline]
    pub fn num_entities(&self) -> usize {
        self.entity_ids.len()
    }

    #[inline]
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    #[inline(always)]
    pub fn row_range(&self, row: usize) -> std::ops::Range<usize> {
        self.indptr[row] as usize..self.indptr[row + 1] as usize
    }

    /// Index of the entity with the given name.
    pub fn index_of(&self, entity_id: &str) -> Option<usize> {
        let lookup = self.lookup.get_or_init(|| {
            let mut map = FxHashMap::with_capacity_and_hasher(self.entity_ids.len(), FxBuildHasher);
            for (index, id) in self.entity_ids.iter().enumerate() {
                // On a (practically impossible) hash collision the first entity keeps the slot.
                map.entry(hash_entity(id)).or_insert(index as u32);
            }
            map
        });
        match lookup.get(&hash_entity(entity_id)) {
            Some(&index) if self.entity_ids[index as usize] == entity_id => Some(index as usize),
            Some(_) => self.entity_ids.iter().position(|id| id == entity_id),
            None => None,
        }
    }

    /// Replaces the entity names (and forgets everything derived from them).
    pub fn replace_entity_ids(&mut self, entity_ids: Vec<String>) {
        self.entity_ids = entity_ids;
        self.lookup = OnceLock::new();
        self.py_entity_ids = PyOnceLock::new();
    }

    /// Entities without any neighbour other than themselves, in increasing order.
    pub fn isolated_entities(&self) -> Vec<usize> {
        (0..self.num_entities())
            .filter(|&row| self.row_range(row).all(|k| self.indices[k] as usize == row))
            .collect()
    }

    /// Mean number of neighbours (entities other than itself) of an entity.
    pub fn mean_neighbours(&self) -> f64 {
        let rows = self.num_entities();
        if rows == 0 {
            return 0.0;
        }
        let diagonal: usize = (0..rows)
            .map(|row| {
                self.row_range(row)
                    .filter(|&k| self.indices[k] as usize == row)
                    .count()
            })
            .sum();
        (self.nnz() - diagonal) as f64 / rows as f64
    }

    /// Share of the total weight that lies on the diagonal: how lazy the random walk
    /// over this matrix is. 0 for a plain adjacency matrix, 1/2 for a reflexive column
    /// of two-entity hyperedges. Summed in row order, so the result does not depend
    /// on the number of threads.
    pub fn self_weight_share(&self) -> f64 {
        let total: f64 = self.row_sums.iter().sum();
        if total <= 0.0 {
            return 0.0;
        }
        let diagonal: f64 = (0..self.num_entities())
            .flat_map(|row| self.row_range(row).map(move |k| (row, k)))
            .filter(|&(row, k)| self.indices[k] as usize == row)
            .map(|(_, k)| self.values[k] as f64)
            .sum();
        diagonal / total
    }

    /// Edge value under the requested normalisation.
    #[inline(always)]
    pub fn normalized_value(&self, row: usize, k: usize, kind: Normalization) -> f32 {
        let raw = self.values[k];
        match kind {
            Normalization::Raw => raw,
            Normalization::Left => raw * self.inv_row[row],
            Normalization::Symmetric => {
                raw * self.inv_sqrt_row[row] * self.inv_sqrt_row[self.indices[k] as usize]
            }
        }
    }

    pub fn to_state_bytes(&self) -> Result<Vec<u8>, String> {
        let state = State {
            column_names: self.column_names.clone(),
            relations: self.relations.clone(),
            entity_ids: self.entity_ids.clone(),
            column_ids: self.column_ids.clone(),
            row_sums: self.row_sums.clone(),
            indptr: self.indptr.clone(),
            indices: self.indices.clone(),
            values: self.values.clone(),
        };
        let mut bytes = STATE_MAGIC.to_vec();
        bincode::serialize_into(&mut bytes, &state)
            .map_err(|e| format!("Serialization failed: {}", e))?;
        Ok(bytes)
    }

    pub fn from_state_bytes(bytes: &[u8]) -> Result<SparseMatrix, String> {
        if let Some(payload) = bytes.strip_prefix(STATE_MAGIC) {
            let state: State = bincode::deserialize(payload)
                .map_err(|e| format!("Deserialization failed: {}", e))?;
            let matrix = SparseMatrix::from_parts(
                state.column_names,
                state.relations,
                state.entity_ids,
                state.column_ids,
                state.row_sums,
                state.indptr,
                state.indices,
                state.values,
            );
            matrix.validate()?;
            return Ok(matrix);
        }
        Self::from_legacy_state_bytes(bytes)
    }

    /// Reads a matrix pickled by pycleora <= 3.2.
    fn from_legacy_state_bytes(bytes: &[u8]) -> Result<SparseMatrix, String> {
        let legacy: LegacyState =
            bincode::deserialize(bytes).map_err(|e| format!("Deserialization failed: {}", e))?;
        let n = legacy.entity_ids.len();
        if legacy.entities.len() != n || legacy.column_ids.len() != n || legacy.slices.len() > n {
            return Err("Deserialization failed: inconsistent legacy state".to_string());
        }
        // Old matrices stored one slice per entity, in entity order.
        let mut indptr = vec![0u64; n + 1];
        for (row, (start, end)) in legacy.slices.iter().enumerate() {
            if start > end || *end > legacy.edges.len() {
                return Err("Deserialization failed: inconsistent legacy state".to_string());
            }
            indptr[row] = *start as u64;
            indptr[row + 1] = *end as u64;
        }
        for row in legacy.slices.len()..n {
            indptr[row + 1] = indptr[row];
        }
        let row_sums: Vec<f64> = legacy.entities.iter().map(|e| e.row_sum as f64).collect();
        let mut indices = Vec::with_capacity(legacy.edges.len());
        let mut values = Vec::with_capacity(legacy.edges.len());
        for row in 0..n {
            for edge in &legacy.edges[indptr[row] as usize..indptr[row + 1] as usize] {
                indices.push(edge.other_entity_ix);
                values.push(edge.left_markov_value * legacy.entities[row].row_sum);
            }
        }
        let d = legacy.descriptor;
        let (column_names, relations, column_ids) = if d.col_a_name == d.col_b_name
            && d.col_a_id != d.col_b_id
            && legacy.column_ids.iter().all(|c| *c == d.col_a_id)
        {
            // a single reflexive column
            (
                vec![d.col_a_name],
                vec![Relation { a: 0, b: 0 }],
                vec![0u8; n],
            )
        } else {
            let column_ids = legacy
                .column_ids
                .iter()
                .map(|c| if *c == d.col_b_id { 1 } else { 0 })
                .collect();
            (
                vec![d.col_a_name, d.col_b_name],
                vec![Relation { a: 0, b: 1 }],
                column_ids,
            )
        };
        let matrix = SparseMatrix::from_parts(
            column_names,
            relations,
            legacy.entity_ids,
            column_ids,
            row_sums,
            indptr,
            indices,
            values,
        );
        matrix.validate()?;
        Ok(matrix)
    }

    fn validate(&self) -> Result<(), String> {
        let n = self.entity_ids.len();
        let consistent = self.column_ids.len() == n
            && self.row_sums.len() == n
            && self.indptr.len() == n + 1
            && self.indptr.first() == Some(&0)
            && self.indptr.windows(2).all(|w| w[0] <= w[1])
            && self.indptr.last().copied() == Some(self.indices.len() as u64)
            && self.indices.len() == self.values.len()
            && self.indices.iter().all(|&j| (j as usize) < n);
        if consistent {
            Ok(())
        } else {
            Err("Deserialization failed: inconsistent matrix state".to_string())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalization {
    Raw,
    Left,
    Symmetric,
}

impl Normalization {
    pub fn parse(name: Option<&str>) -> Result<Normalization, String> {
        match name.unwrap_or("left") {
            "left" => Ok(Normalization::Left),
            "symmetric" => Ok(Normalization::Symmetric),
            "raw" | "none" => Ok(Normalization::Raw),
            other => Err(format!(
                "Unknown markov_type '{}'. Use 'left', 'symmetric' or 'raw'.",
                other
            )),
        }
    }
}
