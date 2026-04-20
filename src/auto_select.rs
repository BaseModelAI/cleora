//! Spectral probe for automatic operator selection.
//!
//! Computes two cheap diagnostic numbers on the left-Markov matrix
//! M^(L) Cleora builds from its hypergraph clique expansion:
//!
//!   * `sigma_1`  — top singular value (power iteration on M^T M).
//!   * `cos_v1_d` — cosine between the absolute top right singular
//!     vector and the node-degree vector.
//!
//! The ratio `cos_v1_d / sigma_1` is a useful discriminator for
//! choosing between propagation strategies on real recsys graphs: in
//! a 9-dataset full-pipeline MRR study, a threshold of ~0.10 cleanly
//! separated graphs where deflation-style anti-collapse methods won
//! from graphs where plain left-Markov propagation was sufficient.
//!
//! This module contributes only the probe primitive; users compose
//! it with their preferred embedding method.

use std::hash::Hasher;

use twox_hash::XxHash64;

use crate::sparse_matrix::SparseMatrix;

/// Default number of power-iteration rounds.  30 converges tightly
/// enough for the top singular vector on graphs up to ~1M nodes.
pub const DEFAULT_PROBE_ITERS: usize = 30;

/// Diagnostic output of `spectral_probe`.
#[derive(Clone, Debug)]
pub struct ProbeResult {
    pub sigma_1: f64,
    pub cos_v1_d: f64,
    /// Convenience: `cos_v1_d / sigma_1`.  Returns 0 if sigma_1 is 0.
    pub ratio: f64,
    pub probe_iters: usize,
}

#[inline]
fn normalize(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm > 1e-30 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

#[inline]
fn spmv_left(sm: &SparseMatrix, x: &[f64], y: &mut [f64]) {
    y.iter_mut().for_each(|yi| *yi = 0.0);
    for (row, &(s, e)) in sm.slices.iter().enumerate() {
        let mut acc = 0.0_f64;
        for edge in &sm.edges[s..e] {
            acc += (edge.left_markov_value as f64) * x[edge.other_entity_ix as usize];
        }
        y[row] = acc;
    }
}

#[inline]
fn spmv_left_t(sm: &SparseMatrix, x: &[f64], y: &mut [f64]) {
    y.iter_mut().for_each(|yi| *yi = 0.0);
    for (row, &(s, e)) in sm.slices.iter().enumerate() {
        let xi = x[row];
        for edge in &sm.edges[s..e] {
            y[edge.other_entity_ix as usize] += (edge.left_markov_value as f64) * xi;
        }
    }
}

fn row_degrees(sm: &SparseMatrix) -> Vec<f64> {
    sm.slices.iter().map(|(s, e)| (e - s) as f64).collect()
}

/// Run the spectral probe.  One-shot top-1 power iteration over
/// `M^(L)`, returns σ₁, cos(|v₁|, d), and their ratio.
///
/// Deterministic under a given `seed`.  Complexity is
/// O(`probe_iters` × nnz) — 2 SpMVs per round.
pub fn spectral_probe(sm: &SparseMatrix, seed: u64, probe_iters: usize) -> ProbeResult {
    let n = sm.slices.len();
    if n == 0 {
        return ProbeResult {
            sigma_1: 0.0,
            cos_v1_d: 0.0,
            ratio: 0.0,
            probe_iters: 0,
        };
    }

    // Hash-based deterministic init in (-1, 1).
    let mut v: Vec<f64> = (0..n)
        .map(|i| {
            let mut h = XxHash64::with_seed(seed);
            h.write_u64(i as u64);
            let u = (h.finish() as f64) / (u64::MAX as f64);
            u * 2.0 - 1.0
        })
        .collect();
    normalize(&mut v);

    let mut y = vec![0.0_f64; n];
    let mut w = vec![0.0_f64; n];

    for _ in 0..probe_iters {
        spmv_left(sm, &v, &mut y);
        spmv_left_t(sm, &y, &mut w);
        std::mem::swap(&mut v, &mut w);
        normalize(&mut v);
    }

    spmv_left(sm, &v, &mut y);
    let sigma_1: f64 = y.iter().map(|&x| x * x).sum::<f64>().sqrt();

    let d = row_degrees(sm);
    let mut abs_v: Vec<f64> = v.iter().map(|x| x.abs()).collect();
    let norm_v: f64 = abs_v.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let norm_d: f64 = d.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let cos_v1_d = if norm_v > 1e-30 && norm_d > 1e-30 {
        abs_v
            .iter_mut()
            .zip(d.iter())
            .map(|(a, dv)| *a * dv)
            .sum::<f64>()
            / (norm_v * norm_d)
    } else {
        0.0
    };

    let ratio = if sigma_1 > 1e-30 {
        cos_v1_d / sigma_1
    } else {
        0.0
    };

    ProbeResult {
        sigma_1,
        cos_v1_d,
        ratio,
        probe_iters,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_matrix::{Edge, Entity, SparseMatrixDescriptor};

    fn make_sm(n: usize, adj: Vec<Vec<(u32, f32)>>) -> SparseMatrix {
        let descriptor = SparseMatrixDescriptor {
            col_a_id: 0,
            col_a_name: "a".into(),
            col_b_id: 0,
            col_b_name: "a".into(),
        };
        let mut edges: Vec<Edge> = Vec::new();
        let mut slices: Vec<(usize, usize)> = Vec::with_capacity(n);
        for neighbours in &adj {
            let start = edges.len();
            for &(other, w) in neighbours {
                edges.push(Edge {
                    other_entity_ix: other,
                    left_markov_value: w,
                    symmetric_markov_value: w,
                });
            }
            slices.push((start, edges.len()));
        }
        SparseMatrix {
            descriptor,
            entity_ids: (0..n).map(|i| format!("n{}", i)).collect(),
            entities: vec![Entity { row_sum: 1.0 }; n],
            edges,
            slices,
            column_ids: vec![0; n],
        }
    }

    #[test]
    fn probe_empty_graph_returns_zero() {
        let sm = make_sm(0, vec![]);
        let r = spectral_probe(&sm, 0, 10);
        assert_eq!(r.sigma_1, 0.0);
        assert_eq!(r.cos_v1_d, 0.0);
        assert_eq!(r.ratio, 0.0);
    }

    #[test]
    fn probe_is_deterministic() {
        let n = 20usize;
        let w = 1.0 / ((n - 1) as f32);
        let mut adj: Vec<Vec<(u32, f32)>> = Vec::new();
        for i in 0..n {
            adj.push(
                (0..n as u32)
                    .filter(|&j| j != i as u32)
                    .map(|j| (j, w))
                    .collect(),
            );
        }
        let sm = make_sm(n, adj);
        let a = spectral_probe(&sm, 42, 30);
        let b = spectral_probe(&sm, 42, 30);
        assert!((a.sigma_1 - b.sigma_1).abs() < 1e-10);
        assert!((a.cos_v1_d - b.cos_v1_d).abs() < 1e-10);
    }

    #[test]
    fn probe_on_clique_finds_perron() {
        let n = 20usize;
        let w = 1.0 / ((n - 1) as f32);
        let mut adj: Vec<Vec<(u32, f32)>> = Vec::new();
        for i in 0..n {
            adj.push(
                (0..n as u32)
                    .filter(|&j| j != i as u32)
                    .map(|j| (j, w))
                    .collect(),
            );
        }
        let sm = make_sm(n, adj);
        let r = spectral_probe(&sm, 42, 40);
        // Row-stochastic clique has σ₁ = 1 (Perron) and v₁ = 1 → cos = 1.
        assert!((r.sigma_1 - 1.0).abs() < 1e-3);
        assert!(r.cos_v1_d > 0.99);
    }
}
