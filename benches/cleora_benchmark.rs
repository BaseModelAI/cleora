//! Benchmarks of the kernels that dominate a Cleora run: building the sparse matrix,
//! one propagation step, row normalisation, the covariance behind whitening, and a
//! complete classic embedding.
//!
//! Run with `cargo bench`. The graph is a fixed pseudo-random one, so numbers are
//! comparable between runs and commits (but not with numbers of real graphs: random
//! graphs have none of the locality real ones have).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use pycleora::configuration::Configuration;
use pycleora::embedding::{
    column_mean_and_covariance, embed, initialize, l2_normalize_rows, propagate, thread_pool,
    MarkovType,
};
use pycleora::pipeline::{build_graph_from_int_edges, build_graph_from_lines};
use pycleora::sparse_matrix::SparseMatrix;

const NODES: i64 = 100_000;
const EDGES: usize = 500_000;
const DIM: usize = 128;

/// SplitMix64: a tiny deterministic generator, so the benchmark needs no extra crate.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn random_edges() -> (Vec<i64>, Vec<i64>) {
    let mut rng = Rng(7);
    let mut sources = Vec::with_capacity(EDGES);
    let mut targets = Vec::with_capacity(EDGES);
    for _ in 0..EDGES {
        sources.push((rng.next() % NODES as u64) as i64);
        targets.push((rng.next() % NODES as u64) as i64);
    }
    (sources, targets)
}

fn configuration(num_workers: usize) -> Configuration {
    Configuration::new("complex::reflexive::node", 16, Some(num_workers))
        .expect("valid column definition")
}

fn build_graph(num_workers: usize) -> SparseMatrix {
    let (sources, targets) = random_edges();
    build_graph_from_int_edges(&configuration(num_workers), &sources, &targets, None)
        .expect("graph builds")
        .0
}

fn worker_counts() -> Vec<usize> {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut counts = vec![1];
    if available > 1 {
        counts.push(available);
    }
    counts
}

fn bench_graph_construction(c: &mut Criterion) {
    let (sources, targets) = random_edges();
    let lines: Vec<String> = sources
        .iter()
        .zip(&targets)
        .map(|(s, t)| format!("{} {}", s, t))
        .collect();

    let mut group = c.benchmark_group("graph_construction");
    group.sample_size(10);
    group.throughput(Throughput::Elements(EDGES as u64));
    for workers in worker_counts() {
        let config = configuration(workers);
        group.bench_with_input(BenchmarkId::new("from_edges", workers), &workers, |b, _| {
            b.iter(|| build_graph_from_int_edges(&config, &sources, &targets, None).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("from_lines", workers), &workers, |b, _| {
            b.iter(|| build_graph_from_lines(&config, &lines, None).unwrap())
        });
    }
    group.finish();
}

fn bench_propagation(c: &mut Criterion) {
    let graph = build_graph(worker_counts().into_iter().max().unwrap_or(1));
    let input = initialize(&graph.entity_ids, DIM, 0);
    let mut output = vec![0f32; input.len()];

    let mut group = c.benchmark_group("propagation");
    group.sample_size(20);
    group.throughput(Throughput::Elements(graph.nnz() as u64));
    for workers in worker_counts() {
        let pool = thread_pool(workers);
        for (name, kind) in [
            ("left", MarkovType::Left),
            ("symmetric", MarkovType::Symmetric),
        ] {
            group.bench_with_input(BenchmarkId::new(name, workers), &workers, |b, _| {
                b.iter(|| pool.install(|| propagate(&graph, &input, &mut output, DIM, kind)))
            });
        }
    }
    group.finish();
}

fn bench_postprocessing(c: &mut Criterion) {
    let graph = build_graph(worker_counts().into_iter().max().unwrap_or(1));
    let rows = graph.num_entities();
    let embeddings = initialize(&graph.entity_ids, DIM, 0);

    let mut group = c.benchmark_group("postprocessing");
    group.sample_size(20);
    group.throughput(Throughput::Elements(rows as u64));
    for workers in worker_counts() {
        let pool = thread_pool(workers);
        group.bench_with_input(
            BenchmarkId::new("l2_normalize", workers),
            &workers,
            |b, _| {
                let mut scratch = embeddings.clone();
                b.iter(|| pool.install(|| l2_normalize_rows(&mut scratch, DIM)))
            },
        );
        group.bench_with_input(BenchmarkId::new("covariance", workers), &workers, |b, _| {
            b.iter(|| pool.install(|| column_mean_and_covariance(&embeddings, rows, DIM)))
        });
    }
    group.finish();
}

fn bench_embedding(c: &mut Criterion) {
    let graph = build_graph(worker_counts().into_iter().max().unwrap_or(1));

    let mut group = c.benchmark_group("classic_embedding");
    group.sample_size(10);
    for workers in worker_counts() {
        let pool = thread_pool(workers);
        group.bench_with_input(
            BenchmarkId::new("8_iterations", workers),
            &workers,
            |b, _| b.iter(|| pool.install(|| embed(&graph, DIM, 0, MarkovType::Left, 8, 0.0, 0.0))),
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_graph_construction,
    bench_propagation,
    bench_postprocessing,
    bench_embedding
);
criterion_main!(benches);
