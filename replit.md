# pycleora

A Python library for efficient, scalable graph embeddings using sparse hypergraph structure and Markov propagation. Built with Rust (via PyO3/maturin) for high performance on CPU, with optional GPU acceleration via PyTorch.

## Overview

**pycleora** is a graph embedding library/SDK, not a web application. It provides the most comprehensive set of graph embedding features available in a single lightweight package — 81 tested features across 13 modules.

## Project Structure

```
pycleora/               Python package
  __init__.py            Main API: embed, attention, supervised, weighted, directed, streaming, edge features
  algorithms.py          Alternative algorithms: ProNE, RandNE, HOPE, NetMF, GraRep, DeepWalk, Node2Vec
  classify.py            Classification: Label Propagation, MLP, GCN (pure numpy/scipy)
  community.py           Community detection: k-means, spectral, Louvain + modularity
  datasets.py            Built-in datasets: 14 graphs (Karate Club to SNAP com-Friendster scale)
  hetero.py              Heterogeneous graphs: HeteroGraph, per-relation, metapath embedding
  io_utils.py            I/O: NetworkX, PyG, DGL, save/load (npz/csv/tsv/parquet)
  metrics.py             Evaluation: AUC/MRR/Hits@K, MAP@K, nDCG, ARI, Silhouette, cross-validation
  sampling.py            Graph sampling: node, edge, neighborhood, subgraph, GraphSAINT, negative sampling
  generators.py          Graph generation: Erdos-Renyi, Barabasi-Albert, SBM, Watts-Strogatz
  tuning.py              Hyperparameter tuning: grid search, random search
  benchmark.py           Benchmarking suite: algorithm comparison, dataset benchmarks, table formatting
  cli.py                 CLI tool: embed, info, benchmark, similar commands
  viz.py                 Visualization: t-SNE, PCA, UMAP + matplotlib plotting
  pycleora.pyi           Type stubs
  pycleora.cpython-*.so  Compiled Rust extension
  __main__.py            CLI entry point (python -m pycleora)
src/                    Rust source code
  lib.rs                 PyO3 bindings (16 methods)
  embedding.rs           Optimized SpMM (adaptive serial/parallel)
  pipeline.rs            Multi-format parser (TSV/CSV/space)
  sparse_matrix.rs       Core CSR data structure
  sparse_matrix_builder.rs  Parallel node indexing
  configuration.rs       Column DSL parsing
  entity.rs              XxHash64 entity hashing
setup.py                setuptools-rust install path
pyproject.toml          maturin build config (primary)
run.py                  Full feature demo (81 tests)
benchmark_report.py     7-algorithm x 3-SNAP-dataset benchmark comparison
website/                Marketing website (Flask)
  app.py                Flask server (port 5000)
  static/style.css      Dark theme CSS
  templates/
    base.html           Shared layout (nav, footer)
    index.html           Landing page (features, comparison, code examples)
    docs.html            Documentation (installation, all APIs, guides)
    api.html             API Reference (all functions, params, returns)
    benchmarks.html      Benchmark results with interactive Chart.js visualizations (3 SNAP datasets: ego-Facebook, roadNet-CA, soc-LiveJournal1)
  static/benchmarks.js   Chart.js benchmark charts (accuracy, speed, memory, scatter, cross-validation)
    changelog.html       Version history (v1.0.0 through v3.0.0)
```

## Build

```bash
pip install maturin==1.2.3
maturin build --release -i python3
pip install target/wheels/pycleora-*.whl --force-reinstall
cp ~/.pythonlibs/lib/python3.12/site-packages/pycleora/pycleora.cpython-312-x86_64-linux-gnu.so pycleora/
```

## Complete Feature Set (81 features across 18 parts)

### Core Embedding (`pycleora/__init__.py`)
- `embed()` - Standard Cleora propagation (left/symmetric, l2/l1/spectral/none normalization)
- `embed_multiscale()` - Multi-depth concatenated embeddings
- `embed_with_node_features()` - Combine graph structure with node features
- `embed_using_baseline_cleora()` - Original Cleora algorithm
- `embed_with_attention()` - Attention-weighted propagation
- `embed_weighted()` - Edge-weight-aware embedding
- `embed_directed()` - Directed graph embedding
- `embed_streaming()` - Stream processing for batch-by-batch ingestion
- `embed_inductive()` - Transfer embeddings to new nodes
- `embed_edge_features()` - Multi-dimensional edge feature embedding (concat/mean/edge_only)
- `supervised_refine()` - Fine-tune with positive/negative pairs
- `update_graph()` / `remove_edges()` - Dynamic graph modification
- `predict_links()` - Top-K link prediction
- `find_most_similar()` / `cosine_similarity()` - Similarity queries

### Alternative Algorithms (`algorithms.py`) — 7 total
- ProNE, RandNE, HOPE, NetMF, GraRep, DeepWalk, Node2Vec

### Classification (`classify.py`)
- Label Propagation (semi-supervised)
- MLP Classifier (pure numpy)
- GCN Classifier (2-layer Graph Convolutional Network, pure numpy/scipy)

### Enhanced Metrics (`metrics.py`)
- Link prediction: AUC, MRR, Hits@K, Average Precision
- Ranking: MAP@K, nDCG@K
- Clustering: NMI, Purity, Adjusted Rand Index (ARI), Silhouette Score
- Cross-validation: k-fold CV for node classification

### Graph Sampling (`sampling.py`)
- `sample_nodes()` - Random node sampling
- `sample_edges()` - Random edge sampling
- `sample_neighborhood()` - K-hop neighborhood sampling with max neighbor limit
- `sample_subgraph()` - Subgraph extraction (random_walk, random_node, bfs)
- `graphsaint_sample()` - GraphSAINT-style mini-batching
- `negative_sampling()` - Generate negative edges avoiding existing
- `train_test_split_edges()` - Split edges for link prediction

### Graph Generation (`generators.py`)
- `erdos_renyi()` - Erdos-Renyi random graphs
- `barabasi_albert()` - Preferential attachment (scale-free)
- `stochastic_block_model()` - Community-structured graphs
- `planted_partition()` - Equal-sized community blocks
- `watts_strogatz()` - Small-world networks

### Hyperparameter Tuning (`tuning.py`)
- `grid_search()` - Exhaustive parameter grid search
- `random_search()` - Randomized parameter search

### Benchmarking (`benchmark.py`)
- `benchmark_algorithms()` - Compare algorithms (time, memory, accuracy)
- `benchmark_datasets()` - Compare across datasets
- `format_benchmark_table()` / `format_dataset_table()` - Formatted output

### CLI (`cli.py`)
- `pycleora embed` - Generate embeddings from edge files
- `pycleora info` - Graph statistics
- `pycleora benchmark` - Run benchmarks
- `pycleora similar` - Find similar entities

## Architecture Notes

- **Louvain**: Uses binary adjacency (w=1.0 per edge, no self-loops), NOT Markov transition values. Same for modularity().
- **Datasets**: Cora, CiteSeer etc. are synthetically generated (community-structured random graphs), cached in `~/.pycleora_datasets/`. SNAP datasets (ego-Facebook, roadNet-CA, soc-LiveJournal1, com-Orkut, com-Friendster) are downloaded from snap.stanford.edu on first use, streamed from .gz, and cached as .npz. Benchmark uses ego-Facebook, roadNet-CA, and soc-LiveJournal1.
- **DeepWalk/Node2Vec**: Use `_build_adj_list()` + random walks + PMI SVD.
- **GCN**: Pure numpy/scipy implementation with symmetric normalization (D^{-1/2} A_hat D^{-1/2}), dropout, multi-layer support.
- **Edge Feature Embedding**: Aggregates edge features to nodes, propagates through graph structure, combines with structural embedding.

## Key Technical Details

- `to_sparse_csr()` returns row-stochastic (Markov) values; community detection and walk-based algorithms use binary adjacency
- `num_nodes = len(labels)` for datasets (not from edge count)
- Build requires maturin 1.2.3; CLI entrypoint configured in both pyproject.toml and setup.py
