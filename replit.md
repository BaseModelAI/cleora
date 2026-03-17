# pycleora

A Python library for efficient, scalable graph embeddings using sparse hypergraph structure and Markov propagation. Built with Rust (via PyO3/maturin) for high performance on CPU, with optional GPU acceleration via PyTorch.

## Overview

**pycleora** is a graph embedding library/SDK, not a web application. It provides the most comprehensive set of graph embedding features available in a single lightweight package — 54 tested features across 8 modules.

## Project Structure

```
pycleora/               Python package
  __init__.py            Main API: embed, attention, supervised, weighted, directed, streaming, etc.
  algorithms.py          Alternative algorithms: ProNE, RandNE, HOPE, NetMF, GraRep, DeepWalk, Node2Vec
  classify.py            Classification: Label Propagation, MLP classifier (numpy-based)
  community.py           Community detection: k-means, spectral, Louvain + modularity
  datasets.py            Built-in datasets: 12 graphs (Karate Club → Reddit scale)
  hetero.py              Heterogeneous graphs: HeteroGraph, per-relation, metapath embedding
  io_utils.py            I/O: NetworkX, PyG, DGL, save/load (npz/csv/tsv/parquet)
  metrics.py             Evaluation: link prediction (AUC/MRR/Hits@K), node classification, clustering
  viz.py                 Visualization: t-SNE, PCA, UMAP + matplotlib plotting
  pycleora.pyi           Type stubs
  pycleora.cpython-*.so  Compiled Rust extension
src/                    Rust source code
  lib.rs                 PyO3 bindings (16 methods)
  embedding.rs           Optimized SpMM (adaptive serial/parallel)
  pipeline.rs            Multi-format parser (TSV/CSV/space)
  sparse_matrix.rs       Core CSR data structure
  sparse_matrix_builder.rs  Parallel node indexing
  configuration.rs       Column DSL parsing
  entity.rs              XxHash64 entity hashing
run.py                  Full feature demo (54 tests)
```

## Build

```bash
pip install maturin==1.2.3
maturin build --release -i python3
pip install target/wheels/pycleora-*.whl --force-reinstall
cp ~/.pythonlibs/lib/python3.12/site-packages/pycleora/pycleora.cpython-312-x86_64-linux-gnu.so pycleora/
```

## Complete Feature Set (52 features)

### Core Embedding (`pycleora`)
- `embed()` - Standard Cleora propagation (left/symmetric, l2/l1/spectral/none normalization)
- `embed_multiscale()` - Multi-depth concatenated embeddings
- `embed_with_node_features()` - Combine graph structure with node features
- `embed_using_baseline_cleora()` - Original Cleora algorithm

### Advanced Embedding Methods (`pycleora`)
- `embed_with_attention()` - Attention-weighted propagation (GAT-style without neural net training)
- `embed_weighted()` - Weighted edge support (different edge importance)
- `embed_directed()` - Directed graph embedding (asymmetric propagation)
- `supervised_refine()` - Contrastive learning refinement with margin loss

### Alternative Algorithms (`pycleora.algorithms`)
- `embed_prone()` - ProNE: spectral propagation with Chebyshev polynomials
- `embed_randne()` - RandNE: random projection embedding (fastest)
- `embed_hope()` - HOPE: asymmetric, good for directed graphs
- `embed_netmf()` - NetMF: theoretical generalization of DeepWalk
- `embed_grarep()` - GraRep: multi-scale matrix factorization
- `embed_deepwalk()` - DeepWalk: random walk + SVD (the original graph embedding)
- `embed_node2vec()` - Node2Vec: biased random walk with p,q parameters (BFS/DFS balance)

### Graph Updates (`pycleora`)
- `update_graph()` - Add new edges (preserves original hyperedges)
- `remove_edges()` - Remove edges dynamically
- `embed_inductive()` - Inductive embedding for new entities
- `embed_streaming()` - Process edges in batches (out-of-core capable)

### Classification (`pycleora.classify`)
- `label_propagation()` - Semi-supervised label propagation
- `label_propagation_predict()` - Label propagation with train/test split
- `mlp_classify()` - 2-layer MLP classifier (pure numpy, no PyTorch needed)

### Heterogeneous Graphs (`pycleora.hetero`)
- `HeteroGraph` - Multi-type node/edge graph container
- `embed_per_relation()` - Per-relation embeddings with concat/mean combining
- `embed_metapath()` - Metapath-based embedding (e.g., user→product→store)
- `to_homogeneous_edges()` - Convert to flat edge list

### Analysis (`pycleora`)
- `predict_links()` - Link prediction with ranked results
- `find_most_similar()` - Top-K similarity search
- `cosine_similarity()` - Pairwise similarity

### Community Detection (`pycleora.community`)
- `detect_communities_kmeans()` - K-means on embeddings
- `detect_communities_spectral()` - Spectral clustering
- `detect_communities_louvain()` - Louvain algorithm (optimized O(n·degree))
- `modularity()` - Modularity score

### Evaluation Metrics (`pycleora.metrics`)
- `link_prediction_scores()` - AUC, MRR, Hits@K, Average Precision
- `node_classification_scores()` - Accuracy, Macro-F1, Weighted-F1
- `clustering_scores()` - NMI, Purity, Intra-cluster similarity

### Datasets (`pycleora.datasets`) — 12 total
Small (built-in):
- Karate Club (34 nodes, 2 classes)
- Dolphins (62 nodes, 3 classes)
- Les Miserables (77 nodes, 7 classes)
- Football (32 nodes, 3 classes)

Medium (generated with community structure, cached):
- Cora (2708 nodes, 7 classes)
- CiteSeer (3312 nodes, 6 classes)
- PubMed (19717 nodes, 3 classes)
- Amazon Computers (13752 nodes, 10 classes)
- Amazon Photo (7650 nodes, 8 classes)
- PPI (3890 nodes, 50 classes)
- DBLP (4057 nodes, 4 classes)
- Reddit (10000 nodes, 41 classes)

### I/O (`pycleora.io_utils`)
- NetworkX import/export
- PyG Data export
- DGL Graph export
- Save/load embeddings (npz, csv, tsv, parquet)
- Edge list export

### Visualization (`pycleora.viz`)
- t-SNE dimensionality reduction (built-in, no sklearn needed)
- PCA reduction (built-in)
- UMAP reduction (optional)
- matplotlib-based plotting with labels, colors, save-to-file

### GPU Acceleration
- `propagate_gpu()` - Optional PyTorch sparse matrix multiplication (CUDA/CPU)

## Dependencies

- numpy (required)
- scipy (required for algorithms, attention, weighted, directed, classify)
- matplotlib (required for visualization)
- networkx (required for graph export)
- maturin 1.2.3 (build only)
- torch (optional, GPU propagation)
- pyarrow (optional, parquet export)
- umap-learn (optional, UMAP reduction)
- torch-geometric (optional, PyG export)
- dgl (optional, DGL export)

## Workflow

- **Start application**: `python3 run.py` — demo script (console output, 54 feature tests)
