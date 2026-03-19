# pycleora vs Competitors — Feature Comparison

## Libraries Compared

| Library | Language | Backend | Focus | Status |
|---------|----------|---------|-------|--------|
| **pycleora 2.3** | Rust + Python | CPU (Rust SpMM) | Full-stack graph embedding SDK | Active |
| **PyG (PyTorch Geometric)** | Python | GPU (PyTorch) | GNN training framework | Active |
| **KarateClub** | Python | CPU (numpy/scipy) | Traditional graph embedding | Active |
| **Cleora (BaseModelAI)** | Rust | CPU | Sparse embedding only | Minimal |
| **Node2Vec (Eliorc)** | Python | CPU | Random walk embedding | Active |
| **DGL** | Python | GPU (PyTorch/MXNet) | GNN training framework | Active |
| **StellarGraph** | Python | GPU (TensorFlow/Keras) | GNN & graph ML library | Archived |
| **GEM** | Python | CPU (numpy/scipy) | Graph embedding methods | Inactive |
| **GraphVite** | C++/Python | GPU (CUDA) | High-speed graph embedding | Inactive |
| **DeepWalk** | Python | CPU (gensim) | Random walk + Skip-gram | Inactive |
| **LINE** | C++/Python | CPU | 1st/2nd order proximity | Inactive |
| **SDNE** | Python | GPU (TF/Keras) | Autoencoder-based embedding | Inactive |
| **graspologic** | Python | CPU (numpy/scipy) | Spectral graph statistics | Active |
| **GraphSAGE** | Python | GPU (TF) | Inductive node embedding | Inactive |
| **Struc2Vec** | Python | CPU | Structural identity embedding | Inactive |
| **VERSE** | C++/Python | CPU | Versatile graph embedding | Inactive |
| **NetSMF** | C++/Python | CPU | Sparse matrix factorization | Inactive |

> **Sources**: [StellarGraph repo](https://github.com/stellargraph/stellargraph) (archived 2022), [GEM repo](https://github.com/palash1992/GEM) (last commit 2020), [GraphVite repo](https://github.com/DeepGraphLearning/graphvite) (last commit 2020), [Node2Vec/eliorc](https://github.com/eliorc/node2vec), [DeepWalk](https://github.com/phanein/deepwalk) (last commit 2019), [LINE](https://github.com/tangjianpku/LINE) (last commit 2017), [SDNE](https://github.com/suanrong/SDNE) (last commit 2018), [graspologic](https://github.com/microsoft/graspologic) (Microsoft, active), [GraphSAGE](https://github.com/williamleif/GraphSAGE) (reference impl, last commit 2019), [Struc2Vec](https://github.com/leoribeiro/struc2vec) (last commit 2018), [VERSE](https://github.com/xgfs/verse) (last commit 2019), [NetSMF](https://github.com/xptree/NetSMF) (last commit 2019). Install sizes estimated from pip download + dependencies. Timing estimates are order-of-magnitude based on published benchmarks and documentation.

---

## Feature Matrix

### Embedding Algorithms

| Feature | pycleora | PyG | KarateClub | Cleora (Base) | Node2Vec | DGL | StellarGraph | GEM | GraphVite | DeepWalk | LINE | SDNE | graspologic | GraphSAGE | Struc2Vec | VERSE | NetSMF |
|---------|:--------:|:---:|:----------:|:-------------:|:--------:|:---:|:------------:|:---:|:---------:|:--------:|:----:|:----:|:-----------:|:---------:|:---------:|:-----:|:------:|
| Cleora (Markov propagation) | **Yes** | No | No | **Yes** | No | No | No | No | No | No | No | No | No | No | No | No | No |
| ProNE | **Yes** | No | **Yes** | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| RandNE | **Yes** | No | **Yes** | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| HOPE | **Yes** | No | **Yes** | No | No | No | No | **Yes** | No | No | No | No | No | No | No | No | No |
| NetMF | **Yes** | No | **Yes** | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| GraRep | **Yes** | No | **Yes** | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| Node2Vec/DeepWalk | **Yes** | **Yes** | **Yes** | No | **Yes** | No | **Yes** | **Yes** | **Yes** | **Yes** | No | No | No | No | No | No | No |
| LINE | No | No | No | No | No | No | No | No | **Yes** | No | **Yes** | No | No | No | No | No | No |
| SDNE | No | No | No | No | No | No | No | **Yes** | No | No | No | **Yes** | No | No | No | No | No |
| Laplacian Eigenmaps | No | No | **Yes** | No | No | No | No | **Yes** | No | No | No | No | **Yes** | No | No | No | No |
| GCN/GAT/GraphSAGE | No | **Yes** | No | No | No | **Yes** | **Yes** | No | No | No | No | No | No | **Yes** | No | No | No |
| Attention-weighted embed | **Yes** | Partial | No | No | No | Partial | Partial | No | No | No | No | No | No | No | No | No | No |
| Multiscale embedding | **Yes** | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| Supervised refinement | **Yes** | **Yes** | No | No | No | **Yes** | **Yes** | No | No | No | No | No | No | **Yes** | No | No | No |
| Struc2Vec | No | No | No | No | No | No | No | No | No | No | No | No | No | No | **Yes** | No | No |
| VERSE | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | **Yes** | No |
| NetSMF | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | **Yes** |
| ASE/LSE (spectral) | No | No | No | No | No | No | No | No | No | No | No | No | **Yes** | No | No | No | No |
| **Total algorithms** | **11** | **50+** | **30+** | **1** | **1** | **50+** | **~15** | **~8** | **~5** | **1** | **1** | **1** | **~5** | **1** | **1** | **1** | **1** |

### Graph Types

| Feature | pycleora | PyG | KarateClub | Cleora (Base) | Node2Vec | DGL | StellarGraph | GEM | GraphVite | DeepWalk | LINE | SDNE | graspologic | GraphSAGE | Struc2Vec | VERSE | NetSMF |
|---------|:--------:|:---:|:----------:|:-------------:|:--------:|:---:|:------------:|:---:|:---------:|:--------:|:----:|:----:|:-----------:|:---------:|:---------:|:-----:|:------:|
| Undirected graphs | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |
| Directed graphs | **Yes** | **Yes** | Partial | No | No | **Yes** | **Yes** | **Yes** | **Yes** | No | **Yes** | **Yes** | **Yes** | No | No | **Yes** | No |
| Weighted edges | **Yes** | **Yes** | Partial | No | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | No | **Yes** | **Yes** | **Yes** | No | No | **Yes** | **Yes** |
| Hypergraph/bipartite | **Yes** | **Yes** | No | **Yes** | No | **Yes** | **Yes** | No | No | No | No | No | No | **Yes** | No | No | No |
| Heterogeneous graphs | **Yes** | **Yes** | No | No | No | **Yes** | **Yes** | No | No | No | No | No | No | **Yes** | No | No | No |
| Node features | **Yes** | **Yes** | No | No | No | **Yes** | **Yes** | No | No | No | No | No | No | **Yes** | No | No | No |

### Dynamic/Streaming

| Feature | pycleora | PyG | KarateClub | Cleora (Base) | Node2Vec | DGL | StellarGraph | GEM | GraphVite | DeepWalk | LINE | SDNE | graspologic | GraphSAGE | Struc2Vec | VERSE | NetSMF |
|---------|:--------:|:---:|:----------:|:-------------:|:--------:|:---:|:------------:|:---:|:---------:|:--------:|:----:|:----:|:-----------:|:---------:|:---------:|:-----:|:------:|
| Incremental update (add edges) | **Yes** | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| Edge removal | **Yes** | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| Inductive embedding | **Yes** | **Yes** | No | No | No | **Yes** | **Yes** | No | No | No | No | No | No | **Yes** | No | No | No |
| Streaming (batch-by-batch) | **Yes** | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No |

### Classification

| Feature | pycleora | PyG | KarateClub | Cleora (Base) | Node2Vec | DGL | StellarGraph | GEM | GraphVite | DeepWalk | LINE | SDNE | graspologic | GraphSAGE | Struc2Vec | VERSE | NetSMF |
|---------|:--------:|:---:|:----------:|:-------------:|:--------:|:---:|:------------:|:---:|:---------:|:--------:|:----:|:----:|:-----------:|:---------:|:---------:|:-----:|:------:|
| Label Propagation | **Yes** | **Yes** | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| MLP classifier (built-in) | **Yes** | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| GNN classifiers | No | **Yes** | No | No | No | **Yes** | **Yes** | No | No | No | No | No | No | **Yes** | No | No | No |

### Community Detection

| Feature | pycleora | PyG | KarateClub | Cleora (Base) | Node2Vec | DGL | StellarGraph | GEM | GraphVite | DeepWalk | LINE | SDNE | graspologic | GraphSAGE | Struc2Vec | VERSE | NetSMF |
|---------|:--------:|:---:|:----------:|:-------------:|:--------:|:---:|:------------:|:---:|:---------:|:--------:|:----:|:----:|:-----------:|:---------:|:---------:|:-----:|:------:|
| K-Means clustering | **Yes** | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| Spectral clustering | **Yes** | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| Louvain algorithm | **Yes** | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| Modularity score | **Yes** | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No |

### Evaluation Metrics

| Feature | pycleora | PyG | KarateClub | Cleora (Base) | Node2Vec | DGL | StellarGraph | GEM | GraphVite | DeepWalk | LINE | SDNE | graspologic | GraphSAGE | Struc2Vec | VERSE | NetSMF |
|---------|:--------:|:---:|:----------:|:-------------:|:--------:|:---:|:------------:|:---:|:---------:|:--------:|:----:|:----:|:-----------:|:---------:|:---------:|:-----:|:------:|
| Link prediction (AUC/MRR/Hits@K) | **Yes** | Partial | No | No | No | No | Partial | No | Partial | No | No | No | No | No | No | No | No |
| Node classification (Acc/F1) | **Yes** | Partial | **Yes** | No | No | No | Partial | No | Partial | No | No | No | No | No | No | No | No |
| Clustering (NMI/Purity) | **Yes** | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No |

### Built-in Datasets

| Feature | pycleora | PyG | KarateClub | Cleora (Base) | Node2Vec | DGL | StellarGraph | GEM | GraphVite | DeepWalk | LINE | SDNE | graspologic | GraphSAGE | Struc2Vec | VERSE | NetSMF |
|---------|:--------:|:---:|:----------:|:-------------:|:--------:|:---:|:------------:|:---:|:---------:|:--------:|:----:|:----:|:-----------:|:---------:|:---------:|:-----:|:------:|
| Small graphs (Karate, etc.) | **4** | **4+** | **4+** | **0** | **0** | **4+** | **3+** | **2** | **0** | **0** | **0** | **0** | **3+** | **0** | **0** | **0** | **0** |
| Citation networks (Cora, etc.) | **4** | **4+** | **0** | **0** | **0** | **4+** | **3+** | **1** | **3+** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |
| Amazon co-purchase | **2** | **2** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |
| Large-scale (PPI, Reddit, DBLP) | **2** | **5+** | **0** | **0** | **0** | **5+** | **4+** | **0** | **2+** | **0** | **0** | **0** | **2+** | **0** | **0** | **0** | **0** |
| **Total datasets** | **14** | **70+** | **~5** | **0** | **0** | **40+** | **~10** | **~3** | **~5** | **0** | **0** | **0** | **~5** | **0** | **0** | **0** | **0** |

### I/O & Interoperability

| Feature | pycleora | PyG | KarateClub | Cleora (Base) | Node2Vec | DGL | StellarGraph | GEM | GraphVite | DeepWalk | LINE | SDNE | graspologic | GraphSAGE | Struc2Vec | VERSE | NetSMF |
|---------|:--------:|:---:|:----------:|:-------------:|:--------:|:---:|:------------:|:---:|:---------:|:--------:|:----:|:----:|:-----------:|:---------:|:---------:|:-----:|:------:|
| NetworkX import/export | **Yes** | **Yes** | **Yes** | No | **Yes** | **Yes** | **Yes** | **Yes** | No | **Yes** | No | No | **Yes** | No | **Yes** | No | No |
| PyG Data export | **Yes** | N/A | No | No | No | **Yes** | No | No | No | No | No | No | No | No | No | No | No |
| DGL Graph export | **Yes** | **Yes** | No | No | No | N/A | No | No | No | No | No | No | No | No | No | No | No |
| Save/load NPZ | **Yes** | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| Save/load CSV/TSV | **Yes** | No | No | **Yes** | No | No | No | No | No | No | No | No | No | No | No | No | No |
| Save/load Parquet | **Yes** | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| Pickle serialization | **Yes** | **Yes** | **Yes** | No | **Yes** | **Yes** | **Yes** | **Yes** | No | No | No | No | **Yes** | No | No | No | No |

### Visualization

| Feature | pycleora | PyG | KarateClub | Cleora (Base) | Node2Vec | DGL | StellarGraph | GEM | GraphVite | DeepWalk | LINE | SDNE | graspologic | GraphSAGE | Struc2Vec | VERSE | NetSMF |
|---------|:--------:|:---:|:----------:|:-------------:|:--------:|:---:|:------------:|:---:|:---------:|:--------:|:----:|:----:|:-----------:|:---------:|:---------:|:-----:|:------:|
| t-SNE (built-in) | **Yes** | No | No | No | No | No | No | No | Partial | No | No | No | No | No | No | No | No |
| PCA (built-in) | **Yes** | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| UMAP support | **Yes** | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| Plot with labels/colors | **Yes** | No | No | No | No | No | No | No | No | No | No | No | **Yes** | No | No | No | No |

### Performance & Setup

| Feature | pycleora | PyG | KarateClub | Cleora (Base) | Node2Vec | DGL | StellarGraph | GEM | GraphVite | DeepWalk | LINE | SDNE | graspologic | GraphSAGE | Struc2Vec | VERSE | NetSMF |
|---------|:--------:|:---:|:----------:|:-------------:|:--------:|:---:|:------------:|:---:|:---------:|:--------:|:----:|:----:|:-----------:|:---------:|:---------:|:-----:|:------:|
| No GPU required | **Yes** | No | **Yes** | **Yes** | **Yes** | No | Optional | **Yes** | No | **Yes** | **Yes** | Optional | **Yes** | Optional | **Yes** | **Yes** | **Yes** |
| Optional GPU support | **Yes** | **Yes** | No | No | No | **Yes** | **Yes** | No | **Yes** | No | No | **Yes** | No | **Yes** | No | No | No |
| Multi-GPU support | Planned | **Yes** | No | No | No | **Yes** | Limited | No | **Yes** | No | No | No | No | No | No | No | No |
| Rust-native performance | **Yes** | C++/CUDA | No | **Yes** | No | C++/CUDA | No | No | C++/CUDA | No | C++ | No | No | No | No | C++ | C++ |
| pip install (no build) | No | **Yes** | **Yes** | No | **Yes** | **Yes** | **Yes** | No | No | **Yes** | No | No | **Yes** | No | **Yes** | No | No |
| Minimal dependencies | **Yes** | No | **Yes** | **Yes** | **Yes** | No | No | No | No | **Yes** | **Yes** | No | No | No | **Yes** | **Yes** | **Yes** |
| Embedding 1M nodes (time) | ~10s | ~30s+GPU | ~60s | ~10s | ~300s | ~30s+GPU | ~60s+GPU | ~120s | ~5s+GPU | ~300s | ~60s | ~120s+GPU | ~30s | ~60s+GPU | ~600s | ~30s | ~20s |
| Actively maintained | **Yes** | **Yes** | **Yes** | Minimal | **Yes** | **Yes** | Archived | Inactive | Inactive | Inactive | Inactive | Inactive | **Yes** | Inactive | Inactive | Inactive | Inactive |

---

## Summary: Competitive Position

### Where pycleora leads:

1. **All-in-one SDK** — No other single library combines: embedding + 5 algorithms + classification + community detection + evaluation + 14 datasets + visualization + heterogeneous graphs + streaming + I/O
2. **CPU performance** — Rust backend matches or beats pure-Python alternatives by 10-100x
3. **Dynamic graphs** — Only library with incremental update, edge removal, streaming, AND inductive embedding
4. **Zero-GPU deployment** — Full feature set works on CPU; GPU is optional
5. **Built-in evaluation pipeline** — Link prediction, node classification, and clustering metrics without sklearn
6. **Lightweight footprint** — ~5 MB install vs 200-600 MB for GPU-dependent competitors

### Where competitors lead:

1. **PyG/DGL** — 50+ GNN architectures (GCN, GAT, GraphSAGE, etc.) with GPU training
2. **KarateClub** — 30+ traditional algorithms (community detection, diffusion, etc.)
3. **PyG/DGL** — 80+ real benchmark datasets (not synthetically generated)
4. **PyG/DGL** — Active community, papers, tutorials, production deployments
5. **Node2Vec** — Simplest API for random-walk based embeddings
6. **GraphVite** — Fastest GPU-accelerated embedding (multi-GPU, C++/CUDA core)
7. **StellarGraph** — Keras-based API for accessible GNN experimentation (though now archived)
8. **graspologic** — Microsoft-maintained spectral methods (ASE, LSE, OMNI) with strong statistical foundations
9. **GraphSAGE** — Pioneered inductive graph representation learning

### Feature counts:

| Category | pycleora | PyG | KarateClub | Cleora (Base) | Node2Vec | DGL | StellarGraph | GEM | GraphVite | DeepWalk | LINE | SDNE | graspologic | GraphSAGE | Struc2Vec | VERSE | NetSMF |
|----------|:--------:|:---:|:----------:|:-------------:|:--------:|:---:|:------------:|:---:|:---------:|:--------:|:----:|:----:|:-----------:|:---------:|:---------:|:-----:|:------:|
| Public functions | **52** | 200+ | 40+ | **8** | ~5 | 200+ | ~80 | ~15 | ~10 | ~5 | ~3 | ~5 | 50+ | ~10 | ~5 | ~3 | ~3 |
| Algorithms | 11 | 50+ | 30+ | 1 | 1 | 50+ | ~15 | ~8 | ~5 | 1 | 1 | 1 | ~5 | 1 | 1 | 1 | 1 |
| Datasets | 14 | 70+ | ~5 | 0 | 0 | 40+ | ~10 | ~3 | ~5 | 0 | 0 | 0 | ~5 | 0 | 0 | 0 | 0 |
| Graph types | 6 | 6 | 3 | 2 | 2 | 6 | 5 | 3 | 3 | 1 | 2 | 2 | 3 | 4 | 1 | 3 | 2 |
| Downstream tasks | 7 | 3 | 2 | 0 | 0 | 3 | 4 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| I/O formats | 6 | 3 | 2 | 1 | 1 | 3 | 2 | 1 | 0 | 1 | 0 | 0 | 2 | 0 | 1 | 0 | 0 |
| Dependencies (required) | 3 | 10+ | 5+ | 0 | 2 | 10+ | 8+ | 5+ | 3+ | 3 | 1 | 5+ | 8+ | 5+ | 3 | 1 | 2 |
| Install size | ~5 MB | ~500 MB+ | ~15 MB | ~3 MB | ~2 MB | ~400 MB+ | ~600 MB+ | ~50 MB | ~200 MB+ | ~5 MB | ~5 MB | ~300 MB+ | ~50 MB | ~500 MB+ | ~5 MB | ~5 MB | ~10 MB |
