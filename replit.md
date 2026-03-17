# pycleora

A Python library for efficient, scalable graph embeddings using sparse hypergraph structure and Markov propagation. Built with Rust (via PyO3/maturin) for high performance on CPU, with optional GPU acceleration via PyTorch.

## Overview

**pycleora** is a Python library exposing Rust-based graph embedding functionality. It is NOT a web application — it is a library/SDK.

## Project Structure

```
pycleora/           - Python package
  __init__.py       - Full API: embed, embed_with_attention, embed_multiscale,
                      embed_inductive, supervised_refine, update_graph,
                      propagate_gpu, find_most_similar, cosine_similarity
  pycleora.pyi      - Type stubs
  pycleora.cpython-312-x86_64-linux-gnu.so  - Compiled Rust extension
src/                - Rust source code
  lib.rs            - PyO3 bindings, to_sparse_csr, get_neighbors, entity lookup
  embedding.rs      - Optimized SpMM (adaptive serial/parallel per node degree)
  pipeline.rs       - Multi-format parser (TSV/CSV/space), safe file handling
  sparse_matrix.rs  - Core CSR data structure
  sparse_matrix_builder.rs - Safe parallel node indexing (no unsafe)
  configuration.rs  - Column DSL parsing
  entity.rs         - XxHash64 entity hashing
run.py              - Full feature demo (14 tests)
```

## Build

```bash
pip install maturin==1.2.3
maturin build --release -i python3
pip install target/wheels/pycleora-*.whl --force-reinstall
cp ~/.pythonlibs/lib/python3.12/site-packages/pycleora/pycleora.cpython-312-x86_64-linux-gnu.so pycleora/
```

## Key Features

### Core (Rust)
- **embed()** - Configurable propagation (left/symmetric), normalization (l2/l1/spectral/none)
- **embed_multiscale()** - Multi-depth concatenated embeddings
- **to_sparse_csr()** - Export transition matrix for custom workflows
- **get_neighbors()** - Query graph structure per entity

### Attention (Python + scipy)
- **embed_with_attention()** - Computes attention weights from embedding similarity between connected nodes, applies softmax-normalized attention to propagation. Approximates GAT without neural network training.

### Supervised Learning (Python)
- **supervised_refine()** - Contrastive learning with margin loss on positive/negative entity pairs. Refines embeddings using gradient-based optimization.

### Incremental Updates (Python)
- **update_graph()** - Merges new edges into existing graph structure
- **embed_inductive()** - Generates embeddings for new entities by initializing from trained embeddings and re-propagating. Existing entities retain learned knowledge.

### GPU Acceleration (Optional, requires PyTorch)
- **propagate_gpu()** - PyTorch sparse matrix multiplication on CUDA/CPU. Falls back gracefully when torch not installed.

## Dependencies

- numpy (required)
- scipy (required, for attention and sparse export)
- maturin 1.2.3 (build only)
- torch (optional, for GPU propagation)

## Workflow

- **Start application**: Runs `python3 run.py` — demo script (console output)
