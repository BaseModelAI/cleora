# pycleora

A Python library for efficient, scalable graph embeddings using a sparse hypergraph structure and Markov propagation. Built with Rust (via PyO3/maturin) for high performance on CPU.

## Overview

**pycleora** is a Python library exposing Rust-based graph embedding functionality. It is NOT a web application — it is a library/SDK that users import in their own Python code.

## Project Structure

```
pycleora/           - Python package (includes the compiled .so extension)
  __init__.py       - Public API (SparseMatrix, embed, embed_multiscale, find_most_similar, cosine_similarity)
  pycleora.pyi      - Type stubs
  pycleora.cpython-312-x86_64-linux-gnu.so  - Compiled Rust extension
src/                - Rust source code
  lib.rs            - PyO3 bindings, SparseMatrix Python methods, entity lookup, repr
  configuration.rs  - Column/field parsing
  embedding.rs      - Optimized Markov propagation (adaptive serial/parallel per row)
  entity.rs         - Entity hashing (XxHash64)
  pipeline.rs       - Graph building from files or iterator, multi-format parser (TSV/CSV/space)
  sparse_matrix.rs  - Core sparse matrix data structure
  sparse_matrix_builder.rs - Safe parallel node indexing (no unsafe ptr::write)
examples/           - Python usage examples
files/samples/      - Sample edge list TSV files
legacy/             - Legacy Rust CLI version of Cleora
run.py              - Full feature demo
```

## Build System

- **Language**: Rust + Python (PyO3 bindings)
- **Build tool**: maturin 1.2.3
- **Python**: 3.12
- **Rust**: stable

## Setup / Rebuild

```bash
pip install maturin==1.2.3
maturin build --release -i python3
pip install target/wheels/pycleora-*.whl --force-reinstall
cp ~/.pythonlibs/lib/python3.12/site-packages/pycleora/pycleora.cpython-312-x86_64-linux-gnu.so pycleora/
```

## Key Features

- **embed()** - Configurable embedding with propagation type (left/symmetric), normalization (l2/l1/spectral/none), progress callbacks
- **embed_multiscale()** - Multi-scale embedding concatenating representations at different propagation depths
- **find_most_similar()** - Cosine similarity search across all entities
- **get_entity_index() / get_entity_indices()** - Fast entity lookup
- **Pickle support** - Full serialization/deserialization of graph structures
- **Multi-format input** - TSV, CSV, TXT files + Python iterators
- **Proper error handling** - All errors raise Python ValueError/RuntimeError instead of crashing

## Workflow

- **Start application**: Runs `python3 run.py` — demo script (console output)

## Dependencies

- numpy
- maturin (build only)
