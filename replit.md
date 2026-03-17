# pycleora

A Python library for efficient, scalable graph embeddings using a sparse hypergraph structure and Markov propagation. Built with Rust (via PyO3/maturin) for high performance.

## Overview

**pycleora** is a Python library exposing Rust-based graph embedding functionality. It is NOT a web application — it is a library/SDK that users import in their own Python code.

## Project Structure

```
pycleora/           - Python package (includes the compiled .so extension)
  __init__.py       - Public API (SparseMatrix, embed_using_baseline_cleora)
  pycleora.pyi      - Type stubs
  pycleora.cpython-312-x86_64-linux-gnu.so  - Compiled Rust extension (built with maturin)
src/                - Rust source code
  lib.rs            - PyO3 bindings and SparseMatrix Python methods
  configuration.rs  - Column/field parsing
  embedding.rs      - Markov propagation (left, symmetric)
  entity.rs         - Entity hashing
  pipeline.rs       - Graph building from files or iterator
  sparse_matrix.rs  - Core sparse matrix data structure
  sparse_matrix_builder.rs
examples/           - Python usage examples
files/samples/      - Sample edge list TSV files
legacy/             - Legacy Rust CLI version of Cleora
run.py              - Demo/showcase script for Replit
```

## Build System

- **Language**: Rust + Python (PyO3 bindings)
- **Build tool**: maturin (>=1.2.3)
- **Python**: 3.12
- **Rust**: stable

## Setup

To rebuild the library after Rust source changes:
```bash
pip install maturin==1.2.3
maturin build --release -i python3
pip install target/wheels/pycleora-*.whl --force-reinstall
cp ~/.pythonlibs/lib/python3.12/site-packages/pycleora/pycleora.cpython-312-x86_64-linux-gnu.so pycleora/
```

The compiled `.so` file must be copied into the local `pycleora/` directory because that directory takes precedence in Python's import path over site-packages.

## Usage

```python
from pycleora import SparseMatrix, embed_using_baseline_cleora
import numpy as np

# From file
graph = SparseMatrix.from_files(["edges.tsv"], "complex::reflexive::product")

# From iterator
edges = ["user1 item1 item2", "user2 item2 item3"]
graph = SparseMatrix.from_iterator(iter(edges), "complex::reflexive::product")

# Embed
embeddings = embed_using_baseline_cleora(graph, feature_dim=64, iter=3)
```

## Workflow

- **Start application**: Runs `python3 run.py` — a demo script showing the library works
- No web server / no port needed (console output type)

## Dependencies

- numpy
- maturin (build only)
