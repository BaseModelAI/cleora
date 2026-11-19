"""Minimal check that an installed pycleora wheel works. CI runs it against the
freshly built wheel on every platform it can execute on (the full test suite in
tests/python runs on Linux)."""

import numpy as np

import pycleora
from pycleora import SparseMatrix, embed

lines = [f"n{i % 50} n{(i * 7 + 1) % 50} n{(i * 13 + 2) % 50}" for i in range(200)]
graph = SparseMatrix.from_iterator(iter(lines), "complex::reflexive::node")
assert graph.num_entities == 50, graph

default = embed(graph, 16)
whitened = embed(graph, 16, 4, whiten=True)
assert default.shape == whitened.shape == (50, 16)
assert np.isfinite(default).all() and np.isfinite(whitened).all()
np.testing.assert_allclose(np.linalg.norm(default, axis=1), 1.0, atol=1e-5)
np.testing.assert_array_equal(default, embed(graph, 16, graph.suggest_iterations(), num_workers=1))

print(f"pycleora {pycleora.__version__}: ok")
