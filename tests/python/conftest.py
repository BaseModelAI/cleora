"""Shared helpers: a slow, obviously-correct reference implementation of the graph
construction, and generators of random hypergraphs."""

import numpy as np
import pytest

from pycleora import SparseMatrix
from pycleora.pycleora import entity_hash


def reference_graph(rows, relations, trim=16, weights=None):
    """Brute-force construction of the co-occurrence matrix.

    ``rows[i][c]`` is the list of entity names of entity column ``c`` in hyperedge
    ``i``; ``relations`` lists pairs of column indices (``(c, c)`` is reflexive).
    Returns ``(entity_ids, raw_matrix, row_sums)``.
    """
    ids, index = [], {}
    used = [w is None or w > 0 for w in (weights if weights is not None else [None] * len(rows))]
    for row, keep in zip(rows, used):
        if not keep:
            continue
        for field in row:
            for token in field:
                if token not in index:
                    index[token] = len(ids)
                    ids.append(token)
    n = len(ids)
    occurrence = np.zeros(n, dtype=np.int64)
    row_sums = np.zeros(n)
    raw = np.zeros((n, n))
    deferred = []

    def element(x, y, value):
        if x == y:
            raw[x, x] += 2 * value
        else:
            raw[x, y] += value
            raw[y, x] += value

    for r, (row, keep) in enumerate(zip(rows, used)):
        if not keep:
            continue
        weight = 1.0 if weights is None else float(weights[r])
        for a, b in relations:
            side_a = [index[t] for t in row[a]]
            side_b = [index[t] for t in row[b]]
            for x in side_a:
                occurrence[x] += len(side_b)
                row_sums[x] += weight / len(side_b)
            for y in side_b:
                occurrence[y] += len(side_a)
                row_sums[y] += weight / len(side_a)
            value = weight / (len(side_a) * len(side_b))
            if len(side_a) > trim and len(side_b) > trim:
                deferred.append((side_a, side_b, value))
            else:
                for x in side_a:
                    for y in side_b:
                        element(x, y, value)

    def split(nodes):
        ranked = sorted(nodes, key=lambda i: (-occurrence[i], entity_hash(ids[i])))
        return ranked[:trim], ranked[trim:]

    for side_a, side_b, value in deferred:
        a_high, a_low = split(side_a)
        b_high, b_low = split(side_b)
        for xs, ys in ((a_high, b_high), (a_high, b_low), (a_low, b_high)):
            for x in xs:
                for y in ys:
                    element(x, y, value)
    return ids, raw, row_sums


def dense(graph: SparseMatrix, markov_type: str = "raw") -> np.ndarray:
    """The graph's matrix as a dense float64 array."""
    from scipy.sparse import csr_matrix

    indptr, indices, data = graph.to_csr(markov_type)
    n = graph.num_entities
    return csr_matrix((data.astype(np.float64), indices, indptr), shape=(n, n)).toarray()


def lines_from_rows(rows):
    """Text form of tokenised rows: fields separated by tabs, entities by spaces."""
    return ["\t".join(" ".join(field) for field in row) for row in rows]


def random_rows(rng, n_rows, columns, vocabulary=40, max_len=6, prefix_per_column=True):
    """Random hyperedges. ``columns`` lists, per entity column, whether it is complex."""
    rows = []
    for _ in range(n_rows):
        row = []
        for c, is_complex in enumerate(columns):
            length = int(rng.integers(1, max_len + 1)) if is_complex else 1
            prefix = f"c{c}_" if prefix_per_column else "e"
            row.append([f"{prefix}{int(rng.integers(0, vocabulary))}" for _ in range(length)])
        rows.append(row)
    return rows


@pytest.fixture
def rng():
    return np.random.default_rng(20261118)


@pytest.fixture
def small_graph():
    rng = np.random.default_rng(7)
    lines = [f"n{int(rng.integers(0, 120))} n{int(rng.integers(0, 120))}" for _ in range(600)]
    return SparseMatrix.from_iterator(iter(lines), "complex::reflexive::node"), lines
