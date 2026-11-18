"""Same input, same output - bit for bit, for any number of worker threads."""

import numpy as np
import pytest

from pycleora import SparseMatrix, embed

from conftest import lines_from_rows, random_rows


def graph_bytes(graph):
    indptr, indices, data = graph.to_csr("raw")
    return (
        tuple(graph.entity_ids),
        indptr.tobytes(),
        indices.tobytes(),
        data.tobytes(),
        np.asarray(graph.entity_degrees).tobytes(),
    )


@pytest.fixture(scope="module")
def baskets():
    rng = np.random.default_rng(99)
    # many repeated pairs and many hyperedges above the trim threshold of 16
    rows = random_rows(rng, 4000, [True], vocabulary=500, max_len=45)
    return lines_from_rows(rows)


def test_graph_is_identical_for_any_number_of_workers(baskets):
    reference = graph_bytes(SparseMatrix.from_iterator(iter(baskets), "complex::reflexive::item", 16, 1))
    for workers in (2, 3, 8):
        for _ in range(3):
            graph = SparseMatrix.from_iterator(iter(baskets), "complex::reflexive::item", 16, workers)
            assert graph_bytes(graph) == reference


def test_weighted_bipartite_graph_is_identical_for_any_number_of_workers():
    rng = np.random.default_rng(5)
    rows = random_rows(rng, 3000, [True, True], vocabulary=200, max_len=30)
    lines = lines_from_rows(rows)
    weights = rng.uniform(0.01, 50.0, size=len(lines))
    reference = graph_bytes(SparseMatrix.from_iterator(iter(lines), "complex::a complex::b", 5, 1, weights=weights))
    for workers in (2, 7):
        graph = SparseMatrix.from_iterator(iter(lines), "complex::a complex::b", 5, workers, weights=weights)
        assert graph_bytes(graph) == reference


def test_embeddings_are_identical_for_any_number_of_workers(baskets):
    results = []
    for workers in (1, 2, 5):
        graph = SparseMatrix.from_iterator(iter(baskets), "complex::reflexive::item", 16, workers)
        results.append((
            graph.embed_fast(32, 5, num_workers=workers),
            embed(graph, 32, 5, num_workers=workers),
            embed(graph, 32, 4, propagation="symmetric", whiten=False, num_workers=workers,
                  callback=lambda i, e: None),
        ))
    for fast, default, looped in results[1:]:
        np.testing.assert_array_equal(fast, results[0][0])
        np.testing.assert_array_equal(default, results[0][1])
        np.testing.assert_array_equal(looped, results[0][2])


def test_repeated_runs_are_identical(baskets):
    graph = SparseMatrix.from_iterator(iter(baskets), "complex::reflexive::item")
    first = embed(graph, 24, 6)
    second = embed(graph, 24, 6)
    np.testing.assert_array_equal(first, second)


def test_order_of_lines_does_not_change_the_matrix(baskets):
    """Only the row order follows the input order; the matrix itself is the same."""
    shuffled = list(baskets)
    np.random.default_rng(1).shuffle(shuffled)
    a = SparseMatrix.from_iterator(iter(baskets), "complex::reflexive::item", 16, 2)
    b = SparseMatrix.from_iterator(iter(shuffled), "complex::reflexive::item", 16, 2)
    assert sorted(a.entity_ids) == sorted(b.entity_ids)
    order_a = np.argsort(a.entity_ids)
    order_b = np.argsort(b.entity_ids)
    from conftest import dense
    matrix_a = dense(a)[np.ix_(order_a, order_a)]
    matrix_b = dense(b)[np.ix_(order_b, order_b)]
    np.testing.assert_array_equal(matrix_a, matrix_b)
    # and so are the embeddings of every entity, because the initial vector of an
    # entity depends on its name only (up to the order in which neighbours are summed)
    emb_a = a.embed_fast(16, 3)[order_a]
    emb_b = b.embed_fast(16, 3)[order_b]
    np.testing.assert_allclose(emb_a, emb_b, atol=2e-6)
