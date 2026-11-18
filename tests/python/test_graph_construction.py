"""The sparse matrix built by the Rust core against a brute-force reference."""

import numpy as np
import pytest

from pycleora import SparseMatrix

from conftest import dense, lines_from_rows, random_rows, reference_graph

CASES = [
    # (columns definition, complex flag per entity column, relations, shared vocabulary)
    ("complex::reflexive::item", [True], [(0, 0)], True),
    ("user product", [False, False], [(0, 1)], False),
    ("user complex::product", [False, True], [(0, 1)], False),
    ("complex::a complex::b", [True, True], [(0, 1)], False),
    ("src dst", [False, False], [(0, 1)], True),  # the same names in both columns
    ("user complex::reflexive::product", [False, True], [(0, 1), (1, 1)], False),
    ("user complex::product brand", [False, True, False], [(0, 1), (0, 2), (1, 2)], False),
    ("complex::reflexive::a complex::reflexive::b", [True, True], [(0, 0), (0, 1), (1, 1)], True),
]


@pytest.mark.parametrize("columns,complexity,relations,shared", CASES)
@pytest.mark.parametrize("num_workers", [1, 3])
def test_matches_reference(rng, columns, complexity, relations, shared, num_workers):
    rows = random_rows(rng, 300, complexity, vocabulary=35, max_len=5, prefix_per_column=not shared)
    graph = SparseMatrix.from_iterator(iter(lines_from_rows(rows)), columns, 16, num_workers)
    ids, raw, row_sums = reference_graph(rows, relations)

    assert list(graph.entity_ids) == ids
    np.testing.assert_allclose(dense(graph, "raw"), raw, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(graph.entity_degrees, row_sums, rtol=1e-6)
    np.testing.assert_allclose(dense(graph, "left"), raw / row_sums[:, None], rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(
        dense(graph, "symmetric"), raw / np.sqrt(np.outer(row_sums, row_sums)), rtol=1e-5, atol=1e-9
    )


@pytest.mark.parametrize("columns,complexity,relations,shared", [CASES[0], CASES[3], CASES[5]])
@pytest.mark.parametrize("trim", [1, 3, 8])
@pytest.mark.parametrize("num_workers", [1, 4])
def test_oversized_hyperedges_match_reference(rng, columns, complexity, relations, shared, trim, num_workers):
    # hyperedges much longer than the trim threshold, with repeated entities
    rows = random_rows(rng, 120, complexity, vocabulary=60, max_len=25, prefix_per_column=not shared)
    weights = rng.uniform(0.2, 3.0, size=len(rows))
    graph = SparseMatrix.from_iterator(
        iter(lines_from_rows(rows)), columns, trim, num_workers, weights=weights
    )
    ids, raw, row_sums = reference_graph(rows, relations, trim=trim, weights=weights.astype(np.float32))

    assert list(graph.entity_ids) == ids
    np.testing.assert_allclose(dense(graph, "raw"), raw, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(graph.entity_degrees, row_sums, rtol=1e-5)


def test_matrix_is_symmetric_with_sorted_rows(rng):
    rows = random_rows(rng, 400, [True], vocabulary=80, max_len=30)
    graph = SparseMatrix.from_iterator(iter(lines_from_rows(rows)), "complex::reflexive::x", 4)
    matrix = dense(graph, "raw")
    np.testing.assert_array_equal(matrix, matrix.T)
    indptr, indices, _ = graph.to_csr("raw")
    for row in range(graph.num_entities):
        columns = indices[indptr[row]:indptr[row + 1]]
        assert np.all(np.diff(columns) > 0)


def test_untrimmed_rows_are_stochastic(rng):
    rows = random_rows(rng, 300, [True], vocabulary=50, max_len=10)
    graph = SparseMatrix.from_iterator(iter(lines_from_rows(rows)), "complex::reflexive::x", 16)
    np.testing.assert_allclose(dense(graph, "left").sum(axis=1), 1.0, rtol=1e-5)


def test_coordinate_export_agrees_with_csr(small_graph):
    graph, _ = small_graph
    for kind in ("left", "symmetric", "raw"):
        rows, cols, values, n, m = graph.to_sparse_csr(kind)
        assert n == m == graph.num_entities
        assert len(values) == graph.num_edges
        rebuilt = np.zeros((n, n))
        rebuilt[rows, cols] = values
        np.testing.assert_array_equal(rebuilt, dense(graph, kind).astype(np.float32))
    np.testing.assert_array_equal(graph.to_sparse_csr()[2], graph.to_sparse_csr("left")[2])
    with pytest.raises(ValueError, match="markov_type"):
        graph.to_sparse_csr("bogus")


def test_column_masks():
    graph = SparseMatrix.from_iterator(iter(["u1\tp1 p2", "u2\tp2"]), "user complex::product")
    ids = np.array(graph.entity_ids)
    assert set(ids[graph.get_entity_column_mask("user")]) == {"u1", "u2"}
    assert set(ids[graph.get_entity_column_mask("product")]) == {"p1", "p2"}
    assert graph.column_names == ["user", "product"]
    with pytest.raises(ValueError, match="not found"):
        graph.get_entity_column_mask("brand")

    # a single reflexive column used to give an all-False mask
    reflexive = SparseMatrix.from_iterator(iter(["a b", "b c"]), "complex::reflexive::item")
    assert reflexive.get_entity_column_mask("item").all()


def test_repr_and_sizes():
    graph = SparseMatrix.from_iterator(iter(["a b c", "c d"]), "complex::reflexive::item")
    assert repr(graph) == "SparseMatrix(entities=4, edges=12, columns=('item', 'item'))"
    assert len(graph) == graph.num_entities == 4
    assert graph.num_edges == 12
    bipartite = SparseMatrix.from_iterator(iter(["u\tp"]), "user product")
    assert repr(bipartite) == "SparseMatrix(entities=2, edges=2, columns=('user', 'product'))"
    three = SparseMatrix.from_iterator(iter(["u\tp\tb"]), "user product brand")
    assert repr(three) == "SparseMatrix(entities=3, edges=6, columns=('user', 'product', 'brand'))"


def test_neighbors(small_graph):
    graph, _ = small_graph
    entity = graph.entity_ids[3]
    neighbors = dict(graph.get_neighbors(entity))
    row = dense(graph, "left")[3]
    expected = {graph.entity_ids[j]: row[j] for j in np.flatnonzero(row)}
    assert neighbors.keys() == expected.keys()
    for name, value in expected.items():
        assert neighbors[name] == pytest.approx(value, rel=1e-6)
    with pytest.raises(ValueError, match="not found"):
        graph.get_neighbors("no such entity")
