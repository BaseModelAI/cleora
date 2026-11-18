"""SparseMatrix.from_cliques, isolated_entities and in-place normalised propagation."""

import numpy as np
import pytest

from pycleora import SparseMatrix
from pycleora import pycleora as core


def dense(graph, kind="left"):
    indptr, indices, data = graph.to_csr(kind)
    matrix = np.zeros((graph.num_entities, graph.num_entities))
    for row in range(graph.num_entities):
        matrix[row, indices[indptr[row]:indptr[row + 1]]] = data[indptr[row]:indptr[row + 1]]
    return matrix


def test_cliques_equal_lines_of_the_same_entities():
    cliques = [["a", "b", "c"], ["b", "d"], ["e"], ["a", "d", "e", "f"]]
    from_cliques = SparseMatrix.from_cliques(cliques)
    from_lines = SparseMatrix.from_iterator(iter(" ".join(c) for c in cliques), "complex::reflexive::CliqueNode")
    assert from_cliques.entity_ids == from_lines.entity_ids
    assert from_cliques.column_names == ["CliqueNode"]
    np.testing.assert_array_equal(dense(from_cliques, "raw"), dense(from_lines, "raw"))
    np.testing.assert_array_equal(from_cliques.entity_degrees, from_lines.entity_degrees)
    # any iterable of sequences works: a generator of tuples
    again = SparseMatrix.from_cliques(tuple(c) for c in cliques)
    np.testing.assert_array_equal(dense(again, "raw"), dense(from_cliques, "raw"))


def test_clique_names_may_contain_any_character():
    cliques = [["New York", "tab\there", "a,b"], ["a,b", "ünï cödé"]]
    graph = SparseMatrix.from_cliques(cliques)
    assert graph.entity_ids == ("New York", "tab\there", "a,b", "ünï cödé")
    assert [name for name, _ in graph.get_neighbors("a,b")] == ["New York", "tab\there", "a,b", "ünï cödé"]


def test_clique_weights_and_trimming():
    cliques = [["a", "b"], ["b", "c"], ["a", "c"]]
    weighted = SparseMatrix.from_cliques(cliques, weights=[2.0, 1.0, 0.0])
    repeated = SparseMatrix.from_cliques([["a", "b"], ["a", "b"], ["b", "c"]])
    np.testing.assert_allclose(dense(weighted, "raw"), dense(repeated, "raw"))
    big = [[f"n{i}" for i in range(30)]]
    trimmed = SparseMatrix.from_cliques(big, hyperedge_trim_n=4)
    full = SparseMatrix.from_cliques(big, hyperedge_trim_n=64)
    assert trimmed.num_edges < full.num_edges == 30 * 30


def test_cliques_reject_bad_input():
    with pytest.raises(ValueError, match="sequence of entity names"):
        SparseMatrix.from_cliques(["a b", "c d"])
    with pytest.raises(ValueError, match="empty"):
        SparseMatrix.from_cliques([["a", "b"], []])
    with pytest.raises(ValueError, match="strings"):
        SparseMatrix.from_cliques([["a", 3]])
    with pytest.raises(ValueError, match="weights"):
        SparseMatrix.from_cliques([["a", "b"]], weights=[1.0, 2.0])


@pytest.mark.parametrize("lines, expected", [
    (["0", "1 2"], ["0"]),
    (["0", "1 2", "0 1"], []),
    (["0", "1 2", "2", "3", "1", "0"], ["0", "3"]),
    (["0 1", "2"], ["2"]),
])
def test_isolated_entities(lines, expected):
    graph = SparseMatrix.from_iterator(iter(lines), "complex::reflexive::node")
    assert [graph.entity_ids[i] for i in graph.isolated_entities] == expected


def test_no_entity_of_a_two_column_graph_is_isolated():
    graph = SparseMatrix.from_iterator(iter(["u1\tp1", "u2\tp1"]), "user product")
    assert graph.isolated_entities == []


@pytest.mark.parametrize("kind", ["left", "symmetric"])
def test_inplace_iteration_matches_the_explicit_loop(kind):
    rng = np.random.default_rng(0)
    graph = SparseMatrix.from_iterator(
        iter(f"n{a} n{b}" for a, b in rng.integers(0, 80, (400, 2))), "complex::reflexive::n")
    step = graph.left_markov_propagate if kind == "left" else graph.symmetric_markov_propagate
    inplace = getattr(graph, f"{kind}_markov_propagate_normalized_inplace")
    start = graph.initialize_deterministically(16, seed=5)
    for rounds in (0, 1, 2, 5):
        expected = start.copy()
        for _ in range(rounds):
            expected = step(expected)
            core.l2_normalize_inplace(expected)
        x = start.copy()
        assert inplace(x, rounds) is None
        np.testing.assert_array_equal(x, expected)
    # the result does not depend on the number of threads
    one, many = start.copy(), start.copy()
    inplace(one, 3, num_workers=1)
    inplace(many, 3, num_workers=5)
    np.testing.assert_array_equal(one, many)


def test_inplace_iteration_validates_its_argument():
    graph = SparseMatrix.from_iterator(iter(["a b", "b c"]), "complex::reflexive::n")
    with pytest.raises(ValueError, match="rows"):
        graph.left_markov_propagate_normalized_inplace(np.zeros((5, 4), dtype=np.float32), 1)
    with pytest.raises(TypeError, match="float32"):
        graph.left_markov_propagate_normalized_inplace(np.zeros((3, 4), dtype=np.float64), 1)
    with pytest.raises(ValueError, match="C-contiguous"):
        graph.left_markov_propagate_normalized_inplace(np.zeros((3, 4), dtype=np.float32, order="F"), 1)
    frozen = np.zeros((3, 4), dtype=np.float32)
    frozen.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        graph.left_markov_propagate_normalized_inplace(frozen, 1)
