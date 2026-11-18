"""Edge / hyperedge weights are part of the matrix."""

import numpy as np
import pytest

import pycleora
from pycleora import SparseMatrix
from pycleora.io_utils import from_arrays, from_edge_list, from_numpy, from_pandas, from_scipy_sparse

from conftest import dense, lines_from_rows, random_rows, reference_graph

COLUMNS = "complex::reflexive::item"


def test_weights_match_reference(rng):
    rows = random_rows(rng, 250, [True], vocabulary=30, max_len=6)
    weights = rng.uniform(0.1, 9.0, size=len(rows)).astype(np.float32)
    graph = SparseMatrix.from_iterator(iter(lines_from_rows(rows)), COLUMNS, weights=weights)
    ids, raw, row_sums = reference_graph(rows, [(0, 0)], weights=weights)
    assert list(graph.entity_ids) == ids
    np.testing.assert_allclose(dense(graph), raw, rtol=1e-6)
    np.testing.assert_allclose(graph.entity_degrees, row_sums, rtol=1e-6)


def test_integer_weight_equals_repeating_the_line(rng):
    lines = lines_from_rows(random_rows(rng, 200, [True], vocabulary=25, max_len=5))
    counts = rng.integers(1, 5, size=len(lines))
    weighted = SparseMatrix.from_iterator(iter(lines), COLUMNS, weights=counts.astype(np.float64))
    repeated = SparseMatrix.from_iterator(
        iter([line for line, count in zip(lines, counts) for _ in range(count)]), COLUMNS
    )
    assert weighted.entity_ids == repeated.entity_ids
    np.testing.assert_allclose(dense(weighted), dense(repeated), rtol=1e-6)


def test_zero_weight_removes_the_line():
    lines = ["a b", "c d", "a c"]
    graph = SparseMatrix.from_iterator(iter(lines), COLUMNS, weights=[1.0, 0.0, 2.0])
    reference = SparseMatrix.from_iterator(iter(["a b", "a c"]), COLUMNS, weights=[1.0, 2.0])
    assert graph.entity_ids == reference.entity_ids == ("a", "b", "c")
    np.testing.assert_array_equal(dense(graph), dense(reference))


def test_common_factor_does_not_change_the_walk(rng):
    lines = lines_from_rows(random_rows(rng, 200, [True], vocabulary=25, max_len=5))
    weights = rng.uniform(0.5, 2.0, size=len(lines))
    a = SparseMatrix.from_iterator(iter(lines), COLUMNS, weights=weights)
    b = SparseMatrix.from_iterator(iter(lines), COLUMNS, weights=weights * 37.5)
    np.testing.assert_allclose(dense(a, "left"), dense(b, "left"), rtol=1e-5)
    np.testing.assert_allclose(dense(a, "symmetric"), dense(b, "symmetric"), rtol=1e-5)


def test_weights_change_the_embedding(rng):
    lines = lines_from_rows(random_rows(rng, 300, [True], vocabulary=40, max_len=4))
    random_weights = rng.uniform(0.01, 100.0, size=len(lines))
    _, uniform = pycleora.embed_weighted([(line, 1.0) for line in lines], COLUMNS, feature_dim=16, num_iterations=4)
    _, weighted = pycleora.embed_weighted(list(zip(lines, random_weights)), COLUMNS, feature_dim=16, num_iterations=4)
    plain = pycleora.embed(SparseMatrix.from_iterator(iter(lines), COLUMNS), 16, 4)
    np.testing.assert_array_equal(uniform, plain)
    assert np.abs(weighted - uniform).max() > 1e-2


def test_weight_column():
    lines = ["u1\tp1 p2\t2.5", "u2\tp2\t0.5", "u1\tp3\t1"]
    by_column = SparseMatrix.from_iterator(iter(lines), "user complex::product weight::w")
    by_argument = SparseMatrix.from_iterator(
        iter(["u1\tp1 p2", "u2\tp2", "u1\tp3"]), "user complex::product", weights=[2.5, 0.5, 1.0]
    )
    assert by_column.entity_ids == by_argument.entity_ids
    np.testing.assert_array_equal(dense(by_column), dense(by_argument))
    assert by_column.column_names == ["user", "product"]


def test_weight_column_in_files(tmp_path):
    path = tmp_path / "edges.tsv"
    path.write_text("a b\t3\nb c\t1.5\n")
    graph = SparseMatrix.from_files([str(path)], "complex::reflexive::x weight::w")
    reference = SparseMatrix.from_iterator(iter(["a b", "b c"]), "complex::reflexive::x", weights=[3.0, 1.5])
    np.testing.assert_array_equal(dense(graph), dense(reference))


@pytest.mark.parametrize("bad", [[1.0], [1.0, -1.0], [1.0, float("nan")], [1.0, float("inf")], [1.0, 1e12]])
def test_invalid_weights_are_rejected(bad):
    with pytest.raises(ValueError, match="weights"):
        SparseMatrix.from_iterator(iter(["a b", "b c"]), COLUMNS, weights=bad)


def test_weights_argument_and_column_are_exclusive():
    with pytest.raises(ValueError, match="not both"):
        SparseMatrix.from_iterator(iter(["a b\t1"]), "complex::reflexive::x weight::w", weights=[1.0])


def test_weight_types():
    lines = ["a b", "b c", "c d"]
    expected = dense(SparseMatrix.from_iterator(iter(lines), COLUMNS, weights=[1.0, 2.0, 3.0]))
    for weights in (
        (1, 2, 3),
        np.array([1, 2, 3]),
        np.array([1, 2, 3], dtype=np.float32),
        np.array([1.0, 2.0, 3.0])[::1],
        np.array([1.0, 9.0, 2.0, 9.0, 3.0])[::2],
    ):
        np.testing.assert_array_equal(dense(SparseMatrix.from_iterator(iter(lines), COLUMNS, weights=weights)), expected)
    with pytest.raises(TypeError):
        SparseMatrix.from_iterator(iter(lines), COLUMNS, weights="abc")


def test_helpers_carry_weights():
    pd = pytest.importorskip("pandas")
    scipy_sparse = pytest.importorskip("scipy.sparse")

    reference = dense(from_arrays(["a", "b", "a"], ["b", "c", "c"], weights=[2.0, 0.5, 4.0]))
    assert reference[0, 1] != reference[1, 2]

    frame = pd.DataFrame({"s": ["a", "b", "a", "x"], "t": ["b", "c", "c", None], "w": [2.0, 0.5, 4.0, 1.0]})
    np.testing.assert_array_equal(dense(from_pandas(frame, "s", "t", weight_col="w")), reference)

    np.testing.assert_array_equal(dense(from_edge_list([("a", "b", 2.0), ("b", "c", 0.5), ("a", "c", 4.0)])), reference)

    adjacency = np.array([[0, 2.0, 4.0], [2.0, 0, 0.5], [4.0, 0.5, 0]])
    names = ["a", "b", "c"]
    np.testing.assert_array_equal(dense(from_numpy(adjacency, names, use_weights=True)), reference)
    np.testing.assert_array_equal(
        dense(from_scipy_sparse(scipy_sparse.csr_matrix(adjacency), names, use_weights=True)), reference
    )
    # without use_weights the values are ignored, as before
    unweighted = dense(from_arrays(["a", "a", "b"], ["b", "c", "c"]))
    np.testing.assert_array_equal(dense(from_numpy(adjacency, names)), unweighted)
    np.testing.assert_array_equal(dense(from_scipy_sparse(scipy_sparse.csr_matrix(adjacency), names)), unweighted)
