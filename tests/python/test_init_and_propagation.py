"""Initial vectors and the propagation kernel."""

import numpy as np
import pytest

from pycleora import SparseMatrix
from pycleora.pycleora import entity_hash, l2_normalize_inplace

from conftest import dense

MASK = (1 << 64) - 1


def splitmix64(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def reference_init(entity_id: str, dim: int, seed: int) -> np.ndarray:
    seed_key = splitmix64(seed & MASK)
    values = []
    for column in range(dim):
        column_key = splitmix64(column ^ seed_key)
        bits = splitmix64(entity_hash(entity_id) ^ column_key) >> 40
        values.append(bits / 8388608.0 - 1.0)
    return np.array(values, dtype=np.float32)


def test_initial_vectors_follow_the_documented_formula(small_graph):
    graph, _ = small_graph
    for seed in (0, 1, -3, 2**40):
        init = graph.initialize_deterministically(12, seed)
        assert init.dtype == np.float32 and init.shape == (graph.num_entities, 12)
        for row in (0, 17, graph.num_entities - 1):
            np.testing.assert_array_equal(init[row], reference_init(graph.entity_ids[row], 12, seed))


def test_initial_vector_depends_on_the_name_only(small_graph):
    graph, _ = small_graph
    other = SparseMatrix.from_iterator(iter([f"{graph.entity_ids[9]} somebody_else"]), "complex::reflexive::x")
    np.testing.assert_array_equal(
        other.initialize_deterministically(20, 4)[0], graph.initialize_deterministically(20, 4)[9]
    )


@pytest.mark.parametrize("dim", [64, 256, 1024])
def test_initial_vectors_are_statistically_independent(dim):
    # 3.2 gave pairwise cosines with a standard deviation of 0.45 and an effective
    # rank of ~5 for every dimension
    n = 3000
    graph = SparseMatrix.from_iterator(iter(f"e{i} e{(i * 37) % n}" for i in range(n)), "complex::reflexive::x")
    init = graph.initialize_deterministically(dim).astype(np.float64)
    assert -1.0 <= init.min() and init.max() < 1.0
    assert abs(init.mean()) < 5e-3 and abs(init.var() - 1 / 3) < 5e-3

    unit = init / np.linalg.norm(init, axis=1, keepdims=True)
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, n, 50000), rng.integers(0, n, 50000)
    keep = a != b
    cosines = np.einsum("ij,ij->i", unit[a[keep]], unit[b[keep]])
    assert cosines.std() == pytest.approx(1 / np.sqrt(dim), rel=0.05)
    assert np.abs(cosines).max() < 8 / np.sqrt(dim)

    spectrum = np.linalg.svd(init, compute_uv=False) ** 2
    participation = spectrum.sum() ** 2 / (spectrum ** 2).sum()
    assert participation > 0.7 * min(dim, n)


def test_seeds_give_unrelated_vectors(small_graph):
    graph, _ = small_graph
    a = graph.initialize_deterministically(128, 0).astype(np.float64)
    b = graph.initialize_deterministically(128, 1).astype(np.float64)
    for shift in range(4):
        left, right = a[:, shift:], b[:, : 128 - shift]
        assert not np.array_equal(left, right)
        correlation = (left * right).mean() * 3
        assert abs(correlation) < 0.05


@pytest.mark.parametrize("propagation", ["left", "symmetric"])
def test_propagation_matches_dense_multiplication(small_graph, propagation, rng):
    graph, _ = small_graph
    x = rng.standard_normal((graph.num_entities, 7)).astype(np.float32)
    propagate = graph.left_markov_propagate if propagation == "left" else graph.symmetric_markov_propagate
    expected = dense(graph, propagation) @ x.astype(np.float64)
    result = propagate(x)
    assert result.dtype == np.float32 and result.flags.c_contiguous and result.flags.writeable
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)
    for workers in (1, 2, 7):
        np.testing.assert_array_equal(propagate(x, num_workers=workers), result)


def test_propagation_into_a_given_buffer(small_graph, rng):
    graph, _ = small_graph
    x = rng.standard_normal((graph.num_entities, 5)).astype(np.float32)
    expected = graph.left_markov_propagate(x)
    out = np.full_like(x, 123.0)
    returned = graph.left_markov_propagate(x, out=out)
    assert returned is out
    np.testing.assert_array_equal(out, expected)

    with pytest.raises(ValueError, match="share memory"):
        graph.left_markov_propagate(x, out=x)
    with pytest.raises(ValueError, match="shape"):
        graph.left_markov_propagate(x, out=np.zeros((3, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="C-contiguous"):
        graph.left_markov_propagate(x, out=np.asfortranarray(np.zeros_like(x)))
    read_only = np.zeros_like(x)
    read_only.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        graph.left_markov_propagate(x, out=read_only)


def test_propagating_selected_rows(small_graph, rng):
    graph, _ = small_graph
    n = graph.num_entities
    x = rng.standard_normal((n, 6)).astype(np.float32)
    rows = np.array([5, 0, 5, n - 1, 17])
    many = rng.integers(0, n, size=1000)
    for propagate in (graph.left_markov_propagate, graph.symmetric_markov_propagate):
        full = propagate(x)
        np.testing.assert_array_equal(propagate(x, rows=rows), full[rows])
        np.testing.assert_array_equal(propagate(x, rows=[5, 0]), full[[5, 0]])
        np.testing.assert_array_equal(propagate(x, rows=rows.astype(np.int32)), full[rows])
        np.testing.assert_array_equal(propagate(x, rows=many, num_workers=3), full[many])
        assert propagate(x, rows=[]).shape == (0, 6)
        out = np.empty((len(rows), 6), dtype=np.float32)
        assert propagate(x, rows=rows, out=out) is out
        np.testing.assert_array_equal(out, full[rows])

    for bad in ([n], [-1], [0, 10**12]):
        with pytest.raises(ValueError, match="out of range"):
            graph.left_markov_propagate(x, rows=bad)
    for bad in ("abc", 5, [1.5]):
        with pytest.raises(TypeError, match="rows"):
            graph.left_markov_propagate(x, rows=bad)
    with pytest.raises(ValueError, match="shape"):
        graph.left_markov_propagate(x, rows=[1, 2], out=np.zeros((3, 6), dtype=np.float32))


def test_propagation_accepts_any_memory_layout(small_graph, rng):
    # Fortran-ordered and strided arrays used to crash with a Rust panic
    graph, _ = small_graph
    wide = rng.standard_normal((graph.num_entities, 12)).astype(np.float32)
    for view in (np.asfortranarray(wide), wide[:, ::2], wide[:, ::-1], wide[::-1][::-1]):
        expected = graph.left_markov_propagate(np.ascontiguousarray(view))
        np.testing.assert_array_equal(graph.left_markov_propagate(view), expected)
        np.testing.assert_array_equal(
            graph.symmetric_markov_propagate(view), graph.symmetric_markov_propagate(np.ascontiguousarray(view))
        )


def test_propagation_rejects_wrong_input_clearly(small_graph):
    graph, _ = small_graph
    n = graph.num_entities
    with pytest.raises(TypeError, match="float32.*dtype float64"):
        graph.left_markov_propagate(np.zeros((n, 4)))
    with pytest.raises(TypeError, match="1-dimensional"):
        graph.left_markov_propagate(np.zeros(n, dtype=np.float32))
    with pytest.raises(TypeError, match="not a numpy array"):
        graph.left_markov_propagate([[0.0] * 4] * n)
    with pytest.raises(ValueError, match="rows but graph has"):
        graph.left_markov_propagate(np.zeros((n + 1, 4), dtype=np.float32))


def test_zero_width_and_isolated_cases():
    graph = SparseMatrix.from_iterator(iter(["a b", "c"]), "complex::reflexive::x")
    assert graph.left_markov_propagate(np.zeros((3, 0), dtype=np.float32)).shape == (3, 0)
    x = np.eye(3, dtype=np.float32)
    # "c" only co-occurs with itself: its row is a self-loop
    np.testing.assert_allclose(graph.left_markov_propagate(x)[2], [0.0, 0.0, 1.0])


def test_l2_normalisation(small_graph, rng):
    graph, _ = small_graph
    x = rng.standard_normal((graph.num_entities, 9)).astype(np.float32)
    x[4] = 0.0
    normalised = graph.l2_normalize(x)
    norms = np.linalg.norm(normalised, axis=1)
    np.testing.assert_allclose(np.delete(norms, 4), 1.0, rtol=1e-5)
    assert not normalised[4].any()
    assert not np.shares_memory(normalised, x)
    in_place = x.copy()
    l2_normalize_inplace(in_place)
    np.testing.assert_array_equal(in_place, normalised)


def test_entity_lookup(small_graph):
    graph, _ = small_graph
    ids = graph.entity_ids
    assert isinstance(ids, tuple) and graph.entity_ids is ids  # built once
    for index in (0, 11, len(ids) - 1):
        assert graph.get_entity_index(ids[index]) == index
    assert graph.get_entity_indices([ids[3], ids[0], ids[3]]) == [3, 0, 3]
    with pytest.raises(ValueError, match="not found"):
        graph.get_entity_index("missing")
    with pytest.raises(ValueError, match="not found"):
        graph.get_entity_indices([ids[0], "missing"])


def test_entity_ids_can_be_replaced():
    graph = SparseMatrix.from_iterator(iter(["a b", "b c"]), "complex::reflexive::x")
    assert graph.get_entity_index("b") == 1
    graph.entity_ids = ["x", "y", "z"]
    assert graph.entity_ids == ("x", "y", "z")
    assert graph.get_entity_index("y") == 1
    with pytest.raises(ValueError):
        graph.get_entity_index("b")
    with pytest.raises(ValueError, match="3 elements"):
        graph.entity_ids = ["only", "two"]
