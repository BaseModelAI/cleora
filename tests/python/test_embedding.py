"""embed(), whitening and the helpers built on top of them, against NumPy references."""

import numpy as np
import pytest

import pycleora
from pycleora import SparseMatrix, embed, whiten_embeddings
from pycleora.pycleora import center_project, center_project_inplace, column_mean_cov

from conftest import dense

COLUMNS = "complex::reflexive::node"


@pytest.fixture(scope="module")
def graph():
    rng = np.random.default_rng(11)
    # 4 loose communities, so the embedding has real structure
    lines = []
    for _ in range(2500):
        community = int(rng.integers(0, 4))
        a = community * 100 + int(rng.integers(0, 100))
        b = (community if rng.random() < 0.9 else int(rng.integers(0, 4))) * 100 + int(rng.integers(0, 100))
        lines.append(f"n{a} n{b}")
    return SparseMatrix.from_iterator(iter(lines), COLUMNS)


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-10)


def reference_whiten(x):
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(axis=0)
    covariance = centered.T @ centered / (len(x) - 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    pivot = np.argmax(np.abs(vectors), axis=0)
    vectors = vectors * np.sign(vectors[pivot, np.arange(vectors.shape[1])])
    return centered @ (vectors / np.sqrt(np.maximum(values, 1e-10)))


@pytest.mark.parametrize("propagation", ["left", "symmetric"])
def test_classic_embedding_matches_numpy(graph, propagation):
    transition = dense(graph, propagation)
    expected = graph.initialize_deterministically(24, 3).astype(np.float64)
    for _ in range(5):
        expected = l2(transition @ expected)
    result = embed(graph, 24, 5, propagation=propagation, seed=3, whiten=False)
    assert result.dtype == np.float32 and result.shape == (graph.num_entities, 24)
    np.testing.assert_allclose(result, expected, atol=2e-5)
    np.testing.assert_array_equal(result, graph.embed_fast(24, 5, propagation=propagation, seed=3))


def test_default_embedding_matches_numpy(graph):
    transition = dense(graph, "left")
    expected = graph.initialize_deterministically(16).astype(np.float64)
    for _ in range(4):
        expected = reference_whiten(l2(transition @ expected))
    result = embed(graph, 16, 4)
    np.testing.assert_allclose(result, expected, atol=2e-3)


def test_default_embedding_is_whitened(graph):
    embeddings = embed(graph, 32, 6).astype(np.float64)
    np.testing.assert_allclose(embeddings.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.cov(embeddings.T), np.eye(32), atol=1e-4)


def test_rust_and_python_loops_agree(graph):
    """The same algorithm runs in Rust (fast path) and as a Python loop over the Rust
    kernels (when a callback or initial embeddings are given)."""
    seen = []
    looped = embed(graph, 20, 4, whiten=False, callback=lambda i, e: seen.append((i, e)))
    np.testing.assert_array_equal(looped, embed(graph, 20, 4, whiten=False))
    assert [i for i, _ in seen] == [0, 1, 2, 3]
    np.testing.assert_array_equal(seen[-1][1], looped)
    assert not np.array_equal(seen[0][1], seen[1][1])  # callbacks get their own copies

    init = graph.initialize_deterministically(20)
    np.testing.assert_array_equal(
        embed(graph, 20, 4, whiten=False, initial_embeddings=init), embed(graph, 20, 4, whiten=False)
    )
    with_residual = embed(graph, 20, 4, whiten=False, residual_weight=0.25)
    np.testing.assert_allclose(
        embed(graph, 20, 4, whiten=False, residual_weight=0.25, callback=lambda i, e: None),
        with_residual, atol=1e-6,
    )
    assert np.abs(with_residual - embed(graph, 20, 4, whiten=False)).max() > 1e-3


def test_residual_matches_numpy(graph):
    transition = dense(graph, "left")
    expected = graph.initialize_deterministically(12).astype(np.float64)
    for _ in range(3):
        expected = l2(0.7 * (transition @ expected) + 0.3 * expected)
    np.testing.assert_allclose(embed(graph, 12, 3, whiten=False, residual_weight=0.3), expected, atol=2e-5)


def test_initial_embeddings_are_not_modified(graph):
    rng = np.random.default_rng(0)
    for init in (
        rng.standard_normal((graph.num_entities, 8)).astype(np.float32),
        rng.standard_normal((graph.num_entities, 8)),  # float64
        np.asfortranarray(rng.standard_normal((graph.num_entities, 8)).astype(np.float32)),
    ):
        before = init.copy()
        for whiten in (True, False):
            result = embed(graph, 8, 3, initial_embeddings=init, whiten=whiten)
            assert result.dtype == np.float32 and result.shape == init.shape
            assert not np.shares_memory(result, init)
        np.testing.assert_array_equal(init, before)
    with pytest.raises(ValueError, match="rows but graph has"):
        embed(graph, 8, 2, initial_embeddings=np.zeros((3, 8), dtype=np.float32))


def test_zero_iterations_return_the_initial_vectors(graph):
    init = graph.initialize_deterministically(8, 5)
    np.testing.assert_array_equal(embed(graph, 8, 0, seed=5), init)
    np.testing.assert_array_equal(embed(graph, 8, 0, seed=5, whiten=False), init)


def test_convergence_stops_early(graph):
    full, iterations = graph.embed_fast_convergence(8, 50, convergence_threshold=1e-2)
    assert 1 < iterations < 50
    np.testing.assert_array_equal(full, graph.embed_fast(8, iterations))
    np.testing.assert_array_equal(embed(graph, 8, 50, whiten=False, convergence_threshold=1e-2), full)
    _, never = graph.embed_fast_convergence(8, 6, convergence_threshold=0.0)
    assert never == 6


def test_other_normalisations(graph):
    transition = dense(graph, "left")
    init = graph.initialize_deterministically(10).astype(np.float64)

    expected = init
    for _ in range(3):
        expected = transition @ expected
        expected = expected / np.maximum(np.abs(expected).sum(axis=1, keepdims=True), 1e-10)
    np.testing.assert_allclose(embed(graph, 10, 3, normalization="l1", whiten=False), expected, atol=1e-5)

    expected = init
    for _ in range(3):
        expected = transition @ expected
    np.testing.assert_allclose(embed(graph, 10, 3, normalization="none", whiten=False), expected, atol=1e-5)

    spectral = embed(graph, 10, 2, normalization="spectral", whiten=False)
    assert spectral.shape == (graph.num_entities, 10) and np.isfinite(spectral).all()


def test_argument_validation(graph):
    with pytest.raises(ValueError, match="propagation"):
        embed(graph, 8, 2, propagation="sideways")
    with pytest.raises(ValueError, match="normalization"):
        embed(graph, 8, 2, normalization="l3")
    with pytest.raises(ValueError, match="negative"):
        embed(graph, 8, -1)
    with pytest.raises(ValueError, match="'auto'"):
        embed(graph, 8, "many")
    assert embed(graph, 8, "auto", whiten=False).shape == (graph.num_entities, 8)


def test_whitening_more_dimensions_than_entities_is_reported():
    """Whitening N points in N - 1 or more dimensions makes them all equidistant."""
    rng = np.random.default_rng(31)
    lines = [f"n{i % 20} n{int(rng.integers(0, 20))} n{int(rng.integers(0, 20))}" for i in range(60)]
    graph = SparseMatrix.from_iterator(iter(lines), COLUMNS)
    assert graph.num_entities == 20

    with pytest.warns(RuntimeWarning, match="20 entities and the embedding 32 dimensions"):
        useless = embed(graph, 32, 5)
    unit = l2(useless.astype(np.float64))
    cosines = (unit @ unit.T)[~np.eye(20, dtype=bool)]
    np.testing.assert_allclose(cosines, -1 / 19, atol=1e-4)  # a regular simplex: no information

    for function in (
        lambda: pycleora.embed_multiscale(graph, 32, [1, 2]),
        lambda: pycleora.embed_with_attention(graph, 32, 2),
        lambda: pycleora.embed_directed(lines, COLUMNS, feature_dim=32, num_iterations=2),
    ):
        with pytest.warns(RuntimeWarning, match="Whitening cannot work"):
            function()

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fine = embed(graph, 8, 5)  # enough entities for 8 dimensions
        embed(graph, 32, 5, whiten=False)  # no whitening, no problem
    unit = l2(fine.astype(np.float64))
    assert (unit @ unit.T)[~np.eye(20, dtype=bool)].std() > 0.1


# --- whitening -----------------------------------------------------------------

def test_whitening_matches_numpy():
    rng = np.random.default_rng(3)
    mixing = rng.standard_normal((12, 12))
    data = (rng.standard_normal((4000, 12)) @ mixing + rng.standard_normal(12) * 5).astype(np.float32)
    result = whiten_embeddings(data)
    np.testing.assert_allclose(result, reference_whiten(data), atol=5e-4)
    np.testing.assert_allclose(np.cov(result.astype(np.float64).T), np.eye(12), atol=1e-4)
    for workers in (1, 2, 5):
        np.testing.assert_array_equal(whiten_embeddings(data, num_workers=workers), result)


def test_whitening_leaves_the_input_alone_unless_asked():
    rng = np.random.default_rng(4)
    data = rng.standard_normal((500, 6)).astype(np.float32)
    before = data.copy()
    result = whiten_embeddings(data)
    np.testing.assert_array_equal(data, before)
    assert not np.shares_memory(result, data)

    in_place = whiten_embeddings(data, inplace=True)
    assert in_place is data
    np.testing.assert_array_equal(in_place, result)


def test_whitening_accepts_any_layout_and_dtype():
    rng = np.random.default_rng(5)
    data = rng.standard_normal((300, 5)).astype(np.float32)
    expected = whiten_embeddings(data)
    for variant in (np.asfortranarray(data), data.astype(np.float64), np.hstack([data, data])[:, :5], data.tolist()):
        np.testing.assert_allclose(whiten_embeddings(variant), expected, atol=1e-5)
    strided = np.repeat(data, 2, axis=1)[:, ::2]
    assert not strided.flags.c_contiguous
    np.testing.assert_array_equal(whiten_embeddings(strided), expected)


def test_whitening_can_reduce_the_dimension():
    rng = np.random.default_rng(6)
    data = (rng.standard_normal((2000, 10)) * np.arange(10, 0, -1)).astype(np.float32)
    reduced = whiten_embeddings(data, n_components=3)
    assert reduced.shape == (2000, 3)
    np.testing.assert_allclose(reduced, whiten_embeddings(data)[:, :3], atol=1e-5)
    # in-place is impossible when the shape changes: the input must stay intact
    before = data.copy()
    assert whiten_embeddings(data, n_components=3, inplace=True).shape == (2000, 3)
    np.testing.assert_array_equal(data, before)


def test_whitening_edge_cases():
    single = np.ones((1, 4), dtype=np.float32)
    np.testing.assert_array_equal(whiten_embeddings(single), single)
    with pytest.raises(ValueError, match="2-dimensional"):
        whiten_embeddings(np.ones(4, dtype=np.float32))
    constant = np.ones((50, 3), dtype=np.float32)
    assert np.isfinite(whiten_embeddings(constant)).all()


def test_covariance_and_projection_kernels():
    rng = np.random.default_rng(8)
    data = rng.standard_normal((1234, 7)).astype(np.float32)
    mean, covariance = column_mean_cov(data)
    assert mean.dtype == covariance.dtype == np.float64
    np.testing.assert_allclose(mean, data.astype(np.float64).mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(covariance, np.cov(data.astype(np.float64).T), atol=1e-10)
    np.testing.assert_array_equal(covariance, covariance.T)
    for workers in (1, 2, 3, 8):
        other_mean, other_covariance = column_mean_cov(data, workers)
        np.testing.assert_array_equal(other_mean, mean)
        np.testing.assert_array_equal(other_covariance, covariance)
    fortran_mean, fortran_covariance = column_mean_cov(np.asfortranarray(data))
    np.testing.assert_array_equal(fortran_covariance, covariance)

    shift = rng.standard_normal(7).astype(np.float32)
    transform = rng.standard_normal((7, 4)).astype(np.float32)
    projected = center_project(data, shift, transform)
    np.testing.assert_allclose(projected, (data - shift) @ transform, atol=1e-4)
    np.testing.assert_array_equal(center_project(np.asfortranarray(data), shift, np.asfortranarray(transform)), projected)

    square = rng.standard_normal((7, 7)).astype(np.float32)
    target = data.copy()
    center_project_inplace(target, shift, square)
    np.testing.assert_array_equal(target, center_project(data, shift, square))
    with pytest.raises(ValueError, match="C-contiguous"):
        center_project_inplace(np.asfortranarray(data), shift, square)
    with pytest.raises(ValueError, match="square"):
        center_project_inplace(data.copy(), shift, transform)
    with pytest.raises(ValueError):
        center_project(data, shift[:3], transform)


# --- inductive / streaming -------------------------------------------------------

def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.mark.parametrize("whiten", [True, False])
def test_inductive_update_keeps_known_entities_where_they_are(whiten):
    rng = np.random.default_rng(21)
    existing = [f"n{int(rng.integers(0, 60))} n{int(rng.integers(0, 60))}" for _ in range(400)]
    graph = SparseMatrix.from_iterator(iter(existing), COLUMNS)
    embeddings = embed(graph, 16, 4, whiten=whiten)
    new_edges = ["n1 fresh_a", "n2 fresh_b", "n3 fresh_b", "fresh_a fresh_chain", "island_1 island_2"]

    updated_graph, updated = pycleora.embed_inductive(graph, embeddings, existing, new_edges, COLUMNS)
    assert updated.dtype == np.float32 and updated.shape == (graph.num_entities + 5, 16)
    old_rows = updated_graph.get_entity_indices(list(graph.entity_ids))
    np.testing.assert_array_equal(updated[old_rows], embeddings)

    def vector(name, source=updated, source_graph=updated_graph):
        return source[source_graph.get_entity_index(name)].astype(np.float64)

    # one known neighbour: the new entity lands on it; a chain is reached in the next round
    assert cosine(vector("fresh_a"), vector("n1")) > 0.9999
    assert cosine(vector("fresh_chain"), vector("n1")) > 0.9999
    # two known neighbours: in between them
    middle = vector("n2") + vector("n3")
    assert cosine(vector("fresh_b"), middle) > 0.9999
    # no path to anything known: no information, zero vector
    assert not vector("island_1").any() and not vector("island_2").any()
    if not whiten:
        np.testing.assert_allclose(np.linalg.norm(vector("fresh_b")), 1.0, atol=1e-5)


def test_inductive_update_keeps_a_trained_model_valid():
    """A classifier trained before the update must still work after it."""
    rng = np.random.default_rng(22)
    nodes = 300
    label = np.arange(nodes) % 3

    def edge():
        a = int(rng.integers(0, nodes))
        same = np.flatnonzero(label == label[a])
        b = int(rng.choice(same)) if rng.random() < 0.9 else int(rng.integers(0, nodes))
        return f"v{a} v{b}"

    existing = [edge() for _ in range(3000)]
    graph = SparseMatrix.from_iterator(iter(existing), COLUMNS)
    embeddings = embed(graph, 32, 5)
    classes = np.array([label[int(name[1:])] for name in graph.entity_ids])
    centroids = np.stack([embeddings[classes == c].mean(axis=0) for c in range(3)])

    def accuracy(matrix, matrix_graph):
        truth = np.array([label[int(name[1:])] for name in matrix_graph.entity_ids])
        return float((np.argmax(matrix @ centroids.T, axis=1) == truth).mean())

    before = accuracy(embeddings, graph)
    assert before > 0.9
    more = [edge() for _ in range(300)]
    updated_graph, updated = pycleora.embed_inductive(graph, embeddings, existing, more, COLUMNS)
    assert accuracy(updated, updated_graph) >= before - 0.02


def test_inductive_modes_and_validation():
    existing = ["a b", "b c", "c d", "d a"]
    graph = SparseMatrix.from_iterator(iter(existing), COLUMNS)
    embeddings = embed(graph, 4, 3, whiten=False)
    retrained_graph, retrained = pycleora.embed_inductive(
        graph, embeddings, existing, ["d e"], COLUMNS, num_iterations=3, mode="retrain", whiten=False
    )
    assert retrained.shape == (5, 4) and np.isfinite(retrained).all()
    again = pycleora.embed_inductive(
        graph, embeddings, existing, ["d e"], COLUMNS, num_iterations=3, mode="retrain", whiten=False
    )[1]
    np.testing.assert_array_equal(retrained, again)  # no hidden randomness

    with pytest.raises(ValueError, match="Unknown mode"):
        pycleora.embed_inductive(graph, embeddings, existing, ["d e"], COLUMNS, mode="magic")
    with pytest.raises(ValueError, match="rows but graph has"):
        pycleora.embed_inductive(graph, embeddings[:2], existing, ["d e"], COLUMNS)


def test_streaming_keeps_coordinates_stable():
    rng = np.random.default_rng(23)
    batches = [[f"n{int(rng.integers(0, 80 + 40 * b))} n{int(rng.integers(0, 80 + 40 * b))}" for _ in range(500)]
               for b in range(3)]
    snapshots = []
    graph, embeddings = pycleora.embed_streaming(
        batches, COLUMNS, feature_dim=12, num_iterations=4,
        batch_callback=lambda i, g, e: snapshots.append((g.entity_ids, e.copy())),
    )
    assert len(snapshots) == 3 and embeddings.shape == (graph.num_entities, 12)
    first_ids, first = snapshots[0]
    np.testing.assert_array_equal(embeddings[graph.get_entity_indices(list(first_ids))], first)
    np.testing.assert_array_equal(snapshots[0][1], embed(SparseMatrix.from_iterator(iter(batches[0]), COLUMNS), 12, 4))

    refreshed_graph, refreshed = pycleora.embed_streaming(batches, COLUMNS, feature_dim=12, num_iterations=4, refresh_every=1)
    assert refreshed.shape == embeddings.shape and np.isfinite(refreshed).all()
    assert not np.array_equal(refreshed, embeddings)
    with pytest.raises(ValueError, match="refresh_every"):
        pycleora.embed_streaming(batches, COLUMNS, refresh_every=0)


def test_alignment_recovers_a_rotation():
    rng = np.random.default_rng(24)
    old = rng.standard_normal((200, 6)).astype(np.float32)
    rotation, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    ids = [f"e{i}" for i in range(200)]
    rotated = (old @ rotation).astype(np.float32)
    # the new embedding has extra entities and a different order
    order = rng.permutation(200)
    new_ids = [ids[i] for i in order] + ["extra"]
    new = np.vstack([rotated[order], rng.standard_normal((1, 6)).astype(np.float32)])
    aligned = pycleora._align_to_previous(new, new_ids, old, ids)
    np.testing.assert_allclose(aligned[:200], old[order], atol=1e-4)


# --- similarity search / link prediction ----------------------------------------------

def test_find_most_similar_matches_brute_force(graph):
    embeddings = embed(graph, 16, 4)
    unit = l2(embeddings.astype(np.float64))
    query = graph.entity_ids[42]
    similarities = unit @ unit[42]
    similarities[42] = -np.inf
    expected = np.argsort(-similarities)[:7]
    found = pycleora.find_most_similar(graph, embeddings, query, top_k=7)
    assert [hit["index"] for hit in found] == expected.tolist()
    assert [hit["entity_id"] for hit in found] == [graph.entity_ids[i] for i in expected]
    np.testing.assert_allclose([hit["similarity"] for hit in found], similarities[expected], atol=1e-5)

    with_self = pycleora.find_most_similar(graph, embeddings, query, top_k=3, exclude_self=False)
    assert with_self[0]["entity_id"] == query and with_self[0]["similarity"] == pytest.approx(1.0, abs=1e-5)
    assert pycleora.find_most_similar(graph, embeddings, query, top_k=0) == []
    assert len(pycleora.find_most_similar(graph, embeddings, query, top_k=10**6)) == graph.num_entities
    with pytest.raises(ValueError, match="not found"):
        pycleora.find_most_similar(graph, embeddings, "nobody")


@pytest.mark.parametrize("exclude_existing", [True, False])
def test_predict_links_matches_brute_force(graph, exclude_existing):
    embeddings = embed(graph, 16, 4)
    unit = l2(embeddings.astype(np.float64))
    scores = unit @ unit.T
    np.fill_diagonal(scores, -np.inf)
    if exclude_existing:
        scores[dense(graph) > 0] = -np.inf

    sources = [graph.entity_ids[i] for i in (3, 77, 150)]
    rows = [3, 77, 150]
    flat = [(scores[r, c], r, c) for r in rows for c in range(graph.num_entities) if np.isfinite(scores[r, c])]
    flat.sort(reverse=True)
    expected = flat[:9]
    found = pycleora.predict_links(graph, embeddings, top_k=9, exclude_existing=exclude_existing, source_entities=sources)
    assert [(p["source"], p["target"]) for p in found] == [(graph.entity_ids[r], graph.entity_ids[c]) for _, r, c in expected]
    np.testing.assert_allclose([p["score"] for p in found], [s for s, _, _ in expected], atol=1e-5)

    everything = pycleora.predict_links(graph, embeddings, top_k=5, exclude_existing=exclude_existing)
    best = np.sort(scores[np.isfinite(scores)])[::-1][:5]
    np.testing.assert_allclose([p["score"] for p in everything], best, atol=1e-5)
    assert pycleora.predict_links(graph, embeddings, top_k=0) == []


# --- the other embedding variants --------------------------------------------------------

def test_directed_embedding_follows_the_arrows():
    lines = ["a b", "a c", "b c"]
    graph, embeddings = pycleora.embed_directed(lines, COLUMNS, feature_dim=6, num_iterations=1, whiten=False)
    init = graph.initialize_deterministically(6).astype(np.float64)
    row = {name: i for i, name in enumerate(graph.entity_ids)}
    # "a" points to b and c, "b" to c, "c" to nothing
    np.testing.assert_allclose(embeddings[row["a"]], l2((init[row["b"]] + init[row["c"]])[None])[0], atol=1e-5)
    np.testing.assert_allclose(embeddings[row["b"]], l2(init[row["c"]][None])[0], atol=1e-5)
    assert not embeddings[row["c"]].any()

    hyper = ["x y z w", "w x"]
    graph, embeddings = pycleora.embed_directed(hyper, COLUMNS, feature_dim=2, num_iterations=3)
    assert embeddings.shape == (4, 2) and np.isfinite(embeddings).all()


def test_multiscale_attention_and_estimator(graph):
    multi = pycleora.embed_multiscale(graph, 8, [1, 3], whiten=False)
    np.testing.assert_array_equal(multi[:, :8], embed(graph, 8, 1, whiten=False))
    np.testing.assert_array_equal(multi[:, 8:], embed(graph, 8, 3, whiten=False))

    attention = pycleora.embed_with_attention(graph, 8, 3)
    assert attention.shape == (graph.num_entities, 8) and np.isfinite(attention).all()
    np.testing.assert_array_equal(attention, pycleora.embed_with_attention(graph, 8, 3))

    estimator = pycleora.CleoraEmbedder(feature_dim=4, num_iterations=3, columns=COLUMNS)
    lines = [f"n{i % 20} n{(i * 3 + 1) % 20} n{(i * 7 + 2) % 20}" for i in range(60)]
    fitted = estimator.fit_transform(lines)
    np.testing.assert_array_equal(fitted, embed(SparseMatrix.from_iterator(iter(lines), COLUMNS), 4, 3))


def test_heterogeneous_graph_relations_carry_weights():
    from pycleora.hetero import HeteroGraph

    def build(weights):
        hetero = HeteroGraph()
        hetero.add_node_type("user")
        hetero.add_node_type("item")
        hetero.add_edge_type(
            "buys", "user", "item", [("u1", "i1"), ("u2", "i1"), ("u2", "i2"), ("u3", "i2")], weights=weights
        )
        hetero.add_edge_type("follows", "user", "user", [("u1", "u2"), ("u2", "u3")])
        return hetero

    for combine, width in (("concat", 16), ("mean", 8)):
        graphs, embeddings, combined = build(None).embed_per_relation(
            feature_dim=8, num_iterations=3, combine=combine, whiten=False
        )
        assert set(graphs) == set(embeddings) == {"buys", "follows"}
        assert combined.shape == (5, width) and np.isfinite(combined).all()

    plain = build(None).embed_per_relation(8, 3, whiten=False)[1]["buys"]
    weighted = build([5.0, 1.0, 1.0, 5.0]).embed_per_relation(8, 3, whiten=False)[1]["buys"]
    assert np.abs(plain - weighted).max() > 1e-3
