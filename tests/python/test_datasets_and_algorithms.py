"""Datasets say what they are, and the bundled baselines actually work."""

import os

import numpy as np
import pytest

from pycleora import SparseMatrix, algorithms, datasets
from pycleora.io_utils import from_arrays

REFLEXIVE = "complex::reflexive::node"


# --- datasets --------------------------------------------------------------------

def test_registry_marks_every_dataset():
    listed = datasets.list_datasets()
    names = [entry["name"] for entry in listed]
    assert len(names) == len(set(names))
    assert {"karate_club", "cora", "citeseer", "pubmed", "ppi", "email_eu_core"} <= set(names)
    for entry in listed:
        assert entry["synthetic"] is False, entry
        assert entry["nodes"] > 0 and entry["edges"] > 0


def test_karate_club_is_the_real_network():
    data = datasets.load_dataset("karate_club")
    assert data["synthetic"] is False and data["source"]
    assert data["num_nodes"] == 34 and data["num_edges"] == 78 and data["num_classes"] == 2
    graph = SparseMatrix.from_iterator(iter(data["edges"]), data["columns"])
    assert graph.num_entities == 34
    degrees = {}
    for edge in data["edges"]:
        for node in edge.split():
            degrees[node] = degrees.get(node, 0) + 1
    # the two best connected members of Zachary's club: the instructor and the administrator
    assert sorted(degrees.values())[-2:] == [16, 17]
    assert sorted(set(data["labels"].values())) == [0, 1]
    assert sum(1 for label in data["labels"].values() if label == data["labels"]["0"]) == 17


def test_names_that_used_to_be_synthetic_now_say_so():
    for name in ("flickr", "yelp", "dblp", "reddit"):
        with pytest.raises(ValueError, match="not available"):
            datasets.load_dataset(name)
    with pytest.raises(ValueError, match="Unknown dataset"):
        datasets.load_dataset("no_such_dataset")


def test_synthetic_graphs_are_labelled_synthetic(tmp_path, monkeypatch):
    monkeypatch.setenv("PYCLEORA_DATA", str(tmp_path))
    available = datasets.list_synthetic()
    assert "cora" in available
    data = datasets.load_synthetic("cora")
    assert data["synthetic"] is True
    assert "ynthetic" in data["name"] and "planted" in data["description"].lower() + data["name"].lower()
    assert data["num_nodes"] == 2708 and len(data["labels"]) == 2708
    again = datasets.load_synthetic("cora")  # second call comes from the cache
    assert again["edges"] == data["edges"] and again["synthetic"] is True
    with pytest.raises(ValueError):
        datasets.load_synthetic("karate_club")


def test_wikipedia_pos_keeps_the_edge_weights(tmp_path, monkeypatch):
    """The published POS.mat is weighted (0.5 to 1941.5, two entries in three are not 1);
    the loader returned its edges without the weights."""
    import scipy.io
    import scipy.sparse

    network = np.array([
        [3.0, 0.5, 0.0, 7.0],  # 3.0: a self-loop, which is not an edge
        [0.5, 0.0, 2.5, 0.0],
        [0.0, 2.5, 0.0, 0.0],
        [7.0, 0.0, 1.5, 0.0],  # 1.5: stored in one direction only
    ])
    groups = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    for filename in ("node2vec_POS.mat", "node2vec_Homo_sapiens.mat"):
        scipy.io.savemat(
            str(tmp_path / filename),
            {"network": scipy.sparse.csc_matrix(network), "group": scipy.sparse.csc_matrix(groups)},
        )
    monkeypatch.setattr(datasets, "_CACHE_DIR", str(tmp_path))

    def refuse(*args, **kwargs):
        raise AssertionError("the files are in the cache: nothing is to be downloaded")

    monkeypatch.setattr(datasets, "_download_with_progress", refuse)

    data = datasets.load_wikipedia_pos()
    assert list(data["edges"]) == ["0 1", "0 3", "1 2", "2 3"]
    assert data["weights"] == [0.5, 7.0, 2.5, 1.5]
    assert data["num_edges"] == 4 and data["multilabel"] is True

    # one weight per edge, in the order of the edges: the graph built from them is the file's
    graph = SparseMatrix.from_iterator(iter(data["edges"]), data["columns"], weights=data["weights"])
    adjacency = algorithms._graph_to_adjacency(graph).toarray()
    row = [graph.get_entity_index(str(node)) for node in range(4)]
    expected = np.array([
        [0.0, 0.5, 0.0, 7.0],
        [0.5, 0.0, 2.5, 0.0],
        [0.0, 2.5, 0.0, 1.5],
        [7.0, 0.0, 1.5, 0.0],
    ])
    reordered = adjacency[np.ix_(row, row)]
    np.testing.assert_allclose(reordered / reordered[0, 1], expected / expected[0, 1])

    # the protein network of the same collection has no weights to report
    assert "weights" not in datasets.load_ppi()


NETWORK = pytest.mark.skipif(
    os.environ.get("PYCLEORA_TEST_DOWNLOADS") != "1",
    reason="downloads public datasets; set PYCLEORA_TEST_DOWNLOADS=1 to run",
)


@NETWORK
@pytest.mark.parametrize("name", ["dolphins", "les_miserables", "football", "cora", "citeseer", "email_eu_core"])
def test_downloaded_datasets_match_the_registry(name):
    registry = {entry["name"]: entry for entry in datasets.list_datasets()}[name]
    data = datasets.load_dataset(name)
    assert data["synthetic"] is False and data["source"]
    assert data["num_nodes"] == registry["nodes"] and data["num_edges"] == registry["edges"]
    graph = SparseMatrix.from_iterator(iter(data["edges"]), data["columns"])
    assert graph.num_entities == registry["nodes"]
    if data.get("labels"):
        assert set(data["labels"]) <= set(graph.entity_ids)
        assert len(set(data["labels"].values())) == registry["classes"]


# --- baselines ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_communities():
    """Stochastic block model: 2 x 70 nodes, dense inside, sparse between."""
    rng = np.random.default_rng(1)
    n = 140
    block = np.arange(n) % 2
    src, dst = [], []
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < (0.25 if block[a] == block[b] else 0.02):
                src.append(a)
                dst.append(b)
    graph = from_arrays(np.array(src), np.array(dst))
    labels = np.array([block[int(name)] for name in graph.entity_ids])
    return graph, labels


def separation(embeddings, labels):
    """Leave-one-out nearest-centroid accuracy."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    correct = 0
    for i in range(len(labels)):
        mask = np.ones(len(labels), dtype=bool)
        mask[i] = False
        centroids = np.stack([embeddings[mask & (labels == c)].mean(axis=0) for c in (0, 1)])
        correct += int(np.argmax(centroids @ embeddings[i]) == labels[i])
    return correct / len(labels)


ALGORITHMS = [
    ("embed_prone", {}),
    ("embed_randne", {}),
    ("embed_hope", {}),
    ("embed_netmf", {}),
    ("embed_grarep", {}),
    ("embed_deepwalk", {"num_walks": 10, "walk_length": 20}),
    ("embed_node2vec", {"num_walks": 10, "walk_length": 20, "p": 0.5, "q": 2.0}),
]


@pytest.mark.parametrize("name,kwargs", ALGORITHMS)
def test_baselines_recover_planted_communities(two_communities, name, kwargs):
    graph, labels = two_communities
    function = getattr(algorithms, name)
    embeddings = function(graph, feature_dim=16, **kwargs)
    assert embeddings.shape == (graph.num_entities, 16)
    assert embeddings.dtype == np.float32 and np.isfinite(embeddings).all()
    # earlier versions of several baselines were at chance level (0.5) on such graphs
    assert separation(embeddings, labels) > 0.9, name
    np.testing.assert_array_equal(function(graph, feature_dim=16, **kwargs), embeddings)


def test_every_listed_algorithm_exists():
    listed = algorithms.list_algorithms()
    assert {entry["name"] for entry in listed} == {"prone", "randne", "hope", "netmf", "grarep", "deepwalk", "node2vec"}
    for entry in listed:
        assert callable(getattr(algorithms, entry["function"]))


def test_dense_baselines_refuse_graphs_that_are_too_large(two_communities):
    graph, _ = two_communities
    for name in ("embed_hope", "embed_netmf", "embed_grarep"):
        with pytest.raises(ValueError, match="max_nodes"):
            getattr(algorithms, name)(graph, feature_dim=8, max_nodes=50)


def test_adjacency_export_is_the_raw_graph():
    graph = from_arrays(["a", "b", "a"], ["b", "c", "c"], weights=[2.0, 1.0, 4.0])
    adjacency = algorithms._graph_to_adjacency(graph).toarray()
    assert not adjacency.diagonal().any()
    np.testing.assert_array_equal(adjacency, adjacency.T)
    row = {name: i for i, name in enumerate(graph.entity_ids)}
    assert adjacency[row["a"], row["c"]] == 4 * adjacency[row["b"], row["c"]]
    assert adjacency[row["a"], row["b"]] == 2 * adjacency[row["b"], row["c"]]
    with_loops = algorithms._graph_to_adjacency(graph, self_loops=True).toarray()
    assert with_loops.diagonal().all()
