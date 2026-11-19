"""The ``pycleora embed`` command line."""

import numpy as np
import pytest

from pycleora import SparseMatrix, cli, embed

LINES = [f"n{i % 60} n{(i * 7 + 1) % 60}" for i in range(300)]


def run_embed(tmp_path, monkeypatch, *arguments):
    edges = tmp_path / "edges.txt"
    edges.write_text("\n".join(LINES) + "\n")
    output = tmp_path / "out.npz"
    monkeypatch.setattr("sys.argv", ["pycleora", "embed", "-i", str(edges), "-o", str(output), "-d", "16", *arguments])
    cli.main()
    with np.load(output, allow_pickle=True) as archive:
        return archive["embeddings"], list(archive["entity_ids"])


def test_embed_defaults_are_those_of_the_library(tmp_path, monkeypatch):
    graph = SparseMatrix.from_iterator(iter(LINES), "complex::reflexive::node")
    vectors, entity_ids = run_embed(tmp_path, monkeypatch)
    assert entity_ids == list(graph.entity_ids)
    np.testing.assert_array_equal(vectors, embed(graph, 16))


def test_embed_takes_iterations_propagation_and_whitening(tmp_path, monkeypatch):
    graph = SparseMatrix.from_iterator(iter(LINES), "complex::reflexive::node")
    vectors, _ = run_embed(tmp_path, monkeypatch, "-n", "3", "-p", "symmetric")
    np.testing.assert_array_equal(vectors, embed(graph, 16, 3, propagation="symmetric"))
    vectors, _ = run_embed(tmp_path, monkeypatch, "--iterations", "auto", "--whiten")
    np.testing.assert_array_equal(vectors, embed(graph, 16, "auto", whiten=True))


@pytest.mark.parametrize("value", ["many", "-1", "2.5"])
def test_embed_rejects_a_bad_number_of_iterations(tmp_path, monkeypatch, capsys, value):
    with pytest.raises(SystemExit):
        run_embed(tmp_path, monkeypatch, f"--iterations={value}")
    assert "--iterations" in capsys.readouterr().err


def test_node2vec_is_not_deepwalk(tmp_path, monkeypatch):
    """--algorithm node2vec ran with p = q = 1: DeepWalk under another name."""
    from pycleora.algorithms import embed_deepwalk, embed_node2vec

    graph = SparseMatrix.from_iterator(iter(LINES), "complex::reflexive::node")
    vectors, _ = run_embed(tmp_path, monkeypatch, "-a", "node2vec")
    np.testing.assert_array_equal(vectors, embed_node2vec(graph, 16, p=1.0, q=0.5))
    assert not np.array_equal(vectors, embed_deepwalk(graph, 16))

    # --p is the return parameter; -p stays the propagation
    vectors, _ = run_embed(tmp_path, monkeypatch, "-a", "node2vec", "--p", "0.5", "--q", "2", "-p", "symmetric")
    np.testing.assert_array_equal(vectors, embed_node2vec(graph, 16, p=0.5, q=2.0))
    vectors, _ = run_embed(tmp_path, monkeypatch, "-a", "node2vec", "--p", "1", "--q", "1", "--seed", "3")
    np.testing.assert_array_equal(vectors, embed_deepwalk(graph, 16, seed=3))


def test_similar_ranks_the_entities_of_a_small_graph(tmp_path, monkeypatch, capsys):
    """12 entities at the default 256 dimensions: two cliques of 6 joined by one edge."""
    lines = [f"{clique}{i} {clique}{j}" for clique in "ab" for i in range(6) for j in range(i + 1, 6)]
    edges = tmp_path / "edges.txt"
    edges.write_text("\n".join(lines + ["a0 b0"]) + "\n")

    def similar(*arguments):
        monkeypatch.setattr("sys.argv", ["pycleora", "similar", "-i", str(edges), "-e", "a3", *arguments])
        cli.main()
        rows = [line.split() for line in capsys.readouterr().out.splitlines()]
        return [(name, float(similarity.split("=")[1])) for name, similarity in rows]

    found = similar()
    names = [name for name, _ in found]
    similarity = dict(found)
    assert len(found) == 10 and "a3" not in names
    # its own clique first, then the end of the bridge, then the rest of the other clique
    assert set(names[:5]) == {"a0", "a1", "a2", "a4", "a5"} and names[5] == "b0"
    assert min(similarity[name] for name in names[:5]) > 0.9
    assert 0.5 > similarity["b0"] > max(similarity[name] for name in names[6:]) + 0.05

    # asked for more than there are: the 11 others, not the query with a similarity of -1
    everyone = similar("-k", "50")
    assert len(everyone) == 11 and "a3" not in [name for name, _ in everyone]
    assert min(value for _, value in everyone) > 0


def test_benchmark_command_on_a_small_graph(monkeypatch, capsys):
    """Karate club has 34 members: the whitened row needs fewer dimensions than asked
    for, node2vec is not a second DeepWalk row, and no memory is reported."""
    import warnings

    from pycleora import benchmark
    from pycleora.algorithms import embed_node2vec

    seen = {}
    measure = benchmark.benchmark_algorithms

    def spy(graph, labels, algorithms, **kwargs):
        seen.update(graph=graph, algorithms=algorithms)
        return measure(graph, labels, algorithms, **kwargs)

    monkeypatch.setattr(benchmark, "benchmark_algorithms", spy)
    monkeypatch.setattr("sys.argv", ["pycleora", "benchmark", "--dataset", "karate_club", "--dim", "64"])
    with warnings.catch_warnings():
        # as an error, the warning would be what the row of cleora-whitened reports
        warnings.filterwarnings("error", message="Whitening cannot work")
        cli.main()
    output = capsys.readouterr().out

    assert "Using 8 dimensions instead of 64" in output
    assert "ERROR" not in output and "Memory" not in output
    table = output[output.index("Algorithm"):].splitlines()
    assert table[0].split() == ["Algorithm", "Time", "(s)", "Accuracy"]
    assert [line.split()[0] for line in table[2:]] == [
        "cleora", "cleora-whitened", "deepwalk", "node2vec", "prone", "randne",
    ]
    assert all(len(line.split()) == 3 for line in table[2:])

    graph, algorithms = seen["graph"], seen["algorithms"]
    node2vec = algorithms["node2vec"](graph)
    np.testing.assert_array_equal(node2vec, embed_node2vec(graph, 8, p=1.0, q=0.5))
    assert not np.array_equal(node2vec, algorithms["deepwalk"](graph))
