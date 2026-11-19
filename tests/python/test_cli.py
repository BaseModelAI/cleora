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
