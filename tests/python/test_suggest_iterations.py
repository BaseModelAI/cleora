"""SparseMatrix.suggest_iterations: the rule behind ``num_iterations="auto"``."""

import numpy as np
import pytest

from pycleora import SparseMatrix


def ring(nodes, step=1):
    source = np.arange(nodes)
    return source, (source + step) % nodes


def circulant(nodes, steps):
    """Every node linked to the ``steps`` next ones: ``2 * steps`` neighbours each."""
    parts = [ring(nodes, step) for step in range(1, steps + 1)]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def test_denser_graphs_get_fewer_iterations():
    sparse = SparseMatrix.from_edges(*ring(200), columns="src dst")
    dense = SparseMatrix.from_edges(*circulant(200, 8), columns="src dst")
    # 2 neighbours: 16 / sqrt(2) = 11.3; 16 neighbours: 16 / 4 = 4
    assert sparse.suggest_iterations() == 11
    assert dense.suggest_iterations() == 4


def test_purposes():
    graph = SparseMatrix.from_edges(*circulant(200, 2), columns="src dst")  # 4 neighbours
    assert graph.suggest_iterations(purpose="similarity") == 6       # 12 / 2
    assert graph.suggest_iterations(purpose="general") == 8          # 16 / 2
    assert graph.suggest_iterations(purpose="classification") == 12  # 24 / 2
    assert graph.suggest_iterations() == graph.suggest_iterations("left", "general")
    with pytest.raises(ValueError, match="purpose"):
        graph.suggest_iterations(purpose="fun")


def test_a_reflexive_column_is_a_lazy_walk():
    pair = SparseMatrix.from_edges(*ring(200), columns="src dst")
    lazy = SparseMatrix.from_edges(*ring(200), columns="complex::reflexive::node")
    # it moves half of the time: twice the iterations
    assert lazy.suggest_iterations() == round(2 * 16 / np.sqrt(2))
    assert lazy.suggest_iterations() == pytest.approx(2 * pair.suggest_iterations(), abs=1)
    # and however dense, it needs 8 iterations to forget its initial vectors (0.5 ** 8 = 1 / 256)
    dense = SparseMatrix.from_edges(*circulant(400, 150), columns="complex::reflexive::node")
    assert dense.suggest_iterations(purpose="similarity") == 8
    assert dense.suggest_iterations(purpose="classification") == 8


def test_very_dense_plain_graphs_get_two_iterations():
    dense = SparseMatrix.from_edges(*circulant(400, 150), columns="src dst")  # 300 neighbours
    assert dense.suggest_iterations() == 2


def test_symmetric_propagation_gets_more_iterations():
    graph = SparseMatrix.from_edges(*ring(200), columns="src dst")
    assert graph.suggest_iterations("symmetric") == round(1.5 * 16 / np.sqrt(2))
    assert graph.suggest_iterations("left") == graph.suggest_iterations()


def test_edge_weights_do_not_change_the_number_of_neighbours():
    source, target = ring(200)
    plain = SparseMatrix.from_edges(source, target, columns="complex::reflexive::node")
    weighted = SparseMatrix.from_edges(source, target, columns="complex::reflexive::node",
                                       weights=np.linspace(0.5, 4.0, len(source)))
    assert weighted.suggest_iterations() == plain.suggest_iterations()


def test_unknown_propagation_is_rejected():
    graph = SparseMatrix.from_edges(*ring(10), columns="src dst")
    with pytest.raises(ValueError, match="propagation"):
        graph.suggest_iterations("sideways")
