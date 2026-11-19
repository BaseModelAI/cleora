"""mlp_classify and gcn_classify score the test entities; they do not choose the model with them."""

import numpy as np
import pytest

from pycleora import SparseMatrix, classify, embed
from pycleora.io_utils import from_arrays

CLASSIFIERS = [classify.mlp_classify, classify.gcn_classify]


@pytest.fixture(scope="module")
def noise():
    """A random graph whose labels have nothing to do with it."""
    rng = np.random.default_rng(5)
    lines = [f"n{int(rng.integers(0, 300))} n{int(rng.integers(0, 300))}" for _ in range(1500)]
    graph = SparseMatrix.from_iterator(iter(lines), "complex::reflexive::node")
    labels = {name: int(rng.integers(0, 2)) for name in graph.entity_ids}
    return graph, embed(graph, 16, 3), labels


@pytest.fixture(scope="module")
def communities():
    """Stochastic block model: 2 x 60 nodes, dense inside, sparse between."""
    rng = np.random.default_rng(1)
    block = np.arange(120) % 2
    pairs = [
        (a, b) for a in range(120) for b in range(a + 1, 120)
        if rng.random() < (0.25 if block[a] == block[b] else 0.02)
    ]
    graph = from_arrays(np.array([a for a, _ in pairs]), np.array([b for _, b in pairs]))
    labels = {name: int(block[int(name)]) for name in graph.entity_ids}
    return graph, embed(graph, 16, 3), labels


@pytest.mark.parametrize("classifier", CLASSIFIERS)
def test_test_labels_do_not_choose_the_model(noise, classifier):
    """The predictions for the test entities may not depend on their labels, so
    flipping those labels has to turn an accuracy of a into 1 - a.

    The checkpoint used to be the one with the best test accuracy: both runs then
    reported their best checkpoint and the two accuracies added up to more than 1."""
    graph, embeddings, labels = noise
    names = list(labels)
    # the test entities of seed=42, train_ratio=0.8: the end of the seed's permutation
    test = np.random.default_rng(42).permutation(len(names))[int(len(names) * 0.8):]
    flipped = dict(labels)
    for position in test:
        flipped[names[position]] = 1 - labels[names[position]]

    accuracy = classifier(graph, embeddings, labels, seed=42, train_ratio=0.8)["accuracy"]
    accuracy_flipped = classifier(graph, embeddings, flipped, seed=42, train_ratio=0.8)["accuracy"]
    assert accuracy + accuracy_flipped == pytest.approx(1.0)


@pytest.mark.parametrize("classifier", CLASSIFIERS)
def test_classifiers_learn_planted_communities(communities, classifier):
    graph, embeddings, labels = communities
    result = classifier(graph, embeddings, labels, seed=0)
    assert result["accuracy"] > 0.9 and result["macro_f1"] > 0.9
    assert (result["train_size"], result["test_size"], result["num_classes"]) == (96, 24, 2)
    assert classifier(graph, embeddings, labels, seed=0) == result  # the split comes from the seed


def test_validation_is_held_out_of_the_training_part():
    fit, validation, test = classify._split_labelled(np.random.default_rng(0), 100, 0.8)
    assert (len(fit), len(validation), len(test)) == (64, 16, 20)
    assert sorted(np.concatenate([fit, validation, test]).tolist()) == list(range(100))
    # the test part is the one there was before a validation part existed
    np.testing.assert_array_equal(test, np.random.default_rng(0).permutation(100)[80:])

    # the smallest training part that can be divided
    fit, validation, test = classify._split_labelled(np.random.default_rng(0), 4, 0.5)
    assert (len(fit), len(validation), len(test)) == (1, 1, 2)


@pytest.mark.parametrize("classifier", CLASSIFIERS)
def test_a_training_part_too_small_to_divide_is_an_error(communities, classifier):
    graph, embeddings, labels = communities
    few = dict(list(labels.items())[:4])
    with pytest.raises(ValueError, match="at least 2 training entities"):
        classifier(graph, embeddings, few, train_ratio=0.3)
