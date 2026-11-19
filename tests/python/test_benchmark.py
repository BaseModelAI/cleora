"""pycleora.benchmark: time and quality of every algorithm, a failure in its own row."""

import pytest

from pycleora import benchmark, embed


@pytest.fixture
def labelled(small_graph):
    graph, _ = small_graph
    return graph, {name: i % 2 for i, name in enumerate(graph.entity_ids)}


def test_results_are_time_and_scores(labelled):
    """There was a memory column too. It showed 0.00 MB for calls that allocate megabytes:
    memory the process already holds is reused, and the peak of the process does not move."""
    graph, labels = labelled
    results = benchmark.benchmark_algorithms(graph, labels, {"cleora": lambda g: embed(g, 8, 2)}, num_runs=2)
    assert set(results["cleora"]) == {"avg_time", "std_time", "scores", "num_runs"}
    assert results["cleora"]["avg_time"] > 0 and results["cleora"]["num_runs"] == 2
    assert 0.0 <= results["cleora"]["scores"]["accuracy"] <= 1.0

    table = benchmark.format_benchmark_table(results).splitlines()
    assert table[0].split() == ["Algorithm", "Time", "(s)", "Accuracy"]
    name, seconds, accuracy = table[2].split()
    assert name == "cleora" and float(seconds) >= 0
    assert float(accuracy) == pytest.approx(results["cleora"]["scores"]["accuracy"], abs=1e-4)


def test_a_failing_algorithm_is_reported_in_its_row(labelled):
    graph, labels = labelled

    def broken(g):
        raise RuntimeError("no such luck")

    results = benchmark.benchmark_algorithms(graph, labels, {"broken": broken, "cleora": lambda g: embed(g, 8, 2)})
    assert results["broken"] == {"error": "RuntimeError: no such luck"}
    assert "accuracy" in results["cleora"]["scores"]  # the others are still measured
    table = benchmark.format_benchmark_table(results)
    assert "broken          ERROR: RuntimeError: no such luck" in table


def test_a_failing_dataset_is_reported_in_its_row():
    results = benchmark.benchmark_datasets(["no_such_dataset"], lambda g: embed(g, 8, 2))
    assert results["no_such_dataset"]["error"].startswith("ValueError: Unknown dataset: 'no_such_dataset'")
    assert "ERROR: ValueError: Unknown dataset" in benchmark.format_dataset_table(results)
