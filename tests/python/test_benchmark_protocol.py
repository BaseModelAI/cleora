"""The evaluation protocol of the benchmark suite: what is ranked against what. It runs
on PyTorch, as the suite does, and is skipped where PyTorch is not installed."""

import pathlib
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "benchmarks"))

from cleora_bench import tasks  # noqa: E402


def test_retrieval_from_an_official_list_of_pairs_queries_each_distinct_pair_from_both_ends():
    # 60 nodes in 30 pairs with one direction per pair: the nearest node is the partner
    num_nodes = 60
    vectors = torch.eye(30).repeat_interleave(2, dim=0)
    pairs = np.arange(num_nodes, dtype=np.int64).reshape(30, 2)
    test = np.concatenate([pairs[:10], pairs[:1]])  # the first pair is listed twice
    graph = {"meta": {"num_nodes": num_nodes, "official_link_split": True}, "edges": pairs[10:],
             "lp_valid_pos": np.zeros((0, 2), dtype=np.int64), "lp_test_pos": test}
    split = {"test_pos": test, "embedded": np.ones(num_nodes, dtype=bool)}

    result = tasks._retrieval(vectors, split, graph, seed=0)

    assert result["num_queries"] == 20
    assert result["all_nodes_mrr"] == 1.0
    assert result["recall@10"] == 1.0  # 0.5 for the repeated pair if it counted as two targets
