#!/usr/bin/env python
"""Measures candidate exponents of the popularity prior on validation data.

    python benchmarks/popularity_exponent.py --data ~/cleora-bench [--tiers SM] [--datasets a,b] [--tasks lp,nc]

This is how the exponents of ``pycleora.scale_by_popularity`` were chosen; it is not a step
of the benchmark itself. For every graph and task, Cleora as the benchmark runs it for the
task, with each exponent of ``popularity.EXPONENTS`` in place of the configured one:

* ``lp``  the link split of seed 0; validation scores of the dot product (MRR against the
          sampled candidates, recall@10 among all nodes);
* ``nc``  graphs with node labels; the validation score of logistic regression (seed 0,
          three splits).

Test edges and test labels are never scored. The results go to ``<data>/popularity``;
``make_report.py`` shows them as the tables ``popularity_exponent`` and
``popularity_exponent_classification``.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cleora_bench import configurations, datasets, display, popularity, probes, runner, tasks  # noqa: E402


def measure(dataset, task, data, device, threads, timeout=3600):
    """Writes ``{exponent: {metric: validation score}}`` of a graph and task, unless it is
    there. Returns the path, or None for a graph without labels and the task ``nc``."""
    path = popularity.path(data, task, dataset)
    if os.path.exists(path):
        return path
    graph = datasets.load(dataset, os.path.join(data, "datasets"))
    if task == "nc" and "labels" not in graph:
        return None
    cache = os.path.join(data, "cache")
    if task == "lp":
        split = runner.link_split(graph, 0, cache)
        edges, weights = split["embed"], None
    else:
        edges, weights = graph["edges"], graph.get("weights")
    method, parameters = configurations.resolve(display.cleora_configuration("hyperedge", task))
    scores = {}
    for exponent in popularity.EXPONENTS:
        work = tempfile.mkdtemp(prefix=f"{dataset}_popularity_", dir=cache)
        try:
            vectors, record = runner.embed(method, dict(parameters, popularity=exponent), edges,
                                           graph["meta"]["num_nodes"], weights, display.MAIN_DIM, 0, threads, work, timeout)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        if vectors is None:
            raise RuntimeError(f"{dataset}: {record['reason']}")
        if task == "lp":
            validation = tasks.validation_scores(vectors, split, graph, device, 0)
            scores[str(exponent)] = {metric: validation[metric] for metric in popularity.METRICS["lp"]}
        else:
            chosen = tasks.node_classification(vectors, graph, device, (0, 1, 2), ("logistic",))["logistic"]["chosen"]
            scores[str(exponent)] = {"score": float(np.mean([split["validation_score"] for split in chosen]))}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(scores, handle, indent=1)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True, help="directory of the benchmark (datasets, caches)")
    parser.add_argument("--datasets", default=None, help="comma-separated names (default: all of --tiers)")
    parser.add_argument("--tiers", default="SM", help="dataset tiers when --datasets is not given")
    parser.add_argument("--tasks", default="lp,nc")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    args = parser.parse_args()
    data = os.path.abspath(os.path.expanduser(args.data))
    device = probes.pick_device()
    probes.torch.set_num_threads(min(args.threads, 4))  # the small probes that are fitted on the CPU
    for dataset in args.datasets.split(",") if args.datasets else datasets.names(args.tiers):
        for task in args.tasks.split(","):
            print(measure(dataset, task, data, device, args.threads) or f"{dataset}: no node labels", flush=True)
    print(popularity.link_table(data), popularity.classification_table(data), sep="\n\n")


if __name__ == "__main__":
    main()
