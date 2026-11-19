#!/usr/bin/env python
"""Measures candidate exponents of the popularity prior on validation edges.

    python benchmarks/popularity_exponent.py --data ~/cleora-bench [--tiers SM] [--datasets a,b]

This is how ``pycleora.DEFAULT_POPULARITY_EXPONENT`` was chosen; it is not a step of the
benchmark itself. For every graph: the link split of seed 0, Cleora as the benchmark runs
it for similarity with each exponent of ``popularity.EXPONENTS`` in place of the default,
and the validation scores of the dot product. Test edges are never scored. The results go
to ``<data>/popularity``; ``make_report.py`` shows them as the table ``popularity_exponent``.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cleora_bench import configurations, datasets, display, popularity, probes, runner, tasks  # noqa: E402


def measure(dataset, data, device, threads, timeout=3600):
    """Writes ``{exponent: {metric: validation score}}`` of a graph, unless it is there."""
    path = popularity.path(data, dataset)
    if os.path.exists(path):
        return path
    graph = datasets.load(dataset, os.path.join(data, "datasets"))
    cache = os.path.join(data, "cache")
    split = runner.link_split(graph, 0, cache)
    method, parameters = configurations.resolve(display.cleora_configuration("hyperedge", "lp"))
    scores = {}
    for exponent in popularity.EXPONENTS:
        work = tempfile.mkdtemp(prefix=f"{dataset}_popularity_", dir=cache)
        try:
            vectors, record = runner.embed(method, dict(parameters, popularity=exponent), split["embed"],
                                           graph["meta"]["num_nodes"], None, display.MAIN_DIM, 0, threads, work, timeout)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        if vectors is None:
            raise RuntimeError(f"{dataset}: {record['reason']}")
        validation = tasks.validation_scores(vectors, split, graph, device, 0)
        scores[str(exponent)] = {metric: validation[metric] for metric in popularity.METRICS}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(scores, handle, indent=1)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True, help="directory of the benchmark (datasets, caches)")
    parser.add_argument("--datasets", default=None, help="comma-separated names (default: all of --tiers)")
    parser.add_argument("--tiers", default="SM", help="dataset tiers when --datasets is not given")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    args = parser.parse_args()
    data = os.path.abspath(os.path.expanduser(args.data))
    device = probes.pick_device()
    for dataset in args.datasets.split(",") if args.datasets else datasets.names(args.tiers):
        print(measure(dataset, data, device, args.threads), flush=True)
    print(popularity.markdown(os.path.join(data, popularity.DIRECTORY)))


if __name__ == "__main__":
    main()
