#!/usr/bin/env python
"""Downloads the benchmark datasets from their original sources and stores them in the
canonical form of ``cleora_bench.datasets``.

    python benchmarks/prepare_datasets.py --data ~/cleora-bench --tiers SM

Datasets that are already prepared are skipped. A dataset whose source no longer yields
the graph the benchmark was run on fails with a message; the others are still prepared.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cleora_bench import datasets  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True, help="directory for datasets, caches and results")
    parser.add_argument("--datasets", default=None, help="comma-separated names (default: all of --tiers)")
    parser.add_argument("--tiers", default="S", help="S (small), M (medium), L (large); e.g. SM")
    args = parser.parse_args()
    root = os.path.join(os.path.abspath(os.path.expanduser(args.data)), "datasets")
    failed = []
    for name in args.datasets.split(",") if args.datasets else datasets.names(args.tiers):
        started = time.time()
        try:
            meta = datasets.load(name, root)["meta"]
        except Exception as error:  # report and go on with the other datasets; the exit code says it happened
            failed.append(name)
            print(f"FAILED {name}: {type(error).__name__}: {error}", flush=True)
            continue
        labels = f"{meta['num_classes']} {'labels' if meta.get('multilabel') else 'classes'}" if meta["num_classes"] else "no labels"
        print(f"{name:18} {meta['num_nodes']:>9,} nodes {meta['num_edges']:>11,} edges  {labels:12} "
              f"[{time.time() - started:.0f}s]", flush=True)
    if failed:
        sys.exit(f"Could not prepare: {', '.join(failed)}")


if __name__ == "__main__":
    main()
