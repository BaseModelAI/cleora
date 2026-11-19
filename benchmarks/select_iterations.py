#!/usr/bin/env python
"""Chooses Cleora's iteration counts from the sweep, on validation scores.

    python benchmarks/select_iterations.py --data ~/cleora-bench

Reads ``<data>/sweep`` (written by ``run_benchmark.py --results sweep`` with the
configurations of ``cleora_bench.configurations.sweep``) and writes
``<data>/selected_iterations.json``, which ``--methods cleora_tuned`` and the report use.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cleora_bench import configurations, selection  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    data = os.path.abspath(os.path.expanduser(args.data))
    for dataset, tasks in sorted(selection.write(data).items()):
        for task, families in sorted(tasks.items()):
            print(f"{dataset:18} {task}  " + "  ".join(f"{family}: {configurations.iterations(name)}" for family, name in sorted(families.items())))


if __name__ == "__main__":
    main()
