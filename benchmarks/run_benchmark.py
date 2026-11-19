#!/usr/bin/env python
"""Runs benchmark jobs: every requested method on every requested dataset and task.

    python benchmarks/run_benchmark.py --data ~/cleora-bench --datasets cora,citeseer \\
        --methods cleora,deepwalk,prone --dim 1024 --seeds 0,1,2

Results are written as JSON under ``<data>/results`` (one file per job; finished jobs are
skipped, so the command can be repeated or run in several processes on disjoint
datasets). ``benchmarks/make_report.py`` turns them into the tables of the README.

The method ``cleora_tuned`` stands for the fixed-iteration configurations that
``benchmarks/select_iterations.py`` chose for the dataset from a sweep.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cleora_bench import configurations, datasets, probes, runner, selection, tasks  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True, help="directory for datasets, caches and results")
    parser.add_argument("--datasets", default=None, help="comma-separated names (default: all of --tiers)")
    parser.add_argument("--tiers", default="S", help="dataset tiers to run when --datasets is not given: S, M, L")
    parser.add_argument("--methods", default=",".join(configurations.DEFAULT), help="comma-separated configuration names")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--seeds", default="0,1,2", help="seeds of the embedding methods and of the link splits")
    parser.add_argument("--tasks", default="nc,lp,viz",
                        help="nc (node classification), lp (link prediction), viz (vectors of the whole graph "
                             "for the figures, where nc does not already provide them), time (embedding time "
                             "and memory only: run it alone on an idle machine, with --results timing)")
    parser.add_argument("--node-probes", default=",".join(tasks.NODE_PROBES))
    parser.add_argument("--pair-probes", default=",".join(tasks.PAIR_PROBES))
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--timeout", type=float, default=6.0, help="hours a method may take on one graph")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--results", default="results", help="name of the results directory under --data")
    args = parser.parse_args()

    args.data = os.path.abspath(os.path.expanduser(args.data))
    names = args.datasets.split(",") if args.datasets else datasets.names(args.tiers)
    device = probes.pick_device()
    probes.torch.set_num_threads(min(args.threads, 4))  # the small probes that are fitted on the CPU
    for dataset in names:
        meta = datasets.load(dataset, os.path.join(args.data, "datasets"))["meta"]
        for task in args.tasks.split(","):
            if task == "nc" and meta["num_classes"] == 0:
                print(f"{dataset}: no node labels - link task only", flush=True)
                continue
            if task == "viz" and meta["num_classes"] > 0:
                continue  # the node-classification run saves the sample
            method_ids = []
            for method_id in args.methods.split(","):
                method_ids += selection.configurations_for(args.data, dataset, task) if method_id == selection.TUNED else [method_id]
            for method_id in method_ids:
                method, params = configurations.resolve(method_id)
                for seed in [int(s) for s in args.seeds.split(",")]:
                    if task == "viz" and seed != 0:
                        continue
                    started = time.time()
                    result = runner.run_job(
                        dataset, method_id, method, params, args.dim, seed, task,
                        data_root=os.path.join(args.data, "datasets"), results=os.path.join(args.data, args.results),
                        cache=os.path.join(args.data, "cache"), threads=args.threads, timeout=args.timeout * 3600,
                        device=device, node_probes=tuple(args.node_probes.split(",")),
                        pair_probes=tuple(args.pair_probes.split(",")), retry_failed=args.retry_failed)
                    embedding = result["embedding"]
                    if embedding["status"] != "ok":
                        summary = f"{embedding['status']}: {embedding['reason']}"
                    elif task == "viz":
                        summary = "sample saved"
                    elif task == "time":
                        summary = f"peak memory {embedding['peak_rss_mb'] / 1024:.2f} GB"
                    elif task == "nc":
                        first = result["probes"][next(iter(result["probes"]))]["metrics"]
                        key = "accuracy" if "accuracy" in first[0] else "micro_f1"
                        summary = " ".join(f"{probe}={sum(m[key] for m in r['metrics']) / len(r['metrics']):.4f}"
                                           for probe, r in result["probes"].items()) + f" ({key})"
                    else:
                        summary = " ".join(f"{probe}={r['metrics']['mrr']:.4f}" for probe, r in result["probes"].items()) + " (MRR)"
                    seconds = embedding.get("seconds")
                    print(f"{dataset:18} {task} {method_id:22} seed={seed} embed={seconds if seconds is None else round(seconds, 1)}s "
                          f"{summary} [{time.time() - started:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
