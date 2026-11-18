#!/usr/bin/env python
"""Reproducible node-classification benchmark on real, public graphs.

    pip install "pycleora[full]" scikit-learn
    python benchmarks/run_benchmarks.py                      # the tables in the README
    python benchmarks/run_benchmarks.py --datasets cora,ppi --methods cleora,prone
    python benchmarks/run_benchmarks.py --out results.json --resume   # continue an interrupted run

What is measured
----------------
For every dataset and method the script embeds the graph (structure only, node
features are never used), then trains a classifier on the embeddings:

* embedding time (wall clock, best of three runs for the Cleora variants),
* the additional peak memory of the embedding call (resident set size on Linux,
  Python allocations elsewhere). It is measured in a second, separate process in
  which the C allocator maps every large block on its own and returns it as soon
  as it is freed; otherwise memory left over from loading the graph is silently
  reused and the figure comes out too low. That setting slows allocation-heavy
  code down, which is why time is not taken from the same run,
* accuracy and macro-F1 for single-label graphs, micro-F1 and macro-F1 for
  multi-label graphs.

Protocol: features are standardised, the classifier is L2-regularised logistic
regression (one-vs-rest for multi-label graphs, where - as in the DeepWalk and
node2vec papers - the number of labels to predict for a node is taken from the
ground truth). The regularisation strength is chosen on a validation part of the
training set. Results are averaged over random splits; every method sees the same
splits. Everything is seeded, so the numbers are reproducible on the same machine
and library versions.

The datasets are downloaded by ``pycleora.datasets`` from their original public
sources; none of them is generated.
"""

import argparse
import ctypes
import gc
import importlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np

import pycleora
from pycleora import SparseMatrix, datasets, embed
from pycleora import algorithms
from pycleora.benchmark import _MemoryTracker

DEFAULT_DATASETS = [
    "cora", "citeseer", "pubmed", "email_eu_core", "amazon_photo", "amazon_computers", "ppi", "wikipedia_pos",
]

# name -> (label, function(graph, dim), needs a dense n x n matrix)
METHODS = {
    "cleora": ("Cleora (default: 40 it. + whitening)", lambda g, d: embed(g, d), False),
    "cleora-classic": ("Cleora classic (8 it., no whitening)", lambda g, d: embed(g, d, 8, whiten=False), False),
    "prone": ("ProNE", lambda g, d: algorithms.embed_prone(g, d), False),
    "randne": ("RandNE", lambda g, d: algorithms.embed_randne(g, d), False),
    "deepwalk": ("DeepWalk (PPMI + SVD)", lambda g, d: algorithms.embed_deepwalk(g, d), False),
    "node2vec": ("node2vec (p=0.5, q=2; PPMI + SVD)",
                 lambda g, d: algorithms.embed_node2vec(g, d, p=0.5, q=2.0), False),
    "netmf": ("NetMF", lambda g, d: algorithms.embed_netmf(g, d), True),
    "hope": ("HOPE", lambda g, d: algorithms.embed_hope(g, d), True),
    "grarep": ("GraRep", lambda g, d: algorithms.embed_grarep(g, d), True),
}


def load(name):
    """Graph, labelled rows and their targets (a vector, or a 0/1 matrix)."""
    data = datasets.load_dataset(name)
    assert data["synthetic"] is False, name
    graph = SparseMatrix.from_iterator(iter(data["edges"]), data["columns"])
    row_of = {entity: i for i, entity in enumerate(graph.entity_ids)}
    if data.get("multilabel"):
        keep = [i for i, entity in enumerate(data["label_entities"]) if entity in row_of]
        rows = np.array([row_of[data["label_entities"][i]] for i in keep])
        targets = np.asarray(data["label_matrix"])[keep].astype(np.int64)
        labelled = targets.sum(axis=1) > 0
        return data, graph, rows[labelled], targets[labelled], True
    entities = [entity for entity in data["labels"] if entity in row_of]
    rows = np.array([row_of[entity] for entity in entities])
    targets = np.array([data["labels"][entity] for entity in entities])
    return data, graph, rows, targets, False


def fit_and_score(train_x, train_y, test_x, test_y, multilabel, strength):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.multiclass import OneVsRestClassifier

    if multilabel:
        present = train_y.sum(axis=0) > 0
        model = OneVsRestClassifier(LogisticRegression(C=strength, max_iter=300))
        model.fit(train_x, train_y[:, present])
        scores = np.full(test_y.shape, -np.inf)
        scores[:, present] = model.predict_proba(test_x)
        predicted = np.zeros_like(test_y)
        order = np.argsort(-scores, axis=1)
        for i, count in enumerate(test_y.sum(axis=1)):
            predicted[i, order[i, :count]] = 1
        return (
            f1_score(test_y, predicted, average="micro", zero_division=0),
            f1_score(test_y, predicted, average="macro", zero_division=0),
        )
    model = LogisticRegression(C=strength, max_iter=500).fit(train_x, train_y)
    predicted = model.predict(test_x)
    return float((predicted == test_y).mean()), f1_score(test_y, predicted, average="macro", zero_division=0)


def evaluate(features, targets, multilabel, splits, train_ratio, fixed_strength=None):
    from sklearn.preprocessing import StandardScaler

    grid = (0.003, 0.03, 0.3) if multilabel else (0.001, 0.01, 0.1, 1.0)
    results = []
    for split in range(splits):
        rng = np.random.default_rng(1000 + split)
        order = rng.permutation(len(features))
        cut = int(len(features) * train_ratio)
        train, test = order[:cut], order[cut:]
        scaler = StandardScaler().fit(features[train])
        train_x, test_x = scaler.transform(features[train]), scaler.transform(features[test])
        train_y, test_y = targets[train], targets[test]
        if fixed_strength is None:
            inner = rng.permutation(len(train))
            validation, fit = inner[: len(train) // 5], inner[len(train) // 5:]
            strength = max(
                grid,
                key=lambda c: fit_and_score(
                    train_x[fit], train_y[fit], train_x[validation], train_y[validation], multilabel, c
                )[0],
            )
        else:
            strength = fixed_strength
        results.append(fit_and_score(train_x, train_y, test_x, test_y, multilabel, strength))
    results = np.array(results)
    return {
        "score": float(results[:, 0].mean()),
        "score_std": float(results[:, 0].std()),
        "macro_f1": float(results[:, 1].mean()),
    }


def release_free_memory():
    """Returns freed heap to the operating system (glibc), so that a later peak is
    not hidden by memory the process still holds from earlier work."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def measure(function, graph, dim, repeats):
    """Best wall time of `repeats` runs and the additional peak memory of the first."""
    best = float("inf")
    memory = method = None
    embeddings = None
    for _ in range(repeats):
        embeddings = None
        release_free_memory()
        with _MemoryTracker() as tracker:
            start = time.perf_counter()
            embeddings = function(graph, dim)
            elapsed = time.perf_counter() - start
        best = min(best, elapsed)
        if memory is None:
            memory, method = tracker.peak_mb, tracker.method
    return embeddings, best, memory, method


def worker(name, key, dim, output, single_run):
    """Runs one method on one dataset in this (fresh) process."""
    # Import the optional dependencies up front: the baselines load scipy lazily,
    # and neither its import time nor its memory belongs to their measurement.
    for module in ("scipy.sparse", "scipy.sparse.linalg", "scipy.linalg"):
        try:
            importlib.import_module(module)
        except ImportError:
            pass
    _, graph, _, _, _ = load(name)
    repeats = 3 if key.startswith("cleora") and graph.num_entities < 50000 and not single_run else 1
    embeddings, seconds, memory, memory_method = measure(METHODS[key][1], graph, dim, repeats)
    if output:
        np.save(output, embeddings)
    print(json.dumps({"time_s": seconds, "memory_mb": memory, "memory_method": memory_method}))


# glibc: serve every large allocation from a mapping of its own and give freed
# memory back at once, so that resident memory follows what the code really uses.
MEMORY_RUN_ENVIRONMENT = {
    "MALLOC_MMAP_THRESHOLD_": "65536",
    "MALLOC_TRIM_THRESHOLD_": "65536",
    "MALLOC_TOP_PAD_": "0",
}


def run_worker(name, key, dim, output=None, memory_run=False):
    command = [sys.executable, os.path.abspath(__file__), "--worker", name, key, "--dim", str(dim)]
    if output:
        command += ["--worker-output", output]
    environment = None
    if memory_run:
        command += ["--worker-single-run"]
        environment = dict(os.environ, **MEMORY_RUN_ENVIRONMENT)
    completed = subprocess.run(command, capture_output=True, text=True, env=environment)
    if completed.returncode != 0:
        lines = completed.stderr.strip().splitlines()
        return {"failed": lines[-1][:200] if lines else "failed"}
    return json.loads(completed.stdout.strip().splitlines()[-1])


def run_isolated(name, key, dim, keep_embeddings=True):
    """Embeddings and time from one fresh process, memory from another."""
    embeddings = None
    with tempfile.TemporaryDirectory() as directory:
        output = os.path.join(directory, "embeddings.npy") if keep_embeddings else None
        timed = run_worker(name, key, dim, output=output)
        if "failed" in timed:
            return None, timed
        if keep_embeddings:
            embeddings = np.load(output)
    measured = {"time_s": timed["time_s"]}
    memory = run_worker(name, key, dim, memory_run=True)
    if "failed" in memory:
        return None, memory
    measured.update(memory_mb=memory["memory_mb"], memory_method=memory["memory_method"])
    return embeddings, measured


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--splits", type=int, default=5, help="random train/test splits (single-label graphs)")
    parser.add_argument("--multilabel-splits", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--dense-limit", type=int, default=5000,
                        help="largest graph (nodes) for the methods that build a dense n x n matrix")
    parser.add_argument("--fixed-c", type=float, default=None,
                        help="skip the search for the regularisation strength and use this value")
    parser.add_argument("--no-eval", action="store_true", help="measure time and memory only")
    parser.add_argument("--out", default=None, help="write the results as JSON (updated after every method)")
    parser.add_argument("--resume", action="store_true",
                        help="keep the results already in --out and only run what is missing")
    parser.add_argument("--worker", nargs=2, metavar=("DATASET", "METHOD"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    parser.add_argument("--worker-single-run", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    if args.worker:
        worker(args.worker[0], args.worker[1], args.dim, args.worker_output, args.worker_single_run)
        return
    try:
        import sklearn
    except ImportError:
        if not args.no_eval:
            sys.exit("scikit-learn is needed for the evaluation: pip install scikit-learn")
        sklearn = None

    report = {
        "pycleora": pycleora.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": getattr(sklearn, "__version__", None),
        "machine": platform.machine(),
        "processor_count": os.cpu_count(),
        "dim": args.dim,
        "train_ratio": args.train_ratio,
        "results": [],
    }
    done = {}
    if args.resume and args.out and os.path.exists(args.out):
        with open(args.out) as handle:
            report["results"] = json.load(handle)["results"]
        done = {(entry["dataset"], entry["method"]): entry for entry in report["results"]}
    print(json.dumps({k: v for k, v in report.items() if k != "results"}))

    def record(entry):
        report["results"].append(entry)
        if args.out:
            with open(args.out + ".tmp", "w") as handle:
                json.dump(report, handle, indent=1)
            os.replace(args.out + ".tmp", args.out)

    def show(entry):
        if "skipped" in entry:
            print(f"| {entry['label']} | - | - | - | - |")
        elif "failed" in entry:
            print(f"| {entry['label']} | failed | - | - | - |")
        elif "score" not in entry:
            print(f"| {entry['label']} | - | - | {entry['time_s']:.2f} | {entry['memory_mb']:.0f} |")
        else:
            print(f"| {entry['label']} | {entry['score']:.3f} ± {entry['score_std']:.3f} | "
                  f"{entry['macro_f1']:.3f} | {entry['time_s']:.2f} | {entry['memory_mb']:.0f} |")
        sys.stdout.flush()

    for name in [d for d in args.datasets.split(",") if d]:
        start = time.perf_counter()
        data, graph, rows, targets, multilabel = load(name)
        build = time.perf_counter() - start
        # the workers load the graph themselves: keep only what the report needs
        nodes, edges, classes = graph.num_entities, data["num_edges"], data["num_classes"]
        del data, graph
        splits = args.multilabel_splits if multilabel else args.splits
        metric = "micro-F1" if multilabel else "accuracy"
        print(f"\n## {name}: {nodes} nodes, {edges} edges, "
              f"{classes} classes ({'multi' if multilabel else 'single'}-label), "
              f"{len(rows)} labelled, loaded and built in {build:.2f} s")
        print(f"| method | {metric} | macro-F1 | time (s) | peak memory (MB) |")
        print("|---|---|---|---|---|")
        for key in [m for m in args.methods.split(",") if m]:
            label, _, dense = METHODS[key]
            if (name, key) in done:
                show(done[(name, key)])
                continue
            entry = {"dataset": name, "method": key, "label": label, "nodes": nodes,
                     "edges": edges, "metric": metric}
            if dense and nodes > args.dense_limit:
                entry["skipped"] = f"dense n x n method, graph has more than {args.dense_limit} nodes"
            else:
                embeddings, measured = run_isolated(name, key, args.dim, keep_embeddings=not args.no_eval)
                entry.update(measured)
                if embeddings is not None and not args.no_eval:
                    entry.update(evaluate(embeddings[rows], targets, multilabel, splits, args.train_ratio, args.fixed_c))
                del embeddings
            show(entry)
            record(entry)

    if args.out:
        print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    main()
