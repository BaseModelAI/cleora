"""
pycleora 3.0 — Full Benchmark Report
Compares all embedding algorithms across SNAP datasets.
"""
import numpy as np
import time
import tracemalloc
import sys

from pycleora import SparseMatrix, embed
from pycleora.algorithms import (
    embed_prone, embed_randne, embed_hope, embed_netmf,
    embed_grarep, embed_deepwalk, embed_node2vec,
)
from pycleora.classify import gcn_classify, mlp_classify
from pycleora.metrics import (
    node_classification_scores, clustering_scores, cross_validate,
    adjusted_rand_index, silhouette_score,
)
from pycleora.community import detect_communities_louvain, modularity
from pycleora.datasets import load_dataset


DATASETS = [
    "facebook",
    "ppi_large",
    "flickr",
    "ogbn_arxiv",
    "yelp",
    "roadnet",
    "livejournal",
]

DIM = 1024

LARGE_GRAPH_THRESHOLD = 50_000
HUGE_GRAPH_THRESHOLD = 3_000_000


def _make_algorithms(n_nodes):
    algos = {}

    algos["Cleora"] = lambda g: embed(g, DIM, 4)
    algos["Cleora-sym"] = lambda g: embed(g, DIM, 4, propagation="symmetric")
    algos["ProNE"] = lambda g: embed_prone(g, DIM)
    algos["RandNE"] = lambda g: embed_randne(g, DIM)

    if n_nodes <= LARGE_GRAPH_THRESHOLD:
        algos["NetMF"] = lambda g: embed_netmf(g, DIM)

    if n_nodes <= 500:
        algos["HOPE"] = lambda g: embed_hope(g, DIM)
        algos["GraRep"] = lambda g: embed_grarep(g, DIM)
        algos["DeepWalk"] = lambda g: embed_deepwalk(g, DIM, num_walks=20, walk_length=40)
        algos["Node2Vec"] = lambda g: embed_node2vec(g, DIM, num_walks=20, walk_length=40, p=1.0, q=0.5)
    elif n_nodes <= LARGE_GRAPH_THRESHOLD:
        algos["DeepWalk"] = lambda g: embed_deepwalk(g, DIM, num_walks=10, walk_length=20)
        algos["Node2Vec"] = lambda g: embed_node2vec(g, DIM, num_walks=10, walk_length=20, p=1.0, q=0.5)

    return algos


def _generate_community_labels(graph):
    sys.stderr.write("  Generating community labels via Louvain...\n")
    sys.stderr.flush()
    labels = detect_communities_louvain(graph)
    unique_labels = set(labels.values())
    num_classes = len(unique_labels)
    sys.stderr.write(f"  Found {num_classes} communities\n")
    sys.stderr.flush()
    return labels, num_classes


def run_benchmark():
    sep = "=" * 90
    thin = "-" * 90

    print(sep)
    print("  pycleora 3.0 — BENCHMARK REPORT")
    print(f"  Comparing algorithms across {len(DATASETS)} SNAP datasets")
    print(sep)
    print()

    all_results = {}

    for ds_name in DATASETS:
        print(f"\n{sep}")
        print(f"  DATASET: {ds_name.upper()}")
        print(sep)

        ds = load_dataset(ds_name)
        n_nodes = ds["num_nodes"]

        if n_nodes > HUGE_GRAPH_THRESHOLD:
            print(f"  Nodes: {n_nodes:<10,d}  Edges: {ds['num_edges']:<10,d}")
            print(f"  Skipping benchmark — graph too large for this environment.")
            print(f"  Use dedicated hardware with >16GB RAM for graphs of this scale.")
            all_results[ds_name] = {}
            continue

        graph = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
        labels = ds["labels"]
        num_classes = ds["num_classes"]
        print(f"  Nodes: {n_nodes:<10d}  Edges: {ds['num_edges']:<10d}  Classes: {num_classes}")

        has_labels = len(labels) >= 4

        if not has_labels and n_nodes <= LARGE_GRAPH_THRESHOLD:
            labels, num_classes = _generate_community_labels(graph)
            has_labels = len(labels) >= 4

        print()

        algos = _make_algorithms(n_nodes)

        header = (
            f"  {'Algorithm':<12s} {'Acc':>6s} {'MacF1':>6s} {'Time':>8s} "
            f"{'Mem MB':>10s} {'Sil':>6s}"
        )
        print(header)
        print(f"  {thin}")

        ds_results = {}

        for algo_name, algo_fn in algos.items():
            try:
                tracemalloc.start()
                t0 = time.time()
                emb = algo_fn(graph)
                elapsed = time.time() - t0
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                mem_mb = peak / 1024 / 1024

                if emb.shape[0] != graph.num_entities:
                    raise ValueError(f"Shape mismatch: {emb.shape[0]} vs {graph.num_entities}")

                result_entry = {
                    "time": elapsed, "memory_mb": mem_mb,
                }

                if has_labels:
                    nc = node_classification_scores(graph, emb, labels, seed=42)
                    acc = nc["accuracy"]
                    f1 = nc["macro_f1"]

                    true_arr = np.array([labels.get(eid, 0) for eid in graph.entity_ids])
                    sil = silhouette_score(emb, true_arr)

                    result_entry.update({
                        "accuracy": acc, "macro_f1": f1, "silhouette": sil,
                    })

                    print(
                        f"  {algo_name:<12s} {acc:>6.3f} {f1:>6.3f} {elapsed:>7.3f}s "
                        f"{mem_mb:>9.2f} {sil:>6.3f}"
                    )
                else:
                    print(
                        f"  {algo_name:<12s}    —      —   {elapsed:>7.3f}s "
                        f"{mem_mb:>9.2f}    —  "
                    )

                ds_results[algo_name] = result_entry

            except Exception as e:
                try:
                    tracemalloc.stop()
                except Exception:
                    pass
                print(f"  {algo_name:<12s} {'ERROR':>6s}  {str(e)[:60]}")
                ds_results[algo_name] = {"error": str(e)}

        all_results[ds_name] = ds_results

        successful = [(name, r) for name, r in ds_results.items() if "error" not in r]
        if successful:
            if has_labels:
                best_algo = max(
                    [(name, r["accuracy"]) for name, r in ds_results.items() if "accuracy" in r],
                    key=lambda x: x[1],
                )
                print(f"\n  Best: {best_algo[0]} (accuracy={best_algo[1]:.4f})")
            else:
                fastest = min(
                    [(name, r["time"]) for name, r in ds_results.items() if "time" in r],
                    key=lambda x: x[1],
                )
                print(f"\n  Fastest: {fastest[0]} ({fastest[1]:.3f}s)")

    print(f"\n\n{sep}")
    print("  CROSS-VALIDATION (5-fold on Cleora embeddings)")
    print(sep)

    for ds_name in DATASETS:
        ds = load_dataset(ds_name)
        if ds["num_nodes"] > LARGE_GRAPH_THRESHOLD:
            print(f"  {ds_name:<16s}  (skipped — too large for cross-validation)")
            continue
        graph = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
        labels = ds["labels"]

        if len(labels) < 4:
            labels, _ = _generate_community_labels(graph)

        emb = embed(graph, DIM, 4)

        cv = cross_validate(graph, emb, labels, k_folds=5, seed=42)
        print(
            f"  {ds_name:<16s}  Acc={cv['mean_accuracy']:.4f} +/- {cv['std_accuracy']:.4f}  "
            f"F1={cv['mean_macro_f1']:.4f} +/- {cv['std_macro_f1']:.4f}  "
            f"folds={cv['fold_accuracies']}"
        )

    print(f"\n\n{sep}")
    print("  SUMMARY TABLE — Best Accuracy per Dataset")
    print(sep)
    all_algo_names = sorted(set(a for ds_r in all_results.values() for a in ds_r))
    print(f"\n  {'Dataset':<16s}", end="")
    for algo_name in all_algo_names:
        print(f" {algo_name:>9s}", end="")
    print()
    print(f"  {thin}")

    for ds_name in DATASETS:
        print(f"  {ds_name:<16s}", end="")
        ds_r = all_results.get(ds_name, {})
        accs_with_vals = [r.get("accuracy", 0) for r in ds_r.values() if "accuracy" in r]
        best_acc = max(accs_with_vals) if accs_with_vals else 0
        for algo_name in all_algo_names:
            r = ds_r.get(algo_name, {})
            if "accuracy" in r:
                acc = r["accuracy"]
                marker = "*" if abs(acc - best_acc) < 0.001 else " "
                print(f" {acc:>8.3f}{marker}", end="")
            elif "error" in r:
                print(f"     ERR  ", end="")
            elif "time" in r:
                print(f"       —  ", end="")
            else:
                print(f"       —  ", end="")
        print()

    print(f"\n  * = best on this dataset")
    print(f"\n{sep}")
    print("  Benchmark complete!")
    print(sep)


if __name__ == "__main__":
    run_benchmark()
