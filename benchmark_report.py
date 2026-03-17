"""
pycleora 3.0 — Full Benchmark Report
Compares all embedding algorithms across public datasets.
"""
import numpy as np
import time
import tracemalloc

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
    "karate_club",
    "dolphins",
    "les_miserables",
    "football",
    "cora",
    "citeseer",
]

DIM = 128

def _make_algorithms(n_nodes):
    algos = {
        "Cleora":     lambda g: embed(g, DIM, 4),
        "Cleora-sym": lambda g: embed(g, DIM, 4, propagation="symmetric"),
        "ProNE":      lambda g: embed_prone(g, DIM),
        "RandNE":     lambda g: embed_randne(g, DIM),
        "NetMF":      lambda g: embed_netmf(g, DIM),
    }
    if n_nodes <= 500:
        algos["HOPE"] = lambda g: embed_hope(g, DIM)
        algos["GraRep"] = lambda g: embed_grarep(g, DIM)
        algos["DeepWalk"] = lambda g: embed_deepwalk(g, DIM, num_walks=20, walk_length=40)
        algos["Node2Vec"] = lambda g: embed_node2vec(g, DIM, num_walks=20, walk_length=40, p=1.0, q=0.5)
    else:
        algos["DeepWalk"] = lambda g: embed_deepwalk(g, DIM, num_walks=3, walk_length=15)
        algos["Node2Vec"] = lambda g: embed_node2vec(g, DIM, num_walks=3, walk_length=15, p=1.0, q=0.5)
    return algos


def run_benchmark():
    sep = "=" * 90
    thin = "-" * 90

    print(sep)
    print("  pycleora 3.0 — BENCHMARK REPORT")
    print("  Comparing 9 algorithms across 6 datasets")
    print(sep)
    print()

    all_results = {}

    for ds_name in DATASETS:
        print(f"\n{sep}")
        print(f"  DATASET: {ds_name.upper()}")
        print(sep)

        ds = load_dataset(ds_name)
        graph = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
        labels = ds["labels"]
        print(f"  Nodes: {ds['num_nodes']:<8d}  Edges: {ds['num_edges']:<8d}  Classes: {ds['num_classes']}")
        print()

        algos = _make_algorithms(ds["num_nodes"])

        header = (
            f"  {'Algorithm':<12s} {'Acc':>6s} {'MacF1':>6s} {'Time':>8s} "
            f"{'Mem MB':>8s} {'Sil':>6s} {'Mod':>6s}"
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

                nc = node_classification_scores(graph, emb, labels, seed=42)
                acc = nc["accuracy"]
                f1 = nc["macro_f1"]

                true_arr = np.array([labels.get(eid, 0) for eid in graph.entity_ids])
                sil = silhouette_score(emb, true_arr)

                comms = detect_communities_louvain(graph)
                mod = modularity(graph, comms)

                print(
                    f"  {algo_name:<12s} {acc:>6.3f} {f1:>6.3f} {elapsed:>7.3f}s "
                    f"{mem_mb:>7.2f} {sil:>6.3f} {mod:>6.3f}"
                )

                ds_results[algo_name] = {
                    "accuracy": acc, "macro_f1": f1,
                    "time": elapsed, "memory_mb": mem_mb,
                    "silhouette": sil, "modularity": mod,
                }

            except Exception as e:
                tracemalloc.stop()
                print(f"  {algo_name:<12s} {'ERROR':>6s}  {str(e)[:50]}")
                ds_results[algo_name] = {"error": str(e)}

        all_results[ds_name] = ds_results

        best_algo = max(
            [(name, r["accuracy"]) for name, r in ds_results.items() if "accuracy" in r],
            key=lambda x: x[1],
        )
        print(f"\n  Best: {best_algo[0]} (accuracy={best_algo[1]:.4f})")

    print(f"\n\n{sep}")
    print("  CLASSIFIER COMPARISON (MLP vs GCN on top of Cleora embeddings)")
    print(sep)

    for ds_name in ["karate_club", "dolphins", "football", "cora"]:
        ds = load_dataset(ds_name)
        graph = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
        emb = embed(graph, DIM, 4)
        labels = ds["labels"]

        nc_nearest = node_classification_scores(graph, emb, labels, seed=42)

        mlp_result = mlp_classify(graph, emb, labels, hidden_dim=64, num_epochs=300, seed=42)

        gcn_result = gcn_classify(graph, emb, labels, hidden_dim=64, num_epochs=200, num_layers=2, seed=42)

        print(f"\n  {ds_name.upper()} ({ds['num_nodes']} nodes, {ds['num_classes']} classes):")
        print(f"    Nearest Centroid:  Acc={nc_nearest['accuracy']:.4f}  F1={nc_nearest['macro_f1']:.4f}")
        print(f"    MLP (1 hidden):    Acc={mlp_result['accuracy']:.4f}  F1={mlp_result['macro_f1']:.4f}")
        print(f"    GCN (2 layers):    Acc={gcn_result['accuracy']:.4f}  F1={gcn_result['macro_f1']:.4f}")

    print(f"\n\n{sep}")
    print("  CROSS-VALIDATION (5-fold on Cleora embeddings)")
    print(sep)

    for ds_name in ["karate_club", "dolphins", "football", "cora"]:
        ds = load_dataset(ds_name)
        graph = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
        emb = embed(graph, DIM, 4)
        labels = ds["labels"]

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
        best_acc = max((r.get("accuracy", 0) for r in ds_r.values() if "accuracy" in r), default=0)
        for algo_name in all_algo_names:
            r = ds_r.get(algo_name, {})
            if "accuracy" in r:
                acc = r["accuracy"]
                marker = "*" if abs(acc - best_acc) < 0.001 else " "
                print(f" {acc:>8.3f}{marker}", end="")
            else:
                print(f"     ERR  ", end="")
        print()

    print(f"\n  * = best on this dataset")
    print(f"\n{sep}")
    print("  Benchmark complete!")
    print(sep)


if __name__ == "__main__":
    run_benchmark()
