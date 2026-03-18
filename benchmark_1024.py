import numpy as np
import time
import tracemalloc
import sys

from pycleora import SparseMatrix, embed
from pycleora.algorithms import embed_prone, embed_randne
from pycleora.metrics import node_classification_scores
from pycleora.datasets import load_dataset

DATASETS = ["ppi_large", "flickr", "ogbn_arxiv"]
DIM = 1024


def run_single(ds_name):
    print(f"\n{'='*80}")
    print(f"  Dataset: {ds_name} (dim={DIM})")
    print(f"{'='*80}")

    data = load_dataset(ds_name)
    n = data['num_nodes']
    print(f"  Nodes: {n:,}  Edges: {data['num_edges']:,}  Classes: {data['num_classes']}")

    t0 = time.time()
    graph = SparseMatrix.from_iterator(iter(data["edges"]), data["columns"])
    print(f"  Graph build: {time.time()-t0:.2f}s")

    labels = data.get("labels", {})

    algos = [
        ("Cleora", lambda g: embed(g, DIM, 4)),
        ("Cleora-sym", lambda g: embed(g, DIM, 4, propagation="symmetric")),
        ("RandNE", lambda g: embed_randne(g, DIM)),
        ("ProNE", lambda g: embed_prone(g, DIM)),
    ]

    for algo_name, algo_fn in algos:
        print(f"\n  --- {algo_name} ---")
        try:
            tracemalloc.start()
            t0 = time.time()
            emb = algo_fn(graph)
            elapsed = time.time() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = peak / (1024 * 1024)

            print(f"  Time: {elapsed:.3f}s  Memory: {peak_mb:.2f} MB")

            if labels and data["num_classes"] > 1:
                try:
                    scores = node_classification_scores(
                        graph, emb, labels, train_ratio=0.8
                    )
                    print(f"  Accuracy: {scores['accuracy']:.4f}  Macro F1: {scores['macro_f1']:.4f}")
                except Exception as e:
                    print(f"  Classification error: {e}")
        except Exception as e:
            print(f"  FAILED: {e}")
            try:
                tracemalloc.stop()
            except:
                pass


if __name__ == "__main__":
    for ds in DATASETS:
        run_single(ds)
    print("\n\nDone with medium datasets!")
