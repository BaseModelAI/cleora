"""Full benchmark: Cleora base vs Cleora(whiten) on ALL datasets"""
import numpy as np
import time
import tracemalloc
import sys
import gc

from pycleora import SparseMatrix, embed
from pycleora.metrics import node_classification_scores
from pycleora.community import detect_communities_louvain
from pycleora.datasets import load_dataset
from pycleora.classify import mlp_classify
from pycleora.algorithms import embed_prone, embed_randne

DIM = 256

def measure(fn, graph):
    gc.collect()
    tracemalloc.start()
    t0 = time.time()
    result = fn(graph)
    elapsed = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()
    return result, elapsed, peak / 1024 / 1024

DATASETS = ["facebook", "ppi_large", "flickr", "ogbn_arxiv", "yelp", "roadnet"]

print(f"{'Dataset':<16s} {'Algorithm':<20s} {'NC_Acc':>8s} {'NC_F1':>8s} {'Time':>8s} {'Mem_MB':>8s}")
print("=" * 80)

for ds_name in DATASETS:
    sys.stderr.write(f"\n--- Loading {ds_name} ---\n")
    sys.stderr.flush()

    ds = load_dataset(ds_name)
    graph = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
    labels = ds.get("labels", {})
    if len(labels) < 4:
        labels = detect_communities_louvain(graph)

    has_labels = len(labels) >= 4
    n_nodes = ds["num_nodes"]
    sys.stderr.write(f"  Nodes: {n_nodes}, Labels: {len(labels)}\n")
    sys.stderr.flush()

    configs = [
        ("Cleora(base,4it)", lambda g: embed(g, DIM, 4)),
        ("Cleora(w,16it)", lambda g: embed(g, DIM, 16, whiten=True)),
        ("ProNE", lambda g: embed_prone(g, DIM)),
        ("RandNE", lambda g: embed_randne(g, DIM)),
    ]

    for name, fn in configs:
        try:
            emb, t, mem = measure(fn, graph)
            if has_labels:
                scores = node_classification_scores(graph, emb, labels, seed=42)
                acc = scores["accuracy"]
                f1 = scores["macro_f1"]
            else:
                acc = f1 = float('nan')
            print(f"{ds_name:<16s} {name:<20s} {acc:>8.4f} {f1:>8.4f} {t:>7.3f}s {mem:>8.1f}")
            sys.stdout.flush()
        except Exception as e:
            print(f"{ds_name:<16s} {name:<20s} {'FAIL':>8s} {'':>8s} {'':>8s} {str(e)[:30]}")
            sys.stdout.flush()

        del emb
        gc.collect()

    del graph, labels
    gc.collect()

print("\n" + "=" * 80)
print("DONE")
