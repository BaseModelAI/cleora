import numpy as np, time, tracemalloc, sys, gc
from pycleora import SparseMatrix, embed
from pycleora.algorithms import embed_prone, embed_randne, embed_netmf, embed_deepwalk
from pycleora.classify import mlp_classify
from pycleora.metrics import node_classification_scores
from pycleora.community import detect_communities_louvain
from pycleora.datasets import load_dataset

DIM = 256

def measure(fn, g):
    gc.collect(); tracemalloc.start()
    t0 = time.time(); r = fn(g); t = time.time() - t0
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return r, t, peak/1024/1024

ds = load_dataset("facebook")
g = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
lb = detect_communities_louvain(g)

algos = [
    ("NetMF", lambda g: embed_netmf(g, DIM)),
    ("DeepWalk", lambda g: embed_deepwalk(g, DIM, num_walks=10, walk_length=20)),
    ("ProNE", lambda g: embed_prone(g, DIM)),
    ("RandNE", lambda g: embed_randne(g, DIM)),
]

print(f"{'Algo':<14s} {'NC_Acc':>8s} {'NC_F1':>8s} {'MLP_Acc':>8s} {'MLP_F1':>8s} {'Time':>8s} {'Mem':>8s}")
for name, fn in algos:
    try:
        emb, t, mem = measure(fn, g)
        nc = node_classification_scores(g, emb, lb, seed=42)
        mlp = mlp_classify(g, emb, lb, hidden_dim=128, num_epochs=200, learning_rate=0.01, seed=42)
        print(f"{name:<14s} {nc['accuracy']:>8.4f} {nc['macro_f1']:>8.4f} {mlp['accuracy']:>8.4f} {mlp['macro_f1']:>8.4f} {t:>7.3f}s {mem:>7.1f}")
    except Exception as e:
        print(f"{name:<14s} ERROR: {str(e)[:60]}")
    sys.stdout.flush(); gc.collect()
