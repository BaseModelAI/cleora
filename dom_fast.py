import numpy as np, time, tracemalloc, sys, gc
from pycleora import SparseMatrix, embed
from pycleora.algorithms import embed_prone, embed_randne
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

def run(name, fn, g, lb):
    emb, t, mem = measure(fn, g)
    nc = node_classification_scores(g, emb, lb, seed=42)
    mlp = mlp_classify(g, emb, lb, hidden_dim=128, num_epochs=200, learning_rate=0.01, seed=42)
    print(f"{name:<18s} {nc['accuracy']:>8.4f} {nc['macro_f1']:>8.4f} {mlp['accuracy']:>8.4f} {mlp['macro_f1']:>8.4f} {t:>7.3f}s {mem:>7.1f}")
    sys.stdout.flush(); gc.collect()

# Facebook - remaining algos
print("=== FACEBOOK (ProNE, RandNE) ===")
ds = load_dataset("facebook")
g = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
lb = detect_communities_louvain(g)
run("ProNE", lambda g: embed_prone(g, DIM), g, lb)
run("RandNE", lambda g: embed_randne(g, DIM), g, lb)

# PPI-large 
print("\n=== PPI-LARGE ===")
ds = load_dataset("ppi_large")
g = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
lb = detect_communities_louvain(g)
print(f"Nodes: {ds['num_nodes']}, Labels: {len(lb)}")
print(f"{'Algo':<18s} {'NC_Acc':>8s} {'NC_F1':>8s} {'MLP_Acc':>8s} {'MLP_F1':>8s} {'Time':>8s} {'Mem':>8s}")
run("Cleora(w,8it)", lambda g: embed(g, DIM, 8, whiten=True), g, lb)
run("Cleora(base)", lambda g: embed(g, DIM, 4), g, lb)
run("ProNE", lambda g: embed_prone(g, DIM), g, lb)
run("RandNE", lambda g: embed_randne(g, DIM), g, lb)

# Flickr
print("\n=== FLICKR ===")
ds = load_dataset("flickr")
g = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
lb = detect_communities_louvain(g)
print(f"Nodes: {ds['num_nodes']}, Labels: {len(lb)}")
print(f"{'Algo':<18s} {'NC_Acc':>8s} {'NC_F1':>8s} {'MLP_Acc':>8s} {'MLP_F1':>8s} {'Time':>8s} {'Mem':>8s}")
run("Cleora(w,8it)", lambda g: embed(g, DIM, 8, whiten=True), g, lb)
run("Cleora(base)", lambda g: embed(g, DIM, 4), g, lb)
run("ProNE", lambda g: embed_prone(g, DIM), g, lb)
run("RandNE", lambda g: embed_randne(g, DIM), g, lb)

print("\nDONE!")
