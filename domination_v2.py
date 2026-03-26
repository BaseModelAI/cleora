"""FOCUSED DOMINATION — all algorithms, all datasets, MLP classifier"""
import numpy as np, time, tracemalloc, sys, gc

from pycleora import SparseMatrix, embed, embed_multiscale
from pycleora.algorithms import embed_prone, embed_randne, embed_netmf, embed_deepwalk
from pycleora.classify import mlp_classify
from pycleora.metrics import node_classification_scores, cross_validate
from pycleora.community import detect_communities_louvain
from pycleora.datasets import load_dataset

DIM = 256

def measure(fn, graph):
    gc.collect()
    tracemalloc.start()
    t0 = time.time()
    r = fn(graph)
    elapsed = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return r, elapsed, peak / 1024 / 1024

def get_labels(ds, graph):
    labels = ds.get("labels", {})
    if len(labels) < 4:
        labels = detect_communities_louvain(graph)
    return labels

def run_algo(name, fn, graph, labels, mlp_h=128, mlp_e=300, mlp_lr=0.01):
    emb, t, mem = measure(fn, graph)
    nc = node_classification_scores(graph, emb, labels, seed=42)
    mlp = mlp_classify(graph, emb, labels, hidden_dim=mlp_h, num_epochs=mlp_e, learning_rate=mlp_lr, seed=42)
    return {
        "nc_acc": nc["accuracy"], "nc_f1": nc["macro_f1"],
        "mlp_acc": mlp["accuracy"], "mlp_f1": mlp["macro_f1"],
        "time": t, "mem": mem
    }

# === ego-Facebook ===
print("=== EGO-FACEBOOK ===")
ds = load_dataset("facebook")
g = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
lb = get_labels(ds, g)

algos = [
    ("Cleora(w,8it)", lambda g: embed(g, DIM, 8, whiten=True)),
    ("Cleora(w,16it)", lambda g: embed(g, DIM, 16, whiten=True)),
    ("Cleora(base)", lambda g: embed(g, DIM, 4)),
    ("NetMF", lambda g: embed_netmf(g, DIM)),
    ("DeepWalk", lambda g: embed_deepwalk(g, DIM, num_walks=20, walk_length=40)),
    ("ProNE", lambda g: embed_prone(g, DIM)),
    ("RandNE", lambda g: embed_randne(g, DIM)),
]

print(f"{'Algo':<18s} {'NC_Acc':>8s} {'NC_F1':>8s} {'MLP_Acc':>8s} {'MLP_F1':>8s} {'Time':>8s} {'Mem':>8s}")
print("-" * 80)
for name, fn in algos:
    try:
        r = run_algo(name, fn, g, lb, mlp_h=256, mlp_e=400, mlp_lr=0.005)
        print(f"{name:<18s} {r['nc_acc']:>8.4f} {r['nc_f1']:>8.4f} {r['mlp_acc']:>8.4f} {r['mlp_f1']:>8.4f} {r['time']:>7.3f}s {r['mem']:>7.1f}")
    except Exception as e:
        print(f"{name:<18s} ERROR: {str(e)[:60]}")
    sys.stdout.flush()
    gc.collect()

# === PPI-large ===
print("\n=== PPI-LARGE ===")
ds = load_dataset("ppi_large")
g = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
lb = get_labels(ds, g)
print(f"Nodes: {ds['num_nodes']}, Labels: {len(lb)}")

algos2 = [
    ("Cleora(w,8it)", lambda g: embed(g, DIM, 8, whiten=True)),
    ("Cleora(base)", lambda g: embed(g, DIM, 4)),
    ("ProNE", lambda g: embed_prone(g, DIM)),
    ("RandNE", lambda g: embed_randne(g, DIM)),
]

print(f"{'Algo':<18s} {'NC_Acc':>8s} {'MLP_Acc':>8s} {'MLP_F1':>8s} {'Time':>8s} {'Mem':>8s}")
print("-" * 70)
for name, fn in algos2:
    try:
        r = run_algo(name, fn, g, lb)
        print(f"{name:<18s} {r['nc_acc']:>8.4f} {r['mlp_acc']:>8.4f} {r['mlp_f1']:>8.4f} {r['time']:>7.3f}s {r['mem']:>7.1f}")
    except Exception as e:
        print(f"{name:<18s} ERROR: {str(e)[:60]}")
    sys.stdout.flush()
    gc.collect()

# === Flickr ===
print("\n=== FLICKR ===")
ds = load_dataset("flickr")
g = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
lb = get_labels(ds, g)
print(f"Nodes: {ds['num_nodes']}, Labels: {len(lb)}")

for name, fn in algos2:
    try:
        r = run_algo(name, fn, g, lb)
        print(f"{name:<18s} {r['nc_acc']:>8.4f} {r['mlp_acc']:>8.4f} {r['mlp_f1']:>8.4f} {r['time']:>7.3f}s {r['mem']:>7.1f}")
    except Exception as e:
        print(f"{name:<18s} ERROR: {str(e)[:60]}")
    sys.stdout.flush()
    gc.collect()

# === ogbn-arxiv ===
print("\n=== OGBN-ARXIV ===")
ds = load_dataset("ogbn_arxiv")
g = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
lb = get_labels(ds, g)
print(f"Nodes: {ds['num_nodes']}, Labels: {len(lb)}")

for name, fn in algos2:
    try:
        r = run_algo(name, fn, g, lb)
        print(f"{name:<18s} {r['nc_acc']:>8.4f} {r['mlp_acc']:>8.4f} {r['mlp_f1']:>8.4f} {r['time']:>7.3f}s {r['mem']:>7.1f}")
    except Exception as e:
        print(f"{name:<18s} ERROR: {str(e)[:60]}")
    sys.stdout.flush()
    gc.collect()

# === Scale: Yelp & roadNet ===
print("\n=== SCALE TEST ===")
for ds_name in ["yelp", "roadnet"]:
    ds = load_dataset(ds_name)
    g = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
    emb, t, mem = measure(lambda g: embed(g, DIM, 4), g)
    print(f"{ds_name}: nodes={ds['num_nodes']}, time={t:.3f}s, mem={mem:.1f}MB")
    gc.collect()

print("\nDONE!")
