"""
DOMINATION BENCHMARK — Find the absolute best Cleora config
Tests: MLP classifier, multiscale, residual weights
"""
import numpy as np
import time
import tracemalloc
import sys
import gc

from pycleora import SparseMatrix, embed, embed_multiscale
from pycleora.algorithms import (
    embed_prone, embed_randne, embed_netmf, embed_deepwalk,
)
from pycleora.classify import mlp_classify
from pycleora.metrics import node_classification_scores
from pycleora.community import detect_communities_louvain
from pycleora.datasets import load_dataset

DIM = 256

def measure(fn, graph):
    gc.collect()
    tracemalloc.start()
    t0 = time.time()
    result = fn(graph)
    elapsed = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / 1024 / 1024

def get_labels(ds, graph):
    labels = ds.get("labels", {})
    if len(labels) < 4:
        sys.stderr.write("  Generating Louvain community labels...\n")
        sys.stderr.flush()
        labels = detect_communities_louvain(graph)
    return labels

def classify_nc(graph, emb, labels):
    return node_classification_scores(graph, emb, labels, seed=42)

def classify_mlp(graph, emb, labels, hidden=128, epochs=300, lr=0.01):
    return mlp_classify(graph, emb, labels, hidden_dim=hidden, 
                       num_epochs=epochs, learning_rate=lr, seed=42)

# ============================================
# PHASE 1: Cleora hyperparameter sweep on Facebook
# ============================================
print("=" * 80)
print("PHASE 1: CLEORA HYPERPARAMETER SWEEP on ego-Facebook")
print("=" * 80)

ds = load_dataset("facebook")
graph = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
labels = get_labels(ds, graph)
print(f"Nodes: {ds['num_nodes']}, Labels: {len(labels)}")

configs = [
    ("Cleora(w,4it)", lambda g: embed(g, DIM, 4, whiten=True)),
    ("Cleora(w,8it)", lambda g: embed(g, DIM, 8, whiten=True)),
    ("Cleora(w,12it)", lambda g: embed(g, DIM, 12, whiten=True)),
    ("Cleora(w,16it)", lambda g: embed(g, DIM, 16, whiten=True)),
    ("Cleora(w,8,r0.1)", lambda g: embed(g, DIM, 8, whiten=True, residual_weight=0.1)),
    ("Cleora(w,8,r0.3)", lambda g: embed(g, DIM, 8, whiten=True, residual_weight=0.3)),
    ("Cleora(w,8,r0.5)", lambda g: embed(g, DIM, 8, whiten=True, residual_weight=0.5)),
    ("Cleora(ms[2,4,8],w)", lambda g: embed_multiscale(g, DIM, scales=[2,4,8], whiten=True)),
    ("Cleora(ms[1,2,4,8],w)", lambda g: embed_multiscale(g, DIM, scales=[1,2,4,8], whiten=True)),
    ("Cleora(ms[4,8,12],w)", lambda g: embed_multiscale(g, DIM, scales=[4,8,12], whiten=True)),
]

print(f"\n{'Config':<24s} {'NC_Acc':>8s} {'NC_F1':>8s} {'MLP_Acc':>8s} {'MLP_F1':>8s} {'Time':>8s}")
print("-" * 80)

for name, fn in configs:
    try:
        emb, t, mem = measure(fn, graph)
        nc = classify_nc(graph, emb, labels)
        mlp = classify_mlp(graph, emb, labels, hidden=128, epochs=300)
        print(f"{name:<24s} {nc['accuracy']:>8.4f} {nc['macro_f1']:>8.4f} {mlp['accuracy']:>8.4f} {mlp['macro_f1']:>8.4f} {t:>7.3f}s")
        sys.stdout.flush()
    except Exception as e:
        print(f"{name:<24s} ERROR: {e}")
        sys.stdout.flush()

# ============================================
# PHASE 2: MLP tuning on best config
# ============================================
print("\n" + "=" * 80)
print("PHASE 2: MLP HYPERPARAMETER TUNING")
print("=" * 80)

emb_w8 = embed(graph, DIM, 8, whiten=True)

mlp_cfgs = [
    ("h64,e200", 64, 200, 0.01),
    ("h128,e300", 128, 300, 0.01),
    ("h256,e300", 256, 300, 0.01),
    ("h128,e500", 128, 500, 0.005),
    ("h256,e500", 256, 500, 0.005),
    ("h256,e500,lr0.01", 256, 500, 0.01),
    ("h512,e500", 512, 500, 0.005),
]

print(f"\n{'MLP Config':<20s} {'Accuracy':>10s} {'Macro_F1':>10s}")
print("-" * 45)

for name, h, e, lr in mlp_cfgs:
    mlp = classify_mlp(graph, emb_w8, labels, hidden=h, epochs=e, lr=lr)
    print(f"{name:<20s} {mlp['accuracy']:>10.4f} {mlp['macro_f1']:>10.4f}")
    sys.stdout.flush()

# ============================================
# PHASE 3: ALL ALGORITHMS with best classifier — Facebook
# ============================================
print("\n" + "=" * 80)
print("PHASE 3: ALL ALGORITHMS — ego-Facebook (NC + MLP)")
print("=" * 80)

algos_fb = [
    ("Cleora(w,8it)", lambda g: embed(g, DIM, 8, whiten=True)),
    ("Cleora(ms,w)", lambda g: embed_multiscale(g, DIM, scales=[2,4,8], whiten=True)),
    ("Cleora(base)", lambda g: embed(g, DIM, 4)),
    ("NetMF", lambda g: embed_netmf(g, DIM)),
    ("DeepWalk", lambda g: embed_deepwalk(g, DIM, num_walks=20, walk_length=40)),
    ("ProNE", lambda g: embed_prone(g, DIM)),
    ("RandNE", lambda g: embed_randne(g, DIM)),
]

print(f"\n{'Algorithm':<18s} {'NC_Acc':>8s} {'NC_F1':>8s} {'MLP_Acc':>8s} {'MLP_F1':>8s} {'Time':>8s} {'Mem_MB':>8s}")
print("-" * 85)

for name, fn in algos_fb:
    try:
        emb, t, mem = measure(fn, graph)
        nc = classify_nc(graph, emb, labels)
        mlp = classify_mlp(graph, emb, labels, hidden=256, epochs=500, lr=0.005)
        print(f"{name:<18s} {nc['accuracy']:>8.4f} {nc['macro_f1']:>8.4f} {mlp['accuracy']:>8.4f} {mlp['macro_f1']:>8.4f} {t:>7.3f}s {mem:>7.1f}")
    except Exception as e:
        print(f"{name:<18s} ERROR: {str(e)[:60]}")
    sys.stdout.flush()
    gc.collect()

# ============================================
# PHASE 4: FULL DATASET SWEEP
# ============================================
print("\n" + "=" * 80)
print("PHASE 4: FULL DATASET SWEEP")
print("=" * 80)

datasets = ["ppi_large", "flickr", "ogbn_arxiv"]

for ds_name in datasets:
    print(f"\n--- {ds_name.upper()} ---")
    ds = load_dataset(ds_name)
    graph = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
    labels = get_labels(ds, graph)
    n = ds['num_nodes']
    print(f"Nodes: {n}, Labels: {len(labels)}")
    
    test_algos = [
        ("Cleora(w,8it)", lambda g: embed(g, DIM, 8, whiten=True)),
        ("Cleora(base)", lambda g: embed(g, DIM, 4)),
        ("ProNE", lambda g: embed_prone(g, DIM)),
        ("RandNE", lambda g: embed_randne(g, DIM)),
    ]
    
    if n <= 60000:
        test_algos.append(("NetMF", lambda g: embed_netmf(g, DIM)))
        test_algos.append(("DeepWalk", lambda g: embed_deepwalk(g, DIM, num_walks=10, walk_length=20)))
    
    print(f"{'Algorithm':<18s} {'NC_Acc':>8s} {'NC_F1':>8s} {'MLP_Acc':>8s} {'MLP_F1':>8s} {'Time':>8s} {'Mem_MB':>8s}")
    print("-" * 85)
    
    for aname, afn in test_algos:
        try:
            emb, t, mem = measure(afn, graph)
            nc = classify_nc(graph, emb, labels)
            mlp = classify_mlp(graph, emb, labels, hidden=128, epochs=300)
            print(f"{aname:<18s} {nc['accuracy']:>8.4f} {nc['macro_f1']:>8.4f} {mlp['accuracy']:>8.4f} {mlp['macro_f1']:>8.4f} {t:>7.3f}s {mem:>7.1f}")
        except Exception as e:
            print(f"{aname:<18s} ERROR: {str(e)[:60]}")
        sys.stdout.flush()
        gc.collect()

# ============================================
# PHASE 5: SCALE TEST — Yelp & roadNet
# ============================================
print("\n" + "=" * 80)
print("PHASE 5: SCALE — Yelp & roadNet (Cleora only)")
print("=" * 80)

for ds_name in ["yelp", "roadnet"]:
    print(f"\n--- {ds_name.upper()} ---")
    ds = load_dataset(ds_name)
    graph = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
    n = ds['num_nodes']
    print(f"Nodes: {n}, Edges: {ds['num_edges']}")
    
    emb, t, mem = measure(lambda g: embed(g, DIM, 4), graph)
    print(f"Cleora(base): time={t:.3f}s, mem={mem:.1f}MB")
    gc.collect()

print("\n\nDOMINATION BENCHMARK COMPLETE!")
