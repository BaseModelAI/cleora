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

def run(name, fn, g, lb, mlp_h=64, mlp_e=100):
    emb, t, mem = measure(fn, g)
    nc = node_classification_scores(g, emb, lb, seed=42)
    try:
        mlp = mlp_classify(g, emb, lb, hidden_dim=mlp_h, num_epochs=mlp_e, learning_rate=0.01, seed=42)
        mlp_acc = mlp['accuracy']
        mlp_f1 = mlp['macro_f1']
    except:
        mlp_acc = -1; mlp_f1 = -1
    print(f"{name:<18s} {nc['accuracy']:>8.4f} {nc['macro_f1']:>8.4f} {mlp_acc:>8.4f} {mlp_f1:>8.4f} {t:>7.3f}s {mem:>7.1f}")
    sys.stdout.flush(); gc.collect()

for ds_name in ["ppi_large", "flickr", "ogbn_arxiv"]:
    print(f"\n=== {ds_name.upper()} ===")
    ds = load_dataset(ds_name)
    g = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
    lb = detect_communities_louvain(g)
    print(f"Nodes: {ds['num_nodes']}, Labels: {len(lb)}")
    algos = [
        ("Cleora(w,8it)", lambda g: embed(g, DIM, 8, whiten=True)),
        ("Cleora(base)", lambda g: embed(g, DIM, 4)),
        ("ProNE", lambda g: embed_prone(g, DIM)),
        ("RandNE", lambda g: embed_randne(g, DIM)),
    ]
    for name, fn in algos:
        run(name, fn, g, lb)

print("\nDONE!")
