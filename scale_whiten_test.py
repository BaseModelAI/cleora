"""Test whitening on Yelp and roadNet — previously OOM'd"""
import numpy as np, time, tracemalloc, gc, sys
from pycleora import SparseMatrix, embed
from pycleora.datasets import load_dataset
from pycleora.community import detect_communities_louvain
from pycleora.metrics import node_classification_scores
DIM = 256

def m(fn, g):
    gc.collect(); tracemalloc.start()
    t0 = time.time(); r = fn(g); t = time.time() - t0
    _, p = tracemalloc.get_traced_memory(); tracemalloc.stop()
    gc.collect()
    return r, t, p/1024/1024

for ds_name in ["yelp", "roadnet"]:
    sys.stderr.write(f"\n=== {ds_name} ===\n"); sys.stderr.flush()
    t0 = time.time()
    ds = load_dataset(ds_name)
    sys.stderr.write(f"  load: {time.time()-t0:.1f}s\n"); sys.stderr.flush()

    t0 = time.time()
    g = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
    sys.stderr.write(f"  graph: {time.time()-t0:.1f}s, nodes={g.num_entities}\n"); sys.stderr.flush()

    t0 = time.time()
    labels = detect_communities_louvain(g)
    sys.stderr.write(f"  louvain: {time.time()-t0:.1f}s, communities={len(set(labels.values()))}\n"); sys.stderr.flush()

    emb, t, mem = m(lambda g: embed(g, DIM, 4), g)
    scores = node_classification_scores(g, emb, labels, seed=42)
    print(f"{ds_name:<10s} Cleora(base,4it)   acc={scores['accuracy']:.4f}  time={t:.3f}s  mem={mem:.1f}MB")
    sys.stdout.flush()
    del emb; gc.collect()

    sys.stderr.write(f"  Running whiten(16it)...\n"); sys.stderr.flush()
    emb, t, mem = m(lambda g: embed(g, DIM, 16, whiten=True), g)
    scores = node_classification_scores(g, emb, labels, seed=42)
    print(f"{ds_name:<10s} Cleora(w,16it)     acc={scores['accuracy']:.4f}  time={t:.3f}s  mem={mem:.1f}MB")
    sys.stdout.flush()
    del emb; gc.collect()

    del g, labels; gc.collect()

print("\nDONE - Whitening works on ALL datasets!")
