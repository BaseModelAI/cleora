import numpy as np, time, tracemalloc, gc, sys
from pycleora import SparseMatrix, embed
from pycleora.datasets import load_dataset

DIM = 256
print("Loading yelp...", flush=True)
ds = load_dataset('yelp')
print(f"Loaded: {ds['num_nodes']} nodes, {len(ds['edges'])} edges", flush=True)

print("Building graph...", flush=True)
g = SparseMatrix.from_iterator(iter(ds['edges']), ds['columns'])
print(f"Graph: {g.num_entities} entities", flush=True)

print("Running base embed(4it)...", flush=True)
gc.collect()
tracemalloc.start()
t0 = time.time()
emb = embed(g, DIM, 4)
t = time.time() - t0
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"  Base(4it): shape={emb.shape}, time={t:.1f}s, mem={peak/1e6:.0f}MB", flush=True)
del emb; gc.collect()

print("Running whiten embed(16it)...", flush=True)
gc.collect()
tracemalloc.start()
t0 = time.time()
emb = embed(g, DIM, 16, whiten=True)
t = time.time() - t0
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"  Whiten(16it): shape={emb.shape}, time={t:.1f}s, mem={peak/1e6:.0f}MB", flush=True)
del emb; gc.collect()

print("SUCCESS - Yelp whitening completed without OOM!", flush=True)
