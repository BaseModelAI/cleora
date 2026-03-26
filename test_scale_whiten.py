import numpy as np, time, tracemalloc, gc, sys
from pycleora import SparseMatrix, embed
from pycleora.datasets import load_dataset

DIM = 256

for ds_name in ["yelp", "roadnet"]:
    print(f"\n=== {ds_name} ===", flush=True)
    ds = load_dataset(ds_name)
    print(f"Loaded: {ds['num_nodes']} nodes, {len(ds['edges'])} edges", flush=True)
    
    g = SparseMatrix.from_iterator(iter(ds['edges']), ds['columns'])
    print(f"Graph: {g.num_entities} entities", flush=True)
    
    gc.collect()
    tracemalloc.start()
    t0 = time.time()
    emb = embed(g, DIM, 4)
    t_base = time.time() - t0
    _, peak_base = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  Base(4it): time={t_base:.1f}s, mem={peak_base/1e6:.0f}MB", flush=True)
    del emb; gc.collect()
    
    gc.collect()
    tracemalloc.start()
    t0 = time.time()
    emb = embed(g, DIM, 4, whiten=True)
    t_w = time.time() - t0
    _, peak_w = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  Whiten(4it): time={t_w:.1f}s, mem={peak_w/1e6:.0f}MB", flush=True)
    del emb; gc.collect()

    gc.collect()
    tracemalloc.start()
    t0 = time.time()
    emb = embed(g, DIM, 8, whiten=True)
    t_w8 = time.time() - t0
    _, peak_w8 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  Whiten(8it): time={t_w8:.1f}s, mem={peak_w8/1e6:.0f}MB", flush=True)
    del emb; gc.collect()

    gc.collect()
    tracemalloc.start()
    t0 = time.time()
    emb = embed(g, DIM, 16, whiten=True)
    t_w16 = time.time() - t0
    _, peak_w16 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  Whiten(16it): time={t_w16:.1f}s, mem={peak_w16/1e6:.0f}MB", flush=True)
    del emb; gc.collect()
    
    del g; gc.collect()

print("\nSUCCESS - All scale tests passed!", flush=True)
