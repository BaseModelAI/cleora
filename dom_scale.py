import numpy as np, time, tracemalloc, gc
from pycleora import SparseMatrix, embed
from pycleora.datasets import load_dataset
DIM = 256

def m(fn, g):
    gc.collect(); tracemalloc.start()
    t0 = time.time(); r = fn(g); t = time.time() - t0
    _, p = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return r, t, p/1024/1024

# Test 16 iterations on yelp and roadnet
for ds_name in ["yelp", "roadnet"]:
    ds = load_dataset(ds_name)
    g = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
    print(f"{ds_name}: nodes={ds['num_nodes']}, edges={ds['num_edges']}")
    
    # 4 iterations (baseline)
    emb, t, mem = m(lambda g: embed(g, DIM, 4), g)
    print(f"  4 iter: t={t:.3f}s, mem={mem:.1f}MB")
    del emb; gc.collect()
    
    # 8 iterations
    emb, t, mem = m(lambda g: embed(g, DIM, 8), g)
    print(f"  8 iter: t={t:.3f}s, mem={mem:.1f}MB")
    del emb; gc.collect()
    
    # 16 iterations 
    emb, t, mem = m(lambda g: embed(g, DIM, 16), g)
    print(f"  16 iter: t={t:.3f}s, mem={mem:.1f}MB")
    del emb; gc.collect()
