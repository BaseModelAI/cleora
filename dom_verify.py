import numpy as np, time, tracemalloc, gc
from pycleora import SparseMatrix, embed
from pycleora.algorithms import embed_netmf, embed_deepwalk
from pycleora.classify import mlp_classify
from pycleora.metrics import node_classification_scores, cross_validate
from pycleora.community import detect_communities_louvain
from pycleora.datasets import load_dataset
DIM = 256

def m(fn, g):
    gc.collect(); tracemalloc.start()
    t0 = time.time(); r = fn(g); t = time.time() - t0
    _, p = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return r, t, p/1024/1024

ds = load_dataset("facebook")
g = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
lb = detect_communities_louvain(g)

# Verify Cleora(w,16it) 3 runs for consistency  
print("=== CLEORA(w,16it) - 3 RUNS ===")
for run in range(3):
    emb, t, mem = m(lambda g: embed(g, DIM, 16, whiten=True), g)
    nc = node_classification_scores(g, emb, lb, seed=42)
    print(f"Run {run+1}: NC={nc['accuracy']:.4f} F1={nc['macro_f1']:.4f} t={t:.3f}s mem={mem:.1f}MB")
    gc.collect()

# Verify MLP stability
print("\n=== CLEORA(w,16it) MLP - 3 seeds ===")
emb16, t, mem = m(lambda g: embed(g, DIM, 16, whiten=True), g)
for seed in [42, 123, 777]:
    mlp = mlp_classify(g, emb16, lb, hidden_dim=256, num_epochs=400, learning_rate=0.005, seed=seed)
    print(f"Seed {seed}: MLP_acc={mlp['accuracy']:.4f} MLP_f1={mlp['macro_f1']:.4f}")

# Cross-validation on Cleora(w,16it)
print("\n=== CLEORA(w,16it) CROSS-VALIDATION ===")
cv = cross_validate(g, emb16, lb, k_folds=5, seed=42)
print(f"CV Acc: {cv['mean_accuracy']:.4f} ± {cv['std_accuracy']:.4f}")
print(f"CV F1:  {cv['mean_macro_f1']:.4f} ± {cv['std_macro_f1']:.4f}")

# Also check: does Cleora(w,8it) win on NC with slightly different seed?
print("\n=== STABILITY: Cleora(w,8it) vs NetMF NC ===")
for seed in [42, 0, 123]:
    emb8, _, _ = m(lambda g: embed(g, DIM, 8, whiten=True), g)
    nc8 = node_classification_scores(g, emb8, lb, seed=seed)
    print(f"Cleora(w,8it) seed={seed}: NC={nc8['accuracy']:.4f}")

print("\nDONE!")
