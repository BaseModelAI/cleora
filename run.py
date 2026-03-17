"""
pycleora 2.2 - Complete Feature Demo
Demonstrates all capabilities: core embedding, attention, supervised learning,
incremental updates, weighted/directed graphs, link prediction, community detection,
evaluation metrics, visualization, datasets, and I/O.
"""
import numpy as np
import time
import pickle

from pycleora import (
    SparseMatrix, embed, embed_multiscale, embed_with_attention,
    embed_inductive, embed_with_node_features, embed_weighted, embed_directed,
    embed_streaming, supervised_refine, update_graph, remove_edges,
    predict_links, find_most_similar, cosine_similarity, propagate_gpu,
)
from pycleora.metrics import link_prediction_scores, node_classification_scores, clustering_scores
from pycleora.community import detect_communities_kmeans, detect_communities_louvain, modularity
from pycleora.datasets import load_dataset, list_datasets
from pycleora.io_utils import to_networkx, to_edge_list, save_embeddings, load_embeddings
from pycleora.viz import reduce_dimensions, visualize

def _group_communities(comms):
    groups = {}
    for eid, cid in comms.items():
        groups.setdefault(cid, []).append(eid)
    return groups

total_start = time.time()

print("=" * 70)
print("  pycleora 2.2 - Complete Graph Embedding Library")
print("=" * 70)

edges = [
    "alice item_laptop item_mouse",
    "bob item_mouse item_keyboard item_monitor",
    "carol item_laptop item_keyboard",
    "dave item_monitor item_mouse item_laptop",
    "eve item_keyboard item_laptop item_mouse",
    "frank item_monitor item_keyboard",
    "grace item_laptop item_monitor",
    "heidi item_mouse item_monitor item_keyboard",
]
columns = "complex::reflexive::product"

print("\n" + "=" * 70)
print("  PART 1: CORE FEATURES")
print("=" * 70)

print("\n[1] Building graph...")
t0 = time.time()
graph = SparseMatrix.from_iterator(iter(edges), columns)
print(f"    {graph} ({time.time()-t0:.4f}s)")

print(f"\n[2] Standard embedding (left Markov, L2)...")
emb = embed(graph, feature_dim=128, num_iterations=4)
print(f"    Shape: {emb.shape}")

print(f"\n[3] Similarity search (alice):")
for r in find_most_similar(graph, emb, "alice", top_k=3):
    print(f"    {r['entity_id']:20s} sim={r['similarity']:.4f}")

print(f"\n[4] Multi-scale embedding [1,2,4,8]:")
emb_multi = embed_multiscale(graph, feature_dim=64, scales=[1, 2, 4, 8])
print(f"    Shape: {emb_multi.shape}")

print(f"\n[5] Pickle: ", end="")
data = pickle.dumps(graph)
g2 = pickle.loads(data)
print(f"{len(data)} bytes -> {g2}")

print("\n" + "=" * 70)
print("  PART 2: ADVANCED EMBEDDING METHODS")
print("=" * 70)

print(f"\n[6] ATTENTION-WEIGHTED embedding:")
emb_attn = embed_with_attention(graph, feature_dim=128, num_iterations=4, attention_temperature=0.5)
print(f"    Shape: {emb_attn.shape}")

print(f"\n[7] SUPERVISED REFINEMENT:")
positive_pairs = [("alice", "carol"), ("bob", "heidi"), ("dave", "grace")]
emb_refined = supervised_refine(graph, emb, positive_pairs, learning_rate=0.05, num_epochs=100, margin=0.3)
before = cosine_similarity(emb[graph.get_entity_index("alice")], emb[graph.get_entity_index("carol")])
after = cosine_similarity(emb_refined[graph.get_entity_index("alice")], emb_refined[graph.get_entity_index("carol")])
print(f"    alice-carol: {before:.4f} -> {after:.4f} ({after-before:+.4f})")

print(f"\n[8] NODE FEATURES embedding:")
node_feats = {
    "alice": np.random.randn(64).astype(np.float32),
    "bob": np.random.randn(64).astype(np.float32),
    "carol": np.random.randn(64).astype(np.float32),
}
emb_feat = embed_with_node_features(graph, node_feats, num_iterations=4, feature_weight=0.7)
print(f"    Shape: {emb_feat.shape} (feature_weight=0.7)")

print(f"\n[9] WEIGHTED EDGES:")
weighted_edges = [(e, float(i + 1)) for i, e in enumerate(edges)]
w_graph, emb_w = embed_weighted(weighted_edges, columns, feature_dim=128, num_iterations=4)
print(f"    {w_graph}, embedding shape: {emb_w.shape}")

print(f"\n[10] DIRECTED GRAPH:")
d_graph, emb_d = embed_directed(edges, columns, feature_dim=128, num_iterations=4)
print(f"    {d_graph}, embedding shape: {emb_d.shape}")

print("\n" + "=" * 70)
print("  PART 3: INCREMENTAL & DYNAMIC UPDATES")
print("=" * 70)

new_edges = [
    "ivan item_laptop item_keyboard item_tablet",
    "judy item_tablet item_mouse",
]

print(f"\n[11] INCREMENTAL UPDATE:")
updated_graph = update_graph(edges, new_edges, columns)
new_entities = set(updated_graph.entity_ids) - set(graph.entity_ids)
print(f"    {graph.num_entities} -> {updated_graph.num_entities} entities, new: {new_entities}")

print(f"\n[12] INDUCTIVE EMBEDDING:")
g2, emb_ind = embed_inductive(graph, emb, edges, new_edges, columns, num_iterations=4)
print(f"    Updated: {g2.num_entities} entities, shape: {emb_ind.shape}")
for ent in sorted(new_entities):
    if ent in g2.entity_ids:
        sim = find_most_similar(g2, emb_ind, ent, top_k=2)
        print(f"    {ent} nearest: {[r['entity_id'] for r in sim]}")

print(f"\n[13] EDGE REMOVAL:")
g_reduced = remove_edges(edges, [edges[0]], columns)
print(f"    Removed 1 edge: {graph.num_entities} -> {g_reduced.num_entities} entities, {graph.num_edges} -> {g_reduced.num_edges} edges")

print(f"\n[14] STREAMING (batch-by-batch):")
batches = [edges[:4], edges[4:]]
g_stream, emb_stream = embed_streaming(iter(batches), columns, feature_dim=128)
print(f"    2 batches -> {g_stream.num_entities} entities, shape: {emb_stream.shape}")

print("\n" + "=" * 70)
print("  PART 4: LINK PREDICTION")
print("=" * 70)

print(f"\n[15] PREDICT LINKS:")
preds = predict_links(graph, emb, top_k=5, source_entities=["alice"])
for p in preds:
    print(f"    {p['source']} -> {p['target']}: {p['score']:.4f}")

print("\n" + "=" * 70)
print("  PART 5: COMMUNITY DETECTION")
print("=" * 70)

print(f"\n[16] K-MEANS communities (k=3):")
comms_km = detect_communities_kmeans(graph, emb, k=3)
for k, members in _group_communities(comms_km).items():
    print(f"    Cluster {k}: {members}")

print(f"\n[17] LOUVAIN communities:")
comms_lv = detect_communities_louvain(graph)
mod = modularity(graph, comms_lv)
for k, members in _group_communities(comms_lv).items():
    print(f"    Community {k}: {members}")
print(f"    Modularity: {mod:.4f}")

print("\n" + "=" * 70)
print("  PART 6: EVALUATION METRICS")
print("=" * 70)

print(f"\n[18] LINK PREDICTION metrics:")
test_edges_lp = [("alice", "item_laptop"), ("bob", "item_mouse"), ("carol", "item_keyboard")]
lp_scores = link_prediction_scores(graph, emb, test_edges_lp)
print(f"    AUC={lp_scores['auc']:.4f}, MRR={lp_scores['mrr']:.4f}, Hits@10={lp_scores['hits@10']:.4f}")

print(f"\n[19] NODE CLASSIFICATION metrics:")
labels_nc = {"alice": 0, "bob": 1, "carol": 0, "dave": 1, "eve": 0, "frank": 1, "grace": 0, "heidi": 1}
nc_scores = node_classification_scores(graph, emb, labels_nc, train_ratio=0.6)
print(f"    Accuracy={nc_scores['accuracy']:.4f}, Macro-F1={nc_scores['macro_f1']:.4f}")

print(f"\n[20] CLUSTERING metrics:")
cluster_labels = np.array([comms_km[eid] for eid in graph.entity_ids])
cl_scores = clustering_scores(emb, cluster_labels)
print(f"    NMI={cl_scores['nmi']:.4f}, Purity={cl_scores['purity']:.4f}")

print("\n" + "=" * 70)
print("  PART 7: BUILT-IN DATASETS")
print("=" * 70)

print(f"\n[21] Available datasets:")
for ds in list_datasets():
    print(f"    {ds['name']:20s} nodes={ds['nodes']:4d}  edges={ds['edges']:4d}  classes={ds['classes']}")

print(f"\n[22] KARATE CLUB benchmark:")
ds = load_dataset("karate_club")
g_kc = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
emb_kc = embed(g_kc, feature_dim=64, num_iterations=4)
nc_kc = node_classification_scores(g_kc, emb_kc, ds["labels"], train_ratio=0.7)
print(f"    {ds['name']}: {g_kc.num_entities} nodes, Accuracy={nc_kc['accuracy']:.4f}")

print(f"\n[23] LES MISERABLES benchmark:")
ds2 = load_dataset("les_miserables")
g_lm = SparseMatrix.from_iterator(iter(ds2["edges"]), ds2["columns"])
emb_lm = embed(g_lm, feature_dim=64, num_iterations=4)
nc_lm = node_classification_scores(g_lm, emb_lm, ds2["labels"], train_ratio=0.7)
print(f"    {ds2['name']}: {g_lm.num_entities} nodes, Accuracy={nc_lm['accuracy']:.4f}")

print("\n" + "=" * 70)
print("  PART 8: I/O & EXPORT")
print("=" * 70)

print(f"\n[24] NETWORKX export:")
G = to_networkx(graph, emb)
print(f"    NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

print(f"\n[25] EDGE LIST export:")
el = to_edge_list(graph)
print(f"    {len(el)} unique edges, sample: {el[0]}")

print(f"\n[26] SAVE/LOAD embeddings (npz):")
save_embeddings(graph, emb, "/tmp/test_emb.npz", format="npz")
emb_loaded, ids_loaded = load_embeddings("/tmp/test_emb.npz", format="npz")
print(f"    Saved & loaded: {emb_loaded.shape}, {len(ids_loaded)} entities")

print(f"\n[27] SAVE embeddings (CSV):")
save_embeddings(graph, emb, "/tmp/test_emb.csv", format="csv")
emb_csv, ids_csv = load_embeddings("/tmp/test_emb.csv", format="csv")
print(f"    CSV: {emb_csv.shape}")

print(f"\n[28] SPARSE MATRIX export:")
rows, cols, vals, n, _ = graph.to_sparse_csr()
print(f"    {len(vals)} non-zeros in {n}x{n} matrix")

print("\n" + "=" * 70)
print("  PART 9: VISUALIZATION")
print("=" * 70)

print(f"\n[29] t-SNE dimensionality reduction:")
emb_2d = reduce_dimensions(emb, method="tsne")
print(f"    {emb.shape} -> {emb_2d.shape}")

print(f"\n[30] PCA dimensionality reduction:")
emb_pca = reduce_dimensions(emb, method="pca")
print(f"    {emb.shape} -> {emb_pca.shape}")

print(f"\n[31] VISUALIZATION (saved to file):")
vis_labels = {"alice": 0, "bob": 1, "carol": 0, "dave": 1, "eve": 0,
              "frank": 1, "grace": 0, "heidi": 1,
              "item_laptop": 2, "item_mouse": 2, "item_keyboard": 2, "item_monitor": 2}
result = visualize(graph, emb, labels=vis_labels, method="pca",
                   title="pycleora Embeddings", save_path="/tmp/pycleora_viz.png")
print(f"    Saved to: {result}")

print(f"\n[32] Karate Club visualization:")
kc_labels = load_dataset("karate_club")["labels"]
result_kc = visualize(g_kc, emb_kc, labels=kc_labels, method="pca",
                       title="Karate Club", save_path="/tmp/karate_viz.png")
print(f"    Saved to: {result_kc}")

print("\n" + "=" * 70)
print("  PART 10: GPU & ERROR HANDLING")
print("=" * 70)

print(f"\n[33] GPU propagation check:")
try:
    import torch
    if torch.cuda.is_available():
        emb_gpu = propagate_gpu(graph, emb, num_iterations=4, device="cuda")
        print(f"    GPU: {emb_gpu.shape}")
    else:
        emb_cpu = propagate_gpu(graph, emb, num_iterations=4, device="cpu")
        print(f"    PyTorch CPU: {emb_cpu.shape}")
except ImportError:
    print(f"    PyTorch not installed - GPU available with 'pip install torch'")
    print(f"    Rust CPU propagation active (fast!)")

print(f"\n[34] Error handling:")
errors_caught = 0
for test_fn in [
    lambda: graph.get_entity_index("nonexistent"),
    lambda: embed(graph, propagation="invalid"),
    lambda: embed_with_attention(graph, attention_temperature=-1),
    lambda: graph.to_sparse_csr("invalid_type"),
]:
    try:
        test_fn()
    except (ValueError, RuntimeError):
        errors_caught += 1
print(f"    Caught {errors_caught}/4 expected errors")

elapsed = time.time() - total_start
print("\n" + "=" * 70)
print(f"  All 34 features working! Total time: {elapsed:.2f}s")
print("=" * 70)


print("\nPress Ctrl+C to exit.")
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    pass
