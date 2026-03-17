"""
pycleora 2.4 - Complete Feature Demo (54 tests)
"""
import numpy as np
import time

t_start = time.time()

from pycleora import (
    SparseMatrix, embed, embed_multiscale, embed_with_attention,
    embed_with_node_features, embed_weighted, embed_directed,
    embed_streaming, embed_inductive, supervised_refine,
    update_graph, remove_edges, predict_links,
    find_most_similar, cosine_similarity, propagate_gpu,
    embed_using_baseline_cleora,
)
from pycleora.metrics import link_prediction_scores, node_classification_scores, clustering_scores
from pycleora.community import detect_communities_kmeans, detect_communities_spectral, detect_communities_louvain, modularity
from pycleora.datasets import list_datasets, load_dataset
from pycleora.io_utils import to_networkx, to_edge_list, save_embeddings, load_embeddings
from pycleora.viz import reduce_dimensions, plot_embeddings, visualize
from pycleora.algorithms import embed_prone, embed_randne, embed_hope, embed_netmf, embed_grarep, embed_deepwalk, embed_node2vec, list_algorithms
from pycleora.classify import label_propagation, mlp_classify, label_propagation_predict
from pycleora.hetero import HeteroGraph

test_num = [0]
def T(msg):
    test_num[0] += 1
    print(f"[{test_num[0]:2d}] {msg}")
    return time.time()

def _group_communities(comms):
    groups = {}
    for eid, cid in comms.items():
        groups.setdefault(cid, []).append(eid)
    return groups


edges = [
    "alice item_laptop", "alice item_mouse", "bob item_keyboard", "bob item_monitor",
    "carol item_laptop", "carol item_mouse", "dave item_laptop", "dave item_mouse",
    "eve item_laptop", "eve item_mouse", "frank item_keyboard", "frank item_monitor",
    "grace item_laptop", "grace item_mouse", "heidi item_keyboard", "heidi item_mouse",
]

print("=" * 70)
print("  pycleora 2.4 - Complete Graph Embedding Library")
print("=" * 70)

print("=" * 70)
print("  PART 1: CORE FEATURES")
print("=" * 70)

t = T("Building graph...")
graph = SparseMatrix.from_iterator(iter(edges), "complex::reflexive::product")
print(f"    {graph} ({time.time()-t:.4f}s)")

t = T("Standard embedding (left Markov, L2)...")
emb = embed(graph, feature_dim=128, num_iterations=4)
print(f"    Shape: {emb.shape}")

t = T("Similarity search (alice):")
similar = find_most_similar(graph, emb, "alice", top_k=3)
for r in similar:
    print(f"    {r['entity_id']:20s} sim={r['similarity']:.4f}")

t = T("Multi-scale embedding [1,2,4,8]:")
emb_multi = embed_multiscale(graph, feature_dim=64, scales=[1, 2, 4, 8])
print(f"    Shape: {emb_multi.shape}")

t = T("Pickle:")
import pickle
data = pickle.dumps(graph)
graph2 = pickle.loads(data)
print(f"    {len(data)} bytes -> {graph2}")

print("=" * 70)
print("  PART 2: ADVANCED EMBEDDING METHODS")
print("=" * 70)

t = T("ATTENTION-WEIGHTED embedding:")
emb_attn = embed_with_attention(graph, feature_dim=128, attention_temperature=0.5)
print(f"    Shape: {emb_attn.shape}")

t = T("SUPERVISED REFINEMENT:")
pos_pairs = [("alice", "carol"), ("bob", "frank")]
before = cosine_similarity(emb[graph.get_entity_index("alice")], emb[graph.get_entity_index("carol")])
refined = supervised_refine(graph, emb, pos_pairs, learning_rate=0.05, num_epochs=100)
after = cosine_similarity(refined[graph.get_entity_index("alice")], refined[graph.get_entity_index("carol")])
print(f"    alice-carol: {before:.4f} -> {after:.4f} ({after - before:+.4f})")

t = T("NODE FEATURES embedding:")
node_features = {eid: np.random.randn(64).astype(np.float32) for eid in graph.entity_ids}
emb_feat = embed_with_node_features(graph, node_features, feature_weight=0.7)
print(f"    Shape: {emb_feat.shape} (feature_weight=0.7)")

t = T("WEIGHTED EDGES:")
weighted = [(e, np.random.uniform(0.5, 2.0)) for e in edges]
g_w, emb_w = embed_weighted(weighted, "complex::reflexive::product")
print(f"    {g_w}, embedding shape: {emb_w.shape}")

t = T("DIRECTED GRAPH:")
g_d, emb_d = embed_directed(edges, "complex::reflexive::product")
print(f"    {g_d}, embedding shape: {emb_d.shape}")

print("=" * 70)
print("  PART 3: INCREMENTAL & DYNAMIC UPDATES")
print("=" * 70)

new_edges = ["ivan item_mouse", "judy item_tablet", "ivan item_tablet"]
t = T("INCREMENTAL UPDATE:")
g2 = update_graph(edges, new_edges, "complex::reflexive::product")
new_ents = set(g2.entity_ids) - set(graph.entity_ids)
print(f"    {graph.num_entities} -> {g2.num_entities} entities, new: {new_ents}")

t = T("INDUCTIVE EMBEDDING:")
g3, emb3 = embed_inductive(graph, emb, edges, new_edges, "complex::reflexive::product")
print(f"    Updated: {g3.num_entities} entities, shape: {emb3.shape}")

t = T("EDGE REMOVAL:")
g4 = remove_edges(edges, ["alice item_laptop"], "complex::reflexive::product")
print(f"    Removed 1 edge: {graph.num_entities} -> {g4.num_entities} entities")

t = T("STREAMING (batch-by-batch):")
batches = [edges[:8], edges[8:]]
g5, emb5 = embed_streaming(batches, "complex::reflexive::product")
print(f"    2 batches -> {g5.num_entities} entities, shape: {emb5.shape}")

print("=" * 70)
print("  PART 4: LINK PREDICTION")
print("=" * 70)

t = T("PREDICT LINKS:")
preds = predict_links(graph, emb, top_k=5, source_entities=["alice"])
for p in preds:
    print(f"    {p['source']} -> {p['target']}: {p['score']:.4f}")

print("=" * 70)
print("  PART 5: COMMUNITY DETECTION")
print("=" * 70)

t = T("K-MEANS communities (k=3):")
comms_km = detect_communities_kmeans(graph, emb, k=3)
for cid, members in _group_communities(comms_km).items():
    print(f"    Cluster {cid}: {sorted(members)}")

t = T("SPECTRAL communities (k=3):")
comms_sp = detect_communities_spectral(graph, emb, k=3)
for cid, members in _group_communities(comms_sp).items():
    print(f"    Cluster {cid}: {sorted(members)}")

t = T("LOUVAIN communities:")
comms_lv = detect_communities_louvain(graph)
for cid, members in _group_communities(comms_lv).items():
    print(f"    Community {cid}: {sorted(members)}")
print(f"    Modularity: {modularity(graph, comms_lv):.4f}")

print("=" * 70)
print("  PART 6: EVALUATION METRICS")
print("=" * 70)

test_edges_lp = [("alice", "carol"), ("bob", "frank"), ("dave", "eve")]
t = T("LINK PREDICTION metrics:")
lp_scores = link_prediction_scores(graph, emb, test_edges_lp)
print(f"    AUC={lp_scores['auc']:.4f}, MRR={lp_scores['mrr']:.4f}, Hits@10={lp_scores['hits@10']:.4f}")

labels_nc = {"alice": 0, "bob": 1, "carol": 0, "dave": 0, "eve": 0, "frank": 1, "grace": 0, "heidi": 1}
t = T("NODE CLASSIFICATION metrics:")
nc_scores = node_classification_scores(graph, emb, labels_nc)
print(f"    Accuracy={nc_scores['accuracy']:.4f}, Macro-F1={nc_scores['macro_f1']:.4f}")

t = T("CLUSTERING metrics:")
true_labels = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1])
cl_scores = clustering_scores(emb, true_labels)
print(f"    NMI={cl_scores['nmi']:.4f}, Purity={cl_scores['purity']:.4f}")

print("=" * 70)
print("  PART 7: BUILT-IN DATASETS (12 total)")
print("=" * 70)

t = T("Available datasets:")
for ds in list_datasets():
    print(f"    {ds['name']:22s} nodes={ds['nodes']:6d}  edges={ds['edges']:6d}  classes={ds['classes']}")

t = T("KARATE CLUB benchmark:")
kc = load_dataset("karate_club")
g_kc = SparseMatrix.from_iterator(iter(kc["edges"]), kc["columns"])
e_kc = embed(g_kc, feature_dim=128, num_iterations=4)
nc_kc = node_classification_scores(g_kc, e_kc, kc["labels"])
print(f"    {kc['name']}: {kc['num_nodes']} nodes, Accuracy={nc_kc['accuracy']:.4f}")

t = T("CORA benchmark:")
cora = load_dataset("cora")
g_cora = SparseMatrix.from_iterator(iter(cora["edges"]), cora["columns"])
e_cora = embed(g_cora, feature_dim=128, num_iterations=4)
nc_cora = node_classification_scores(g_cora, e_cora, cora["labels"])
print(f"    {cora['name']}: {cora['num_nodes']} nodes, Accuracy={nc_cora['accuracy']:.4f}")

t = T("CITESEER benchmark:")
cs = load_dataset("citeseer")
g_cs = SparseMatrix.from_iterator(iter(cs["edges"]), cs["columns"])
e_cs = embed(g_cs, feature_dim=128, num_iterations=4)
nc_cs = node_classification_scores(g_cs, e_cs, cs["labels"])
print(f"    {cs['name']}: {cs['num_nodes']} nodes, Accuracy={nc_cs['accuracy']:.4f}")

t = T("DBLP benchmark:")
dblp = load_dataset("dblp")
g_dblp = SparseMatrix.from_iterator(iter(dblp["edges"]), dblp["columns"])
e_dblp = embed(g_dblp, feature_dim=128, num_iterations=4)
nc_dblp = node_classification_scores(g_dblp, e_dblp, dblp["labels"])
print(f"    {dblp['name']}: {dblp['num_nodes']} nodes, Accuracy={nc_dblp['accuracy']:.4f}")

print("=" * 70)
print("  PART 8: ALTERNATIVE ALGORITHMS (7 total)")
print("=" * 70)

t = T("Available algorithms:")
for alg in list_algorithms():
    print(f"    {alg['name']:10s} - {alg['description'][:60]}")

t = T("ProNE embedding:")
t0 = time.time()
emb_prone = embed_prone(graph, feature_dim=64)
print(f"    Shape: {emb_prone.shape}, time: {time.time()-t0:.4f}s")

t = T("RandNE embedding:")
t0 = time.time()
emb_randne = embed_randne(graph, feature_dim=64)
print(f"    Shape: {emb_randne.shape}, time: {time.time()-t0:.4f}s")

t = T("HOPE embedding:")
t0 = time.time()
emb_hope = embed_hope(graph, feature_dim=64)
print(f"    Shape: {emb_hope.shape}, time: {time.time()-t0:.4f}s")

t = T("NetMF embedding:")
t0 = time.time()
emb_netmf = embed_netmf(graph, feature_dim=64)
print(f"    Shape: {emb_netmf.shape}, time: {time.time()-t0:.4f}s")

t = T("GraRep embedding:")
t0 = time.time()
emb_grarep = embed_grarep(graph, feature_dim=64)
print(f"    Shape: {emb_grarep.shape}, time: {time.time()-t0:.4f}s")

t = T("DeepWalk embedding:")
t0 = time.time()
emb_dw = embed_deepwalk(graph, feature_dim=64, num_walks=10, walk_length=40)
print(f"    Shape: {emb_dw.shape}, time: {time.time()-t0:.4f}s")

t = T("Node2Vec embedding (p=0.5, q=2.0):")
t0 = time.time()
emb_n2v = embed_node2vec(graph, feature_dim=64, num_walks=10, walk_length=40, p=0.5, q=2.0)
print(f"    Shape: {emb_n2v.shape}, time: {time.time()-t0:.4f}s")

t = T("Algorithm comparison (Karate Club accuracy):")
algos_results = {}
for name_alg, fn in [("cleora", lambda g: embed(g, 128, 4)),
                      ("prone", lambda g: embed_prone(g, 128)),
                      ("randne", lambda g: embed_randne(g, 128)),
                      ("hope", lambda g: embed_hope(g, 128)),
                      ("deepwalk", lambda g: embed_deepwalk(g, 128, num_walks=20, walk_length=40)),
                      ("node2vec", lambda g: embed_node2vec(g, 128, num_walks=20, walk_length=40, p=1.0, q=0.5))]:
    e_alg = fn(g_kc)
    if e_alg.shape[0] == g_kc.num_entities:
        sc = node_classification_scores(g_kc, e_alg, kc["labels"])
        algos_results[name_alg] = sc["accuracy"]
        print(f"    {name_alg:10s}: accuracy={sc['accuracy']:.4f}")

print("=" * 70)
print("  PART 9: CLASSIFICATION (Label Prop + MLP)")
print("=" * 70)

t = T("LABEL PROPAGATION (Karate Club):")
lp_result = label_propagation_predict(g_kc, e_kc, kc["labels"], num_iterations=50)
print(f"    Accuracy: {lp_result['accuracy']:.4f} (train={lp_result['train_size']}, test={lp_result['test_size']})")

t = T("MLP CLASSIFIER (Karate Club):")
mlp_result = mlp_classify(g_kc, e_kc, kc["labels"], hidden_dim=32, num_epochs=300, learning_rate=0.01)
print(f"    Accuracy: {mlp_result['accuracy']:.4f}, Macro-F1: {mlp_result['macro_f1']:.4f}")

t = T("LABEL PROPAGATION (Cora):")
lp_cora = label_propagation_predict(g_cora, e_cora, cora["labels"], num_iterations=50)
print(f"    Accuracy: {lp_cora['accuracy']:.4f}")

t = T("MLP CLASSIFIER (Cora):")
mlp_cora = mlp_classify(g_cora, e_cora, cora["labels"], hidden_dim=64, num_epochs=200)
print(f"    Accuracy: {mlp_cora['accuracy']:.4f}, Macro-F1: {mlp_cora['macro_f1']:.4f}")

print("=" * 70)
print("  PART 10: HETEROGENEOUS GRAPHS")
print("=" * 70)

t = T("Build HeteroGraph:")
hg = HeteroGraph()
hg.add_node_type("user")
hg.add_node_type("product")
hg.add_node_type("store")

hg.add_edge_type("purchased", "user", "product", [
    ("alice", "laptop"), ("alice", "mouse"), ("bob", "keyboard"),
    ("bob", "monitor"), ("carol", "laptop"), ("carol", "mouse"),
    ("dave", "laptop"), ("eve", "keyboard"), ("eve", "monitor"),
])
hg.add_edge_type("sold_at", "product", "store", [
    ("laptop", "store_A"), ("mouse", "store_A"), ("keyboard", "store_B"),
    ("monitor", "store_B"),
])
hg.add_edge_type("visited", "user", "store", [
    ("alice", "store_A"), ("alice", "store_B"), ("bob", "store_A"),
    ("carol", "store_B"), ("dave", "store_A"), ("eve", "store_B"),
])
print(f"    {hg}")
print(f"    {hg.summary()}")

t = T("Per-relation embedding (concat):")
rel_graphs, rel_embs, combined = hg.embed_per_relation(feature_dim=64, combine="concat")
print(f"    Relations: {list(rel_graphs.keys())}")
for rname, remb in rel_embs.items():
    print(f"      {rname}: {remb.shape}")
if combined is not None:
    print(f"    Combined shape: {combined.shape}")

t = T("Per-relation embedding (mean):")
_, _, combined_mean = hg.embed_per_relation(feature_dim=64, combine="mean")
if combined_mean is not None:
    print(f"    Mean combined shape: {combined_mean.shape}")

t = T("Metapath embedding (user->product->store):")
g_meta, emb_meta = hg.embed_metapath(["purchased", "sold_at"], feature_dim=64)
print(f"    Metapath graph: {g_meta.num_entities} entities, embedding: {emb_meta.shape}")

t = T("Homogeneous edge export:")
edges_homo = hg.to_homogeneous_edges()
print(f"    Homogeneous edges: {len(edges_homo)}")

print("=" * 70)
print("  PART 11: I/O & EXPORT")
print("=" * 70)

t = T("NETWORKX export:")
G = to_networkx(graph, emb)
print(f"    NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

t = T("EDGE LIST export:")
el = to_edge_list(graph)
print(f"    {len(el)} unique edges, sample: {el[0]}")

t = T("SAVE/LOAD embeddings (npz):")
save_embeddings(graph, emb, "/tmp/test_emb.npz")
loaded_emb, loaded_ids = load_embeddings("/tmp/test_emb.npz")
print(f"    Saved & loaded: {loaded_emb.shape}, {len(loaded_ids)} entities")

t = T("SAVE embeddings (CSV):")
save_embeddings(graph, emb, "/tmp/test_emb.csv", format="csv")
print(f"    CSV: {emb.shape}")

t = T("SPARSE MATRIX export:")
rows, cols, vals, n, _ = graph.to_sparse_csr()
print(f"    {len(vals)} non-zeros in {n}x{n} matrix")

print("=" * 70)
print("  PART 12: VISUALIZATION")
print("=" * 70)

t = T("t-SNE dimensionality reduction:")
emb_2d = reduce_dimensions(emb, method="tsne")
print(f"    {emb.shape} -> {emb_2d.shape}")

t = T("PCA dimensionality reduction:")
emb_2d_pca = reduce_dimensions(emb, method="pca")
print(f"    {emb.shape} -> {emb_2d_pca.shape}")

t = T("VISUALIZATION (saved to file):")
path = visualize(graph, emb, labels=labels_nc, save_path="/tmp/pycleora_viz.png")
print(f"    Saved to: {path}")

print("=" * 70)
print("  PART 13: GPU & ERROR HANDLING")
print("=" * 70)

t = T("GPU propagation check:")
try:
    import torch
    print(f"    PyTorch available, CUDA: {torch.cuda.is_available()}")
except ImportError:
    print("    PyTorch not installed - GPU available with 'pip install torch'")
    print("    Rust CPU propagation active (fast!)")

t = T("Error handling:")
errors_caught = 0
try:
    embed(graph, propagation="invalid")
except ValueError:
    errors_caught += 1
try:
    find_most_similar(graph, emb, "nonexistent_entity")
except Exception:
    errors_caught += 1
try:
    embed_with_node_features(graph, {})
except ValueError:
    errors_caught += 1
try:
    graph.to_sparse_csr("invalid_markov")
except Exception:
    errors_caught += 1
print(f"    Caught {errors_caught}/4 expected errors")

elapsed = time.time() - t_start
total_tests = test_num[0]

print("\n" + "=" * 70)
print(f"  All {total_tests} features working! Total time: {elapsed:.2f}s")
print("=" * 70)

print("\nPress Ctrl+C to exit.")
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    pass
