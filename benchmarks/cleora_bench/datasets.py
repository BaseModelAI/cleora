"""Benchmark datasets: fetched from their original public sources, stored in one canonical form.

Concern of this module: turning a dataset *name* into a canonical graph on disk.
Nothing here knows about embedding methods, probes or metrics.

Canonical form (``<root>/<name>/graph.npz`` + ``meta.json``)
------------------------------------------------------------
``edges``        int64 [E, 2]   undirected, ``u < v``, unique, no self-loops. Node ids are
                                ``0..N-1`` and every node has at least one edge.
``weights``      float32 [E]    only for graphs whose source carries edge weights.
``labels``       int64 [N]      single-label graphs, ``-1`` = unlabelled; or
                 bool  [N, C]   multi-label graphs. Absent when the dataset has no node labels:
                                such graphs support the link/similarity tasks only.
``split_train`` / ``split_valid`` / ``split_test``
                 int64 [*]      official node split, when the dataset defines one.
``lp_*``                        official link-prediction split, when the dataset defines one
                                (OGB link datasets): ``lp_valid_pos``, ``lp_valid_neg``,
                                ``lp_test_pos``, ``lp_test_neg`` (int64 [*, 2]) or, for
                                ogbl-citation2, ``lp_{valid,test}_{source,target,target_neg}``.
                                ``edges`` then holds the official *training* edges only.
``num_left``     int64 scalar   bipartite graphs: nodes ``0..num_left-1`` are the left side
                                (users), the others the right side (items).

Nodes without any edge cannot be embedded from structure, so they are removed (and the ids
compacted) for every method alike; ``meta.json`` records how many were dropped.

The loaders use PyTorch Geometric and OGB where those libraries ship the dataset, and the
original files otherwise. Every source URL is recorded in ``meta.json``.
"""

import contextlib
import gzip
import hashlib
import io
import json
import os
import urllib.request
import zipfile

import numpy as np

# name -> (loader function, keyword arguments, tier, one-line description)
# tier: "S" up to about a million edges (three seeds), "M" up to about 60 million and
# "L" millions of nodes (one seed each). Methods that need dense n x n matrices are
# limited by the number of nodes (see methods.METHODS), whatever the tier.
REGISTRY = {}


def _register(name, tier, description, **kwargs):
    def decorate(loader):
        REGISTRY[name] = (loader, kwargs, tier, description)
        return loader
    return decorate


# Size of every canonical graph (nodes with an edge, undirected edges). A source that
# starts serving different data is noticed when the dataset is prepared.
EXPECTED_SIZE = {
    "cora": (2708, 5278), "citeseer": (3279, 4552), "pubmed": (19717, 44324), "cora_full": (19793, 63421),
    "dblp": (17716, 52867), "amazon_photo": (7535, 119081), "amazon_computers": (13471, 245861),
    "coauthor_cs": (18333, 81894), "coauthor_physics": (34493, 247962), "ppi": (3860, 37845),
    "wikipedia_pos": (4777, 92295), "blogcatalog": (10312, 333983), "facebook": (4039, 88234),
    "flickr": (89250, 449878), "yelp": (716847, 6618986), "reddit": (232965, 57307946),
    "ogbn_arxiv": (169343, 1157799), "ogbn_products": (2400608, 61859012), "ogbl_collab": (235868, 967632),
    "ogbl_ddi": (4267, 1067911), "ogbl_ppa": (576289, 21231931), "ogbl_citation2": (2926620, 30351880),
    "roadnet_ca": (1965206, 2766607), "livejournal": (3997962, 34681189), "movielens_1m": (9746, 1000209),
    "movielens_20m": (165237, 20000263),
}


# ----------------------------------------------------------------------------- helpers


def _download(url, path):
    """Downloads ``url`` to ``path`` (atomically) unless it is already there."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "cleora-bench"})
    with urllib.request.urlopen(request, timeout=600) as response, open(path + ".part", "wb") as out:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
    os.replace(path + ".part", path)
    return path


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@contextlib.contextmanager
def _legacy_torch_load():
    """OGB 1.3.6 caches its processed files with pickle and reads them back with
    ``torch.load``; PyTorch >= 2.6 refuses that unless ``weights_only=False``. The files are
    written by OGB itself from the downloaded csv files, a few lines earlier."""
    import torch

    original = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = load
    try:
        yield
    finally:
        torch.load = original


def canonical_edges(src, dst, weights=None):
    """Undirected, unique, self-loop-free edge list with ``u < v``, sorted.

    Duplicate edges (including the two directions of one edge) are merged; with weights
    their *maximum* is kept, so a symmetric weighted matrix keeps its weights unchanged.
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    keep = src != dst
    low = np.minimum(src[keep], dst[keep])
    high = np.maximum(src[keep], dst[keep])
    order = np.lexsort((high, low))
    low, high = low[order], high[order]
    first = np.ones(len(low), dtype=bool)
    first[1:] = (low[1:] != low[:-1]) | (high[1:] != high[:-1])
    edges = np.stack([low[first], high[first]], axis=1)
    if weights is None:
        return edges, None
    weights = np.asarray(weights, dtype=np.float64)[keep][order]
    group = np.cumsum(first) - 1
    merged = np.full(len(edges), -np.inf)
    np.maximum.at(merged, group, weights)
    return edges, merged.astype(np.float32)


def _compact(graph):
    """Drops nodes without edges and renumbers the rest ``0..N-1`` (order preserved)."""
    edges = graph["edges"]
    num_original = int(graph.pop("num_nodes"))
    present = np.zeros(num_original, dtype=bool)
    present[edges.ravel()] = True
    new_id = np.cumsum(present) - 1
    new_id[~present] = -1
    graph["edges"] = new_id[edges]
    if "labels" in graph:
        graph["labels"] = graph["labels"][present]
    for key in ("split_train", "split_valid", "split_test"):
        if key in graph:
            mapped = new_id[graph[key]]
            graph[key] = mapped[mapped >= 0]
    # Official link splits. A node without a training edge has no embedding to rank with:
    # positive and negative pairs that touch one are dropped (the counts go to meta.json).
    dropped = {}
    for part in ("valid", "test"):
        if f"lp_{part}_pos" not in graph:
            continue
        positives = new_id[graph[f"lp_{part}_pos"]]
        keep = (positives >= 0).all(axis=1)
        graph[f"lp_{part}_pos"] = positives[keep]
        dropped[f"{part}_pos"] = int((~keep).sum())
        if f"lp_{part}_neg" in graph:  # one shared list of negative pairs
            negatives = new_id[graph[f"lp_{part}_neg"]]
            keep_negative = (negatives >= 0).all(axis=1)
            graph[f"lp_{part}_neg"] = negatives[keep_negative]
            dropped[f"{part}_neg"] = int((~keep_negative).sum())
        if f"lp_{part}_target_neg" in graph:  # negatives per positive, row-aligned; -1 = absent
            graph[f"lp_{part}_target_neg"] = new_id[graph[f"lp_{part}_target_neg"]][keep]
    if "num_left" in graph:
        graph["num_left"] = np.int64(present[: int(graph["num_left"])].sum())
    return graph, num_original, int(present.sum()), dropped


def _from_pyg(data):
    edge_index = data.edge_index.numpy()
    return edge_index[0], edge_index[1], int(data.num_nodes)


def _mask_split(data):
    return {
        "split_train": np.flatnonzero(data.train_mask.numpy()),
        "split_valid": np.flatnonzero(data.val_mask.numpy()),
        "split_test": np.flatnonzero(data.test_mask.numpy()),
    }


# ----------------------------------------------------------------------------- loaders


@_register("cora", "S", "Planetoid citation graph, 7 classes", pyg="Cora")
@_register("citeseer", "S", "Planetoid citation graph, 6 classes", pyg="CiteSeer")
@_register("pubmed", "S", "Planetoid citation graph, 3 classes", pyg="PubMed")
def _planetoid(raw, pyg):
    from torch_geometric.datasets import Planetoid

    data = Planetoid(os.path.join(raw, "planetoid"), pyg)[0]
    src, dst, n = _from_pyg(data)
    # The fixed 20-labels-per-class Planetoid split is made for feature-based GNNs; for
    # structure-only embeddings the benchmark draws its own stratified splits.
    return {"src": src, "dst": dst, "num_nodes": n, "labels": data.y.numpy().astype(np.int64),
            "source": f"torch_geometric.datasets.Planetoid('{pyg}')",
            "citation": "Yang, Cohen, Salakhutdinov, ICML 2016; Sen et al., AI Magazine 2008"}


@_register("cora_full", "S", "Full Cora citation graph, 70 classes", pyg="Cora")
@_register("dblp", "S", "DBLP citation graph, 4 classes", pyg="DBLP")
def _citation_full(raw, pyg):
    from torch_geometric.datasets import CitationFull

    data = CitationFull(os.path.join(raw, "citation_full"), pyg)[0]
    src, dst, n = _from_pyg(data)
    return {"src": src, "dst": dst, "num_nodes": n, "labels": data.y.numpy().astype(np.int64),
            "source": f"torch_geometric.datasets.CitationFull('{pyg}')",
            "citation": "Bojchevski and Günnemann, ICLR 2018"}


@_register("amazon_photo", "S", "Amazon co-purchase graph, 8 classes", pyg="Photo")
@_register("amazon_computers", "S", "Amazon co-purchase graph, 10 classes", pyg="Computers")
def _amazon(raw, pyg):
    from torch_geometric.datasets import Amazon

    data = Amazon(os.path.join(raw, "amazon"), pyg)[0]
    src, dst, n = _from_pyg(data)
    return {"src": src, "dst": dst, "num_nodes": n, "labels": data.y.numpy().astype(np.int64),
            "source": f"torch_geometric.datasets.Amazon('{pyg}')",
            "citation": "Shchur et al., Pitfalls of GNN Evaluation, 2018; McAuley et al., SIGIR 2015"}


@_register("coauthor_cs", "S", "Co-authorship graph (MAG), 15 classes", pyg="CS")
@_register("coauthor_physics", "S", "Co-authorship graph (MAG), 5 classes", pyg="Physics")
def _coauthor(raw, pyg):
    from torch_geometric.datasets import Coauthor

    data = Coauthor(os.path.join(raw, "coauthor"), pyg)[0]
    src, dst, n = _from_pyg(data)
    return {"src": src, "dst": dst, "num_nodes": n, "labels": data.y.numpy().astype(np.int64),
            "source": f"torch_geometric.datasets.Coauthor('{pyg}')",
            "citation": "Shchur et al., Pitfalls of GNN Evaluation, 2018"}


@_register("ppi", "S", "Protein interactions (node2vec version), 50 labels, multi-label",
           url="https://snap.stanford.edu/node2vec/Homo_sapiens.mat",
           citation="Grover and Leskovec, node2vec, KDD 2016")
@_register("wikipedia_pos", "S", "Word co-occurrence (node2vec version), 40 labels, multi-label, weighted",
           url="https://snap.stanford.edu/node2vec/POS.mat",
           citation="Grover and Leskovec, node2vec, KDD 2016")
@_register("blogcatalog", "S", "BlogCatalog social graph (DeepWalk version), 39 labels, multi-label",
           url="https://github.com/phanein/deepwalk/raw/master/example_graphs/blogcatalog.mat",
           citation="Tang and Liu, KDD 2009; Perozzi et al., DeepWalk, KDD 2014")
def _matlab_network(raw, url, citation):
    import scipy.io
    import scipy.sparse

    path = _download(url, os.path.join(raw, "mat", os.path.basename(url)))
    content = scipy.io.loadmat(path)
    network = scipy.sparse.coo_matrix(content["network"])
    labels = np.asarray(scipy.sparse.csr_matrix(content["group"]).todense()) > 0
    graph = {"src": network.row, "dst": network.col, "num_nodes": network.shape[0], "labels": labels,
             "source": url, "citation": citation, "sha256": _sha256(path)}
    if not np.all(network.data == network.data[0]):
        graph["weights"] = network.data
    return graph


@_register("flickr", "S", "Flickr image graph (GraphSAINT), 7 classes, official split")
def _flickr(raw):
    from torch_geometric.datasets import Flickr

    data = Flickr(os.path.join(raw, "flickr"))[0]
    src, dst, n = _from_pyg(data)
    return {"src": src, "dst": dst, "num_nodes": n, "labels": data.y.numpy().astype(np.int64),
            **_mask_split(data), "source": "torch_geometric.datasets.Flickr",
            "citation": "Zeng et al., GraphSAINT, ICLR 2020"}


@_register("yelp", "M", "Yelp friendship graph (GraphSAINT), 100 labels, multi-label, official split")
def _yelp(raw):
    from torch_geometric.datasets import Yelp

    data = Yelp(os.path.join(raw, "yelp"))[0]
    src, dst, n = _from_pyg(data)
    return {"src": src, "dst": dst, "num_nodes": n, "labels": data.y.numpy() > 0,
            **_mask_split(data), "source": "torch_geometric.datasets.Yelp",
            "citation": "Zeng et al., GraphSAINT, ICLR 2020"}


@_register("reddit", "M", "Reddit post graph (GraphSAGE), 41 classes, official split")
def _reddit(raw):
    from torch_geometric.datasets import Reddit

    # PyTorch Geometric fetches https://data.dgl.ai/dataset/reddit.zip; the same archive
    # is served by the bucket behind that name, which is reachable where the alias is not.
    root = os.path.join(raw, "reddit")
    raw_files = [os.path.join(root, "raw", name) for name in ("reddit_data.npz", "reddit_graph.npz")]
    if not all(os.path.exists(path) for path in raw_files):
        archive = _download("https://dgl-data.s3-accelerate.amazonaws.com/dataset/reddit.zip",
                            os.path.join(root, "raw", "reddit.zip"))
        with zipfile.ZipFile(archive) as content:
            content.extractall(os.path.join(root, "raw"))
    data = Reddit(root)[0]
    src, dst, n = _from_pyg(data)
    return {"src": src, "dst": dst, "num_nodes": n, "labels": data.y.numpy().astype(np.int64),
            **_mask_split(data), "source": "torch_geometric.datasets.Reddit",
            "citation": "Hamilton, Ying, Leskovec, GraphSAGE, NeurIPS 2017"}


@_register("ogbn_arxiv", "S", "OGB arXiv citation graph, 40 classes, official (time) split", ogb="ogbn-arxiv")
@_register("ogbn_products", "L", "OGB Amazon co-purchase graph, 47 classes, official split", ogb="ogbn-products")
def _ogb_node(raw, ogb):
    from ogb.nodeproppred import NodePropPredDataset

    with _legacy_torch_load():
        dataset = NodePropPredDataset(ogb, root=os.path.join(raw, "ogb"))
        split = dataset.get_idx_split()
    graph, labels = dataset[0]
    return {"src": graph["edge_index"][0], "dst": graph["edge_index"][1], "num_nodes": int(graph["num_nodes"]),
            "labels": labels.reshape(-1).astype(np.int64),
            "split_train": split["train"], "split_valid": split["valid"], "split_test": split["test"],
            "source": f"ogb.nodeproppred.NodePropPredDataset('{ogb}')",
            "citation": "Hu et al., Open Graph Benchmark, NeurIPS 2020"}


@_register("ogbl_collab", "S", "OGB author collaboration graph, official (time) link split, Hits@50", ogb="ogbl-collab")
@_register("ogbl_ddi", "S", "OGB drug-drug interaction graph, official link split, Hits@20", ogb="ogbl-ddi")
@_register("ogbl_ppa", "M", "OGB protein association graph, official link split, Hits@100", ogb="ogbl-ppa")
@_register("ogbl_citation2", "L", "OGB citation graph, official link split, MRR", ogb="ogbl-citation2")
def _ogb_link(raw, ogb):
    from ogb.linkproppred import LinkPropPredDataset

    with _legacy_torch_load():
        dataset = LinkPropPredDataset(ogb, root=os.path.join(raw, "ogb"))
        split = dataset.get_edge_split()
    num_nodes = int(dataset[0]["num_nodes"])
    result = {"num_nodes": num_nodes, "source": f"ogb.linkproppred.LinkPropPredDataset('{ogb}')",
              "citation": "Hu et al., Open Graph Benchmark, NeurIPS 2020"}
    if "edge" in split["train"]:
        train = np.asarray(split["train"]["edge"])
        result.update(src=train[:, 0], dst=train[:, 1])
        for part in ("valid", "test"):
            result[f"lp_{part}_pos"] = np.asarray(split[part]["edge"], dtype=np.int64)
            result[f"lp_{part}_neg"] = np.asarray(split[part]["edge_neg"], dtype=np.int64)
    else:  # ogbl-citation2: one positive target and 1000 negative targets per source node
        result.update(src=np.asarray(split["train"]["source_node"]), dst=np.asarray(split["train"]["target_node"]))
        for part in ("valid", "test"):
            source = np.asarray(split[part]["source_node"], dtype=np.int64)
            target = np.asarray(split[part]["target_node"], dtype=np.int64)
            negatives = np.asarray(split[part]["target_node_neg"], dtype=np.int64)
            result[f"lp_{part}_pos"] = np.stack([source, target], axis=1)
            result[f"lp_{part}_target_neg"] = negatives
    return result


@_register("facebook", "S", "Facebook ego networks (SNAP), no labels",
           url="https://snap.stanford.edu/data/facebook_combined.txt.gz", separator=" ",
           citation="McAuley and Leskovec, NeurIPS 2012")
@_register("roadnet_ca", "L", "California road network (SNAP), no labels",
           url="https://snap.stanford.edu/data/roadNet-CA.txt.gz", separator="\t",
           citation="Leskovec et al., Internet Mathematics 2009")
@_register("livejournal", "L", "LiveJournal friendship graph (SNAP com-lj), no labels used",
           url="https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz", separator="\t",
           citation="Yang and Leskovec, ICDM 2012")
def _snap_edge_list(raw, url, separator, citation):
    import pandas

    path = _download(url, os.path.join(raw, "snap", os.path.basename(url)))
    pairs = pandas.read_csv(path, sep=separator, comment="#", header=None, dtype=np.int64).to_numpy()
    # SNAP node ids are arbitrary integers: renumber them densely, in sorted order
    ids, inverse = np.unique(pairs, return_inverse=True)
    inverse = inverse.reshape(pairs.shape)
    return {"src": inverse[:, 0], "dst": inverse[:, 1], "num_nodes": len(ids),
            "source": url, "citation": citation, "sha256": _sha256(path)}


@_register("movielens_1m", "S", "MovieLens-1M user-movie ratings as a bipartite graph, no labels",
           url="https://files.grouplens.org/datasets/movielens/ml-1m.zip", member="ml-1m/ratings.dat", separator="::")
@_register("movielens_20m", "L", "MovieLens-20M user-movie ratings as a bipartite graph, no labels",
           url="https://files.grouplens.org/datasets/movielens/ml-20m.zip", member="ml-20m/ratings.csv", separator=",")
def _movielens(raw, url, member, separator):
    import pandas

    path = _download(url, os.path.join(raw, "movielens", os.path.basename(url)))
    with zipfile.ZipFile(path) as archive:
        content = io.BytesIO(archive.read(member))
    header = 0 if member.endswith(".csv") else None
    table = pandas.read_csv(content, sep=separator, header=header, engine="python" if len(separator) > 1 else "c",
                            usecols=[0, 1]).to_numpy(dtype=np.int64)
    # every rating is one user-movie interaction (implicit feedback); users first, then movies
    users, user_index = np.unique(table[:, 0], return_inverse=True)
    movies, movie_index = np.unique(table[:, 1], return_inverse=True)
    return {"src": user_index, "dst": movie_index + len(users), "num_nodes": len(users) + len(movies),
            "num_left": np.int64(len(users)), "source": url, "sha256": _sha256(path),
            "citation": "Harper and Konstan, The MovieLens Datasets, ACM TiiS 2015"}


# ----------------------------------------------------------------------------- public API


def prepare(name, root):
    """Builds the canonical files of dataset ``name`` under ``root`` (no-op if present)."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Known: {', '.join(sorted(REGISTRY))}")
    directory = os.path.join(root, name)
    graph_path = os.path.join(directory, "graph.npz")
    if os.path.exists(graph_path):
        return directory
    loader, kwargs, tier, description = REGISTRY[name]
    loaded = loader(os.path.join(root, "_raw"), **kwargs)

    meta = {"name": name, "tier": tier, "description": description}
    for key in ("source", "citation", "sha256"):
        if key in loaded:
            meta[key] = loaded.pop(key)
    src, dst = loaded.pop("src"), loaded.pop("dst")
    edges, weights = canonical_edges(src, dst, loaded.pop("weights", None))
    graph = dict(loaded, edges=edges)
    graph, num_original, num_nodes, dropped_pairs = _compact(graph)
    if weights is not None:
        graph["weights"] = weights

    meta.update(num_nodes=num_nodes, num_edges=int(len(graph["edges"])),
                nodes_without_edges_dropped=num_original - num_nodes,
                official_link_pairs_dropped=dropped_pairs,
                weighted=weights is not None, bipartite="num_left" in graph,
                official_node_split="split_train" in graph,
                official_link_split=any(k.startswith("lp_") for k in graph))
    size = (meta["num_nodes"], meta["num_edges"])
    if size != EXPECTED_SIZE[name]:
        raise RuntimeError(f"{name}: the source now yields {size[0]} nodes and {size[1]} edges, the benchmark was "
                           f"run on {EXPECTED_SIZE[name][0]} and {EXPECTED_SIZE[name][1]}. Results would not be comparable.")
    if "labels" in graph:
        labels = graph["labels"]
        meta["multilabel"] = labels.ndim == 2
        meta["num_classes"] = int(labels.shape[1]) if labels.ndim == 2 else int(labels.max()) + 1
        meta["num_labelled"] = int(labels.any(axis=1).sum()) if labels.ndim == 2 else int((labels >= 0).sum())
    else:
        meta["num_classes"] = 0  # no node labels: link / similarity tasks only

    os.makedirs(directory, exist_ok=True)
    np.savez(graph_path + ".tmp.npz", **graph)
    os.replace(graph_path + ".tmp.npz", graph_path)
    with open(os.path.join(directory, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=1)
    return directory


def load(name, root):
    """Canonical arrays of a dataset (prepared on first use) plus its ``meta`` dictionary."""
    directory = prepare(name, root)
    with np.load(os.path.join(directory, "graph.npz")) as archive:
        graph = {key: archive[key] for key in archive.files}
    with open(os.path.join(directory, "meta.json")) as handle:
        graph["meta"] = json.load(handle)
    return graph


def meta(name, root):
    """The ``meta`` dictionary of a prepared dataset, without loading its arrays."""
    with open(os.path.join(root, name, "meta.json")) as handle:
        return json.load(handle)


def names(tiers="SML"):
    return [name for name, (_, _, tier, _) in REGISTRY.items() if tier in tiers]
