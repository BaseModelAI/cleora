"""Embedding methods under test, behind one signature.

Concern of this module: given a graph, produce one vector per node with a named method.
Every method - Cleora and the baselines alike - is called as

    embed(edges, num_nodes, dim, seed, threads, weights) -> float32 [num_nodes, dim]

``edges`` is the canonical edge list of :mod:`cleora_bench.datasets` (undirected, unique,
``u < v``); row ``i`` of the result is the vector of node ``i``.

The baselines are *not* reimplemented here. Each one runs its authors' code or a widely
used library, with the hyper-parameters its authors recommend:

==============  =================================================================  ========
method          implementation                                                     env
==============  =================================================================  ========
deepwalk        PecanPy 2.0.9 walks + gensim skip-gram (negative sampling)          walks
node2vec        same, biased walks                                                  walks
prone           authors' proNE.py (github.com/THUDM/ProNE)                          main
netmf           authors' netmf.py (github.com/xptree/NetMF), Theano calls -> NumPy  main
fastrp          authors' fastrp.py (github.com/GTmac/FastRP)                        main
randne          authors' RandNE.py (github.com/ZW-ZHANG/RandNE)                     main
grarep          karateclub 1.3.3                                                    karate
hope            karateclub 1.3.3                                                    karate
walklets        karateclub 1.3.3                                                    karate
==============  =================================================================  ========

``env`` names the Python environment (see ``benchmarks/envs``) the method runs in; the
reference repositories are cloned at pinned commits by ``benchmarks/setup.sh`` into the
directory named by ``CLEORA_BENCH_BASELINES``.

The module only needs NumPy at import time and runs on Python >= 3.10, because it is
imported inside every one of those environments.
"""

import os
import sys

import numpy as np

# name -> (function, environment, label, largest graph in nodes (None = unlimited), note)
METHODS = {}

# What a method wants recorded about the run besides time and memory (the worker stores
# it with the statistics). One run per process, so a module-level dictionary is enough.
RUN_INFO = {}


def _method(name, env, label, max_nodes=None, note=""):
    def decorate(function):
        METHODS[name] = (function, env, label, max_nodes, note)
        return function
    return decorate


def embed(name, edges, num_nodes, dim, seed, threads, weights=None, **params):
    function = METHODS[name][0]
    vectors = np.ascontiguousarray(function(edges, num_nodes, dim, seed, threads, weights, **params), dtype=np.float32)
    if vectors.shape != (num_nodes, dim):
        raise RuntimeError(f"{name} returned shape {vectors.shape}, expected {(num_nodes, dim)}")
    return vectors


# ----------------------------------------------------------------------------- helpers


def adjacency(edges, num_nodes, weights=None):
    """Symmetric adjacency matrix (float64 CSR, sorted indices, no self-loops)."""
    from scipy.sparse import csr_matrix

    values = np.ones(len(edges)) if weights is None else np.asarray(weights, dtype=np.float64)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    matrix = csr_matrix((np.concatenate([values, values]), (rows, cols)), shape=(num_nodes, num_nodes))
    matrix.sort_indices()
    return matrix


def _reference(repository, *subdirectory):
    """Makes the pinned checkout of a reference implementation importable."""
    root = os.environ.get("CLEORA_BENCH_BASELINES")
    if not root:
        raise RuntimeError("Set CLEORA_BENCH_BASELINES to the directory benchmarks/setup.sh cloned the baselines into")
    path = os.path.join(root, repository, *subdirectory)
    if not os.path.isdir(path):
        raise RuntimeError(f"Reference implementation not found: {path} (run benchmarks/setup.sh)")
    if path not in sys.path:
        sys.path.insert(0, path)


def _pad(vectors, dim):
    """Methods that cannot produce ``dim`` columns on a small graph are zero-padded."""
    if vectors.shape[1] >= dim:
        return vectors[:, :dim]
    return np.concatenate([vectors, np.zeros((vectors.shape[0], dim - vectors.shape[1]), vectors.dtype)], axis=1)


# ----------------------------------------------------------------------------- Cleora


def _cleora_graph(edges, threads, weights, columns):
    from pycleora import SparseMatrix

    graph = SparseMatrix.from_edges(edges[:, 0], edges[:, 1], columns=columns, weights=weights, num_workers=threads)
    return graph, np.array(graph.entity_ids, dtype=np.int64)


def _node_order(vectors, row_node, num_nodes):
    out = np.zeros((num_nodes, vectors.shape[1]), dtype=np.float32)
    out[row_node] = vectors
    return out


@_method("cleora_classic", "main", "Cleora", note="iterations: a number, or 'auto' = SparseMatrix.suggest_iterations(purpose)")
def cleora_classic(edges, num_nodes, dim, seed, threads, weights, iterations="auto", propagation="left",
                   columns="complex::reflexive::node", purpose="general"):
    import pycleora

    graph, row_node = _cleora_graph(edges, threads, weights, columns)
    if iterations == "auto":
        iterations = graph.suggest_iterations(propagation, purpose)
    RUN_INFO["iterations"] = iterations
    vectors = pycleora.embed(graph, dim, iterations, propagation=propagation, seed=seed, num_workers=threads, whiten=False)
    return _node_order(vectors, row_node, num_nodes)


@_method("cleora_whiten40", "main", "Cleora 3.2 / PR #81 default", note="40 iterations, PCA whitening after every one")
def cleora_whiten40(edges, num_nodes, dim, seed, threads, weights):
    import warnings

    import pycleora

    graph, row_node = _cleora_graph(edges, threads, weights, "complex::reflexive::node")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vectors = pycleora.embed(graph, dim, 40, seed=seed, num_workers=threads, whiten=True)
    return _node_order(vectors, row_node, num_nodes)


# ----------------------------------------------------------------------------- random walks


def _skipgram_walks(edges, num_nodes, dim, seed, threads, weights, p, q, epochs):
    from pecanpy import pecanpy

    matrix = adjacency(edges, num_nodes, weights)
    # PecanPy's own choice of walk engine: plain neighbour sampling for first-order walks
    # (alias tables when edges are weighted), on-the-fly second-order probabilities for
    # node2vec. All of them sample the same walks; the engines differ in speed only.
    if p == 1.0 and q == 1.0:
        engine = pecanpy.FirstOrderUnweighted if weights is None else pecanpy.PreCompFirstOrder
    else:
        engine = pecanpy.SparseOTF
    graph = engine(p=p, q=q, workers=threads, verbose=False, random_state=seed)
    graph.set_node_ids([str(node) for node in range(num_nodes)])
    graph.indptr = matrix.indptr.astype(np.uint64)
    graph.indices = matrix.indices.astype(np.uint32)
    graph.data = matrix.data.astype(np.float32)
    # node2vec's settings: 10 walks of length 80 per node, window 10 (Grover & Leskovec, 2016)
    return graph.embed(dim=dim, num_walks=10, walk_length=80, window_size=10, epochs=epochs)


@_method("deepwalk", "walks", "DeepWalk", note="PecanPy first-order walks + gensim skip-gram")
def deepwalk(edges, num_nodes, dim, seed, threads, weights, epochs=1):
    return _skipgram_walks(edges, num_nodes, dim, seed, threads, weights, 1.0, 1.0, epochs)


@_method("node2vec", "walks", "node2vec", note="PecanPy biased walks + gensim skip-gram")
def node2vec(edges, num_nodes, dim, seed, threads, weights, p=1.0, q=0.5, epochs=1):
    return _skipgram_walks(edges, num_nodes, dim, seed, threads, weights, p, q, epochs)


# ----------------------------------------------------------------------------- matrix methods


@_method("prone", "main", "ProNE", note="authors' code; mu = 0.2, theta = 0.5, 10 Chebyshev steps")
def prone(edges, num_nodes, dim, seed, threads, weights):
    _reference("ProNE")
    import proNE

    matrix = adjacency(edges, num_nodes, weights)
    # The reference class reads an edge-list file with networkx in its constructor; the
    # two algorithm steps only need the adjacency matrix, the node count and the dimension.
    model = proNE.ProNE.__new__(proNE.ProNE)
    model.node_number, model.dimension = num_nodes, min(dim, num_nodes - 1)
    np.random.seed(seed)  # its randomized SVD draws from the global NumPy generator
    features = model.pre_factorization(matrix, matrix)
    return _pad(model.chebyshev_gaussian(matrix, features, 10, 0.2, 0.5), dim)


NETMF_DENSE_FALLBACK_NODES = 20_000


def _leading_eigenpairs_dense(matrix, rank):
    """What NetMF's ``approximate_normalized_graph_laplacian`` returns - the ``rank``
    largest eigenvalues of D^-1/2 A D^-1/2 and D^-1/2 times their eigenvectors - computed
    with a dense symmetric eigensolver."""
    from scipy.sparse import csgraph, diags, identity

    laplacian, d_rt = csgraph.laplacian(matrix, normed=True, return_diag=True)
    values, vectors = np.linalg.eigh((identity(matrix.shape[0]) - laplacian).toarray())
    keep = np.argsort(values)[-rank:]
    return values[keep], diags(d_rt ** -1).dot(vectors[:, keep])


@_method("netmf", "main", "NetMF", max_nodes=60_000,
         note="authors' algorithm (large-window variant, T = 10, b = 1); needs a dense n x n matrix")
def netmf(edges, num_nodes, dim, seed, threads, weights, window=10, negative=1.0):
    import types

    # netmf.py imports Theano (unmaintained since 2020) for two element-wise expressions.
    # A placeholder satisfies the import; the two functions that use it are restated
    # below with NumPy, line by line.
    if "theano" not in sys.modules:
        placeholder = types.ModuleType("theano")
        placeholder.config = types.SimpleNamespace(exception_verbosity="high", floatX="float32")
        placeholder.tensor = types.ModuleType("theano.tensor")
        sys.modules["theano"], sys.modules["theano.tensor"] = placeholder, placeholder.tensor
    _reference("NetMF")
    import netmf as reference
    import scipy.sparse.linalg  # noqa: F401  (the reference calls sparse.linalg without importing it)
    from scipy import sparse

    matrix = adjacency(edges, num_nodes, weights)
    rank = min(max(256, dim), num_nodes - 2)
    volume = float(matrix.sum())
    try:
        eigenvalues, d_rt_inv_u = reference.approximate_normalized_graph_laplacian(matrix, rank=rank, which="LA")
    except sparse.linalg.ArpackError:
        # ARPACK gives up when the leading eigenvalue is repeated very often (a graph in
        # many pieces). The same eigenpairs from a dense decomposition, where that is affordable:
        if num_nodes > NETMF_DENSE_FALLBACK_NODES:
            raise
        eigenvalues, d_rt_inv_u = _leading_eigenpairs_dense(matrix, rank)
    # approximate_deepwalk_matrix, with T.log(T.maximum(m m^T * vol / b, 1)) in NumPy:
    eigenvalues = reference.deepwalk_filter(eigenvalues, window=window)
    x = sparse.diags(np.sqrt(eigenvalues)).dot(d_rt_inv_u.T).T.astype(np.float32)
    deepwalk_matrix = x @ x.T
    deepwalk_matrix *= volume / negative
    np.maximum(deepwalk_matrix, 1, out=deepwalk_matrix)
    np.log(deepwalk_matrix, out=deepwalk_matrix)
    np.random.seed(seed)
    # svd_deepwalk_matrix: U * sqrt(S) of the leading singular triplets
    u, s, _ = sparse.linalg.svds(deepwalk_matrix, min(dim, num_nodes - 2), return_singular_vectors="u")
    return _pad(np.asarray(sparse.diags(np.sqrt(s)).dot(u.T).T), dim)


@_method("fastrp", "main", "FastRP",
         note="authors' code and published configuration: weights (0, 0, 1, 6.67), beta = -0.64")
def fastrp(edges, num_nodes, dim, seed, threads, weights, order_weights=(0.0, 0.0, 1.0, 6.67), beta=-0.64):
    _reference("FastRP")
    import fastrp as reference
    from scipy.sparse import spdiags

    matrix = adjacency(edges, num_nodes, weights)
    # The reference builds the (degree-scaled, very sparse) projection and the first
    # product, and merges the powers. Its loop over the higher powers multiplies sparse
    # matrices that fill in completely; here those products are dense, which computes
    # the same numbers. The reference fixes the seed of the projection (random_state=42).
    first = reference.fastrp_projection(matrix, q=1, dim=dim, projection_method="sparse",
                                        input_matrix="trans", alpha=beta)[0]
    current = np.asarray(first.todense()) if hasattr(first, "todense") else np.asarray(first)
    degrees = np.asarray(matrix.sum(axis=1)).ravel()
    transition = spdiags(1.0 / degrees, 0, num_nodes, num_nodes) @ matrix
    powers = [current]
    for _ in range(len(order_weights) - 1):
        current = transition @ current
        powers.append(current)
    return np.asarray(reference.fastrp_merge(powers, list(order_weights), True))


@_method("randne", "main", "RandNE",
         note="authors' code and classification configuration: transition matrix, q = 3, weights (1, 1e2, 1e4, 1e5)")
def randne(edges, num_nodes, dim, seed, threads, weights):
    _reference("RandNE", "Python")
    import RandNE as reference
    from sklearn.preprocessing import normalize

    transition = normalize(adjacency(edges, num_nodes, weights), norm="l1", axis=1)
    parts = reference.Projection(transition, 3, dim, False, seed)
    return normalize(reference.Combine(parts, [1, 1e2, 1e4, 1e5]), norm="l2", axis=1)


# ----------------------------------------------------------------------------- karateclub


def _karate(model, edges, num_nodes, weights):
    import networkx

    graph = networkx.Graph()
    graph.add_nodes_from(range(num_nodes))
    if weights is None:
        graph.add_edges_from(edges.tolist())
    else:
        graph.add_weighted_edges_from([(int(u), int(v), float(w)) for (u, v), w in zip(edges, weights)])
    model.fit(graph)
    return model.get_embedding()


@_method("grarep", "karate", "GraRep", max_nodes=40_000, note="karateclub; 4 orders x dim/4; dense n x n")
def grarep(edges, num_nodes, dim, seed, threads, weights):
    from karateclub import GraRep

    order = 4
    return _pad(_karate(GraRep(dimensions=min(dim // order, num_nodes - 1), order=order, seed=seed), edges, num_nodes, weights), dim)


@_method("hope", "karate", "HOPE", max_nodes=40_000, note="karateclub (common-neighbours proximity)")
def hope(edges, num_nodes, dim, seed, threads, weights):
    from karateclub import HOPE

    return _pad(_karate(HOPE(dimensions=min(dim, num_nodes - 1), seed=seed), edges, num_nodes, weights), dim)


@_method("walklets", "karate", "Walklets", max_nodes=40_000, note="karateclub; 4 scales x dim/4")
def walklets(edges, num_nodes, dim, seed, threads, weights):
    from karateclub import Walklets

    scales = 4
    return _karate(Walklets(dimensions=dim // scales, window_size=scales, workers=threads, seed=seed), edges, num_nodes, weights)
