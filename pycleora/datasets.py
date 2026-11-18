"""Benchmark graphs.

Every loader in this module returns real, published data, downloaded on first use
and cached under ``~/.pycleora_datasets`` (override with the ``PYCLEORA_DATA``
environment variable). The returned dictionary always carries
``"synthetic": False`` and the ``"source"`` it was fetched from.

Generated graphs live in :func:`load_synthetic` (and :mod:`pycleora.generators`);
they are always marked ``"synthetic": True`` and never carry the name of a real
dataset.
"""

import os
import sys
import gzip
import functools
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections.abc import Sequence


_CACHE_DIR = os.environ.get("PYCLEORA_DATA") or os.path.join(os.path.expanduser("~"), ".pycleora_datasets")


def _real(source: str, citation: Optional[str] = None):
    """Marks a loader as returning real, published data."""
    def decorate(loader):
        @functools.wraps(loader)
        def wrapper(*args, **kwargs):
            dataset = loader(*args, **kwargs)
            dataset["synthetic"] = False
            dataset["source"] = source
            if citation:
                dataset["citation"] = citation
            return dataset
        return wrapper
    return decorate


class _LazyEdgeList(Sequence):
    __slots__ = ("_src", "_dst", "_len")

    def __init__(self, src: np.ndarray, dst: np.ndarray):
        self._src = src
        self._dst = dst
        self._len = len(src)

    @property
    def src(self) -> np.ndarray:
        """Source node of every edge (integer ids; the entity name is ``str(id)``)."""
        return self._src

    @property
    def dst(self) -> np.ndarray:
        """Target node of every edge."""
        return self._dst

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [f"{self._src[i]} {self._dst[i]}" for i in range(*idx.indices(self._len))]
        if idx < 0:
            idx += self._len
        if idx < 0 or idx >= self._len:
            raise IndexError(f"index {idx} out of range")
        return f"{self._src[idx]} {self._dst[idx]}"

    def __iter__(self):
        src = self._src
        dst = self._dst
        for i in range(self._len):
            yield f"{src[i]} {dst[i]}"

    def __repr__(self):
        return f"_LazyEdgeList(len={self._len:,})"


def _ensure_cache_dir():
    os.makedirs(_CACHE_DIR, exist_ok=True)


def _download_file(url: str, filepath: str):
    import urllib.request
    import ssl
    ctx = ssl.create_default_context()
    urllib.request.urlretrieve(url, filepath, context=ctx)


def _download_with_progress(url: str, filepath: str, description: str = "Downloading"):
    import urllib.request
    import ssl

    ctx = ssl.create_default_context()
    # some hosts refuse the default urllib user agent
    req = urllib.request.Request(url, headers={"User-Agent": "pycleora-datasets"})
    response = urllib.request.urlopen(req, context=ctx)
    total_size = response.headers.get("Content-Length")
    total_size = int(total_size) if total_size else None

    block_size = 1024 * 1024
    downloaded = 0

    with open(filepath, "wb") as f:
        while True:
            chunk = response.read(block_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = downloaded / total_size * 100
                mb_done = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stderr.write(f"\r{description}: {mb_done:.1f}/{mb_total:.1f} MB ({pct:.1f}%)")
            else:
                mb_done = downloaded / (1024 * 1024)
                sys.stderr.write(f"\r{description}: {mb_done:.1f} MB")
            sys.stderr.flush()
    sys.stderr.write("\n")
    sys.stderr.flush()


def _load_snap_edge_list(name: str, url: str, display_name: str, description: str,
                         expected_nodes: int, expected_edges: int,
                         size_warning: Optional[str] = None) -> Dict:
    import tempfile

    _ensure_cache_dir()
    cache_path = os.path.join(_CACHE_DIR, f"{name}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        src_arr = data["src"]
        dst_arr = data["dst"]
        num_nodes = int(data["num_nodes"])
        num_edges = int(data["num_edges"])
        return {
            "name": display_name,
            "edges": _LazyEdgeList(src_arr, dst_arr),
            "labels": {},
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_classes": 0,
            "columns": "complex::reflexive::node",
            "description": description,
        }

    if size_warning:
        sys.stderr.write(f"WARNING: {size_warning}\n")
        sys.stderr.flush()

    gz_path = os.path.join(_CACHE_DIR, f"{name}.txt.gz")
    gz_tmp_path = gz_path + ".tmp"
    if not os.path.exists(gz_path):
        _download_with_progress(url, gz_tmp_path, description=f"Downloading {display_name}")
        os.rename(gz_tmp_path, gz_path)

    sys.stderr.write(f"Parsing {display_name} edges (streaming from .gz)...\n")
    sys.stderr.flush()

    dtype = np.int64 if expected_nodes > 2_147_483_647 else np.int32
    chunk_size = 1_000_000
    src_chunks = []
    dst_chunks = []
    src_buf = np.empty(chunk_size, dtype=dtype)
    dst_buf = np.empty(chunk_size, dtype=dtype)
    buf_idx = 0
    edge_count = 0

    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line or line[0] == "#" or line[0] == "\n":
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            src_buf[buf_idx] = int(parts[0])
            dst_buf[buf_idx] = int(parts[1])
            buf_idx += 1
            edge_count += 1
            if buf_idx == chunk_size:
                src_chunks.append(src_buf[:buf_idx].copy())
                dst_chunks.append(dst_buf[:buf_idx].copy())
                buf_idx = 0
                if edge_count % 5_000_000 == 0:
                    sys.stderr.write(f"\r  Parsed {edge_count:,} edges...")
                    sys.stderr.flush()

    if buf_idx > 0:
        src_chunks.append(src_buf[:buf_idx].copy())
        dst_chunks.append(dst_buf[:buf_idx].copy())

    sys.stderr.write(f"\r  Parsed {edge_count:,} edges total. Caching...\n")
    sys.stderr.flush()

    del src_buf, dst_buf

    src_arr = np.concatenate(src_chunks) if src_chunks else np.array([], dtype=dtype)
    dst_arr = np.concatenate(dst_chunks) if dst_chunks else np.array([], dtype=dtype)
    del src_chunks, dst_chunks

    if len(src_arr) > 0:
        src_unique = np.unique(src_arr)
        dst_unique = np.unique(dst_arr)
        all_unique = np.union1d(src_unique, dst_unique)
        num_nodes = len(all_unique)
        del src_unique, dst_unique, all_unique
    else:
        num_nodes = 0
    num_edges = len(src_arr)

    edge_drift = abs(num_edges - expected_edges) / max(expected_edges, 1)
    if edge_drift > 0.20:
        raise ValueError(
            f"{display_name}: parsed {num_edges:,} edges but expected ~{expected_edges:,} "
            f"(drift {edge_drift:.1%}). The download may be corrupt. "
            f"Delete {gz_path} and retry."
        )
    if edge_drift > 0.01 or num_nodes != expected_nodes:
        sys.stderr.write(
            f"  Note: parsed {num_nodes:,} nodes / {num_edges:,} edges "
            f"(expected ~{expected_nodes:,} / ~{expected_edges:,})\n"
        )
        sys.stderr.flush()

    fd, tmp_cache_path = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".npz")
    os.close(fd)
    try:
        np.savez(tmp_cache_path,
                 src=src_arr, dst=dst_arr,
                 num_nodes=num_nodes, num_edges=num_edges)
        os.rename(tmp_cache_path, cache_path)
    except BaseException:
        try:
            os.remove(tmp_cache_path)
        except OSError:
            pass
        raise

    try:
        os.remove(gz_path)
    except OSError:
        pass

    sys.stderr.write(f"  {display_name}: {num_nodes:,} nodes, {num_edges:,} edges cached.\n")
    sys.stderr.flush()

    return {
        "name": display_name,
        "edges": _LazyEdgeList(src_arr, dst_arr),
        "labels": {},
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_classes": 0,
        "columns": "complex::reflexive::node",
        "description": description,
    }


@_real("https://snap.stanford.edu/data/facebook_combined.txt.gz")
def load_facebook() -> Dict:
    return _load_snap_edge_list(
        name="facebook",
        url="https://snap.stanford.edu/data/facebook_combined.txt.gz",
        display_name="ego-Facebook",
        description="Facebook ego networks (SNAP). ~4k nodes, ~88k edges.",
        expected_nodes=4_039,
        expected_edges=88_234,
    )


@_real("https://snap.stanford.edu/data/roadNet-CA.txt.gz")
def load_roadnet() -> Dict:
    return _load_snap_edge_list(
        name="roadnet",
        url="https://snap.stanford.edu/data/roadNet-CA.txt.gz",
        display_name="roadNet-CA",
        description="California road network (SNAP). ~2M nodes, ~2.8M roads (5.5M directed edge lines).",
        expected_nodes=1_965_206,
        expected_edges=5_533_214,
        size_warning="roadNet-CA is a large dataset (~18MB compressed, ~5.5M edge lines).",
    )


@_real("https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz")
def load_livejournal() -> Dict:
    return _load_snap_edge_list(
        name="livejournal",
        url="https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz",
        display_name="soc-LiveJournal1",
        description="LiveJournal online social network (SNAP). ~4.8M nodes, ~69M edges.",
        expected_nodes=4_847_571,
        expected_edges=68_993_773,
        size_warning="soc-LiveJournal1 is a very large dataset (~250MB compressed, ~69M edges). "
                     "Download and parsing may take a long time and require significant memory.",
    )


@_real("https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz")
def load_com_orkut() -> Dict:
    return _load_snap_edge_list(
        name="com_orkut",
        url="https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz",
        display_name="com-Orkut",
        description="Orkut online social network (SNAP). ~3M nodes, ~117M edges.",
        expected_nodes=3_072_441,
        expected_edges=117_185_083,
    )


@_real("https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz")
def load_com_friendster() -> Dict:
    return _load_snap_edge_list(
        name="com_friendster",
        url="https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz",
        display_name="com-Friendster",
        description="Friendster online social network (SNAP). ~65.6M nodes, ~1.8B edges.",
        expected_nodes=65_608_366,
        expected_edges=1_806_067_135,
        size_warning="com-Friendster is a very large dataset (~1.2GB compressed download, ~1.8B edges). "
                     "Download and parsing may take a long time and require significant memory.",
    )


@_real("embedded", "W. W. Zachary, J. Anthropol. Res. 33, 452-473 (1977)")
def load_karate_club() -> Dict:
    edges = [
        "0 1", "0 2", "0 3", "0 4", "0 5", "0 6", "0 7", "0 8", "0 10", "0 11",
        "0 12", "0 13", "0 17", "0 19", "0 21", "0 31",
        "1 2", "1 3", "1 7", "1 13", "1 17", "1 19", "1 21", "1 30",
        "2 3", "2 7", "2 8", "2 9", "2 13", "2 27", "2 28", "2 32",
        "3 7", "3 12", "3 13",
        "4 6", "4 10",
        "5 6", "5 10", "5 16",
        "6 16",
        "8 30", "8 32", "8 33",
        "9 33",
        "13 33",
        "14 32", "14 33",
        "15 32", "15 33",
        "18 32", "18 33",
        "19 33",
        "20 32", "20 33",
        "22 32", "22 33",
        "23 25", "23 27", "23 29", "23 32", "23 33",
        "24 25", "24 27", "24 31",
        "25 31",
        "26 29", "26 33",
        "27 33",
        "28 31", "28 33",
        "29 32", "29 33",
        "30 32", "30 33",
        "31 32", "31 33",
        "32 33",
    ]

    labels = {}
    club_0 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 21]
    club_1 = [9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    for n in club_0:
        labels[str(n)] = 0
    for n in club_1:
        labels[str(n)] = 1

    return {
        "name": "Zachary's Karate Club",
        "edges": edges,
        "labels": labels,
        "num_nodes": 34,
        "num_edges": len(edges),
        "num_classes": 2,
        "columns": "complex::reflexive::member",
        "description": "Social network of a university karate club (Zachary, 1977). 34 members, 2 communities.",
    }


def _download_to_cache(url: str, filename: str, description: str) -> str:
    """Downloads ``url`` into the cache once and returns the local path."""
    _ensure_cache_dir()
    path = os.path.join(_CACHE_DIR, filename)
    if not os.path.exists(path):
        tmp_path = path + ".tmp"
        _download_with_progress(url, tmp_path, description=f"Downloading {description}")
        os.replace(tmp_path, path)
    return path


def _undirected_unique(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One (low, high) pair per undirected edge, self-loops dropped, sorted."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    low = np.minimum(src, dst)
    high = np.maximum(src, dst)
    keep = low != high
    pairs = np.unique(np.stack([low[keep], high[keep]], axis=1), axis=0)
    dtype = np.int32 if (pairs.size == 0 or pairs.max() <= np.iinfo(np.int32).max) else np.int64
    return pairs[:, 0].astype(dtype), pairs[:, 1].astype(dtype)


def _parse_gml(text: str):
    """Minimal reader for the GML files of M. Newman's network collection.

    Returns ``(nodes, edges)``: ``nodes`` maps node id to its attributes,
    ``edges`` is a list of attribute dictionaries with ``source`` and ``target``.
    """
    import re

    def attributes(body: str) -> Dict:
        result = {}
        for key, value in re.findall(r'(\w+)\s+("[^"]*"|[-+]?[\d.]+(?:[eE][-+]?\d+)?)', body):
            if value.startswith('"'):
                result[key] = value[1:-1]
            else:
                number = float(value)
                result[key] = int(number) if number.is_integer() else number
        return result

    nodes = {}
    edges = []
    for kind, body in re.findall(r"\b(node|edge)\s*\[(.*?)\]", text, flags=re.S):
        attrs = attributes(body)
        if kind == "node":
            nodes[attrs["id"]] = attrs
        else:
            edges.append(attrs)
    return nodes, edges


_NEWMAN_URL = "https://websites.umich.edu/~mejn/netdata/{name}.zip"


def _load_newman(name: str, display_name: str, description: str, column: str,
                 labelled: bool = False, weighted: bool = False) -> Dict:
    import zipfile

    path = _download_to_cache(_NEWMAN_URL.format(name=name), f"newman_{name}.zip", display_name)
    with zipfile.ZipFile(path) as archive:
        member = _find_zip_member(archive, ".gml")
        text = archive.read(member).decode("utf-8", errors="replace")
    nodes, raw_edges = _parse_gml(text)

    def node_name(node_id) -> str:
        return "_".join(str(nodes[node_id].get("label", node_id)).split())

    edges = []
    weights = []
    for edge in raw_edges:
        if edge["source"] == edge["target"]:
            continue
        edges.append(f"{node_name(edge['source'])} {node_name(edge['target'])}")
        weights.append(float(edge.get("value", 1.0)))

    labels = {}
    if labelled:
        labels = {node_name(node_id): int(attrs["value"]) for node_id, attrs in nodes.items() if "value" in attrs}

    dataset = {
        "name": display_name,
        "edges": edges,
        "labels": labels,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "num_classes": len(set(labels.values())),
        "columns": f"complex::reflexive::{column}",
        "description": description,
    }
    if weighted:
        dataset["weights"] = weights
    return dataset


@_real(_NEWMAN_URL.format(name="dolphins"), "D. Lusseau et al., Behav. Ecol. Sociobiol. 54, 396-405 (2003)")
def load_dolphins() -> Dict:
    return _load_newman(
        "dolphins", "Dolphins Social Network",
        "Social network of 62 bottlenose dolphins in Doubtful Sound, New Zealand (Lusseau et al., 2003). "
        "159 frequent associations. No labels.",
        column="dolphin",
    )


@_real(_NEWMAN_URL.format(name="lesmis"), "D. E. Knuth, The Stanford GraphBase (1993)")
def load_les_miserables() -> Dict:
    return _load_newman(
        "lesmis", "Les Miserables",
        "Co-appearance network of the 77 characters of Victor Hugo's Les Miserables (Knuth, 1993). "
        "254 edges; 'weights' holds the number of chapters two characters share. No labels.",
        column="character", weighted=True,
    )


@_real(_NEWMAN_URL.format(name="football"), "M. Girvan and M. E. J. Newman, PNAS 99, 7821-7826 (2002)")
def load_football() -> Dict:
    return _load_newman(
        "football", "American College Football",
        "Games between the 115 Division IA college football teams in the 2000 season "
        "(Girvan & Newman, 2002). 613 games; labels are the 12 conferences.",
        column="team", labelled=True,
    )


_PLANETOID_URL = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/ind.{name}.{part}"


def _load_planetoid_pickle(path: str):
    """Reads one of the Planetoid data files.

    The files are Python 2 pickles. Unpickling can execute arbitrary code, so only
    the handful of constructors these files actually use are allowed.
    """
    import importlib
    import pickle

    allowed = {
        ("__builtin__", "list"): ("builtins", "list"),
        ("builtins", "list"): ("builtins", "list"),
        ("collections", "defaultdict"): ("collections", "defaultdict"),
        ("numpy", "dtype"): ("numpy", "dtype"),
        ("numpy", "ndarray"): ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"): (None, "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"): (None, "_reconstruct"),
    }

    class RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            target = allowed.get((module, name))
            if target is None:
                raise pickle.UnpicklingError(f"refusing to load '{module}.{name}' from a dataset file")
            if target[0] is None:
                try:
                    multiarray = importlib.import_module("numpy._core.multiarray")
                except ImportError:
                    multiarray = importlib.import_module("numpy.core.multiarray")
                return getattr(multiarray, target[1])
            return getattr(importlib.import_module(target[0]), target[1])

    with open(path, "rb") as f:
        return RestrictedUnpickler(f, encoding="latin1").load()


def _load_planetoid(name: str, display_name: str, description: str, num_classes: int) -> Dict:
    _ensure_cache_dir()
    cache_path = os.path.join(_CACHE_DIR, f"planetoid_{name}.npz")
    if not os.path.exists(cache_path):
        parts = {}
        for part in ("graph", "ally", "ty", "test.index"):
            parts[part] = _download_to_cache(
                _PLANETOID_URL.format(name=name, part=part), f"ind.{name}.{part}", f"{display_name} ({part})"
            )
        graph = _load_planetoid_pickle(parts["graph"])
        ally = np.asarray(_load_planetoid_pickle(parts["ally"]))
        ty = np.asarray(_load_planetoid_pickle(parts["ty"]))
        with open(parts["test.index"]) as f:
            test_index = np.array([int(line) for line in f if line.strip()], dtype=np.int64)

        # Test nodes are stored in shuffled order; CiteSeer additionally has test
        # indices without any data (isolated papers), which stay unlabelled.
        test_sorted = np.sort(test_index)
        full_range = np.arange(test_sorted[0], test_sorted[-1] + 1)
        ty_full = np.zeros((len(full_range), ty.shape[1]), dtype=ty.dtype)
        ty_full[test_sorted - test_sorted[0]] = ty
        one_hot = np.vstack([ally, ty_full])
        one_hot[test_index] = one_hot[test_sorted]
        has_label = one_hot.sum(axis=1) > 0
        label_of = np.where(has_label, one_hot.argmax(axis=1), -1).astype(np.int32)

        src = np.fromiter((node for node, nbrs in graph.items() for _ in nbrs), dtype=np.int64)
        dst = np.fromiter((nbr for nbrs in graph.values() for nbr in nbrs), dtype=np.int64)
        src, dst = _undirected_unique(src, dst)
        tmp_path = cache_path + ".tmp.npz"
        np.savez(tmp_path, src=src, dst=dst, labels=label_of)
        os.replace(tmp_path, cache_path)

    data = np.load(cache_path, allow_pickle=False)
    src, dst, label_of = data["src"], data["dst"], data["labels"]
    in_graph = np.union1d(src, dst)
    labels = {str(int(node)): int(label_of[node]) for node in in_graph if label_of[node] >= 0}
    return {
        "name": display_name,
        "edges": _LazyEdgeList(src, dst),
        "labels": labels,
        "num_nodes": int(len(in_graph)),
        "num_edges": int(len(src)),
        "num_classes": num_classes,
        "columns": "complex::reflexive::paper",
        "description": description,
    }


_PLANETOID_CITATION = "Z. Yang, W. Cohen, R. Salakhutdinov, ICML 2016 (data: Sen et al., AI Magazine 2008)"


@_real(_PLANETOID_URL.format(name="cora", part="*"), _PLANETOID_CITATION)
def load_cora() -> Dict:
    return _load_planetoid(
        "cora", "Cora Citation Network",
        "Citation network of 2,708 machine-learning papers (Planetoid version): "
        "5,278 undirected citation links, 7 subject classes.",
        num_classes=7,
    )


@_real(_PLANETOID_URL.format(name="citeseer", part="*"), _PLANETOID_CITATION)
def load_citeseer() -> Dict:
    return _load_planetoid(
        "citeseer", "CiteSeer Citation Network",
        "Citation network of computer-science papers (Planetoid version): "
        "3,279 papers with at least one link, 4,552 undirected citation links, 6 subject classes.",
        num_classes=6,
    )


@_real(_PLANETOID_URL.format(name="pubmed", part="*"), _PLANETOID_CITATION)
def load_pubmed() -> Dict:
    return _load_planetoid(
        "pubmed", "PubMed Diabetes Citation Network",
        "Citation network of 19,717 diabetes publications (Planetoid version): "
        "44,324 undirected citation links, 3 classes.",
        num_classes=3,
    )


_AMAZON_URL = "https://raw.githubusercontent.com/shchur/gnn-benchmark/master/data/npz/amazon_electronics_{name}.npz"


def _load_amazon(name: str, display_name: str, description: str) -> Dict:
    path = _download_to_cache(_AMAZON_URL.format(name=name), f"amazon_electronics_{name}.npz", display_name)
    data = np.load(path, allow_pickle=False)
    indptr = data["adj_indptr"]
    src = np.repeat(np.arange(len(indptr) - 1, dtype=np.int64), np.diff(indptr))
    src, dst = _undirected_unique(src, data["adj_indices"])
    label_of = np.asarray(data["labels"], dtype=np.int64)
    in_graph = np.union1d(src, dst)
    return {
        "name": display_name,
        "edges": _LazyEdgeList(src, dst),
        "labels": {str(int(node)): int(label_of[node]) for node in in_graph},
        "num_nodes": int(len(in_graph)),
        "num_edges": int(len(src)),
        "num_classes": int(len(np.unique(label_of))),
        "columns": "complex::reflexive::product",
        "description": description,
    }


_AMAZON_CITATION = "O. Shchur et al., Pitfalls of Graph Neural Network Evaluation (2018); data: McAuley et al., SIGIR 2015"


@_real(_AMAZON_URL.format(name="computers"), _AMAZON_CITATION)
def load_amazon_computers() -> Dict:
    return _load_amazon(
        "computers", "Amazon Computers",
        "Amazon co-purchase graph of computer products: nodes are products, edges link "
        "products that are frequently bought together, labels are 10 product categories.",
    )


@_real(_AMAZON_URL.format(name="photo"), _AMAZON_CITATION)
def load_amazon_photo() -> Dict:
    return _load_amazon(
        "photo", "Amazon Photo",
        "Amazon co-purchase graph of photo products: nodes are products, edges link "
        "products that are frequently bought together, labels are 8 product categories.",
    )


def _load_node2vec_mat(filename: str, display_name: str, description: str, column: str) -> Dict:
    """The multi-label networks distributed with node2vec (MATLAB files)."""
    import scipy.io
    import scipy.sparse

    url = f"https://snap.stanford.edu/node2vec/{filename}"
    path = _download_to_cache(url, f"node2vec_{filename}", display_name)
    content = scipy.io.loadmat(path)
    adjacency = scipy.sparse.coo_matrix(content["network"])
    src, dst = _undirected_unique(adjacency.row, adjacency.col)
    label_matrix = np.asarray(scipy.sparse.csr_matrix(content["group"]).todense()).astype(np.uint8)
    in_graph = np.union1d(src, dst)
    return {
        "name": display_name,
        "edges": _LazyEdgeList(src, dst),
        # every node may carry several labels: see "label_matrix"
        "labels": {},
        "multilabel": True,
        "label_entities": [str(int(node)) for node in in_graph],
        "label_matrix": label_matrix[in_graph],
        "num_nodes": int(len(in_graph)),
        "num_edges": int(len(src)),
        "num_classes": int(label_matrix.shape[1]),
        "columns": f"complex::reflexive::{column}",
        "description": description,
    }


_NODE2VEC_CITATION = "A. Grover and J. Leskovec, node2vec, KDD 2016"


@_real("https://snap.stanford.edu/node2vec/Homo_sapiens.mat", _NODE2VEC_CITATION)
def load_ppi() -> Dict:
    return _load_node2vec_mat(
        "Homo_sapiens.mat", "Protein-Protein Interaction (Homo sapiens)",
        "Protein-protein interaction network of Homo sapiens (node2vec version): 3,860 proteins with "
        "at least one interaction, 37,845 interactions, 50 biological-state labels. Multi-label: a protein can carry several labels "
        "('label_matrix' has one row per entry of 'label_entities').",
        column="protein",
    )


@_real("https://snap.stanford.edu/node2vec/POS.mat", _NODE2VEC_CITATION)
def load_wikipedia_pos() -> Dict:
    return _load_node2vec_mat(
        "POS.mat", "Wikipedia word co-occurrence (POS)",
        "Co-occurrence network of 4,777 words from a Wikipedia dump (node2vec version) with "
        "40 part-of-speech labels. Multi-label ('label_matrix' / 'label_entities').",
        column="word",
    )


@_real("https://snap.stanford.edu/data/email-Eu-core.html",
       "J. Leskovec, J. Kleinberg, C. Faloutsos (2007); H. Yin et al., KDD 2017")
def load_email_eu_core() -> Dict:
    edges_path = _download_to_cache(
        "https://snap.stanford.edu/data/email-Eu-core.txt.gz", "email-Eu-core.txt.gz", "email-Eu-core")
    labels_path = _download_to_cache(
        "https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz",
        "email-Eu-core-department-labels.txt.gz", "email-Eu-core labels")
    pairs = np.loadtxt(edges_path, dtype=np.int64, ndmin=2)
    src, dst = _undirected_unique(pairs[:, 0], pairs[:, 1])
    departments = np.loadtxt(labels_path, dtype=np.int64, ndmin=2)
    department_of = dict(zip(departments[:, 0].tolist(), departments[:, 1].tolist()))
    in_graph = np.union1d(src, dst)
    return {
        "name": "email-Eu-core",
        "edges": _LazyEdgeList(src, dst),
        "labels": {str(int(node)): int(department_of[int(node)]) for node in in_graph if int(node) in department_of},
        "num_nodes": int(len(in_graph)),
        "num_edges": int(len(src)),
        "num_classes": int(len(set(department_of.values()))),
        "columns": "complex::reflexive::person",
        "description": "E-mail network of a European research institution (SNAP): nodes are people, an edge "
                       "means at least one e-mail was exchanged, labels are the 42 departments.",
    }


@_real("https://snap.stanford.edu/graphsage/ppi.zip", "W. Hamilton, R. Ying, J. Leskovec, GraphSAGE, NeurIPS 2017")
def load_ppi_large() -> Dict:
    import json
    import zipfile

    path = _download_to_cache("https://snap.stanford.edu/graphsage/ppi.zip", "graphsage_ppi.zip", "PPI (GraphSAGE)")
    with zipfile.ZipFile(path) as archive:
        graph = json.loads(archive.read(_find_zip_member(archive, "ppi-G.json")))
        class_map = json.loads(archive.read(_find_zip_member(archive, "ppi-class_map.json")))
    src = np.fromiter((link["source"] for link in graph["links"]), dtype=np.int64)
    dst = np.fromiter((link["target"] for link in graph["links"]), dtype=np.int64)
    src, dst = _undirected_unique(src, dst)
    in_graph = np.union1d(src, dst)
    label_matrix = np.array([class_map[str(int(node))] for node in in_graph], dtype=np.uint8)
    return {
        "name": "PPI (GraphSAGE)",
        "edges": _LazyEdgeList(src, dst),
        "labels": {},
        "multilabel": True,
        "label_entities": [str(int(node)) for node in in_graph],
        "label_matrix": label_matrix,
        "num_nodes": int(len(in_graph)),
        "num_edges": int(len(src)),
        "num_classes": int(label_matrix.shape[1]),
        "columns": "complex::reflexive::protein",
        "description": "Protein-protein interaction graphs of 24 human tissues (GraphSAGE version): "
                       "56,688 proteins with at least one interaction, 121 gene-ontology labels. Multi-label "
                       "('label_matrix' / 'label_entities').",
    }


@_real("https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip", "W. Hu et al., Open Graph Benchmark, NeurIPS 2020")
def load_ogbn_arxiv() -> Dict:
    return _load_ogb_dataset(
        name="ogb_arxiv",
        display_name="ogbn-arxiv",
        description="OGB arxiv citation network: 169,343 computer-science papers, 1.17M citations, "
                    "40 subject areas.",
        zip_url="https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip",
        edge_csv_path_in_zip="arxiv/raw/edge.csv.gz",
        expected_nodes=169_343,
        expected_edges=1_166_243,
        label_csv_path_in_zip="arxiv/raw/node-label.csv.gz",
        num_classes=40,
        columns="complex::reflexive::paper",
    )


@_real("https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv")
def load_reddit_hyperlink() -> Dict:
    import tempfile
    import csv

    name = "reddit_hyperlink"
    display_name = "Reddit Hyperlink Network"
    description = "Reddit hyperlink network (SNAP). Subreddits as nodes, hyperlinks between posts as edges. ~55K nodes, ~858K edges."

    _ensure_cache_dir()
    cache_path = os.path.join(_CACHE_DIR, f"{name}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        src_arr = data["src"]
        dst_arr = data["dst"]
        num_nodes = int(data["num_nodes"])
        num_edges = int(data["num_edges"])
        return {
            "name": display_name,
            "edges": _LazyEdgeList(src_arr, dst_arr),
            "labels": {},
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_classes": 0,
            "columns": "complex::reflexive::subreddit",
            "description": description,
        }

    url = "https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv"
    tsv_path = os.path.join(_CACHE_DIR, f"{name}.tsv")
    tsv_tmp_path = tsv_path + ".tmp"
    if not os.path.exists(tsv_path):
        _download_with_progress(url, tsv_tmp_path, description=f"Downloading {display_name}")
        os.rename(tsv_tmp_path, tsv_path)

    sys.stderr.write(f"Parsing {display_name} edges from TSV...\n")
    sys.stderr.flush()

    node_map = {}
    next_id = 0
    src_list = []
    dst_list = []

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            source_sub = row[0].strip()
            target_sub = row[1].strip()
            if source_sub not in node_map:
                node_map[source_sub] = next_id
                next_id += 1
            if target_sub not in node_map:
                node_map[target_sub] = next_id
                next_id += 1
            src_list.append(node_map[source_sub])
            dst_list.append(node_map[target_sub])

    src_arr = np.array(src_list, dtype=np.int32)
    dst_arr = np.array(dst_list, dtype=np.int32)
    del src_list, dst_list

    num_nodes = len(node_map)
    num_edges = len(src_arr)
    del node_map

    sys.stderr.write(f"  {display_name}: {num_nodes:,} nodes, {num_edges:,} edges. Caching...\n")
    sys.stderr.flush()

    fd, tmp_cache_path = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".npz")
    os.close(fd)
    try:
        np.savez(tmp_cache_path,
                 src=src_arr, dst=dst_arr,
                 num_nodes=num_nodes, num_edges=num_edges)
        os.rename(tmp_cache_path, cache_path)
    except BaseException:
        try:
            os.remove(tmp_cache_path)
        except OSError:
            pass
        raise

    try:
        os.remove(tsv_path)
    except OSError:
        pass

    return {
        "name": display_name,
        "edges": _LazyEdgeList(src_arr, dst_arr),
        "labels": {},
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_classes": 0,
        "columns": "complex::reflexive::subreddit",
        "description": description,
    }


def _find_zip_member(zf, target_suffix: str) -> str:
    for member in zf.namelist():
        if member.endswith(target_suffix):
            return member
    raise KeyError(f"No zip member ending with '{target_suffix}' found. "
                   f"Available: {zf.namelist()[:20]}")


def _load_ogb_dataset(name: str, display_name: str, description: str,
                      zip_url: str, edge_csv_path_in_zip: str,
                      expected_nodes: int, expected_edges: int,
                      label_csv_path_in_zip: Optional[str] = None,
                      num_classes: int = 0,
                      columns: str = "complex::reflexive::node") -> Dict:
    import tempfile
    import zipfile
    import io

    _ensure_cache_dir()
    cache_path = os.path.join(_CACHE_DIR, f"{name}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        src_arr = data["src"]
        dst_arr = data["dst"]
        num_nodes = int(data["num_nodes"])
        num_edges = int(data["num_edges"])
        labels = {}
        if "label_keys" in data and "label_vals" in data:
            labels = {str(k): int(v) for k, v in zip(data["label_keys"].tolist(), data["label_vals"].tolist())}
        return {
            "name": display_name,
            "edges": _LazyEdgeList(src_arr, dst_arr),
            "labels": labels,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_classes": num_classes,
            "columns": columns,
            "description": description,
        }

    zip_path = os.path.join(_CACHE_DIR, f"{name}.zip")
    zip_tmp_path = zip_path + ".tmp"
    if not os.path.exists(zip_path):
        _download_with_progress(zip_url, zip_tmp_path, description=f"Downloading {display_name}")
        os.rename(zip_tmp_path, zip_path)

    sys.stderr.write(f"Extracting {display_name} edges from zip...\n")
    sys.stderr.flush()

    dtype = np.int64 if expected_nodes > 2_147_483_647 else np.int32
    chunk_size = 1_000_000
    src_chunks = []
    dst_chunks = []
    src_buf = np.empty(chunk_size, dtype=dtype)
    dst_buf = np.empty(chunk_size, dtype=dtype)
    buf_idx = 0
    edge_count = 0
    max_node_id = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        edge_member = _find_zip_member(zf, edge_csv_path_in_zip.split("/", 1)[-1])

        with zf.open(edge_member) as ef:
            if edge_member.endswith(".gz"):
                stream = gzip.open(ef, "rt", encoding="utf-8")
            else:
                stream = io.TextIOWrapper(ef, encoding="utf-8")

            for line in stream:
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                try:
                    s = int(parts[0])
                    t = int(parts[1])
                except ValueError:
                    continue
                src_buf[buf_idx] = s
                dst_buf[buf_idx] = t
                if s > max_node_id:
                    max_node_id = s
                if t > max_node_id:
                    max_node_id = t
                buf_idx += 1
                edge_count += 1
                if buf_idx == chunk_size:
                    src_chunks.append(src_buf[:buf_idx].copy())
                    dst_chunks.append(dst_buf[:buf_idx].copy())
                    buf_idx = 0
                    if edge_count % 5_000_000 == 0:
                        sys.stderr.write(f"\r  Parsed {edge_count:,} edges...")
                        sys.stderr.flush()

        if buf_idx > 0:
            src_chunks.append(src_buf[:buf_idx].copy())
            dst_chunks.append(dst_buf[:buf_idx].copy())
        del src_buf, dst_buf

        labels = {}
        label_keys = []
        label_vals = []
        if label_csv_path_in_zip:
            try:
                label_suffix = label_csv_path_in_zip.split("/", 1)[-1]
                label_member = _find_zip_member(zf, label_suffix)
                with zf.open(label_member) as lf:
                    if label_member.endswith(".gz"):
                        lstream = gzip.open(lf, "rt", encoding="utf-8")
                    else:
                        lstream = io.TextIOWrapper(lf, encoding="utf-8")
                    for node_id, lline in enumerate(lstream):
                        lline = lline.strip()
                        if lline:
                            try:
                                label_val = int(lline.split(",")[0])
                                label_keys.append(str(node_id))
                                label_vals.append(label_val)
                                labels[str(node_id)] = label_val
                            except ValueError:
                                continue
            except (KeyError, FileNotFoundError):
                sys.stderr.write(f"  Warning: label file not found in zip, skipping labels.\n")
                sys.stderr.flush()

    src_arr = np.concatenate(src_chunks) if src_chunks else np.array([], dtype=dtype)
    dst_arr = np.concatenate(dst_chunks) if dst_chunks else np.array([], dtype=dtype)
    del src_chunks, dst_chunks

    num_nodes_actual = max_node_id + 1 if len(src_arr) > 0 else 0
    num_edges_actual = len(src_arr)

    sys.stderr.write(f"\r  {display_name}: {num_nodes_actual:,} nodes, {num_edges_actual:,} edges. Caching...\n")
    sys.stderr.flush()

    fd, tmp_cache_path = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".npz")
    os.close(fd)
    try:
        save_kwargs = dict(src=src_arr, dst=dst_arr,
                          num_nodes=num_nodes_actual, num_edges=num_edges_actual)
        if label_keys:
            save_kwargs["label_keys"] = np.array(label_keys)
            save_kwargs["label_vals"] = np.array(label_vals)
        np.savez(tmp_cache_path, **save_kwargs)
        os.rename(tmp_cache_path, cache_path)
    except BaseException:
        try:
            os.remove(tmp_cache_path)
        except OSError:
            pass
        raise

    try:
        os.remove(zip_path)
    except OSError:
        pass

    return {
        "name": display_name,
        "edges": _LazyEdgeList(src_arr, dst_arr),
        "labels": labels,
        "num_nodes": num_nodes_actual,
        "num_edges": num_edges_actual,
        "num_classes": num_classes,
        "columns": columns,
        "description": description,
    }


@_real("https://snap.stanford.edu/ogb/data/nodeproppred/ogbn-products.zip", "W. Hu et al., Open Graph Benchmark, NeurIPS 2020")
def load_ogbn_products() -> Dict:
    return _load_ogb_dataset(
        name="ogbn_products",
        display_name="ogbn-products",
        description="OGB products co-purchasing graph. 2.4M product nodes, 62M edges, 47 categories.",
        zip_url="https://snap.stanford.edu/ogb/data/nodeproppred/ogbn-products.zip",
        edge_csv_path_in_zip="ogbn-products/raw/edge.csv.gz",
        expected_nodes=2_449_029,
        expected_edges=61_859_140,
        label_csv_path_in_zip="ogbn-products/raw/node-label.csv.gz",
        num_classes=47,
        columns="complex::reflexive::product",
    )


@_real("https://snap.stanford.edu/ogb/data/linkproppred/ogbl-citation2.zip", "W. Hu et al., Open Graph Benchmark, NeurIPS 2020")
def load_ogbl_citation2() -> Dict:
    return _load_ogb_dataset(
        name="ogbl_citation2",
        display_name="ogbl-citation2",
        description="OGB citation2 graph. 2.9M papers, 30M citation edges. Link prediction benchmark.",
        zip_url="https://snap.stanford.edu/ogb/data/linkproppred/ogbl-citation2.zip",
        edge_csv_path_in_zip="ogbl-citation2/raw/edge.csv.gz",
        expected_nodes=2_927_963,
        expected_edges=30_561_187,
        num_classes=0,
        columns="complex::reflexive::paper",
    )


@_real("https://nrvis.com/download/data/soc/soc-twitter.zip")
def load_twitter() -> Dict:
    import tempfile
    import zipfile

    name = "twitter"
    display_name = "Twitter-2010"
    description = "Twitter-2010 follower network. ~41.7M users, ~1.47B edges."
    expected_nodes = 41_652_230
    expected_edges = 1_468_365_182

    _ensure_cache_dir()
    cache_path = os.path.join(_CACHE_DIR, f"{name}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        src_arr = data["src"]
        dst_arr = data["dst"]
        num_nodes = int(data["num_nodes"])
        num_edges = int(data["num_edges"])
        return {
            "name": display_name,
            "edges": _LazyEdgeList(src_arr, dst_arr),
            "labels": {},
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_classes": 0,
            "columns": "complex::reflexive::user",
            "description": description,
        }

    sys.stderr.write(
        "WARNING: Twitter-2010 is a very large dataset (~6GB compressed, ~1.47B edges). "
        "Download and parsing may take a long time and require significant memory.\n"
    )
    sys.stderr.flush()

    zip_url = "https://nrvis.com/download/data/soc/soc-twitter.zip"
    zip_path = os.path.join(_CACHE_DIR, f"{name}.zip")
    zip_tmp_path = zip_path + ".tmp"
    if not os.path.exists(zip_path):
        _download_with_progress(zip_url, zip_tmp_path, description=f"Downloading {display_name}")
        os.rename(zip_tmp_path, zip_path)

    sys.stderr.write(f"Parsing {display_name} edges (streaming from zip)...\n")
    sys.stderr.flush()

    dtype = np.int32
    chunk_size = 1_000_000
    src_chunks = []
    dst_chunks = []
    src_buf = np.empty(chunk_size, dtype=dtype)
    dst_buf = np.empty(chunk_size, dtype=dtype)
    buf_idx = 0
    edge_count = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        edge_file = None
        for zi in zf.namelist():
            if zi.endswith(".edges") or zi.endswith(".mtx") or zi.endswith(".txt") or zi.endswith(".csv"):
                edge_file = zi
                break
        if edge_file is None:
            edge_file = [n for n in zf.namelist() if not n.endswith("/")][0]

        import io
        header_skipped = False
        with zf.open(edge_file) as ef:
            reader = io.TextIOWrapper(ef, encoding="utf-8")
            for line in reader:
                if not line or line[0] == "%" or line[0] == "#" or line[0] == "\n":
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                if not header_skipped and len(parts) >= 3:
                    try:
                        int(parts[0])
                        int(parts[1])
                        int(parts[2])
                        header_skipped = True
                        continue
                    except ValueError:
                        header_skipped = True
                if not header_skipped:
                    header_skipped = True
                try:
                    s = int(parts[0])
                    t = int(parts[1])
                except ValueError:
                    continue
                src_buf[buf_idx] = s
                dst_buf[buf_idx] = t
                buf_idx += 1
                edge_count += 1
                if buf_idx == chunk_size:
                    src_chunks.append(src_buf[:buf_idx].copy())
                    dst_chunks.append(dst_buf[:buf_idx].copy())
                    buf_idx = 0
                    if edge_count % 5_000_000 == 0:
                        sys.stderr.write(f"\r  Parsed {edge_count:,} edges...")
                        sys.stderr.flush()

    if buf_idx > 0:
        src_chunks.append(src_buf[:buf_idx].copy())
        dst_chunks.append(dst_buf[:buf_idx].copy())

    sys.stderr.write(f"\r  Parsed {edge_count:,} edges total. Caching...\n")
    sys.stderr.flush()

    del src_buf, dst_buf

    src_arr = np.concatenate(src_chunks) if src_chunks else np.array([], dtype=dtype)
    dst_arr = np.concatenate(dst_chunks) if dst_chunks else np.array([], dtype=dtype)
    del src_chunks, dst_chunks

    if len(src_arr) > 0:
        num_nodes = int(max(src_arr.max(), dst_arr.max())) + 1
    else:
        num_nodes = 0
    num_edges = len(src_arr)

    edge_drift = abs(num_edges - expected_edges) / max(expected_edges, 1)
    if edge_drift > 0.20:
        raise ValueError(
            f"{display_name}: parsed {num_edges:,} edges but expected ~{expected_edges:,} "
            f"(drift {edge_drift:.1%}). The download may be corrupt. "
            f"Delete {zip_path} and retry."
        )
    if edge_drift > 0.01 or num_nodes != expected_nodes:
        sys.stderr.write(
            f"  Note: parsed {num_nodes:,} nodes / {num_edges:,} edges "
            f"(expected ~{expected_nodes:,} / ~{expected_edges:,})\n"
        )
        sys.stderr.flush()

    fd, tmp_cache_path = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".npz")
    os.close(fd)
    try:
        np.savez(tmp_cache_path,
                 src=src_arr, dst=dst_arr,
                 num_nodes=num_nodes, num_edges=num_edges)
        os.rename(tmp_cache_path, cache_path)
    except BaseException:
        try:
            os.remove(tmp_cache_path)
        except OSError:
            pass
        raise

    try:
        os.remove(zip_path)
    except OSError:
        pass

    sys.stderr.write(f"  {display_name}: {num_nodes:,} nodes, {num_edges:,} edges cached.\n")
    sys.stderr.flush()

    return {
        "name": display_name,
        "edges": _LazyEdgeList(src_arr, dst_arr),
        "labels": {},
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_classes": 0,
        "columns": "complex::reflexive::user",
        "description": description,
    }


# ---------------------------------------------------------------------------
# Synthetic graphs
#
# pycleora <= 3.2 returned the graphs below from loaders named after real
# datasets (load_cora, load_ppi, load_flickr, ...). They are planted-partition
# random graphs that merely share the node / edge / class counts of those
# datasets; their labels are the planted communities. They are kept, under an
# explicit name, so that earlier experiments can be reproduced.
# ---------------------------------------------------------------------------

def _generate_citation_graph(name, num_classes, seed=42):
    rng = np.random.default_rng(seed)

    configs = {
        "cora": {"nodes": 2708, "edges": 5429, "feat_dim": 1433},
        "citeseer": {"nodes": 3312, "edges": 4732, "feat_dim": 3703},
        "pubmed": {"nodes": 19717, "edges": 44338, "feat_dim": 500},
    }

    cfg = configs[name]
    n = cfg["nodes"]
    num_edges = cfg["edges"]
    feat_dim = cfg["feat_dim"]

    community_assignments = rng.integers(0, num_classes, size=n)
    labels = {}
    for i in range(n):
        labels[f"p{i}"] = int(community_assignments[i])

    edge_set = set()
    for i in range(n):
        comm = community_assignments[i]
        num_neighbors = rng.poisson(lam=num_edges * 2 / n)
        num_neighbors = max(1, min(num_neighbors, 20))

        for _ in range(num_neighbors):
            if rng.random() < 0.7:
                same_comm = np.where(community_assignments == comm)[0]
                j = int(rng.choice(same_comm))
            else:
                j = int(rng.integers(0, n))
            if i != j:
                edge_set.add((min(i, j), max(i, j)))

            if len(edge_set) >= num_edges:
                break
        if len(edge_set) >= num_edges:
            break

    while len(edge_set) < num_edges:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i != j:
            edge_set.add((min(i, j), max(i, j)))

    edges = [f"p{i} p{j}" for i, j in edge_set]

    features = rng.standard_normal((n, min(feat_dim, 64))).astype(np.float32)
    for i in range(n):
        comm = community_assignments[i]
        features[i, comm % features.shape[1]] += 2.0

    return edges, labels, features


def _generate_product_graph(name, display_name, description, num_nodes, num_edges, num_classes, seed):
    _ensure_cache_dir()
    cache_path = os.path.join(_CACHE_DIR, f"synthetic_{name}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        return {
            "name": display_name,
            "edges": data["edges"].tolist(),
            "labels": dict(zip(data["label_keys"].tolist(), data["label_vals"].tolist())),
            "num_nodes": int(data["num_nodes"]),
            "num_edges": int(data["num_edges"]),
            "num_classes": int(data["num_classes"]),
            "columns": "complex::reflexive::product",
            "description": description,
        }

    rng = np.random.default_rng(seed)
    community = rng.integers(0, num_classes, size=num_nodes)
    labels = {f"prod{i}": int(community[i]) for i in range(num_nodes)}

    edge_set = set()
    for i in range(num_nodes):
        comm = community[i]
        num_nb = rng.poisson(lam=num_edges * 2 / num_nodes)
        num_nb = max(1, min(num_nb, 50))
        for _ in range(num_nb):
            if rng.random() < 0.65:
                same = np.where(community == comm)[0]
                j = int(rng.choice(same))
            else:
                j = int(rng.integers(0, num_nodes))
            if i != j:
                edge_set.add((min(i, j), max(i, j)))
            if len(edge_set) >= num_edges:
                break
        if len(edge_set) >= num_edges:
            break

    while len(edge_set) < num_edges:
        i, j = int(rng.integers(0, num_nodes)), int(rng.integers(0, num_nodes))
        if i != j:
            edge_set.add((min(i, j), max(i, j)))

    edges = [f"prod{i} prod{j}" for i, j in edge_set]

    np.savez(cache_path,
             edges=np.array(edges),
             label_keys=np.array(list(labels.keys())),
             label_vals=np.array(list(labels.values())),
             num_nodes=num_nodes, num_edges=len(edges), num_classes=num_classes)

    return {
        "name": display_name,
        "edges": edges,
        "labels": labels,
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "num_classes": num_classes,
        "columns": "complex::reflexive::product",
        "description": description,
    }


def _generate_large_community_graph(name, display_name, description, num_nodes, num_edges,
                                     num_classes, columns, seed, intra_prob=0.6):
    _ensure_cache_dir()
    cache_path = os.path.join(_CACHE_DIR, f"synthetic_{name}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        return {
            "name": display_name,
            "edges": data["edges"].tolist(),
            "labels": dict(zip(data["label_keys"].tolist(), data["label_vals"].tolist())),
            "num_nodes": int(data["num_nodes"]),
            "num_edges": int(data["num_edges"]),
            "num_classes": int(data["num_classes"]),
            "columns": columns,
            "description": description,
        }

    sys.stderr.write(f"Generating {display_name} ({num_nodes:,} nodes, {num_edges:,} edges)...\n")
    sys.stderr.flush()

    rng = np.random.default_rng(seed)

    community = rng.integers(0, num_classes, size=num_nodes)

    comm_members = {}
    for c in range(num_classes):
        comm_members[c] = np.where(community == c)[0]

    edge_set = set()
    batch = max(num_edges // 20, 100000)

    while len(edge_set) < num_edges:
        remaining = num_edges - len(edge_set)
        gen_count = min(remaining * 2, batch * 2)
        srcs = rng.integers(0, num_nodes, size=gen_count)
        is_intra = rng.random(size=gen_count) < intra_prob

        for k in range(gen_count):
            i = int(srcs[k])
            if is_intra[k]:
                members = comm_members[community[i]]
                j = int(members[rng.integers(0, len(members))])
            else:
                j = int(rng.integers(0, num_nodes))
            if i != j:
                edge_set.add((min(i, j), max(i, j)))
            if len(edge_set) >= num_edges:
                break

        if len(edge_set) % 500000 < batch:
            sys.stderr.write(f"\r  Generated {len(edge_set):,}/{num_edges:,} edges...")
            sys.stderr.flush()

    sys.stderr.write(f"\r  Generated {len(edge_set):,} edges total. Caching...\n")
    sys.stderr.flush()

    prefix = name.replace("_", "")[:3]
    edges = [f"{prefix}{i} {prefix}{j}" for i, j in edge_set]
    labels = {f"{prefix}{i}": int(community[i]) for i in range(num_nodes)}

    np.savez(cache_path,
             edges=np.array(edges),
             label_keys=np.array(list(labels.keys())),
             label_vals=np.array(list(labels.values())),
             num_nodes=num_nodes, num_edges=len(edges), num_classes=num_classes)

    return {
        "name": display_name,
        "edges": edges,
        "labels": labels,
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "num_classes": num_classes,
        "columns": columns,
        "description": description,
    }


def _generate_coauthor_graph() -> Dict:
    _ensure_cache_dir()
    cache_path = os.path.join(_CACHE_DIR, "synthetic_dblp.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        return {
            "name": "DBLP",
            "edges": data["edges"].tolist(),
            "labels": dict(zip(data["label_keys"].tolist(), data["label_vals"].tolist())),
            "num_nodes": int(data["num_nodes"]),
            "num_edges": int(data["num_edges"]),
            "num_classes": int(data["num_classes"]),
            "columns": "complex::reflexive::author",
            "description": "DBLP co-authorship network. 4 research areas.",
            "is_heterogeneous": True,
        }

    rng = np.random.default_rng(400)
    num_authors = 4057
    num_papers = 14328
    num_classes = 4

    author_area = rng.integers(0, num_classes, size=num_authors)
    labels = {f"author{i}": int(author_area[i]) for i in range(num_authors)}

    author_edges = set()
    author_paper_edges = []

    for p in range(num_papers):
        area = rng.integers(0, num_classes)
        same_area = np.where(author_area == area)[0]
        num_authors_per_paper = rng.integers(2, 5)
        if len(same_area) >= num_authors_per_paper:
            paper_authors = rng.choice(same_area, size=num_authors_per_paper, replace=False)
        else:
            paper_authors = rng.choice(num_authors, size=num_authors_per_paper, replace=False)

        for a in paper_authors:
            author_paper_edges.append((f"author{a}", f"paper{p}"))

        for i in range(len(paper_authors)):
            for j in range(i + 1, len(paper_authors)):
                a1, a2 = int(paper_authors[i]), int(paper_authors[j])
                author_edges.add((min(a1, a2), max(a1, a2)))

    edges = [f"author{i} author{j}" for i, j in author_edges]

    np.savez(cache_path,
             edges=np.array(edges),
             label_keys=np.array(list(labels.keys())),
             label_vals=np.array(list(labels.values())),
             num_nodes=num_authors, num_edges=len(edges),
             num_classes=num_classes)

    return {
        "name": "DBLP",
        "edges": edges,
        "labels": labels,
        "num_nodes": num_authors,
        "num_edges": len(edges),
        "num_classes": num_classes,
        "columns": "complex::reflexive::author",
        "description": "DBLP co-authorship network. 4 research areas.",
        "is_heterogeneous": True,
    }



_SYNTHETIC_SPECS = {
    # name: (generator, arguments)
    "cora": ("citation", dict(num_classes=7)),
    "citeseer": ("citation", dict(num_classes=6)),
    "pubmed": ("citation", dict(num_classes=3)),
    "amazon_computers": ("product", dict(num_nodes=13752, num_edges=245861, num_classes=10, seed=100)),
    "amazon_photo": ("product", dict(num_nodes=7650, num_edges=119081, num_classes=8, seed=200)),
    "ppi": ("product", dict(num_nodes=3890, num_edges=76584, num_classes=50, seed=300)),
    "reddit": ("product", dict(num_nodes=10000, num_edges=100000, num_classes=41, seed=500)),
    "dblp": ("coauthor", dict()),
    "ogbn_arxiv": ("community", dict(num_nodes=169343, num_edges=1166243, num_classes=40,
                                     columns="complex::reflexive::paper", seed=1001, intra_prob=0.65)),
    "flickr": ("community", dict(num_nodes=89250, num_edges=899756, num_classes=7,
                                 columns="complex::reflexive::image", seed=1002, intra_prob=0.55)),
    "ppi_large": ("community", dict(num_nodes=56944, num_edges=818716, num_classes=121,
                                    columns="complex::reflexive::protein", seed=1003, intra_prob=0.50)),
    "yelp": ("community", dict(num_nodes=716847, num_edges=6977410, num_classes=100,
                               columns="complex::reflexive::business", seed=1004, intra_prob=0.55)),
}


def list_synthetic() -> List[str]:
    """Names accepted by :func:`load_synthetic`."""
    return list(_SYNTHETIC_SPECS)


def load_synthetic(name: str) -> Dict:
    """A planted-partition random graph with the size of the named dataset.

    These graphs are NOT the datasets they are sized after: edges are random, with
    a fixed share falling inside randomly assigned communities, and the labels are
    those communities. Classification accuracy on them says how well a method
    recovers planted communities, nothing about the real dataset.
    """
    if name not in _SYNTHETIC_SPECS:
        raise ValueError(f"Unknown synthetic graph: '{name}'. Available: {', '.join(_SYNTHETIC_SPECS)}")
    kind, arguments = _SYNTHETIC_SPECS[name]
    title = f"Synthetic planted-partition graph (sized like {name})"
    note = "Synthetic planted-partition graph; labels are the planted communities."
    if kind == "citation":
        edges, labels, features = _generate_citation_graph(name, arguments["num_classes"])
        dataset = {
            "edges": edges,
            "labels": labels,
            "num_nodes": len(labels),
            "num_edges": len(edges),
            "num_classes": arguments["num_classes"],
            "columns": "complex::reflexive::paper",
            "features": features,
        }
    elif kind == "product":
        dataset = _generate_product_graph(name, title, note, **arguments)
    elif kind == "coauthor":
        dataset = _generate_coauthor_graph()
    else:
        dataset = _generate_large_community_graph(name, title, note, **arguments)
    dataset["name"] = title
    dataset["description"] = note
    dataset["synthetic"] = True
    dataset["source"] = "generated"
    return dataset


_DATASETS = {
    # name: (loader, nodes, edges, classes, description)
    "karate_club": (load_karate_club, 34, 78, 2, "Zachary's Karate Club social network"),
    "dolphins": (load_dolphins, 62, 159, 0, "Bottlenose dolphins social network (no labels)"),
    "les_miserables": (load_les_miserables, 77, 254, 0, "Les Miserables character co-appearances (weighted, no labels)"),
    "football": (load_football, 115, 613, 12, "American college football games, 12 conferences"),
    "email_eu_core": (load_email_eu_core, 986, 16064, 42, "E-mail network of a research institution (SNAP), 42 departments"),
    "cora": (load_cora, 2708, 5278, 7, "Cora citation network (Planetoid)"),
    "citeseer": (load_citeseer, 3279, 4552, 6, "CiteSeer citation network (Planetoid)"),
    "pubmed": (load_pubmed, 19717, 44324, 3, "PubMed diabetes citation network (Planetoid)"),
    "ppi": (load_ppi, 3860, 37845, 50, "Protein-protein interaction network (node2vec), multi-label"),
    "wikipedia_pos": (load_wikipedia_pos, 4777, 92295, 40, "Wikipedia word co-occurrence network (node2vec), multi-label"),
    "amazon_photo": (load_amazon_photo, 7535, 119081, 8, "Amazon co-purchase graph (photo)"),
    "amazon_computers": (load_amazon_computers, 13471, 245861, 10, "Amazon co-purchase graph (computers)"),
    "ppi_large": (load_ppi_large, 56688, 793632, 121, "PPI graphs of 24 tissues (GraphSAGE), multi-label"),
    "ogbn_arxiv": (load_ogbn_arxiv, 169343, 1166243, 40, "OGB arxiv citation network"),
    "facebook": (load_facebook, 4039, 88234, 0, "Facebook ego networks (SNAP, ~4k nodes, ~88k edges)"),
    "reddit_hyperlink": (load_reddit_hyperlink, 55863, 858490, 0, "Reddit hyperlink network (SNAP, ~55K subreddits, ~858K edges)"),
    "roadnet": (load_roadnet, 1965206, 5533214, 0, "California road network (SNAP, ~2M nodes, ~5.5M edge lines)"),
    "ogbn_products": (load_ogbn_products, 2449029, 61859140, 47, "OGB products co-purchasing graph (2.4M nodes, 62M edges, 47 classes)"),
    "ogbl_citation2": (load_ogbl_citation2, 2927963, 30561187, 0, "OGB citation2 graph (2.9M nodes, 30M edges, link prediction)"),
    "com_orkut": (load_com_orkut, 3072441, 117185083, 0, "Orkut online social network (SNAP, ~3M nodes, ~117M edges)"),
    "livejournal": (load_livejournal, 4847571, 68993773, 0, "LiveJournal social network (SNAP, ~4.8M nodes, ~69M edges)"),
    "twitter": (load_twitter, 41652230, 1468365182, 0, "Twitter-2010 follower network (~41.7M nodes, ~1.47B edges)"),
    "com_friendster": (load_com_friendster, 65608366, 1806067135, 0, "Friendster online social network (SNAP, ~65.6M nodes, ~1.8B edges)"),
}

# Names that pycleora <= 3.2 served with a synthetic stand-in and for which no
# stable public download of the real data exists.
_UNAVAILABLE = {
    "flickr": "the GraphSAINT Flickr graph",
    "yelp": "the GraphSAINT Yelp graph",
    "dblp": "the DBLP heterogeneous graph",
    "reddit": "the GraphSAGE Reddit graph",
}


def list_datasets() -> List[Dict]:
    """The real datasets that can be loaded with :func:`load_dataset`."""
    return [
        {"name": name, "nodes": nodes, "edges": edges, "classes": classes,
         "description": description, "synthetic": False}
        for name, (_, nodes, edges, classes, description) in _DATASETS.items()
    ]


def load_dataset(name: str) -> Dict:
    """Loads a real dataset by name (see :func:`list_datasets`)."""
    if name in _DATASETS:
        return _DATASETS[name][0]()
    if name in _UNAVAILABLE:
        raise ValueError(
            f"'{name}' is not available: pycleora cannot download {_UNAVAILABLE[name]}. "
            f"Earlier versions returned a synthetic random graph under this name; "
            f"if that is what you want, call load_synthetic('{name}')."
        )
    available = ", ".join(_DATASETS)
    raise ValueError(f"Unknown dataset: '{name}'. Available: {available}")
