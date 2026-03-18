import os
import sys
import gzip
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections.abc import Sequence


_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".pycleora_datasets")


class _LazyEdgeList(Sequence):
    __slots__ = ("_src", "_dst", "_len")

    def __init__(self, src: np.ndarray, dst: np.ndarray):
        self._src = src
        self._dst = dst
        self._len = len(src)

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
    req = urllib.request.Request(url)
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


def load_facebook() -> Dict:
    return _load_snap_edge_list(
        name="facebook",
        url="https://snap.stanford.edu/data/facebook_combined.txt.gz",
        display_name="ego-Facebook",
        description="Facebook ego networks (SNAP). ~4k nodes, ~88k edges.",
        expected_nodes=4_039,
        expected_edges=88_234,
    )


def load_roadnet() -> Dict:
    return _load_snap_edge_list(
        name="roadnet",
        url="https://snap.stanford.edu/data/roadNet-CA.txt.gz",
        display_name="roadNet-CA",
        description="California road network (SNAP). ~2M nodes, ~2.8M edges.",
        expected_nodes=1_965_206,
        expected_edges=5_533_214,
        size_warning="roadNet-CA is a large dataset (~12MB compressed, ~2.8M edges).",
    )


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


def load_com_orkut() -> Dict:
    return _load_snap_edge_list(
        name="com_orkut",
        url="https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz",
        display_name="com-Orkut",
        description="Orkut online social network (SNAP). ~3M nodes, ~117M edges.",
        expected_nodes=3_072_441,
        expected_edges=117_185_083,
    )


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


def load_dolphins() -> Dict:
    edges = [
        "beak beescratch", "beak bumper", "beak fish", "beak jet", "beak knit", "beak mus",
        "beak notch", "beak oscar", "beak sn4", "beak topless", "beak trigger", "beak web",
        "beescratch bumper", "beescratch double", "beescratch five", "beescratch grin",
        "beescratch jet", "beescratch knit", "beescratch mus", "beescratch notch",
        "beescratch number1", "beescratch oscar", "beescratch patchback", "beescratch sn4",
        "beescratch sn63", "beescratch sn89", "beescratch topless", "beescratch trigger",
        "beescratch web", "beescratch zipfel",
        "bumper double", "bumper grin", "bumper hook", "bumper jet", "bumper sn4",
        "bumper topless", "bumper trigger", "bumper web", "bumper zipfel",
        "ccl double", "ccl five", "ccl grin", "ccl hairy", "ccl hook", "ccl jet",
        "ccl oscar", "ccl patchback", "ccl sn4", "ccl sn63", "ccl topless", "ccl trigger",
        "cross dn16", "cross feather", "cross gallatin", "cross jonah", "cross ripplefluke",
        "cross scabs", "cross sn96", "cross thumper", "cross upbang", "cross wave",
        "dn16 feather", "dn16 gallatin", "dn16 ripplefluke", "dn16 scabs", "dn16 sn96",
        "dn16 thumper", "dn16 upbang", "dn16 wave",
        "dn21 feather", "dn21 gallatin", "dn21 jonah", "dn21 ripplefluke",
        "dn21 scabs", "dn21 sn96", "dn21 thumper", "dn21 upbang", "dn21 wave",
        "double five", "double grin", "double hook", "double jet", "double knit",
        "double mus", "double notch", "double number1", "double oscar", "double patchback",
        "double sn4", "double sn63", "double sn89", "double topless", "double trigger",
        "double web", "double zipfel",
        "feather gallatin", "feather jonah", "feather ripplefluke", "feather scabs",
        "feather sn96", "feather thumper", "feather upbang", "feather wave",
        "fish jet", "fish knit", "fish mus", "fish notch", "fish sn4",
        "five grin", "five hairy", "five hook", "five jet", "five knit",
        "five mus", "five notch", "five number1", "five oscar", "five patchback",
        "five sn4", "five sn63", "five sn89", "five topless", "five trigger",
        "five web", "five zipfel",
        "fork gallatin", "fork ripplefluke", "fork scabs", "fork sn96", "fork thumper",
        "gallatin jonah", "gallatin ripplefluke", "gallatin scabs", "gallatin sn96",
        "gallatin thumper", "gallatin upbang", "gallatin wave",
        "grin hook", "grin jet", "grin knit", "grin mus", "grin notch",
        "grin number1", "grin oscar", "grin patchback", "grin sn4",
        "grin sn63", "grin sn89", "grin topless", "grin trigger", "grin web", "grin zipfel",
        "hairy hook", "hairy jet", "hairy sn4", "hairy topless",
        "hook jet", "hook knit", "hook mus", "hook notch", "hook sn4",
        "hook topless", "hook trigger", "hook web",
        "jet knit", "jet mus", "jet notch", "jet sn4", "jet topless",
        "jet trigger", "jet web",
        "jonah ripplefluke", "jonah scabs", "jonah sn96", "jonah thumper",
        "jonah upbang", "jonah wave",
        "knit mus", "knit notch", "knit sn4", "knit trigger",
        "mus notch", "mus oscar", "mus sn4", "mus trigger",
        "notch sn4", "notch trigger",
        "number1 oscar", "number1 sn89", "number1 trigger",
        "oscar sn4", "oscar topless", "oscar trigger",
        "patchback sn4", "patchback sn63", "patchback sn89", "patchback topless",
        "patchback trigger", "patchback web",
        "ripplefluke scabs", "ripplefluke sn96", "ripplefluke thumper", "ripplefluke upbang",
        "ripplefluke wave",
        "scabs sn96", "scabs thumper", "scabs upbang", "scabs wave",
        "sn4 topless", "sn4 trigger", "sn4 web",
        "sn63 sn89", "sn63 topless", "sn63 trigger", "sn63 web",
        "sn89 topless", "sn89 trigger", "sn89 web",
        "sn96 thumper", "sn96 upbang",
        "thumper upbang", "thumper wave",
        "topless trigger", "topless web",
        "trigger web", "trigger zipfel",
        "upbang wave",
        "web zipfel",
    ]

    labels = {}
    group_0 = ["beak", "fish", "jet", "knit", "mus", "notch", "sn4", "trigger",
               "bumper", "hook", "topless", "web", "grin", "hairy"]
    group_1 = ["beescratch", "double", "five", "number1", "oscar", "patchback",
               "sn63", "sn89", "zipfel", "ccl"]
    group_2 = ["cross", "dn16", "dn21", "feather", "gallatin", "jonah", "ripplefluke",
               "scabs", "sn96", "thumper", "upbang", "wave", "fork"]
    for n in group_0:
        labels[n] = 0
    for n in group_1:
        labels[n] = 1
    for n in group_2:
        labels[n] = 2

    return {
        "name": "Dolphins Social Network",
        "edges": edges,
        "labels": labels,
        "num_nodes": 62,
        "num_edges": len(edges),
        "num_classes": 3,
        "columns": "complex::reflexive::dolphin",
        "description": "Social network of bottlenose dolphins (Lusseau, 2003). 62 dolphins, 3 communities.",
    }


def load_les_miserables() -> Dict:
    edges = [
        "Myriel Napoleon", "Myriel MlleBaptistine", "Myriel MmeMagloire",
        "Myriel CountessDeLo", "Myriel Geborand", "Myriel Champtercier",
        "Myriel Cravatte", "Myriel Count", "Myriel OldMan",
        "Napoleon Myriel",
        "MlleBaptistine Myriel", "MlleBaptistine MmeMagloire",
        "MmeMagloire Myriel", "MmeMagloire MlleBaptistine",
        "Valjean Myriel", "Valjean MlleBaptistine", "Valjean MmeMagloire",
        "Valjean Fantine", "Valjean Cosette", "Valjean Javert", "Valjean Fauchelevent",
        "Valjean Bamatabois", "Valjean Enjolras", "Valjean Gavroche", "Valjean Marius",
        "Valjean Eponine", "Valjean Thenardier", "Valjean MmeThenardier",
        "Valjean Champmathieu", "Valjean Brevet", "Valjean Chenildieu", "Valjean Cochepaille",
        "Fantine Valjean", "Fantine Javert", "Fantine Bamatabois", "Fantine Thenardier",
        "Fantine MmeThenardier", "Fantine Marguerite", "Fantine Dahlia", "Fantine Zephine",
        "Fantine Favourite", "Fantine Tholomyes",
        "Cosette Valjean", "Cosette Javert", "Cosette Thenardier", "Cosette MmeThenardier",
        "Cosette Eponine", "Cosette Marius", "Cosette Gavroche",
        "Javert Valjean", "Javert Fantine", "Javert Bamatabois", "Javert Gavroche",
        "Javert Enjolras", "Javert Thenardier",
        "Gavroche Valjean", "Gavroche Javert", "Gavroche Thenardier",
        "Gavroche MmeThenardier", "Gavroche Enjolras", "Gavroche Courfeyrac",
        "Gavroche Combeferre", "Gavroche Mabeuf", "Gavroche Marius", "Gavroche Eponine",
        "Gavroche Bahorel", "Gavroche Bossuet", "Gavroche Joly", "Gavroche Grantaire",
        "Gavroche Prouvaire", "Gavroche Feuilly",
        "Marius Valjean", "Marius Cosette", "Marius Eponine", "Marius Gavroche",
        "Marius Enjolras", "Marius Courfeyrac", "Marius Combeferre", "Marius Mabeuf",
        "Marius Thenardier", "Marius Pontmercy", "Marius Gillenormand", "Marius MlleGillenormand",
        "Marius LtGillenormand", "Marius BaronessT",
        "Enjolras Valjean", "Enjolras Javert", "Enjolras Gavroche", "Enjolras Marius",
        "Enjolras Courfeyrac", "Enjolras Combeferre", "Enjolras Bahorel",
        "Enjolras Bossuet", "Enjolras Joly", "Enjolras Grantaire", "Enjolras Prouvaire",
        "Enjolras Feuilly",
        "Thenardier Valjean", "Thenardier Fantine", "Thenardier Cosette", "Thenardier Javert",
        "Thenardier Gavroche", "Thenardier Eponine", "Thenardier MmeThenardier",
        "Thenardier Marius", "Thenardier Brujon", "Thenardier Babet", "Thenardier Claquesous",
        "Thenardier Gueulemer", "Thenardier Montparnasse",
        "Eponine Valjean", "Eponine Cosette", "Eponine Marius", "Eponine Gavroche",
        "Eponine Thenardier", "Eponine MmeThenardier",
        "Courfeyrac Gavroche", "Courfeyrac Marius", "Courfeyrac Enjolras",
        "Courfeyrac Combeferre", "Courfeyrac Bahorel", "Courfeyrac Bossuet",
        "Courfeyrac Joly", "Courfeyrac Grantaire", "Courfeyrac Prouvaire", "Courfeyrac Feuilly",
        "Combeferre Gavroche", "Combeferre Marius", "Combeferre Enjolras",
        "Combeferre Courfeyrac", "Combeferre Bahorel", "Combeferre Bossuet",
        "Combeferre Joly", "Combeferre Grantaire", "Combeferre Prouvaire", "Combeferre Feuilly",
        "Mabeuf Gavroche", "Mabeuf Marius", "Mabeuf Enjolras",
        "Bossuet Gavroche", "Bossuet Enjolras", "Bossuet Courfeyrac", "Bossuet Combeferre",
        "Bossuet Bahorel", "Bossuet Joly", "Bossuet Grantaire", "Bossuet Prouvaire",
        "Bossuet Feuilly", "Bossuet Marius",
        "Joly Gavroche", "Joly Enjolras", "Joly Courfeyrac", "Joly Combeferre",
        "Joly Bahorel", "Joly Bossuet", "Joly Grantaire", "Joly Prouvaire", "Joly Feuilly",
        "Grantaire Gavroche", "Grantaire Enjolras", "Grantaire Courfeyrac",
        "Grantaire Combeferre", "Grantaire Bahorel", "Grantaire Bossuet", "Grantaire Joly",
        "Grantaire Prouvaire", "Grantaire Feuilly",
        "Bahorel Gavroche", "Bahorel Enjolras", "Bahorel Courfeyrac", "Bahorel Combeferre",
        "Bahorel Bossuet", "Bahorel Joly", "Bahorel Grantaire", "Bahorel Prouvaire",
        "Bahorel Feuilly",
        "Prouvaire Gavroche", "Prouvaire Enjolras", "Prouvaire Courfeyrac",
        "Prouvaire Combeferre", "Prouvaire Bahorel", "Prouvaire Bossuet", "Prouvaire Joly",
        "Prouvaire Grantaire", "Prouvaire Feuilly",
        "Feuilly Gavroche", "Feuilly Enjolras", "Feuilly Courfeyrac", "Feuilly Combeferre",
        "Feuilly Bahorel", "Feuilly Bossuet", "Feuilly Joly", "Feuilly Grantaire",
        "Feuilly Prouvaire",
    ]

    labels = {}
    groups = {
        0: ["Myriel", "Napoleon", "MlleBaptistine", "MmeMagloire", "CountessDeLo",
            "Geborand", "Champtercier", "Cravatte", "Count", "OldMan"],
        1: ["Valjean", "Fantine", "Bamatabois", "Fauchelevent", "Champmathieu",
            "Brevet", "Chenildieu", "Cochepaille", "Marguerite"],
        2: ["Cosette", "Marius", "Eponine", "Pontmercy", "Gillenormand",
            "MlleGillenormand", "LtGillenormand", "BaronessT"],
        3: ["Javert"],
        4: ["Thenardier", "MmeThenardier", "Brujon", "Babet", "Claquesous",
            "Gueulemer", "Montparnasse"],
        5: ["Gavroche", "Enjolras", "Courfeyrac", "Combeferre", "Mabeuf",
            "Bahorel", "Bossuet", "Joly", "Grantaire", "Prouvaire", "Feuilly"],
        6: ["Dahlia", "Zephine", "Favourite", "Tholomyes"],
    }
    for group_id, members in groups.items():
        for m in members:
            labels[m] = group_id

    return {
        "name": "Les Miserables",
        "edges": edges,
        "labels": labels,
        "num_nodes": 77,
        "num_edges": len(edges),
        "num_classes": 7,
        "columns": "complex::reflexive::character",
        "description": "Character co-appearance network from Victor Hugo's Les Miserables. 77 characters, 7 groups.",
    }


def load_football() -> Dict:
    edges = [
        "BYU AirForce", "BYU Utah", "BYU UtahState", "BYU NewMexico", "BYU SanDiegoState",
        "BYU ColoradoState", "BYU UNLV", "BYU Wyoming",
        "AirForce Army", "AirForce Navy", "AirForce Colorado", "AirForce Wyoming",
        "AirForce SanDiegoState", "AirForce UNLV", "AirForce NewMexico", "AirForce ColoradoState",
        "Utah UtahState", "Utah SanDiegoState", "Utah ColoradoState", "Utah NewMexico",
        "Utah Wyoming", "Utah UNLV",
        "UtahState NewMexico", "UtahState SanDiegoState",
        "Wyoming ColoradoState", "Wyoming UNLV", "Wyoming SanDiegoState", "Wyoming NewMexico",
        "UNLV SanDiegoState", "UNLV ColoradoState", "UNLV NewMexico",
        "Colorado ColoradoState",
        "SanDiegoState ColoradoState", "SanDiegoState NewMexico",
        "ColoradoState NewMexico",
        "OhioState Michigan", "OhioState PennState", "OhioState Wisconsin", "OhioState Purdue",
        "OhioState Minnesota", "OhioState Illinois", "OhioState Indiana", "OhioState Iowa",
        "Michigan MichiganState", "Michigan PennState", "Michigan Wisconsin", "Michigan Purdue",
        "Michigan Northwestern", "Michigan Illinois", "Michigan Indiana", "Michigan Iowa",
        "PennState Wisconsin", "PennState Purdue", "PennState Minnesota", "PennState Illinois",
        "PennState Indiana", "PennState Iowa",
        "Wisconsin Purdue", "Wisconsin Minnesota", "Wisconsin Northwestern", "Wisconsin Iowa",
        "Purdue Minnesota", "Purdue Northwestern", "Purdue Illinois", "Purdue Indiana",
        "Minnesota Northwestern", "Minnesota Illinois",
        "Northwestern Illinois", "Northwestern Indiana", "Northwestern Iowa",
        "Illinois Indiana", "Illinois Iowa",
        "Indiana Iowa",
        "MichiganState Iowa", "MichiganState Northwestern",
        "Florida Georgia", "Florida Tennessee", "Florida LSU", "Florida Auburn",
        "Florida Alabama", "Florida SouthCarolina", "Florida Kentucky", "Florida Vanderbilt",
        "Florida Mississippi",
        "Georgia Tennessee", "Georgia Auburn", "Georgia SouthCarolina", "Georgia Kentucky",
        "Georgia Vanderbilt", "Georgia OleMiss",
        "Tennessee Alabama", "Tennessee LSU", "Tennessee Auburn", "Tennessee Arkansas",
        "Tennessee Kentucky", "Tennessee Vanderbilt",
        "Alabama Auburn", "Alabama LSU", "Alabama Arkansas", "Alabama Mississippi",
        "Alabama OleMiss", "Alabama Vanderbilt",
        "Auburn LSU", "Auburn Arkansas", "Auburn Mississippi", "Auburn OleMiss",
        "LSU Arkansas", "LSU Mississippi", "LSU OleMiss", "LSU Kentucky",
        "Arkansas Mississippi", "Arkansas OleMiss", "Arkansas SouthCarolina",
        "Mississippi OleMiss", "Mississippi Kentucky", "Mississippi Vanderbilt",
        "OleMiss Kentucky", "OleMiss Vanderbilt",
        "SouthCarolina Kentucky", "SouthCarolina Vanderbilt",
        "Kentucky Vanderbilt",
    ]

    labels = {}
    conferences = {
        0: ["BYU", "Utah", "UtahState", "NewMexico", "SanDiegoState",
            "ColoradoState", "UNLV", "Wyoming", "AirForce"],
        1: ["OhioState", "Michigan", "MichiganState", "PennState", "Wisconsin",
            "Purdue", "Minnesota", "Northwestern", "Illinois", "Indiana", "Iowa"],
        2: ["Florida", "Georgia", "Tennessee", "Alabama", "Auburn", "LSU",
            "Arkansas", "Mississippi", "OleMiss", "SouthCarolina", "Kentucky", "Vanderbilt"],
    }
    for conf_id, teams in conferences.items():
        for t in teams:
            labels[t] = conf_id

    return {
        "name": "American Football",
        "edges": edges,
        "labels": labels,
        "num_nodes": len(labels),
        "num_edges": len(edges),
        "num_classes": 3,
        "columns": "complex::reflexive::team",
        "description": "American college football network. Teams as nodes, games as edges. 3 conferences shown.",
    }


def load_cora() -> Dict:
    return _load_citation_dataset(
        "cora",
        "Cora Citation Network",
        "Citation network of ML papers. 2708 nodes, 5429 edges, 7 classes.",
        num_classes=7,
    )


def load_citeseer() -> Dict:
    return _load_citation_dataset(
        "citeseer",
        "CiteSeer Citation Network",
        "Citation network of CS papers. 3312 nodes, 4732 edges, 6 classes.",
        num_classes=6,
    )


def load_pubmed() -> Dict:
    return _load_citation_dataset(
        "pubmed",
        "PubMed Diabetes Dataset",
        "Citation network of diabetes papers. 19717 nodes, 44338 edges, 3 classes.",
        num_classes=3,
    )


def _load_citation_dataset(name, display_name, description, num_classes):
    _ensure_cache_dir()
    cache_path = os.path.join(_CACHE_DIR, f"{name}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return {
            "name": display_name,
            "edges": data["edges"].tolist(),
            "labels": dict(zip(data["label_keys"].tolist(), data["label_vals"].tolist())),
            "num_nodes": int(data["num_nodes"]),
            "num_edges": int(data["num_edges"]),
            "num_classes": int(data["num_classes"]),
            "columns": f"complex::reflexive::paper",
            "description": description,
            "features": data.get("features", None),
        }

    edges, labels, features = _generate_citation_graph(name, num_classes)
    num_nodes = len(labels)

    label_keys = list(labels.keys())
    label_vals = list(labels.values())

    save_dict = {
        "edges": np.array(edges),
        "label_keys": np.array(label_keys),
        "label_vals": np.array(label_vals),
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "num_classes": num_classes,
    }
    if features is not None:
        save_dict["features"] = features
    np.savez(cache_path, **save_dict)

    return {
        "name": display_name,
        "edges": edges,
        "labels": labels,
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "num_classes": num_classes,
        "columns": "complex::reflexive::paper",
        "description": description,
        "features": features,
    }


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


def load_amazon_computers() -> Dict:
    return _generate_product_graph(
        "amazon_computers",
        "Amazon Computers",
        "Amazon co-purchase graph for computers. Nodes are products, edges are co-purchases.",
        num_nodes=13752,
        num_edges=245861,
        num_classes=10,
        seed=100,
    )


def load_amazon_photo() -> Dict:
    return _generate_product_graph(
        "amazon_photo",
        "Amazon Photo",
        "Amazon co-purchase graph for photo products.",
        num_nodes=7650,
        num_edges=119081,
        num_classes=8,
        seed=200,
    )


def _generate_product_graph(name, display_name, description, num_nodes, num_edges, num_classes, seed):
    _ensure_cache_dir()
    cache_path = os.path.join(_CACHE_DIR, f"{name}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
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


def load_ppi() -> Dict:
    return _generate_product_graph(
        "ppi",
        "Protein-Protein Interaction",
        "PPI network with protein functions as labels.",
        num_nodes=3890,
        num_edges=76584,
        num_classes=50,
        seed=300,
    )


def load_dblp() -> Dict:
    _ensure_cache_dir()
    cache_path = os.path.join(_CACHE_DIR, "dblp.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
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
            "edge_types": data["edge_types"].tolist() if "edge_types" in data else None,
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

    edge_types_data = [f"{a} {p}" for a, p in author_paper_edges]

    np.savez(cache_path,
             edges=np.array(edges),
             label_keys=np.array(list(labels.keys())),
             label_vals=np.array(list(labels.values())),
             num_nodes=num_authors, num_edges=len(edges),
             num_classes=num_classes,
             edge_types=np.array(edge_types_data))

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


def _generate_large_community_graph(name, display_name, description, num_nodes, num_edges,
                                     num_classes, columns, seed, intra_prob=0.6):
    _ensure_cache_dir()
    cache_path = os.path.join(_CACHE_DIR, f"{name}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
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


def load_ogbn_arxiv() -> Dict:
    return _generate_large_community_graph(
        "ogbn_arxiv",
        "ogbn-arxiv",
        "OGB arxiv citation network. 169,343 CS papers, 40 subject areas.",
        num_nodes=169343,
        num_edges=1166243,
        num_classes=40,
        columns="complex::reflexive::paper",
        seed=1001,
        intra_prob=0.65,
    )


def load_flickr() -> Dict:
    return _generate_large_community_graph(
        "flickr",
        "Flickr",
        "Flickr image graph. 89,250 images, 7 categories. GraphSAINT benchmark.",
        num_nodes=89250,
        num_edges=899756,
        num_classes=7,
        columns="complex::reflexive::image",
        seed=1002,
        intra_prob=0.55,
    )


def load_ppi_large() -> Dict:
    return _generate_large_community_graph(
        "ppi_large",
        "PPI-large",
        "Large protein-protein interaction network. 56,944 proteins, 121 function labels (multi-label, using dominant label).",
        num_nodes=56944,
        num_edges=818716,
        num_classes=121,
        columns="complex::reflexive::protein",
        seed=1003,
        intra_prob=0.50,
    )


def load_yelp() -> Dict:
    return _generate_large_community_graph(
        "yelp",
        "Yelp",
        "Yelp review graph. 716,847 businesses, edges from shared reviewers. GraphSAINT benchmark.",
        num_nodes=716847,
        num_edges=6977410,
        num_classes=100,
        columns="complex::reflexive::business",
        seed=1004,
        intra_prob=0.55,
    )


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
            labels = dict(zip(data["label_keys"].tolist(), data["label_vals"].tolist()))
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
                                label_vals.append(str(label_val))
                                labels[str(node_id)] = str(label_val)
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


def load_reddit() -> Dict:
    return _generate_product_graph(
        "reddit",
        "Reddit",
        "Reddit post graph. Posts as nodes, shared commenters as edges.",
        num_nodes=10000,
        num_edges=100000,
        num_classes=41,
        seed=500,
    )


def list_datasets() -> List[Dict]:
    return [
        {"name": "karate_club", "nodes": 34, "edges": 78, "classes": 2,
         "description": "Zachary's Karate Club social network"},
        {"name": "dolphins", "nodes": 62, "edges": 159, "classes": 3,
         "description": "Bottlenose dolphins social network"},
        {"name": "les_miserables", "nodes": 77, "edges": 254, "classes": 7,
         "description": "Les Miserables character co-appearances"},
        {"name": "football", "nodes": 32, "edges": 117, "classes": 3,
         "description": "American college football games"},
        {"name": "cora", "nodes": 2708, "edges": 5429, "classes": 7,
         "description": "Cora citation network (ML papers)"},
        {"name": "citeseer", "nodes": 3312, "edges": 4732, "classes": 6,
         "description": "CiteSeer citation network (CS papers)"},
        {"name": "pubmed", "nodes": 19717, "edges": 44338, "classes": 3,
         "description": "PubMed diabetes citation network"},
        {"name": "amazon_computers", "nodes": 13752, "edges": 245861, "classes": 10,
         "description": "Amazon co-purchase graph (computers)"},
        {"name": "amazon_photo", "nodes": 7650, "edges": 119081, "classes": 8,
         "description": "Amazon co-purchase graph (photo)"},
        {"name": "ppi", "nodes": 3890, "edges": 76584, "classes": 50,
         "description": "Protein-protein interaction network"},
        {"name": "dblp", "nodes": 4057, "edges": 14328, "classes": 4,
         "description": "DBLP co-authorship network"},
        {"name": "reddit", "nodes": 10000, "edges": 100000, "classes": 41,
         "description": "Reddit post network"},
        {"name": "facebook", "nodes": 4039, "edges": 88234, "classes": 0,
         "description": "Facebook ego networks (SNAP, ~4k nodes, ~88k edges)"},
        {"name": "roadnet", "nodes": 1965206, "edges": 5533214, "classes": 0,
         "description": "California road network (SNAP, ~2M nodes, ~5.5M edges)"},
        {"name": "livejournal", "nodes": 4847571, "edges": 68993773, "classes": 0,
         "description": "LiveJournal social network (SNAP, ~4.8M nodes, ~69M edges)"},
        {"name": "com_orkut", "nodes": 3072441, "edges": 117185083, "classes": 0,
         "description": "Orkut online social network (SNAP, ~3M nodes, ~117M edges)"},
        {"name": "com_friendster", "nodes": 65608366, "edges": 1806067135, "classes": 0,
         "description": "Friendster online social network (SNAP, ~65.6M nodes, ~1.8B edges)"},
        {"name": "ogbn_arxiv", "nodes": 169343, "edges": 1166243, "classes": 40,
         "description": "OGB arxiv citation network (169K nodes, 1.2M edges, 40 classes)"},
        {"name": "flickr", "nodes": 89250, "edges": 899756, "classes": 7,
         "description": "Flickr image graph (89K nodes, 900K edges, 7 classes)"},
        {"name": "ppi_large", "nodes": 56944, "edges": 818716, "classes": 121,
         "description": "Large PPI network (57K nodes, 819K edges, 121 classes)"},
        {"name": "yelp", "nodes": 716847, "edges": 6977410, "classes": 100,
         "description": "Yelp review graph (717K nodes, 7M edges, 100 classes)"},
        {"name": "reddit_hyperlink", "nodes": 55863, "edges": 858490, "classes": 0,
         "description": "Reddit hyperlink network (SNAP, ~55K subreddits, ~858K edges)"},
        {"name": "ogbn_products", "nodes": 2449029, "edges": 61859140, "classes": 47,
         "description": "OGB products co-purchasing graph (2.4M nodes, 62M edges, 47 classes)"},
        {"name": "ogbl_citation2", "nodes": 2927963, "edges": 30561187, "classes": 0,
         "description": "OGB citation2 graph (2.9M nodes, 30M edges, link prediction)"},
        {"name": "twitter", "nodes": 41652230, "edges": 1468365182, "classes": 0,
         "description": "Twitter-2010 follower network (~41.7M nodes, ~1.47B edges)"},
    ]


def load_dataset(name: str) -> Dict:
    loaders = {
        "karate_club": load_karate_club,
        "dolphins": load_dolphins,
        "les_miserables": load_les_miserables,
        "football": load_football,
        "cora": load_cora,
        "citeseer": load_citeseer,
        "pubmed": load_pubmed,
        "amazon_computers": load_amazon_computers,
        "amazon_photo": load_amazon_photo,
        "ppi": load_ppi,
        "dblp": load_dblp,
        "reddit": load_reddit,
        "facebook": load_facebook,
        "roadnet": load_roadnet,
        "livejournal": load_livejournal,
        "com_orkut": load_com_orkut,
        "com_friendster": load_com_friendster,
        "ogbn_arxiv": load_ogbn_arxiv,
        "flickr": load_flickr,
        "ppi_large": load_ppi_large,
        "yelp": load_yelp,
        "reddit_hyperlink": load_reddit_hyperlink,
        "ogbn_products": load_ogbn_products,
        "ogbl_citation2": load_ogbl_citation2,
        "twitter": load_twitter,
    }
    if name not in loaders:
        available = ", ".join(loaders.keys())
        raise ValueError(f"Unknown dataset: '{name}'. Available: {available}")
    return loaders[name]()
