"""Package metadata stays consistent."""

import pathlib
import re

import pytest

import pycleora

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_version_is_exposed():
    assert re.fullmatch(r"\d+\.\d+\.\d+([a-z0-9.+-]*)", pycleora.__version__)
    assert pycleora.pycleora.__version__ == pycleora.__version__


def test_installed_metadata_has_the_same_version():
    from importlib import metadata

    try:
        installed = metadata.version("pycleora")
    except metadata.PackageNotFoundError:
        pytest.skip("pycleora is not installed as a distribution")
    assert installed == pycleora.__version__


def test_version_matches_cargo_toml():
    cargo = ROOT / "Cargo.toml"
    if not cargo.exists():
        pytest.skip("not running from a source checkout")
    version = re.search(r'^version\s*=\s*"([^"]+)"', cargo.read_text(), flags=re.MULTILINE).group(1)
    assert pycleora.__version__ == version


def test_declared_dependencies_cover_the_imports():
    pyproject = ROOT / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("not running from a source checkout")
    text = pyproject.read_text()
    dependencies = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, flags=re.MULTILINE | re.DOTALL).group(1)
    # numpy is imported by `import pycleora`; scipy only by optional parts
    assert "numpy" in dependencies
    assert re.search(r'^scipy\s*=\s*\["scipy', text, flags=re.MULTILINE)
    assert 'requires-python = ">=3.10"' in text


def test_core_api_does_not_need_scipy():
    """`pip install pycleora` installs numpy only: the core must work without scipy."""
    import subprocess
    import sys

    program = """
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("scipy is blocked in this test")
sys.meta_path.insert(0, Block())
import numpy as np
import pycleora
from pycleora import SparseMatrix, embed
lines = [f"n{i % 30} n{(i * 3 + 1) % 30} n{(i * 7 + 2) % 30}" for i in range(90)]
columns = "complex::reflexive::node"
graph = SparseMatrix.from_iterator(iter(lines), columns)
emb = embed(graph, 8, 3)
classic = embed(graph, 8, 3, whiten=False)
pycleora.whiten_embeddings(classic)
pycleora.find_most_similar(graph, emb, "n0", top_k=2)
pycleora.predict_links(graph, emb, top_k=2)
pycleora.embed_weighted([(line, 2.0) for line in lines], columns, feature_dim=8, num_iterations=2)
pycleora.embed_multiscale(graph, 8, [1, 2])
g2, e2 = pycleora.embed_inductive(graph, emb, lines, ["n3 new"], columns)
assert e2.shape == (31, 8)
pycleora.embed_streaming([lines, ["n3 new"], ["new newer"]], columns, feature_dim=8, num_iterations=2, refresh_every=2)
from pycleora.io_utils import from_arrays, from_edge_list
from_arrays([1, 2], [2, 3]); from_edge_list([("a", "b")])
assert "scipy" not in sys.modules
print("ok")
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_public_api_is_importable():
    for name in (
        "SparseMatrix", "embed", "whiten_embeddings", "embed_weighted", "embed_directed", "embed_inductive",
        "embed_streaming", "embed_multiscale", "embed_with_attention", "predict_links", "find_most_similar",
        "CleoraEmbedder", "update_graph", "remove_edges", "cosine_similarity",
    ):
        assert hasattr(pycleora, name), name
    from pycleora import algorithms, datasets, io_utils  # noqa: F401
    from pycleora.io_utils import from_arrays, from_pandas, from_networkx, from_scipy_sparse  # noqa: F401
