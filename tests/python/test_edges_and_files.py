"""Every way of feeding the same graph gives the same matrix."""

import pickle

import numpy as np
import pytest

from pycleora import SparseMatrix
from pycleora.io_utils import (
    from_arrays, from_edge_list, from_networkx, from_numpy, from_pandas, from_scipy_sparse,
    load_embeddings, save_embeddings, to_edge_list, to_networkx,
)

from conftest import dense, lines_from_rows, random_rows


def same(a: SparseMatrix, b: SparseMatrix) -> bool:
    return (
        a.entity_ids == b.entity_ids
        and all(np.array_equal(x, y) for x, y in zip(a.to_csr("raw"), b.to_csr("raw")))
        and np.array_equal(a.entity_degrees, b.entity_degrees)
    )


def test_integer_edge_arrays_equal_lines(rng):
    src = rng.integers(-50, 300, size=2000)
    dst = rng.integers(-50, 300, size=2000)
    lines = [f"{s} {d}" for s, d in zip(src, dst)]
    assert same(from_arrays(src, dst), SparseMatrix.from_iterator(iter(lines), "complex::reflexive::node"))
    tab_lines = [f"{s}\t{d}" for s, d in zip(src, dst)]
    assert same(from_arrays(src, dst, columns="user item"), SparseMatrix.from_iterator(iter(tab_lines), "user item"))


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.uint16, np.uint64, np.int64])
def test_integer_dtypes(dtype):
    src = np.array([1, 2, 3, 1], dtype=dtype)
    dst = np.array([2, 3, 4, 4], dtype=dtype)
    graph = from_arrays(src, dst)
    assert graph.entity_ids == ("1", "2", "3", "4")
    assert same(graph, SparseMatrix.from_iterator(iter(["1 2", "2 3", "3 4", "1 4"]), "complex::reflexive::node"))


def test_string_edge_arrays_equal_lines(rng):
    src = [f"a{int(i)}" for i in rng.integers(0, 80, size=500)]
    dst = [f"a{int(i)}" for i in rng.integers(0, 80, size=500)]
    lines = [f"{s} {d}" for s, d in zip(src, dst)]
    reference = SparseMatrix.from_iterator(iter(lines), "complex::reflexive::node")
    assert same(from_arrays(src, dst), reference)
    assert same(from_arrays(np.array(src), np.array(dst)), reference)
    assert same(from_arrays(np.array(src, dtype=object), np.array(dst, dtype=object)), reference)
    assert same(SparseMatrix.from_edges(src, dst), reference)


def test_names_in_edge_arrays_may_contain_any_character():
    graph = from_arrays(["New York", "a,b"], ["Los Angeles", "tab\there"], columns="city other")
    assert graph.entity_ids == ("New York", "Los Angeles", "a,b", "tab\there")
    assert graph.get_entity_index("tab\there") == 3


def test_non_contiguous_edge_arrays():
    src = np.arange(20, dtype=np.int64)[::2]
    dst = np.arange(20, dtype=np.int64)[1::2]
    assert same(from_arrays(src, dst), from_arrays(src.copy(), dst.copy()))
    assert same(SparseMatrix.from_edges(src, dst), from_arrays(src.copy(), dst.copy()))


@pytest.mark.parametrize("num_workers", [1, 2, 6])
def test_files_equal_lines_whatever_the_number_of_workers(tmp_path, rng, num_workers):
    rows = random_rows(rng, 3000, [True], vocabulary=400, max_len=30)
    lines = lines_from_rows(rows)
    paths = []
    for part in range(5):
        path = tmp_path / f"part{part}.txt"
        chunk = lines[part * 600:(part + 1) * 600]
        # mixed line endings, blank lines and a missing final newline
        text = ("\r\n" if part % 2 else "\n").join(chunk)
        path.write_text(text + ("" if part == 4 else "\n\n"), newline="")
        paths.append(str(path))
    from_files = SparseMatrix.from_files(paths, "complex::reflexive::x", 16, num_workers)
    from_lines = SparseMatrix.from_iterator(iter(lines), "complex::reflexive::x", 16, 1)
    assert same(from_files, from_lines)


def test_large_file_is_read_in_batches(tmp_path):
    # > 1 MB, so the reader has to split the file into several batches
    lines = [f"node_{i} node_{(i * 7919) % 50000} node_{(i * 104729) % 50000}" for i in range(60000)]
    path = tmp_path / "big.txt"
    path.write_text("\n".join(lines) + "\n")
    assert path.stat().st_size > (1 << 20)
    assert same(
        SparseMatrix.from_files([str(path)], "complex::reflexive::x", 16, 3),
        SparseMatrix.from_iterator(iter(lines), "complex::reflexive::x", 16, 2),
    )


def test_any_file_extension_is_accepted(tmp_path):
    path = tmp_path / "graph.edgelist"
    path.write_text("a b\nb c\n")
    assert SparseMatrix.from_files([str(path)], "complex::reflexive::x").entity_ids == ("a", "b", "c")


def test_pandas_scipy_numpy_networkx_agree():
    pd = pytest.importorskip("pandas")
    nx = pytest.importorskip("networkx")
    from scipy.sparse import csr_matrix

    pairs = [(0, 1), (1, 2), (0, 3), (3, 4), (2, 4)]
    reference = from_arrays([a for a, _ in pairs], [b for _, b in pairs])

    frame = pd.DataFrame({"s": [a for a, _ in pairs], "t": [b for _, b in pairs], "extra": [0.5] * len(pairs)})
    assert same(from_pandas(frame, "s", "t"), reference)

    adjacency = np.zeros((5, 5))
    for a, b in pairs:
        adjacency[a, b] = adjacency[b, a] = 1.0
    G = nx.Graph()
    G.add_edges_from(pairs)
    # these emit the edges in their own order, so compare after sorting the entities
    ref_order = np.argsort(reference.entity_ids)
    expected = dense(reference)[np.ix_(ref_order, ref_order)]
    for graph in (from_numpy(adjacency), from_scipy_sparse(csr_matrix(adjacency)), from_networkx(G)):
        order = np.argsort(graph.entity_ids)
        assert sorted(graph.entity_ids) == sorted(reference.entity_ids)
        np.testing.assert_array_equal(dense(graph)[np.ix_(order, order)], expected)

    assert same(from_edge_list(pairs), reference)

    exported = to_networkx(reference)
    assert set(map(frozenset, exported.edges())) >= {frozenset((str(a), str(b))) for a, b in pairs}
    assert {frozenset((a, b)) for a, b, _ in to_edge_list(reference) if a != b} == {frozenset((str(a), str(b))) for a, b in pairs}


def test_pandas_missing_values_and_integer_ids():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"s": [1, 2, None, 3], "t": [2, 3, 4, None], "w": [1.0, 2.0, 3.0, 4.0]})
    graph = from_pandas(frame.astype({"s": "Int64", "t": "Int64"}), "s", "t")
    assert graph.entity_ids == ("1", "2", "3")
    with pytest.raises(ValueError, match="not found"):
        from_pandas(frame, "s", "missing")
    empty = pd.DataFrame({"s": [None], "t": [None]})
    with pytest.raises(ValueError, match="No valid edges"):
        from_pandas(empty, "s", "t")


def test_embeddings_round_trip(tmp_path, small_graph):
    graph, _ = small_graph
    embeddings = graph.embed_fast(8, 2)
    for fmt in ("npz", "csv", "tsv"):
        path = str(tmp_path / f"emb.{fmt}")
        save_embeddings(graph, embeddings, path, format=fmt)
        loaded, ids = load_embeddings(path, format=fmt)
        assert list(ids) == list(graph.entity_ids)
        np.testing.assert_allclose(loaded, embeddings, atol=1e-6)


def test_pickle_round_trip(small_graph):
    graph, _ = small_graph
    restored = pickle.loads(pickle.dumps(graph))
    assert same(graph, restored)
    assert repr(restored) == repr(graph)
    np.testing.assert_array_equal(restored.embed_fast(8, 2), graph.embed_fast(8, 2))
    assert restored.get_entity_index(graph.entity_ids[5]) == 5


def test_pickles_written_by_3_2_still_load():
    import base64
    import json
    import pathlib

    fixture = json.loads((pathlib.Path(__file__).parent / "data" / "legacy_3_2_1.json").read_text())
    for case in fixture.values():
        graph = pickle.loads(base64.b64decode(case["pickle"]))
        assert list(graph.entity_ids) == case["entity_ids"]
        rows, cols, left, n, _ = graph.to_sparse_csr("left")
        assert rows.tolist() == case["rows"] and cols.tolist() == case["cols"]
        np.testing.assert_allclose(left, case["left"], rtol=1e-5)
        np.testing.assert_allclose(graph.to_sparse_csr("symmetric")[2], case["symmetric"], rtol=1e-5)
        np.testing.assert_allclose(graph.entity_degrees, case["degrees"], rtol=1e-6)
        # and it describes the same graph as building it again from the lines
        rebuilt = SparseMatrix.from_iterator(iter(case["lines"]), case["columns"])
        assert rebuilt.entity_ids == graph.entity_ids
        np.testing.assert_allclose(dense(rebuilt, "left"), dense(graph, "left"), rtol=1e-5)
        assert rebuilt.column_names == graph.column_names


def test_corrupt_pickle_state_is_rejected():
    graph = SparseMatrix.from_iterator(iter(["a b"]), "complex::reflexive::x")
    with pytest.raises(RuntimeError, match="Deserialization failed"):
        graph.__setstate__(b"PYCLEORA\x02 definitely not a matrix")
    with pytest.raises(RuntimeError, match="Deserialization failed"):
        graph.__setstate__(b"\x00\x01\x02")
