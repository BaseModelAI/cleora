"""Bad input is reported, never silently turned into an empty or partial graph."""

import warnings

import numpy as np
import pytest

from pycleora import SparseMatrix
from pycleora.io_utils import from_arrays

from conftest import dense

COLUMNS = "complex::reflexive::node"


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="definitely_not_here"):
        SparseMatrix.from_files(["/tmp/definitely_not_here.tsv"], COLUMNS)


def test_no_files_and_compressed_files_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="At least one file"):
        SparseMatrix.from_files([], COLUMNS)
    archive = tmp_path / "edges.tsv.gz"
    archive.write_bytes(b"\x1f\x8b not really gzip")
    with pytest.raises(ValueError, match="Unsupported file format"):
        SparseMatrix.from_files([str(archive)], COLUMNS)


def test_malformed_lines_warn_with_the_first_offender(tmp_path):
    path = tmp_path / "edges.tsv"
    path.write_text("u1\tp1\nu2\tp2\nu3 p3\nu4\tp4\textra\n")
    with pytest.warns(RuntimeWarning, match=r"2 of 4 input line\(s\) were skipped.*line 3.*u3 p3"):
        graph = SparseMatrix.from_files([str(path)], "user product")
    assert set(graph.entity_ids) == {"u1", "p1", "u2", "p2"}


def test_nothing_usable_is_an_error():
    # two columns need a tab (or comma) between them; a space does not separate fields
    with pytest.raises(ValueError, match=r"None of the 2 input line\(s\) could be used.*expected 2 field"):
        SparseMatrix.from_iterator(iter(["u1 p1", "u2 p2"]), "user product")


def test_comma_separated_fields_work():
    by_comma = SparseMatrix.from_iterator(iter(["u1,p1 p2", "u2, p2"]), "user complex::product")
    by_tab = SparseMatrix.from_iterator(iter(["u1\tp1 p2", "u2\tp2"]), "user complex::product")
    assert by_comma.entity_ids == by_tab.entity_ids
    np.testing.assert_array_equal(dense(by_comma), dense(by_tab))


def test_blank_lines_and_stray_spaces_are_harmless():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        messy = SparseMatrix.from_iterator(iter(["a  b ", "", "   ", " b c\n", "c   d\r\n"]), COLUMNS)
        clean = SparseMatrix.from_iterator(iter(["a b", "b c", "c d"]), COLUMNS)
    assert messy.entity_ids == clean.entity_ids == ("a", "b", "c", "d")
    np.testing.assert_array_equal(dense(messy), dense(clean))


def test_empty_input_gives_an_empty_graph():
    graph = SparseMatrix.from_iterator(iter([]), COLUMNS)
    assert graph.num_entities == 0 and graph.num_edges == 0
    assert graph.embed_fast(8, 2).shape == (0, 8)


def test_non_string_lines_are_rejected():
    with pytest.raises(ValueError, match="must be strings"):
        SparseMatrix.from_iterator(iter(["a b", 5]), COLUMNS)


def test_exception_inside_the_iterator_propagates():
    def lines():
        yield "a b"
        raise KeyError("boom")

    with pytest.raises(KeyError, match="boom"):
        SparseMatrix.from_iterator(lines(), COLUMNS)


@pytest.mark.parametrize("columns,message", [
    ("node", "at least one relation"),
    ("", "empty"),
    ("reflexive::node", "REFLEXIVE"),
    ("a bogus::b", "Unrecognized"),
    ("a b weight::x weight::y", "At most one"),
])
def test_bad_column_definitions(columns, message):
    with pytest.raises(ValueError, match=message):
        SparseMatrix.from_iterator(iter(["a b"]), columns)


def test_bad_trim_value():
    with pytest.raises(ValueError, match="hyperedge_trim_n"):
        SparseMatrix.from_iterator(iter(["a b"]), COLUMNS, 0)


def test_invalid_utf8_lines_are_reported(tmp_path):
    path = tmp_path / "edges.txt"
    path.write_bytes(b"a b\n\xff\xfe broken\nb c\n")
    with pytest.warns(RuntimeWarning, match="not valid UTF-8"):
        graph = SparseMatrix.from_files([str(path)], COLUMNS)
    assert graph.entity_ids == ("a", "b", "c")


def test_direct_construction_is_refused():
    with pytest.raises(ValueError, match="cannot be constructed directly"):
        SparseMatrix("anything")


def test_edge_arrays_of_different_length():
    with pytest.raises(ValueError, match="src has 2 element"):
        from_arrays([1, 2], [1])
    with pytest.raises(ValueError, match="1-dimensional"):
        from_arrays(np.zeros((2, 2), dtype=np.int64), np.zeros((2, 2), dtype=np.int64))
    with pytest.raises(TypeError, match="int64 numpy arrays"):
        SparseMatrix.from_edges(np.array([1.5, 2.5]), np.array([1.5, 2.5]))
