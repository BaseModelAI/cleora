"""pycleora in a forked child process, after the parent has used it."""

import multiprocessing
import sys

import numpy as np
import pytest

from pycleora import SparseMatrix, embed

LINES = [f"n{i % 80} n{(i * 7 + 1) % 80} n{(i * 13 + 2) % 80}" for i in range(400)]


def _work(connection):
    graph = SparseMatrix.from_iterator(iter(LINES), "complex::reflexive::node", num_workers=3)
    vectors = embed(graph, 16, 4, num_workers=3)
    connection.send(graph.left_markov_propagate(vectors, num_workers=2))
    connection.close()


@pytest.mark.skipif(sys.platform == "win32", reason="no fork on Windows")
@pytest.mark.filterwarnings("ignore:.*fork.*:DeprecationWarning")  # Python warns about forking with threads
def test_a_forked_child_can_use_the_library():
    """The worker threads of the parent do not exist in a forked child; it used to wait
    for them forever."""
    graph = SparseMatrix.from_iterator(iter(LINES), "complex::reflexive::node", num_workers=3)
    expected = graph.left_markov_propagate(embed(graph, 16, 4, num_workers=3), num_workers=2)

    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(target=_work, args=(sender,))
    child.start()
    sender.close()
    try:
        assert receiver.poll(60), "the forked child did not answer"
        np.testing.assert_array_equal(receiver.recv(), expected)
    finally:
        child.join(5)
        if child.is_alive():
            child.kill()
    assert child.exitcode == 0
