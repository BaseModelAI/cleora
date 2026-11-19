#!/usr/bin/env python
"""Times the two things pycleora does - building the graph and one embedding
iteration - on a synthetic power-law graph, with whatever pycleora is installed.

    python benchmarks/kernel_speed.py                       # 2^20 nodes, 12M edges, 1024 dimensions
    python benchmarks/kernel_speed.py --scale 18 --edges 2500000 --threads 10,4,1

It only uses calls that exist in pycleora 3.2 as well (``SparseMatrix.from_files``,
``embed_fast``), so the same script compares releases: run it once per installed version.

The graph is an R-MAT graph (Chakrabarti et al., 2004; a = 0.57, b = c = 0.19), whose
skewed degrees are what makes the propagation hard to parallelise. It is written to a
file as one two-node hyperedge per line and read with the column
``complex::reflexive::node``.

Reported, best of ``--repeats`` runs each:

* build           file to sparse matrix
* per iteration   (embed with ``--iterations`` iterations - embed with none) / iterations:
                  one propagation and row normalisation of the whole embedding matrix
"""

import argparse
import os
import tempfile
import time

import numpy as np

COLUMNS = "complex::reflexive::node"


def rmat_edges(scale, num_edges, seed=0, a=0.57, b=0.19, c=0.19):
    """Edges of an R-MAT graph with ``2 ** scale`` node ids; self-loops removed."""
    generator = np.random.default_rng(seed)
    source = np.zeros(num_edges, dtype=np.int64)
    target = np.zeros(num_edges, dtype=np.int64)
    for _ in range(scale):
        draw = generator.random(num_edges)
        source = (source << 1) | (draw >= a + b)
        target = (target << 1) | (((draw >= a) & (draw < a + b)) | (draw >= a + b + c))
    keep = source != target
    return source[keep], target[keep]


def write_edge_list(path, source, target, block=1_000_000):
    with open(path, "w") as handle:
        for start in range(0, len(source), block):
            lines = np.char.add(np.char.add(source[start:start + block].astype(str), " "),
                                target[start:start + block].astype(str))
            handle.write("\n".join(lines.tolist()) + "\n")


def best(function, repeats):
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        times.append(time.perf_counter() - started)
    return min(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scale", type=int, default=20, help="2 ** scale node ids (default 20)")
    parser.add_argument("--edges", type=int, default=12_000_000)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", default=None,
                        help="comma-separated thread counts (pycleora >= 4.0); default: the library's own default")
    args = parser.parse_args()

    import pycleora
    from pycleora import SparseMatrix

    version = getattr(pycleora, "__version__", "3.x")
    source, target = rmat_edges(args.scale, args.edges)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "rmat.tsv")
        write_edge_list(path, source, target)
        for threads in [None] if args.threads is None else [int(t) for t in args.threads.split(",")]:
            workers = {} if threads is None else {"num_workers": threads}
            graphs = []
            build = best(lambda: graphs.append(SparseMatrix.from_files([path], COLUMNS, **workers)), args.repeats)
            graph = graphs[-1]
            nothing = best(lambda: graph.embed_fast(args.dim, 0, **workers), args.repeats)
            rounds = best(lambda: graph.embed_fast(args.dim, args.iterations, **workers), args.repeats)
            print(f"pycleora {version}  threads={'default' if threads is None else threads:>7}  "
                  f"{graph.num_entities:,} nodes  {len(source):,} edges  dim {args.dim}  "
                  f"build {build:.2f} s  per iteration {(rounds - nothing) / args.iterations:.3f} s", flush=True)


if __name__ == "__main__":
    main()
