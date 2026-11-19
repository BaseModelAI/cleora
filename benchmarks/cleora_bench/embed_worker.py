"""Runs one embedding method on one graph in this (fresh) process.

    python -m cleora_bench.embed_worker --method prone --graph graph.npz --dim 1024 \\
        --seed 0 --threads 32 --out vectors.npy --stats stats.json

``graph.npz`` holds ``edges`` (int64 [E, 2]), ``num_nodes`` and optionally ``weights``.
The statistics are the wall time of the method call (from edge arrays in memory to the
embedding matrix in memory) and the peak resident memory of the process - interpreter
and libraries included; ``rss_before_mb`` is the peak before the method was called.

Use :func:`run` to start it in the Python environment the method needs.
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "NUMBA_NUM_THREADS", "RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _peak_rss_mb():
    """Peak resident memory of this process.

    On Linux ``ru_maxrss`` survives ``exec``: a freshly started worker would report the
    memory of the process that launched it. ``VmHWM`` belongs to the current address
    space only, so that is what is read there."""
    if sys.platform.startswith("linux"):
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024  # kB
        raise RuntimeError("VmHWM not found in /proc/self/status")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1 << 20) if sys.platform == "darwin" else peak / 1024  # bytes on macOS, kB elsewhere


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--method", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--params", default="{}", help="JSON object of method parameters")
    parser.add_argument("--out", required=True)
    parser.add_argument("--stats", required=True)
    args = parser.parse_args()

    from cleora_bench import methods

    with np.load(args.graph) as archive:
        edges = archive["edges"]
        num_nodes = int(archive["num_nodes"])
        weights = archive["weights"] if "weights" in archive.files else None
    before = _peak_rss_mb()
    start = time.perf_counter()
    vectors = methods.embed(args.method, edges, num_nodes, args.dim, args.seed, args.threads, weights,
                            **json.loads(args.params))
    seconds = time.perf_counter() - start
    peak = _peak_rss_mb()
    if not np.isfinite(vectors).all():
        raise RuntimeError(f"{args.method} produced non-finite values")
    np.save(args.out, vectors)
    with open(args.stats, "w") as handle:
        json.dump({"seconds": seconds, "peak_rss_mb": peak, "rss_before_mb": before, "threads": args.threads,
                   **methods.RUN_INFO}, handle)


def run(method, graph_path, dim, seed, threads, out_path, stats_path, params=None, timeout=None):
    """Runs the worker in the environment of ``method``. Returns ``None`` on success or a
    short failure reason ("timeout", "out of memory", or the last line of the error)."""
    from cleora_bench import methods

    environment_name = methods.METHODS[method][1]
    environments = os.environ.get("CLEORA_BENCH_ENVS")
    python = os.path.join(environments, environment_name, "bin", "python") if environments else sys.executable
    environment = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for variable in THREAD_VARIABLES:
        environment[variable] = str(threads)
    command = [python, "-m", "cleora_bench.embed_worker", "--method", method, "--graph", graph_path,
               "--dim", str(dim), "--seed", str(seed), "--threads", str(threads),
               "--params", json.dumps(params or {}), "--out", out_path, "--stats", stats_path]
    try:
        completed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout"
    if completed.returncode == 0:
        return None
    if completed.returncode in (-9, 137) or "MemoryError" in completed.stderr:
        return "out of memory"
    lines = [line for line in completed.stderr.strip().splitlines() if line.strip()]
    return (lines[-1] if lines else f"exit code {completed.returncode}")[:300]


if __name__ == "__main__":
    main()
