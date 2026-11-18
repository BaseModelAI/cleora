import numpy as np
import time
import tracemalloc
from typing import Dict, List, Optional, Callable


class _MemoryTracker:
    """Peak additional memory of a block of code, in MB.

    On Linux this is the peak resident set size of the process, which includes
    native (Rust, BLAS) allocations. Elsewhere it falls back to ``tracemalloc``,
    which only sees allocations made through Python and therefore under-reports
    native code; ``method`` tells which one was used.
    """

    def __init__(self):
        self.method = "rss" if self._reset_peak() else "tracemalloc"
        self.peak_mb = 0.0

    @staticmethod
    def _reset_peak() -> bool:
        try:
            with open("/proc/self/clear_refs", "w") as f:
                f.write("5")
            return True
        except OSError:
            return False

    @staticmethod
    def _status_kb(field: str) -> Optional[int]:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith(field + ":"):
                        return int(line.split()[1])
        except OSError:
            pass
        return None

    def __enter__(self):
        if self.method == "rss":
            self._reset_peak()
            self._start_kb = self._status_kb("VmRSS")
            if self._start_kb is None:
                self.method = "tracemalloc"
        if self.method == "tracemalloc":
            tracemalloc.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.method == "rss":
            peak_kb = self._status_kb("VmHWM")
            if peak_kb is not None:
                self.peak_mb = max(peak_kb - self._start_kb, 0) / 1024
        else:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.peak_mb = peak / 1024 / 1024
        return False


def benchmark_algorithms(
    graph,
    labels: Dict[str, int],
    algorithms: Dict[str, Callable],
    metrics_fn: Optional[Callable] = None,
    num_runs: int = 1,
    seed: int = 42,
) -> Dict:
    from .metrics import node_classification_scores

    if metrics_fn is None:
        metrics_fn = lambda g, emb, lbls: node_classification_scores(g, emb, lbls, seed=seed)

    results = {}
    for name, algo_fn in algorithms.items():
        times = []
        scores_list = []
        memory_peaks = []

        memory_method = None
        for run in range(num_runs):
            try:
                with _MemoryTracker() as tracker:
                    t0 = time.perf_counter()
                    emb = algo_fn(graph)
                    elapsed = time.perf_counter() - t0
                memory_method = tracker.method

                scores = metrics_fn(graph, emb, labels)
                times.append(elapsed)
                scores_list.append(scores)
                memory_peaks.append(tracker.peak_mb)
            except Exception as e:
                results[name] = {"error": f"{type(e).__name__}: {e}"}
                break

        if name not in results:
            avg_scores = {}
            if scores_list:
                for key in scores_list[0]:
                    vals = [s[key] for s in scores_list if isinstance(s.get(key), (int, float))]
                    if vals:
                        avg_scores[key] = float(np.mean(vals))

            results[name] = {
                "avg_time": float(np.mean(times)),
                "std_time": float(np.std(times)) if len(times) > 1 else 0.0,
                "avg_memory_mb": float(np.mean(memory_peaks)),
                "memory_method": memory_method,
                "scores": avg_scores,
                "num_runs": num_runs,
            }

    return results


def benchmark_datasets(
    dataset_names: List[str],
    embed_fn: Callable,
    feature_dim: int = 256,
    seed: int = 42,
) -> Dict:
    from .datasets import load_dataset
    from .metrics import node_classification_scores
    from .pycleora import SparseMatrix

    results = {}
    for ds_name in dataset_names:
        try:
            ds = load_dataset(ds_name)
            t0 = time.perf_counter()
            graph = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])
            emb = embed_fn(graph)
            elapsed = time.perf_counter() - t0

            scores = node_classification_scores(graph, emb, ds["labels"], seed=seed)
            results[ds_name] = {
                "num_nodes": ds["num_nodes"],
                "num_edges": ds["num_edges"],
                "num_classes": ds["num_classes"],
                "time": elapsed,
                "scores": scores,
            }
        except Exception as e:
            results[ds_name] = {"error": str(e)}

    return results


def format_benchmark_table(results: Dict, metric: str = "accuracy") -> str:
    lines = []
    header = f"{'Algorithm':<15} {'Time (s)':<12} {'Memory (MB)':<14} {metric.capitalize():<12}"
    lines.append(header)
    lines.append("-" * len(header))

    for name, data in sorted(results.items()):
        if "error" in data:
            lines.append(f"{name:<15} ERROR: {data['error']}")
        else:
            t = data.get("avg_time", 0)
            m = data.get("avg_memory_mb", 0)
            s = data.get("scores", {}).get(metric, 0)
            lines.append(f"{name:<15} {t:<12.4f} {m:<14.2f} {s:<12.4f}")

    return "\n".join(lines)


def format_dataset_table(results: Dict, metric: str = "accuracy") -> str:
    lines = []
    header = f"{'Dataset':<20} {'Nodes':<8} {'Edges':<10} {'Time (s)':<12} {metric.capitalize():<12}"
    lines.append(header)
    lines.append("-" * len(header))

    for name, data in sorted(results.items()):
        if "error" in data:
            lines.append(f"{name:<20} ERROR: {data['error']}")
        else:
            n = data.get("num_nodes", 0)
            e = data.get("num_edges", 0)
            t = data.get("time", 0)
            s = data.get("scores", {}).get(metric, 0)
            lines.append(f"{name:<20} {n:<8} {e:<10} {t:<12.4f} {s:<12.4f}")

    return "\n".join(lines)
