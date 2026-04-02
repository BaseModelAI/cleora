import argparse
import sys
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        prog="pycleora",
        description="pycleora - Graph Embedding CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    embed_parser = subparsers.add_parser("embed", help="Generate graph embeddings")
    embed_parser.add_argument("--input", "-i", required=True, help="Input edge file (TSV/CSV/space-separated)")
    embed_parser.add_argument("--output", "-o", required=True, help="Output file (npz/csv/tsv)")
    embed_parser.add_argument("--dim", "-d", type=int, default=256, help="Embedding dimension (default: 256)")
    embed_parser.add_argument("--iterations", "-n", type=int, default=40, help="Number of iterations (default: 40)")
    embed_parser.add_argument("--propagation", "-p", choices=["left", "symmetric"], default="left")
    embed_parser.add_argument("--normalization", choices=["l2", "l1", "none"], default="l2")
    embed_parser.add_argument("--columns", "-c", default="complex::reflexive::node", help="Column definition")
    embed_parser.add_argument("--algorithm", "-a", default="cleora",
                              choices=["cleora", "prone", "randne", "hope", "netmf", "grarep", "deepwalk", "node2vec"])
    embed_parser.add_argument("--seed", type=int, default=0)
    embed_parser.add_argument("--verbose", "-v", action="store_true")

    info_parser = subparsers.add_parser("info", help="Show graph information")
    info_parser.add_argument("--input", "-i", required=True, help="Input edge file")
    info_parser.add_argument("--columns", "-c", default="complex::reflexive::node")

    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument("--dataset", "-d", default="karate_club", help="Dataset name")
    bench_parser.add_argument("--dim", type=int, default=256)

    similar_parser = subparsers.add_parser("similar", help="Find similar entities")
    similar_parser.add_argument("--input", "-i", required=True)
    similar_parser.add_argument("--columns", "-c", default="complex::reflexive::node")
    similar_parser.add_argument("--entity", "-e", required=True, help="Query entity")
    similar_parser.add_argument("--top-k", "-k", type=int, default=10)
    similar_parser.add_argument("--dim", "-d", type=int, default=256)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "embed":
        _cmd_embed(args)
    elif args.command == "info":
        _cmd_info(args)
    elif args.command == "benchmark":
        _cmd_benchmark(args)
    elif args.command == "similar":
        _cmd_similar(args)


def _read_edges(filepath):
    edges = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                edges.append(line)
    return edges


def _cmd_embed(args):
    from .pycleora import SparseMatrix
    from . import embed
    from .io_utils import save_embeddings
    from .algorithms import (embed_prone, embed_randne, embed_hope,
                             embed_netmf, embed_grarep, embed_deepwalk, embed_node2vec)

    if args.verbose:
        print(f"Reading edges from {args.input}...")

    edges = _read_edges(args.input)

    if args.verbose:
        print(f"  {len(edges)} edges loaded")
        print(f"Building graph (columns={args.columns})...")

    t0 = time.time()
    graph = SparseMatrix.from_iterator(iter(edges), args.columns)

    if args.verbose:
        print(f"  {graph.num_entities} entities, {graph.num_edges} edges ({time.time()-t0:.2f}s)")
        print(f"Generating {args.dim}-dim embeddings using {args.algorithm}...")

    t0 = time.time()
    algo_map = {
        "cleora": lambda: embed(graph, args.dim, args.iterations, args.propagation, args.normalization, args.seed),
        "prone": lambda: embed_prone(graph, args.dim, seed=args.seed),
        "randne": lambda: embed_randne(graph, args.dim, seed=args.seed),
        "hope": lambda: embed_hope(graph, args.dim),
        "netmf": lambda: embed_netmf(graph, args.dim),
        "grarep": lambda: embed_grarep(graph, args.dim),
        "deepwalk": lambda: embed_deepwalk(graph, args.dim, seed=args.seed),
        "node2vec": lambda: embed_node2vec(graph, args.dim, seed=args.seed),
    }

    emb = algo_map[args.algorithm]()

    if args.verbose:
        print(f"  Shape: {emb.shape} ({time.time()-t0:.2f}s)")
        print(f"Saving to {args.output}...")

    fmt = "npz"
    if args.output.endswith(".csv"):
        fmt = "csv"
    elif args.output.endswith(".tsv"):
        fmt = "tsv"

    save_embeddings(graph, emb, args.output, format=fmt)

    if args.verbose:
        print("Done!")
    else:
        print(f"{graph.num_entities} entities -> {emb.shape} saved to {args.output}")


def _cmd_info(args):
    from .pycleora import SparseMatrix

    edges = _read_edges(args.input)
    graph = SparseMatrix.from_iterator(iter(edges), args.columns)

    print(f"Graph: {graph.num_entities} entities, {graph.num_edges} edges")
    print(f"Columns: {args.columns}")

    degrees = graph.entity_degrees
    print(f"Degree stats: min={degrees.min():.0f}, max={degrees.max():.0f}, "
          f"mean={degrees.mean():.1f}, median={np.median(degrees):.1f}")


def _cmd_benchmark(args):
    from .datasets import load_dataset
    from .pycleora import SparseMatrix
    from . import embed
    from .algorithms import embed_prone, embed_randne, embed_deepwalk, embed_node2vec
    from .benchmark import benchmark_algorithms, format_benchmark_table

    ds = load_dataset(args.dataset)
    graph = SparseMatrix.from_iterator(iter(ds["edges"]), ds["columns"])

    algorithms = {
        "cleora": lambda g: embed(g, args.dim, 40),
        "prone": lambda g: embed_prone(g, args.dim),
        "randne": lambda g: embed_randne(g, args.dim),
        "deepwalk": lambda g: embed_deepwalk(g, args.dim),
        "node2vec": lambda g: embed_node2vec(g, args.dim),
    }

    print(f"Benchmarking on {ds['name']} ({ds['num_nodes']} nodes)...")
    results = benchmark_algorithms(graph, ds["labels"], algorithms)
    print(format_benchmark_table(results))


def _cmd_similar(args):
    from .pycleora import SparseMatrix
    from . import embed, find_most_similar

    edges = _read_edges(args.input)
    graph = SparseMatrix.from_iterator(iter(edges), args.columns)
    emb = embed(graph, args.dim)

    results = find_most_similar(graph, emb, args.entity, top_k=args.top_k)
    for r in results:
        print(f"  {r['entity_id']:<30s} similarity={r['similarity']:.4f}")


if __name__ == "__main__":
    main()
