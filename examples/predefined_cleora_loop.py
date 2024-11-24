import time

from pycleora import embed_using_baseline_cleora, SparseMatrix

start_time = time.time()
graph = SparseMatrix.from_files(["perf_inputs/0.tsv", "perf_inputs/1.tsv", "perf_inputs/2.tsv", "perf_inputs/3.tsv", "perf_inputs/4.tsv", "perf_inputs/5.tsv", "perf_inputs/6.tsv", "perf_inputs/7.tsv"], "complex::reflexive::name")
embeddings = embed_using_baseline_cleora(graph, 128, 3)
print(f"Took {time.time() - start_time} seconds ")
