"""What the tables and figures show, and under which names.

Concern of this module: the rows (datasets), the columns (methods and their variants)
and their labels. It holds no numbers and computes none.

A *method* of a table may stand for several configurations (*variants*); for every
dataset, task and probe the report shows the variant with the best validation score
(:func:`cleora_bench.report.select_variant`). The variants are:

- baselines: the benchmark dimensionality and the 128 dimensions their papers use;
- ``cleora``: the four ways of running the library - the graph as one reflexive column
  or as two columns, left or symmetric normalisation - each with the number of iterations
  the library suggests for the task (``TASK_PURPOSE``). Nothing is tuned per dataset
  beyond that choice;
- ``cleora_tuned``: additionally the iteration counts found by the sweep (chosen on
  validation, see ``select_iterations.py``).
"""

from . import configurations

MAIN_DIM = 1024
PAPER_DIM = 128

DATASET_LABELS = {
    "cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed", "cora_full": "Cora-Full", "dblp": "DBLP",
    "amazon_photo": "Amazon Photo", "amazon_computers": "Amazon Computers", "coauthor_cs": "Coauthor CS",
    "coauthor_physics": "Coauthor Physics", "ppi": "PPI", "wikipedia_pos": "Wikipedia POS",
    "blogcatalog": "BlogCatalog", "facebook": "Facebook (ego)", "flickr": "Flickr", "yelp": "Yelp", "reddit": "Reddit",
    "ogbn_arxiv": "ogbn-arxiv", "ogbn_products": "ogbn-products", "ogbl_collab": "ogbl-collab", "ogbl_ddi": "ogbl-ddi",
    "ogbl_ppa": "ogbl-ppa", "ogbl_citation2": "ogbl-citation2", "livejournal": "LiveJournal",
    "roadnet_ca": "roadNet-CA", "movielens_1m": "MovieLens-1M", "movielens_20m": "MovieLens-20M",
}
# order of the rows: by number of edges
DATASET_ORDER = (
    "citeseer", "cora", "ppi", "pubmed", "dblp", "cora_full", "facebook", "coauthor_cs", "amazon_photo",
    "wikipedia_pos", "amazon_computers", "coauthor_physics", "blogcatalog", "flickr", "movielens_1m", "ogbl_ddi",
    "ogbn_arxiv", "ogbl_collab", "roadnet_ca", "yelp", "reddit", "movielens_20m", "ogbl_ppa", "ogbl_citation2",
    "livejournal", "ogbn_products",
)

# what SparseMatrix.suggest_iterations is asked for, by task
TASK_PURPOSE = {"nc": "classification", "lp": "similarity"}


def cleora_variants(task):
    return [(name, MAIN_DIM) for name in configurations.automatic(TASK_PURPOSE[task])]


_BASELINES = (
    ("deepwalk", "DeepWalk"), ("node2vec", "node2vec"), ("walklets", "Walklets"), ("netmf", "NetMF"),
    ("grarep", "GraRep"), ("hope", "HOPE"), ("prone", "ProNE"), ("fastrp", "FastRP"), ("randne", "RandNE"),
)

# key -> label, variants ((configuration, dim), ...; None: they depend on the task and the
# dataset, see make_report.Results.variants), configurations drawn in the figures (the
# first one that has vectors: labelled graphs are drawn from the classification run)
METHODS = {
    "cleora": {"label": "Cleora", "variants": None,
               "figure": [configurations.AUTO[("hyperedge", "left", purpose)] for purpose in ("classification", "similarity")]},
    "cleora_tuned": {"label": "Cleora, tuned", "variants": None, "figure": []},
    "cleora_3_2": {"label": "pycleora 3.2 default", "variants": [("cleora_whiten40", 256), ("cleora_whiten40", MAIN_DIM)],
                   "figure": ["cleora_whiten40"]},
}
for _name, _label in _BASELINES:
    METHODS[_name] = {"label": _label, "variants": [(_name, MAIN_DIM), (_name, PAPER_DIM)], "figure": [_name]}

MAIN_COLUMNS = ("cleora", "cleora_tuned") + tuple(name for name, _ in _BASELINES)
BASELINES = tuple(name for name, _ in _BASELINES)

# the four ways of feeding and normalising a graph, for the ablation tables
FAMILIES = {
    "one column, left": ("hyperedge", "left"), "one column, symmetric": ("hyperedge", "symmetric"),
    "two columns, left": ("pair", "left"), "two columns, symmetric": ("pair", "symmetric"),
}

FIGURE_METHODS = [{"label": METHODS[key]["label"], "configurations": METHODS[key]["figure"]}
                  for key in ("cleora",) + BASELINES + ("cleora_3_2",)]

NODE_PROBES = {"logistic": "logistic regression", "svm": "linear SVM", "mlp": "MLP (AdamC)", "knn": "k-NN (cosine)"}
PAIR_PROBES = {
    "dot": "dot product", "cosine": "cosine", "bilinear": "bilinear x'My (AdamC)",
    "mlp": "concat -> MLP -> sigmoid (AdamC)", "logistic": "logistic regression on x*y",
    "svm": "linear SVM on x*y",
}
