"""What the tables and figures show, and under which names.

Concern of this module: the rows (datasets), the columns (methods), the measurements
and their labels. It holds no numbers and computes none.

Every column of the main tables is one configuration at ``MAIN_DIM`` dimensions - the
same for every graph, never chosen per dataset:

- a baseline: its authors' settings;
- ``cleora`` / ``cleora_pair``: the graph as one reflexive column / as two columns, with
  the number of iterations the library suggests for the task (``TASK_PURPOSE``) and the
  normalisation the README recommends for it (``TASK_PROPAGATION``).

``cleora_tuned`` is not a column of the main tables: it is what choosing the iteration
count and the family on validation data adds, shown in tables of its own.
"""

from . import configurations

MAIN_DIM = 1024

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
# The normalisation by task, for every graph alike. It is the one with the better mean
# validation score over the graphs (table "families_validation"); test scores had no say.
TASK_PROPAGATION = {"nc": "left", "lp": "symmetric"}


def cleora_configuration(setup, task):
    """The one Cleora configuration of a setup ("hyperedge" or "pair") for a task."""
    return configurations.AUTO[(setup, TASK_PROPAGATION[task], TASK_PURPOSE[task])]


_BASELINES = (
    ("deepwalk", "DeepWalk"), ("node2vec", "node2vec"), ("walklets", "Walklets"), ("netmf", "NetMF"),
    ("grarep", "GraRep"), ("hope", "HOPE"), ("prone", "ProNE"), ("fastrp", "FastRP"), ("randne", "RandNE"),
)
BASELINES = tuple(name for name, _ in _BASELINES)

# key -> label, and for the methods whose configuration does not depend on the task, its name
METHODS = {
    "cleora": {"label": "Cleora", "setup": "hyperedge"},
    "cleora_pair": {"label": "Cleora, two columns", "setup": "pair"},
    "cleora_tuned": {"label": "Cleora, tuned"},
    "cleora_3_2": {"label": "pycleora 3.2 default", "configuration": "cleora_whiten40"},
}
for _name, _label in _BASELINES:
    METHODS[_name] = {"label": _label, "configuration": _name}

MAIN_COLUMNS = ("cleora", "cleora_pair") + BASELINES
# columns that are the same method fed differently: they are not ranked against each other
SAME_METHOD = ("cleora", "cleora_pair")

# the four ways of feeding and normalising a graph, for the ablation tables
FAMILIES = {
    "one column, left": ("hyperedge", "left"), "one column, symmetric": ("hyperedge", "symmetric"),
    "two columns, left": ("pair", "left"), "two columns, symmetric": ("pair", "symmetric"),
}

# panels of the figures: label and the configurations to look for (labelled graphs are drawn
# from the classification run, the others from the similarity run)
FIGURE_METHODS = [{"label": "Cleora", "configurations": [cleora_configuration("hyperedge", task) for task in ("nc", "lp")]}]
FIGURE_METHODS += [{"label": METHODS[name]["label"], "configurations": [name]} for name in BASELINES]
FIGURE_METHODS.append({"label": METHODS["cleora_3_2"]["label"], "configurations": ["cleora_whiten40"]})

NODE_PROBES = {"logistic": "logistic regression", "mlp": "MLP (AdamC)"}
PAIR_PROBES = {"dot": "dot product", "cosine": "cosine", "logistic": "logistic regression on x*y"}
# metrics of the retrieval among all nodes that get a table
RETRIEVAL_METRICS = {"recall@10": "recall@10", "recall@50": "recall@50", "ndcg@10": "nDCG@10", "map@10": "MAP@10",
                     "all_nodes_mrr": "MRR"}

# The uses of the vectors the report is organised by, and the measurement each is headlined
# with: (label, how it is measured, task, probe, metric). The probe "similarity" is the
# better of dot product and cosine on validation; a metric of None is accuracy on
# single-label graphs and micro-F1 on multi-label ones.
SCENARIOS = {
    "retrieval": ("Similarity search among all nodes", "dot product / cosine, recall@10", "lp", "similarity", "recall@10"),
    "nc": ("Node classification", "logistic regression, accuracy / micro-F1", "nc", "logistic", None),
    "lp": ("Link ranking against sampled candidates", "dot product / cosine, MRR", "lp", "similarity", "mrr"),
}
# the graphs the summary at the top of the README takes its time and its memory figure from:
# the largest ones on which the timing pass ran every method that scales
SUMMARY_TIME_GRAPH, SUMMARY_MEMORY_GRAPH = "ogbn_arxiv", "yelp"

# "Cleora, tuned" is chosen on what the sweep validates - the classification score and the
# MRR against sampled candidates - so these are the scenarios it is reported for
TUNED_SCENARIOS = ("nc", "lp")
