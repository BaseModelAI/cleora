"""Named method configurations: what a row of a results table is.

Concern of this module: the mapping from a configuration name to an implementation in
:mod:`methods` and its parameters. Baselines run with the settings their authors
recommend (see the notes in :mod:`methods`); nothing here is tuned on test data.
"""

# name -> (method in methods.METHODS, parameters)
CONFIGURATIONS = {
    "deepwalk": ("deepwalk", {}),
    "node2vec": ("node2vec", {}),
    "prone": ("prone", {}),
    "netmf": ("netmf", {}),
    "fastrp": ("fastrp", {}),
    "randne": ("randne", {}),
    "grarep": ("grarep", {}),
    "hope": ("hope", {}),
    "walklets": ("walklets", {}),
    "cleora_whiten40": ("cleora_whiten40", {}),
}

# Cleora with a fixed number of iterations: the sweep behind the iteration heuristic and
# the "iterations chosen on validation" rows. A homogeneous graph can be given to Cleora
# in two ways, and both are benchmarked:
#
#   hyperedge  one column, "complex::reflexive::node": every edge is a two-node hyperedge.
#              The column is related to itself, so every node is also its own neighbour
#              (a lazy random walk).
#   pair       two columns, "src dst": the plain adjacency matrix, without self-loops.
SWEEP_ITERATIONS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
SETUPS = {"hyperedge": "complex::reflexive::node", "pair": "src dst"}
# What the README recommends on top of normalisation and iteration count when the vectors
# are used to find or rank links: the popularity prior of pycleora.scale_by_popularity, at
# the library's default exponent (written out here so that a result names what produced it).
SIMILARITY = {"popularity": 0.0625}
for _setup, _columns in SETUPS.items():
    for _propagation, _tag in (("left", "left"), ("symmetric", "sym")):
        for _iterations in SWEEP_ITERATIONS:
            # the hyperedge names predate the pair setup and stay as they were
            _name = f"cleora_{_tag}_k{_iterations}" if _setup == "hyperedge" else f"cleora_pair_{_tag}_k{_iterations}"
            _parameters = {"iterations": _iterations, "propagation": _propagation, "columns": _columns}
            CONFIGURATIONS[_name] = ("cleora_classic", _parameters)
            CONFIGURATIONS[_name + "_sim"] = ("cleora_classic", {**_parameters, **SIMILARITY})


# The library's own choice of the number of iterations (SparseMatrix.suggest_iterations):
# nothing is chosen per dataset. "general" is what embed() does by default; the other
# two are the library's suggestion when the use of the vectors is known.
PURPOSES = {"general": "auto", "similarity": "sim", "classification": "cls"}
AUTO = {}  # (setup, propagation, purpose) -> configuration name
for _setup, _columns in SETUPS.items():
    for _propagation, _tag in (("left", "left"), ("symmetric", "sym")):
        for _purpose, _suffix in PURPOSES.items():
            _name = f"cleora_{_tag}_{_suffix}" if _setup == "hyperedge" else f"cleora_pair_{_tag}_{_suffix}"
            CONFIGURATIONS[_name] = ("cleora_classic", {
                "iterations": "auto", "purpose": _purpose, "propagation": _propagation, "columns": _columns,
                **(SIMILARITY if _purpose == "similarity" else {})})
            AUTO[(_setup, _propagation, _purpose)] = _name


def automatic(purpose):
    """Names of the four automatic configurations (setup x normalisation) of a purpose."""
    return [name for (_, _, p), name in AUTO.items() if p == purpose]


def sweep(setup, propagation=None):
    """Names of the fixed-iteration Cleora configurations of one setup."""
    prefix = "cleora_" if setup == "hyperedge" else "cleora_pair_"
    tags = ("left", "sym") if propagation is None else (propagation,)
    return [f"{prefix}{tag}_k{k}" for tag in tags for k in SWEEP_ITERATIONS]


def for_purpose(name, purpose):
    """The fixed-iteration configuration ``name`` of the sweep as it is run for a purpose:
    with the extras of ``SIMILARITY`` when the vectors are used for similarity."""
    return name + "_sim" if purpose == "similarity" else name


def iterations(name):
    """The iteration count of a fixed-iteration configuration."""
    return CONFIGURATIONS[name][1]["iterations"]


CONFIGURATIONS["deepwalk_e5"] = ("deepwalk", {"epochs": 5})


DEFAULT = ("deepwalk", "node2vec", "prone", "netmf", "fastrp", "randne", "grarep", "hope", "walklets")


def resolve(name):
    if name not in CONFIGURATIONS:
        raise KeyError(f"Unknown configuration '{name}'. Known: {', '.join(sorted(CONFIGURATIONS))}")
    return CONFIGURATIONS[name]
