# The benchmark

How the numbers in the main [README](../README.md) were produced, and how to produce
them again. All result tables are in [RESULTS.md](RESULTS.md); both files are filled
by `make_report.py` from the job files, no number in them is typed by hand.

```bash
benchmarks/replicate.sh ~/cleora-bench            # everything: environments, data, runs, tables, figures
benchmarks/replicate.sh ~/cleora-bench --datasets cora,citeseer --no-timing   # a small part of it
```

- [What is measured](#what-is-measured)
- [Datasets](#datasets)
- [Methods](#methods)
- [Node classification](#node-classification)
- [Link ranking](#link-ranking)
- [One configuration per method](#one-configuration-per-method)
- [Number of iterations](#number-of-iterations)
- [Time and memory](#time-and-memory)
- [Figures](#figures)
- [Replication](#replication)
- [What the benchmark does not show](#what-the-benchmark-does-not-show)

## What is measured

Every method turns a graph into one vector per node, without seeing node labels or
node features. The vectors are then frozen and judged by what small supervised models
("probes") can do with them:

| Task | Graphs | Question |
|---|---|---|
| node classification | those with node labels | can a classifier predict the label of a node from its vector? |
| link ranking | all | does the true neighbour of a node score higher than non-neighbours? |

Not every graph has node labels: Facebook, MovieLens and the OGB link datasets take
part in link ranking only.

Every method produces **1024-dimensional** vectors, and every column of the main
tables is **one configuration** of a method, the same for every graph: nothing is
chosen per dataset, for Cleora or for the baselines.

## Datasets

All graphs are downloaded from their original public sources by
`prepare_datasets.py` (through PyTorch Geometric and OGB where those ship the
dataset) and stored as undirected, unweighted-or-weighted edge lists without
self-loops and duplicate edges. Nodes without any edge cannot be embedded from
structure and are removed for every method alike. The number of nodes and edges of
every prepared graph is checked against the numbers below; a source that starts
serving different data makes the preparation fail instead of silently changing the
benchmark.

<!-- benchmark:datasets -->
| Dataset | Nodes | Edges | Mean degree | Node labels | Seeds | Source |
|---|---:|---:|---:|---|---:|---|
| CiteSeer | 3,279 | 4,552 | 2.8 | 6 classes | 3 | Yang, Cohen, Salakhutdinov, ICML 2016; Sen et al., AI Magazine 2008 |
| Cora | 2,708 | 5,278 | 3.9 | 7 classes | 3 | Yang, Cohen, Salakhutdinov, ICML 2016; Sen et al., AI Magazine 2008 |
| PPI | 3,860 | 37,845 | 19.6 | 50 labels (multi-label) | 3 | Grover and Leskovec, node2vec, KDD 2016 |
| PubMed | 19,717 | 44,324 | 4.5 | 3 classes | 3 | Yang, Cohen, Salakhutdinov, ICML 2016; Sen et al., AI Magazine 2008 |
| DBLP | 17,716 | 52,867 | 6.0 | 4 classes | 3 | Bojchevski and Günnemann, ICLR 2018 |
| Cora-Full | 19,793 | 63,421 | 6.4 | 70 classes | 3 | Bojchevski and Günnemann, ICLR 2018 |
| Facebook (ego) | 4,039 | 88,234 | 43.7 | – | 3 | McAuley and Leskovec, NeurIPS 2012 |
| Coauthor CS | 18,333 | 81,894 | 8.9 | 15 classes | 3 | Shchur et al., Pitfalls of GNN Evaluation, 2018 |
| Amazon Photo | 7,535 | 119,081 | 31.6 | 8 classes | 3 | Shchur et al., Pitfalls of GNN Evaluation, 2018; McAuley et al., SIGIR 2015 |
| Wikipedia POS | 4,777 | 92,295 | 38.6 | 40 labels (multi-label) | 3 | Grover and Leskovec, node2vec, KDD 2016; weighted |
| Amazon Computers | 13,471 | 245,861 | 36.5 | 10 classes | 3 | Shchur et al., Pitfalls of GNN Evaluation, 2018; McAuley et al., SIGIR 2015 |
| Coauthor Physics | 34,493 | 247,962 | 14.4 | 5 classes | 3 | Shchur et al., Pitfalls of GNN Evaluation, 2018 |
| BlogCatalog | 10,312 | 333,983 | 64.8 | 39 labels (multi-label) | 3 | Tang and Liu, KDD 2009; Perozzi et al., DeepWalk, KDD 2014 |
| Flickr | 89,250 | 449,878 | 10.1 | 7 classes | 3 | Zeng et al., GraphSAINT, ICLR 2020; official node split |
| MovieLens-1M | 9,746 | 1,000,209 | 205.3 | – | 3 | Harper and Konstan, The MovieLens Datasets, ACM TiiS 2015; bipartite |
| ogbl-ddi | 4,267 | 1,067,911 | 500.5 | – | 3 | Hu et al., Open Graph Benchmark, NeurIPS 2020; official link split |
| ogbn-arxiv | 169,343 | 1,157,799 | 13.7 | 40 classes | 3 | Hu et al., Open Graph Benchmark, NeurIPS 2020; official node split |
| ogbl-collab | 235,868 | 967,632 | 8.2 | – | 3 | Hu et al., Open Graph Benchmark, NeurIPS 2020; official link split |
| roadNet-CA | 1,965,206 | 2,766,607 | 2.8 | – | 1 | Leskovec et al., Internet Mathematics 2009 |
| Yelp | 716,847 | 6,618,986 | 18.5 | 100 labels (multi-label) | 1 | Zeng et al., GraphSAINT, ICLR 2020; official node split |
| Reddit | 232,965 | 57,307,946 | 492.0 | 41 classes | 1 | Hamilton, Ying, Leskovec, GraphSAGE, NeurIPS 2017; official node split |
| MovieLens-20M | 165,237 | 20,000,263 | 242.1 | – | 1 | Harper and Konstan, The MovieLens Datasets, ACM TiiS 2015; bipartite |
| ogbl-ppa | 576,289 | 21,231,931 | 73.7 | – | 1 | Hu et al., Open Graph Benchmark, NeurIPS 2020; official link split |
| ogbl-citation2 | 2,926,620 | 30,351,880 | 20.7 | – | 1 | Hu et al., Open Graph Benchmark, NeurIPS 2020; official link split |
| LiveJournal | 3,997,962 | 34,681,189 | 17.3 | – | 1 | Yang and Leskovec, ICDM 2012 |
| ogbn-products | 2,400,608 | 61,859,012 | 51.5 | 47 classes | 1 | Hu et al., Open Graph Benchmark, NeurIPS 2020; official node split |
<!-- /benchmark:datasets -->

Graphs up to about a million edges are run with three seeds, larger ones with one.

## Methods

**Cleora** is `pycleora.embed(graph, 1024, k)` on a graph built with
`SparseMatrix.from_edges`. A homogeneous graph can be given to Cleora in two ways, and
both are benchmarked:

| Setup | Columns | Matrix |
|---|---|---|
| one column | `complex::reflexive::node` | every edge is a two-node hyperedge of a column related to itself: every node is also its own neighbour, the walk stays where it is half of the time |
| two columns | `src dst` | the plain adjacency matrix |

Each can be run with the left (`D⁻¹A`) and the symmetric (`D⁻½AD⁻½`) normalisation:
four *families*. The columns of the main tables:

- **Cleora** (one column) and **Cleora, two columns**: the settings the README
  recommends for the task, for every graph alike -
  `embed(graph, 1024, graph.suggest_iterations(propagation, purpose), propagation)`
  with `propagation="left"`, `purpose="classification"` for node classification and
  `propagation="symmetric"`, `purpose="similarity"` for link ranking. The two columns
  are the same method fed in two ways; they are shown side by side and are not ranked
  against each other.

Outside the main tables, in tables of their own:

- **Cleora, tuned**: the number of iterations (from a sweep over 1, 2, 3, 4, 6, 8, 12,
  16, 24, 32, 48, 64, `select_iterations.py`) and the family chosen on validation
  data - what a user with a validation set can add.
- **The four families**, each on its own.
- **pycleora 3.2 default**: 40 iterations with PCA whitening after each. This was the
  default of `embed()` until this release.

**Baselines** run their authors' code or a widely used library, with the settings the
authors recommend, at 1024 dimensions. None of them is re-implemented here;
`cleora_bench/methods.py` contains the exact calls.

| Method | Implementation | Settings |
|---|---|---|
| DeepWalk | [PecanPy](https://github.com/krishnanlab/PecanPy) 2.0.9 walks + gensim skip-gram | 10 walks of length 80 per node, window 10, negative sampling |
| node2vec | same, biased walks | p = 1, q = 0.5, otherwise as DeepWalk |
| Walklets | [karateclub](https://github.com/benedekrozemberczki/karateclub) 1.3.3 | 4 scales × dim/4 |
| NetMF | [authors' code](https://github.com/xptree/NetMF) (`77286b8`) | large-window variant, T = 10, b = 1, rank 1024 |
| GraRep | karateclub 1.3.3 | 4 orders × dim/4 |
| HOPE | karateclub 1.3.3 | common-neighbours proximity |
| ProNE | [authors' code](https://github.com/THUDM/ProNE) (`873630a`) | μ = 0.2, θ = 0.5, 10 Chebyshev steps |
| FastRP | [authors' code](https://github.com/GTmac/FastRP) (`3a6a71c`) | weights (0, 0, 1, 6.67), β = −0.64 |
| RandNE | [authors' code](https://github.com/ZW-ZHANG/RandNE) (`bf06bd7`) | transition matrix, q = 3, weights (1, 10², 10⁴, 10⁵) |

Three adaptations were unavoidable and do not change what is computed. NetMF imports
Theano, which has been unmaintained since 2020: its two element-wise expressions are
restated in NumPy. Its sparse eigensolver (ARPACK) gives up on graphs that fall into
many pieces, as the sparsest graphs do once a quarter of their edges is held out: the
same eigenpairs are then computed with a dense solver. FastRP's loop over the higher
powers multiplies sparse matrices that fill in completely: the same products are
computed densely.

NetMF, GraRep, HOPE and Walklets (in these implementations) build dense n × n matrices
or Python-level graph objects; they are run on graphs of up to 60,000 (NetMF) and
40,000 (the karateclub methods) nodes and marked `n/a (size)` beyond. Every run has a
time limit of six hours; a method that exceeds it, or the memory of the machine, is
marked as such.

On the five largest graphs (ogbn-products, ogbl-citation2, LiveJournal, roadNet-CA,
MovieLens-20M) the limit is three hours per run; Cleora needs one to three minutes
there. ProNE cannot produce 1024 dimensions for more than about two million nodes: its
randomised SVD hands SciPy's LU a matrix of more than 2³¹ elements, which is reported
as `out of memory`.

Every method runs in a fresh process with the same number of threads.

## Node classification

The vectors come from the whole graph. Labelled nodes are split into train, validation
and test: the official split where the dataset defines one, otherwise 10% / 10% / 80%
at random (stratified for single-label graphs), three splits per embedding. With three
seeds that makes nine evaluations per number.

Four probes, each fitted on train, tuned on validation and scored once on test:

| Probe | Model | Tuned on validation |
|---|---|---|
| logistic regression | multinomial (single-label) or one per label (multi-label), L-BFGS to convergence | L2 penalty from 10² … 10⁻⁸, feature scaling |
| linear SVM | squared hinge loss, one-vs-rest, L-BFGS to convergence | the same |
| MLP | one hidden layer of 512 units, dropout 0.5, 100 epochs of AdamC (warm-up, cosine decay) | weight decay 10⁻² or 10⁻¹, the epoch, feature scaling |
| k-NN | cosine vote of the k nearest training nodes; no training | k from 1, 5, 10, 20, 50 |

*Feature scaling* is one of: none, unit-norm rows, standardised columns. Embedding
methods differ in what the scale of their coordinates means - standardising helps
some and destroys the spectrum of an SVD-based embedding - so the scaling is chosen on
validation for every method like any other hyper-parameter of the probe.

Metrics (`pycleora.metrics`, checked against scikit-learn in the test suite):

- single-label: accuracy, balanced accuracy, macro-F1, weighted F1, Matthews
  correlation, Cohen's κ, top-3 and top-5 accuracy, macro one-vs-rest ROC-AUC and
  average precision;
- multi-label: micro-, macro- and sample-averaged F1 at a score threshold of zero,
  the same with the number of labels of every node given (the protocol of the
  DeepWalk and node2vec papers, `*_known_count`), subset accuracy, Hamming loss,
  micro and macro ROC-AUC and average precision, label-ranking average precision,
  precision@1.

## Link ranking

The edges are split at random into four disjoint sets:

| Set | Share | Used for |
|---|---:|---|
| embed | 75% | the only edges the embedding method sees |
| probe | 10% | positives for training the pair probes |
| validation | 5% | model selection |
| test | 10% | the reported numbers |

The probe positives are deliberately *not* in the embedded graph: a probe trained on
edges the vectors were fitted to learns to recognise memorised edges, not to predict
missing ones. On large graphs the three held-out sets are capped (500,000 / 20,000 /
50,000 edges). Datasets with an official link split (OGB) keep their official
positives and negatives; only the probe positives are taken out of the official
training edges.

Every validation or test edge (u, v) is ranked, in both directions, against 500 random
nodes that are not neighbours of the query node in the full graph. All methods and all
probes rank the same candidates.

| Probe | Score of a pair (x, y) | Training |
|---|---|---|
| dot product | xᵀy | none |
| cosine | xᵀy / (‖x‖‖y‖) | none |
| bilinear | xᵀMy, M initialised to the identity | AdamC, 30 epochs |
| MLP | concatenation → 512 hidden units → sigmoid | AdamC, 30 epochs |
| logistic regression | linear model on the element-wise product x∘y | L-BFGS |
| linear SVM | the same, squared hinge loss | L-BFGS |

The trained probes see the probe positives and three sampled negatives per positive
(the two linear probes at most one million of these pairs); their weight decay / L2 penalty
is chosen on validation MRR.

Metrics: MRR, mean and median rank, Hits@1/3/10/50/100, ROC-AUC and average precision
against the 500 sampled candidates. For the two training-free scores additionally
*retrieval among all nodes*: the held-out neighbours of 10,000 query nodes are ranked
among all nodes of the graph with the known neighbours filtered out, which is what a
nearest-neighbour index over the vectors does - recall@k, precision@k, hit rate@k,
nDCG@k and MAP@k for k = 10, 20, 50, and the MRR and Hits@k of the full ranking.

Edge weights (Wikipedia POS) are used for node classification and ignored in the link
task, whose split is over unweighted edges.

## One configuration per method

Every column of the main tables is a single configuration at 1024 dimensions, applied
unchanged to every graph. Validation data is used inside the probes (their
regularisation, feature scaling, number of neighbours) and nowhere else; test scores
are never used to choose anything.

Two decisions about Cleora were made once, for all graphs, and not on test scores:

- *The normalisation per task.* Left for classification, symmetric for link ranking:
  the one with the better mean validation score over the graphs.

<!-- benchmark:families_validation -->
| Task | Graphs | one column, left | one column, symmetric | two columns, left | two columns, symmetric |
|---|---:|---:|---:|---:|---:|
| node classification (logistic regression) | 17 | 0.6606 (4) | 0.6490 (7) | 0.6592 (2) | 0.6539 (4) |
| link ranking (cosine) | 26 | 0.4243 (3) | 0.4574 (21) | 0.3870 (1) | 0.4033 (1) |
<!-- /benchmark:families_validation -->

- *The number of iterations.* A rule on two statistics of the graph, fitted on
  validation curves ([next section](#number-of-iterations)).

For the training-free link scores a method is given the better of dot product and
cosine on validation; for Cleora, whose vectors have unit length, the two coincide.

A number is the mean over seeds and splits; ± is the standard deviation over them.
The best number of a row is bold, the second best underlined. *mean rank* is the
average rank of a method over the rows of a table (a method without a result ranks
after all that have one); *best on* counts the rows a method wins. Rows on which
only one method has a result are left out of both. The two Cleora columns are the
same method fed in two ways and do not compete with each other in these two lines:
every baseline is ranked against Cleora with one column, and each Cleora column
against the baselines only.

What choosing the number of iterations and the family on validation data adds is
reported separately:

<!-- benchmark:tuned_nc -->
| Dataset | Cleora | Cleora, tuned | gain | Best baseline |
|---|---:|---:|---:|---|
| CiteSeer | 0.566 | **0.579** | +1.3 | NetMF 0.572 |
| Cora | 0.785 | **0.791** | +0.6 | NetMF 0.766 |
| PubMed | 0.811 | **0.814** | +0.3 | NetMF 0.806 |
| DBLP | 0.808 | **0.817** | +0.9 | node2vec 0.803 |
| Cora-Full | 0.586 | **0.587** | +0.1 | Walklets 0.573 |
| Coauthor CS | 0.876 | **0.882** | +0.6 | node2vec 0.869 |
| Amazon Photo | 0.917 | **0.924** | +0.7 | RandNE 0.913 |
| Amazon Computers | 0.892 | **0.901** | +0.9 | FastRP 0.890 |
| Coauthor Physics | 0.928 | **0.938** | +1.0 | NetMF 0.932 |
| Flickr | 0.539 | **0.544** | +0.6 | ProNE 0.535 |
| ogbn-arxiv | 0.681 | **0.702** | +2.1 | node2vec 0.685 |
| Reddit | 0.941 | **0.945** | +0.4 | RandNE 0.940 |
| ogbn-products | **0.645** | 0.637 | -0.8 | RandNE 0.562 |
| PPI | 0.127 | **0.131** | +0.3 | RandNE 0.129 |
| Wikipedia POS | 0.383 | **0.428** | +4.5 | RandNE 0.409 |
| BlogCatalog | 0.259 | 0.299 | +4.0 | FastRP 0.332 |
| Yelp | 0.217 | 0.255 | +3.8 | ProNE 0.284 |
| *mean gain (points)* | | | +1.3 | |
<!-- /benchmark:tuned_nc -->

<!-- benchmark:tuned_lp -->
| Dataset | Cleora | Cleora, tuned | gain | Best baseline |
|---|---:|---:|---:|---|
| CiteSeer | 0.440 | **0.445** | +0.6 | NetMF 0.421 |
| Cora | 0.387 | **0.388** | +0.2 | NetMF 0.364 |
| PPI | 0.204 | **0.207** | +0.3 | RandNE 0.183 |
| PubMed | **0.428** | 0.427 | -0.0 | Walklets 0.420 |
| DBLP | 0.524 | **0.529** | +0.5 | RandNE 0.492 |
| Cora-Full | 0.643 | **0.649** | +0.6 | RandNE 0.614 |
| Facebook (ego) | 0.595 | **0.608** | +1.3 | ProNE 0.601 |
| Coauthor CS | 0.634 | **0.657** | +2.3 | RandNE 0.648 |
| Amazon Photo | 0.508 | **0.520** | +1.2 | HOPE 0.456 |
| Wikipedia POS | **0.251** | **0.251** | +0.0 | HOPE 0.242 |
| Amazon Computers | 0.483 | **0.493** | +1.0 | HOPE 0.421 |
| Coauthor Physics | 0.737 | **0.765** | +2.9 | RandNE 0.752 |
| BlogCatalog | **0.218** | **0.218** | +0.0 | HOPE 0.209 |
| Flickr | 0.283 | **0.297** | +1.4 | DeepWalk 0.244 |
| MovieLens-1M | 0.148 | 0.148 | +0.0 | HOPE 0.149 |
| ogbl-ddi | 0.027 | 0.027 | +0.0 | HOPE 0.083 |
| ogbn-arxiv | **0.671** | **0.671** | +0.0 | RandNE 0.606 |
| ogbl-collab | 0.057 | 0.055 | -0.2 | node2vec 0.099 |
| roadNet-CA | 0.752 | 0.752 | +0.0 | ProNE 0.778 |
| Yelp | 0.495 | 0.504 | +0.8 | RandNE 0.513 |
| Reddit | 0.605 | **0.645** | +4.1 | ProNE 0.626 |
| MovieLens-20M | 0.209 | 0.209 | +0.0 | ProNE 0.227 |
| ogbl-ppa | **0.069** | **0.069** | +0.0 | FastRP 0.034 |
| ogbl-citation2 | 0.802 | **0.838** | +3.6 | RandNE 0.829 |
| LiveJournal | 0.756 | **0.799** | +4.3 | RandNE 0.776 |
| ogbn-products | **0.909** | **0.909** | +0.0 | RandNE 0.906 |
| *mean gain (points)* | | | +1.0 | |
<!-- /benchmark:tuned_lp -->

## Number of iterations

Cleora has one hyper-parameter that matters: the number of iterations. Each iteration
moves every vector one step along a random walk. Too few and only the direct
neighbourhood is visible; too many and all vectors converge to the same direction.
The sweep (`run_benchmark.py --results sweep`) shows where the optimum is: between 1
and 64, later for classification than for link ranking, earlier on dense graphs, and
about twice as late in the one-column setup.

`SparseMatrix.suggest_iterations(propagation, purpose)` predicts it from two numbers
that are known as soon as the graph is built - `d`, the mean number of neighbours, and
`s`, the share of the weight on the diagonal of the matrix (0 for two columns, 1/2 for
one reflexive column):

```text
mixing = c(purpose) / ((1 - s) · √d)            × 1.5 for the symmetric normalisation
forget = ln(1/256) / ln(s)                       0 when s = 0
iterations = max(mixing, forget, 2)              at most 64
```

`c` is 12 for `purpose="similarity"`, 24 for `"classification"` and 16 for `"general"`
(what `num_iterations="auto"` uses when the purpose is not known).

- *mixing*: denser graphs mix faster. A walk that stays where it is with probability
  `s` needs `1 / (1 - s)` times the iterations for the same number of moves.
- *forget*: with a self-loop, every iteration keeps a share `s` of what the vector was,
  and what it was at the start is random. `forget` is the number of iterations after
  which that share has decayed to 1/256: 8 for the one-column setup, however dense
  the graph. (On Reddit, with 490 neighbours per node, the mixing term alone suggests
  one iteration; accuracy after 1, 2, 4 and 8 iterations is 0.15, 0.28, 0.87 and 0.94.)

How the form was chosen: candidate rules were fitted leave-one-graph-out on the
validation curves of the sweep, and judged by the test score they give up against the
iteration count that validated best.

<!-- benchmark:iteration_rule_fit -->
| Rule | Task | Graphs | Curves | Mean loss | Median loss | Worst loss |
|---|---|---:|---:|---:|---:|---:|
| one constant per setup and normalisation | classification | 16 | 64 | 1.36 | 0.46 | 9.18 |
| one constant per setup and normalisation | link ranking | 20 | 80 | 1.80 | 0.26 | 30.68 |
| mixing term only: c / ((1 - s) sqrt(d)) | classification | 16 | 64 | 2.12 | 0.20 | 31.08 |
| mixing term only: c / ((1 - s) sqrt(d)) | link ranking | 20 | 80 | 2.75 | 0.48 | 58.62 |
| mixing term and floors (the form that ships) | classification | 16 | 64 | 0.88 | 0.16 | 8.28 |
| mixing term and floors (the form that ships) | link ranking | 20 | 80 | 1.11 | 0.18 | 19.34 |

- *one constant per setup and normalisation*, fitted on all graphs: link ranking, hyperedge/left: 8; link ranking, hyperedge/sym: 8; link ranking, pair/left: 2; link ranking, pair/sym: 3; classification, hyperedge/left: 16; classification, hyperedge/sym: 32; classification, pair/left: 8; classification, pair/sym: 12
- *mixing term only: c / ((1 - s) sqrt(d))*, fitted on all graphs: c = 12 (link ranking), c = 40 (classification), symmetric x1.5
- *mixing term and floors (the form that ships)*, fitted on all graphs: c = 10 (link ranking), c = 40 (classification), symmetric x1.5, remainder 1/256, minimum 1
<!-- /benchmark:iteration_rule_fit -->

The optimum of `c` is flat, and it is not the same for the two tasks (loss in points,
all graphs of the sweep). The library uses 24 for classification and 12 for
similarity; 16, the general-purpose value, is the price of not knowing the task. A fit
on all graphs would take 40 and 10, which is within about a tenth of a point of 24 and
12, and a minimum of one iteration instead of two: Reddit fed as two columns ranks
links best after a single iteration (MRR 0.745, against 0.551 after two), while
classification on the same graph needs the second one (accuracy 0.82 after one, 0.94
after two). The library keeps the minimum of two.

<!-- benchmark:iteration_rule_constants -->
| c | classification: mean | median | worst | link ranking: mean | median | worst |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 7.21 | 5.77 | 31.07 | 1.37 | 0.35 | 18.76 |
| 8 | 5.17 | 4.00 | 20.66 | 0.66 | 0.15 | 5.41 |
| 10 | 3.66 | 2.79 | 20.66 | 0.61 | 0.13 | 4.86 |
| 12 | 2.93 | 1.92 | 20.66 | 0.70 | 0.12 | 4.86 |
| 14 | 2.20 | 1.39 | 20.66 | 0.96 | 0.41 | 8.19 |
| 16 | 1.87 | 0.99 | 20.66 | 1.40 | 0.60 | 8.19 |
| 20 | 1.41 | 0.65 | 20.66 | 2.32 | 1.54 | 11.03 |
| 24 | 0.96 | 0.41 | 12.07 | 3.39 | 2.60 | 14.16 |
| 28 | 0.98 | 0.35 | 12.07 | 4.34 | 3.59 | 16.98 |
| 32 | 0.94 | 0.26 | 12.07 | 5.23 | 4.70 | 19.59 |
| 40 | 0.84 | 0.16 | 8.28 | 7.20 | 6.33 | 23.28 |
| 48 | 1.03 | 0.17 | 9.05 | 8.67 | 8.20 | 27.56 |
<!-- /benchmark:iteration_rule_constants -->

What the rule that ships costs on the benchmark itself, for each of the four families
(three seeds, the number of iterations the library suggested against the one the
sweep chose):

<!-- benchmark:iterations_nc -->
| Dataset | one column, left (auto) | one column, left (tuned) | one column, left (loss) | one column, symmetric (auto) | one column, symmetric (tuned) | one column, symmetric (loss) | two columns, left (auto) | two columns, left (tuned) | two columns, left (loss) | two columns, symmetric (auto) | two columns, symmetric (tuned) | two columns, symmetric (loss) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CiteSeer | 29 | 64 | +1.3 | 43 | 64 | -0.5 | 14 | 48 | +1.6 | 22 | 64 | +1.7 |
| Cora | 24 | 48 | +0.2 | 36 | 48 | +0.0 | 12 | 12 | +0.0 | 18 | 32 | +0.0 |
| PPI | 11 | 12 | +0.3 | 16 | 12 | -0.1 | 5 | 4 | -0.2 | 8 | 6 | +0.3 |
| PubMed | 23 | 32 | +0.2 | 34 | 64 | +0.8 | 11 | 16 | +0.2 | 17 | 32 | +0.7 |
| DBLP | 20 | 32 | +0.8 | 29 | 64 | +1.6 | 10 | 16 | +0.8 | 15 | 48 | +1.1 |
| Cora-Full | 19 | 16 | +0.1 | 28 | 24 | -0.1 | 9 | 8 | -0.0 | 14 | 12 | +0.0 |
| Coauthor CS | 16 | 24 | +0.2 | 24 | 32 | +0.2 | 8 | 12 | +0.1 | 12 | 16 | +0.2 |
| Amazon Photo | 9 | 8 | -0.1 | 13 | 16 | +0.2 | 4 | 6 | +0.1 | 6 | 8 | +0.1 |
| Wikipedia POS | 8 | 48 | +0.2 | 12 | 32 | -0.1 | 4 | 3 | +1.8 | 6 | 3 | +0.9 |
| Amazon Computers | 8 | 12 | +0.0 | 12 | 24 | +0.3 | 4 | 4 | +0.0 | 6 | 8 | -0.0 |
| Coauthor Physics | 13 | 32 | +0.8 | 19 | 64 | +0.4 | 6 | 16 | +0.8 | 9 | 24 | +0.4 |
| BlogCatalog | 8 | 12 | +1.5 | 9 | 24 | +3.8 | 3 | 4 | +0.7 | 4 | 4 | +0.0 |
| Flickr | 15 | 32 | +0.4 | 23 | 16 | -0.4 | 8 | 16 | -0.0 | 11 | 8 | +0.0 |
| ogbn-arxiv | 13 | 24 | +0.7 | 19 | 48 | +2.7 | 6 | 12 | +0.8 | 10 | 24 | +1.2 |
| Yelp | 11 | 8 | +2.8 | 17 | 16 | +0.4 | 6 | 2 | +6.7 | 8 | 4 | +1.5 |
| Reddit | 8 | 12 | +0.1 | 8 | 24 | +3.2 | 2 | 3 | +0.4 | 2 | 8 | +2.5 |
| ogbn-products | 8 | – | – | 10 | – | – | 3 | – | – | 5 | – | – |
| *mean loss* |  |  | +0.6 |  |  | +0.8 |  |  | +0.9 |  |  | +0.7 |
<!-- /benchmark:iterations_nc -->

<!-- benchmark:iterations_lp -->
| Dataset | one column, left (auto) | one column, left (tuned) | one column, left (loss) | one column, symmetric (auto) | one column, symmetric (tuned) | one column, symmetric (loss) | two columns, left (auto) | two columns, left (tuned) | two columns, left (loss) | two columns, symmetric (auto) | two columns, symmetric (tuned) | two columns, symmetric (loss) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CiteSeer | 16 | 8 | +0.5 | 23/24 | 8 | +0.6 | 8 | 6 | -0.1 | 12 | 16 | -0.1 |
| Cora | 14 | 12 | +0.3 | 20 | 16 | +0.2 | 7 | 24 | -1.4 | 10 | 32 | -1.0 |
| PPI | 8 | 8 | +0.0 | 9 | 6 | +0.3 | 3 | 3 | +0.0 | 5 | 3 | +0.6 |
| PubMed | 12 | 12 | +0.0 | 18 | 16 | -0.0 | 6 | 8 | -0.1 | 9 | 16 | -0.2 |
| DBLP | 11 | 8 | +0.3 | 16 | 12 | +0.5 | 5 | 6 | -0.5 | 8 | 6 | +0.1 |
| Cora-Full | 11 | 8 | +0.5 | 16 | 12 | +0.6 | 5 | 4 | -0.2 | 8 | 6 | +0.2 |
| Facebook (ego) | 8 | 6 | +0.7 | 8 | 6 | +1.3 | 2 | 2 | +0.0 | 3 | 2 | +4.7 |
| Coauthor CS | 9 | 6 | +0.9 | 14 | 8 | +2.3 | 5 | 3 | +2.8 | 7 | 4 | +3.0 |
| Amazon Photo | 8 | 6 | +0.2 | 8 | 6 | +1.2 | 2 | 2 | +0.0 | 4 | 2 | +1.7 |
| Wikipedia POS | 8 | 12 | +0.7 | 8 | 8 | +0.0 | 2 | 2 | +0.0 | 3 | 3 | +0.0 |
| Amazon Computers | 8 | 6 | +0.1 | 8 | 6 | +1.0 | 2 | 2 | +0.0 | 3 | 2 | +0.5 |
| Coauthor Physics | 8 | 6 | +1.1 | 11 | 6 | +2.9 | 4 | 2 | +4.0 | 5 | 3 | +3.5 |
| BlogCatalog | 8 | 12 | -0.1 | 8 | 8 | +0.0 | 2 | 2 | +0.0 | 3 | 3 | +0.0 |
| Flickr | 9 | 16 | +0.9 | 13 | 16 | +1.4 | 4 | 8 | +1.1 | 7 | 8 | +0.3 |
| MovieLens-1M | 8 | 12 | +1.2 | 8 | 8 | +0.0 | 2 | 64 | +4.2 | 2 | 64 | +1.8 |
| ogbl-ddi | 8 | 12 | -0.0 | 8 | 8 | +0.0 | 2 | 2 | +0.0 | 2 | 2 | +0.0 |
| ogbn-arxiv | 8 | 8 | +0.0 | 11 | 12 | -0.3 | 4 | 3 | +0.1 | 5 | 4 | +0.1 |
| ogbl-collab | 9 | 16 | -0.0 | 13 | 2 | +0.3 | 4 | 8 | +0.0 | 7 | 2 | +0.1 |
| roadNet-CA | 15 | – | – | 23 | – | – | 8 | – | – | 11 | – | – |
| Yelp | 8 | 8 | +0.0 | 9 | 12 | +0.0 | 3 | 2 | +0.1 | 4 | 4 | +0.0 |
| Reddit | 8 | 8 | – | 8 | 8 | – | 2 | 1 | – | 2 | 1 | – |
| MovieLens-20M | 8 | – | – | 8 | – | – | 2 | – | – | 2 | – | – |
| ogbl-ppa | 8 | – | – | 8 | – | – | 2 | – | – | 2 | – | – |
| ogbl-citation2 | 8 | – | – | 8 | – | – | 3 | – | – | 4 | – | – |
| LiveJournal | 8 | – | – | 9 | – | – | 3 | – | – | 4 | – | – |
| ogbn-products | 8 | – | – | 8 | – | – | 2 | – | – | 3 | – | – |
| *mean loss* |  |  | +0.4 |  |  | +0.6 |  |  | +0.5 |  |  | +0.8 |
<!-- /benchmark:iterations_lp -->

The rule was derived on the graphs of the sweep, so for them the *Cleora* columns are
not a blind test of the rule: the leave-one-graph-out table above is the estimate of what it
costs on a new graph. ogbl-ppa and the five largest graphs (ogbn-products,
ogbl-citation2, LiveJournal, roadNet-CA, MovieLens-20M) were never swept; the rule was
fixed before they were run.

Limits of the rule, visible in these tables: it knows the graph but not the labels.
Where the labels follow a much finer or much coarser structure than usual (Yelp's 100
business categories; CiteSeer's six broad fields) the best number of iterations is
far from the suggestion, and a validation set is worth using: `embed(...,
callback=...)` hands out the vectors after every iteration, so one run covers all
candidates.

## Time and memory

Quality runs share a machine with other jobs, so their timings are not used. Time and
peak memory come from a separate pass (`--tasks time --results timing`): one method at
a time on an otherwise idle machine (Intel Core Ultra 9 275HX, 24 threads, 183 GB of
memory), whole graph, 1024 dimensions, three runs, median reported - a single run for
DeepWalk and node2vec on ogbn-arxiv and for the baselines on Yelp, where one run takes
up to 100 minutes. The clock runs from edge arrays in memory to the embedding matrix
in memory; memory is the peak resident set of the worker process, interpreter and
libraries included.

## Figures

`make_figures.py` draws a 2-D UMAP (umap-learn 0.6.0, cosine metric, 15 neighbours,
fixed seed) of the same random sample of up to 15,000 nodes for every method.
Points are coloured by node class - the eight largest classes, the others grey - or,
on graphs without labels, by degree. The pictures illustrate; whether one method
separates classes better than another is what the tables measure.

## Replication

```bash
benchmarks/replicate.sh <directory> [--tiers S|SM|SML] [--datasets a,b,c] [--dim 1024] [--threads N]
                        [--no-setup] [--no-baselines] [--no-previous] [--no-tuned]
                        [--no-timing] [--no-figures] [--no-report]
```

runs, in this order and skipping whatever is already there:

| Step | Script | Writes |
|---|---|---|
| environments and baseline checkouts | `setup.sh` | `envs/`, `baselines/` |
| datasets | `prepare_datasets.py` | `datasets/` |
| Cleora, baselines, pycleora 3.2 default | `run_benchmark.py` | `results/` |
| iteration sweep and selection | `run_benchmark.py --results sweep`, `select_iterations.py` | `sweep/`, `selected_iterations.json` |
| Cleora, tuned | `run_benchmark.py --methods cleora_tuned` | `results/` |
| time and memory | `run_benchmark.py --tasks time --results timing` | `timing/` |
| figures | `make_figures.py` | `benchmarks/figures/` |
| tables | `make_report.py` | `benchmarks/RESULTS.md`, `benchmarks/results.csv.gz` (every aggregated number), the READMEs |

Needs [uv](https://docs.astral.sh/uv/), git and a Rust toolchain. The probes use a CUDA
GPU when there is one (Apple MPS or the CPU otherwise).

Three Python environments are needed because the reference implementations pin
incompatible versions of NumPy:

| Environment | Python | Runs | Pinned in |
|---|---|---|---|
| `main` | 3.12 | pycleora, probes, report, figures, ProNE, NetMF, FastRP, RandNE | [`envs/main.txt`](envs/main.txt) (every transitive version: [`envs/main.lock-linux-x86_64-cuda13.txt`](envs/main.lock-linux-x86_64-cuda13.txt)) |
| `walks` | 3.11 | DeepWalk, node2vec | [`envs/walks.txt`](envs/walks.txt) |
| `karate` | 3.10 | GraRep, HOPE, Walklets | [`envs/karate.txt`](envs/karate.txt) |

Everything is seeded: the initial vectors of Cleora depend only on node names and the
seed, the splits and the sampled negatives only on the dataset and the seed. Cleora's
output is bit-identical between runs and thread counts. Skip-gram training with several
threads is not deterministic, so DeepWalk, node2vec and Walklets vary slightly from run
to run; the standard deviations show by how much.

## What the benchmark does not show

- **Tuned baselines.** The baselines run with their authors' settings. DeepWalk's
  window, node2vec's p and q or ProNE's filter could be tuned per dataset as Cleora's
  iterations are in the *tuned* tables; at 10² to 10⁴ times Cleora's cost per run this
  was not done, which is why the tuned numbers are kept out of the main tables.
- **Other dimensionalities.** Everything is at 1024 dimensions. Most baselines were
  published at 128, and some of them are better there than at 1024.
- **Methods that use node features or labels** (GCN, GraphSAGE, ...). All methods here
  see the graph only; on feature-rich graphs such as Flickr or Yelp, feature-based
  models are far ahead of every structural embedding.
- **Heterogeneous and hypergraph data**, which is what Cleora's column modifiers are
  for. The public datasets with labels are homogeneous graphs; MovieLens (bipartite)
  is the only exception here.
