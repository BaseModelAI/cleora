<p align="center">

![Cleora logo](files/images/cleora.png)

</p>

<h1 align="center">The Graph Embedding Engine</h1>

<p align="center">
Cleora computes <b>all possible random walks in a single matrix multiplication</b>.<br>
No negative sampling. No GPU. No noise. Just fast, deterministic, production-grade embeddings.
</p>

<p align="center">
  <a href="https://cleora.ai">Website</a> &nbsp;·&nbsp;
  <a href="https://cleora.ai/docs">Documentation</a> &nbsp;·&nbsp;
  <a href="https://cleora.ai/api">API Reference</a> &nbsp;·&nbsp;
  <a href="https://cleora.ai/benchmarks">Benchmarks</a>
</p>

<p align="center">
  <code>pip install pycleora</code>
</p>

---

<p align="center">
  <b>Measured on real graphs. Reproducible with one script.</b><br>
  On <b>8 public datasets</b> against <b>7 other embedding methods</b>, Cleora is the best method — or within one standard deviation of it — on <b>5</b>,<br>and the fastest and leanest on all of them. <a href="#benchmarks">See the numbers</a>.
</p>

<p align="center">
  <b>2M nodes</b> embedded in 8 s on 2 CPU cores &nbsp;·&nbsp;
  <b>Bit-for-bit</b> reproducible &nbsp;·&nbsp;
  <b>&lt; 1 MB</b> wheel, NumPy only &nbsp;·&nbsp;
  <b>0</b> GPUs required
</p>

---

## Achievements

:one:st place at [SIGIR eCom Challenge 2020](https://sigir-ecom.github.io/ecom20DCPapers/SIGIR_eCom20_DC_paper_1.pdf)

:two:nd place and Best Paper Award at [WSDM Booking.com Challenge 2021](http://ceur-ws.org/Vol-2855/challenge_short_3.pdf)

:two:nd place at [Twitter Recsys Challenge 2021](https://recsys-twitter.com/competition_leaderboard/latest)

:three:rd place at [KDD Cup 2021](https://ogb.stanford.edu/paper/kddcup2021/mag240m_SyneriseAI.pdf)

---

## Installation

```bash
pip install pycleora
```

Requires Python 3.10 or newer. NumPy is the only required dependency: building graphs, `embed()`, whitening, inductive updates, similarity search and link prediction need nothing else.

Optional extras:

```bash
pip install "pycleora[scipy]"   # baseline algorithms (ProNE, NetMF, ...), attention / directed embeddings, graph statistics
pip install "pycleora[viz]"     # matplotlib for visualization
pip install "pycleora[full]"    # scipy + matplotlib + networkx + tqdm
```

## Quick Start

```python
from pycleora import SparseMatrix, embed, find_most_similar

# one line per shopping basket: the products that were bought together
baskets = [
    "laptop mouse keyboard",
    "laptop mouse monitor",
    "keyboard mouse",
    "espresso_machine coffee_beans",
    "coffee_beans mug",
    "espresso_machine coffee_beans mug",
]
graph = SparseMatrix.from_iterator(iter(baskets), "complex::reflexive::product")

# a toy graph: a few dimensions and classic mode are all it needs
embeddings = embed(graph, feature_dim=8, num_iterations=3, whiten=False)

similar = find_most_similar(graph, embeddings, "laptop", top_k=3)
for r in similar:
    print(f"{r['entity_id']}: {r['similarity']:.4f}")
```

On real data, start with the defaults: `embed(graph)` uses `feature_dim=256`, `num_iterations=40`, and whitening after every propagation step. Whitening needs clearly more entities than dimensions — on a graph with at most `feature_dim + 1` entities it erases the structure (all vectors end up equidistant) and pycleora warns about it — which is why the toy example above turns it off.

**Classic Cleora** — propagation and L2 normalisation only, no whitening — is one argument away. It runs entirely in the Rust core, needs just two embedding matrices of memory and only a few iterations (4–16):

```python
embeddings = embed(graph, feature_dim=256, num_iterations=8, whiten=False)
```

Which of the two is better depends on the graph; the classic mode is one to two orders of magnitude faster, so it costs little to try both (see [Benchmarks](#benchmarks)).

**Edges from arrays, with weights.** Graphs do not have to go through strings, and a weight per edge (or per hyperedge) is part of the transition matrix:

```python
import numpy as np
from pycleora.io_utils import from_arrays

src = np.array([0, 0, 1, 2])
dst = np.array([1, 2, 2, 3])
graph = from_arrays(src, dst, weights=[1.0, 0.5, 2.0, 1.0])

# the same for text input: one weight per line, or a `weight::` column in the file
graph = SparseMatrix.from_iterator(iter(baskets), "complex::reflexive::product", weights=[1, 3, 1, 1, 2, 1])
```

**New data without retraining.** `embed_inductive` adds edges and entities to an embedded graph. Vectors that already exist stay exactly as they are — models trained on them remain valid — and every new entity is placed at the weighted average of its neighbours:

```python
from pycleora import embed_inductive

graph, embeddings = embed_inductive(
    graph, embeddings, existing_edges=baskets, new_edges=["mug tea_bags"],
    columns="complex::reflexive::product",
)
```

### Step-by-Step Example

The high-level `embed()` function wraps the Markov propagation loop for convenience. Here's the full manual version, which gives you complete control over the process:

```python
from pycleora import SparseMatrix, whiten_embeddings
import numpy as np
import pandas as pd
import random

customers = [f"Customer_{i}" for i in range(1, 300)]
products = [f"Product_{j}" for j in range(1, 200)]

data = {
    "customer": random.choices(customers, k=3000),
    "product": random.choices(products, k=3000),
}

df = pd.DataFrame(data)
customer_products = df.groupby('customer')['product'].apply(list).values
cleora_input = map(lambda x: ' '.join(x), customer_products)

mat = SparseMatrix.from_iterator(cleora_input, columns='complex::reflexive::product')

print(mat.entity_ids[:10])

# 199 products: 32 dimensions leave whitening enough entities to work with
embeddings = mat.initialize_deterministically(32)

NUM_ITERATIONS = 40

for i in range(NUM_ITERATIONS):
    embeddings = mat.left_markov_propagate(embeddings)
    embeddings /= np.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)
    embeddings = whiten_embeddings(embeddings)

for entity, embedding in list(zip(mat.entity_ids, embeddings))[:3]:
    print(entity, embedding)

print(np.dot(embeddings[0], embeddings[1]))
```

### CLI

```bash
pycleora embed --input graph.tsv --output embeddings.npz --dim 256 --iterations 40
pycleora info --input graph.tsv
pycleora similar --input graph.tsv --entity alice --top-k 10
pycleora benchmark --dataset karate_club
```

---

## Key Advantages

### No Negative Sampling
Unlike DeepWalk, Node2Vec, and LINE, Cleora doesn't approximate random walks with negative sampling. It computes **all walks exactly** via matrix multiplication. Less noise, perfect reproducibility.

### 240x Faster Than GraphSAGE
Zomato reported embedding generation in **under 5 minutes** with Cleora, compared to **20 hours with GraphSAGE** on the same dataset. Rust core with adaptive parallelism makes every CPU cycle count.

### Deterministic Embeddings
Same input always produces the same output — bit for bit, whatever the number of threads. The graph is accumulated in fixed-point arithmetic and every parallel reduction has a fixed order, so there is no stochastic variation and no "run it 5 times and average" workflow. `seed` selects a different, equally reproducible starting point. (The whitening of the default mode calls LAPACK through NumPy, so across machines the default mode is as reproducible as that library; the classic mode does not depend on it.)

### Heterogeneous Hypergraphs
Natively handles multi-type nodes and edges, bipartite graphs, and hypergraphs. TSV input with typed columns like `complex::reflexive::product`. No graph preprocessing needed.

### Under 1 MB, NumPy Only
The wheel is under 1 MB and NumPy is its only required dependency. Compare: PyTorch Geometric is 500 MB+, DGL is 400 MB+. Cleora ships as a single compiled Rust extension that works on every Python from 3.10 on. No CUDA, no cuDNN, no GPU driver headaches.

### Stable & Inductive
Embeddings are identical across runs, and new nodes can be embedded without retraining the entire graph: `embed_inductive` and `embed_streaming` leave every existing vector untouched and place new entities next to their neighbours, so a model trained on yesterday's embeddings keeps working today.

---

## Supported Algorithms

| Algorithm | Type | Description |
|-----------|------|-------------|
| **Cleora** | Spectral / Random Walk | Iterative Markov propagation — all random walks in one matrix multiplication. Whitening after every iteration by default, `whiten=False` for the classic algorithm |
| **ProNE** | Spectral | Fast spectral propagation with Chebyshev polynomial approximation |
| **RandNE** | Random Projection | Gaussian random projection for very fast, approximate embeddings |
| **NetMF** | Matrix Factorization | Network Matrix Factorization — factorizes the DeepWalk matrix explicitly (dense, for graphs up to ~20K nodes) |
| **DeepWalk** | Random Walk | Random walks factorized with PPMI + SVD — the matrix-factorization view of DeepWalk, no skip-gram training |
| **Node2Vec** | Random Walk | Biased random walks with tunable BFS/DFS exploration (`p`, `q`), factorized the same way |
| **HOPE** | Matrix Factorization | High-Order Proximity preserved Embedding, Katz index (dense, small graphs) |
| **GraRep** | Matrix Factorization | Graph Representations with Global Structural Information (dense, small graphs) |
| **MLP** | Neural Classifier | 2-layer MLP classifier in pure numpy/scipy — no PyTorch needed |

All algorithms are unified under a single API. Switch between methods by changing one parameter:

```bash
pycleora embed --input graph.tsv --output out.npz --algorithm cleora
pycleora embed --input graph.tsv --output out.npz --algorithm prone
pycleora embed --input graph.tsv --output out.npz --algorithm node2vec
```

### Advanced Embedding Modes

Beyond the standard algorithms, Cleora supports several advanced embedding strategies:

- **Multiscale embeddings** — concatenates embeddings from different iteration depths (e.g. scales `[10, 20, 30, 40]`) to capture both local and global graph structure simultaneously
- **Attention-weighted propagation** — uses softmax-normalized dot-product attention during propagation, dynamically weighting neighbor contributions
- **Supervised refinement** — fine-tunes unsupervised embeddings using positive/negative entity pairs with a triplet margin loss
- **Directed graph embeddings** — handles asymmetric relationships where edge direction matters
- **Weighted graph embeddings** — a weight per edge or hyperedge (`weights=` or a `weight::` column) scales its share of the transition matrix; an integer weight `k` equals repeating the line `k` times
- **Inductive and streaming updates** — `embed_inductive` / `embed_streaming` add entities without moving the existing vectors; `refresh_every` periodically re-embeds and rotates the result back onto the old coordinates
- **Node feature integration** — initializes embeddings with external features (text, image, numeric) before propagation
- **PCA whitening** — built-in whitening after every iteration by default to decorrelate embedding dimensions and improve downstream task performance

---

## Batteries Included

pycleora ships with a comprehensive set of built-in modules:

| Module | What it does |
|--------|-------------|
| `pycleora.community` | Community detection (Louvain) |
| `pycleora.classify` | MLP and Label Propagation classifiers — no PyTorch needed |
| `pycleora.sampling` | 6 graph sampling methods |
| `pycleora.tuning` | Grid search and random search for hyperparameter tuning |
| `pycleora.compress` | Embedding compression (PCA, random projection, product quantization) |
| `pycleora.io_utils` | Graphs from NumPy / pandas / SciPy / NetworkX (`from_arrays`, `from_pandas`, ...), save/load embeddings (NPZ, CSV, TSV) |
| `pycleora.viz` | Embedding visualization (UMAP, t-SNE projections) |
| `pycleora.metrics` | Evaluation metrics for embeddings |
| `pycleora.benchmark` | Compare algorithms with time, memory, and accuracy metrics |
| `pycleora.ensemble` | Combine embeddings from multiple algorithms |
| `pycleora.align` | Embedding alignment across graphs |
| `pycleora.search` | Nearest-neighbor entity search |
| `pycleora.stats` | Graph statistics and degree analysis |
| `pycleora.preprocess` | Graph preprocessing and filtering |
| `pycleora.hetero` | Heterogeneous graph utilities |
| `pycleora.generators` | Synthetic graph generators for testing |
| `pycleora.datasets` | Real benchmark graphs downloaded from their original sources (Cora, CiteSeer, PubMed, PPI, email-Eu-core, ogbn-arxiv, Facebook, roadNet-CA, and more); every dataset carries `"synthetic": False` and its `source` |

See the [full API reference](https://cleora.ai/api) for details on every function and parameter.

---

## Case Study: Zomato

**From 20 hours to under 5 minutes** — powering recommendations for 80M+ users across 500+ cities.

Zomato's ML team needed graph embeddings to power "People Like You" restaurant recommendations. Their initial approach with **GraphSAGE took ~20 hours** just to process customer-restaurant interaction data for a single city region — making it impossible to scale across 500+ cities.

**Pipeline:**
1. **Customer-Restaurant Graph** — Bipartite graph of customer orders and restaurant interactions
2. **Cleora Embeddings** (< 5 minutes) — 197x faster than DeepWalk, no sampling of positive/negative examples
3. **EMDE Density Estimation** — Customer preferences modeled as probability density functions
4. **Production Recommendations** — Restaurant recommendations, search ranking, dish suggestions, and "People Like You" lookalikes

**Results:**

| Metric | Value |
|--------|-------|
| Speed vs DeepWalk | **197x** faster |
| Embedding generation | **< 5 min** |
| Cities scaled to | **500+** |
| GPUs required | **0** |

[Read the full Zomato blog post →](https://www.zomato.com/blog/connecting-the-dots-strengthening-recommendations-for-our-customers-part-two/)

---

## Benchmarks

Node classification on eight public graphs, from the graph structure alone (no node features), at 256 dimensions. Every number in this section is produced by [`benchmarks/run_benchmarks.py`](benchmarks/run_benchmarks.py):

```bash
pip install "pycleora[full]" scikit-learn
python benchmarks/run_benchmarks.py
```

**Protocol.** The embeddings are standardized and fed to a logistic regression. Half of the nodes are used for training, the regularization strength is chosen on a validation part of the training set, and the result is the mean over 5 random splits (3 for the two multi-label graphs, PPI and Wikipedia POS, which use the top-k protocol of the DeepWalk and node2vec papers and report micro-F1). All methods see the same splits. The baselines are the implementations in `pycleora.algorithms`: NumPy/SciPy code that follows the papers, with DeepWalk and node2vec as their matrix-factorization variants (random walks → PPMI → SVD) rather than skip-gram. NetMF, HOPE and GraRep build a dense n × n matrix and are run on graphs up to 5,000 nodes. Measured with pycleora 4.0.0, Python 3.13 and NumPy 2.5 on a 2-core x86-64 cloud VM with 8 GB of memory: times will differ on your hardware, and so may the order of methods that are close.

### Classification Accuracy

| Dataset | Nodes | Edges | Metric | Cleora<br>(default) | Cleora<br>(classic, 8 it.) | ProNE | RandNE | DeepWalk | node2vec | NetMF | HOPE | GraRep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Cora** | 2.7K | 5.3K | accuracy | 0.800 | **0.812** | 0.789 | 0.709 | **0.813** | 0.803 | **0.806** | 0.775 | 0.801 |
| **CiteSeer** | 3.3K | 4.6K | accuracy | **0.648** | 0.617 | 0.595 | 0.481 | 0.624 | 0.618 | 0.625 | 0.550 | 0.564 |
| **PubMed** | 19.7K | 44.3K | accuracy | 0.818 | 0.730 | 0.811 | 0.615 | **0.827** | 0.822 | — | — | — |
| **email-Eu-core** | 1.0K | 16.1K | accuracy | 0.630 | **0.753** | 0.674 | **0.759** | 0.641 | 0.664 | 0.669 | 0.683 | **0.755** |
| **Amazon Photo** | 7.5K | 119K | accuracy | **0.933** | **0.932** | 0.925 | 0.919 | 0.929 | 0.929 | — | — | — |
| **Amazon Computers** | 13.5K | 246K | accuracy | **0.908** | 0.899 | 0.902 | 0.889 | 0.905 | 0.904 | — | — | — |
| **PPI** | 3.9K | 37.8K | micro-F1 | 0.216 | 0.234 | 0.232 | 0.220 | 0.219 | 0.217 | 0.219 | 0.207 | **0.240** |
| **Wikipedia POS** | 4.8K | 92.3K | micro-F1 | 0.472 | 0.477 | 0.530 | 0.449 | 0.440 | 0.443 | 0.454 | **0.538** | 0.529 |

Bold: the best method on that graph and every method within one standard deviation (over the splits) of it.

- **No method wins everywhere.** One of the two Cleora modes is the best method, or within one standard deviation of it, on five of the eight graphs. On PubMed the default mode is 0.009 behind DeepWalk, on PPI the classic mode is 0.006 behind GraRep. On Wikipedia POS, a word co-occurrence network labelled with parts of speech, both are clearly behind HOPE, ProNE and GraRep (0.48 against 0.53–0.54).
- **The two Cleora modes are different tools.** The whitened default is ahead on CiteSeer, PubMed and Amazon Computers; classic Cleora with 8 iterations is ahead on Cora, email-Eu-core and PPI. Classic mode takes 1/25 to 1/100 of the time, so trying both costs little.

### Time and Memory

Embedding time in seconds (best of three runs for Cleora, one run for the others):

| Dataset | Cleora<br>(default) | Cleora<br>(classic, 8 it.) | ProNE | RandNE | DeepWalk | node2vec | NetMF | HOPE | GraRep |
|---|---|---|---|---|---|---|---|---|---|
| **Cora** | 0.79 | **0.01** | 0.81 | 0.09 | 1.93 | 2.44 | 6.70 | 9.95 | 38 |
| **CiteSeer** | 0.94 | **0.01** | 1.10 | 0.14 | 1.96 | 2.66 | 11 | 17 | 66 |
| **PubMed** | 3.48 | **0.06** | 8.54 | 0.95 | 32 | 34 | — | — | — |
| **email-Eu-core** | 0.61 | **0.01** | 0.38 | 0.04 | 0.87 | 1.44 | 0.39 | 0.64 | 2.04 |
| **Amazon Photo** | 1.72 | **0.06** | 3.54 | 0.42 | 14 | 17 | — | — | — |
| **Amazon Computers** | 3.10 | **0.13** | 7.33 | 0.94 | 43 | 47 | — | — | — |
| **PPI** | 1.08 | **0.02** | 1.65 | 0.19 | 7.15 | 8.32 | 20 | 30 | 115 |
| **Wikipedia POS** | 1.35 | **0.05** | 2.30 | 0.31 | 15 | 16 | 37 | 64 | 254 |

Additional peak memory of the embedding step in MB:

| Dataset | Cleora<br>(default) | Cleora<br>(classic, 8 it.) | ProNE | RandNE | DeepWalk | node2vec | NetMF | HOPE | GraRep |
|---|---|---|---|---|---|---|---|---|---|
| **Cora** | 11 | **5** | 70 | 30 | 119 | 113 | 569 | 508 | 691 |
| **CiteSeer** | 13 | **6** | 83 | 36 | 131 | 131 | 833 | 715 | 954 |
| **PubMed** | 53 | **38** | 426 | 198 | 1.0 GB | 908 | — | — | — |
| **email-Eu-core** | 8 | **2** | 33 | 14 | 59 | 58 | 80 | 77 | 99 |
| **Amazon Photo** | 23 | **15** | 190 | 82 | 548 | 480 | — | — | — |
| **Amazon Computers** | 38 | **26** | 331 | 146 | 1.3 GB | 1.2 GB | — | — | — |
| **PPI** | 14 | **7** | 98 | 42 | 304 | 285 | 1.1 GB | 1.0 GB | 1.4 GB |
| **Wikipedia POS** | 16 | **9** | 126 | 54 | 479 | 458 | 1.7 GB | 1.6 GB | 2.2 GB |

Classic Cleora is the fastest and smallest method on every graph: 6–16 times faster than RandNE, 45–130 times faster than ProNE and more than 100 times faster than the DeepWalk variant, in 2–38 MB. The whitened default needs 0.6–3.5 s and 8–53 MB here: about as fast as ProNE on the three smallest graphs and 1.5–2.5 times faster on the others, with 4–9 times less memory.

### Larger Graphs

**ogbn-arxiv** — 169K nodes, 1.2M edges, 40 classes. One random 50/50 split with a fixed regularization strength (`--splits 1 --fixed-c 0.1`); the DeepWalk variants and the dense methods were not run at this size.

| Method | Accuracy | Time | Peak memory |
|--------|----------|------|-------------|
| **Cleora (default)** | **0.691** | 33 s | 351 MB |
| Cleora (classic, 8 it.) | 0.548 | **2.0 s** | **330 MB** |
| ProNE | 0.646 | 138 s | 3.5 GB |
| RandNE | 0.381 | 17 s | 1.7 GB |

**roadNet-CA** — 1.97M nodes, 5.5M edges, no labels, so time and memory only. Classic Cleora with 8 iterations takes **8 seconds**; the whitened default with 40 iterations takes 276 seconds. Both need the two embedding matrices they work on — 3.8 GB at 256 dimensions — and little else.

---

## Library Comparison

| Feature | **pycleora 4.0** | PyG | KarateClub | DGL | Node2Vec | StellarGraph |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| CPU-only (no GPU needed) | **Yes** | Optional | Yes | Optional | Yes | Optional |
| Rust-powered core | **Yes** | No (C++) | No | No (C++) | No | No (TF) |
| No negative sampling needed | **Yes** | No | No | No | No | No |
| Deterministic output | **Yes** | No | No | No | No | No |
| Node2Vec / DeepWalk | **Built-in** | Yes | Yes | Yes | Yes | Yes |
| MLP classifier (no PyTorch) | **MLP** | Requires PyTorch | No | Requires PyTorch | No | Requires TF |
| Graph sampling | **6 methods** | Yes | No | Yes | No | Yes |
| Hyperparameter tuning | **Grid + Random** | Manual | No | Manual | No | Manual |
| Install size | **< 1 MB** (+ NumPy) | ~500 MB+ | ~15 MB | ~400 MB+ | ~2 MB | ~600 MB+ |
| Actively maintained | **Yes** | Yes | Yes | Yes | Yes | Archived |

---

## Use Cases

- **Recommendation Systems** — Products, content, restaurants, videos
- **Knowledge Graphs** — Entity and relation embeddings
- **Customer Lookalikes** — Find users with similar behavior patterns
- **Entity Resolution** — Match entities across data sources
- **Fraud Detection** — Detect anomalous patterns in transaction graphs
- **Social Networks** — Community detection and link prediction
- **Drug Discovery** — Molecule and protein interaction networks
- **Supply Chain** — Supplier and logistics graph analysis

See [cleora.ai/use-cases](https://cleora.ai/use-cases) for detailed walkthroughs with code examples.

---

## How It Works

1. **Input Data** — Feed edge lists, interaction logs, or knowledge triples. Cleora accepts any TSV with typed columns.
2. **Hypergraph Construction** — Builds a heterogeneous hypergraph where a single edge can connect multiple entities of different types.
3. **Sparse Markov Matrix** — Constructs a sparse transition matrix (99%+ sparse). Rows normalized so each row sums to 1.
4. **Single Matrix Multiplication = All Walks** — One sparse matrix multiplication captures *every possible random walk* of a given length. No sampling, no noise.
5. **L2-Normalized + Whitened Propagation** — Each iteration replaces every node's embedding with the L2-normalized average of its neighbors and then whitens the embedding space. The default configuration runs 40 iterations at 256 dimensions.
6. **Embeddings Ready** — Dense, deterministic embedding vectors for every entity. Same input always yields same output.

---

## Also Used By

**Synerise** — AI/ML platform processing billions of e-commerce events daily. Cleora powers core recommendation and personalization: product embeddings from terabytes of transactions, substitute vs. complement detection, customer segmentation, cold-start solving — all on CPU in minutes.

**Dailymotion** — Video platform with 350M+ monthly visitors. Personalized video recommendations with improved relevance and catalog coverage.

**ML Competitions** — Cleora-powered solutions achieved top placements in KDD Cup 2021, WSDM WebTour 2021, and SIGIR eCom 2020 — beating deep learning approaches on travel, e-commerce, and web recommendation benchmarks.

---

## FAQ

**Q: What should I embed?**

A: Any entities that interact with each other, co-occur or can be said to be present together in a given context. Examples can include: products in a shopping basket, locations frequented by the same people at similar times, employees collaborating together, chemical molecules being present in specific circumstances, proteins produced by the same bacteria, drug interactions, co-authors of the same academic papers, companies occurring together in the same LinkedIn profiles.

**Q: How should I construct the input?**

A: What works best is grouping entities co-occurring in a similar context, and feeding them in whitespace-separated lines using `complex::reflexive` modifier is a good idea. E.g. if you have product data, you can group the products by shopping baskets or by users. If you have urls, you can group them by browser sessions, or by (user, time window) pairs. Check out the usage example above. Grouping products by customers is just one possibility.

**Q: Can I embed users and products simultaneously, to compare them with cosine similarity?**

A: No, this is a methodologically wrong approach, stemming from outdated matrix factorization approaches. What you should do is come up with good product embeddings first, then create user embeddings from them. Feeding two columns e.g. `user product` into cleora will result in a bipartite graph. Similar products will be close to each other, similar users will be close to each other, but users and products will not necessarily be similar to each other.

**Q: What embedding dimensionality to use?**

A: The default is **256**. For larger production systems we often work from _1024_ to _4096_, but `256` is the baseline shipped by the library.

**Q: How many iterations of Markov propagation should I use?**

A: The default is **40** whitening-enhanced propagation steps. If you want more local, co-occurrence-style behavior you can dial that down manually; higher values bias more toward contextual similarity. Without whitening (`whiten=False`) nothing counteracts over-smoothing, so use far fewer iterations there: 4–16 is the useful range, and all of them cost a fraction of the default.

**Q: How do I incorporate external information, e.g. entity metadata, images, texts into the embeddings?**

A: Just initialize the embedding matrix with your own vectors coming from a VIT, sentence-transformers, or a random projection of your numeric features. In that scenario fewer Markov iterations than the default `40` often work best.

**Q: My embeddings don't fit in memory, what do I do?**

A: Classic Cleora (`whiten=False`) operates on dimensions independently, so you can compute the embedding in slices: run it with a smaller number of dimensions and a different `seed` for every slice, persist each result to disk, and concatenate the slices afterwards — but remember to normalize the rows again! Use a new `seed` each time: the same seed reproduces the same columns. (The whitened default mixes the dimensions and cannot be split this way.) The classic mode needs memory for only two embedding matrices to begin with.

**Q: Is there a minimum number of entity occurrences?**

A: No, an entity `A` co-occurring just 1 time with some other entity `B` will get a proper embedding, i.e. `B` will be the most similar to `A`. The other way around, `A` will be highly ranked among nearest neighbors of `B`, which may or may not be desirable, depending on your use case. Feel free to prune your input to Cleora to eliminate low-frequency items.

**Q: Are there any edge cases where Cleora can fail?**

A: Cleora works best for relatively sparse hypergraphs. If all your hyperedges contain some very common entity `X`, e.g. a _shopping bag_, then it will degrade the quality of embeddings by degenerating shortest paths in the random walk. It is a good practice to remove such entities from the hypergraph.

**Q: How can Cleora be so fast and accurate at the same time?**

A: Not using negative sampling is a great boon. By constructing the (sparse) Markov transition matrix, Cleora explicitly performs all possible random walks in a hypergraph in one big step (a single matrix multiplication). That's what we call a single _iteration_. The default configuration performs 40 such iterations with whitening after every step. Negative sampling or randomly selecting random walks tend to introduce a lot of noise - Cleora is free of those burdens.

---

## Resources

- **Website**: [cleora.ai](https://cleora.ai)
- **API Reference**: [cleora.ai/api](https://cleora.ai/api)
- **Benchmarks**: [cleora.ai/benchmarks](https://cleora.ai/benchmarks)
- **Whitepaper**: ["Cleora: A Simple, Strong and Scalable Graph Embedding Scheme"](https://arxiv.org/abs/2102.02302)
- **GitHub**: [github.com/BaseModelAI/cleora](https://github.com/BaseModelAI/cleora)
- **PyPI**: [pypi.org/project/pycleora](https://pypi.org/project/pycleora/)

## Cite

Please cite [our paper](https://arxiv.org/abs/2102.02302) (and the respective papers of the methods used) if you use this code in your own work:

```
@article{DBLP:journals/corr/abs-2102-02302,
  author    = {Barbara Rychalska, Piotr Babel, Konrad Goluchowski, Andrzej Michalowski, Jacek Dabrowski},
  title     = {Cleora: {A} Simple, Strong and Scalable Graph Embedding Scheme},
  journal   = {CoRR},
  year      = {2021}
}
```

## License

MIT licensed. See [LICENSE](LICENSE) for details.

## Contributing

Pull requests are welcome. For major changes, please open an issue first. Contact: cleora@synerise.com
