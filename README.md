<p align="center">

![Cleora logo](files/images/cleora.png)

</p>

<h3 align="center">Graph Embeddings. Blazing Fast.</h3>

<p align="center">
The only graph embedding library that performs <b>all possible random walks in a single matrix multiplication</b>.<br>
No negative sampling. No GPU. No noise. Just fast, deterministic, production-grade embeddings.
</p>

<p align="center">
  <b>240x</b> Faster Than GraphSAGE &nbsp;·&nbsp;
  <b>9</b> Embedding Algorithms &nbsp;·&nbsp;
  <b>14</b> Built-in Datasets &nbsp;·&nbsp;
  <b>5 MB</b> Total Install Size
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

## Quick Start

```python
from pycleora import SparseMatrix, embed, find_most_similar

# Build graph from edge list
edges = ["alice item_laptop", "alice item_mouse", "bob item_keyboard"]
graph = SparseMatrix.from_iterator(iter(edges), "complex::reflexive::product")

# Generate 1024-dimensional embeddings
embeddings = embed(graph, feature_dim=1024, num_iterations=4)

# Find similar entities
similar = find_most_similar(graph, embeddings, "alice", top_k=5)
for r in similar:
    print(f"{r['entity_id']}: {r['similarity']:.4f}")
```

### Full Usage Example

```python
from pycleora import SparseMatrix
import numpy as np
import pandas as pd
import random

# Generate example data
customers = [f"Customer_{i}" for i in range(1, 20)]
products = [f"Product_{j}" for j in range(1, 20)]

data = {
    "customer": random.choices(customers, k=100),
    "product": random.choices(products, k=100),
}

# Create DataFrame
df = pd.DataFrame(data)

# Create hyperedges
customer_products = df.groupby('customer')['product'].apply(list).values

# Convert to Cleora input format
cleora_input = map(lambda x: ' '.join(x), customer_products)

# Create Markov transition matrix for the hypergraph
mat = SparseMatrix.from_iterator(cleora_input, columns='complex::reflexive::product')

# Look at entity ids in the matrix, corresponding to embedding vectors
print(mat.entity_ids)

# Initialize embedding vectors externally, using text, image, random vectors
# embeddings = ...

# Or use built-in random deterministic initialization
embeddings = mat.initialize_deterministically(1024)

# Perform Markov random walk, then normalize however many times we want

NUM_WALKS = 3   # The optimal number depends on the graph, typically between 3 and 7 yields good results
                # lower values tend to capture co-occurrence, higher iterations capture substitutability in a context

for i in range(NUM_WALKS):
    # Can propagate with a symmetric matrix as well, but left Markov is a great default
    embeddings = mat.left_markov_propagate(embeddings)
    # Normalize with L2 norm by default, for the embeddings to reside on a hypersphere. Can use standardization instead.
    embeddings /= np.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)

# We're done, here are our embeddings

for entity, embedding in zip(mat.entity_ids, embeddings):
    print(entity, embedding)

# We can now compare our embeddings with dot product (since they are L2 normalized)

print(np.dot(embeddings[0], embeddings[1]))
print(np.dot(embeddings[0], embeddings[2]))
print(np.dot(embeddings[0], embeddings[3]))
```

### CLI

```bash
pycleora embed --input graph.tsv --output embeddings.npz --dim 1024
```

---

## Key Advantages

### No Negative Sampling
Unlike DeepWalk, Node2Vec, and LINE, Cleora doesn't approximate random walks with negative sampling. It computes **all walks exactly** via matrix multiplication. Less noise, higher accuracy, perfect reproducibility.

### 240x Faster Than GraphSAGE
Zomato reported embedding generation in **under 5 minutes** with Cleora, compared to **20 hours with GraphSAGE** on the same dataset. Rust core with adaptive parallelism makes every CPU cycle count.

### Deterministic Embeddings
Same input always produces the same output. No random seeds, no stochastic variation, no "run it 5 times and average" workflows. Critical for reproducible research and production ML pipelines.

### Heterogeneous Hypergraphs
Natively handles multi-type nodes and edges, bipartite graphs, and hypergraphs. TSV input with typed columns like `complex::reflexive::product`. No graph preprocessing needed.

### 5 MB, Zero Dependencies
The entire library is ~5 MB. Compare: PyTorch Geometric is 500 MB+, DGL is 400 MB+. Cleora ships as a single compiled Rust extension. No CUDA, no cuDNN, no GPU driver headaches.

### Stable & Inductive
Embeddings are stable across runs and support inductive learning: new nodes can be embedded without retraining the entire graph. Production-ready from day one.

---

## Supported Algorithms

| Algorithm | Type | Description |
|-----------|------|-------------|
| **Cleora** | Spectral / Random Walk | Iterative Markov propagation with L2 normalization — all random walks in one matrix multiplication |
| **ProNE** | Spectral | Fast spectral propagation with Chebyshev polynomial approximation |
| **RandNE** | Random Projection | Gaussian random projection for very fast, approximate embeddings |
| **NetMF** | Matrix Factorization | Network Matrix Factorization — factorizes the DeepWalk matrix explicitly |
| **DeepWalk** | Random Walk | Classic random walk + skip-gram approach |
| **Node2Vec** | Random Walk | Biased random walks with tunable BFS/DFS exploration |
| **HOPE** | Matrix Factorization | High-Order Proximity preserved Embedding |
| **GraRep** | Matrix Factorization | Graph Representations with Global Structural Information |
| **GCN** | Mini-GNN | 2-layer Graph Convolutional Network classifier in pure numpy/scipy — no PyTorch needed |

All 9 algorithms are unified under a single API. Switch between methods by changing one parameter.

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

Tested on real-world graphs from 4K to 2M+ nodes. Cleora wins on accuracy, speed, and memory.

### Link Prediction Accuracy (AUC)

| Dataset | Cleora | NetMF | Node2Vec | DeepWalk | Cleora Time |
|---------|--------|-------|----------|----------|-------------|
| **ego-Facebook** (4K nodes, 88K edges) | **0.964** | 0.944 | 0.918 | 0.912 | 0.74s |
| **Flickr** (89K nodes, 899K edges) | **0.158** | OOM | OOM | OOM | 0.47s |
| **ogbn-arxiv** (169K nodes, 1.2M edges) | **0.038** | OOM | OOM | OOM | — |

### Speed Comparison

| Dataset | Cleora | RandNE | ProNE | NetMF |
|---------|--------|--------|-------|-------|
| **PPI-large** (57K nodes) | **0.33s** | 1.07s | 8.34s | OOM |
| **Yelp** (717K nodes) | **3.3s** | OOM | OOM | OOM |
| **roadNet-CA** (2M nodes) | **4.2s** | 9.0s | 57.7s | OOM |

### Memory Efficiency

| Dataset | Cleora | Runner-up | Factor |
|---------|--------|-----------|--------|
| PPI-large (57K) | **28 MB** | 458 MB | 16x less |
| Flickr (89K) | **44 MB** | 701 MB | 16x less |
| ogbn-arxiv (169K) | **83 MB** | 1.3 GB | 16x less |
| Yelp (717K) | **350 MB** | OOM | Only one that finished |
| roadNet (2M) | **1.9 GB** | 14.6 GB | ~8x less |

> 500x more nodes with only ~19x runtime increase — from 0.22s to 4.2s.

---

## Library Comparison

| Feature | **pycleora 3.0** | PyG | KarateClub | DGL | Node2Vec | StellarGraph |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| CPU-only (no GPU needed) | **Yes** | Optional | Yes | Optional | Yes | Optional |
| Rust-powered core | **Yes** | No (C++) | No | No (C++) | No | No (TF) |
| No negative sampling needed | **Yes** | No | No | No | No | No |
| Deterministic output | **Yes** | No | No | No | No | No |
| Node2Vec / DeepWalk | **Built-in** | Yes | Yes | Yes | Yes | Yes |
| GNN classifier (no PyTorch) | **GCN** | Requires PyTorch | No | Requires PyTorch | No | Requires TF |
| Built-in datasets | **14** | 70+ | ~5 | 40+ | No | ~10 |
| Graph sampling | **6 methods** | Yes | No | Yes | No | Yes |
| Hyperparameter tuning | **Grid + Random** | Manual | No | Manual | No | Manual |
| Install size | **~5 MB** | ~500 MB+ | ~15 MB | ~400 MB+ | ~2 MB | ~600 MB+ |
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

---

## How It Works

1. **Input Data** — Feed edge lists, interaction logs, or knowledge triples. Cleora accepts any TSV with typed columns.
2. **Hypergraph Construction** — Builds a heterogeneous hypergraph where a single edge can connect multiple entities of different types.
3. **Sparse Markov Matrix** — Constructs a sparse transition matrix (99%+ sparse). Rows normalized so each row sums to 1.
4. **Single Matrix Multiplication = All Walks** — One sparse matrix multiplication captures *every possible random walk* of a given length. No sampling, no noise.
5. **L2-Normalized Propagation** — Each iteration replaces every node's embedding with the L2-normalized average of its neighbors. 3-4 iterations for co-occurrence similarity, 7+ for contextual similarity.
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

A: What works best is grouping entities co-occurring in a similar context, and feeding them in whitespace-separated lines using `complex::reflexive` modifier is a good idea. E.g. if you have product data, you can group the products by shopping baskets or by users. If you have urls, you can group them by browser sessions, of by (user, time window) pairs. Check out the usage example above. Grouping products by customers is just one possibility.

**Q: Can I embed users and products simultaneously, to compare them with cosine similarity?**

A: No, this is a methodologically wrong approach, stemming from outdated matrix factorization approaches. What you should do is come up with good product embeddings first, then create user embeddings from them. Feeding two columns e.g. `user product` into cleora will result in a bipartite graph. Similar products will be close to each other, similar users will be close to each other, but users and products will not necessarily be similar to each other.

**Q: What embedding dimensionality to use?**

A: The more, the better, but we typically work from _1024_ to _4096_. Memory is cheap and machines are powerful, so don't skimp on embedding size.

**Q: How many iterations of Markov propagation should I use?**

A: Depends on what you want to achieve. Low iterations (3) tend to approximate the co-occurrence matrix, while high iterations (7+) tend to give contextual similarity (think skip-gram but much more accurate and faster).

**Q: How do I incorporate external information, e.g. entity metadata, images, texts into the embeddings?**

A: Just initialize the embedding matrix with your own vectors coming from a VIT, sentence-transformers, or a random projection of your numeric features. In that scenario low numbers of Markov iterations (1 to 3) tend to work best.

**Q: My embeddings don't fit in memory, what do I do?**

A: Cleora operates on dimensions independently. Initialize your embeddings with a smaller number of dimensions, run Cleora, persist to disk, then repeat. You can concatenate your resulting embedding vectors afterwards, but remember to normalize them afterwards!

**Q: Is there a minimum number of entity occurrences?**

A: No, an entity `A` co-occurring just 1 time with some other entity `B` will get a proper embedding, i.e. `B` will be the most similar to `A`. The other way around, `A` will be highly ranked among nearest neighbors of `B`, which may or may not be desirable, depending on your use case. Feel free to prune your input to Cleora to eliminate low-frequency items.

**Q: Are there any edge cases where Cleora can fail?**

A: Cleora works best for relatively sparse hypergraphs. If all your hyperedges contain some very common entity `X`, e.g. a _shopping bag_, then it will degrade the quality of embeddings by degenerating shortest paths in the random walk. It is a good practice to remove such entities from the hypergraph.

**Q: How can Cleora be so fast and accurate at the same time?**

A: Not using negative sampling is a great boon. By constructing the (sparse) Markov transition matrix, Cleora explicitly performs all possible random walks in a hypergraph in one big step (a single matrix multiplication). That's what we call a single _iteration_. We perform 3+ such iterations. Thanks to a highly efficient implementation in Rust, with special care for concurrency, memory layout and cache coherence, it is blazingly fast. Negative sampling or randomly selecting random walks tend to introduce a lot of noise - Cleora is free of those burdens.

---

## Resources

- **Whitepaper**: ["Cleora: A Simple, Strong and Scalable Graph Embedding Scheme"](https://arxiv.org/abs/2102.02302)
- **Documentation**: [cleora.readthedocs.io](https://cleora.readthedocs.io/)
- **Benchmarks**: [Full benchmark results](https://cleora.readthedocs.io/)
- **GitHub**: [github.com/BaseModelAI/cleora](https://github.com/BaseModelAI/cleora)

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

Synerise Cleora is MIT licensed, as found in the [LICENSE](LICENSE) file.

## How to Contribute

Pull requests are welcome. For details contact us at cleora@synerise.com
