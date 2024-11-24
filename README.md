<p align="center">

![Cleora logo](files/images/cleora.png)

</p>

## Achievements

:one:st place at [SIGIR eCom Challenge 2020](https://sigir-ecom.github.io/ecom20DCPapers/SIGIR_eCom20_DC_paper_1.pdf)
 
:two:nd place  and Best Paper Award at [WSDM Booking.com Challenge 2021](http://ceur-ws.org/Vol-2855/challenge_short_3.pdf)

:two:nd place at [Twitter Recsys Challenge 2021](https://recsys-twitter.com/competition_leaderboard/latest)

:three:rd place at [KDD Cup 2021](https://ogb.stanford.edu/paper/kddcup2021/mag240m_SyneriseAI.pdf)


# Cleora

_**Cleora** is a genus of moths in the family **Geometridae**. Their scientific name derives from the Ancient Greek geo γῆ or γαῖα "the earth", and metron μέτρον "measure" in reference to the way their larvae, or "inchworms", appear to "**measure the earth**" as they move along in a looping fashion._

Cleora is a general-purpose model for efficient, scalable learning of stable and inductive entity embeddings for heterogeneous relational data.

**Cleora** is now available as a python package _pycleora_. Key improvements compared to the previous version:
* _performance optimizations_: 10x faster embedding times
* _performance optimizations_: reduced memory usage
* _latest research_: significantly improved embedding quality
* _new feature_: can create graphs from a Python iterator in addition to tsv files
* _new feature_: seamless integration with _NumPy_
* _new feature_: item attributes support via custom embeddings initialization
* _new feature_: adjustable vector projection / normalization after each propagation step

**Breaking changes:**
* _transient_ modifier not supported any more - creating _complex::reflexive_ columns for hypergraph embeddings, grouped by the transient entity gives better results.


**Example usage:**

```
import pycleora
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
mat = pycleora.SparseMatrix.from_iterator(cleora_input, columns='complex::reflexive::product')

# Look at entity ids in the matrix, corresponding to embedding vectors
print(mat.entity_ids)
# ['Product_5', 'Product_3', 'Product_2', 'Product_4', 'Product_1']

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

**Read the whitepaper ["Cleora: A Simple, Strong and Scalable Graph Embedding Scheme"](https://arxiv.org/abs/2102.02302)**

Cleora embeds entities in *n-dimensional spherical spaces* utilizing extremely fast stable, iterative random projections, which allows for unparalleled performance and scalability. 

Types of data which can be embedded include for example:
- heterogeneous undirected graphs
- heterogeneous undirected hypergraphs
- text and other categorical array data
- any combination of the above


Key competitive advantages of Cleora:
* more than **197x faster than DeepWalk**
* **~4x-8x faster than [PyTorch-BigGraph](https://ai.facebook.com/blog/open-sourcing-pytorch-biggraph-for-faster-embeddings-of-extremely-large-graphs/)** (depends on use case)
* star expansion, clique expansion, and no expansion support for hypergraphs
* **quality of results outperforming or competitive** with other embedding frameworks like [PyTorch-BigGraph](https://ai.facebook.com/blog/open-sourcing-pytorch-biggraph-for-faster-embeddings-of-extremely-large-graphs/), GOSH, DeepWalk, LINE
* can embed extremely large graphs & hypergraphs on a single machine

Embedding times - example:

<table>
<tr>
<td> <b>Algorithm</b>
<td> <b>FB dataset</b>
<td> <b>RoadNet dataset</b>
<td> <b>LiveJournal dataset</b>
</tr>

<tr>
<td> Cleora
<td> 00:00:43 h
<td> 00:21:59 h
<td> 01:31:42 h
</tr>

<tr>
<td> PyTorch-BigGraph
<td> 00:04.33 h
<td> 00:31:11 h
<td> 07:10:00 h
</tr>

</table>

Link Prediction results - example:
<table>
  <tr>
    <td>
    <!-- <td rowspan="2">&nbsp;</td> -->
    <td colspan="2"><b>FB dataset</b></td>
    <td colspan="2"><b>RoadNet dataset</b></td>
    <td colspan="2"><b>LiveJournal dataset</b></td>
  </tr>
  <tr>
    <td> <b>Algorithm</b>
    <td> <b>MRR</b>
    <td> <b>HitRate@10</b>
    <td> <b>MRR</b>
    <td> <b>HitRate@10</b>
    <td> <b>MRR</b>
    <td> <b>HitRate@10</b>
  </tr>
  <tr>
    <td> Cleora
    <td> 0.072
    <td> 0.172
    <td> 0.929
    <td> 0.942
    <td> 0.586
    <td> 0.627
  </tr>
  <tr>
  <td> PyTorch-BigGraph
  <td> 0.035
  <td> 0.072
  <td> 0.850
  <td> 0.866
  <td> 0.565
  <td> 0.672
  </tr>

  <!-- <tr>
  <td> LINE
  <td> 0.075
  <td> 0.192
  <td> 0.962
  <td> 0.983
  <td> 0.553
  <td> 0.648
  </tr> -->
</table>

## Cleora design principles
Cleora is built as a multi-purpose "just embed it" tool, suitable for many different data types and formats.

Cleora ingests a relational table of rows representing a typed and undirected heterogeneous hypergraph, which can contain multiple:
- typed categorical columns
- typed categorical array columns

For example a relational table representing shopping baskets may have the following columns:

    user <\t> product <\t> store

With the input file containing values:

    user_id <\t> product_id product_id product_id <\t> store_id

Every column has a type, which is used to determine whether spaces of identifiers between different columns are shared or distinct. It is possible for two columns to share a type, which is the case for homogeneous graphs:

    user <\t> user

Based on the column format specification, Cleora performs:
 - Star decomposition of hyper-edges
 - Creation of pairwise graphs for all pairs of entity types
 - Embedding of each graph

The final output of Cleora consists of multiple files for each (undirected) pair of entity types in the table.

Those embeddings can then be utilized in a novel way thanks to their dim-wise independence property, which is described further below.

## Key technical features of Cleora embeddings
The embeddings produced by Cleora are different from those produced by Node2vec, Word2vec, DeepWalk or other systems in this class by a number of key properties:
 - **efficiency** - Cleora is two orders of magnitude faster than Node2Vec or DeepWalk
 - **inductivity** - as Cleora embeddings of an entity are defined only by interactions with other entities, vectors for new entities can be computed on-the-fly
 - **updatability** - refreshing a Cleora embedding for an entity is a very fast operation allowing for real-time updates without retraining
 - **stability** - all starting vectors for entities are deterministic, which means that Cleora embeddings on similar datasets will end up being similar. Methods like Word2vec, Node2vec or DeepWalk return different results with every run.
 - **cross-dataset compositionality** - thanks to stability of Cleora embeddings, embeddings of the same entity on multiple datasets can be combined by averaging, yielding meaningful vectors
 - **dim-wise independence** - thanks to the process producing Cleora embeddings, every dimension is independent of others. This property allows for efficient and low-parameter method for combining multi-view embeddings with Conv1d layers.
 - **extreme parallelism and performance** - Cleora is written in Rust utilizing thread-level parallelism for all calculations except input file loading. In practice this means that the embedding process is often faster than loading the input data.

## Key usability features of Cleora embeddings

The technical properties described above imply good production-readiness of Cleora, which from the end-user perspective can be summarized as follows:
- heterogeneous relational tables can be embedded without any artificial data pre-processing
- mixed interaction + text datasets can be embedded with ease
- cold start problem for new entities is non-existent
- real-time updates of the embeddings do not require any separate solutions
- multi-view embeddings work out of the box
- temporal, incremental embeddings are stable out of the box, with no need for re-alignment, rotations or other methods
- extremely large datasets are supported and can be embedded within seconds / minutes

## Documentation

More information can be found in [the full documentation](https://cleora.readthedocs.io/).

For details contact us at cleora@synerise.com

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

You are welcomed to contribute to this open-source toolbox. The detailed instructions will be released soon as issues.
