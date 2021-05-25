.. why-cleora:

Why worth use Synerise Cleora?
===============================================

Key technical features of Cleora embeddings
--------------------------------------------------------------------------------------------

The embeddings produced by Cleora are different from those produced by Node2vec, Word2vec, DeepWalk or other systems in this class by a number of key properties:

- **efficiency** - Cleora is two orders of magnitude faster than Node2Vec or DeepWalk
- **inductivity** - as Cleora embeddings of an entity are defined only by interactions with other entities, vectors for new entities can be computed on-the-fly
- **updatability** - refreshing a Cleora embedding for an entity is a very fast operation allowing for real-time updates without retraining
- **stability** - all starting vectors for entities are deterministic, which means that Cleora embeddings on similar datasets will end up being similar. Methods like Word2vec, Node2vec or DeepWalk return different results with every run.
- **cross-dataset compositionality** - thanks to stability of Cleora embeddings, embeddings of the same entity on multiple datasets can be combined by averaging, yielding meaningful vectors
- **dim-wise independence** - thanks to the process producing Cleora embeddings, every dimension is independent of others. This property allows for efficient and low-parameter method for combining multi-view embeddings with Conv1d layers.
- **extreme parallelism and performance** - Cleora is written in Rust utilizing thread-level parallelism for all calculations except input file loading. In practice this means that the embedding process is often faster than loading the input data.

Key usability features of Cleora embeddings
--------------------------------------------------------------------------------------------

The technical properties described above imply good production-readiness of Cleora, which from the end-user perspective can be summarized as follows:

- heterogeneous relational tables can be embedded without any artificial data pre-processing
- mixed interaction + text datasets can be embedded with ease
- cold start problem for new entities is non-existent
- real-time updates of the embeddings do not require any separate solutions
- multi-view embeddings work out of the box
- temporal, incremental embeddings are stable out of the box, with no need for re-alignment, rotations or other methods
- extremely large datasets are supported and can be embedded within seconds / minutes
