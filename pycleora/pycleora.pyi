from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

__version__: str


class SparseMatrix:
    """Symmetric sparse co-occurrence matrix of a hypergraph."""

    #: Entity names in matrix row order (immutable, cached).
    entity_ids: Tuple[str, ...]
    num_entities: int
    #: Number of stored (directed) entries of the matrix.
    num_edges: int
    #: Weighted degree of every entity.
    entity_degrees: NDArray[np.float32]
    #: Names of the entity columns the matrix was built from.
    column_names: List[str]
    #: Indices of the entities that have no neighbour other than themselves.
    isolated_entities: List[int]

    def __new__(cls, *args: Any) -> "SparseMatrix": ...
    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...

    @staticmethod
    def from_iterator(
        hyperedges: Iterable[str],
        columns: str,
        hyperedge_trim_n: int = 16,
        num_workers: Optional[int] = None,
        weights: Optional[Union[Sequence[float], NDArray[Any]]] = None,
    ) -> "SparseMatrix": ...

    @staticmethod
    def from_cliques(
        cliques: Iterable[Sequence[str]],
        hyperedge_trim_n: int = 16,
        num_workers: Optional[int] = None,
        weights: Optional[Union[Sequence[float], NDArray[Any]]] = None,
    ) -> "SparseMatrix": ...

    @staticmethod
    def from_files(
        filepaths: List[str],
        columns: str,
        hyperedge_trim_n: int = 16,
        num_workers: Optional[int] = None,
    ) -> "SparseMatrix": ...

    @staticmethod
    def from_edges(
        src: Union[NDArray[np.int64], Sequence[str]],
        dst: Union[NDArray[np.int64], Sequence[str]],
        columns: str = "complex::reflexive::node",
        weights: Optional[Union[Sequence[float], NDArray[Any]]] = None,
        num_workers: Optional[int] = None,
    ) -> "SparseMatrix": ...

    def left_markov_propagate(
        self,
        x: NDArray[np.float32],
        num_workers: Optional[int] = None,
        out: Optional[NDArray[np.float32]] = None,
        rows: Optional[Union[Sequence[int], NDArray[np.integer[Any]]]] = None,
    ) -> NDArray[np.float32]: ...
    def symmetric_markov_propagate(
        self,
        x: NDArray[np.float32],
        num_workers: Optional[int] = None,
        out: Optional[NDArray[np.float32]] = None,
        rows: Optional[Union[Sequence[int], NDArray[np.integer[Any]]]] = None,
    ) -> NDArray[np.float32]: ...
    def left_markov_propagate_normalized_inplace(
        self, x: NDArray[np.float32], num_iterations: int, num_workers: Optional[int] = None
    ) -> None: ...
    def symmetric_markov_propagate_normalized_inplace(
        self, x: NDArray[np.float32], num_iterations: int, num_workers: Optional[int] = None
    ) -> None: ...
    def embed_fast(
        self,
        feature_dim: int,
        num_iterations: int,
        propagation: str = "left",
        seed: int = 0,
        residual_weight: float = 0.0,
        num_workers: Optional[int] = None,
    ) -> NDArray[np.float32]: ...
    def embed_fast_convergence(
        self,
        feature_dim: int,
        max_iterations: int,
        propagation: str = "left",
        seed: int = 0,
        residual_weight: float = 0.0,
        convergence_threshold: float = 0.0,
        num_workers: Optional[int] = None,
    ) -> Tuple[NDArray[np.float32], int]: ...
    def l2_normalize(self, x: NDArray[np.float32], num_workers: Optional[int] = None) -> NDArray[np.float32]: ...
    def initialize_deterministically(self, feature_dim: int, seed: int = 0) -> NDArray[np.float32]: ...
    def get_entity_column_mask(self, column_name: str) -> NDArray[np.bool_]: ...
    def get_entity_index(self, entity_id: str) -> int: ...
    def get_entity_indices(self, entity_ids: Sequence[str]) -> List[int]: ...
    def get_neighbors(self, entity_id: str) -> List[Tuple[str, float]]: ...
    def to_sparse_csr(
        self, markov_type: Optional[str] = None
    ) -> Tuple[NDArray[np.uint32], NDArray[np.uint32], NDArray[np.float32], int, int]: ...
    def to_csr(
        self, markov_type: Optional[str] = None
    ) -> Tuple[NDArray[np.integer[Any]], NDArray[np.integer[Any]], NDArray[np.float32]]: ...
    def __getstate__(self) -> bytes: ...
    def __setstate__(self, state: bytes) -> None: ...


def column_mean_cov(
    x: NDArray[np.float32], num_workers: Optional[int] = None
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def center_project(
    x: NDArray[np.float32],
    mean: NDArray[np.float32],
    transform: NDArray[np.float32],
    num_workers: Optional[int] = None,
) -> NDArray[np.float32]: ...
def center_project_inplace(
    x: NDArray[np.float32],
    mean: NDArray[np.float32],
    transform: NDArray[np.float32],
    num_workers: Optional[int] = None,
) -> None: ...
def l2_normalize_inplace(x: NDArray[np.float32], num_workers: Optional[int] = None) -> None: ...
def entity_hash(entity_id: str) -> int: ...
