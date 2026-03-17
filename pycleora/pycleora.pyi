from typing import Any, Iterable, List, Optional, Self

import numpy as np
from numpy.typing import NDArray


class SparseMatrix:
    entity_ids: List[str]
    num_entities: int
    num_edges: int
    entity_degrees: NDArray[np.float32]

    def __new__(cls, *args: Any) -> Self:
        pass

    def __repr__(self) -> str:
        pass

    def __len__(self) -> int:
        pass

    @classmethod
    def from_iterator(
        cls, hyperedges: Iterable[str], columns: str, hyperedge_trim_n: int = 16, num_workers: Optional[int] = None
    ) -> Self:
        pass

    @classmethod
    def from_files(
        cls, filepaths: list[str], columns: str, hyperedge_trim_n: int = 16, num_workers: Optional[int] = None
    ) -> Self:
        pass

    def left_markov_propagate(self, x: NDArray[np.float32], num_workers: Optional[int] = None) -> NDArray[np.float32]:
        pass

    def symmetric_markov_propagate(
        self, x: NDArray[np.float32], num_workers: Optional[int] = None
    ) -> NDArray[np.float32]:
        pass

    def get_entity_column_mask(self, column_name: str) -> NDArray[np.bool_]:
        pass

    def get_entity_index(self, entity_id: str) -> int:
        pass

    def get_entity_indices(self, entity_ids: List[str]) -> List[int]:
        pass

    def initialize_deterministically(self, feature_dim: int, seed: int = 0) -> NDArray[np.float32]:
        pass

    def __getstate__(self) -> bytes:
        pass

    def __setstate__(self, state: bytes) -> None:
        pass
