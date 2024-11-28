from typing import Any, Iterable, Optional, Self

import numpy as np
from numpy.typing import NDArray


class SparseMatrix:
    def __new__(cls, *args: Any) -> Self:
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

    def get_entity_column_mask(self, column_name: str) -> NDArray[np.bool]:
        pass

    def entity_degrees(self) -> NDArray[np.float32]:
        pass

    def initialize_deterministically(self, feature_dim: int, seed: int = 0) -> NDArray[np.float32]:
        pass
