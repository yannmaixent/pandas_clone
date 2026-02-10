from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Union, overload

import numpy as np

Key = Union[int, slice, np.ndarray]

@dataclass(frozen=True)
class Index:
    """
    Minimal Pandas-like Index (immutable, ID)
    Backed by a Numpy array
    """

    _values: np.ndarray

    def __init__(self, data: Iterable):
        arr = np.asarray(list(data), dtype=object)
        if arr.ndim != 1:
            raise ValueError("Index must be 1-dimensional")
        object.__setattr__(self, "_values", arr)

    def __len__(self) -> int:
        return int(self._values.size)
    
    @overload
    def __getitem__(self, key: int) -> object: ...
    @overload
    def __getitem__(self, key: slice) -> "Index": ...
    @overload
    def __getitem__(self, key: np.ndarray) -> "Index": ...

    def __getitem__(self, key: Key):
        out = self._values[key]
        
        # scalar
        if np.isscalar(out):
            return out
        
        # masque booléen
        if isinstance(key, np.ndarray) and key.dtype == bool:
            return Index(out)
        
        # slice / fancy indexing
        return Index(out)
    
    def to_numpy(self) -> np.ndarray:
        return self._values.copy()
    
    def __repr__(self) -> str:
        return f"Index({self._values.tolist()})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Index):
            return False
        return bool(np.array_equal(self._values, other._values))
    

