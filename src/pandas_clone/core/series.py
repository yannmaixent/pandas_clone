from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Union, overload

import numpy as np

from .index import Index

Key = Union[int, slice, np.ndarray]

@dataclass
class Series:
    """
    Minimal Pandas-like Series: 1D data + Index.
    Backed by a Numpy array
    """

    _values: np.ndarray
    index: Index

    def __init__(self, data: Iterable, index:Index | None = None):
        values = np.asarray(list(data), dtype=float)
        if values.ndim !=1:
            raise ValueError("Series data must be 1-dimensional")
        
        if index is None:
            index = Index(range(len(values)))

        if len(index) != len(values):
            raise ValueError("Index and data must be the same lenght")
        
        self._values = values
        self.index = index

    def __len__(self) -> int:
        return int(self._values.size)
    
    @overload
    def __getitem__(self, key: int) -> float: ...
    @overload
    def __getitem__(self, key: slice) -> "Series": ...
    @overload
    def __getitem__(self, key: np.ndarray) -> "Series": ...

    def __getitem__(self, key: Key):
        out = self._values[key]
        if np.isscalar(out):
            return float(out)
        return Series(out, self.index[key])
    
    def sum(self) -> float:
        return float(np.sum(self._values))

    def mean(self) -> float:
        return float(np.mean(self._values))
    
    def apply(self, func: Callable[[float], float]) -> "Series":
        # simple, explicit, predictable
        out = np.array([func(float(x)) for x in self._values], dtype=float)
        return Series(out, self.index)
    
    def to_numpy(self) -> np.ndarray:
        return self._values.copy()
    
    def __repr__(self) -> str:
        return f"Series(values={self._values.tolist()}, index={self.index})"
    