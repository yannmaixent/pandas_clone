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
    
    def reindex(self, new_index: Index) -> "Series":
        """
        Align values to new_index. Missing labels become NaN
        """
        old_pos = self.index.position_map()
        out = np.full(len(new_index), np.nan, dtype=float)

        for j, label in enumerate(new_index):
            i = old_pos.get(label)
            if i is not None:
                out[j] = float(self._values[i])
        
        return Series(out, new_index)
    
    def __add__(self, other:"Series") -> "Series":
        """
        Aligned addition; union of indexes + NaN for missing.
        """
        if not isinstance(other, Series):
            return NotImplemented
        
        new_index = self.index.union(other.index)
        a = self.reindex(new_index)
        b = other.reindex(new_index)

        # numpy addition: NaN propagates naturally
        return Series(a._values + b._values, new_index)
    
    @property
    def loc(self):
        return _LocIndexer(self)
    
    @property
    def iloc(self):
        return _IlocIndexer(self)


class _LocIndexer:
    def __init__(self, series):
        self._series = series
    
    def __getitem__(self, key):
        pos =self._series.index.position_map().get(key)
        if pos is None:
            raise KeyError(key)
        return float(self._series._values[pos])
    
class _IlocIndexer:
    def __init__(self, series):
        self._series = series
    
    def __getitem__(self, key):
        return float(self._series._values[key])

