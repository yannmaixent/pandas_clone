from __future__ import annotations

from typing import Dict, Iterable

from .index import Index
from .series import Series

class DataFrame:
    """
    Minimal Pandas-like DataFrame (column-oriented).
    """

    def __init__(self, data: Dict[str, Iterable], index: Index | None = None):
        if not isinstance(data, dict):
            raise TypeError("DataFrame data must be a dict of column -> iterable")
        
        if len(data) == 0:
            raise ValueError("DataFrame cannot be empty")
        
        # infer length
        first_col = next(iter(data.values()))
        length = len(list(first_col))

        if index is None:
            index = Index(range(length))

        if len(index) != length:
            raise ValueError("Index length must macth data length")
        
        self.index = index
        self._data: Dict[str, Series] = {}

        for col, values in data.items():
            series = Series(values, index)
            self._data[col] = series

    
    #-------------------------------------------
    # Basic API
    #-------------------------------------------

    def __len__(self) -> int:
        return len(self.index)
    
    @property
    def columns(self):
        return list(self._data.keys())
    
    def __getitem__(self, key):
        # Column selection
        if isinstance(key, str):
            return self._data[key]
        
        # Row slicing
        return DataFrame(
            {col: series[key].to_numpy() for col, series in self._data.items()},
            index=self.index[key],
        )
    
    def __repr__(self) -> str:
        col_str = ", ".join(self.columns)
        return f"DataFrame(columns=[{col_str}], index={self.index})"
    
