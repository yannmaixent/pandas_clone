import numpy as np
from pandas_clone.core import Index

def test_index_len():
    idx = Index([1, 2, 3])
    assert len(idx) == 3

def test_index_getitem_scalar():
    idx = Index(["a", "b", "c"])
    assert idx[1] == "b"

def test_index_slice():
    idx = Index([10, 20, 30])
    assert idx[1:] == Index([20, 30])