import numpy as np
from pandas_clone.core import Series, Index

def test_series_sum_mean():
    s = Series([1, 2, 3])
    assert s.sum() == 6.0
    assert s.mean() == 2.0

def test_series_with_custom_index():
    idx = Index(["a", "b", "c"])
    s = Series([10, 20, 30], idx)
    assert len(s) == 3
    assert s.index == idx


def test_series_slice_keeps_index():
    s = Series([1, 2, 3])
    sub = s[1:]
    assert sub.sum() == 5.0
    assert len(sub.index) == 2


def test_series_mask():
    s = Series([1, 2, 3, 4])
    mask = np.array([True, False, True, False])
    sub = s[mask]
    assert np.array_equal(sub.to_numpy(), np.array([1.0, 3.0]))

    def test_series_apply():
        s = Series([1, 2, 3])
        s2 = s.apply(lambda x: x *10)
        assert np.array_equal(s2.to_numpy(), np.array([10.0, 20.0, 30.0]))
