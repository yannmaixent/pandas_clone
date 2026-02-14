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

def test_series_reindex_introduces_nan():
    s= Series([10, 20, 30], Index(["a", "b", "c"]))
    new = Index(["a", "c", "d"])
    r = s.reindex(new)
    assert np.array_equal(r.index.to_numpy(), new.to_numpy())
    assert np.isnan(r.to_numpy()[2]) # label "d" missing


def test_series_add_aligns_by_index():
    s1 = Series([10, 20, 30], Index(["a", "b", "c"]))
    s2 = Series([1, 2, 3], Index(["b", "c", "d"]))

    r = s1 + s2
    assert r.index == Index(["a", "b", "c", "d"])

    vals = r.to_numpy()
    assert np.isnan(vals[0])    # "a" missing in s2
    assert vals[1] == 21.0      # 20 + 1
    assert vals[2] == 32.0      # 30 + 2
    assert np.isnan(vals[3])    # "d" missing in s1

def test_series_loc():
    s = Series([10, 20, 30], Index(["a", "b", "c"]))
    assert s.loc["b"] == 20.0

def test_series_iloc():
    s = Series([10, 20, 30])
    assert s.iloc[1] == 20.0
