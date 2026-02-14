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


def test_series_sub_and_mul():
    s1 = Series([10, 20, 30], Index(["a", "b", "c"]))
    s2 = Series([1, 2, 3], Index(["b", "c", "d"]))

    r_sub = s1 - s2
    r_mul = s1 * s2

    assert r_sub.index == Index(["a", "b", "c", "d"])
    assert r_mul.index == Index(["a", "b", "c", "d"])

    vals_sub = r_sub.to_numpy()
    vals_mul = r_mul.to_numpy()

    #a missing in s2 -> NaN
    assert np.isnan(vals_sub[0]) and np.isnan(vals_mul[0])

    # b: 20 - 1 , 20*1
    assert vals_sub[1] == 19.0
    assert vals_mul[1] == 20.0

    # c: 30-2, 30*2
    assert vals_sub[2] == 28.0
    assert vals_mul[2] == 60.0

    # d : missing in s1 -> NaN
    assert np.isnan(vals_sub[3]) and np.isnan(vals_mul[3])