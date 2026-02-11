import numpy as np
from pandas_clone.core import DataFrame, Series

def test_dataframe_creation():
    df = DataFrame(
        {
            "price": [10, 20, 30],
            "volume": [100, 200, 300],
        }
    )

    assert len(df) == 3
    assert set(df.columns) == {"price", "volume"}


def test_column_selection():
    df = DataFrame(
        {
            "price": [10, 20, 30],
            "volume": [100, 200, 300],
        }
    )

    price = df["price"]
    assert isinstance(price, Series)
    assert price.sum() == 60.

def test_row_slice():
    df = DataFrame(
        {
            "price": [10, 20, 30],
            "volume": [100, 200, 300],
        }
    )

    sub = df[1:]
    assert len(sub) == 2
    assert np.array_equal(sub["price"].to_numpy(), np.array([20.0, 30.0]))