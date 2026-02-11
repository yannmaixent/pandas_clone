from pandas_clone.core import Series, Index, DataFrame

idx = Index(["a", "b", "c"])
s = Series([1, 2, 3], idx)

df = DataFrame(
    {
        "price": [10, 20, 30],
        "volume": [100, 200, 300],
    }
)

print(s.mean())
print(s[1:])

print(df)
print(df["price"])
print(df[1:])