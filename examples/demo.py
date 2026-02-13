from pandas_clone.core import Series, Index, DataFrame

print("DEMO START")

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

s1 = Series([10, 20, 30], Index(["a", "b", "c"]))
s2 = Series([1, 2, 3], Index(["b", "c", "d"]))


print("s1:", s1)
print("s2", s2)
print("s1+s2:", s1 + s2)
print("DEMO END")
