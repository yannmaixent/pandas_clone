from pandas_clone.core import Series, Index

idx = Index(["a", "b", "c"])
s = Series([1, 2, 3], idx)

print(s.mean())
print(s[1:])