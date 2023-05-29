# %%
class TestIterator:
    def __init__(self):
        pass

    def __iter__(self):
        self.a = 1
        return self

    def __next__(self):
        x = self.a
        self.a += 1
        return x
# %%
test = TestIterator()
a = iter(test)

print(next(a))
print(next(a))
print(next(a))
print(next(a))
# %%
def foo():
    for i in range(10):
        yield i

g = foo()
print([i for i in g]) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print([i for i in g]) # []

# %%
