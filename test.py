import itertools
from re import A
import string

l = [[], [], [1, 2, 3]]

print(list(itertools.permutations([[1, 2], [3], []], 3)))


def string_repr(state):
    pegs = 3
    discs = 3
    s = ["" for _ in range(discs)]

    for i in range(len(state)):
        for l in state[i]:
            s[l - 1] = chr(i+97)
    return tuple(s)


def generate_dict():
    pegs = 3
    discs = 3

    s = list(string.ascii_lowercase[:pegs])

    d = {}

    product = [p for p in itertools.product(s, repeat=3)]
    a = 0
    for p in product:
        d[p] = a
        a += 1
    return d


s = string_repr(l)
print(s)
d = generate_dict()
print(d[s])
print(d)
