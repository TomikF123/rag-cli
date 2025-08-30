import os
from builtins import range


# print(list(range(10)))
# print(list(enumerate(list(range(10)))))
func = lambda x: x + 1


def func2(x):
    return x + 2


# print(*(map(func2, list(range(10)))))

(*x,) = map(lambda x: x, list(range(10)))
(*y,) = map(lambda x: x if x % 2 == 1 else None, list(range(12)))
even_nums = [x for x in range(12) if x % 2 == 0]
even_nums2 = filter(lambda x: x % 2 == 0, x)
print(type(x), x)
print(*even_nums2, "EVEN NUMS 2")
print(*x, y)
nested_list = [[1, 2, 3], [3, 4, 5]]
print(nested_list[1][1])
import timeit
from itertools import accumulate

print(list(accumulate(x)))
from itertools import combinations

_combinations = combinations(x, r=2)
print(*_combinations)
print([*zip([1, 2, 3, 4], [1, 2, 3, 4])])
