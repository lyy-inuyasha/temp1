from typing import Iterable

def make_pair(size):
    return size if isinstance(size, Iterable) else (size, size, size)
