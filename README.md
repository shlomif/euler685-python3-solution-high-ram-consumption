# About

This a demo repository for a problem I am having with [cpython3](https://en.wikipedia.org/wiki/CPython)
and [pypy3](https://en.wikipedia.org/wiki/PyPy) where the [program](./685-v1.py) consumes
a lot of RAM quickly on [Fedora 31](https://en.wikipedia.org/wiki/Fedora_%28operating_system%29).

# To reproduce

Run `gmake run`.

## Attempts to fix:

I tried using https://pypi.org/project/memory-profiler/ but it slowed down the code and did not
display a significant memory increment.

I tried using https://docs.python.org/3/library/tracemalloc.html but its reported allocations were small.
