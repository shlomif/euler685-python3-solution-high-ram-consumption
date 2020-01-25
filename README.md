# About

This a demo repository for a problem I am having with [cpython3](https://en.wikipedia.org/wiki/CPython)
and [pypy3](https://en.wikipedia.org/wiki/PyPy) where the [program](./685-v1.py) consumes
a lot of RAM quickly on [Fedora 31](https://en.wikipedia.org/wiki/Fedora_%28operating_system%29).

It is a work-in-progress solution to [this project euler challenge](https://projecteuler.net/problem=685).

# To reproduce

Run `gmake run`.

# How much is "high RAM consumption"?

You can find a [screenshot](./images/high-consume-evidence-on-8-GB-computer/pypy.png) of htop,
and a similar [text transcript of the terminal](./images/high-consume-evidence-on-8-GB-computer/pypy.txt),
where it shows it consumes over 1-2 GB of RAM well before reaching k=1,000 (out of max k=10,000).

## Attempts to fix:

I tried using https://pypi.org/project/memory-profiler/ but it slowed down the code and did not
display a significant memory increment.

I tried using https://docs.python.org/3/library/tracemalloc.html but its reported allocations were small.
