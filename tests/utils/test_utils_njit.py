"""utils/_njit.fma: a * b + c rounded once, on every path the indicators take."""

import math
import random
from fractions import Fraction

import pytest

from pandas_ta_classic.utils import _njit
from pandas_ta_classic.utils._njit import fma, njit

# (a, b, c) and the IEEE result C's fma returns for it
EDGE = [
    ((0.1, 10.0, -1.0), 5.551115123125783e-17),  # rounded once, where a * b + c reads 0.0
    ((-0.0, 1.0, -0.0), -0.0),  # an exact zero keeps IEEE's sign
    ((0.0, -1.0, 0.0), 0.0),
    ((2e154, 1e154, -1.5e308), float(Fraction(2e154) * Fraction(1e154) + Fraction(-1.5e308))),  # the product overflows, the sum does not
    ((1e308, 10.0, 0.0), math.inf),  # the result overflows: math.fma raises here
    ((-1e308, 10.0, 0.0), -math.inf),
    ((math.inf, 1.0, 1.0), math.inf),
    ((math.inf, 0.0, 1.0), math.nan),  # inf * 0
    ((math.nan, 1.0, 1.0), math.nan),
    ((5e-324, 0.5, 5e-324), 1e-323),  # subnormal: 1.5 units of the last place, a tie, rounds to the even 2
]


def _same(a, b):
    return (math.isnan(a) and math.isnan(b)) or (a == b and math.copysign(1, a) == math.copysign(1, b))


def _random(n=20000):
    rng = random.Random(7)
    return [(rng.uniform(-1e3, 1e3), rng.uniform(-1e3, 1e3), rng.uniform(-1e6, 1e6)) for _ in range(n)]


@pytest.mark.parametrize("path", ["math.fma or fractions", "fractions"])
def test_fma_returns_the_ieee_result_at_the_edges(path, monkeypatch):
    if path == "fractions":
        monkeypatch.setattr(_njit, "_math_fma", None)
    bad = [(t, fma(*t), want) for t, want in EDGE if not _same(fma(*t), want)]
    assert not bad, bad


def test_the_compiled_path_returns_the_ieee_result_at_the_edges():
    compiled = njit(lambda a, b, c: fma(a, b, c))
    bad = [(t, compiled(*t), want) for t, want in EDGE if not _same(compiled(*t), want)]
    assert not bad, bad


@pytest.mark.skipif(not hasattr(math, "fma"), reason="math.fma is the reference; Python 3.13+")
def test_the_fraction_path_and_the_compiled_path_round_as_math_fma(monkeypatch):
    compiled = njit(lambda a, b, c: fma(a, b, c))
    monkeypatch.setattr(_njit, "_math_fma", None)
    bad = [t for t in _random() if not (fma(*t) == compiled(*t) == math.fma(*t))]
    assert not bad, bad[:5]
