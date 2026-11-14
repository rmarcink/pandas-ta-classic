"""Numba njit decorator with graceful fallback.

When numba is installed (``pip install numba``), re-exports the real
``njit``.  Otherwise provides a no-op decorator so that the same
``@njit(cache=True)`` syntax works everywhere without ImportError.
"""

import inspect
import math
import os
from fractions import Fraction


def fma(a: float, b: float, c: float) -> float:
    """``a * b + c`` rounded once, as C's ``fma``; TA-Lib's recursions use it.

    ``math.fma`` (Python 3.13+) does it in C. Before 3.13 the exact sum of
    fractions is rounded once by ``float()``. A non-finite operand, an exact
    zero (whose sign the fractions drop) and an overflow take the IEEE result
    on both paths, as C's fma and the compiled path return it: ``math.fma``
    raises ``OverflowError`` where they read inf. Under numba the call
    compiles to LLVM's ``fma`` (below).
    """
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
        return a * b + c
    if _math_fma is not None:
        try:
            return _math_fma(a, b, c)
        except OverflowError:
            return math.inf if Fraction(a) * Fraction(b) + Fraction(c) > 0 else -math.inf
    exact = Fraction(a) * Fraction(b) + Fraction(c)
    if exact == 0:
        return a * b + c  # the product is exactly -c, so this is exact and signs the zero as IEEE does
    try:
        return float(exact)
    except OverflowError:
        return math.inf if exact > 0 else -math.inf


_math_fma = getattr(math, "fma", None)

try:
    from numba import njit as _numba_njit
except ImportError:

    def njit(*args, **kwargs):  # type: ignore[misc]
        """No-op decorator mimicking ``numba.njit``."""

        def _wrap(f):
            return f

        return _wrap if not args or not callable(args[0]) else args[0]

else:

    def njit(*args, **kwargs):  # type: ignore[misc]
        """``numba.njit``, without the on-disk cache for code imported from a zip archive.

        numba places the cache of a function whose source is not a file in the
        user cache directory, ignoring ``NUMBA_CACHE_DIR``, and raises
        ``PermissionError`` on the first call when that directory is read-only.
        The cache only saves compile time, so such functions compile in memory.
        """

        def decorate(f):
            options = dict(kwargs)
            if options.get("cache") and not os.path.isfile(inspect.getsourcefile(f) or ""):
                options["cache"] = False
            return _numba_njit(**options)(f)

        return decorate(args[0]) if args and callable(args[0]) else decorate

    from numba.core import types as _types
    from numba.extending import intrinsic as _intrinsic, overload as _overload

    @_intrinsic
    def _llvm_fma(typingctx, a, b, c):  # type: ignore[no-untyped-def]
        def codegen(context, builder, signature, args):  # type: ignore[no-untyped-def]
            return builder.fma(*args)

        return _types.float64(_types.float64, _types.float64, _types.float64), codegen

    @_overload(fma)
    def _fma_nopython(a, b, c):  # type: ignore[no-untyped-def]
        if all(isinstance(t, _types.Float) for t in (a, b, c)):
            return lambda a, b, c: _llvm_fma(float(a), float(b), float(c))  # type: ignore[call-arg]  # numba supplies typingctx
        return None


__all__ = ["fma", "njit"]
