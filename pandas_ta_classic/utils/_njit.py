"""Numba njit decorator with graceful fallback.

When numba is installed (``pip install numba``), re-exports the real
``njit``.  Otherwise provides a no-op decorator so that the same
``@njit(cache=True)`` syntax works everywhere without ImportError.

Running without numba is a supported configuration and stays silent. An
*installed but unimportable* numba is not: it is normally a NumPy version
mismatch, it is user-fixable, and left unreported it silently drops every
``@njit`` indicator onto the plain-Python path at a fraction of the speed.
:data:`NUMBA_ACTIVE` and :data:`NUMBA_ERROR` record which case applies.
"""

from __future__ import annotations

import warnings
from importlib.util import find_spec
from typing import Optional

#: True when ``@njit`` really compiles; False when it is the no-op fallback.
NUMBA_ACTIVE: bool
#: The import failure that caused the fallback, or None when numba is active or absent.
NUMBA_ERROR: Optional[str]


def _numba_installed() -> bool:
    """Whether a numba distribution is present, without importing it.

    ``find_spec`` raises rather than returns for a module that has been
    blocked or half-imported (``sys.modules["numba"] = None``, as benchmark
    harnesses do to force the fallback path); treat that as "not installed"
    so the caller sees the quiet, supported configuration.
    """
    try:
        return find_spec("numba") is not None
    except (ImportError, ValueError):
        return False


try:
    from numba import njit

    NUMBA_ACTIVE, NUMBA_ERROR = True, None
except ImportError as exc:
    NUMBA_ACTIVE, NUMBA_ERROR = False, str(exc)

    if _numba_installed():
        warnings.warn(
            f"numba is installed but could not be imported, so every @njit indicator falls back to plain Python "
            f"(often 20-190x slower). Fix or remove the numba install to silence this. Import error: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    def njit(*args, **kwargs):  # type: ignore[misc]
        """No-op decorator mimicking ``numba.njit``."""

        def _wrap(f):
            return f

        return _wrap if not args or not callable(args[0]) else args[0]


__all__ = ["njit", "NUMBA_ACTIVE", "NUMBA_ERROR"]
