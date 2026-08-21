"""Whether ``@njit`` compiles is decided once, at import time.

These tests therefore run each configuration in a fresh subprocess: numba
present, numba absent, and numba installed but unimportable. Only the last one
is expected to warn -- running without numba is supported and must stay quiet.
"""

import subprocess
import sys
from pathlib import Path
from unittest import TestCase

from pandas_ta_classic.utils._njit import NUMBA_ACTIVE, NUMBA_ERROR, njit

REPO_ROOT = Path(__file__).resolve().parents[1]

# Reports NUMBA_ACTIVE / NUMBA_ERROR on stdout; any warning lands on stderr.
_PROBE = "from pandas_ta_classic.utils._njit import NUMBA_ACTIVE, NUMBA_ERROR, njit; print(NUMBA_ACTIVE, njit.__module__)"

# Blocks the real numba the way a machine without it behaves at import time.
_BLOCK_NUMBA = "import sys; sys.modules['numba'] = None; "


def _run(code: str, extra_path: Path | None = None) -> subprocess.CompletedProcess:
    env = {**dict(__import__("os").environ)}
    if extra_path is not None:
        env["PYTHONPATH"] = str(extra_path)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT, env=env)


class TestNjitFallback(TestCase):

    def test_fallback_decorator_supports_both_call_forms(self):
        """The no-op has to accept ``@njit`` and ``@njit(cache=True)`` alike.

        Runs in a numba-blocked subprocess: with numba active the real decorator
        returns a dispatcher, so the fallback is only reachable that way.
        """
        code = _BLOCK_NUMBA + (
            "from pandas_ta_classic.utils._njit import njit\n"
            "f = lambda x: x\n"
            "assert njit(f) is f, 'bare form did not return the function'\n"
            "assert njit(cache=True)(f) is f, 'parametrised form did not return the function'\n"
            "print('both forms ok')"
        )
        proc = _run(code)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("both forms ok", proc.stdout)

    def test_state_flags_agree_with_the_decorator(self):
        if NUMBA_ACTIVE:
            self.assertIsNone(NUMBA_ERROR)
            self.assertNotEqual(njit.__module__, "pandas_ta_classic.utils._njit")
        else:
            self.assertIsInstance(NUMBA_ERROR, str)
            self.assertEqual(njit.__module__, "pandas_ta_classic.utils._njit")

    def test_absent_numba_falls_back_silently(self):
        proc = _run(_BLOCK_NUMBA + _PROBE)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("False pandas_ta_classic.utils._njit", proc.stdout)
        self.assertNotIn("RuntimeWarning", proc.stderr)

    def test_broken_numba_warns(self):
        """An installed-but-unimportable numba must not degrade silently."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "numba"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("raise ImportError('Numba needs NumPy 2.0 or less. Got NumPy 2.5.')\n")

            proc = _run(_PROBE, extra_path=Path(tmp))

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("False pandas_ta_classic.utils._njit", proc.stdout)
        self.assertIn("RuntimeWarning", proc.stderr)
        self.assertIn("numba is installed but could not be imported", proc.stderr)
        self.assertIn("Got NumPy 2.5.", proc.stderr)  # the underlying cause is passed through
