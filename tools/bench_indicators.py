"""Benchmark indicator runtime across the three execution paths, with profiling and complexity fits.

A reproducible harness for the performance figures cited in ``CHANGELOG.md`` and
a general-purpose way to find the current slowest indicators. It builds a
fixed-seed synthetic OHLCV frame, warms each indicator once (so numba JIT
compilation and import side effects are excluded), then reports the median
wall-clock time over several runs -- slowest first, so bottlenecks surface on
their own without any hard-coded list of "interesting" indicators.

Every run measures three variants of each indicator:

``plain``
    Pure Python/pandas path. ``numba`` is blocked at import time, so every
    ``@njit`` decorator falls back to the no-op in ``utils/_njit.py``. This is
    what a user without numba installed gets, and numba is not a hard dependency.
``numba``
    The same code with numba installed and JIT compilation active.
``talib``
    The ``talib=True`` passthrough, for indicators that expose the parameter and
    only when TA-Lib is importable.

Because numba is an import-time decision (``from numba import njit`` runs once,
when the package is first imported), the variants cannot share a process. This
script re-executes itself as two workers -- one with numba blocked, one without
-- and merges their JSON results. That is also why ``pandas_ta_classic`` is
imported lazily instead of at module scope.

Usage::

    python tools/bench_indicators.py                     # every indicator, slowest first
    python tools/bench_indicators.py --top 20            # only the 20 slowest
    python tools/bench_indicators.py rsi macd jma        # specific names
    python tools/bench_indicators.py --rows 20000        # a larger frame
    python tools/bench_indicators.py --profile 25        # cProfile each variant separately
    python tools/bench_indicators.py --scaling jma psar  # derive O() from a size sweep
"""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
import time
import warnings
from math import log
from statistics import median
from typing import Any

import numpy as np
import pandas as pd

# ``pandas_ta_classic`` is deliberately NOT imported here: the numba-blocked
# worker has to poison ``sys.modules`` before the package -- and therefore
# ``utils/_njit`` -- is first imported. ``_import_ta()`` sets this global.
ta: Any = None

#: Which variants each worker process measures. The numba worker also covers
#: ``talib``, whose passthrough never reaches the njit code paths.
WORKERS = {"plain": ("plain",), "numba": ("numba", "talib")}
VARIANTS = ("plain", "numba", "talib")

#: Candidate growth models for the ``--scaling`` fit.
MODELS = {
    "O(1)": lambda n: 1.0,
    "O(n)": lambda n: float(n),
    "O(n log n)": lambda n: n * log(n),
    "O(n^2)": lambda n: float(n) * n,
}


def _import_ta(block_numba: bool):
    """Import the package, optionally with numba made unimportable first."""
    global ta
    if block_numba:
        sys.modules["numba"] = None  # type: ignore[assignment]  # any ``from numba import njit`` below now raises ImportError
    import pandas_ta_classic

    ta = pandas_ta_classic
    return ta


def _jit_state() -> tuple[bool, str | None]:
    """Whether ``@njit`` really compiles, and the import error when it does not."""
    from pandas_ta_classic.utils._njit import NUMBA_ACTIVE, NUMBA_ERROR

    return NUMBA_ACTIVE, NUMBA_ERROR


def all_indicator_names() -> list[str]:
    """Every indicator name discovered across the category registry."""
    return sorted({n for v in ta.Category.values() for n in v})


def make_ohlcv(rows: int, seed: int = 42) -> pd.DataFrame:
    """Fixed-seed random-walk OHLCV frame with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(rows))
    high = close + rng.random(rows) * 2
    low = close - rng.random(rows) * 2
    open_ = close + rng.standard_normal(rows)
    volume = rng.integers(1_000, 100_000, rows).astype(float)
    idx = pd.date_range("2000-01-01", periods=rows, freq="min")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _time(fn, repeats: int) -> float:
    """Median wall-clock time in milliseconds over *repeats* runs."""
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return median(samples) * 1000.0


def _has_talib_param(name: str) -> bool:
    fn = getattr(ta, name, None)
    if fn is None:
        return False
    try:
        return "talib" in inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return False


def _callable_for(df: pd.DataFrame, name: str, variant: str):
    """Return a zero-arg callable for *name* in *variant*, or a status string explaining why not."""
    acc = getattr(df.ta, name, None)
    if acc is None:
        return "no-accessor"
    if variant != "talib":
        return acc
    if not ta.Imports["talib"]:
        return "no-talib"
    if not _has_talib_param(name):
        return "no-param"
    return lambda: acc(talib=True)


def bench(names: list[str], rows: int, repeats: int, variants: tuple[str, ...]) -> list[dict]:
    """Median runtime of every *name* in every *variant* on a *rows*-bar frame."""
    df = make_ohlcv(rows)
    results = []
    for variant in variants:
        for name in names:
            fn = _callable_for(df, name, variant)
            if isinstance(fn, str):
                results.append({"name": name, "variant": variant, "status": fn})
                continue
            try:
                fn()  # warmup (also triggers numba JIT compilation)
                fn()
                ms = _time(fn, repeats)
            except Exception as exc:  # noqa: BLE001
                results.append({"name": name, "variant": variant, "status": f"err:{type(exc).__name__}"})
                continue
            results.append({"name": name, "variant": variant, "status": "ok", "ms": ms})
    return results


def profile(names: list[str], rows: int, top: int, variant: str) -> str:
    """cProfile one variant of the whole indicator set once and rank hot functions.

    Surfaces per-call overhead (Python callbacks, rolling.apply, per-bar loops)
    that a pure wall-clock ranking under-weights. High ``ncalls`` on a project
    function is a vectorise/njit candidate.
    """
    import cProfile
    import io
    import pstats

    df = make_ohlcv(rows)
    fns = [_callable_for(df, n, variant) for n in names]
    fns = [f for f in fns if not isinstance(f, str)]
    for f in fns:  # warm up (numba JIT + import side effects out of the profile)
        try:
            f()
        except Exception:  # noqa: BLE001
            pass

    pr = cProfile.Profile()
    pr.enable()
    for f in fns:
        try:
            f()
        except Exception:  # noqa: BLE001
            pass
    pr.disable()

    out = io.StringIO()
    stats = pstats.Stats(pr, stream=out)
    for sort_key in ("tottime", "ncalls"):
        out.write(f"\n=== [{variant}] top {top} by {sort_key} ===\n")
        stats.sort_stats(sort_key).print_stats(top)
    return out.getvalue()


def fit_complexity(sizes: list[int], times: list[float]) -> dict:
    """Derive an empirical growth order from a size sweep.

    Two independent readings of the same samples: ``exp`` is the slope of a
    least-squares line through ``log(size)`` vs ``log(time)``, and ``model`` is
    whichever of :data:`MODELS` reproduces the measurements with the smallest
    relative error once scaled. A large ``err`` means the timings are dominated
    by fixed overhead or noise and neither reading should be trusted.
    """
    if len(sizes) < 2 or any(t <= 0 for t in times):
        return {"exp": None, "model": None, "err": None}

    xs = [log(n) for n in sizes]
    ys = [log(t) for t in times]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    exp = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom else None

    best_name, best_err = None, None
    for label, f in MODELS.items():
        pred = [f(s) for s in sizes]
        scale = sum(t / p for t, p in zip(times, pred)) / n  # scale factor in relative (not absolute) space
        err = (sum(((scale * p - t) / t) ** 2 for p, t in zip(pred, times)) / n) ** 0.5
        if best_err is None or err < best_err:
            best_name, best_err = label, err
    return {"exp": exp, "model": best_name, "err": best_err}


def scaling(names: list[str], sizes: list[int], repeats: int, variants: tuple[str, ...]) -> list[dict]:
    """Time every variant at each size in *sizes* and fit a growth order to the result."""
    frames = {s: make_ohlcv(s) for s in sizes}
    results = []
    for variant in variants:
        for name in names:
            times, ok = [], True
            for size in sizes:
                fn = _callable_for(frames[size], name, variant)
                if isinstance(fn, str):
                    results.append({"name": name, "variant": variant, "status": fn})
                    ok = False
                    break
                try:
                    fn()  # warm per size: JIT compiles once, but caches and buffers do not carry over
                    times.append(_time(fn, repeats))
                except Exception as exc:  # noqa: BLE001
                    results.append({"name": name, "variant": variant, "status": f"err:{type(exc).__name__}"})
                    ok = False
                    break
            if ok:
                results.append({"name": name, "variant": variant, "status": "ok", "times": times, **fit_complexity(sizes, times)})
    return results


def _parse_sizes(spec: str) -> list[int]:
    sizes = sorted({int(s) for s in spec.split(",") if s.strip()})
    if len(sizes) < 2:
        raise ValueError(f"--scaling-rows needs at least two distinct sizes, got {spec!r}")
    return sizes


def _run_worker(worker: str, names: list[str], args) -> dict:
    """Re-execute this file as *worker* and return its JSON payload."""
    cmd = [sys.executable, __file__, "--worker", worker, "--rows", str(args.rows), "--repeats", str(args.repeats)]
    if args.profile is not None:
        cmd += ["--profile", str(args.profile)]
    if args.scaling:
        cmd += ["--scaling", "--scaling-rows", args.scaling_rows, "--scaling-repeats", str(args.scaling_repeats)]
    proc = subprocess.run(cmd, input=json.dumps(names), capture_output=True, text=True)
    empty = {"rows": [], "scaling": [], "profiles": {}, "numba": None, "numba_error": None, "talib": None}
    if proc.returncode != 0:
        print(f"worker {worker} failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}", file=sys.stderr)
        return empty
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f"worker {worker} produced unparseable output:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}", file=sys.stderr)
        return empty


def worker_main(args) -> int:
    """Entry point inside a spawned worker: measure this worker's variants, emit JSON on stdout."""
    variants = WORKERS[args.worker]
    _import_ta(block_numba=(args.worker == "plain"))
    names = json.loads(sys.stdin.read()) or all_indicator_names()

    payload = {
        "rows": bench(names, args.rows, args.repeats, variants),
        "scaling": scaling(names, _parse_sizes(args.scaling_rows), args.scaling_repeats, variants) if args.scaling else [],
        "profiles": {v: profile(names, args.rows, args.profile, v) for v in variants} if args.profile is not None else {},
        "talib": bool(ta.Imports["talib"]),
    }
    payload["numba"], payload["numba_error"] = _jit_state()
    json.dump(payload, sys.stdout)
    return 0


def _index(rows: list[dict]) -> dict:
    return {(r["name"], r["variant"]): r for r in rows}


def _cell(rec: dict | None) -> str:
    if rec is None:
        return "-"
    if rec["status"] != "ok":
        return rec["status"]
    return f"{rec['ms']:.3f}"


def _speedup(base: dict | None, other: dict | None) -> str:
    if not base or not other or base["status"] != "ok" or other["status"] != "ok" or not other["ms"]:
        return "-"
    return f"{base['ms'] / other['ms']:.1f}x"


def print_table(rows: list[dict], names: list[str], top: int | None) -> None:
    idx = _index(rows)
    ranked = [n for n in names if idx.get((n, "plain"), {}).get("status") == "ok"]
    ranked.sort(key=lambda n: idx[(n, "plain")]["ms"], reverse=True)
    ranked += [n for n in names if n not in ranked]  # unmeasurable ones last, in name order
    shown = ranked[:top] if top else ranked

    print(f"{'indicator':22} {'plain_ms':>10} {'numba_ms':>10} {'talib_ms':>10} {'plain/numba':>12} {'plain/talib':>12}")
    for name in shown:
        plain, numba, talib = (idx.get((name, v)) for v in VARIANTS)
        print(f"{name:22} {_cell(plain):>10} {_cell(numba):>10} {_cell(talib):>10} {_speedup(plain, numba):>12} {_speedup(plain, talib):>12}")


def print_scaling(rows: list[dict], sizes: list[int], names: list[str]) -> None:
    idx = _index(rows)
    print(f"\n=== complexity fit over rows={','.join(str(s) for s in sizes)} ===")
    print(f"{'indicator':22} {'variant':8} {'exponent':>9} {'model':>12} {'rel_err':>9}   times_ms")
    for name in names:
        for variant in VARIANTS:
            rec = idx.get((name, variant))
            if rec is None:
                continue
            if rec["status"] != "ok":
                print(f"{name:22} {variant:8} {rec['status']:>9}")
                continue
            exp = f"{rec['exp']:.2f}" if rec["exp"] is not None else "-"
            err = f"{rec['err']:.3f}" if rec["err"] is not None else "-"
            times = " ".join(f"{t:.2f}" for t in rec["times"])
            print(f"{name:22} {variant:8} {exp:>9} {rec['model'] or '-':>12} {err:>9}   {times}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="indicator names to benchmark (default: all)")
    parser.add_argument("--rows", type=int, default=5000, help="number of bars (default: 5000)")
    parser.add_argument("--repeats", type=int, default=7, help="timed runs per indicator (default: 7)")
    parser.add_argument("--top", type=int, default=None, help="print only the N slowest (default: all)")
    parser.add_argument(
        "--profile",
        type=int,
        nargs="?",
        const=25,
        default=None,
        metavar="N",
        help="cProfile every variant separately and print its top N hot functions by tottime and ncalls (default N: 25)",
    )
    parser.add_argument("--scaling", action="store_true", help="time each variant at several sizes and derive an empirical O()")
    parser.add_argument("--scaling-rows", default="2500,5000,10000,20000", help="comma-separated sizes for --scaling")
    parser.add_argument("--scaling-repeats", type=int, default=3, help="timed runs per size for --scaling (default: 3)")
    parser.add_argument("--worker", choices=sorted(WORKERS), default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore")

    if args.worker:
        return worker_main(args)

    sizes = _parse_sizes(args.scaling_rows) if args.scaling else []

    _import_ta(block_numba=False)
    names = args.names or all_indicator_names()
    print(f"rows: {args.rows}   repeats: {args.repeats}   indicators: {len(names)}   variants: {', '.join(VARIANTS)}")
    print("one worker process per variant group (numba is an import-time decision)\n")

    rows, scaling_rows, profiles, flags = [], [], {}, {}
    for worker in WORKERS:
        payload = _run_worker(worker, names, args)
        rows += payload["rows"]
        scaling_rows += payload["scaling"]
        profiles.update(payload["profiles"])
        flags[worker] = payload

    numba_worker = flags.get("numba", {})
    if numba_worker.get("numba") is False:
        reason = numba_worker.get("numba_error") or "not installed"
        print(f"warning: numba JIT is INACTIVE ({reason}) - the 'numba' column repeats the plain-Python path\n")
    if numba_worker.get("talib") is False:
        print("warning: TA-Lib is NOT installed - the 'talib' column is empty\n")

    print_table(rows, names, args.top)
    if args.scaling:
        print_scaling(scaling_rows, sizes, names)
    for text in profiles.values():
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
