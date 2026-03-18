"""Benchmark hot-loop indicators."""
import time
import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000
close = pd.Series(np.cumsum(np.random.randn(n)) + 100, name="close")
high = close + np.abs(np.random.randn(n))
low = close - np.abs(np.random.randn(n))
open_ = close + np.random.randn(n) * 0.5
volume = pd.Series(np.random.randint(1000, 100000, n), dtype=float, name="volume")

import pandas_ta_classic as ta

BENCHMARKS = {
    "psar": (ta.psar, (high, low, close), {}),
    "qqe": (ta.qqe, (close,), {}),
    "hwc": (ta.hwc, (close,), {}),
    "rsx": (ta.rsx, (close,), {}),
    "lrsi": (ta.lrsi, (close,), {}),
    "fisher": (ta.fisher, (high, low,), {}),
    "alma": (ta.alma, (close,), {}),
    "supertrend": (ta.supertrend, (high, low, close), {}),
    "hwma": (ta.hwma, (close,), {}),
    "mcgd": (ta.mcgd, (close,), {}),
    "ssf": (ta.ssf, (close,), {}),
    "fwma": (ta.fwma, (close,), {}),
    "pwma": (ta.pwma, (close,), {}),
    "sinwma": (ta.sinwma, (close,), {}),
    "cg": (ta.cg, (close,), {}),
}

RUNS = 20

print(f"{'Indicator':<15} {'Mean (ms)':>10} {'Min (ms)':>10} {'Runs':>5}")
print("-" * 45)

for name, (func, args, kwargs) in BENCHMARKS.items():
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    mean_ms = np.mean(times)
    min_ms = np.min(times)
    print(f"{name:<15} {mean_ms:>10.2f} {min_ms:>10.2f} {RUNS:>5}")
