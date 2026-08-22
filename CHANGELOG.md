# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
* **backtesting.py integration** (PR #124 by @Adhyansinghgupta): Bridge function (`ta_bridge`), `SMACrossover` example strategy (`examples/backtesting_py_strategy.py`), and integration tutorial (`docs/tutorials/backtesting_py.md`). Added `backtesting` to `integration` optional dependencies.
* **backtrader integration** (`docs/tutorials/backtrader.md`, `examples/backtrader_strategy.py`): Precompute-then-feed pattern using dynamic `PandasData` subclass (`make_feed()`); covers single-output, multi-output (MACD), and OHLCV-dependent (ATR) indicators. Added `backtrader` to `integration` optional dependencies.
* **vectorbt integration tutorial** (`docs/tutorials/vectorbt.md`): Documented the `df.ta.tsignals()` → `vbt.Portfolio.from_signals()` integration pattern, including multi-output indicator handling and benchmark comparison.
* **Type stubs for `AnalysisIndicators`** (`pandas_ta_classic/core.pyi`): IDE autocomplete and mypy now see full method signatures for every indicator.
* **`ichimoku()` single-DataFrame return** (`as_dataframe=True`): opt in to a single DataFrame of the visible period instead of the legacy `(visible, span)` tuple. Pass `append_span=True` to also append the future-dated projected span rows. `append_span` is now a real parameter on the underlying `ichimoku()` function (previously accessor-only) and is forwarded by `df.ta.ichimoku()`.

### Changed
* **`signed_series` ~5.5× faster, lifting `obv`, `nvi`, `pvi`, `kvo`, `aobv` and `vp` (pure numpy)**: the shared sign-of-difference helper mapped its differences with two boolean assignments (`sign[sign > 0] = 1`, `sign[sign < 0] = -1`), each of which builds a mask, aligns it and copies the block. `np.sign` produces the same 1/0/-1 mapping in one pass and leaves NaN as NaN, which is what the two assignments did. Per 5000-bar call the helper drops `0.354 ms` → `0.064 ms`, and its callers follow: `obv` `0.33` → `0.12 ms`, `nvi` `0.64` → `0.41 ms`, `pvi` `0.64` → `0.42 ms`, `kvo` `1.55` → `1.26 ms`, `aobv` `2.59` → `1.97 ms`, `vp` `1.33` → `1.08 ms`. Output is **bit-for-bit identical** including dtype, name and the `-0.0`, all-NaN and integer-input edge cases. No numba involved.
* **Broken numba installs no longer degrade silently**: `utils/_njit.py` caught the `ImportError` and installed its no-op decorator without a word, so an installed-but-unimportable numba (usually a NumPy version mismatch, e.g. `Numba needs NumPy 2.0 or less`) was indistinguishable from no numba at all — every `@njit` indicator quietly ran the plain-Python path 20–190× slower. That case now emits a `RuntimeWarning` naming the underlying import error. Running *without* numba stays silent, as before. New `NUMBA_ACTIVE` / `NUMBA_ERROR` flags in `pandas_ta_classic.utils._njit` expose the state.
* **`tools/bench_indicators.py` benchmarks all three execution paths**: every run now reports plain (numba blocked at import), numba (JIT active) and TA-Lib passthrough side by side, using one worker subprocess per variant because numba is an import-time decision. `--profile N` cProfiles each variant separately; `--scaling` sweeps several frame sizes and derives an empirical growth order (log-log exponent plus a best-fit O(1)/O(n)/O(n log n)/O(n²) model).
* **`td_seq` ~3500× faster (numba)**: the Tom DeMark Sequential consecutive-run count replaced `rolling(13).apply(python_callback)` with a numba `@njit` loop (`~1912 ms` → `~0.5 ms` per 5000-bar call, the single largest indicator bottleneck). Bit-for-bit identical; falls back to a plain Python loop without numba. Reproduce with `python tools/bench_indicators.py`.
* **`hilo` ~250× faster (numba)**: the Gann HiLo per-bar `.iloc[]` state loop moved into a numba `@njit` kernel over numpy arrays (`~130 ms` → `~0.5 ms` per 5000-bar call). Output is identical for all `mamode`s; falls back to a plain Python loop without numba.
* **`ht_*` family ~110–145× faster (numba)**: the shared Ehlers Hilbert Transform per-bar loop (`cycles/_hilbert.py::_hilbert_transform_loop`) is now a numba `@njit(cache=True)` kernel. It backs all six `ht_dcperiod`, `ht_dcphase`, `ht_phasor`, `ht_sine`, `ht_trendmode` and `ht_trendline`, which together accounted for `~456 ms` of the `~1021 ms` plain-Python cost of a full 224-indicator sweep — by far the largest remaining bottleneck. Per 5000-bar call: `ht_sine` `99.2 ms` → `0.70 ms` (142×), `ht_phasor` `80.1 ms` → `0.72 ms` (111×), `ht_trendmode` `79.6 ms` → `0.67 ms` (120×), `ht_dcperiod` `79.3 ms` → `0.57 ms` (139×), `ht_trendline` `76.9 ms` → `0.53 ms` (145×), `ht_dcphase` `75.7 ms` → `0.54 ms` (140×). The kernel body is unchanged — only the decorator was added — so output is **bit-for-bit identical** on both paths (verified across 200/1000/5000-bar frames for all 24 output columns of the six indicators); without numba it stays the same plain Python loop as before. First call in a fresh process pays a one-off JIT compile of `~1.3 s`, cached to disk by `cache=True`.
* **`jma` ~130× faster (numba)**: the Jurik Moving Average per-bar recursion (volatility bands + 3-stage filter) moved into a numba `@njit` loop with scalar coefficients precomputed by the caller (`~50 ms` → `~0.4 ms` per 5000-bar call). The trailing-window average uses `np.mean` in place of the original `np.average`; for an unweighted call numpy computes these bit-for-bit identically, so the **numba-off fallback is bit-identical to the previous output**. Under numba (JIT on), `np.mean` compiles to numba's own reduction, whose summation order differs from numpy's at the ULP level (~4e-16 relative, full float64 precision); compounded through the recursive `jma[i] = jma[i-1] + det1` update the JIT path lands within ~1 ULP of the original but not bit-for-bit identical. Inconsequential for any practical use; `jma` is the only optimised indicator here whose JIT and numba-off paths are not bit-identical to each other (every other one is).
* **`wma` ~132× / `swma` ~149× / `hma` ~90× faster (pure numpy)**: `wma` and `swma` were the last two weighted moving averages still calling `rolling(length).apply(weights(w), raw=True)`, a Python callback invoked once per window (~25k `np.dot` calls per 5000-bar sweep, 26 ms of the 54 ms profiled across the family). Both now use the `utils._core._sliding_weighted_ma` helper that `alma`, `cg`, `fwma`, `pwma` and `sinwma` already share. `hma` is three chained `wma` calls and gets the speedup for free. Per 5000-bar call: `wma` `4.93 ms` → `0.04 ms`, `swma` `3.49 ms` → `0.02 ms`, `hma` `15.04 ms` → `0.17 ms`; `wma` is now within ~1.7× of the TA-Lib passthrough (was 137× slower). No numba involved — the win applies to every install. See the `_sliding_weighted_ma` entry below for the numerical difference.
* **`msw` ~54× faster (pure numpy)**: the Mesa Sine Wave DFT ran a per-bar Python loop with two `np.dot` calls and scalar phase branching. The two weighted sums are now one `np.correlate` each and the phase wrapping is `np.where`, preserving the original branch order (`11.26 ms` → `0.21 ms` per 5000-bar call). The NaN quirk is preserved on purpose: a NaN in the window makes `abs(rp) > 0.001` false, so the bar takes the ±π branch and still emits a finite phase, exactly as the scalar loop did. `msw` is ill-conditioned by construction — its phase is `arctan(ip / rp)` and `rp` passes through zero at a cycle turn — so a 1-ULP shift in `rp` moves the output by ~`1e-13`: the new values differ from the old by at most `2.5e-13` absolute on the `[-1, 1]` sine/lead outputs. For scale, merely switching the OpenBLAS kernel (`OPENBLAS_CORETYPE=NEHALEM`) moved the *old* implementation by `5.0e-13` on the same machine. Verified over periods 2–30, 200/1000/5000-bar frames and NaN-containing input; the NaN mask is unchanged. The Tulipy passthrough is untouched.
* **`_sliding_weighted_ma` uses `np.correlate` instead of a sliding-window matmul**: the shared weighted-MA helper (`alma`, `cg`, `fwma`, `pwma`, `sinwma`, and now `wma`/`swma`) no longer materialises an `O(len(close) * length)` window copy — 320 MB for a 200k-bar frame at `length=200`, now zero. It is also **closer to the pre-vectorisation `rolling().apply()` output**: for `length >= 12` NumPy's `correlate` performs the same per-window `np.dot`, so those results are bit-for-bit identical to what `rolling().apply()` produced; below that it uses its own C loop and lands within ~2 ULP (`wma` at `length=10`: `4.6e-16` relative, versus `1.3e-14` for the matmul it replaces). Timings per 5000-bar call at default lengths: `alma` `0.040` → `0.024 ms`, `fwma` `0.050` → `0.033 ms`, `pwma` `0.039` → `0.026 ms`, `cg` `0.127` → `0.118 ms`. The one regression is `sinwma`, whose default `length=14` falls in the 12–31 band where `correlate` calls BLAS per window: `0.042` → `0.061 ms`. Accepted — 19 µs, in exchange for the band where the output is bit-exact.
  For context on why none of these paths is canonical: the pre-vectorisation `np.dot` result is itself BLAS-dependent. On this machine `np.dot` matches a sequential-FMA sum in 100% of cases at `length` 3/5/10, and forcing a pre-FMA kernel with `OPENBLAS_CORETYPE=NEHALEM` shifts it by `4.1e-16` relative — the same order as the change described here. Measured against exact rational arithmetic, every variant lands within 4 ULP of the correctly-rounded weighted sum.
* **`true_range` ~5.7× faster (pure numpy), lifting 15 dependants**: True Range built a three-column DataFrame with `concat([hl, high - prev_close, prev_close - low], axis=1)` and reduced it with `.abs().max(axis=1)` — an intermediate frame plus a row-wise reduction on every call. It now reduces the same three ranges with a chained `np.fmax`; `fmax` skips NaN operands exactly like `DataFrame.max(axis=1)`'s default `skipna=True` and yields NaN only where all three are NaN, so the result is **bit-for-bit identical**, NaN mask included (verified on clean input, scattered internal NaNs, fully-NaN rows and an all-NaN frame). Per 5000-bar call `0.72 ms` → `0.13 ms`. Because `atr` → `true_range` sits under most of the volatility and trend families, the win propagates: `atr` `0.88` → `0.25 ms`, `natr` `0.93` → `0.30 ms`, `supertrend` `1.19` → `0.48 ms`, `pmax` `1.18` → `0.55 ms`, `kc` `1.30` → `0.66 ms`, `vortex` `1.24` → `0.66 ms`, `chop` `1.36` → `0.71 ms`, `cksp` `1.69` → `1.02 ms`, `adx` `3.39` → `2.75 ms`, `squeeze` `3.10` → `2.31 ms`, plus `aberration`, `ce`, `pgo`, `adxr` and `dx`. No numba involved — the win applies to every install.
* **`squeeze_pro` ~2.5× faster (no redundant work)**: the Squeeze PRO called `kc()` three times with identical `length`, `mamode` and `tr`, differing only in the scalar — so the true range and both moving averages were computed three times to produce three pairs of bands that are all `basis ± scalar * band`. The shared `range_`/`basis`/`band` are now computed once and the six bounds derived from them with the same expressions `kc()` uses, so the classification is **bit-for-bit identical** (verified across default, `mamode="ema"`, `tr=False`, `detailed=True` and `asint=False`, on NaN-containing input). Per 5000-bar call `5.77 ms` → `2.31 ms`; roughly `2.0 ms` of that comes from the three `kc()` calls collapsing into one and the rest from the `true_range` entry above.
* **`mavp` ~22× faster (pure numpy)**: the variable-period SMA ran a Python loop calling `close[i - p + 1 : i + 1].mean()` once per bar — 5000 `ndarray.mean()` round trips, 55% of the indicator's profile. Bars sharing a window size are now grouped by a single stable `argsort` and each group gathers its windows in one strided fancy-index and reduces with `mean(axis=1)`. The windows are the same slices in the same order, so the output is **bit-for-bit identical**, NaN mask included (verified for the default linear ramp, custom `minperiod`/`maxperiod`, a random integer period series, NaN-containing input, and frames too short to produce any value). Per 5000-bar call `8.71 ms` → `0.40 ms`; at 50000 bars `~83 ms` → `3.7 ms`. Cost now scales with the number of *distinct* periods rather than with the number of bars, which is bounded by `maxperiod - minperiod + 1` (29 at the defaults). The trade-off is at the exotic end: with ~1000 or more distinct periods the per-group gather overtakes the per-bar loop — at `maxperiod=2000` over 5000 bars it is `9.0` → `14.9 ms`. No numba involved.
* **`macd` / `macdfix` ~15× faster (numba)**: `_ema_aligned` — the TA-Lib-aligned EMA seeded with an SMA at a given bar — ran a per-bar Python loop and is called three times per `macd()` (fast, slow, signal), accounting for ~96% of the indicator's profile. It is now a numba `@njit(cache=True)` kernel; the body is unchanged, only the decorator was added. Per 5000-bar call: `macd` `3.60 ms` → `0.24 ms`, `macdfix` (which delegates to `macd`) `3.60 ms` → `0.34 ms`, `macd(asmode=True)` `4.13 ms` → `0.81 ms`. At the default `12/26/9` the output is **bit-for-bit identical**; at other periods the seed SMA can differ by 1 ULP, because numba's `.mean()` sums sequentially where NumPy uses pairwise summation — at `5/35/5` that is `1.4e-14` absolute on the MACD line (1 ULP at the ~100 price scale; median difference 0, NaN mask unchanged), propagated forward by the recursion. Without numba it stays the same plain Python loop as before.
* **`adx` ~2.1× / `dx` ~2.5× / `dm` ~2.8× faster (vectorised), lifting `adxr`, `plus_dm`, `minus_dm` and `signals`**: the near-zero clamp was applied with `Series.apply(zero)`, a Python callback invoked once per bar — two of them per directional-movement indicator, i.e. 10000 Python-level `abs()` comparisons per 5000-bar `adx()`, and the largest single entry in its profile. A new `utils.zero_series()` does the same clamp as `x.where(~(x.abs() < sflt.epsilon), 0)`; the comparison is negated rather than inverted so that NaN — which never compares below epsilon — passes through untouched, exactly as `apply(zero)` did. `psar`/`sarext` only ever used the last bar of the clamped series and now clamp that one scalar instead. Per 5000-bar call: `adx` `2.94` → `1.41 ms`, `adxr` `3.19` → `1.99 ms`, `dx` `2.09` → `0.84 ms`, `dm` `1.97` → `0.71 ms`, `minus_dm` `1.02` → `0.51 ms`, `plus_dm` `1.04` → `0.46 ms`, and `macd(signal_indicators=True)` `5.00` → `2.08 ms` through `utils._signals`. Output is **bit-for-bit identical** for all of them. No numba involved — the win applies to every install.
* **`skew` / `kurtosis` ~2× faster (pure numpy)**: `np_rolling_moments` raised the mean-centred window matrix to each requested order with `dev ** k`, calling `np.power` over an `(n - length + 1, length)` array once per order and recomputing the squared deviations both times. A `_dev_power_sum` helper now derives the third and fourth powers from one shared `dev * dev` (`dev**3` → `sq * dev`, `dev**4` → `sq * sq`). Per 5000-bar call: `skew` `2.41` → `1.22 ms`, `kurtosis` `2.40` → `1.14 ms`. Repeated multiplication rounds twice where `np.power` rounds once, so ~58% (`skew`) and ~74% (`kurtosis`) of bars stay bit-identical and the rest move by at most `6.7e-16` and `3.6e-15` absolute — 3 and 16 ULP of 1.0, on outputs spanning `[-2.5, 7.1]`, with a median difference of 0 and an unchanged NaN mask. The much larger *relative* figures (up to `6.5e-13`) occur only at bars where the statistic itself is ~`1e-16`. For scale, that absolute band is the same order as switching the OpenBLAS kernel under the weighted-MA helper (see the `_sliding_weighted_ma` entry). The summation order is unchanged. No numba involved.
* **`vidya` ~4.8× faster (numba)**: the Variable Index Dynamic Average per-bar recursion (`alpha * cmo[i] * close[i] + vidya[i-1] * (1 - alpha * cmo[i])`) moved into a numba `@njit(cache=True)` kernel over numpy arrays; the vectorised CMO prep stays in pandas. Per 5000-bar call `3.01 ms` → `0.63 ms`. Output is **bit-for-bit identical**; falls back to a plain Python loop without numba.
* **`adosc` ~9× faster (numba)**: the Accumulation/Distribution Oscillator propagated its two AD EMAs (seeded with `AD[0]`, matching TA-Lib) in a per-bar Python loop. That loop is now a numba `@njit(cache=True)` kernel and the function-local `import numpy as np` moved to module scope. Per 5000-bar call `2.20 ms` → `0.24 ms`. Output is **bit-for-bit identical**; falls back to a plain Python loop without numba.
* **`cdl_pattern(name="all")` ~18% faster (no redundant work)**: every native `cdl_*` pattern re-derived the same nine OHLC helper arrays (body high/low, real body, upper/lower shadow, ranges, colour) from the same four `to_numpy()` conversions, so a single `cdl_pattern` call built 60 identical `CandleArrays` — the largest entry in its profile once the detection loops were JIT-compiled. `cdl_pattern` now builds them once and hands them down through a `candle_arrays` kwarg that `run_pattern` pops; a pattern invoked on its own still builds its own. Per 5000-bar call `8.89 ms` → `7.32 ms` (with `scalar=1`, `10.40` → `8.11 ms`). Output is **bit-for-bit identical** for all 60 patterns, verified for the default call, `scalar=1`, `offset=2`, and the standalone `cdl_hammer` / `cdl_doji` / `cdl_inside`. No numba involved.
* **`ha` ~50× faster (numba)**: the recursive Heikin-Ashi open (`df.iat[]` scalar loop) moved into a numba `@njit` scan over the precomputed HA_close array (`~75 ms` → `~1.5 ms` per 5000-bar call). Output is identical; falls back to a plain Python loop without numba.
* **`stc` ~40× faster (numba)**: the two Schaff Trend Cycle stochastic-smoothing loops moved into a shared numba `@njit` kernel, parametrised by the gate series (pass 1 gates on the rolling minimum, pass 2 on the rolling range) (`~82 ms` → `~2 ms` per 5000-bar call). Output is bit-for-bit identical (the cumulative 8-decimal rounding is preserved inside the kernel); falls back to a plain Python loop without numba.
* **`wad` ~15× faster**: Williams A/D replaced `Series.combine(prev_close, max/min)` (a per-element Python callback) with vectorised `np.fmax`/`np.fmin` (`~16 ms` → `~1 ms` per 5000-bar call). Output is identical; pure numpy, no numba needed.
* **`wilder_smooth` / `dm` ~5× faster (numba)**: Wilder's cumulative smoothing (`utils/_wilder.py`, used by `dm`) moved its per-bar recursion into a numba `@njit` kernel. The seed (`np.nansum(raw[1:length])`) is still computed with numpy in the caller and passed in, so the cumulative sum matches bit-for-bit — the kernel runs only the deterministic scalar recursion. `dm` `~10.5 ms` → `~2.0 ms` per 5000-bar call. Bit-for-bit identical; falls back to a plain Python loop without numba.
* **`pmax` ~4.6× faster (numba)**: the PMAX adaptive-band / trend-direction per-bar loop is now `@njit`-compiled with its Python-list accumulators replaced by numpy arrays (`~5.4 ms` → `~1.2 ms` per 5000-bar call; the residual is the ATR/MA prep). The loop is pure comparisons + `min`/`max` + assignment, so output is bit-for-bit identical; falls back to a plain Python loop without numba.
* **`npround` ~400× faster (vectorised)**: replaced the per-element `close.apply(np.round)` Python callback with a single vectorised `np.round(close)` (`~6 ms` → `~0.02 ms` per 5000-bar call). Output is bit-for-bit identical.
* **`aroon`, `maxindex`, `minindex`, `minmaxindex` ~9–17× faster (vectorised)**: the rolling `argmax`/`argmin` arg-position loops (`Series.rolling(length).apply(np.argmax/argmin, raw=True)`) now use a shared `_sliding_argextreme()` helper built on `numpy.lib.stride_tricks.sliding_window_view` (mirrors the existing `_sliding_weighted_ma`). aroon `~12 ms` → `~0.7 ms`, minmaxindex `~8 ms` → `~0.8 ms`, maxindex/minindex `~4 ms` → `~0.4 ms` per 5000-bar call. Integer arg-positions → bit-for-bit identical output; pure numpy, no numba needed.
* **`ebsw` ~850× faster (numba)**: the Even Better SineWave per-bar `.iloc[]` HighPass + SuperSmoother recursion moved into a numba `@njit` kernel over a numpy array, with the (bar-invariant) filter coefficients hoisted out of the loop and the 2-element `FilterHist` list replaced by two scalars (`~34 ms` → `~0.04 ms` per 5000-bar call). Output is bit-for-bit identical; falls back to a plain Python loop without numba.
* **`kama` ~90× faster (numba)**: Kaufman's Adaptive MA per-bar smoothing recursion moved into a numba `@njit` loop over numpy arrays (the vectorised efficiency-ratio/smoothing-constant prep stays in pandas) (`~30 ms` → `~0.3 ms` per 5000-bar call). Output is bit-for-bit identical; falls back to a plain Python loop without numba.
* **`mama` ~55× faster (numba)**: the MESA Adaptive MA / FAMA inline Hilbert-Transform loop is now `@njit`-compiled (`~16 ms` → `~0.3 ms` per 5000-bar call). Output is bit-for-bit identical (verified against the pure-Python `.py_func`); falls back to a plain Python loop without numba.
* **Candlestick patterns ~32× faster (numba)**: all 60 native `cdl_*` pattern per-bar detection loops (the TA-Lib incremental running-window algorithm) moved into numba `@njit` kernels over numpy arrays, with the average-factor constants and range arrays hoisted by the caller. `cdl_pattern(name="all")` drops from `~288 ms` to `~9 ms` per 5000-bar call. Output is bit-for-bit identical for every pattern; falls back to a plain Python loop without numba.
* **Faster cold import (~3–4×)**: `import pandas_ta_classic` takes ~280 ms (was ~930 ms) because indicator functions load on demand. Public API unchanged.
* **Math operators individually importable**: Each math/trig operator (`add`, `sub`, `mult`, `div`, `rolling_max/min/sum`, `acos`, `cos`, `exp`, `ln`, …) now lives in its own submodule, e.g. `from pandas_ta_classic.math.add import add`.
* **`combination()` delegates to `math.comb`**: Replaces the hand-rolled nCr loop. Signature and results unchanged; the unused `multichoose` kwarg alias was dropped (use `repetition`).
* **`npNaN` alias removed internally**: All modules now use `np.nan` directly. No public API change.
* **All `npX` numpy import aliases removed** (26 aliases, e.g. `npSqrt`, `npArange`, `npNdArray`): modules now use `import numpy as np` + `np.x` directly. Internal only; no public API change.
* **Explicit package namespace**: `pandas_ta_classic/__init__.py`, `core.py`, and `utils/__init__.py` no longer use star imports — every re-export is explicit with `__all__` defined. All documented names (utility functions, metrics, `Strategy`, lazy-loaded indicators) are unchanged. Incidental namespace leaks (e.g. `ta.np`, `ta.pd`, `ta.DataFrame`, `ta.logger`) are gone; import these from their real homes.
* **Lint-clean repo**: `ruff check .` passes with zero errors and zero per-file ignores. Ambiguous variables renamed (`O` → `O_` in candle loops, `l` → `low` in tests); `sys.path` bootstrap hacks removed from `tests/fixtures/`, `tools/`, and `docs/conf.py` (the editable install makes them redundant); notebook cells reordered to import-first.
* **`linear_regression()` always uses the numpy implementation**: the `"r"` key is now consistently the Pearson correlation coefficient. Previously, environments with scikit-learn installed silently used `LinearRegression.score()` (R²) instead, so results depended on the environment. Affects `pure_profit_score`.
* **`optimal_leverage()` returns a float**: the result is no longer truncated with `int()`, the `capital` argument is documented, the unused `**kwargs` parameter is removed, and the "Incomplete. Do NOT use" docstring warning is removed. Zero-variance input now raises `ValueError` (previously crashed with `OverflowError`).
* **Optional-dependency extras restructured**: the `integration` extra is split into `data` (`yfinance`, `alpha-vantage` — the only integrations the library imports, lazily via `utils/data`) and `backtest` (`backtesting`, `vectorbt`, `backtrader` — docs/examples only). `integration` is retained as an alias for `data,backtest`. The `[all]` extra no longer pulls `data`/`backtest`, so `pip install "pandas-ta-classic[all]"` stays installable on newer Python/OS combinations that lack wheels for these unmaintained, platform-fragile packages; install them explicitly when needed.
* **tulipy oracle frozen to a golden file**: `tulipy` (unmaintained, no wheels for CPython ≥3.12) is removed from the `oracle` extra and no longer imported at test time. Its deterministic outputs are snapshotted in `tests/fixtures/tulipy_oracle.json` (regenerate with `tests/fixtures/generate_tulipy_oracle.py` on CPython <3.12), and `test_oracle_tulipy.py` now compares native output against the frozen arrays on **every** Python version. This fixes the `testing-oracle` CI job on Python 3.12–3.14, where the tulipy install previously failed. `ta-lib` remains the live primary oracle. (The MSW oracle tolerance was loosened to `1e-3` to stay robust to ~`1e-4` numpy-version float drift in that recursive trig indicator.)
* **`tal_ma()` no longer needs TA-Lib and fails fast**: the MA-name→`MA_Type` mapping is now a plain dict of the (frozen) TA-Lib enum values, so `tal_ma("ema") == 1` regardless of whether TA-Lib is installed (previously it silently returned `0`/SMA when TA-Lib was absent). Unknown names now raise `ValueError` and non-string input raises `TypeError` instead of silently defaulting to SMA. Callers (`bbands`, `ppo`, `apo`) already normalise `mamode` to a valid string, so their behaviour is unchanged.

### Deprecated
* **Built-in data fetching (`df.ta.ticker()`, `ta.yf()`, `ta.av()`)**: emit a `FutureWarning` and will be removed in a future release. Data fetching is out of scope for a technical-analysis library. Fetch OHLCV with `yfinance` / `alpha-vantage` directly and pass the DataFrame to pandas-ta-classic; see the new `examples/fetch_market_data.py` for the replacement patterns. The `data` optional extra remains available during the deprecation window.
* **`ichimoku()` tuple return**: returning a `(visible, span)` tuple is deprecated and now emits a `DeprecationWarning`. Pass `as_dataframe=True` to opt in to the single-DataFrame return (add `append_span=True` for the projected span rows), or `as_dataframe=False` to keep the tuple without warning. The accessor `df.ta.ichimoku()` already returns a DataFrame and is unaffected.
* **`CDL_PATTERN_NAMES`**: Use `ALL_PATTERNS` instead. Accessing the old name emits a `DeprecationWarning`.
* **`tsignals` `drift` parameter**: Passing `drift` now emits a `DeprecationWarning` and the value is ignored — trade signals are always computed with a 1-period difference. The parameter will be removed in a future release.
* **Inactive `drift` parameters on `cfo`, `inertia`, `kst`, `rsx`, `chop`, `accbands`, `kvo`**: These indicators accepted a `drift` parameter that never entered their calculation (dead since the upstream template). Passing `drift` now emits a `DeprecationWarning`; the value has never affected the result and the parameter will be removed in a future release. Misleading docstrings were corrected (e.g. `cfo`'s "short period", `chop`'s `ATR(drift)` pseudocode).

### Removed
* **Dead code (audit, zero callers)**: `BasePandasObject` and `_check_na_columns()` in `core.py`; CPR helpers `round_to_strike`, `calculate_option_strikes`, `detect_cpr_breakout`, `detect_cpr_rejection`; math utils `geometric_mean`, `log_geometric_mean`, and the hand-rolled `erf`; time utils `df_dates`, `df_month_to_date`/`mtd`, `df_quarter_to_date`/`qtd`; `category_files()`; six unused `CandleArrays` helper methods; the `pkg_resources` Python < 3.8 fallback in `_meta.py`; `tests/context.py` and other unused test scaffolding.
* **`crossany()`**: zero callers and zero tests; use `cross(a, b, above=True) | cross(a, b, above=False)` if needed. Docs entry removed.
* **`high_low_range()` / `real_body()` wrappers** (`utils/_candles.py`): call sites now use `non_zero_range()` directly (`real_body(open_, close)` ≡ `non_zero_range(close, open_)`).
* **`ytd` alias**: use `df_year_to_date()`. Internal `_camelCase2Title` inlined at its single call site.
* **Unused dev/optional dependencies**: `stochastic` (never imported), `cython` (no `.pyx` sources), `pytest-cov` and `pytest-benchmark` (no coverage/benchmark runs), `isort` (black owns formatting). Corresponding `Imports` probes and the Makefile isort step removed.
* **Unused hard dependencies**: `scipy`, `scikit-learn`, and `statsmodels` dropped from `[project.dependencies]` — nothing in the package imports them anymore. `_linear_regression_sklearn()` removed (see Changed). Stale `Imports` probes pruned (`backtrader`, `matplotlib`, `mplfinance`, `numba`, `scipy`, `sklearn`, `statsmodels`, `vectorbt`, `yaml`); numba acceleration is unaffected (`utils/_njit.py` self-detects numba).

### Fixed
* **`tsignals` `drift` corrupted trade signals**: `trends.diff(drift)` computed entries/exits against the state `drift` bars ago instead of the previous bar, so `drift != 1` reported spurious entries mid-trend and missed real crossovers (contradicting the documented `diff()` behavior). Now always uses a 1-period difference. Output at the default (`drift=1`) is unchanged.
* **Cross-package indicator imports**: A submodule import could overwrite a re-exported function of the same name on the parent package (e.g. `from pandas_ta_classic.volatility import atr` occasionally returned the `atr` *module* instead of the function). Resolved.
* **Test-fixture regen no longer drops `cmo_14` entries** when running tests without tulipy installed. Generators now merge with existing JSON instead of overwriting.

## [0.6.52] - 2026-06-25

### Added
* **SMC Liquidity Sweep** (`smc_sweep`) (PR #123 by @Adhyansinghgupta): Bullish and bearish smart-money liquidity sweep indicator for momentum analysis. Available as `df.ta.smc_sweep()`.
* **ichimoku `append_span`** (PR #120): New `append_span` parameter on the ichimoku accessor for span-appending control.

### Changed
* **CPR numeric encoding** (PR #117): `calculate_price_position()` and `calculate_cpr_width()` now return `np.int8` instead of string labels. `CPR_POSITION`: `1` (above TC), `0` (inside CPR), `-1` (below BC). `CPR_WIDTH_CLASS`: `1` (wide), `0` (medium), `-1` (narrow). All indicator columns are now numeric.
* **`AnalysisIndicators.__call__` fail-fast**: Removed `except BaseException: pass` swallowing all indicator errors. Indicator exceptions now propagate to callers instead of silently returning `None`. Code that relied on `df.ta.rsi()` never raising must add its own error handling.

### Fixed
* **ichimoku accessor returns DataFrame** (PR #116): `df.ta.ichimoku()` now returns a DataFrame instead of a tuple.
* **macdext silent double-fallback** (PR #121): Eliminated silent double-fallback for KAMA/MAMA matypes in `macdext`.

### Documentation
* **long_run / short_run / xsignals** (PR #119): Clarified `fast`/`slow` as pre-computed Series; added `xa`/`xb` range guidance.
* **beta / correl benchmark** (PR #118): Documented that `benchmark` is required for non-`None` output.

### Dependencies
* **actions/checkout v7** (PR #122): Bumped from v6 to v7.

---

## [0.6.20] - 2026-05-21

### Added
* **`apply_offset` / `apply_fill` helpers** (PR #105): Extracted into `utils/` and all indicators migrated to use them. Removes ~200 lines of duplicated offset/fill logic.
* **Comprehensive candle pattern tests** (PR #106): 59 TA-Lib candle patterns covered by CI tests.
* **Fluent API chaining** (PR #113): DataFrame accessor supports method chaining.
* **Property-based testing** (PR #114): Hypothesis-based tests added to test suite.
* **Full Indicator Name Comments**: All indicator files include full indicator names as comments on line 1 (format: `# Full Indicator Name (ABBREVIATION)`).
* **UV Package Manager Support**: All documentation includes `uv` install instructions alongside `pip`.
* **Native Candlestick Patterns**: Added native `cdl_doji` and `cdl_inside` implementations (no TA-Lib required). Accessible via `df.ta.cdl_doji()`, `df.ta.cdl_inside()`, or `df.ta.cdl_pattern()`.

### Changed
* **Wilder smoothing shared utility** (PR #112 remediation): Extracted `wilder_smooth()` into `utils/_wilder.py` for TA-Lib-exact cumulative smoothing. Used by `dm.py` for PLUS_DM/MINUS_DM parity.
* **Chained EMA helper** (PR #112 remediation): Added `_ema_chain()` to `overlap/ema.py` for consistent NaN-stripping in DEMA, TEMA, T3. Reduces repetitive boilerplate by ~60 lines.
* **Fixture auto-regeneration**: `tests/__init__.py` now regenerates `expected_values.json` and `regression_snapshots.json` on import when TA-Lib is available. `make fixtures` and `make test-all` targets added to `Makefile`.
* **`tal` → `talib` rename**: All test files now import `talib` directly instead of aliasing as `tal`, matching source code convention.
* **Dead code removal**: Removed ~370 instances of unused imports (F401), 8 unused local variables (F841), 1 duplicate method (`test_custom_a`), 65 useless f-string prefixes (F541), and 279 unnecessary UTF-8 encoding declarations (UP009).
* **Code modernization**: Applied pyupgrade (UP) and flake8-return (RET) fixes — `Optional[X]`→`X|None`, `List`→`list`, removed redundant `else` after `return`.
* **Linreg TA-Lib dispatch**: Replaced 6-branch `if`/`elif` chain with `_TALIB_DISPATCH` lookup dictionary.
* **PSAR cleanup**: Removed `import numpy as _np` from function body; uses module-level `np` instead.
* **test_strategy**: Fixed duplicate `test_custom_a` method (was shadowed, never ran). Fixed stale column count assertion.

### Fixed
* **TA-Lib-exact native implementations** (PR #112): Corrected native paths for many indicators to match TA-Lib output exactly.
* **stdev variance calculation** (PR #109): `ddof` default updated to `False` for population variance.
* **Modularity refactor** (PR #108): Issue #46 modularity improvements.
* **Accessor usability** (PR #107): Issue #48 usability and docs fixes.
* **Alpha Vantage integration** (PRs #110, #111): Support for both `alpha_vantage` and `alphaVantage` libraries; improved import checks and error handling.
* **Feature fix PR #112** (PR #115): Follow-up corrections to the TA-Lib-exact implementation from PR #112.

---

## [0.5.44] - 2026-04-30

### Added
* **TA-Lib / tulipy parity indicator set** (PR #104): Added wrappers and native implementations for `msw`, `fosc`, `macdext`, `macdfix`, `rocp`, `rocr`, `rocr100`, `stochf`, `avgprice`, `medprice`, `typprice`, `linregangle`, `linregintercept`, `linregslope`, `mavp`, `md`, `stderr`, `dx`, `edecay`, `plus_dm`, `minus_dm`, `sarext`, `avolume`, `cvi`, `hvol`, `emv`, `marketfi`, `vosc`, and `wad`.
* **Math operator namespace**: Added `pandas_ta_classic/math/` exposing arithmetic operators (`add`, `sub`, `mult`, `div`), rolling operators (`rolling_max`, `rolling_min`, `rolling_sum`), and math transforms for TA-Lib/tulipy compatibility.
* **Oracle parity suites**: Added `tests/test_oracle_talib.py` and `tests/test_oracle_tulipy.py` for cross-library validation on shared SPY fixtures.
* **60 native CDL pattern files** (PR #87): Added `candles/cdl_*.py` implementations for 60 patterns. Combined with `cdl_doji` and `cdl_inside`, total accessible via `cdl_pattern()` is **62**. TA-Lib is **never** used for CDL patterns — native implementations take priority regardless of TA-Lib installation. Added shared `_cdl_math.py` helper.
* **5 Hilbert Transform cycle indicators** (PR #83): `ht_dcperiod`, `ht_dcphase`, `ht_phasor`, `ht_sine`, `ht_trendmode`. Shared `_hilbert.py` helper. Cycles category grows from 2 to 7.
* **MAMA / FAMA** (PR #84): MESA Adaptive Moving Average with FAMA output. Uses Ehlers' adaptive phase computation.
* **HT_TRENDLINE** (PR #84): Hilbert Transform Instantaneous Trendline. Added to overlap category.
* **TSF** (PR #85): Time Series Forecast — linear regression projected one period ahead. Matches TA-Lib TSF.
* **Beta** (PR #86): Asset volatility relative to a benchmark series. Matches TA-Lib BETA.
* **CORREL** (PR #86): Pearson Correlation Coefficient between two series. Matches TA-Lib CORREL.
* **ADXR** (PR #89): Average Directional Movement Index Rating — smoothed average of ADX. Matches TA-Lib ADXR.
* **CPR** (PR #77): Central Pivot Range with 4 calculation methods (classic, camarilla, fibonacci, woodie).
* **Chandelier Exit** (`ce`): Trailing stop. `CE_L = rolling_max(high, length) - multiplier × ATR`, `CE_S = rolling_min(low, length) + multiplier × ATR`. Default `length=22`, `multiplier=3.0`. Returns DataFrame with `CE_L_{length}_{multiplier}` and `CE_S_{length}_{multiplier}`.
* **9 New Technical Indicators** (Issue #29): LRSI (Laguerre RSI), PMAX (Price Max), VFI (Volume Flow Indicator), MMAR (Madrid Moving Average Ribbon), Rainbow (Rainbow Charts), PO (Projection Oscillator), DSP (Detrended Synthetic Price), TRIXH (TRIX Histogram), VWMACD (Volume Weighted MACD).

### Changed
* **QQE output columns** (PR #97): `qqe()` now returns 6 columns instead of 3. New columns: `QQEb_l` (long band), `QQEb_s` (short band), `QQEd` (±1 trend direction). **Breaking change**: code relying on a fixed column count or positional indexing must be updated.
* **Updated indicator counts**: 164 indicators in Category (was 151); total 224 with native CDL patterns (was 213).
* **`linreg` breaking default change**: `degrees` kwarg now defaults to `True` (was `False`) to match TA-Lib. Callers using `linreg(close, angle=True)` without `degrees=False` will now receive degrees instead of radians.
* **`stdev`/`variance` breaking default change**: `ddof` now defaults to `0` (population, was `1` sample) to match TA-Lib. Callers relying on sample variance must pass `ddof=1` explicitly.
* **`natr` breaking default change**: `mamode` now defaults to `'rma'` (Wilder smoothing, was `'ema'`) to match TA-Lib. Callers relying on EMA-based NATR must pass `mamode='ema'` explicitly.
* **Native preferred by default** for all indicators with `talib` parameter: Native implementation used by default across all 59 indicators. Callers wanting TA-Lib output must pass `talib=True` explicitly.
* **Enhanced RVGI**: Relative Vigor Index now includes histogram column (RVGI − Signal).
* **Dynamic Category Discovery**: `Category` dict in `_meta.py` now built dynamically from filesystem structure. Auto-discovers previously undocumented indicators (`cdl_doji`, `cdl_inside`, `hwma`, `ma`, `drawdown`, `dm`, `vp`).
* **Python Version Support**: Rolling 5-version support policy (LATEST-4 through LATEST). Managed dynamically via `LATEST_PYTHON_VERSION` in `.github/workflows/ci.yml`.
* **Development Status**: Changed from Beta to Production/Stable in `pyproject.toml`.

### Performance
* **numpy vectorization** (PR #88): 15 indicators (QQE, PSAR, HWC, HT_TRENDLINE, SSF, squeeze, squeeze_pro, RVGI, TD_SEQ, TOS_STDEVALL, ALMA, SINWMA, SWMA, TRIMA, VIDYA) now use `sliding_window_view` instead of pandas `.iloc` loops. Adds shared `_sliding_weighted_ma()` utility.
* **Numba JIT acceleration** (PR #99): 10 indicators (SSF, MCGD, HWMA, RSX, PSAR, Supertrend, QQE, and 3 more) gain optional `@njit(cache=True)` via numba. Graceful no-op fallback in `utils/_njit.py`. Enable with `pip install pandas-ta-classic[performance]`. Speedups: RSX 230×, HWMA 70×, MCGD 43×, SSF 42×, Supertrend 13×, QQE 10×, PSAR 6×.

### Fixed
* **Oracle test policy**: Replaced skip-based oracle tests with explicit assertions for value equivalence or documented divergence. Zero skipped tests.
* **TA-Lib compatibility paths**: Added/updated `talib=True` behavior for `macdfix`, `psar`, `stochrsi`, `plus_dm`, `minus_dm`.
* **Indicator formula parity**: Corrected `edecay` (multiplicative exponential decay), `emv` scaling (`divisor=10000`).

---

## [0.4.47] - 2026-03-17

### Fixed
* **Dependency cleanup, pandas 3.0 compat, Windows pool fix** (PR #79).
* **Initialization and edge-case bugs** across 11 indicators (PR #80).
* **Numerical bugs** in `linreg`, `tsi`, `brar`, `bbands`, `cti` (PR #94).
* **TA-Lib reference alignment** for 8 indicators (PR #81).
* **Rolling stats cross-version determinism**: Replaced pandas rolling stats with numpy (PR #82).
* **None-guards** to prevent crashes on short/invalid input across 26 files (PR #95).
* **print() → logging** across library code (PR #93). **Breaking**: code catching print output must switch to log capture.
* **Strategy dataclass bugs** (PR #96). **Breaking**: some Strategy API surface changed.
* **Code of Conduct Contact**: Updated enforcement contact from original maintainer email to GitHub Issues.
* **PyPI Release Version**: Set `SETUPTOOLS_SCM_PRETEND_VERSION` from the release tag to prevent `.dev0` versions being published.
* **Version Scheme**: Changed from post-release to default scheme. Tagged releases get clean versions (e.g., `0.4.47`); commits after a tag get `.devN` suffix.
* **PyPI Image Display**: Updated README.md to use absolute GitHub URLs so images render correctly on the PyPI package page.
* **CI/CD Shallow Clone**: Added `fetch-depth: 0` to all checkout steps to ensure full git history for setuptools-scm.
* **Version Fallback**: Changed fallback from `0.0.0.dev0` to `0.0.0` (PEP 440 compliant).

### Added
* **Automatic Version Management**: Version now determined from git tags via `setuptools-scm`. Development builds get `.dev` suffix; tagged releases use the tag exactly. See CONTRIBUTING.md for documentation.

---

## [0.3.78] - 2026-02-27

### Changed
* **Type hints** (PR #67 by @rmarcink): Type annotations added across all public function signatures.

---

## Pre-0.3.78 — Historical

> Items below describe the state of the library as inherited from the original `pandas-ta` project and pre-2026 development. Preserved for historical reference.

### General
* A __Strategy__ Class to help name and group your favorite indicators.
* If a **TA Lib** is already installed, Pandas TA will run TA Lib's version. (**BETA**)
* Some indicators have had their ```mamode``` _kwarg_ updated with more _moving average_ choices with the **Moving Average Utility** function ```ta.ma()```. For simplicity, all _choices_ are single source _moving averages_. This is primarily an internal utility used by indicators that have a ```mamode``` _kwarg_. This includes indicators: _accbands_, _amat_, _aobv_, _atr_, _bbands_, _bias_, _efi_, _hilo_, _kc_, _natr_, _qqe_, _rvi_, and _thermo_; the default ```mamode``` parameters have not changed. However, ```ta.ma()``` can be used by the user as well if needed. For more information: ```help(ta.ma)```
    * **Moving Average Choices**: dema, ema, fwma, hma, linreg, midpoint, pwma, rma, sinwma, sma, swma, t3, tema, trima, vidya, wma, zlma.
* An _experimental_ and independent __Watchlist__ Class located in the [Examples](https://github.com/xgboosted/pandas-ta-classic/tree/main/examples/watchlist.py) Directory that can be used in conjunction with the new __Strategy__ Class.
* _Linear Regression_ (**linear_regression**) is a new utility method for Simple Linear Regression using _Numpy_ or _Scikit Learn_'s implementation.
* Added utility/convience function, ```to_utc```, to convert the DataFrame index to UTC. See: ```help(ta.to_utc)``` **Now** as a Pandas TA DataFrame Property to easily convert the DataFrame index to UTC.

<br />

### Breaking / Depreciated Indicators
* _Trend Return_ (**trend_return**) has been removed and replaced with **tsignals**. When given a trend Series like ```close > sma(close, 50)``` it returns the Trend, Trade Entries and Trade Exits of that trend to make it compatible with [**vectorbt**](https://github.com/polakowo/vectorbt) by setting ```asbool=True``` to get boolean Trade Entries and Exits. See ```help(ta.tsignals)```

<br/>

### New Indicators
* _Arnaud Legoux Moving Average_ (**alma**) uses the curve of the Normal (Gauss) distribution to allow regulating the smoothness and high sensitivity of the indicator. See: ```help(ta.alma)```
* _Draw Down_ (**drawdown**) calculates the percentage decline from the peak equity of a trading account, or fund. See ```help(ta.drawdown)```
* _Candle Patterns_ (**cdl_pattern**) If TA Lib is installed, then all those Candle Patterns are available. See the list and examples above on how to call the patterns. See ```help(ta.cdl_pattern)```
* _Candle Z Score_ (**cdl_z**) normalizes OHLC Candles with a rolling Z Score. See ```help(ta.cdl_z)```
* _Correlation Trend Indicator_ (**cti**) is an oscillator created by John Ehler in 2020. See ```help(ta.cti)```
* _Cross Signals_ (**xsignals**) was created by Kevin Johnson. It is a wrapper of Trade Signals that returns Trends, Trades, Entries and Exits. Cross Signals are commonly used for **bbands**, **rsi**, **zscore** crossing some value either above or below two values at different times. See ```help(ta.xsignals)```
* _Directional Movement_ (**dm**) developed by J. Welles Wilder in 1978 attempts to determine which direction the price of an asset is moving. See ```help(ta.dm)```
* _Even Better Sinewave_ (**ebsw**) measures market cycles and uses a low pass filter to remove noise. See: ```help(ta.ebsw)```
* _Jurik Moving Average_ (**jma**) attempts to eliminate noise to see the "true" underlying activity.. See: ```help(ta.jma)```
* _Klinger Volume Oscillator_ (**kvo**) was developed by Stephen J. Klinger. It is designed to predict price reversals in a market by comparing volume to price.. See ```help(ta.kvo)```
* _Schaff Trend Cycle_ (**stc**) is an evolution of the popular MACD incorportating two cascaded stochastic calculations with additional smoothing. See ```help(ta.stc)```
* _Squeeze Pro_ (**squeeze_pro**) is an extended version of "TTM Squeeze" from John Carter. See ```help(ta.squeeze_pro)```
* _Tom DeMark's Sequential_ (**td_seq**) attempts to identify a price point where an uptrend or a downtrend exhausts itself and reverses. Currently exlcuded from ```df.ta.strategy()``` for performance reasons. See ```help(ta.td_seq)```
* _Think or Swim Standard Deviation All_ (**tos_stdevall**) indicator which
returns the standard deviation of data for the entire plot or for the interval
of the last bars defined by the length parameter. See ```help(ta.tos_stdevall)```
* _Vertical Horizontal Filter_ (**vhf**) was created by Adam White to identify trending and ranging markets.. See ```help(ta.vhf)```

<br/>

### Updated Indicators

* _Acceleration Bands_ (**accbands**) Argument ```mamode``` renamed to ```mode```. See ```help(ta.accbands)```.
* _ADX_ (**adx**): Added ```mamode``` with default "**RMA**" and with the same ```mamode``` options as TradingView. New argument ```lensig``` so it behaves like TradingView's builtin ADX indicator. See ```help(ta.adx)```.
* _Archer Moving Averages Trends_ (**amat**): Added ```drift``` argument and more descriptive column names.
* _Average True Range_ (**atr**): The default ```mamode``` is now "**RMA**" and with the same ```mamode``` options as TradingView. See ```help(ta.atr)```.
* _Bollinger Bands_ (**bbands**): New argument ```ddoff``` to control the Degrees of Freedom. Also included BB Percent (BBP) as the final column. Default is 0. See ```help(ta.bbands)```.
* _Choppiness Index_ (**chop**): New argument ```ln``` to use Natural Logarithm (True) instead of the Standard Logarithm (False). Default is False.  See ```help(ta.chop)```.
* _Chande Kroll Stop_ (**cksp**): Added ```tvmode``` with default ```True```. When ```tvmode=False```, **cksp** implements "The New Technical Trader" with default values. See ```help(ta.cksp)```.
* _Chande Momentum Oscillator_ (**cmo**): New argument ```talib``` will use TA Lib's version and if TA Lib is installed. Default is True. See ```help(ta.cmo)```.
* _Decreasing_ (**decreasing**): New argument ```strict``` checks if the series is continuously decreasing over period ```length``` with a faster calculation. Default: ```False```. The ```percent``` argument has also been added with default None. See ```help(ta.decreasing)```.
* _Increasing_ (**increasing**): New argument ```strict``` checks if the series is continuously increasing over period ```length``` with a faster calculation. Default: ```False```. The ```percent``` argument has also been added with default None. See ```help(ta.increasing)```.
* _Klinger Volume Oscillator_ (**kvo**): Implements TradingView's Klinger Volume Oscillator version. See ```help(ta.kvo)```.
* _Linear Regression_ (**linreg**): Checks **numpy**'s version to determine whether to utilize the ```as_strided``` method or the newer ```sliding_window_view``` method. This should resolve Issues with Google Colab and it's delayed dependency updates as well as TensorFlow's dependencies as discussed in Issues [#285](https://github.com/twopirllc/pandas-ta/issues/285) and [#329](https://github.com/twopirllc/pandas-ta/issues/329).
* _Moving Average Convergence Divergence_ (**macd**): New argument ```asmode``` enables AS version of MACD. Default is False.  See ```help(ta.macd)```.
* _Parabolic Stop and Reverse_ (**psar**): Bug fix and adjustment to match TradingView's ```sar```. New argument ```af0``` to initialize the Acceleration Factor. See ```help(ta.psar)```.
* _Percentage Price Oscillator_ (**ppo**): Included new argument ```mamode``` as an option. Default is **sma** to match TA Lib. See ```help(ta.ppo)```.
* _True Strength Index_ (**tsi**): Added ```signal``` with default ```13``` and Signal MA Mode ```mamode``` with default **ema** as arguments. See ```help(ta.tsi)```.
* _Volume Profile_ (**vp**): Calculation improvements. See [Pull Request #320](https://github.com/twopirllc/pandas-ta/pull/320) See ```help(ta.vp)```.
* _Volume Weighted Moving Average_ (**vwma**): Fixed bug in DataFrame Extension call. See ```help(ta.vwma)```.
* _Volume Weighted Average Price_ (**vwap**): Added a new parameter called ```anchor```. Default: "D" for "Daily". See [Timeseries Offset Aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases) for additional options. **Requires** the DataFrame index to be a DatetimeIndex. See ```help(ta.vwap)```.
* _Volume Weighted Moving Average_ (**vwma**): Fixed bug in DataFrame Extension call. See ```help(ta.vwma)```.
* _Z Score_ (**zscore**): Changed return column name from ```Z_length``` to ```ZS_length```. See ```help(ta.zscore)```.
