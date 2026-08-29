"""Guards against future-data leakage (lookahead bias).

An indicator is *causal* when the value it reports at bar ``t`` depends only on
bars ``<= t``.  A causal indicator must therefore return identical values for
the first ``K`` bars whether it is evaluated over ``K`` bars or over the whole
series.  Any difference means a bar after ``K`` changed an earlier row, so a
backtest run over the full history would use information that did not exist at
the time the signal is claimed to have been produced.

Covered:
  1. Issue #149 -- ``non_zero_range`` added its epsilon to *every* row as soon as any
row in the batch had a zero range, so appending one flat bar shifted the whole
history.  27 indicators inherited the leak.
  2. ``mavp`` -- the default ``periods`` was ``linspace(min, max, len(close))``,
     so every bar's window length depended on the total number of bars.
  3. ``cdl_z(full=True)`` -- a whole-series window plus ``bfill()`` copied the
     final bar's Z Score onto every earlier row.

A leak is only visible if the input contains whatever the leaking branch tests
for -- issue #149 was invisible on the SPY sample data because that data has no
flat bar.  ``causal_test_data`` therefore plants a set of pathological bars,
all of them *after* the last cut point, so a causal indicator cannot see them
in any prefix.  Add to that set whenever a new batch-global branch appears.

Run:
    pytest tests/test_lookahead.py
"""

from sys import float_info as sflt

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta
from pandas_ta_classic.utils import non_zero_range

BARS = 400
CUTS = (100, 250)
CUT = CUTS[-1]


def causal_test_data(bars: int = BARS) -> pd.DataFrame:
    """OHLCV frame whose pathological bars all sit after the last cut point.

    Each planted feature is the trigger for a branch that has historically been
    written batch-globally:

    * flat bars (``high == low``) -- the epsilon substitution in ``non_zero_range``
    * zero-volume bars            -- volume-weighted divisions
    * a constant-price run        -- zero standard deviation / zero range windows
    * a single extreme bar        -- whole-series min/max normalisation

    Because they all live past ``max(CUTS)``, a causal indicator can never let
    them influence a prefix result.
    """
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 1.0, bars))
    high = close + rng.uniform(0.1, 1.0, bars)
    low = close - rng.uniform(0.1, 1.0, bars)
    open_ = close + rng.normal(0, 0.3, bars)
    volume = rng.integers(1_000, 10_000, bars).astype(float)

    for i in (bars - 1, bars - 5, bars - 40):
        open_[i] = high[i] = low[i] = close[i]

    volume[bars - 3] = 0.0
    volume[bars - 12] = 0.0

    flat_run = slice(bars - 60, bars - 55)
    open_[flat_run] = high[flat_run] = low[flat_run] = close[flat_run] = close[bars - 60]

    high[bars - 25] *= 5.0
    low[bars - 25] /= 5.0
    volume[bars - 25] *= 100.0

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2020-01-01", periods=bars, freq="D"),
    )


def declined(result, source: pd.DataFrame) -> bool:
    """True when the accessor echoed the source frame instead of a result.

    An indicator that returns None because the prefix is shorter than its
    warm-up gets the input DataFrame back from the accessor; that is not a
    leak, just too little data.
    """
    return isinstance(result, pd.DataFrame) and list(result.columns) == list(source.columns)


def prefix_deviation(full, prefix):
    """Compare two results on their shared index; return a reason or None.

    Values are required to match exactly: a causal indicator replays the same
    operations on the same inputs for the shared rows, so any non-zero delta is
    a leak rather than accumulated floating point noise.
    """
    common = full.index.intersection(prefix.index)
    if len(common) == 0:
        return "no overlapping index"

    a = np.asarray(full.loc[common], dtype=float)
    b = np.asarray(prefix.loc[common], dtype=float)
    if a.shape != b.shape:
        return f"shape {a.shape} != {b.shape}"

    nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
    if nan_mismatch:
        return f"{nan_mismatch} NaN placement mismatches"

    finite = ~np.isnan(a)
    if finite.any():
        delta = np.abs(a[finite] - b[finite]).max()
        if delta > 0:
            return f"max deviation {delta:.3e}"
    return None


@pytest.fixture(scope="module")
def frames():
    """The full frame plus one prefix per cut point."""
    df = causal_test_data()
    return df, {cut: df.head(cut).copy() for cut in CUTS}



def call(source: pd.DataFrame, name: str, kwargs: dict):
    return getattr(source.ta, name)(**kwargs)


def deviations(frames, name: str, kwargs: dict) -> list[str]:
    """Every cut point at which *name* disagrees with its own prefix."""
    df, prefixes = frames
    found = []
    full = call(df, name, kwargs)
    if full is None or declined(full, df):
        return found
    for cut, source in prefixes.items():
        prefix = call(source, name, kwargs)
        if prefix is None or declined(prefix, source):
            continue
        deviation = prefix_deviation(full, prefix)
        if deviation is not None:
            found.append(f"cut={cut}: {deviation}")
    return found



# --- issue #149: the epsilon must apply to the flat row only ---------------


def test_flat_bar_does_not_change_other_rows():
    high = pd.Series([10.0, 11.0, 12.0, 13.0])
    low = pd.Series([9.0, 10.0, 11.0, 13.0])

    result = non_zero_range(high, low)

    np.testing.assert_array_equal(result.iloc[:3].to_numpy(), np.array([1.0, 1.0, 1.0]))
    assert result.iloc[3] == sflt.epsilon


def test_appending_flat_bar_leaves_history_untouched():
    high = pd.Series([10.0, 11.0, 12.0])
    low = pd.Series([9.0, 10.0, 11.0])

    before = non_zero_range(high, low)
    after = non_zero_range(
        pd.concat([high, pd.Series([13.0], index=[3])]),
        pd.concat([low, pd.Series([13.0], index=[3])]),
    )

    pd.testing.assert_series_equal(before, after.iloc[:3])


def test_nan_rows_are_preserved():
    high = pd.Series([np.nan, 11.0, 12.0])
    low = pd.Series([np.nan, 11.0, 11.0])

    result = non_zero_range(high, low)

    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == sflt.epsilon
    assert result.iloc[2] == 1.0


@pytest.mark.parametrize("name", ["pdist", "brar", "true_range", "atr", "ad"])
def test_indicators_downstream_of_the_epsilon(frames, name):
    """The indicators named in issue #149, plus their closest neighbours."""
    assert deviations(frames, name, {}) == []



# --- the individually fixed cases ------------------------------------------


def test_mavp_refuses_to_invent_a_schedule(frames):
    """A `periods` default derived from len(close) would be a leak."""
    df, _ = frames
    with pytest.raises(TypeError):
        ta.mavp(df["close"])


def test_cdl_z_full_is_anchored(frames):
    """full=True is anchored, not whole-sample + bfill (see issue #149 follow-up)."""
    assert deviations(frames, "cdl_z", {"full": True}) == []
