"""Integer and nullable inputs give the float64 answer.

Feeds deliver prices in ticks (``int64``), and pandas readers hand out the
nullable extension dtypes (``Int64``, ``Float64``) as soon as a column has a
missing value or ``dtype_backend="numpy_nullable"`` is set. The numbers are the
same, so the result must be too. Two defects of this kind are already known:
``non_zero_range`` downcast its epsilon to 0 on ``int64`` (0.8.32), and the
nullable dtypes failed in numpy-backed kernels (``fix/nullable-numeric-input``).

Each input is compared against the same values as ``float64``, for every
registered indicator: the NaN mask must match and the values agree to 1e-12.

Indicators are discovered through ``ta.Category``.
"""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta

COLUMNS = {"open_": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
NOT_OHLCV = {"add", "sub", "mult", "div", "above", "above_value", "below", "below_value", "cross", "cross_value"}
NOT_OHLCV |= {"long_run", "short_run", "tsignals", "xsignals", "beta", "correl", "mavp", "ma", "vp"}
INDICATORS = sorted({name for names in ta.Category.values() for name in names} - NOT_OHLCV)
DTYPES = ("int64", "Int64", "Float64")
RTOL = 1e-12

# (indicator, dtype) pairs that break the contract today, as strict xfails: one
# that starts passing fails, so the table cannot go stale.
_NULLABLE = "OPEN DEFECT, fixed on branch fix/nullable-numeric-input (all 48 pass there): verify_series keeps the extension dtype"
_FAILS_ON_FLOAT64_EXT = {"cdl_doji", "cdl_inside", "cdl_pattern", "decreasing", "increasing", "smc_sweep", "td_seq", "ttm_trend"}
_FAILS_ON_INT64_EXT = _FAILS_ON_FLOAT64_EXT | {
    "aberration", "amat", "aobv", "atr", "ce", "cksp", "dema", "efi", "ema", "eri", "kc", "macdext", "massi", "mmar",
    "natr", "pgo", "pmax", "pvo", "qqe", "rma", "rsi", "smi", "stc", "stochrsi", "supertrend", "t3", "tema", "thermo",
    "trix", "trixh", "tsi", "zlma",
}  # fmt: skip
OPEN_FINDINGS: dict[tuple[str, str], str] = {
    **{(name, "Float64"): _NULLABLE for name in _FAILS_ON_FLOAT64_EXT},
    **{(name, "Int64"): _NULLABLE for name in _FAILS_ON_INT64_EXT},
}


@pytest.fixture(scope="module")
def ticks() -> pd.DataFrame:
    """SPY_D in cents, as integers, so every dtype holds exactly the same numbers."""
    df = pd.read_csv(Path(__file__).parent.parent / "examples" / "data" / "SPY_D.csv", index_col="date", parse_dates=True)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df.columns = df.columns.str.lower()
    df = df.iloc[-1000:]
    return (df[["open", "high", "low", "close"]] * 100).round().join(df[["volume"]].round()).astype("int64")


def _call(name: str, df: pd.DataFrame) -> pd.DataFrame:
    fn = getattr(ta, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fn(**{p: df[c] for p, c in COLUMNS.items() if p in inspect.signature(fn).parameters})
    return result.to_frame() if isinstance(result, pd.Series) else result


def _params():
    for dtype in DTYPES:
        for name in INDICATORS:
            reason = OPEN_FINDINGS.get((name, dtype))
            marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
            yield pytest.param(name, dtype, marks=marks, id=f"{name}-{dtype}")


@pytest.mark.parametrize(("name", "dtype"), list(_params()))
def test_integer_and_nullable_input_give_the_float_answer(name: str, dtype: str, ticks: pd.DataFrame) -> None:
    expected = _call(name, ticks.astype("float64"))
    result = _call(name, ticks.astype(dtype))
    assert list(result.columns) == list(expected.columns), f"{name}: columns {list(result.columns)} for {dtype}"
    for column in expected.columns:
        e = expected[column].to_numpy(float, na_value=np.nan)
        r = result[column].to_numpy(float, na_value=np.nan)
        np.testing.assert_array_equal(np.isnan(r), np.isnan(e), err_msg=f"{name} {column}: NaN mask differs for {dtype}")
        np.testing.assert_allclose(
            r, e, rtol=RTOL, atol=RTOL * float(np.nanmax(np.abs(e), initial=0.0)), equal_nan=True, err_msg=f"{name} {column}: {dtype}"
        )
