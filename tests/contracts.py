"""Shared harness of the registry-wide contract tests.

Every ``test_*_contract.py`` module, ``test_degenerate_input.py`` and
``test_oracle_registry.py`` runs each registered indicator against one class of
input. They share three things, kept here so they cannot drift apart:

* the data: :func:`spy` returns the last *bars* rows of ``SPY_D``;
* the registry: :data:`OHLCV_INDICATORS` and :func:`call`, which feeds an
  indicator the OHLCV columns its signature names;
* known exceptions: :func:`params` and :func:`pin` turn a table of
  ``{key: reason}`` into strict xfails, so an entry that starts passing fails
  and the table cannot go stale.

Every reason starts with one of :data:`REASON_PREFIXES`:

* ``OPEN:`` a defect, to be fixed;
* ``KEPT:`` a decided behaviour that breaks the contract on purpose;
* ``TALIB:`` a difference that is TA-Lib's, not ours.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import pandas_ta_classic as ta

SPY_CSV = Path(__file__).parent.parent / "examples" / "data" / "SPY_D.csv"

COLUMNS = {"open_": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
# Not a function of one OHLCV frame: two-series math, signal helpers, a
# required benchmark or periods series, the ma() dispatcher.
NOT_OHLCV = {"add", "sub", "mult", "div", "above", "above_value", "below", "below_value", "cross", "cross_value"}
NOT_OHLCV |= {"long_run", "short_run", "tsignals", "xsignals", "beta", "correl", "mavp", "ma"}
# vp returns price bins, not one row per bar.
NOT_TIME_SERIES = {"vp"}
OHLCV_INDICATORS = sorted({name for names in ta.Category.values() for name in names} - NOT_OHLCV - NOT_TIME_SERIES)

REASON_PREFIXES = ("OPEN:", "KEPT:", "TALIB:")


@cache
def _spy_csv() -> pd.DataFrame:
    df = pd.read_csv(SPY_CSV, index_col="date", parse_dates=True)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df.columns = df.columns.str.lower()
    return df


def spy(bars: int | None = None) -> pd.DataFrame:
    """The last *bars* rows of SPY_D (all of it for None), as a copy the caller may change."""
    df = _spy_csv()
    return (df if bars is None else df.iloc[-bars:]).copy()


def call(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Run indicator *name* on the columns of *df* its signature names; a Series comes back as a frame."""
    fn = getattr(ta, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fn(**{p: df[c] for p, c in COLUMNS.items() if p in inspect.signature(fn).parameters})
    return result.to_frame() if isinstance(result, pd.Series) else result


def _xfail(reason: str) -> pytest.MarkDecorator:
    if not reason.startswith(REASON_PREFIXES):
        raise ValueError(f"a known-exception reason must start with one of {REASON_PREFIXES}, got {reason!r}")
    return pytest.mark.xfail(strict=True, reason=reason)


def params(
    rows: Iterable[tuple],
    known: Mapping[Hashable, str],
    *,
    key: Callable[[tuple], Hashable] = lambda row: row,
    id: Callable[[tuple], str] = lambda row: "-".join(map(str, row)),
) -> list[Any]:
    """One ``pytest.param`` per row; a row whose key is in *known* is a strict xfail with its reason."""
    out = []
    for row in rows:
        reason = known.get(key(row))
        out.append(pytest.param(*row, marks=[_xfail(reason)] if reason else [], id=id(row)))
    return out


def pin(request: pytest.FixtureRequest, known: Mapping[Hashable, str], key: Hashable) -> None:
    """:func:`params` for a test whose parameters are built elsewhere: mark this run when *key* is known."""
    reason = known.get(key)
    if reason:
        request.applymarker(_xfail(reason))
