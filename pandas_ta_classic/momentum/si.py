# Wilder Swing Index (SI)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _name_tokens, _pos_float, nan_on_short_input


def _swing_index(open_: Series, high: Series, low: Series, close: Series, limit: float) -> Series:
    """Wilder's swing index of each bar against the bar before it."""
    prev_close, prev_open = close.shift(1), open_.shift(1)
    to_high = (high - prev_close).abs()
    to_low = (low - prev_close).abs()
    bar_range = high - low
    prev_body = (prev_close - prev_open).abs()
    # R takes the largest of the three moves, with the high's move first on a tie, then the low's.
    high_largest = (to_high >= to_low) & (to_high >= bar_range)
    low_largest = ~high_largest & (to_low >= bar_range)
    r = np.where(
        high_largest,
        to_high - 0.5 * to_low + 0.25 * prev_body,
        np.where(low_largest, to_low - 0.5 * to_high + 0.25 * prev_body, bar_range + 0.25 * prev_body),
    )
    r = Series(r, index=close.index)
    k = np.maximum(to_high, to_low)
    move = close - prev_close + 0.5 * (close - open_) + 0.25 * (prev_close - prev_open)
    # A bar with no move to weigh (R == 0) swings by nothing, as TA-Lib's SI reads.
    si = (50.0 * move / r.where(r != 0) * k / limit).mask(r == 0, 0.0)
    return si.mask(prev_close.isna() | r.isna())


@nan_on_short_input
def si(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    limit: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Wilder Swing Index (SI)"""
    # Validate Arguments
    limit = _pos_float(limit, 3.0, "limit")
    open_ = verify_series(open_, 2)
    high = verify_series(high, 2)
    low = verify_series(low, 2)
    close = verify_series(close, 2)
    offset = get_offset(offset)

    if open_ is None or high is None or low is None or close is None:
        return None

    # Calculate Result
    si = _swing_index(open_, high, low, close, limit)

    # Offset
    si = apply_offset(si, offset)

    si = apply_fill(si, **kwargs)

    # Name and Categorize it
    si.name = "SI" + _name_tokens(("limit", limit, 3.0))
    si.category = "momentum"

    return si


si.__doc__ = """Wilder Swing Index (SI)

J. Welles Wilder's measure of each bar's swing against the bar before it,
from the opens, highs, lows and closes, scaled by the limit move.

Sources:
    Wilder, J. W. (1978), "New Concepts in Technical Trading Systems".

Calculation:
    Default Inputs:
        limit=3.0
    A = |high - close[1]|, B = |low - close[1]|, C = high - low, D = |close[1] - open[1]|
    R = A - B / 2 + D / 4 if A is the largest, B - A / 2 + D / 4 if B is,
        else C + D / 4
    K = max(A, B)
    SI = 50 * (close - close[1] + (close - open) / 2 + (close[1] - open[1]) / 4) / R * K / limit

Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    limit (float): The limit move of the market. Default: 3.0
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
