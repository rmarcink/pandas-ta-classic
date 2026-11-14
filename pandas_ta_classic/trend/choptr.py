# Choppiness Index, true range box (CHOPTR)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input
from pandas_ta_classic.volatility.true_range import true_range


@nan_on_short_input
def choptr(
    high: Series,
    low: Series,
    close: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Choppiness Index, true range box (CHOPTR)"""
    # Validate Arguments
    length = _pos_int(length, 14, "length", gt=1)
    high = verify_series(high, length + 1)
    low = verify_series(low, length + 1)
    close = verify_series(close, length + 1)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    prev_close = close.shift(1)
    true_high = np.fmax(high, prev_close).mask(prev_close.isna())
    true_low = np.fmin(low, prev_close).mask(prev_close.isna())
    tr = true_range(high, low, close)
    if tr is None:
        return None
    tr_sum = tr.rolling(length).sum()
    box = true_high.rolling(length).max() - true_low.rolling(length).min()
    # The box spans every true range in it, so it is zero only where they all are:
    # such a window reads 100, as TA-Lib's CHOPTR does.
    with np.errstate(divide="ignore", invalid="ignore"):
        choptr = (100 * np.log10(tr_sum / box.where(box != 0)) / np.log10(length)).mask(box == 0, 100.0)

    # Offset
    choptr = apply_offset(choptr, offset)

    choptr = apply_fill(choptr, **kwargs)

    # Name and Categorize it
    choptr.name = f"CHOPTR_{length}"
    choptr.category = "trend"

    return choptr


choptr.__doc__ = """Choppiness Index, true range box (CHOPTR)

The Choppiness Index with its box drawn around the true ranges: the highest
true high (the high or the previous close) less the lowest true low. Unlike
``chop``, whose box is the window's high-low range, a gap cannot push the
index above 100.

Sources:
    https://ta-lib.org/functions/  (CHOPTR)

Calculation:
    Default Inputs:
        length=14
    TRUE_HIGH = max(high, close[1]), TRUE_LOW = min(low, close[1])
    CHOPTR = 100 * log10(SUM(TR, length) / (MAX(TRUE_HIGH, length) - MIN(TRUE_LOW, length))) / log10(length)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
