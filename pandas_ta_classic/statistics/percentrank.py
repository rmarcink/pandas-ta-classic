# Percent Rank (PERCENTRANK)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input


@nan_on_short_input
def percentrank(
    close: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Percent Rank (PERCENTRANK)"""
    # Validate Arguments
    length = _pos_int(length, 100, "length")
    close = verify_series(close, length + 1)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    values = close.to_numpy(dtype=float)
    result = np.full(values.size, np.nan)
    if values.size > length:
        windows = np.lib.stride_tricks.sliding_window_view(values, length + 1)
        below = (windows[:, :-1] < windows[:, -1:]).sum(axis=1)
        rank = 100.0 * below / length
        # A missing bar in the window or on the bar itself has no rank.
        rank[np.isnan(windows).any(axis=1)] = np.nan
        result[length:] = rank
    percentrank = Series(result, index=close.index)

    # Offset
    percentrank = apply_offset(percentrank, offset)

    percentrank = apply_fill(percentrank, **kwargs)

    # Name and Categorize it
    percentrank.name = f"PERCENTRANK_{length}"
    percentrank.category = "statistics"

    return percentrank


percentrank.__doc__ = """Percent Rank (PERCENTRANK)

The share of the previous ``length`` values that are strictly below the
current one, in percent: 100 is a new high of the window, 0 a value no
earlier value was below.

Sources:
    https://ta-lib.org/functions/  (PERCENTRANK)

Calculation:
    Default Inputs:
        length=100
    PERCENTRANK = 100 * COUNT(close[i] < close, i = 1..length) / length

Args:
    close (pd.Series): Series of 'close's
    length (int): Number of earlier values compared. Default: 100
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
