# Percent Rank (PERCENTRANK)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _bool_param, _pos_int, nan_on_short_input


@nan_on_short_input
def percentrank(
    close: Series,
    length: int | None = None,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Percent Rank (PERCENTRANK)"""
    # Validate Arguments
    length = _pos_int(length, 100, "length", gt=1)
    close = verify_series(close, length + 1)
    offset = get_offset(offset)
    mode_talib = _bool_param(talib, False, "talib")

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_talib:
        from talib import PERCENTRANK

        percentrank = PERCENTRANK(close, length)
    else:
        values = close.to_numpy(dtype=float)
        result = np.full(values.size, np.nan)
        if values.size > length:
            windows = np.lib.stride_tricks.sliding_window_view(values, length + 1)
            # Compared in blocks of rows, so a temporary holds about 2**20 values at most;
            # in one piece, a million bars at length 500 took 500 MiB.
            step = max(1, 2**20 // (length + 1))
            below = np.concatenate(
                [(block[:, :-1] < block[:, -1:]).sum(axis=1) for block in (windows[i : i + step] for i in range(0, len(windows), step))]
            )
            rank = 100.0 * below / length
            # A missing bar in the window or on the bar itself has no rank.
            missing = np.concatenate(([0], np.cumsum(np.isnan(values))))
            rank[missing[length + 1 :] - missing[: -length - 1] > 0] = np.nan
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
    length (int): Number of earlier values compared, at least 2. Default: 100
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
