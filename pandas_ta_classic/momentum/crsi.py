# Connors Relative Strength Index (CRSI)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.momentum.roc import roc
from pandas_ta_classic.momentum.rsi import rsi
from pandas_ta_classic.statistics.percentrank import percentrank
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, _positive_inputs, nan_on_short_input
from pandas_ta_classic.utils._njit import njit


@njit(cache=True)
def _streak(close: np.ndarray) -> np.ndarray:
    """Consecutive up closes counted positive, down closes negative, 0 on an unchanged close.

    A bar without a previous close (the first, or one after a missing bar)
    starts the count again at 0.
    """
    result = np.full(close.size, np.nan)
    for i in range(close.size):
        if np.isnan(close[i]):
            continue
        if i == 0 or np.isnan(close[i - 1]):
            result[i] = 0.0
        elif close[i] > close[i - 1]:
            previous = result[i - 1]
            result[i] = previous + 1.0 if previous > 0 else 1.0
        elif close[i] < close[i - 1]:
            previous = result[i - 1]
            result[i] = previous - 1.0 if previous < 0 else -1.0
        else:
            result[i] = 0.0
    return result


@nan_on_short_input
@_positive_inputs("close")
def crsi(
    close: Series,
    length: int | None = None,
    streak: int | None = None,
    rank: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Connors Relative Strength Index (CRSI)"""
    # Validate Arguments
    length = _pos_int(length, 3, "length")
    streak = _pos_int(streak, 2, "streak")
    rank = _pos_int(rank, 100, "rank")
    close = verify_series(close, max(length, streak, rank) + 1)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    price_rsi = rsi(close, length)
    streak_rsi = rsi(Series(_streak(close.to_numpy(dtype=float)), index=close.index), streak)
    change = roc(close, 1)
    if price_rsi is None or streak_rsi is None or change is None:
        return None
    change_rank = percentrank(change, rank)
    if change_rank is None:
        return None
    crsi = (price_rsi + streak_rsi + change_rank) / 3.0

    # Offset
    crsi = apply_offset(crsi, offset)

    crsi = apply_fill(crsi, **kwargs)

    # Name and Categorize it
    crsi.name = f"CRSI_{length}_{streak}_{rank}"
    crsi.category = "momentum"

    return crsi


crsi.__doc__ = """Connors Relative Strength Index (CRSI)

Larry Connors' composite of three oscillators: a short RSI of the close, an
RSI of the up/down streak length and the percent rank of the one-bar change.

Sources:
    https://www.tradingview.com/support/solutions/43000502017-connors-rsi-crsi/

Calculation:
    Default Inputs:
        length=3, streak=2, rank=100
    STREAK = consecutive up closes (+1, +2, ...), down closes (-1, -2, ...), 0 unchanged
    CRSI = (RSI(close, length) + RSI(STREAK, streak) + PERCENTRANK(ROC(close, 1), rank)) / 3

Args:
    close (pd.Series): Series of 'close's
    length (int): The RSI period of the close. Default: 3
    streak (int): The RSI period of the streak. Default: 2
    rank (int): The percent rank period of the one-bar change. Default: 100
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
