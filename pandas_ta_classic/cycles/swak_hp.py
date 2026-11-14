# Swiss Army Knife High-Pass Filter (SWAK_HP)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input, skip_leading_nan
from pandas_ta_classic.utils._njit import njit


@njit(cache=True)
def _swak_hp_loop(x: np.ndarray, c0: float, a1: float) -> np.ndarray:
    """y = c0 * (x - x[1]) + a1 * y[1], with x[1] = x[0] and y[1] = 0 before the first bar."""
    result = np.empty(x.size)
    x1 = x[0]
    y1 = 0.0
    for i in range(x.size):
        y = a1 * y1 + c0 * (x[i] - x1)
        x1 = x[i]
        y1 = y
        result[i] = y
    # Bar 0 is the seed (0.0 whatever the data), not a value; TA-Lib publishes it.
    result[0] = np.nan
    return result


@nan_on_short_input
@skip_leading_nan("close", interior=True)
def swak_hp(
    close: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Swiss Army Knife High-Pass Filter (SWAK_HP)"""
    # Validate Arguments
    length = _pos_int(length, 20, "length", gt=1)
    close = verify_series(close)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    h = np.pi / length
    sh, ch = np.sin(h), np.cos(h)
    alpha = 2.0 * sh * (ch - sh) / (1.0 - 2.0 * sh * sh)
    hp = Series(_swak_hp_loop(close.to_numpy(dtype=float), 1.0 - alpha / 2.0, 1.0 - alpha), index=close.index)

    # Offset
    hp = apply_offset(hp, offset)

    hp = apply_fill(hp, **kwargs)

    # Name and Categorize it
    hp.name = f"SWAKHP_{length}"
    hp.category = "cycles"

    return hp


swak_hp.__doc__ = """Swiss Army Knife High-Pass Filter (SWAK_HP)

John Ehlers' one-pole high-pass filter from his "Swiss Army Knife" family: it
removes the cycles longer than ``length`` bars, the trend. Seeded as TA-Lib's
SWAK_HP: the bar before the first repeats it, so a constant reads 0.

Sources:
    Ehlers, J. F. (2006), "Swiss Army Knife Indicator", Technical Analysis
    of Stocks & Commodities 24(1).

Calculation:
    Default Inputs:
        length=20
    H = pi / length
    ALPHA = 2 * sin(H) * (cos(H) - sin(H)) / (1 - 2 * sin(H) ** 2)
    HP = (1 - ALPHA / 2) * (close - close[1]) + (1 - ALPHA) * HP[1]

Args:
    close (pd.Series): Series of 'close's
    length (int): The cutoff period. Default: 20
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
