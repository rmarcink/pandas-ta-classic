# Swiss Army Knife Two-Pole High-Pass Filter (SWAK_2PHP)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input, skip_leading_nan
from pandas_ta_classic.utils._njit import njit


def _two_pole_alpha(length: int) -> float:
    """The smoothing constant of Ehlers' two-pole filters (Gaussian, Butterworth, high-pass) for *length*."""
    s = np.sin(np.pi / length)
    b = 2.415 * (2.0 * s * s)
    return float(-b + np.sqrt(b * b + 2.0 * b))


@njit(cache=True)
def _swak_2php_loop(x: np.ndarray, c0: float, a1: float, a2: float) -> np.ndarray:
    """y = c0 * (x - 2 x[1] + x[2]) + a1 * y[1] + a2 * y[2], with x[1] = x[2] = x[0] and y[1] = y[2] = 0 before the first bar."""
    result = np.empty(x.size)
    x1 = x2 = x[0]
    y1 = y2 = 0.0
    for i in range(x.size):
        y = a1 * y1 + (a2 * y2 + c0 * (x[i] - 2.0 * x1 + x2))
        x2, x1 = x1, x[i]
        y2, y1 = y1, y
        result[i] = y
    # Bar 0 is the seed (0.0 whatever the data), not a value; TA-Lib publishes it.
    result[0] = np.nan
    return result


@nan_on_short_input
@skip_leading_nan("close", interior=True)
def swak_2php(
    close: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Swiss Army Knife Two-Pole High-Pass Filter (SWAK_2PHP)"""
    # Validate Arguments
    length = _pos_int(length, 20, "length", gt=1)
    close = verify_series(close)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    alpha = _two_pole_alpha(length)
    om = 1.0 - alpha
    hp = Series(_swak_2php_loop(close.to_numpy(dtype=float), (1.0 - alpha / 2.0) ** 2, 2.0 * om, -(om * om)), index=close.index)

    # Offset
    hp = apply_offset(hp, offset)

    hp = apply_fill(hp, **kwargs)

    # Name and Categorize it
    hp.name = f"SWAK2PHP_{length}"
    hp.category = "cycles"

    return hp


swak_2php.__doc__ = """Swiss Army Knife Two-Pole High-Pass Filter (SWAK_2PHP)

John Ehlers' two-pole high-pass filter from his "Swiss Army Knife" family: a
sharper cut of the cycles longer than ``length`` bars than ``swak_hp``. Seeded
as TA-Lib's SWAK_2PHP: the bars before the first repeat it, so a constant
reads 0.

Sources:
    Ehlers, J. F. (2006), "Swiss Army Knife Indicator", Technical Analysis
    of Stocks & Commodities 24(1).

Calculation:
    Default Inputs:
        length=20
    B = 2.415 * (1 - cos(2 * pi / length)), ALPHA = -B + sqrt(B ** 2 + 2 * B)
    HP = (1 - ALPHA / 2) ** 2 * (close - 2 * close[1] + close[2])
         + 2 * (1 - ALPHA) * HP[1] - (1 - ALPHA) ** 2 * HP[2]

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
