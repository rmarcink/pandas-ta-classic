# Swiss Army Knife Band-Pass Filter (SWAK_BP)
import math
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _bool_param, _pos_float, _pos_int, nan_on_short_input, skip_leading_nan
from pandas_ta_classic.utils._njit import fma, njit


@njit(cache=True)
def _swak_bp_loop(x: np.ndarray, c0: float, a1: float, a2: float) -> np.ndarray:
    """y = c0 * (x - x[2]) + a1 * y[1] + a2 * y[2], with x[1] = x[2] = x[0] and y[1] = y[2] = 0 before the first bar."""
    result = np.empty(x.size)
    x1 = x2 = x[0]
    y1 = y2 = 0.0
    for i in range(x.size):
        y = fma(a1, y1, fma(a2, y2, c0 * (x[i] - x2)))
        x2, x1 = x1, x[i]
        y2, y1 = y1, y
        result[i] = y
    # Bar 0 is the seed (0.0 whatever the data), not a value; TA-Lib publishes it.
    result[0] = np.nan
    return result


@nan_on_short_input
@skip_leading_nan("close", interior=True)
def swak_bp(
    close: Series,
    length: int | None = None,
    delta: float | None = None,
    talib: bool | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Swiss Army Knife Band-Pass Filter (SWAK_BP)"""
    # Validate Arguments
    # TA-Lib's lower limits. They also keep the pass band's half width 4 * pi * delta / length
    # below a quarter turn, which needs delta < length / 8.
    length = _pos_int(length, 20, "length", gt=4)
    delta = _pos_float(delta, 0.1, "delta", gt=None, ge=0.05)
    if delta > 0.5:
        raise ValueError(f"swak_bp() delta must be <= 0.5, got {delta}")
    close = verify_series(close)
    offset = get_offset(offset)
    mode_talib = _bool_param(talib, False, "talib")

    if close is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_talib:
        from talib import SWAK_BP

        bp = SWAK_BP(close, length, float(delta))
    else:
        beta = math.cos(2.0 * math.pi / length)
        t = 4.0 * math.pi * delta / length
        # (1 - sin t) / cos t is Ehlers' gamma - sqrt(gamma**2 - 1) without its cancellation as t -> 0.
        alpha = (1.0 - math.sin(t)) / math.cos(t)
        bp = Series(_swak_bp_loop(close.to_numpy(dtype=float), (1.0 - alpha) / 2.0, beta * (1.0 + alpha), -alpha), index=close.index)

    # Offset
    bp = apply_offset(bp, offset)

    bp = apply_fill(bp, **kwargs)

    # Name and Categorize it
    bp.name = f"SWAKBP_{length}_{delta}"
    bp.category = "cycles"

    return bp


swak_bp.__doc__ = """Swiss Army Knife Band-Pass Filter (SWAK_BP)

John Ehlers' band-pass filter from his "Swiss Army Knife" family: it keeps the
cycles near ``length`` bars and removes the trend and the noise. Seeded as
TA-Lib's SWAK_BP: the bars before the first repeat it, so bar 0 reads NaN and
a constant reads 0 from bar 1 on.

Sources:
    Ehlers, J. F. (2006), "Swiss Army Knife Indicator", Technical Analysis
    of Stocks & Commodities 24(1).

Calculation:
    Default Inputs:
        length=20, delta=0.1
    BETA = cos(2 * pi / length), T = 4 * pi * delta / length
    ALPHA = (1 - sin(T)) / cos(T)
    BP = (1 - ALPHA) / 2 * (close - close[2]) + BETA * (1 + ALPHA) * BP[1] - ALPHA * BP[2]

Args:
    close (pd.Series): Series of 'close's
    length (int): The cycle period the filter passes, at least 5. Default: 20
    delta (float): The half width of the pass band, as a fraction of the
        period, from 0.05 to 0.5. Default: 0.1
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: False
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
