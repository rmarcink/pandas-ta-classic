# Swiss Army Knife Gaussian Filter (SWAK_GAUSS)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.cycles.swak_2php import _two_pole_alpha
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input, skip_leading_nan
from pandas_ta_classic.utils._njit import njit


@njit(cache=True)
def _swak_gauss_loop(x: np.ndarray, c0: float, a1: float, a2: float) -> np.ndarray:
    """y = c0 * x + a1 * y[1] + a2 * y[2], with both earlier outputs equal to x[0]."""
    result = np.empty(x.size)
    y1 = y2 = x[0]
    for i in range(x.size):
        y = a1 * y1 + (a2 * y2 + c0 * x[i])
        y2, y1 = y1, y
        result[i] = y
    return result


@nan_on_short_input
@skip_leading_nan("close", interior=True)
def swak_gauss(
    close: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Swiss Army Knife Gaussian Filter (SWAK_GAUSS)"""
    # Validate Arguments
    length = _pos_int(length, 20, "length", gt=1)
    close = verify_series(close)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    alpha = _two_pole_alpha(length)
    om = 1.0 - alpha
    gauss = Series(_swak_gauss_loop(close.to_numpy(dtype=float), alpha * alpha, 2.0 * om, -(om * om)), index=close.index)

    # Offset
    gauss = apply_offset(gauss, offset)

    gauss = apply_fill(gauss, **kwargs)

    # Name and Categorize it
    gauss.name = f"SWAKGAUSS_{length}"
    gauss.category = "overlap"

    return gauss


swak_gauss.__doc__ = """Swiss Army Knife Gaussian Filter (SWAK_GAUSS)

John Ehlers' two-pole Gaussian low-pass filter from his "Swiss Army Knife"
family: two EMAs in cascade with the same constant. Seeded as TA-Lib's
SWAK_GAUSS: the bars before the first repeat it, so it starts on the first
close.

Sources:
    Ehlers, J. F. (2006), "Swiss Army Knife Indicator", Technical Analysis
    of Stocks & Commodities 24(1).

Calculation:
    Default Inputs:
        length=20
    B = 2.415 * (1 - cos(2 * pi / length)), ALPHA = -B + sqrt(B ** 2 + 2 * B)
    GAUSS = ALPHA ** 2 * close + 2 * (1 - ALPHA) * GAUSS[1] - (1 - ALPHA) ** 2 * GAUSS[2]

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
