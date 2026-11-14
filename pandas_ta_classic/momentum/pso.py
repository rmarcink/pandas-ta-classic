# Premier Stochastic Oscillator (PSO)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.overlap.ema import ema
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input


@nan_on_short_input
def pso(
    high: Series,
    low: Series,
    close: Series,
    length: int | None = None,
    smooth: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Premier Stochastic Oscillator (PSO)"""
    # Validate Arguments
    length = _pos_int(length, 8, "length")
    smooth = _pos_int(smooth, 5, "smooth")
    _length = length + 2 * (smooth - 1)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    lowest = low.rolling(length).min()
    window_range = high.rolling(length).max() - lowest
    # A window with no range puts the close in the middle (50), as TA-Lib's PSO does.
    stoch_k = (100 * (close - lowest) / window_range.where(window_range != 0)).mask(window_range == 0, 50.0)
    smoothed = ema(0.1 * (stoch_k - 50), smooth)
    smoothed = None if smoothed is None else ema(smoothed, smooth)
    if smoothed is None:
        return None
    growth = np.exp(smoothed)
    pso = (growth - 1) / (growth + 1)

    # Offset
    pso = apply_offset(pso, offset)

    pso = apply_fill(pso, **kwargs)

    # Name and Categorize it
    pso.name = f"PSO_{length}_{smooth}"
    pso.category = "momentum"

    return pso


pso.__doc__ = """Premier Stochastic Oscillator (PSO)

Lee Leibfarth's stochastic, normalized around zero, smoothed twice and
squashed into -1..1.

Sources:
    Leibfarth, L. (2008), "Premier Stochastic Oscillator", Technical Analysis
    of Stocks & Commodities 26(8).

Calculation:
    Default Inputs:
        length=8, smooth=5
    NSK = 0.1 * (STOCHF %K(length) - 50)
    S = EMA(EMA(NSK, smooth), smooth)
    PSO = (exp(S) - 1) / (exp(S) + 1)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): The stochastic period. Default: 8
    smooth (int): The period of both EMAs. Default: 5
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
