# Fractal Adaptive Moving Average (FRAMA)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input, skip_leading_nan
from pandas_ta_classic.utils._njit import njit


@njit(cache=True)
def _frama_loop(price: np.ndarray, alpha: np.ndarray, start: int) -> np.ndarray:
    """FRAMA = alpha * price + (1 - alpha) * FRAMA[1], seeded with the price on bar ``start - 1``."""
    result = np.full(price.size, np.nan)
    previous = price[start - 1]
    for i in range(start, price.size):
        previous = alpha[i] * price[i] + (1.0 - alpha[i]) * previous
        result[i] = previous
    return result


@nan_on_short_input
@skip_leading_nan("high", "low", interior=True)
def frama(
    high: Series,
    low: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Fractal Adaptive Moving Average (FRAMA)"""
    # Validate Arguments
    length = _pos_int(length, 16, "length", gt=1)
    if length % 2:
        raise ValueError(f"frama() length must be an even integer, got {length}")
    high = verify_series(high, length + 1)
    low = verify_series(low, length + 1)
    offset = get_offset(offset)

    if high is None or low is None:
        return None

    # Calculate Result
    half = length // 2
    price = (high + low) / 2
    high_half, low_half = high.rolling(half).max(), low.rolling(half).min()
    n1 = (high_half - low_half) / half  # the recent half of the window
    n2 = n1.shift(half)  # the older half
    n3 = (high.rolling(length).max() - low.rolling(length).min()) / length
    with np.errstate(divide="ignore", invalid="ignore"):
        dimension = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
    # A window without a range in either half has no dimension: TA-Lib's FRAMA
    # then follows the price (alpha 1).
    alpha = np.exp(-4.6 * (dimension - 1)).where((n1 > 0) & (n2 > 0) & (n3 > 0), 1.0).clip(0.01, 1.0)
    frama = Series(_frama_loop(price.to_numpy(dtype=float), alpha.to_numpy(dtype=float), length), index=high.index)

    # Offset
    frama = apply_offset(frama, offset)

    frama = apply_fill(frama, **kwargs)

    # Name and Categorize it
    frama.name = f"FRAMA_{length}"
    frama.category = "overlap"

    return frama


frama.__doc__ = """Fractal Adaptive Moving Average (FRAMA)

John Ehlers' EMA whose smoothing follows the fractal dimension of the window:
fast in a trend (dimension near 1), slow in a range (near 2).

Sources:
    Ehlers, J. F. (2005), "Fractal Adaptive Moving Average", Technical
    Analysis of Stocks & Commodities 23(10).

Calculation:
    Default Inputs:
        length=16 (even)
    N1 = (MAX(high, n/2) - MIN(low, n/2)) / (n/2) over the recent half
    N2 = the same over the older half, N3 = (MAX(high, n) - MIN(low, n)) / n
    D = (log(N1 + N2) - log(N3)) / log(2)
    ALPHA = exp(-4.6 * (D - 1)), clipped to 0.01..1
    FRAMA = ALPHA * (high + low) / 2 + (1 - ALPHA) * FRAMA[1]

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    length (int): It's period, an even number. Default: 16
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
