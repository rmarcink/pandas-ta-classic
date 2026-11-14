# Derivative Oscillator (DOSC)
from typing import Any

from pandas import Series

from pandas_ta_classic.momentum.rsi import rsi
from pandas_ta_classic.overlap.ema import ema
from pandas_ta_classic.overlap.sma import sma
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input


@nan_on_short_input
def dosc(
    close: Series,
    length: int | None = None,
    first: int | None = None,
    second: int | None = None,
    signal: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Derivative Oscillator (DOSC)"""
    # Validate Arguments
    length = _pos_int(length, 14, "length")
    first = _pos_int(first, 5, "first")
    second = _pos_int(second, 3, "second")
    signal = _pos_int(signal, 9, "signal")
    close = verify_series(close, length + first + second + signal - 2)
    offset = get_offset(offset)

    if close is None:
        return None

    # Calculate Result
    rsi_ = rsi(close, length)
    smoothed = None if rsi_ is None else ema(rsi_, first)
    smoothed = None if smoothed is None else ema(smoothed, second)
    signal_ = None if smoothed is None else sma(smoothed, signal)
    if smoothed is None or signal_ is None:
        return None
    dosc = smoothed - signal_

    # Offset
    dosc = apply_offset(dosc, offset)

    dosc = apply_fill(dosc, **kwargs)

    # Name and Categorize it
    dosc.name = f"DOSC_{length}_{first}_{second}_{signal}"
    dosc.category = "momentum"

    return dosc


dosc.__doc__ = """Derivative Oscillator (DOSC)

Constance Brown's oscillator: an RSI smoothed twice by EMAs, less a simple
moving average of itself.

Sources:
    Brown, C. (1999), "Technical Analysis for the Trading Professional".

Calculation:
    Default Inputs:
        length=14, first=5, second=3, signal=9
    S = EMA(EMA(RSI(close, length), first), second)
    DOSC = S - SMA(S, signal)

Args:
    close (pd.Series): Series of 'close's
    length (int): The RSI period. Default: 14
    first (int): The period of the first EMA. Default: 5
    second (int): The period of the second EMA. Default: 3
    signal (int): The period of the SMA subtracted. Default: 9
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
