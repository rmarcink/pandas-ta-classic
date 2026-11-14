# Stochastic Momentum Index (STOCH_MI)
from typing import Any

from pandas import DataFrame, Series

from pandas_ta_classic.overlap.ema import ema
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, _windows_in_order, nan_on_short_input


@nan_on_short_input
def stoch_mi(
    high: Series,
    low: Series,
    close: Series,
    length: int | None = None,
    fast: int | None = None,
    slow: int | None = None,
    signal: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame | None:
    """Indicator: Stochastic Momentum Index (STOCH_MI)"""
    # Validate Arguments
    length = _pos_int(length, 13, "length")
    fast = _pos_int(fast, 2, "fast")
    slow = _pos_int(slow, 25, "slow")
    _windows_in_order("stoch_mi", fast=fast, slow=slow)
    signal = _pos_int(signal, 9, "signal")
    _length = length + fast + slow + signal - 3
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    highest = high.rolling(length).max()
    lowest = low.rolling(length).min()
    # The slow EMA first, then the fast one, as TA-Lib's SMI: the order decides the seeds.
    distance = ema(close - 0.5 * (highest + lowest), slow)
    distance = None if distance is None else ema(distance, fast)
    half_range = ema(highest - lowest, slow)
    half_range = None if half_range is None else ema(half_range, fast)
    if distance is None or half_range is None:
        return None
    half_range = 0.5 * half_range
    # A window that never had a range reads 0, as TA-Lib's SMI does.
    smi = (100 * distance / half_range.where(half_range > 0)).mask(half_range <= 0, 0.0)
    smi_signal = ema(smi, signal)
    if smi_signal is None:
        return None

    # Offset
    smi, smi_signal = apply_offset([smi, smi_signal], offset)

    smi, smi_signal = apply_fill([smi, smi_signal], **kwargs)

    # Name and Categorize it
    _props = f"_{length}_{fast}_{slow}_{signal}"
    smi.name = f"STOCHMI{_props}"
    smi_signal.name = f"STOCHMIs{_props}"
    smi.category = smi_signal.category = "momentum"

    df = DataFrame({smi.name: smi, smi_signal.name: smi_signal})
    df.name = f"STOCHMI{_props}"
    df.category = "momentum"

    return df


stoch_mi.__doc__ = """Stochastic Momentum Index (STOCH_MI)

William Blau's Stochastic Momentum Index: the close's distance from the
middle of the window's range, as a percentage of half that range, both
smoothed by two EMAs. TA-Lib calls it SMI; ``smi`` here is a different
indicator, Blau's SMI Ergodic of the TSI.

Sources:
    Blau, W. (1993), "Stochastic Momentum", Technical Analysis of Stocks &
    Commodities 11(1).

Calculation:
    Default Inputs:
        length=13, fast=2, slow=25, signal=9
    HH = MAX(high, length), LL = MIN(low, length)
    STOCHMI = 100 * EMA(EMA(close - (HH + LL) / 2, slow), fast)
                  / (0.5 * EMA(EMA(HH - LL, slow), fast))
    STOCHMIs = EMA(STOCHMI, signal)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): The window of the highest high and lowest low. Default: 13
    fast (int): The period of the second EMA. Default: 2
    slow (int): The period of the first EMA. Default: 25
    signal (int): The period of the signal EMA. Default: 9
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: STOCHMI and STOCHMIs columns.
"""
