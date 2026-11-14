# WaveTrend Oscillator (WAVETREND)
from typing import Any

from pandas import DataFrame, Series

from pandas_ta_classic.overlap.ema import ema
from pandas_ta_classic.overlap.sma import sma
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input


@nan_on_short_input
def wavetrend(
    high: Series,
    low: Series,
    close: Series,
    channel: int | None = None,
    average: int | None = None,
    signal: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> DataFrame | None:
    """Indicator: WaveTrend Oscillator (WAVETREND)"""
    # Validate Arguments
    channel = _pos_int(channel, 10, "channel")
    average = _pos_int(average, 21, "average")
    signal = _pos_int(signal, 4, "signal")
    _length = 2 * channel + average + signal - 3
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    close = verify_series(close, _length)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    typical = (high + low + close) / 3
    esa = ema(typical, channel)
    deviation = None if esa is None else ema((typical - esa).abs(), channel)
    if esa is None or deviation is None:
        return None
    # A price that never left its average has no deviation: CI reads 0, as TA-Lib's WAVETREND does.
    ci = ((typical - esa) / (0.015 * deviation.where(deviation != 0))).mask(deviation == 0, 0.0)
    wt = ema(ci, average)
    wt_signal = None if wt is None else sma(wt, signal)
    if wt is None or wt_signal is None:
        return None

    # Offset
    wt, wt_signal = apply_offset([wt, wt_signal], offset)

    wt, wt_signal = apply_fill([wt, wt_signal], **kwargs)

    # Name and Categorize it
    _props = f"_{channel}_{average}_{signal}"
    wt.name = f"WT{_props}"
    wt_signal.name = f"WTs{_props}"
    wt.category = wt_signal.category = "momentum"

    df = DataFrame({wt.name: wt, wt_signal.name: wt_signal})
    df.name = f"WAVETREND{_props}"
    df.category = "momentum"

    return df


wavetrend.__doc__ = """WaveTrend Oscillator (WAVETREND)

LazyBear's WaveTrend: the typical price's distance from its EMA in units of
its mean deviation, as in a CCI, smoothed by an EMA, with an SMA signal line.

Sources:
    https://www.tradingview.com/script/2KE8wTuF-Indicator-WaveTrend-Oscillator-WT/

Calculation:
    Default Inputs:
        channel=10, average=21, signal=4
    AP = (high + low + close) / 3
    ESA = EMA(AP, channel)
    D = EMA(|AP - ESA|, channel)
    CI = (AP - ESA) / (0.015 * D)
    WT = EMA(CI, average)
    WTs = SMA(WT, signal)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    channel (int): The period of ESA and D. Default: 10
    average (int): The period of the EMA of CI. Default: 21
    signal (int): The period of the signal SMA. Default: 4
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: WT and WTs columns.
"""
