# Mesa Sine Wave (MSW)
from typing import Any, Optional

import numpy as np
from pandas import DataFrame, Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series


def msw(
    close: Series,
    period: Optional[int] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[DataFrame]:
    """Indicator: Mesa Sine Wave (MSW)

    Identifies cycles using a DFT-based approach from Ehlers (2001).
    Returns two oscillator series: sine and lead (sine + 45°).
    """
    # Validate Arguments
    period = int(period) if period and period > 1 else 5
    close = verify_series(close, period + 1)
    offset = get_offset(offset)

    if close is None:
        return None

    # Tulipy passthrough
    mode_tu = kwargs.get("tulipy", True)
    if Imports["tulipy"] and mode_tu:
        import tulipy as tu

        result = tu.msw(np.array(close, dtype=float), period=period)
        _size = result[0].size
        _pad = len(close) - _size
        sine_arr = np.concatenate([[np.nan] * _pad, result[0]])
        lead_arr = np.concatenate([[np.nan] * _pad, result[1]])
    else:
        sine_arr, lead_arr = _msw_native(np.array(close, dtype=float), period)

    sine = Series(sine_arr, index=close.index)
    lead = Series(lead_arr, index=close.index)

    # Offset
    sine, lead = apply_offset([sine, lead], offset)

    # Handle fills
    sine, lead = apply_fill([sine, lead], **kwargs)

    # Name and Categorize
    _params = f"_{period}"
    sine.name = f"MSW_SINE{_params}"
    lead.name = f"MSW_LEAD{_params}"
    sine.category = lead.category = "cycles"

    df = DataFrame({sine.name: sine, lead.name: lead}, index=close.index)
    df.name = f"MSW{_params}"
    df.category = "cycles"
    return df


def _msw_native(arr: np.ndarray, period: int):
    """Pure numpy Mesa Sine Wave — matches Tulip Indicators algorithm."""
    pi = np.pi
    tpi = 2.0 * pi
    size = len(arr)
    sine = np.full(size, np.nan)
    lead = np.full(size, np.nan)

    j_arr = np.arange(period, dtype=float)
    cos_arr = np.cos(tpi * j_arr / period)
    sin_arr = np.sin(tpi * j_arr / period)

    # The DFT window is read newest-first (j=0 is arr[i]), so the correlation
    # — which is oldest-first — runs against reversed coefficients. Element k
    # ends at bar k + period - 1; dropping element 0 starts at bar period,
    # matching the original loop's range(period, size).
    rp = np.correlate(arr, cos_arr[::-1], mode="valid")[1:]
    ip = np.correlate(arr, sin_arr[::-1], mode="valid")[1:]

    # NaN inputs make abs(rp) > 0.001 False, so they take the ±pi branch and
    # yield a finite phase — the same quirk as the original scalar loop.
    with np.errstate(divide="ignore", invalid="ignore"):
        phase = np.where(np.abs(rp) > 0.001, np.arctan(ip / rp), (tpi / 2.0) * np.where(ip < 0.0, -1.0, 1.0))

    phase = np.where(rp < 0.0, phase + pi, phase)
    phase += pi / 2.0
    phase = np.where(phase < 0.0, phase + tpi, phase)
    phase = np.where(phase > tpi, phase - tpi, phase)

    sine[period:] = np.sin(phase)
    lead[period:] = np.sin(phase + pi / 4.0)

    return sine, lead


msw.__doc__ = """
Mesa Sine Wave (MSW)

Introduced by John F. Ehlers in "Rocket Science For Traders" (2001).
Uses a DFT of the recent ``period`` bars to estimate phase and outputs
two oscillators that help identify cycle turning points.

Sources:
    Tulip Indicators: https://tulipindicators.org/msw
    Ehlers, John F. (2001) Rocket Science For Traders

Args:
    close (pd.Series): Close price series.
    period (int): Lookback period. Default: 5.
    offset (int): Number of periods to offset. Default: 0.

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: Columns MSW_SINE_{period}, MSW_LEAD_{period}.

Example:
    df[['MSW_SINE_5', 'MSW_LEAD_5']] = df.ta.msw()
"""
