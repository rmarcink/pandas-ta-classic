# Rogers-Satchell Volatility (ROGERSSATCHELL)
from typing import Any

import numpy as np
from pandas import Series

from pandas_ta_classic.overlap.sma import sma
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _name_tokens, _pos_float, _pos_int, _positive_inputs, nan_on_short_input


@nan_on_short_input
@_positive_inputs("open_", "high", "low", "close")
def rogerssatchell(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    length: int | None = None,
    annualization: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Rogers-Satchell Volatility (ROGERSSATCHELL)"""
    # Validate Arguments
    length = _pos_int(length, 10, "length")
    annualization = _pos_float(annualization, 252, "annualization")
    open_ = verify_series(open_, length)
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    offset = get_offset(offset)

    if open_ is None or high is None or low is None or close is None:
        return None

    # Calculate Result
    log_hc, log_ho = np.log(high / close), np.log(high / open_)
    log_lc, log_lo = np.log(low / close), np.log(low / open_)
    variance = sma(log_hc * log_ho + log_lc * log_lo, length)
    if variance is None:
        return None
    rs = np.sqrt(variance * annualization)

    # Offset
    rs = apply_offset(rs, offset)

    rs = apply_fill(rs, **kwargs)

    # Name and Categorize it
    rs.name = f"RS_{length}" + _name_tokens(("annualization", annualization, 252))
    rs.category = "volatility"

    return rs


rogerssatchell.__doc__ = """Rogers-Satchell Volatility (ROGERSSATCHELL)

An estimator of volatility from each bar's open, high, low and close that
stays unbiased when the price drifts. Annualized, as a fraction (0.2 is 20 %),
as TA-Lib's ROGERSSATCHELL returns it.

Sources:
    Rogers, L.C.G., Satchell, S.E. (1991), "Estimating variance from high,
    low and closing prices", Annals of Applied Probability 1(4).

Calculation:
    Default Inputs:
        length=10, annualization=252
    RS_t = ln(high / close) * ln(high / open) + ln(low / close) * ln(low / open)
    ROGERSSATCHELL = sqrt(SMA(RS, length) * annualization)

Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    annualization (float): Periods per year. Default: 252
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
