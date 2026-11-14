# Average Day Range (ADR)
from typing import Any

from pandas import Series

from pandas_ta_classic.overlap.sma import sma
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input


@nan_on_short_input
def adr(
    high: Series,
    low: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Average Day Range (ADR)"""
    # Validate Arguments
    length = _pos_int(length, 14, "length")
    high = verify_series(high, length)
    low = verify_series(low, length)
    offset = get_offset(offset)

    if high is None or low is None:
        return None

    # Calculate Result
    adr = sma(high - low, length)
    if adr is None:
        return None

    # Offset
    adr = apply_offset(adr, offset)

    adr = apply_fill(adr, **kwargs)

    # Name and Categorize it
    adr.name = f"ADR_{length}"
    adr.category = "volatility"

    return adr


adr.__doc__ = """Average Day Range (ADR)

The average of the bars' high-low ranges. Unlike the average true range it
ignores gaps from the previous close.

Sources:
    https://ta-lib.org/functions/  (ADR)

Calculation:
    Default Inputs:
        length=14
    ADR = SMA(high - low, length)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    length (int): It's period. Default: 14
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
