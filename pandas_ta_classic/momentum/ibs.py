# Internal Bar Strength (IBS)
from typing import Any

from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import nan_on_short_input


@nan_on_short_input
def ibs(
    high: Series,
    low: Series,
    close: Series,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Internal Bar Strength (IBS)"""
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    offset = get_offset(offset)

    if high is None or low is None or close is None:
        return None

    # Calculate Result
    bar_range = high - low
    # A bar with no range has no position in it: TA-Lib's IBS reads the middle, 0.5.
    ibs = ((close - low) / bar_range.where(bar_range != 0)).mask(bar_range == 0, 0.5)

    # Offset
    ibs = apply_offset(ibs, offset)

    ibs = apply_fill(ibs, **kwargs)

    # Name and Categorize it
    ibs.name = "IBS"
    ibs.category = "momentum"

    return ibs


ibs.__doc__ = """Internal Bar Strength (IBS)

Where the close sits within the bar's range: 0 at the low, 1 at the high.
A bar with no range reads 0.5, as TA-Lib's IBS does.

Sources:
    https://ta-lib.org/functions/  (IBS)

Calculation:
    IBS = (close - low) / (high - low)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
