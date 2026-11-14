# Wilder Accumulative Swing Index (ASI)
from typing import Any

from pandas import Series

from pandas_ta_classic.momentum.si import _swing_index
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _name_tokens, _pos_float, nan_on_short_input


@nan_on_short_input
def asi(
    open_: Series,
    high: Series,
    low: Series,
    close: Series,
    limit: float | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Wilder Accumulative Swing Index (ASI)"""
    # Validate Arguments
    limit = _pos_float(limit, 3.0, "limit")
    open_ = verify_series(open_, 2)
    high = verify_series(high, 2)
    low = verify_series(low, 2)
    close = verify_series(close, 2)
    offset = get_offset(offset)

    if open_ is None or high is None or low is None or close is None:
        return None

    # Calculate Result
    swing = _swing_index(open_, high, low, close, limit)
    # A bar without a swing (the first, or one next to a missing bar) adds
    # nothing and reads NaN; the sum carries on after it.
    asi = swing.cumsum(skipna=True).mask(swing.isna())

    # Offset
    asi = apply_offset(asi, offset)

    asi = apply_fill(asi, **kwargs)

    # Name and Categorize it
    asi.name = "ASI" + _name_tokens(("limit", limit, 3.0))
    asi.category = "momentum"

    return asi


asi.__doc__ = """Wilder Accumulative Swing Index (ASI)

The running sum of Wilder's swing index (see ``si``).

Sources:
    Wilder, J. W. (1978), "New Concepts in Technical Trading Systems".

Calculation:
    Default Inputs:
        limit=3.0
    ASI = cumulative sum of SI(open, high, low, close, limit)

Args:
    open_ (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    limit (float): The limit move of the market. Default: 3.0
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
