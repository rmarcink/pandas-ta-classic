# Relative Volume (RVOL)
from typing import Any

from pandas import Series

from pandas_ta_classic.overlap.sma import sma
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input


@nan_on_short_input
def rvol(
    volume: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Relative Volume (RVOL)"""
    # Validate Arguments
    length = _pos_int(length, 20, "length")
    volume = verify_series(volume, length + 1)
    offset = get_offset(offset)

    if volume is None:
        return None

    # Calculate Result
    average = sma(volume.shift(1), length)
    if average is None:
        return None
    # A plain division, as TA-Lib's RVOL: no volume over no volume is NaN,
    # volume after a window without any is inf.
    rvol = volume / average

    # Offset
    rvol = apply_offset(rvol, offset)

    rvol = apply_fill(rvol, **kwargs)

    # Name and Categorize it
    rvol.name = f"RVOL_{length}"
    rvol.category = "volume"

    return rvol


rvol.__doc__ = """Relative Volume (RVOL)

The bar's volume as a multiple of the average volume of the bars before it.
The bar itself is not part of its average.

Sources:
    https://ta-lib.org/functions/  (RVOL)

Calculation:
    Default Inputs:
        length=20
    RVOL = volume / SMA(volume[1], length)

Args:
    volume (pd.Series): Series of 'volume's
    length (int): Number of earlier bars averaged. Default: 20
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
