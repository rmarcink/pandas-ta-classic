# Intraday Momentum Index (IMI)
from typing import Any

from pandas import Series

from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, nan_on_short_input


@nan_on_short_input
def imi(
    open_: Series,
    close: Series,
    length: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Intraday Momentum Index (IMI)"""
    # Validate Arguments
    length = _pos_int(length, 14, "length")
    open_ = verify_series(open_, length)
    close = verify_series(close, length)
    offset = get_offset(offset)

    if open_ is None or close is None:
        return None

    # Calculate Result
    body = close - open_
    gains = body.clip(lower=0).rolling(length).sum()
    losses = (-body).clip(lower=0).rolling(length).sum()
    total = gains + losses
    # A window of bodiless bars has no momentum either way: TA-Lib's IMI reads 50.
    imi = (100 * gains / total.where(total != 0)).mask(total == 0, 50.0)

    # Offset
    imi = apply_offset(imi, offset)

    imi = apply_fill(imi, **kwargs)

    # Name and Categorize it
    imi.name = f"IMI_{length}"
    imi.category = "momentum"

    return imi


imi.__doc__ = """Intraday Momentum Index (IMI)

Tushar Chande's RSI of the bodies: the share of the window's white bodies
in all its bodies. A window of bodiless bars reads 50, as TA-Lib's IMI does.

Sources:
    https://www.investopedia.com/terms/i/intraday-momentum-index-imi.asp

Calculation:
    Default Inputs:
        length=14
    GAIN = SUM(max(close - open, 0), length)
    LOSS = SUM(max(open - close, 0), length)
    IMI = 100 * GAIN / (GAIN + LOSS)

Args:
    open_ (pd.Series): Series of 'open's
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 14
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
