# Accelerator Oscillator (AC)
from typing import Any

from pandas import Series

from pandas_ta_classic.momentum.ao import ao
from pandas_ta_classic.overlap.sma import sma
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series
from pandas_ta_classic.utils._core import _pos_int, _windows_in_order, nan_on_short_input


@nan_on_short_input
def ac(
    high: Series,
    low: Series,
    fast: int | None = None,
    slow: int | None = None,
    signal: int | None = None,
    offset: int | None = None,
    **kwargs: Any,
) -> Series | None:
    """Indicator: Accelerator Oscillator (AC)"""
    # Validate Arguments
    fast = _pos_int(fast, 5, "fast")
    slow = _pos_int(slow, 34, "slow")
    _windows_in_order("ac", fast=fast, slow=slow)
    signal = _pos_int(signal, 5, "signal")
    _length = slow + signal - 1
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    offset = get_offset(offset)

    if high is None or low is None:
        return None

    # Calculate Result
    ao_ = ao(high, low, fast, slow)
    if ao_ is None:
        return None
    ao_signal = sma(ao_, signal)
    if ao_signal is None:
        return None
    ac = ao_ - ao_signal

    # Offset
    ac = apply_offset(ac, offset)

    ac = apply_fill(ac, **kwargs)

    # Name and Categorize it
    ac.name = f"AC_{fast}_{slow}_{signal}"
    ac.category = "momentum"

    return ac


ac.__doc__ = """Accelerator Oscillator (AC)

Bill Williams' Accelerator/Decelerator Oscillator: the Awesome Oscillator less
its own simple moving average, so it turns before AO does.

Sources:
    https://www.tradingview.com/support/solutions/43000501837-accelerator-oscillator-ac/

Calculation:
    Default Inputs:
        fast=5, slow=34, signal=5
    AO = SMA(hl2, fast) - SMA(hl2, slow)
    AC = AO - SMA(AO, signal)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    fast (int): The short period of AO. Default: 5
    slow (int): The long period of AO. Default: 34
    signal (int): The period of AO's moving average. Default: 5
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
