# Moving Average with Variable Period (MAVP)
import warnings
from typing import Any, Optional

import numpy as np
from pandas import Series

from pandas_ta_classic import Imports
from pandas_ta_classic.utils import apply_fill, apply_offset, get_offset, verify_series

#: A period group is gathered only while both of these hold; otherwise its
#: bars are sliced one at a time, which is what the pre-vectorisation
#: implementation did.  So no shape of input is slower than it was before the
#: grouping was introduced.
#:
#: ``_MAVP_GATHER_MAX_PERIOD`` bounds the copying: a gather copies ``p`` values
#: per bar where a slice is a view, and that copy overtakes one Python
#: iteration at ``p ~= 1000``.  ``_MAVP_GATHER_MIN_BARS`` bounds the fixed
#: cost: each gather is one fancy-index plus one reduction, which only pays off
#: once a few bars share the period (a 200-bar frame with periods around 50 has
#: ~2 bars per period, and gathering them loses 2.3x).
_MAVP_GATHER_MAX_PERIOD = 1000
_MAVP_GATHER_MIN_BARS = 4


def _mavp_sma_values(close_arr, per_arr):
    """Compute a variable-period SMA for each bar.

    For each bar ``i`` the window is ``close_arr[i - p + 1 : i + 1]`` where
    ``p = per_arr[i]``.  Bars where fewer than ``p`` preceding values exist
    are left as ``NaN``.

    Bars are handled in groups that share a window size.  A group is either
    gathered -- all its windows indexed at once and reduced with one
    ``mean(axis=1)`` -- or sliced bar by bar, whichever is cheaper for its size
    and width (see the thresholds above).  Both take the mean of the identical
    ``close_arr[i - p + 1 : i + 1]`` slice, so they agree bit-for-bit and a
    single call can mix the two.

    Args:
        close_arr (np.ndarray): 1-D float64 price array.
        per_arr (np.ndarray): Integer array of per-bar window sizes,
            already clipped to ``[minperiod, maxperiod]``.

    Returns:
        np.ndarray: Rolling variable-SMA values.
    """
    n = len(close_arr)
    result = np.full(n, np.nan)
    rows = np.flatnonzero(np.arange(n) + 1 >= per_arr)
    if rows.size == 0:
        return result

    # Sorting puts the bars sharing a window size next to each other; the
    # windows are still the exact same slices, in the same order.
    row_periods = per_arr[rows]
    order = np.argsort(row_periods, kind="stable")
    rows = rows[order]
    periods, starts = np.unique(row_periods[order], return_index=True)
    bounds = np.append(starts, rows.size)

    for k, p in enumerate(periods.tolist()):  # Python ints: numpy scalars cost more to index with
        block = rows[bounds[k] : bounds[k + 1]]
        if p > _MAVP_GATHER_MAX_PERIOD or block.size < _MAVP_GATHER_MIN_BARS:
            # ``tolist()`` so the slice bounds are Python ints: indexing an
            # array with numpy scalars costs more per bar than the slice does.
            for i in block.tolist():
                result[i] = close_arr[i - p + 1 : i + 1].mean()
        else:
            result[block] = close_arr[block[:, None] - np.arange(p - 1, -1, -1)].mean(axis=1)
    return result


def mavp(
    close: Series,
    periods: Optional[Series] = None,
    minperiod: Optional[int] = None,
    maxperiod: Optional[int] = None,
    mamode: Optional[int] = None,
    talib: Optional[bool] = None,
    offset: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Series]:
    """Indicator: Moving Average with Variable Period (MAVP)"""
    # Validate Arguments
    minperiod = int(minperiod) if minperiod and minperiod >= 2 else 2
    maxperiod = int(maxperiod) if maxperiod and maxperiod > minperiod else 30
    # mamode: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3
    # For native fallback we only support SMA (0)
    mamode = int(mamode) if mamode is not None else 0
    close = verify_series(close, maxperiod)
    offset = get_offset(offset)
    mode_talib = bool(talib) if isinstance(talib, bool) else False

    if close is None:
        return None

    # Resolve variable-period series
    periods = (
        Series(
            np.linspace(minperiod, maxperiod, len(close)),
            index=close.index,
            dtype=float,
        )
        if periods is None
        else verify_series(periods)
    )
    if periods is None:
        return None

    # Calculate Result
    if Imports["talib"] and mode_talib:
        from talib import MAVP as TAMAVP

        mavp_ = TAMAVP(
            close,
            periods.astype(float),
            minperiod=minperiod,
            maxperiod=maxperiod,
            matype=mamode,
        )
    else:
        # Native: simple moving average with per-bar variable window
        # Only SMA (mamode=0) is supported natively; other MA types require TA-Lib
        if mamode != 0:
            warnings.warn(
                f"MAVP native fallback only supports SMA (mamode=0); " f"mamode={mamode} requires TA-Lib. Results will use SMA.",
                UserWarning,
                stacklevel=2,
            )
        close_arr = close.to_numpy(dtype=float)
        per_arr = np.clip(periods.to_numpy(dtype=float).round().astype(int), minperiod, maxperiod)
        mavp_ = Series(_mavp_sma_values(close_arr, per_arr), index=close.index)

    # Offset
    mavp_ = apply_offset(mavp_, offset)

    mavp_ = apply_fill(mavp_, **kwargs)

    # Name and Categorize it
    mavp_.name = f"MAVP_{minperiod}_{maxperiod}"
    mavp_.category = "overlap"

    return mavp_


mavp.__doc__ = """Moving Average with Variable Period (MAVP)

Moving Average with Variable Period computes a moving average where the
lookback period is different for each bar. The periods Series determines
the window size at each data point. When periods is None, a linearly
interpolated range from minperiod to maxperiod is used.

Sources:
    https://mrjbq7.github.io/ta-lib/func_groups/overlap_studies.html

Args:
    close (pd.Series): Close price series.
    periods (pd.Series): Variable period series (optional). Default: linearly spaced [minperiod, maxperiod]
    minperiod (int): Minimum allowed period. Default: 2
    maxperiod (int): Maximum allowed period. Default: 30
    mamode (int): MA type (TA-Lib convention). Default: 0 (SMA).
        Only mamode=0 (SMA) is supported natively. All other values
        require TA-Lib and emit a UserWarning if talib=False.
    talib (bool): Use TA-Lib if installed. Default: False
    offset (int): Result offset. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: MAVP values.
"""
