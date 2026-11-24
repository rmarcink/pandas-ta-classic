# Moving Average (MA)
from typing import Any

from pandas import Series

from pandas_ta_classic.utils._core import Bound, _validate_number, declared_params, nan_on_short_input

from .dema import dema
from .ema import ema
from .fwma import fwma
from .hma import hma
from .linreg import linreg
from .midpoint import midpoint
from .pwma import pwma
from .rma import rma
from .sinwma import sinwma
from .sma import sma
from .swma import swma
from .t3 import t3
from .tema import tema
from .trima import trima
from .vidya import vidya
from .wma import wma
from .zlma import zlma

# Dispatch table mapping lower-case MA name → function.
# Keeping this at module level means it is built only once and makes
# adding a new MA type a one-line change.
_MA_DISPATCH = {
    "dema": dema,
    "ema": ema,
    "fwma": fwma,
    "hma": hma,
    "linreg": linreg,
    "midpoint": midpoint,
    "pwma": pwma,
    "rma": rma,
    "sinwma": sinwma,
    "sma": sma,
    "swma": swma,
    "t3": t3,
    "tema": tema,
    "trima": trima,
    "vidya": vidya,
    "wma": wma,
    "zlma": zlma,
}


def ma_window(mamode: str, value: int, name: str) -> int:
    """Check *value*, a window an indicator hands to ``ma(mamode, length=...)``, against
    the bound the chosen MA declares for its ``length``, so the error names the caller's
    parameter (``stoch() d must be an integer > 1 with mamode='hma', got 1``)."""
    declared = declared_params(_MA_DISPATCH[mamode]).get("length")
    if declared is None or not isinstance(declared.marker, Bound):
        return value
    bound = declared.marker
    return _validate_number(value, value, name, integer=True, gt=bound.gt, ge=bound.ge, lt=bound.lt, note=f" with mamode={mamode!r}")


@nan_on_short_input
def ma(name: str | None = None, source: Series | None = None, **kwargs: Any) -> Series | None:
    """Simple MA Utility for easier MA selection

    Available MAs:
        dema, ema, fwma, hma, linreg, midpoint, pwma, rma,
        sinwma, sma, swma, t3, tema, trima, vidya, wma, zlma

    Examples:
        ema8 = ta.ma("ema", df.close, length=8)
        sma50 = ta.ma("sma", df.close, length=50)
        wma10 = ta.ma("wma", df.close, length=10, asc=False)

    Args:
        name (str): One of the Available MAs. Default: "ema"
        source (pd.Series): The 'source' Series.

    Kwargs:
        Any additional kwargs the MA may require.

    Returns:
        pd.Series: New feature generated.
    """
    if name is None and source is None:
        return list(_MA_DISPATCH.keys())

    if name is None:
        return ema(source, **kwargs)
    if not isinstance(name, str) or name.lower() not in _MA_DISPATCH:
        raise ValueError(f"ma() name must be one of {sorted(_MA_DISPATCH)}, got {name!r}")
    return _MA_DISPATCH[name.lower()](source, **kwargs)
