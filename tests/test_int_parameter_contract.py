"""A small whole-number parameter is rejected or gives clean values on long input.

`kama(drift=2)` published only its seed bar on 1000 bars: its efficiency
ratio needs `length` price changes `drift` bars apart, the first of which
does not exist until bar `drift`, and the recursion started at bar
`length` from a NaN ratio, so the NaN rode it to the end. `linreg(length=1)`
fit a line through one point: 0/0 on every bar, a RuntimeWarning and an
empty column. Every int parameter of every registered indicator is set to
0, 1, 2, 3 and 5. A value the indicator accepts may not come back with a
column the default call fills at most one value, with an infinity the
default call does not have, or with a RuntimeWarning from the package; a
value its formula cannot use is rejected by its validation, naming the
parameter.
"""

import inspect
import typing
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta
from tests.contracts import INDICATORS, NOT_OHLCV, inputs, params, spy

VALUES = (0, 1, 2, 3, 5)
PACKAGE = str(Path(ta.__file__).parent)
CASES = [
    (name, param)
    for name in INDICATORS
    if name not in NOT_OHLCV
    for param, spec in inspect.signature(getattr(ta, name)).parameters.items()
    if param != "offset" and (spec.annotation is int or int in typing.get_args(spec.annotation))
]
_LINE_THROUGH_ONE_POINT = "OPEN: length=1 fits a line through one point, 0/0 in linreg on every bar; the formula needs length > 1"
_NAMES_THE_CALLEE = "OPEN: {param}={value} raises, but the error names {callee}(), not {name}() and {param}"
KNOWN = {
    ("vidya", "drift"): "OPEN: the CMO starts at bar length + drift - 1, the SMA seed sits at bar length - 1; where to seed is undecided",
    **{
        (name, "length"): _LINE_THROUGH_ONE_POINT
        for name in ("cfo", "cti", "fosc", "inertia", "linreg", "linregangle", "linregintercept", "linregslope", "po", "tsf")
    },
    ("inertia", "rvi_length"): _NAMES_THE_CALLEE.format(param="rvi_length", value=1, callee="rvi", name="inertia"),
    ("squeeze", "bb_length"): _NAMES_THE_CALLEE.format(param="bb_length", value=1, callee="bbands", name="squeeze"),
    ("squeeze_pro", "bb_length"): _NAMES_THE_CALLEE.format(param="bb_length", value=1, callee="bbands", name="squeeze_pro"),
    ("trixh", "drift"): _NAMES_THE_CALLEE.format(param="drift", value=0, callee="trix", name="trixh"),
    ("stderr", "length"): "OPEN: length=1 divides by length - 1 = 0, an empty column without a warning; the formula needs length > 1",
    ("hvol", "length"): "OPEN: length=1 takes the standard deviation of one return, an empty column without a warning",
    ("chop", "length"): "OPEN: length=1 divides by log10(1) = 0, +inf on every bar without a warning",
    ("brar", "length"): "OPEN: x/0 when every bar of a short window sits on one side of the previous close: +inf at length 1, 2 and 3",
    ("vp", "width"): "KEPT: width=1 is one price bin, a one-row result, not an empty column",
}


@pytest.fixture(scope="module")
def frame():
    return spy(1000)


def _frame(result) -> pd.DataFrame:
    return result.to_frame() if isinstance(result, pd.Series) else result


def _infinite(column: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(column) and bool(np.isinf(column.to_numpy(float, na_value=np.nan)).any())


def test_sweep_found_the_parameters():
    assert ("kama", "drift") in CASES and ("linreg", "length") in CASES and len(CASES) > 150


@pytest.mark.parametrize(("name", "param"), params(CASES, KNOWN, id=lambda row: "-".join(row)))
def test_a_small_int_parameter_is_rejected_or_clean(name, param, frame):
    func = getattr(ta, name)
    default = _frame(func(**inputs(name, frame)))
    found = []
    for value in VALUES:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            try:
                result = _frame(func(**inputs(name, frame), **{param: value}))
            except ValueError as error:  # outside the parameter's range (ddof >= length, poles not 2 or 3, ...)
                if f"{name}()" not in str(error) or param not in str(error):
                    found.append(f"{param}={value}: rejected, but the error does not name {name}() and {param}: {error}")
                continue
        where = f"{param}={value}"
        found += sorted({f"{where}: RuntimeWarning at {Path(w.filename).name}:{w.lineno}" for w in caught if issubclass(w.category, RuntimeWarning) and w.filename.startswith(PACKAGE)})
        # The parameter is often in the column name (LR_1 for LR_14): pair the columns by position.
        pairs = zip(default.columns, result.columns) if default.shape[1] == result.shape[1] else ((c, c) for c in result.columns if c in default)
        for d, c in pairs:
            if default[d].notna().any() and result[c].notna().sum() <= 1:
                found.append(f"{where}: {c} blank")
            if _infinite(result[c]) and not _infinite(default[d]):
                found.append(f"{where}: {c} infinite")
    assert not found, f"{name}: {found}"
