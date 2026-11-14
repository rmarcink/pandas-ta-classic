"""talib=True must not silently ignore a parameter TA-Lib cannot express.

wma(asc=False, talib=True) used to return TA-Lib's ascending WMA; a sweep then
found 37 more parameters in 23 indicators (scalar, drift, mamode, ddof, uo's
weights) that changed the native result but were dropped on the TA-Lib path.
Those indicators now run natively whenever such a parameter is not at its
default, so talib=True and talib=False agree on it.
"""

import inspect
import re
import warnings

import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as ta
from pandas_ta_classic._indicator_loader import _find_indicator_func
from pandas_ta_classic.overlap.ma import _MA_DISPATCH
from pandas_ta_classic.utils._core import _TAL_MA_TYPES
from pandas_ta_classic.utils._keywords import _MA_OPTIONS
from tests.config import get_sample_data
from tests.contracts import INDICATORS, inputs, spy as _spy
from tests.meta.test_docstring_params import _accepted

pytestmark = pytest.mark.skipif(not ta.Imports["talib"], reason="needs TA-Lib")

_SERIES = {"open_": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
_SKIP = {"offset", "talib", "fillna", "fill_method"}
# Tried in order; the first value that changes the native result is used.
_CANDIDATES = [2, 3, 7, 21, 0.5, 1.5, 50.0, "ema", "wma", "sma", True, False]


def _talib_indicators():
    return [n for n in INDICATORS if "talib" in inspect.signature(_find_indicator_func(n)).parameters]


def _same(a, b):
    """Compare values only: `DataFrame.equals` also compares column labels, and a
    parameter that is part of the name (`ATRe_14`, `CCI_14_0.5`) made every pair
    differ, so a talib=True path that ignored the value still passed."""
    a, b = pd.DataFrame(a).to_numpy(dtype=float), pd.DataFrame(b).to_numpy(dtype=float)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def _options(func):
    """The options of *func*: its keyword parameters, a price column it takes as an
    optional input (`ad(open_=...)` switches to the Open form), and the options its
    module reads from ``**kwargs`` (`sma(min_periods=...)`), which the signature hides,
    including those of the functions it hands its ``**kwargs`` to (`tema` -> `ema(adjust=...)`)."""
    params = inspect.signature(func).parameters
    found = {}
    for param, spec in params.items():
        if param in _SKIP or spec.kind in (spec.VAR_KEYWORD, spec.VAR_POSITIONAL):
            continue
        if param in _SERIES:
            if spec.default is None:
                found[param] = [_SERIES[param]]
            continue
        found[param] = _CANDIDATES
    source = inspect.getsource(inspect.getmodule(inspect.unwrap(func)))
    for param in re.findall(r'kwargs\.(?:get|pop|setdefault)\("(\w+)"', source):
        if param not in _SKIP and param not in found:
            found[param] = _CANDIDATES
    for param in sorted(_accepted(func) - set(params) - _SKIP - set(found)):
        found[param] = _CANDIDATES
    return found


@pytest.mark.parametrize("name", _talib_indicators())
def test_talib_true_honours_every_parameter(name):
    frame = get_sample_data().iloc[:600]
    func = _find_indicator_func(name)
    params = inspect.signature(func).parameters
    series = {p: frame[c] for p, c in _SERIES.items() if p in params and params[p].default is not None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            native_default = func(**series, talib=False)
            talib_default = func(**series, talib=True)
        except Exception:  # noqa: BLE001 - indicators that need extra inputs are out of scope
            pytest.skip("needs inputs beyond price columns")
        ignored = []
        for param, candidates in _options(func).items():
            for label in candidates:
                value = frame[label] if param in _SERIES else label
                try:
                    native = func(**series, talib=False, **{param: value})
                except (ValueError, TypeError):
                    continue
                if _same(native, native_default):
                    continue
                talib = func(**series, talib=True, **{param: value})
                if _same(talib, talib_default):
                    ignored.append(f"{param}={label!r}")
                break
    assert not ignored, f"{name}(talib=True) ignores {ignored}"


def test_non_default_parameter_runs_natively():
    close = get_sample_data().close.iloc[:300]
    assert ta.rsi(close, scalar=50, talib=True).equals(ta.rsi(close, scalar=50, talib=False))
    assert not ta.rsi(close, talib=True).equals(ta.rsi(close, talib=False))  # defaults still use TA-Lib


def test_macdfix_keeps_its_signal_columns_on_the_talib_path():
    """The sweep above reads the signature, so a `**kwargs` option is invisible to it.

    `macdfix(talib=True)` builds its own frame from TA-Lib's three lines instead
    of delegating to `macd()`, so `signal_indicators=True` used to come back with
    the three columns and no signals at all -- silently. TA-Lib has no signal
    columns, so that call runs natively, like every other parameter TA-Lib cannot
    express.
    """
    close = get_sample_data().close.iloc[:300]
    expected = [
        "MACDFIX_9_9",
        "MACDFIXh_9_9",
        "MACDFIXs_9_9",
        "MACDFIXh_9_9_XA_0",
        "MACDFIXh_9_9_XB_0",
        "MACDFIX_9_9_A_0",
    ]

    native = ta.macdfix(close, signal_indicators=True, talib=False)
    talib_path = ta.macdfix(close, signal_indicators=True, talib=True)

    assert list(native.columns) == expected
    assert list(talib_path.columns) == expected
    assert talib_path.equals(native), "signal_indicators=True must give the same result on both paths"
    # Without the keyword, talib=True still uses TA-Lib -- which seeds its EMAs
    # differently, so the two paths must not agree there.
    assert not ta.macdfix(close, talib=True).equals(ta.macdfix(close, talib=False))


# Columns whose first value TA-Lib publishes on another bar than the native path, by
# design of TA-Lib: the reason is part of the table so a change on either side shows up here.
_ANOTHER_BAR_ON_TALIB = {
    "asi": ({"ASI"}, "TA-Lib's ASI reads 0 on bar 0, where native has no previous close and reads NaN"),
    "macd": ({"MACD_12_26_9"}, "TA-Lib holds the MACD line back until its signal line exists"),
    "macdext": ({"MACDEXT_12_26_9"}, "TA-Lib holds the MACD line back until its signal line exists"),
    "macdfix": ({"MACDFIX_9_9"}, "TA-Lib holds the MACD line back until its signal line exists"),
    "mavp": ({"MAVP_2_30"}, "TA-Lib waits for maxperiod on every bar, whatever that bar's period"),
    "stoch": ({"STOCHk_14_3_3"}, "TA-Lib holds %K back until %D exists"),
    "stochf": ({"STOCHFk_5_3"}, "TA-Lib holds %K back until %D exists"),
    "stoch_mi": ({"STOCHMI_13_2_25_9"}, "TA-Lib holds the SMI line back until its signal line exists"),
    "wavetrend": ({"WT_10_21_4"}, "TA-Lib holds WT back until its signal line exists"),
    "swak_2php": ({"SWAK2PHP_20"}, "TA-Lib publishes the filter's seed, 0.0, on bar 0; native reads NaN there"),
    "swak_bp": ({"SWAKBP_20_0.1"}, "TA-Lib publishes the filter's seed, 0.0, on bar 0; native reads NaN there"),
    "swak_hp": ({"SWAKHP_20"}, "TA-Lib publishes the filter's seed, 0.0, on bar 0; native reads NaN there"),
}


@pytest.mark.parametrize("name", _talib_indicators())
def test_native_and_talib_publish_from_the_same_bar(name):
    """Native kama published its seed (the close of bar length-1) one bar before
    TA-Lib's KAMA, which starts at bar length."""
    frame = get_sample_data().iloc[:600]
    func = _find_indicator_func(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        native = pd.DataFrame(func(**inputs(name, frame), talib=False))
        talib = pd.DataFrame(func(**inputs(name, frame), talib=True))
    assert list(native.columns) == list(talib.columns)
    differ = {column for column in native if native[column].first_valid_index() != talib[column].first_valid_index()}
    assert differ == _ANOTHER_BAR_ON_TALIB.get(name, (set(), ""))[0], f"{name}: first value on another bar in {sorted(differ)}"


def test_accessor_options_keep_the_talib_path():
    """append/col_names/prefix come from the accessor, not for the moving average:
    bbands counted them as MA options and left TA-Lib for every strategy() run."""
    df = get_sample_data().iloc[:300].copy()
    expected = ta.bbands(df.close, talib=True)
    appended = df.ta.bbands(talib=True, append=True, prefix="X")
    assert np.array_equal(appended.to_numpy(), expected.to_numpy(), equal_nan=True)
    assert not np.array_equal(expected.to_numpy(), ta.bbands(df.close, talib=False).to_numpy(), equal_nan=True)


def test_bbands_knows_every_ma_option():
    """bbands leaves TA-Lib for these options only; one it does not list is ignored on that path."""
    internal = {"close", "source", "series", "depth", "length", "talib", "offset", "fillna", "fill_method"}
    read = {name: _accepted(func) - internal for name, func in _MA_DISPATCH.items()}
    assert {name: options for name, options in read.items() if options} == _MA_OPTIONS


def test_an_option_the_ma_does_not_read_raises():
    """Only vidya reads drift: bbands(talib=True, drift=2) left TA-Lib although its
    SMA never reads drift, then ignored it. strategy(drift=N) now hands drift only
    to the indicators that accept it, so the direct call can raise."""
    close = get_sample_data().close.iloc[:300]
    with pytest.raises(TypeError, match=r"^bbands\(\) got an unexpected keyword argument 'drift'$"):
        ta.bbands(close, talib=True, drift=2)
    assert ta.bbands(close, mamode="vidya", drift=2) is not None


# Indicators that hand mamode to TA-Lib's MA_Type enum (utils.tal_ma).
_TAL_MA_INDICATORS = ("apo", "bbands", "ppo")


@pytest.mark.parametrize("talib", [False, True])
@pytest.mark.parametrize("mamode", sorted(set(_TAL_MA_TYPES) - set(_MA_DISPATCH)))
@pytest.mark.parametrize("name", _TAL_MA_INDICATORS)
def test_a_mamode_ma_lacks_raises_on_both_paths(name, mamode, talib):
    """talib=True handed kama and mama to TA-Lib, while the native path refused
    them in ma(); both paths accept the same values (AGENTS.md rule 6)."""
    close = get_sample_data().close.iloc[:300]
    with pytest.raises(ValueError, match=rf"^{name}\(\) mamode must be one of .*got '{mamode}'$"):
        getattr(ta, name)(close, mamode=mamode, talib=talib)


@pytest.mark.parametrize("mamode", sorted(set(_MA_DISPATCH) - set(_TAL_MA_TYPES)))
@pytest.mark.parametrize("name", _TAL_MA_INDICATORS)
def test_a_mamode_talib_lacks_runs_natively(name, mamode):
    """talib=True with hma, rma, zlma, ... raised from tal_ma(); TA-Lib cannot
    express that MA, so the call computes natively, as for any other parameter."""
    close = get_sample_data().close.iloc[:300]
    func = getattr(ta, name)
    assert _same(func(close, mamode=mamode, talib=True), func(close, mamode=mamode, talib=False))


# ---------------------------------------------------------------------------
# stochrsi TA-Lib path honours k and rsi_length: talib=True uses TA-Lib only
# when it can express them, so it equals the native result.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [{}, {"rsi_length": 7}, {"k": 1}, {"k": 1, "rsi_length": 7}], ids=["defaults", "rsi_length", "k1", "k1-rsi_length"])
def test_stochrsi_talib_path_matches_native(kwargs):
    close = _spy().close
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fast, native = ta.stochrsi(close, talib=True, **kwargs), ta.stochrsi(close, talib=False, **kwargs)
    for col in range(2):
        a, b = fast.iloc[:, col].to_numpy(float), native.iloc[:, col].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        assert both.sum() > 5000
        np.testing.assert_allclose(a[both], b[both], rtol=0, atol=1e-9)


# talib=True at these defaults computes natively, by rule 6: TA-Lib cannot express a default.
_NATIVE_AT_DEFAULTS = {"stochrsi": "TA-Lib's STOCHRSI has no %K smoothing; it runs only at k=1, and the default is k=3"}


@pytest.mark.parametrize("name", [n for n in _talib_indicators() if n not in _NATIVE_AT_DEFAULTS])
def test_talib_true_calls_talib(name, monkeypatch):
    """With TA-Lib installed, talib=True at the defaults runs a TA-Lib function, not the native formula."""
    import talib

    called = []

    def _recording(fname, real):
        def wrapper(*args, **kwargs):
            called.append(fname)
            return real(*args, **kwargs)

        return wrapper

    for fname in talib.get_functions():
        monkeypatch.setattr(talib, fname, _recording(fname, getattr(talib, fname)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        func = _find_indicator_func(name)
        params = inspect.signature(func).parameters
        # An optional open_ selects ad's and adosc's Open form, which TA-Lib lacks.
        series = {k: v for k, v in inputs(name, _spy(300)).items() if params[k].default is inspect.Parameter.empty}
        func(**series, talib=True)
    assert called, f"{name}(talib=True) never called TA-Lib"
