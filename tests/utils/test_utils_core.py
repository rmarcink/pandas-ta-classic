"""Tests of pandas_ta_classic/utils/_core.py: series checks, offset and fill, parameter validation."""

import numpy as np
import pytest
from pandas import Series

import pandas_ta_classic as ta
from pandas_ta_classic.utils._core import _bool_param, _number, _pos_float, _pos_int, _str_param


@pytest.mark.parametrize("bad", [0, -5, 2.7, True, np.nan, float("inf"), "14"])
def test_pos_int_rejects(bad):
    def sma(length=None):
        return _pos_int(length, 10, "length")

    with pytest.raises(ValueError, match=r"sma\(\) length must be an integer > 0"):
        sma(bad)


def test_pos_helpers_accept_none_and_valid_values():
    assert _pos_int(None, 10, "length") == 10
    assert _pos_int(14, 10, "length") == 14
    assert _pos_int(14.0, 10, "length") == 14
    assert _pos_int(np.int64(5), 10, "length") == 5
    assert _pos_float(0.5, 0.2, "na", lt=1) == 0.5
    assert _pos_int(0, 1, "ddof", gt=None, ge=0, lt=30) == 0


def test_a_whole_number_keeps_the_type_of_an_int_default():
    assert type(_pos_float(2.0, 2, "scalar")) is int
    assert type(_pos_float(2, 2.0, "std")) is float
    assert type(_pos_float(2.5, 2, "scalar")) is float
    assert type(_number(100.0, 100, "scalar")) is int


def test_number_accepts_zero_and_negatives_but_not_junk():
    assert _number(None, 100, "scalar") == 100
    assert _number(0, 100, "scalar") == 0.0  # used to become 100
    assert _number(-2, 100, "scalar") == -2.0
    for bad in (np.nan, float("inf"), True, "100"):
        with pytest.raises(ValueError, match="scalar must be a number"):
            _number(bad, 100, "scalar")


def test_bool_param_rejects_non_bools():
    def ema(talib=None):
        return _bool_param(talib, False, "talib")

    assert ema() is False
    assert ema(True) is True
    assert ema(np.bool_(True)) is True
    for bad in (1, 0, "True", []):  # talib=1 used to mean False
        with pytest.raises(ValueError, match=r"ema\(\) talib must be True or False"):
            ema(bad)


def test_str_param_rejects_non_strings_and_unknown_choices():
    def cpr(method=None):
        return _str_param(method, "classic", "method", choices={"classic", "woodie"})

    assert cpr() == "classic"
    assert cpr("WOODIE") == "woodie"
    for bad in (1, "", "fibonaci"):
        with pytest.raises(ValueError, match=r"cpr\(\) method must be"):
            cpr(bad)


def test_bad_fill_method_leaves_the_input_unfilled():
    """fill_method is validated before fillna runs: a rejected call must not have
    filled the caller's Series in place on its way to the error (AGENTS rule 19)."""
    series = Series([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="fill_method"):
        ta.utils.apply_fill(series, fillna=0, fill_method="bogus")
    assert series.isna().tolist() == [False, True, False]
