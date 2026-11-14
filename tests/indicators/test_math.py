from unittest import TestCase

import numpy as np
from pandas import DataFrame, Series

import pandas_ta_classic as pandas_ta
from tests.assertions import (
    IndicatorSpec,
    assert_indicator_standard,
)
from tests.config import get_sample_data


class TestMath(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()
        cls.data.columns = cls.data.columns.str.lower()
        cls.open = cls.data["open"]
        cls.high = cls.data["high"]
        cls.low = cls.data["low"]
        cls.close = cls.data["close"]
        cls.volume = cls.data["volume"]

    @classmethod
    def tearDownClass(cls):
        del cls.open, cls.high, cls.low, cls.close, cls.volume, cls.data

    # -----------------------------------------------------------------------
    # Two-series operators
    # -----------------------------------------------------------------------

    def test_add(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.add,
                args=[self.close, self.open],
                expected_name="ADD",
            ),
        )

    def test_sub(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.sub,
                args=[self.close, self.open],
                expected_name="SUB",
            ),
        )

    def test_div(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.div,
                args=[self.close, self.open],
                expected_name="DIV",
            ),
        )

    def test_mult(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.mult,
                args=[self.close, self.open],
                expected_name="MULT",
            ),
        )

    # -----------------------------------------------------------------------
    # Rolling operators
    # -----------------------------------------------------------------------

    def test_rolling_max(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.rolling_max,
                args=[self.close],
                expected_name="MAX_30",
            ),
        )

    def test_rolling_min(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.rolling_min,
                args=[self.close],
                expected_name="MIN_30",
            ),
        )

    def test_rolling_sum(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.rolling_sum,
                args=[self.close],
                expected_name="SUM_30",
            ),
        )

    def test_rolling_max_length_invalid_raises(self):
        with self.assertRaises(ValueError):
            pandas_ta.rolling_max(self.close, length=0)
        with self.assertRaises(ValueError):
            pandas_ta.rolling_max(self.close, length=-5)

    def test_rolling_min_length_invalid_raises(self):
        with self.assertRaises(ValueError):
            pandas_ta.rolling_min(self.close, length=0)
        with self.assertRaises(ValueError):
            pandas_ta.rolling_min(self.close, length=-5)

    def test_rolling_sum_length_invalid_raises(self):
        with self.assertRaises(ValueError):
            pandas_ta.rolling_sum(self.close, length=0)
        with self.assertRaises(ValueError):
            pandas_ta.rolling_sum(self.close, length=-5)

    def test_window_index_convention(self):
        # position counted from the oldest bar of the window, not "bars back"; the oldest wins a tie
        s = Series([1.0, 5.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(pandas_ta.maxindex(s, length=5).tolist()[4:], [1.0, 0.0])
        self.assertEqual(pandas_ta.minindex(s, length=5).tolist()[4:], [0.0, 1.0])
        self.assertEqual(pandas_ta.minmaxindex(s, length=5).iloc[4:].to_numpy().tolist(), [[0.0, 1.0], [1.0, 0.0]])
        # the same tie rule on the minimum side: two lows of 1, at positions 1 and 3
        lows = Series([5.0, 1.0, 3.0, 1.0, 4.0])
        self.assertEqual(pandas_ta.minindex(lows, length=5).iloc[-1], 1.0)
        self.assertEqual(pandas_ta.minmaxindex(lows, length=5).iloc[-1].tolist(), [1.0, 0.0])

    def test_window_index_window_with_a_missing_bar_is_nan(self):
        # argmax would point at the NaN; a window holding one has no position to report
        s = Series([1.0, 5.0, np.nan, 3.0, 4.0, 2.0, 6.0])
        for func in (pandas_ta.maxindex, pandas_ta.minindex):
            self.assertEqual(np.isnan(func(s, length=3).to_numpy()).tolist(), [True] * 5 + [False] * 2)


    def test_maxindex(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.maxindex,
                args=[self.close],
                expected_name="MAXINDEX_30",
            ),
        )

    def test_minindex(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.minindex,
                args=[self.close],
                expected_name="MININDEX_30",
            ),
        )

    def test_minmax(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.minmax,
                args=[self.close],
                expected_name="MINMAX_30",
                expected_type=DataFrame,
                expected_columns=["MIN_30", "MAX_30"],
            ),
        )

    def test_minmaxindex(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.minmaxindex,
                args=[self.close],
                expected_name="MINMAXINDEX_30",
                expected_type=DataFrame,
                expected_columns=["MINIDX_30", "MAXIDX_30"],
            ),
        )

    # -----------------------------------------------------------------------
    # Math transforms
    # -----------------------------------------------------------------------

    def test_acos(self):
        # close prices are out of arccos domain [-1, 1]; result is all-NaN but not None
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.acos,
                args=[self.close],
                expected_name="ACOS",
            ),
        )

    def test_asin(self):
        # close prices are out of arcsin domain [-1, 1]; result is all-NaN but not None
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.asin,
                args=[self.close],
                expected_name="ASIN",
            ),
        )

    def test_atan(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.atan,
                args=[self.close],
                expected_name="ATAN",
            ),
        )

    def test_ceil(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ceil,
                args=[self.close],
                expected_name="CEIL",
            ),
        )

    def test_cos(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cos,
                args=[self.close],
                expected_name="COS",
            ),
        )

    def test_cosh(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.cosh,
                args=[self.close],
                expected_name="COSH",
            ),
        )

    def test_exp(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.exp,
                args=[self.close],
                expected_name="EXP",
            ),
        )

    def test_floor(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.floor,
                args=[self.close],
                expected_name="FLOOR",
            ),
        )

    def test_ln(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.ln,
                args=[self.close],
                expected_name="LN",
            ),
        )

    def test_log10(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.log10,
                args=[self.close],
                expected_name="LOG10",
            ),
        )

    def test_sin(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.sin,
                args=[self.close],
                expected_name="SIN",
            ),
        )

    def test_sinh(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.sinh,
                args=[self.close],
                expected_name="SINH",
            ),
        )

    def test_sqrt(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.sqrt,
                args=[self.close],
                expected_name="SQRT",
            ),
        )

    def test_tan(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.tan,
                args=[self.close],
                expected_name="TAN",
            ),
        )

    def test_tanh(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.tanh,
                args=[self.close],
                expected_name="TANH",
            ),
        )

    # -----------------------------------------------------------------------
    # tulipy extras
    # -----------------------------------------------------------------------

    def test_npabs(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.npabs,
                args=[self.close],
                expected_name="ABS",
            ),
        )

    def test_npround(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.npround,
                args=[self.close],
                expected_name="ROUND",
            ),
        )

    def test_trunc(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.trunc,
                args=[self.close],
                expected_name="TRUNC",
            ),
        )

    def test_todeg(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.todeg,
                args=[self.close],
                expected_name="TODEG",
            ),
        )

    def test_torad(self):
        assert_indicator_standard(
            self,
            IndicatorSpec(
                func=pandas_ta.torad,
                args=[self.close],
                expected_name="TORAD",
            ),
        )
