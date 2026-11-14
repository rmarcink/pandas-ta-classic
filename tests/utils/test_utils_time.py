"""Tests of pandas_ta_classic/utils/_time.py: time ranges and time zones."""

from unittest import TestCase

import pandas as pd
import pytest
from hypothesis import assume, given, strategies as st
from pandas import DataFrame

import pandas_ta_classic as pandas_ta
from tests.config import get_sample_data


class TestTimeUtilities(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()

    @classmethod
    def tearDownClass(cls):
        del cls.data

    def setUp(self):
        self.utils = pandas_ta.utils

    def tearDown(self):
        del self.utils

    def test_df_year_to_date(self):
        result = self.utils.df_year_to_date(self.data)
        self.assertIsInstance(result, DataFrame)
        self.assertLessEqual(len(result), len(self.data))

    def test_to_utc(self):
        result = self.utils.to_utc(self.data.copy())
        self.assertTrue(isinstance(result.index.dtype, pd.DatetimeTZDtype))

    def test_total_time(self):
        result = self.utils.total_time(self.data)
        self.assertEqual(20.824093086926762, result)

        result = self.utils.total_time(self.data, "months")
        self.assertEqual(250.05753361606995, result)

        result = self.utils.total_time(self.data, "weeks")
        self.assertEqual(1086.5714285714287, result)

        result = self.utils.total_time(self.data, "days")
        self.assertEqual(7606, result)

        result = self.utils.total_time(self.data, "hours")
        self.assertEqual(182544, result)

        result = self.utils.total_time(self.data, "minutes")
        self.assertEqual(10952640.0, result)

        result = self.utils.total_time(self.data, "seconds")
        self.assertEqual(657158400.0, result)

    def test_total_time_needs_an_ascending_index(self):
        # A descending index gave -20.8 years, which the metrics turned into NaN.
        with self.assertRaisesRegex(ValueError, r"total_time\(\) needs an index sorted in ascending order"):
            self.utils.total_time(self.data.iloc[::-1])
        # A single row, or rows sharing one timestamp, truly span 0.
        self.assertEqual(self.utils.total_time(self.data.iloc[:1]), 0)
        twice = pd.concat([self.data.iloc[:1], self.data.iloc[:1]])
        self.assertEqual(self.utils.total_time(twice, "days"), 0)


class TestTotalTimeProperties(TestCase):
    """Properties of total_time."""

    @given(st.integers(min_value=2, max_value=60), st.data())
    def test_total_time_rejects_any_unsorted_index(self, n, data):
        """A descending index gave a negative span; any reordering now raises."""
        index = pd.date_range("2021-01-04", periods=n, freq="D")
        order = data.draw(st.permutations(range(n)))
        assume(list(order) != list(range(n)))
        with pytest.raises(ValueError, match="sorted in ascending order"):
            pandas_ta.utils.total_time(pd.Series(range(n), index=index[list(order)]))
        assert pandas_ta.utils.total_time(pd.Series(range(n), index=index), "days") == n - 1
