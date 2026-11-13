from unittest import TestCase

from pandas import DataFrame

import pandas_ta_classic as pandas_ta
from tests.assertions import AccessorValueCase
from tests.config import get_sample_data


class TestPerformaceExtension(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()
        cls.islong = cls.data["close"] > pandas_ta.sma(cls.data["close"], length=50)

    @classmethod
    def tearDownClass(cls):
        del cls.data
        del cls.islong

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_log_return_ext(self):
        self.data.ta.log_return(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "LOGRET_1")

    def test_cum_log_return_ext(self):
        self.data.ta.log_return(append=True, cumulative=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CUMLOGRET_1")

    def test_percent_return_ext(self):
        self.data.ta.percent_return(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "PCTRET_1")

    def test_cum_percent_return_ext(self):
        self.data.ta.percent_return(append=True, cumulative=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CUMPCTRET_1")

    def test_drawdown_ext(self):
        self.data.ta.drawdown(append=True)
        self.assertIsInstance(self.data, DataFrame)
        # Drawdown returns 3 columns: DD, DD_PCT, DD_LOG
        self.assertEqual(list(self.data.columns[-3:]), ["DD", "DD_PCT", "DD_LOG"])


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class TestExtPerformanceDeeper(AccessorValueCase):

    def test_log_return_ext(self):
        df = self.fresh()
        df.ta.log_return(append=True)
        lr_cols = [c for c in df.columns if c.startswith("LOGRET")]
        self.assertTrue(len(lr_cols) >= 1, "No LOGRET column appended")
        self._check_appended(df, lr_cols[0])

    def test_percent_return_ext(self):
        df = self.fresh()
        df.ta.percent_return(append=True)
        pct_cols = [c for c in df.columns if c.startswith("PCTRET")]
        self.assertTrue(len(pct_cols) >= 1, "No PCTRET column appended")
        self._check_appended(df, pct_cols[0])
