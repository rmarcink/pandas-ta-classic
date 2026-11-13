from unittest import TestCase

from pandas import DataFrame

import pandas_ta_classic  # noqa: F401  (registers the df.ta accessor)
from tests.assertions import AccessorValueCase
from tests.config import get_sample_data


class TestVolatilityExtension(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()

    @classmethod
    def tearDownClass(cls):
        del cls.data

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_aberration_ext(self):
        self.data.ta.aberration(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(
            list(self.data.columns[-4:]),
            ["ABER_ZG_5_15", "ABER_SG_5_15", "ABER_XG_5_15", "ABER_ATR_5_15"],
        )

    def test_accbands_ext(self):
        self.data.ta.accbands(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(list(self.data.columns[-3:]), ["ACCBL_20", "ACCBM_20", "ACCBU_20"])

    def test_atr_ext(self):
        self.data.ta.atr(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "ATRr_14")

    def test_bbands_ext(self):
        self.data.ta.bbands(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(
            list(self.data.columns[-5:]),
            ["BBL_5_2.0", "BBM_5_2.0", "BBU_5_2.0", "BBB_5_2.0", "BBP_5_2.0"],
        )

    def test_ce_ext(self):
        self.data.ta.ce(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(list(self.data.columns[-2:]), ["CE_L_22_3.0", "CE_S_22_3.0"])

    def test_donchian_ext(self):
        self.data.ta.donchian(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(list(self.data.columns[-3:]), ["DCL_20_20", "DCM_20_20", "DCU_20_20"])

    def test_kc_ext(self):
        self.data.ta.kc(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(list(self.data.columns[-3:]), ["KCLe_20_2", "KCBe_20_2", "KCUe_20_2"])

    def test_massi_ext(self):
        self.data.ta.massi(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "MASSI_9_25")

    def test_natr_ext(self):
        self.data.ta.natr(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "NATR_14")

    def test_pdist_ext(self):
        self.data.ta.pdist(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "PDIST")

    def test_rvi_ext(self):
        self.data.ta.rvi(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "RVI_14")

    def test_rvi_refined_ext(self):
        self.data.ta.rvi(refined=True, append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "RVIr_14")

    def test_rvi_thirds_ext(self):
        self.data.ta.rvi(thirds=True, append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "RVIt_14")

    def test_thermo_ext(self):
        self.data.ta.thermo(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(
            list(self.data.columns[-4:]),
            [
                "THERMO_20_2_0.5",
                "THERMOma_20_2_0.5",
                "THERMOl_20_2_0.5",
                "THERMOs_20_2_0.5",
            ],
        )

    def test_true_range_ext(self):
        self.data.ta.true_range(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "TRUERANGE_1")

    def test_ui_ext(self):
        self.data.ta.ui(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "UI_14")

        self.data.ta.ui(append=True, everget=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "UIe_14")

    def test_hwc_ext(self):
        self.data.ta.hwc(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(list(self.data.columns[-3:]), ["HWM", "HWU", "HWL"])

    def test_avolume_ext(self):
        self.data.ta.avolume(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "AVOLUME_20")

    def test_cvi_ext(self):
        self.data.ta.cvi(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CVI_10")

    def test_hvol_ext(self):
        self.data.ta.hvol(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "HVOL_20")


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


class TestExtVolatilityDeeper(AccessorValueCase):

    def test_atr_ext(self):
        df = self.fresh()
        df.ta.atr(append=True)
        atr_cols = [c for c in df.columns if c.startswith("ATR")]
        self.assertTrue(len(atr_cols) >= 1, "No ATR column appended")
        self._check_appended(df, atr_cols[0])
        # ATR must be non-negative
        valid = df[atr_cols[0]].dropna()
        self.assertTrue((valid >= 0).all(), "ATR has negative values")

    def test_chop_ext(self):
        df = self.fresh()
        df.ta.chop(append=True)
        self._check_appended(df, "CHOP_14_1_100")
        valid = df["CHOP_14_1_100"].dropna()
        self.assertTrue((valid >= 0).all() and (valid <= 100).all(), "CHOP out of [0,100]")

    def test_donchian_ext(self):
        df = self.fresh()
        df.ta.donchian(append=True)
        self._check_appended(df, "DCL_20_20")
        self._check_appended(df, "DCM_20_20")
        self._check_appended(df, "DCU_20_20")

    def test_kc_ext(self):
        df = self.fresh()
        df.ta.kc(append=True)
        kc_cols = [c for c in df.columns if c.startswith("KC")]
        self.assertTrue(len(kc_cols) >= 1, "No KC columns appended")
        for col in kc_cols:
            self._check_appended(df, col)

    def test_pdist_ext(self):
        df = self.fresh()
        df.ta.pdist(append=True)
        self._check_appended(df, "PDIST")
        valid = df["PDIST"].dropna()
        self.assertTrue((valid >= 0).all(), "PDIST has negative values")

    def test_squeeze_ext(self):
        df = self.fresh()
        df.ta.squeeze(append=True)
        sqz_cols = [c for c in df.columns if c.startswith("SQZ")]
        self.assertTrue(len(sqz_cols) >= 1, "No SQZ columns appended")
        for col in sqz_cols:
            self._check_appended(df, col)

    def test_thermo_ext(self):
        df = self.fresh()
        df.ta.thermo(append=True)
        self._check_last_col(df)

    def test_ui_ext(self):
        df = self.fresh()
        df.ta.ui(append=True)
        self._check_appended(df, "UI_14")
        valid = df["UI_14"].dropna()
        self.assertTrue((valid >= 0).all(), "UI has negative values")
