from unittest import TestCase

from pandas import DataFrame

import pandas_ta_classic  # noqa: F401  (registers the df.ta accessor)
from tests.assertions import AccessorValueCase
from tests.config import get_sample_data


class TestVolumeExtension(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = get_sample_data()
        cls.open = cls.data["open"]

    @classmethod
    def tearDownClass(cls):
        del cls.data
        del cls.open

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ad_ext(self):
        self.data.ta.ad(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "AD")

    def test_ad_open_ext(self):
        self.data.ta.ad(open_=self.open, append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "ADo")

    def test_adosc_ext(self):
        self.data.ta.adosc(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "ADOSC_3_10")

    def test_aobv_ext(self):
        self.data.ta.aobv(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(
            list(self.data.columns[-7:]),
            [
                "OBV",
                "OBV_min_2",
                "OBV_max_2",
                "OBVe_4",
                "OBVe_12",
                "AOBV_LR_2",
                "AOBV_SR_2",
            ],
        )
        # Remove "OBV" so it does not interfere with test_obv_ext()
        del self.data["OBV"]  # the frame is shared by the class, so in place

    def test_cmf_ext(self):
        self.data.ta.cmf(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "CMF_20")

    def test_efi_ext(self):
        self.data.ta.efi(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "EFI_13")

    def test_eom_ext(self):
        self.data.ta.eom(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "EOM_14_100000000")

    def test_kvo_ext(self):
        self.data.ta.kvo(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(list(self.data.columns[-2:]), ["KVO_34_55_13", "KVOs_34_55_13"])

    def test_mfi_ext(self):
        self.data.ta.mfi(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "MFI_14")

    def test_nvi_ext(self):
        self.data.ta.nvi(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "NVI_1")

    def test_obv_ext(self):
        self.data.ta.obv(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "OBV")

    def test_pvi_ext(self):
        self.data.ta.pvi(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "PVI_1")

    def test_pvol_ext(self):
        self.data.ta.pvol(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "PVOL")

    def test_pvr_ext(self):
        self.data.ta.pvr(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "PVR")

    def test_pvt_ext(self):
        self.data.ta.pvt(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "PVT")

    def test_vp_ext(self):
        result = self.data.ta.vp()
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "VP_10")

    def test_vfi_ext(self):
        self.data.ta.vfi(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "VFI_130")

    def test_emv_ext(self):
        self.data.ta.emv(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "EMV")

    def test_marketfi_ext(self):
        self.data.ta.marketfi(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "MARKETFI")

    def test_vosc_ext(self):
        self.data.ta.vosc(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "VOSC_14_28")

    def test_wad_ext(self):
        self.data.ta.wad(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "WAD")


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


class TestExtVolumeDeeper(AccessorValueCase):

    def test_ad_ext(self):
        df = self.fresh()
        df.ta.ad(append=True)
        ad_cols = [c for c in df.columns if c.startswith("AD")]
        self.assertTrue(len(ad_cols) >= 1, "No AD column appended")
        self._check_appended(df, ad_cols[0])

    def test_aobv_ext(self):
        df = self.fresh()
        df.ta.aobv(append=True)
        self._check_appended(df, "OBV")

    def test_cmf_ext(self):
        df = self.fresh()
        df.ta.cmf(append=True)
        self._check_last_col(df)

    def test_efi_ext(self):
        df = self.fresh()
        df.ta.efi(append=True)
        self._check_appended(df, "EFI_13")

    def test_mfi_ext(self):
        df = self.fresh()
        df.ta.mfi(append=True)
        mfi_cols = [c for c in df.columns if c.startswith("MFI")]
        self.assertTrue(len(mfi_cols) >= 1, "No MFI column appended")
        self._check_appended(df, mfi_cols[0])
        # MFI in [0, 100]
        valid = df[mfi_cols[0]].dropna()
        self.assertTrue((valid >= 0).all() and (valid <= 100).all(), "MFI out of [0,100]")

    def test_obv_ext(self):
        df = self.fresh()
        df.ta.obv(append=True)
        obv_cols = [c for c in df.columns if c.startswith("OBV")]
        self.assertTrue(len(obv_cols) >= 1, "No OBV column appended")
        self._check_appended(df, obv_cols[0])

    def test_pvol_ext(self):
        df = self.fresh()
        df.ta.pvol(append=True)
        self._check_appended(df, "PVOL")
        valid = df["PVOL"].dropna()
        self.assertTrue((valid >= 0).all(), "PVOL has negative values")

    def test_pvt_ext(self):
        df = self.fresh()
        df.ta.pvt(append=True)
        self._check_appended(df, "PVT")

    def test_vwmacd_ext(self):
        df = self.fresh()
        df.ta.vwmacd(append=True)
        vwm_cols = [c for c in df.columns if c.startswith("VWMACD")]
        self.assertTrue(len(vwm_cols) >= 1, "No VWMACD columns appended")
        for col in vwm_cols:
            self._check_appended(df, col)
