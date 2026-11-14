import pandas_ta_classic  # noqa: F401  (registers the df.ta accessor)
from tests.assertions import AccessorValueCase


class TestExtCyclesDeeper(AccessorValueCase):
    def test_ebsw_ext(self):
        df = self.fresh()
        df.ta.ebsw(append=True)
        self._check_last_col(df)

    def test_swak_2php_ext(self):
        df = self.fresh()
        df.ta.swak_2php(append=True)
        self._check_appended(df, "SWAK2PHP_20")

    def test_swak_bp_ext(self):
        df = self.fresh()
        df.ta.swak_bp(append=True)
        self._check_appended(df, "SWAKBP_20_0.1")

    def test_swak_hp_ext(self):
        df = self.fresh()
        df.ta.swak_hp(append=True)
        self._check_appended(df, "SWAKHP_20")
