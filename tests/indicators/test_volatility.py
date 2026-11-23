
import numpy as np
import pandas as pd
import pytest

import pandas_ta_classic as pandas_ta
from tests.config import get_sample_data


class TestVolatility:
    @classmethod
    def setup_class(cls):
        cls.data = get_sample_data()
        cls.data.columns = cls.data.columns.str.lower()
        cls.open = cls.data["open"]
        cls.high = cls.data["high"]
        cls.low = cls.data["low"]
        cls.close = cls.data["close"]
        if "volume" in cls.data.columns:
            cls.volume = cls.data["volume"]

    @classmethod
    def teardown_class(cls):
        del cls.open
        del cls.high
        del cls.low
        del cls.close
        if hasattr(cls, "volume"):
            del cls.volume
        del cls.data

    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    def test_cvi(self):
        assert pandas_ta.cvi(self.high, None) is None

    def test_rvi_mode_arguments_fail_fast(self, subtests):
        # refined/thirds used to be bool()-coerced, so rvi(refined="yes") ran
        # the refined branch without a word.
        for kwarg in ("refined", "thirds"):
            with subtests.test(kwarg=kwarg), pytest.raises(ValueError, match=rf"rvi\(\) {kwarg} must be True or False"):
                pandas_ta.rvi(self.close, self.high, self.low, **{kwarg: "yes"})

        # They are alternative modes; refined used to win silently.
        with pytest.raises(ValueError, match="alternative modes"):
            pandas_ta.rvi(self.close, self.high, self.low, refined=True, thirds=True)

        # Both modes are defined over high/low. Omitting them used to return
        # None, which reads as an ordinary short-input result.
        for kwarg in ("refined", "thirds"):
            with subtests.test(kwarg=kwarg), pytest.raises(ValueError, match=rf"rvi\(\) {kwarg}=True needs both high and low"):
                pandas_ta.rvi(self.close, **{kwarg: True})
        with pytest.raises(ValueError, match="needs both high and low"):
            pandas_ta.rvi(self.close, high=self.high, refined=True)


class TestTalib08Volatility:
    @classmethod
    def setup_class(cls):
        cls.data = get_sample_data()

    def test_adr_is_the_sma_of_the_bar_range(self):
        d = self.data
        np.testing.assert_allclose(pandas_ta.adr(d.high, d.low, 14), (d.high - d.low).rolling(14).mean(), rtol=1e-12, equal_nan=True)

    def test_rogerssatchell_does_not_depend_on_the_price_unit(self):
        d = self.data
        base = pandas_ta.rogerssatchell(d.open, d.high, d.low, d.close)
        scaled = pandas_ta.rogerssatchell(d.open * 1e-4, d.high * 1e-4, d.low * 1e-4, d.close * 1e-4)
        np.testing.assert_allclose(scaled, base, rtol=1e-9, equal_nan=True)

    def test_rogerssatchell_is_zero_on_bars_that_never_move(self):
        flat = pd.Series(100.0, index=range(30))
        assert (pandas_ta.rogerssatchell(flat, flat, flat, flat).dropna() == 0.0).all()

    def test_rogerssatchell_annualization_scales_by_its_root(self):
        d = self.data
        daily = pandas_ta.rogerssatchell(d.open, d.high, d.low, d.close, annualization=1)
        yearly = pandas_ta.rogerssatchell(d.open, d.high, d.low, d.close)
        np.testing.assert_allclose(yearly, daily * np.sqrt(252), rtol=1e-12, equal_nan=True)
        assert daily.name == "RS_10_annualization1"
