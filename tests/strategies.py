"""Hypothesis strategies shared by the property-based tests.

We use ``hypothesis.extra.numpy`` and ``hypothesis.extra.pandas`` to generate
realistic price series while also injecting pathological values (NaN, ±Inf,
zero, negative) at controlled rates to probe boundary behaviour.
"""

import numpy as np
import pandas as pd
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

# ---------------------------------------------------------------------------
# Custom Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def price_series(
    draw,
    min_size: int = 2,
    max_size: int = 200,
    allow_nan: bool = False,
    nan_prob: float = 0.0,
) -> pd.Series:
    """Generate a realistic price-like Series.

    Parameters
    ----------
    min_size, max_size : int
        Length bounds.
    allow_nan : bool
        If True, each value is NaN with probability *nan_prob*.
    nan_prob : float
        Probability of a value being NaN, in (0, 1). Required with
        *allow_nan*; a value above 0 without it is rejected.
    """
    if allow_nan != (nan_prob > 0) or not 0 <= nan_prob < 1:
        raise ValueError(f"price_series() needs allow_nan=True with 0 < nan_prob < 1, or neither; got allow_nan={allow_nan}, nan_prob={nan_prob}")
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    raw = draw(arrays(dtype=np.float64, shape=n, elements=st.floats(min_value=1.0, max_value=1000.0, width=64)))
    if allow_nan:
        # ``floats`` cannot combine allow_nan with min/max, so draw finite
        # prices and replace each with NaN with probability nan_prob. The mask
        # comes from a drawn seed: element-wise Hypothesis floats shrink towards
        # 0.0 and would turn the whole series NaN.
        rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2**32 - 1)))
        raw[rng.random(n) < nan_prob] = np.nan
    return pd.Series(raw, name="close")


@st.composite
def ohlcv_dataframe(draw, min_size: int = 20, max_size: int = 200):
    """Generate a realistic OHLCV DataFrame compatible with ``df.ta`` accessor."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))

    # Generate a base price and derive OHLC from it
    base = (
        np.cumsum(
            draw(
                arrays(
                    dtype=np.float64,
                    shape=n,
                    elements=st.floats(min_value=-2.0, max_value=2.0, width=64, allow_nan=False),
                )
            )
        )
        + 100.0
    )

    df = pd.DataFrame(index=pd.RangeIndex(n))
    df["open"] = base + draw(
        arrays(
            dtype=np.float64,
            shape=n,
            elements=st.floats(min_value=-1.0, max_value=1.0, width=64),
        )
    )
    df["high"] = df["open"] + np.abs(
        draw(
            arrays(
                dtype=np.float64,
                shape=n,
                elements=st.floats(min_value=0.1, max_value=3.0, width=64),
            )
        )
    )
    df["low"] = df["open"] - np.abs(
        draw(
            arrays(
                dtype=np.float64,
                shape=n,
                elements=st.floats(min_value=0.1, max_value=3.0, width=64),
            )
        )
    )
    df["close"] = draw(
        arrays(
            dtype=np.float64,
            shape=n,
            elements=st.floats(min_value=df["low"].min(), max_value=df["high"].max(), width=64),
        )
    )
    df["volume"] = np.abs(
        draw(
            arrays(
                dtype=np.float64,
                shape=n,
                elements=st.floats(min_value=0.0, max_value=1e7, width=64),
            )
        )
    )

    # Ensure high >= low and close in range
    df["high"] = np.maximum(df["high"], df[["open", "close"]].max(axis=1))
    df["low"] = np.minimum(df["low"], df[["open", "close"]].min(axis=1))
    # Real prices are positive, and a walk of 200 steps of -2 from 100 reaches -300.
    # Lift the frame so its lowest price is 1: a shift keeps every bar-to-bar
    # difference, and a level-dependent indicator reads NaN and warns at or below
    # zero (#345), which is not what these tests are about.
    prices = ["open", "high", "low", "close"]
    df[prices] += max(0.0, 1.0 - float(df[prices].min().min()))
    return df


@st.composite
def constant_price_series(draw, min_size: int = 20, max_size: int = 200):
    """Generate a constant-valued Series — tests degenerate arithmetic."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    value = draw(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, width=64))
    return pd.Series([value] * n, dtype=np.float64)
