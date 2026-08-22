Testing
=======

**Pandas TA Classic** uses a multi-layered testing strategy to ensure
indicator correctness, robustness, and reliability.

.. contents::
   :local:
   :depth: 1


Unit Tests
----------

**Why:** Verify individual indicators produce correct values for known inputs.

**Files:** ``test_indicator_candle.py``, ``test_indicator_cycles.py``,
``test_indicator_momentum.py``, ``test_indicator_overlap.py``,
``test_indicator_performance.py``, ``test_indicator_statistics.py``,
``test_indicator_trend.py``, ``test_indicator_volatility.py``,
``test_indicator_volume.py``, ``test_indicator_math.py``.

Uses ``IndicatorSpec``-based assertions (``assert_indicator_standard``)
against real market data from ``SPY_D.csv``.

**Run:** ``python -m pytest tests/test_indicator_momentum.py -v``


Extension API Tests
-------------------

**Why:** Confirm indicators work correctly through the ``df.ta`` DataFrame
accessor with ``append=True``.

**Files:** ``test_ext_indicator_candle.py``, ``test_ext_indicator_cycles.py``,
``test_ext_indicator_momentum.py``, ``test_ext_indicator_overlap_ext.py``,
``test_ext_indicator_performance.py``, ``test_ext_indicator_statistics.py``,
``test_ext_indicator_trend.py``, ``test_ext_indicator_volatility.py``,
``test_ext_indicator_volume.py``.

**Run:** ``python -m pytest tests/test_ext_indicator_momentum.py -v``


Accessor API Tests
------------------

**Why:** Validate DataFrame accessor metadata and utilities: ``prefix``/``suffix``
naming, ``indicators()`` discovery, ``ticker()`` data fetching, time range
filtering, and ``constants()``.

**Files:** ``test_accessor_api.py``, ``test_ext_assertions.py``,
``test_accessor_conformance.py`` (every indicator reachable through the
accessor returns a Series or DataFrame).

**Run:** ``python -m pytest tests/test_accessor_api.py -v``


Oracle / Comparison Tests
-------------------------

**Why:** Compare native (``talib=False``) implementations against
TA-Lib (C library) and tulipy outputs to catch numerical divergence.

The two oracles work differently:

* ``test_oracle_talib.py`` imports TA-Lib at test time and skips when it is
  missing.  Install it with ``pip install -e ".[oracle]"``.
* ``test_oracle_tulipy.py`` does **not** import tulipy.  tulipy is
  unmaintained and ships no wheels for CPython 3.12+, so its values are
  frozen in ``tests/fixtures/tulipy_oracle.json`` and compared on every
  Python version.  Regenerating that snapshot is the only thing that needs a
  tulipy install (see `Fixture Files`_).

**Run:** ``python -m pytest tests/test_oracle_talib.py tests/test_oracle_tulipy.py -v``


Native Indicator Tests
----------------------

**Why:** Cover indicators that have no TA-Lib alternative, validating
return type, non-NaN row count, value finiteness, and mathematical bounds.

**Files:** ``test_native_indicators.py``.

**Run:** ``python -m pytest tests/test_native_indicators.py -v``


Regression Tests
----------------

**Why:** Prevent reintroduction of known bugs and catch silent value drift.

- ``test_regression.py`` — Spot-checks indicator values at 5 fixed indices
  (50, 200, 500, 1500, 3000) against stored fixture data.
- ``test_regression_bugfixes.py`` — Pins ~12 documented fixes from CHANGELOG.
- ``test_indicator_values.py`` — Golden fixture tests: checks last non-NaN
  values and per-column NaN counts against snapshots in ``tests/fixtures/``.

**Run:** ``python -m pytest tests/test_regression.py -v``


Edge-Case Tests
---------------

**Why:** Verify indicators don't crash on degenerate inputs.

- ``test_indicator_edge_cases.py`` — All-NaN series, constant-price series,
  ±Inf injection at mid-series positions, and mismatched OHLCV lengths.
- ``test_nan_behaviour.py`` — NaN prefix warmup periods, minimum length
  requirements, boundary conditions.

**Run:** ``python -m pytest tests/test_indicator_edge_cases.py -v``


Integration / E2E Tests
-----------------------

**Why:** Exercise full workflows end-to-end.

**Files:** ``test_integration_e2e.py`` — Multi-indicator chaining,
Strategy execution with ``df.ta.strategy()``, plugin binding, and
category-strategy runs.

**Run:** ``python -m pytest tests/test_integration_e2e.py -v``


Fluent API Tests
----------------

**Why:** Validate the ``df.ta.chain()`` fluent programming API.

**Files:** ``test_fluent_chaining.py`` — Chained indicator calls,
auto-append behaviour, ``unchain()``.

**Run:** ``python -m pytest tests/test_fluent_chaining.py -v``


Strategy Tests
--------------

**Why:** Confirm the ``Strategy`` class executes correctly, including
multi-core processing.

**Files:** ``test_strategy.py``.

**Run:** ``python -m pytest tests/test_strategy.py -v``


Custom / Plugin Tests
---------------------

**Why:** Verify the custom indicator registration system.

**Files:** ``test_custom.py`` — ``bind()`` and ``import_dir()`` from
``pandas_ta_classic.custom``, module loading, and custom indicator discovery.

**Run:** ``python -m pytest tests/test_custom.py -v``


Property-Based Tests
--------------------

**Why:** Randomized input testing using `Hypothesis
<https://hypothesis.readthedocs.io/en/latest/>`_ to discover edge cases that
deterministic tests miss — overflow conditions, NaN propagation bugs,
boundary violations.

**Files:** ``test_property_based.py``.

**What's tested:**

* **Output invariants** — Type correctness, length preservation, naming.
* **Mathematical invariants** — Bollinger Band ordering, ATR/STDEV
  non-negativity, MOM/ROC relationship.
* **Core utilities** — ``verify_series``, ``apply_offset``, ``apply_fill``.
* **None-guard safety** — Indicators return ``None`` for ``None`` input.
* **NaN propagation** — All-NaN input → all-NaN output, no crash.
* **Idempotence** — Same args twice → identical result.
* **Category discovery** — Dynamic discovery stays consistent.
* **Boundedness** — RSI, stochastic oscillator within expected ranges
  (where input assumptions hold).

**Strategies used:**

* Random walks — Cumulative sum of normal increments.
* OHLCV DataFrames — Derived OHLC with high ≥ low, close ∈ [low, high].
* Constant series — Degenerate arithmetic testing.
* Controlled NaN injection — Finite floats with proportionally sampled NaN.

**Run:**

.. code-block:: bash

   python -m pytest tests/test_property_based.py -v
   python -m pytest tests/test_property_based.py -v --hypothesis-show-statistics
   python -m pytest tests/test_property_based.py -v --hypothesis-profile=ci

**Adding property tests for a new indicator:**

.. code-block:: python

   import hypothesis.strategies as st
   from hypothesis import assume, given, settings

   @given(price_series(min_size=30, max_size=200), st.integers(min_value=2, max_value=20))
   @settings(max_examples=100)
   def test_my_indicator_output_invariant(s, length):
       assume(len(s) >= length + 2)
       result = ta.my_indicator(s, length=length)
       assert isinstance(result, pd.Series)
       assert len(result) == len(s)
       assert str(length) in result.name


Infrastructure Tests
--------------------

**Why:** Cover the machinery around the indicators — lazy imports and the
optional numba acceleration — which no indicator test exercises directly.

- ``test_lazy_core.py`` — Lazy subpackage loading: ``__getattr__`` /
  ``__dir__`` on the package and its subpackages, indicator resolution and
  wrapper caching, and the regression that a cross-package import returns the
  function rather than the module.
- ``test_njit_fallback.py`` — Whether ``@njit`` compiles is decided once at
  import time, so each configuration runs in a fresh subprocess: numba
  present, numba absent, and numba installed but unimportable.  Only the last
  one must warn; running without numba is supported and stays quiet.

**Run:** ``python -m pytest tests/test_lazy_core.py tests/test_njit_fallback.py -v``


Utility Tests
-------------

**Files:** ``test_utils.py`` (``verify_series``, ``apply_offset``,
``apply_fill``, cross detection), ``test_utils_metrics.py`` (Sharpe ratio,
drawdown, CAGR, Jensen's alpha), ``test_utils_data_alphavantage.py``
(AlphaVantage data fetching).

**Run:** ``python -m pytest tests/test_utils.py -v``


Running All Tests
-----------------

.. code-block:: bash

   # Full test suite (primary — this is what CI runs)
   python -m pytest tests/ -v

   # Core suite only, as CI's testing-core job invokes it
   python -m pytest tests/        --ignore=tests/test_oracle_talib.py        --ignore=tests/test_oracle_tulipy.py -v

   # Regenerate fixture JSONs only (requires TA-Lib installed)
   make fixtures

   # Regenerate fixtures, then run the suite through unittest (Makefile target)
   make test-all

   # With coverage
   python -m pytest --cov=pandas_ta_classic --cov-report=html tests/


Fixture Files
-------------

``tests/fixtures/expected_values.json`` and
``tests/fixtures/regression_snapshots.json`` are **generated** files.
They are rebuilt automatically when ``tests/`` is imported (before any
test runs) if TA-Lib is available.  Manual regeneration:

.. code-block:: bash

   python -m tests.fixtures.generate_fixtures
   python -m tests.fixtures.generate_regression_snapshots

Both scripts can also be invoked directly (``python tests/fixtures/generate_*.py``)
and require the project root to be on ``sys.path``.

``tests/fixtures/tulipy_oracle.json`` is generated too, but **not**
automatically — it is a committed snapshot that the oracle tests read on every
Python version.  Regenerate it only when the tulipy comparison itself changes,
on a CPython version below 3.12 with tulipy installed:

.. code-block:: bash

   python tests/fixtures/generate_tulipy_oracle.py
