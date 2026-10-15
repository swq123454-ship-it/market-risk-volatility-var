"""Tests for realised measures and the lagged design matrix.

Two things are being defended here.

The first is the *measures*: bipower variation must be unbiased for the diffusion
variance and must be insensitive to a jump, which is the entire basis for
separating the two components of quadratic variation.

The second, and the more important, is the *alignment*. A design matrix that
contains the day it is predicting produces a forecast that looks better than it
is, and no diagnostic in the rest of the pipeline will notice. So the tests do not
check that a shift was applied; they corrupt one day's data and check that the days
before it are unaffected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketrisk import config as cfg
from marketrisk import features as feat


def _bridge_path(rng: np.random.Generator, n: int, sigma: float, bars: int = 78) -> tuple[np.ndarray, np.ndarray]:
    """One Brownian-bridge path per day, plus the day's aggregate diffusion move.

    The bridge is centred and rescaled so that its intraday bars sum to the daily
    diffusion move exactly; that identity is what makes the aggregate of the simulated
    intraday returns equal ``diffusion`` to machine precision, which is the property
    every realised-measure test below leans on.
    """
    diffusion = rng.standard_normal(n) * sigma
    bridge = rng.standard_normal((n, bars))
    bridge = (bridge - bridge.mean(axis=1, keepdims=True)) / np.sqrt((bars - 1.0) / bars)
    return diffusion[:, None] / bars + sigma * np.sqrt(bars - 1.0) / bars * bridge, diffusion


# ---------------------------------------------------------------------------
# Bipower variation
# ---------------------------------------------------------------------------
def test_bipower_is_unbiased_on_a_pure_diffusion_path():
    rng = np.random.default_rng(3)
    sigma = 0.011
    intraday, _ = _bridge_path(rng, 20_000, sigma)
    rv = feat.realised_variance(intraday)
    bv = feat.bipower_variation(intraday)
    # Both estimate the same quantity when nothing jumps.
    assert rv.mean() / sigma**2 == pytest.approx(1.0, rel=0.02)
    assert bv.mean() / sigma**2 == pytest.approx(1.0, rel=0.03)


def test_bipower_ignores_a_jump_that_realised_variance_absorbs():
    """A single large bar inflates RV by an order of magnitude and moves BV by ~25%.

    The magnitudes are set so the contrast is unambiguous: a jump every twentieth
    day of fifteen intraday standard deviations contributes about twelve times the
    whole diffusion variance to RV, and only the two products that straddle the
    jump bar to BV.
    """
    rng = np.random.default_rng(4)
    sigma = 0.01
    intraday, _ = _bridge_path(rng, 5_000, sigma)
    rv_before = feat.realised_variance(intraday).mean()
    bv_before = feat.bipower_variation(intraday).mean()

    jumped = intraday.copy()
    jumped[::20, 39] += 0.15

    rv_after = feat.realised_variance(jumped).mean()
    bv_after = feat.bipower_variation(jumped).mean()
    assert rv_after > 5.0 * rv_before
    # BV does move a little: the jump sits in two of the 77 products on its day.
    assert bv_after < 1.5 * bv_before


def test_jump_variation_is_non_negative():
    rv = np.array([1.0, 0.5, 2.0])
    bv = np.array([0.9, 0.6, 1.0])
    jv = feat.jump_variation(rv, bv)
    assert np.all(jv >= 0.0)
    assert np.allclose(jv, [0.1, 0.0, 1.0])


def test_jump_ratio_statistic_is_standardised():
    """Under the null the statistic has roughly unit variance."""
    rng = np.random.default_rng(5)
    intraday, _ = _bridge_path(rng, 30_000, 0.01)
    stat = feat.jump_ratio_statistic(feat.realised_variance(intraday), feat.bipower_variation(intraday))
    assert abs(stat.mean()) < 0.1
    assert 0.5 < stat.std() < 2.0


# ---------------------------------------------------------------------------
# Range estimators
# ---------------------------------------------------------------------------
def test_parkinson_scales_with_the_range_not_the_level():
    high = pd.Series([110.0, 121.0])
    low = pd.Series([100.0, 110.0])
    value = feat.parkinson(high, low).to_numpy()
    assert value[0] == pytest.approx(value[1], rel=1e-9)


def test_garman_klass_and_rogers_satchell_are_drift_robust_here():
    """A pure uptrend with no range gives zero variance for RS, not a negative one."""
    open_ = pd.Series([100.0, 200.0])
    close = pd.Series([200.0, 400.0])
    high = pd.Series([200.0, 400.0])
    low = pd.Series([100.0, 200.0])
    assert (feat.garman_klass(open_, high, low, close).to_numpy() >= 0).all()
    assert (feat.rogers_satchell(open_, high, low, close).to_numpy() >= 0).all()


# ---------------------------------------------------------------------------
# HAR components
# ---------------------------------------------------------------------------
def test_har_components_average_in_logs():
    """The weekly regressor is the mean of log RV, not the log of mean RV.

    The series has to be longer than the quarterly window, or the quarterly
    component is simply ``NaN`` and the comparison against a truncated manual mean
    is meaningless.
    """
    values = pd.Series(np.exp(np.linspace(0.0, 1.0, 200)))
    log_rv = np.log(values)
    out = feat.har_components(log_rv, {"d": 1, "w": 5, "m": 22})
    assert out["log_rv_w"].iloc[-1] == pytest.approx(log_rv.iloc[-5:].mean(), rel=1e-12)
    assert out["log_rv_m"].iloc[-1] == pytest.approx(log_rv.iloc[-22:].mean(), rel=1e-12)
    assert out["log_rv_q"].iloc[-1] == pytest.approx(log_rv.iloc[-88:].mean(), rel=1e-12)
    # Averaging in logs is not the same as logging the average, and on a series this
    # steep the two differ by enough to matter.
    assert out["log_rv_w"].iloc[-1] != pytest.approx(np.log(values.iloc[-5:].mean()), rel=1e-6)


def test_har_components_are_nan_before_the_window_is_full():
    log_rv = pd.Series(np.log(np.linspace(1e-4, 1e-3, 30)))
    out = feat.har_components(log_rv, {"d": 1, "w": 5, "m": 22})
    assert np.isnan(out["log_rv_m"].iloc[10])
    assert np.isfinite(out["log_rv_m"].iloc[-1])


# ---------------------------------------------------------------------------
# The alignment invariant
# ---------------------------------------------------------------------------
def test_every_declared_feature_is_present(panel):
    for column in feat.FEATURE_COLUMNS:
        assert column in panel.features.columns
    assert feat.TARGET_COLUMN in panel.features.columns


def test_features_are_dated_for_the_day_being_forecast(panel):
    """Row ``t`` holds the return realised on ``t-1``, and the target of ``t``."""
    frame = panel.features
    raw = panel.raw
    assert np.allclose(frame["ret_1"].dropna().to_numpy(), raw["ret"].shift(1).dropna().to_numpy())
    assert np.allclose(
        frame[feat.TARGET_COLUMN].dropna().to_numpy(), np.log(raw["realised_variance"]).dropna().to_numpy()
    )


def test_contaminating_one_day_cannot_change_earlier_rows(panel):
    """The alignment test: poison day ``d`` and check days ``d`` and earlier.

    This is the assertion the module exists to make. Note what is checked and
    what is not: the target *on* day ``d`` is expected to move, because the target
    is day ``d``'s realised variance. The predictors on day ``d`` are expected not
    to move, because day ``d``'s predictors are known at the close of ``d-1``. A
    design matrix built without the final shift passes none of this.
    """
    raw = panel.raw.copy()
    d = 900
    poisoned = raw.copy()
    poisoned.iloc[d, poisoned.columns.get_loc("ret")] *= 7.0
    poisoned.iloc[d, poisoned.columns.get_loc("realised_variance")] *= 12.0
    poisoned.iloc[d, poisoned.columns.get_loc("bipower")] *= 12.0
    for column in ("open", "high", "low", "close"):
        poisoned.iloc[d, poisoned.columns.get_loc(column)] *= 1.4

    before = feat.build_features(raw)
    after = feat.build_features(poisoned)

    for column in feat.FEATURE_COLUMNS:
        left = before[column].iloc[: d + 1].to_numpy()
        right = after[column].iloc[: d + 1].to_numpy()
        both = np.isfinite(left) & np.isfinite(right)
        assert np.allclose(left[both], right[both], atol=0), column

    # Sanity: the poison has to be visible somewhere, or the test proves nothing.
    assert after[feat.TARGET_COLUMN].iloc[d] != before[feat.TARGET_COLUMN].iloc[d]
    changed = [c for c in feat.FEATURE_COLUMNS if after[c].iloc[d + 1] != before[c].iloc[d + 1]]
    assert changed, "contaminating day d must show up in day d+1's predictors"


def test_contaminating_the_future_cannot_change_a_past_forecast(returns, features):
    """The model-level version of the same invariant.

    A model fitted on data up to ``T`` and asked for a forecast on ``T`` must be
    unaffected by anything that happens after ``T``. If it is not, every
    out-of-sample number in the study is optimistic.
    """
    from marketrisk import volmodels

    t = 1_400

    def fitted_forecast(series: pd.Series) -> float:
        model = volmodels.HAR(use_vix=True)
        model.fit(series.to_numpy(dtype=float), features, t)
        return float(model.forecast(series.to_numpy(dtype=float), features, t, t + 1)[0])

    baseline = fitted_forecast(returns)
    poisoned = returns.copy()
    poisoned.iloc[t + 50 : t + 150] *= -8.0
    assert fitted_forecast(poisoned) == pytest.approx(baseline, rel=1e-12)


def test_sparse_rows_are_dropped_not_zero_filled(panel):
    """The first rows are incomplete, and they are dropped rather than filled."""
    frame = panel.features
    usable = frame.loc[:, list(feat.FEATURE_COLUMNS) + [feat.TARGET_COLUMN]].notna().all(axis=1)
    assert not usable.iloc[0]
    assert usable.iloc[-1]
    # If NaNs had been filled with zero, every row would be usable.
    assert usable.sum() < len(frame)
    assert cfg.HAR_LAGS["m"] - 1 <= int(np.argmax(usable.to_numpy()))
