"""Tests for the scoring rules and the forecast-evaluation tables.

QLIKE's defining property is scale invariance, and that is the first test: if
doubling every forecast and every proxy changes the loss, the loss is not the one
the volatility literature uses. The Mincer-Zarnowitz tests check the joint
restriction rather than the individual coefficients, because a slope of one with an
intercept of two is still a badly calibrated forecast. The Diebold-Mariano tests
check the sign convention, which is the only thing about DM that is easy to get
backwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketrisk import config as cfg
from marketrisk import evaluate, losses, var


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def test_qlike_is_minimised_at_the_true_value():
    truth = np.array([1e-4, 4e-4])
    assert np.allclose(losses.qlike(truth, truth), 0.0)


def test_qlike_is_scale_invariant():
    """The property that makes it the right loss for a latent variance."""
    forecast = np.array([1e-4, 2e-4, 5e-4])
    proxy = np.array([1.2e-4, 1.7e-4, 6e-4])
    assert np.allclose(losses.qlike(forecast, proxy), losses.qlike(forecast * 100.0, proxy * 100.0))


def test_qlike_is_non_negative():
    rng = np.random.default_rng(51)
    forecast = rng.uniform(1e-5, 1e-3, 500)
    proxy = rng.uniform(1e-5, 1e-3, 500)
    assert (losses.qlike(forecast, proxy) >= 0).all()


def test_qlike_penalises_a_downward_bias_more_than_an_upward_one():
    """The asymmetry that matters for risk: under-forecasting variance is worse."""
    truth = np.full(1, 1e-4)
    low = losses.qlike(np.array([0.5e-4]), truth)[0]
    high = losses.qlike(np.array([2.0e-4]), truth)[0]
    assert low > high


def test_mse_variance_and_volatility_disagree_about_units():
    forecast = np.array([1e-4])
    proxy = np.array([4e-4])
    assert losses.mse_variance(forecast, proxy)[0] == pytest.approx((3e-4) ** 2)
    assert losses.mse_volatility(forecast, proxy)[0] == pytest.approx((0.01 - 0.02) ** 2)


def test_tick_loss_is_minimised_at_the_true_quantile():
    """The defining property, and the one the sign of the coefficient breaks.

    Written to be robust to sampling noise: rather than demanding that a grid search
    land on the exact sample quantile - the loss is nearly flat near its minimum, so
    the argmin wanders - it asserts that the true quantile achieves the lowest mean
    loss over a wide grid, and that moving 40% in either direction costs more.
    """
    rng = np.random.default_rng(52)
    r = rng.standard_normal(200_000) * 0.01
    level = 0.99
    truth = float(-np.quantile(r, 1.0 - level))

    def mean_loss(candidate: float) -> float:
        return float(np.mean(losses.tick_loss(r, np.full(r.size, candidate), level)))

    grid = np.linspace(truth * 0.4, truth * 1.6, 121)
    curve = np.array([mean_loss(c) for c in grid])
    assert mean_loss(truth) <= curve.min() + 1e-7
    assert mean_loss(truth * 0.6) > mean_loss(truth)
    assert mean_loss(truth * 1.4) > mean_loss(truth)


def test_the_inverted_tick_loss_would_be_minimised_at_zero():
    """The bug this test file was written for, pinned so it cannot come back.

    Scoring with the coverage level instead of the tail probability makes the loss
    increase almost linearly in the reported VaR, so the best score goes to whatever
    reports the least risk. That is the opposite of the intent, and the resulting
    table still sorts and still names a winner.
    """
    rng = np.random.default_rng(520)
    r = rng.standard_normal(200_000) * 0.01
    truth = float(-np.quantile(r, 0.01))
    inverted = lambda v: float(np.mean((0.99 - (r < -v).astype(float)) * (r + v)))  # noqa: E731
    assert inverted(0.0) < inverted(truth)
    assert inverted(0.0) < inverted(0.5 * truth)


def test_a_failing_var_is_penalised_more_than_an_over_conservative_one():
    """The asymmetry the tick loss exists to encode.

    A VaR that is too conservative costs opportunity; a VaR that is too aggressive
    lets losses through. Squared error would treat them symmetrically.
    """
    rng = np.random.default_rng(53)
    r = rng.standard_normal(50_000) * 0.01
    truth = float(-np.quantile(r, 0.01))
    too_tight = float(np.mean(losses.tick_loss(r, np.full(r.size, truth * 0.6), 0.99)))
    too_loose = float(np.mean(losses.tick_loss(r, np.full(r.size, truth * 1.6), 0.99)))
    assert too_tight > too_loose


def test_tick_loss_scales_to_basis_points():
    r = np.array([-0.01, 0.01])
    v = np.array([0.02, 0.02])
    assert np.allclose(losses.tick_loss_scaled(r, v, 0.99), losses.tick_loss(r, v, 0.99) * 1e4)


# ---------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------
def test_diebold_mariano_detects_a_clear_winner():
    """A model that forecasts the proxy exactly beats one that does not."""
    rng = np.random.default_rng(54)
    proxy = rng.uniform(1e-4, 1e-3, 4_000)
    bad = losses.qlike(np.full(4_000, 1e-5), proxy)
    perfect = losses.qlike(proxy, proxy)
    statistic, p_value = losses.diebold_mariano(bad, perfect, hac_lags=5)
    assert statistic > 0  # the bad model loses more
    assert p_value < 0.01
    assert np.allclose(perfect, 0.0)


def test_diebold_mariano_is_symmetric_under_swap():
    rng = np.random.default_rng(55)
    a = rng.standard_normal(2_000)
    b = rng.standard_normal(2_000) + 0.01
    forward = losses.diebold_mariano(a, b, hac_lags=3)
    backward = losses.diebold_mariano(b, a, hac_lags=3)
    assert forward[0] == pytest.approx(-backward[0], rel=1e-9)
    assert forward[1] == pytest.approx(backward[1], rel=1e-9)


def test_newey_west_lags_shrink_the_statistic():
    """Serially correlated loss differences need the HAC variance, which is larger."""
    rng = np.random.default_rng(56)
    n = 3_000
    common = np.cumsum(rng.standard_normal(n)) * 0.01
    a = common + rng.standard_normal(n) * 0.1
    b = common + rng.standard_normal(n) * 0.1 + 0.05
    naive = abs(losses.diebold_mariano(a, b, hac_lags=0)[0])
    hac = abs(losses.diebold_mariano(a, b, hac_lags=30)[0])
    assert hac < naive


def test_diebold_mariano_needs_a_few_observations():
    assert not np.isfinite(losses.diebold_mariano(np.array([1.0]), np.array([2.0]))[0])


# ---------------------------------------------------------------------------
# Mincer-Zarnowitz
# ---------------------------------------------------------------------------
def test_mincer_zarnowitz_recovers_the_coefficients_of_a_perfect_forecast():
    """Point estimates on an exact forecast, and why its p-value is not usable.

    With no residual variance the Wald statistic is ``0 / 0`` in effect: any
    deviation, however small in economic terms, is infinitely significant. The
    coefficients are the meaningful output here, and the joint p-value is only
    interpretable when the forecast has genuine error, which the next test covers.
    """
    rng = np.random.default_rng(57)
    truth = np.exp(rng.normal(-9.0, 0.7, 2_000))
    result = evaluate.mincer_zarnowitz(truth, truth)
    assert result.slope == pytest.approx(1.0, abs=1e-6)
    assert result.intercept == pytest.approx(0.0, abs=1e-9)
    assert result.r_squared == pytest.approx(1.0, abs=1e-9)
    assert result.observations == 2_000


def test_mincer_zarnowitz_detects_a_scaling_bias():
    """A forecast that is wrong by a known factor gives a predictable slope."""
    rng = np.random.default_rng(570)
    truth = np.exp(rng.normal(-9.0, 0.7, 4_000))
    result = evaluate.mincer_zarnowitz(truth**1.3, truth)
    assert result.slope == pytest.approx(1.0 / 1.3, rel=0.05)
    assert not result.calibrated


def test_mincer_zarnowitz_slope_measures_attenuation_from_forecast_noise():
    """Noise in the regressor attenuates the slope, and the test says so.

    This is the reason the study reports ``mz_slope`` next to ``mz_calibrated``
    rather than instead of it: with thousands of observations the joint Wald test
    rejects almost any deviation, including a slope of 0.99, so the pass/fail flag
    is a near-constant and the coefficient is the informative part.
    """
    rng = np.random.default_rng(571)
    truth = np.exp(rng.normal(-9.0, 0.7, 4_000))
    noisy = truth * np.exp(rng.normal(0.0, 0.35, 4_000))
    result = evaluate.mincer_zarnowitz(noisy, truth)
    assert 0.5 < result.slope < 0.95
    assert result.r_squared > 0.5


def test_mincer_zarnowitz_rejects_a_biased_forecast():
    rng = np.random.default_rng(58)
    truth = np.exp(rng.normal(-9.0, 0.7, 2_000))
    result = evaluate.mincer_zarnowitz(truth * 2.0, truth)
    assert result.slope == pytest.approx(1.0, abs=0.02)
    assert not result.calibrated  # intercept is far from zero


def test_mincer_zarnowitz_rejects_a_compressed_forecast():
    rng = np.random.default_rng(59)
    truth = np.exp(rng.normal(-9.0, 1.0, 2_000))
    centre = truth.mean()
    result = evaluate.mincer_zarnowitz(centre + 0.3 * (truth - centre), truth)
    assert result.slope == pytest.approx(1.0 / 0.3, rel=0.05)
    assert not result.calibrated


def test_mincer_zarnowitz_joint_test_is_a_chi_square_two():
    rng = np.random.default_rng(60)
    truth = np.exp(rng.normal(-9.0, 0.7, 500))
    result = evaluate.mincer_zarnowitz(truth * np.exp(rng.normal(0, 0.05, 500)), truth)
    assert result.joint_statistic >= 0.0
    assert 0.0 <= result.joint_p_value <= 1.0
    assert result.calibrated == (result.joint_p_value > 0.05)


# ---------------------------------------------------------------------------
# Score tables
# ---------------------------------------------------------------------------
def test_volatility_scores_ranks_by_qlike_and_reports_both_targets(returns, realised, features):
    r = returns.to_numpy(dtype=float)
    index = returns.index
    forecasts = pd.DataFrame(
        {
            "flat": np.full(len(index), float(np.var(r))),
            "too_small": np.full(len(index), float(np.var(r)) / 4.0),
        },
        index=index,
    )
    table = evaluate.volatility_scores(forecasts, realised, latent=features["rv_next"])
    assert table["model"].iloc[0] == "flat"
    assert table["qlike"].iloc[0] < table["qlike"].iloc[1]
    assert "qlike_latent" in table.columns
    assert table["observations"].min() > 0


def test_volatility_scores_omits_the_latent_columns_without_a_latent_series(returns, realised):
    r = returns.to_numpy(dtype=float)
    forecasts = pd.DataFrame({"flat": np.full(len(returns), float(np.var(r)))}, index=returns.index)
    table = evaluate.volatility_scores(forecasts, realised)
    assert "qlike_latent" not in table.columns
    assert "mz_slope" in table.columns


def test_pairwise_diebold_mariano_is_a_square_symmetric_matrix(returns, realised):
    r = returns.to_numpy(dtype=float)
    index = returns.index
    forecasts = pd.DataFrame(
        {
            "a": np.full(len(index), float(np.var(r))),
            "b": np.full(len(index), float(np.var(r)) * 1.5),
            "c": np.full(len(index), float(np.var(r)) * 0.7),
        },
        index=index,
    )
    table = evaluate.pairwise_diebold_mariano(forecasts, realised, hac_lags=5)
    assert list(table.columns) == ["a", "b", "c"]
    assert np.allclose(table.to_numpy(), table.to_numpy().T, equal_nan=True)
    # A model against itself is the null of no difference with zero variance, so the
    # statistic is undefined. Reporting NaN says that; reporting 1.0 would claim the
    # diagonal was tested and found insignificant.
    assert np.isnan(np.diag(table.to_numpy())).all()
    assert table.notna().sum().sum() == 6


def test_qlike_by_regime_reports_one_column_per_regime_plus_calm(returns, realised):
    r = returns.to_numpy(dtype=float)
    index = returns.index
    forecasts = pd.DataFrame({"flat": np.full(len(index), float(np.var(r)))}, index=index)
    table = evaluate.qlike_by_regime(forecasts, realised)
    assert len(table) == 1  # one row per model
    for name in cfg.STRESS_WINDOWS:
        assert name in table.columns
    assert "calm (all other days)" in table.columns


# ---------------------------------------------------------------------------
# VaR-specific tables
# ---------------------------------------------------------------------------
def _suite(returns: pd.Series, level: float = 0.99, horizon: int = 1) -> var.VaRSuite:
    rng = np.random.default_rng(61)
    sigma = np.full(len(returns), 0.01)
    r = returns.to_numpy(dtype=float)
    suite = var.build_suite(
        level,
        horizon,
        returns,
        {"gjr": pd.Series(sigma, index=returns.index)},
        None,
        None,
        rng,
        paths=200,
    )
    _ = r
    return suite


def test_tick_loss_table_covers_every_method(returns):
    suite = _suite(returns)
    table = evaluate.tick_loss_table(suite, returns)
    assert set(table["method"]) == set(suite.forecasts)
    assert table["tick_loss_bp"].is_monotonic_increasing
    assert (table["mean_var_bp"] > 0).all()


def test_tick_loss_table_uses_the_suite_horizon(returns):
    one = evaluate.tick_loss_table(_suite(returns, horizon=1), returns)
    ten = evaluate.tick_loss_table(_suite(returns, horizon=10), returns)
    # The ten-day mean VaR is larger and there are fewer usable observations.
    assert ten["mean_var_bp"].max() > one["mean_var_bp"].max()
    assert ten["observations"].max() < one["observations"].max()


def test_es_backtest_table_reports_a_z_score_and_a_p_value(returns):
    suite = _suite(returns)
    table = evaluate.es_backtest_table(suite, returns)
    assert not table.empty
    for column in ("method", "breaches", "z1", "z1_p", "es_adequate"):
        assert column in table.columns
    assert table["z1_p"].between(0.0, 1.0).all()


def test_feature_importance_handles_an_unfitted_model():
    from marketrisk import volmodels

    table = evaluate.feature_importance(volmodels.GBM())
    assert list(table.columns) == ["feature", "gain"]
    assert table.empty
