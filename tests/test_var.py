"""Tests for the VaR and ES routes.

The convention this file defends is stated once, in the module docstring it is
testing: returns are log returns, and a VaR is reported as a **positive loss
magnitude**, so a breach is ``r_t < -VaR_t``. Every downstream table, plot and
backtest inherits that convention, and getting the sign wrong is the kind of error
that produces a 0% breach rate and looks like an extremely good model.

The other thing worth testing here is the horizon convention. ``horizon_returns``
is forward-looking by design, and the difference between forward and backward is
invisible in every summary statistic except the one that matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketrisk import config as cfg
from marketrisk import var, volmodels


# ---------------------------------------------------------------------------
# Quantiles and expected shortfall
# ---------------------------------------------------------------------------
def test_normal_quantile_matches_the_textbook_value():
    assert var.normal_quantile(0.99) == pytest.approx(2.326347874, rel=1e-8)
    assert var.normal_quantile(0.95) == pytest.approx(1.644853627, rel=1e-8)


def test_student_t_quantile_is_unit_variance():
    """The ``sqrt((dof-2)/dof)`` factor, whose absence inflates the 99% quantile.

    For a standard (non-unit-variance) t at ``dof = 5`` the 99% quantile is about
    3.36; scaled to unit variance it is about 2.60. Dropping the factor makes every
    parametric-t VaR about 29% too conservative.
    """
    from scipy import stats

    for dof in (3.0, 5.0, 12.0):
        scaled = var.student_t_quantile(0.99, dof)
        unscaled = float(-stats.t.ppf(0.01, dof))
        assert scaled == pytest.approx(unscaled * np.sqrt((dof - 2.0) / dof), rel=1e-9)
        assert scaled < unscaled


def test_student_t_quantile_converges_to_the_normal():
    assert var.student_t_quantile(0.99, 500.0) == pytest.approx(var.normal_quantile(0.99), rel=1e-9)
    assert var.student_t_quantile(0.99, np.inf) == pytest.approx(var.normal_quantile(0.99), rel=1e-9)


def test_fat_tails_give_a_larger_quantile_at_the_same_scale():
    assert var.student_t_quantile(0.99, 4.0) > var.student_t_quantile(0.99, 30.0)
    assert var.student_t_quantile(0.99, 30.0) > var.normal_quantile(0.99) * 0.99


def test_expected_shortfall_exceeds_var_everywhere():
    for level in (0.90, 0.95, 0.975, 0.99, 0.999):
        for dof in (3.0, 5.0, 8.0, 20.0, np.inf):
            q = var.student_t_quantile(level, dof)
            e = var.student_t_es(level, dof)
            assert e > q > 0.0, (level, dof)


def test_normal_es_matches_its_closed_form():
    from scipy import stats

    for level in (0.95, 0.99):
        z = stats.norm.ppf(1.0 - level)
        assert var.normal_es(level) == pytest.approx(stats.norm.pdf(z) / (1.0 - level), rel=1e-12)


def test_tail_dof_recovery():
    """``fit_tail_dof`` recovers the truth on t draws and rejects the normal."""
    rng = np.random.default_rng(31)
    raw = rng.standard_t(5.0, size=20_000)
    standardised = raw / np.sqrt(5.0 / (5.0 - 2.0))
    assert var.fit_tail_dof(standardised) == pytest.approx(5.0, abs=1.5)

    normal = rng.standard_normal(20_000)
    assert var.fit_tail_dof(normal) > 20.0


# ---------------------------------------------------------------------------
# Historical simulation
# ---------------------------------------------------------------------------
def test_historical_var_is_the_empirical_quantile():
    rng = np.random.default_rng(32)
    r = rng.standard_normal(2_000) * 0.01
    v, e = var.historical_var(r, 0.99, window=250, horizon=1)
    t = 600
    expected = -float(np.quantile(r[t - 250 : t], 0.01))
    assert v[t] == pytest.approx(expected, rel=1e-9)
    assert e[t] > v[t]


def test_historical_var_leaves_the_burn_in_empty():
    rng = np.random.default_rng(33)
    r = rng.standard_normal(500) * 0.01
    v, e = var.historical_var(r, 0.99, window=250, horizon=1)
    assert np.isnan(v[:250]).all()
    assert np.isfinite(v[250:]).all()
    assert np.isnan(e[:250]).all()
    assert np.isfinite(e[250:]).all()


def test_historical_var_shifts_the_burn_in_with_the_horizon():
    """An h-day empirical quantile needs h more observations before it exists."""
    rng = np.random.default_rng(330)
    r = rng.standard_normal(800) * 0.01
    one, _ = var.historical_var(r, 0.99, window=250, horizon=1)
    ten, _ = var.historical_var(r, 0.99, window=250, horizon=10)
    first_one = int(np.argmax(np.isfinite(one)))
    first_ten = int(np.argmax(np.isfinite(ten)))
    assert first_ten - first_one == 9


def test_age_weighted_var_with_no_decay_agrees_with_plain_historical():
    """A decay of one makes BRW the flat-window estimator up to its interpolation.

    The two are not bit-identical: ``historical_var`` reads an order statistic via
    ``np.quantile``, while BRW locates the same point by walking the cumulative
    weight, and the two interpolate differently between adjacent order statistics.
    What must hold is that they agree to well inside the spacing of those
    statistics, which is what makes the comparison between them a comparison of
    *weighting* rather than of estimator.
    """
    rng = np.random.default_rng(34)
    r = rng.standard_normal(1_200) * 0.01
    plain, _ = var.historical_var(r, 0.99, window=250, horizon=1)
    weighted, _ = var.age_weighted_var(r, 0.99, window=250, decay=1.0, horizon=1)
    both = np.isfinite(plain) & np.isfinite(weighted) & (plain > 0)
    assert both.sum() > 800
    relative = np.abs(weighted[both] - plain[both]) / plain[both]
    assert float(np.median(relative)) < 0.05
    assert float(np.mean(relative)) < 0.15


def test_age_weighting_reacts_faster_than_a_flat_window():
    """A large shock entering the window moves the BRW quantile more than the flat one."""
    rng = np.random.default_rng(340)
    base = rng.standard_normal(640) * 0.01
    r = np.concatenate([base, [-0.09]])
    plain, _ = var.historical_var(r, 0.99, window=250, horizon=1)
    weighted, _ = var.age_weighted_var(r, 0.99, window=250, decay=0.99, horizon=1)
    assert weighted[-1] > plain[-1]


# ---------------------------------------------------------------------------
# Parametric
# ---------------------------------------------------------------------------
def test_parametric_var_scales_with_sigma_and_horizon():
    sigma = np.array([0.01, 0.02])
    v, e = var.parametric_var(sigma, 0.99, "normal", horizon=1)
    assert np.allclose(v, sigma * var.normal_quantile(0.99))
    assert np.allclose(e, sigma * var.normal_es(0.99))

    v10, _ = var.parametric_var(sigma, 0.99, "normal", horizon=10)
    assert np.allclose(v10, v * np.sqrt(10.0))


def test_parametric_var_rejects_an_unknown_distribution():
    with pytest.raises(ValueError, match="unknown distribution"):
        var.parametric_var(np.array([0.01]), 0.99, "cauchy")


def test_fat_tailed_parametric_beats_gaussian_at_the_same_scale():
    sigma = np.array([0.01])
    normal_v, _ = var.parametric_var(sigma, 0.99, "normal")
    t_v, _ = var.parametric_var(sigma, 0.99, "t", dof=5.0)
    assert t_v[0] > normal_v[0]


# ---------------------------------------------------------------------------
# Filtered historical simulation
# ---------------------------------------------------------------------------
def test_fhs_is_the_conditional_scale_times_a_residual_quantile():
    rng = np.random.default_rng(35)
    n = 1_500
    sigma = np.full(n, 0.01)
    r = rng.standard_normal(n) * sigma
    v, e = var.filtered_historical_var(r, sigma, 0.99, window=250, horizon=1)
    t = 600
    residual = r[t - 250 : t] / sigma[t - 250 : t]
    expected = -float(np.quantile(residual, 0.01)) * sigma[t]
    assert v[t] == pytest.approx(expected, rel=1e-9)
    assert e[t] > v[t]


def test_fhs_evt_tail_is_not_systematically_below_the_empirical_quantile():
    """The regression test for the moments estimator of the GPD shape.

    With the wrong formula the shape comes out negative on an approximately
    exponential tail, which implies a *bounded* tail, and the EVT VaR lands about
    20% below the empirical 99% quantile - which showed up as a breach rate of
    2.8% against a nominal 1%. A correct fit puts the two in the same place.
    """
    rng = np.random.default_rng(36)
    n = 5_000
    sigma = np.full(n, 0.01)
    r = rng.standard_t(6.0, n) * sigma / np.sqrt(6.0 / 4.0)

    empirical, _ = var.filtered_historical_var(r, sigma, 0.99, window=750, horizon=1, tail="empirical")
    evt, _ = var.filtered_historical_var(r, sigma, 0.99, window=750, horizon=1, tail="evt")
    both = np.isfinite(empirical) & np.isfinite(evt)
    ratio = np.median(evt[both] / empirical[both])
    assert 0.85 < ratio < 1.35


def test_gpd_shape_is_zero_on_an_exponential_tail():
    """Directly: the shape parameter of exponential exceedances is zero.

    The wrong moments formula returns -0.25 here. The sign is what makes it
    dangerous - a negative shape describes a tail with a finite upper bound, and
    the quantile it implies is far too small.
    """

    def implied_shape(sample: np.ndarray, threshold_quantile: float = 0.90) -> float:
        loss = -sample
        threshold = float(np.quantile(loss, threshold_quantile))
        exceedances = loss[loss > threshold] - threshold
        return 0.5 * (1.0 - exceedances.mean() ** 2 / exceedances.var(ddof=1))

    rng = np.random.default_rng(37)
    exponential = -rng.exponential(1.0, 40_000)
    shape = implied_shape(exponential)
    assert abs(shape) < 0.06, shape
    # The natural-looking but wrong form, for contrast.
    loss = -exponential
    threshold = float(np.quantile(loss, 0.90))
    exceedances = loss[loss > threshold] - threshold
    wrong = 0.5 * (exceedances.mean() ** 2 / np.mean(exceedances**2) - 1.0)
    assert wrong < -0.15


def test_evt_degrades_to_the_empirical_quantile_when_the_tail_is_too_thin():
    """With too few exceedances to fit, the EVT path must become the empirical one.

    A quiet 250-day window does not supply eight observations above the threshold,
    and the fit is then abandoned. What it is abandoned *to* matters: the fallback
    has to return the same object on the same scale as the fitted path, which here
    means the negative residual quantile.
    """
    rng = np.random.default_rng(38)
    sample = rng.standard_normal(250)

    threshold = 0.975  # ~6 exceedances in 250 points, so the fit is skipped
    loss = -sample
    n_excess = int((loss > float(np.quantile(loss, threshold))).sum())
    assert n_excess < 8

    quantile_z, es_z = var._gpd_tail(sample, 0.99, threshold)
    assert quantile_z < 0.0 and es_z < 0.0

    # Bit-for-bit what the empirical tail produces, so the degradation is exact and
    # not merely in the right direction.
    empirical = float(np.quantile(sample, 0.01))
    assert quantile_z == pytest.approx(empirical, abs=1e-12)
    assert es_z == pytest.approx(float(sample[sample <= empirical].mean()), abs=1e-12)


def test_the_evt_fallback_would_report_a_zero_var_if_the_sign_flipped():
    """The consequence of getting that sign wrong, pinned so it cannot come back.

    The caller negates the residual quantile and multiplies by ``sigma_t``, then
    clamps at zero. Feed it a positive quantile and the clamp does not complain - it
    returns exactly zero VaR and zero ES, i.e. "no risk today", on every window that
    takes the fallback branch. The damage surfaces somewhere else entirely: a zero ES
    becomes the denominator of the Acerbi-Szekely statistic, which came out at
    2.1e8, and a zero VaR silently counts every down day as a breach.
    """
    rng = np.random.default_rng(39)
    sample = rng.standard_normal(250)
    quantile_z, es_z = var._gpd_tail(sample, 0.99, 0.975)
    sigma = 0.01

    # As returned: the caller's `max(-q * sigma, 0)` is a positive loss magnitude.
    assert max(-quantile_z * sigma, 0.0) > 0.0
    assert max(-es_z * sigma, 0.0) > 0.0

    # What the caller would have made of the old, sign-flipped fallback.
    flipped_z, flipped_es = -quantile_z, -es_z
    assert max(-flipped_z * sigma, 0.0) == 0.0
    assert max(-flipped_es * sigma, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def _fitted_garch(returns: np.ndarray, features: pd.DataFrame) -> volmodels.GARCH:
    model = volmodels.GARCH(kind="gjr")
    model.fit(returns, features, returns.size)
    return model


def test_monte_carlo_one_day_matches_the_closed_form(returns, features, rng):
    """At h = 1 the recursion has one step, so MC must agree with the parametric leg."""
    r = returns.to_numpy(dtype=float)
    garch = _fitted_garch(r, features)
    p = garch.params
    persistence = p["alpha"] + p["beta"] + 0.5 * p["gamma"]
    omega = (1.0 - persistence) * p["unconditional_variance"]
    from marketrisk import kernels

    path = kernels.garch_variance_path(omega, p["alpha"], p["gamma"], p["beta"], r, float(np.var(r)))

    mc, _ = var.monte_carlo_var(garch, r, path, 0.99, rng, paths=40_000, horizon=1)
    analytic, _ = var.parametric_var(np.sqrt(path), 0.99, "t", dof=p["dof"], horizon=1)
    both = np.isfinite(mc) & np.isfinite(analytic)
    ratio = np.median(mc[both] / analytic[both])
    assert 0.9 < ratio < 1.15, ratio


def test_monte_carlo_multi_day_is_below_the_square_root_rule_when_vol_is_high(features):
    """The headline mechanism: a mean-reverting recursion grows slower than sqrt(h).

    When the current variance sits above its long-run level the recursion pulls it
    back down, so the h-day variance is less than h times the one-day variance and
    the square-root rule overstates the risk. The opposite happens in a calm
    period, which is why the study measures the gap rather than assuming its sign.
    """
    rng = np.random.default_rng(38)
    truth = {"alpha": 0.06, "gamma": 0.09, "beta": 0.86}
    persistence = truth["alpha"] + truth["beta"] + 0.5 * truth["gamma"]
    sigma2_long = 0.01**2
    omega = (1.0 - persistence) * sigma2_long

    from marketrisk import kernels

    z = rng.standard_t(6.0, 20_000) / np.sqrt(6.0 / 4.0)
    path = kernels.garch_simulate_path(omega, truth["alpha"], truth["gamma"], truth["beta"], z, sigma2_long)
    r = np.sqrt(path) * z
    model = volmodels.GARCH(kind="gjr")
    model.fit(r, features.iloc[: r.size], r.size)

    n = r.size
    elevated = np.full(n, 16.0 * sigma2_long)  # four times the long-run volatility
    one_day, _ = var.monte_carlo_var(model, r, elevated, 0.99, rng, paths=20_000, horizon=1)
    ten_day, _ = var.monte_carlo_var(model, r, elevated, 0.99, rng, paths=20_000, horizon=10)
    both = np.isfinite(one_day) & np.isfinite(ten_day) & (one_day > 0)
    gap = var.horizon_gap(one_day, ten_day, 10)
    finite = gap[np.isfinite(gap) & both]
    # The gap is a signed fraction, so "below the scaling rule" is a negative number.
    assert np.median(finite) < 0.0


def test_monte_carlo_expects_shortfall_above_var(returns, features, rng):
    r = returns.to_numpy(dtype=float)
    garch = _fitted_garch(r, features)
    from marketrisk import kernels

    path = kernels.garch_variance_path(
        (1.0 - garch.params["alpha"] - garch.params["beta"] - 0.5 * garch.params["gamma"])
        * garch.params["unconditional_variance"],
        garch.params["alpha"],
        garch.params["gamma"],
        garch.params["beta"],
        r,
        float(np.var(r)),
    )
    v, e = var.monte_carlo_var(garch, r, path, 0.99, rng, paths=2_000, horizon=5)
    both = np.isfinite(v) & np.isfinite(e)
    assert (e[both] > v[both]).all()


def test_monte_carlo_is_reproducible_for_a_fixed_seed(returns, features):
    r = returns.to_numpy(dtype=float)
    garch = _fitted_garch(r, features)
    from marketrisk import kernels

    path = kernels.garch_variance_path(1e-6, 0.06, 0.09, 0.86, r, float(np.var(r)))
    a, _ = var.monte_carlo_var(garch, r, path, 0.99, np.random.default_rng(5), paths=2_000, horizon=3)
    b, _ = var.monte_carlo_var(garch, r, path, 0.99, np.random.default_rng(5), paths=2_000, horizon=3)
    assert np.allclose(a, b, equal_nan=True)


# ---------------------------------------------------------------------------
# Horizon convention
# ---------------------------------------------------------------------------
def test_horizon_returns_looks_forward():
    """The convention that a 10-day VaR issued at ``t`` refers to ``t..t+9``."""
    r = pd.Series(np.arange(10.0) * 0.001)
    forward = var.horizon_returns(r, 3)
    assert forward.iloc[0] == pytest.approx(r.iloc[0] + r.iloc[1] + r.iloc[2])
    assert forward.iloc[4] == pytest.approx(r.iloc[4] + r.iloc[5] + r.iloc[6])
    # The last h-1 entries are undefined: there is not yet a full window ahead.
    assert forward.iloc[-2:].isna().all()
    assert np.isfinite(forward.iloc[-3])


def test_horizon_returns_is_a_no_op_at_one_day():
    r = pd.Series(np.arange(6.0))
    assert var.horizon_returns(r, 1).equals(r)


def test_horizon_gap_is_zero_under_exact_square_root_scaling():
    """``horizon_gap`` returns the *fractional* deviation, not the ratio."""
    one = np.array([0.01, 0.02])
    assert np.allclose(var.horizon_gap(one, one * np.sqrt(10.0), 10), 0.0)
    assert np.allclose(var.horizon_gap(one, one * 1.5 * np.sqrt(10.0), 10), 0.5)


def test_horizon_gap_is_not_defined_on_a_zero_reference():
    gap = var.horizon_gap(np.array([0.0]), np.array([0.01]), 10)
    assert not np.isfinite(gap).any()


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------
def test_build_suite_always_includes_monte_carlo(returns, features, rng):
    """A path-count knob, not a presence flag.

    An earlier version dropped the Monte-Carlo leg under a speed flag. It was the
    only method whose multi-day VaR is not a scaled one-day VaR, so the report's
    horizon-scaling section came out empty while everything else looked fine.
    """
    r = returns.to_numpy(dtype=float)
    garch = _fitted_garch(r, features)
    sigma = np.sqrt(garch.forecast(r, features, 0, r.size))
    sigma2 = sigma**2

    small = var.build_suite(
        0.99,
        1,
        returns,
        {"gjr": pd.Series(sigma, index=returns.index)},
        garch,
        pd.Series(sigma2, index=returns.index),
        rng,
        paths=500,
    )
    assert "monte_carlo" in small.forecasts

    big = var.build_suite(
        0.99,
        1,
        returns,
        {"gjr": pd.Series(sigma, index=returns.index)},
        garch,
        pd.Series(sigma2, index=returns.index),
        np.random.default_rng(cfg.SEED),
        paths=20_000,
    )
    assert "monte_carlo" in big.forecasts
    both = np.isfinite(small.forecasts["monte_carlo"].var) & np.isfinite(big.forecasts["monte_carlo"].var)
    # A 500-path quantile carries a standard error of a few percent of the tail
    # width, so this is a check that the two runs estimate the same quantity and
    # not a check that they agree to the last digit.
    assert (
        np.corrcoef(
            small.forecasts["monte_carlo"].var[both].to_numpy(),
            big.forecasts["monte_carlo"].var[both].to_numpy(),
        )[0, 1]
        > 0.95
    )


def test_build_suite_omits_monte_carlo_only_without_a_filter(returns, features):
    r = returns.to_numpy(dtype=float)
    sigma = np.full(r.size, 0.01)
    suite = var.build_suite(
        0.99,
        1,
        returns,
        {"gjr": pd.Series(sigma, index=returns.index)},
        None,
        None,
        np.random.default_rng(1),
        paths=500,
    )
    assert "monte_carlo" not in suite.forecasts
    assert {"hs_250", "hs_brw", "parametric_normal_gjr", "parametric_t_gjr", "fhs_gjr", "fhs_evt_gjr"} <= set(
        suite.forecasts
    )


def test_the_suite_matches_the_declared_roster(returns, features, rng):
    """The seven routes that run are the seven the roster names, in that order.

    The order is part of it: the report, the tables and the README all render routes in
    suite order, so a route that moves is a route that appears somewhere else in every
    document - which is the kind of change nobody notices and everybody misreads.
    """
    r = returns.to_numpy(dtype=float)
    garch = _fitted_garch(r, features)
    sigma = np.sqrt(garch.forecast(r, features, 0, r.size))
    suite = var.build_suite(
        0.99,
        1,
        returns,
        {"gjr": pd.Series(sigma, index=returns.index)},
        garch,
        pd.Series(sigma**2, index=returns.index),
        rng,
        paths=500,
    )
    assert suite.names() == list(cfg.VAR_METHOD_NAMES)


def test_suite_frames_all_share_the_index(returns, features, rng):
    r = returns.to_numpy(dtype=float)
    garch = _fitted_garch(r, features)
    sigma = np.sqrt(garch.forecast(r, features, 0, r.size))
    suite = var.build_suite(
        0.99,
        1,
        returns,
        {"gjr": pd.Series(sigma, index=returns.index)},
        garch,
        pd.Series(sigma**2, index=returns.index),
        rng,
        paths=500,
    )
    frame = suite.var_frame()
    assert frame.index.equals(returns.index)
    assert list(frame.columns) == suite.names()
    assert suite.es_frame().shape == frame.shape


def test_var_and_es_are_strictly_positive_wherever_they_are_defined(returns, features, rng):
    """The sign convention, checked on every route at once - and strictly.

    ``>= 0`` is the assertion this test used to make and it was too weak to be worth
    anything: zero passed. A VaR of zero is not a border case, it is the absence of a
    risk number, and it slipped through here while a tail-fitting branch returned its
    quantile with the wrong sign and the caller's clamp turned it into one. Every
    route here is a positive scale times a positive quantile, so a defined value is
    strictly positive, and the strict form is the one that catches a sign error.
    """
    r = returns.to_numpy(dtype=float)
    garch = _fitted_garch(r, features)
    sigma = np.sqrt(garch.forecast(r, features, 0, r.size))
    suite = var.build_suite(
        0.99,
        1,
        returns,
        {"gjr": pd.Series(sigma, index=returns.index)},
        garch,
        pd.Series(sigma**2, index=returns.index),
        rng,
        paths=500,
    )
    for name, forecast in suite.forecasts.items():
        values = forecast.var.dropna()
        assert (values > 0).all(), f"{name} produced a non-positive VaR"
        es_values = forecast.es.dropna()
        assert (es_values > 0).all(), f"{name} produced a non-positive ES"
        # And the ES describes a worse outcome than the VaR it accompanies, which is
        # the one ordering the two numbers must always satisfy.
        both = forecast.var.notna() & forecast.es.notna()
        assert (forecast.es[both] >= forecast.var[both]).all(), f"{name} has ES below VaR"


def test_multi_day_var_exceeds_one_day_var(returns, features, rng):
    """Every route at ten days is a larger number than the same route at one day.

    Asserted on the *mean* level rather than day by day. For the routes that scale
    explicitly the multiple is exactly sqrt(10) and the comparison would hold daily,
    but the empirical routes read a ten-day quantile off overlapping windows, and
    that sample is not a monotonically rescaled version of the daily one - so a
    handful of individual days can go either way while the level clearly rises.
    """
    r = returns.to_numpy(dtype=float)
    garch = _fitted_garch(r, features)
    sigma = np.sqrt(garch.forecast(r, features, 0, r.size))
    kwargs = dict(
        sigma_paths={"gjr": pd.Series(sigma, index=returns.index)},
        garch_model=garch,
        sigma2_garch=pd.Series(sigma**2, index=returns.index),
    )
    one = var.build_suite(0.99, 1, returns, rng=np.random.default_rng(3), paths=2_000, **kwargs)
    ten = var.build_suite(0.99, 10, returns, rng=np.random.default_rng(3), paths=2_000, **kwargs)
    for name in one.forecasts:
        a = one.forecasts[name].var.dropna()
        b = ten.forecasts[name].var.dropna()
        both = a.index.intersection(b.index)
        assert b.loc[both].mean() > 2.0 * a.loc[both].mean(), name
        assert (b.loc[both] > a.loc[both]).mean() > 0.9, name
