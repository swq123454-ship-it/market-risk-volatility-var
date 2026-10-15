"""Tests for the synthetic market.

The generator is not a toy. Every downstream conclusion is conditioned on it, so
the properties it claims are asserted here directly rather than being inferred
from the fact that the pipeline runs: the crisis profile peaks and decays, the
Brownian bridge preserves the daily return, the range measures satisfy the OHLC
ordering, jumps are systematic rather than per-asset, and the whole thing is
bit-for-bit reproducible from a seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketrisk import config as cfg
from marketrisk import generator


# ---------------------------------------------------------------------------
# Crisis profile
# ---------------------------------------------------------------------------
def test_crisis_multiplier_is_one_outside_the_schedule():
    """A day outside every declared window - ramp and decay included - is unstressed.

    2014 is the safest such window in this calendar: the euro sovereign-debt decay
    of early 2012 is long finished and the China/oil ramp does not begin until 40
    days before its July 2015 onset.
    """
    dates = pd.bdate_range("2014-01-02", "2014-12-31")
    mult = generator.crisis_multiplier(dates)
    assert np.allclose(mult, 1.0)


def test_crisis_multiplier_peaks_at_the_scheduled_level():
    """Each window reaches its nominal multiplier, deep inside the hold."""
    for name, start, end, level in cfg.CRISIS_SCHEDULE:
        dates = pd.bdate_range(start, end)
        peak = float(generator.crisis_multiplier(dates).max())
        assert peak == pytest.approx(level, rel=1e-9), name


def test_crisis_multiplier_ramps_rather_than_steps():
    """The window climbs into its level over ``CRISIS_RAMP_DAYS`` before the onset.

    A step in the variance *target* does not produce a step in realised variance,
    because the recursion only pulls a fraction of the way towards its target each
    day. Asserting the ramp is the cheap way to keep someone from "simplifying" it
    back to a step and silently halving every crisis in the study.
    """
    _, start, _, level = cfg.CRISIS_SCHEDULE[0]
    onset = pd.Timestamp(start)
    ramp_start = onset - pd.Timedelta(days=cfg.CRISIS_RAMP_DAYS)
    dates = pd.bdate_range(ramp_start - pd.Timedelta(days=10), onset + pd.Timedelta(days=60))
    mult = generator.crisis_multiplier(dates)

    assert mult[0] == pytest.approx(1.0)  # still ahead of the ramp
    ramp = mult[dates < onset]
    assert np.all(np.diff(ramp) >= -1e-12)
    assert ramp[0] < 1.5
    assert float(mult[dates == onset][0]) == pytest.approx(level, rel=1e-9)


def test_crisis_multiplier_decays_after_the_window():
    """Stress bleeds out rather than snapping back to one on the closing date.

    The window has to stop before the *next* scheduled crisis begins its ramp,
    otherwise the assertion is measuring two overlapping episodes.
    """
    _, _, end, level = cfg.CRISIS_SCHEDULE[0]
    close = pd.Timestamp(end)
    dates = pd.bdate_range(close + pd.Timedelta(days=1), close + pd.Timedelta(days=600))
    mult = generator.crisis_multiplier(dates)
    assert mult[0] > 1.5
    assert mult[0] < level
    assert np.all(np.diff(mult) <= 1e-12)
    # 600 days is 24 e-folding times at the configured 25-day decay.
    assert mult[-1] == pytest.approx(1.0, abs=1e-6)


def test_crisis_multiplier_is_deterministic():
    dates = pd.bdate_range("2007-01-01", "2010-12-31")
    assert np.array_equal(generator.crisis_multiplier(dates), generator.crisis_multiplier(dates))


def test_stress_windows_are_more_stressed_than_a_calm_year():
    """The declared stress windows must actually be the volatile ones.

    This is the assertion that a previous calibration failed: with the original
    multiplier levels and a seven-month GFC window, the recursion never climbed
    far enough and ``STRESS_WINDOWS`` pointed at slices that were *quieter* than
    2017. Reporting those as stress slices would have been a false claim dressed
    as a chart.
    """
    dates = pd.bdate_range(cfg.DEFAULT_START, cfg.DEFAULT_END)
    mult = pd.Series(generator.crisis_multiplier(dates), index=dates)
    calm = mult["2017-01-01":"2017-12-31"].mean()
    for name, (start, end) in cfg.STRESS_WINDOWS.items():
        assert mult[start:end].mean() > 3.0 * calm, name


# ---------------------------------------------------------------------------
# Intraday bridge
# ---------------------------------------------------------------------------
def test_brownian_bridge_sums_to_the_daily_return():
    """``sum(r_i)`` is exactly the pinned daily diffusion return, by construction."""
    rng = np.random.default_rng(0)
    n, bars = 300, cfg.INTRADAY_BARS
    diffusion = rng.standard_normal(n) * 0.01
    sigma2 = np.full(n, 0.01**2)
    bridge = rng.standard_normal((n, bars))
    bridge = (bridge - bridge.mean(axis=1, keepdims=True)) / np.sqrt((bars - 1.0) / bars)

    intraday = generator._intraday_from_daily(diffusion, sigma2, bridge, bars, drift=0.0)
    assert np.allclose(intraday.sum(axis=1), diffusion, atol=1e-14)


def test_brownian_bridge_bar_variance_matches_theory():
    """``Var(r_i)`` is ``sigma^2 / M`` up to the bridge's own sampling noise."""
    rng = np.random.default_rng(1)
    n, bars = 20_000, 78
    sigma = 0.012
    diffusion = rng.standard_normal(n) * sigma
    sigma2 = np.full(n, sigma**2)
    bridge = rng.standard_normal((n, bars))
    bridge = (bridge - bridge.mean(axis=1, keepdims=True)) / np.sqrt((bars - 1.0) / bars)

    intraday = generator._intraday_from_daily(diffusion, sigma2, bridge, bars, drift=0.0)
    realised = np.sum(intraday**2, axis=1)
    # RV is unbiased for sigma^2 by construction; the tolerance reflects the
    # 2.5% standard error of a single day, not a fitting fudge.
    assert realised.mean() / sigma**2 == pytest.approx(1.0, rel=0.02)


def test_realised_variance_tracks_true_variance_in_logs():
    """The estimator the study scores models against is not its own noise source.

    The correlation is measured in logs. On the level scale a handful of tiny-
    variance days dominate the ratio's standard deviation and the diagnostic reads
    about 0.71 for a generator whose diffusion part is exactly unbiased; in logs the
    same data reads 0.978, which is the number that means something.

    The mean ratio sits above one because ``true_variance`` is the ex-ante variance
    of the *diffusion* part and RV also contains the jump contribution. That gap is
    the reason the study reports a separate latent-target score rather than
    pretending the two targets are the same quantity.
    """
    out = generator.simulate_market(start="2005-01-03", end="2024-12-31", seed=cfg.SEED)
    path = out.assets[cfg.PRIMARY_ASSET]
    log_rv = np.log(path.realised_variance.to_numpy())
    log_var = np.log(path.true_variance.to_numpy())
    assert np.corrcoef(log_rv, log_var)[0, 1] > 0.93
    ratio = float(np.mean(np.exp(log_rv - log_var)))
    assert 1.0 <= ratio < 1.30
    # On the diffusion-only part the estimator is unbiased; assemble it by removing
    # the days the simulator actually placed a jump on.
    jump = np.asarray(path.jump)
    diffusion = np.flatnonzero(jump == 0.0)
    clean = float(np.mean(np.exp(log_rv[diffusion] - log_var[diffusion])))
    assert clean == pytest.approx(1.0, rel=0.10)


# ---------------------------------------------------------------------------
# OHLC and jumps
# ---------------------------------------------------------------------------
def test_ohlc_ordering_holds_on_every_bar(panel):
    """``low <= open, close <= high`` - an obvious invariant that was once broken.

    Without an explicit clamp the high of a session in which every bar falls comes
    out below the open, because the bridge path is measured from the session's own
    starting point and a monotone decline never revisits it.
    """
    raw = panel.raw
    assert (raw["low"] <= raw["open"] + 1e-9).all()
    assert (raw["open"] <= raw["high"] + 1e-9).all()
    assert (raw["low"] <= raw["close"] + 1e-9).all()
    assert (raw["close"] <= raw["high"] + 1e-9).all()
    assert (raw["low"] <= raw["high"] + 1e-9).all()


def test_returns_match_the_close_to_close_change(panel):
    raw = panel.raw
    implied = np.log(raw["close"]).diff().dropna()
    assert np.allclose(implied.to_numpy(), raw["ret"].iloc[1:].to_numpy(), atol=1e-12)


def test_jumps_are_sparse_and_shared():
    """Jumps are a market-wide event: they land on the same days for every asset.

    Every non-primary instrument carries the *common factor's* jump scaled by its
    loading. It does not roll its own Poisson process - the cost of getting that
    wrong was EUR/USD carrying a 23% jump share, which no FX desk would recognise -
    but it does inherit the systematic shock, because a crash is not a private
    event. So the invariant is not "only the index jumps"; it is "the jump days are
    the same days and the magnitudes follow the loadings".
    """
    out = generator.simulate_market(start="2005-01-03", end="2024-12-31", seed=cfg.SEED)
    primary = np.asarray(out.assets[cfg.PRIMARY_ASSET].jump)
    jump_days = np.flatnonzero(primary != 0.0)
    assert 0 < jump_days.size < 0.05 * primary.size

    for key, path in out.assets.items():
        jump = np.asarray(path.jump)
        assert np.array_equal(np.flatnonzero(jump != 0.0), jump_days), key
        if key == cfg.PRIMARY_ASSET:
            continue
        loading = float(cfg.ASSETS[key]["beta"])
        ratio = float(np.median(jump[jump_days] / primary[jump_days]))
        assert ratio == pytest.approx(loading, rel=1e-6), key


def test_equity_correlations_exceed_the_fx_correlation():
    """The factor structure has to survive into the realised series."""
    out = generator.simulate_market(start="2005-01-03", end="2024-12-31", seed=cfg.SEED)
    frame = pd.DataFrame({key: path.returns for key, path in out.assets.items()}).dropna()
    corr = frame.corr()
    assert corr.loc["SPX", "FTSE"] > 0.7
    assert corr.loc["SPX", "ISEQ"] > 0.7
    assert corr.loc["SPX", "EURUSD"] < corr.loc["SPX", "FTSE"]


def test_annualised_volatility_is_in_a_plausible_band():
    """SPX sits in the low twenties and the euro crosses in single digits."""
    out = generator.simulate_market(start="2005-01-03", end="2024-12-31", seed=cfg.SEED)
    annual = {key: float(path.returns.std() * np.sqrt(cfg.TRADING_DAYS)) for key, path in out.assets.items()}
    assert 0.15 < annual["SPX"] < 0.30
    assert 0.12 < annual["FTSE"] < 0.30
    assert annual["EURUSD"] < 0.13
    # Equity index volatility exceeds FX volatility, which is the ordering anyone
    # who trades either would expect.
    assert annual["EURUSD"] < annual["SPX"]


def test_leverage_effect_has_the_right_sign():
    """Negative returns raise next-day variance more than positive ones."""
    out = generator.simulate_market(start="2005-01-03", end="2024-12-31", seed=cfg.SEED)
    r = out.assets[cfg.PRIMARY_ASSET].returns.to_numpy()
    corr = float(np.corrcoef(r[:-1], r[1:] ** 2)[0, 1])
    assert corr < -0.02


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def test_same_seed_reproduces_exactly():
    a = generator.simulate_market(start="2015-01-01", end="2017-12-31", seed=123)
    b = generator.simulate_market(start="2015-01-01", end="2017-12-31", seed=123)
    for key in a.assets:
        assert np.array_equal(a.assets[key].returns.to_numpy(), b.assets[key].returns.to_numpy())
        assert np.array_equal(a.assets[key].intraday, b.assets[key].intraday)


def test_different_seeds_diverge():
    a = generator.simulate_market(start="2015-01-01", end="2017-12-31", seed=1)
    b = generator.simulate_market(start="2015-01-01", end="2017-12-31", seed=2)
    assert not np.array_equal(
        a.assets[cfg.PRIMARY_ASSET].returns.to_numpy(), b.assets[cfg.PRIMARY_ASSET].returns.to_numpy()
    )


def test_manifest_records_the_seed_and_the_crisis_schedule():
    out = generator.simulate_market(start="2015-01-01", end="2017-12-31", seed=99)
    manifest = out.manifest()
    assert manifest["seed"] == 99
    assert len(manifest["crisis_schedule"]) == len(cfg.CRISIS_SCHEDULE)
    assert manifest["parameters"]["intraday_bars"] == cfg.INTRADAY_BARS
    assert manifest["parameters"]["unconditional_vol_daily"] == cfg.SIM_UNCOND_VOL
    assert manifest["trading_days"] > 0


# ---------------------------------------------------------------------------
# Slow component
# ---------------------------------------------------------------------------
def test_slow_component_is_normalised_to_mean_one(rng):
    m = generator._slow_component(40_000, rng, cfg.SIM_LONG_PHI, cfg.SIM_LONG_SIGMA)
    assert m.mean() == pytest.approx(1.0, rel=1e-9)
    assert (m > 0).all()
    # The *stationary* log standard deviation is sigma_l / sqrt(1 - phi^2). The
    # sample standard deviation of a single near-unit-root realisation is a much
    # noisier object than that formula suggests: at phi = 0.995 the effective sample
    # size of 40,000 days is only about 100, so the estimate wanders by tens of
    # percent and a tight tolerance here would fail on a different seed rather than
    # catch a bug. The mean, above, is the sharp assertion.
    expected = cfg.SIM_LONG_SIGMA / np.sqrt(1.0 - cfg.SIM_LONG_PHI**2)
    assert 0.5 * expected < float(np.std(np.log(m))) < 2.0 * expected
