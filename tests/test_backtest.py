"""Tests for the VaR backtests and the Basel traffic light.

The tests come in three kinds. Known-answer tests pin the statistics to values
that can be checked by hand. Discrimination tests check that each test *rejects
the failure mode it exists to detect* - a Kupiec test that does not reject a
breach rate of 5% against a nominal 1% is not a test. And a small group of tests
covers the conventions that are easy to get wrong and invisible when wrong: which
direction the horizon aggregation runs in, and which levels the Basel columns are
meaningful at.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from marketrisk import backtest
from marketrisk import config as cfg


# ---------------------------------------------------------------------------
# Kupiec
# ---------------------------------------------------------------------------
def test_kupiec_is_zero_when_the_rate_is_exactly_nominal():
    hits = np.zeros(1_000)
    hits[:10] = 1.0  # exactly 1% at the 99% level
    result = backtest.kupiec_pof(hits, 0.99)
    assert result.statistic == pytest.approx(0.0, abs=1e-9)
    assert result.p_value == pytest.approx(1.0, abs=1e-9)
    assert not result.reject


def test_kupiec_matches_the_hand_computed_likelihood_ratio():
    hits = np.zeros(200)
    hits[:7] = 1.0  # 3.5% observed against 1% nominal
    result = backtest.kupiec_pof(hits, 0.99)
    p_hat = 7.0 / 200.0
    p_target = 0.01
    loglik = lambda p: 193.0 * np.log(1.0 - p) + 7.0 * np.log(p)  # noqa: E731
    expected = -2.0 * (loglik(p_target) - loglik(p_hat))
    assert result.statistic == pytest.approx(expected, rel=1e-9)
    assert result.p_value == pytest.approx(1.0 - stats.chi2.cdf(expected, 1), rel=1e-9)


def test_kupiec_rejects_a_rate_five_times_too_high():
    hits = np.zeros(2_000)
    hits[:100] = 1.0  # 5% against 1%
    assert backtest.kupiec_pof(hits, 0.99).reject


def test_kupiec_is_two_sided_and_rejects_zero_breaches():
    """A model that never breaches is mispriced, not conservative.

    The rejection is the point. A one-sided test would accept a 99.9% VaR wearing
    a 99% label, and the capital that wastes is real money.
    """
    hits = np.zeros(3_000)
    result = backtest.kupiec_pof(hits, 0.99)
    assert result.reject
    assert result.statistic > 0


def test_kupiec_accepts_a_small_imbalance():
    hits = np.zeros(3_000)
    hits[:36] = 1.0  # 1.2% vs 1%
    assert not backtest.kupiec_pof(hits, 0.99).reject


def test_kupiec_handles_an_empty_series():
    result = backtest.kupiec_pof(np.array([]), 0.99)
    assert not np.isfinite(result.statistic)
    assert not result.reject


# ---------------------------------------------------------------------------
# Christoffersen
# ---------------------------------------------------------------------------
def test_independence_accepts_an_iid_hit_sequence():
    rng = np.random.default_rng(41)
    hits = (rng.random(5_000) < 0.01).astype(float)
    assert not backtest.christoffersen_independence(hits).reject


def test_independence_rejects_a_clustered_hit_sequence():
    """The failure a static historical simulation exhibits.

    Twenty breaches inside one run have the same average rate as twenty spread
    evenly - Kupiec cannot tell them apart - and only the second model is useful,
    because a risk limit that is breached twenty days in a row has failed when the
    desk cannot absorb it. The clustering is what Christoffersen detects.
    """
    hits = np.zeros(2_000)
    hits[1_000:1_020] = 1.0
    assert backtest.christoffersen_independence(hits).reject
    # The unconditional rate is exactly nominal, which is why the two tests are run
    # together rather than instead of each other.
    assert not backtest.kupiec_pof(hits, 0.99).reject


def test_independence_cannot_be_assessed_on_a_degenerate_sequence():
    """One state never occurs, so there is no transition to test.

    The result is reported as "not rejected" with a statistic of zero *and a note
    explaining it*. The distinction matters: a model that never breaches has not
    passed an independence test, it has failed to supply the data one would need.
    """
    for hits in (np.zeros(500), np.ones(500)):
        result = backtest.christoffersen_independence(hits)
        assert result.statistic == 0.0
        assert not result.reject
        assert "cannot be assessed" in result.note


def test_conditional_coverage_has_two_degrees_of_freedom():
    hits = np.zeros(1_000)
    hits[:40] = 1.0
    result = backtest.christoffersen_cc(hits, 0.99)
    assert result.degrees_of_freedom == 2
    assert 0.0 <= result.p_value <= 1.0


# ---------------------------------------------------------------------------
# Dynamic quantile
# ---------------------------------------------------------------------------
def test_dynamic_quantile_accepts_iid_hits_at_the_right_rate():
    rng = np.random.default_rng(42)
    n = 3_000
    var = np.full(n, 0.02)
    hits = (rng.random(n) < 0.01).astype(float)
    assert not backtest.dynamic_quantile(hits, var, 0.99).reject


def test_dynamic_quantile_detects_hits_that_track_the_forecast():
    """Hits concentrated exactly where the VaR is smallest are not a quantile."""
    n = 3_000
    var = np.linspace(0.01, 0.03, n)
    hits = np.zeros(n)
    hits[np.argsort(var)[:60]] = 1.0
    assert backtest.dynamic_quantile(hits, var, 0.99).reject


# ---------------------------------------------------------------------------
# Basel zone mapping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("exceptions", "zone"),
    [(0, "green"), (4, "green"), (5, "yellow"), (9, "yellow"), (10, "red"), (25, "red")],
)
def test_basel_zone_boundaries(exceptions, zone):
    assert backtest.basel_zone(exceptions) == zone


def test_basel_plus_factor_is_non_decreasing_and_jumps_at_the_boundaries():
    factors = [backtest.basel_plus_factor(i) for i in range(0, 13)]
    # Not `strict=True`: this deliberately compares each element with the next, so the
    # two operands differ in length by one by construction.
    assert all(b >= a for a, b in zip(factors, factors[1:], strict=False))
    assert factors[0] == 0.0
    assert factors[4] == 0.0
    assert factors[5] == pytest.approx(0.40)
    assert factors[9] == pytest.approx(0.85)
    assert factors[10] == pytest.approx(cfg.BASEL_PLUS_FACTOR_RED)
    # The 4 -> 5 step costs 40% of the capital charge: a model that is "slightly"
    # over its breach rate is not slightly expensive.
    assert factors[5] - factors[4] == pytest.approx(0.40)


def test_rolling_traffic_light_leaves_a_partial_window_empty():
    """A window that is not yet full must not be reported as a low count."""
    index = pd.bdate_range("2020-01-01", periods=600)
    hits = np.zeros(600)
    hits[:30] = 1.0
    frame = backtest.basel_traffic_light(hits, index, window=250)
    assert frame["exceptions"].iloc[:249].isna().all()
    assert frame["exceptions"].iloc[249:].notna().all()
    assert frame["zone"].iloc[:249].eq("n/a").all()


def test_rolling_traffic_light_counts_the_trailing_window():
    index = pd.bdate_range("2020-01-01", periods=300)
    hits = np.zeros(300)
    hits[260:270] = 1.0  # ten breaches, all inside the final window
    frame = backtest.basel_traffic_light(hits, index, window=250)
    assert frame["exceptions"].iloc[-1] == 10
    assert frame["zone"].iloc[-1] == "red"
    assert frame["plus_factor"].iloc[-1] == pytest.approx(cfg.BASEL_PLUS_FACTOR_RED)


# ---------------------------------------------------------------------------
# Coverage report conventions
# ---------------------------------------------------------------------------
def test_coverage_report_counts_breaches_on_the_loss_side():
    rng = np.random.default_rng(43)
    n = 2_000
    r = rng.standard_normal(n) * 0.01
    var = np.full(n, 0.02)
    report = backtest.coverage_report("m", "parametric", 0.99, 1, r, var)
    expected = int(np.sum(r < -0.02))
    assert report.breaches == expected
    assert report.expected_rate == pytest.approx(0.01)


def test_coverage_report_aggregates_the_horizon_internally():
    """A 10-day VaR compared against daily returns is the failure this prevents.

    The deterministic part of the assertion is the observation count: an h-day
    return does not exist until day ``h``, so ten-day evaluation loses exactly nine
    usable observations. If the horizon were ignored the two counts would be equal,
    and an earlier version of this function did exactly that - it produced zero
    breaches in 4,200 days for every method, which read as an outstanding result.

    The breach counts are checked for comparability rather than for an ordering.
    With iid returns and a correctly scaled VaR the two rates are the same in
    expectation, so which count is larger is sampling noise and asserting a
    direction would be asserting a coin flip.
    """
    rng = np.random.default_rng(44)
    n = 3_000
    r = rng.standard_normal(n) * 0.01
    one_day = backtest.coverage_report("m", "historical", 0.99, 1, r, np.full(n, 0.023))
    ten_day = backtest.coverage_report("m", "historical", 0.99, 10, r, np.full(n, 0.023 * np.sqrt(10.0)))
    assert one_day.observations - ten_day.observations == 9
    assert ten_day.breaches > 5
    assert 0.4 * one_day.breaches < ten_day.breaches < 2.5 * one_day.breaches


def test_basel_columns_are_blanked_below_99_percent():
    """At 95% a correct model is in the red zone by construction.

    Roughly 12.5 breaches a year against a green threshold of 4 means the traffic
    light says nothing about a 95% model's quality. Reporting it would turn a
    regulatory convention into a false finding about six good models.
    """
    rng = np.random.default_rng(45)
    r = rng.standard_normal(3_500) * 0.01
    var = np.full(r.size, 0.001)
    at_95 = backtest.coverage_report("m", "historical", 0.95, 1, r, var)
    assert at_95.basel_zone == "n/a"
    assert at_95.basel_exceptions == -1
    assert not np.isfinite(at_95.basel_plus_factor)

    at_99 = backtest.coverage_report("m", "historical", 0.99, 1, r, var)
    assert at_99.basel_zone in {"green", "yellow", "red"}
    assert at_99.basel_exceptions >= 0


def test_coverage_report_start_trims_the_burn_in():
    rng = np.random.default_rng(46)
    r = rng.standard_normal(2_000) * 0.01
    var = np.full(r.size, 0.02)
    full = backtest.coverage_report("m", "historical", 0.99, 1, r, var, start=0)
    trimmed = backtest.coverage_report("m", "historical", 0.99, 1, r, var, start=1_500)
    assert trimmed.observations == 500
    assert full.observations == 2_000


def test_coverage_report_row_has_every_reported_column():
    r = np.zeros(600)
    report = backtest.coverage_report("m", "historical", 0.99, 1, r, np.full(600, 0.01))
    row = report.as_row()
    for key in ("method", "level", "horizon", "breaches", "kupiec_p", "ind_p", "cc_p", "dq_p", "basel_zone"):
        assert key in row


def test_a_correct_model_is_not_rejected_at_99_percent():
    """The end-to-end sanity check: simulate from a known law, backtest its own VaR."""
    rng = np.random.default_rng(47)
    n = 6_000
    sigma = 0.01
    r = rng.standard_t(6.0, n) * sigma / np.sqrt(6.0 / 4.0)
    quantile = -stats.t.ppf(0.01, 6.0) * np.sqrt(4.0 / 6.0)
    report = backtest.coverage_report("truth", "parametric", 0.99, 1, r, np.full(n, sigma * quantile))
    assert not report.kupiec.reject
    assert not report.independence.reject
    assert abs(report.observed_rate - 0.01) < 0.004


def test_a_deliberately_wrong_model_is_rejected():
    rng = np.random.default_rng(48)
    n = 6_000
    r = rng.standard_t(6.0, n) * 0.01 / np.sqrt(6.0 / 4.0)
    report = backtest.coverage_report("too_tight", "parametric", 0.99, 1, r, np.full(n, 0.015))
    assert report.kupiec.reject


# ---------------------------------------------------------------------------
# Stress slices
# ---------------------------------------------------------------------------
def test_stress_window_coverage_reports_one_row_per_declared_window():
    index = pd.bdate_range(cfg.DEFAULT_START, cfg.DEFAULT_END)
    rng = np.random.default_rng(49)
    r = rng.standard_normal(len(index)) * 0.01
    var = np.full(len(index), 0.02)
    frame = backtest.stress_window_coverage(r, var, index, 0.99, 1)
    assert len(frame) == len(cfg.STRESS_WINDOWS)
    assert set(frame["window"]) == set(cfg.STRESS_WINDOWS)
    for column in ("observations", "breaches", "breach_rate", "expected_rate", "realised_vol_annual"):
        assert column in frame.columns


def test_stress_window_coverage_skips_a_window_outside_the_sample():
    """A window with no overlapping observations is omitted, not reported as zero."""
    index = pd.bdate_range("2015-01-01", "2016-12-31")
    r = np.zeros(len(index))
    frame = backtest.stress_window_coverage(r, np.full(len(index), 0.01), index, 0.99, 1)
    assert len(frame) < len(cfg.STRESS_WINDOWS)


def test_stress_window_coverage_finds_breaches_where_the_losses_are():
    index = pd.bdate_range("2007-01-01", "2009-12-31")
    r = np.zeros(len(index))
    var = np.full(len(index), 0.01)
    # Put large losses inside the GFC window and leave the rest quiet.
    start = index.searchsorted(pd.Timestamp("2008-09-15"))
    r[start : start + 40] = -0.05
    frame = backtest.stress_window_coverage(r, var, index, 0.99, 1).set_index("window")
    assert frame.loc["GFC 2008-09", "breaches"] == 40
