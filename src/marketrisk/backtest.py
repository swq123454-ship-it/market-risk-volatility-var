"""Backtests for VaR: unconditional coverage, independence, and the regulator's view.

Three questions, and a VaR model can fail any of them independently.

1. **Is the breach rate right?** Kupiec's proportion-of-failures test. A 99% VaR
   should be breached about 2.5 times a year; 12 times is a failure, and so is
   zero - a model that never breaches is not conservative, it is mispriced, and
   it is costing the desk capital for nothing. The test is two-sided for that
   reason.
2. **Are the breaches independent?** Christoffersen's test. A model that breaches
   in runs - three days in a row, then nothing for a quarter - has the right
   average and is still useless: the breaches arrive exactly when the desk cannot
   absorb them, which is the whole point of a risk limit. This is the failure
   mode a GARCH model exists to prevent, and the one a static historical
   simulation exhibits.
3. **Do the breaches depend on anything observable?** Engle and Manganelli's
   dynamic quantile test regresses the hit indicator on its own lags and on the
   VaR itself. If either has predictive power, the quantile is not a quantile.
   This is strictly stronger than the first two, because it can detect a
   misspecification that leaves both the rate and the clustering intact.

Then the **Basel traffic light**, which is not a statistical test but the rule
that actually determines how much capital a bank holds. It counts exceptions over
a rolling 250-day window and maps them to a zone; the yellow zone carries a
multiplicative add-on to the capital charge that is not linear in the count - the
jump from 4 to 5 exceptions costs 40% of the charge, and from 9 to 10 costs the
entire charge again. Reporting the count without the plus factor would miss the
part with money attached.

A note on the word "pass"
-------------------------
None of these tests is a proof. At 99% over a 3,000-day sample there are about 30
breaches, so the Kupiec test has almost no power: the 95% acceptance region spans
roughly 15 to 50 breaches per 3,000 days, which is a wide band. The right reading
of a failure is "this model's breach rate is inconsistent with the level", and of
a pass is "not enough evidence to say otherwise" - never "this model is correct".
The study reports the statistics and the observed rates side by side so the
reader can see how much slack the test is allowing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg

EPS = 1e-12


# ---------------------------------------------------------------------------
# Test statistics
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    """One hypothesis test, with the numbers needed to interpret it."""

    name: str
    statistic: float
    p_value: float
    degrees_of_freedom: int
    reject: bool
    note: str = ""


def kupiec_pof(hits: np.ndarray, level: float, alpha: float = cfg.ALPHA_SIGNIFICANCE) -> TestResult:
    """Kupiec's unconditional-coverage (proportion of failures) likelihood ratio.

    ``LR = -2 [ n0 ln(1-p) + n1 ln p - n0 ln(1-pi) - n1 ln pi ]`` with ``pi`` the
    observed breach rate. Under the null it is chi-squared with one degree of
    freedom.

    Reported two-sided: the alternative is "the rate is wrong", not "the rate is
    too high". A model with zero breaches in 3,000 days is rejected, correctly -
    it is a 99.9% VaR wearing a 99% label, and the capital it wastes is real.
    """
    h = np.asarray(hits, dtype=float)
    h = h[np.isfinite(h)]
    n = h.size
    if n == 0:
        return TestResult("kupiec_pof", float("nan"), float("nan"), 1, False, "no observations")
    n1 = float(h.sum())
    n0 = n - n1
    pi = n1 / n
    target = 1.0 - level

    def loglik(rate: float) -> float:
        rate = float(np.clip(rate, 1e-12, 1.0 - 1e-12))
        return n0 * np.log(1.0 - rate) + n1 * np.log(rate)

    statistic = max(-2.0 * (loglik(target) - loglik(pi)), 0.0)
    p_value = float(1.0 - stats.chi2.cdf(statistic, 1))
    return TestResult(
        "kupiec_pof",
        float(statistic),
        p_value,
        1,
        p_value < alpha,
        f"observed {pi:.4%} vs target {target:.4%} ({int(n1)} breaches in {n})",
    )


def christoffersen_independence(hits: np.ndarray, alpha: float = cfg.ALPHA_SIGNIFICANCE) -> TestResult:
    """Christoffersen's independence test on the hit sequence.

    A first-order Markov chain on the hit indicator: the null is that
    ``P(hit_t | hit_{t-1}) = P(hit_t)``, i.e. that a breach today tells you
    nothing about tomorrow. The alternative is clustering, which is precisely the
    failure a model without a volatility filter exhibits.

    If one of the two states never occurs - no breaches at all, or a breach on every
    day - the transition counts are degenerate and clustering cannot be assessed. The
    result is then reported as "not rejected" with a statistic of zero and a note
    saying why, rather than as a fabricated p-value. The distinction matters: a model
    that never breaches has not passed an independence test, it has failed to provide
    the data one would need.
    """
    h = np.asarray(hits, dtype=float)
    h = h[np.isfinite(h)]
    if h.size < 3:
        return TestResult("christoffersen_ind", float("nan"), float("nan"), 1, False, "too few observations")
    previous, current = h[:-1], h[1:]
    n00 = float(np.sum((previous == 0) & (current == 0)))
    n01 = float(np.sum((previous == 0) & (current == 1)))
    n10 = float(np.sum((previous == 1) & (current == 0)))
    n11 = float(np.sum((previous == 1) & (current == 1)))
    n0, n1 = n00 + n01, n10 + n11
    if n0 == 0 or n1 == 0:
        return TestResult(
            "christoffersen_ind",
            0.0,
            1.0,
            1,
            False,
            "one state never occurs, so clustering cannot be assessed",
        )
    pi01 = n01 / n0
    pi11 = n11 / n1
    pi = (n01 + n11) / (n0 + n1)

    def loglik(a: float, b: float) -> float:
        total = 0.0
        for count, rate in ((n00, 1 - a), (n01, a), (n10, 1 - b), (n11, b)):
            if count > 0:
                total += count * np.log(max(rate, 1e-12))
        return total

    statistic = max(-2.0 * (loglik(pi, pi) - loglik(pi01, pi11)), 0.0)
    p_value = float(1.0 - stats.chi2.cdf(statistic, 1))
    return TestResult(
        "christoffersen_ind",
        float(statistic),
        p_value,
        1,
        p_value < alpha,
        f"P(hit|hit)={pi11:.3f}, P(hit|no hit)={pi01:.3f}",
    )


def christoffersen_cc(hits: np.ndarray, level: float, alpha: float = cfg.ALPHA_SIGNIFICANCE) -> TestResult:
    """Conditional coverage: Kupiec plus independence, on two degrees of freedom."""
    pof = kupiec_pof(hits, level, alpha)
    ind = christoffersen_independence(hits, alpha)
    if not np.isfinite(pof.statistic) or not np.isfinite(ind.statistic):
        return TestResult("christoffersen_cc", float("nan"), float("nan"), 2, False, "undefined")
    statistic = pof.statistic + ind.statistic
    p_value = float(1.0 - stats.chi2.cdf(statistic, 2))
    return TestResult(
        "christoffersen_cc",
        float(statistic),
        p_value,
        2,
        p_value < alpha,
        "conditional coverage",
    )


def dynamic_quantile(
    hits: np.ndarray, var: np.ndarray, level: float, lags: int = 4, alpha: float = cfg.ALPHA_SIGNIFICANCE
) -> TestResult:
    """Engle-Manganelli dynamic quantile test.

    Regresses ``hit_t - (1 - level)`` on a constant, ``lags`` lagged hits and the
    VaR itself, and forms ``DQ = beta' X'X beta / (p(1-p))``, which is chi-squared
    with as many degrees of freedom as there are regressors. Under the null the
    hits are unpredictable from anything in the information set used, so both a
    breach that follows a breach and a breach that follows a high VaR are
    rejections.

    The VaR enters in *volatility* units rather than return units purely to keep
    the regression matrix well conditioned; it is the same test either way.
    """
    h = np.asarray(hits, dtype=float)
    v = np.asarray(var, dtype=float)
    mask = np.isfinite(h) & np.isfinite(v)
    h, v = h[mask], v[mask]
    n = h.size
    if n < lags + 20:
        return TestResult("dynamic_quantile", float("nan"), float("nan"), lags + 2, False, "too few observations")
    p = 1.0 - level
    demeaned = h - p
    columns = [np.ones(n - lags), v[lags:]]
    for lag in range(1, lags + 1):
        columns.append(demeaned[lags - lag : n - lag])
    x = np.column_stack(columns)
    target = demeaned[lags:]
    beta, *_ = np.linalg.lstsq(x, target, rcond=None)
    fitted = x @ beta
    statistic = float(fitted @ fitted / max(p * (1.0 - p), EPS))
    dof = x.shape[1]
    p_value = float(1.0 - stats.chi2.cdf(statistic, dof))
    return TestResult(
        "dynamic_quantile",
        statistic,
        p_value,
        dof,
        p_value < alpha,
        f"{dof} regressors, {n} observations",
    )


# ---------------------------------------------------------------------------
# Basel traffic light
# ---------------------------------------------------------------------------
def basel_zone(exceptions: int) -> str:
    """Green / yellow / red for an exception count over the Basel window."""
    for low, high, name in cfg.BASEL_ZONES:
        if low <= exceptions <= high:
            return name
    return "red"


def basel_plus_factor(exceptions: int) -> float:
    """Capital add-on implied by the exception count.

    The table is the reason the traffic light matters: five exceptions in a year
    costs 40% more capital than four, and ten costs twice as much as five. A
    model that is "only slightly" over its breach rate is not slightly expensive.
    """
    if exceptions in cfg.BASEL_PLUS_FACTOR:
        return cfg.BASEL_PLUS_FACTOR[exceptions]
    return cfg.BASEL_PLUS_FACTOR_RED if exceptions > 9 else 0.0


def basel_traffic_light(hits: np.ndarray, index: pd.DatetimeIndex, window: int = cfg.BASEL_WINDOW) -> pd.DataFrame:
    """Rolling Basel window: exception count, zone and capital add-on.

    The count is over a full window of *valid* observations, so a window that
    straddles the start of the evaluation sample is left as ``NaN`` rather than
    reported as a low count - a partially observed window would make a model look
    green for no reason.
    """
    h = pd.Series(np.asarray(hits, dtype=float), index=index)
    counts = h.rolling(window, min_periods=window).sum()
    zones = [basel_zone(int(c)) if np.isfinite(c) else "n/a" for c in counts.to_numpy()]
    factors = [basel_plus_factor(int(c)) if np.isfinite(c) else np.nan for c in counts.to_numpy()]
    return pd.DataFrame({"exceptions": counts, "zone": zones, "plus_factor": factors}, index=index)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
@dataclass
class CoverageReport:
    """Everything the coverage tables need for one method at one level."""

    method: str
    family: str
    level: float
    horizon: int
    observations: int
    breaches: int
    observed_rate: float
    expected_rate: float
    kupiec: TestResult
    independence: TestResult
    conditional: TestResult
    dynamic: TestResult
    worst_breach: float
    mean_es_excess: float
    basel_exceptions: int
    basel_zone: str
    basel_plus_factor: float

    def as_row(self) -> dict[str, object]:
        return {
            "method": self.method,
            "family": self.family,
            "level": self.level,
            "horizon": self.horizon,
            "observations": self.observations,
            "breaches": self.breaches,
            "observed_rate": self.observed_rate,
            "expected_rate": self.expected_rate,
            "kupiec_lr": self.kupiec.statistic,
            "kupiec_p": self.kupiec.p_value,
            "kupiec_reject": self.kupiec.reject,
            "ind_lr": self.independence.statistic,
            "ind_p": self.independence.p_value,
            "ind_reject": self.independence.reject,
            "cc_lr": self.conditional.statistic,
            "cc_p": self.conditional.p_value,
            "cc_reject": self.conditional.reject,
            "dq_stat": self.dynamic.statistic,
            "dq_p": self.dynamic.p_value,
            "dq_reject": self.dynamic.reject,
            "worst_breach": self.worst_breach,
            "mean_es_excess": self.mean_es_excess,
            "basel_exceptions": self.basel_exceptions,
            "basel_zone": self.basel_zone,
            "basel_plus_factor": self.basel_plus_factor,
        }


def coverage_report(
    method: str,
    family: str,
    level: float,
    horizon: int,
    returns: np.ndarray | pd.Series,
    var: np.ndarray,
    es: np.ndarray | None = None,
    start: int = 0,
) -> CoverageReport:
    """Run every coverage test for one method over one evaluation slice.

    ``returns`` is always the **daily** series and ``horizon`` says how to
    aggregate it. Accepting the daily series rather than an already-aggregated one
    is deliberate: the h-day VaR and the h-day return have to line up, and the
    only way to guarantee that is to derive one of them from the other here. The
    first version of this function took whatever returns it was handed and
    produced 0 breaches in 4,200 days for every 10-day method, which looks like a
    spectacular result and is a units error.

    ``start`` trims the burn-in. Every method needs a different amount of history
    before its first forecast exists, and comparing methods over different samples
    is the easiest way to produce a flattering table by accident; the caller
    passes a common start index instead.

    The Basel columns are only meaningful at the 99% level, which is what the
    regulation is written for. At 95% a correct model breaches about 12.5 times a
    year and would be in the red zone by construction; the columns are therefore
    blanked rather than reported as a damning finding about a model that is fine.
    """
    from .var import horizon_returns

    daily = pd.Series(np.asarray(returns, dtype=float))
    aggregated = horizon_returns(daily, horizon)
    r = aggregated.to_numpy(dtype=float)[start:]
    v = np.asarray(var, dtype=float)[start:]
    mask = np.isfinite(r) & np.isfinite(v)
    r, v = r[mask], v[mask]
    breaches = r < -v
    hits = breaches.astype(float)
    n = hits.size
    level_alpha = cfg.ALPHA_SIGNIFICANCE
    kup = kupiec_pof(hits, level, level_alpha)
    ind = christoffersen_independence(hits, level_alpha)
    cc = christoffersen_cc(hits, level, level_alpha)
    dq = dynamic_quantile(hits, v, level, lags=4, alpha=level_alpha)
    tail = r[breaches]
    mean_es_excess = float(np.mean(-tail - v[breaches])) if tail.size else 0.0
    is_basel_level = abs(level - 0.99) < 1e-9
    window = min(cfg.BASEL_WINDOW, n)
    recent = int(hits[-window:].sum()) if (is_basel_level and window > 0) else -1
    return CoverageReport(
        method=method,
        family=family,
        level=level,
        horizon=horizon,
        observations=int(n),
        breaches=int(hits.sum()),
        observed_rate=float(hits.mean()) if n else float("nan"),
        expected_rate=1.0 - level,
        kupiec=kup,
        independence=ind,
        conditional=cc,
        dynamic=dq,
        worst_breach=float(tail.min()) if tail.size else 0.0,
        mean_es_excess=mean_es_excess,
        basel_exceptions=recent,
        basel_zone=basel_zone(recent) if is_basel_level else "n/a",
        basel_plus_factor=basel_plus_factor(recent) if is_basel_level else float("nan"),
    )


def stress_window_coverage(
    returns: np.ndarray | pd.Series,
    var: np.ndarray,
    index: pd.DatetimeIndex,
    level: float,
    horizon: int = 1,
) -> pd.DataFrame:
    """Breach rate inside each declared stress window.

    The aggregate breach rate is an average over calm and violent days, and a
    model can hit it exactly while being far too tight in the crisis and far too
    loose everywhere else. Slicing by regime is what makes that visible.

    Returns are aggregated to the horizon here as well, for the same reason as in
    :func:`coverage_report`.
    """
    from .var import horizon_returns

    daily = pd.Series(np.asarray(returns, dtype=float), index=index)
    r = horizon_returns(daily, horizon)
    v = pd.Series(np.asarray(var, dtype=float), index=index)
    rows = []
    for name, (start, end) in cfg.STRESS_WINDOWS.items():
        slice_r = r.loc[start:end]
        slice_v = v.loc[start:end]
        mask = slice_r.notna() & slice_v.notna()
        slice_r, slice_v = slice_r[mask], slice_v[mask]
        if slice_r.empty:
            continue
        breach = (slice_r < -slice_v).astype(float)
        rows.append(
            {
                "window": name,
                "observations": int(len(slice_r)),
                "breaches": int(breach.sum()),
                "breach_rate": float(breach.mean()),
                "expected_rate": 1.0 - level,
                "realised_vol_annual": float(slice_r.std() * np.sqrt(cfg.TRADING_DAYS)),
                "worst_return": float(slice_r.min()),
                "mean_var": float(slice_v.mean()),
            }
        )
    return pd.DataFrame(rows)
