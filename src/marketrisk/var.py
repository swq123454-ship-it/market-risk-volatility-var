"""Value-at-Risk and Expected Shortfall, by five routes.

The question this module answers is narrow and exact: for day ``t``, what loss
will not be exceeded with probability ``level``, using only information available
at the close of ``t-1``? Every method below is a different answer to that
question, and the study's job is to find out which answers survive contact with
the data.

The five routes
---------------
**Historical simulation.** The empirical quantile of the last ``N`` returns. No
model, no parameters, no distributional assumption - and the assumption it does
make is the strong one: that the next day is a draw from the same distribution as
the last ``N`` days. Its failure is exactly the one that matters, because it
responds to a volatility regime change only after the change is already in the
window.

**Age-weighted historical simulation** (Boudoukh-Richardson-Whitelaw). The same
empirical quantile, with the observations exponentially down-weighted by age and
the quantile found by cumulative weight rather than by rank. It is the cheapest
possible fix for HS's staleness and it works surprisingly well.

**Parametric, Gaussian.** ``sigma_t * z_alpha``. Fast, transparent, the number
everyone's limit system starts from, and wrong in the tail by construction.

**Parametric, Student-t.** The same with a fat-tailed innovation and the degrees
of freedom estimated from the data rather than assumed. Fixes the *unconditional*
tail but not the conditional one, because the model is misspecified on purpose:
the simulated returns are negatively skewed and a symmetric t cannot see it.

**Filtered historical simulation.** ``sigma_t`` times an empirical quantile of
*standardised* residuals. This is the method that gets both parts right - the
conditional scale from a GARCH filter, the shape of the tail from the data - and
it is the one a modern desk actually uses. The residual pool is taken from days
strictly before ``t``, so no residual is ever scaled by a volatility estimate that
already contains it.

**Monte Carlo.** Simulate the fitted variance recursion forward and take the
quantile of the simulated cumulative returns. It is the only method here that
gets the *multi-day* distribution right, because it iterates the recursion
instead of scaling it. The square-root-of-time rule that everything else uses is
a statement about i.i.d. returns, and GARCH returns are the opposite of i.i.d. by
construction; the gap between the two is one of the study's reported results.

Expected Shortfall is computed alongside every VaR, because a quantile says
nothing about how bad the breach is. For the empirical methods it is the average
of the losses beyond the quantile; for the parametric ones it is the closed form;
for the tail-modelled ones (EVT) it follows from the fitted generalised Pareto
tail.

A note on signs
---------------
Returns are log returns. A VaR is reported as a **positive loss magnitude**, so a
breach is ``r_t < -VaR_t``. ES is likewise positive. Every internal computation
that would otherwise flip sign is commented, because this is the single most
common source of a silently mirrored result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg
from . import kernels, volmodels

EPS = 1e-12


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------
def normal_quantile(level: float) -> float:
    """Lower-tail standard-normal quantile, returned as a positive loss."""
    return float(-stats.norm.ppf(1.0 - level))


def normal_es(level: float) -> float:
    """E[-Z | Z <= z_{1-level}] for a standard normal, positive."""
    z = stats.norm.ppf(1.0 - level)
    return float(stats.norm.pdf(z) / (1.0 - level))


def student_t_quantile(level: float, dof: float) -> float:
    """Lower-tail quantile of a **unit-variance** Student-t, positive loss.

    The ``sqrt((dof-2)/dof)`` factor is what makes the innovation unit-variance.
    Dropping it is a common and consequential error: the standard t has variance
    ``dof/(dof-2)``, so at ``dof = 5`` the unscaled quantile is about 29% too
    large and the VaR would be badly over-conservative.
    """
    if not np.isfinite(dof) or dof > 200:
        return normal_quantile(level)
    quantile = stats.t.ppf(1.0 - level, dof)
    return float(-quantile * np.sqrt((dof - 2.0) / dof))


def student_t_es(level: float, dof: float) -> float:
    """E[-Z | Z <= q] for a unit-variance Student-t, positive.

    Uses the closed form for the partial mean of a Student-t,
    ``E[T | T <= t_q] = -((dof + t_q^2) / (dof - 1)) * f(t_q) / alpha``, then
    rescales to unit variance.
    """
    if not np.isfinite(dof) or dof > 200:
        return normal_es(level)
    alpha = 1.0 - level
    t_q = stats.t.ppf(alpha, dof)
    pdf = float(stats.t.pdf(t_q, dof))
    conditional = -((dof + t_q**2) / (dof - 1.0)) * pdf / alpha
    return float(-conditional * np.sqrt((dof - 2.0) / dof))


def fit_tail_dof(standardised: np.ndarray, lower: float = 2.05, upper: float = 60.0) -> float:
    """Degrees of freedom of a unit-variance t, by maximum likelihood.

    Estimated rather than fixed, because the parametric-t VaR is only as good as
    this number and a hard-coded ``dof = 5`` would be a hidden assumption. The
    search is over a small bracket because the likelihood is flat above about 30,
    where the t is indistinguishable from the normal.
    """
    z = np.asarray(standardised, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < 50:
        return 5.0
    grid = np.concatenate([np.linspace(lower, 12.0, 46), np.linspace(13.0, upper, 24)])

    def neg_loglik(dof: float) -> float:
        return -float(np.sum(kernels.student_t_logpdf(z, dof)))

    losses = np.array([neg_loglik(d) for d in grid])
    return float(grid[int(np.argmin(losses))])


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------
@dataclass
class RiskForecast:
    """A VaR/ES path plus the metadata a table needs to interpret it."""

    method: str
    level: float
    horizon: int
    var: pd.Series
    es: pd.Series
    #: Method family, used to group the report tables.
    family: str = ""

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "var": self.var,
                "es": self.es,
                "method": self.method,
                "level": self.level,
                "horizon": self.horizon,
                "family": self.family,
            },
            index=self.var.index,
        )


# ---------------------------------------------------------------------------
# Historical simulation
# ---------------------------------------------------------------------------
def historical_var(returns: np.ndarray, level: float, window: int, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Rolling empirical VaR/ES on the raw return distribution.

    The window holds ``window`` *horizons*, i.e. overlapping h-day returns, so
    that a 10-day historical VaR is a genuine 10-day empirical quantile rather
    than a one-day number scaled up. Overlapping windows are the standard
    construction and they make the historical quantile at longer horizons
    genuinely data-driven, at the cost of the observations being dependent -
    which matters for the standard errors of the backtest, and is why the
    Christoffersen independence test is reported next to Kupiec.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    var = np.full(n, np.nan)
    es = np.full(n, np.nan)
    if horizon > 1:
        cumulative = np.full(n, np.nan)
        for i in range(horizon - 1, n):
            cumulative[i] = r[i - horizon + 1 : i + 1].sum()
    else:
        cumulative = r.copy()

    for t in range(window + horizon - 1, n):
        sample = cumulative[t - window : t]
        sample = sample[np.isfinite(sample)]
        if sample.size < 20:
            continue
        quantile = float(np.quantile(sample, 1.0 - level))
        tail = sample[sample <= quantile]
        var[t] = max(-quantile, 0.0)
        # ES is the mean of the tail, not the mean of the exceedances beyond the
        # quantile value: with a small window the two differ materially and the
        # latter is biased towards the quantile itself.
        es[t] = max(-float(tail.mean()), 0.0) if tail.size else var[t]
    return var, es


def age_weighted_var(
    returns: np.ndarray, level: float, window: int, decay: float = cfg.HS_DECAY, horizon: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """BRW age-weighted historical simulation.

    Weights decay as ``decay^k`` with ``k`` the age in days, then the quantile is
    the observation at which the cumulative weight first reaches ``1 - level``.
    Because the weights are not uniform, this is not a rank statistic and cannot
    be obtained by sorting and indexing - a detail that is easy to get wrong and
    produces a VaR that is silently just the plain historical one.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    var = np.full(n, np.nan)
    es = np.full(n, np.nan)
    if horizon > 1:
        cumulative = np.full(n, np.nan)
        for i in range(horizon - 1, n):
            cumulative[i] = r[i - horizon + 1 : i + 1].sum()
    else:
        cumulative = r.copy()

    ages = np.arange(window)[::-1]  # 0 for the most recent day
    weights = decay**ages
    weights = weights / weights.sum()
    target = 1.0 - level

    for t in range(window + horizon - 1, n):
        sample = cumulative[t - window : t]
        if not np.isfinite(sample).all():
            continue
        order = np.argsort(sample)
        sorted_sample = sample[order]
        sorted_weights = weights[order]
        cumulative_weight = np.cumsum(sorted_weights)
        index = int(np.searchsorted(cumulative_weight, target))
        index = min(max(index, 0), sorted_sample.size - 1)
        quantile = float(sorted_sample[index])
        var[t] = max(-quantile, 0.0)
        tail = sorted_sample[: index + 1]
        tail_weights = sorted_weights[: index + 1]
        if tail.size and tail_weights.sum() > 0:
            es[t] = max(-float(np.average(tail, weights=tail_weights)), 0.0)
        else:
            es[t] = var[t]
    return var, es


# ---------------------------------------------------------------------------
# Parametric
# ---------------------------------------------------------------------------
def parametric_var(
    sigma: np.ndarray,
    level: float,
    distribution: str = "normal",
    dof: float = 5.0,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """``sigma_t * quantile`` with a square-root-of-time horizon scaling.

    The scaling rule is stated rather than buried because it is an assumption
    with a direction of error. When current volatility is above its long-run
    level - the state that matters - the variance recursion pulls it back down, so
    ``sqrt(h)`` overstates the h-day risk. When it is below, it understates it.
    The Monte-Carlo method measures that gap.
    """
    s = np.asarray(sigma, dtype=float)
    scale = np.sqrt(float(horizon))
    if distribution == "normal":
        quantile, expected = normal_quantile(level), normal_es(level)
    elif distribution == "t":
        quantile, expected = student_t_quantile(level, dof), student_t_es(level, dof)
    else:
        raise ValueError(f"unknown distribution {distribution!r}")
    return s * quantile * scale, s * expected * scale


# ---------------------------------------------------------------------------
# Filtered historical simulation
# ---------------------------------------------------------------------------
def filtered_historical_var(
    returns: np.ndarray,
    sigma: np.ndarray,
    level: float,
    window: int = cfg.HS_WINDOWS[0],
    horizon: int = 1,
    tail: str = "empirical",
    evt_threshold: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    """FHS: scale the conditional volatility by an empirical residual quantile.

    ``z_s = r_s / sigma_s`` is the standardised residual. The quantile of ``z``
    over the window ending at ``t-1`` is then multiplied by ``sigma_t``. This is
    the method that separates the *scale* of tomorrow's risk from the *shape* of
    the tail, and it is why it works: a GARCH filter captures the clustering, and
    the empirical residual pool captures the fat tail and the skew that a
    symmetric parametric form would miss.

    ``tail='evt'`` replaces the empirical quantile with a generalised-Pareto fit to
    the exceedances above ``evt_threshold``. At 99% and above with a 250-day
    window there are only about 2.5 observations in the tail, so the empirical
    quantile is essentially interpolation between noise; the GPD extrapolates a
    fitted tail instead. That difference is one of the things the backtest is
    asked to adjudicate.
    """
    r = np.asarray(returns, dtype=float)
    s = np.asarray(sigma, dtype=float)
    n = r.size
    var = np.full(n, np.nan)
    es = np.full(n, np.nan)
    standardised = np.divide(r, s, out=np.full(n, np.nan), where=s > 0)

    for t in range(max(window, 60) + 1, n):
        if not np.isfinite(s[t]) or s[t] <= 0:
            continue
        sample = standardised[t - window : t]
        sample = sample[np.isfinite(sample)]
        if sample.size < 60:
            continue
        if tail == "evt":
            quantile_z, es_z = _gpd_tail(sample, level, evt_threshold)
        else:
            quantile_z = float(np.quantile(sample, 1.0 - level))
            tail_values = sample[sample <= quantile_z]
            es_z = float(tail_values.mean()) if tail_values.size else quantile_z
        # The horizon enters through the same square-root rule as the parametric
        # method, but applied to the *residual* quantile: the residual scale is
        # already the conditional volatility, so scaling both would double-count.
        scale = np.sqrt(float(horizon))
        var[t] = max(-quantile_z * s[t] * scale, 0.0)
        es[t] = max(-es_z * s[t] * scale, 0.0)
    return var, es


def _gpd_tail(sample: np.ndarray, level: float, threshold_quantile: float) -> tuple[float, float]:
    """Peaks-over-threshold quantile and ES from a generalised Pareto fit.

    Exceedances of the ``threshold_quantile`` quantile are fitted to a GPD by
    **method of moments**, which is closed-form and - unlike maximum likelihood on
    a handful of exceedances - does not fail to converge on the quiet windows that
    make up most of the sample.

    The moments estimator is ``xi = (1 - mean^2 / variance) / 2``, derived from
    ``E[Y] = beta/(1-xi)`` and ``Var(Y) = beta^2 / ((1-xi)^2 (1-2xi))``. Anything
    built on ``mean(y^2)`` instead of the variance is wrong by a factor that looks
    harmless and is not: the natural-looking ``(mean^2 / mean_square - 1) / 2``
    returns ``-0.25`` for an exponential tail whose true shape is zero, and a
    negative shape means a *bounded* tail. The resulting VaR came out about 20%
    below the empirical 99% quantile and produced twice the expected breach rate.

    Returns the quantile and the ES of the *standardised residual*, both on the
    loss side (i.e. negative return scale). Both are negative, on every path out of
    this function; the caller negates them and multiplies by ``sigma_t``. When there
    are too few exceedances to estimate a shape and a scale - fewer than eight, which
    a quiet 250-day window does produce - the fit is abandoned and the empirical
    quantile is returned in its place, so the method degrades into the ``empirical``
    tail rather than into a number of its own invention.
    """
    loss = -sample  # work on the loss side so the tail is the upper tail
    threshold = float(np.quantile(loss, threshold_quantile))
    exceedances = loss[loss > threshold] - threshold
    n = loss.size
    n_excess = exceedances.size
    # The empirical quantile on the *residual* scale, which is negative, and which is
    # both what this function returns and what the caller multiplies by sigma_t. The
    # sign is load-bearing: the caller clamps at zero, so returning the positive
    # magnitude instead does not raise an error - it reports a VaR and an ES of
    # exactly zero, i.e. "no risk today", on precisely the quiet windows that take the
    # fallback branch. A zero ES then becomes the denominator of the Acerbi-Szekely
    # statistic and the reported value explodes to order 1e8, which is how the mistake
    # was found: not by a wrong number, but by an impossible one.
    empirical = float(np.quantile(sample, 1.0 - level))

    def empirical_tail() -> tuple[float, float]:
        tail = sample[sample <= empirical]
        return empirical, float(tail.mean()) if tail.size else empirical

    if n_excess < 8:
        # Too few exceedances to estimate two GPD parameters. Degrade to the empirical
        # quantile - the same thing ``tail='empirical'`` would have returned - rather
        # than to some third object with its own sign convention.
        return empirical_tail()
    mean_excess = float(exceedances.mean())
    variance = float(exceedances.var(ddof=1))
    if variance <= 0 or mean_excess <= 0:
        return empirical_tail()
    xi = 0.5 * (1.0 - mean_excess**2 / variance)
    xi = float(np.clip(xi, -0.5, 0.45))
    beta = float(max(mean_excess * (1.0 - xi), 1e-8))
    ratio = float(np.clip((n / n_excess) * (1.0 - level), 1e-8, 0.999))
    if abs(xi) < 1e-6:
        quantile_loss = threshold + beta * (-np.log(ratio))
    else:
        quantile_loss = threshold + beta / xi * (ratio ** (-xi) - 1.0)
    es_loss = quantile_loss + (beta + xi * (quantile_loss - threshold)) / (1.0 - xi)
    return -quantile_loss, -es_loss


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def monte_carlo_var(
    garch: volmodels.GARCH,
    returns: np.ndarray,
    sigma2_path: np.ndarray,
    level: float,
    rng: np.random.Generator,
    paths: int = cfg.MC_PATHS,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the fitted recursion forward and read the quantile off the paths.

    Two things this gets right that the closed-form methods cannot.

    First, the **multi-day** distribution. The recursion is iterated, so the
    variance of the h-day sum reflects the fact that volatility itself is random
    and mean-reverting. The ``sqrt(h)`` rule assumes it is a constant.

    Second, the **asymmetry of the path**. A large negative first-day shock raises
    the variance for every subsequent day, so the left tail of the h-day sum is
    fatter than a scaled one-day tail. This is the mechanism behind the well-known
    result that the square-root rule understates multi-day downside risk even when
    it is calibrated on the right one-day number.

    The innovation is drawn from the model's assumed law - a unit-variance t with
    the fitted degrees of freedom - so the method is only as good as the filter it
    is built on, and the reported table says which filter that was.
    """
    r = np.asarray(returns, dtype=float)
    s2 = np.asarray(sigma2_path, dtype=float)
    n = r.size
    dof = float(garch.params.get("dof", 5.0))
    if not np.isfinite(dof):
        dof = 200.0
    alpha = float(garch.params.get("alpha", 0.05))
    gamma = float(garch.params.get("gamma", 0.0))
    beta = float(garch.params.get("beta", 0.9))
    persistence = alpha + beta + 0.5 * gamma
    omega = (1.0 - persistence) * max(float(garch.params.get("unconditional_variance", np.var(r))), EPS)

    var = np.full(n, np.nan)
    es = np.full(n, np.nan)
    for t in range(1, n):
        start = s2[t]
        if not np.isfinite(start) or start <= 0:
            continue
        z = _draw_innovations(rng, paths, horizon, dof)
        cumulative, _terminal = kernels.garch_terminal_paths(omega, alpha, gamma, beta, z, start, horizon)
        quantile = float(np.quantile(cumulative, 1.0 - level))
        var[t] = max(-quantile, 0.0)
        tail = cumulative[cumulative <= quantile]
        es[t] = max(-float(tail.mean()), 0.0) if tail.size else var[t]
    return var, es


def _draw_innovations(rng: np.random.Generator, paths: int, horizon: int, dof: float) -> np.ndarray:
    """Standardised Student-t draws, shaped (paths, horizon)."""
    if dof > 200:
        return rng.standard_normal((paths, horizon))
    raw = rng.standard_t(dof, size=(paths, horizon))
    return raw * np.sqrt((dof - 2.0) / dof)


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------
@dataclass
class VaRSuite:
    """Every VaR/ES path the study reports, keyed by method name."""

    forecasts: dict[str, RiskForecast]
    index: pd.DatetimeIndex
    level: float
    horizon: int

    def names(self) -> list[str]:
        return list(self.forecasts)

    def var_frame(self) -> pd.DataFrame:
        return pd.DataFrame({name: f.var for name, f in self.forecasts.items()}, index=self.index)

    def es_frame(self) -> pd.DataFrame:
        return pd.DataFrame({name: f.es for name, f in self.forecasts.items()}, index=self.index)


def build_suite(
    level: float,
    horizon: int,
    returns: pd.Series,
    sigma_paths: dict[str, pd.Series],
    garch_model: volmodels.GARCH | None,
    sigma2_garch: pd.Series | None,
    rng: np.random.Generator,
    hs_window: int = cfg.HS_WINDOWS[0],
    paths: int = cfg.MC_PATHS,
) -> VaRSuite:
    """Assemble every VaR method at one confidence level and horizon.

    ``sigma_paths`` maps a volatility-model name to its one-step-ahead volatility
    forecast; the parametric and FHS methods are each run against the model chosen
    for them, and the report says which. Historical simulation needs no model and
    is run once.

    The Monte-Carlo leg is always produced when a GJR filter is available. ``paths``
    is a precision knob, not a switch: a caller that wants a cheaper run lowers it
    and accepts a noisier 99% quantile. Making it a boolean instead once removed the
    only method whose multi-day VaR is not a scaled one-day VaR, which silently
    emptied the horizon-scaling section of the report.
    """
    r = returns.to_numpy(dtype=float)
    index = returns.index
    forecasts: dict[str, RiskForecast] = {}

    def add(name: str, family: str, var: np.ndarray, es: np.ndarray) -> None:
        forecasts[name] = RiskForecast(
            method=name,
            level=level,
            horizon=horizon,
            var=pd.Series(var, index=index, name=name),
            es=pd.Series(es, index=index, name=name),
            family=family,
        )

    var, es = historical_var(r, level, hs_window, horizon)
    add(f"hs_{hs_window}", "historical", var, es)

    var, es = age_weighted_var(r, level, hs_window, cfg.HS_DECAY, horizon)
    add("hs_brw", "historical", var, es)

    # The filtered and parametric legs are attached to the GJR filter, which the
    # model-comparison table selects on QLIKE; running each against every filter
    # would multiply the tables without adding a finding.
    reference = "gjr" if "gjr" in sigma_paths else next(iter(sigma_paths))
    sigma = sigma_paths[reference].to_numpy(dtype=float)

    # Emission order is the declared roster order in ``config.VAR_METHOD_NAMES``, which
    # is grouped by family - historical, filtered, parametric, simulation. Iteration
    # order of ``forecasts`` reaches the coverage table and the frames, so this is the
    # order those are built in, and the roster test pins the two together. Grouping by
    # family puts the two routes that share a construction next to each other, which is
    # the comparison the tables are read for.
    #
    # Only the Monte-Carlo leg below draws from ``rng``, so reordering any of the
    # deterministic legs leaves every number identical. The simulation stays last
    # anyway: moving it would change which draws it gets while leaving the suite
    # looking unchanged, which is the kind of silent difference worth not creating.
    var, es = filtered_historical_var(r, sigma, level, hs_window, horizon, tail="empirical")
    add(f"fhs_{reference}", "filtered", var, es)

    var, es = filtered_historical_var(r, sigma, level, hs_window, horizon, tail="evt")
    add(f"fhs_evt_{reference}", "filtered", var, es)

    var, es = parametric_var(sigma, level, "normal", horizon=horizon)
    add(f"parametric_normal_{reference}", "parametric", var, es)

    dof = float(garch_model.params.get("dof", 5.0)) if garch_model is not None else 5.0
    if not np.isfinite(dof):
        dof = 200.0
    var, es = parametric_var(sigma, level, "t", dof=dof, horizon=horizon)
    add(f"parametric_t_{reference}", "parametric", var, es)

    if garch_model is not None and sigma2_garch is not None:
        var, es = monte_carlo_var(garch_model, r, sigma2_garch.to_numpy(dtype=float), level, rng, paths, horizon)
        add("monte_carlo", "simulation", var, es)

    return VaRSuite(forecasts=forecasts, index=index, level=level, horizon=horizon)


def horizon_returns(returns: pd.Series, horizon: int) -> pd.Series:
    """The h-day return that a VaR at index ``t`` is a statement about.

    The convention is **forward**: the value at index ``t`` is the sum of the
    returns realised on ``t, t+1, ..., t+h-1``, which is what a risk number issued
    at the close of ``t-1`` refers to. The final ``h-1`` entries are undefined and
    are left as ``NaN``.

    This function exists because getting it wrong is silent and produces a
    spectacular result. Comparing a 10-day VaR against a *daily* return makes
    every method look absurdly conservative - essentially zero breaches instead of
    42 in 4,200 days - and the failure looks like a finding rather than a bug.
    Every backtest in this package therefore takes the daily series and the
    horizon and aggregates internally, so the two can never drift apart.
    """
    if horizon <= 1:
        return returns
    forward = returns.rolling(horizon).sum().shift(-(horizon - 1))
    return forward


def horizon_gap(one_day: np.ndarray, ten_day: np.ndarray, horizon: int = 10) -> np.ndarray:
    """How far the h-day VaR departs from the square-root-of-time scaling, in percent.

    Returned as ``h_day_var / (sqrt(h) * one_day_var) - 1``, i.e. a **signed
    fraction**, not a ratio. Positive means the scaling rule *understates* risk -
    the case the literature warns about - and negative means it overstates it,
    which is what happens coming out of a crisis when volatility is high and
    expected to fall. The zero-centred form is what the report's "share of days the
    scaling rule understates" statistic counts, so the sign convention is load
    bearing rather than cosmetic.
    """
    scaled = np.asarray(one_day, dtype=float) * np.sqrt(float(horizon))
    return (
        np.divide(np.asarray(ten_day, dtype=float), scaled, out=np.full_like(scaled, np.nan), where=scaled > EPS) - 1.0
    )
