"""Forecast scoring rules, in one place.

Two kinds of forecast are scored in this study and they need different losses.
Conflating them is the most common way a volatility comparison ends up ranking
models by their outliers.

**Variance forecasts.** ``MSE`` on variance is dominated by the handful of
largest observations: an error of 5 variance units on a crisis day outweighs a
thousand tiny errors on calm days, so the ranking is decided by three or four
observations. The volatility literature therefore uses ``QLIKE``,

    QLIKE(sigma2_hat, proxy) = proxy / sigma2_hat - log(proxy / sigma2_hat) - 1

which is the loss implied by a Gaussian likelihood, is robust to noise in the
proxy (Patton 2011), and is minimised in expectation at the true conditional
variance. Both are reported; neither is treated as the single truth.

**Quantile forecasts.** A VaR is a quantile, so it is scored with the asymmetric
tick (quantile) loss

    TL(alpha) = (alpha - 1[r < -VaR]) * (r + VaR)

which is minimised by the true quantile. Squared error is the wrong loss here:
a VaR that is too conservative is penalised quadratically even though it costs
the desk only the opportunity cost of capital, while a VaR that is too
aggressive - the failure that actually matters - is penalised only through the
exceedance it happens to produce. Exceedance counts alone cannot rank two models
that both hit their target rate, which is why the tick loss is needed on top of
the coverage tests.

``proxy`` is deliberately a separate argument everywhere. Comparing a forecast
against a *measured* variance (realised variance) and against the *latent*
conditional variance are different questions, and in synthetic mode this study
can ask both.
"""

from __future__ import annotations

import numpy as np


def qlike(forecast: np.ndarray, proxy: np.ndarray) -> np.ndarray:
    """QLIKE loss, per observation. Lower is better.

    ``proxy / forecast - log(proxy / forecast) - 1``, evaluated on the ratio so
    that it is scale-free: doubling every forecast and every proxy leaves the
    loss unchanged.
    """
    f = np.asarray(forecast, dtype=float)
    p = np.asarray(proxy, dtype=float)
    ratio = np.divide(p, f, out=np.ones_like(p), where=(f > 0.0) & (p > 0.0))
    ratio = np.maximum(ratio, 1e-12)
    return ratio - np.log(ratio) - 1.0


def mse_variance(forecast: np.ndarray, proxy: np.ndarray) -> np.ndarray:
    """Squared error on variance. Reported next to QLIKE, never instead of it."""
    return (np.asarray(forecast, dtype=float) - np.asarray(proxy, dtype=float)) ** 2


def mse_volatility(forecast: np.ndarray, proxy: np.ndarray) -> np.ndarray:
    """Squared error on standard deviation - the units a desk quotes in."""
    f = np.sqrt(np.maximum(np.asarray(forecast, dtype=float), 0.0))
    p = np.sqrt(np.maximum(np.asarray(proxy, dtype=float), 0.0))
    return (f - p) ** 2


def tick_loss(returns: np.ndarray, var: np.ndarray, level: float) -> np.ndarray:
    """Asymmetric quantile loss for a VaR forecast, in return units.

    ``var`` is entered as a *positive* loss magnitude, so the breach condition is
    ``r < -var``, and ``level`` is the coverage probability (0.99 for a 99% VaR).
    The loss is the standard tick loss for the ``1 - level`` quantile of the return,

        TL = ((1 - level) - 1[r < -var]) * (r + var)

    Note which of the two probabilities appears in the coefficient. It is the
    **tail** probability, ``1 - level``, and not the coverage level. Substituting
    ``level`` is a one-token change that inverts the objective: the non-breach
    weight becomes 0.99, the loss then grows almost linearly in ``var``, and the
    minimiser moves to ``var = 0``. A table scored that way ranks methods by how
    little risk they report - the exact opposite of the intent - and nothing about
    it looks broken, because the table still sorts and still names a winner. On a
    sample from a known distribution the correct form is minimised at the sample
    quantile and the inverted one is minimised at zero; both are asserted in the
    test suite.
    """
    r = np.asarray(returns, dtype=float)
    v = np.asarray(var, dtype=float)
    tail = 1.0 - float(level)
    breach = (r < -v).astype(float)
    return (tail - breach) * (r + v)


def tick_loss_scaled(returns: np.ndarray, var: np.ndarray, level: float, scale: float = 1e4) -> np.ndarray:
    """Tick loss in basis points, for reporting."""
    return tick_loss(returns, var, level) * scale


def mean_or_nan(values: np.ndarray) -> float:
    """Mean of a possibly empty array, without a warning."""
    values = np.asarray(values, dtype=float)
    return float(values.mean()) if values.size else float("nan")


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, hac_lags: int = 0) -> tuple[float, float]:
    """Diebold-Mariano test that two forecast losses have equal expectation.

    Returns ``(statistic, two_sided_p_value)``. The variance is Newey-West with
    ``hac_lags`` lags, which matters here because the loss differentials are
    serially correlated through volatility clustering: a naive standard error is
    far too small and would declare almost everything significant.

    Positive statistic means model A has the *higher* loss, i.e. B is better.
    """
    from scipy import stats

    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 4:
        return float("nan"), float("nan")
    mean = d.mean()
    demeaned = d - mean
    gamma0 = float(np.dot(demeaned, demeaned) / n)
    variance = gamma0
    for lag in range(1, min(hac_lags, n - 1) + 1):
        cov = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
        variance += 2.0 * (1.0 - lag / (hac_lags + 1.0)) * cov
    if variance <= 0.0:
        return float("nan"), float("nan")
    statistic = mean / np.sqrt(variance / n)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(statistic)))
    return float(statistic), float(p_value)
