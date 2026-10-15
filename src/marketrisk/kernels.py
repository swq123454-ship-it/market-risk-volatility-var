"""Inner recursive kernels, in one place, with an optional Numba fast path.

The conditional-variance recursions are scalar loops: sigma^2_t depends on
sigma^2_{t-1}, so they cannot be vectorised the way most of this codebase can.
On a 5,000-day sample a plain Python loop costs about 3 ms, which sounds
harmless until you remember that a maximum-likelihood fit evaluates the loop a
hundred times and the walk-forward backtest refits the model two hundred times.
That is the difference between a three-minute study and a thirty-minute one.

Numba is therefore used when it is importable and silently skipped when it is
not. It is deliberately *not* a required dependency: the fallback is exact (the
tests assert the two paths agree to machine precision), so a reviewer who does
not want a JIT toolchain installed loses speed and nothing else.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

try:  # pragma: no cover - exercised only when numba is installed
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        """No-op stand-in for ``numba.njit``."""

        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrap(func):
            return func

        return wrap


#: E|z| for a standard normal, sqrt(2/pi). The EGARCH kernel needs it, and so
#: does the analytic expected-shortfall formula for the Gaussian case.
E_ABS_NORMAL = float(np.sqrt(2.0 / np.pi))


@njit(cache=False, fastmath=True)
def garch_variance_path(omega, alpha, gamma, beta, returns, backcast):
    """GJR-GARCH(1,1,1) conditional-variance path.

    ``sigma2[t]`` is the *conditional variance of return t* estimated from
    returns up to ``t-1``. The recursion is

        sigma2_t = omega + (alpha + gamma * 1[r_{t-1} < 0]) * r_{t-1}^2
                   + beta * sigma2_{t-1}

    Setting ``gamma = 0`` reduces this to plain GARCH(1,1), which is why one
    kernel serves both models. The seed variance is a decay-weighted backcast of
    the sample, the convention the ``rugarch`` / ``arch`` packages use; seeding
    with the sample variance instead would make the first few hundred
    likelihood evaluations depend on data that is not yet in the information set.
    """
    n = returns.shape[0]
    sigma2 = np.empty(n)
    prev = backcast
    for t in range(n):
        r = returns[t - 1] if t > 0 else 0.0
        val = omega + alpha * r * r + beta * prev
        if r < 0.0:
            val += gamma * r * r
        if val <= 1e-12:
            val = 1e-12
        sigma2[t] = val
        prev = val
    return sigma2


@njit(cache=False, fastmath=True)
def egarch_log_variance_path(omega, alpha, gamma, beta, returns, backcast_log, e_abs):
    """EGARCH(1,1) log-variance path.

        ln sigma2_t = omega + beta * ln sigma2_{t-1}
                      + alpha * (|z_{t-1}| - E|z|) + gamma * z_{t-1}

    The recursion is in logs, so variance stays positive without parameter
    constraints, and the ``gamma * z`` term makes the leverage effect a *level*
    effect in log variance rather than the quadratic one GJR imposes.

    ``e_abs`` is E|z| under the assumed innovation law: sqrt(2/pi) for Gaussian
    shocks, and the Student-t analogue otherwise. Getting this wrong shifts the
    whole log-variance path by a constant, which the intercept happily absorbs -
    the fit still looks fine and the reported ``omega`` is silently wrong.
    """
    n = returns.shape[0]
    log_sigma2 = np.empty(n)
    prev = backcast_log
    for t in range(n):
        if t == 0:
            val = omega + beta * prev
        else:
            z = returns[t - 1] * np.exp(-0.5 * prev)
            val = omega + beta * prev + alpha * (abs(z) - e_abs) + gamma * z
        if val > 30.0:
            val = 30.0
        elif val < -30.0:
            val = -30.0
        log_sigma2[t] = val
        prev = val
    return log_sigma2


@njit(cache=False, fastmath=True)
def ewma_variance_path(lam, returns, init_var, demean):
    """RiskMetrics / EWMA variance path: sigma2_t = lam * sigma2_{t-1} + (1-lam) * r_{t-1}^2."""
    n = returns.shape[0]
    sigma2 = np.empty(n)
    prev = init_var
    for t in range(n):
        r = returns[t - 1] - demean if t > 0 else 0.0
        val = lam * prev + (1.0 - lam) * r * r
        if val <= 1e-12:
            val = 1e-12
        sigma2[t] = val
        prev = val
    return sigma2


@njit(cache=False, fastmath=True)
def garch_simulate_path(omega, alpha, gamma, beta, z, sigma2_0):
    """Forward-simulate a GJR-GARCH variance path from standardised shocks ``z``."""
    n = z.shape[0]
    sigma2 = np.empty(n)
    prev = sigma2_0
    for t in range(n):
        val = omega + beta * prev
        if t > 0:
            r = np.sqrt(prev) * z[t - 1]
            val += alpha * r * r + gamma * (r * r if r < 0.0 else 0.0)
        if val <= 1e-12:
            val = 1e-12
        sigma2[t] = val
        prev = val
    return sigma2


@njit(cache=False, fastmath=True)
def garch_terminal_paths(omega, alpha, gamma, beta, z, sigma2_start, horizon):
    """Simulate ``horizon``-step-ahead variance shocks across many Monte-Carlo paths.

    ``z`` has shape (n_paths, horizon). Returns cumulative log return and the
    terminal variance per path, which is all the multi-step VaR needs. Iterating
    the variance recursion explicitly is the entire reason Monte-Carlo and the
    square-root-of-time rule disagree: the recursion mean-reverts, so the
    h-day variance grows *slower* than h times the one-day variance.
    """
    n_paths = z.shape[0]
    cum_ret = np.empty(n_paths)
    terminal = np.empty(n_paths)
    for i in range(n_paths):
        prev = sigma2_start
        total = 0.0
        for h in range(horizon):
            sigma = np.sqrt(prev)
            r = sigma * z[i, h]
            total += r
            val = omega + beta * prev + alpha * r * r + gamma * (r * r if r < 0.0 else 0.0)
            if val <= 1e-12:
                val = 1e-12
            prev = val
        cum_ret[i] = total
        terminal[i] = prev
    return cum_ret, terminal


def standardised_abs_expectation(dof: float) -> float:
    """E|z| for a Student-t innovation standardised to unit variance.

    Needed by the EGARCH recursion and by the fat-tail VaR formula. The naive
    ``sqrt(2/(pi*(nu-2)))`` that circulates online is wrong for small ``nu``;
    this evaluates the exact integral
    ``E|z| = 2 sqrt(nu-2) Gamma((nu+1)/2) / ((nu-1) sqrt(pi) Gamma(nu/2))``.
    """
    if not np.isfinite(dof) or dof > 200:
        return float(np.sqrt(2.0 / np.pi))
    ln = (
        np.log(2.0)
        + 0.5 * np.log(dof - 2.0)
        + gammaln((dof + 1.0) / 2.0)
        - np.log(dof - 1.0)
        - 0.5 * np.log(np.pi)
        - gammaln(dof / 2.0)
    )
    return float(np.exp(ln))


def student_t_logpdf(z: np.ndarray, dof: float) -> np.ndarray:
    """Log density of a unit-variance Student-t, evaluated at standardised ``z``."""
    if dof > 200:
        return -0.5 * (np.log(2.0 * np.pi) + z * z)
    c = gammaln((dof + 1.0) / 2.0) - gammaln(dof / 2.0) - 0.5 * np.log(np.pi * (dof - 2.0))
    return c - 0.5 * (1.0 + dof) * np.log1p(z * z / (dof - 2.0))


def normal_logpdf(z: np.ndarray) -> np.ndarray:
    """Log density of a standard normal."""
    return -0.5 * (np.log(2.0 * np.pi) + z * z)


def variance_backcast(returns: np.ndarray, decay: float = 0.94) -> float:
    """Decay-weighted unconditional variance used to seed the recursions."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 1e-8
    n = r.size
    weights = decay ** np.arange(n)[::-1]
    weights = weights / weights.sum()
    return float(max(np.dot(weights, r * r), 1e-12))
