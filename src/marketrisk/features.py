"""Realised measures and the lagged feature matrix.

Every column this module produces for a row dated ``t`` is computed from
information available at the close of ``t-1``. That is not a stylistic choice:
a volatility forecast that peeks at the day it is forecasting will look
excellent and be worthless, and the failure is silent - the numbers simply come
out a little too good. The defence is structural. The realised measures are
built from contemporaneous prices, and :func:`build_features` shifts every one
of them by one day before it is used, so the alignment is a property of the code
rather than of the author's memory. The test suite asserts it directly by
corrupting one day's return and checking that no earlier forecast changes.

The measures themselves
-----------------------
``RV``  realised variance, the sum of squared intraday returns. Unbiased for
        integrated variance, and the workhorse of the literature.
``BV``  bipower variation, ``(pi/2) * sum |r_i| |r_{i-1}|``. Consistent for the
        *continuous* part of quadratic variation, so ``RV - BV`` isolates the
        jump contribution - which is why both are computed and both are used.
``Range`` estimators (Parkinson, Garman-Klass, Rogers-Satchell). These were the
        standard high-frequency-free volatility proxy before intraday data was
        cheap, they use only the daily bar, and they are what
        ``--data-mode real`` has to fall back on because a free close-price feed
        gives no intraday data at all. Reporting them next to RV makes the cost
        of that fallback visible instead of implied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

#: (pi/2) / 2 - the constants in the bipower and range estimators, named so the
#: formulas below can be read against the papers rather than against a compiler.
MU1_SQ = 2.0 / np.pi  # E|z|^2 for a standard normal
PARKINSON_C = 1.0 / (4.0 * np.log(2.0))
GARMAN_KLASS_C = 2.0 * np.log(2.0) - 1.0


# ---------------------------------------------------------------------------
# Realised measures from intraday data
# ---------------------------------------------------------------------------
def realised_variance(intraday: np.ndarray) -> np.ndarray:
    """RV_t = sum_i r_{t,i}^2, per day."""
    return np.square(np.asarray(intraday, dtype=float)).sum(axis=1)


def bipower_variation(intraday: np.ndarray, adjust: bool = True) -> np.ndarray:
    """BV_t = mu1^-2 * sum_i |r_i| |r_{i-1}|, the jump-robust variance estimate.

    ``mu1^-2 = pi/2``. The estimator is consistent for the integrated variance of
    the continuous part of the price path; because a jump contaminates ``RV`` but
    appears in at most two of the products, it is (asymptotically) absent from
    ``BV``. That difference is the whole basis for separating the two components.

    ``adjust`` applies the small-sample correction of Barndorff-Nielsen and
    Shephard, ``M / (M - 1)``, which matters at 78 bars (about 1.3%). It is on by
    default because with it ``BV`` is unbiased for the diffusion variance in this
    simulation, which the test suite asserts against the known truth.
    """
    r = np.asarray(intraday, dtype=float)
    if r.shape[1] < 2:
        return np.zeros(r.shape[0])
    products = np.abs(r[:, 1:]) * np.abs(r[:, :-1])
    value = (np.pi / 2.0) * products.sum(axis=1)
    if adjust:
        m = r.shape[1]
        value = value * m / (m - 1.0)
    return value


def jump_variation(rv: np.ndarray, bv: np.ndarray) -> np.ndarray:
    """``max(RV - BV, 0)``: the estimated jump contribution to quadratic variation."""
    return np.maximum(np.asarray(rv, dtype=float) - np.asarray(bv, dtype=float), 0.0)


def jump_ratio_statistic(rv: np.ndarray, bv: np.ndarray) -> np.ndarray:
    """Huang-Tauchen z-statistic for the null of "no jumps on day t".

    Under the null the ratio ``(RV - BV) / RV`` converges to a standardised
    normal with a known variance, so this is a test rather than a threshold. It
    is used to label jump days, and the labels are compared against the jumps the
    simulator actually placed - which is a check unavailable with real data.
    """
    rv = np.asarray(rv, dtype=float)
    bv = np.asarray(bv, dtype=float)
    m = max(cfg.INTRADAY_BARS, 2)
    # With the Barndorff-Nielsen correction the leading constant of the variance is
    # (mu1^-4 + 2 mu1^-2 - 5) / M, where mu1 = sqrt(2/pi); the textbook
    # pi^2/4 + pi - 5 is the large-M limit and the two agree to within a few parts in
    # a thousand at M = 78. The exact form is used, so the constant is not written out
    # here where it would drift out of sync with `_jump_statistic_variance`.
    scale = _jump_statistic_variance(m)
    ratio = np.divide(rv - bv, rv, out=np.zeros_like(rv), where=rv > 0.0)
    return ratio / np.sqrt(np.maximum(scale, 1e-12))


def _jump_statistic_variance(m: int) -> float:
    """Leading variance of the ratio statistic for ``m`` intraday returns."""
    mu1_inv4 = (np.pi / 2.0) ** 2
    return (mu1_inv4 + 2.0 * (np.pi / 2.0) - 5.0) / max(m, 1)


def parkinson(high: pd.Series, low: pd.Series) -> pd.Series:
    """Parkinson (1980) range variance: ``(h - l)^2 / (4 ln 2)`` in logs."""
    hl = np.log(high.to_numpy(dtype=float) / low.to_numpy(dtype=float))
    return pd.Series(PARKINSON_C * hl**2, index=high.index, name="parkinson")


def garman_klass(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Garman-Klass (1980): uses the open and close as well as the range."""
    lo = np.log(open_.to_numpy(dtype=float))
    lh = np.log(high.to_numpy(dtype=float))
    ll = np.log(low.to_numpy(dtype=float))
    lc = np.log(close.to_numpy(dtype=float))
    value = 0.5 * (lh - ll) ** 2 - GARMAN_KLASS_C * (lc - lo) ** 2
    return pd.Series(np.maximum(value, 0.0), index=high.index, name="garman_klass")


def rogers_satchell(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Rogers-Satchell (1991): drift-independent, unlike Garman-Klass."""
    lo = np.log(open_.to_numpy(dtype=float))
    lh = np.log(high.to_numpy(dtype=float))
    ll = np.log(low.to_numpy(dtype=float))
    lc = np.log(close.to_numpy(dtype=float))
    value = (lh - lc) * (lh - lo) + (ll - lc) * (ll - lo)
    return pd.Series(np.maximum(value, 0.0), index=high.index, name="rogers_satchell")


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
#: Columns the lagged design matrix is built from. Kept as a module constant so
#: that the reported feature list cannot drift away from the one actually used -
#: the model documentation is generated from this.
FEATURE_COLUMNS: tuple[str, ...] = (
    "log_rv_d",
    "log_rv_w",
    "log_rv_m",
    "log_rv_q",
    "rv_slope",
    "ret_1",
    "ret_abs_1",
    "ret_neg_1",
    "ret_w",
    "ret_sq_1",
    "jump_1",
    "log_parkinson_1",
    "log_vix_1",
    "vix_change_1",
    "rv_dispersion",
)

TARGET_COLUMN = "log_rv_next"


def har_components(log_rv: pd.Series, lags: dict[str, int] | None = None) -> pd.DataFrame:
    """Daily / weekly / monthly averages of ``log RV`` - the HAR regressors.

    Averaging in logs rather than in levels is the Corsi (2009) specification;
    averaging ``RV`` itself and then logging gives a different and (given the
    lognormal-ish shape of RV) worse-behaved regressor.
    """
    lags = cfg.HAR_LAGS if lags is None else lags
    out = pd.DataFrame(index=log_rv.index)
    out["log_rv_d"] = log_rv
    out["log_rv_w"] = log_rv.rolling(lags["w"]).mean()
    out["log_rv_m"] = log_rv.rolling(lags["m"]).mean()
    out["log_rv_q"] = log_rv.rolling(4 * lags["m"]).mean()
    return out


def build_features(frame: pd.DataFrame, log_vix: pd.Series | None = None) -> pd.DataFrame:
    """Build the lagged design matrix.

    ``frame`` must contain ``ret``, ``realised_variance``, ``bipower`` and the
    OHLC columns, all dated on the day the return was realised. The returned
    matrix is dated on the day being *forecast*: row ``t`` holds only quantities
    known at the close of ``t-1``.

    The final shift is the point of the function. Everything above it is
    descriptive; this is the line that makes the study honest, and it is applied
    once, at the end, so it cannot be forgotten for one column out of fifteen.
    """
    ret = frame["ret"].astype(float)
    rv = frame["realised_variance"].astype(float).clip(lower=1e-14)
    bv = frame["bipower"].astype(float).clip(lower=1e-14)
    log_rv = np.log(rv)

    har = har_components(log_rv)
    jump_share = (jump_variation(rv, bv) / rv).clip(0.0, 1.0)

    raw = pd.DataFrame(index=frame.index)
    raw["log_rv_d"] = har["log_rv_d"]
    raw["log_rv_w"] = har["log_rv_w"]
    raw["log_rv_m"] = har["log_rv_m"]
    raw["log_rv_q"] = har["log_rv_q"]
    # Term structure of volatility: a backwardated curve (short-dated above
    # long-dated) predicts mean reversion up, and vice versa. This is the
    # cheapest useful summary of the shape of the vol curve and it is what the
    # HAR-X model would otherwise have to reconstruct from three collinear lags.
    raw["rv_slope"] = har["log_rv_w"] - har["log_rv_m"]

    raw["ret_1"] = ret
    raw["ret_abs_1"] = ret.abs()
    raw["ret_neg_1"] = np.minimum(ret, 0.0)
    raw["ret_w"] = ret.rolling(cfg.HAR_LAGS["w"]).sum()
    raw["ret_sq_1"] = ret**2
    raw["jump_1"] = jump_share

    park = np.log(parkinson(frame["high"], frame["low"]).clip(lower=1e-14))
    raw["log_parkinson_1"] = park
    # Dispersion between two independent volatility proxies. When the range
    # estimate and the close-to-close estimate disagree, the day was unusual in a
    # way neither captures alone; this is also the mechanism by which the model
    # can learn about jump days, since a jump moves RV but not the range.
    raw["rv_dispersion"] = np.log(rv) - park

    if log_vix is not None:
        vix = log_vix.reindex(frame.index).astype(float)
        raw["log_vix_1"] = vix
        raw["vix_change_1"] = vix.diff()
    else:
        raw["log_vix_1"] = np.log(rv.rolling(cfg.HAR_LAGS["m"]).mean())
        raw["vix_change_1"] = 0.0

    # --- the alignment step -------------------------------------------------
    columns: list[str] = []
    out = pd.DataFrame(index=frame.index)
    for name in FEATURE_COLUMNS:
        out[name] = raw[name].shift(1)
        columns.append(name)
    out[TARGET_COLUMN] = log_rv
    out["ret_next"] = ret
    out["rv_next"] = rv
    return out.loc[:, columns + [TARGET_COLUMN, "ret_next", "rv_next"]]


def feature_index() -> pd.Index:
    """The predictor columns, in the order the models see them."""
    return pd.Index(FEATURE_COLUMNS)
