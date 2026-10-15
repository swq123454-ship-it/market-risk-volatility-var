"""Model comparison: scoring, ranking, and the tests that say whether a gap is real.

Ranking a dozen volatility models produces a table in which the top few are always
close together, and the question a reader actually has is whether the ordering
means anything. Four devices are used here and each answers a different part of
that question.

**QLIKE and MSE.** Two losses, both reported. QLIKE is the one to rank on because
MSE is decided by a handful of crisis days, but MSE is included because it is the
loss a risk committee recognises and because a model that wins on QLIKE while
losing badly on MSE is telling you something about where its errors sit.

**Mincer-Zarnowitz regression.** ``proxy = a + b * forecast + e``. The joint test
``a = 0, b = 1`` is the formal statement of "the forecast is unbiased and
correctly scaled". A model can have a good QLIKE and still fail it, which means it
is right on average for the wrong reasons - and that is worth knowing before it is
used for anything.

**Diebold-Mariano with a Newey-West variance.** Volatility loss differentials are
strongly serially correlated, so a naive standard error is far too small and would
declare the top six models significantly different from each other. The lag length
is set from the forecast horizon and reported.

**Ranking by regime.** An aggregate score is an average over calm and violent
days. A model can win the average by being excellent in the quiet 70% and useless
in the 5% that matters. Splitting the QLIKE by regime is what stops a ranking from
being decided by a sample the desk does not care about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg
from . import losses


@dataclass
class MincerZarnowitz:
    """Slope, intercept and the joint test that the forecast is calibrated."""

    intercept: float
    slope: float
    intercept_se: float
    slope_se: float
    r_squared: float
    joint_statistic: float
    joint_p_value: float
    observations: int

    @property
    def calibrated(self) -> bool:
        return self.joint_p_value > cfg.ALPHA_SIGNIFICANCE


def mincer_zarnowitz(forecast: np.ndarray, proxy: np.ndarray) -> MincerZarnowitz:
    """Regress the proxy on the forecast and jointly test ``a = 0, b = 1``.

    Run in logs, which is where a variance forecast should be assessed. A
    regression in levels is dominated by the same few observations as MSE, and
    its ``R^2`` mostly measures whether the model got the crisis right.
    """
    f = np.asarray(forecast, dtype=float)
    p = np.asarray(proxy, dtype=float)
    mask = np.isfinite(f) & np.isfinite(p) & (f > 0) & (p > 0)
    f, p = np.log(f[mask]), np.log(p[mask])
    n = f.size
    if n < 30:
        return MincerZarnowitz(*(float("nan"),) * 7, observations=n)
    x = np.column_stack([np.ones(n), f])
    beta, *_ = np.linalg.lstsq(x, p, rcond=None)
    residual = p - x @ beta
    dof = n - 2
    sigma2 = float(residual @ residual / dof)
    xtx_inv = np.linalg.inv(x.T @ x)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    r_squared = 1.0 - float(residual @ residual) / float(((p - p.mean()) ** 2).sum())
    # Wald test of the two linear restrictions.
    r = np.array([0.0, 1.0])
    diff = beta - r
    statistic = float(diff @ np.linalg.inv(xtx_inv * sigma2) @ diff)
    p_value = float(1.0 - stats.chi2.cdf(statistic, 2))
    return MincerZarnowitz(
        intercept=float(beta[0]),
        slope=float(beta[1]),
        intercept_se=float(se[0]),
        slope_se=float(se[1]),
        r_squared=float(r_squared),
        joint_statistic=statistic,
        joint_p_value=p_value,
        observations=n,
    )


def volatility_scores(
    forecasts: pd.DataFrame,
    proxy: pd.Series,
    latent: pd.Series | None = None,
) -> pd.DataFrame:
    """Score every model's variance forecast against a proxy, and optionally the truth.

    Two columns sets are produced when ``latent`` is available.

    The **proxy** is realised variance - what a desk can observe, and the standard
    benchmark in the literature.

    The **latent** column is the simulated conditional variance, which only exists
    because the data are synthetic. It is reported because it separates two
    failure modes that the proxy conflates: a model can be badly wrong about the
    *level* of risk, or merely unable to track the noise in its own measure. A
    GARCH filter estimates the diffusion variance and the RV proxy also contains
    jumps, so against the latent target it carries a known upward bias of about
    8%; that is a property of the comparison and is stated rather than corrected.
    """
    common = forecasts.dropna(how="all").index
    proxy_aligned = proxy.reindex(common)
    rows = []
    for name in forecasts.columns:
        f = forecasts[name].reindex(common)
        mask = f.notna() & proxy_aligned.notna()
        ql = losses.qlike(f[mask].to_numpy(), proxy_aligned[mask].to_numpy())
        se = losses.mse_variance(f[mask].to_numpy(), proxy_aligned[mask].to_numpy())
        sv = losses.mse_volatility(f[mask].to_numpy(), proxy_aligned[mask].to_numpy())
        mz = mincer_zarnowitz(f[mask].to_numpy(), proxy_aligned[mask].to_numpy())
        row = {
            "model": name,
            "observations": int(mask.sum()),
            "qlike": float(np.nanmean(ql)),
            "mse_variance": float(np.nanmean(se)),
            "mse_vol_bp": float(np.nanmean(sv)) * 1e8,
            "mz_slope": mz.slope,
            "mz_r2_log": mz.r_squared,
            "mz_joint_p": mz.joint_p_value,
            "mz_calibrated": mz.calibrated,
            "mean_forecast_vol_annual": float(np.sqrt(np.nanmean(f[mask]) * cfg.TRADING_DAYS)),
        }
        if latent is not None:
            lat = latent.reindex(common)
            mask2 = f.notna() & lat.notna()
            row["qlike_latent"] = float(np.nanmean(losses.qlike(f[mask2].to_numpy(), lat[mask2].to_numpy())))
            row["mz_r2_log_latent"] = mincer_zarnowitz(f[mask2].to_numpy(), lat[mask2].to_numpy()).r_squared
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sort_values("qlike").reset_index(drop=True)


def pairwise_diebold_mariano(forecasts: pd.DataFrame, proxy: pd.Series, hac_lags: int = 10) -> pd.DataFrame:
    """Matrix of DM p-values on QLIKE differences, models in the same order as columns.

    Reported as p-values rather than statistics because the useful reading is
    "which of these gaps survives its own standard error", and a matrix of
    statistics invites the reader to compare magnitudes across pairs, which is not
    meaningful.
    """
    names = list(forecasts.columns)
    proxy_aligned = proxy.reindex(forecasts.index)
    qlikes = {}
    for name in names:
        f = forecasts[name]
        mask = f.notna() & proxy_aligned.notna()
        values = np.full(len(f), np.nan)
        values[mask.to_numpy()] = losses.qlike(f[mask].to_numpy(), proxy_aligned[mask].to_numpy())
        qlikes[name] = values
    out = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pair = np.isfinite(qlikes[a]) & np.isfinite(qlikes[b])
            if pair.sum() < 30:
                continue
            _stat, p_value = losses.diebold_mariano(qlikes[a][pair], qlikes[b][pair], hac_lags)
            out.loc[a, b] = p_value
            out.loc[b, a] = p_value
    return out


def qlike_by_regime(
    forecasts: pd.DataFrame, proxy: pd.Series, windows: dict[str, tuple[str, str]] | None = None
) -> pd.DataFrame:
    """Average QLIKE inside each declared regime, plus the calm remainder.

    The "calm" row is everything outside the stress windows, which is the largest
    part of the sample and the part an aggregate score mostly reflects.
    """
    windows = cfg.STRESS_WINDOWS if windows is None else windows
    regimes: dict[str, pd.Index] = {}
    all_index = forecasts.index
    covered = pd.Index([])
    for name, (start, end) in windows.items():
        slice_index = all_index[(all_index >= pd.Timestamp(start)) & (all_index <= pd.Timestamp(end))]
        regimes[name] = slice_index
        covered = covered.union(slice_index)
    regimes["calm (all other days)"] = all_index.difference(covered)

    proxy_aligned = proxy.reindex(all_index)
    rows = []
    for name in forecasts.columns:
        row = {"model": name}
        for regime, idx in regimes.items():
            f = forecasts[name].reindex(idx)
            p = proxy_aligned.reindex(idx)
            mask = f.notna() & p.notna()
            row[regime] = (
                float(np.nanmean(losses.qlike(f[mask].to_numpy(), p[mask].to_numpy()))) if mask.any() else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def tick_loss_table(suite, returns: pd.Series) -> pd.DataFrame:
    """Rank VaR methods by tick loss, with the mean VaR level for context.

    ``returns`` is the daily series; the h-day aggregation is done here from
    ``suite.horizon`` so that the loss and the forecast refer to the same window.

    The mean VaR column matters when reading the ranking: a method can win on tick
    loss by being systematically conservative, and the tick loss already penalises
    that - but the column makes the trade visible instead of implicit.
    """
    from .var import horizon_returns

    r = horizon_returns(returns, suite.horizon).to_numpy(dtype=float)
    rows = []
    for name, forecast in suite.forecasts.items():
        v = forecast.var.to_numpy(dtype=float)
        mask = np.isfinite(v) & np.isfinite(r)
        if mask.sum() < 30:
            continue
        tl = losses.tick_loss_scaled(r[mask], v[mask], suite.level)
        rows.append(
            {
                "method": name,
                "family": forecast.family,
                "level": suite.level,
                "horizon": suite.horizon,
                "observations": int(mask.sum()),
                "tick_loss_bp": float(np.nanmean(tl)),
                "mean_var_bp": float(np.nanmean(v[mask])) * 1e4,
                "mean_es_bp": float(np.nanmean(forecast.es.to_numpy(dtype=float)[mask])) * 1e4,
            }
        )
    return pd.DataFrame(rows).sort_values("tick_loss_bp").reset_index(drop=True)


def es_backtest_table(suite, returns: pd.Series) -> pd.DataFrame:
    """Acerbi-Szekely Z1 test that the ES is consistent with the tail it describes.

    ES is only backtestable jointly with VaR: the statistic is the average, over
    breach days, of ``-r / ES - 1``, which should be centred on zero when the ES
    is right. A model whose ES is right but whose VaR is too tight will fail the
    coverage test and pass this one - and that combination means the losses are
    larger than the VaR admits but the amount beyond it is correctly sized, which
    is a diagnosis no single test provides.

    Under the null the statistic is asymptotically normal with variance
    ``1 / (n * alpha)``; the reported p-value is two-sided.
    """
    from .var import horizon_returns

    r = horizon_returns(returns, suite.horizon).to_numpy(dtype=float)
    rows = []
    for name, forecast in suite.forecasts.items():
        v = forecast.var.to_numpy(dtype=float)
        e = forecast.es.to_numpy(dtype=float)
        mask = np.isfinite(v) & np.isfinite(e) & np.isfinite(r)
        r_m, v_m, e_m = r[mask], v[mask], e[mask]
        breach = r_m < -v_m
        n = breach.sum()
        if n < 5:
            continue
        ratio = (-r_m[breach] / np.maximum(e_m[breach], 1e-12)) - 1.0
        statistic = float(ratio.mean() / np.sqrt(1.0 / (n * (1.0 - suite.level))))
        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(statistic))))
        rows.append(
            {
                "method": name,
                "family": forecast.family,
                "level": suite.level,
                "horizon": suite.horizon,
                "breaches": int(n),
                "mean_es_bp": float(np.nanmean(e_m)) * 1e4,
                "mean_excess_bp": float(np.mean(-r_m[breach] - e_m[breach])) * 1e4,
                "z1": statistic,
                "z1_p": p_value,
                "es_adequate": p_value > cfg.ALPHA_SIGNIFICANCE,
            }
        )
    # Sorted within the (level, horizon) block and not across it: the four blocks are
    # different questions - the same statistic under a different level and horizon is
    # not comparable - and a single sort merges them into a table whose rows cannot be
    # told apart, since two of them share a method name and nothing else distinguishes
    # a 95%/1-day row from a 99%/10-day one.
    return pd.DataFrame(rows).sort_values(["level", "horizon", "z1"]).reset_index(drop=True)


def feature_importance(model) -> pd.DataFrame:
    """Gain-based importance for the tree model, sorted."""
    inner = getattr(model, "model", None)
    columns = [c for c in _FEATURE_ORDER]
    if inner is None:
        return pd.DataFrame(columns=["feature", "gain"])
    booster = inner.booster_
    gain = booster.feature_importance(importance_type="gain")
    names = booster.feature_name()
    frame = pd.DataFrame({"feature": names, "gain": gain}).sort_values("gain", ascending=False)
    _ = columns
    return frame.reset_index(drop=True)


_FEATURE_ORDER: tuple[str, ...] = (
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
