"""Figures.

One rule shapes everything here: **every axis carries units and every figure
states the sample it was drawn on.** A volatility chart with an unlabelled y-axis
is decoration, and a chart drawn on a different sample from the table next to it
is worse than that. The sample is passed into each function rather than
discovered, so the caller has to decide.

Colours are drawn from a small palette so that a method keeps the same colour
across every figure it appears in - ``gjr`` is the same blue in the forecast path,
the QLIKE ranking and the coverage chart. Reading across figures is then possible
without a legend.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as cfg  # noqa: E402

INK = "#1b1b1f"
MUTED = "#6b6b76"
GRID = "#d8d8e0"
ACCENT = "#1f6feb"
ACCENT_2 = "#d1495b"
ACCENT_3 = "#2a9d8f"
ACCENT_4 = "#e8871e"
ACCENT_5 = "#7d5ba6"
PAPER = "#ffffff"

#: Stable colour per model, so a model keeps its colour across figures.
MODEL_COLOURS: dict[str, str] = {
    "rolling_21": MUTED,
    "rolling_63": "#a8a8b3",
    "ewma_94": ACCENT_4,
    "ewma_fitted": "#f0b357",
    "garch": ACCENT_5,
    "gjr": ACCENT,
    "egarch": "#4f9dea",
    "har": ACCENT_3,
    "harx": "#1b7f74",
    "gbm": ACCENT_2,
    "gru": "#b5179e",
}


def _model_colour(name: str, position: int = 0) -> str:
    """Colour for a model, falling back to the palette by position."""
    if name in MODEL_COLOURS:
        return MODEL_COLOURS[name]
    palette = (ACCENT, ACCENT_2, ACCENT_3, ACCENT_4, ACCENT_5, MUTED)
    return palette[position % len(palette)]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 11.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "font.size": 9.5,
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
        }
    )


def _shade_regimes(ax, index: pd.DatetimeIndex) -> None:
    """Shade the declared stress windows so every figure marks the same episodes."""
    for name, (start, end) in cfg.STRESS_WINDOWS.items():
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        if hi < index[0] or lo > index[-1]:
            continue
        ax.axvspan(max(lo, index[0]), min(hi, index[-1]), color=ACCENT_2, alpha=0.07, linewidth=0)
        _ = name


def _footnote(fig, text: str) -> None:
    fig.text(0.005, -0.012, text, ha="left", va="top", fontsize=7.6, color=MUTED)


# ---------------------------------------------------------------------------
# Data figures
# ---------------------------------------------------------------------------
def plot_market_history(panel, path: str) -> None:
    """Price level with stress shading, and the volatility that produced it."""
    _style()
    raw = panel.raw
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.15]})
    axes[0].plot(raw.index, raw["close"], color=INK, linewidth=0.9)
    axes[0].set_ylabel("index level (start = 100)")
    axes[0].set_title(f"{panel.label}: simulated price path and its conditional volatility")
    _shade_regimes(axes[0], raw.index)

    realised_vol = np.sqrt(raw["realised_variance"] * cfg.TRADING_DAYS) * 100
    axes[1].plot(
        raw.index, realised_vol.rolling(21).mean(), color=ACCENT, linewidth=0.9, label="realised vol (21-day mean)"
    )
    if panel.true_variance is not None:
        latent = np.sqrt(panel.true_variance * cfg.TRADING_DAYS) * 100
        axes[1].plot(raw.index, latent, color=ACCENT_2, linewidth=0.8, alpha=0.75, label="true conditional vol")
    axes[1].set_ylabel("annualised volatility (%)")
    axes[1].set_xlabel("date")
    axes[1].legend(loc="upper left", ncols=2)
    _shade_regimes(axes[1], raw.index)
    _footnote(
        fig,
        f"synthetic market, seed {cfg.SEED}, {raw.index[0].date()} to {raw.index[-1].date()}, {len(raw)} trading days. Shaded bands are the declared stress windows.",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_stylised_facts(panel, path: str) -> None:
    """The four facts a volatility model has to reproduce."""
    _style()
    r = panel.raw["ret"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.8))

    lags = range(1, 81)
    acf_sq = [float(pd.Series(r**2).autocorr(lag)) for lag in lags]
    acf_abs = [float(pd.Series(np.abs(r)).autocorr(lag)) for lag in lags]
    acf_ret = [float(pd.Series(r).autocorr(lag)) for lag in lags]
    axes[0, 0].bar(lags, acf_sq, color=ACCENT, width=0.9, label=r"$r_t^2$")
    axes[0, 0].plot(lags, acf_abs, color=ACCENT_2, linewidth=1.1, label=r"$|r_t|$")
    axes[0, 0].plot(lags, acf_ret, color=MUTED, linewidth=1.0, label=r"$r_t$")
    axes[0, 0].axhline(0.0, color=INK, linewidth=0.6)
    axes[0, 0].set_title("Autocorrelation: returns are unpredictable, their size is not")
    axes[0, 0].set_xlabel("lag (trading days)")
    axes[0, 0].set_ylabel("autocorrelation")
    axes[0, 0].legend(ncols=3)

    standardised = (r - r.mean()) / r.std()
    quantiles = np.linspace(0.001, 0.999, 300)
    from scipy import stats

    axes[0, 1].plot(stats.norm.ppf(quantiles), np.quantile(standardised, quantiles), color=ACCENT, linewidth=1.2)
    axes[0, 1].plot([-4.5, 4.5], [-4.5, 4.5], color=MUTED, linewidth=0.9, linestyle="--", label="normal")
    axes[0, 1].set_title(f"QQ against the normal (excess kurtosis {pd.Series(r).kurtosis():.1f})")
    axes[0, 1].set_xlabel("theoretical normal quantile")
    axes[0, 1].set_ylabel("standardised return quantile")
    axes[0, 1].set_xlim(-4.5, 4.5)
    axes[0, 1].legend()

    prev, nxt = r[:-1], r[1:]
    axes[1, 0].scatter(prev, np.abs(nxt), s=3.5, alpha=0.25, color=ACCENT, linewidths=0)
    bins = np.quantile(prev, np.linspace(0, 1, 13))
    centres, means = [], []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (prev >= lo) & (prev <= hi)
        if mask.sum() > 8:
            centres.append(prev[mask].mean())
            means.append(np.abs(nxt[mask]).mean())
    axes[1, 0].plot(centres, means, color=ACCENT_2, linewidth=1.6, marker="o", markersize=3)
    axes[1, 0].set_title("Leverage: tomorrow's size against today's return")
    axes[1, 0].set_xlabel("return on day t (%)")
    axes[1, 0].set_ylabel("|return| on day t+1 (%)")
    axes[1, 0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}"))

    rv = np.log(panel.raw["realised_variance"].clip(lower=1e-14))
    axes[1, 1].plot(rv.index, rv, color=ACCENT_3, linewidth=0.7)
    axes[1, 1].set_title("log realised variance: level shifts, not just spikes")
    axes[1, 1].set_xlabel("date")
    axes[1, 1].set_ylabel("log RV")
    _shade_regimes(axes[1, 1], rv.index)

    _footnote(
        fig,
        f"{panel.label}, synthetic, seed {cfg.SEED}, {len(r)} observations. The leverage panel bins day-t returns by decile and plots the mean |return| on day t+1.",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_realised_measures(panel, path: str) -> None:
    """Every volatility proxy on the same axes, against the latent truth."""
    _style()
    raw = panel.raw
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.4), sharex=True)
    window = 21

    series = {
        "realised variance (5-min)": raw["realised_variance"],
        "bipower variation (jump-robust)": raw["bipower"],
        "Parkinson range": raw["parkinson"],
    }
    for (label, values), colour in zip(series.items(), (ACCENT, ACCENT_3, ACCENT_4), strict=True):
        axes[0].plot(
            raw.index,
            np.sqrt(values.rolling(window).mean() * cfg.TRADING_DAYS) * 100,
            label=label,
            color=colour,
            linewidth=0.9,
        )
    if panel.true_variance is not None:
        axes[0].plot(
            raw.index,
            np.sqrt(panel.true_variance.rolling(window).mean() * cfg.TRADING_DAYS) * 100,
            color=ACCENT_2,
            linewidth=1.2,
            label="true conditional vol (latent)",
        )
    axes[0].set_ylabel("annualised vol (%), 21-day mean")
    axes[0].set_title("Volatility proxies against the latent truth")
    axes[0].legend(ncols=2)

    if panel.true_variance is not None:
        ratio = raw["realised_variance"] / panel.true_variance
        axes[1].plot(raw.index, ratio.rolling(window).mean(), color=ACCENT, linewidth=0.9, label="RV / true variance")
        ratio_bv = raw["bipower"] / panel.true_variance
        axes[1].plot(
            raw.index, ratio_bv.rolling(window).mean(), color=ACCENT_3, linewidth=0.9, label="BV / true variance"
        )
        axes[1].axhline(1.0, color=INK, linewidth=0.8, linestyle="--")
        axes[1].set_ylim(0.5, 1.8)
        axes[1].set_ylabel("ratio to latent variance")
        axes[1].set_xlabel("date")
        axes[1].legend(ncols=2)
        axes[1].set_title("The ratio above one is the jump contribution; BV removes it")
    _shade_regimes(axes[0], raw.index)
    _footnote(
        fig,
        f"{panel.label}: 78 five-minute bars per day. RV is unbiased for diffusion plus jump variance; BV is consistent for the continuous part alone.",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Volatility-model figures
# ---------------------------------------------------------------------------
def plot_forecast_paths(forecasts: pd.DataFrame, panel, path: str, models: tuple[str, ...] | None = None) -> None:
    """Forecast volatility against what was realised, for a readable subset."""
    _style()
    models = tuple(models) if models is not None else tuple(forecasts.columns[:6])
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.6), sharex=True)

    proxy = np.sqrt(panel.raw["realised_variance"] * cfg.TRADING_DAYS) * 100
    latent = np.sqrt(panel.true_variance * cfg.TRADING_DAYS) * 100 if panel.true_variance is not None else None
    sample = slice("2008-01-01", "2010-06-30")

    for position, name in enumerate(models):
        if name not in forecasts.columns:
            continue
        series = np.sqrt(forecasts[name].clip(lower=1e-14) * cfg.TRADING_DAYS) * 100
        axes[0].plot(
            series.loc[sample].index, series.loc[sample], label=name, color=_model_colour(name, position), linewidth=1.0
        )
    if latent is not None:
        axes[0].plot(
            latent.loc[sample].index, latent.loc[sample], color=INK, linewidth=1.3, label="true conditional vol"
        )
    axes[0].set_ylabel("annualised vol (%)")
    axes[0].set_title("One-step-ahead volatility forecasts through the GFC")
    axes[0].legend(ncols=3)

    axes[1].plot(proxy.loc[sample].index, proxy.loc[sample], color=MUTED, linewidth=0.7, label="realised vol (daily)")
    axes[1].plot(
        proxy.loc[sample].rolling(21).mean().index,
        proxy.loc[sample].rolling(21).mean(),
        color=ACCENT,
        linewidth=1.1,
        label="realised vol (21-day mean)",
    )
    axes[1].set_ylabel("annualised vol (%)")
    axes[1].set_xlabel("date")
    axes[1].set_title("The target: realised volatility over the same window")
    axes[1].legend(ncols=2)
    _footnote(
        fig,
        f"{panel.label}, {sample.start} to {sample.stop}, synthetic seed {cfg.SEED}. Forecasts are conditional on information available at the previous close.",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_qlike_ranking(scores: pd.DataFrame, path: str) -> None:
    """Bar chart of QLIKE, with the proxy-based and latent-based scores side by side."""
    _style()
    frame = scores.sort_values("qlike", ascending=True)
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    positions = np.arange(len(frame))
    ax.barh(positions, frame["qlike"], color=[_model_colour(m, i) for i, m in enumerate(frame["model"])], height=0.62)
    if "qlike_latent" in frame.columns:
        ax.barh(
            positions,
            frame["qlike_latent"],
            color=INK,
            height=0.20,
            alpha=0.55,
            label="against the latent conditional variance",
        )
        ax.legend(loc="lower right")
    ax.set_yticks(positions, frame["model"])
    ax.invert_yaxis()
    ax.set_xlabel("QLIKE loss (lower is better)")
    ax.set_title(f"Volatility forecast accuracy, {int(frame['observations'].max()):,} out-of-sample days")
    _footnote(
        fig,
        "Solid bars: loss against realised variance. Dark inset: loss against the simulated latent variance, available only because the data are synthetic.",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_qlike_by_regime(regimes: pd.DataFrame, path: str) -> None:
    """Grouped bars: does the ranking survive a change of regime?"""
    _style()
    frame = regimes.set_index("model")
    columns = list(frame.columns)
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    positions = np.arange(len(frame))
    width = 0.8 / len(columns)
    colours = [ACCENT, ACCENT_2, ACCENT_4, ACCENT_3, MUTED]
    for i, (column, colour) in enumerate(zip(columns, colours, strict=False)):
        ax.bar(positions + i * width - 0.4 + width / 2, frame[column], width=width * 0.92, label=column, color=colour)
    ax.set_xticks(positions, frame.index, rotation=35, ha="right")
    ax.set_ylabel("QLIKE loss")
    ax.set_title("QLIKE by regime: an aggregate score is mostly the calm days")
    ax.legend(ncols=len(columns))
    _footnote(
        fig,
        "Regimes are the declared stress windows plus every remaining day. A model that wins by being excellent in the quiet sample is visible here.",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_calibration(scores: pd.DataFrame, forecasts: pd.DataFrame, panel, path: str) -> None:
    """Mincer-Zarnowitz scatter for the best model, plus slope diagnostics."""
    _style()
    best = str(scores.iloc[0]["model"])
    f = forecasts[best].reindex(panel.raw.index).clip(lower=1e-14)
    p = panel.raw["realised_variance"].clip(lower=1e-14)
    mask = f.notna() & p.notna()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3))

    axes[0].scatter(np.log(f[mask]), np.log(p[mask]), s=2.5, alpha=0.16, color=ACCENT, linewidths=0)
    lo, hi = float(np.log(f[mask]).quantile(0.001)), float(np.log(f[mask]).quantile(0.999))
    axes[0].plot([lo, hi], [lo, hi], color=INK, linewidth=0.9, linestyle="--", label="perfect calibration")
    axes[0].set_xlabel(f"log forecast variance ({best})")
    axes[0].set_ylabel("log realised variance")
    axes[0].set_title("Mincer-Zarnowitz: forecast against outcome")
    axes[0].legend()

    frame = scores.sort_values("qlike")
    axes[1].barh(
        np.arange(len(frame)),
        frame["mz_slope"],
        color=[_model_colour(m, i) for i, m in enumerate(frame["model"])],
        height=0.62,
    )
    axes[1].axvline(1.0, color=INK, linewidth=1.0, linestyle="--")
    axes[1].set_yticks(np.arange(len(frame)), frame["model"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mincer-Zarnowitz slope (1.0 = correctly scaled)")
    axes[1].set_title("Slope below one means the forecast is too timid")
    _footnote(
        fig,
        "The joint test that intercept = 0 and slope = 1 is reported per model in the results tables; the slope alone is shown here.",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# VaR figures
# ---------------------------------------------------------------------------
def plot_var_exceptions(suite, returns: pd.Series, path: str, method: str | None = None) -> None:
    """VaR against returns with the breach days marked."""
    _style()
    from .var import horizon_returns

    names = list(suite.forecasts)
    chosen = method if method in suite.forecasts else names[0]
    forecast = suite.forecasts[chosen]
    r = horizon_returns(returns, suite.horizon) * 100

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.6), sharex=True)
    axes[0].plot(r.index, r, color=MUTED, linewidth=0.6, label=f"{suite.horizon}-day return")
    axes[0].plot(
        forecast.var.index,
        -forecast.var * 100,
        color=ACCENT_2,
        linewidth=0.9,
        label=f"{chosen} VaR ({suite.level:.0%})",
    )
    breach = r < -forecast.var * 100
    axes[0].scatter(
        r.index[breach.fillna(False)],
        r[breach.fillna(False)],
        s=9,
        color=ACCENT_2,
        zorder=5,
        linewidths=0,
        label=f"{int(breach.sum())} breaches",
    )
    axes[0].set_ylabel("return (%)")
    axes[0].set_title(f"Value-at-Risk and its exceptions: {chosen} at {suite.level:.0%}, {suite.horizon}-day horizon")
    axes[0].legend(ncols=3)

    window = 250
    rolling = breach.fillna(False).astype(float).rolling(window).mean() * 100
    axes[1].plot(rolling.index, rolling, color=ACCENT, linewidth=1.0)
    axes[1].axhline(
        (1 - suite.level) * 100,
        color=INK,
        linewidth=1.0,
        linestyle="--",
        label=f"target {(1 - suite.level) * 100:.1f}%",
    )
    axes[1].set_ylabel(f"breach rate (%), {window}-day window")
    axes[1].set_xlabel("date")
    axes[1].set_title("Rolling breach rate: agreement on average can hide long disagreement")
    axes[1].legend()
    _footnote(
        fig,
        f"Synthetic market, seed {cfg.SEED}, {suite.horizon}-day horizon. The rolling rate uses a {window}-day window and is only plotted where the window is full.",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_coverage_summary(coverage: pd.DataFrame, path: str) -> None:
    """Observed against expected breach rate, with the test verdicts."""
    _style()
    frame = coverage.copy()
    frame["excess"] = frame["observed_rate"] / frame["expected_rate"]
    frame = frame.sort_values("excess")
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    positions = np.arange(len(frame))
    colours = [ACCENT_3 if not r else ACCENT_2 for r in frame["kupiec_reject"]]
    ax.barh(positions, frame["observed_rate"] * 100, color=colours, height=0.62)
    ax.axvline(frame["expected_rate"].iloc[0] * 100, color=INK, linewidth=1.2, linestyle="--", label="expected rate")
    for position, (_, row) in enumerate(frame.iterrows()):
        label = f"{row['breaches']} breaches"
        if row["kupiec_reject"]:
            label += "  (Kupiec rejects)"
        ax.text(row["observed_rate"] * 100, position, f"  {label}", va="center", fontsize=8, color=INK)
    ax.set_yticks(positions, frame["method"])
    ax.invert_yaxis()
    ax.set_xlabel("observed breach rate (%)")
    ax.set_title(
        f"Unconditional coverage, {frame['level'].iloc[0]:.0%} VaR, {int(frame['horizon'].iloc[0])}-day horizon"
    )
    ax.legend(loc="lower right")
    _footnote(
        fig,
        "Green passes Kupiec's unconditional coverage test at 5%; red fails it. Passing means the data are consistent with the level, not that the model is right.",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_traffic_light(traffic: pd.DataFrame, path: str, method: str) -> None:
    """Rolling 250-day exception count with the Basel zone bands."""
    _style()
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    counts = traffic["exceptions"].dropna()
    ax.plot(counts.index, counts, color=INK, linewidth=0.9)
    ax.axhspan(0, 4.5, color=ACCENT_3, alpha=0.10, linewidth=0)
    ax.axhspan(4.5, 9.5, color=ACCENT_4, alpha=0.13, linewidth=0)
    ax.axhspan(9.5, max(12.5, float(counts.max()) + 1), color=ACCENT_2, alpha=0.13, linewidth=0)
    ax.axhline(4.5, color=MUTED, linewidth=0.8, linestyle="--")
    ax.axhline(9.5, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_ylabel(f"exceptions in the last {cfg.BASEL_WINDOW} days")
    ax.set_xlabel("date")
    ax.set_title(f"Basel traffic light, 99% VaR: {method}")
    ax.text(0.008, 0.94, "green (0-4)", transform=ax.transAxes, fontsize=8, color=ACCENT_3, va="top")
    ax.text(0.008, 0.70, "yellow (5-9)", transform=ax.transAxes, fontsize=8, color=ACCENT_4, va="top")
    ax.text(0.008, 0.46, "red (10+)", transform=ax.transAxes, fontsize=8, color=ACCENT_2, va="top")
    _footnote(
        fig,
        "The zone is not merely a label: 5 exceptions cost 40% more capital than 4 under the Basel add-on, and 10 double it.",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_horizon_gap(gap_series: pd.Series, path: str, method: str) -> None:
    """Monte-Carlo 10-day VaR against the square-root-of-time scaling."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    axes[0].plot(gap_series.index, gap_series * 100, color=ACCENT_5, linewidth=0.8)
    axes[0].axhline(0.0, color=INK, linewidth=1.0, linestyle="--")
    axes[0].set_ylabel("deviation (%)")
    axes[0].set_xlabel("date")
    axes[0].set_title("Simulated 10-day VaR vs sqrt(10) scaling")

    axes[1].hist(gap_series.dropna() * 100, bins=45, color=ACCENT_5, alpha=0.85)
    axes[1].axvline(0.0, color=INK, linewidth=1.0, linestyle="--")
    axes[1].set_xlabel("deviation (%)")
    axes[1].set_ylabel("days")
    axes[1].set_title("Above zero: the scaling rule understates risk")
    _footnote(
        fig,
        f"{method}: (simulated 10-day VaR) / (sqrt(10) x 1-day VaR) - 1, over the out-of-sample period. Negative values are the state after a shock, when volatility is high and expected to fall.",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_es_against_var(suite, returns: pd.Series, path: str) -> None:
    """The tail beyond the quantile, by method."""
    _style()
    from .var import horizon_returns

    r = horizon_returns(returns, suite.horizon)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    names = list(suite.forecasts)
    es_means = [float(suite.forecasts[n].es.mean()) * 1e4 for n in names]
    var_means = [float(suite.forecasts[n].var.mean()) * 1e4 for n in names]
    positions = np.arange(len(names))
    axes[0].barh(positions - 0.18, var_means, height=0.34, color=ACCENT, label="mean VaR")
    axes[0].barh(positions + 0.18, es_means, height=0.34, color=ACCENT_2, label="mean ES")
    axes[0].set_yticks(positions, names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("basis points")
    axes[0].set_title(f"VaR and ES levels, {suite.level:.0%}, {suite.horizon}-day")
    axes[0].legend()

    worst = r.dropna().nsmallest(40) * 1e4
    axes[1].plot(
        np.arange(len(worst)),
        np.sort(worst.to_numpy()),
        color=INK,
        linewidth=1.4,
        marker="o",
        markersize=3,
        label="40 largest losses",
    )
    for position, name in enumerate(names[:4]):
        axes[1].axhline(
            float(suite.forecasts[name].var.mean()) * 1e4,
            color=_model_colour(name, position),
            linewidth=1.0,
            linestyle="--",
            label=f"{name} mean VaR",
        )
    axes[1].set_xlabel("rank (1 = largest loss)")
    axes[1].set_ylabel("loss (basis points)")
    axes[1].set_title("The losses that matter, against each VaR level")
    axes[1].legend()
    _footnote(
        fig,
        "A quantile says nothing about how far beyond it a breach goes; that is what ES is for, and the two are reported together throughout.",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_stress_coverage(stress: pd.DataFrame, path: str, method: str) -> None:
    """Breach rate by stress window against the target."""
    _style()
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    positions = np.arange(len(stress))
    ax.bar(positions, stress["breach_rate"] * 100, color=ACCENT_2, width=0.5, label="observed breach rate")
    ax.axhline(stress["expected_rate"].iloc[0] * 100, color=INK, linewidth=1.2, linestyle="--", label="target")
    for position, (_, row) in enumerate(stress.iterrows()):
        ax.text(
            position,
            row["breach_rate"] * 100,
            f"  {int(row['breaches'])}/{int(row['observations'])}",
            va="bottom",
            fontsize=8,
            color=INK,
        )
    ax.set_xticks(
        positions,
        [
            f"{w}\nrealised vol {v:.0f}%"
            for w, v in zip(stress["window"], stress["realised_vol_annual"] * 100, strict=True)
        ],
    )
    ax.set_ylabel("breach rate (%)")
    ax.set_title(f"Coverage inside the stress windows: {method}")
    ax.legend()
    _footnote(
        fig,
        "An aggregate breach rate is an average over calm and violent days. A model can hit the average exactly while being far too tight in every crisis.",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
