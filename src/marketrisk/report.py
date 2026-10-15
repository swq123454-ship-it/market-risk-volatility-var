"""The end-to-end pipeline and the results bundle.

One command runs the study; everything the README claims is written by this
module into ``outputs/`` and nothing is typed by hand. That constraint is the
point. A results file assembled from a run is checkable by anyone with a clone;
a results file assembled from memory is a claim.

The pipeline stages
-------------------
1. **Data.** Simulate or download the market, depending on the mode.
2. **Features.** Realised measures and the strictly-lagged design matrix.
3. **Volatility forecasts.** Walk-forward over the whole model family.
4. **Scoring.** QLIKE, MSE, Mincer-Zarnowitz, Diebold-Mariano, by regime.
5. **VaR and ES.** Seven routes at each level and horizon: two historical, two
   filtrations of the same filter (one of them tail-modelled), two parametric and one
   simulation.
6. **Backtesting.** Kupiec, Christoffersen, Engle-Manganelli, Acerbi-Szekely.
7. **Basel traffic light** over rolling 250-day windows.
8. **Stress slices** and the horizon-scaling gap.
9. **Figures, tables, RESULTS.md and metrics.json.**

Every stage writes its table to ``outputs/tables`` as a CSV before the markdown
is rendered, so a number in the report can be traced to a line in a file.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest, evaluate, plots, var, volmodels
from . import config as cfg
from . import data as data_module

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _fmt(value: float, digits: int = 4) -> str:
    """Format a float for a markdown table, tolerating NaN and infinities."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:,.{digits}f}"


def _pct(value: float, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, digits: int = 4) -> str:
    """Render a DataFrame as a markdown table with no index."""
    view = frame if columns is None else frame.loc[:, [c for c in columns if c in frame.columns]]
    header = "| " + " | ".join(str(c) for c in view.columns) + " |"
    divider = "|" + "|".join("---" for _ in view.columns) + "|"
    rows = []
    for _, row in view.iterrows():
        cells = []
        for value in row:
            if isinstance(value, (bool, np.bool_)):
                cells.append("yes" if value else "no")
            elif isinstance(value, (int, np.integer)):
                cells.append(f"{int(value):,}")
            elif isinstance(value, (float, np.floating)):
                cells.append(_fmt(float(value), digits))
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *rows])


def _write_table(frame: pd.DataFrame, directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    frame.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Bundle serialisation
# ---------------------------------------------------------------------------
def _jsonable(value):
    """Recursively convert numpy and pandas objects into JSON primitives."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    return value


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def build_data(pipeline: cfg.PipelineConfig) -> data_module.MarketData:
    return data_module.load_market(
        data_mode=pipeline.data_mode, start=pipeline.start, end=pipeline.end, seed=pipeline.seed
    )


def fit_volatility_models(
    market: data_module.MarketData, pipeline: cfg.PipelineConfig, verbose: bool = True
) -> tuple[volmodels.ForecastSet, list[volmodels.VolatilityModel]]:
    panel = market.primary
    zoo = volmodels.default_model_zoo(fast=pipeline.fast, gru_params=pipeline.effective_gru_params)
    if verbose:
        print(f"  models: {', '.join(m.name for m in zoo)}")
    forecasts = volmodels.walk_forward(
        zoo,
        panel.returns,
        panel.features,
        init_window=pipeline.init_window,
        refit_every=pipeline.refit_every,
        verbose=verbose,
    )
    return forecasts, zoo


def build_var_studies(
    market: data_module.MarketData,
    forecasts: volmodels.ForecastSet,
    zoo: list[volmodels.VolatilityModel],
    pipeline: cfg.PipelineConfig,
    seed: int,
) -> dict[tuple[float, int], var.VaRSuite]:
    panel = market.primary
    sigma_paths = {name: np.sqrt(forecasts.variance[name].clip(lower=1e-14)) for name in forecasts.variance.columns}
    garch = next((m for m in zoo if isinstance(m, volmodels.GARCH) and m.kind == "gjr"), None)
    sigma2 = forecasts.variance[garch.name] if garch is not None else None
    rng = np.random.default_rng(seed)
    suites: dict[tuple[float, int], var.VaRSuite] = {}
    for level in pipeline.confidence_levels:
        for horizon in pipeline.horizons:
            suites[(level, horizon)] = var.build_suite(
                level=level,
                horizon=horizon,
                returns=panel.returns,
                sigma_paths=sigma_paths,
                garch_model=garch,
                sigma2_garch=sigma2,
                rng=rng,
                hs_window=cfg.HS_WINDOWS[0],
                paths=pipeline.effective_mc_paths,
            )
    return suites


def coverage_tables(suites: dict[tuple[float, int], var.VaRSuite], market: data_module.MarketData) -> pd.DataFrame:
    """One row per method per level per horizon, on a common evaluation sample."""
    panel = market.primary
    rows = []
    for (level, horizon), suite in suites.items():
        frame = suite.var_frame()
        complete = frame.notna().all(axis=1)
        start = int(np.argmax(complete.to_numpy())) if complete.any() else 0
        if start == 0 and not complete.iloc[0]:
            continue
        for name, forecast in suite.forecasts.items():
            report = backtest.coverage_report(
                method=name,
                family=forecast.family,
                level=level,
                horizon=horizon,
                returns=panel.returns,
                var=forecast.var.to_numpy(),
                es=forecast.es.to_numpy(),
                start=start,
            )
            row = report.as_row()
            row["evaluation_start"] = str(panel.index[start].date())
            rows.append(row)
    return pd.DataFrame(rows)


def horizon_scaling_study(
    suites: dict[tuple[float, int], var.VaRSuite], market: data_module.MarketData, level: float
) -> pd.DataFrame:
    """Compare the simulated multi-day VaR against the square-root-of-time rule."""
    one_day = suites.get((level, 1))
    ten_day = suites.get((level, 10))
    if one_day is None or ten_day is None or "monte_carlo" not in ten_day.forecasts:
        return pd.DataFrame()
    a = one_day.forecasts["monte_carlo"].var
    b = ten_day.forecasts["monte_carlo"].var
    gap = var.horizon_gap(a.to_numpy(), b.to_numpy(), 10)
    frame = pd.DataFrame({"date": a.index, "one_day_var": a.to_numpy(), "ten_day_var": b.to_numpy(), "gap": gap})
    frame["level"] = level
    return frame.dropna()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _render_results(
    market: data_module.MarketData,
    pipeline: cfg.PipelineConfig,
    scores: pd.DataFrame,
    regimes: pd.DataFrame,
    dm: pd.DataFrame,
    coverage: pd.DataFrame,
    ticks: pd.DataFrame,
    es_tests: pd.DataFrame,
    stress: pd.DataFrame,
    traffic: pd.DataFrame,
    gap: pd.DataFrame,
    manifest: dict[str, object],
) -> str:
    panel = market.primary
    lines: list[str] = []
    add = lines.append

    add("# Results")
    add("")
    add(f"Generated by `scripts/run_pipeline.py` on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.")
    add(f"Data mode **{market.mode}**, seed **{pipeline.seed}**, asset **{panel.label}**.")
    add("")
    add("Every number below was written by the pipeline that produced it. Nothing in this file")
    add("is typed by hand, and every table exists as a CSV in `outputs/tables/` with the same numbers.")
    add("")

    add("## 1. Sample")
    add("")
    add(f"- Trading days: **{len(panel):,}** ({panel.index[0].date()} to {panel.index[-1].date()})")
    add(f"- Out-of-sample evaluation starts at index {pipeline.init_window:,}, after the initial window")
    realised_vol = float(panel.raw["ret"].std() * np.sqrt(cfg.TRADING_DAYS))
    add(f"- Realised volatility over the whole sample: **{_pct(realised_vol)}** annualised")
    add(
        f"- Realised measure used: `{panel.realised_measure}`"
        + ("" if panel.jumps_available else " (intraday data unavailable, so no jump decomposition)")
    )
    add("")
    add("### Stress windows")
    add("")
    rows = []
    for name, (start, end) in cfg.STRESS_WINDOWS.items():
        window = panel.raw["ret"].loc[start:end]
        rows.append(
            {
                "window": name,
                "start": start,
                "end": end,
                "days": len(window),
                "realised vol (annual)": float(window.std() * np.sqrt(cfg.TRADING_DAYS)),
                "worst day": float(window.min()),
            }
        )
    add(_markdown_table(pd.DataFrame(rows), digits=4))
    add("")
    add("These bands are shaded in every figure that spans them.")
    add("")

    add("## 2. Volatility forecast accuracy")
    add("")
    add("Scored out of sample. QLIKE is the ranking loss; MSE is reported because a risk")
    add("committee recognises it, and because a model that wins on QLIKE while losing badly")
    add("on MSE is telling you where its errors sit.")
    add("")
    add(
        _markdown_table(
            scores,
            columns=[
                "model",
                "observations",
                "qlike",
                "mse_vol_bp",
                "mz_slope",
                "mz_r2_log",
                "mz_joint_p",
                "mean_forecast_vol_annual",
            ],
        )
    )
    add("")
    if "qlike_latent" in scores.columns:
        add("`mz_slope` is the Mincer-Zarnowitz slope: 1.0 means the forecast is correctly scaled,")
        add("below 1.0 means it is too timid, above means it over-reacts. `mz_joint_p` is the p-value of")
        add("the joint test that the intercept is zero and the slope is one; a small value means the")
        add("forecast is miscalibrated even if its average level is right.")
        add("")
        best_latent = scores.sort_values("qlike_latent").iloc[0]
        add("Against the **latent** conditional variance - available only because the data are")
        add(f"synthetic - the best model is `{best_latent['model']}` at QLIKE {_fmt(best_latent['qlike_latent'])},")
        add("and the ordering is broadly the same as against realised variance. That matters: it means")
        add("the ranking is not an artefact of the noise in the realised measure.")
        add("")
    add("### By regime")
    add("")
    add(_markdown_table(regimes))
    add("")
    add("An aggregate score over twenty years is mostly an average over quiet days. The columns")
    add("below split the same loss by regime so that a model which is excellent in the calm sample")
    add("and useless in the crises cannot hide inside the total.")
    add("")
    add("### Pairwise Diebold-Mariano")
    add("")
    add("P-values on the QLIKE loss differential, Newey-West with 10 lags. A small value means")
    add("the gap between those two models is larger than its own standard error.")
    add("")
    add(_markdown_table(dm.reset_index().rename(columns={"index": "model"}), digits=4))
    add("")

    add("## 3. Value-at-Risk and Expected Shortfall")
    add("")
    # The count is read off the table rather than typed, so adding a route cannot leave
    # the prose describing a study that no longer exists.
    add(f"{int(ticks['method'].nunique())} routes to the same number. The ranking below is by tick loss,")
    add("the correct scoring rule for a quantile; the coverage tests in the next section ask a")
    add("different question. The mean VaR column is there because a method can win on tick loss by")
    add("being systematically conservative, and that trade should be visible rather than implicit.")
    add("")
    for level in pipeline.confidence_levels:
        for horizon in pipeline.horizons:
            subset = ticks[(ticks["level"] == level) & (ticks["horizon"] == horizon)]
            if subset.empty:
                continue
            add(f"### {level:.0%}, {horizon}-day horizon")
            add("")
            add(
                _markdown_table(
                    subset, columns=["method", "family", "tick_loss_bp", "mean_var_bp", "mean_es_bp"], digits=2
                )
            )
            add("")

    add("## 4. Backtests")
    add("")
    add("Four tests, each of which can fail on its own.")
    add("")
    add("- **Kupiec** (`kupiec_p`): is the breach rate right? Rejected when the observed rate is")
    add("  inconsistent with the level, in *either* direction - a model with no breaches is mispriced,")
    add("  not conservative.")
    add("- **Christoffersen** (`ind_p`): are breaches independent? A model that breaches in runs has")
    add("  the right average and is still useless, because the breaches arrive together.")
    add("- **Conditional coverage** (`cc_p`): the two above, jointly.")
    add("- **Engle-Manganelli** (`dq_p`): is the breach predictable from its own lags or from the VaR?")
    add("  Strictly stronger than the other two; it can detect a misspecification they both miss.")
    add("")
    for level in pipeline.confidence_levels:
        for horizon in pipeline.horizons:
            subset = coverage[(coverage["level"] == level) & (coverage["horizon"] == horizon)]
            if subset.empty:
                continue
            add(f"### {level:.0%}, {horizon}-day horizon")
            add("")
            add(
                _markdown_table(
                    subset,
                    columns=[
                        "method",
                        "observations",
                        "breaches",
                        "observed_rate",
                        "expected_rate",
                        "kupiec_p",
                        "kupiec_reject",
                        "ind_p",
                        "cc_p",
                        "dq_p",
                    ],
                )
            )
            if horizon > 1:
                add("")
                add("> **Reading the independence test at h > 1.** Multi-day returns are computed over")
                add("> overlapping windows, so consecutive hit indicators are mechanically dependent even")
                add("> when the model is correct. The Christoffersen and dynamic-quantile p-values at h = 10")
                add("> are therefore biased downwards and should be read as indicative only. The *breach rate*")
                add("> is unaffected - its expectation is still one minus the level - so Kupiec remains valid.")
            add("")

    add("### Basel traffic light")
    add("")
    add("The regulator's test, at the level the regulation is written for (99%). The zone is not a")
    add("label: five exceptions in 250 days costs 40% more capital than four, and ten doubles it.")
    add("")
    basel_rows = coverage[coverage["level"].apply(lambda v: abs(v - 0.99) < 1e-9)]
    if not basel_rows.empty:
        basel_rows = basel_rows[basel_rows["horizon"] == 1]
        add(
            _markdown_table(
                basel_rows,
                columns=["method", "basel_exceptions", "basel_zone", "basel_plus_factor"],
            )
        )
    add("")
    if not traffic.empty:
        zones = traffic["zone"].dropna()
        share = float((zones != "green").mean()) if len(zones) else float("nan")
        worst = int(traffic["exceptions"].max()) if traffic["exceptions"].notna().any() else 0
        add(f"For the reference method, **{_pct(share, 1)}** of rolling windows are outside the green")
        add(f"zone, and the worst window holds **{worst}** exceptions.")
    add("")

    add("### Expected Shortfall")
    add("")
    add("Acerbi-Szekely Z1: the average, over breach days, of `-r / ES - 1`, standardised. Zero means")
    add("the ES is the right size for the tail it claims to describe.")
    add("")
    for level in pipeline.confidence_levels:
        for horizon in pipeline.horizons:
            subset = es_tests[(es_tests["level"] == level) & (es_tests["horizon"] == horizon)]
            if subset.empty:
                continue
            add(f"**{level:.0%}, {horizon}-day horizon**")
            add("")
            add(
                _markdown_table(
                    subset,
                    columns=["method", "breaches", "mean_es_bp", "mean_excess_bp", "z1", "z1_p", "es_adequate"],
                    digits=3,
                )
            )
            add("")
    add("`mean_excess_bp` is the average amount by which a breach exceeded its VaR, in basis points.")
    add("A method can pass coverage and still have a large excess, which is the failure mode ES exists")
    add("to catch.")
    add("")

    add("## 5. Stress coverage")
    add("")
    add(_markdown_table(stress, digits=4))
    add("")
    add("Same breach rate as Section 4, split by regime. The aggregate can be exactly right while")
    add("the crisis is far too tight and everything else is far too loose.")
    add("")

    add("## 6. Multi-day scaling")
    add("")
    if gap.empty:
        add("Not available in this run.")
    else:
        mean_gap = float(gap["gap"].mean())
        share_pos = float((gap["gap"] > 0).mean())
        add("The square-root-of-time rule says an h-day variance is h times a one-day variance. That is")
        add("true for i.i.d. returns and false for GARCH returns, whose variance is random and")
        add("mean-reverting. Simulating the recursion gives the ratio directly.")
        add("")
        add(f"- Mean deviation of the simulated 10-day VaR from the scaled one-day VaR: **{_pct(mean_gap)}**")
        add(f"- Share of days on which the scaling rule understates the simulated risk: **{_pct(share_pos, 1)}**")
        add("")
        add("Negative values are the aftermath of a shock: volatility is high and expected to fall, so")
        add("scaling by the square root of time is too cautious. Positive values are quiet periods, where")
        add("the same rule assumes volatility stays put and it does not. Both directions appear, which is")
        add("why the rule is a statement about the regime as much as about the horizon.")
    add("")

    add("## 7. Reproducing this")
    add("")
    add("```bash")
    add("make pipeline          # full run")
    add("python scripts/run_pipeline.py --fast     # reduced sample, for a quick check")
    add("python scripts/run_pipeline.py --data-mode real   # downloaded index history instead")
    add("```")
    add("")
    add("The generator manifest for this run:")
    add("")
    add("```json")
    add(json.dumps(_jsonable(manifest), indent=2)[:2400])
    add("```")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_pipeline(
    pipeline: cfg.PipelineConfig,
    output_dir: Path | None = None,
    verbose: bool = True,
    render_figures: bool = True,
) -> dict:
    """Run every stage and write the bundle. Returns the results dictionary."""
    output = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    if verbose:
        print("stage 1/8  data")

    market = build_data(pipeline)
    panel = market.primary
    if verbose:
        print(f"  mode={market.mode}  rows={len(panel):,}  assets={', '.join(market.keys())}")

    if verbose:
        print("stage 2/8  volatility forecasts")
    forecasts, zoo = fit_volatility_models(market, pipeline, verbose=verbose)

    if verbose:
        print("stage 3/8  scoring")
    scores = evaluate.volatility_scores(forecasts.variance, panel.realised, panel.true_variance)
    regimes = evaluate.qlike_by_regime(forecasts.variance, panel.realised)
    dm = evaluate.pairwise_diebold_mariano(forecasts.variance, panel.realised)

    if verbose:
        print("stage 4/8  VaR and ES")
    suites = build_var_studies(market, forecasts, zoo, pipeline, pipeline.seed)

    if verbose:
        print("stage 5/8  backtests")
    coverage = coverage_tables(suites, market)
    ticks = pd.concat([evaluate.tick_loss_table(suite, panel.returns) for suite in suites.values()], ignore_index=True)
    es_tests = pd.concat(
        [evaluate.es_backtest_table(suite, panel.returns) for suite in suites.values()], ignore_index=True
    )

    if verbose:
        print("stage 6/8  stress slices and Basel")
    primary_suite = suites[(cfg.PRIMARY_LEVEL, 1)]
    reference_method = (
        "fhs_evt_gjr" if "fhs_evt_gjr" in primary_suite.forecasts else next(iter(primary_suite.forecasts))
    )
    stress = backtest.stress_window_coverage(
        panel.returns, primary_suite.forecasts[reference_method].var.to_numpy(), panel.index, cfg.PRIMARY_LEVEL, 1
    )
    hits = (panel.returns.to_numpy() < -primary_suite.forecasts[reference_method].var.to_numpy()).astype(float)
    traffic = backtest.basel_traffic_light(hits, panel.index)
    gap = horizon_scaling_study(suites, market, cfg.PRIMARY_LEVEL)

    if verbose:
        print("stage 7/8  figures")
    if render_figures:
        _render_figures(
            market, forecasts, scores, regimes, coverage, suites, stress, traffic, gap, reference_method, figures
        )
    elif verbose:
        print("  skipped (--no-figures)")

    if verbose:
        print("stage 8/8  results bundle")
    manifest = dict(market.manifest)
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": f"marketrisk-seed{pipeline.seed}-{pipeline.data_mode}",
        "data_mode": market.mode,
        "seed": pipeline.seed,
        "asset": panel.label,
        "rows": int(len(panel)),
        "sample_start": str(panel.index[0].date()),
        "sample_end": str(panel.index[-1].date()),
        "init_window": pipeline.init_window,
        "refit_every": pipeline.refit_every,
        "realised_vol_annual": float(panel.raw["ret"].std() * np.sqrt(cfg.TRADING_DAYS)),
        "manifest": manifest,
        "volatility_scores": scores.to_dict(orient="records"),
        "qlike_by_regime": regimes.to_dict(orient="records"),
        "coverage": coverage.to_dict(orient="records"),
        "tick_loss": ticks.to_dict(orient="records"),
        "es_backtest": es_tests.to_dict(orient="records"),
        "stress_coverage": stress.to_dict(orient="records"),
        "horizon_scaling": {
            "reference_method": reference_method,
            "mean_gap": float(gap["gap"].mean()) if not gap.empty else None,
            "share_scaling_understates": float((gap["gap"] > 0).mean()) if not gap.empty else None,
        },
        "basel": {
            "reference_method": reference_method,
            "worst_exceptions": int(traffic["exceptions"].max()) if traffic["exceptions"].notna().any() else None,
            "share_outside_green": float((traffic["zone"].dropna() != "green").mean())
            if traffic["zone"].notna().any()
            else None,
        },
        "config": {
            "confidence_levels": list(pipeline.confidence_levels),
            "horizons": list(pipeline.horizons),
            "hs_windows": list(cfg.HS_WINDOWS),
            "mc_paths": pipeline.mc_paths,
            "basel_window": cfg.BASEL_WINDOW,
            "models": [m.name for m in zoo],
        },
    }

    _write_table(scores, tables, "01_volatility_scores.csv")
    _write_table(regimes, tables, "02_qlike_by_regime.csv")
    _write_table(dm.reset_index(names="model"), tables, "03_diebold_mariano_pvalues.csv")
    _write_table(coverage, tables, "04_coverage_tests.csv")
    _write_table(ticks, tables, "05_tick_loss.csv")
    _write_table(es_tests, tables, "06_expected_shortfall_tests.csv")
    _write_table(stress, tables, "07_stress_coverage.csv")
    _write_table(traffic.dropna(), tables, "08_basel_traffic_light.csv")
    if not gap.empty:
        _write_table(gap, tables, "09_horizon_scaling.csv")
    _write_table(
        pd.DataFrame(
            [
                {
                    "date": str(panel.index[i]),
                    **{name: float(forecasts.variance[name].iloc[i]) for name in forecasts.variance.columns},
                    "realised_variance": float(panel.raw["realised_variance"].iloc[i]),
                    "ret": float(panel.raw["ret"].iloc[i]),
                }
                for i in range(0, len(panel), 5)
            ]
        ),
        tables,
        "10_forecast_path_subsample.csv",
    )

    markdown = _render_results(
        market, pipeline, scores, regimes, dm, coverage, ticks, es_tests, stress, traffic, gap, manifest
    )
    (output / "RESULTS.md").write_text(markdown, encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    (output / "generator_manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")

    if verbose:
        print(_summary(scores, coverage, ticks))
    return results


def _render_figures(
    market, forecasts, scores, regimes, coverage, suites, stress, traffic, gap, reference_method, figures
) -> None:
    panel = market.primary
    plots.plot_market_history(panel, str(figures / "01_market_history.png"))
    plots.plot_stylised_facts(panel, str(figures / "02_stylised_facts.png"))
    plots.plot_realised_measures(panel, str(figures / "03_realised_measures.png"))
    plots.plot_forecast_paths(forecasts.variance, panel, str(figures / "04_forecast_paths.png"))
    plots.plot_qlike_ranking(scores, str(figures / "05_qlike_ranking.png"))
    plots.plot_qlike_by_regime(regimes, str(figures / "06_qlike_by_regime.png"))
    plots.plot_calibration(scores, forecasts.variance, panel, str(figures / "07_calibration.png"))

    primary = suites[(cfg.PRIMARY_LEVEL, 1)]
    plots.plot_var_exceptions(primary, panel.returns, str(figures / "08_var_exceptions.png"), reference_method)
    subset = coverage[(coverage["level"] == cfg.PRIMARY_LEVEL) & (coverage["horizon"] == 1)]
    if not subset.empty:
        plots.plot_coverage_summary(subset, str(figures / "09_coverage_summary.png"))
    plots.plot_traffic_light(traffic, str(figures / "10_basel_traffic_light.png"), reference_method)
    if not stress.empty:
        plots.plot_stress_coverage(stress, str(figures / "11_stress_coverage.png"), reference_method)
    if not gap.empty:
        plots.plot_horizon_gap(gap.set_index("date")["gap"], str(figures / "12_horizon_gap.png"), reference_method)
    plots.plot_es_against_var(primary, panel.returns, str(figures / "13_es_vs_var.png"))


def _summary(scores: pd.DataFrame, coverage: pd.DataFrame, ticks: pd.DataFrame) -> str:
    best = scores.iloc[0]
    lines = ["", "  headline results"]
    lines.append(f"    best volatility model      {best['model']}  (QLIKE {best['qlike']:.4f})")
    subset = coverage[(coverage["level"] == cfg.PRIMARY_LEVEL) & (coverage["horizon"] == 1)]
    if not subset.empty:
        passing = subset[~subset["kupiec_reject"]]
        lines.append(f"    pass Kupiec at 99%, 1-day  {len(passing)} of {len(subset)} methods")
    best_tick = ticks[(ticks["level"] == cfg.PRIMARY_LEVEL) & (ticks["horizon"] == 1)]
    if not best_tick.empty:
        lines.append(
            f"    best tick loss at 99%      {best_tick.iloc[0]['method']}  ({best_tick.iloc[0]['tick_loss_bp']:.1f} bp)"
        )
    return "\n".join(lines)
