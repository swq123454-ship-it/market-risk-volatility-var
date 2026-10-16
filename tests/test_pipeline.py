"""End-to-end tests for the pipeline and the results bundle.

The point of these tests is not that the pipeline runs - a pipeline that runs and
produces five of six VaR methods is exactly the failure mode that motivated them.
They check that every deliverable exists and that the numbers in the report are the
numbers in the bundle, so a README claim can be traced to a file.

Marked ``slow``; deselect with ``-m "not slow"`` when iterating on a module.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from marketrisk import config as cfg
from marketrisk.report import run_pipeline

pytestmark = pytest.mark.slow

#: Sample chosen so that at least one declared stress window overlaps it: the
#: 2022 rate shock and the Q4 2018 selloff both fall inside. The figure that shows
#: stress slices is conditional on that, and a test that never produces it is not
#: testing it.
START = "2017-01-03"
END = "2022-12-30"

PIPELINE = cfg.PipelineConfig(
    fast=True,
    start=START,
    end=END,
    init_window=400,
    refit_every=63,
)

UNCONDITIONAL_FIGURES = [
    "01_market_history.png",
    "02_stylised_facts.png",
    "03_realised_measures.png",
    "04_forecast_paths.png",
    "05_qlike_ranking.png",
    "06_qlike_by_regime.png",
    "07_calibration.png",
    "08_var_exceptions.png",
    "09_coverage_summary.png",
    "10_basel_traffic_light.png",
    "12_horizon_gap.png",
    "13_es_vs_var.png",
]

TABLES = [
    "01_volatility_scores.csv",
    "02_qlike_by_regime.csv",
    "03_diebold_mariano_pvalues.csv",
    "04_coverage_tests.csv",
    "05_tick_loss.csv",
    "06_expected_shortfall_tests.csv",
    "07_stress_coverage.csv",
    "08_basel_traffic_light.csv",
    "09_horizon_scaling.csv",
    "10_forecast_path_subsample.csv",
]


@pytest.fixture(scope="module")
def bundle(scratch):
    output = scratch("p3_bundle")
    run_pipeline(PIPELINE, output_dir=output, verbose=False)
    return output


def test_results_markdown_and_metrics_json_exist(bundle):
    assert (bundle / "RESULTS.md").exists()
    assert (bundle / "metrics.json").exists()
    assert (bundle / "generator_manifest.json").exists()


def test_every_table_is_written(bundle):
    for name in TABLES:
        path = bundle / "tables" / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name


def test_every_figure_is_written(bundle):
    for name in UNCONDITIONAL_FIGURES:
        path = bundle / "figures" / name
        assert path.exists(), name
        assert path.stat().st_size > 5_000, name
    # 11 is conditional on the sample overlapping a declared stress window, which
    # this sample does.
    assert (bundle / "figures" / "11_stress_coverage.png").exists()


def test_metrics_bundle_has_every_section(bundle):
    metrics = json.loads((bundle / "metrics.json").read_text())
    for key in (
        "volatility_scores",
        "qlike_by_regime",
        "coverage",
        "tick_loss",
        "es_backtest",
        "stress_coverage",
        "horizon_scaling",
        "basel",
        "manifest",
        "config",
    ):
        assert key in metrics, key
    assert metrics["data_mode"] == "synthetic"
    assert metrics["seed"] == cfg.SEED


def test_every_model_and_every_var_method_is_reported(bundle):
    """The completeness check.

    A speed flag that silently drops a contestant turns the comparison into a
    different comparison under the same title. This asserts the roster.
    """
    metrics = json.loads((bundle / "metrics.json").read_text())
    models = {row["model"] for row in metrics["volatility_scores"]}
    methods = {row["method"] for row in metrics["coverage"]}
    assert set(metrics["config"]["models"]) == models
    assert {
        "rolling_21",
        "rolling_63",
        "ewma_94",
        "ewma_fitted",
        "garch",
        "gjr",
        "egarch",
        "har",
        "harx",
        "gbm",
        "gru",
    } == models
    assert {
        "hs_250",
        "hs_brw",
        "parametric_normal_gjr",
        "parametric_t_gjr",
        "fhs_gjr",
        "fhs_evt_gjr",
        "monte_carlo",
    } == methods


def test_the_horizon_scaling_section_is_populated(bundle):
    """It is empty unless the Monte-Carlo leg is present, which was once the case."""
    metrics = json.loads((bundle / "metrics.json").read_text())
    scaling = metrics["horizon_scaling"]
    assert scaling["mean_gap"] is not None
    assert scaling["share_scaling_understates"] is not None
    assert 0.0 <= scaling["share_scaling_understates"] <= 1.0
    assert (bundle / "tables" / "09_horizon_scaling.csv").stat().st_size > 0


def test_the_summary_table_is_sorted_by_qlike(bundle):
    metrics = json.loads((bundle / "metrics.json").read_text())
    qlike = [row["qlike"] for row in metrics["volatility_scores"]]
    assert qlike == sorted(qlike)


def test_the_ranking_is_not_dominated_by_the_naive_benchmarks(bundle):
    """The study's claim is that a model earns its place. Check it can be earned.

    If a trailing sample variance beat every fitted model the honest thing would be
    to say so, but the reverse is the expected result and a silent inversion would
    mean the walk-forward engine is misaligned.
    """
    metrics = json.loads((bundle / "metrics.json").read_text())
    scores = {row["model"]: row["qlike"] for row in metrics["volatility_scores"]}
    best = min(scores, key=scores.get)
    assert best not in {"rolling_21", "rolling_63"}


def test_coverage_tables_are_complete_across_levels_and_horizons(bundle):
    metrics = json.loads((bundle / "metrics.json").read_text())
    seen = {(row["level"], row["horizon"]) for row in metrics["coverage"]}
    expected = {(level, horizon) for level in PIPELINE.confidence_levels for horizon in PIPELINE.horizons}
    assert seen == expected
    for row in metrics["coverage"]:
        assert row["observations"] > 200
        assert 0.0 <= row["observed_rate"] <= 1.0


def test_basel_columns_are_only_filled_at_the_ninety_nine_percent_level(bundle):
    metrics = json.loads((bundle / "metrics.json").read_text())
    for row in metrics["coverage"]:
        if abs(row["level"] - 0.99) < 1e-9:
            assert row["basel_zone"] != "n/a"
            assert row["basel_exceptions"] >= 0
        else:
            assert row["basel_zone"] == "n/a"
            assert row["basel_plus_factor"] is None  # NaN is not JSON-representable


def test_every_declared_stress_window_is_reported(bundle):
    metrics = json.loads((bundle / "metrics.json").read_text())
    windows = {row["window"] for row in metrics["stress_coverage"]}
    assert windows <= set(cfg.STRESS_WINDOWS)
    assert windows, "no stress window overlapped the sample"


def test_the_report_quotes_the_numbers_in_the_bundle(bundle):
    """A claim that cannot be traced to a file is a claim, not evidence."""
    metrics = json.loads((bundle / "metrics.json").read_text())
    report = (bundle / "RESULTS.md").read_text()
    best = metrics["volatility_scores"][0]
    assert best["model"] in report
    assert f"{best['qlike']:.4f}" in report
    assert f"{metrics['sample_start']}" in report
    assert f"{metrics['sample_end']}" in report


def test_the_report_does_not_leak_a_missing_value(bundle):
    """A markdown table with ``NaN`` in it is a rendering failure, not a result."""
    report = (bundle / "RESULTS.md").read_text()
    assert "NaN" not in report
    assert "nan" not in report.split()
    assert "None" not in report


def test_the_bundle_is_reproducible(bundle, scratch):
    """Same seed, same numbers. The seed is in the manifest for a reason."""
    second = scratch("p3_bundle_repeat")
    run_pipeline(PIPELINE, output_dir=second, verbose=False, render_figures=False)
    first_metrics = json.loads((bundle / "metrics.json").read_text())
    second_metrics = json.loads((second / "metrics.json").read_text())
    for key in ("volatility_scores", "coverage", "tick_loss", "stress_coverage", "basel", "horizon_scaling"):
        assert first_metrics[key] == second_metrics[key], key


def test_figures_can_be_skipped_without_losing_the_numbers(scratch):
    output = scratch("p3_bundle_nofig")
    run_pipeline(PIPELINE, output_dir=output, verbose=False, render_figures=False)
    assert (output / "metrics.json").exists()
    assert (output / "RESULTS.md").exists()
    assert not any((output / "figures").glob("*.png"))


def test_the_every_table_identifies_its_level_and_horizon(bundle):
    """Rows that cannot be told apart are not a table.

    The ES table used to report one row per method per level per horizon with neither
    key printed, so the same method appeared four times and nothing said which was
    which. The same goes for the tick-loss table. Both now carry the keys, and this
    asserts they cover every combination the configuration declares.
    """
    metrics = json.loads((bundle / "metrics.json").read_text())
    expected = {(level, horizon) for level in PIPELINE.confidence_levels for horizon in PIPELINE.horizons}
    for key in ("es_backtest", "tick_loss", "coverage"):
        seen = {(row["level"], row["horizon"]) for row in metrics[key]}
        assert seen == expected, key


@pytest.fixture(scope="module")
def documentation(bundle, scratch):
    """README.md and MODEL_DOCUMENTATION.md, rendered from the test run's bundle."""
    docs = scratch("p3_docs")
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_docs.py"
    subprocess.run(
        [sys.executable, str(script), "--metrics", str(bundle / "metrics.json"), "--output", str(docs)],
        check=True,
        capture_output=True,
        text=True,
    )
    return docs


#: Tokens a *numeric* formatter emits when it is handed a value it cannot render. Only the
#: NaN/infinity family belongs here. ``none`` and ``null`` deliberately do not, even though
#: they are what a Python ``None`` looks like printed: this document legitimately tabulates
#: ``none`` as the parameter count of a model with nothing to estimate and ``null`` as the
#: column of hypothesis being tested, and a token list that flags those is a token list
#: that gets switched off. A leaked Python ``None`` is caught separately, by the exact-word
#: check in the test below.
#:
#: Checking these inside *table cells* rather than anywhere in the document is the rest of
#: the point: ``MODEL_DOCUMENTATION.md`` uses the word NaN as a definition, when it explains
#: why the Diebold-Mariano diagonal is undefined, and a document-wide substring search flags
#: that correct sentence as a defect while missing nothing that a cell check misses.
MISSING_TOKENS = frozenset({"nan", "-nan", "+nan", "inf", "-inf", "+inf", "infinity", "-infinity"})


def _cells_holding_missing_values(text: str) -> list[str]:
    """Every table cell whose entire content is a missing-value token.

    Markup and a trailing percent sign are stripped first, because those are what the
    formatters wrap a value in: a failure through ``pct`` arrives as ``nan%`` and one
    through a bolded cell as ``**nan**``, and neither is caught by comparing the raw
    cell against the token.
    """
    cells: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        for cell in stripped.strip("|").split("|"):
            token = cell.strip().strip("`*_ ").removesuffix("%").strip()
            if token.lower() in MISSING_TOKENS:
                cells.append(cell.strip())
    return cells


def test_the_documentation_renders_from_the_bundle(documentation):
    """The renderer is part of the contract, so it is exercised with the pipeline.

    Run as a subprocess rather than by importing it: that is how a reader runs it, and
    it is the only way to catch a wiring error between the script's arguments and the
    paths it writes - the kind of thing that works in an interactive session and not
    from a Makefile.
    """
    readme = (documentation / "README.md").read_text()
    model_doc = (documentation / "MODEL_DOCUMENTATION.md").read_text()
    assert len(readme.splitlines()) > 150
    assert len(model_doc.splitlines()) > 150
    for text in (readme, model_doc):
        # A number the renderer could not format is a hole in a published table, and a
        # hole reads as a zero to anyone skimming. "n/a" is not in the token list because
        # it is a deliberate, defined rendering - it is what the pipeline writes for a
        # quantity that does not exist, such as the diagonal of the Diebold-Mariano
        # matrix, rather than a formatter failure.
        assert not _cells_holding_missing_values(text)
        assert "None" not in text
        # The documents are published, so an absolute path from the machine that
        # rendered them is both a leak and a broken link for every reader.
        assert str(Path.home()) not in text
        assert "/Users/" not in text
        assert "/private/var/" not in text


def test_the_documentation_names_every_model_and_every_route(documentation):
    readme = (documentation / "README.md").read_text()
    for model in cfg.MODEL_NAMES:
        assert f"`{model}`" in readme, model
    for method in cfg.VAR_METHOD_NAMES:
        assert f"`{method}`" in readme, method


def test_the_documentation_quotes_the_run_it_was_rendered_from(bundle, documentation):
    """The traceability claim, checked: the README's numbers are the bundle's numbers."""
    metrics = json.loads((bundle / "metrics.json").read_text())
    readme = (documentation / "README.md").read_text()
    best = metrics["volatility_scores"][0]
    assert f"{best['qlike']:.4f}" in readme
    assert metrics["sample_start"] in readme
    assert metrics["sample_end"] in readme
    assert f"{metrics['seed']}" in readme
    assert f"{metrics['rows']:,}" in readme


def test_the_documentation_embeds_figures_that_exist(documentation):
    readme = (documentation / "README.md").read_text()
    embedded = [line for line in readme.splitlines() if line.startswith("![")]
    assert len(embedded) >= 10
    docs_dir = documentation.resolve()
    for line in embedded:
        target = line.split("](", 1)[1].rstrip(")")
        # Shields.io badges are hosted images, not repository files; everything
        # else the document embeds must resolve to a file a clone actually has.
        if target.startswith(("http://", "https://")):
            continue
        assert (docs_dir / target).resolve().exists(), target
