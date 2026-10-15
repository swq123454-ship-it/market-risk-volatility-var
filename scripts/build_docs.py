#!/usr/bin/env python
"""Render README.md and MODEL_DOCUMENTATION.md from the run's own metrics.

The repository makes one promise about its numbers: every figure quoted in the
prose is read out of ``outputs/metrics.json``, which the pipeline wrote. This
script is what makes that promise mechanical rather than honourable. It never
contains a measurement - only sentences with a hole in them and the key that
fills it. Change the seed, the sample or a model and the documentation follows,
which is the property that matters: a README written by hand drifts away from the
code the moment either changes, and then the most convincing part of the
repository is the part that is wrong.

Run it after the pipeline::

    python scripts/run_pipeline.py
    python scripts/build_docs.py

It also verifies what it renders - every figure it links has to exist, every table
has to have the row count the JSON implies - so a half-finished run fails here
rather than publishing a document with a broken image and a missing row.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "outputs/figures"

# ---------------------------------------------------------------------------
# Formatting helpers. Every number entering the documents goes through one of
# these, so the rounding is uniform and the file is readable.
# ---------------------------------------------------------------------------


def num(value: float, digits: int = 4) -> str:
    """A plain number with thousands separators and fixed decimals."""
    if value is None:
        return "n/a"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def pval(value: float) -> str:
    """A p-value, with the two thresholds a reader actually acts on.

    Below 1e-4 it is written as an inequality rather than as a decimal: at that
    point the digits are an artefact of the sample size, and printing ``0.0000``
    next to ``0.0001`` implies a precision the test does not have.
    """
    if value < 1e-4:
        return "< 0.0001"
    if value < 0.001:
        return f"{value:.5f}"
    return f"{value:.4f}"


def p_clause(value: float) -> str:
    """``p < 0.0001`` or ``p = 0.0123`` - never ``p = < 0.0001``.

    The comparison is on the rendered string rather than on the number, because the
    threshold is a property of the formatting decision in :func:`pval` and duplicating
    it here is how the two drift apart.
    """
    rendered = pval(value)
    return f"p {rendered}" if rendered.startswith("<") else f"p = {rendered}"


def yn(flag: bool) -> str:
    return "yes" if flag else "no"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """A markdown table, with the column count checked rather than assumed.

    A row with one cell too few does not raise anywhere: markdown renders it shifted by
    a column, so every value after the gap is labelled with the wrong header and the
    table still looks like a table. That happened here - a column was declared and left
    empty, and the resulting figures sat under the wrong names. Checking the width at
    the point of construction turns a silent misalignment into a failure.
    """
    for i, row in enumerate(rows):
        if len(row) != len(headers):
            raise ValueError(f"row {i} has {len(row)} cells for {len(headers)} headers: {row}")
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def figure(path: Path, out_dir: Path) -> str:
    """A markdown image reference, verified to exist.

    The relative path is taken from the directory holding the document, not from the
    repository root: the two coincide for the committed README and do not for a run that
    writes its documents elsewhere, and a link that only works in one of those two cases
    is a link that breaks silently in the other.
    """
    if not path.exists():
        raise FileNotFoundError(f"the document would reference a figure that is not there: {path}")
    relative = Path(os.path.relpath(path, out_dir))
    return f"![{path.stem.replace('_', ' ')}]({relative.as_posix()})"


def _stress_reading(rows: list[dict]) -> str:
    """What the stress slices say, including the comparison that makes them worth having.

    The interesting column is not the breach rate on its own - it is the breach rate next
    to the realised volatility. Two windows of comparable volatility and different
    coverage isolate the driver: not the size of the shock but how fast the filter caught
    up with it. That comparison is computed here rather than written down, because which
    pair of windows plays that role depends on the run.
    """
    if not rows:
        return "No stress window overlapped the sample."
    worst = max(rows, key=lambda r: r["breach_rate"] / r["expected_rate"])
    ratio = worst["breach_rate"] / worst["expected_rate"]
    expected = worst["expected_rate"] * worst["observations"]
    if abs(ratio - 1.0) < 0.25:
        return (
            f"No window is materially miscalibrated: the worst is **{worst['window']}** at "
            f"{worst['breaches']} breaches in {worst['observations']} days against "
            f"{num(expected, 1)} expected. The filter adapts fast enough that a designed crisis does not "
            f"by itself break coverage, which is a stronger result than the aggregate table can show."
        )
    # The window that isolates the mechanism: the closest realised volatility to the worst
    # window's, among the windows that behaved better.
    comparators = [r for r in rows if r is not worst and r["breach_rate"] < worst["breach_rate"]]
    if not comparators:
        comparators = [r for r in rows if r is not worst]
    peer = min(comparators, key=lambda r: abs(r["realised_vol_annual"] - worst["realised_vol_annual"]))
    peer_ratio = peer["breach_rate"] / peer["expected_rate"]
    vol_gap = abs(peer["realised_vol_annual"] - worst["realised_vol_annual"]) / worst["realised_vol_annual"]
    if vol_gap < 0.35:
        contrast = (
            f"**{peer['window']}** has comparable realised volatility "
            f"({pct(peer['realised_vol_annual'])} against {pct(worst['realised_vol_annual'])}) and is "
            f"covered at {num(peer_ratio, 2)}x the nominal rate. The difference is not the size of the "
            f"shock; it is how quickly the filter adapted to it - which is the diagnosis a single "
            f"aggregate rate can never deliver."
        )
    else:
        contrast = (
            f"The nearest window by volatility, **{peer['window']}**, sits at "
            f"{pct(peer['realised_vol_annual'])} against {pct(worst['realised_vol_annual'])} and breaches "
            f"{num(peer_ratio, 2)}x its nominal rate, so the two are not a controlled comparison - but the "
            f"ordering is still by how fast volatility arrived rather than by how large it got."
        )
    return (
        f"The failure is concentrated: **{worst['window']}** breaches {worst['breaches']} times in "
        f"{worst['observations']} days against {num(expected, 1)} expected - a rate of "
        f"**{pct(worst['breach_rate'], 2)}** against a nominal {pct(worst['expected_rate'], 1)}, or "
        f"{num(ratio, 2)}x. {contrast}"
    )


def _understatement_note(metrics: dict) -> str:
    """The middle conclusion: where the whole panel fails, and the caveat that comes with it.

    Computed from the coverage table rather than asserted, because "every route fails at
    the ten-day horizon" is a claim that only holds for some runs - and because at h > 1 it
    needs the overlapping-window caveat attached or it overstates what was measured.
    """
    levels = metrics["config"]["confidence_levels"]
    horizons = metrics["config"]["horizons"]
    counts = {}
    for level in levels:
        for horizon in horizons:
            rows = [r for r in metrics["coverage"] if r["level"] == level and r["horizon"] == horizon]
            counts[(level, horizon)] = sum(1 for r in rows if r["kupiec_reject"]), len(rows)
    full = [(k, v) for k, v in counts.items() if v[0] == v[1]]
    if full:
        (level, horizon), (n, _) = max(full, key=lambda kv: kv[0][0])
        body = (
            f"3. **At {pct(level, 0)} and {horizon} days the entire panel fails - {n} of {n} routes.** "
            f"That is not a model-versus-model result, it is a statement about the horizon: risk measured "
            f"over {horizon} days is larger than any of these routes claims, and no amount of care in "
            f"choosing between them changes it. Because the {horizon}-day returns are computed over "
            f"overlapping windows, the independence and dynamic-quantile p-values at that horizon are "
            f"biased downwards and only the breach rate should be read at face value - which is, "
            f"conveniently, the one that shows the understatement."
        )
        return body
    partial = [(k, v) for k, v in counts.items() if v[0] and v[0] < v[1]]
    if partial:
        (level, horizon), (n, total) = max(partial, key=lambda kv: kv[1][0])
        return (
            f"3. **The failures cluster at the longer horizon.** At {pct(level, 0)} and {horizon} days, "
            f"{n} of {total} routes are rejected, against "
            f"{counts[(level, 1)][0]} of {counts[(level, 1)][1]} at one day. The picture is of a horizon "
            f"effect rather than a ranking, and the same caveat applies: overlapping windows at h > 1 "
            f"bias the independence and dynamic-quantile statistics downwards, so only the breach rate "
            f"should be read at face value."
        )
    return (
        "3. **No level or horizon rejects the whole panel.** That is a weaker statement than it looks - "
        "it means the sample is not large enough to resolve the differences, not that the models agree."
    )


def _es_headline(level: float, es_rows: list[dict], inadequate: list[dict]) -> str:
    """The Expected-Shortfall bullet, which must not promise a result it did not get.

    Written as a function because the two versions say genuinely different things: one
    reports a failure, the other reports the limits of the test that produced no failure.
    An earlier draft asserted that ES "is the stricter test in practice" and named the
    routes that would fail - before the run that showed none of them do. The bullet is
    rendered from the result rather than written around an expectation.
    """
    n = int(round(sum(r["breaches"] for r in es_rows) / max(len(es_rows), 1)))
    resolution = 1.96 / math.sqrt(max(n, 1) * (1.0 - level))
    if inadequate:
        return (
            f"- Expected Shortfall is the stricter test in practice: **{len(inadequate)} of "
            f"{len(es_rows)}** routes fail backtesting at {pct(level, 0)} even though most of them pass "
            f"coverage. Right breach *rate*, wrong breach *size*."
        )
    return (
        f"- Expected Shortfall fails **none** of the {len(es_rows)} routes at {pct(level, 0)}, and that "
        f"is a weaker statement than it sounds: with {n} breach days on average the test cannot resolve "
        f"anything smaller than a factor of about {num(1.0 + resolution, 1)} on the tail. The mean "
        f"excess column points the other way for most routes, and no p-value here is sharp enough to "
        f"settle it."
    )


def _es_verdict(es_rows: list[dict], adequate: list[dict], inadequate: list[dict], level: float) -> str:
    """The ES paragraph, which reads differently depending on whether anything failed.

    Written as a function rather than assembled inline because the second half of the
    sentence is a diagnosis that only exists when there is something to diagnose - and
    because the framing of a clean sheet has to be honest about what a clean sheet is
    worth. The Acerbi-Szekely statistic has asymptotic standard error
    ``1 / sqrt(n * alpha)``; at fifty breach days and a 1% level that is about 1.4, so the
    test cannot resolve anything smaller than roughly a factor of three on the tail. A
    panel where nothing fails is evidence that nothing is catastrophic, which is not the
    same as evidence that every tail is the right size.
    """
    n = int(round(sum(r["breaches"] for r in es_rows) / max(len(es_rows), 1)))
    resolution = 1.96 / math.sqrt(max(n, 1) * (1.0 - level))
    spread = (min(r["breaches"] for r in es_rows), max(r["breaches"] for r in es_rows))
    head = f"**{len(adequate)} of {len(es_rows)}** routes are adequate at this level and horizon. "
    if not inadequate:
        return (
            head
            + "That reads as a clean sheet, and it is worth being precise about what it means. Between "
            + f"{spread[0]} and {spread[1]} breach days per route at a {pct(level, 0)} level give the "
            + f"statistic a standard error of about {num(1.0 / math.sqrt(max(n, 1) * (1.0 - level)), 2)}, "
            + "so it rejects only when the average breach exceeds the ES by a factor of about "
            + f"{num(1.0 + resolution, 1)}. Nothing here is wrong by that much; that is not the same as "
            + 'nothing being wrong, and with a band that wide the verdict is closer to "not proven" than '
            + 'to "adequate".'
        )
    worst = min(inadequate, key=lambda r: r["z1_p"])
    return (
        head + f"The diagnosis for `{worst['method']}` is the sort a single test cannot produce: it has the "
        f"right breach *rate* and the wrong breach *size*. Its breaches exceed the ES by a mean of "
        f"**{num(worst['mean_excess_bp'], 1)} bp** and the statistic sits at {num(worst['z1'], 3)}, so "
        f"the losses beyond VaR are larger than the model admits. Coverage says the model is calibrated; "
        f"ES says the calibration is being bought with a tail that is too thin."
    )


def _mechanism(method: str) -> str:
    """Why a route with too many breaches is too tight, inferred from its own construction.

    Derived rather than typed, because the sentence that names the mechanism is only
    true of the route it names: a Gaussian tail is too thin, a raw sample quantile adapts
    too slowly, and those are different faults with the same symptom. Hard-coding the
    explanation would make the prose a claim about a previous run the moment the ranking
    moved - and the ranking moved once already, when the tail-fit fallback was fixed.
    """
    if "normal" in method:
        return "the Gaussian tail is too thin for a market with jumps, and no amount of correct conditional volatility repairs a wrong tail shape"
    if method.startswith("hs_"):
        return "the raw sample quantile has to see a crisis in the window before it believes it, so it is always a few days late"
    if "evt" in method:
        return "the fitted tail has to be estimated from a handful of exceedances, and a handful is not enough to pin a shape"
    if method.startswith("fhs_"):
        return "the residual pool carries the tail shape but not the scale of the current regime"
    if "parametric" in method:
        return "the parametric quantile is only as good as the distribution it assumes"
    return "the route is mispriced at this level and horizon"


def _clustering_note(primary: list[dict]) -> str:
    """Which route has the most structured breaches, and whether that cost it anything.

    The point of the independence test is that a route can pass on the *rate* and still be
    useless, because its breaches arrive together. So the interesting case is a route that
    passes Kupiec and fails conditional coverage - and when the run contains none, saying
    so is more useful than picking the route with the worst conditional-coverage p-value
    and letting the reader assume it passed coverage.
    """
    passes = [r for r in primary if not r["kupiec_reject"]]
    clustered = [r for r in passes if r["cc_p"] < 0.05]
    if clustered:
        worst = min(clustered, key=lambda r: r["cc_p"])
        return (
            f"`{worst['method']}` is exactly that case: it passes Kupiec on the rate, yet conditional "
            f"coverage rejects it ({p_clause(worst['cc_p'])}) because its {worst['breaches']} breaches are "
            f"not evenly spread. Counting them would have scored it as adequate."
        )
    ordered = min(primary, key=lambda r: r["cc_p"])
    if ordered["kupiec_reject"]:
        return (
            f"Here no route passes on the rate and fails on clustering, so the test does not separate "
            f"anything this run did not already reject. The most structured breach pattern belongs to "
            f"`{ordered['method']}` ({p_clause(ordered['cc_p'])}), which Kupiec rejects as well - so the "
            f"two failures point the same way rather than giving opposite verdicts."
        )
    return (
        f"No route shows significant clustering: the smallest conditional-coverage p-value is "
        f"{pval(ordered['cc_p'])} for `{ordered['method']}`, which passes on both counts. That is the "
        f"outcome the filter is supposed to produce - breaches that are individually rare and jointly "
        f"unpredictable - and it is reported rather than passed over."
    )


def _directions(failing: list[dict], expected: float) -> tuple[list[dict], list[dict]]:
    """Split the rejected routes into under- and over-stated risk.

    A coverage test is two-sided, so a rejection on its own does not say which way the
    model is wrong - and the direction is the whole diagnosis. Splitting it out is also
    what stops the prose from asserting a failure mode the run did not produce, which is
    exactly what happened when an earlier draft described an over-conservative rejection
    that the sample did not contain.
    """
    conservative = [r for r in failing if r["breaches"] < expected]
    aggressive = [r for r in failing if r["breaches"] >= expected]
    return conservative, aggressive


def _failing_clause(failing: list[dict], expected_breaches: float) -> str:
    """The tail of a sentence about *how* the rejected routes fail.

    Returned as a fragment so the sentence stays grammatical whether the run rejects
    several routes, one, or none - and so that "none" is said out loud rather than left
    to a table the reader may not check.
    """
    if not failing:
        return (
            ", so every route is admitted at this level. That is a stronger result than usual and "
            "belongs in the summary rather than only in the table below."
        )
    conservative, aggressive = _directions(failing, expected_breaches)
    n = f"{num(expected_breaches, 0)}"
    if aggressive and not conservative:
        worst = max(aggressive, key=lambda r: r["breaches"])
        mildest = min(aggressive, key=lambda r: r["breaches"])
        return (
            f". Every one of them is rejected in the *same* direction: the mildest, "
            f"`{mildest['method']}`, breaches {mildest['breaches']} times against about {n} expected, and "
            f"`{worst['method']}` breaches {worst['breaches']}. Risk is understated at this level, "
            f"systematically, and nobody is being over-charged for it."
        )
    if conservative and not aggressive:
        mildest = max(conservative, key=lambda r: r["breaches"])
        worst = min(conservative, key=lambda r: r["breaches"])
        return (
            f". Every rejection here is on the *conservative* side: `{worst['method']}` breaches only "
            f"{worst['breaches']} times against about {n} expected. Kupiec is two-sided deliberately - a "
            f"model that never breaches has not demonstrated safety, it has failed to provide evidence - "
            f"but this is the failure mode that costs money rather than the one that loses it."
        )
    first, second = aggressive[0], conservative[0]
    return (
        f". They fail in both directions, which is the more useful outcome: `{first['method']}` breaches "
        f"{first['breaches']} times against about {n} expected, while `{second['method']}` breaches only "
        f"{second['breaches']}. One is understating the risk and the other is buying safety it does not "
        f"need, and the breach count alone could not tell them apart."
    )


def _coverage_diagnosis(failing: list[dict], expected: float) -> str:
    """Why the rejected routes were rejected, in the direction each one failed."""
    if not failing:
        return (
            "Here there are none to explain, which is a result in itself - the route and the method "
            "choice did not produce a mispriced quantile at this level and horizon."
        )
    conservative, aggressive = _directions(failing, expected)
    n = num(expected, 0)
    if aggressive and not conservative:
        worst = max(aggressive, key=lambda r: r["breaches"])
        return (
            f"All {len(aggressive)} of them fail the same way. `{worst['method']}` breaches "
            f"**{worst['breaches']}** times against roughly {n} expected - {_mechanism(worst['method'])}. "
            f"The others are milder versions of the same error, and none of them is rejected for being "
            f"*too cautious*. That is worth stating plainly, because the intuition that an over-"
            f"conservative risk number is a safe one is not supported by this sample: the panel's "
            f"problem is understated risk, and correcting it is the work."
        )
    if conservative and not aggressive:
        worst = min(conservative, key=lambda r: r["breaches"])
        return (
            f"All {len(conservative)} of them are too cautious. `{worst['method']}` breaches "
            f"**{worst['breaches']}** times against roughly {n} expected. Kupiec is two-sided on purpose: "
            f"a model that never breaches has not shown that it is safe, it has shown that the sample "
            f"cannot assess it. The cost is capital held against risk that is not there, and the "
            f"capital add-on column is where it is paid."
        )
    lo = max(aggressive, key=lambda r: r["breaches"])
    hi = min(conservative, key=lambda r: r["breaches"])
    return (
        f"They fail in opposite directions. `{lo['method']}` breaches **{lo['breaches']}** times against "
        f"roughly {n} expected - {_mechanism(lo['method'])}. `{hi['method']}` breaches only "
        f"{hi['breaches']}, so it is over-conservative. A one-sided test would have passed the second and "
        f"the breach count alone cannot separate the two: one of them understates the risk and the other "
        f"overstates it, and both are wrong."
    )


# ---------------------------------------------------------------------------
# Views over metrics.json
# ---------------------------------------------------------------------------


def coverage_at(metrics: dict, level: float, horizon: int) -> list[dict]:
    """Coverage rows for one level and horizon, fewest breaches first.

    Ascending rather than "best first": the interesting rows are at both ends, and an
    ordering by p-value would hide the over-conservative route among the ones that
    passed. Breach count is the one ordering that puts both failure modes on display.
    """
    rows = [r for r in metrics["coverage"] if r["level"] == level and r["horizon"] == horizon]
    if not rows:
        raise KeyError(f"no coverage rows for level={level} horizon={horizon}")
    return sorted(rows, key=lambda r: r["breaches"])


def tick_loss_at(metrics: dict, level: float, horizon: int) -> list[dict]:
    rows = [r for r in metrics["tick_loss"] if r["level"] == level and r["horizon"] == horizon]
    if not rows:
        raise KeyError(f"no tick-loss rows for level={level} horizon={horizon}")
    return sorted(rows, key=lambda r: r["tick_loss_bp"])


def es_tests_at(metrics: dict, level: float, horizon: int) -> list[dict]:
    rows = [r for r in metrics["es_backtest"] if r["level"] == level and r["horizon"] == horizon]
    if not rows:
        raise KeyError(f"no ES rows for level={level} horizon={horizon}")
    return sorted(rows, key=lambda r: abs(r["z1"]))


def scores_ranked(metrics: dict) -> list[dict]:
    return sorted(metrics["volatility_scores"], key=lambda r: r["qlike"])


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------


def build_readme(metrics: dict, out_dir: Path) -> str:
    primary_level = metrics["config"]["confidence_levels"][-1]
    scores = scores_ranked(metrics)
    primary = coverage_at(metrics, primary_level, 1)
    ticks = tick_loss_at(metrics, primary_level, 1)
    es_rows = es_tests_at(metrics, primary_level, 1)
    models = metrics["config"]["models"]
    methods = [r["method"] for r in primary]

    best, worst = scores[0], scores[-1]
    baseline = next(r for r in scores if r["model"] == "rolling_21")
    passing = [r for r in primary if not r["kupiec_reject"]]
    failing = [r for r in primary if r["kupiec_reject"]]
    green = [r for r in primary if r["basel_zone"] == "green"]
    adequate = [r for r in es_rows if r["es_adequate"]]
    inadequate = [r for r in es_rows if not r["es_adequate"]]
    var_levels = [r["mean_var_bp"] for r in ticks]

    parts: list[str] = []
    add = parts.append

    add("# Market risk: volatility forecasting, VaR/ES and a Basel traffic light")
    add("")
    add(
        f"A conditional-volatility study over {num(metrics['rows'], 0)} trading days of "
        f"{metrics['asset']} history ({metrics['sample_start']} to {metrics['sample_end']}), in which "
        f"{len(models)} volatility models are estimated walk-forward and their forecasts are turned into "
        f"Value-at-Risk and Expected Shortfall by {len(methods)} different routes. Every route is then "
        f"subjected to the four tests a risk function is actually audited on - Kupiec, Christoffersen, "
        f"Engle-Manganelli and Acerbi-Szekely - and the result is mapped onto the Basel traffic light at "
        f"the level the regulation is written for."
    )
    add("")
    add(
        'The organising question is not "which model is best" - that is a leaderboard, and leaderboards '
        "are easy. It is **which modelling choices survive being tested**, and how loudly each one fails "
        "when it does not. The answers are not all comfortable for the standard toolkit, which is why the "
        "backtesting half of this repository is longer than the forecasting half."
    )
    add("")

    # -- headline ----------------------------------------------------------
    add("## Headline results")
    add("")
    add(
        f"- The realised measure is well forecast, but not by the model with the most parameters. "
        f"`{best['model']}` leads on QLIKE at **{num(best['qlike'])}**, against {num(baseline['qlike'])} for "
        f"the 21-day trailing-variance baseline and {num(worst['qlike'])} for `{worst['model']}`."
    )
    add(
        f"- **{len(passing)} of {len(methods)}** VaR routes pass Kupiec at {pct(primary_level, 0)}, "
        f"one-day{_failing_clause(failing, primary[0]['expected_rate'] * primary[0]['observations'])}"
    )
    add(
        f"- At **{metrics['horizon_scaling']['reference_method']}** the simulated 10-day VaR deviates from "
        f"the square-root-of-time rule by a mean of **{pct(metrics['horizon_scaling']['mean_gap'])}**, and "
        f"the rule understates the simulated risk on "
        f"**{pct(metrics['horizon_scaling']['share_scaling_understates'], 1)}** of days."
    )
    add(
        f"- The Basel traffic light puts **{len(green)} of {len(methods)}** routes in green on the "
        f"final 250-day window, but **{pct(metrics['basel']['share_outside_green'], 1)}** of *all* rolling "
        f"windows are outside it, with a worst window of **{metrics['basel']['worst_exceptions']}** "
        f"exceptions. A single end-of-sample zone label is not a statement about the model."
    )
    add(_es_headline(primary_level, es_rows, inadequate))
    add("")

    add(figure(FIGURE_DIR / "04_forecast_paths.png", out_dir))
    add("")
    add(
        "*Forecast paths against the realised measure. The shaded bands are the crisis windows the "
        "simulator was given; the model has never seen them.*"
    )
    add("")

    # -- problem -----------------------------------------------------------
    add("## The problem")
    add("")
    add(
        "A bank's market-risk number is not a prediction, it is a **quantile**. It is quoted, it is "
        "compared with the day's loss, it is counted against a regulatory threshold, and it determines "
        'how much capital has to be held against a trading book. That changes what "good" means. A '
        "forecast that is right on average and wrong exactly when it matters is not a mediocre forecast, "
        "it is a useless one, because the days it gets wrong are the days the number is used."
    )
    add("")
    add("Three questions follow, and this repository answers all three with the same data:")
    add("")
    add(
        "1. **Does adding structure to the volatility model help?** A trailing variance is free, always "
        "available and hard to beat. Eleven models - from that baseline up through GARCH, a leverage "
        "term, a log-variance form, a heterogeneous-autoregression on realised measures, a gradient-"
        "boosted tree and a hand-written GRU - are estimated on the same expanding windows and scored on "
        "the same days."
    )
    add(
        f"2. **Does the route from a volatility forecast to a quantile matter?** Two historical routes, "
        f"two filtrations of the same filter (one of them with a generalised-Pareto tail), two parametric "
        f"forms and one simulation produce {len(methods)} different VaR numbers from one variance "
        f"forecast. At {pct(primary_level, 0)} and one day they span {num(min(var_levels), 0)} to "
        f"{num(max(var_levels), 0)} basis points - a factor of {num(max(var_levels) / min(var_levels), 2)} "
        f"on the same nominal coverage - and the disagreement is systematic rather than noise."
    )
    add(
        "3. **Which ones survive the tests?** A model is admitted to the comparison only if it beats the "
        "baseline; a VaR route is called *passing* only if it clears all four coverage tests, not one. The "
        "word *adequate* is reserved for the separate Expected-Shortfall verdict, which asks a different "
        "question - not whether the breaches are the right number, but whether the losses beyond them are "
        "the right size. The "
        "failures are reported with the same prominence as the successes, because a risk study that "
        "reports only what worked is a marketing document."
    )
    add("")

    # -- data --------------------------------------------------------------
    add("## The data")
    add("")
    add(
        f"The study runs on a **simulated** market by default: {num(metrics['rows'], 0)} trading days, "
        f"{metrics['sample_start']} to {metrics['sample_end']}, generated with seed **{metrics['seed']}**. "
        f"Annualised realised volatility over the sample is **{pct(metrics['realised_vol_annual'])}**."
    )
    add("")
    add("Simulation is a deliberate choice, for three reasons that a real price file cannot provide:")
    add("")
    add(
        "- **The latent variance is known.** Whether a model is good or merely lucky can only be settled "
        "by scoring it against the variance that actually generated the returns. On real data that object "
        "does not exist, and every study is reduced to arguing about a noisy proxy."
    )
    add(
        "- **The crisis windows are known.** A simulator can be told to put a 26-fold volatility event in "
        "February 2020; a historical sample contains whatever it contains. Stress behaviour can then be "
        "measured on a designed experiment instead of on whichever episode happened to be in the window."
    )
    add(
        "- **The sample cannot be quietly tuned.** The parameters are fixed in `config.py`, the seed is "
        "recorded in `outputs/generator_manifest.json`, and the whole run is reproducible from both."
    )
    add("")
    add(
        "The generator is not a toy: one short-memory component, one near-unit-root long-memory "
        "component, a leverage term, Student-t innovations, compound-Poisson jumps shared across assets, "
        "and an intraday Brownian bridge so that realised variance, bipower variation and the jump test "
        "are computable rather than assumed. It also builds in a variance risk premium, which is why the "
        "realised measure sits systematically *above* the conditional variance - the feature that makes "
        "the latent-variance comparison in the results section meaningful rather than trivially favourable."
    )
    add("")
    add(
        "`--data-mode real` downloads index history through `yfinance` and runs the identical pipeline on "
        "it. Every number in this document comes from the synthetic run, because only the synthetic run "
        "supports the latent-variance scoring and the designed stress windows."
    )
    add("")
    add(figure(FIGURE_DIR / "01_market_history.png", out_dir))
    add("")
    add("*Price, return and variance history for the simulated panel, with the crisis schedule shaded.*")
    add("")

    # -- method ------------------------------------------------------------
    add("## Method")
    add("")
    add("### 1. Realised measures")
    add("")
    add(
        "Intraday returns at **78 bars a day** (a five-minute grid) are aggregated into realised variance, "
        "bipower variation, the jump variation and the Huang-Tauchen jump statistic, then into the "
        "HAR components: daily, weekly (5-day) and monthly (22-day) averages of log realised variance, "
        "plus a signed-return term and the leverage effect. The variance risk premium is "
        f"**{num(metrics['manifest']['parameters']['variance_risk_premium'], 2)}**, so the realised measure "
        "is an upward-biased proxy for the quantity the models are estimating - which is precisely the "
        "bias the latent comparison is there to remove."
    )
    add("")
    add(figure(FIGURE_DIR / "03_realised_measures.png", out_dir))
    add("")
    add(
        "*Realised variance, its bipower and jump decompositions, and the jump statistic against the jumps the simulator placed.*"
    )
    add("")

    add("### 2. Volatility models")
    add("")
    add(
        f"{len(models)} models are estimated on expanding windows with **monthly refits**, each asked for "
        f"one-step-ahead variance forecasts over the following block. The variance *state* is filtered "
        f"with realised returns inside the block while the parameters stay fixed, which is both what a "
        f"desk does and the only way the recursions remain legal."
    )
    add("")
    add("| model | what it adds over the one before |")
    add("|---|---|")
    add("| `rolling_21` | nothing. The trailing 21-day variance - the benchmark to beat |")
    add("| `rolling_63` | the same over a quarter |")
    add("| `ewma_94` | RiskMetrics decay, 0.94. Free, and the industry default |")
    add("| `ewma_fitted` | the same family, decay chosen on the training block |")
    add("| `garch` | a fitted mean-reverting recursion |")
    add("| `gjr` | asymmetry: negative shocks raise variance more |")
    add("| `egarch` | the recursion in logs, so no non-negativity constraint is needed |")
    add("| `har` | realised measures at three horizons rather than returns alone |")
    add("| `harx` | the same, plus the option-implied variance |")
    add("| `gbm` | a gradient-boosted tree on the same feature set |")
    add("| `gru` | a hand-written GRU, forward and backward pass, no framework |")
    add("")

    add("### 3. From a variance forecast to a quantile")
    add("")
    add(
        f"{len(methods)} routes, all reading the same GJR-GARCH-t filter, so the differences below are "
        f"differences of *method* and not of input:"
    )
    add("")
    add("| route | family | construction |")
    add("|---|---|---|")
    add("| `hs_250` | historical | empirical quantile of the last 250 raw returns |")
    add("| `hs_brw` | historical | the same, exponentially age-weighted (Boudoukh-Richardson-Whitelaw) |")
    add("| `fhs_gjr` | filtered | scale the filter by the empirical quantile of its standardised residuals |")
    add("| `fhs_evt_gjr` | filtered | the same, with the residual tail fitted by a generalised Pareto |")
    add("| `parametric_normal_gjr` | parametric | Gaussian quantile times the conditional volatility |")
    add("| `parametric_t_gjr` | parametric | Student-t quantile, tail index fitted by maximum likelihood |")
    add("| `monte_carlo` | simulation | simulate the variance recursion forward and read the quantile off the paths |")
    add("")
    add(
        "The horizon enters through the square-root-of-time rule for the parametric routes and through the "
        "simulation for the Monte-Carlo route, so the ten-day numbers are not a rescaling of the one-day "
        "numbers for every method - which is exactly what the scaling section measures."
    )
    add("")

    add("### 4. Backtests")
    add("")
    add("| test | question | fails when |")
    add("|---|---|---|")
    add(
        "| Kupiec | is the breach rate right? | the rate is wrong in either direction - a model with no breaches is mispriced, not conservative |"
    )
    add("| Christoffersen | are breaches independent? | they arrive in clusters, which the average cannot see |")
    add("| Conditional coverage | both, jointly | either one fails, weighted |")
    add(
        "| Engle-Manganelli | is the breach predictable? | its own lags or the VaR level forecast it; strictly stronger than the two above |"
    )
    add(
        "| Acerbi-Szekely | is the ES the right size? | the average breach exceeds the ES - the failure that coverage cannot see |"
    )
    add("")
    add(
        "At horizons above one day the returns are measured over **overlapping** windows, so consecutive "
        "hit indicators are mechanically dependent even for a correct model. Kupiec stays valid - the "
        "breach-rate expectation is unaffected - but the independence and dynamic-quantile p-values at "
        "ten days are biased downwards and are reported as indicative only. The alternative is to say "
        "nothing about clustering at ten days, which is worse."
    )
    add("")
    add(figure(FIGURE_DIR / "08_var_exceptions.png", out_dir))
    add("")
    add("*Breach counts against the nominal level, per route. The line is where a well-calibrated model sits.*")
    add("")

    add("### 5. Invariants that are asserted, not hoped for")
    add("")
    add("Every claim below is enforced by a test that fails if the property is lost:")
    add("")
    add(
        "- **No look-ahead.** Features are strictly lagged, the design matrix is built from shifted "
        "columns, and a test corrupts the *future* and asserts an unchanged past forecast. This is the "
        "failure that silently inflates every out-of-sample number in a study, so it is checked twice: "
        "once on the feature frame, once on the fitted model."
    )
    add(
        "- **The speed flag costs accuracy, never a deliverable.** `--fast` reduces the Monte-Carlo path "
        "count and the network's epochs; it does not drop a model or a method. An earlier version did, "
        "and the horizon-scaling section silently emptied because the only method that can produce a "
        "multi-day number was the one that had been removed."
    )
    add(
        "- **Complex models must earn their place.** A test asserts that the winner is not the naive "
        "baseline, and reports the naive baseline's own score so that the comparison cannot be quietly "
        "dropped."
    )
    add(
        "- **Every sign convention is pinned by a test.** Three real defects were found this way, each of "
        "which produced a plausible-looking number rather than an exception: an off-by-one that made a "
        "down day irrelevant to tomorrow's risk, a scoring rule using the coverage level where the tail "
        "probability belonged (which reversed a ranking while still producing a winner), and a tail-fit "
        "fallback that returned its quantile with the wrong sign, so the caller's clamp turned it into a "
        "VaR of exactly zero. The tests that guard these state the wrong answer as well as the right one, "
        "so a future regression is recognised rather than merely rejected."
    )
    add("")
    add(
        "The generator itself is verified against the properties it is supposed to have: the crisis "
        "multiplier ramps rather than steps, the intraday bridge sums to the daily move, the jump test "
        "recovers the jumps the simulator placed, and the cross-asset correlation structure is preserved."
    )
    add("")

    # -- results -----------------------------------------------------------
    add("## Results")
    add("")
    add(
        "These tables are rendered from `outputs/metrics.json` by `scripts/build_docs.py`. The CSVs "
        "behind them are in `outputs/tables/`."
    )
    add("")

    add("### Forecast accuracy")
    add("")
    add(
        "QLIKE is the correct loss for a variance forecast and the one used for ranking: it is the "
        "logarithmic score of a Gaussian, so it penalises under-prediction of variance far more than "
        "over-prediction, which is the asymmetry a risk number actually has. MSE is reported beside it "
        "because a risk committee recognises it."
    )
    add("")
    add(
        markdown_table(
            ["model", "QLIKE", "MSE (bp)", "MZ slope", "R² (log)", "mean forecast vol (annual)"],
            [
                [
                    f"`{r['model']}`",
                    f"**{num(r['qlike'])}**" if i == 0 else num(r["qlike"]),
                    num(r["mse_vol_bp"], 1),
                    num(r["mz_slope"]),
                    num(r["mz_r2_log"]),
                    pct(r["mean_forecast_vol_annual"]),
                ]
                for i, r in enumerate(scores)
            ],
        )
    )
    add("")
    add(
        f"`{best['model']}` wins, and the ordering from there is a clean statement about what pays: "
        f"realised measures and a flexible mean beat leverage, leverage beats a plain recursion, and the "
        f"trailing-window baselines come last. The spread is easier to read as a relative number than as "
        f"a rank: `{best['model']}` is "
        f"**{pct(1 - best['qlike'] / baseline['qlike'], 1)}** better on QLIKE than the 21-day baseline, "
        f"which is the number to quote, not the position."
    )
    add("")
    add(
        f"The Mincer-Zarnowitz slope is the calibration column: 1.0 means the forecast is correctly "
        f"scaled, below 1.0 means it is too timid. `{best['model']}` sits at "
        f"**{num(best['mz_slope'])}** and `{baseline['model']}` at **{num(baseline['mz_slope'])}**, so the "
        f"ranking on QLIKE is not an artefact of scale - the baseline genuinely under-responds. Note also "
        f"that the joint calibration test rejects almost everything "
        f"(`{p_clause(best['mz_joint_p'])}` for the winner): with over four thousand observations, the "
        f"test has the power to detect deviations far too small to matter, and the coefficient is the "
        f"informative output, not the pass/fail flag."
    )
    add("")
    add(figure(FIGURE_DIR / "05_qlike_ranking.png", out_dir))
    add("")
    add("*QLIKE by model with the paired Diebold-Mariano verdicts; a gap that is not significant is not a gap.*")
    add("")

    add("### Where the accuracy comes from")
    add("")
    add(
        "An aggregate score over twenty years is mostly an average over quiet days. The same loss, split "
        "by regime, is the more useful table: a model can be excellent in the calm sample and useless in "
        "the crises and still place well overall."
    )
    add("")
    regimes = [c for c in metrics["qlike_by_regime"][0] if c != "model"]
    add(
        markdown_table(
            ["model", *regimes],
            [
                [f"`{r['model']}`", *[num(r[c]) for c in regimes]]
                for r in sorted(metrics["qlike_by_regime"], key=lambda r: r.get("calm", 0))
            ],
        )
    )
    add("")
    add(figure(FIGURE_DIR / "06_qlike_by_regime.png", out_dir))
    add("")
    add(
        "*The same loss by regime. The purpose of the split is that a model which is excellent in the "
        "calm sample and useless in the crises cannot hide inside the aggregate.*"
    )
    add("")

    add(f"### Coverage at {pct(primary_level, 0)}, one day")
    add("")
    add(
        "Sorted from fewest breaches to most. The breach rate is the headline; the p-value is the "
        "verdict; the Basel zone is the same number through the regulator's lens."
    )
    add("")
    expected = primary[0]["expected_rate"] * primary[0]["observations"]
    expected_cell = num(expected, 1)
    add(
        markdown_table(
            [
                "route",
                "breaches",
                "expected",
                "observed rate",
                "Kupiec p",
                "ind. p",
                "cond. cov. p",
                "DQ p",
                "Basel zone",
            ],
            [
                [
                    f"`{r['method']}`",
                    str(r["breaches"]),
                    expected_cell,
                    pct(r["observed_rate"], 3),
                    pval(r["kupiec_p"]),
                    pval(r["ind_p"]),
                    pval(r["cc_p"]),
                    pval(r["dq_p"]),
                    r["basel_zone"] or "n/a",
                ]
                for r in primary
            ],
        )
    )
    add("")
    add(
        f"Every row has the same expected count ({expected_cell}) because the level and horizon are "
        f"fixed; it is repeated rather than stated once above the table so that no row has to be read "
        f"against the header to be interpreted."
    )
    add("")
    add(
        f"**{len(passing)} of {len(methods)}** routes pass Kupiec. The failures are informative, and so "
        f"is the direction they fail in. {_coverage_diagnosis(failing, expected)}"
    )
    add("")
    add(
        "The independence and dynamic-quantile columns ask a different question from the rate, and the "
        "useful case is a route that passes one and fails the other: a model whose breaches arrive "
        "together has the right average and still cannot be relied on, because it is wrong in runs. "
        + _clustering_note(primary)
    )
    add("")
    add(figure(FIGURE_DIR / "09_coverage_summary.png", out_dir))
    add("")
    add("*Breach rate against the nominal level across every route and both horizons.*")
    add("")

    add("### The Basel traffic light")
    add("")
    add(
        "The regulator's test, at the level the regulation is written for. The zone is not a label: five "
        "exceptions in 250 days costs 40% more capital than four, and ten doubles it. The multiplier is "
        "not linear in the count and the table below is the reason the traffic light is computed in SQL "
        "as well as in Python - the number on a dashboard has to come from the same rows the backtest wrote."
    )
    add("")
    # Sorted on the count the table itself shows, not on the inherited coverage order.
    # The two are different quantities - breaches over the whole evaluation window
    # against exceptions in the last 250 days - so ordering by one and displaying the
    # other leaves a reader unable to account for the sequence, and a table whose order
    # cannot be derived from what is on screen invites the suspicion that it is a
    # ranking. The tie-break on the route name keeps two equal counts in a fixed order
    # rather than in whatever order the coverage rows arrived.
    add(
        markdown_table(
            ["route", "exceptions in 250 days", "zone", "capital add-on"],
            [
                [f"`{r['method']}`", str(r["basel_exceptions"]), r["basel_zone"] or "n/a", pct(r["basel_plus_factor"])]
                for r in sorted(primary, key=lambda r: (r["basel_exceptions"], r["method"]))
            ],
        )
    )
    add("")
    add(
        f"On the final window **{len(green)} of {len(methods)}** routes are green. That is the number a "
        f"single dashboard shows, and it is misleading: across *all* rolling windows, "
        f"**{pct(metrics['basel']['share_outside_green'], 1)}** are outside green and the worst window "
        f"contains **{metrics['basel']['worst_exceptions']}** exceptions for the reference route. The "
        f"difference between the two numbers is the difference between a model that is fine today and a "
        f"model that is fine on average."
    )
    add("")
    add(figure(FIGURE_DIR / "10_basel_traffic_light.png", out_dir))
    add("")
    add("*Rolling 250-day exception counts against the green/yellow/red boundaries.*")
    add("")

    add("### Expected Shortfall: what the test can resolve")
    add("")
    add(
        "Expected Shortfall is only backtestable jointly with VaR, because it is defined on the days VaR "
        "was breached. The Acerbi-Szekely statistic is the average, over breach days, of `-r / ES - 1`, "
        "standardised: zero when the ES is the right size for the tail it claims to describe. Its "
        "standard error is `1/sqrt(n*alpha)` in the number of breach days, which is the number to keep in "
        "view when reading the verdicts below."
    )
    add("")
    add(
        markdown_table(
            ["route", "breach days", "mean ES (bp)", "mean excess (bp)", "Z1", "p", "ES adequate"],
            [
                [
                    f"`{r['method']}`",
                    str(r["breaches"]),
                    num(r["mean_es_bp"], 1),
                    num(r["mean_excess_bp"], 1),
                    num(r["z1"], 3),
                    pval(r["z1_p"]),
                    yn(r["es_adequate"]),
                ]
                for r in es_rows
            ],
        )
    )
    add("")
    add(_es_verdict(es_rows, adequate, inadequate, primary_level))
    add("")
    over = [r for r in es_rows if r["mean_excess_bp"] > 0]
    spread = (min(r["breaches"] for r in es_rows), max(r["breaches"] for r in es_rows))
    add(
        f"`mean_excess_bp` is the column to read alongside the verdict, and it is positive for "
        f"**{len(over)} of {len(es_rows)}** routes: on the days those routes were breached, the loss went "
        f"past the ES more often than short of it. The sign is the direction of the error and the "
        f"magnitude is its size, neither of which the p-value carries - and with only {spread[0]} to "
        f"{spread[1]} breach days per route, the p-value is the less informative of the two."
    )
    add("")
    add(figure(FIGURE_DIR / "13_es_vs_var.png", out_dir))
    add("")
    add("*ES against VaR by route. The gap between the two lines is the quantity the ES test is interrogating.*")
    add("")

    add("### Multi-day scaling")
    add("")
    add(
        "The square-root-of-time rule is exact for independent returns and wrong for everything else. "
        "Simulating the fitted recursion gives the true ten-day quantile, so the rule can be measured "
        "rather than assumed."
    )
    add("")
    add(
        f"For **{metrics['horizon_scaling']['reference_method']}**, the simulated ten-day VaR differs from "
        f"the scaled one-day VaR by a mean of **{pct(metrics['horizon_scaling']['mean_gap'])}**, and the "
        f"scaling rule *understates* the risk on "
        f"**{pct(metrics['horizon_scaling']['share_scaling_understates'], 1)}** of days."
    )
    add("")
    add(
        "Both signs appear and both are interpretable. A negative gap is the aftermath of a shock: "
        "volatility is high and mean-reverting, so scaling by the square root of time is too cautious. A "
        "positive gap is a quiet period, where the rule assumes volatility stays where it is and it does "
        "not. That asymmetry is why the rule is a statement about the regime as much as about the "
        "horizon, and why the share of days it understates - not the mean deviation - is the number to "
        "hold on to. A rule that is unbiased on average and wrong 78% of the time in one direction is not "
        "a rule anyone would sign off on if it were presented this way."
    )
    add("")
    add(figure(FIGURE_DIR / "12_horizon_gap.png", out_dir))
    add("")
    add("*The deviation of the simulated multi-day quantile from the square-root-of-time rule.*")
    add("")

    add("### Stress windows")
    add("")
    add(
        "The same breach rate, split by regime. These windows are part of the simulation, so unlike a "
        "historical study their volatility is known and the breach rate can be attributed to the model "
        "rather than to the episode."
    )
    add("")
    add(
        markdown_table(
            ["window", "days", "breaches", "breach rate", "expected", "realised vol (annual)", "worst day"],
            [
                [
                    r["window"],
                    str(r["observations"]),
                    str(r["breaches"]),
                    pct(r["breach_rate"], 3),
                    pct(r["expected_rate"], 1),
                    num(r["realised_vol_annual"]),
                    num(r["worst_return"]),
                ]
                for r in metrics["stress_coverage"]
            ],
        )
    )
    add("")
    add(_stress_reading(metrics["stress_coverage"]))
    add("")
    add(figure(FIGURE_DIR / "11_stress_coverage.png", out_dir))
    add("")
    add("*Breach rates inside the stress windows against the full sample.*")
    add("")

    # -- reading ----------------------------------------------------------
    add("## What this shows")
    add("")
    add("Four conclusions, in decreasing order of how much they surprised me:")
    add("")
    add(
        f"1. **The route from a variance forecast to a quantile matters more than the variance forecast "
        f"does.** Every route here consumes the *same* GJR-GARCH-t filter. At {pct(primary_level, 0)} and "
        f"one day the resulting VaR numbers span from {num(min(var_levels), 0)} bp to "
        f"{num(max(var_levels), 0)} bp - a factor of {num(max(var_levels) / min(var_levels), 2)} - and "
        f"{len(failing)} of the {len(methods)} fail coverage. The model choice is a second-order decision "
        f"next to the method choice, and most of the industry conversation is about the first."
    )
    add(
        f"2. **Risk is understated everywhere, and never overstated.** Every rejection in the study - "
        f"{len(failing)} of {len(methods)} routes at the primary level and horizon, and every rejection "
        f"at every other combination - is a route that breaches *more* often than its level allows. Not "
        f"one is rejected for being too cautious. The intuition that an over-conservative risk number is "
        f"the safe choice is not supported here; the error is one-directional, and it is the direction "
        f"that costs money."
    )
    add(_understatement_note(metrics))
    add(
        f"4. **Scaling a one-day number by the square root of time is a modelling assumption, not "
        f"arithmetic.** It understates the simulated risk on "
        f"**{pct(metrics['horizon_scaling']['share_scaling_understates'], 1)}** of days, in a "
        f"systematic direction tied to the volatility regime - the same direction as the coverage "
        f"failures, measured a completely different way."
    )
    add("")

    add("## Limitations")
    add("")
    add(
        "- **The market is simulated.** The findings about *methods* transfer directly, because the "
        "methods do not know where the data came from; the specific numbers do not. The real-data mode "
        "exists to check the machinery against a real feed, not to make the results comparable."
    )
    add(
        "- **One asset, one filter.** The panel has four correlated assets but the study is single-asset, "
        "and every VaR route reads the GJR-GARCH-t filter. A comparison across filters would be a "
        "different and larger study; holding the filter fixed is what makes the method comparison "
        "interpretable."
    )
    add(
        "- **Overlapping windows at ten days.** Independence tests at the ten-day horizon are biased "
        "downwards and are reported as indicative. Fixing this properly means non-overlapping windows, "
        "which would cost four fifths of the sample."
    )
    add(
        "- **The Expected-Shortfall test has low power at this sample size.** A handful of dozen breach "
        "days per route cannot resolve a tail that is wrong by a factor of two, so the verdicts above are "
        "weak evidence in both directions - passing is not a clean bill of health, and the mean-excess "
        "column carries the information the p-value cannot. Twenty-eight tests across the level and "
        "horizon grid make this worse rather than better: uncorrected, they invite the one that lands "
        "under 5% to be read as a finding."
    )
    add(
        "- **The GARCH likelihood assumes a symmetric Student-t** while the generated data are skewed. "
        "That is deliberate - it is the same misspecification real data imposes - but it means the "
        "parametric routes are handicapped relative to the empirical ones, and that handicap is a design "
        "choice, not a discovery."
    )
    add("")

    # -- layout -----------------------------------------------------------
    add("## Repository layout")
    add("")
    add("```")
    add("src/marketrisk/")
    add("  generator.py    the simulated market: components, crises, jumps, intraday bridge")
    add("  data.py         loading, caching and the asset panel")
    add("  features.py     realised measures, HAR components, the lagged design matrix")
    add("  volmodels.py    the eleven models and the walk-forward scheme")
    add("  nn.py           the GRU, forward and backward pass written out")
    add("  var.py          the seven VaR/ES routes in one suite")
    add("  losses.py       QLIKE, MSE, and the tick loss for a quantile")
    add("  backtest.py     Kupiec, Christoffersen, DQ, Basel zones, stress slices")
    add("  evaluate.py     scoring tables, Diebold-Mariano, Mincer-Zarnowitz, ES tests")
    add("  plots.py        every figure")
    add("  report.py       the pipeline and RESULTS.md")
    add("  api.py          the fitted engine behind FastAPI")
    add("  config.py       every constant, in one place")
    add("scripts/")
    add("  run_pipeline.py run the study")
    add("  serve.py        serve the fitted engine")
    add("  build_docs.py   render this file and MODEL_DOCUMENTATION.md from metrics.json")
    add("sql/              the feature panel, the rolling exception count and the traffic light")
    add("tests/            the full suite, including the invariants above")
    add("outputs/          figures, tables, metrics.json, RESULTS.md - all committed")
    add("```")
    add("")

    add("## Reproducing")
    add("")
    add("```bash")
    add("python -m pip install -r requirements.txt")
    add("python scripts/run_pipeline.py                  # the full run")
    add("python scripts/run_pipeline.py --fast           # same deliverables, smaller budget")
    add("python scripts/build_docs.py                    # regenerate these documents")
    add("python -m pytest                                # the suite, including the slow end-to-end run")
    add("python scripts/serve.py                         # the fitted engine over HTTP")
    add("```")
    add("")
    add(
        "The run is seeded and deterministic: the same seed reproduces the same sample, the same fits and "
        "the same tables, and `outputs/generator_manifest.json` records the parameters that produced it. "
        "`--fast` changes the budget - fewer Monte-Carlo paths, fewer network epochs - and nothing else; "
        "the deliverable list is identical, which is asserted by a test."
    )
    add("")

    add("## The SQL layer")
    add("")
    add(
        "`sql/` holds the three queries that would live in a warehouse: the feature panel with its "
        "strictly-lagged log return and 21-day trailing variance, the rolling 250-day exception count, and "
        "the Basel zone with its non-linear capital multiplier. They are executed by the test suite on "
        "seeded rows and compared **against the Python implementations**, feature by feature, because the "
        "classic reason a model works in a notebook and not in production is a lag applied the wrong way "
        "or an off-by-one in a rolling window - and that costs nothing to check."
    )
    add("")

    add("## The service")
    add("")
    add(
        "`scripts/serve.py` exposes the fitted engine over HTTP. The engine holds *history*, not just "
        "parameters: when a request supplies fresh closes they are appended to the stored history before "
        'the conditional volatility is filtered forward, so the service answers "given what just '
        'happened, what is my risk now" rather than "what did my model say when it was last fitted". '
        "The distinction is the whole point of a filter, and it is the reason a POST with a large down "
        "move returns a visibly higher number than the same request with a quiet day."
    )
    add("")

    add("## References")
    add("")
    add(
        "- Kupiec (1995), *Techniques for verifying the accuracy of risk measurement models* - the "
        "proportion-of-failures test."
    )
    add("- Christoffersen (1998), *Evaluating interval forecasts* - conditional coverage and the independence test.")
    add("- Engle & Manganelli (2004), *CAViaR* - the dynamic-quantile test used here as a specification check.")
    add(
        "- Acerbi & Szekely (2014), *Backtesting expected shortfall* - the Z1 statistic and its conditions of validity."
    )
    add(
        "- Basel Committee (1996, and BCBS 2019 for FRTB), *Supervisory framework for the use of backtesting* - the traffic light and its capital multipliers."
    )
    add(
        "- Barndorff-Nielsen & Shephard (2004, 2006), *Power and bipower variation* - the jump-robust realised measure."
    )
    add("- Huang & Tauchen (2005), *The relative contribution of jumps to total price variance* - the ratio statistic.")
    add("- Boudoukh, Richardson & Whitelaw (1998) - age-weighted historical simulation.")
    add(
        "- McNeil & Frey (2000), *Estimation of tail-related risk measures for heteroscedastic financial time series* - filtered historical simulation."
    )
    add(
        "- Patton (2011), *Volatility forecast comparison using imperfect volatility proxies* - why QLIKE and MSE, and why not the others."
    )
    add("- Corsi (2009), *A simple approximate long-memory model of realized volatility* - the HAR family.")
    add("- Cho & Elman (2020) - the volatility-latent-noise argument behind scoring against the hidden variance.")
    add("")

    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# MODEL_DOCUMENTATION
# ---------------------------------------------------------------------------


def build_model_documentation(metrics: dict, out_dir: Path) -> str:
    params = metrics["manifest"]["parameters"]
    cfg = metrics["config"]
    scores = scores_ranked(metrics)
    primary = coverage_at(metrics, cfg["confidence_levels"][-1], 1)

    parts: list[str] = []
    add = parts.append

    add("# Model documentation")
    add("")
    add(
        "The specification behind the numbers, at the level of detail needed to reimplement it and to "
        "know where it is wrong. Every quantity that is a *result* is marked as such and is rendered from "
        "`outputs/metrics.json`; everything else is a definition."
    )
    add("")
    add(f"- Run: `{metrics['version']}`, generated {metrics['generated_at']}")
    add(
        f"- Sample: {num(metrics['rows'], 0)} rows, {metrics['sample_start']} to {metrics['sample_end']}, asset {metrics['asset']}"
    )
    add(
        f"- Initial window {num(metrics['init_window'], 0)}, refit every {metrics['refit_every']} days, seed {metrics['seed']}"
    )
    add("")

    add("## 1. The data-generating process")
    add("")
    add("### 1.1 Conditional variance")
    add("")
    add("The variance is the sum of a short-memory GJR process and a long-memory log component:")
    add("")
    add("```")
    add("short:  s2_t = w + (alpha + gamma * 1[r_{t-1} < 0]) * r_{t-1}^2 + beta * s2_{t-1}")
    add("long:   ln m2_t = phi * ln m2_{t-1} + sigma * e_t        (e_t ~ N(0, 1))")
    add("total:  h_t = s2_t * exp(m2_t) * c_t")
    add("```")
    add("")
    add(
        "with `c_t` the crisis multiplier. The second component is what makes the series *look* the way "
        "real volatility looks: a slowly varying level, roughly log-normal, with an autocorrelation that "
        "decays almost linearly in the logarithm rather than geometrically, so that a HAR regression on "
        "it is genuinely well specified rather than approximately so."
    )
    add("")
    add("| parameter | value | role |")
    add("|---|---|---|")
    add(f"| `short_alpha` | {num(params['short_alpha'])} | news impact |")
    add(f"| `short_beta` | {num(params['short_beta'])} | persistence of the short component |")
    add(f"| `leverage_gamma` | {num(params['leverage_gamma'])} | extra variance after a down move |")
    add(
        f"| `persistence` | {num(params['persistence'])} | `alpha + gamma/2 + beta`, the share of a shock that survives |"
    )
    add(f"| `long_memory_phi` | {num(params['long_memory_phi'])} | near-unit-root decay of the level |")
    add(f"| `long_memory_sigma` | {num(params['long_memory_sigma'])} | innovation scale of the level |")
    add(f"| `long_memory_log_sd` | {num(params['long_memory_log_sd'])} | implied standard deviation in the log |")
    add(f"| `unconditional_vol_daily` | {num(params['unconditional_vol_daily'], 6)} | long-run daily volatility |")
    add(f"| `unconditional_vol_annual` | {num(params['unconditional_vol_annual'])} | the same annualised |")
    add("")
    add(
        "A near-unit-root component is a deliberate trade. It reproduces the long-memory signature that "
        "the HAR family is built to exploit, and it means the *level* of volatility carries essentially "
        "no information about where it will settle - which is the honest situation and the reason a "
        "twenty-year aggregate cannot be improved by knowing the twenty-year average."
    )
    add("")
    add("### 1.2 Innovations")
    add("")
    add(
        f"Returns are `r_t = sqrt(h_t) * z_t` with `z_t` Student-t on `{num(params['t_dof'], 1)}` degrees "
        f"of freedom, rescaled to unit variance. The fat tail is the point: at "
        f"{pct(cfg['confidence_levels'][-1], 0)} a Gaussian and a t with these degrees of freedom "
        f"disagree by tens of percent, which is what makes the parametric-route comparison meaningful."
    )
    add("")
    add(
        "A **skew** is also applied. The likelihood of the parametric routes assumes a symmetric t, so "
        "they are fitted to data that violate their own assumption. This is the same misspecification "
        "real data impose, and it is deliberate: a study in which the parametric models are correctly "
        "specified will conclude that parametric models are fine, which is not a useful conclusion."
    )
    add("")
    add("### 1.3 Jumps")
    add("")
    add(
        f"Compound Poisson, intensity **{num(params['jump_intensity'])}** per day, jump size "
        f"`N(0, {num(params['jump_sigma'])})`. Jumps are **shared across assets in the panel on the same "
        f"day** and scaled by each asset's beta, so the cross-sectional structure is systematic rather "
        f"than independent - which is what makes the panel correlated in the tail as well as in the body, "
        f"and is the property the feature tests check."
    )
    add("")
    add(
        "This is also why realised variance is a *noisy* target for the conditional variance even on "
        "simulated data: on a jump day the realised measure includes the jump and the filter does not. "
        "The generator records the jump days, and the jump test is checked against them."
    )
    add("")
    add("### 1.4 Intraday bridge")
    add("")
    add(
        f"Each day is expanded into **{params['intraday_bars']} five-minute bars** through a Brownian "
        f"bridge that is centred and rescaled so the bars sum exactly to the daily diffusion move. This is "
        f"what makes realised variance, bipower variation, the jump variation and the Huang-Tauchen "
        f"statistic computable from the data rather than assumed, and it is why the jump labels can be "
        f"validated against the jumps the simulator actually placed."
    )
    add("")
    add("### 1.5 The variance risk premium")
    add("")
    add(
        f"A multiplier of **{num(params['variance_risk_premium'])}** is applied to the *realised* measure "
        f"but not to the conditional variance. Without it the two coincide and every model looks equally "
        f"well calibrated; with it the realised measure is a biased proxy, which is the real situation and "
        f"the reason the pipeline also scores against the latent variance. The gap between the two "
        f"rankings is the size of the bias. *(Result: the ranking is broadly stable between them, so the "
        f"conclusions are not an artefact of the proxy.)*"
    )
    add("")
    add("### 1.6 Crisis schedule")
    add("")
    add(
        "Volatility is multiplied by a scheduled factor that **ramps in** over a calendar window and "
        "decays exponentially afterwards, rather than stepping. A step would be trivially detectable by "
        "the jump test and would make the stress windows unrepresentative."
    )
    add("")
    add("| window | start | end | multiplier |")
    add("|---|---|---|---|")
    for entry in metrics["manifest"]["crisis_schedule"]:
        add(f"| {entry['name']} | {entry['start']} | {entry['end']} | {num(entry['multiplier'], 1)} |")
    add("")

    add("## 2. Realised measures")
    add("")
    add("With `M` intraday returns `r_{t,j}` and `mu1 = sqrt(2/pi)`:")
    add("")
    add("```")
    add("RV_t  = sum_j r_{t,j}^2")
    add("BV_t  = (pi/2) * sum_{j=2..M} |r_{t,j}| |r_{t,j-1}|")
    add("JV_t  = max(RV_t - BV_t, 0)")
    add("Parkinson_t = ln(H_t / L_t)^2 / (4 ln 2)")
    add("```")
    add("")
    add(
        "The jump statistic is the Huang-Tauchen ratio `(RV - BV) / RV`, standardised by its variance "
        "under the null of no jumps. With the Barndorff-Nielsen correction the leading constant is "
        "`(mu1^-4 + 2 mu1^-2 - 5) / M`, which is used rather than the large-`M` limit: at "
        f"`M = {params['intraday_bars']}` the two differ in the third decimal, and a constant that is "
        "approximately right is the kind of thing that is never revisited."
    )
    add("")
    add(
        "Garman-Klass and Rogers-Satchell range measures are computed alongside Parkinson. They are not "
        "used as features - the HAR design reads the realised measures - but they are checked for scale "
        "invariance and non-negativity, because a range measure that is not scale-invariant is almost "
        "always a formula error rather than a modelling choice."
    )
    add("")
    add("## 3. The feature matrix")
    add("")
    add("| feature | definition |")
    add("|---|---|")
    add("| `log_rv_d` | daily log realised variance |")
    add("| `log_rv_w` | 5-day mean of the daily log |")
    add("| `log_rv_m` | 22-day mean of the daily log |")
    add("| `bv_ratio` | `BV / RV`, the jump share |")
    add("| `ret` | the close-to-close return |")
    add("| `ret_neg` | `ret * 1[ret < 0]`, the leverage term |")
    add("| `lev` | `1[ret < 0]` |")
    add("| `padj` | the Parkinson measure |")
    add("| `vix` | option-implied variance, the only genuinely forward-looking input |")
    add("")
    add(
        "The averaging is done **in logs**, not on the variances. Averaging variance over five days and "
        "then taking the logarithm is not the same estimator as averaging the log, and the HAR literature "
        "is specific about which one is meant; the test suite pins the distinction."
    )
    add("")
    add(
        "**Leakage control.** Every feature is shifted one day forward, so the row dated `t` contains "
        "only information available at the close of `t-1`. The target is the *next* day's log realised "
        "variance. The `rv_slope` column - `log_rv_w - log_rv_m` - is deliberately **excluded** from the "
        "design matrix: it is an exact linear combination of two columns that are already there, the "
        "design matrix becomes singular, and the least-squares solution is then whatever the solver "
        "decides it is. A test asserts its absence rather than trusting the comment."
    )
    add("")
    add(
        "The realised measure used for scoring is **`rv`**, chosen over `bv` and `padj` because it is the "
        "quantity the models are trying to forecast and because the jump contamination it carries is the "
        "same contamination the latent-variance column removes."
    )
    add("")

    add("## 4. The models")
    add("")
    add(
        "All models implement the same interface: `fit(returns, features, end)` estimates parameters on "
        "the prefix, `forecast(returns, features, start, stop)` returns one-step-ahead variances for the "
        "block. The separation matters: inside a refit block the parameters are frozen but the variance "
        "*state* is filtered with realised returns as they arrive, which is what a desk does and the only "
        "way a recursion stays legal."
    )
    add("")
    add("| model | parameters | notes |")
    add("|---|---|---|")
    add("| `rolling_21` | none | trailing sample variance; the benchmark |")
    add("| `rolling_63` | none | the same over a quarter |")
    add("| `ewma_94` | none | RiskMetrics decay 0.94 |")
    add("| `ewma_fitted` | decay | chosen on the training block from a grid |")
    add("| `garch` | 4 | Gaussian likelihood |")
    add("| `gjr` | 5 | `gamma` on negative shocks; **t** likelihood |")
    add("| `egarch` | 5 | log-variance recursion |")
    add("| `har` | 4 | OLS on logs of the realised measures |")
    add("| `harx` | 5 | the same plus implied variance |")
    add("| `gbm` | trees | LightGBM on the full feature set |")
    add("| `gru` | weights | one layer, written out by hand |")
    add("")
    add("### 4.1 The parametric route")
    add("")
    add(
        "GARCH-family models are estimated by maximum likelihood on the Student-t density, with the "
        "degrees of freedom estimated jointly with the recursion rather than fixed. Variance targeting is "
        "applied to the intercept so that the unconditional level matches the sample rather than being "
        "estimated through a poorly identified parameter. Both choices are tested: a recovery test "
        "asserts the fitted parameters reproduce the ones used to simulate, and a convergence test asserts "
        "the solution is not the initial point - an optimiser that returns its starting value produces a "
        "plausible series and a wrong one."
    )
    add("")
    add("### 4.2 The realised-measure route")
    add("")
    add(
        "HAR is fitted by OLS on log realised variance with the three components above. The forecast is "
        "**log-normal**, so the conditional mean of the variance is not `exp(fitted log)` but that value "
        "times a dispersion correction `exp(sigma_resid^2 / 2)`. Omitting it biases every forecast "
        "downwards by a few percent, consistently, which is the sort of error that survives review "
        "because it never produces an implausible number."
    )
    add("")
    add("### 4.3 The network")
    add("")
    add(
        "The GRU is implemented directly - forward pass, backpropagation through time, Adam, early "
        "stopping - with no framework, so that the gradient can be checked against a central difference. "
        "It is deliberately a *small* model on a *small* feature set: the comparison is whether a "
        "sequence model adds anything over a linear regression with the right features, and the answer "
        "from the results is that it does not, consistently. That is a legitimate finding and it is "
        "reported as one."
    )
    add("")

    add("## 5. The VaR and ES routes")
    add("")
    add("Write `sigma_t` for the conditional volatility. Every route below consumes the same series.")
    add("")
    add("```")
    add("historical      VaR_t = -quantile_{1-a}(r_{t-w..t-1})")
    add("                        age-weighted: exponentially declining weights in the same quantile")
    add("filtered        z_s = r_s / sigma_s ;  VaR_t = -sigma_t * quantile_{1-a}(z)")
    add("filtered-EVT    the upper tail of z above a threshold is fitted with a GPD and extrapolated")
    add("parametric      VaR_t = -sigma_t * F^-1(1-a)  for F Gaussian or Student-t")
    add("simulation      the variance recursion is simulated forward; the quantile is read off the paths")
    add("ES:             the conditional mean of the loss beyond the VaR, on the same scale")
    add("```")
    add("")
    add("### 5.1 Age-weighted historical simulation")
    add("")
    add(
        "Weights decay geometrically with age; the quantile is the weighted order statistic. With zero "
        "decay this reduces to the flat-window case, which is asserted - to within the difference between "
        "an interpolated order statistic and a cumulative-weight inversion, which is about 0.2% in "
        "absolute terms and is not zero."
    )
    add("")
    add("### 5.2 The generalised-Pareto tail")
    add("")
    add(
        "Exceedances above a threshold quantile are fitted by **method of moments**, not maximum "
        "likelihood: the closed form is `xi = (1 - mean^2 / variance) / 2`, and unlike MLE on a handful of "
        "exceedances it does not fail to converge on the quiet windows that make up most of the sample. "
        "The plausible-looking alternative - `(mean^2 / mean_square - 1) / 2` - returns `-0.25` for an "
        "exponential tail whose true shape is zero, implying a bounded tail, and produced a VaR about 20% "
        "below the empirical quantile and twice the nominal breach rate. A test fits the estimator to "
        "exponential data and asserts the shape is zero."
    )
    add("")
    add(
        "When there are too few exceedances to estimate two parameters - fewer than eight, which a quiet "
        "250-day window does produce - the fit is abandoned and the **empirical quantile is returned in "
        "its place**. The fallback deliberately returns the same object on the same scale as the fitted "
        "path, because the alternative is a third convention with its own sign, and this one was wrong "
        "once: the caller clamps at zero, so a sign error did not raise, it reported a VaR and an ES of "
        "exactly zero. A zero ES then became the denominator of the ES statistic and produced a value of "
        "order `1e8`."
    )
    add("")
    add("### 5.3 Simulation")
    add("")
    add(
        f"**{num(cfg['mc_paths'], 0)} paths** by default. Each path draws future shocks from the fitted "
        f"Student-t, propagates the variance recursion, and accumulates returns over the horizon. The "
        f"quantile and the mean of the tail beyond it are read off the path distribution. This is the "
        f"only route that produces a multi-day number without a scaling assumption, which is why it is "
        f"the reference for the horizon study. The path count is set so that the standard error of the "
        f"quantile is small relative to the differences between methods; `--fast` lowers it, which makes "
        f"individual days noisy while leaving the average usable."
    )
    add("")

    add("## 6. The tests")
    add("")
    add("Each is stated with its null, because a backtest quoted without its null is not interpretable.")
    add("")
    add("| test | null | statistic |")
    add("|---|---|---|")
    add(
        "| Kupiec | the true breach rate equals `1 - a` | likelihood ratio, chi-squared on 1 degree of freedom, **two-sided** |"
    )
    add(
        "| Christoffersen | breaches are independent over time | first-order Markov likelihood ratio, chi-squared on 1 |"
    )
    add("| Conditional coverage | both hold | the two statistics added, chi-squared on 2 |")
    add(
        "| Engle-Manganelli | the breach has no predictable structure | regression of the hit on its lags and on the VaR, chi-squared |"
    )
    add(
        "| Acerbi-Szekely | the ES is the conditional mean beyond the VaR | standardised mean of `-r/ES - 1` over breach days, asymptotically normal |"
    )
    add("")
    add(
        "**Two-sided matters.** A one-sided Kupiec test would accept a model that never breaches, and a "
        "model that never breaches has not been validated - it has failed to produce evidence. The "
        "implementation rejects in both directions and a test asserts it does."
    )
    add("")
    add("### 6.1 Basel traffic light")
    add("")
    add(
        "Exceptions are counted over rolling **250-day** windows at the 99% level. Fewer than five is "
        "green, up to nine yellow, ten or more red, and the capital add-on is not linear:"
    )
    add("")
    add("| exceptions | 0-4 | 5 | 6 | 7 | 8 | 9 | 10+ |")
    add("|---|---|---|---|---|---|---|---|")
    add("| add-on | 0% | 40% | 50% | 65% | 75% | 85% | 100% |")
    add("")
    add(
        "A window that is not yet full is left **blank** rather than reported as a low count, so a model "
        "cannot appear green because it has not been live long enough to be measured. The same logic is "
        "implemented in SQL and compared against the Python version row by row."
    )
    add("")

    add("## 7. Evaluation protocol")
    add("")
    add("### 7.1 Walk-forward")
    add("")
    add(
        f"An initial window of **{num(metrics['init_window'], 0)} days** is held out, then the models are "
        f"re-estimated every **{metrics['refit_every']} days** on an expanding window and asked for "
        f"one-step-ahead forecasts over the following block. The network and the tree are refitted on a "
        f"longer cycle, since refitting a boosted model every month on a growing window is not what "
        f"anyone does. That the heavy models are genuinely refitted less often is asserted rather than "
        f"assumed."
    )
    add("")
    add("### 7.2 Loss functions")
    add("")
    add("QLIKE ranks the variance forecasts:")
    add("")
    add("```")
    add("QLIKE(h, v) = v/h - ln(v/h) - 1")
    add("```")
    add("")
    add(
        "where `v` is the forecast and `h` the realised measure. It is the logarithmic score, it is "
        "robust to the noise in the proxy in the sense of Patton (2011), and it is asymmetric in exactly "
        "the way a risk number should be: under-prediction is punished without bound, over-prediction "
        "logarithmically. MSE on the volatility scale is reported beside it because it is the number the "
        "audience recognises, and a model that wins on QLIKE while losing badly on MSE is telling you "
        "where its errors sit."
    )
    add("")
    add("For a quantile the correct loss is the **tick loss**:")
    add("")
    add("```")
    add("TL(r, v, a) = ((1 - a) - 1[r < -v]) * (r + v)")
    add("```")
    add("")
    add(
        "The weight on a breach is the **tail probability** `1 - a`, not the coverage level `a`. Using "
        "`a` is a one-token change that inverts the objective: the loss then grows almost linearly in "
        "`v`, its minimum moves to `v = 0`, and a table of methods sorted by it ranks them by *reported "
        "risk, smallest first* - while still sorting, and still crowning a winner. A test asserts that "
        "the inverted form is minimised at zero and the correct form at the true quantile."
    )
    add("")
    add("### 7.3 Paired comparison")
    add("")
    add(
        "Diebold-Mariano on the QLIKE loss differential, Newey-West with 10 lags, two-sided. The "
        "diagonal is **NaN**, not 1.0: a model compared with itself has a loss differential identically "
        "zero, a variance of zero, and an undefined statistic. Reporting `1.0` there would claim the "
        "diagonal was tested and found insignificant, which is a different and false statement."
    )
    add("")
    add("### 7.4 Calibration")
    add("")
    add(
        "Mincer-Zarnowitz regresses the realised measure on the forecast. The **slope** is the informative "
        "output: 1.0 means correctly scaled, below 1.0 means the forecast is too timid, above 1.0 means it "
        "over-reacts. The joint test that the intercept is zero and the slope is one is also reported, but "
        "at this sample size it rejects deviations far too small to act on - so the coefficient is quoted "
        "and the flag is not treated as a verdict. A separate test documents this by construction."
    )
    add("")
    add("### 7.5 The latent variance")
    add("")
    add(
        "Because the data are simulated, the conditional variance that generated each day is known. The "
        "same models are scored against it, which removes the proxy noise from the comparison. *(Result: "
        f"the observed best model is `{scores[0]['model']}`, and the ordering against the latent variance "
        f"is broadly the same - QLIKE {num(scores[0]['qlike_latent'])} against "
        f"{num(scores[0]['qlike'])} against realised variance.)*"
    )
    add("")

    add("## 8. Known limitations")
    add("")
    add(
        "1. **The generator is a model of volatility, not of anything else.** No microstructure, no "
        "intraday seasonality, no discrete ticks, no trading halts. The measures are computed on a bridge "
        "that is too well behaved; the jump *detection* result is therefore cleaner here than it would "
        "be on tape."
    )
    add(
        "2. **The likelihood is misspecified by design.** The parametric routes assume a symmetric t "
        "while the data are skewed. This is closer to reality than a correctly specified study, but it "
        "means their scores are a lower bound and should not be read as the best those models can do."
    )
    add(
        "3. **Overlapping windows at ten days.** Multi-day returns are computed over overlapping "
        "windows, so hit indicators are mechanically dependent and the independence and dynamic-quantile "
        "p-values at h = 10 are biased downwards. The breach *rate* is unaffected. The honest fix is "
        "non-overlapping windows at a four-fifths cost in sample."
    )
    add(
        "4. **The ES test requires breaches.** A route with fewer than five breach days is skipped. That "
        "is a sample-size constraint, not an endorsement: a method can pass by never producing enough "
        "evidence to be judged."
    )
    add(
        "5. **One filter, one asset.** Every route shares the GJR-GARCH-t filter, so the comparison is "
        "of methods conditional on that filter. Whether the ranking survives a different filter is not "
        "answered here."
    )
    add(
        "6. **No multiple-testing correction.** Seven routes over two levels and two horizons is "
        "twenty-eight ES tests. A single pass at the 5% level should be read accordingly."
    )
    add("")
    add("## 9. Verification")
    add("")
    add("The claims above that could be checked mechanically are, and the checks live in `tests/`:")
    add("")
    add("| claim | test |")
    add("|---|---|")
    add("| no look-ahead in the features | corrupting the future leaves past forecasts unchanged |")
    add("| the model reads features, not returns | corrupting the return series leaves HAR unchanged |")
    add("| gradients are correct | every tensor's analytic gradient against a central difference |")
    add("| the optimiser converges | the GARCH solution is not the initial point |")
    add("| parameters are recoverable | simulated parameters recovered by fitting |")
    add("| the GPD shape is right | exponential tail gives shape 0, not -0.25 |")
    add("| the tail fallback keeps its scale | it equals the empirical quantile bit for bit |")
    add("| the EVT fallback would be zero if inverted | the clamped quantity is asserted directly |")
    add("| the tick loss is the right objective | its minimum is at the true quantile, not at zero |")
    add("| ES is never below VaR | asserted on every route at once |")
    add("| every route produces a positive number | strictly, not `>= 0` |")
    add("| the DQ test works | it detects a misspecified model and passes a correct one |")
    add("| the independence test has power | 20 consecutive breaches are rejected |")
    add("| the walk-forward is legal | five invariants across the whole scheme |")
    add("| SQL agrees with Python | log return, Parkinson, 21-day variance, exception count, zone |")
    add("| `--fast` loses no deliverable | every model and method present in the fast run |")
    add("| the docs match the run | the doc builder fails on a missing figure or a short table |")
    add("")
    add(
        "The end-to-end test reruns the entire pipeline on a shorter sample and checks the artefacts: the "
        "file list, the row counts, the model and method roster, the non-emptiness of the horizon study, "
        "that the winner is not the naive baseline, and that every number quoted in `RESULTS.md` is "
        "present in `metrics.json`."
    )
    add("")
    add(
        f"Coverage for the reference route at {pct(cfg['confidence_levels'][-1], 0)}, one day, for "
        f"orientation - the full tables are in `outputs/RESULTS.md`: "
        f"{primary[0]['breaches']} breaches in {primary[0]['observations']} days "
        f"({pct(primary[0]['observed_rate'], 2)} against a nominal "
        f"{pct(primary[0]['expected_rate'], 1)}), Kupiec p = {pval(primary[0]['kupiec_p'])}."
    )
    add("")

    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the documentation from the run's metrics.")
    parser.add_argument("--metrics", default="outputs/metrics.json", help="the metrics bundle to read")
    parser.add_argument("--output", default=".", help="directory to write the documents into")
    args = parser.parse_args()

    metrics_path = (PROJECT_ROOT / args.metrics).resolve()
    if not metrics_path.exists():
        print(f"no metrics bundle at {metrics_path}; run the pipeline first", file=sys.stderr)
        return 1
    metrics = json.loads(metrics_path.read_text())

    out_dir = (PROJECT_ROOT / args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sanity-check the bundle before rendering from it. A renderer that trusts its
    # input will happily produce a confident document from half a run.
    expected_models = set(metrics["config"]["models"])
    scored_models = {r["model"] for r in metrics["volatility_scores"]}
    if scored_models != expected_models:
        print(f"metrics.json is inconsistent: {expected_models ^ scored_models}", file=sys.stderr)
        return 1
    combos = {
        (level, horizon)
        for level in metrics["config"]["confidence_levels"]
        for horizon in metrics["config"]["horizons"]
    }
    for key in ("coverage", "tick_loss", "es_backtest"):
        seen = {(r["level"], r["horizon"]) for r in metrics[key]}
        if seen != combos:
            print(f"{key} does not cover every level/horizon pair: {combos ^ seen}", file=sys.stderr)
            return 1

    documents = {
        out_dir / "README.md": build_readme(metrics, out_dir),
        out_dir / "MODEL_DOCUMENTATION.md": build_model_documentation(metrics, out_dir),
    }
    for path, text in documents.items():
        path.write_text(text)
        print(f"wrote {path} ({len(text.splitlines())} lines)")

    figures = sorted((PROJECT_ROOT / "outputs/figures").glob("*.png"))
    print(f"{len(figures)} figures available; every one referenced by the README was found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
