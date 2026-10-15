#!/usr/bin/env python
"""Entry point: build the market, fit every model, backtest every VaR route, report it.

Usage
-----
    python scripts/run_pipeline.py                        # synthetic market (default)
    python scripts/run_pipeline.py --data-mode real       # download via yfinance
    python scripts/run_pipeline.py --fast                 # reduced budget, same deliverables
    python scripts/run_pipeline.py --no-figures           # tables and RESULTS.md only
    python scripts/run_pipeline.py --end 2019-12-31       # shorter sample

The run is deterministic for a fixed ``--seed``. ``--fast`` lowers the Monte-Carlo
path count and the network's training budget; it does not remove a method, so the
fast run still produces every figure and every table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketrisk import config as cfg
from marketrisk.report import DEFAULT_OUTPUT, run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the market risk pipeline.")
    parser.add_argument(
        "--data-mode",
        choices=("synthetic", "real"),
        default="synthetic",
        help="synthetic (reproducible, no download) or real (yfinance download)",
    )
    parser.add_argument("--start", default=cfg.DEFAULT_START, help="first date of the sample")
    parser.add_argument("--end", default=cfg.DEFAULT_END, help="last date of the sample")
    parser.add_argument("--asset", default=cfg.PRIMARY_ASSET, choices=sorted(cfg.ASSETS), help="asset to study")
    parser.add_argument("--seed", type=int, default=cfg.SEED, help="random seed")
    parser.add_argument("--refit-every", type=int, default=cfg.REFIT_EVERY, help="trading days between refits")
    parser.add_argument("--mc-paths", type=int, default=cfg.MC_PATHS, help="Monte-Carlo path count")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="reduced path count and training budget; every deliverable is still produced",
    )
    parser.add_argument("--no-figures", action="store_true", help="skip figure rendering")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="output directory")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    arguments = parser.parse_args()

    pipeline = cfg.PipelineConfig(
        start=arguments.start,
        end=arguments.end,
        seed=arguments.seed,
        data_mode=arguments.data_mode,
        asset=arguments.asset,
        refit_every=arguments.refit_every,
        mc_paths=arguments.mc_paths,
        fast=arguments.fast,
    )

    run_pipeline(
        pipeline,
        output_dir=Path(arguments.output),
        verbose=not arguments.quiet,
        render_figures=not arguments.no_figures,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
