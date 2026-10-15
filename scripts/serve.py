#!/usr/bin/env python
"""Serve the fitted risk engine over HTTP.

    python scripts/serve.py                      # fit on the synthetic market, then serve
    python scripts/serve.py --bundle outputs/risk_engine.joblib
    python scripts/serve.py --port 9000

The engine holds history, not just parameters. When a request supplies fresh
closes they are appended to the stored history before the conditional volatility
is filtered forward, so the service answers "given what just happened, what is my
risk now" rather than "what did my model say last time it was fitted".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketrisk import api
from marketrisk import config as cfg
from marketrisk import data as data_module


def build_engine(data_mode: str, seed: int) -> api.RiskEngine:
    market = data_module.load_market(data_mode=data_mode, start=cfg.DEFAULT_START, end=cfg.DEFAULT_END, seed=seed)
    panel = market.primary
    return api.RiskEngine.fit(panel.returns, panel.features, asset=panel.key)


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the market risk engine.")
    parser.add_argument("--bundle", default="", help="path to a saved engine bundle")
    parser.add_argument("--save-bundle", default="", help="write the freshly fitted engine here")
    parser.add_argument("--data-mode", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    arguments = parser.parse_args()

    if arguments.bundle:
        engine = api.load_bundle(arguments.bundle)
        print(f"loaded engine {engine.version} ({engine.observations:,} observations)")
    else:
        engine = build_engine(arguments.data_mode, arguments.seed)
        print(f"fitted engine {engine.version} on {engine.observations:,} observations")
        if arguments.save_bundle:
            api.save_bundle(engine, arguments.save_bundle)
            print(f"saved bundle to {arguments.save_bundle}")

    app = api.create_app(engine)

    import uvicorn

    uvicorn.run(app, host=arguments.host, port=arguments.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
