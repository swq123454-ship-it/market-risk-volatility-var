"""Assemble the price panel the study runs on, in either data mode.

Two modes, one interface
------------------------
``synthetic`` (default) calls :mod:`marketrisk.generator` and returns a panel
whose *true* conditional variance is known. ``real`` downloads index closes
through :mod:`yfinance` and returns the same columns, minus the ones that only
exist because the data were simulated.

The two modes are not interchangeable, and the code says so rather than
pretending otherwise. Real mode cannot provide:

* **a true conditional variance**, so the "does the model track the latent
  process" table is unavailable and the study falls back to proxy-based losses;
* **intraday bars**, so realised variance has to be proxied by a daily range
  estimator and the RV/BV jump decomposition disappears. Every table records
  which realised measure it used.

What it does provide is a check on the synthetic conclusions, and the cross-asset
tables are run on whatever mode is active so that the question "does this survive
a change of market" has an answer either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as cfg
from . import features as feat
from .generator import simulate_market


@dataclass
class AssetPanel:
    """One instrument, in the layout every downstream module expects.

    Two frames, deliberately. ``raw`` is dated on the day a return was realised
    and holds prices, returns and the realised measures. ``features`` is dated on
    the day being *forecast* and holds only lagged information, plus the target
    and the return that is about to be predicted. Keeping them apart is what
    stops an unlagged column from leaking into a design matrix by accident: a
    model that wants a feature has to ask the feature frame, and the feature frame
    cannot contain the day it is predicting.
    """

    key: str
    label: str
    raw: pd.DataFrame
    features: pd.DataFrame
    #: True ex-ante conditional variance; ``None`` in real-data mode.
    true_variance: pd.Series | None = None
    vix: pd.Series | None = None
    #: False when jumps cannot be separated from the continuous part.
    jumps_available: bool = True
    #: Which realised measure was used as the volatility proxy.
    realised_measure: str = "rv"

    @property
    def returns(self) -> pd.Series:
        return self.raw["ret"]

    @property
    def realised(self) -> pd.Series:
        return self.raw["realised_variance"]

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.raw.index

    def __len__(self) -> int:
        return len(self.raw)


@dataclass
class MarketData:
    assets: dict[str, AssetPanel]
    mode: str
    seed: int
    manifest: dict[str, object] = field(default_factory=dict)

    @property
    def primary(self) -> AssetPanel:
        return self.assets[cfg.PRIMARY_ASSET]

    def keys(self) -> list[str]:
        return list(self.assets)


def _synthetic_panel(start: str, end: str, seed: int, assets: tuple[str, ...] | None) -> MarketData:
    simulated = simulate_market(start=start, end=end, seed=seed, assets=assets)
    panels: dict[str, AssetPanel] = {}
    for key, path in simulated.assets.items():
        raw = path.frame.copy()
        intraday = path.intraday
        raw["bipower"] = feat.bipower_variation(intraday)
        raw["realised_variance"] = path.realised_variance.to_numpy()
        raw["parkinson"] = feat.parkinson(raw["high"], raw["low"]).to_numpy()
        log_vix = np.log(simulated.vix.clip(lower=1e-6)) if key == cfg.PRIMARY_ASSET else None
        frame = feat.build_features(raw, log_vix=log_vix)
        panel = AssetPanel(
            key=key,
            label=path.label,
            raw=raw,
            features=frame,
            true_variance=path.true_variance,
            vix=simulated.vix,
            jumps_available=True,
            realised_measure="rv",
        )
        panels[key] = panel
    return MarketData(assets=panels, mode="synthetic", seed=seed, manifest=simulated.manifest())


def _download_closes(symbols: dict[str, str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted closes, with a clear failure rather than a silent empty frame."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("real-data mode needs yfinance; install it or run with --data-mode synthetic") from exc

    frames = []
    for key, symbol in symbols.items():
        data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
        if data is None or len(data) == 0:
            raise RuntimeError(f"yfinance returned no rows for {symbol} ({key})")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        close = data["Close"].rename(key)
        frames.append(close)
    panel = pd.concat(frames, axis=1).dropna(how="all")
    return panel


def _real_panel(start: str, end: str, seed: int) -> MarketData:
    from .features import garman_klass, parkinson

    symbols = {key: str(meta["symbol"]) for key, meta in cfg.ASSETS.items()}
    symbols["VIX"] = cfg.VIX_SYMBOL
    closes = _download_closes(symbols, start, end)
    vix = closes.pop("VIX")

    panels: dict[str, AssetPanel] = {}
    for key in cfg.ASSETS:
        if key not in closes:
            continue
        close = closes[key].dropna()
        raw = pd.DataFrame(index=close.index)
        raw["close"] = close
        raw["ret"] = np.log(close).diff()
        raw = raw.dropna()
        # Real mode has no daily OHLC from this feed, so the high/low/open/close
        # columns are set to the close. That makes the range estimators equal to
        # zero rather than wrong, and `realised_measure` records that the proxy in
        # use is the squared close-to-close return.
        for column in ("open", "high", "low"):
            raw[column] = raw["close"]
        raw["realised_variance"] = raw["ret"] ** 2
        raw["bipower"] = raw["ret"] ** 2
        raw["parkinson"] = parkinson(raw["high"], raw["low"]).to_numpy()
        log_vix = np.log(vix.reindex(raw.index).clip(lower=1e-6)) if key == cfg.PRIMARY_ASSET else None
        panels[key] = AssetPanel(
            key=key,
            label=str(cfg.ASSETS[key]["label"]),
            raw=raw,
            features=feat.build_features(raw, log_vix=log_vix),
            true_variance=None,
            vix=vix if key == cfg.PRIMARY_ASSET else None,
            jumps_available=False,
            realised_measure="squared_return",
        )
    if cfg.PRIMARY_ASSET not in panels:
        raise RuntimeError(f"real-data mode produced no panel for {cfg.PRIMARY_ASSET}")
    _ = garman_klass  # kept importable for callers that prefer it
    return MarketData(
        assets=panels,
        mode="real",
        seed=seed,
        manifest={"source": "yfinance", "start": start, "end": end, "symbols": symbols},
    )


def load_market(
    data_mode: str = "synthetic",
    start: str = cfg.DEFAULT_START,
    end: str = cfg.DEFAULT_END,
    seed: int = cfg.SEED,
    assets: tuple[str, ...] | None = None,
) -> MarketData:
    """Load the price panel in the requested mode."""
    if data_mode == "synthetic":
        return _synthetic_panel(start, end, seed, assets)
    if data_mode == "real":
        return _real_panel(start, end, seed)
    raise ValueError(f"unknown data mode {data_mode!r}; expected 'synthetic' or 'real'")


def usable_rows(frame: pd.DataFrame) -> pd.Series:
    """Rows with a complete predictor set and a target - the sample actually fitted on."""
    columns = [*feat.FEATURE_COLUMNS, feat.TARGET_COLUMN, "ret_next"]
    present = [c for c in columns if c in frame.columns]
    return frame.loc[:, present].notna().all(axis=1)


def log_variance_target(frame: pd.DataFrame) -> pd.Series:
    """``log RV_{t}`` dated on ``t`` - the regression target for HAR/GBM/GRU."""
    return frame[feat.TARGET_COLUMN]
