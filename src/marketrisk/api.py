"""A small risk-engine service over a fitted volatility filter.

The study's output is a comparison of methods; a desk's output is a single number
by 07:00 that someone has to act on. This module closes that gap - it wraps the
same fitted filter and the same VaR methods in an HTTP service that returns the
current risk numbers, so the machinery the study validates is the machinery that
would actually be used.

Two contracts are worth stating because they are the ones that bite in production.

**The engine holds history, not just parameters.** Historical simulation and
filtered historical simulation need the recent return distribution, so the
service carries a reference window. Its size is recorded in the bundle and
reported on every response, because a VaR produced from a two-week window is a
different number from one produced from a year, and a response that does not say
which it used is not auditable.

**Missing is not zero.** A JSON payload cannot carry a NaN, so a sparse feature row
arrives with the key absent rather than with a sentinel value. The service treats an
absent key as missing and refuses to score with it rather than silently
substituting zero - and it says so in the error, because the alternative is a
plausible-looking number computed from a defaulted input, which is the worst
possible failure for a risk system.

The service is deliberately read-only with respect to parameters: ``/risk`` uses
the stored fit, and ``/risk/conditional`` scores a *new* observation with those
same parameters. Refitting is the pipeline's job, not the endpoint's, because an
endpoint that refits on request is an endpoint that can be made to say anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import backtest, kernels, losses, var, volmodels
from . import config as cfg

EPS = 1e-12


@dataclass
class RiskEngine:
    """A fitted filter plus the history the empirical methods need."""

    returns: np.ndarray
    garch: volmodels.GARCH
    ewma: volmodels.EWMA
    rolling: volmodels.RollingVariance
    reference_window: int = cfg.HS_WINDOWS[0]
    version: str = ""
    asset: str = ""
    created: str = ""
    sample_end: str = ""
    observations: int = 0
    #: Per-method VaR history, used for the rolling coverage diagnostic.
    var_history: pd.DataFrame = field(default_factory=pd.DataFrame)

    # -- construction --------------------------------------------------------
    @classmethod
    def fit(
        cls,
        returns: pd.Series,
        features: pd.DataFrame,
        asset: str = "",
        reference_window: int = cfg.HS_WINDOWS[0],
        version: str = "",
    ) -> RiskEngine:
        r = returns.to_numpy(dtype=float)
        garch = volmodels.GARCH(kind="gjr")
        garch.fit(r, features, len(r))
        ewma = volmodels.EWMA(lam=0.94)
        ewma.fit(r, features, len(r))
        rolling = volmodels.RollingVariance(window=21)
        return cls(
            returns=r,
            garch=garch,
            ewma=ewma,
            rolling=rolling,
            reference_window=reference_window,
            version=version or f"gjr-t-seed{cfg.SEED}-{len(r)}obs",
            asset=asset,
            created=pd.Timestamp.utcnow().isoformat(),
            sample_end=str(returns.index[-1].date()),
            observations=int(len(r)),
        )

    # -- the numbers ---------------------------------------------------------
    def volatility_forecast(self, extra: np.ndarray | None = None) -> float:
        """Conditional volatility of the **next** return, given the stored history.

        The recursion is run over the whole history and then advanced one further
        step with the last realised return. That extra step is the difference
        between two different questions, and only one of them is what a desk asks.
        The kernel's final entry is the variance *of* the last observation - which
        is already known and therefore of no use as a risk number - so returning it
        makes an eight-percent down day raise tomorrow's risk by nothing at all:
        the shock moves the next recursion step, which the function never took.
        """
        history = self.returns if extra is None or extra.size == 0 else np.concatenate([self.returns, extra])
        if history.size == 0:
            raise ValueError("the engine holds no observations to filter")
        backcast = kernels.variance_backcast(history, cfg.GARCH_BACKCAST_WEIGHT)
        p = self.garch.params
        persistence = p["alpha"] + p["beta"] + 0.5 * p["gamma"]
        omega = (1.0 - persistence) * max(float(p["unconditional_variance"]), EPS)
        path = kernels.garch_variance_path(omega, p["alpha"], p["gamma"], p["beta"], history, backcast)

        last = float(history[-1])
        advanced = omega + p["alpha"] * last * last + p["beta"] * float(path[-1])
        if last < 0.0:
            advanced += p["gamma"] * last * last
        return float(np.sqrt(max(advanced, EPS)))

    def risk(self, level: float = cfg.PRIMARY_LEVEL, horizon: int = 1, extra: np.ndarray | None = None) -> dict:
        """Current VaR and ES by every method, with the diagnostics a desk needs.

        ``extra`` appends observations to the stored history - the scenario where
        the service is asked to re-price after a new close without refitting.
        """
        history = self.returns if extra is None or extra.size == 0 else np.concatenate([self.returns, extra])
        if history.size < 60:
            raise ValueError(f"need at least 60 observations, got {history.size}")
        sigma = self.volatility_forecast(extra)
        window = self.reference_window
        oldest = float(np.min(history[-window:]))

        methods: dict[str, dict[str, float]] = {}
        # Historical simulation on the reference window.
        sample = history[-window:]
        hs_var = max(-float(np.quantile(sample, 1.0 - level)) * np.sqrt(horizon), 0.0)
        tail = sample[sample <= -hs_var / np.sqrt(horizon)] if hs_var > 0 else sample[:0]
        methods["hs"] = {
            "var": hs_var,
            "es": max(-float(tail.mean()) * np.sqrt(horizon), 0.0) if tail.size else hs_var,
        }
        # Parametric, with the fitted degrees of freedom.
        dof = float(self.garch.params.get("dof", 5.0))
        if not np.isfinite(dof):
            dof = 200.0
        methods["parametric_normal"] = {
            "var": sigma * var.normal_quantile(level) * np.sqrt(horizon),
            "es": sigma * var.normal_es(level) * np.sqrt(horizon),
        }
        methods["parametric_t"] = {
            "var": sigma * var.student_t_quantile(level, dof) * np.sqrt(horizon),
            "es": sigma * var.student_t_es(level, dof) * np.sqrt(horizon),
        }
        # Filtered historical simulation: residual quantile, then rescale.
        residuals = history[-window:] / np.maximum(_filtered_volatility(history, self, window), EPS)
        z_q = float(np.quantile(residuals, 1.0 - level))
        z_tail = residuals[residuals <= z_q]
        methods["fhs"] = {
            "var": max(-z_q * sigma * np.sqrt(horizon), 0.0),
            "es": max(-float(z_tail.mean()) * sigma * np.sqrt(horizon), 0.0) if z_tail.size else max(-z_q * sigma, 0.0),
        }

        annualised = sigma * np.sqrt(cfg.TRADING_DAYS)
        recent = history[-min(cfg.BASEL_WINDOW, history.size) :]
        target = 1.0 - level
        return {
            "asset": self.asset,
            "level": level,
            "horizon": horizon,
            "version": self.version,
            "observations": int(history.size),
            "reference_window": int(window),
            "conditional_vol_daily": sigma,
            "conditional_vol_annual": annualised,
            "degrees_of_freedom": None if dof > 200 else dof,
            "methods": methods,
            "diagnostics": {
                "window_worst_return": oldest,
                "window_realised_vol_annual": float(recent.std() * np.sqrt(cfg.TRADING_DAYS)),
                "expected_breaches_per_year": target * cfg.TRADING_DAYS,
                "converged": bool(self.garch.params.get("converged", False)),
            },
        }

    def info(self) -> dict:
        p = self.garch.params
        return {
            "version": self.version,
            "asset": self.asset,
            "created": self.created,
            "sample_end": self.sample_end,
            "observations": self.observations,
            "reference_window": self.reference_window,
            "parameters": {
                key: (None if isinstance(p.get(key), float) and not np.isfinite(p[key]) else p.get(key))
                for key in ("alpha", "gamma", "beta", "dof", "unconditional_variance")
            },
            "persistence": float(p["alpha"] + p["beta"] + 0.5 * p["gamma"]),
            "half_life_days": float(np.log(0.5) / np.log(max(p["alpha"] + p["beta"] + 0.5 * p["gamma"], 1e-6))),
            "converged": bool(p.get("converged", False)),
        }

    def coverage_history(self, level: float = cfg.PRIMARY_LEVEL, window: int = cfg.BASEL_WINDOW) -> dict:
        """Rolling breach count for the stored history, i.e. the traffic-light line."""
        if self.var_history.empty or str(level) not in self.var_history.columns:
            return {"available": False}
        hits = (self.returns < -self.var_history[str(level)].to_numpy(dtype=float)).astype(float)
        counted = pd.Series(hits).rolling(window, min_periods=window).sum().dropna()
        if counted.empty:
            return {"available": False}
        latest = int(counted.iloc[-1])
        return {
            "available": True,
            "window": window,
            "latest_exceptions": latest,
            "zone": backtest.basel_zone(latest),
            "plus_factor": backtest.basel_plus_factor(latest),
            "worst": int(counted.max()),
        }


def _filtered_volatility(history: np.ndarray, engine: RiskEngine, window: int) -> np.ndarray:
    """EWMA-filtered volatility over the tail of the history, for FHS residuals."""
    backcast = kernels.variance_backcast(history, cfg.GARCH_BACKCAST_WEIGHT)
    path = kernels.ewma_variance_path(engine.ewma.chosen, history, backcast, 0.0)
    return np.sqrt(np.maximum(path, EPS))[-window:]


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def save_bundle(engine: RiskEngine, path) -> None:
    import joblib

    joblib.dump({"engine": engine, "version": engine.version}, path)


def load_bundle(path) -> RiskEngine:
    import joblib

    bundle = joblib.load(path)
    if isinstance(bundle, dict) and "engine" in bundle:
        return bundle["engine"]
    return bundle


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------
def _risk_request_model():
    """The POST body schema, built on demand.

    Not defined at import time on purpose: importing this module must not require
    pydantic to be installed, because the study's core needs no web framework and
    only the serving path does.
    """
    from pydantic import BaseModel, Field

    class RiskRequest(BaseModel):
        """A fresh slice of returns to append to the stored history."""

        returns: list[float] = Field(
            default_factory=list,
            description=(
                "Log returns, most recent last. Appended to the reference window for this "
                "call only; parameters are not refitted. A JSON payload cannot carry NaN, so "
                "omit a day rather than sending a sentinel - the engine treats a missing "
                "observation as missing and will say so rather than assuming zero."
            ),
        )
        level: float = Field(default=cfg.PRIMARY_LEVEL, ge=0.5, le=0.9999)
        horizon: int = Field(default=1, ge=1, le=250)

    return RiskRequest


def create_app(engine: RiskEngine):
    """Build the FastAPI application around a fitted engine.

    Imported lazily so that the study's core does not depend on a web framework
    being installed.
    """
    from fastapi import FastAPI, HTTPException

    app = FastAPI(
        title="Market risk engine",
        description="VaR and Expected Shortfall over a fitted GJR-GARCH-t filter.",
        version=engine.version,
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "observations": engine.observations, "version": engine.version}

    @app.get("/model/info")
    def model_info() -> dict:
        return engine.info()

    @app.get("/risk")
    def current_risk(level: float = cfg.PRIMARY_LEVEL, horizon: int = 1) -> dict:
        if not 0.5 < level < 1.0:
            raise HTTPException(status_code=422, detail="level must lie strictly between 0.5 and 1.0")
        return engine.risk(level=level, horizon=horizon)

    def conditional_risk(request) -> dict:
        extra = np.asarray(request.returns, dtype=float)
        if extra.size and not np.isfinite(extra).all():
            raise HTTPException(
                status_code=422,
                detail="returns must be finite; omit a missing observation instead of sending a sentinel",
            )
        return engine.risk(level=request.level, horizon=request.horizon, extra=extra)

    # The schema is attached to the signature as a *class object*, and the route is
    # registered explicitly rather than with the decorator. ``from __future__ import
    # annotations`` turns every annotation in this file into a string, and FastAPI
    # resolves strings against the module globals - where a function-local class does
    # not exist. Resolution then fails quietly and FastAPI falls back to treating the
    # body as a required *query* parameter, so every well-formed POST comes back 422
    # with ``loc: ["query", "request"]``. Putting the class itself into
    # ``__annotations__`` bypasses the string form entirely.
    conditional_risk.__annotations__["request"] = _risk_request_model()
    app.add_api_route("/risk/conditional", conditional_risk, methods=["POST"])

    @app.get("/coverage")
    def coverage(level: float = cfg.PRIMARY_LEVEL) -> dict:
        return engine.coverage_history(level=level)

    return app


def risk_from_window(
    returns: np.ndarray, level: float, horizon: int, window: int, dof: float
) -> dict[str, dict[str, float]]:
    """Method-level VaR/ES from a raw return window, without a fitted bundle.

    Kept separate from :class:`RiskEngine` so that the arithmetic can be tested
    without constructing a filter, and so that a caller who only wants
    historical-simulation numbers is not forced to fit a GARCH model first.
    """
    r = np.asarray(returns, dtype=float)
    sample = r[-window:]
    sigma = float(np.std(sample[-min(21, sample.size) :]))
    out: dict[str, dict[str, float]] = {}
    hs = var.historical_var(
        r[-window * 2 :] if r.size >= window * 2 else r, level, min(window, max(r.size - 1, 2)), horizon
    )
    out["hs"] = {"var": float(hs[0][-1]), "es": float(hs[1][-1])}
    out["parametric_normal"] = {
        "var": sigma * var.normal_quantile(level) * np.sqrt(horizon),
        "es": sigma * var.normal_es(level) * np.sqrt(horizon),
    }
    out["parametric_t"] = {
        "var": sigma * var.student_t_quantile(level, dof) * np.sqrt(horizon),
        "es": sigma * var.student_t_es(level, dof) * np.sqrt(horizon),
    }
    z = sample / max(sigma, EPS)
    z_q = float(np.quantile(z, 1.0 - level))
    out["fhs"] = {
        "var": max(-z_q * sigma * np.sqrt(horizon), 0.0),
        "es": max(-float(z[z <= z_q].mean()) * sigma * np.sqrt(horizon), 0.0),
    }
    return out


def tick_loss_diagnostics(returns: np.ndarray, var_path: np.ndarray, level: float) -> dict[str, float]:
    """A one-line health check on a VaR path: mean tick loss and breach rate."""
    r = np.asarray(returns, dtype=float)
    v = np.asarray(var_path, dtype=float)
    mask = np.isfinite(r) & np.isfinite(v)
    if mask.sum() < 10:
        return {"tick_loss_bp": float("nan"), "breach_rate": float("nan")}
    tl = losses.tick_loss_scaled(r[mask], v[mask], level)
    return {
        "tick_loss_bp": float(np.nanmean(tl)),
        "breach_rate": float(np.mean(r[mask] < -v[mask])),
    }
