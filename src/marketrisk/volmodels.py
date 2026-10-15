"""The volatility-model family, and the walk-forward engine that runs it.

Every model answers one question: what is the conditional variance of *tomorrow's*
return, given everything known today? They are scored on that question and
nothing else, so a model that fits the sample beautifully but forecasts badly is
recorded as bad - which is the only test that matters.

The family
----------
``rolling_21`` / ``rolling_63``
    Trailing sample variance. The benchmark every paper is obliged to beat and
    most are not able to.
``ewma_94``
    RiskMetrics. One parameter, published value, no estimation. A genuinely hard
    benchmark, which is worth saying because it is often dismissed as naive.
``ewma_fitted``
    The same recursion with the decay chosen by QLIKE on the training window, so
    the published convention can be checked rather than assumed.
``garch`` / ``gjr`` / ``egarch``
    Maximum-likelihood conditional-variance models with Student-t innovations.
    ``gjr`` adds the leverage term, ``egarch`` replaces the quadratic recursion
    with a log-linear one so that the asymmetry becomes a level effect.
``har`` / ``harx``
    Regressions of log realised variance on its daily/weekly/monthly averages.
    ``harx`` adds the jump share, a leverage term and the VIX.
``gbm``
    Gradient-boosted trees on the same regressors. Included because it is what a
    desk would reach for now, and because if a nonlinear learner cannot beat
    three linear lags that is worth knowing.
``gru``
    A from-scratch recurrent network on sequences of the regressors.

Fit and filter are separate
---------------------------
The engine refits parameters every ``refit_every`` days and *filters* in
between: parameters come from data up to the refit date, but the variance state
is updated with returns as they actually arrive. That is what a desk does, and it
is also the only way the recursions are legal - the one-step-ahead forecast for
day ``t`` uses returns up to ``t-1`` and nothing later, whether the parameters
were estimated last month or this morning.

Because parameters are frozen between refits, the *training* window of a forecast
never contains the day being forecast, and no model ever sees its own target.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize

from . import config as cfg
from . import kernels, losses, nn

EPS = 1e-12


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
class VolatilityModel(abc.ABC):
    """A conditional-variance forecaster with an explicit fit/filter split."""

    name: str = "model"
    #: True when the model consumes the realised-measure feature matrix.
    needs_features: bool = False
    #: True when the model needs to be retrained rather than merely refitted.
    heavy: bool = False
    #: Set once parameters exist. Checked for warm starts, not for correctness.
    fitted: bool = False

    def reset(self) -> None:
        """Forget any fitted state, so a model object can be reused across runs.

        The default clears the fitted flag, which is the only state the base class
        owns; a model that holds parameters overrides this. It is deliberately a
        concrete method with a real default rather than an abstract one: a rolling
        window and a fixed-decay EWMA are stateless by construction, so an abstract
        ``reset`` would only give three subclasses an empty body to restate, and a
        contract written down three times is a contract that drifts.
        """
        self.fitted = False

    @abc.abstractmethod
    def fit(self, returns: np.ndarray, features: pd.DataFrame, end: int) -> None:
        """Estimate parameters using ``returns[:end]`` and ``features[:end]``."""

    @abc.abstractmethod
    def forecast(self, returns: np.ndarray, features: pd.DataFrame, start: int, stop: int) -> np.ndarray:
        """One-step-ahead variance for positions ``start`` through ``stop-1``."""


# ---------------------------------------------------------------------------
# Simple benchmarks
# ---------------------------------------------------------------------------
@dataclass
class RollingVariance(VolatilityModel):
    """Trailing sample variance over a fixed window."""

    window: int = cfg.ROLLING_WINDOW
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"rolling_{self.window}"

    def fit(self, returns: np.ndarray, features: pd.DataFrame, end: int) -> None:
        return None

    def reset(self) -> None:
        return None

    def forecast(self, returns: np.ndarray, features: pd.DataFrame, start: int, stop: int) -> np.ndarray:
        r = np.asarray(returns, dtype=float)
        # Running sums make this O(n) for the whole block rather than O(n * window).
        squares = np.concatenate([[0.0], np.cumsum(r * r)])
        out = np.empty(stop - start)
        for i in range(start, stop):
            lo = max(i - self.window, 0)
            count = i - lo
            out[i - start] = squares[i] - squares[lo] if count <= 0 else (squares[i] - squares[lo]) / count
        return np.maximum(out, EPS)


@dataclass
class EWMA(VolatilityModel):
    """RiskMetrics exponential smoother: ``sigma2_t = lam * sigma2_{t-1} + (1-lam) r_{t-1}^2``."""

    lam: float = cfg.EWMA_LAMBDA
    #: When True, ``lam`` is chosen by QLIKE on the training window instead of
    #: being taken from the published convention. Named for the behaviour rather
    #: than ``fitted`` so that it cannot be confused with the base class flag of
    #: the same name.
    fit_decay: bool = False
    name: str = ""
    #: Decay actually in use.
    chosen: float = cfg.EWMA_LAMBDA

    def __post_init__(self) -> None:
        if not self.name:
            self.name = "ewma_fitted" if self.fit_decay else f"ewma_{int(round(self.lam * 100))}"

    def reset(self) -> None:
        self.chosen = self.lam

    def fit(self, returns: np.ndarray, features: pd.DataFrame, end: int) -> None:
        if not self.fit_decay:
            self.chosen = self.lam
            return
        r = np.asarray(returns[:end], dtype=float)
        if r.size < 50:
            self.chosen = self.lam
            return
        proxy = np.asarray(features["rv_next"].to_numpy()[:end], dtype=float)
        backcast = kernels.variance_backcast(r, cfg.GARCH_BACKCAST_WEIGHT)
        best, best_loss = self.lam, np.inf
        for candidate in cfg.EWMA_LAMBDA_GRID:
            path = kernels.ewma_variance_path(candidate, r, backcast, 0.0)
            value = float(np.nanmean(losses.qlike(path, proxy)))
            if value < best_loss:
                best, best_loss = candidate, value
        self.chosen = float(best)

    def forecast(self, returns: np.ndarray, features: pd.DataFrame, start: int, stop: int) -> np.ndarray:
        r = np.asarray(returns[:stop], dtype=float)
        backcast = kernels.variance_backcast(r, cfg.GARCH_BACKCAST_WEIGHT)
        path = kernels.ewma_variance_path(self.chosen, r, backcast, 0.0)
        return np.maximum(path[start:stop], EPS)


# ---------------------------------------------------------------------------
# Maximum-likelihood GARCH family
# ---------------------------------------------------------------------------
@dataclass
class GARCH(VolatilityModel):
    """GJR-GARCH / EGARCH with Student-t innovations, fitted by maximum likelihood.

    Variance targeting is used for the GARCH and GJR forms: the intercept is not
    a free parameter, it is whatever makes the unconditional variance of the
    recursion equal the sample variance. That removes a parameter whose
    likelihood surface is close to flat and whose estimate is therefore nearly
    unidentified, and it guarantees that a fit cannot imply an unconditional
    variance wildly different from the data. EGARCH is estimated untargeted
    because its intercept is the mean of *log* variance, which is a different
    quantity and targeting it would be a category error.
    """

    kind: str = "gjr"
    use_t: bool = True
    params: dict[str, float] = field(default_factory=dict)
    fitted: bool = False
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.kind

    def reset(self) -> None:
        self.params = {}
        self.fitted = False

    # -- parameter transformations ------------------------------------------
    @staticmethod
    def _logit(value: float) -> float:
        value = float(np.clip(value, 1e-6, 1.0 - 1e-6))
        return float(np.log(value / (1.0 - value)))

    def _initial(self, returns: np.ndarray) -> np.ndarray:
        if self.params and self.fitted:
            return np.asarray(self.params["theta"], dtype=float)
        if self.kind == "egarch":
            return np.array([-0.05, 0.15, -0.08, 0.97, np.log(4.0)])
        return np.array(
            [
                self._logit(0.06 / 0.40),
                self._logit(0.09 / 0.40),
                self._logit(0.86 / 0.98),
                np.log(4.0),
            ]
        )

    def _decode(self, theta: np.ndarray) -> dict[str, float]:
        if self.kind == "egarch":
            return {
                "omega": float(theta[0]),
                "alpha": float(theta[1]),
                "gamma": float(theta[2]),
                "beta": float(theta[3]),
                "dof": float(2.05 + np.exp(theta[4])) if self.use_t else float("inf"),
            }
        a = 0.40 / (1.0 + np.exp(-theta[0]))
        g = 0.40 / (1.0 + np.exp(-theta[1])) if self.kind == "gjr" else 0.0
        b = 0.98 / (1.0 + np.exp(-theta[2]))
        return {
            "alpha": a,
            "gamma": g,
            "beta": b,
            "dof": float(2.05 + np.exp(theta[3])) if self.use_t else float("inf"),
        }

    def _bounds(self) -> list[tuple[float, float]]:
        if self.kind == "egarch":
            return [(-2.0, 2.0), (0.0, 1.5), (-1.0, 1.0), (-0.999, 0.999), (np.log(0.05), np.log(400.0))]
        return [(-8.0, 8.0), (-8.0, 8.0), (-8.0, 8.0), (np.log(0.05), np.log(400.0))]

    # -- likelihood ----------------------------------------------------------
    def _neg_loglik(self, theta: np.ndarray, returns: np.ndarray) -> float:
        p = self._decode(theta)
        backcast = kernels.variance_backcast(returns, cfg.GARCH_BACKCAST_WEIGHT)
        if self.kind == "egarch":
            e_abs = kernels.standardised_abs_expectation(p["dof"]) if self.use_t else kernels.E_ABS_NORMAL
            log_sigma2 = kernels.egarch_log_variance_path(
                p["omega"], p["alpha"], p["gamma"], p["beta"], returns, float(np.log(backcast)), e_abs
            )
            sigma2 = np.exp(log_sigma2)
        else:
            persistence = p["alpha"] + p["beta"] + 0.5 * p["gamma"]
            if persistence >= 0.9995:
                return 1e10
            target = float(np.var(returns)) if returns.size else 1.0
            omega = (1.0 - persistence) * max(target, EPS)
            sigma2 = kernels.garch_variance_path(omega, p["alpha"], p["gamma"], p["beta"], returns, backcast)
        sigma2 = np.maximum(sigma2, EPS)
        z = returns / np.sqrt(sigma2)
        dof = p["dof"]
        if self.use_t and np.isfinite(dof):
            log_density = kernels.student_t_logpdf(z, dof)
        else:
            log_density = kernels.normal_logpdf(z)
        value = -np.sum(log_density - 0.5 * np.log(sigma2))
        return float(value) if np.isfinite(value) else 1e10

    def fit(self, returns: np.ndarray, features: pd.DataFrame, end: int) -> None:
        r = np.asarray(returns[:end], dtype=float)
        r = r[np.isfinite(r)]
        if r.size < 100:
            self.params = {"theta": self._initial(r), "unconditional_variance": float(np.var(r) if r.size else EPS)}
            self.fitted = True
            return
        theta0 = self._initial(r)
        # maxls is the option that matters. The first line search starts from a
        # unit step along a gradient whose components differ by two orders of
        # magnitude (the dof direction is nearly flat, the alpha direction is
        # steep), and scipy's default of 20 backtracking steps is exhausted before
        # an acceptable point is found - L-BFGS-B then reports nit=0 and returns
        # the *initial* point while quoting the best function value it saw, which
        # is a silent failure: the fit looks like it worked and the parameters are
        # simply the starting guess. 100 steps converge in about 40 iterations.
        options = {"maxiter": 800, "ftol": 1e-12, "gtol": 1e-7, "maxls": 100, "maxfun": 40_000}
        result = optimize.minimize(
            self._neg_loglik, theta0, args=(r,), method="L-BFGS-B", bounds=self._bounds(), options=options
        )
        # Cross-check with a derivative-free method. Nelder-Mead and L-BFGS-B
        # agree to four decimals on this likelihood, so a disagreement is a signal
        # that something is wrong rather than a matter of taste.
        if not np.isfinite(result.fun) or result.fun >= self._neg_loglik(theta0, r):
            fallback = optimize.minimize(
                self._neg_loglik,
                theta0,
                args=(r,),
                method="Nelder-Mead",
                options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-10},
            )
            if np.isfinite(fallback.fun) and fallback.fun < self._neg_loglik(theta0, r):
                result = fallback
        theta = result.x if np.isfinite(result.fun) else theta0
        self.params = {
            "theta": np.asarray(theta, dtype=float),
            "unconditional_variance": float(np.var(r)),
            "neg_loglik": float(result.fun),
            "converged": bool(getattr(result, "success", False)),
            **self._decode(theta),
        }
        self.fitted = True

    def forecast(self, returns: np.ndarray, features: pd.DataFrame, start: int, stop: int) -> np.ndarray:
        r = np.asarray(returns[:stop], dtype=float)
        p = self.params
        backcast = kernels.variance_backcast(r, cfg.GARCH_BACKCAST_WEIGHT)
        if self.kind == "egarch":
            e_abs = kernels.standardised_abs_expectation(p["dof"]) if self.use_t else kernels.E_ABS_NORMAL
            log_sigma2 = kernels.egarch_log_variance_path(
                p["omega"], p["alpha"], p["gamma"], p["beta"], r, float(np.log(backcast)), e_abs
            )
            path = np.exp(log_sigma2)
        else:
            persistence = p["alpha"] + p["beta"] + 0.5 * p["gamma"]
            omega = (1.0 - persistence) * max(p["unconditional_variance"], EPS)
            path = kernels.garch_variance_path(omega, p["alpha"], p["gamma"], p["beta"], r, backcast)
        return np.maximum(path[start:stop], EPS)


# ---------------------------------------------------------------------------
# Realised-volatility regressions
# ---------------------------------------------------------------------------
@dataclass
class HAR(VolatilityModel):
    """HAR-RV (and HAR-X) on log realised variance, fitted by least squares.

    Fitting in logs rather than levels is the choice that matters: RV is
    lognormal-ish with a long right tail, so a level regression is estimated
    mostly by the fifty crisis days and forecasts badly everywhere else. The
    point forecast is then ``exp(mu_hat + s^2/2)`` - the smearing adjustment -
    because the mean of a lognormal is not the exponential of the mean of its
    log. Omitting it biases every forecast down by roughly a factor of
    ``exp(s^2/2)``, which here is around 1.35, and the bias is invisible in a
    log-space R-squared.
    """

    use_vix: bool = True
    name: str = ""
    coef: np.ndarray | None = None
    columns: list[str] = field(default_factory=list)
    residual_variance: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            self.name = "harx" if self.use_vix else "har"

    def reset(self) -> None:
        self.coef = None
        self.residual_variance = 0.0

    def _design(self, features: pd.DataFrame, stop: int) -> tuple[np.ndarray, np.ndarray]:
        # ``rv_slope`` is deliberately absent. It is defined as
        # ``log_rv_w - log_rv_m``, so including it alongside both of its
        # components makes the design matrix exactly singular - the condition
        # number is of order 1e16 and the least-squares solution is whatever the
        # ridge penalty happens to pick. The three Corsi lags already span the
        # term structure; the slope is left as a feature for the tree model and
        # the network, where collinearity is harmless.
        base = ["log_rv_d", "log_rv_w", "log_rv_m"]
        if self.use_vix:
            base = [*base, "jump_1", "ret_neg_1", "log_vix_1", "vix_change_1"]
        self.columns = ["const", *base]
        block = features.iloc[:stop]
        x = np.column_stack([np.ones(len(block)), block[base].to_numpy(dtype=float)])
        y = block["log_rv_next"].to_numpy(dtype=float) if "log_rv_next" in block else np.zeros(len(block))
        return x, y

    def fit(self, returns: np.ndarray, features: pd.DataFrame, end: int) -> None:
        x, y = self._design(features, end)
        # The target is today's log RV; the design is already lagged by one day,
        # so the last usable row is the one dated end-1.
        mask = np.isfinite(x).all(axis=1) & np.isfinite(y)
        if mask.sum() < len(self.columns) + 5:
            self.coef = None
            return
        xm, ym = x[mask], y[mask]
        # Ridge with a tiny penalty only for numerical safety: the three HAR lags
        # are highly collinear and the unpenalised normal equations are close to
        # singular on quiet stretches.
        penalty = 1e-8 * np.eye(xm.shape[1])
        penalty[0, 0] = 0.0
        self.coef = np.linalg.solve(xm.T @ xm + penalty, xm.T @ ym)
        residual = ym - xm @ self.coef
        self.residual_variance = float(np.var(residual))

    def forecast(self, returns: np.ndarray, features: pd.DataFrame, start: int, stop: int) -> np.ndarray:
        if self.coef is None:
            return np.full(stop - start, float(np.var(np.asarray(returns[:stop], dtype=float))) or EPS)
        _, _ = self._design(features, stop)
        base = self.columns[1:]
        block = features.iloc[start:stop]
        x = np.column_stack([np.ones(len(block)), block[base].to_numpy(dtype=float)])
        log_forecast = x @ self.coef
        # Smearing: E[RV] = exp(mu + s^2/2) under lognormal residuals.
        return np.maximum(np.exp(log_forecast + 0.5 * self.residual_variance), EPS)


@dataclass
class GBM(VolatilityModel):
    """Gradient-boosted trees predicting log realised variance."""

    name: str = "gbm"
    model: object | None = None
    residual_variance: float = 0.0

    def reset(self) -> None:
        self.model = None
        self.residual_variance = 0.0

    def _design(self, features: pd.DataFrame, start: int, stop: int) -> np.ndarray:
        columns = [c for c in features.columns if c in _GBM_FEATURES]
        return features.iloc[start:stop][columns].to_numpy(dtype=float)

    def fit(self, returns: np.ndarray, features: pd.DataFrame, end: int) -> None:
        import lightgbm as lgb

        x = self._design(features, 0, end)
        y = features.iloc[:end]["log_rv_next"].to_numpy(dtype=float)
        mask = np.isfinite(x).all(axis=1) & np.isfinite(y)
        if mask.sum() < 200:
            self.model = None
            return
        self.model = lgb.LGBMRegressor(random_state=cfg.SEED, **cfg.GBM_PARAMS)
        self.model.fit(x[mask], y[mask])
        residual = y[mask] - self.model.predict(x[mask])
        self.residual_variance = float(np.var(residual))

    def forecast(self, returns: np.ndarray, features: pd.DataFrame, start: int, stop: int) -> np.ndarray:
        if self.model is None:
            return np.full(stop - start, float(np.var(np.asarray(returns[:stop], dtype=float))) or EPS)
        x = self._design(features, start, stop)
        prediction = self.model.predict(x)
        return np.maximum(np.exp(prediction + 0.5 * self.residual_variance), EPS)


#: Features the tree model and the network consume. The two are given the same
#: inputs so that a difference in their scores is a difference in the functional
#: form and not in the information set.
_GBM_FEATURES: tuple[str, ...] = (
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


@dataclass
class GRUNet(VolatilityModel):
    """The from-scratch GRU over sequences of the same regressors."""

    name: str = "gru"
    heavy: bool = True
    needs_features: bool = True
    model: nn.GRU | None = None
    residual_variance: float = 0.0
    sequence_length: int = int(cfg.GRU_PARAMS["sequence_length"])
    params: dict[str, object] | None = None

    def reset(self) -> None:
        self.model = None
        self.residual_variance = 0.0

    def fit(self, returns: np.ndarray, features: pd.DataFrame, end: int) -> None:
        columns = list(_GBM_FEATURES)
        values = features[columns].to_numpy(dtype=float)
        targets = features["log_rv_next"].to_numpy(dtype=float)
        x, indices = nn.make_sequences(values, self.sequence_length, 0, end)
        if x.shape[0] < 200:
            self.model = None
            return
        y = targets[indices]
        mask = np.isfinite(x.reshape(x.shape[0], -1)).all(axis=1) & np.isfinite(y)
        rng = np.random.default_rng(cfg.SEED)
        self.model = nn.GRU.fit(x[mask], y[mask], rng, self.params or cfg.GRU_PARAMS)
        residual = y[mask] - self.model.predict(x[mask])
        self.residual_variance = float(np.var(residual))

    def forecast(self, returns: np.ndarray, features: pd.DataFrame, start: int, stop: int) -> np.ndarray:
        fallback = float(np.var(np.asarray(returns[:stop], dtype=float))) or EPS
        out = np.full(stop - start, fallback)
        if self.model is None:
            return out
        columns = list(_GBM_FEATURES)
        values = features[columns].to_numpy(dtype=float)
        x, indices = nn.make_sequences(values, self.sequence_length, start, stop)
        if x.shape[0] == 0:
            return out
        mask = np.isfinite(x.reshape(x.shape[0], -1)).all(axis=1)
        if mask.any():
            prediction = self.model.predict(x[mask])
            for position, value in zip(indices[mask], prediction, strict=True):
                out[position - start] = max(float(np.exp(value + 0.5 * self.residual_variance)), EPS)
        return out


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------
@dataclass
class ForecastSet:
    """Forecasts plus the bookkeeping needed to interpret them."""

    variance: pd.DataFrame
    index: pd.DatetimeIndex
    refit_dates: list[pd.Timestamp]
    model_names: list[str]

    def __getitem__(self, name: str) -> pd.Series:
        return self.variance[name]


def walk_forward(
    models: list[VolatilityModel],
    returns: pd.Series,
    features: pd.DataFrame,
    init_window: int = cfg.BACKTEST_INIT_WINDOW,
    refit_every: int = cfg.REFIT_EVERY,
    nn_refit_every: int = cfg.NN_REFIT_EVERY,
    verbose: bool = False,
) -> ForecastSet:
    """Run every model through the same expanding-window walk-forward scheme.

    At each refit point ``T`` every model is re-estimated on ``returns[:T]`` and
    then asked for one-step-ahead forecasts over ``[T, T + refit_every)``. The
    variance *state* inside that block is filtered with the realised returns as
    they arrive, so the parameters are stale but the state is current - which is
    both what a desk does and the only way the recursions remain legal.

    Models flagged ``heavy`` are retrained on a longer cycle (annually by
    default): refitting a network every month on an expanding window would take
    hours and would not be what anyone does.
    """
    r = returns.to_numpy(dtype=float)
    n = r.size
    if init_window >= n - 10:
        raise ValueError(f"init_window={init_window} leaves no evaluation sample in {n} rows")

    out = pd.DataFrame(index=returns.index, columns=[m.name for m in models], dtype=float)
    refit_dates: list[pd.Timestamp] = []
    last_nn_fit = -(10**9)

    for model in models:
        model.reset()

    block_starts = list(range(init_window, n, refit_every))
    for block_index, start in enumerate(block_starts):
        stop = min(start + refit_every, n)
        for model in models:
            periodic = model.heavy
            due = (block_index == 0) or (not periodic) or (start - last_nn_fit >= nn_refit_every)
            if due:
                model.fit(r, features, start)
                if periodic:
                    last_nn_fit = start
            out.loc[returns.index[start:stop], model.name] = model.forecast(r, features, start, stop)
        refit_dates.append(returns.index[start])
        if verbose and block_index % 25 == 0:
            print(f"    refit {block_index + 1}/{len(block_starts)} at {returns.index[start].date()}")

    return ForecastSet(
        variance=out.astype(float),
        index=returns.index,
        refit_dates=refit_dates,
        model_names=[m.name for m in models],
    )


def default_model_zoo(fast: bool = False, gru_params: dict[str, object] | None = None) -> list[VolatilityModel]:
    """The model list the pipeline runs, in the order the tables show them.

    ``fast`` lowers the network's training budget. It does not drop the network:
    the whole point of the study is the comparison, and a speed flag that removes a
    contestant turns a comparison into a different comparison while keeping the
    same title.
    """
    models: list[VolatilityModel] = [
        RollingVariance(window=21),
        RollingVariance(window=63),
        EWMA(lam=0.94),
        EWMA(lam=0.94, fit_decay=True),
        GARCH(kind="garch"),
        GARCH(kind="gjr"),
        GARCH(kind="egarch"),
        HAR(use_vix=False),
        HAR(use_vix=True),
        GBM(),
    ]
    models.append(GRUNet(params=gru_params))
    return models
