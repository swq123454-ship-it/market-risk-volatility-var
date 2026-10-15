"""The simulated market: a two-component GJR-GARCH with jumps and a crisis schedule.

Why simulate at all
-------------------
A volatility study needs twenty years of daily history, and the honest sources
for that (CRSP, Refinitiv, Bloomberg) are licensed. yfinance will give you index
closes for free, and ``--data-mode real`` uses it, but it gives you *closes*, not
intraday bars, so the realised-variance series has to be proxied from a range
estimator rather than measured.

The default pipeline therefore runs on a process whose parameters are written
down and dumped to ``outputs/generator_manifest.json``. Three things become
possible that real data does not allow:

1. **Forecasts can be scored against the truth.** The simulated ex-ante
   conditional variance sigma^2_t is known exactly. A model can be judged on
   whether it tracks the *latent* variance, not merely on whether it tracks a
   noisy proxy of it.
2. **Estimators can be validated.** Parameters are known, so maximum likelihood
   can be checked for recovery rather than merely for convergence, and the
   conditional-variance recursion can be checked against the process that
   generated it.
3. **The study is reproducible on any machine, in one command, forever.** A
   reviewer can clone, run, and get the same table.

What it is not: evidence about markets. Every table records which mode produced
it, and the README says so again.

The process
-----------
Daily equity returns follow a GJR-GARCH(1,1,1) whose unconditional variance is
*targeted at a time-varying level* given by a slow, near-unit-root component in
log variance:

    sigma^2_t = omega_t + (alpha + gamma * 1[r_{t-1} < 0]) * r_{t-1}^2
                + beta * sigma^2_{t-1}
    omega_t   = (1 - alpha - beta - gamma/2) * kappa_t * m_t * sigma_bar^2
    ln m_t    = phi * ln m_{t-1} + sigma_L * eta_t

Four design choices here are load-bearing and would be wrong to skip:

* **Variance targeting with a time-varying level.** With a constant target the
  process is short-memory, and a HAR-RV model has nothing to find. The slow
  component is what makes the realised-volatility family a genuine competitor
  rather than a straw man.
* **The GJR term.** Negative shocks raise next period's variance more than
  positive shocks of the same size. Without it there is no leverage effect, and
  the corresponding row of the stylised-facts table becomes vacuous.
* **Student-t innovations with nu = 6.** Daily equity returns are not normal, and
  the difference between a Gaussian and a fat-tailed VaR at 99% is the entire
  point of the parametric method.
* **The recursion is driven by the realised return, not by a separate draw.**
  This sounds obvious and is easy to get wrong: if the variance recursion is
  driven by a second, independent series of innovations, the simulated returns
  show no volatility clustering even though the latent variance path looks
  exactly right. The test suite asserts that the autocorrelation of squared
  returns is positive.

Crises enter twice, on purpose. The multiplier kappa_t raises the *target*
variance, so volatility builds through the GARCH recursion the way it does in
practice, and it raises the Poisson jump intensity, so the first day of a crisis
is genuinely violent rather than merely elevated. A crisis that arrived only
through the target would be too smooth to test whether a model reacts fast
enough, which is the question the stress slices ask.

Realised variance is constructed as a **Brownian bridge pinned to the daily
return**, which is the property that makes it usable:

    a_i    = (z_i - mean(z)) / sqrt((M-1)/M),      z ~ N(0, 1),  i = 1..M
    r_t,i  = sigma_t * eps_t / M + sigma_t * sqrt(M-1)/M * a_i

so ``sum_i r_{t,i} = sigma_t * eps_t`` exactly, ``Var(r_{t,i}) = sigma^2_t / M``,
``Cov(r_{t,i}, r_{t,j}) = 0``, and therefore ``RV_t = sum_i r_{t,i}^2`` is
unbiased for sigma^2_t with a relative standard deviation of sqrt(2/(M-1)) ~ 16%.
The daily return ``eps_t`` is a unit-variance Student-t, so the fat tails survive
in the daily series without contaminating the realised measure.

That last point is worth stating plainly because an earlier version of this file
got it wrong. Multiplying the whole day by a single chi-square mixing variable is
a natural way to make the *daily* return heavy-tailed, and it works - but the
mixing factor then enters RV as well, giving it a coefficient of variation of 70%
and dropping the correlation between RV and the true conditional variance to
0.63. HAR-RV would have been fitted to a signal that was mostly its own noise,
and the realised-volatility family would have lost the comparison for a reason
that had nothing to do with markets. The test suite now asserts that RV is
unbiased for the simulated variance *and* that it correlates with it above 0.9.

With M = 78 five-minute bars, the 16% sampling error is large enough that
comparing a *measured* volatility against a *filtered* one is not rigged in
favour of either - which is the honest version of a comparison that the
literature usually reports in favour of the measured one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as cfg


@dataclass
class AssetPath:
    """Simulated history for one instrument."""

    key: str
    label: str
    kind: str
    #: Daily log return, including jumps.
    returns: pd.Series
    #: True *ex-ante* conditional variance sigma^2_t of the diffusion component.
    #: This is the target a volatility model can legitimately be scored against:
    #: jumps are unpredictable by construction, so a model cannot be expected to
    #: forecast them, and scoring against a jump-inclusive target would put an
    #: irreducible error floor under every model.
    true_variance: pd.Series
    #: Sum of squared intraday returns plus jumps squared - what a practitioner
    #: actually observes. Unbiased for (diffusion variance + jump variance).
    realised_variance: pd.Series
    #: Intraday 5-minute log returns, shape (n_days, bars).
    intraday: np.ndarray
    close: pd.Series
    open: pd.Series
    high: pd.Series
    low: pd.Series
    #: Crisis multiplier applied to the variance target on each day.
    multiplier: pd.Series
    #: Realised jump magnitudes; zero on non-jump days.
    jump: pd.Series

    @property
    def frame(self) -> pd.DataFrame:
        """OHLC plus diagnostics, which is what every downstream loader expects."""
        frame = pd.DataFrame(
            {
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "ret": self.returns,
                "true_variance": self.true_variance,
                "realised_variance": self.realised_variance,
                "multiplier": self.multiplier,
                "jump": self.jump,
            }
        )
        frame.index.name = "date"
        return frame


@dataclass
class GeneratorOutput:
    """Everything the simulator produced, plus the parameters that produced it."""

    assets: dict[str, AssetPath]
    vix: pd.Series
    dates: pd.DatetimeIndex
    parameters: dict[str, float] = field(default_factory=dict)
    seed: int = cfg.SEED

    def manifest(self) -> dict[str, object]:
        """Serialisable record of the data-generating process."""
        return {
            "seed": self.seed,
            "start": str(self.dates[0].date()),
            "end": str(self.dates[-1].date()),
            "trading_days": int(len(self.dates)),
            "parameters": dict(self.parameters),
            "crisis_schedule": [
                {"name": name, "start": start, "end": end, "multiplier": mult}
                for name, start, end, mult in cfg.CRISIS_SCHEDULE
            ],
        }


def crisis_multiplier(dates: pd.DatetimeIndex) -> np.ndarray:
    """Deterministic stress multiplier per day: ramp in, hold, decay out.

    The profile matters as much as the level. A multiplier that switches on the
    first day of a crisis and off on the last produces a step in the variance
    target, and because the GARCH recursion only pulls a few percent of the way
    towards its target each day, the *realised* variance is a heavily lagged copy
    of that step: an earlier version raised the GFC target to 4.2x for seven
    months and the window still came out no more volatile than a calm year.

    Three features fix that:

    * a linear **ramp** of ``CRISIS_RAMP_DAYS`` before the onset, so the target
      climbs rather than jumps and the recursion has time to follow;
    * a window that is **wider than the historical episode**, so the shock is
      still in force when the labelled stress slice begins;
    * an **exponential decay** after the end rather than a step back to one.
      Volatility does not normalise the day a crisis is declared over; relaxing
      the multiplier with a ~45-day e-folding time is what the trailing edge of
      2009 and 2020 actually looked like, and it stops a model from being
      rewarded for simply reading the calendar.
    """
    mult = np.ones(len(dates), dtype=float)
    calendar = dates.values.astype("datetime64[D]")
    for _name, start, end, level in cfg.CRISIS_SCHEDULE:
        lo = np.datetime64(pd.Timestamp(start) - pd.Timedelta(days=cfg.CRISIS_RAMP_DAYS), "D")
        onset = np.datetime64(pd.Timestamp(start), "D")
        hi = np.datetime64(pd.Timestamp(end), "D")

        ramping = (calendar >= lo) & (calendar < onset)
        if ramping.any():
            span = (calendar[ramping] - lo) / np.timedelta64(1, "D")
            mult[ramping] = np.maximum(mult[ramping], 1.0 + (level - 1.0) * span / cfg.CRISIS_RAMP_DAYS)

        inside = (calendar >= onset) & (calendar <= hi)
        mult[inside] = np.maximum(mult[inside], level)

        after = calendar > hi
        if after.any():
            days_after = (calendar[after] - hi) / np.timedelta64(1, "D")
            tail = 1.0 + (level - 1.0) * np.exp(-days_after / cfg.CRISIS_DECAY_DAYS)
            mult[after] = np.maximum(mult[after], tail)
    return mult


def _slow_component(n: int, rng: np.random.Generator, phi: float, sigma_l: float) -> np.ndarray:
    """Slow log-variance component m_t, normalised to mean one.

    phi = 0.995 gives a half-life of about 138 trading days, which is where the
    long-memory signature of realised volatility comes from. The stationary
    log standard deviation is ``sigma_l / sqrt(1 - phi^2)``, i.e. about 0.45 for
    the configured values: the level of volatility itself wanders by a factor of
    roughly exp(+/-0.45) over a decade, which is the range the VIX actually spans.
    """
    innovations = rng.standard_normal(n)
    ln_m = np.empty(n)
    state = 0.0
    for t in range(n):
        state = phi * state + sigma_l * innovations[t]
        ln_m[t] = state
    m = np.exp(ln_m)
    return m / m.mean()


def _intraday_from_daily(
    diffusion: np.ndarray, sigma2: np.ndarray, bridge: np.ndarray, bars: int, drift: float
) -> np.ndarray:
    """Brownian-bridge intraday path pinned to a given daily diffusion return.

    Exposed separately from the day loop so the bridge identity
    (``sum(r_i) == diffusion`` and ``Var(r_i) == sigma^2 / M``) can be asserted
    directly in the test suite rather than inferred from a full simulation.
    """
    weight = np.sqrt(bars - 1.0) / bars
    intraday = diffusion[:, None] / bars + np.sqrt(sigma2)[:, None] * weight * bridge
    intraday += drift / bars
    return intraday


def _ohlc_from_intraday(
    intraday: np.ndarray, start_price: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build OHLC bars from the intraday path.

    The session opens where the previous one closed, and the jump - when there is
    one - arrives in the first five minutes, so it sits inside the day's range.
    Real markets gap overnight; the difference does not touch any estimator used
    here, because the range measures and realised variance both include the jump
    and bipower variation excludes it either way.

    The high and low are explicitly clamped to include the open. Without that
    clamp, a session in which every bar falls produces a high below the open and
    an OHLC bar that no data feed would ever print. The test suite asserts
    ``low <= open, close <= high`` on every bar, which is what caught it.
    """
    path = np.cumsum(intraday, axis=1)
    n = path.shape[0]
    log_open = np.zeros(n)
    log_close = path[:, -1]
    log_high = np.maximum(path.max(axis=1), 0.0)
    log_low = np.minimum(path.min(axis=1), 0.0)
    prices = start_price * np.exp(np.cumsum(log_close))
    previous = np.concatenate([[start_price], prices[:-1]])
    return (
        previous * np.exp(log_open),
        previous * np.exp(log_high),
        previous * np.exp(log_low),
        prices,
    )


def simulate_asset(
    kind: str,
    n: int,
    rng: np.random.Generator,
    target_variance: np.ndarray,
    multiplier: np.ndarray,
    idiosyncratic: bool,
    with_jumps: bool = True,
) -> dict[str, np.ndarray]:
    """Simulate one instrument's daily path.

    Returns a dict rather than an ``AssetPath`` because the non-primary
    instruments have to be composed with the common factor before their OHLC can
    be built.

    ``with_jumps`` is False for the idiosyncratic sleeves. Jumps are a
    *systematic* event - a crash is a market-wide gap, not a private one - and
    letting each sleeve roll its own Poisson process would give every asset an
    independent jump leg. That inflates each instrument's jump share, breaks the
    interpretation of the correlation matrix, and hands EUR/USD a 23% jump share
    that no FX desk would recognise.
    """
    bars = cfg.INTRADAY_BARS
    dof = cfg.SIM_T_DOF
    if idiosyncratic:
        scale_factor = float(cfg.SIM_IDIO_VOL.get(kind, 0.006)) / cfg.SIM_UNCOND_VOL
        alpha = cfg.SIM_SHORT_ALPHA
        beta = cfg.SIM_SHORT_BETA * 0.96
        gamma = cfg.SIM_LEVERAGE_GAMMA * 0.85
        target = target_variance * scale_factor**2
    else:
        alpha, beta, gamma = cfg.SIM_SHORT_ALPHA, cfg.SIM_SHORT_BETA, cfg.SIM_LEVERAGE_GAMMA
        target = target_variance

    persistence = alpha + beta + 0.5 * gamma
    # Only the common factor earns a risk premium. In a factor model the sleeve
    # has beta loading and zero alpha, so E[r_i] = beta_i * premium: giving each
    # idiosyncratic sleeve the full equity drift as well would hand the low-beta
    # assets a premium they did not earn and would make the cross-asset
    # comparison meaningless. A factor with drift is convenient because it gives
    # the index a sensible trend, so the bias is removed on the sleeve instead.
    drift = 0.0 if idiosyncratic else cfg.SIM_DRIFT_ANNUAL / cfg.TRADING_DAYS
    # ---- Pre-draw the shocks ------------------------------------------------
    # The daily innovation is a unit-variance Student-t, skewed to the left by a
    # two-piece (Fernandez-Steel style) rescaling:
    #     eps = u * (1 - delta) if u >= 0 else u * (1 + delta),  normalised
    # Downside moves are stretched by (1+delta) and upside moves compressed by
    # (1-delta), and the whole thing is divided by sqrt(1+delta^2) so the variance
    # is unchanged. Skewness is therefore a free parameter that costs nothing in
    # variance.
    #
    # The alternative - getting the negative skew out of asymmetric jumps - was
    # tried first and is a false economy. To reach a skewness of -0.4 purely from
    # jumps, the jumps have to carry roughly 15% of total variance, and because a
    # squared jump has an enormous variance (a 3-sigma jump contributes 900 times
    # the mean), RV then becomes mostly its own noise: the correlation with the
    # latent conditional variance fell to 0.5 and the autocorrelation of squared
    # returns to 0.11. Skewing the diffusion instead keeps RV a clean estimate of
    # integrated variance, which is what the realised-volatility leg of this study
    # depends on.
    #
    # A deliberate side effect: the GARCH-t likelihood fitted later assumes a
    # *symmetric* t. The simulated data are skewed, so the fitted model is
    # misspecified on purpose, in the same direction real models are, and the
    # parametric VaR's left tail fails by more than a symmetric-t calibration
    # would predict. That failure is reported rather than engineered away.
    #
    # The two-piece rescaling is not mean-preserving: E[eps] = -E|eps| * skew, so
    # for the configured skew the raw draw carries a mean of about -0.13. Leaving
    # that in puts a spurious -15 bp/day into the index and compounds the price
    # level to 0.4 over twenty years. The innovation is therefore standardised on
    # the sample, which pins the realised mean to zero and the variance to one
    # exactly, so the only drift in the simulated index is the configured one.
    eps = rng.standard_t(dof, size=n) * np.sqrt((dof - 2.0) / dof)
    skew = cfg.SIM_INNOVATION_SKEW
    eps = np.where(eps >= 0.0, eps * (1.0 - skew), eps * (1.0 + skew))
    eps = (eps - eps.mean()) / eps.std()

    # The intraday path is a Brownian bridge pinned to that daily innovation.
    # Take M iid normals, remove their mean, and rescale:
    #     a_i = (z_i - mean(z)) / sqrt((M-1)/M)   =>   sum(a) = 0,  Var(a_i) = 1
    # then   r_i = sigma*eps/M + sigma*sqrt(M-1)/M * a_i
    # gives   sum(r) = sigma*eps,  Var(r_i) = sigma^2/M,  Cov(r_i, r_j) = 0,
    # and RV = sum(r_i^2) has expectation sigma^2 with a relative standard
    # deviation of sqrt(2/(M-1)) ~ 16%.
    #
    # An earlier version instead multiplied the whole day by a single chi-square
    # mixing variable, which is a much more natural way to make the *daily* return
    # heavy-tailed. It was wrong for this study: the mixing factor went into RV as
    # well, giving it a coefficient of variation of 70% and dropping the
    # correlation between RV and the true conditional variance to 0.63. HAR-RV
    # would then have been fitted to a signal that was mostly its own noise, and
    # the realised-volatility family would have lost for an artefactual reason.
    z = rng.standard_normal((n, bars))
    a = (z - z.mean(axis=1, keepdims=True)) / np.sqrt((bars - 1.0) / bars)

    # ---- Jumps -------------------------------------------------------------
    # Modelled additively, in the spirit of a stochastic-volatility model with
    # correlated jumps: the GARCH recursion below is driven by the *diffusive*
    # return only, so the jump variance is not amplified by the recursion's gain
    # of (alpha + gamma/2)/(1 - persistence) ~ 3.
    #
    # The jump leg is left sparse - zero on most days - because bipower variation
    # and the jump tests below rely on the jump series being a genuine jump
    # indicator. The drift it carries is compensated in ``drift`` further down
    # rather than by subtracting the mean from the jump series, which would make
    # every single day look like a jump day.
    rate = np.clip(cfg.SIM_JUMP_INTENSITY * np.sqrt(multiplier), 0.0, 0.5)
    if with_jumps:
        is_jump = rng.random(n) < rate
        size = np.abs(rng.standard_normal(n)) * cfg.SIM_JUMP_SIGMA
        direction = np.where(rng.random(n) < cfg.SIM_JUMP_DOWN_PROB, -1.0, 1.0)
        jumps = np.where(is_jump, direction * size, 0.0)
    else:
        jumps = np.zeros(n)

    # The index drifts by SIM_DRIFT_ANNUAL. The jump leg is skewed downwards, so
    # it would otherwise subtract a drift of its own; compensating here keeps the
    # equity risk premium a parameter rather than an artefact of jump asymmetry.
    drift = drift - float(np.mean(jumps))

    # ---- The day loop ------------------------------------------------------
    # Only the variance recursion is sequential: sigma^2_t depends on the
    # previous *diffusive* return, which is known as soon as sigma^2_{t-1} is.
    # Everything else is vectorised afterwards.
    sigma2 = np.empty(n)
    previous_variance = float(target[0])
    previous_return = 0.0
    for t in range(n):
        omega = (1.0 - persistence) * target[t]
        value = omega + beta * previous_variance
        if t > 0:
            value += alpha * previous_return**2
            if previous_return < 0.0:
                value += gamma * previous_return**2
        value = max(value, 1e-12)
        sigma2[t] = value
        previous_variance = value
        previous_return = np.sqrt(value) * eps[t]

    diffusion = np.sqrt(sigma2) * eps
    intraday = _intraday_from_daily(diffusion, sigma2, a, bars, drift)
    # The jump arrives in the first five minutes of the session. Putting it
    # inside the intraday path rather than outside is what makes realised
    # variance contain jumps while bipower variation does not - and that
    # difference is the estimator the study uses to separate the two.
    intraday[:, 0] += jumps

    return {
        "returns": intraday.sum(axis=1),
        "sigma2": sigma2,
        "realised": (intraday**2).sum(axis=1),
        "intraday": intraday,
        "jumps": jumps,
        "multiplier": multiplier,
    }


def _simulate_vix(asset: AssetPath, multiplier: np.ndarray, rng: np.random.Generator) -> pd.Series:
    """A VIX-like implied-volatility index.

    The real VIX is the risk-neutral expectation of the next 30 days of
    integrated variance. The simulated one is the *physical* expectation over 22
    trading days multiplied by a variance risk premium that widens when the
    crisis multiplier is high - the empirically observed pattern, and what makes
    the VIX a useful predictor rather than a duplicate of information already
    present in returns.
    """
    sigma2 = asset.true_variance.to_numpy()
    n = sigma2.size
    horizon = cfg.SIM_VIX_HORIZON
    cumulative = np.concatenate([[0.0], np.cumsum(sigma2)])
    lo = np.arange(n) + 1
    hi = np.minimum(np.arange(n) + horizon + 1, n)
    counts = np.maximum(hi - lo, 1)
    forward = (cumulative[hi] - cumulative[lo]) / counts
    forward = np.where(hi > lo, forward, sigma2)
    premium = cfg.SIM_VRP_LEVEL + cfg.SIM_VRP_SLOPE * (multiplier - 1.0)
    noise = 1.0 + cfg.SIM_VIX_NOISE * rng.standard_normal(n)
    vix = 100.0 * np.sqrt(cfg.TRADING_DAYS * forward * premium) * noise
    return pd.Series(np.clip(vix, 5.0, 150.0), index=asset.returns.index, name="vix")


def simulate_market(
    start: str = cfg.DEFAULT_START,
    end: str = cfg.DEFAULT_END,
    seed: int = cfg.SEED,
    assets: tuple[str, ...] | None = None,
) -> GeneratorOutput:
    """Simulate the full instrument universe.

    The common equity factor is simulated first; every other instrument is that
    factor plus a beta loading and its own GJR-GARCH idiosyncratic component. The
    loading structure is what makes the cross-sectional question ("does the
    conclusion survive a change of market") meaningful rather than decorative.
    """
    dates = pd.bdate_range(start, end)
    n = len(dates)
    rng = np.random.default_rng(seed)

    multiplier = crisis_multiplier(dates)
    slow = _slow_component(n, rng, cfg.SIM_LONG_PHI, cfg.SIM_LONG_SIGMA)
    target = multiplier * slow * cfg.SIM_UNCOND_VOL**2

    keys = list(assets) if assets is not None else list(cfg.ASSETS)
    primary = cfg.PRIMARY_ASSET if cfg.PRIMARY_ASSET in keys else keys[0]

    factor: dict[str, np.ndarray] | None = None
    out: dict[str, AssetPath] = {}

    for key in keys:
        meta = cfg.ASSETS[key]
        kind = str(meta["kind"])
        simulated = simulate_asset(
            kind, n, rng, target, multiplier, idiosyncratic=key != primary, with_jumps=key == primary
        )

        if key == primary:
            # The factor *includes* the primary market's jumps: a crash in the
            # index is a systematic event and must propagate to every other
            # instrument, otherwise cross-asset correlation breaks on exactly the
            # days that matter.
            factor = simulated
            intraday = simulated["intraday"]
            jumps = simulated["jumps"]
            sigma2 = simulated["sigma2"]
        else:
            assert factor is not None
            # Compose bar by bar, not just day by day. Composing only the daily
            # returns and then inventing an OHLC bar would silently discard the
            # intraday range, and the range estimators are one of the realised
            # measures under test.
            loading = float(meta["beta"])
            intraday = loading * factor["intraday"] + simulated["intraday"]
            jumps = loading * factor["jumps"] + simulated["jumps"]
            # Factor and idiosyncratic shocks are independent by construction, so
            # the ex-ante variance of the sum is the sum of the ex-ante variances.
            sigma2 = loading**2 * factor["sigma2"] + simulated["sigma2"]

        total = intraday.sum(axis=1)
        realised = (intraday**2).sum(axis=1)
        o, h, low, c = _ohlc_from_intraday(intraday, start_price=100.0)

        out[key] = AssetPath(
            key=key,
            label=str(meta["label"]),
            kind=kind,
            returns=pd.Series(total, index=dates, name=key),
            true_variance=pd.Series(sigma2, index=dates),
            realised_variance=pd.Series(realised, index=dates),
            intraday=intraday,
            close=pd.Series(c, index=dates),
            open=pd.Series(o, index=dates),
            high=pd.Series(h, index=dates),
            low=pd.Series(low, index=dates),
            multiplier=pd.Series(multiplier, index=dates),
            jump=pd.Series(jumps, index=dates),
        )

    vix = _simulate_vix(out[primary], multiplier, rng)

    parameters = {
        "short_alpha": cfg.SIM_SHORT_ALPHA,
        "short_beta": cfg.SIM_SHORT_BETA,
        "leverage_gamma": cfg.SIM_LEVERAGE_GAMMA,
        "persistence": cfg.SIM_SHORT_ALPHA + cfg.SIM_SHORT_BETA + 0.5 * cfg.SIM_LEVERAGE_GAMMA,
        "unconditional_vol_daily": cfg.SIM_UNCOND_VOL,
        "unconditional_vol_annual": cfg.SIM_UNCOND_VOL * np.sqrt(cfg.TRADING_DAYS),
        "long_memory_phi": cfg.SIM_LONG_PHI,
        "long_memory_sigma": cfg.SIM_LONG_SIGMA,
        "long_memory_log_sd": cfg.SIM_LONG_SIGMA / np.sqrt(1.0 - cfg.SIM_LONG_PHI**2),
        "t_dof": cfg.SIM_T_DOF,
        "jump_intensity": cfg.SIM_JUMP_INTENSITY,
        "jump_sigma": cfg.SIM_JUMP_SIGMA,
        "intraday_bars": cfg.INTRADAY_BARS,
        "variance_risk_premium": cfg.SIM_VRP_LEVEL,
    }
    return GeneratorOutput(assets=out, vix=vix, dates=dates, parameters=parameters, seed=seed)
