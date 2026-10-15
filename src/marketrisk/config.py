"""Single source of truth for every tunable in the market-risk study.

Nothing in this package is allowed to hard-code a window length, a confidence
level, a cost or a seed. Everything that a reviewer might reasonably want to
change lives here, is passed explicitly down the call stack, and is echoed into
``outputs/metrics.json`` so that a number in the README can always be traced
back to the configuration that produced it.

The second reason this module exists is honesty about the data. The default
pipeline runs on a **simulated** price history whose data-generating process is
written down in ``generator.py`` and dumped to ``outputs/generator_manifest.json``
next to the results. That is not a substitute for market data: it is a way to
run the whole study reproducibly, on a machine without a market-data licence,
against a process where the *true* volatility path is known. Being able to score
a volatility forecast against the truth (rather than against a noisy proxy) is
something real data cannot give you, and several conclusions below depend on it.
``--data-mode real`` swaps in downloaded index history instead, and every table
records which mode produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

TRADING_DAYS = 252
CALENDAR_DAYS = 365

# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------
# Twenty years of business days. Long enough to contain three genuinely
# different stress regimes (GFC 2008, COVID 2020, the 2022 rate shock) and short
# enough that a full walk-forward re-estimation finishes in minutes.
DEFAULT_START = "2005-01-03"
DEFAULT_END = "2024-12-31"

#: The instrument the headline comparison is run on. Everything else is a
#: cross-check on whether a conclusion survives a change of market.
PRIMARY_ASSET = "SPX"

#: Index / FX universe. ``beta`` is the loading on the common equity factor used
#: by the simulator; it is ignored in real-data mode. EUR/USD loads positively on
#: the equity factor because the euro tends to weaken when equities fall.
ASSETS: dict[str, dict[str, object]] = {
    "SPX": {"symbol": "^GSPC", "label": "S&P 500", "kind": "equity", "beta": 1.00},
    "FTSE": {"symbol": "^FTSE", "label": "FTSE 100", "kind": "equity", "beta": 0.78},
    "ISEQ": {"symbol": "^ISEQ", "label": "ISEQ 20 (Ireland)", "kind": "equity", "beta": 0.68},
    "EURUSD": {"symbol": "EURUSD=X", "label": "EUR/USD", "kind": "fx", "beta": 0.22},
}

VIX_SYMBOL = "^VIX"
VIX_LABEL = "CBOE VIX"

# ---------------------------------------------------------------------------
# Simulator (data-generating process)
# ---------------------------------------------------------------------------
# The simulator is a two-component GJR-GARCH with Student-t innovations, jumps
# and a documented crisis schedule. The two-component structure matters: a
# single-component GARCH has short memory, and a HAR-RV model fitted to it would
# have nothing to find. The slow component is what gives the realised-volatility
# models a genuine long-memory signal to exploit.

#: Five-minute bars per trading session (78 = 6.5 hours). Sets the sampling error
#: of the realised-variance proxy: with M bars, RV estimates integrated variance
#: with a relative standard deviation of sqrt(2/M) ~ 16%.
INTRADAY_BARS = 78

# Component 1: fast, mean-reverting GARCH on the short-run variance.
SIM_SHORT_ALPHA = 0.06
SIM_SHORT_BETA = 0.86
SIM_LEVERAGE_GAMMA = 0.09  # GJR term: extra variance after a negative shock
SIM_UNCOND_VOL = 0.0064  # 0.64% daily on the fast component alone

# Component 2: slow, near-unit-root component in log variance. This is what
# produces the "long memory" of realised volatility: phi = 0.995 is a half-life
# of about 138 trading days, so the level of volatility drifts over years rather
# than reverting in weeks, which is exactly the structure a HAR-RV regression is
# built to exploit.
#
# The *size* of this component is deliberately modest. A near-unit-root process
# has an enormous low-frequency range: with sigma_L = 0.075 the stationary log
# standard deviation is 0.75, and over a twenty-year sample the level of variance
# wanders across a 90x range (0.05x to 4.7x the mean), which completely swamps
# the dated crisis schedule - the GFC window happened to fall on a *low* reading
# and came out no more volatile than the calm years. An earlier calibration made
# exactly that mistake and the stress slices were meaningless.
#
# sigma_L = 0.035 gives a stationary log standard deviation of about 0.35, i.e.
# volatility itself moves by roughly +-40% around its trend, which is what the
# VIX actually does between calm years and normal corrections. The large, dated
# swings are left to the crisis schedule below, where they can be calibrated and
# reported.
SIM_LONG_PHI = 0.995
SIM_LONG_SIGMA = 0.035

#: Heavy tails. nu = 6 is the value the literature typically finds for daily
#: equity index returns; anything above ~30 makes the GARCH-t likelihood
#: indistinguishable from the Gaussian one and would make the fat-tail test
#: vacuous.
SIM_T_DOF = 6.0

#: Jump component. Intensity is the calm-period Poisson rate per day; it is
#: multiplied by the crisis multiplier when a crisis window is active. 75% of
#: jumps are downward, which is what gives the return distribution its negative
#: skew: an equal split would leave skewness at roughly zero and the leverage
#: row of the stylised-facts table would be the only evidence of asymmetry.
#:
#: Jump size is deliberately modest. A larger jump variance does not merely add
#: fat tails - because the variance recursion is driven by the *realised* return,
#: jumps feed back into sigma^2 with gain (alpha + gamma/2)/(1 - persistence),
#: which here is about 3. Jump variance therefore gets multiplied by roughly
#: three on the way into the conditional variance, and a 22% jump share (an
#: earlier calibration) inflated unconditional volatility by 60% and put an
#: implausible -14% annual drift into the index.
SIM_JUMP_INTENSITY = 0.012
SIM_JUMP_SIGMA = 0.020
SIM_JUMP_DOWN_PROB = 0.60

#: Left-skew of the *diffusion* innovation, via a two-piece (Fernandez-Steel
#: style) rescaling of the unit-variance Student-t. Downside draws are stretched
#: by (1 + delta) and upside draws compressed by (1 - delta), then the whole thing
#: is divided by sqrt(1 + delta^2), so skewness is bought without touching the
#: variance.
#:
#: The obvious alternative - getting the negative skew out of asymmetric jumps -
#: was tried and rejected. Reaching a skewness of about -0.4 purely from jumps
#: requires them to carry roughly 15% of total variance; a squared jump has an
#: enormous variance, so RV then becomes mostly its own noise, the correlation
#: between RV and the latent conditional variance collapses, and the
#: realised-volatility leg of the study is crippled for an artefactual reason.
#: Skewing the diffusion keeps RV a clean estimate of integrated variance.
#:
#: There is a deliberate side effect: the GARCH-t likelihood fitted later assumes
#: a *symmetric* t, so the fitted model is misspecified in the same direction real
#: models are, and the parametric left tail fails by more than it otherwise would.
#: That failure is reported, not engineered away.
SIM_INNOVATION_SKEW = 0.10

#: Drift of the diffusion component. The jump leg is de-meaned before it is
#: added, so the index drift is this number and the negative skewness of the
#: return distribution comes from jump *asymmetry* rather than from a downward
#: drift. 5.5% is roughly the price return of a developed equity index over
#: 2005-2024.
SIM_DRIFT_ANNUAL = 0.055

#: Declared stress windows, applied as a multiplicative shock to the variance
#: target. These exist so the stress slices below are guaranteed to be stressful;
#: they are recorded in the manifest rather than hidden.
#:
#: Each tuple is (name, onset, end, level). ``onset`` is deliberately *earlier*
#: than the historical start of the episode and ``end`` later, because the GARCH
#: recursion responds to a change in its target slowly: the AR coefficient of the
#: variance level is random (it depends on the squared innovation) and its
#: *typical* value puts the half-life of the level at roughly 130 trading days,
#: not the 21 that the mean-reversion parameter alpha + beta + gamma/2 suggests.
#: A multiplier switched on for the seven months of the GFC and off again reached
#: only about half its nominal level, and the "stress" windows came out no more
#: volatile than the calm years. The generating window therefore has to be wide.
#:
#: The levels are calibrated against the *achieved* 60-day realised volatility in
#: each labelled window, not against the nominal multiple - the calibration table
#: is printed by the pipeline and the values are what reproduce 45-60% realised
#: volatility in the GFC and COVID windows against a ~13% calm baseline.
CRISIS_SCHEDULE: tuple[tuple[str, str, str, float], ...] = (
    ("GFC", "2007-06-01", "2009-06-30", 20.0),
    ("Euro sovereign debt", "2011-06-01", "2012-01-31", 3.0),
    ("China / oil selloff", "2015-07-01", "2016-02-29", 2.0),
    ("Volmageddon", "2018-01-15", "2018-03-31", 3.0),
    ("Q4 2018 selloff", "2018-10-01", "2019-01-31", 2.4),
    ("COVID-19", "2019-12-01", "2020-07-31", 26.0),
    ("2022 rate shock", "2021-10-01", "2023-03-31", 5.0),
)

#: Trading days over which a crisis multiplier is ramped in before its onset, and
#: the e-folding time of the decay after it ends. Without the ramp the target
#: steps discontinuously and the variance path is a lagged copy of that step;
#: with it, volatility builds the way it does when a market deteriorates.
CRISIS_RAMP_DAYS = 40
CRISIS_DECAY_DAYS = 25

#: Idiosyncratic volatility for the non-primary assets, as a daily standard
#: deviation (equity sleeves and FX separately - EUR/USD is a much quieter
#: series than an equity index and pretending otherwise would be sloppy). These
#: are set so that the equity sleeves come out around 0.80 correlated with the
#: US market, which is where the large developed indices actually sit, and so
#: that EUR/USD lands near its historical 9% annualised.
#:
#: The level here is easy to get wrong in a way that hides: total variance is
#: beta^2 * factor + idio, so an idio sleeve that looks small in absolute terms
#: can dominate once beta is below one. An earlier value of 0.0052 put FTSE's
#: idiosyncratic variance *above* its systematic variance and made the UK index
#: more volatile than the S&P, which is exactly backwards.
SIM_IDIO_VOL = {"equity": 0.0037, "fx": 0.0022}

#: Variance risk premium baked into the simulated VIX: the option market prices
#: expected variance above the physical expectation, and the spread widens in
#: stress. 1.25 means the simulated VIX sits ~25% above the physical forecast.
SIM_VRP_LEVEL = 1.25
SIM_VRP_SLOPE = 0.35
SIM_VIX_HORIZON = 22
SIM_VIX_NOISE = 0.04


# ---------------------------------------------------------------------------
# Volatility-model layer
# ---------------------------------------------------------------------------
#: Rolling-window estimators of daily volatility, in trading days.
ROLLING_WINDOW = 21
ROLLING_WINDOWS = (21, 63)

#: RiskMetrics decay. 0.94 is the published J.P. Morgan convention for daily
#: data; the pipeline also fits the decay by QLIKE so the convention can be
#: checked rather than assumed.
EWMA_LAMBDA = 0.94
EWMA_LAMBDA_GRID = (0.86, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98)

#: Conditional-variance models estimated by maximum likelihood. The GARCH family
#: is fitted on daily returns; HAR on log realised variance.
GARCH_MODELS = ("garch", "gjr", "egarch")
GARCH_BACKCAST_WEIGHT = 0.94  # decay for the variance backcast that seeds the recursion

#: HAR family. ``har`` is the textbook daily/weekly/monthly log-RV regression;
#: ``harx`` adds the jump, leverage and VIX terms.
HAR_LAGS = {"d": 1, "w": 5, "m": 22}

#: Gradient-boosted volatility model. Target is log realised variance; the
#: features are the HAR lags plus the jump, leverage and range terms.
GBM_PARAMS: dict[str, object] = {
    "n_estimators": 400,
    "learning_rate": 0.035,
    "num_leaves": 15,
    "min_child_samples": 40,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "verbose": -1,
}

#: NumPy GRU. Implemented from scratch in ``nn.py`` rather than pulled in from a
#: deep-learning framework: with ~4,000 training sequences a framework buys
#: nothing but a 900 MB dependency, and writing the backward pass makes the
#: gradient check in the test suite possible.
GRU_PARAMS: dict[str, object] = {
    "sequence_length": 20,
    "hidden_size": 12,
    "learning_rate": 0.01,
    "epochs": 120,
    "batch_size": 64,
    "grad_clip": 5.0,
    "dropout": 0.0,
    "patience": 12,
}

#: Reduced training budget for ``PipelineConfig.fast``. Enough epochs for the
#: holdout loss to start moving, so the fast path still exercises the early-stop
#: branch and still produces a GRU column; it just does not produce a *good* one.
FAST_GRU_EPOCHS = 30
FAST_GRU_PATIENCE = 6

#: Walk-forward scheme. Parameters are refreshed every ``refit_every`` trading
#: days and filtered forward in between; a neural network is retrained yearly.
#: Monthly refresh is what a bank actually does, and it keeps the study's
#: runtime in minutes instead of hours.
BACKTEST_INIT_WINDOW = 1000
REFIT_EVERY = 21
NN_REFIT_EVERY = 252


# ---------------------------------------------------------------------------
# VaR / ES layer
# ---------------------------------------------------------------------------
#: Two levels, because 95% and 99% fail in different ways: 95% is a risk-limit
#: number and 99% is a regulatory-capital number, and a model that passes one
#: routinely fails the other.
CONFIDENCE_LEVELS = (0.95, 0.99)
PRIMARY_LEVEL = 0.99

#: Horizon 1 day is the trading limit; 10 days is the Basel market-risk horizon.
HORIZONS = (1, 10)

#: Historical-simulation lookbacks, in trading days.
HS_WINDOWS = (250, 500)
#: BRW (Boudoukh-Richardson-Whitelaw) exponential decay for age-weighted HS.
HS_DECAY = 0.995

#: Monte-Carlo path count. 20,000 gives a 99% quantile with a standard error of
#: roughly 2.5% of the tail width, which is enough to make method comparisons
#: about the methods rather than about the seed.
MC_PATHS = 20_000
MC_PATHS_MULTI = 12_000

#: Path count used when ``PipelineConfig.fast`` is set. The 99% quantile standard
#: error grows to roughly 5.5% of the tail width, which is too coarse to rank
#: methods by but comfortably fine for checking that the pipeline is wired up.
FAST_MC_PATHS = 4_000

#: How far back to go when estimating the tail-index / degrees of freedom used
#: by the parametric method.
TAIL_FIT_WINDOW = 750

#: Two-sided p-value below which a backtest rejects the model.
ALPHA_SIGNIFICANCE = 0.05

#: The two rosters, named in one place.
#:
#: These exist so that "every model" and "every route" are answerable without running
#: anything, and so that the two questions "did a speed flag quietly drop a contestant?"
#: and "does the report still name what ran?" have one answer. ``default_model_zoo`` and
#: ``build_suite`` are checked against these, order included, by their own tests: the
#: order here is the order the pipeline emits, so a model or a route that moves in one of
#: those two places is a red test rather than two documents that disagree with each
#: other. That is the failure these replaced - ``fast=True`` used to drop a competitor
#: and leave a report comparing ten models while claiming eleven.
#:
#: They are deliberately not the renderer's input. ``scripts/build_docs.py`` runs with no
#: third-party imports, so it cannot import this module, and it checks something else
#: instead: that the bundle it was handed scores every model the *bundle's own* config
#: names. That is a check against a truncated run. A roster that is wrong about the code
#: is only visible from the tests, which is why the constants are here and the
#: assertions are there.
MODEL_NAMES = (
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
)

VAR_METHOD_NAMES = (
    "hs_250",
    "hs_brw",
    "fhs_gjr",
    "fhs_evt_gjr",
    "parametric_normal_gjr",
    "parametric_t_gjr",
    "monte_carlo",
)


# ---------------------------------------------------------------------------
# Basel traffic light
# ---------------------------------------------------------------------------
BASEL_WINDOW = 250  # trading days, ~1 year

#: Basel's own mapping from exceptions in the last 250 days to the capital
#: multiplier add-on. The zone boundaries (0-4 green, 5-9 yellow, 10+ red) are
#: the familiar summary; the plus factor is the part that hits the capital
#: number, and it is not linear in the exception count.
BASEL_ZONES: tuple[tuple[int, int, str], ...] = (
    (0, 4, "green"),
    (5, 9, "yellow"),
    (10, 10_000, "red"),
)
BASEL_PLUS_FACTOR: dict[int, float] = {
    0: 0.00,
    1: 0.00,
    2: 0.00,
    3: 0.00,
    4: 0.00,
    5: 0.40,
    6: 0.50,
    7: 0.65,
    8: 0.75,
    9: 0.85,
}
BASEL_PLUS_FACTOR_RED = 1.00


# ---------------------------------------------------------------------------
# Stress windows and rolling diagnostics
# ---------------------------------------------------------------------------
#: Stress slices for the window-level coverage tables.
#:
#: Only three episodes are listed, and that is a deliberate choice rather than a
#: tidying-up. Every crisis in ``CRISIS_SCHEDULE`` is present in the simulated
#: history, but the *reported* slices are the ones where the market is genuinely
#: stressed relative to its own calm baseline (~13% annualised). The 2011 euro
#: episode, for instance, happens to fall on a stretch when the slow volatility
#: component is low, so its multiplier lifts the level without lifting it above
#: the calm average. Reporting a "stress window" that is quieter than 2017 would
#: be misleading. The realised volatility of every slice is printed by the
#: pipeline, so the claim can be checked rather than taken on trust.
STRESS_WINDOWS: dict[str, tuple[str, str]] = {
    "GFC 2008-09": ("2008-09-01", "2009-06-30"),
    "COVID 2020": ("2020-02-19", "2020-06-30"),
    "2022 rate shock": ("2022-01-01", "2022-10-31"),
}

#: Rolling window used for the coverage heat map.
ROLLING_COVERAGE_WINDOW = 250


# ---------------------------------------------------------------------------
# Cost / loss preferences for model selection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LossConfig:
    """How a volatility forecast and a VaR forecast are scored.

    Two different losses are needed and conflating them is a common mistake.

    ``vol_loss`` scores a variance forecast. MSE on variance is dominated by the
    few largest observations, so the volatility literature uses QLIKE, which is
    robust to the noise in the volatility proxy and is the loss implied by a
    Gaussian likelihood. Both are reported.

    ``var_loss`` scores a quantile forecast. The correct scoring rule for a VaR
    is the asymmetric "tick" (quantile) loss, not a squared error: a VaR that is
    too high costs nothing in tick loss terms beyond the level, while a VaR that
    is too low is penalised at the tail probability. Exceedance counts alone
    cannot rank two models that both hit their target rate.
    """

    #: Weight of the QLIKE term when a single ranking number is needed.
    qlike_weight: float = 1.0
    #: Scale applied to tick loss, purely cosmetic (reported in basis points).
    tick_scale: float = 1e4


@dataclass(frozen=True)
class PipelineConfig:
    """Everything the run needs, in one object."""

    start: str = DEFAULT_START
    end: str = DEFAULT_END
    seed: int = SEED
    data_mode: str = "synthetic"  # "synthetic" | "real"
    asset: str = PRIMARY_ASSET
    refit_every: int = REFIT_EVERY
    init_window: int = BACKTEST_INIT_WINDOW
    mc_paths: int = MC_PATHS
    confidence_levels: tuple[float, ...] = CONFIDENCE_LEVELS
    horizons: tuple[int, ...] = HORIZONS
    fast: bool = False
    losses: LossConfig = field(default_factory=LossConfig)

    @property
    def effective_mc_paths(self) -> int:
        """Path count actually used by the Monte-Carlo leg.

        ``fast`` *lowers* the path count, it never removes the method. Dropping a
        method under a speed flag is how a report ends up quietly comparing five
        routes instead of six and printing a horizon-scaling section that is
        empty because the only method that can produce one is missing.
        """
        return min(self.mc_paths, FAST_MC_PATHS) if self.fast else self.mc_paths

    @property
    def effective_gru_params(self) -> dict[str, object]:
        """GRU hyper-parameters actually used. Same rule: cheaper, not absent."""
        params = dict(GRU_PARAMS)
        if self.fast:
            params["epochs"] = FAST_GRU_EPOCHS
            params["patience"] = FAST_GRU_PATIENCE
        return params


def confidence_label(level: float) -> str:
    """``0.99`` -> ``"99"``. Used for column names and file names."""
    return f"{level * 100:.4g}"


def asset_label(key: str) -> str:
    """Human-readable name for an asset key."""
    meta = ASSETS.get(key)
    return key if meta is None else str(meta["label"])
