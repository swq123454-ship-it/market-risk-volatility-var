# Model documentation

The specification behind the numbers, at the level of detail needed to reimplement it and to know where it is wrong. Every quantity that is a *result* is marked as such and is rendered from `outputs/metrics.json`; everything else is a definition.

- Run: `marketrisk-seed42-synthetic`, generated 2026-10-15T23:12:39.064754+00:00
- Sample: 5,217 rows, 2005-01-03 to 2024-12-31, asset S&P 500
- Initial window 1,000, refit every 21 days, seed 42

## 1. The data-generating process

### 1.1 Conditional variance

The variance is the sum of a short-memory GJR process and a long-memory log component:

```
short:  s2_t = w + (alpha + gamma * 1[r_{t-1} < 0]) * r_{t-1}^2 + beta * s2_{t-1}
long:   ln m2_t = phi * ln m2_{t-1} + sigma * e_t        (e_t ~ N(0, 1))
total:  h_t = s2_t * exp(m2_t) * c_t
```

with `c_t` the crisis multiplier. The second component is what makes the series *look* the way real volatility looks: a slowly varying level, roughly log-normal, with an autocorrelation that decays almost linearly in the logarithm rather than geometrically, so that a HAR regression on it is genuinely well specified rather than approximately so.

| parameter | value | role |
|---|---|---|
| `short_alpha` | 0.0600 | news impact |
| `short_beta` | 0.8600 | persistence of the short component |
| `leverage_gamma` | 0.0900 | extra variance after a down move |
| `persistence` | 0.9650 | `alpha + gamma/2 + beta`, the share of a shock that survives |
| `long_memory_phi` | 0.9950 | near-unit-root decay of the level |
| `long_memory_sigma` | 0.0350 | innovation scale of the level |
| `long_memory_log_sd` | 0.3504 | implied standard deviation in the log |
| `unconditional_vol_daily` | 0.006400 | long-run daily volatility |
| `unconditional_vol_annual` | 0.1016 | the same annualised |

A near-unit-root component is a deliberate trade. It reproduces the long-memory signature that the HAR family is built to exploit, and it means the *level* of volatility carries essentially no information about where it will settle - which is the honest situation and the reason a twenty-year aggregate cannot be improved by knowing the twenty-year average.

### 1.2 Innovations

Returns are `r_t = sqrt(h_t) * z_t` with `z_t` Student-t on `6.0` degrees of freedom, rescaled to unit variance. The fat tail is the point: at 99% a Gaussian and a t with these degrees of freedom disagree by tens of percent, which is what makes the parametric-route comparison meaningful.

A **skew** is also applied. The likelihood of the parametric routes assumes a symmetric t, so they are fitted to data that violate their own assumption. This is the same misspecification real data impose, and it is deliberate: a study in which the parametric models are correctly specified will conclude that parametric models are fine, which is not a useful conclusion.

### 1.3 Jumps

Compound Poisson, intensity **0.0120** per day, jump size `N(0, 0.0200)`. Jumps are **shared across assets in the panel on the same day** and scaled by each asset's beta, so the cross-sectional structure is systematic rather than independent - which is what makes the panel correlated in the tail as well as in the body, and is the property the feature tests check.

This is also why realised variance is a *noisy* target for the conditional variance even on simulated data: on a jump day the realised measure includes the jump and the filter does not. The generator records the jump days, and the jump test is checked against them.

### 1.4 Intraday bridge

Each day is expanded into **78 five-minute bars** through a Brownian bridge that is centred and rescaled so the bars sum exactly to the daily diffusion move. This is what makes realised variance, bipower variation, the jump variation and the Huang-Tauchen statistic computable from the data rather than assumed, and it is why the jump labels can be validated against the jumps the simulator actually placed.

### 1.5 The variance risk premium

A multiplier of **1.2500** is applied to the *realised* measure but not to the conditional variance. Without it the two coincide and every model looks equally well calibrated; with it the realised measure is a biased proxy, which is the real situation and the reason the pipeline also scores against the latent variance. The gap between the two rankings is the size of the bias. *(Result: the ranking is broadly stable between them, so the conclusions are not an artefact of the proxy.)*

### 1.6 Crisis schedule

Volatility is multiplied by a scheduled factor that **ramps in** over a calendar window and decays exponentially afterwards, rather than stepping. A step would be trivially detectable by the jump test and would make the stress windows unrepresentative.

| window | start | end | multiplier |
|---|---|---|---|
| GFC | 2007-06-01 | 2009-06-30 | 20.0 |
| Euro sovereign debt | 2011-06-01 | 2012-01-31 | 3.0 |
| China / oil selloff | 2015-07-01 | 2016-02-29 | 2.0 |
| Volmageddon | 2018-01-15 | 2018-03-31 | 3.0 |
| Q4 2018 selloff | 2018-10-01 | 2019-01-31 | 2.4 |
| COVID-19 | 2019-12-01 | 2020-07-31 | 26.0 |
| 2022 rate shock | 2021-10-01 | 2023-03-31 | 5.0 |

## 2. Realised measures

With `M` intraday returns `r_{t,j}` and `mu1 = sqrt(2/pi)`:

```
RV_t  = sum_j r_{t,j}^2
BV_t  = (pi/2) * sum_{j=2..M} |r_{t,j}| |r_{t,j-1}|
JV_t  = max(RV_t - BV_t, 0)
Parkinson_t = ln(H_t / L_t)^2 / (4 ln 2)
```

The jump statistic is the Huang-Tauchen ratio `(RV - BV) / RV`, standardised by its variance under the null of no jumps. With the Barndorff-Nielsen correction the leading constant is `(mu1^-4 + 2 mu1^-2 - 5) / M`, which is used rather than the large-`M` limit: at `M = 78` the two differ in the third decimal, and a constant that is approximately right is the kind of thing that is never revisited.

Garman-Klass and Rogers-Satchell range measures are computed alongside Parkinson. They are not used as features - the HAR design reads the realised measures - but they are checked for scale invariance and non-negativity, because a range measure that is not scale-invariant is almost always a formula error rather than a modelling choice.

## 3. The feature matrix

| feature | definition |
|---|---|
| `log_rv_d` | daily log realised variance |
| `log_rv_w` | 5-day mean of the daily log |
| `log_rv_m` | 22-day mean of the daily log |
| `bv_ratio` | `BV / RV`, the jump share |
| `ret` | the close-to-close return |
| `ret_neg` | `ret * 1[ret < 0]`, the leverage term |
| `lev` | `1[ret < 0]` |
| `padj` | the Parkinson measure |
| `vix` | option-implied variance, the only genuinely forward-looking input |

The averaging is done **in logs**, not on the variances. Averaging variance over five days and then taking the logarithm is not the same estimator as averaging the log, and the HAR literature is specific about which one is meant; the test suite pins the distinction.

**Leakage control.** Every feature is shifted one day forward, so the row dated `t` contains only information available at the close of `t-1`. The target is the *next* day's log realised variance. The `rv_slope` column - `log_rv_w - log_rv_m` - is deliberately **excluded** from the design matrix: it is an exact linear combination of two columns that are already there, the design matrix becomes singular, and the least-squares solution is then whatever the solver decides it is. A test asserts its absence rather than trusting the comment.

The realised measure used for scoring is **`rv`**, chosen over `bv` and `padj` because it is the quantity the models are trying to forecast and because the jump contamination it carries is the same contamination the latent-variance column removes.

## 4. The models

All models implement the same interface: `fit(returns, features, end)` estimates parameters on the prefix, `forecast(returns, features, start, stop)` returns one-step-ahead variances for the block. The separation matters: inside a refit block the parameters are frozen but the variance *state* is filtered with realised returns as they arrive, which is what a desk does and the only way a recursion stays legal.

| model | parameters | notes |
|---|---|---|
| `rolling_21` | none | trailing sample variance; the benchmark |
| `rolling_63` | none | the same over a quarter |
| `ewma_94` | none | RiskMetrics decay 0.94 |
| `ewma_fitted` | decay | chosen on the training block from a grid |
| `garch` | 4 | Gaussian likelihood |
| `gjr` | 5 | `gamma` on negative shocks; **t** likelihood |
| `egarch` | 5 | log-variance recursion |
| `har` | 4 | OLS on logs of the realised measures |
| `harx` | 5 | the same plus implied variance |
| `gbm` | trees | LightGBM on the full feature set |
| `gru` | weights | one layer, written out by hand |

### 4.1 The parametric route

GARCH-family models are estimated by maximum likelihood on the Student-t density, with the degrees of freedom estimated jointly with the recursion rather than fixed. Variance targeting is applied to the intercept so that the unconditional level matches the sample rather than being estimated through a poorly identified parameter. Both choices are tested: a recovery test asserts the fitted parameters reproduce the ones used to simulate, and a convergence test asserts the solution is not the initial point - an optimiser that returns its starting value produces a plausible series and a wrong one.

### 4.2 The realised-measure route

HAR is fitted by OLS on log realised variance with the three components above. The forecast is **log-normal**, so the conditional mean of the variance is not `exp(fitted log)` but that value times a dispersion correction `exp(sigma_resid^2 / 2)`. Omitting it biases every forecast downwards by a few percent, consistently, which is the sort of error that survives review because it never produces an implausible number.

### 4.3 The network

The GRU is implemented directly - forward pass, backpropagation through time, Adam, early stopping - with no framework, so that the gradient can be checked against a central difference. It is deliberately a *small* model on a *small* feature set: the comparison is whether a sequence model adds anything over a linear regression with the right features, and the answer from the results is that it does not, consistently. That is a legitimate finding and it is reported as one.

## 5. The VaR and ES routes

Write `sigma_t` for the conditional volatility. Every route below consumes the same series.

```
historical      VaR_t = -quantile_{1-a}(r_{t-w..t-1})
                        age-weighted: exponentially declining weights in the same quantile
filtered        z_s = r_s / sigma_s ;  VaR_t = -sigma_t * quantile_{1-a}(z)
filtered-EVT    the upper tail of z above a threshold is fitted with a GPD and extrapolated
parametric      VaR_t = -sigma_t * F^-1(1-a)  for F Gaussian or Student-t
simulation      the variance recursion is simulated forward; the quantile is read off the paths
ES:             the conditional mean of the loss beyond the VaR, on the same scale
```

### 5.1 Age-weighted historical simulation

Weights decay geometrically with age; the quantile is the weighted order statistic. With zero decay this reduces to the flat-window case, which is asserted - to within the difference between an interpolated order statistic and a cumulative-weight inversion, which is about 0.2% in absolute terms and is not zero.

### 5.2 The generalised-Pareto tail

Exceedances above a threshold quantile are fitted by **method of moments**, not maximum likelihood: the closed form is `xi = (1 - mean^2 / variance) / 2`, and unlike MLE on a handful of exceedances it does not fail to converge on the quiet windows that make up most of the sample. The plausible-looking alternative - `(mean^2 / mean_square - 1) / 2` - returns `-0.25` for an exponential tail whose true shape is zero, implying a bounded tail, and produced a VaR about 20% below the empirical quantile and twice the nominal breach rate. A test fits the estimator to exponential data and asserts the shape is zero.

When there are too few exceedances to estimate two parameters - fewer than eight, which a quiet 250-day window does produce - the fit is abandoned and the **empirical quantile is returned in its place**. The fallback deliberately returns the same object on the same scale as the fitted path, because the alternative is a third convention with its own sign, and this one was wrong once: the caller clamps at zero, so a sign error did not raise, it reported a VaR and an ES of exactly zero. A zero ES then became the denominator of the ES statistic and produced a value of order `1e8`.

### 5.3 Simulation

**20,000 paths** by default. Each path draws future shocks from the fitted Student-t, propagates the variance recursion, and accumulates returns over the horizon. The quantile and the mean of the tail beyond it are read off the path distribution. This is the only route that produces a multi-day number without a scaling assumption, which is why it is the reference for the horizon study. The path count is set so that the standard error of the quantile is small relative to the differences between methods; `--fast` lowers it, which makes individual days noisy while leaving the average usable.

## 6. The tests

Each is stated with its null, because a backtest quoted without its null is not interpretable.

| test | null | statistic |
|---|---|---|
| Kupiec | the true breach rate equals `1 - a` | likelihood ratio, chi-squared on 1 degree of freedom, **two-sided** |
| Christoffersen | breaches are independent over time | first-order Markov likelihood ratio, chi-squared on 1 |
| Conditional coverage | both hold | the two statistics added, chi-squared on 2 |
| Engle-Manganelli | the breach has no predictable structure | regression of the hit on its lags and on the VaR, chi-squared |
| Acerbi-Szekely | the ES is the conditional mean beyond the VaR | standardised mean of `-r/ES - 1` over breach days, asymptotically normal |

**Two-sided matters.** A one-sided Kupiec test would accept a model that never breaches, and a model that never breaches has not been validated - it has failed to produce evidence. The implementation rejects in both directions and a test asserts it does.

### 6.1 Basel traffic light

Exceptions are counted over rolling **250-day** windows at the 99% level. Fewer than five is green, up to nine yellow, ten or more red, and the capital add-on is not linear:

| exceptions | 0-4 | 5 | 6 | 7 | 8 | 9 | 10+ |
|---|---|---|---|---|---|---|---|
| add-on | 0% | 40% | 50% | 65% | 75% | 85% | 100% |

A window that is not yet full is left **blank** rather than reported as a low count, so a model cannot appear green because it has not been live long enough to be measured. The same logic is implemented in SQL and compared against the Python version row by row.

## 7. Evaluation protocol

### 7.1 Walk-forward

An initial window of **1,000 days** is held out, then the models are re-estimated every **21 days** on an expanding window and asked for one-step-ahead forecasts over the following block. The network and the tree are refitted on a longer cycle, since refitting a boosted model every month on a growing window is not what anyone does. That the heavy models are genuinely refitted less often is asserted rather than assumed.

### 7.2 Loss functions

QLIKE ranks the variance forecasts:

```
QLIKE(h, v) = v/h - ln(v/h) - 1
```

where `v` is the forecast and `h` the realised measure. It is the logarithmic score, it is robust to the noise in the proxy in the sense of Patton (2011), and it is asymmetric in exactly the way a risk number should be: under-prediction is punished without bound, over-prediction logarithmically. MSE on the volatility scale is reported beside it because it is the number the audience recognises, and a model that wins on QLIKE while losing badly on MSE is telling you where its errors sit.

For a quantile the correct loss is the **tick loss**:

```
TL(r, v, a) = ((1 - a) - 1[r < -v]) * (r + v)
```

The weight on a breach is the **tail probability** `1 - a`, not the coverage level `a`. Using `a` is a one-token change that inverts the objective: the loss then grows almost linearly in `v`, its minimum moves to `v = 0`, and a table of methods sorted by it ranks them by *reported risk, smallest first* - while still sorting, and still crowning a winner. A test asserts that the inverted form is minimised at zero and the correct form at the true quantile.

### 7.3 Paired comparison

Diebold-Mariano on the QLIKE loss differential, Newey-West with 10 lags, two-sided. The diagonal is **NaN**, not 1.0: a model compared with itself has a loss differential identically zero, a variance of zero, and an undefined statistic. Reporting `1.0` there would claim the diagonal was tested and found insignificant, which is a different and false statement.

### 7.4 Calibration

Mincer-Zarnowitz regresses the realised measure on the forecast. The **slope** is the informative output: 1.0 means correctly scaled, below 1.0 means the forecast is too timid, above 1.0 means it over-reacts. The joint test that the intercept is zero and the slope is one is also reported, but at this sample size it rejects deviations far too small to act on - so the coefficient is quoted and the flag is not treated as a verdict. A separate test documents this by construction.

### 7.5 The latent variance

Because the data are simulated, the conditional variance that generated each day is known. The same models are scored against it, which removes the proxy noise from the comparison. *(Result: the observed best model is `harx`, and the ordering against the latent variance is broadly the same - QLIKE 0.0178 against 0.1251 against realised variance.)*

## 8. Known limitations

1. **The generator is a model of volatility, not of anything else.** No microstructure, no intraday seasonality, no discrete ticks, no trading halts. The measures are computed on a bridge that is too well behaved; the jump *detection* result is therefore cleaner here than it would be on tape.
2. **The likelihood is misspecified by design.** The parametric routes assume a symmetric t while the data are skewed. This is closer to reality than a correctly specified study, but it means their scores are a lower bound and should not be read as the best those models can do.
3. **Overlapping windows at ten days.** Multi-day returns are computed over overlapping windows, so hit indicators are mechanically dependent and the independence and dynamic-quantile p-values at h = 10 are biased downwards. The breach *rate* is unaffected. The honest fix is non-overlapping windows at a four-fifths cost in sample.
4. **The ES test requires breaches.** A route with fewer than five breach days is skipped. That is a sample-size constraint, not an endorsement: a method can pass by never producing enough evidence to be judged.
5. **One filter, one asset.** Every route shares the GJR-GARCH-t filter, so the comparison is of methods conditional on that filter. Whether the ranking survives a different filter is not answered here.
6. **No multiple-testing correction.** Seven routes over two levels and two horizons is twenty-eight ES tests. A single pass at the 5% level should be read accordingly.

## 9. Verification

The claims above that could be checked mechanically are, and the checks live in `tests/`:

| claim | test |
|---|---|
| no look-ahead in the features | corrupting the future leaves past forecasts unchanged |
| the model reads features, not returns | corrupting the return series leaves HAR unchanged |
| gradients are correct | every tensor's analytic gradient against a central difference |
| the optimiser converges | the GARCH solution is not the initial point |
| parameters are recoverable | simulated parameters recovered by fitting |
| the GPD shape is right | exponential tail gives shape 0, not -0.25 |
| the tail fallback keeps its scale | it equals the empirical quantile bit for bit |
| the EVT fallback would be zero if inverted | the clamped quantity is asserted directly |
| the tick loss is the right objective | its minimum is at the true quantile, not at zero |
| ES is never below VaR | asserted on every route at once |
| every route produces a positive number | strictly, not `>= 0` |
| the DQ test works | it detects a misspecified model and passes a correct one |
| the independence test has power | 20 consecutive breaches are rejected |
| the walk-forward is legal | five invariants across the whole scheme |
| SQL agrees with Python | log return, Parkinson, 21-day variance, exception count, zone |
| `--fast` loses no deliverable | every model and method present in the fast run |
| the docs match the run | the doc builder fails on a missing figure or a short table |

The end-to-end test reruns the entire pipeline on a shorter sample and checks the artefacts: the file list, the row counts, the model and method roster, the non-emptiness of the horizon study, that the winner is not the naive baseline, and that every number quoted in `RESULTS.md` is present in `metrics.json`.

Coverage for the reference route at 99%, one day, for orientation - the full tables are in `outputs/RESULTS.md`: 46 breaches in 4157 days (1.11% against a nominal 1.0%), Kupiec p = 0.4972.

