-- ---------------------------------------------------------------------------
-- Build the per-symbol feature panel: log return, Parkinson range variance and
-- the trailing 21-day variance.
--
-- Two conventions here are carried straight from the research code and are the
-- reason this query is worth having in SQL at all:
--
--   * the return at row `t` is `log(close_t) - log(close_{t-1})` - the value is
--     dated on the day it was *realised*;
--   * every feature is then shifted forward one day by the consumer, so that the
--     row dated `t` contains only information known at the close of `t-1`.
--
-- The second step is a `LAG` in this file and a `.shift(1)` in Python. Splitting
-- them across two systems is how a leak gets introduced, so the trailing variance
-- below is explicitly written against the *already lagged* return column.
-- ---------------------------------------------------------------------------

WITH daily AS (
    SELECT
        symbol,
        trade_date,
        close,
        high,
        low,
        LN(close) - LAG(LN(close)) OVER w AS log_return
    FROM prices
    WINDOW w AS (PARTITION BY symbol ORDER BY trade_date)
)
SELECT
    symbol,
    trade_date,
    close,
    log_return,
    -- Parkinson (1980): (ln(H/L))^2 / (4 ln 2). Uses the daily bar only, which is
    -- what a close-price feed leaves you with.
    POWER(LN(high / low), 2) / (4.0 * LN(2.0))                      AS parkinson,
    AVG(log_return * log_return)
        OVER (PARTITION BY symbol ORDER BY trade_date
              ROWS BETWEEN 20 PRECEDING AND CURRENT ROW)             AS variance_21,
    COUNT(log_return)
        OVER (PARTITION BY symbol ORDER BY trade_date
              ROWS BETWEEN 20 PRECEDING AND CURRENT ROW)             AS observations_21
FROM daily
ORDER BY symbol, trade_date;
