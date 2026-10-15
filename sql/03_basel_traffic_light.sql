-- ---------------------------------------------------------------------------
-- Rolling 250-day exception count and Basel zone per method.
--
-- The traffic light is the one risk number in this repository that a regulator
-- reads, and it is a pure database computation: count the exceptions over a rolling
-- window and map the count to a zone. Keeping it in SQL means the number on a
-- dashboard is derived from the same rows the backtest wrote, rather than
-- recomputed by whatever the dashboard's language happens to be.
--
-- Two details that the Python and SQL implementations must agree on, and do:
--
--   * an exception is `realised < -var_forecast`, because the VaR is stored as a
--     positive loss magnitude;
--   * a window that is not yet full is left as NULL rather than reported as a low
--     count, so a model cannot look green simply because it has not been live long
--     enough to be measured.
-- ---------------------------------------------------------------------------

WITH hits AS (
    SELECT
        method,
        level,
        horizon,
        trade_date,
        CASE WHEN realised < -var_forecast THEN 1 ELSE 0 END AS exception
    FROM var_backtest
),
counted AS (
    SELECT
        method,
        level,
        horizon,
        trade_date,
        SUM(exception)
            OVER (PARTITION BY method, level, horizon ORDER BY trade_date
                  ROWS BETWEEN 249 PRECEDING AND CURRENT ROW) AS window_exceptions,
        COUNT(*)
            OVER (PARTITION BY method, level, horizon ORDER BY trade_date
                  ROWS BETWEEN 249 PRECEDING AND CURRENT ROW) AS window_days
    FROM hits
)
SELECT
    method,
    level,
    horizon,
    trade_date,
    CASE WHEN window_days = 250 THEN window_exceptions END AS exceptions_250d,
    CASE
        WHEN window_days < 250                 THEN NULL
        WHEN window_exceptions <= 4            THEN 'green'
        WHEN window_exceptions <= 9            THEN 'yellow'
        ELSE 'red'
    END AS basel_zone,
    -- The capital add-on is not linear in the count: the step from four exceptions
    -- to five costs 40% of the charge and the step to ten costs it again.
    CASE
        WHEN window_days < 250                 THEN NULL
        WHEN window_exceptions <= 4            THEN 0.00
        WHEN window_exceptions = 5             THEN 0.40
        WHEN window_exceptions = 6             THEN 0.50
        WHEN window_exceptions = 7             THEN 0.65
        WHEN window_exceptions = 8             THEN 0.75
        WHEN window_exceptions = 9             THEN 0.85
        ELSE 1.00
    END AS basel_plus_factor
FROM counted
ORDER BY method, level, horizon, trade_date;
