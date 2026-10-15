-- ---------------------------------------------------------------------------
-- Schema for the daily price panel.
--
-- Everything in this directory is plain SQLite-compatible SQL and is executed by
-- the test suite, so it is not decoration: the log returns, range estimator and
-- trailing variance it produces are checked against the Python implementations in
-- `marketrisk.features` on the same input. A data warehouse that disagrees with the
-- research code is the usual reason a model cannot be put into production, and the
-- check is cheap.
--
-- In a real shop `prices` would be a partitioned external table over the vendor
-- feed and the queries below would be views on top of it. The shape is the same.
-- ---------------------------------------------------------------------------

DROP TABLE IF EXISTS prices;
CREATE TABLE prices (
    symbol  TEXT    NOT NULL,
    trade_date DATE NOT NULL,
    open    REAL    NOT NULL,
    high    REAL    NOT NULL,
    low     REAL    NOT NULL,
    close   REAL    NOT NULL,
    PRIMARY KEY (symbol, trade_date)
);

-- A market-risk query almost always filters to one symbol and then walks time
-- forward, which is what this index serves.
CREATE INDEX idx_prices_symbol_date ON prices (symbol, trade_date);

DROP TABLE IF EXISTS var_backtest;
CREATE TABLE var_backtest (
    method      TEXT    NOT NULL,
    trade_date  DATE    NOT NULL,
    level       REAL    NOT NULL,
    horizon     INTEGER NOT NULL,
    var_forecast REAL   NOT NULL,  -- positive loss magnitude
    realised    REAL    NOT NULL,  -- the return the forecast refers to
    PRIMARY KEY (method, trade_date, level, horizon)
);
