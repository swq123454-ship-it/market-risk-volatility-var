"""Tests for the SQL layer.

The SQL in ``sql/`` is executed here rather than merely linted, and its output is
compared against the Python implementations on the same rows. That comparison is
the whole point of the directory: a warehouse query that quietly disagrees with the
research code - a lag applied the wrong way, an off-by-one in a rolling window - is
the classic reason a model works in a notebook and not in production, and it costs
nothing to check.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marketrisk import backtest
from marketrisk import features as feat

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"

# The final SELECT order of each script, named once. Reading a result set by
# position is unavoidable with the DB-API, and the alternative - scattering bare
# integers through the tests - is how a column gets read one slot to the left after
# somebody adds an output field. If a script's projection changes, these go stale
# loudly, in one place, instead of silently in nine.
FEATURE_COLUMNS = ("symbol", "trade_date", "close", "log_return", "parkinson", "variance_21", "observations_21")
BASEL_COLUMNS = ("method", "level", "horizon", "trade_date", "exceptions_250d", "basel_zone", "basel_plus_factor")


def _column(script_columns: tuple[str, ...], name: str) -> int:
    return script_columns.index(name)


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript((SQL_DIR / "01_schema.sql").read_text())
    return connection


def _insert_prices(
    connection: sqlite3.Connection, symbol: str, rows: list[tuple[str, float, float, float, float]]
) -> None:
    connection.executemany(
        "INSERT INTO prices (symbol, trade_date, open, high, low, close) VALUES (?, ?, ?, ?, ?, ?)",
        [(symbol, *row) for row in rows],
    )


def _price_rows(
    rng: np.random.Generator, n: int, start: float = 100.0
) -> tuple[list[tuple[str, float, float, float, float]], list[str]]:
    """A random-walk price series with a plausible daily bar around each close.

    Returns the rows *and* the date strings, because several tests need to map a
    result row back to its position in the series. Dates come from a business-day
    range rather than a hand-built ``2020-01-{i:02d}`` string: that template quietly
    produces ``2020-01-31`` the day after ``2020-01-30`` and then ``2020-01-32``,
    which SQLite stores happily and ``datetime`` refuses, so the failure surfaces far
    from its cause.
    """
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2020-01-01", periods=n)]
    steps = rng.normal(0.0, 0.01, n)
    closes = start * np.exp(np.cumsum(steps))
    rows = []
    for i in range(n):
        previous = start if i == 0 else closes[i - 1]
        high = max(previous, closes[i]) * (1.0 + abs(rng.normal(0.0, 0.004)))
        low = min(previous, closes[i]) * (1.0 - abs(rng.normal(0.0, 0.004)))
        rows.append((dates[i], previous, high, low, closes[i]))
    return rows, dates


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_the_schema_loads_and_indexes_are_created():
    connection = _connect()
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"prices", "var_backtest"} <= tables
    indexes = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert "idx_prices_symbol_date" in indexes
    connection.close()


def test_the_schema_is_idempotent():
    """Re-running the DDL must not fail: it is executed on every deployment."""
    connection = _connect()
    connection.executescript((SQL_DIR / "01_schema.sql").read_text())
    connection.close()


# ---------------------------------------------------------------------------
# Feature panel
# ---------------------------------------------------------------------------
def test_sql_log_returns_match_the_python_implementation():
    rng = np.random.default_rng(71)
    rows, _ = _price_rows(rng, 80)
    connection = _connect()
    _insert_prices(connection, "SPX", rows)

    query = (SQL_DIR / "02_market_features.sql").read_text()
    frame = connection.execute(query).fetchall()
    connection.close()

    closes = np.array([r[4] for r in rows])
    expected = np.log(closes[1:] / closes[:-1])
    column = _column(FEATURE_COLUMNS, "log_return")
    got = np.array([row[column] for row in frame if row[column] is not None])
    assert np.allclose(got, expected, atol=1e-12)


def test_sql_parkinson_matches_the_python_implementation():
    rng = np.random.default_rng(72)
    rows, _ = _price_rows(rng, 60)
    connection = _connect()
    _insert_prices(connection, "SPX", rows)
    frame = connection.execute((SQL_DIR / "02_market_features.sql").read_text()).fetchall()
    connection.close()

    high = pd.Series([r[2] for r in rows])
    low = pd.Series([r[3] for r in rows])
    expected = feat.parkinson(high, low).to_numpy()
    got = np.array([row[_column(FEATURE_COLUMNS, "parkinson")] for row in frame])
    assert np.allclose(got, expected, atol=1e-12)


def test_sql_trailing_variance_matches_the_python_implementation():
    """A 21-day window, inclusive of the current row, over already-computed returns.

    The window is checked position by position against a NumPy recomputation rather
    than only at the final row, because the interesting failure is a window that is
    right on average - off by one at every single date.
    """
    rng = np.random.default_rng(73)
    rows, dates = _price_rows(rng, 120)
    connection = _connect()
    _insert_prices(connection, "SPX", rows)
    frame = connection.execute((SQL_DIR / "02_market_features.sql").read_text()).fetchall()
    connection.close()

    closes = np.array([r[4] for r in rows])
    returns = np.log(closes[1:] / closes[:-1])
    value_at = _column(FEATURE_COLUMNS, "variance_21")
    count_at = _column(FEATURE_COLUMNS, "observations_21")
    position = {date: i for i, date in enumerate(dates)}

    checked_full = 0
    for row in frame:
        index = position[row[1]]
        if index < 1:
            # The first row has no lagged return, so the window is empty rather than
            # zero-filled and the SQL agrees: NULL, not 0.0.
            assert row[value_at] is None and row[count_at] == 0
            continue
        # `log_return` at row i is returns[i-1]. The window covers rows i-20..i, so
        # it spans returns[i-21 : i] - a half-open slice of size 21, because the
        # current row's own return is the *last* element of the window, not the row
        # count. Writing `index - 20` there is the classic off-by-one: it agrees with
        # the SQL everywhere except the row where the window first fills.
        window = returns[max(0, index - 21) : index]
        assert row[count_at] == window.size
        assert row[value_at] == pytest.approx(float(np.mean(window**2)), rel=1e-12)
        checked_full += window.size == 21
    assert checked_full > 90


def test_sql_returns_are_dated_on_the_day_they_were_realised():
    """The first row has no return, and it is NULL rather than zero.

    A zero would be indistinguishable from a flat day, and the feature builder
    downstream shifts the column, so a fabricated zero would enter the design matrix
    as a real observation of no volatility.
    """
    rng = np.random.default_rng(74)
    rows, _ = _price_rows(rng, 30)
    connection = _connect()
    _insert_prices(connection, "SPX", rows)
    frame = connection.execute((SQL_DIR / "02_market_features.sql").read_text()).fetchall()
    connection.close()
    column = _column(FEATURE_COLUMNS, "log_return")
    assert frame[0][column] is None
    assert frame[1][column] is not None


# ---------------------------------------------------------------------------
# Backtest rollup
# ---------------------------------------------------------------------------
def _insert_backtest(connection: sqlite3.Connection, method: str, exceptions: list[int]) -> None:
    rows = []
    for i, hit in enumerate(exceptions):
        realised = -0.05 if hit else 0.001
        rows.append((method, f"2020-{i // 28 + 1:02d}-{i % 28 + 1:02d}", 0.99, 1, 0.02, realised))
    connection.executemany(
        "INSERT INTO var_backtest (method, trade_date, level, horizon, var_forecast, realised) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )


def test_sql_exception_count_and_zone_match_the_python_implementation():
    """Twenty exceptions spread across the sample, checked against ``basel_zone``."""
    connection = _connect()
    hits = np.zeros(300, dtype=int)
    hits[::15] = 1  # 20 exceptions, the last ten inside the final 250-day window
    _insert_backtest(connection, "m", hits.tolist())

    frame = connection.execute((SQL_DIR / "03_basel_traffic_light.sql").read_text()).fetchall()
    connection.close()

    last = frame[-1]
    expected_count = int(hits[-250:].sum())
    assert last[_column(BASEL_COLUMNS, "exceptions_250d")] == expected_count
    assert last[_column(BASEL_COLUMNS, "basel_zone")] == backtest.basel_zone(expected_count)
    assert last[_column(BASEL_COLUMNS, "basel_plus_factor")] == pytest.approx(
        backtest.basel_plus_factor(expected_count)
    )


def test_sql_leaves_a_partial_window_empty():
    connection = _connect()
    _insert_backtest(connection, "m", np.zeros(300, dtype=int).tolist())
    frame = connection.execute((SQL_DIR / "03_basel_traffic_light.sql").read_text()).fetchall()
    connection.close()
    count_at = _column(BASEL_COLUMNS, "exceptions_250d")
    zone_at = _column(BASEL_COLUMNS, "basel_zone")
    assert all(row[count_at] is None for row in frame[:249])
    assert frame[249][count_at] == 0
    # A model that has not yet been measured is left blank, not called green.
    assert frame[0][zone_at] is None
    assert frame[249][zone_at] == "green"


@pytest.mark.parametrize(
    ("exceptions", "zone", "factor"),
    [(0, "green", 0.00), (4, "green", 0.00), (5, "yellow", 0.40), (9, "yellow", 0.85), (10, "red", 1.00)],
)
def test_sql_zone_boundaries_match_the_python_implementation(exceptions, zone, factor):
    """Every row carries the same count, so the final row exercises the boundary."""
    connection = _connect()
    hits = np.zeros(300, dtype=int)
    hits[-exceptions:] = 1 if exceptions else 0
    _insert_backtest(connection, "m", hits.tolist())
    frame = connection.execute((SQL_DIR / "03_basel_traffic_light.sql").read_text()).fetchall()
    connection.close()
    last = frame[-1]
    assert last[_column(BASEL_COLUMNS, "exceptions_250d")] == exceptions
    assert last[_column(BASEL_COLUMNS, "basel_zone")] == zone == backtest.basel_zone(exceptions)
    assert last[_column(BASEL_COLUMNS, "basel_plus_factor")] == pytest.approx(factor)


def test_sql_separates_methods_and_levels():
    connection = _connect()
    _insert_backtest(connection, "a", np.zeros(300, dtype=int).tolist())
    _insert_backtest(connection, "b", np.ones(300, dtype=int).tolist())
    frame = connection.execute((SQL_DIR / "03_basel_traffic_light.sql").read_text()).fetchall()
    connection.close()
    count_at = _column(BASEL_COLUMNS, "exceptions_250d")
    zone_at = _column(BASEL_COLUMNS, "basel_zone")
    by_method: dict[str, list] = {}
    for row in frame:
        by_method.setdefault(row[0], []).append(row)
    assert len(by_method) == 2
    # All 250 days in the window breached, which is far outside the table's range.
    assert by_method["b"][-1][count_at] == 250
    assert by_method["b"][-1][zone_at] == "red"
    assert by_method["a"][-1][count_at] == 0
    assert by_method["a"][-1][zone_at] == "green"
